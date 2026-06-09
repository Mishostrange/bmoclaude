# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave
import struct

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal

# =========================================================================
# NEW ARCHITECTURE — Phase 0-4 imports
# =========================================================================
from core.logger import setup_logging
from core.config import load_config as load_agent_config, get_ollama_options, AgentConfig
from core.events import get_bus, EventType, Event, STTFinalEvent
from core.state_manager import StateManager, AssistantState
from core.session_manager import SessionManager

# Set up structured logging before anything else
setup_logging()
import logging
logger = logging.getLogger(__name__)

# --- AI ENGINES ---
import openwakeword
from openwakeword.model import Model
import ollama

# --- WEB SEARCH ---
from duckduckgo_search import DDGS

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

# Load config via the new centralized loader
CURRENT_CONFIG: AgentConfig = load_agent_config(CONFIG_FILE)

# Legacy dict-style lookups still used by the legacy pipeline
TEXT_MODEL = CURRENT_CONFIG.text_model
VISION_MODEL = CURRENT_CONFIG.vision_model

# ── Migration flag ────────────────────────────────────────────────────────
# Set use_legacy_pipeline = false in config.json to activate the new stack.
USE_LEGACY_PIPELINE: bool = CURRENT_CONFIG.use_legacy_pipeline

# ── Shared infrastructure (always created) ────────────────────────────────
EVENT_BUS = get_bus()
STATE_MANAGER = StateManager(bus=EVENT_BUS)

# ── New-pipeline components (lazy init in BotGUI.safe_main_execution) ─────
_audio_pipeline = None
_stt_engine = None
_piper_manager = None
_barge_in_ctrl = None

# HARDWARE SETTINGS (legacy)
INPUT_DEVICE_NAME = None

OLLAMA_OPTIONS = get_ollama_options(CURRENT_CONFIG)


def resolve_input_device(config: AgentConfig):
    requested = config.input_device
    if requested in (None, "", "default"):
        return None
    try:
        devices = sd.query_devices()
    except Exception as e:
        logger.warning("[AUDIO] Device query failed: %s", e)
        return None
    if isinstance(requested, int) or (isinstance(requested, str) and str(requested).isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        logger.warning("[AUDIO] Input device index not found: %d", index)
        return None
    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx
    logger.warning("[AUDIO] Input device name not found: %s", requested)
    return None


INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)


def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        device_info = sd.query_devices(device)
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception:
        pass
    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    return 44100


class BotStates:
    """Legacy state constants — kept for GUI compatibility."""
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
    ERROR     = "error"
    CAPTURING = "capturing"
    WARMUP    = "warmup"


# ── Map new AssistantState → legacy BotStates for GUI ─────────────────────
_STATE_MAP = {
    AssistantState.IDLE:          BotStates.IDLE,
    AssistantState.LISTENING:     BotStates.LISTENING,
    AssistantState.PROCESSING:    BotStates.THINKING,
    AssistantState.SPEAKING:      BotStates.SPEAKING,
    AssistantState.INTERRUPTED:   BotStates.IDLE,
    AssistantState.ERROR:         BotStates.ERROR,
    AssistantState.SHUTTING_DOWN: BotStates.IDLE,
}

# ── System prompt ──────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic.

INSTRUCTIONS:
- If the user asks for a physical action (time, search, photo), output JSON.
- If the user just wants to chat, reply with NORMAL TEXT.

### EXAMPLES ###

User: What time is it?
You: {"action": "get_time", "value": "now"}

User: Hello!
You: Hi! I am ready to help!

User: Search for news about robots.
You: {"action": "search_web", "value": "robots news"}

User: What do you see right now?
You: {"action": "capture_image", "value": "environment"}

### END EXAMPLES ###
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.system_prompt_extras

# Sound directories
greeting_sounds_dir = CURRENT_CONFIG.sounds_greeting_dir
ack_sounds_dir      = CURRENT_CONFIG.sounds_ack_dir
thinking_sounds_dir = CURRENT_CONFIG.sounds_thinking_dir
error_sounds_dir    = CURRENT_CONFIG.sounds_error_dir


# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True)
        master.bind('<Escape>', self.exit_fullscreen)

        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)

        # ── Internal state ─────────────────────────────────────────────
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None

        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()

        self.last_ptt_time = 0
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()

        self.tts_queue = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.current_audio_process = None
        self.exiting = False

        # ── New pipeline: session manager ──────────────────────────────
        self.session_manager = SessionManager(
            bus=EVENT_BUS,
            system_prompt=SYSTEM_PROMPT,
            max_history=CURRENT_CONFIG.session_max_history,
        )

        # ── Subscribe GUI to StateManager events ───────────────────────
        EVENT_BUS.subscribe(EventType.STATE_CHANGED, self._on_state_changed)

        # ── Wake Word ──────────────────────────────────────────────────
        logger.info("[INIT] Loading Wake Word…")
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                logger.info("[INIT] Wake Word loaded.")
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    logger.info("[INIT] Wake Word loaded (New API).")
                except Exception as e:
                    logger.error("[CRITICAL] Failed to load wake word model: %s", e)
            except Exception as e:
                logger.error("[CRITICAL] Failed to load wake word model: %s", e)
        else:
            logger.warning("[CRITICAL] Wake word model not found: %s", WAKE_WORD_MODEL)

        # ── GUI widgets ────────────────────────────────────────────────
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility)

        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)

        self.response_text = tk.Text(
            master, height=6, width=60, wrap=tk.WORD,
            state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12)
        )

        # Pipeline mode badge
        pipeline_label = "🆕 New Pipeline" if not USE_LEGACY_PIPELINE else "⚙️ Legacy Pipeline"
        self.status_var = tk.StringVar(value=f"Initializing… [{pipeline_label}]")
        self.status_label = ttk.Label(
            master, textvariable=self.status_var,
            background="#2e2e2e", foreground="white"
        )

        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation()

        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # =========================================================================
    # STATE BRIDGE  (new StateManager → legacy GUI state)
    # =========================================================================

    def _on_state_changed(self, event):
        """Sync GUI state whenever StateManager transitions."""
        new_bot_state = _STATE_MAP.get(event.current, BotStates.IDLE)
        if self.current_state != new_bot_state:
            self.current_state = new_bot_state
            self.master.after(0, lambda: None)  # Trigger animation loop refresh

    # =========================================================================
    # HELPERS
    # =========================================================================

    def extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None
        except Exception:
            return None

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        logger.info("--- SHUTDOWN SEQUENCE ---")

        # Stop new pipeline components if active
        global _audio_pipeline, _piper_manager
        if _audio_pipeline:
            try:
                _audio_pipeline.stop()
            except Exception:
                pass
        if _piper_manager:
            try:
                _piper_manager.stop()
            except Exception:
                pass

        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except Exception:
                pass

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()

        self.session_manager.end_session()
        self.save_chat_history()

        try:
            STATE_MANAGER.go_shutdown(reason="user exit")
        except Exception:
            pass

        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except Exception:
            pass
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.master.quit()
        except Exception:
            pass

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError:
            pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            logger.debug("[PTT] Toggle OFF")
            self.recording_active.clear()
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                logger.debug("[PTT] Toggle ON")
                self.recording_active.set()
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING):
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try:
                    self.current_audio_process.terminate()
                except Exception:
                    pass
            # New pipeline: delegate to PiperManager
            if not USE_LEGACY_PIPELINE and _piper_manager:
                _piper_manager.interrupt()
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"]
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            self.current_frame_index = random.randint(1, len(frames) - 1) if len(frames) > 1 else 0
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg:
                logger.debug("[STATE] %s: %s", state.upper(), msg)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg:
                self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except Exception:
                    pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, (text + "\n") if newline else text)
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, update_text_stream)

    # =========================================================================
    # 3. ACTION ROUTER
    # =========================================================================

    def execute_action_and_get_result(self, action_data):
        raw_action = action_data.get("action", "").lower().strip()
        value = action_data.get("value") or action_data.get("query")

        VALID_TOOLS = {"get_time", "search_web", "capture_image"}
        ALIASES = {
            "google": "search_web", "browser": "search_web", "news": "search_web",
            "search_news": "search_web", "look": "capture_image", "see": "capture_image",
            "check_time": "get_time"
        }

        action = ALIASES.get(raw_action, raw_action)
        logger.debug("ACTION: %s -> %s", raw_action, action)

        if action not in VALID_TOOLS:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."

        elif action == "search_web":
            logger.info("Searching web for: %s", value)
            try:
                with DDGS() as ddgs:
                    results = []
                    try:
                        results = list(ddgs.news(value, region='us-en', max_results=1))
                    except Exception as e:
                        logger.debug("News search error: %s", e)
                    if not results:
                        try:
                            results = list(ddgs.text(value, region='us-en', max_results=1))
                        except Exception as e:
                            logger.debug("Text search error: %s", e)
                    if results:
                        r = results[0]
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        return f"SEARCH RESULTS for '{value}':\nTitle: {title}\nSnippet: {body[:300]}"
                    return "SEARCH_EMPTY"
            except Exception as e:
                logger.warning("Search error: %s", e)
                return "SEARCH_ERROR"

        elif action == "capture_image":
            return "IMAGE_CAPTURE_TRIGGERED"

        return None

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()

            if not USE_LEGACY_PIPELINE:
                self._start_new_pipeline()
            else:
                # Legacy TTS worker
                self.tts_active.set()
                self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
                self.tts_thread.start()

            self.session_manager.start_session()

            while True:
                trigger_source = self.detect_wake_word_or_ptt()
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting…")
                    continue

                self.set_state(BotStates.LISTENING, "I'm listening!")

                if not USE_LEGACY_PIPELINE:
                    user_text = self._wait_for_stt()
                else:
                    audio_file = None
                    if trigger_source == "PTT":
                        audio_file = self.record_voice_ptt()
                    else:
                        audio_file = self.record_voice_adaptive()
                    if not audio_file:
                        self.set_state(BotStates.IDLE, "Heard nothing.")
                        continue
                    user_text = self.transcribe_audio(audio_file)

                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text, img_path=None)

        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    # =========================================================================
    # NEW PIPELINE STARTUP (Phase 4)
    # =========================================================================

    def _start_new_pipeline(self):
        """Initialise and start all new-architecture components."""
        global _audio_pipeline, _stt_engine, _piper_manager, _barge_in_ctrl

        logger.info("[Pipeline] Starting new architecture…")

        # Audio pipeline (mic + VAD)
        from audio.audio_pipeline import AudioPipeline
        _audio_pipeline = AudioPipeline(cfg=CURRENT_CONFIG, bus=EVENT_BUS)
        _audio_pipeline.start()

        # STT engine
        from ai.stt import STTEngine
        _stt_engine = STTEngine(cfg=CURRENT_CONFIG, bus=EVENT_BUS)
        _stt_engine.start()

        # Persistent Piper TTS
        from audio.tts_engine import PiperManager
        _piper_manager = PiperManager(
            cfg=CURRENT_CONFIG,
            state_manager=STATE_MANAGER,
            bus=EVENT_BUS,
        )
        _piper_manager.start()

        # Barge-in controller (Phase 5)
        from audio.barge_in import BargeInController
        _barge_in_ctrl = BargeInController(
            state_manager=STATE_MANAGER,
            bus=EVENT_BUS,
        )

        logger.info("[Pipeline] New architecture running.")

    def _wait_for_stt(self, timeout: float = 30.0) -> str:
        """
        Block until STT_FINAL event arrives (new pipeline).
        Returns the transcribed text or empty string on timeout.
        """
        result_event = threading.Event()
        result_holder: dict = {"text": ""}

        def on_stt_final(event):
            text = getattr(event, "text", "") or (event.data or {}).get("text", "")
            if text:
                result_holder["text"] = text
                result_event.set()

        EVENT_BUS.subscribe(EventType.STT_FINAL, on_stt_final)
        STATE_MANAGER.go_listening(reason="waiting for speech")

        got_result = result_event.wait(timeout=timeout)
        EVENT_BUS.unsubscribe(EventType.STT_FINAL, on_stt_final)

        if not got_result:
            logger.warning("[Pipeline] STT timeout after %.1f s.", timeout)
        return result_holder["text"]

    # =========================================================================
    # WARM UP
    # =========================================================================

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains…")
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            logger.warning("Failed to load %s: %s", TEXT_MODEL, e)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        logger.info("Models loaded.")

    # =========================================================================
    # WAKE WORD DETECTION (unchanged from legacy)
    # =========================================================================

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting…")
        self.ptt_event.clear()

        if self.oww_model:
            self.oww_model.reset()

        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000
        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.input_sample_rate)
        use_resampling = (input_rate != OWW_SAMPLE_RATE)
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        stream_args = {
            "samplerate": input_rate,
            "channels": 1,
            "dtype": 'int16',
            "blocksize": input_chunk_size,
            "device": INPUT_DEVICE_NAME,
        }

        try:
            self._listen_loop(stream_args, input_chunk_size, CHUNK_SIZE, use_resampling)
        except StopIteration as si:
            return str(si)
        except Exception as e:
            logger.warning("[AUDIO] Stream failed: %s. Retrying…", e)
            try:
                stream_args["blocksize"] = 1024
                stream_args["latency"] = "high"
                use_resampling = True
                self._listen_loop(stream_args, 1024, CHUNK_SIZE, use_resampling)
            except StopIteration as si:
                return str(si)
            except Exception as e2:
                logger.error("[CRITICAL] Wake Word Stream Error: %s", e2)
                self.ptt_event.wait()
                return "PTT"

        return "WAKE"

    def _listen_loop(self, stream_args, input_chunk_size, target_chunk_size, use_resampling):
        with sd.InputStream(**stream_args) as stream:
            logger.debug(
                "[AUDIO] Listening — rate=%d  block=%d",
                stream_args['samplerate'], stream_args['blocksize'],
            )
            while True:
                if self.ptt_event.is_set():
                    self.ptt_event.clear()
                    raise StopIteration("PTT")

                rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                if rlist:
                    sys.stdin.readline()
                    raise StopIteration("CLI")

                read_size = input_chunk_size
                if stream_args.get('blocksize') == 0:
                    read_size = 1024

                try:
                    data, overflow = stream.read(read_size)
                    if overflow:
                        raise RuntimeError("Audio Buffer Overflow - Triggering Safe Mode")
                except Exception as e:
                    raise RuntimeError(f"Audio read failed: {e}")

                audio_data = np.frombuffer(data, dtype=np.int16)
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()

                if use_resampling:
                    step = len(audio_data) / target_chunk_size
                    indices = np.arange(0, len(audio_data), step)[:target_chunk_size].astype(int)
                    audio_data = audio_data[indices]

                current_max = np.max(np.abs(audio_data))
                if current_max > 200:
                    prediction = self.oww_model.predict(audio_data)
                    for mdl in self.oww_model.prediction_buffer.keys():
                        score = list(self.oww_model.prediction_buffer[mdl])[-1]
                        if score > WAKE_WORD_THRESHOLD:
                            logger.info("[WAKE] Triggered on '%s' (score=%.2f)", mdl, score)
                            self.oww_model.reset()
                            return

    # =========================================================================
    # LEGACY AUDIO  (unchanged; used when use_legacy_pipeline=true)
    # =========================================================================

    def record_voice_adaptive(self, filename="input.wav"):
        logger.debug("Recording (Adaptive)…")
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.input_sample_rate)

        silence_threshold = 0.006
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)
        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1
            if recorded_chunks < 5:
                return
            if volume_norm < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks:
                    silence_started = True
            else:
                silent_chunks = 0

        try:
            sd.stop()
            time.sleep(0.2)
            with sd.InputStream(
                samplerate=samplerate, channels=1, callback=callback,
                device=INPUT_DEVICE_NAME, blocksize=chunk_size
            ):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as e:
            logger.error("[AUDIO ERROR] Adaptive Recording Failed: %s", e)
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        logger.debug("Recording (PTT)…")
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.input_sample_rate)
        buffer = []

        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())

        try:
            sd.stop()
            time.sleep(0.2)
            with sd.InputStream(
                samplerate=samplerate, channels=1, callback=callback,
                device=INPUT_DEVICE_NAME
            ):
                while self.recording_active.is_set():
                    sd.sleep(50)
        except Exception as e:
            logger.error("[AUDIO ERROR] PTT Recording Failed: %s", e)
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer:
            return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        logger.debug("Transcribing (legacy)…")
        try:
            result = subprocess.run(
                [
                    "./whisper.cpp/build/bin/whisper-cli",
                    "-m", "./whisper.cpp/models/ggml-base.en.bin",
                    "-l", "en", "-t", "4", "-f", filename,
                ],
                capture_output=True, text=True,
            )
            lines = result.stdout.strip().split('\n')
            if lines and lines[-1].strip():
                last = lines[-1].strip()
                transcription = last.split("]")[1].strip() if ']' in last else last
            else:
                transcription = ""
            logger.info("Heard: '%s'", transcription)
            return transcription.strip()
        except Exception as e:
            logger.error("Transcription Error: %s", e)
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching…")
        try:
            subprocess.run(
                ["rpicam-still", "-t", "500", "-n",
                 "--width", "640", "--height", "480", "-o", BMO_IMAGE_FILE],
                check=True,
            )
            rotation = CURRENT_CONFIG.camera_rotation
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE)
                img = img.rotate(rotation, expand=True)
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            logger.error("Camera Error: %s", e)
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def chat_and_respond(self, text, img_path=None):
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.session_manager.reset_session()
            self.save_chat_history()
            with self.tts_queue_lock:
                self.tts_queue.append("Okay. Memory wiped.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        model_to_use = VISION_MODEL if img_path else TEXT_MODEL
        STATE_MANAGER.go_processing(reason="LLM inference")
        self.set_state(BotStates.THINKING, "Thinking…", cam_path=img_path)

        if img_path:
            messages = [{"role": "user", "content": text, "images": [img_path]}]
        else:
            if not USE_LEGACY_PIPELINE:
                # New pipeline uses SessionManager for history
                messages = self.session_manager.get_messages()
                messages.append({"role": "user", "content": text})
            else:
                user_msg = {"role": "user", "content": text}
                messages = self.permanent_memory + self.session_memory + [user_msg]

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        full_response_buffer = ""
        sentence_buffer = ""

        # Import SentenceBuilder for new pipeline
        if not USE_LEGACY_PIPELINE:
            from audio.tts_engine import SentenceBuilder
            sent_builder = SentenceBuilder()
        else:
            sent_builder = None

        try:
            stream = ollama.chat(
                model=model_to_use, messages=messages,
                stream=True, options=OLLAMA_OPTIONS,
            )

            is_action_mode = False

            for chunk in stream:
                if self.interrupted.is_set():
                    break
                content = chunk['message']['content']
                full_response_buffer += content

                if '{"' in content or "action:" in content.lower():
                    is_action_mode = True
                    self.thinking_sound_active.clear()
                    continue

                if is_action_mode:
                    continue

                self.thinking_sound_active.clear()
                STATE_MANAGER.go_speaking(reason="first token")
                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "Speaking…", cam_path=img_path)
                    self.append_to_text("BOT: ", newline=False)

                self._stream_to_text(content)

                if not USE_LEGACY_PIPELINE and sent_builder and _piper_manager:
                    # New pipeline: push tokens through SentenceBuilder → PiperManager
                    for sentence in sent_builder.push(content):
                        _piper_manager.speak(sentence)
                else:
                    # Legacy: buffer and queue sentences
                    sentence_buffer += content
                    if any(punct in content for punct in ".!?\n"):
                        clean = sentence_buffer.strip()
                        if clean and re.search(r'[a-zA-Z0-9]', clean):
                            with self.tts_queue_lock:
                                self.tts_queue.append(clean)
                        sentence_buffer = ""

            # Flush remaining sentence
            if not USE_LEGACY_PIPELINE and sent_builder and _piper_manager:
                remaining = sent_builder.flush()
                if remaining:
                    _piper_manager.speak(remaining)

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if action_data:
                    tool_result = self.execute_action_and_get_result(action_data)
                    self._handle_tool_result(tool_result, text, model_to_use, img_path)
            else:
                self.append_to_text("")
                if not USE_LEGACY_PIPELINE:
                    self.session_manager.add_assistant_message(full_response_buffer)
                else:
                    self.session_memory.append({"role": "assistant", "content": full_response_buffer})

            self.wait_for_tts()
            STATE_MANAGER.go_idle(reason="response complete")
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            logger.error("LLM Error: %s", e)
            STATE_MANAGER.go_error(reason=str(e))
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    def _handle_tool_result(self, tool_result, text, model_to_use, img_path):
        """Centralised handler for all action tool outcomes."""

        def _speak(msg):
            self.thinking_sound_active.clear()
            STATE_MANAGER.go_speaking(reason="tool result")
            self.set_state(BotStates.SPEAKING, "Speaking…", cam_path=img_path)
            self.append_to_text("BOT: ", newline=False)
            self.append_to_text(msg, newline=True)
            if not USE_LEGACY_PIPELINE and _piper_manager:
                _piper_manager.speak(msg)
            else:
                with self.tts_queue_lock:
                    self.tts_queue.append(msg)
            if not USE_LEGACY_PIPELINE:
                self.session_manager.add_assistant_message(msg)
            else:
                self.session_memory.append({"role": "assistant", "content": msg})

        if not tool_result:
            return

        if tool_result.startswith("CHAT_FALLBACK::"):
            _speak(tool_result.split("::", 1)[1])
            self.wait_for_tts()
            STATE_MANAGER.go_idle()
            self.set_state(BotStates.IDLE, "Ready")
            return

        if tool_result == "IMAGE_CAPTURE_TRIGGERED":
            new_img = self.capture_image()
            if new_img:
                self.chat_and_respond(text, img_path=new_img)
            return

        STATIC_REPLIES = {
            "INVALID_ACTION": "I am not sure how to do that.",
            "SEARCH_EMPTY":   "I searched, but I couldn't find any news about that.",
            "SEARCH_ERROR":   "I cannot reach the internet right now.",
        }
        if tool_result in STATIC_REPLIES:
            _speak(STATIC_REPLIES[tool_result])
            return

        # Summarise the search/tool result via LLM
        summary_prompt = [
            {"role": "system", "content": "Summarize this result in one short sentence."},
            {"role": "user", "content": f"RESULT: {tool_result}\nUser Question: {text}"},
        ]
        self.set_state(BotStates.THINKING, "Reading…")
        self.thinking_sound_active.set()
        final_resp = ollama.chat(
            model=model_to_use, messages=summary_prompt,
            stream=False, options=OLLAMA_OPTIONS,
        )
        final_text = final_resp['message']['content']
        _speak(final_text)

    def wait_for_tts(self):
        if not USE_LEGACY_PIPELINE:
            # New pipeline: Piper plays async; we just give it a moment
            time.sleep(0.2)
            return
        while self.tts_queue or self.tts_active.is_set():
            if self.interrupted.is_set():
                break
            time.sleep(0.1)

    # =========================================================================
    # LEGACY TTS WORKER
    # =========================================================================

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                    self.tts_active.set()
            if text:
                self.speak(text)
                self.tts_active.clear()
            else:
                time.sleep(0.05)

    def speak(self, text):
        """Legacy per-sentence Piper invocation (used when use_legacy_pipeline=true)."""
        clean = re.sub(r"[^\w\s,.!?:-]", "", text)
        if not clean.strip():
            return

        logger.debug("[PIPER SPEAKING] '%s'", clean)
        voice_model = CURRENT_CONFIG.voice_model

        try:
            self.current_audio_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            self.current_audio_process.stdin.write(clean.encode() + b'\n')
            self.current_audio_process.stdin.close()

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except Exception:
                native_rate = 48000

            PIPER_RATE = CURRENT_CONFIG.piper_rate
            use_native_rate = False
            try:
                sd.check_output_settings(device=None, samplerate=PIPER_RATE)
            except Exception:
                use_native_rate = True

            with sd.RawOutputStream(
                samplerate=native_rate if use_native_rate else PIPER_RATE,
                channels=1, dtype='int16',
                device=None, latency='low', blocksize=2048,
            ) as stream:
                while True:
                    if self.interrupted.is_set():
                        break
                    data = self.current_audio_process.stdout.read(4096)
                    if not data:
                        break
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        self.current_volume = np.max(np.abs(audio_chunk))
                        if use_native_rate:
                            n = int(len(audio_chunk) * (native_rate / PIPER_RATE))
                            audio_chunk = scipy.signal.resample(audio_chunk, n).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
                    else:
                        self.current_volume = 0
                time.sleep(0.5)

        except Exception as e:
            logger.error("Audio Error: %s", e)
        finally:
            self.current_volume = 0
            if self.current_audio_process:
                if self.current_audio_process.stdout:
                    self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None:
                    self.current_audio_process.terminate()
                self.current_audio_process = None

    # =========================================================================
    # SOUND EFFECTS
    # =========================================================================

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set():
                    return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except Exception:
                native_rate = 48000

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except Exception:
                playback_rate = native_rate
                n = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, n).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait()
        except Exception:
            pass

    # =========================================================================
    # MEMORY
    # =========================================================================

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        if not USE_LEGACY_PIPELINE:
            # Save SessionManager history
            history = self.session_manager.get_history()
            full = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        else:
            full = self.permanent_memory + self.session_memory

        conv = full[1:]
        if len(conv) > 10:
            conv = conv[-10:]
        with open(MEMORY_FILE, "w") as f:
            json.dump([full[0]] + conv, f, indent=4)


# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    logger.info("--- SYSTEM STARTING (pipeline=%s) ---",
                "legacy" if USE_LEGACY_PIPELINE else "new")
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
