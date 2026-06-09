"""
audio/tts_engine.py — Persistent Piper TTS Engine (Phase 3)
Part of the Be More Agent architecture migration.

Architecture
------------
Instead of spawning a new Piper process per sentence, a single persistent
process is started at initialisation. Sentences are fed line-by-line to
stdin, and audio is read continuously from stdout.

Flow:
    LLM tokens
      → SentenceBuilder (yields complete sentences)
      → PiperManager.speak(sentence)
      → PersistentPiperProcess  (stdin → stdout)
      → sounddevice playback

Key improvements over legacy speak():
- No ~50-100ms process spawn overhead per sentence.
- Seamless audio stitching between sentences.
- On INTERRUPTED, process is terminated and immediately respawned.
- Publishes TTS_SPEAKING_START / TTS_SPEAKING_END events.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from core.config import AgentConfig
from core.events import EventBus, EventType, Event, get_bus
from core.state_manager import StateManager, AssistantState

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    import scipy.signal
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


# =========================================================================
# SENTENCE BUILDER
# =========================================================================

class SentenceBuilder:
    """
    Accumulates LLM token chunks and yields complete sentences.

    A sentence boundary is any of: . ! ? \\n
    """

    PUNCT = frozenset(".!?\n")

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> list[str]:
        """Add a token. Returns a list of complete sentences (may be empty)."""
        self._buf += token
        sentences = []
        while True:
            for i, ch in enumerate(self._buf):
                if ch in self.PUNCT:
                    sentence = self._buf[: i + 1].strip()
                    self._buf = self._buf[i + 1:]
                    if sentence and re.search(r"[a-zA-Z0-9]", sentence):
                        sentences.append(sentence)
                    break
            else:
                break
        return sentences

    def flush(self) -> Optional[str]:
        """Flush any remaining partial sentence at stream end."""
        remaining = self._buf.strip()
        self._buf = ""
        if remaining and re.search(r"[a-zA-Z0-9]", remaining):
            return remaining
        return None


# =========================================================================
# PERSISTENT PIPER PROCESS
# =========================================================================

class PersistentPiperProcess:
    """
    Wraps a long-running Piper subprocess.

    - Lines sent to stdin → audio bytes on stdout.
    - Reads stdout in a daemon thread, pushing chunks into an audio queue.
    - Thread-safe: all public methods are safe to call from any thread.
    """

    def __init__(
        self,
        binary: str,
        model: str,
        audio_queue: queue.Queue,
    ) -> None:
        self._binary = binary
        self._model = model
        self._audio_queue = audio_queue
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._alive = threading.Event()

    def spawn(self) -> None:
        """Start the Piper subprocess."""
        if self._proc and self._proc.poll() is None:
            return  # Already running

        cmd = [self._binary, "--model", self._model, "--output-raw"]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._alive.set()
            self._reader_thread = threading.Thread(
                target=self._read_stdout, daemon=True, name="PiperReader"
            )
            self._reader_thread.start()
            logger.info("[Piper] Process spawned (pid=%d).", self._proc.pid)
        except FileNotFoundError:
            logger.error("[Piper] Binary not found: %s", self._binary)
            raise

    def send(self, text: str) -> None:
        """Write a sentence to Piper's stdin."""
        if not self._proc or self._proc.poll() is not None:
            logger.warning("[Piper] Process not running — cannot send text.")
            return
        clean = re.sub(r"[^\w\s,.!?:-]", "", text).strip()
        if not clean:
            return
        try:
            self._proc.stdin.write((clean + "\n").encode())
            self._proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("[Piper] Broken pipe on stdin.")

    def terminate(self) -> None:
        """Kill the process and clear the queue."""
        self._alive.clear()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                pass
            self._proc = None
        # Drain any leftover audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("[Piper] Process terminated.")

    def _read_stdout(self) -> None:
        """Background thread: reads raw audio bytes from stdout into queue."""
        CHUNK_BYTES = 4096
        while self._alive.is_set():
            try:
                data = self._proc.stdout.read(CHUNK_BYTES)
                if not data:
                    break
                self._audio_queue.put(data)
            except Exception as exc:
                logger.debug("[Piper] stdout read error: %s", exc)
                break
        self._alive.clear()
        self._audio_queue.put(None)  # Sentinel: stream ended


# =========================================================================
# PIPER MANAGER  (public interface)
# =========================================================================

class PiperManager:
    """
    Thread-safe TTS interface backed by a PersistentPiperProcess.

    Usage
    -----
    ::

        mgr = PiperManager(cfg, state_manager, bus)
        mgr.start()
        mgr.speak("Hello!")
        mgr.speak("How are you?")
        mgr.stop()

    On INTERRUPTED state, call :meth:`interrupt` to terminate and
    immediately respawn Piper for the next turn.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        state_manager: Optional[StateManager] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._cfg = cfg
        self._sm = state_manager
        self._bus = bus or get_bus()

        self._audio_queue: queue.Queue = queue.Queue(maxsize=32)
        self._piper = PersistentPiperProcess(
            binary=cfg.piper_binary,
            model=cfg.voice_model,
            audio_queue=self._audio_queue,
        )

        self._playback_thread: Optional[threading.Thread] = None
        self._interrupted = threading.Event()
        self._lock = threading.Lock()

        # Subscribe to BARGE_IN so we can self-interrupt
        if self._bus:
            self._bus.subscribe(EventType.BARGE_IN_DETECTED, self._on_barge_in)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn Piper and start the playback thread."""
        try:
            self._piper.spawn()
        except FileNotFoundError:
            logger.error("[PiperManager] Cannot start — binary missing: %s", self._cfg.piper_binary)
            return

        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="PiperPlayback"
        )
        self._playback_thread.start()
        logger.info("[PiperManager] Ready.")

    def stop(self) -> None:
        """Graceful shutdown."""
        self._interrupted.set()
        self._piper.terminate()
        logger.info("[PiperManager] Stopped.")

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Queue text for speech. Returns immediately; audio plays async.
        """
        if self._interrupted.is_set():
            logger.debug("[PiperManager] Interrupted — ignoring speak().")
            return
        with self._lock:
            self._piper.send(text)

    def interrupt(self) -> None:
        """Interrupt current speech, flush queues, respawn Piper."""
        logger.info("[PiperManager] Interrupting…")
        self._interrupted.set()
        with self._lock:
            self._piper.terminate()
            time.sleep(0.05)
            self._interrupted.clear()
            try:
                self._piper.spawn()
            except Exception as exc:
                logger.error("[PiperManager] Respawn failed: %s", exc)

    # ------------------------------------------------------------------
    # Playback loop
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """Reads raw int16 audio from queue and plays via sounddevice."""
        if not _SD_AVAILABLE:
            logger.warning("[PiperManager] sounddevice not available — no audio output.")
            return

        piper_rate = self._cfg.piper_rate

        # Determine output device's native rate
        try:
            dev_info = sd.query_devices(kind="output")
            native_rate = int(dev_info["default_samplerate"])
        except Exception:
            native_rate = 48000
        need_resample = (native_rate != piper_rate)

        try:
            with sd.RawOutputStream(
                samplerate=native_rate if need_resample else piper_rate,
                channels=1,
                dtype="int16",
                device=None,
                latency="low",
                blocksize=2048,
            ) as stream:
                speaking = False
                while True:
                    try:
                        chunk = self._audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if chunk is None:
                        # Sentinel — stream ended
                        if speaking:
                            self._bus.publish(
                                Event(type=EventType.TTS_SPEAKING_END, source="PiperManager")
                            )
                            speaking = False
                        continue

                    if self._interrupted.is_set():
                        continue  # Drain silently

                    audio = np.frombuffer(chunk, dtype=np.int16)
                    if len(audio) == 0:
                        continue

                    if not speaking:
                        speaking = True
                        self._bus.publish(
                            Event(type=EventType.TTS_SPEAKING_START, source="PiperManager")
                        )

                    if need_resample:
                        n = int(len(audio) * native_rate / piper_rate)
                        audio = scipy.signal.resample(audio, n).astype(np.int16)

                    stream.write(audio.tobytes())

        except Exception as exc:
            logger.error("[PiperManager] Playback error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_barge_in(self, event: Event) -> None:
        self.interrupt()
