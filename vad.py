"""
audio/vad.py — Voice Activity Detection (Silero VAD)
Part of the Be More Agent architecture migration (Phase 1).

Wraps Silero VAD to detect speech start/end events and publish them
on the EventBus. Also maintains a rolling audio buffer for STT handoff.

Silero VAD expects:
- 16 kHz or 8 kHz audio
- 512-sample chunks at 16 kHz (32 ms)
- float32 normalised to [-1, 1]
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable, Deque, List, Optional

import numpy as np

from core.events import EventBus, EventType, Event, AudioChunkEvent, get_bus

logger = logging.getLogger(__name__)

# Silero chunk size at 16 kHz
SILERO_CHUNK = 512          # ~32 ms
SILERO_RATE  = 16_000

SPEECH_THRESHOLD = 0.5      # Probability above which speech is detected
SILENCE_THRESHOLD = 0.35    # Below this = silence
SPEECH_PAD_CHUNKS = 8       # Pad before/after speech boundary (~256 ms)
MIN_SPEECH_CHUNKS = 4       # Ignore very short blips (<128 ms)
MAX_BUFFER_S = 30.0         # Hard cap


def _load_silero_vad():
    """Lazy-load Silero VAD via torch.hub (cached after first download)."""
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils
        return model, get_speech_timestamps
    except Exception as exc:
        logger.warning("[VAD] Could not load Silero: %s — VAD disabled.", exc)
        return None, None


class VoiceActivityDetector:
    """
    Consumes 16 kHz int16 audio chunks from the microphone, detects
    speech boundaries, and publishes:

    - ``VAD_SPEECH_START``  — when speech begins
    - ``VAD_SPEECH_END``    — when silence follows speech; event.data is a
                             dict with ``audio`` (np.ndarray int16) and
                             ``sample_rate`` (int).

    Parameters
    ----------
    bus :
        EventBus to publish on (defaults to global bus).
    speech_threshold :
        Silero probability above which the chunk is "speech".
    pad_chunks :
        Number of chunks to pad before/after boundaries.
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        speech_threshold: float = SPEECH_THRESHOLD,
        silence_threshold: float = SILENCE_THRESHOLD,
        pad_chunks: int = SPEECH_PAD_CHUNKS,
        min_speech_chunks: int = MIN_SPEECH_CHUNKS,
        max_buffer_s: float = MAX_BUFFER_S,
    ) -> None:
        self._bus = bus or get_bus()
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.pad_chunks = pad_chunks
        self.min_speech_chunks = min_speech_chunks
        self.max_buffer_s = max_buffer_s

        self._model = None
        self._available = False
        self._lock = threading.Lock()

        # Rolling pre-roll buffer (keeps last N chunks before speech)
        self._pre_buffer: Deque[np.ndarray] = collections.deque(maxlen=pad_chunks)

        # Active speech buffer
        self._speech_buffer: List[np.ndarray] = []
        self._in_speech = False
        self._silence_chunks = 0
        self._speech_chunks = 0

        # Max silence chunks before we declare end of utterance (~800 ms)
        self._max_silence_chunks = int(0.8 * SILERO_RATE / SILERO_CHUNK)

        self._model_load_thread = threading.Thread(
            target=self._load_model, daemon=True, name="SileroLoader"
        )
        self._model_load_thread.start()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model, _ = _load_silero_vad()
        if model is not None:
            self._model = model
            self._available = True
            logger.info("[VAD] Silero VAD loaded.")
        else:
            logger.warning("[VAD] Running without VAD — energy fallback active.")

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """Block until model is loaded or timeout expires."""
        start = time.time()
        while not self._available and time.time() - start < timeout:
            time.sleep(0.1)
        return self._available

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_chunk(self, audio_int16: np.ndarray, sample_rate: int = SILERO_RATE) -> None:
        """
        Feed a 16 kHz int16 chunk. Safe to call from any thread.
        Chunks smaller than SILERO_CHUNK are buffered internally.
        """
        if sample_rate != SILERO_RATE:
            logger.debug("[VAD] Expected %d Hz, got %d Hz — skipping chunk.", SILERO_RATE, sample_rate)
            return

        # Silero needs exact 512-sample chunks; split/buffer accordingly
        with self._lock:
            self._pending = getattr(self, "_pending", np.array([], dtype=np.int16))
            combined = np.concatenate([self._pending, audio_int16])
            while len(combined) >= SILERO_CHUNK:
                chunk = combined[:SILERO_CHUNK]
                combined = combined[SILERO_CHUNK:]
                self._process_silero_chunk(chunk)
            self._pending = combined

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_silero_chunk(self, chunk: np.ndarray) -> None:
        """Process a single exactly-SILERO_CHUNK-sized chunk."""
        prob = self._get_speech_prob(chunk)

        if prob >= self.speech_threshold:
            # Speech detected
            if not self._in_speech:
                # Transition: silence → speech
                self._in_speech = True
                self._speech_chunks = 0
                self._silence_chunks = 0
                # Prepend pre-roll for natural boundary
                self._speech_buffer = list(self._pre_buffer)
                self._bus.publish(Event(type=EventType.VAD_SPEECH_START, source="VAD"))
                logger.debug("[VAD] Speech START  (prob=%.2f)", prob)

            self._speech_buffer.append(chunk)
            self._speech_chunks += 1
            self._silence_chunks = 0

        else:
            # Silence
            if self._in_speech:
                self._speech_buffer.append(chunk)
                self._silence_chunks += 1

                # Check max buffer length
                total_s = len(self._speech_buffer) * SILERO_CHUNK / SILERO_RATE
                if total_s >= self.max_buffer_s:
                    logger.warning("[VAD] Max buffer length reached — forcing end.")
                    self._finalize_speech()
                    return

                if self._silence_chunks >= self._max_silence_chunks:
                    if self._speech_chunks >= self.min_speech_chunks:
                        self._finalize_speech()
                    else:
                        # Too short — discard
                        logger.debug("[VAD] Discarding short blip (%d chunks).", self._speech_chunks)
                        self._reset_speech()
            else:
                # Pre-speech — rolling pre-roll
                self._pre_buffer.append(chunk)

    def _finalize_speech(self) -> None:
        """Package buffer and emit VAD_SPEECH_END."""
        audio = np.concatenate(self._speech_buffer).astype(np.int16)
        duration_s = len(audio) / SILERO_RATE
        logger.debug("[VAD] Speech END — %.2f s (%d chunks)", duration_s, self._speech_chunks)

        event = Event(
            type=EventType.VAD_SPEECH_END,
            data={"audio": audio, "sample_rate": SILERO_RATE, "duration_s": duration_s},
            source="VAD",
        )
        self._bus.publish(event)
        self._reset_speech()

    def _reset_speech(self) -> None:
        self._in_speech = False
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._speech_buffer = []

    def _get_speech_prob(self, chunk: np.ndarray) -> float:
        """Return speech probability (0–1). Falls back to energy if no model."""
        if self._available and self._model is not None:
            try:
                import torch
                tensor = torch.from_numpy(chunk.astype(np.float32) / 32768.0)
                prob = float(self._model(tensor, SILERO_RATE).item())
                return prob
            except Exception as exc:
                logger.debug("[VAD] Silero predict failed: %s", exc)

        # Energy fallback
        energy = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return min(energy / 500.0, 1.0)   # Rough normalisation
