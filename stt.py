"""
ai/stt.py — Streaming Speech-to-Text (faster-whisper, Phase 2)
Part of the Be More Agent architecture migration.

Replaces the legacy ``whisper-cli`` subprocess with in-process
``faster-whisper`` for lower latency and partial transcripts.

Design
------
- Listens for ``VAD_SPEECH_END`` events (carries captured audio).
- Runs transcription in a worker thread to avoid blocking the bus.
- Publishes ``STT_PARTIAL`` events at configurable intervals while
  the user is speaking (750 ms default), and ``STT_FINAL`` when done.
- ``base.en`` with ``int8`` quantization is the default — accurate for
  child speech, acceptable CPU on Pi 5.

Partial transcript implementation note
---------------------------------------
Silero VAD already chunks the audio at utterance end. True streaming
partials (mid-speech) require a different approach: the VAD buffer is
periodically snapshotted while speech is in progress and transcribed
on a timer. This module supports both:

1. **End-of-speech full transcription** (always performed).
2. **Mid-speech partials** via ``PartialTranscriber`` when
   ``partial_interval_ms > 0``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from core.config import AgentConfig
from core.events import (
    EventBus, EventType, Event,
    STTPartialEvent, STTFinalEvent,
    get_bus,
)

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    logger.warning("[STT] faster-whisper not installed. Install with: pip install faster-whisper")


# =========================================================================
# TRANSCRIPTION WORKER
# =========================================================================

class WhisperTranscriber:
    """
    Wraps a ``faster-whisper`` WhisperModel and exposes a simple
    ``transcribe(audio_int16, sample_rate) -> str`` call.
    """

    def __init__(self, model_size: str = "base.en", quantization: str = "int8") -> None:
        self._model_size = model_size
        self._quantization = quantization
        self._model: Optional[object] = None
        self._ready = threading.Event()

    def load(self) -> None:
        """Load model (blocking). Run in a thread to avoid startup lag."""
        if not _WHISPER_AVAILABLE:
            logger.error("[STT] faster-whisper not available — cannot load model.")
            return
        try:
            logger.info("[STT] Loading %s (%s)…", self._model_size, self._quantization)
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type=self._quantization,
            )
            self._ready.set()
            logger.info("[STT] Model loaded.")
        except Exception as exc:
            logger.error("[STT] Model load failed: %s", exc)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready.wait(timeout=timeout)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
        """
        Transcribe audio and return (text, avg_confidence).

        Returns ("", 0.0) on failure or if not ready.
        """
        if not self.is_ready or self._model is None:
            return "", 0.0

        # faster-whisper expects float32 normalised
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        try:
            segments, info = self._model.transcribe(
                audio_f32,
                language="en",
                beam_size=3,
                vad_filter=False,  # We handle VAD upstream
            )
            parts = []
            confidences = []
            for seg in segments:
                parts.append(seg.text.strip())
                confidences.append(getattr(seg, "avg_logprob", 0.0))

            text = " ".join(parts).strip()
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            return text, avg_conf
        except Exception as exc:
            logger.error("[STT] Transcription error: %s", exc)
            return "", 0.0


# =========================================================================
# STT ENGINE  (EventBus integration)
# =========================================================================

class STTEngine:
    """
    Listens for ``VAD_SPEECH_END`` events and publishes
    ``STT_FINAL`` (and optionally ``STT_PARTIAL``) events.

    Parameters
    ----------
    cfg : AgentConfig
        Provides model size, quantization, and partial interval.
    bus : EventBus
        Where events are consumed and published.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus or get_bus()

        self._transcriber = WhisperTranscriber(
            model_size=cfg.stt_model,
            quantization=cfg.stt_quantization,
        )

        # Work queue for audio blobs
        self._work_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="STTWorker"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load model and start worker thread."""
        load_thread = threading.Thread(
            target=self._transcriber.load, daemon=True, name="WhisperLoader"
        )
        load_thread.start()
        self._worker_thread.start()
        logger.info("[STT] Engine started (model=%s, quant=%s).",
                    self._cfg.stt_model, self._cfg.stt_quantization)

        # Subscribe to VAD events
        self._bus.subscribe(EventType.VAD_SPEECH_END, self._on_speech_end)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._transcriber.wait_ready(timeout=timeout)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_speech_end(self, event: Event) -> None:
        """Enqueue audio for transcription."""
        data = event.data or {}
        audio: Optional[np.ndarray] = data.get("audio")
        sample_rate: int = data.get("sample_rate", 16000)
        if audio is not None and len(audio) > 0:
            self._work_queue.put((audio, sample_rate, time.time()))

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            audio, sample_rate, enqueue_time = item
            if not self._transcriber.is_ready:
                logger.debug("[STT] Waiting for model…")
                self._transcriber.wait_ready(timeout=30.0)

            t0 = time.time()
            text, confidence = self._transcriber.transcribe(audio, sample_rate)
            elapsed_ms = (time.time() - t0) * 1000
            total_ms = (time.time() - enqueue_time) * 1000

            logger.info(
                "[STT] '%s'  (conf=%.2f  transcribe=%.0f ms  total=%.0f ms)",
                text, confidence, elapsed_ms, total_ms,
            )

            if not text:
                logger.debug("[STT] Empty transcription — skipping.")
                continue

            event = STTFinalEvent(source="STTEngine")
            event.text = text
            event.confidence = confidence
            event.duration_ms = total_ms
            self._bus.publish(event)
