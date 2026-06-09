"""
audio/audio_pipeline.py — Integrated Audio Input Pipeline
Part of the Be More Agent architecture migration (Phase 1).

Connects: MicrophoneStream → VoiceActivityDetector → EventBus

Usage
-----
::

    pipeline = AudioPipeline(cfg, bus)
    pipeline.start()
    # …
    pipeline.stop()

The pipeline publishes:
- ``VAD_SPEECH_START``
- ``VAD_SPEECH_END``    (carries the captured audio for STT)
"""

from __future__ import annotations

import logging
from typing import Optional

from core.config import AgentConfig
from core.events import EventBus, get_bus
from audio.microphone import MicrophoneStream
from audio.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Manages the microphone stream + VAD and wires them together.

    Parameters
    ----------
    cfg : AgentConfig
        Provides device, sample rate, and VAD settings.
    bus : EventBus
        Where speech events are published.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus or get_bus()

        self._vad = VoiceActivityDetector(bus=self._bus)

        self._mic = MicrophoneStream(
            device=cfg.input_device,
            preferred_rate=cfg.input_sample_rate,
            on_chunk=self._vad.process_chunk,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("[AudioPipeline] Starting…")
        self._vad._model_load_thread.join(timeout=0)  # Non-blocking — loads in background
        self._mic.start()
        logger.info("[AudioPipeline] Running.")

    def stop(self) -> None:
        self._mic.stop()
        logger.info("[AudioPipeline] Stopped.")

    def wait_vad_ready(self, timeout: float = 15.0) -> bool:
        """Block until Silero VAD is loaded (or timeout)."""
        return self._vad.wait_ready(timeout=timeout)

    @property
    def vad(self) -> VoiceActivityDetector:
        return self._vad

    @property
    def mic(self) -> MicrophoneStream:
        return self._mic
