"""
audio/microphone.py — Continuous Microphone Capture Module
Part of the Be More Agent architecture migration (Phase 1).

Replaces the inline record_voice_adaptive() / record_voice_ptt() functions
with a continuous stream that publishes AudioChunkEvents to the EventBus.

Design
------
- Runs in a dedicated daemon thread.
- Auto-detects the device's native sample rate (mirrors legacy helper).
- Publishes 20ms chunks at 16000 Hz to the bus.
- Resamples on the fly (nearest-neighbour, CPU-friendly on Pi 5).
- Thread-safe start/stop control.
"""

from __future__ import annotations

import threading
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — only needed when the new pipeline is active
try:
    import sounddevice as sd
    import scipy.signal as signal
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    logger.warning("[Microphone] sounddevice not available.")


TARGET_RATE = 16000   # Rate the rest of the pipeline expects
CHUNK_MS    = 20      # 20 ms chunks → 320 samples @ 16 kHz


def _choose_sample_rate(device, preferred: Optional[int] = None) -> int:
    """Mirror of legacy choose_input_samplerate() — picks highest-compat rate."""
    if not _SD_AVAILABLE:
        return TARGET_RATE

    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        info = sd.query_devices(device)
        if "default_samplerate" in info:
            candidates.append(int(info["default_samplerate"]))
    except Exception:
        pass
    candidates.extend([48000, 44100, 32000, 16000])

    seen: set = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    return TARGET_RATE


class MicrophoneStream:
    """
    Continuous microphone stream that delivers 16 kHz int16 audio chunks
    via a callback.

    Parameters
    ----------
    device :
        PortAudio device index / name, or None for system default.
    preferred_rate :
        Hint for sample rate selection (passed to _choose_sample_rate).
    chunk_ms :
        Duration of each audio chunk in milliseconds.
    on_chunk :
        Callable(audio_int16: np.ndarray, sample_rate: int) — called for
        every chunk.  Runs in the sounddevice callback thread; keep it fast
        or hand off to a queue.
    """

    def __init__(
        self,
        device=None,
        preferred_rate: Optional[int] = None,
        chunk_ms: int = CHUNK_MS,
        on_chunk=None,
    ) -> None:
        self.device = device
        self.preferred_rate = preferred_rate
        self.chunk_ms = chunk_ms
        self.on_chunk = on_chunk

        self._stream: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # Resolved at start()
        self._input_rate: int = TARGET_RATE
        self._input_chunk_size: int = 320

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not _SD_AVAILABLE:
            raise RuntimeError("sounddevice is not installed.")
        if self._running.is_set():
            logger.warning("[Microphone] Already running.")
            return

        self._input_rate = _choose_sample_rate(self.device, self.preferred_rate)
        target_chunk = int(TARGET_RATE * self.chunk_ms / 1000)
        self._input_chunk_size = int(target_chunk * (self._input_rate / TARGET_RATE))

        logger.info(
            "[Microphone] Starting — device=%s  input_rate=%d  chunk=%d samples",
            self.device, self._input_rate, self._input_chunk_size,
        )

        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="MicrophoneStream"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[Microphone] Stopped.")

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        use_resample = self._input_rate != TARGET_RATE
        target_chunk = int(TARGET_RATE * self.chunk_ms / 1000)

        try:
            with sd.InputStream(
                samplerate=self._input_rate,
                channels=1,
                dtype="int16",
                blocksize=self._input_chunk_size,
                device=self.device,
                latency="low",
            ) as stream:
                logger.info("[Microphone] Stream open.")
                while self._running.is_set():
                    try:
                        data, overflow = stream.read(self._input_chunk_size)
                    except Exception as exc:
                        logger.error("[Microphone] Read error: %s", exc)
                        time.sleep(0.05)
                        continue

                    if overflow:
                        logger.debug("[Microphone] Buffer overflow !")

                    audio = np.frombuffer(data, dtype=np.int16)
                    if audio.ndim > 1:
                        audio = audio.flatten()

                    # Resample to TARGET_RATE using fast nearest-neighbour
                    if use_resample:
                        step = len(audio) / target_chunk
                        indices = np.arange(0, len(audio), step)[:target_chunk].astype(int)
                        audio = audio[indices]

                    if self.on_chunk and len(audio) > 0:
                        try:
                            self.on_chunk(audio, TARGET_RATE)
                        except Exception as exc:
                            logger.error("[Microphone] on_chunk raised: %s", exc)

        except Exception as exc:
            logger.error("[Microphone] Stream failed: %s", exc, exc_info=True)
        finally:
            self._running.clear()
            logger.info("[Microphone] Capture loop exited.")
