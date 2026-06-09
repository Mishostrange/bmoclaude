"""
audio/barge_in.py — Barge-In Controller
Part of the Be More Agent architecture migration (Phase 5).

Monitors VAD_SPEECH_START events while the assistant is in SPEAKING state.
Applies a decay buffer to avoid echo false-positives, then triggers
BARGE_IN_DETECTED which causes the StateManager to transition to INTERRUPTED.

Key rules (from plan):
- Microphone VAD processing is DISABLED while AssistantState == SPEAKING.
  Instead we gate on a 400 ms echo decay window after TTS starts.
- A single VAD_SPEECH_START that persists for >150 ms triggers barge-in.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from core.events import EventBus, EventType, Event, get_bus
from core.state_manager import StateManager, AssistantState

logger = logging.getLogger(__name__)

ECHO_DECAY_MS   = 400    # Ignore VAD for this long after TTS starts
MIN_SPEECH_MS   = 150    # Voice must persist this long to count as barge-in
DEBOUNCE_WINDOW = 1.0    # Seconds to ignore subsequent barge-ins after one fires


class BargeInController:
    """
    Watches VAD_SPEECH_START events. When the assistant is SPEAKING and the
    echo-decay window has passed, triggers BARGE_IN_DETECTED.

    Parameters
    ----------
    state_manager :
        Used to check current state and trigger INTERRUPTED transition.
    bus :
        EventBus to subscribe/publish on.
    echo_decay_ms :
        How long to ignore the mic after TTS begins (echo suppression).
    min_speech_ms :
        How long speech must persist before barge-in fires.
    """

    def __init__(
        self,
        state_manager: StateManager,
        bus: Optional[EventBus] = None,
        echo_decay_ms: int = ECHO_DECAY_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
    ) -> None:
        self._sm = state_manager
        self._bus = bus or get_bus()
        self._echo_decay_s = echo_decay_ms / 1000.0
        self._min_speech_s = min_speech_ms / 1000.0

        self._tts_start_time: float = 0.0
        self._speech_start_time: float = 0.0
        self._last_barge_time: float = 0.0
        self._pending_barge = False

        self._lock = threading.Lock()
        self._check_thread: Optional[threading.Thread] = None

        # Subscribe
        self._bus.subscribe(EventType.TTS_SPEAKING_START, self._on_tts_start)
        self._bus.subscribe(EventType.VAD_SPEECH_START, self._on_speech_start)
        self._bus.subscribe(EventType.VAD_SPEECH_END, self._on_speech_end)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_tts_start(self, event: Event) -> None:
        with self._lock:
            self._tts_start_time = time.time()
            self._pending_barge = False
        logger.debug("[BargeIn] TTS started — echo gate armed (%.0f ms).", self._echo_decay_s * 1000)

    def _on_speech_start(self, event: Event) -> None:
        if not self._sm.is_in(AssistantState.SPEAKING):
            return

        now = time.time()

        with self._lock:
            # Check echo decay window
            if now - self._tts_start_time < self._echo_decay_s:
                logger.debug("[BargeIn] Speech within echo decay window — ignoring.")
                return

            # Check debounce
            if now - self._last_barge_time < DEBOUNCE_WINDOW:
                logger.debug("[BargeIn] Debounce active — ignoring.")
                return

            self._speech_start_time = now
            self._pending_barge = True

        # Start a timer to confirm the speech persists
        t = threading.Timer(self._min_speech_s, self._confirm_barge)
        t.daemon = True
        t.start()

    def _on_speech_end(self, event: Event) -> None:
        with self._lock:
            self._pending_barge = False

    def _confirm_barge(self) -> None:
        with self._lock:
            if not self._pending_barge:
                return
            if not self._sm.is_in(AssistantState.SPEAKING):
                return
            self._last_barge_time = time.time()
            self._pending_barge = False

        logger.info("[BargeIn] ✂ Barge-in confirmed — interrupting TTS.")
        self._bus.publish(Event(type=EventType.BARGE_IN_DETECTED, source="BargeIn"))
        self._sm.go_interrupted(reason="barge-in")
