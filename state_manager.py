"""
core/state_manager.py — Formal State Machine
Part of the Be More Agent architecture migration (Phase 0).

Enforces valid state transitions, emits StateChangedEvent on the EventBus,
and provides thread-safe access to the current state.
"""

from __future__ import annotations

import threading
import logging
from enum import Enum
from typing import Optional, Set, Dict

from core.events import EventBus, EventType, StateChangedEvent, get_bus

logger = logging.getLogger(__name__)


# =========================================================================
# STATE ENUM
# =========================================================================

class AssistantState(Enum):
    IDLE          = "idle"
    LISTENING     = "listening"
    PROCESSING    = "processing"
    SPEAKING      = "speaking"
    INTERRUPTED   = "interrupted"
    ERROR         = "error"
    SHUTTING_DOWN = "shutting_down"


# =========================================================================
# VALID TRANSITIONS
# =========================================================================
#
# Defined as: { FROM_STATE: {set of allowed TO_STATES} }
#
VALID_TRANSITIONS: Dict[AssistantState, Set[AssistantState]] = {
    AssistantState.IDLE: {
        AssistantState.LISTENING,
        AssistantState.ERROR,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.LISTENING: {
        AssistantState.PROCESSING,
        AssistantState.IDLE,           # Timeout / nothing heard
        AssistantState.ERROR,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.PROCESSING: {
        AssistantState.SPEAKING,
        AssistantState.IDLE,           # Empty / error response
        AssistantState.ERROR,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,           # TTS complete
        AssistantState.INTERRUPTED,    # Barge-in
        AssistantState.ERROR,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.INTERRUPTED: {
        AssistantState.LISTENING,      # Reset for next turn
        AssistantState.IDLE,
        AssistantState.ERROR,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.ERROR: {
        AssistantState.IDLE,           # Recovery
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.SHUTTING_DOWN: set(),  # Terminal
}


# =========================================================================
# STATE MANAGER
# =========================================================================

class StateManager:
    """
    Thread-safe state machine for the assistant.

    Usage::

        sm = StateManager(bus=get_bus())
        sm.transition(AssistantState.LISTENING, reason="wake word")
        print(sm.current)   # AssistantState.LISTENING
    """

    def __init__(
        self,
        initial: AssistantState = AssistantState.IDLE,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._bus = bus or get_bus()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current(self) -> AssistantState:
        with self._lock:
            return self._state

    def is_in(self, *states: AssistantState) -> bool:
        return self.current in states

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        new_state: AssistantState,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """
        Attempt to transition to `new_state`.

        Returns True if the transition succeeded, False if it was
        rejected as invalid (unless `force=True`).
        """
        with self._lock:
            previous = self._state

            if previous == new_state:
                return True  # No-op, already there

            allowed = VALID_TRANSITIONS.get(previous, set())

            if new_state not in allowed and not force:
                logger.warning(
                    "[StateManager] REJECTED %s → %s (reason: %s)",
                    previous.value, new_state.value, reason or "none",
                )
                return False

            self._state = new_state

        # Emit event *outside* the lock to avoid re-entrant deadlocks
        label = f" ({reason})" if reason else ""
        logger.info(
            "[StateManager] %s → %s%s",
            previous.value, new_state.value, label,
        )

        event = StateChangedEvent(
            data={"reason": reason},
            source="StateManager",
        )
        event.previous = previous
        event.current = new_state
        self._bus.publish(event)
        return True

    def force_transition(self, new_state: AssistantState, reason: str = "") -> None:
        """Bypass validation — use only for error recovery or shutdown."""
        self.transition(new_state, reason=reason, force=True)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def go_idle(self, reason: str = "") -> bool:
        return self.transition(AssistantState.IDLE, reason=reason)

    def go_listening(self, reason: str = "") -> bool:
        return self.transition(AssistantState.LISTENING, reason=reason)

    def go_processing(self, reason: str = "") -> bool:
        return self.transition(AssistantState.PROCESSING, reason=reason)

    def go_speaking(self, reason: str = "") -> bool:
        return self.transition(AssistantState.SPEAKING, reason=reason)

    def go_interrupted(self, reason: str = "") -> bool:
        return self.transition(AssistantState.INTERRUPTED, reason=reason)

    def go_error(self, reason: str = "") -> bool:
        return self.transition(AssistantState.ERROR, reason=reason, force=True)

    def go_shutdown(self, reason: str = "") -> bool:
        return self.transition(AssistantState.SHUTTING_DOWN, reason=reason, force=True)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<StateManager state={self._state.value}>"
