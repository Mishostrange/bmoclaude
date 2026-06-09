"""
core/session_manager.py — Session Lifecycle & Statistics Tracking
Part of the Be More Agent architecture migration (Phase 0).

Manages the lifecycle of a single interaction session, tracking conversation
history, timing, interrupts, and integration with the memory layer.
"""

from __future__ import annotations

import time
import threading
import logging
from typing import Any, Dict, List, Optional

from core.events import EventBus, EventType, Event, get_bus

logger = logging.getLogger(__name__)


# =========================================================================
# SESSION DATA
# =========================================================================

class SessionStats:
    """Mutable statistics for the current session."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.interaction_count: int = 0      # User utterances processed
        self.interrupt_count: int = 0        # Barge-ins triggered
        self.error_count: int = 0
        self.total_tokens_generated: int = 0
        self.last_activity_time: float = time.time()

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity_time

    def touch(self) -> None:
        self.last_activity_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.elapsed_seconds, 1),
            "interactions": self.interaction_count,
            "interrupts": self.interrupt_count,
            "errors": self.error_count,
            "tokens_generated": self.total_tokens_generated,
        }


# =========================================================================
# SESSION MANAGER
# =========================================================================

class SessionManager:
    """
    Manages the lifecycle of a single interaction session.

    Responsibilities
    ----------------
    - Tracks conversation history (short-term context window).
    - Maintains session statistics.
    - Publishes SESSION_STARTED / SESSION_ENDED events.
    - Integrates with EventBus to auto-update stats on key events.

    Usage
    -----
    ::

        sm = SessionManager(bus=get_bus(), max_history=20)
        sm.start_session()

        sm.add_user_message("Hello!")
        sm.add_assistant_message("Hi there!")

        summary = sm.get_session_summary()
        sm.end_session()
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        max_history: int = 20,
        system_prompt: str = "",
    ) -> None:
        self._bus = bus or get_bus()
        self._lock = threading.RLock()

        self.max_history = max_history
        self.system_prompt = system_prompt

        # Conversation history: list of {"role": ..., "content": ...}
        self._history: List[Dict[str, str]] = []
        self._stats = SessionStats()
        self._active = False

        # Wire event listeners
        self._bus.subscribe(EventType.BARGE_IN_DETECTED, self._on_interrupt)
        self._bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)
        self._bus.subscribe(EventType.STT_FINAL, self._on_stt_final)
        self._bus.subscribe(EventType.LLM_RESPONSE_COMPLETE, self._on_llm_done)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        with self._lock:
            if self._active:
                logger.warning("[SessionManager] start_session() called while already active.")
                return
            self._stats = SessionStats()
            self._active = True
            logger.info("[SessionManager] Session started.")

        self._bus.publish(Event(type=EventType.SESSION_STARTED, source="SessionManager"))

    def end_session(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._stats.end_time = time.time()
            self._active = False
            summary = self._stats.to_dict()
            logger.info("[SessionManager] Session ended: %s", summary)

        self._bus.publish(
            Event(type=EventType.SESSION_ENDED, data=summary, source="SessionManager")
        )

    def reset_session(self) -> None:
        """Clear history and stats, keeping the session active."""
        with self._lock:
            self._history.clear()
            self._stats = SessionStats()
        self._bus.publish(Event(type=EventType.MEMORY_RESET, source="SessionManager"))
        logger.info("[SessionManager] Session reset.")

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._add_message("user", content)
        with self._lock:
            self._stats.interaction_count += 1
            self._stats.touch()

    def add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)
        with self._lock:
            self._stats.touch()

    def _add_message(self, role: str, content: str) -> None:
        with self._lock:
            self._history.append({"role": role, "content": content})
            # Trim to window, always preserving system prompt sentinel
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Return the message list suitable for passing to an LLM."""
        with self._lock:
            history = list(self._history)

        if include_system and self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}] + history
        return history

    def get_history(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_session_summary(self) -> Dict[str, Any]:
        with self._lock:
            return self._stats.to_dict()

    def record_tokens(self, count: int) -> None:
        with self._lock:
            self._stats.total_tokens_generated += count

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_interrupt(self, event: Event) -> None:
        with self._lock:
            self._stats.interrupt_count += 1

    def _on_error(self, event: Event) -> None:
        with self._lock:
            self._stats.error_count += 1

    def _on_stt_final(self, event: Event) -> None:
        # Auto-log final transcripts if event carries text
        text = getattr(event, "text", None)
        if text:
            self.add_user_message(text)

    def _on_llm_done(self, event: Event) -> None:
        self._stats.touch()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<SessionManager active={self._active} "
                f"interactions={self._stats.interaction_count} "
                f"history_len={len(self._history)}>"
            )
