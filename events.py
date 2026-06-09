"""
core/events.py — EventBus and Event Dataclasses
Part of the Be More Agent architecture migration (Phase 0).
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =========================================================================
# EVENT TYPES
# =========================================================================

class EventType(Enum):
    # Audio pipeline
    AUDIO_CHUNK           = auto()   # Raw audio chunk from microphone
    VAD_SPEECH_START      = auto()   # VAD detected speech beginning
    VAD_SPEECH_END        = auto()   # VAD detected end of speech
    BARGE_IN_DETECTED     = auto()   # Barge-in triggered during SPEAKING

    # STT
    STT_PARTIAL           = auto()   # Partial transcript (750ms intervals)
    STT_FINAL             = auto()   # Final confirmed transcript

    # LLM
    LLM_TOKEN             = auto()   # Streaming token from LLM
    LLM_SENTENCE_READY    = auto()   # Complete sentence ready for TTS
    LLM_RESPONSE_COMPLETE = auto()   # Full response done

    # TTS
    TTS_SPEAKING_START    = auto()   # Piper started outputting audio
    TTS_SPEAKING_END      = auto()   # Piper finished speaking
    TTS_INTERRUPTED       = auto()   # TTS was cut off

    # State transitions
    STATE_CHANGED         = auto()   # AssistantState changed

    # Session
    SESSION_STARTED       = auto()
    SESSION_ENDED         = auto()
    MEMORY_RESET          = auto()

    # System
    WAKE_WORD_DETECTED    = auto()
    PTT_PRESSED           = auto()
    SHUTDOWN_REQUESTED    = auto()
    ERROR_OCCURRED        = auto()


# =========================================================================
# EVENT DATACLASSES
# =========================================================================

@dataclass
class Event:
    """Base event class."""
    type: EventType
    data: Any = None
    source: str = ""


@dataclass
class AudioChunkEvent(Event):
    type: EventType = field(default=EventType.AUDIO_CHUNK, init=False)
    audio: Optional[Any] = None     # np.ndarray
    sample_rate: int = 16000


@dataclass
class STTPartialEvent(Event):
    type: EventType = field(default=EventType.STT_PARTIAL, init=False)
    text: str = ""
    confidence: float = 0.0


@dataclass
class STTFinalEvent(Event):
    type: EventType = field(default=EventType.STT_FINAL, init=False)
    text: str = ""
    confidence: float = 0.0
    duration_ms: float = 0.0


@dataclass
class LLMTokenEvent(Event):
    type: EventType = field(default=EventType.LLM_TOKEN, init=False)
    token: str = ""


@dataclass
class LLMSentenceEvent(Event):
    type: EventType = field(default=EventType.LLM_SENTENCE_READY, init=False)
    sentence: str = ""


@dataclass
class StateChangedEvent(Event):
    type: EventType = field(default=EventType.STATE_CHANGED, init=False)
    previous: Any = None
    current: Any = None


@dataclass
class ErrorEvent(Event):
    type: EventType = field(default=EventType.ERROR_OCCURRED, init=False)
    error: Optional[Exception] = None
    message: str = ""
    recoverable: bool = True


# =========================================================================
# EVENT BUS
# =========================================================================

Handler = Callable[[Event], None]


class EventBus:
    """
    Thread-safe publish/subscribe event bus.

    Handlers are called synchronously in the publishing thread unless
    `async_dispatch=True`, in which case each handler runs in its own
    daemon thread. Use async dispatch for heavy handlers (e.g. TTS).
    """

    def __init__(self, async_dispatch: bool = False):
        self._subscribers: Dict[EventType, List[Handler]] = {}
        self._global_subscribers: List[Handler] = []
        self._lock = threading.Lock()
        self._async_dispatch = async_dispatch

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Subscribe handler to a specific event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe handler to ALL events (useful for logging/debug)."""
        with self._lock:
            self._global_subscribers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Dispatch event to all registered handlers."""
        with self._lock:
            specific = list(self._subscribers.get(event.type, []))
            global_h = list(self._global_subscribers)

        all_handlers = specific + global_h
        if not all_handlers:
            return

        if self._async_dispatch:
            for handler in all_handlers:
                t = threading.Thread(
                    target=self._safe_call, args=(handler, event), daemon=True
                )
                t.start()
        else:
            for handler in all_handlers:
                self._safe_call(handler, event)

    def _safe_call(self, handler: Handler, event: Event) -> None:
        try:
            handler(event)
        except Exception as exc:
            logger.error(
                "[EventBus] Handler %s raised for %s: %s",
                handler.__qualname__, event.type, exc,
                exc_info=True,
            )


# =========================================================================
# MODULE-LEVEL SINGLETON (optional convenience)
# =========================================================================

_default_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """Return (and lazily create) the module-level default EventBus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus(async_dispatch=True)
    return _default_bus
