"""
therapy/child_profile.py — Child Profile System
Part of the Be More Agent architecture migration (Phase 9).

Provides a structured ChildProfile dataclass that influences:
- Response length limits
- Vocabulary complexity (via system prompt injection)
- Reinforcement style
- Pacing (pause duration between sentences)
- Interaction structure (max choices, predictable loops)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# CHILD PROFILE
# =========================================================================

@dataclass
class ChildProfile:
    """
    Structured profile controlling how the assistant adapts its
    interaction style for a child user.

    All fields have sensible defaults (neutral / medium settings).
    """

    name: str = "child"
    age: int = 8

    verbal_level: Literal["low", "medium", "high"] = "medium"
    """
    - low:    1-sentence responses, <7 words. Very simple vocabulary.
    - medium: 2-3 sentences. Age-appropriate vocabulary.
    - high:   Multi-sentence exploration. Rich vocabulary.
    """

    attention_span: Literal["short", "medium", "long"] = "medium"
    """Controls pacing pauses between TTS sentences."""

    sensory_sensitivity: Literal["low", "medium", "high"] = "low"
    """High sensitivity → slower pacing, softer audio, minimal surprises."""

    preferred_reinforcement: Literal["praise", "sound", "visual"] = "praise"
    """How positive feedback is delivered."""

    communication_style: Literal["structured", "exploratory", "direct"] = "direct"
    """
    - structured:   Max 2 choices per turn. Predictable narrative loops.
    - exploratory:  Open-ended questions. Free-form interaction.
    - direct:       Clear instructions. No ambiguity.
    """

    enabled: bool = False
    """Master switch — set to True to activate profile-based adaptations."""

    # ------------------------------------------------------------------
    # Derived behaviour
    # ------------------------------------------------------------------

    @property
    def max_response_sentences(self) -> int:
        return {"low": 1, "medium": 3, "high": 6}.get(self.verbal_level, 3)

    @property
    def max_words_per_response(self) -> Optional[int]:
        return {"low": 7, "medium": None, "high": None}.get(self.verbal_level)

    @property
    def sentence_pause_ms(self) -> int:
        """Pause inserted between TTS sentences."""
        base = {"short": 600, "medium": 300, "long": 150}.get(self.attention_span, 300)
        sensitivity_mult = {"high": 1.5, "medium": 1.0, "low": 1.0}.get(
            self.sensory_sensitivity, 1.0
        )
        return int(base * sensitivity_mult)

    @property
    def max_choices(self) -> Optional[int]:
        return 2 if self.communication_style == "structured" else None

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    def build_system_prompt_addon(self) -> str:
        """Return additional system prompt text reflecting this profile."""
        if not self.enabled:
            return ""

        parts = [
            f"\n\n### Child Profile: {self.name}, age {self.age} ###",
            f"Verbal level: {self.verbal_level}.",
        ]

        if self.verbal_level == "low":
            parts.append(
                "Keep every response to a single sentence of at most 7 words. "
                "Use only simple words a young child would know."
            )
        elif self.verbal_level == "medium":
            parts.append("Use simple, clear sentences. Avoid jargon or complex vocabulary.")

        if self.communication_style == "structured":
            parts.append(
                "Always give the child exactly 2 choices to pick from. "
                "Follow a predictable, repetitive structure."
            )
        elif self.communication_style == "direct":
            parts.append("Give clear, direct instructions. Avoid ambiguity.")

        if self.preferred_reinforcement == "praise":
            parts.append("Celebrate correct answers with short, enthusiastic praise.")
        elif self.preferred_reinforcement == "sound":
            parts.append("Signal success with a short sound effect cue (e.g., 'ding!').")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChildProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str) -> "ChildProfile":
        if not os.path.exists(path):
            logger.warning("[ChildProfile] %s not found — using defaults.", path)
            return cls()
        try:
            with open(path, "r") as fh:
                return cls.from_dict(json.load(fh))
        except Exception as exc:
            logger.error("[ChildProfile] Load failed: %s", exc)
            return cls()

    def save(self, path: str) -> None:
        try:
            with open(path, "w") as fh:
                json.dump(self.to_dict(), fh, indent=2)
        except Exception as exc:
            logger.error("[ChildProfile] Save failed: %s", exc)
