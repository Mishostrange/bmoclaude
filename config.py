"""
core/config.py — Centralized Configuration
Part of the Be More Agent architecture migration (Phase 0).

Single source of truth for all configuration values.
Reads from config.json and environment, with documented defaults.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"


# =========================================================================
# CONFIG DATACLASS
# =========================================================================

@dataclass
class AgentConfig:
    # ── Models ────────────────────────────────────────────────────────────
    text_model: str = "gemma3:1b"
    vision_model: str = "moondream"
    voice_model: str = "piper/en_GB-semaine-medium.onnx"

    # ── Wake word ─────────────────────────────────────────────────────────
    wake_word_model: str = "./wakeword.onnx"
    wake_word_threshold: float = 0.5

    # ── Audio hardware ────────────────────────────────────────────────────
    input_device: Optional[Any] = None     # None = system default
    input_sample_rate: Optional[int] = None

    # ── STT (Phase 2) ─────────────────────────────────────────────────────
    stt_model: str = "base.en"             # faster-whisper model size
    stt_quantization: str = "int8"         # "int8" or "float16"
    stt_partial_interval_ms: int = 750     # Partial transcript cadence (500–1000)
    stt_max_audio_buffer_s: float = 30.0   # Hard cap on audio buffer length

    # ── TTS (Phase 3) ─────────────────────────────────────────────────────
    piper_binary: str = "./piper/piper"
    piper_rate: int = 22050                # Must match voice model's sample_rate

    # ── LLM ───────────────────────────────────────────────────────────────
    ollama_keep_alive: str = "-1"
    ollama_num_threads: int = 4
    ollama_temperature: float = 0.7
    ollama_top_k: int = 40
    ollama_top_p: float = 0.9

    # ── Memory ────────────────────────────────────────────────────────────
    chat_memory: bool = True
    memory_file: str = "memory.json"
    session_max_history: int = 20

    # ── Camera ────────────────────────────────────────────────────────────
    camera_rotation: int = 0

    # ── Prompts ───────────────────────────────────────────────────────────
    system_prompt_extras: str = ""

    # ── Migration flags (Phase 4) ─────────────────────────────────────────
    use_legacy_pipeline: bool = True       # Flip to False to enable new stack

    # ── Sound directories ─────────────────────────────────────────────────
    sounds_greeting_dir: str = "sounds/greeting_sounds"
    sounds_ack_dir: str = "sounds/ack_sounds"
    sounds_thinking_dir: str = "sounds/thinking_sounds"
    sounds_error_dir: str = "sounds/error_sounds"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# LOADER
# =========================================================================

def load_config(path: str = CONFIG_FILE) -> AgentConfig:
    """
    Load configuration from *path* (JSON), falling back to defaults
    for any missing key.  Unknown keys are silently ignored.
    """
    cfg = AgentConfig()

    if not os.path.exists(path):
        logger.warning("[Config] %s not found — using defaults.", path)
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
    except Exception as exc:
        logger.error("[Config] Failed to parse %s: %s — using defaults.", path, exc)
        return cfg

    # Apply known keys only
    known = set(cfg.to_dict().keys())
    for key, value in raw.items():
        if key in known:
            setattr(cfg, key, value)
        else:
            logger.debug("[Config] Unknown key ignored: %s", key)

    logger.info("[Config] Loaded from %s", path)
    return cfg


def save_config(cfg: AgentConfig, path: str = CONFIG_FILE) -> None:
    """Persist config to disk (pretty-printed JSON)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
        logger.info("[Config] Saved to %s", path)
    except Exception as exc:
        logger.error("[Config] Failed to save: %s", exc)


# =========================================================================
# OLLAMA OPTIONS HELPER
# =========================================================================

def get_ollama_options(cfg: AgentConfig) -> Dict[str, Any]:
    return {
        "keep_alive": cfg.ollama_keep_alive,
        "num_thread": cfg.ollama_num_threads,
        "temperature": cfg.ollama_temperature,
        "top_k": cfg.ollama_top_k,
        "top_p": cfg.ollama_top_p,
    }
