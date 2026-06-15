"""Centralized automation configuration."""

import os

# ── Cooldowns ────────────────────────────────────────────────────
RETRAIN_COOLDOWN_HOURS = float(os.getenv("RETRAIN_COOLDOWN_HOURS", "1"))
ACTION_COOLDOWN_MINUTES = int(os.getenv("ACTION_COOLDOWN_MINUTES", "0"))

# ── LLM ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── Policy ───────────────────────────────────────────────────────
WHITELISTED_ACTIONS = {"retrain", "rollback", "alert", "scale", "notify", "none"}
