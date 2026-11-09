"""Configuration from environment variables only: this project never loads a .env file (SCOPE.md, Engine)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-3.6-flash"  # override with CLIENT_AGENT_MODEL
BRAIN_URL = "http://127.0.0.1:8000"  # Second Brain API, used with --brain

# Input and output (SCOPE.md, Purpose / Web)
MAX_MESSAGE_CHARS = 2000
TOP_K = 4
MAX_REPLY_WORDS = 180
SIGN_OFF = "Grace, TechByGrace"

# Gemini (SCOPE.md, Engine)
GEMINI_TIMEOUT_S = 60.0  # no reply within 60 s counts as a dropped connection
CLI_RETRY_WAITS_S = (30.0, 60.0)  # command line: 503 / 504 / dropped connection retried twice; 429 never
WEB_RETRY_WAITS_S = (10.0,)  # web: one retry after 10 s
MIN_CALL_SPACING_S = 13.0  # between any two outgoing calls (Gemini or brain), shared pacer

BRAIN_TIMEOUT_S = 60.0


class ConfigError(Exception):
    """A required environment variable is missing or empty."""


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    model: str = DEFAULT_MODEL


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    api_key = env.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("GEMINI_API_KEY is not set (this tool does not read .env files).")
    model = env.get("CLIENT_AGENT_MODEL", "").strip() or DEFAULT_MODEL
    return Config(gemini_api_key=api_key, model=model)
