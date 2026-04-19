from __future__ import annotations

import os
from pathlib import Path


def _load_env_file() -> dict[str, str]:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


_ENV_VALUES = _load_env_file()


def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, _ENV_VALUES.get(name, default))


LLM_PROVIDER = _get_env("LLM_PROVIDER", "groq")
GROQ_MODEL = _get_env("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = _get_env("GROQ_API_KEY", "")
GROQ_BASE_URL = _get_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
LLM_TIMEOUT_SECONDS = float(_get_env("LLM_TIMEOUT_SECONDS", "60"))
LLM_TEMPERATURE = float(_get_env("LLM_TEMPERATURE", "0"))
LLM_SEED = int(_get_env("LLM_SEED", "42"))

EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

QDRANT_URL = _get_env("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = _get_env("COLLECTION_NAME", "pdf_docs")

CHUNK_SIZE = int(_get_env("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", "50"))

TOP_K = int(_get_env("TOP_K", "3"))

# PostgreSQL
POSTGRES_URL = _get_env("POSTGRES_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")

# Redis
REDIS_HOST = _get_env("REDIS_HOST", "localhost")
REDIS_PORT = int(_get_env("REDIS_PORT", "6379"))
REDIS_TTL = int(_get_env("REDIS_TTL", "3600"))  # время жизни кэша в секундах (1 час)
SESSION_MEMORY_TTL = int(_get_env("SESSION_MEMORY_TTL", "14400"))  # время жизни памяти сессии в секундах (4 часа)

# FastAPI
API_HOST = _get_env("API_HOST", "0.0.0.0")
API_PORT = int(_get_env("API_PORT", "8000"))
