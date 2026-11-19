"""Settings, all overridable with environment variables (or a local .env file)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KNOWLEDGE_DIR = ROOT / "knowledge"
INDEX_DIR = ROOT / "index"
MODEL_CACHE = Path(os.getenv("FASTEMBED_CACHE", ROOT / ".cache" / "fastembed"))

# Embeddings run locally (ONNX), so they are free and need no key.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# LLM: Groq free tier. Swap the model with one env var, no code change.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))

TOP_K = int(os.getenv("TOP_K", "4"))

# Only the portfolio (and local dev) may call the API from a browser.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://harshilmayurpatel.github.io,http://localhost:8765,http://127.0.0.1:8765",
    ).split(",")
    if o.strip()
]

# Abuse limits per visitor IP (in memory; resets when the Space restarts).
RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "8"))
RATE_PER_DAY = int(os.getenv("RATE_PER_DAY", "60"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "400"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# Mic intent: cosine similarity needed for the no-LLM fallback to act
INTENT_MIN_SIM = float(os.getenv("INTENT_MIN_SIM", "0.80"))
INTENT_PER_MINUTE = int(os.getenv("INTENT_PER_MINUTE", "20"))
