"""Configuration for the Cathay travel insurance RAG prototype.

Notes:
- API keys (e.g., OpenAI) should be provided via environment variables and
  **never** hard‑coded in this file.
- This module centralizes model names and index paths so they can be
  changed in one place.
"""

import os
from pathlib import Path

# === Models ===

# LLM used to generate answers
LLM_MODEL = os.environ.get("CATHAY_RAG_LLM_MODEL", "gpt-4o-mini")

# Embedding model for semantic search
EMBEDDING_MODEL = os.environ.get("CATHAY_RAG_EMBEDDING_MODEL", "text-embedding-3-small")


# === Paths ===

# Project root = directory containing this file's parent (src/..)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
POLICY_RAW_DIR = DATA_DIR / "policy_raw"
POLICY_CLEAN_DIR = DATA_DIR / "policy_clean"

# Vector index path (for Chroma or similar)
INDEX_DIR = DATA_DIR / "index"

# Default cleaned policy file (you can change this when you have the doc)
DEFAULT_POLICY_CLEAN_PATH = POLICY_CLEAN_DIR / "policy_clean.txt"
