"""Configuration for the Cathay travel insurance RAG prototype."""

import os
from pathlib import Path

# === Models ===

# LLM used to generate answers
LLM_MODEL = os.environ.get("CATHAY_RAG_LLM_MODEL", "gpt-4.1")

# Embedding model for semantic search
EMBEDDING_MODEL = os.environ.get("CATHAY_RAG_EMBEDDING_MODEL", "text-embedding-3-small")


# === Retrieval ===

# Number of chunks retrieved for each user question by default.
RETRIEVER_TOP_K = int(os.environ.get("CATHAY_RAG_RETRIEVER_TOP_K", "5"))

# Number of previous conversation turns to include in the chat prompt.
CHAT_HISTORY_WINDOW = int(os.environ.get("CATHAY_RAG_CHAT_HISTORY_WINDOW", "3"))

# Supported policy sections for section-aware retrieval.
INSURANCE_SECTIONS = (
    "旅程取消保險",
    "班機延誤保險",
    "旅程更改保險",
    "行李延誤保險",
    "行李損失保險",
    "旅行文件損失保險",
    "個人海外旅行補償保險",
    "劫機保險",
    "食品中毒保險",
    "現金竊盜保險",
    "信用卡盜用保險",
    "居家竊盜保險",
    "租車事故保險",
    "特殊活動取消慰問保險",
    "賽事取消慰問保險",
    "行動電話被竊損失保險",
    "海外旅行急難救助保險",
)

# Informal user phrases mapped to the formal policy section names.
SECTION_ALIASES = {
    "航班延誤": "班機延誤保險",
    "班機晚點": "班機延誤保險",
    "班機延誤": "班機延誤保險",
    "旅遊取消": "旅程取消保險",
    "取消旅程": "旅程取消保險",
    "更改旅程": "旅程更改保險",
    "改行程": "旅程更改保險",
    "行李不見": "行李損失保險",
    "行李遺失": "行李損失保險",
    "行李被偷": "行李損失保險",
    "行李延誤": "行李延誤保險",
    "文件遺失": "旅行文件損失保險",
    "護照遺失": "旅行文件損失保險",
    "信用卡被盜刷": "信用卡盜用保險",
    "信用卡盜刷": "信用卡盜用保險",
    "現金被偷": "現金竊盜保險",
    "手機被偷": "行動電話被竊損失保險",
    "租車事故": "租車事故保險",
}


# === Chunking ===

# Default chunking settings for policy ingestion.
# This project uses article-aware chunking first, then sub-splits long articles.
CHUNK_SIZE = int(os.environ.get("CATHAY_RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.environ.get("CATHAY_RAG_CHUNK_OVERLAP", "100"))


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
