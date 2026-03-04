"""Semantic (embedding‑based) retriever.

This module defines a baseline semantic retriever using a vector store
(e.g., Chroma) over policy chunks.

Future work:
- Add BM25 / keyword retriever in `bm25.py`.
- Add hybrid retriever combining semantic + keyword in `hybrid.py`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config import EMBEDDING_MODEL, INDEX_DIR, INDEX_METADATA_FILENAME, RETRIEVER_TOP_K, get_index_dir


def index_exists(index_dir: Path) -> bool:
    """Return True when the target directory looks like a built Chroma index."""

    return (index_dir / INDEX_METADATA_FILENAME).exists() or (index_dir / "chroma.sqlite3").exists()


def build_reindex_command(embedding_model: str) -> str:
    """Return the shell command needed to rebuild the index for a model."""

    if embedding_model == EMBEDDING_MODEL:
        return "python -m src.ingestion"
    return f'CATHAY_RAG_EMBEDDING_MODEL="{embedding_model}" python -m src.ingestion'


def resolve_index_dir(embedding_model: str) -> Path:
    """Resolve the preferred index directory, with a fallback to the legacy layout."""

    preferred_dir = get_index_dir(embedding_model)
    if index_exists(preferred_dir):
        return preferred_dir

    # Backward compatibility: older runs wrote directly into data/index/.
    if preferred_dir != INDEX_DIR and index_exists(INDEX_DIR):
        return INDEX_DIR

    raise FileNotFoundError(
        "找不到對應的向量索引。"
        f"\n目前 embedding model: {embedding_model}"
        f"\n請先執行：{build_reindex_command(embedding_model)}"
    )


def get_semantic_retriever(
    k: int | None = None,
    section: str | None = None,
    embedding_model: str | None = None,
):
    """Return a LangChain retriever over the policy index.

    Assumes that the index has been built and persisted under the model-specific
    directory resolved from `INDEX_DIR`.
    """

    k = k or RETRIEVER_TOP_K
    embedding_model = embedding_model or EMBEDDING_MODEL
    index_dir = resolve_index_dir(embedding_model)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectordb = Chroma(
        collection_name="travel_policy",
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )

    search_kwargs = {"k": k}
    if section:
        search_kwargs["filter"] = {"section": section}

    return vectordb.as_retriever(search_kwargs=search_kwargs)
