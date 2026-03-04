"""Semantic (embedding‑based) retriever.

This module defines a baseline semantic retriever using a vector store
(e.g., Chroma) over policy chunks.

Future work:
- Add BM25 / keyword retriever in `bm25.py`.
- Add hybrid retriever combining semantic + keyword in `hybrid.py`.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config import EMBEDDING_MODEL, INDEX_DIR, RETRIEVER_TOP_K


def get_semantic_retriever(
    k: int | None = None,
    section: str | None = None,
    embedding_model: str | None = None,
):
    """Return a LangChain retriever over the policy index.

    Assumes that the index has been built and persisted under `INDEX_DIR`.
    """

    k = k or RETRIEVER_TOP_K
    embedding_model = embedding_model or EMBEDDING_MODEL
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectordb = Chroma(
        collection_name="travel_policy",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )

    search_kwargs = {"k": k}
    if section:
        search_kwargs["filter"] = {"section": section}

    return vectordb.as_retriever(search_kwargs=search_kwargs)
