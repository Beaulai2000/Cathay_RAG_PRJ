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

from config import EMBEDDING_MODEL, INDEX_DIR, RETRIEVER_TOP_K


def get_semantic_retriever(k: int | None = None):
    """Return a LangChain retriever over the policy index.

    Assumes that the index has been built and persisted under `INDEX_DIR`.
    """

    k = k or RETRIEVER_TOP_K
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name="travel_policy",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )

    return vectordb.as_retriever(search_kwargs={"k": k})
