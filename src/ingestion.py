"""Policy ingestion and chunking for the Cathay travel insurance RAG prototype.

This module is responsible for:

1. Reading the cleaned policy text (e.g., `data/policy_clean/policy_clean.txt`).
2. Splitting the text into chunks suitable for retrieval.
3. (Optional) Building the vector index from these chunks.

For now, we only implement a simple paragraph‑aware chunking strategy and
leave semantic/recursive chunking as future work.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import DEFAULT_POLICY_CLEAN_PATH, EMBEDDING_MODEL, INDEX_DIR


def read_policy_text(path: Path | None = None) -> str:
    """Read the cleaned policy text file and return it as a single string."""

    path = path or DEFAULT_POLICY_CLEAN_PATH
    if not path.exists():
        raise FileNotFoundError(f"Policy text not found: {path}")
    return path.read_text(encoding="utf-8")


def naive_paragraph_chunk(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[str]:
    """Very simple paragraph‑aware chunking.

    - Split text by double newlines into paragraphs.
    - Concatenate paragraphs until chunk_size is reached, then start a new chunk.
    - Add a configurable overlap between chunks.

    This is only a baseline; future work includes:
    - recursive character splitting
    - semantic chunking based on sentence embeddings
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            # flush current chunk
            chunks.append("\n\n".join(current))
            # start new chunk with overlap (last part of previous chunk)
            overlap_text = chunks[-1][-overlap:]
            current = [overlap_text, para]
            current_len = len(overlap_text) + len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_index() -> None:
    """Build (or rebuild) the semantic index for the policy text.

    Steps:
    1. Load cleaned policy text from DEFAULT_POLICY_CLEAN_PATH.
    2. Chunk it with naive_paragraph_chunk.
    3. Create a Chroma vector store with OpenAI embeddings and persist
       it under INDEX_DIR.

    This is a simple baseline: one document chunk = one Chroma document,
    with metadata including a sequential chunk_id.
    """

    print(f"[INFO] Reading cleaned policy from {DEFAULT_POLICY_CLEAN_PATH}")
    text = read_policy_text()
    chunks = naive_paragraph_chunk(text)
    print(f"[INFO] Split policy into {len(chunks)} chunks")

    # Prepare documents and metadata
    from langchain_core.documents import Document

    docs = []
    for i, chunk in enumerate(chunks):
        metadata = {"chunk_id": i}
        docs.append(Document(page_content=chunk, metadata=metadata))

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building Chroma index in {INDEX_DIR}")
    # Rebuild the collection from scratch each time for now.
    vectordb = Chroma(
        collection_name="travel_policy",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    # Wipe existing collection contents (if any)
    vectordb.delete_collection()

    # Create a fresh store with the new docs
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="travel_policy",
        persist_directory=str(INDEX_DIR),
    )
    vectordb.persist()
    print("[INFO] Index build complete.")


if __name__ == "__main__":
    build_index()
