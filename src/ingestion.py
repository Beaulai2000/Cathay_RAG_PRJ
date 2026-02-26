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
from typing import Iterable, List, Tuple

from config import DEFAULT_POLICY_CLEAN_PATH


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


def build_index():
    """Placeholder for index‑building logic.

    When you are ready to ingest the actual policy, this function should:

    1. Call `read_policy_text()` to load the cleaned policy.
    2. Use `naive_paragraph_chunk()` (or a more advanced splitter) to
       split the text into chunks.
    3. Embed each chunk and store them in a vector store (e.g., Chroma),
       together with metadata such as article/section IDs.

    For the coding assignment you can implement this step once you have
    your policy document ready.
    """

    raise NotImplementedError("build_index() is a placeholder for now.")
