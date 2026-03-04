"""Policy ingestion and chunking for the Cathay travel insurance RAG prototype.

This module is responsible for:

1. Reading the cleaned policy text (e.g., `data/policy_clean/policy_clean.txt`).
2. Splitting the text into chunks suitable for retrieval.
3. (Optional) Building the vector index from these chunks.

For now, we only implement a simple paragraph‑aware chunking strategy and
leave semantic/recursive chunking as future work.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_POLICY_CLEAN_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INSURANCE_SECTIONS,
)


def read_policy_text(path: Path | None = None) -> str:
    """Read the cleaned policy text file and return it as a single string."""

    path = path or DEFAULT_POLICY_CLEAN_PATH
    if not path.exists():
        raise FileNotFoundError(f"Policy text not found: {path}")
    return path.read_text(encoding="utf-8")


ARTICLE_HEADING_RE = re.compile(r"(?m)^(第[一二三四五六七八九十百千0-9]+條[^\n]*)$")


def naive_paragraph_chunk(
    text: str,
    chunk_size: int = 400,
    overlap: int = 75,
) -> List[str]:
    """Very simple paragraph‑aware chunking.

    - Split text by double newlines into paragraphs.
    - Concatenate paragraphs until chunk_size is reached, then start a new chunk.
    - Add a configurable overlap between chunks.

    This is only a baseline; future work includes:
    - recursive character splitting
    - semantic chunking based on sentence embeddings
    """

    # Basic fallback splitter: blank lines define paragraph boundaries.
    # This works when the source text already has clean paragraph spacing.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            # Store the current chunk once it is about to exceed the target size.
            chunks.append("\n\n".join(current))
            # Carry over a small tail so boundary sentences are less likely to break.
            overlap_text = chunks[-1][-overlap:]
            current = [overlap_text, para]
            current_len = len(overlap_text) + len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def article_aware_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split policy text by article headings, then sub-split long articles.

    This works better for the current policy text, which mostly uses single
    newlines and article headings such as "第二十七條 ...", instead of blank
    lines between paragraphs.
    """

    # Split on article headings first, since this policy is structured like clauses.
    matches = list(ARTICLE_HEADING_RE.finditer(text))
    if not matches:
        # Fall back to paragraph splitting if no article heading is detected.
        return naive_paragraph_chunk(text, chunk_size=chunk_size, overlap=overlap)

    chunks: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        if not article_text:
            continue

        if len(article_text) <= chunk_size:
            # Keep short articles intact so retrieval preserves the full clause context.
            chunks.append(article_text)
            continue

        # If one article is too long, split it line by line instead of slicing raw characters.
        lines = [line.strip() for line in article_text.splitlines() if line.strip()]
        current: List[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + (1 if current else 0)
            if current and current_len + line_len > chunk_size:
                chunk_text = "\n".join(current)
                chunks.append(chunk_text)
                # Keep a small overlap so trailing conditions or exceptions
                # remain visible in the next chunk.
                overlap_text = chunk_text[-overlap:].strip()
                current = [overlap_text, line] if overlap_text else [line]
                current_len = sum(len(part) for part in current) + max(len(current) - 1, 0)
            else:
                current.append(line)
                current_len += line_len

        if current:
            chunks.append("\n".join(current))

    return chunks


def extract_article_id(chunk: str) -> str:
    """Extract article heading from the chunk for citation metadata."""

    # The first line is usually the article heading, which is useful for citations later.
    first_line = chunk.splitlines()[0].strip() if chunk.strip() else ""
    return first_line if ARTICLE_HEADING_RE.match(first_line) else "(未標註條款編號)"


def infer_section(text: str) -> str:
    """Infer the policy section from known insurance section names."""

    for section in INSURANCE_SECTIONS:
        if section in text:
            return section
    return "未分類"


def preview_chunks(chunks: List[str], limit: int = 3, preview_len: int = 160) -> None:
    """Print a short preview of the first few chunks for quick manual inspection."""

    for i, chunk in enumerate(chunks[:limit]):
        article_id = extract_article_id(chunk)
        snippet = chunk[:preview_len].replace("\n", " ")
        if len(chunk) > preview_len:
            snippet += "..."
        print(f"[INFO] Chunk {i}: article={article_id} chars={len(chunk)}")
        print(f"[INFO] Preview: {snippet}")


def build_index(
    chunk_size: int | None = None,
    overlap: int | None = None,
    preview_limit: int = 3,
) -> None:
    """Build (or rebuild) the semantic index for the policy text.

    Steps:
    1. Load cleaned policy text from DEFAULT_POLICY_CLEAN_PATH.
    2. Chunk it with article-aware splitting.
    3. Create a Chroma vector store with OpenAI embeddings and persist
       it under INDEX_DIR.

    This is a simple baseline: one document chunk = one Chroma document,
    with metadata including a sequential chunk_id.
    """

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    print(f"[INFO] Reading cleaned policy from {DEFAULT_POLICY_CLEAN_PATH}")
    text = read_policy_text()
    chunks = article_aware_chunk(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[INFO] Chunk settings: chunk_size={chunk_size}, overlap={overlap}")
    print(f"[INFO] Split policy into {len(chunks)} chunks")
    if preview_limit > 0:
        preview_chunks(chunks, limit=preview_limit)

    # Prepare documents and metadata
    from langchain_core.documents import Document

    docs = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            # Store the article heading so downstream answers can cite the clause.
            "article_id": extract_article_id(chunk),
            "section": infer_section(chunk),
        }
        docs.append(Document(page_content=chunk, metadata=metadata))

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building Chroma index in {INDEX_DIR}")
    # Open the collection first so old data can be cleared before rebuilding.
    vectordb = Chroma(
        collection_name="travel_policy",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    # Rebuild from scratch each run to avoid mixing old and new chunk sets.
    vectordb.delete_collection()

    # Write the fresh chunk set into Chroma for retrieval.
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="travel_policy",
        persist_directory=str(INDEX_DIR),
    )
    # Note: Chroma.from_documents() with a persist_directory will write
    # the index to disk; an explicit persist() call is not needed with
    # the langchain_chroma integration we are using.
    print("[INFO] Index build complete.")


if __name__ == "__main__":
    build_index()
