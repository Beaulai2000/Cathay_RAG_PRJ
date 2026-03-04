"""Run a small evaluation sweep across chunking configurations.

This script rebuilds the vector index for each chunking configuration,
asks a fixed set of questions, and saves the answers for comparison.

Run from project root:

    python -m src.evaluate_chunk_configs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR
from .ingestion import build_index
from .rag_pipeline import RAGPipeline


QUESTIONS = (
    "什麼情況下可以申請班機延誤理賠？",
    "班機延誤有哪些不保事項？",
    "行李遺失如何申請理賠？",
    "信用卡盜用有哪些不可理賠範圍？",
    "特殊活動取消慰問保險的承保範圍是什麼？",
)

CHUNK_CONFIGS = (
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 700, "overlap": 100},
    {"chunk_size": 900, "overlap": 120},
)

EVAL_DIR = DATA_DIR / "evals"


@dataclass
class QuestionResult:
    question: str
    answer: str


@dataclass
class ChunkEvalResult:
    chunk_size: int
    overlap: int
    results: list[QuestionResult]


def run_chunk_eval(chunk_size: int, overlap: int) -> ChunkEvalResult:
    """Build an index for one chunk config and run the evaluation questions."""

    print(f"[INFO] Evaluating chunk_size={chunk_size}, overlap={overlap}")
    build_index(chunk_size=chunk_size, overlap=overlap, preview_limit=0)

    pipeline = RAGPipeline()
    results: list[QuestionResult] = []
    for question in QUESTIONS:
        print(f"[INFO] Asking: {question}")
        answer = pipeline.answer(question)
        results.append(QuestionResult(question=question, answer=answer))

    return ChunkEvalResult(chunk_size=chunk_size, overlap=overlap, results=results)


def write_results(results: list[ChunkEvalResult]) -> tuple[Path, Path]:
    """Write evaluation results to JSON and Markdown."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_DIR / f"chunk_eval_{timestamp}.json"
    md_path = EVAL_DIR / f"chunk_eval_{timestamp}.md"

    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# Chunk Evaluation Results", ""]
    for result in results:
        md_lines.append(f"## chunk_size={result.chunk_size}, overlap={result.overlap}")
        md_lines.append("")
        for item in result.results:
            md_lines.append(f"### Q: {item.question}")
            md_lines.append("")
            md_lines.append(item.answer)
            md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path


def main() -> None:
    all_results: list[ChunkEvalResult] = []
    for config in CHUNK_CONFIGS:
        all_results.append(
            run_chunk_eval(
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
            )
        )

    json_path, md_path = write_results(all_results)
    print(f"[INFO] Saved JSON results to {json_path}")
    print(f"[INFO] Saved Markdown results to {md_path}")


if __name__ == "__main__":
    main()
