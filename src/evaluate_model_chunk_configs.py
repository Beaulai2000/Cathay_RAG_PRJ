"""Run evaluation sweeps across LLM, embedding, and chunking configurations.

This script rebuilds the vector index for each (embedding, chunk) combination,
runs a fixed set of questions through the chosen LLM, and saves the answers.

Run from project root:

    python -m src.evaluate_model_chunk_configs
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

MODEL_CONFIGS = (
    {
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "label": "baseline_low_cost",
    },
    {
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "label": "better_embedding_same_llm",
    },
    {
        "llm_model": "gpt-4.1",
        "embedding_model": "text-embedding-3-small",
        "label": "better_llm_same_embedding",
    },
    {
        "llm_model": "gpt-4.1",
        "embedding_model": "text-embedding-3-large",
        "label": "high_quality_41_large",
    },
    {
        "llm_model": "gpt-5",
        "embedding_model": "text-embedding-3-small",
        "label": "better_llm_gpt5_small_embedding",
    },
    {
        "llm_model": "gpt-5",
        "embedding_model": "text-embedding-3-large",
        "label": "high_quality_gpt5_large",
    },
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
class EvalResult:
    label: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    overlap: int
    results: list[QuestionResult]


def run_eval(
    *,
    llm_model: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    label: str,
) -> EvalResult:
    """Run one experiment for a given model pair and chunk config."""

    print(
        "[INFO] Evaluating "
        f"label={label} llm_model={llm_model} embedding_model={embedding_model} "
        f"chunk_size={chunk_size} overlap={overlap}"
    )
    build_index(
        chunk_size=chunk_size,
        overlap=overlap,
        preview_limit=0,
        embedding_model=embedding_model,
    )

    pipeline = RAGPipeline(llm_model=llm_model, embedding_model=embedding_model)
    results: list[QuestionResult] = []
    for question in QUESTIONS:
        print(f"[INFO] Asking: {question}")
        answer = pipeline.answer(question)
        results.append(QuestionResult(question=question, answer=answer))

    return EvalResult(
        label=label,
        llm_model=llm_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        results=results,
    )


def write_results(results: list[EvalResult]) -> tuple[Path, Path]:
    """Write evaluation results to JSON and Markdown."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_DIR / f"model_chunk_eval_{timestamp}.json"
    md_path = EVAL_DIR / f"model_chunk_eval_{timestamp}.md"

    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# Model and Chunk Evaluation Results", ""]
    for result in results:
        md_lines.append(
            "## "
            f"{result.label} | llm={result.llm_model} | embedding={result.embedding_model} "
            f"| chunk_size={result.chunk_size} | overlap={result.overlap}"
        )
        md_lines.append("")
        for item in result.results:
            md_lines.append(f"### Q: {item.question}")
            md_lines.append("")
            md_lines.append(item.answer)
            md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    all_results: list[EvalResult] = []
    for model_config in MODEL_CONFIGS:
        for chunk_config in CHUNK_CONFIGS:
            all_results.append(
                run_eval(
                    label=model_config["label"],
                    llm_model=model_config["llm_model"],
                    embedding_model=model_config["embedding_model"],
                    chunk_size=chunk_config["chunk_size"],
                    overlap=chunk_config["overlap"],
                )
            )

    json_path, md_path = write_results(all_results)
    print(f"[INFO] Saved JSON results to {json_path}")
    print(f"[INFO] Saved Markdown results to {md_path}")


if __name__ == "__main__":
    main()
