"""Run retrieval-level evaluation for the v2 scenarios.

This script evaluates retrieval quality before final answer generation.
It records, for each user turn:

- what query was actually sent to retrieval
- which top-k chunks were retrieved
- whether the retrieved chunks match the expected policy section

Run from project root:

    python -m src.evaluate_retrieval_v2
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import DATA_DIR, RETRIEVER_TOP_K
from .evaluate_model_chunk_configs import CHUNK_CONFIGS, MODEL_CONFIGS
from .ingestion import build_index
from .rag_pipeline import (
    build_delay_clarification_message,
    build_section_clarification_message,
    infer_requested_section,
    is_ambiguous_delay_question,
    is_ambiguous_section_question,
    normalize_question_terms,
    rewrite_followup_question,
)
from .retrievers.semantic import get_semantic_retriever


@dataclass(frozen=True)
class Scenario:
    name: str
    focus: str
    turns: tuple[str, ...]
    expected_sections: tuple[str | None, ...]


SCENARIOS = (
    Scenario(
        name="ambiguous_delay_then_pick_flight",
        focus="delay clarification + follow-up rewrite",
        turns=(
            "旅遊延誤賠償怎麼算？",
            "班機延誤保險",
        ),
        expected_sections=(None, "班機延誤保險"),
    ),
    Scenario(
        name="ambiguous_exclusion_then_pick_credit_card",
        focus="section clarification + follow-up rewrite",
        turns=(
            "哪些原因屬於不可理賠範圍？",
            "信用卡盜用保險",
        ),
        expected_sections=(None, "信用卡盜用保險"),
    ),
    Scenario(
        name="credit_card_fraud_synonym",
        focus="synonym mapping for 信用卡盜刷 -> 信用卡盜用保險",
        turns=("信用卡盜刷有哪些不能賠？",),
        expected_sections=("信用卡盜用保險",),
    ),
    Scenario(
        name="passport_loss_synonym",
        focus="synonym mapping for 護照遺失 -> 旅行文件損失保險",
        turns=("護照遺失怎麼申請理賠？",),
        expected_sections=("旅行文件損失保險",),
    ),
    Scenario(
        name="flight_delay_exception_direct",
        focus="condition + exception handling",
        turns=("如果沒有搭航空公司提供的第一班替代交通工具，還能申請班機延誤理賠嗎？",),
        expected_sections=("班機延誤保險",),
    ),
    Scenario(
        name="flight_delay_exception_followup",
        focus="history-aware follow-up reasoning",
        turns=(
            "班機延誤保險有哪些不可理賠範圍？",
            "那如果是因為不可抗力沒搭第一班替代交通工具呢？",
        ),
        expected_sections=("班機延誤保險", "班機延誤保險"),
    ),
    Scenario(
        name="mobile_phone_theft_synonym",
        focus="synonym mapping for 手機被偷 -> 行動電話被竊損失保險",
        turns=("手機被偷可以怎麼申請理賠？",),
        expected_sections=("行動電話被竊損失保險",),
    ),
    Scenario(
        name="cash_theft_synonym",
        focus="synonym mapping for 現金被偷 -> 現金竊盜保險",
        turns=("現金被偷可以理賠嗎？",),
        expected_sections=("現金竊盜保險",),
    ),
)

EVAL_DIR = DATA_DIR / "evals"


@dataclass
class TurnRetrievalResult:
    question: str
    processed_query: str
    inferred_section: str | None
    expected_section: str | None
    retrieved_chunk_ids: list[int]
    retrieved_article_ids: list[str]
    retrieved_sections: list[str]
    relevant: str
    precision_at_k: float | None
    recall_hit_at_k: bool | None
    notes: str


@dataclass
class ScenarioResult:
    name: str
    focus: str
    turns: list[TurnRetrievalResult]


@dataclass
class EvalSummary:
    evaluated_turns: int
    average_precision_at_k: float | None
    recall_hit_rate_at_k: float | None


@dataclass
class EvalResult:
    embedding_model: str
    chunk_size: int
    overlap: int
    k: int
    scenarios: list[ScenarioResult]
    summary: EvalSummary


def unique_embedding_models() -> tuple[str, ...]:
    """Return de-duplicated embedding models from MODEL_CONFIGS."""

    models: list[str] = []
    seen: set[str] = set()
    for config in MODEL_CONFIGS:
        model = config["embedding_model"]
        if model not in seen:
            models.append(model)
            seen.add(model)
    return tuple(models)


def evaluate_turn_retrieval(
    *,
    question: str,
    history: Sequence[tuple[str, str]],
    expected_section: str | None,
    embedding_model: str,
    k: int,
) -> tuple[TurnRetrievalResult, str]:
    """Evaluate retrieval for one user turn and return the assistant history placeholder."""

    rewritten = rewrite_followup_question(question, history=history)
    processed_query = normalize_question_terms(rewritten)
    section = infer_requested_section(processed_query, history=history)

    if not section and is_ambiguous_delay_question(processed_query):
        clarification = build_delay_clarification_message()
        return (
            TurnRetrievalResult(
                question=question,
                processed_query=processed_query,
                inferred_section=None,
                expected_section=expected_section,
                retrieved_chunk_ids=[],
                retrieved_article_ids=[],
                retrieved_sections=[],
                relevant="N/A",
                precision_at_k=None,
                recall_hit_at_k=None,
                notes="Clarification required before retrieval (delay type is ambiguous).",
            ),
            clarification,
        )

    if not section and is_ambiguous_section_question(processed_query):
        clarification = build_section_clarification_message()
        return (
            TurnRetrievalResult(
                question=question,
                processed_query=processed_query,
                inferred_section=None,
                expected_section=expected_section,
                retrieved_chunk_ids=[],
                retrieved_article_ids=[],
                retrieved_sections=[],
                relevant="N/A",
                precision_at_k=None,
                recall_hit_at_k=None,
                notes="Clarification required before retrieval (section is ambiguous).",
            ),
            clarification,
        )

    retriever = get_semantic_retriever(k=k, section=section, embedding_model=embedding_model)
    docs = retriever.invoke(processed_query)

    chunk_ids: list[int] = []
    article_ids: list[str] = []
    sections: list[str] = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if isinstance(chunk_id, int):
            chunk_ids.append(chunk_id)
        elif isinstance(chunk_id, str) and chunk_id.isdigit():
            chunk_ids.append(int(chunk_id))
        article_ids.append(str(doc.metadata.get("article_id", "(未標註條款編號)")))
        sections.append(str(doc.metadata.get("section", "未分類")))

    if expected_section is None:
        relevant = "N/A"
        precision_at_k = None
        recall_hit_at_k = None
        notes = (
            f"Retrieved with inferred section={section}."
            if section
            else "Retrieved without explicit section filter."
        )
    else:
        relevant_hits = [doc_section == expected_section for doc_section in sections]
        relevant_count = sum(relevant_hits)
        if relevant_count == 0:
            relevant = "No"
        elif relevant_count == len(sections):
            relevant = "Yes"
        else:
            relevant = "Partial"

        precision_at_k = (relevant_count / len(sections)) if sections else 0.0
        recall_hit_at_k = relevant_count > 0

        mismatched_articles = [
            article_id
            for article_id, doc_section in zip(article_ids, sections)
            if doc_section != expected_section
        ]
        if not sections:
            notes = "No chunks were retrieved."
        elif mismatched_articles:
            notes = (
                f"Expected section={expected_section}; "
                f"mismatched clauses: {', '.join(mismatched_articles[:3])}"
            )
        else:
            notes = f"All retrieved chunks match expected section={expected_section}."

    result = TurnRetrievalResult(
        question=question,
        processed_query=processed_query,
        inferred_section=section,
        expected_section=expected_section,
        retrieved_chunk_ids=chunk_ids,
        retrieved_article_ids=article_ids,
        retrieved_sections=sections,
        relevant=relevant,
        precision_at_k=precision_at_k,
        recall_hit_at_k=recall_hit_at_k,
        notes=notes,
    )
    return result, "[retrieval-level evaluation placeholder]"


def run_eval(
    *,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    k: int,
) -> EvalResult:
    """Run one retrieval-level evaluation experiment."""

    print(
        "[INFO] Retrieval eval "
        f"embedding_model={embedding_model} chunk_size={chunk_size} overlap={overlap} k={k}"
    )
    build_index(
        chunk_size=chunk_size,
        overlap=overlap,
        preview_limit=0,
        embedding_model=embedding_model,
    )

    scenario_results: list[ScenarioResult] = []
    precision_values: list[float] = []
    recall_hits: list[bool] = []

    for scenario in SCENARIOS:
        print(f"[INFO] Scenario: {scenario.name}")
        assert len(scenario.turns) == len(scenario.expected_sections), "Scenario turns and expected sections must align."

        history: list[tuple[str, str]] = []
        turn_results: list[TurnRetrievalResult] = []
        for question, expected_section in zip(scenario.turns, scenario.expected_sections):
            print(f"[INFO] User: {question}")
            result, assistant_placeholder = evaluate_turn_retrieval(
                question=question,
                history=history,
                expected_section=expected_section,
                embedding_model=embedding_model,
                k=k,
            )
            turn_results.append(result)
            history.append((question, assistant_placeholder))

            if result.precision_at_k is not None:
                precision_values.append(result.precision_at_k)
            if result.recall_hit_at_k is not None:
                recall_hits.append(result.recall_hit_at_k)

        scenario_results.append(
            ScenarioResult(
                name=scenario.name,
                focus=scenario.focus,
                turns=turn_results,
            )
        )

    summary = EvalSummary(
        evaluated_turns=len(precision_values),
        average_precision_at_k=(sum(precision_values) / len(precision_values)) if precision_values else None,
        recall_hit_rate_at_k=(sum(recall_hits) / len(recall_hits)) if recall_hits else None,
    )
    return EvalResult(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        k=k,
        scenarios=scenario_results,
        summary=summary,
    )


def write_results(results: list[EvalResult]) -> tuple[Path, Path]:
    """Write retrieval-level evaluation outputs to JSON and Markdown."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_DIR / f"retrieval_eval_v2_{timestamp}.json"
    md_path = EVAL_DIR / f"retrieval_eval_v2_{timestamp}.md"

    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# Retrieval-Level Evaluation Results v2", ""]
    md_lines.append("## Summary by Configuration")
    md_lines.append("")
    md_lines.append("| Embedding Model | Chunk Size | Overlap | k | Avg Precision@k | Recall-Hit@k |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for result in results:
        avg_precision = (
            f"{result.summary.average_precision_at_k:.3f}"
            if result.summary.average_precision_at_k is not None
            else "N/A"
        )
        recall_rate = (
            f"{result.summary.recall_hit_rate_at_k:.3f}"
            if result.summary.recall_hit_rate_at_k is not None
            else "N/A"
        )
        md_lines.append(
            f"| {result.embedding_model} | {result.chunk_size} | {result.overlap} | {result.k} | "
            f"{avg_precision} | {recall_rate} |"
        )
    md_lines.append("")

    for result in results:
        md_lines.append(
            "## "
            f"embedding={result.embedding_model} | chunk_size={result.chunk_size} | overlap={result.overlap} | k={result.k}"
        )
        md_lines.append("")
        md_lines.append(f"- Evaluated turns (with expected section): {result.summary.evaluated_turns}")
        if result.summary.average_precision_at_k is not None:
            md_lines.append(f"- Avg precision@k: {result.summary.average_precision_at_k:.3f}")
        else:
            md_lines.append("- Avg precision@k: N/A")
        md_lines.append(
            f"- Recall-hit@k: {result.summary.recall_hit_rate_at_k:.3f}"
            if result.summary.recall_hit_rate_at_k is not None
            else "- Recall-hit@k: N/A"
        )
        md_lines.append("")

        for scenario in result.scenarios:
            md_lines.append(f"### Scenario: {scenario.name}")
            md_lines.append(f"Focus: {scenario.focus}")
            md_lines.append("")
            md_lines.append("| Question | Retrieved Chunk IDs | Relevant? | Notes |")
            md_lines.append("|---|---|---|---|")
            for turn in scenario.turns:
                chunk_ids = ", ".join(str(chunk_id) for chunk_id in turn.retrieved_chunk_ids) or "-"
                md_lines.append(
                    f"| {turn.question} | {chunk_ids} | {turn.relevant} | {turn.notes} |"
                )
            md_lines.append("")
            for turn in scenario.turns:
                article_ids = ", ".join(turn.retrieved_article_ids) if turn.retrieved_article_ids else "-"
                sections = ", ".join(turn.retrieved_sections) if turn.retrieved_sections else "-"
                precision = f"{turn.precision_at_k:.3f}" if turn.precision_at_k is not None else "N/A"
                recall = str(turn.recall_hit_at_k) if turn.recall_hit_at_k is not None else "N/A"
                md_lines.append(f"- Question: {turn.question}")
                md_lines.append(f"  Processed Query: {turn.processed_query}")
                md_lines.append(f"  Expected Section: {turn.expected_section or 'N/A'}")
                md_lines.append(f"  Inferred Section: {turn.inferred_section or 'N/A'}")
                md_lines.append(f"  Retrieved Articles: {article_ids}")
                md_lines.append(f"  Retrieved Sections: {sections}")
                md_lines.append(f"  Precision@k: {precision}")
                md_lines.append(f"  Recall-Hit@k: {recall}")
                md_lines.append("")

    md_lines.append(
        "Note: recall here is `recall-hit@k` (whether at least one relevant chunk appears in top-k), "
        "not full corpus recall."
    )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    results: list[EvalResult] = []
    embedding_models = unique_embedding_models()
    for embedding_model in embedding_models:
        for chunk_config in CHUNK_CONFIGS:
            results.append(
                run_eval(
                    embedding_model=embedding_model,
                    chunk_size=chunk_config["chunk_size"],
                    overlap=chunk_config["overlap"],
                    k=RETRIEVER_TOP_K,
                )
            )

    json_path, md_path = write_results(results)
    print(f"[INFO] Saved JSON results to {json_path}")
    print(f"[INFO] Saved Markdown results to {md_path}")


if __name__ == "__main__":
    main()
