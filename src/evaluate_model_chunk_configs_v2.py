"""Run a harder second-pass evaluation across model and chunking configurations.

This version focuses on:
- ambiguous queries that should trigger clarification
- follow-up questions that rely on history
- synonym handling
- condition / exception questions

Run from project root:

    python -m src.evaluate_model_chunk_configs_v2
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR
from .evaluate_model_chunk_configs import CHUNK_CONFIGS, MODEL_CONFIGS
from .ingestion import build_index
from .rag_pipeline import RAGPipeline


@dataclass(frozen=True)
class Scenario:
    name: str
    focus: str
    turns: tuple[str, ...]


SCENARIOS = (
    Scenario(
        name="ambiguous_delay_then_pick_flight",
        focus="delay clarification + follow-up rewrite",
        turns=(
            "旅遊延誤賠償怎麼算？",
            "班機延誤保險",
        ),
    ),
    Scenario(
        name="ambiguous_exclusion_then_pick_credit_card",
        focus="section clarification + follow-up rewrite",
        turns=(
            "哪些原因屬於不可理賠範圍？",
            "信用卡盜用保險",
        ),
    ),
    Scenario(
        name="credit_card_fraud_synonym",
        focus="synonym mapping for 信用卡盜刷 -> 信用卡盜用保險",
        turns=("信用卡盜刷有哪些不能賠？",),
    ),
    Scenario(
        name="passport_loss_synonym",
        focus="synonym mapping for 護照遺失 -> 旅行文件損失保險",
        turns=("護照遺失怎麼申請理賠？",),
    ),
    Scenario(
        name="flight_delay_exception_direct",
        focus="condition + exception handling",
        turns=("如果沒有搭航空公司提供的第一班替代交通工具，還能申請班機延誤理賠嗎？",),
    ),
    Scenario(
        name="flight_delay_exception_followup",
        focus="history-aware follow-up reasoning",
        turns=(
            "班機延誤保險有哪些不可理賠範圍？",
            "那如果是因為不可抗力沒搭第一班替代交通工具呢？",
        ),
    ),
    Scenario(
        name="mobile_phone_theft_synonym",
        focus="synonym mapping for 手機被偷 -> 行動電話被竊損失保險",
        turns=("手機被偷可以怎麼申請理賠？",),
    ),
    Scenario(
        name="cash_theft_synonym",
        focus="synonym mapping for 現金被偷 -> 現金竊盜保險",
        turns=("現金被偷可以理賠嗎？",),
    ),
)

EVAL_DIR = DATA_DIR / "evals"


@dataclass
class TurnResult:
    user: str
    assistant: str


@dataclass
class ScenarioResult:
    name: str
    focus: str
    turns: list[TurnResult]


@dataclass
class EvalResult:
    label: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    overlap: int
    scenarios: list[ScenarioResult]


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
    scenario_results: list[ScenarioResult] = []

    for scenario in SCENARIOS:
        print(f"[INFO] Scenario: {scenario.name}")
        history: list[tuple[str, str]] = []
        turn_results: list[TurnResult] = []
        for turn in scenario.turns:
            print(f"[INFO] User: {turn}")
            answer = pipeline.answer(turn, history=history)
            turn_results.append(TurnResult(user=turn, assistant=answer))
            history.append((turn, answer))
        scenario_results.append(
            ScenarioResult(
                name=scenario.name,
                focus=scenario.focus,
                turns=turn_results,
            )
        )

    return EvalResult(
        label=label,
        llm_model=llm_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        scenarios=scenario_results,
    )


def write_results(results: list[EvalResult]) -> tuple[Path, Path]:
    """Write evaluation results to JSON and Markdown."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_DIR / f"model_chunk_eval_v2_{timestamp}.json"
    md_path = EVAL_DIR / f"model_chunk_eval_v2_{timestamp}.md"

    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# Model and Chunk Evaluation Results v2", ""]
    for result in results:
        md_lines.append(
            "## "
            f"{result.label} | llm={result.llm_model} | embedding={result.embedding_model} "
            f"| chunk_size={result.chunk_size} | overlap={result.overlap}"
        )
        md_lines.append("")
        for scenario in result.scenarios:
            md_lines.append(f"### Scenario: {scenario.name}")
            md_lines.append(f"Focus: {scenario.focus}")
            md_lines.append("")
            for i, turn in enumerate(scenario.turns, start=1):
                md_lines.append(f"#### Turn {i} - User")
                md_lines.append("")
                md_lines.append(turn.user)
                md_lines.append("")
                md_lines.append(f"#### Turn {i} - Assistant")
                md_lines.append("")
                md_lines.append(turn.assistant)
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
