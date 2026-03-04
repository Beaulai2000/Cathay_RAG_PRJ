"""RAG pipeline glue: Question → Retriever → LLM → Answer (with sources).

This module wires together:
- a retriever (currently semantic vector search over policy chunks)
- an LLM for answer generation

The goal is to keep the core flow simple:

    User Question → Retriever → Top‑k Chunks → LLM Answer (with clause citations)
"""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain_openai import ChatOpenAI

from .config import CHAT_HISTORY_WINDOW, INSURANCE_SECTIONS, LLM_MODEL, RETRIEVER_TOP_K, SECTION_ALIASES
from .retrievers.semantic import get_semantic_retriever


class RAGPipeline:
    """Simple RAG pipeline for the travel insurance policy."""

    def __init__(self, k: int | None = None) -> None:
        self.k = k or RETRIEVER_TOP_K
        self.retriever = get_semantic_retriever(k=self.k)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def answer(
        self,
        question: str,
        history: Sequence[Any] | None = None,
    ) -> str:
        """Retrieve relevant clauses and generate an answer with citations."""

        question = rewrite_followup_question(question, history=history)
        question = normalize_question_terms(question)

        # 1. Retrieve top‑k chunks
        section = infer_requested_section(question, history=history)
        if not section and is_ambiguous_delay_question(question):
            return build_delay_clarification_message()
        if not section and is_ambiguous_section_question(question):
            return build_section_clarification_message()

        retriever = get_semantic_retriever(k=self.k, section=section) if section else self.retriever
        docs = retriever.invoke(question)

        if not docs:
            return "目前找不到與條款內容相符的資訊，建議洽詢客服或查看正式保單。"

        # 2. Build context string with clause IDs and text
        context_lines: List[str] = ["條款內容："]
        for d in docs:
            article_id = d.metadata.get("article_id", "(未標註條款編號)")
            context_lines.append(f"[{article_id}]\n{d.page_content}\n")

        context = "\n".join(context_lines)

        # 3. Compose chat-style messages
        messages = [
            (
                "system",
                (
                    "你是一位保險專員，負責解釋「旅遊不便險」的條款內容。"
                    "你只能根據提供的條款內容回答問題，不要加入條款以外的推測。"
                    "若條款未明確規範，請說明"
                    "「條款未明確規範，建議洽詢客服或查看正式保單」。"
                    "請用條理清楚的中文回答，並在最後列出你引用的條款編號與關鍵原文。"
                ),
            ),
        ]

        recent_history = list(history or [])[-CHAT_HISTORY_WINDOW:]
        for item in recent_history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content:
                    messages.append((role, content))
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_message, assistant_message = item[0], item[1]
                if user_message:
                    messages.append(("user", user_message))
                if assistant_message:
                    messages.append(("assistant", assistant_message))

        messages.append(("user", f"{context}\n\n問題：{question}"))

        resp = self.llm.invoke(messages)
        return resp.content


def infer_requested_section(question: str, history: Sequence[Any] | None = None) -> str | None:
    """Infer whether the user explicitly mentioned a supported insurance section."""

    current_section = infer_section_from_text(question)
    if current_section:
        return current_section

    history_text = build_history_text(history)
    return infer_section_from_text(history_text)


def rewrite_followup_question(question: str, history: Sequence[Any] | None = None) -> str:
    """Rewrite short follow-ups like a bare section name into a full question."""

    section = infer_section_from_text(question)
    if not section:
        return question

    normalized_question = question.strip()
    if normalized_question not in {section, section.removesuffix("保險")}:
        return question

    if not last_assistant_message_is_clarification(history):
        return question

    if section in {"班機延誤保險", "行李延誤保險"} and last_assistant_message_is_delay_clarification(history):
        return f"{section}什麼情況下可以申請理賠？"

    return f"{section}有哪些不可理賠範圍？"


def infer_section_from_text(text: str) -> str | None:
    """Infer a section name from arbitrary text."""

    for alias, section in SECTION_ALIASES.items():
        if alias in text:
            return section

    for section in INSURANCE_SECTIONS:
        aliases = {section}
        if section.endswith("保險"):
            aliases.add(section.removesuffix("保險"))

        if any(alias and alias in text for alias in aliases):
            return section

    return None


def build_history_text(history: Sequence[Any] | None) -> str:
    """Flatten recent history into plain text for section inference."""

    if not history:
        return ""

    texts: List[str] = []
    for item in list(history)[-CHAT_HISTORY_WINDOW:]:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
            continue

        if isinstance(item, (list, tuple)):
            for value in item[:2]:
                if isinstance(value, str) and value:
                    texts.append(value)

    return "\n".join(texts)


def normalize_question_terms(question: str) -> str:
    """Rewrite informal phrases to the formal section names used in the policy."""

    normalized = question
    for alias, section in SECTION_ALIASES.items():
        if alias in normalized and section not in normalized:
            normalized = normalized.replace(alias, section)
    return normalized


def last_assistant_message_is_clarification(history: Sequence[Any] | None) -> bool:
    """Check whether the last assistant turn was asking the user to pick a section."""

    if not history:
        return False

    for item in reversed(list(history)):
        if isinstance(item, dict):
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            return isinstance(content, str) and (
                "請問你想問的是哪一種保險" in content or "請問你想問的是哪一種？" in content
            )

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            assistant_message = item[1]
            if isinstance(assistant_message, str) and assistant_message:
                return "請問你想問的是哪一種保險" in assistant_message or "請問你想問的是哪一種？" in assistant_message

    return False


def last_assistant_message_is_delay_clarification(history: Sequence[Any] | None) -> bool:
    """Check whether the last assistant turn was asking the user to clarify the delay type."""

    if not history:
        return False

    for item in reversed(list(history)):
        if isinstance(item, dict):
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            return isinstance(content, str) and "延誤賠償" in content and "班機延誤保險" in content

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            assistant_message = item[1]
            if isinstance(assistant_message, str) and assistant_message:
                return "延誤賠償" in assistant_message and "班機延誤保險" in assistant_message

    return False


def is_ambiguous_section_question(question: str) -> bool:
    """Return True when the user asks about exclusions without naming a section."""

    ambiguous_keywords = (
        "不可理賠",
        "不理賠",
        "不能賠",
        "可理賠",
        "不保事項",
        "除外責任",
        "哪些原因",
        "哪些情況",
    )
    return any(keyword in question for keyword in ambiguous_keywords)


def is_ambiguous_delay_question(question: str) -> bool:
    """Return True when the user asks about delay compensation without naming the delay type."""

    delay_keywords = ("旅遊延誤", "延誤賠償", "旅遊延誤賠償")
    explicit_delay_sections = ("班機延誤", "航班延誤", "行李延誤")
    return any(keyword in question for keyword in delay_keywords) and not any(
        section in question for section in explicit_delay_sections
    )


def build_section_clarification_message() -> str:
    """Ask the user to clarify which insurance section they mean."""

    options = "\n".join(f"- {section}" for section in INSURANCE_SECTIONS)
    return (
        "你的問題目前還不夠具體，因為條款裡有很多不同保險項目的不保事項。\n"
        "請問你想問的是哪一種保險？例如：\n"
        f"{options}\n\n"
        "你也可以直接問，例如：「班機延誤保險有哪些不可理賠範圍？」"
    )


def build_delay_clarification_message() -> str:
    """Ask the user to clarify which type of delay they mean."""

    return (
        "你提到的是「延誤賠償」，但條款裡可能是不同類型的延誤。\n"
        "請問你想問的是哪一種？\n"
        "- 班機延誤保險\n"
        "- 行李延誤保險\n\n"
        "你也可以直接問，例如：「班機延誤保險什麼情況下可以申請理賠？」"
    )
