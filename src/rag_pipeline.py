"""RAG pipeline glue: Question → Retriever → LLM → Answer (with sources).

This module wires together:
- a retriever (currently semantic vector search over policy chunks)
- an LLM for answer generation

The goal is to keep the core flow simple:

    User Question → Retriever → Top‑k Chunks → LLM Answer (with clause citations)
"""

from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI

from config import LLM_MODEL, RETRIEVER_TOP_K
from retrievers.semantic import get_semantic_retriever


class RAGPipeline:
    """Simple RAG pipeline for the travel insurance policy."""

    def __init__(self, k: int | None = None) -> None:
        k = k or RETRIEVER_TOP_K
        self.retriever = get_semantic_retriever(k=k)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def answer(self, question: str) -> str:
        """Retrieve relevant clauses and generate an answer with citations."""

        # 1. Retrieve top‑k chunks
        docs = self.retriever.get_relevant_documents(question)

        if not docs:
            return "目前找不到與條款內容相符的資訊，建議洽詢客服或查看正式保單。"

        # 2. Build context string with clause IDs and text
        context_lines: List[str] = ["條款內容："]
        for d in docs:
            article_id = d.metadata.get("article_id", "(未標註條款編號)")
            context_lines.append(f"[{article_id}]\n{d.page_content}\n")

        context = "\n".join(context_lines)

        # 3. Compose prompt
        prompt = f"""
你是一位保險專員，負責解釋「旅遊不便險」的條款內容。
你只能根據下列條款內容回答問題，不要加入條款以外的推測。
若條款未明確規範，請說明「條款未明確規範，建議洽詢客服或查看正式保單」。

請用條理清楚的中文回答，並在最後列出你引用的條款編號與關鍵原文。

{context}

問題：{question}
""".strip()

        resp = self.llm.invoke(prompt)
        return resp.content
