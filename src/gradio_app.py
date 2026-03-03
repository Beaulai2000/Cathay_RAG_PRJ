"""Gradio UI for the Cathay travel insurance RAG prototype.

This is similar in spirit to the RAG interview coach Gradio app:
- left side: chat history
- right side: model answers based on policy clauses

Prereqs:
- You have built the vector index (see ingestion.build_index).
- OPENAI_API_KEY is set in your environment.

Run from project root (after activating venv):

    python -m src.gradio_app
"""

from __future__ import annotations

import gradio as gr

from rag_pipeline import RAGPipeline


def build_pipeline() -> RAGPipeline:
    # You can tweak k here if needed
    return RAGPipeline(k=5)


PIPELINE = build_pipeline()


def chat_fn(message: str, history: list[tuple[str, str]]) -> str:
    """Gradio chat handler.

    We ignore the history for now and treat each question independently.
    Later you can extend RAGPipeline to use multi‑turn context if desired.
    """

    if not message.strip():
        return "請輸入問題，例如：什麼情況下可以申請旅遊延誤賠償？"

    answer = PIPELINE.answer(message)
    return answer


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Cathay Travel Insurance RAG",
        description=(
            "使用旅遊不便險條款作為知識庫的 RAG Chatbot。"\
            " 問題範例：\n"
            "- 什麼情況下可以申請旅遊延誤賠償？\n"
            "- 行李遺失後應該如何申請理賠？\n"
            "- 哪些原因屬於不可理賠範圍？"
        ),
    )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
