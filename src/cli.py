"""Simple CLI demo for the Cathay travel insurance RAG prototype.

Usage (from project root):

    # Activate your venv and set OPENAI_API_KEY, then:
    python -m src.cli
"""

from __future__ import annotations

from rag_pipeline import RAGPipeline


def main() -> None:
    rag = RAGPipeline(k=5)

    print("Cathay Travel Insurance RAG Demo")
    print("輸入問題，按 Enter 送出（空行離開）。\n")

    while True:
        try:
            q = input("問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            print("Bye.")
            break

        ans = rag.answer(q)
        print("\n=== 回答 ===")
        print(ans)
        print("\n")


if __name__ == "__main__":
    main()
