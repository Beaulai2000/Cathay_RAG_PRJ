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

from .config import CHAT_HISTORY_WINDOW, CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVER_TOP_K
from .ingestion import build_index
from .rag_pipeline import RAGPipeline

PRESET_CONFIGS = {
    "Baseline（推薦）": {
        "k": 5,
        "chunk_size": 700,
        "overlap": 100,
        "history_window": 3,
        "rebuild": False,
    },
    "Precision（k=3）": {
        "k": 3,
        "chunk_size": 700,
        "overlap": 100,
        "history_window": 3,
        "rebuild": False,
    },
    "Recall（k=8）": {
        "k": 8,
        "chunk_size": 700,
        "overlap": 100,
        "history_window": 3,
        "rebuild": False,
    },
    "Long Context（history=5）": {
        "k": 5,
        "chunk_size": 700,
        "overlap": 100,
        "history_window": 5,
        "rebuild": False,
    },
    "Small Chunk（500/100，重建）": {
        "k": 5,
        "chunk_size": 500,
        "overlap": 100,
        "history_window": 3,
        "rebuild": True,
    },
    "Large Chunk（900/120，重建）": {
        "k": 5,
        "chunk_size": 900,
        "overlap": 120,
        "history_window": 3,
        "rebuild": True,
    },
}


def build_pipeline(k: int, history_window: int) -> RAGPipeline:
    return RAGPipeline(k=k, history_window=history_window)


def chat_fn(message: str, history: list[dict[str, str]] | None, runtime_params: dict):
    """Append one chat turn using the current pipeline state."""

    history = history or []

    if not message.strip():
        return history, ""

    pipeline = build_pipeline(
        k=int(runtime_params.get("k", RETRIEVER_TOP_K)),
        history_window=int(runtime_params.get("history_window", CHAT_HISTORY_WINDOW)),
    )
    answer = pipeline.answer(message, history=history)
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return new_history, ""


def apply_parameters(
    k: int,
    chunk_size: int,
    overlap: int,
    history_window: int,
    rebuild_index_flag: bool,
    _runtime_params: dict,
):
    """Apply retrieval/chat parameters and optionally rebuild the index."""

    k = int(k)
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    history_window = int(history_window)

    try:
        status_lines = [
            "[OK] 參數已更新",
            f"- k: {k}",
            f"- history_window: {history_window}",
            f"- chunk_size: {chunk_size}",
            f"- overlap: {overlap}",
        ]

        if rebuild_index_flag:
            build_index(chunk_size=chunk_size, overlap=overlap, preview_limit=0)
            status_lines.append("- 索引：已依 chunk_size/overlap 重建")
        else:
            status_lines.append("- 索引：未重建（chunk_size/overlap 不會生效）")

        new_runtime_params = {
            "k": k,
            "history_window": history_window,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        return "\n".join(status_lines), new_runtime_params
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] 套用失敗：{exc}", _runtime_params


def apply_preset(preset_name: str, runtime_params: dict):
    """Apply one preset and sync all controls."""

    preset = PRESET_CONFIGS.get(preset_name)
    if preset is None:
        return (
            RETRIEVER_TOP_K,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            CHAT_HISTORY_WINDOW,
            False,
            "[ERROR] 找不到 preset",
            runtime_params,
        )

    status, new_runtime_params = apply_parameters(
        k=preset["k"],
        chunk_size=preset["chunk_size"],
        overlap=preset["overlap"],
        history_window=preset["history_window"],
        rebuild_index_flag=preset["rebuild"],
        _runtime_params=runtime_params,
    )
    return (
        preset["k"],
        preset["chunk_size"],
        preset["overlap"],
        preset["history_window"],
        preset["rebuild"],
        f"[Preset] {preset_name}\n{status}",
        new_runtime_params,
    )


def main() -> None:
    with gr.Blocks(title="Cathay Travel Insurance RAG") as demo:
        runtime_state = gr.State(
            {
                "k": RETRIEVER_TOP_K,
                "history_window": CHAT_HISTORY_WINDOW,
                "chunk_size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
            }
        )

        gr.Markdown(
            "## Cathay Travel Insurance RAG\n"
            "使用旅遊不便險條款作為知識庫的 RAG Chatbot。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="對話視窗",
                    height=620,
                )
                message_box = gr.Textbox(
                    label="問題",
                    placeholder="輸入問題，例如：什麼情況下可以申請旅遊延誤賠償？",
                )
                with gr.Row():
                    send_btn = gr.Button("送出", variant="primary")
                    clear_btn = gr.Button("清除對話")

                gr.Examples(
                    examples=[
                        "什麼情況下可以申請旅遊延誤賠償？",
                        "行李遺失後應該如何申請理賠？",
                        "哪些原因屬於不可理賠範圍？",
                    ],
                    inputs=message_box,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 參數控制")

                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_CONFIGS.keys()),
                    value="Baseline（推薦）",
                    label="快速 Preset",
                )
                preset_apply_btn = gr.Button("一鍵套用 Preset")

                k_slider = gr.Slider(minimum=1, maximum=12, value=RETRIEVER_TOP_K, step=1, label="k（檢索 chunk 數）")
                history_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=CHAT_HISTORY_WINDOW,
                    step=1,
                    label="history_window（保留對話輪數）",
                )
                chunk_slider = gr.Slider(
                    minimum=300,
                    maximum=1200,
                    value=CHUNK_SIZE,
                    step=50,
                    label="chunk_size（重建索引用）",
                )
                overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=CHUNK_OVERLAP,
                    step=10,
                    label="overlap（重建索引用）",
                )

                rebuild_checkbox = gr.Checkbox(
                    value=False,
                    label="套用時重建索引（chunk_size/overlap 變更必勾）",
                )
                apply_btn = gr.Button("套用參數", variant="primary")
                status_box = gr.Textbox(
                    label="參數狀態",
                    lines=8,
                    value=(
                        "[Init] 預設參數\n"
                        f"- k: {RETRIEVER_TOP_K}\n"
                        f"- history_window: {CHAT_HISTORY_WINDOW}\n"
                        f"- chunk_size: {CHUNK_SIZE}\n"
                        f"- overlap: {CHUNK_OVERLAP}"
                    ),
                    interactive=False,
                )

                gr.Markdown(
                    "提示：\n"
                    "- 改 `k` / `history_window` 可直接生效。\n"
                    "- 改 `chunk_size` / `overlap` 要勾選「重建索引」才會生效。\n"
                    "- 你也可用「一鍵套用 Preset」快速切換測試組合。"
                )

        send_btn.click(
            chat_fn,
            inputs=[message_box, chatbot, runtime_state],
            outputs=[chatbot, message_box],
        )
        message_box.submit(
            chat_fn,
            inputs=[message_box, chatbot, runtime_state],
            outputs=[chatbot, message_box],
        )
        clear_btn.click(lambda: [], outputs=chatbot, queue=False)

        apply_btn.click(
            apply_parameters,
            inputs=[
                k_slider,
                chunk_slider,
                overlap_slider,
                history_slider,
                rebuild_checkbox,
                runtime_state,
            ],
            outputs=[status_box, runtime_state],
        )
        preset_apply_btn.click(
            apply_preset,
            inputs=[preset_dropdown, runtime_state],
            outputs=[
                k_slider,
                chunk_slider,
                overlap_slider,
                history_slider,
                rebuild_checkbox,
                status_box,
                runtime_state,
            ],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
