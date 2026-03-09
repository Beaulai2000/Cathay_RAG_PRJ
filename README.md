# Cathay Travel Insurance RAG

這是一個針對「國泰旅遊不便險條款」打造的 RAG 專案。  
目標是讓使用者用自然語言提問，系統能根據條款內容回答，並附上條款依據。

## 1. 目前你已完成的核心內容

你目前不是只有做基本 RAG，而是已經完成一個可 demo、可評估、可調參的版本：

- 條款抽取與清洗流程（PDF -> cleaned text）
- 條文感知切塊（article-aware chunking）
- 向量索引建立（Chroma + OpenAI embeddings）
- section-aware retrieval（可依險種過濾）
- 多輪對話支援（history）
- 模糊問題澄清（先問險種/延誤類型再回答）
- follow-up rewrite（使用者只回險種時自動補完整問題）
- 同義詞正規化（例：信用卡盜刷 -> 信用卡盜用保險）
- Gradio UI（含可調 `k/chunk/history` 與 preset）
- 系統化評估腳本（chunk/model/對話情境/retrieval-level）

## 2. 專案流程

```text
Raw Policy -> Cleaned Text -> Chunking -> Chroma Index
                                      -> Retriever (top-k + section filter)
                                      -> LLM (ChatOpenAI)
                                      -> Answer + Clause Citations
```

## 3. 專案結構

```text
src/
  config.py                         # 模型、chunk、retrieval、路徑設定
  extract_policy_from_pdf.py        # PDF 抽取與清洗
  ingestion.py                      # chunking + 建立向量索引
  rag_pipeline.py                   # 核心 RAG 與對話控制
  gradio_app.py                     # Gradio 介面（含參數控制）
  cli.py                            # CLI 問答
  retrievers/
    semantic.py                     # Chroma retriever
  evaluate_chunk_configs.py         # chunk sweep
  evaluate_model_chunk_configs.py   # model + chunk sweep
  evaluate_model_chunk_configs_v2.py# v2 多輪/模糊問題 benchmark
  evaluate_retrieval_v2.py          # retrieval-level 評估

data/
  policy_clean/policy_clean.txt
  evals/
```

## 4. 預設設定（目前主力）

定義在 [`src/config.py`](src/config.py)：

- `LLM_MODEL = "gpt-4.1"`
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `CHUNK_SIZE = 700`
- `CHUNK_OVERLAP = 100`
- `RETRIEVER_TOP_K = 5`
- `CHAT_HISTORY_WINDOW = 3`

## 5. 快速開始

1. 安裝依賴

```bash
pip install -r requirements.txt
```

2. 設定 API key

```bash
export OPENAI_API_KEY="your_key"
```

3. 建立索引

```bash
python -m src.ingestion
```

4. CLI 問答

```bash
python -m src.cli
```

5. Gradio 問答

```bash
python -m src.gradio_app
```

## 6. Gradio 參數控制

Gradio 目前可直接調：

- `k`（top-k chunks）
- `history_window`（對話保留輪數）
- `chunk_size / overlap`（可選擇重建索引）

並提供 preset：

- Baseline（推薦）
- Precision（k=3）
- Recall（k=8）
- Long Context（history=5）
- Small Chunk（500/100，需重建）
- Large Chunk（900/120，需重建）

## 7. 評估腳本

1. chunk 參數掃描

```bash
python -m src.evaluate_chunk_configs
```

2. model + chunk 組合掃描

```bash
python -m src.evaluate_model_chunk_configs
```

3. v2 對話/模糊問題 benchmark

```bash
python -m src.evaluate_model_chunk_configs_v2
```

4. retrieval-level evaluation（先看抓到的 chunks 對不對）

```bash
python -m src.evaluate_retrieval_v2
```

輸出會寫到 `data/evals/`。

## 8. 常見問題

### `Collection expecting embedding with dimension ...` 錯誤

代表「目前 embedding model」和「既有 index 維度」不一致。  
先重建索引即可：

```bash
python -m src.ingestion
```

若你切換 embedding model，請用該 model 重建一次索引。

## 9. 現況結論（依目前 benchmark）

- 目前最平衡組合：`gpt-4.1 + text-embedding-3-small + 700/100`
- 低成本 baseline：`gpt-4o-mini + text-embedding-3-small + 700/100`
- 你的品質提升主要來自 pipeline 設計（澄清/rewrite/section routing），不只是換更大模型

## 10. 相關文件

- 完整工程工作紀錄：[`CATHAY_RAG_WORKFLOW.md`](CATHAY_RAG_WORKFLOW.md)
