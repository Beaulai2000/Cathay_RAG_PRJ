# Cathay Travel Insurance RAG Workflow

> Status: updated on 2026-03-09.

This file is the working memory for the project. It records:

- what the current architecture does
- why specific design decisions were made
- how to run ingestion / CLI / Gradio / evaluation scripts
- what configuration is currently recommended

## 0. Change summary (code-only, for quick recap)

This summary intentionally excludes:

- presentation assets (`assets/*.svg`, `assets/*.png`)
- slide/template generators for PPT
- test/eval output files under `data/evals/*.json` and `data/evals/*.md`

Core code changes completed so far:

1. Project execution flow was standardized to module mode:
   - `python -m src.ingestion`
   - `python -m src.cli`
   - `python -m src.gradio_app`
   - internal imports were updated to package-relative style.

2. Configuration was centralized in [`src/config.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/config.py):
   - model defaults
   - chunking parameters
   - retriever top-k
   - chat history window
   - section names and alias mapping.

3. Ingestion was upgraded in [`src/ingestion.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/ingestion.py):
   - switched from naive paragraph split to article-aware chunking
   - added metadata (`chunk_id`, `article_id`, `section`)
   - added chunk preview logging
   - added model-specific index directories and `index_meta.json`.

4. Retrieval logic was improved in [`src/retrievers/semantic.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/retrievers/semantic.py):
   - supports section filter
   - uses current retriever API (`invoke`)
   - resolves index path by embedding model
   - returns actionable rebuild instructions when index is missing.

5. RAG pipeline was expanded in [`src/rag_pipeline.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/rag_pipeline.py):
   - system/user chat-role structure
   - multi-turn history support
   - section-aware query routing
   - ambiguous question clarification (section and delay type)
   - follow-up rewrite for short replies
   - synonym normalization
   - embedding/index mismatch handling.

6. User-facing entrypoints were aligned:
   - CLI in [`src/cli.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/cli.py)
   - Gradio UI in [`src/gradio_app.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/gradio_app.py) now uses a Blocks layout with runtime parameter controls.

7. Default serving configuration was finalized as:
   - `gpt-4.1`
   - `text-embedding-3-small`
   - `chunk_size=700`, `overlap=100`.

## 1. Project goal

Build a RAG chatbot for the Cathay travel inconvenience insurance policy that:

- answers policy questions in Chinese
- cites the relevant clause headings
- handles ambiguous user phrasing better than a naive single-turn semantic search

## 2. Current high-level flow

1. Extract and clean policy text
2. Chunk the cleaned policy with article-aware chunking
3. Build a Chroma vector index
4. Retrieve relevant chunks with optional section filtering
5. Generate an answer with an OpenAI chat model
6. Expose the flow through CLI and Gradio

## 3. Current default production config

The current default configuration in [`src/config.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/config.py) is:

- `LLM_MODEL = "gpt-4.1"`
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `CHUNK_SIZE = 700`
- `CHUNK_OVERLAP = 100`
- `RETRIEVER_TOP_K = 5`
- `CHAT_HISTORY_WINDOW = 3`

Why this config:

- `gpt-4.1` gave the best answer quality / cost balance in the current evals
- `text-embedding-3-small` was enough for this single-policy knowledge base
- `700 / 100` was the most balanced chunk setup across the benchmark runs

Low-cost fallback:

- `gpt-4o-mini + text-embedding-3-small + 700 / 100`

## 4. Key files

- [`src/config.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/config.py)
  Central configuration for models, chunking, retrieval, sections, and paths.

- [`src/ingestion.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/ingestion.py)
  Reads the cleaned policy text, splits it into chunks, assigns metadata, and builds the Chroma index.

- [`src/retrievers/semantic.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/retrievers/semantic.py)
  Loads the semantic retriever with optional section filtering.

- [`src/rag_pipeline.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/rag_pipeline.py)
  Main RAG orchestration: query normalization, clarification, retrieval, history handling, and answer generation.

- [`src/gradio_app.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/gradio_app.py)
  Gradio chat UI.

- [`src/cli.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/cli.py)
  CLI entrypoint.

## 5. Ingestion and indexing

### 5.1 Chunking strategy

The project does not primarily rely on blank-paragraph chunking anymore.

Current approach:

- first split by article headings such as `第二十七條 ...`
- keep short clauses intact
- sub-split long clauses line by line with overlap

Metadata stored per chunk:

- `chunk_id`
- `article_id`
- `section`

### 5.2 Model-specific index directories

Indexes are now stored per embedding model under [`data/index`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/data/index).

Examples:

- `data/index/text_embedding_3_small/`
- `data/index/text_embedding_3_large/`

Reason:

- evaluation scripts were rebuilding the same Chroma directory with different embedding dimensions
- that caused runtime failures such as `Collection expecting embedding with dimension of 3072, got 1536`
- separating indexes by embedding model prevents eval runs from breaking the default Gradio app

Each built index also writes a small `index_meta.json` file with:

- embedding model
- chunk size
- overlap
- chunk count

### 5.3 Build command

Default build:

```bash
python -m src.ingestion
```

Build for another embedding model:

```bash
CATHAY_RAG_EMBEDDING_MODEL="text-embedding-3-large" python -m src.ingestion
```

## 6. Retrieval design

The retriever is semantic search over Chroma with top-k retrieval.

Current retriever behavior:

- loads the index matching the active embedding model
- can apply `filter={"section": ...}` when a section is inferred
- uses `.invoke(query)` with the current LangChain retriever API

If the correct index does not exist, the retriever now returns a clear rebuild instruction instead of failing with an opaque traceback.

## 7. RAG pipeline behavior

[`src/rag_pipeline.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/rag_pipeline.py) now does more than naive retrieve-then-answer.

### 7.1 Query normalization

The pipeline rewrites informal terms into the policy's formal section names.

Examples:

- `信用卡盜刷` -> `信用卡盜用保險`
- `護照遺失` -> `旅行文件損失保險`
- `手機被偷` -> `行動電話被竊損失保險`

### 7.2 Section-aware retrieval

If the user clearly refers to a specific policy section, retrieval is filtered to that section.

This reduces cross-section contamination for questions like:

- `信用卡盜用有哪些不可理賠範圍？`
- `特殊活動取消慰問保險的承保範圍是什麼？`

### 7.3 Clarification for ambiguous questions

The pipeline does not always answer immediately.

It first asks for clarification when the question is too broad, for example:

- `哪些原因屬於不可理賠範圍？`
- `旅遊延誤賠償怎麼算？`

There are currently two clarification types:

- generic section clarification
- delay clarification (`班機延誤保險` vs `行李延誤保險`)

### 7.4 Follow-up rewrite

If the user only replies with a section name after a clarification question, the pipeline rewrites that short reply into a full retrieval query.

Examples:

- user: `哪些原因屬於不可理賠範圍？`
- assistant: asks which insurance section
- user: `信用卡盜用保險`
- rewritten query: `信用卡盜用保險有哪些不可理賠範圍？`

### 7.5 Conversation history

The pipeline accepts `history` and injects the recent turns into the chat messages sent to the LLM.

History length can be configured in two ways:

- default from `CHAT_HISTORY_WINDOW` in config
- overridden at runtime via `RAGPipeline(history_window=...)`

This supports:

- follow-up clarification
- history-aware answers
- multi-turn chat in Gradio

### 7.6 Error handling

The pipeline now catches index/model mismatch cases and returns a rebuild instruction instead of exposing a raw Chroma dimension traceback to the user.

## 8. CLI and Gradio

CLI:

```bash
python -m src.cli
```

Gradio:

```bash
python -m src.gradio_app
```

Notes:

- Gradio currently launches with `share=True`
- UI is implemented with `gr.Blocks` instead of a plain `ChatInterface`
- runtime controls are available for:
  - `k`
  - `history_window`
  - `chunk_size`
  - `overlap`
- presets can be applied with one click (`Baseline`, `Precision`, `Recall`, etc.)
- index rebuild can be triggered from the UI when changing `chunk_size` / `overlap`
- `chat_fn` passes `history` and runtime parameters into the pipeline

## 9. Evaluation workflow

### 9.1 Chunk-only sweep

[`src/evaluate_chunk_configs.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/evaluate_chunk_configs.py)

Tests:

- fixed 5 questions
- chunk configs:
  - `500 / 100`
  - `700 / 100`
  - `900 / 120`

Outputs:

- `data/evals/chunk_eval_*.json`
- `data/evals/chunk_eval_*.md`

### 9.2 Model + chunk sweep

[`src/evaluate_model_chunk_configs.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/evaluate_model_chunk_configs.py)

Tests:

- 6 model combinations
- 3 chunk configurations

Current model combinations:

- `gpt-4o-mini + text-embedding-3-small`
- `gpt-4o-mini + text-embedding-3-large`
- `gpt-4.1 + text-embedding-3-small`
- `gpt-4.1 + text-embedding-3-large`
- `gpt-5 + text-embedding-3-small`
- `gpt-5 + text-embedding-3-large`

Outputs:

- `data/evals/model_chunk_eval_*.json`
- `data/evals/model_chunk_eval_*.md`

### 9.3 Harder v2 benchmark

[`src/evaluate_model_chunk_configs_v2.py`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/src/evaluate_model_chunk_configs_v2.py)

Tests:

- ambiguous queries
- clarification behavior
- follow-up rewriting
- synonym handling
- condition / exception questions

Plan doc:

- [`data/evals/model_chunk_eval_v2_plan.md`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/data/evals/model_chunk_eval_v2_plan.md)

Analysis docs:

- [`data/evals/model_chunk_eval_analysis_v1.md`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/data/evals/model_chunk_eval_analysis_v1.md)
- [`data/evals/model_chunk_eval_analysis_v2.md`](/Users/laipoyu/Desktop/LLM_Projects/Cathay_Rag_PRJ/Cathay_RAG_PRJ/data/evals/model_chunk_eval_analysis_v2.md)

## 10. Current benchmark conclusions

From the current v1 and v2 evals:

- `700 / 100` is still the best default chunk setup
- `text-embedding-3-large` did not show enough gain to justify becoming the default
- `gpt-4.1` is the best current main model for answer quality / cost balance
- `gpt-4o-mini` remains the cheapest usable baseline
- much of the system quality now comes from pipeline logic, not just from swapping to bigger models

## 11. Known next experiments

Good next steps:

- test `k = 3 / 5 / 7`
- create a tougher v3 benchmark that stresses retrieval failures and citation accuracy
- make citations more deterministic from retrieved metadata instead of relying on the model
- consider query rewriting with a dedicated LLM step before retrieval
