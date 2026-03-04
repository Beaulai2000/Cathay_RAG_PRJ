# Cathay Travel Insurance RAG Prototype

This project is a Retrieval‑Augmented Generation (RAG) prototype built on top of a travel inconvenience insurance policy document.

The goal is to:

1. Ingest the policy clauses (travel delay, baggage loss, exclusions, etc.).
2. Build a retriever over the policy text.
3. Allow a user to ask natural‑language questions (e.g. "在什麼情況下可以申請旅遊延誤賠償？")
4. Use the retriever + LLM to generate answers **with explicit clause citations**.

---

## High‑level flow (current version)

```text
User Question
     ↓
Retriever (vector search)
     ↓
Top‑k Chunks (policy clauses)
     ↓
LLM generates answer (with sources)
```

Later we can extend this to an agent‑style flow:

```text
User Question
     ↓
Agent (LLM reasoning)
     ↓
Decide: do we need to consult the policy?
     ↓
Retriever(s)
     ↓
Agent integrates retrieved evidence
     ↓
Final answer
```

The current prototype focuses on the simpler RAG pipeline, but the design leaves room to plug in more sophisticated retrievers and agent logic later.

---

## Repository layout

```text
Cathay_RAG_PRJ/
  ├─ data/
  │   ├─ policy_raw/       # raw policy docs (PDF/Word) – not under version control
  │   └─ policy_clean/     # cleaned text / preprocessed files (e.g. policy_clean.txt)
  ├─ notebooks/
  │   └─ exploration.ipynb # optional: for manual experiments (chunking, prompts…)
  ├─ src/
  │   ├─ config.py         # model + path configuration (reads API key from env)
  │   ├─ ingestion.py      # policy ingestion + chunking
  │   ├─ retrievers/
  │   │   ├─ semantic.py   # embedding‑based semantic retriever (current baseline)
  │   │   ├─ bm25.py       # keyword/BM25 retriever (future work)
  │   │   └─ hybrid.py     # hybrid retriever combining semantic + keyword (future)
  │   ├─ rag_pipeline.py   # glue code: User Question → Retriever → LLM → Answer
  │   └─ cli.py            # simple CLI interface for demo/testing
  ├─ requirements.txt
  └─ README.md
```

> Note: policy documents themselves (e.g. the actual PDF/Word travel insurance policy) should **not** be committed if they are confidential. Use placeholders or local files under `data/policy_raw/`.

---

## 1. Ingestion & chunking (`src/ingestion.py`)

The ingestion step takes the raw policy document and turns it into a set of **chunks** suitable for retrieval.

### 1.1 Cleaning

- Convert the original policy (PDF/Word) into a plain text file, e.g. `policy_clean.txt`.
- Remove page headers/footers, page numbers, and formatting artefacts.
- Preserve:
  - Article/section titles (e.g. "旅遊延誤保障", "行李損失", "不保事項").
  - Clause IDs (e.g. "第 3 條", "3-1", "第八條" 等)。

### 1.2 Chunking strategy

For the first version we can use a simple **paragraph‑aware chunking**:

- Base chunk size: ~300–500 characters (or words) with ~50–100 overlap.
- Use article/section boundaries as preferred split points (don\'t mix unrelated clauses in one chunk).

Possible future improvements:

- **Recursive character splitter**: start with larger blocks (articles), recursively split into smaller units if too long.
- **Semantic chunking**: group sentences based on semantic similarity so that each chunk is topically coherent.

Each chunk should be stored together with metadata, e.g.:

- `article_id` – clause/section identifier
- `section` – high‑level category (travel delay, baggage, exclusions…)
- `source` / `page` – optional reference back to the original document

This metadata will be used later by the retriever and for citing sources in the answer.

---

## 2. Retriever design (`src/retrievers/`)

The retriever is responsible for finding relevant chunks given a user question.

### 2.1 Semantic vector retriever (baseline)

- Use an embedding model such as `text-embedding-3-small` to embed each chunk.
- Store embeddings + metadata in a vector store (e.g. ChromaDB).
- At query time:
  1. Embed the user question.
  2. Run a similarity search (cosine similarity) to obtain top‑k relevant chunks.

This is the **current baseline** and is usually enough to get a working RAG prototype.

### 2.2 Keyword / BM25 retriever (future work)

- Implement a keyword/BM25 retriever (e.g. using `rank-bm25` or a search engine like Elasticsearch).
- Strengths:
  - Very precise when the policy uses concrete terms (e.g. "不保事項", "除外責任").
  - Good at "hard matching" specific phrases.

Potential uses:

- First filter clauses that contain key terms (e.g. "行李遺失", "延誤", "不保").
- Then apply semantic ranking on the filtered set.

### 2.3 Hybrid retriever (future work)

Combine semantic and keyword signals:

- Either via **score combination**:
  - `score = α * semantic_score + (1-α) * keyword_score`
- Or via a **two-stage retrieval**:
  - Stage 1: keyword/BM25 filter by relevant sections.
  - Stage 2: semantic ranking within those sections.

Motivation:

- Semantic retrieval helps with paraphrases (e.g. "飛機晚點" vs "航班延誤").
- Keyword retrieval ensures we don\'t miss critical phrases that must be present.

### 2.4 Metadata filtering

Use metadata to narrow down the search space before vector search, for example:

- Filter by `section` (travel delay / baggage / exclusions).
- Filter by `version` / `effective_date` when multiple policy versions exist.

This becomes important when the same system supports multiple products or policy versions.

---

## 3. Answer generation (`src/rag_pipeline.py`)

Given a user question and top‑k chunks from the retriever, we use an LLM to generate an answer **with sources**.

### 3.1 Prompt design

- **System prompt**:
  - Define the assistant role, e.g. "You are an assistant that explains a travel inconvenience insurance policy." 
  - Instructions:
    - Only answer based on the provided clauses.
    - If information is not found in the clauses, say so (e.g. "條款未明確規範，建議洽詢客服") instead of guessing.
    - Always include the clause IDs and key excerpts used as evidence.

- **Context**:
  - Inject the top‑k chunks with their clause IDs and titles, e.g.:
    ```text
    條款內容：
    [第 3 條 旅遊延誤保障]
    ...條文...

    [第 8 條 不保事項]
    ...條文...
    ```

- **User prompt**: the original user question in Chinese.

### 3.2 Response format

Encourage the model to respond in two parts:

1. A natural language answer:
   ```text
   回答：
   若因航空公司可歸責因素導致航班延誤超過 X 小時，且符合條款規定之條件，您可以申請旅遊延誤賠償…
   ```

2. Cited clauses:
   ```text
   引用條款：
   - 第 3 條 第 1 項：「…」
   - 第 3 條 第 2 項：「…」
   ```

This makes the system more transparent and safer for an insurance use case.

### 3.3 Future improvements

- **Answer verification**: use a second LLM or rule‑based check to ensure the answer does not contradict the retrieved clauses.
- **Multi‑turn context**: carry forward important parameters (e.g. trip dates, destination) across turns and include them in the prompt.

---

## 4. Limitations & improvement directions

Some important limitations and potential improvements (for the assignment and interviews):

- **Hallucinations**: even with RAG, LLMs may still invent details. Mitigation: strong prompt constraints, explicit citation of clauses, and possibly an answer verification step.
- **Chunking sensitivity**: retrieval quality heavily depends on how we chunk the policy. Future work includes paragraph‑aware, semantic, and recursive chunking.
- **Policy versioning**: real insurance policies change over time. Metadata (product ID, version, effective date) and index rebuild pipelines are needed in production.
- **Retrieval recall vs precision**: semantic vs keyword vs hybrid retrieval trade‑offs.
- **Legal / compliance**: the chatbot should not be treated as a legal authority; answers should explicitly defer to official policy and customer service when unsure.

---

## 5. Running the prototype (to be implemented)

Once the ingestion and retrieval modules are implemented, the basic demo flow will be:

1. Build the index:
   ```bash
   python -m src.ingestion
   ```

2. Start a CLI chat:
   ```bash
   python -m src.cli
   ```

3. Ask questions such as:
   - "什麼情況下可以申請旅遊延誤賠償？"
   - "行李遺失後應該如何申請理賠？"
   - "哪些原因屬於不可理賠範圍？"

The system will retrieve relevant clauses and generate answers with clause citations.

---

## 6. Intent / Section Detection and Query Rewriting (Future Work)

As the project evolves from a simple single-turn RAG into a more conversational, product-like assistant, there are three important layers to think about:

### 6.1 Intent / section detection

For travel insurance, many user questions其實都在問「哪一個險種」或「哪一類資訊」，例如：

- 班機延誤
- 行李延誤
- 行李損失
- 信用卡盜用
- 不保事項（除外責任）
- 理賠文件
- 承保範圍

在未來版本中，可以在 RAG pipeline 前加上一層 **intent / section detection**，做的事情包括：

- 從自然語言問題中判斷：
  - 這一題主要在問哪一個保險項目（section）
  - 使用者想知道的是：
    - 承保範圍嗎？
    - 不保事項 / 除外責任嗎？
    - 理賠流程 / 理賠文件嗎？
- 再根據這個意圖，選擇或過濾對應的條款區域（例如只檢索「班機延誤保險」相關的 chunks）。

這可以採用兩種方式：

- 規則 + alias mapping：
  - 例如把「飛機晚點」「航班延遲」都 map 成「班機延誤保險」。
- LLM classifier：
  - 給模型一小段 system prompt，請它輸出：`section` + `query_type`（承保範圍 / 不保事項 / 理賠文件…）。

### 6.2 Query rewriting（查詢重寫）

在保險條款這種正式用語很多的文件中，**直接拿使用者原始問句去做 semantic search 不一定理想**。更穩的做法是：

1. 先把使用者的口語問題改寫成「比較接近保單說法」的查詢。
2. 再用這個改寫後的查詢送進 retriever。

舉例：

- 原始問題：
  - 「什麼情況下可以申請旅遊延誤賠償？」
- 改寫後（更接近保單語氣）：
  - 「班機延誤保險的承保範圍是什麼？什麼情況下可申請理賠？」

RAG flow 會變成：

```text
User Question
   ↓
(1) 規則 / LLM 協助的 query rewrite
   ↓
Rewritten Query（正式、明確、對齊條款用語）
   ↓
Retriever
   ↓
Top‑k Chunks
   ↓
LLM Answer（with sources）
```

這一層可以分成三種強度：

1. **規則式 preprocessing（現在已開始做）**
   - alias mapping（把口語對應到正式險種名稱）。
   - section clarification（問題不清楚時請使用者選：班機延誤 / 行李延誤 / …）。
   - follow‑up rewrite（使用者只回「班機延誤保險」，自動補成完整問題）。

   優點：穩定、可控、容易 debug。
   缺點：規則要慢慢累積。

2. **LLM‑assisted query rewriting（推薦的下一步）**
   - 在 retriever 前加一個小小的 `rewrite_query(question, history)`：
     - 先讓 LLM 看歷史對話與當前問句。
     - 產生一個更正式、更清楚、更接近條款用語的 rewritten query。
   - 然後用 rewritten query 去做 semantic retrieval。

   特別對這些情況有幫助：
   - 模糊問法（「這樣也可以理賠嗎？」）。
   - 口語同義詞（「飛機晚點」↔「班機延誤」）。
   - 多輪追問（「那行李呢？」）。

3. **真正的 agent（之後如果要再進化）**
   - 讓 LLM 不只改寫 query，還能自己決定：
     - 要不要先澄清使用者要問的險種？
     - 這題需不需要先查承保範圍，再查不保事項？
     - 是否要查理賠文件、理賠流程等其他條款？
     - 是否要分兩次 retrieval、合併多個 section 的資訊？

   這種做法比較重、也比較難控，適合更成熟階段再導入。

### 6.3 建議的演進路線

對目前這個 Cathay RAG 專案，一條實際又穩定的路線是：

1. 先把 **規則式 preprocessing** 做扎實：
   - alias mapping + section clarification + follow‑up rewrite。
2. 再加一層輕量的 `rewrite_query(question, history)`：
   - 讓 LLM 幫忙把模糊的使用者問題改成適合檢索的 written query。
3. 等以上都穩定了，再考慮是否要導入真正的 agent 流程。

這樣，你既能享受 LLM 的「語意理解」優勢，又不會把整個系統交給一個難以 debug 的 agent 黑盒子。
