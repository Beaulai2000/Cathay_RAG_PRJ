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
   python src/ingestion.py
   ```

2. Start a CLI chat:
   ```bash
   python src/cli.py
   ```

3. Ask questions such as:
   - "什麼情況下可以申請旅遊延誤賠償？"
   - "行李遺失後應該如何申請理賠？"
   - "哪些原因屬於不可理賠範圍？"

The system will retrieve relevant clauses and generate answers with clause citations.
