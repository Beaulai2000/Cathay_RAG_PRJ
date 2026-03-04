# Cathay Travel Insurance RAG – Project Workflow & Architecture

> 狀態：截至 2026-03-04 晚上（你剛完成 history + section-aware RAG、Gradio share 等改動）

這份檔案是給**未來的你**看的，讓你一眼了解：

- 這個專案現在長什麼樣子
- 你已經做了哪些設計 / 決策
- 要怎麼跑整套 RAG（CLI + Gradio）
- 之後可以往哪裡優化

README.md 保留原本的課程 / idea；這份是你的**工程實作紀錄**。

---

## 1. 專案目標 & 總體架構

### 1.1 專案目標

- 針對 **國泰旅遊不便險條款**，打造一個：
  - 可以自然語言提問的 Q&A chatbot
  - 背後使用 **RAG（Retrieval-Augmented Generation）**，從條款文本中找答案
  - 能附上 **「第幾條」** 的引用，讓使用者知道依據來自哪個條文

### 1.2 高層架構

整體 flow 可以簡化成：

1. **Ingestion（索引建立）**
   - 從 PDF / 原始保單 → 清洗後文字檔 → 切 chunk（條文感知） → 建立 Chroma 向量索引

2. **Retrieval（檢索）**
   - 根據使用者問題（＋歷史上下文）
   - 使用 semantic search，找出最相關的條款 chunk
   - 支援「保險項目（section）」的意圖判斷與篩選

3. **Answer Generation（回答）**
   - 系統訊息：保險專員角色 + 嚴格依據條款回答的指示
   - 使用者訊息：條款 context ＋ 問題（＋過往對話）
   - 使用 OpenAI Chat 模型產生中文回答
   - 引導模型在答案中附上引用條款

4. **介面**
   - CLI：終端機互動
   - Gradio：web-based chat UI，可 share link demo

---

## 2. 專案結構概覽

專案根目錄的重點：

- `README.md` – 原始課程/idea 說明（保留，不覆蓋）
- `requirements.txt` – 依賴套件（含 langchain、chroma、gradio 等）
- `data/`
  - `policy_raw/` – 原始保單 PDF 或 text（透過 `.gitignore` 排除版本控制）
  - `policy_clean/` – 清洗後的條款純文字
  - `index/` – Chroma 向量索引（亦被 `.gitignore` 排除）
- `notebooks/` – 可放實驗 notebook（目前乾淨）
- `src/`
  - `config.py` – 專案各種設定（模型、路徑、chunking、section、history window 等）
  - `extract_policy_from_pdf.py` – 從 PDF 抽取 + 清洗條款
  - `ingestion.py` – 將清洗後條款切 chunk、建立向量索引
  - `retrievers/semantic.py` – 取得 semantic retriever，支援 section filter
  - `rag_pipeline.py` – **RAG 核心 pipeline**（history + section-aware + clarification）
  - `cli.py` – CLI 入口（`python -m src.cli`）
  - `gradio_app.py` – Gradio UI 入口（`python -m src.gradio_app`）

---

## 3. 設定與共用配置（`src/config.py`）

你把許多「之後會想調整」的東西抽到 `config.py`：

- **模型設定**：
  - `LLM_MODEL` – 例如 `gpt-4o-mini`（或未來的其他模型）
  - `EMBEDDING_MODEL` – 例如 `text-embedding-3-small`

- **Chunking 設定**：
  - `CHUNK_SIZE` – 預設 `700` 字元
  - `CHUNK_OVERLAP` – 預設 `100` 字元
  - 可以透過環境變數 `CATHAY_RAG_CHUNK_SIZE`、`CATHAY_RAG_CHUNK_OVERLAP` 調整

- **路徑設定**：
  - `PROJECT_ROOT`
  - `DATA_DIR`
  - `DEFAULT_POLICY_CLEAN_PATH`
  - `INDEX_DIR` – Chroma 向量庫儲存路徑

- **RAG 相關設定**：
  - `RETRIEVER_TOP_K` – 一次取幾個 chunk（預設 5）
  - `CHAT_HISTORY_WINDOW` – LLM 看到的最近對話輪數（避免 prompt 過長）
  - `INSURANCE_SECTIONS` – 條款中不同保險項目（例如班機延誤保險、行李延誤保險...）
  - `SECTION_ALIASES` – 各種口語說法 → 正式 section 名稱的 mapping

這讓你之後要調整行為，只要改 `config.py` 或環境變數即可，不必重寫 code。

---

## 4. Ingestion：條文切分與索引建立（`src/ingestion.py`）

### 4.1 條文讀取

- `read_policy_text(path: Path | None = None) -> str`
  - 預設從 `DEFAULT_POLICY_CLEAN_PATH` 讀取清洗後的條款文字

### 4.2 Chunking 策略

你設計了兩層的 chunking：

1. **`naive_paragraph_chunk`**（備用方案）
   - 以空行分段（適用於有明顯段落的文本）
   - 使用 `chunk_size` + `overlap` 控制 chunk 長度與重疊

2. **`article_aware_chunk`**（主要方案）
   - 使用正則 `ARTICLE_HEADING_RE` 偵測條文標題，例如：
     - `第X條 ...`
   - 先按條文標題切成「條文為單位」的段落
   - 若單一條文太長，再以行為單位做次級切分，並加上重疊（避免句子被硬切斷）
   - 若完全偵測不到條文標題，就 fallback 回 `naive_paragraph_chunk`

### 4.3 Metadata & 預覽

- `extract_article_id(chunk: str) -> str`
  - 回傳 chunk 的第一行條文標題（如果符合條文 regex）
  - 用於 metadata 的 `article_id`

- `preview_chunks(chunks: List[str], ...)`
  - 印出前幾個 chunk 的長度與前幾十個字，方便你 check 切得好不好

### 4.4 建立 Chroma 向量庫

- `build_index()` 主要流程：

  1. 讀取清洗後條款文本
  2. 使用 `article_aware_chunk` 依 `CHUNK_SIZE` / `CHUNK_OVERLAP` 切 chunk
  3. 把每個 chunk 包成 `Document`：
     - `page_content = chunk`
     - `metadata = {"chunk_id": i, "article_id": extract_article_id(chunk)}`
  4. 建立 `OpenAIEmbeddings(model=EMBEDDING_MODEL)`
  5. 使用 `Chroma` 建立/重建 collection：
     - 每次先 `delete_collection()` 再從頭加 documents（避免舊版 chunk 混進來）

執行方式：

```bash
python -m src.ingestion
```

---

## 5. Retrieval：語意檢索與保險項目（`src/retrievers/semantic.py`）

> 這裡只記設計概念，詳細實作看檔案。

- `get_semantic_retriever(k: int = RETRIEVER_TOP_K, section: str | None = None)`
  - 從 `INDEX_DIR` 載入 Chroma 向量庫
  - 若有指定 `section`，就只檢索該 section 相關的 chunks
  - 回傳一個 retriever，可透過 `.invoke(query)` 或 `.get_relevant_documents(query)` 使用

這個設計讓你之後可以：

- 對不同保險項目（section）建立同一個 collection，但在 query 時做過濾
- 或未來延伸為多個 collection / 多政策切分

---

## 6. RAG Pipeline：History + Section-aware + Clarification（`src/rag_pipeline.py`）

這是目前你花最多心力的部分，也是面試可以重點講的地方。

### 6.1 Pipeline 初始化

```python
class RAGPipeline:
    def __init__(self, k: int | None = None) -> None:
        self.k = k or RETRIEVER_TOP_K
        self.retriever = get_semantic_retriever(k=self.k)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
```

- 預設使用 config 中的 `RETRIEVER_TOP_K` 與 `LLM_MODEL`
- `retriever` 可以被 section-aware 版本覆蓋（見下）

### 6.2 Answer 流程總覽

```python
def answer(self, question: str, history: Sequence[Any] | None = None) -> str:
    # 1) 處理 follow-up 問題與術語正規化
    question = rewrite_followup_question(question, history=history)
    question = normalize_question_terms(question)

    # 2) 判斷保險 section 並處理模糊問句
    section = infer_requested_section(question, history=history)
    if not section and is_ambiguous_delay_question(question):
        return build_delay_clarification_message()
    if not section and is_ambiguous_section_question(question):
        return build_section_clarification_message()

    retriever = get_semantic_retriever(k=self.k, section=section) if section else self.retriever
    docs = retriever.invoke(question)
    if not docs:
        return "目前找不到與條款內容相符的資訊，建議洽詢客服或查看正式保單。"

    # 3) 組條款 context（含 article_id）
    ...

    # 4) 組 messages：system + history + 當前 context+question
    ...

    resp = self.llm.invoke(messages)
    return resp.content
```

### 6.3 History 支援

- `answer` 接受 `history: Sequence[Any] | None`
- 支援兩種格式：
  - Gradio 風格：`[(user, assistant), ...]`
  - LangChain 風格：`{"role": "user"/"assistant", "content": "..."}`
- 只取最近 `CHAT_HISTORY_WINDOW` 輪
- 組 messages 時：
  - 先加上 `system` 指示（保險專員、只依據條款、不亂猜、最後附條款編號等）
  - 再把 history 轉成 chat messages（依序添加 user / assistant）
  - 最後加入這一輪的 `user`：包含條款 context＋問題

### 6.4 保險項目（section）意圖判斷

輔助函式：

- `infer_section_from_text(text: str) -> str | None`
  - 利用 `SECTION_ALIASES` 與 `INSURANCE_SECTIONS`，從任意文字中猜使用者指的是哪個保險項目

- `build_history_text(history) -> str`
  - 把最近幾輪對話的文字串起來，用於意圖判斷

- `infer_requested_section(question, history) -> str | None`
  - 先看當前問題，有就直接回 section
  - 若沒有，再看 history 串起來的文字

### 6.5 Follow-up 問題改寫（Rewrite）

- `rewrite_followup_question(question, history)` 用來處理：
  - 上一輪 assistant 問：「你想問的是哪一種保險？」
  - 這一輪 user 只回：「班機延誤保險」或「行李延誤保險」
- 流程：
  1. 用 `infer_section_from_text` 判斷這是不是一個純粹的 section 名稱
  2. 確認上一輪 assistant 是否剛好是在做澄清（`last_assistant_message_is_clarification` / `last_assistant_message_is_delay_clarification`）
  3. 若是，就重寫成完整 query：
     - 延誤相關：`"班機延誤保險什麼情況下可以申請理賠？"`
     - 其他情況：`"班機延誤保險有哪些不可理賠範圍？"`

### 6.6 模糊問題澄清

- `is_ambiguous_section_question(question)`：
  - 抓出關鍵字："不可理賠"、"不保事項"、"除外責任"、"哪些原因" 等
  - 若問這類問題但沒說哪個保險項目 → 視為模糊
- `build_section_clarification_message()`：
  - 回一段訊息：列出 `INSURANCE_SECTIONS` 中支援的保險項目，請使用者選

- `is_ambiguous_delay_question(question)`：
  - 若有 "旅遊延誤"、"延誤賠償" 等字眼
  - 但沒有「班機延誤」或「行李延誤」
- `build_delay_clarification_message()`：
  - 回一段訊息：請使用者說清楚是「班機延誤保險」還是「行李延誤保險」，並提供範例問法

---

## 7. 介面：CLI 與 Gradio（`src/cli.py`, `src/gradio_app.py`）

### 7.1 CLI

- `src/cli.py`：
  - 使用 `from .rag_pipeline import RAGPipeline`
  - 跑法：

    ```bash
    python -m src.cli
    ```

  - 每輪：
    - 從輸入讀取一行問題
    - 呼叫 `rag.answer(q)`（未來可擴充歷史支援）
    - 印出回答

### 7.2 Gradio UI

- `src/gradio_app.py`：
  - `build_pipeline()` 建立 `RAGPipeline(k=5)`
  - `PIPELINE = build_pipeline()`（全域 pipeline）
  - `chat_fn(message, history)`：
    - 現在會把 `history` 傳進 `PIPELINE.answer`（在某次改動中已完成）
    - 回傳 answer 給 Gradio
  - `main()` 建立 `gr.ChatInterface`，`demo.launch(share=True)`：
    - 本機 UI + public share link

- 執行方式：

  ```bash
  python -m src.gradio_app
  ```

---

## 8. 執行整套 RAG 的步驟

1. **安裝依賴**

   ```bash
   cd Cathay_RAG_PRJ
   pip install -r requirements.txt
   ```

2. **設定環境變數**

   ```bash
   export OPENAI_API_KEY="你的_API_Key"
   # 可選：
   # export CATHAY_RAG_CHUNK_SIZE=700
   # export CATHAY_RAG_CHUNK_OVERLAP=100
   ```

3. **建立索引**

   ```bash
   python -m src.ingestion
   ```

4. **跑 CLI Demo**

   ```bash
   python -m src.cli
   ```

5. **跑 Gradio UI**

   ```bash
   python -m src.gradio_app
   ```

---

## 9. 之後可以做的事（TODO / 想法）

- [ ] 在答案中**由程式**依 `docs` 的 metadata 自動列出引用條款，而不是完全依賴 LLM。
- [ ] 實作 query rewrite：
  - 使用 LLM + history，把 follow-up 問題改寫成完整 query，再做 retrieval。
- [ ] 增加更多 domain-specific test cases，用小腳本跑 QA sanity check。
- [ ] 把這個專案整理成 portfolio：
  - 清楚描述：條文感知 chunking、section-aware retrieval、clarification strategies、history handling。
- [ ] 若有需要，再包裝成 Docker image 或部署成簡單的 web service。

---

> 如果之後架構有大改，可以在這份檔案下面再加「版本變更紀錄」，幫自己記錄你是怎麼從 v1 → v2 → v3 演進的。