# Model and Chunk Evaluation Analysis v1

## Model-by-model interpretation

### `gpt-4o-mini + text-embedding-3-small`

- The most stable low-cost baseline.
- Answers are short, accurate, and do not include too much extra detail.
- For a single-policy document like this project, this setup is already usable.
- Across the three chunk settings, there is no major difference in answer quality.

### `gpt-4o-mini + text-embedding-3-large`

- Adds a bit more detail than the baseline, but the improvement is limited.
- Sometimes includes extra clause references that are not clearly necessary.
- Based on these 5 benchmark questions alone, upgrading to `text-embedding-3-large` does not look very cost-effective.

### `gpt-4.1 + text-embedding-3-small`

- Shows a noticeable quality improvement.
- Answers are more complete and feel closer to a formal customer-service style response.
- The tradeoff is that the answers start getting longer, especially for:
  - `行李遺失如何申請理賠？`
  - `信用卡盜用有哪些不可理賠範圍？`
- If the goal is a more polished production-style answer, this setup is worth considering.

### `gpt-4.1 + text-embedding-3-large`

- Still strong overall.
- However, compared with `gpt-4.1 + text-embedding-3-small`, the improvement is not large in this benchmark.
- In the current test set, the extra embedding cost does not show a clearly proportional benefit.
- This may only become more worthwhile if future questions are more ambiguous or complex.

### `gpt-5 + text-embedding-3-small`

- One of the longest-answering setups.
- The upside is completeness.
- The downside is that it can over-explain for insurance-policy QA.
- If the UI goal is a concise insurance assistant, this setup is probably overkill.

### `gpt-5 + text-embedding-3-large`

- The most expensive high-quality setup in this batch.
- But from the current results, it does not outperform `gpt-4.1` by a large enough margin.
- Based on the current benchmark only, this should not be the first priority choice.

## Chunk configuration analysis

### `chunk_size=500, overlap=100`

- Tighter chunking.
- Answers are often more direct.
- Works well for this document because the clauses are already fairly structured and explicit.
- The risk is that future questions involving many exceptions or cross-sentence conditions may become easier to split apart.

### `chunk_size=700, overlap=100`

- The most balanced setting.
- Almost all model combinations performed stably here.
- This is the recommended default setting to keep for now.

### `chunk_size=900, overlap=120`

- Did not show a clear quality improvement.
- In some cases, answers became longer or included more extra information.
- For the current policy file, this setting does not seem necessary.

## Recommendations

### Best balance of cost and quality

- `gpt-4.1 + text-embedding-3-small + chunk_size=700, overlap=100`

### Cheapest usable baseline

- `gpt-4o-mini + text-embedding-3-small + chunk_size=700, overlap=100`

### Good demo / low-cost setup

- `gpt-4o-mini + text-embedding-3-small + chunk_size=500, overlap=100`
