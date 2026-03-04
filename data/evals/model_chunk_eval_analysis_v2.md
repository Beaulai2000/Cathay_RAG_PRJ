# Model and Chunk Evaluation Analysis v2

## Scope

This analysis is based on:

- [`model_chunk_eval_v2_20260305_005102.md`](./model_chunk_eval_v2_20260305_005102.md)
- the matching JSON result file for the same run

The v2 benchmark was designed to stress:

- ambiguous questions that require clarification
- follow-up turns that rely on history
- synonym handling
- condition and exception clauses

## High-level conclusion

The second-pass benchmark is more realistic than v1, but the gap between model combinations is still not very large.

The main reason is that the current RAG pipeline already contains strong rule-based support:

- section clarification
- follow-up question rewriting
- synonym mapping
- ambiguous delay clarification

Because these steps resolve much of the difficulty before retrieval and generation, the model differences are compressed.

## Main findings

### 1. Almost every configuration passes the important behaviors

Across the 18 evaluated combinations, nearly all of them correctly handled:

- asking the user to clarify delay type
- asking the user to clarify insurance section
- mapping `信用卡盜刷` to `信用卡盜用保險`
- mapping `護照遺失` to `旅行文件損失保險`
- handling the exception for `不可抗力` in the flight-delay clause
- answering `現金被偷可以理賠嗎？`

This means the current system design is doing more work than the model choice itself.

### 2. `gpt-4o-mini` remains stronger than expected

Even under v2 benchmark conditions, `gpt-4o-mini` still performs well.

Why:

- the rule-based clarification logic is already strong
- the synonym mapping layer is working
- the section-aware retrieval is helping a lot

Practical implication:

- for a production-minded but cost-sensitive version, `gpt-4o-mini` is still a realistic choice

### 3. `text-embedding-3-large` still does not show a major advantage

In v2, the larger embedding model still does not clearly separate itself from `text-embedding-3-small`.

Interpretation:

- for the current single-policy corpus
- with article-aware chunking
- and section-aware retrieval

the smaller embedding model is already enough for most retrieval cases in the test set.

Practical implication:

- upgrading embeddings should not be the first optimization priority

### 4. `gpt-4.1` mainly improves answer quality, not pass/fail behavior

`gpt-4.1` usually produces:

- more polished phrasing
- more formal customer-service style answers
- better structured explanations

However, it does not dramatically outperform `gpt-4o-mini` in raw scenario success within this benchmark.

Practical implication:

- choose `gpt-4.1` if you care about response quality and production tone
- not because the benchmark shows a huge correctness gap

### 5. `gpt-5` still looks too expensive for the current setup

`gpt-5` responses are often:

- longer
- more elaborate
- sometimes more expanded than necessary

But the benchmark does not show a strong enough quality advantage over `gpt-4.1`.

Practical implication:

- `gpt-5` does not currently justify its likely extra cost for this insurance-policy QA use case

## Chunking conclusions

The v2 benchmark supports the same chunking conclusion as v1.

### `chunk_size=500, overlap=100`

- works
- often gives direct answers
- may become riskier if future tasks depend on long exception chains or cross-sentence reasoning

### `chunk_size=700, overlap=100`

- still the most balanced configuration
- consistently stable across model combinations
- best default to keep for now

### `chunk_size=900, overlap=120`

- still does not show a clear quality gain
- sometimes encourages longer answers without a proportional benefit
- not necessary for the current policy corpus

## Recommended setups

### Best balance of quality and cost

- `gpt-4.1 + text-embedding-3-small + chunk_size=700, overlap=100`

Why:

- strong answer quality
- stable behavior
- no clear need to pay for larger embeddings

### Best low-cost baseline

- `gpt-4o-mini + text-embedding-3-small + chunk_size=700, overlap=100`

Why:

- very stable even under v2 benchmark conditions
- significantly cheaper than higher-end model combinations
- likely good enough for a first production or demo version

### Low-cost demo option

- `gpt-4o-mini + text-embedding-3-small + chunk_size=500, overlap=100`

Why:

- direct answers
- low cost
- acceptable for lightweight demos

## What v2 still does not fully test

Although v2 is better than v1, it still does not fully separate strong and weak configurations.

The current benchmark still under-tests:

- retrieval confusion between multiple similar clauses
- citation accuracy versus answer content
- failure under intentionally misleading wording
- questions that require combining multiple sections at once
- cases where retrieval returns partially relevant but incomplete evidence

## Final interpretation

The main lesson from v2 is:

- the current rule-based RAG design is already doing a lot of the heavy lifting
- model selection matters, but not as much as pipeline design
- chunking matters, but the best setting is already fairly clear

If the next goal is to discover stronger differences between model combinations, the benchmark itself needs to become harder.

For now, the best practical recommendation remains:

- keep `chunk_size=700, overlap=100`
- use `text-embedding-3-small`
- choose between `gpt-4o-mini` and `gpt-4.1` depending on cost tolerance and desired answer style
