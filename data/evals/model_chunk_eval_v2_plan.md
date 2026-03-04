# Model and Chunk Evaluation Plan v2

## Goal

This second-pass benchmark is designed to create more separation between:

- different LLM quality levels
- different embedding models
- different chunking settings

The v1 benchmark mostly used direct factual questions. In that setup, most model combinations already performed well.
This v2 benchmark focuses on more difficult cases:

- ambiguous user wording
- clarification behavior
- follow-up handling with chat history
- synonym mapping
- condition and exception clauses

## Scenario Set

### 1. `ambiguous_delay_then_pick_flight`

Focus:
- delay clarification
- follow-up rewrite

Turns:
- `旅遊延誤賠償怎麼算？`
- `班機延誤保險`

Expected behavior:
- first answer should ask whether the user means flight delay or baggage delay
- second answer should continue correctly from the clarification and answer the flight-delay case

### 2. `ambiguous_exclusion_then_pick_credit_card`

Focus:
- section clarification
- follow-up rewrite

Turns:
- `哪些原因屬於不可理賠範圍？`
- `信用卡盜用保險`

Expected behavior:
- first answer should ask which insurance section the user means
- second answer should use the selected section and answer the exclusion question

### 3. `credit_card_fraud_synonym`

Focus:
- synonym mapping

Turns:
- `信用卡盜刷有哪些不能賠？`

Expected behavior:
- should map `信用卡盜刷` to `信用卡盜用保險`
- should answer exclusions rather than saying the policy is unclear

### 4. `passport_loss_synonym`

Focus:
- synonym mapping

Turns:
- `護照遺失怎麼申請理賠？`

Expected behavior:
- should map to `旅行文件損失保險`
- should answer the claim / required documents path correctly

### 5. `flight_delay_exception_direct`

Focus:
- condition and exception handling

Turns:
- `如果沒有搭航空公司提供的第一班替代交通工具，還能申請班機延誤理賠嗎？`

Expected behavior:
- should mention the default exclusion
- should also mention the exception for force majeure /不可抗力

### 6. `flight_delay_exception_followup`

Focus:
- history-aware reasoning

Turns:
- `班機延誤保險有哪些不可理賠範圍？`
- `那如果是因為不可抗力沒搭第一班替代交通工具呢？`

Expected behavior:
- second turn should correctly inherit the flight-delay context
- should surface the exception clause instead of treating the follow-up as a fresh unrelated question

### 7. `mobile_phone_theft_synonym`

Focus:
- synonym mapping

Turns:
- `手機被偷可以怎麼申請理賠？`

Expected behavior:
- should map to `行動電話被竊損失保險`

### 8. `cash_theft_synonym`

Focus:
- synonym mapping

Turns:
- `現金被偷可以理賠嗎？`

Expected behavior:
- should map to `現金竊盜保險`

## How to Run

```bash
python -m src.evaluate_model_chunk_configs_v2
```

## Output

The script saves:

- `data/evals/model_chunk_eval_v2_*.json`
- `data/evals/model_chunk_eval_v2_*.md`
