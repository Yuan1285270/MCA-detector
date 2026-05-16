# MCA Scoring Module

這個資料夾負責把已經產生好的 account features 和 graph features 合成 account-level MCA review priority score。

它不生 adjacency graph，也不做 group expansion。

```text
account features + graph summaries -> MCA signal table -> ranked accounts
```

## Signals

目前 MCA score 使用四個 signal：

```text
Manipulative
- avg_rhetorical_score
- non_neutral_post_ratio
- oppositional_stance_ratio

Coordinative
- co_target
- co_negative_target
- trigger_response_frequency

Interaction reach
- outgoing_volume
- incoming_attention
- interaction_breadth

Automatic behavior
- isolation tree anomaly score
```

`A_tag_similarity` 不進 MCA 主分數；它留給 `coordination-expansion/` 做 rhetoric-similarity evidence。

## Run

```bash
.venv/bin/python mca-scoring/score_accounts.py
```

預設輸出：

```text
mca-scoring/output/
├── account_mca_scores.csv
├── top_accounts_primary.csv
└── top_accounts_alt.csv
```

## Default Weights

Primary:

```text
Manipulative:       0.40
Coordinative:       0.40
Interaction reach:  0.10
Automatic behavior: 0.10
```

Alternative:

```text
Manipulative:       0.30
Coordinative:       0.35
Interaction reach:  0.15
Automatic behavior: 0.20
```

Interpretation:

```text
MCA score = review priority, not a final verdict.
```
