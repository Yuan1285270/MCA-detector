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

`A_tag_similarity` 不進 MCA 主分數。tag similarity 目前只作為 EDA / visualization / expansion context，不作為正式 MCA 分數或 Stage 2 verification evidence。

MCA score 的正式定位：

```text
MCA score = seed selection + review priority, not a final verdict.
```

## Run

```bash
.venv/bin/python mca-scoring/score_accounts.py
```

加上 `--min-score` 可以讓 `top_accounts_*.csv` 只輸出分數達標的帳號，避免資料集整體分數偏低時仍強制產出 top-N：

```bash
.venv/bin/python mca-scoring/score_accounts.py --min-score 0.60
```

預設 `--min-score 0.0`（不過濾），行為與之前相同。

可以切換內建 review mode，但預設仍是論文中的 primary weights：

```bash
.venv/bin/python mca-scoring/score_accounts.py --weight-profile behavior
```

目前內建 profiles：

```text
primary       = paper default, coordination-oriented hybrid score
coordination  = stronger co-target / trigger-response emphasis
behavior      = stronger automatic-behavior anomaly emphasis
rhetoric      = stronger manipulative/rhetorical signal emphasis
equal         = equal-weight baseline
```

也可以直接給自訂權重，順序是：

```text
Manipulative Coordinative InteractionReach AutomaticBehavior
```

例如 behavior-anomaly-heavy：

```bash
.venv/bin/python mca-scoring/score_accounts.py --primary-weights 0.20 0.25 0.15 0.40
```

預設輸出：

```text
mca-scoring/output/
├── account_mca_scores.csv
├── top_accounts_primary.csv
├── top_accounts_alt.csv
├── top_accounts_view_<profile>.csv
└── mca_weight_config.json
```

## Default Weights

Primary:

```text
Manipulative:       0.30
Coordinative:       0.35
Interaction reach:  0.15
Automatic behavior: 0.20
```

Alternative:

```text
Manipulative:       0.40
Coordinative:       0.40
Interaction reach:  0.10
Automatic behavior: 0.10
```

Interpretation:

```text
MCA score = review priority, not a final verdict.
```

`mca_weight_config.json` records the exact weights used for the current run so
ranking artifacts remain reproducible. `top_accounts_primary.csv` always uses
the active primary score: the default paper weights unless `--weight-profile` or
`--primary-weights` is provided.
