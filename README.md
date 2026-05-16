# MCA-detector

Collaborative capstone repository for detecting suspicious cryptocurrency social media accounts that may coordinate to steer public opinion.

The project does not claim that an account is a confirmed bot, paid operator, or real-world cyber troop. It produces evidence-backed review candidates based on manipulative rhetoric, coordination proxies, interaction reach, and automation signals.

## Modules

- `behavior/`
  - Behavior analysis module.
  - See `behavior/README.md` for details.

- `llm/`
  - LLM-based Reddit post/comment analysis pipeline.
  - Includes data cleaning, Gemini / Vertex AI batch analysis, local Ollama experiments, and exploratory post clustering.
  - See `llm/README.md` for details.

- `adjacency/`
  - Account-level graph construction module.
  - Builds single-graph and multi-graph sparse adjacency artifacts from LLM-analyzed Reddit comment feedback and account rhetoric features.
  - See `adjacency/README.md` for details.

- `mca-scoring/`
  - Account-level MCA review-priority scoring module.
  - Combines manipulative, coordinative, interaction reach, and automation signals.
  - See `mca-scoring/README.md` for details.

- `coordination-expansion/`
  - Seed expansion and group discovery module.
  - Turns graph layers into reviewable coordination evidence tables.
  - See `coordination-expansion/README.md` for details.

## Full Pipeline Runner

目前 runner 拆成前半、後半、整份三個入口。

前半：從 raw posts/comments 跑到 analyzed exports、account feature matrix、adjacency graphs：

```bash
.venv/bin/python run_front_pipeline.py \
  --raw-posts llm/data-cleaning/source_data/reddit_posts_2025.csv \
  --raw-comments llm/data-cleaning/source_data/reddit_comments_2025.csv \
  --llm-provider ollama \
  --llm-workers 2
```

後半：從已完成的 graph / feature artifacts 跑到 group summary / account roles：

```bash
.venv/bin/python run_back_pipeline.py --top-n-seeds 20
```

整份：前半 + 後半一次跑完：

```bash
.venv/bin/python run_full_pipeline.py \
  --raw-posts llm/data-cleaning/source_data/reddit_posts_2025.csv \
  --raw-comments llm/data-cleaning/source_data/reddit_comments_2025.csv \
  --llm-provider ollama \
  --llm-workers 2 \
  --top-n-seeds 20
```

可選 LLM backend：

```text
--llm-provider ollama   # local Ollama，之後主要使用
--llm-provider gemini   # Vertex/Gemini
--llm-provider none     # 不跑 LLM，直接使用既有 llm/Export/*.csv.gz
```

LLM worker 數可調：

```text
--llm-workers 1   # 串行，最穩
--llm-workers 2   # posts/comments 各自切成 2 個 batch 平行跑
--llm-workers 4   # 更快，但需要本機 Ollama / API quota 撐得住
```

輸出終點：

```text
coordination-expansion/output/final-summary/final_group_summary.csv
coordination-expansion/output/account-roles/account_role_table.csv
```

## Branch Workflow

- `main`: shared stable baseline
- `feature/llm-data-pipeline`: LLM data pipeline work
- `terry-behavior`: behavior analysis work

Feature branches should be merged back into `main` after review.
