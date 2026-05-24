# MCA-detector

Collaborative capstone repository for detecting suspicious cryptocurrency social media accounts that may coordinate to steer public opinion.

The project does not claim that an account is a confirmed bot, paid operator, or real-world cyber troop. It produces evidence-backed review candidates based on manipulative rhetoric, coordination proxies, interaction reach, automation signals, graph expansion, and temporal synchrony.

Current core pipeline:

```text
raw posts/comments
  -> LLM post/comment analysis
  -> account feature matrix + adjacency graphs
  -> MCA seed ranking
  -> coordination expansion
  -> temporal verification
  -> group summary + account roles
```

The main research claim is not "MCA score alone finds cyber troops." The claim is:

```text
MCA score helps choose suspicious seed accounts.
Graph expansion turns those seeds into candidate coordination groups.
Temporal synchrony helps separate same-ideology activity from stronger coordinated action evidence.
```

## Modules

- `llm/`
  - LLM-based Reddit post/comment analysis pipeline.
  - Includes data cleaning, local Ollama analysis, Gemini / Vertex AI compatibility, and exploratory post clustering.
  - See `llm/README.md` for details.

- `adjacency/`
  - Account-level graph construction module.
  - Builds single-graph and multi-graph sparse adjacency artifacts from LLM-analyzed Reddit comment feedback and account rhetoric features.
  - See `adjacency/README.md` for details.

- `mca-scoring/`
  - Account-level MCA review-priority scoring module.
  - Combines manipulative, coordinative, interaction reach, and automation signals.
  - Used mainly for seed selection and review priority, not as a final verdict.
  - See `mca-scoring/README.md` for details.

- `coordination-expansion/`
  - Seed expansion and group discovery module.
  - Turns graph layers into reviewable coordination groups, then verifies them with temporal synchrony.
  - See `coordination-expansion/README.md` for details.

- `paper/`
  - TCSE paper source, final screenshots, literature audit notes, and lightweight reviewer-response experiments.

Legacy modules such as the old standalone `behavior/` analysis and removed demo-site source are kept locally under `Archive/` when needed, but are not part of the current GitHub-facing pipeline.

## Project vs Visualization

The core project is the reproducible analysis pipeline:

```text
raw data -> LLM analysis -> features / graphs -> scoring -> expansion -> verification -> output tables
```

The demo website is a separate presentation layer outside the current core repository. It reads generated output tables and turns them into a clearer review experience. The demo site should not be treated as the research method, the source of truth, or a replacement for the pipeline outputs.

## Project Manual

For a detailed handoff-style explanation of the full project, see:

```text
docs/project_manual.md
output/pdf/mca_detector_project_manual.pdf
```

The manual explains the project goal, module responsibilities, pipeline runners, graph layers, MCA scoring, seed expansion, temporal verification, demo-site boundary, current outputs, case studies, and known limitations.

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
- `terry-behavior`: historical behavior analysis branch

Feature branches should be merged back into `main` after review.
