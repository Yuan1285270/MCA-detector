# MCA Detector Decision Log

這份文件記錄專題中的大決策，避免之後忘記「為什麼當初這樣做」。

每次改到以下內容，都要新增一筆：

- pipeline 架構
- graph / scoring / expansion / verification 規則
- 權重與閾值
- 輸出 schema
- PPT 敘事主線
- demo 跑法

格式：

```text
Date:
Decision:
Why:
Alternatives considered:
Impact:
Status:
```

---

## 2026-05-17 — MCA Score 改為 Seed Selection，不當最終判決

Decision:
MCA score 只用來排序高風險 seed accounts，後續還要透過 seed expansion、temporal verification、candidate validation 才形成最終可審查輸出。

Why:
單一帳號分數容易把正常高活躍用戶、立場強烈用戶也排很高。專題目標不是直接宣稱「誰是網軍」，而是縮小人工審查範圍並提供可解釋證據。

Alternatives considered:
- 直接用 MCA top-k 當最終可疑名單
- 用一個加權總分把所有 evidence 混成 final bot score

Impact:
PPT 與報告要把 MCA 定義成 investigation entry point，而不是 guilt score。最後成果應該講 group-level evidence table。

Status:
Accepted.

---

## 2026-05-17 — Graph Evidence 分成 Discovery 與 Verification

Decision:
Graph layers 先用來發現候選協同群，不能單獨證明同操控者。co-negative-target / tag-similarity / trigger-response 用於 expansion，temporal synchrony 用於第二階段驗證。

Why:
co-negative-target 只能說多個帳號攻擊同一批目標；這可能是意識形態重疊，不一定是網軍或同操控者。需要第二階段證據降低 false positive。

Alternatives considered:
- 直接把 co-negative-target expansion 的群當作可疑網軍群
- 只用 Louvain/group discovery 做群體發現

Impact:
系統輸出要分清楚：
Stage 1 = candidate coordination discovery
Stage 2 = temporal synchrony verification

Status:
Accepted.

---

## 2026-05-17 — Runner 拆成前半、後半、整份

Decision:
保留 full runner，但正式拆出：

- `run_front_pipeline.py`
- `run_back_pipeline.py`
- `run_full_pipeline.py`

Why:
前半段包含 raw cleaning、LLM analysis、export merge、feature matrix、adjacency graph，最不穩也最耗時。後半段相對穩定，適合常常重跑結果與調整規則。

Alternatives considered:
- 只保留一個 full runner
- 只保留 coordination-expansion 內的後半 runner

Impact:
日常調參與 demo 可優先使用後半 runner；需要重建資料時再跑前半或 full runner。

Status:
Accepted.

---

## 2026-05-17 — LLM Backend 可選，預設 Ollama

Decision:
Full/front runner 支援：

- `--llm-provider ollama`
- `--llm-provider gemini`
- `--llm-provider none`

預設使用 Ollama。

Why:
專題之後分析應以本地 Ollama 為主，Gemini/Vertex 作為舊 pipeline 或備用。`none` 讓我們可以直接使用現有 `llm/Export`，避免每次重跑昂貴且不穩的 LLM 分析。

Alternatives considered:
- 只支援 Gemini
- 只支援 Ollama
- 手動切換不同腳本

Impact:
runner 可跨 LLM backend，但所有 provider 最後都必須輸出相同 schema 給 downstream pipeline。

Status:
Accepted.

---

## 2026-05-17 — LLM Analysis 支援可調 Parallel Workers

Decision:
runner 加入 `--llm-workers N`，用 row ranges 切 batch 平行跑 LLM analysis。`N=1` 為串行，`N>1` 為平行。

Why:
LLM analysis 是最慢的階段。手動平行跑容易出錯，所以讓 runner 控制 batch range 與後續 export merge。

Alternatives considered:
- 永遠單線程
- 永遠平行
- 寫獨立 parallel resume script，但不接 runner

Impact:
demo 或小測可用 `--llm-workers 1` 保守跑；長跑可調成 2 或 4，但要注意 Ollama 本機資源或 API quota。

Status:
Accepted.

---

## 2026-05-17 — Temporal Label 需要 Reliability Layer

Decision:
保留原本 temporal label：

- `strong_temporal_sync`
- `moderate_temporal_sync`
- `weak_temporal_overlap`
- `no_temporal_sync`

但新增 temporal reliability/confidence layer，區分：

- `robust`
- `moderate_review`
- `fragile_single_event`
- `fragile_long_median`

Why:
目前 `strong_temporal_sync` 只要有一次 within 5 minutes 就成立，可能只是熱門貼文下的活躍用戶剛好同時出現。人工驗證時需要知道 temporal evidence 是穩定重複，還是單次巧合。

Alternatives considered:
- 直接收緊 strong/moderate 的原始定義
- 完全不改，只在報告中口頭提醒

Impact:
candidate validation 和 group ranking 應優先使用 robust/moderate_review temporal evidence。PPT 可說明 temporal sync 有用，但單次同步需要 reliability filter。

Status:
Accepted and implemented.
