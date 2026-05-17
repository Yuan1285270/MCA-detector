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

---

## 2026-05-17 — Text Fingerprint 視為 Topic Noise 並移出正式證據

Decision:
`text_fingerprint_distance` 不再作為 Stage 2 verification evidence，也不在 demo site 主畫面呈現。Stage 2 的正式驗證訊號改為：

- `temporal synchrony`
- `temporal confidence`
- `account_lifecycle_overlap` / activation window overlap

程式輸出可暫時保留 `text_fingerprint_distance` 欄位作為相容欄位，但填入空值，不參與 ranking、validation、demo 解讀或成果敘事。

Why:
在 r/Bitcoin 這種單一主題資料集中，TF-IDF cosine distance 容易測到「大家都在講同一個主題」，而不是同操控者的寫作指紋。反過來，真正有意識的操控者也可能刻意變換語氣。因此 text fingerprint 在目前資料上更像噪音，放進正式證據會混淆結果。

Alternatives considered:
- 保留 text fingerprint 作為輔助加分
- 改用更複雜的 stylometry
- 等更多 ground truth 後再校準 threshold

Impact:
Stage 2 敘事更乾淨：核心看 temporal synchrony，輔助看 activation window overlap。報告可誠實說明 text fingerprint 曾被測試，但因單一主題社群的 topic similarity 問題而移除。

Status:
Accepted and implemented.

---

## 2026-05-17 — Demo Output 分成群體協同與單一異常帳號

Decision:
demo website 和報告輸出分成兩條結果線：

- `Suspicious Coordination Groups`
- `Individual Abnormal Accounts`

Why:
MCA/anomaly features 可能抓到值得注意的單一異常帳號，例如 spam/scam-like account，但這不代表它一定屬於 coordinated manipulation group。把兩類結果分開，可以保留有價值的異常發現，同時避免把所有異常帳號都硬說成網軍。

Alternatives considered:
- 只輸出 group-level suspicious coordination
- 把所有高 MCA 或 extreme outlier 都塞進同一個網軍排名

Impact:
網站第一層會同時呈現 group-level evidence 和 account-level risk。PPT/demo 應明確說明：系統主要找 coordinated behavior，但也能額外標出 individual abnormal manipulation accounts。

Status:
Accepted and implemented in `MCA-demo-site`.

---

## 2026-05-17 — Demo Site 預設為 Client Mode

Decision:
`MCA-demo-site` 預設使用中文 `Client mode`，並提供 `Client/Demo` 與 `中文/EN` 切換。

Why:
給外部觀眾或老師看的第一畫面應該先回答「哪些群組需要優先處理、為什麼、下一步是什麼」，而不是先展示 pipeline 內部術語。Demo mode 保留 Stage 1、Stage 2、MCA 等研究語言，方便我們自己講方法與 debug。

Alternatives considered:
- 只做研究 demo，不做 client view
- 只做 client view，把研究細節移除
- 只翻譯文字，不區分 audience mode

Impact:
網站第一層敘事以風險審查與處置優先級為主；研究細節仍可用 Demo mode 切回。這讓同一套輸出可以同時服務商業展示與專題答辯。

Status:
Accepted and implemented.

---

## 2026-05-17 — Stage 2 補上 Text Fingerprint 與 Lifecycle Evidence

Decision:
在 `stage2_temporal_verification.py` 裡正式填入原本預留的兩個欄位：

- `text_fingerprint_distance`
- `account_lifecycle_overlap`

`text_fingerprint_distance` 使用同一 group 內帳號 comments 的 TF-IDF cosine distance。`account_lifecycle_overlap` 使用帳號觀測到的 comment 活躍時間區間重疊比例。

Why:
只靠 temporal synchrony 可能把同時在線的正常活躍用戶誤判成可疑。text fingerprint 可以補「語言/模板相似」證據，lifecycle overlap 可以補「帳號活動週期是否重疊」證據。這讓 Stage 2 不只看時間同步，也能提供更完整的 verification evidence。

Alternatives considered:
- 繼續把兩個欄位保留為 `NaN`
- 把 text fingerprint 做成獨立腳本，不接進 Stage 2
- 等網站端才做文字相似分析

Impact:
`stage2_verification_evidence.csv` 會從 temporal-only evidence 升級成 multi-signal verification table。PPT 可以說明我們有 temporal synchrony 與 text fingerprint 兩種驗證訊號。

Status:
Partially superseded. Lifecycle evidence remains accepted; text fingerprint was later removed from formal evidence because it behaved like topic-noise in this dataset.
