# MCA Detector 專案說明書

版本：2026-05-17  
分支：feature/llm-data-pipeline  
用途：專題交接、論文撰寫、demo 展示前說明

## 1. 專案定位

MCA Detector 是一套針對社群平台中「可疑協同行為」的分析流程。它的目標不是直接宣稱某個帳號是機器人、網軍或真實世界中的操控者，而是產生可重跑、可解釋、可人工審查的高風險候選群。

核心問題是：

> 單一帳號看起來可疑，不代表它屬於協同群；多個帳號攻擊同一批目標，也不一定代表它們由同一操控者控制。

因此專案採用兩階段邏輯：

1. 先找候選協同群：用 MCA 分數找 seed accounts，再用 adjacency graph 做 seed expansion。
2. 再做時間驗證：用 temporal synchrony 判斷群內帳號是否在同一討論串短時間共同出現。

## 2. 專案與 Demo 網站的界線

本專案分成「分析 pipeline」與「展示層」兩部分。

分析 pipeline 是研究方法本體：

```text
raw data -> LLM analysis -> features / graphs -> MCA scoring
         -> seed expansion -> temporal verification -> output tables
```

`MCA-demo-site/` 是展示層：

```text
coordination-expansion/output -> MCA-demo-site/data/demo-data.js -> dashboard
```

Demo 網站不計算 MCA 分數、不發現群組、不做 temporal verification，也不是 final verdict。它只是把 pipeline 已產生的 evidence tables 包裝成 client-facing 或 demo-facing 介面。

## 3. 主要資料流

整體 pipeline：

```text
raw posts/comments
  -> LLM post/comment analysis
  -> account feature matrix
  -> adjacency graphs
  -> MCA seed ranking
  -> coordination expansion
  -> Stage 2 temporal verification
  -> candidate validation table
  -> final group summary
  -> account role table
```

目前 runner 分成三個入口：

| Runner | 用途 |
|---|---|
| `run_front_pipeline.py` | 從 raw posts/comments 跑到 analyzed exports、feature matrix、adjacency graphs |
| `run_back_pipeline.py` | 從既有 graph/features 跑 MCA、expansion、verification、summary |
| `run_full_pipeline.py` | 前半與後半一次跑完 |

常用指令：

```bash
.venv/bin/python run_front_pipeline.py \
  --raw-posts llm/data-cleaning/source_data/reddit_posts_2025.csv \
  --raw-comments llm/data-cleaning/source_data/reddit_comments_2025.csv \
  --llm-provider ollama \
  --llm-workers 2
```

```bash
.venv/bin/python run_back_pipeline.py --top-n-seeds 20
```

```bash
.venv/bin/python run_full_pipeline.py \
  --raw-posts llm/data-cleaning/source_data/reddit_posts_2025.csv \
  --raw-comments llm/data-cleaning/source_data/reddit_comments_2025.csv \
  --llm-provider ollama \
  --llm-workers 2 \
  --top-n-seeds 20
```

LLM backend：

| Provider | 說明 |
|---|---|
| `ollama` | 目前建議使用的本地 LLM backend |
| `gemini` | Vertex/Gemini 舊流程或雲端備用 |
| `none` | 不重跑 LLM，直接使用既有 `llm/Export` |

## 4. 模組說明

### 4.1 `llm/`

負責對 Reddit 貼文與留言做 LLM 分析。

貼文分析輸出：

- `sentiment_score`
- `manipulative_rhetoric_score`
- `manipulative_rhetoric_reason`
- `rhetoric_tags`

留言分析輸出：

- `feedback_label`
- `feedback_score`
- `edge_weight`

留言會被轉成 account-to-account edge：

```text
comment_author -> post_author
```

若留言反對或攻擊原貼文，則形成負向互動；若支持或補充，則形成正向互動。

### 4.2 `behavior/`（legacy）

早期負責建立帳號行為特徵與 anomaly label。現在正式 pipeline 不再以此資料夾作為主入口；需要的行為欄位已整合到新版 account feature matrix / MCA scoring 流程。主要特徵包括：

- `comment_count`
- `post_count`
- `comments_per_day`
- `posts_per_day`
- `active_days`
- `burst_ratio`
- `night_activity_ratio`
- `weekend_activity_ratio`
- `comment_post_ratio`

注意：behavior anomaly 不是 bot verdict。它只能說某個帳號的活動模式異常，不能單獨證明該帳號屬於協同群。

### 4.3 `adjacency/`

負責建立多層 account-level graph。主要輸出包括：

| Graph | 意義 | 正式用途 |
|---|---|---|
| `A_count` | 純留言次數 | baseline / EDA |
| `A_signed` | 支持/反對方向與強度 | interaction graph |
| `A_positive` | 正向互動 | visualization / EDA |
| `A_negative` | 負向互動 | visualization / EDA |
| `A_trigger_response` | A 發文後 B 是否固定回應 | expansion support |
| `A_co_target` | 兩帳號是否留言到同一批目標 | support signal |
| `A_co_negative_target` | 兩帳號是否共同攻擊同一批目標 | candidate discovery 主訊號 |
| `A_tag_similarity` | rhetoric tag profile 是否相似 | EDA / visualization，不進正式分數 |

最重要的是 `A_co_negative_target`。它回答：

> 兩個帳號是否常常反對或攻擊同一批作者？

但這不是 final verdict，因為正常使用者也可能因立場相同而共同批評同一批人。

### 4.4 `mca-scoring/`

負責建立 account-level MCA review priority score。MCA 不是最終判決，而是 seed selection 與 review priority。

四個 signal：

| Signal | 權重 | 主要特徵 |
|---|---:|---|
| Manipulative | 0.30 | 平均修辭分數、非中立貼文比例、反對立場比例 |
| Coordinative | 0.35 | co-target、co-negative-target、trigger-response frequency |
| Interaction reach | 0.15 | outgoing volume、incoming attention、interaction breadth |
| Automatic behavior | 0.20 | Isolation Forest anomaly score |

這些權重是常理型初始權重，不是 supervised optimal weights。因為目前沒有大規模 ground truth，不適合宣稱已找到最佳權重。

### 4.5 `coordination-expansion/`

負責把 MCA seed 與 graph artifacts 轉成可審查的候選群。

流程：

```text
MCA top accounts
  -> seed expansion / group discovery
  -> Stage 2 temporal verification
  -> candidate validation table
  -> final group summary
  -> account roles
```

Seed selection 預設取 MCA primary ranking top 20。

Tiered expansion：

| Tier | 納入條件 | 意義 |
|---|---|---|
| Tier 1 | `co_negative_target >= 0.20` | 共同負向目標高度重疊 |
| Tier 2 | `tag_similarity >= 0.90` 且有結構訊號 | 語言/修辭近似，但需支持 |
| Tier 3 | `trigger_response >= 0.50` 且有 co-negative 或 tag support | 固定回應模式 |
| Tier 4 | 2-hop co-negative，且連到至少 2 個已納入成員 | 外圍候選成員 |

`co_target` 是輔助訊號，不單獨納入。

## 5. Stage 2 Temporal Verification

Stage 1 找到的是 candidate coordination groups，不是同操控者證據。Stage 2 檢查群內帳號是否在同一篇 post 下短時間共同出現。

Pair-level labels：

| Label | 定義 |
|---|---|
| `strong_temporal_sync` | 至少一次同串留言時間差小於 5 分鐘 |
| `moderate_temporal_sync` | 至少兩次同串留言時間差小於 30 分鐘 |
| `weak_temporal_overlap` | 有同串共現，但沒有短時間同步 |
| `no_temporal_sync` | 沒有同串共現 |

Temporal confidence：

| Confidence | 意義 |
|---|---|
| `robust` | 重複或跨多篇貼文的短時間同步 |
| `moderate_review` | 有可審查同步證據，但仍需人工確認 |
| `fragile` | 單次或典型延遲太長，只作 context |
| `none` | 沒有可用同步證據 |

正式 Stage 2 只使用：

- `temporal synchrony`
- `temporal confidence`

已移除的 formal evidence：

- `text_fingerprint_distance`
- `account_lifecycle_overlap`

移除原因：

1. Text fingerprint 在單一主題社群中容易測到 topic similarity，而不是 same operator。
2. Lifecycle / activation overlap 在 harvested 與 JG87919 類型案例中無法分開正反樣本。
3. 操控者可以刻意改變文字風格與帳號生命週期，因此這些訊號不穩定。

欄位仍可保留在 CSV schema 中作相容用途，但正式 ranking、validation、demo 解讀不使用它們。

## 6. 目前主要輸出

重要輸出檔：

| 檔案 | 說明 |
|---|---|
| `coordination-expansion/output/selected_seeds.csv` | 被選為 expansion 起點的 seed accounts |
| `coordination-expansion/output/seeds/<seed>/tiered_expansion_members.csv` | 每個 seed 的擴張成員與納入理由 |
| `coordination-expansion/output/stage2-verification/stage2_verification_evidence.csv` | pair-level temporal evidence |
| `coordination-expansion/output/candidate-validation/candidate_validation_table.csv` | account-level review priority table |
| `coordination-expansion/output/final-summary/final_group_summary.csv` | group-level review priority summary |
| `coordination-expansion/output/account-roles/account_role_table.csv` | group member role labels |

目前 pipeline output 摘要：

| 指標 | 數值 |
|---|---:|
| Selected seed accounts | 20 |
| Candidate seed groups | 20 |
| Group membership rows | 178 |
| Unique candidate accounts | 172 |
| Strong temporal sync pairs | 13 |
| Moderate temporal sync pairs | 8 |
| Robust temporal pairs | 3 |
| Moderate-review temporal pairs | 44 |

注意：這些數字是 review output，不是 accuracy。

## 7. 案例解讀

### 7.1 `harvested` 群組

`harvested` 是目前最清楚的高優先案例之一。它在 Stage 1 中具有：

- 8 名 group members
- 7 名 Tier 1 co-negative direct members
- 9 條 internal coordination edges
- 11 個 shared negative targets

Stage 2 中，`NectarineDirect936` 與 `harvested` 在 17 篇共同貼文中出現，包含：

- 2 次 5 分鐘內同步
- 4 次 30 分鐘內同步
- temporal label: `strong_temporal_sync`
- temporal confidence: `robust`

解讀：共同攻擊目標與穩定時間同步同時存在，因此是高優先審查群。

### 7.2 `JG87919` 類型案例

`JG87919` 類型案例用來提醒 Stage 1 不能直接當作最終判斷。它在 Stage 1 中形成：

- 15 名候選成員
- 34 條 internal coordination edges
- 20 個 shared negative targets

但後續檢查顯示，它更像同一社群中立場相近的獨立使用者，而不是同操控來源。這說明 co-negative-target 可以找到「共同攻擊同一批人」的群，但不能直接證明「同一操控者」。

## 8. Account Roles

為了讓群組結果更容易人工閱讀，系統產生簡單角色標註：

| Role | 中文說明 | 規則 |
|---|---|---|
| `leader_instigator` | 帶頭起鬨 | seed 本人 |
| `comment_attacker` | 留言攻擊者 | oppositional comment ratio >= 0.50 |
| `comment_supporter` | 留言支持者 | supportive comment ratio >= 0.50 |
| `context_member` | 背景成員 | 其他被 expansion 保留者 |

角色定位不是人格判斷，也不是 final verdict，只是幫助審查者理解群內行為分工。

## 9. Demo Site

`MCA-demo-site/` 是多頁靜態展示網站，包含：

| Page | 用途 |
|---|---|
| `index.html` | Executive overview 與 review priorities |
| `groups.html` | Risk group queue、關係圖、pair evidence |
| `accounts.html` | Individual abnormal account stream |
| `methodology.html` | Pipeline 與 signal pruning 說明 |

網站支援：

- 中文 / English 切換
- Client mode / Demo mode 切換
- MCA Sentinel 品牌與企業風險面板視覺

網站主要給老師、client 或非技術觀眾快速理解結果；研究方法仍以 pipeline 與 output tables 為準。

## 10. 重要設計決策

| 決策 | 原因 |
|---|---|
| MCA score 不作 final verdict | 單一帳號分數不能證明協同或操控 |
| co-negative-target 用於 discovery，不作 verdict | 正常同立場使用者也可能共同攻擊同一批人 |
| Stage 2 收斂為 pure temporal verification | text fingerprint 與 lifecycle 在本資料集中像噪音 |
| Demo site 與 project pipeline 分開 | 網站是展示層，不是分析方法本體 |
| 使用 Ollama 作主要 LLM backend | 之後較適合本地重跑與跨平台擴充 |
| Runner 拆成前半、後半、整份 | 目前 pipeline 尚在調整，拆開有利 debug 與重跑 |

## 11. 專題與論文主張

本專題不應說：

> 我們用 MCA 分數找到了網軍。

建議說：

> 我們提出一套可審查的協同行為偵測流程。先用 MCA 分數找可疑 seed，再用共同負向目標圖擴張為候選群，最後用短時間同步驗證，區分同立場活躍使用者與更可疑的協同行動。

這樣比較準確，也比較不會過度宣稱。

## 12. 目前限制

1. 缺乏大規模 ground truth，因此不能宣稱 supervised accuracy。
2. Temporal synchrony 仍可能受到熱門貼文與高活躍使用者影響。
3. Text fingerprint 與 lifecycle 已被測試但不適合作 formal evidence。
4. Reddit 原始資料缺少部分跨平台需要的欄位，例如完整 subreddit distribution。
5. 不同平台的 action traces 不同，未來若做插件或跨平台工具，需要重新定義平台行為。

## 13. 下一步建議

短期：

1. 將本文說明書內容整理進 4-6 頁投稿論文。
2. 將 final group summary 做成更清楚的結果表。
3. 為 harvested 與 JG87919 類型案例補上圖示與審查說明。
4. Demo site 保持展示層，不再塞入分析邏輯。

中期：

1. 建立人工標註的小型正反樣本集。
2. 測試 temporal threshold 在其他資料集上的穩定性。
3. 擴充不同平台的 action trace，例如 repost、share link、hashtag、like、reply chain。
4. 將 pipeline 包裝成可定期重跑的 monitoring workflow。

## 14. 快速交接摘要

一句話版本：

> MCA Detector 不是直接抓網軍，而是從可疑帳號出發，找出可能一起行動的群，再用時間同步證據排序審查優先級。

三層邏輯：

1. Account suspiciousness: MCA score
2. Group candidate discovery: co-negative-target expansion
3. Verification: temporal synchrony

最重要的輸出：

- `final_group_summary.csv`
- `stage2_verification_evidence.csv`
- `candidate_validation_table.csv`
- `account_role_table.csv`

最重要的觀念：

> 協同不等於網軍；共同攻擊目標不等於同操控者；因此我們用 temporal verification 來降低誤判。
