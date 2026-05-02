# SocialMedia_LLM

這個專案主要用來分析加密貨幣相關的 Reddit 討論，重點分成兩層：

- 貼文層級：用 Gemini + Vertex AI + RAG 評估每篇貼文的情緒傾向與操縱式修辭強度。
- 互動層級：分析留言對文章的支持/反對關係，之後可轉成帳號間 signed weighted adjacency matrix。

目前的資料流分成三段：`data-cleaning/` 負責原始資料清理與共用 processed data，`gemini-cloud/` 負責正式 Gemini / Vertex AI 批次分析，`ollama-local/` 保留作為本地 Ollama 替代實驗。

## 專案重點

- 主要分析對象：Reddit 貼文與直回文章的留言
- 貼文任務：為每篇貼文產生
  - `sentiment_score`
  - `manipulative_rhetoric_score`
  - `rhetoric_tags`
- 留言任務：為每則直回文章的留言產生
  - `feedback_label`
  - `feedback_score`
  - `edge_weight`
- 主要模型：Gemini on Vertex AI
- 主要輔助方式：RAG retrieval，提供背景參考但不直接替貼文貼標

## 目前建議使用的流程

1. 將原始 Reddit 貼文與留言放進 `data-cleaning/source_data/`
2. 用 `data-cleaning/` 產生共用 processed data
3. 將 processed data 送進 `gemini-cloud/` 做正式 Gemini / Vertex AI 批次分析
4. 如需本地替代實驗，將同一份 processed data 送進 `ollama-local/`
5. 最終產出可繼續統計、人工審查或建 network matrix 的 CSV

## 目錄說明

### 主要資料夾

- `data-cleaning/`
  - 原始資料清理、共用 processed data 產生
  - 也保留 post-level clustering 作為探索性分析
- `gemini-cloud/`
  - 正式 Gemini / Vertex AI 批次分析、續跑與輸出結果

### 實驗性資料夾

- `ollama-local/`
  - 本地 Ollama + PDF RAG 的實驗版本
- `RAG DATA/`
  - RAG 相關參考資料與 PDF

### 本機資料與輸出

以下資料夾預設不放進 GitHub，請依需要在本機或雲端環境自行準備：

- `Export/`
  - 最終正式輸出的完整 CSV / gzip 檔
- `data-cleaning/source_data/`
  - 原始 Reddit 匯入資料
- `data-cleaning/processed_data/`
  - 清理後、尚未送模型或作為模型輸入的中間資料
- `data-cleaning/post-clustering/output/`
  - post-level clustering 的探索性輸出
- `gemini-cloud/output/`
  - Gemini 批次分析輸出
- `Archive/`
  - 舊腳本、工作檔、備份與本機 SDK

## `data-cleaning/` 在做什麼

`data-cleaning/` 先把原始資料整理成 Gemini 和 Ollama 都可以共用的 processed data：

- `preprocess_posts.py`
  - 讀取 `source_data/reddit_posts_2025.csv`
  - 清理 `title` / `selftext`
  - 合併成可分析文字
  - 去掉空值、過短內容、重複貼文
  - 輸出 `processed_data/processed_data.csv`

- `preprocess_comments.py`
  - 讀取 `source_data/reddit_comments_2025.csv`
  - 用 `processed_data/processed_data.csv` 的 `post_id` 當保留文章清單
  - 移除掛在已剔除文章底下的留言
  - 移除空白、`[removed]`、`[deleted]` 留言
  - 主輸出只保留直回文章的留言
  - comment-to-comment replies 另存備用

- `post-clustering/`
  - post-level clustering 的探索性分析
  - 不作為正式 suspicious account group detection 主線

## `gemini-cloud/` 在做什麼

`gemini-cloud/` 有兩個核心分析檔案：

- `analyze_posts_with_gemini.py`
  - 主力批次分析腳本
  - 適合大量資料長時間執行
  - 支援區間批次、續跑、定期 autosave
  - 會把每篇貼文送進 Gemini + RAG 分析

- `analyze_comment_feedback_with_gemini.py`
  - 分析留言對原文章的 stance / feedback
  - 不使用 RAG，直接看文章內容與留言內容
  - 輸出可轉成 adjacency matrix 的 edge 欄位

## 資料流程

```text
data-cleaning/source_data/reddit_posts_2025.csv
    -> preprocess_posts.py
    -> data-cleaning/processed_data/processed_data.csv
    -> analyze_posts_with_gemini.py
    -> gemini-cloud/output/post_analysis_<START>_<END>.csv
```

留言互動流程：

```text
data-cleaning/source_data/reddit_comments_2025.csv
    + data-cleaning/processed_data/processed_data.csv
    -> preprocess_comments.py
    -> data-cleaning/processed_data/processed_comments.csv
    -> analyze_comment_feedback_with_gemini.py
    -> gemini-cloud/output/comments/comment_feedback_<START>_<END>.csv
```

comment-to-comment replies 目前不進主 adjacency matrix，會先保留在：

```text
data-cleaning/temp/comment_replies/comment_replies.csv
```

## 輸入資料格式

目前主流程假設貼文資料至少有以下欄位：

- `post_id`
- `author`
- `created_utc`
- `title`
- `selftext`
- `num_comments`

## 貼文分析輸出內容

分析後的輸出 CSV 會包含原本貼文欄位，並新增像下面這些結果：

- `rag_analysis`
- `sentiment_score`
- `sentiment_reason`
- `manipulative_rhetoric_score`
- `manipulative_rhetoric_reason`
- `rhetoric_tags`

其中：

- `sentiment_score`：情緒極性與強度，範圍約 `-100 ~ 100`
- `manipulative_rhetoric_score`：操縱式修辭強度，範圍約 `0 ~ 100`
- `rhetoric_tags`：修辭標籤，例如 `urgency`、`fear`、`overconfidence`、`authority_claim`、`bandwagon`、`us_vs_them`、`call_to_action`、`emotional_amplification`、`analytical_neutral`

## 留言互動輸出內容

`data-cleaning/processed_data/processed_comments.csv` 目前只保留直回文章的留言，也就是 `parent_type = post`。這樣每則留言都可以明確轉成一條 account-to-account edge：

```text
source_author -> target_author
```

其中：

- `source_author`：留言作者
- `target_author`：文章作者
- `edge_type`：目前固定為 `comment_to_post`
- `feedback_label`：留言對文章的 stance / feedback 類別
- `feedback_score`：連續分數，範圍 `-100 ~ 100`
- `edge_weight`：`feedback_score / 100`，範圍 `-1.0 ~ 1.0`

`feedback_label` 定義：

- `supportive`：留言支持、同意、鼓勵、建設性回答、驗證或補充文章。
- `oppositional`：留言反對、批評、嘲諷、攻擊、警告、否定或削弱文章。
- `neutral`：留言主要是事實性、資訊性、無關，或沒有明確評價文章。
- `mixed`：留言同時包含支持與批評。
- `unclear`：無法從文字判斷其對文章的立場。

`feedback_score` 定義：

- `-100`：強烈反對、敵意或否定文章
- `-50`：明確負向/批評性回饋
- `0`：中性、無關或不明確
- `50`：明確正向/支持性回饋
- `100`：強烈支持、認同或驗證文章

設計上，`feedback_score` / `edge_weight` 是之後建 weighted adjacency matrix 的主要數值欄位；`feedback_label` 則用於解釋、檢查與統計。

## 快速開始

### 1. 準備環境

你需要：

- Python 虛擬環境
- 可用的 Google Cloud / Vertex AI 專案
- 已啟用 Vertex AI
- 能夠存取對應的 RAG corpus

如果要使用 GCP 認證，先確認本機已完成登入與授權。

### 2. 清理資料

在 repo 根目錄進入 `llm/` 後執行：

```bash
cd llm
../.venv/bin/python data-cleaning/preprocess_posts.py
```

執行完成後會得到：

```text
data-cleaning/processed_data/processed_data.csv
```

### 3. 清理留言資料

在 repo 根目錄進入 `llm/` 後執行：

```bash
cd llm
../.venv/bin/python data-cleaning/preprocess_comments.py
```

執行完成後會得到：

```text
data-cleaning/processed_data/processed_comments.csv
data-cleaning/temp/comment_replies/comment_replies.csv
```

### 4. 執行 Gemini 貼文分析

```bash
cd gemini-cloud
../../.venv/bin/python analyze_posts_with_gemini.py
```

預設情況下，`analyze_posts_with_gemini.py` 會：

- 讀取 `../data-cleaning/processed_data/processed_data.csv`
- 分析 `START_ROW` 到 `END_ROW`
- 每隔幾筆自動存檔
- 若輸出檔已存在，則從中斷處續跑

輸出位置：

```text
gemini-cloud/output/post_analysis_<START_ROW>_<END_ROW>.csv
```

### 5. 執行留言 feedback 分析

```bash
cd gemini-cloud
../../.venv/bin/python analyze_comment_feedback_with_gemini.py
```

預設情況下，`analyze_comment_feedback_with_gemini.py` 會：

- 讀取 `../data-cleaning/processed_data/processed_comments.csv`
- merge `../data-cleaning/processed_data/processed_data.csv` 取得文章作者與文章內容
- 分析 `START_ROW` 到 `END_ROW`
- 每隔幾筆自動存檔
- 若輸出檔已存在，則從中斷處續跑

輸出位置：

```text
gemini-cloud/output/comments/comment_feedback_<START_ROW>_<END_ROW>.csv
```

## 常用可調參數

`gemini-cloud/analyze_posts_with_gemini.py` 內目前最常調整的是：

- `PROJECT_ID`
- `LOCATION`
- `RAG_CORPUS_PATH`
- `MODEL_NAME`
- `START_ROW`
- `END_ROW`
- `MAX_CHARS`
- `SAVE_EVERY`
- `SLEEP_SECONDS`

如果要分批跑大資料，通常只要修改：

- `START_ROW`
- `END_ROW`

例如先跑 `0-5000`，再跑 `5000-10000`。

`gemini-cloud/analyze_comment_feedback_with_gemini.py` 也使用同樣的 batch 設定方式：

- `START_ROW`
- `END_ROW`
- `SAVE_EVERY`
- `SLEEP_SECONDS`
- `POST_MAX_CHARS`
- `COMMENT_MAX_CHARS`

## 使用上的設計原則

這個流程的 prompt 有幾個明確限制：

- 只分析文字本身
- 不直接推斷作者是不是 bot
- 不直接推定惡意或協同行為
- 不做超出文本證據的事實判定
- RAG 只作背景參考，不直接拿檢索內容當答案

換句話說，這個專案比較像是在做：

- 文字層級的修辭風險分析
- 可疑話術強度量化
- 帳號互動的支持/反對關係估計
- 後續人工審查前的前處理

而不是直接做帳號層級的 bot detection。

## 研究設計依據

### 貼文修辭分析

貼文分析模組定位為 text-level rhetoric analysis。它只分析文本本身的情緒、操縱式修辭與修辭標籤，不直接推論作者是不是 bot，也不直接判斷帳號是否惡意。

目前 production tags 採用 9 個 text-level rhetoric labels：

- `urgency`
- `fear`
- `overconfidence`
- `authority_claim`
- `bandwagon`
- `us_vs_them`
- `call_to_action`
- `emotional_amplification`
- `analytical_neutral`

### 留言 feedback / stance 分析

留言分析模組不是一般 sentiment analysis，而是 target-specific feedback / stance detection。目標是判斷留言者對「原文章」的支持或反對，而不是只判斷留言本身情緒。

這個設計參考 stance detection 的研究問題設定。SemEval-2016 Task 6 將 stance detection 定義為：給定文本與 target，判斷作者是 in favor of target、against target，或 neither。該文也指出 stance 和 sentiment 不同：負面情緒文字可以支持 target，正面情緒文字也可以反對 target。

參考文獻：

- Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., & Cherry, C. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. ACL Anthology. https://aclanthology.org/S16-1003/

### Signed weighted network 設計

留言 feedback 之後會轉成 account-level directed edges：

```text
source_author -> target_author
```

其中 `edge_weight = feedback_score / 100`。這使得互動關係可以形成 signed weighted directed network：正值代表支持，負值代表反對，絕對值代表強度。

這個設計和 signed network / trust network 研究的資料形式相近。例如 SNAP 的 Bitcoin Alpha / Bitcoin OTC datasets 使用 `SOURCE, TARGET, RATING, TIME` 形式記錄帶方向、帶正負權重的使用者關係。

參考資料：

- SNAP Bitcoin Alpha signed network dataset: https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- SNAP Bitcoin OTC signed network dataset: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- Kumar, S., Spezzano, F., Subrahmanian, V. S., & Faloutsos, C. (2016). Edge Weight Prediction in Weighted Signed Networks. IEEE ICDM.

## 目前狀態

目前 LLM 模組的實際主線是 `gemini-cloud/`。如果只是想跑得通、延續既有結果，建議先不要碰其他實驗資料夾，直接專注在這條流程即可。

## 後續可補強方向

- 補上 `requirements.txt` 或 `pyproject.toml`
- 把 `gemini-cloud/` 再拆成更清楚的 pipeline modules
- 補上結果彙整與評估腳本
- 將批次設定改成 CLI 參數，而不是直接改 Python 常數
- 將 comment feedback 輸出聚合成 adjacency matrix / edge list
