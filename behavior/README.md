# Behavior Analysis Module

這個資料夾是 MCA-detector 專案中的 behavior analysis 模組，主要用來分析 Reddit 帳號的活動行為，以及留言方向與 BTC 價格變化之間的關係。

它和 `../llm/` 裡的 LLM 分析流程是分開的模組。

## 模組目的

behavior 模組主要關注帳號層級的行為特徵，例如：

- 發文與留言頻率
- 活躍天數與每日活動量
- 留言/發文比例
- 短時間爆發式活動
- 夜間與週末活動比例
- 使用 Isolation Forest 做異常帳號評分
- 將方向性加密貨幣留言與未來 BTC 報酬對齊
- 排名較具 predictive 或 contrarian 特徵的帳號

## 資料夾結構

```text
behavior/
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── merge_data.py
│   ├── behavior_analysis.py
│   ├── market_alignment.py
│   ├── account_prediction_performance.py
│   ├── standardize.py
│   ├── io_utils.py
│   └── column_mapping.py
├── data/      # local raw and processed data, not committed
└── output/    # generated tables and figures, not committed
```

其中：

- `src/`：主要程式碼
- `data/`：本機資料，不放進 GitHub
- `output/`：分析產生的表格與圖表，不放進 GitHub

## 本機資料需求

請將原始資料放在：

```text
behavior/data/raw/
```

預期檔名：

```text
reddit_posts_2025.csv
reddit_comments_2025.csv
BTCUSDT_1h_close_2025.csv
```

這些資料檔會被 Git 忽略，不會推上 GitHub。

## 執行流程

以下指令請在 `behavior/` 資料夾中執行，這樣相對路徑才會正確。

### 1. 清理 Reddit 原始資料

```bash
cd behavior
../.venv/bin/python src/preprocess.py
```

輸出：

```text
data/processed/cleaned_posts_2025.csv
data/processed/cleaned_comments_2025.csv
```

### 2. 合併貼文與留言資料

```bash
../.venv/bin/python src/merge_data.py
```

輸出：

```text
data/processed/merged_reddit_2025.csv
```

### 3. 建立帳號行為特徵並偵測異常

```bash
../.venv/bin/python src/behavior_analysis.py
```

輸出：

```text
output/tables/behavior_features.csv
output/tables/suspicious_accounts_behavior.csv
output/figures/
```

### 4. 將方向性留言與 BTC 報酬對齊

```bash
../.venv/bin/python src/market_alignment.py
```

輸出：

```text
output/tables/directional_comments_with_returns.csv
```

### 5. 排名 predictive / contrarian 帳號

```bash
../.venv/bin/python src/account_prediction_performance.py
```

輸出：

```text
output/tables/account_prediction_performance.csv
output/tables/top_predictive_accounts.csv
output/tables/top_contrarian_accounts.csv
```

## 補充說明

- `src/config.py` 集中管理輸入與輸出路徑。
- `src/column_mapping.py` 負責將原始 CSV 欄位名稱對應到內部標準欄位。
- `src/main.py` 目前只是簡單檢查 raw data 的腳本，不是主要 pipeline runner。
- 產生的資料、輸出表格、圖表與本機環境都不會放進 GitHub。
