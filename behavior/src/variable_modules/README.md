## ######################################################################################

## column_mapping.py

`column_mapping.py` 是欄位對應設定檔，負責集中管理不同原始資料欄位與專案標準欄位之間的對應關係。

本專案資料包含 Reddit 貼文、Reddit 留言與 BTC 價格資料。不同資料來源的欄位名稱可能不一致，因此本檔案透過 mapping dictionary 定義各資料表需要使用的欄位，讓前處理程式可以依照此設定統一欄位名稱。

## 主要功能

- 定義 Reddit 貼文資料欄位對應
- 定義 Reddit 留言資料欄位對應
- 定義 BTC 價格資料欄位對應
- 將資料來源相關的欄位設定集中管理
- 降低更換資料來源時對核心分析程式的影響

## 欄位對應表

### POST_SOURCE_COLUMNS

`POST_SOURCE_COLUMNS` 定義 Reddit 貼文資料需要使用的欄位。

| 標準欄位 | 原始欄位 | 說明 |
|---|---|---|
| `post_id` | `post_id` | 貼文 ID |
| `author` | `author` | 貼文作者 |
| `created_utc` | `created_utc` | 貼文建立時間 |
| `title` | `title` | 貼文標題 |
| `selftext` | `selftext` | 貼文內文 |
| `num_comments` | `num_comments` | 貼文留言數 |

### COMMENT_SOURCE_COLUMNS

`COMMENT_SOURCE_COLUMNS` 定義 Reddit 留言資料需要使用的欄位。

| 標準欄位 | 原始欄位 | 說明 |
|---|---|---|
| `comment_id` | `comment_id` | 留言 ID |
| `link_id` | `link_id` | 留言所屬貼文的識別欄位 |
| `parent_id` | `parent_id` | 上層留言或貼文 ID |
| `author` | `author` | 留言作者 |
| `body` | `body` | 留言內容 |
| `created_utc` | `created_utc` | 留言建立時間 |

### BTC_SOURCE_COLUMNS

`BTC_SOURCE_COLUMNS` 定義 BTC 價格資料需要使用的欄位。

| 標準欄位 | 原始欄位 | 說明 |
|---|---|---|
| `timestamp` | `timestamp` | 價格時間 |
| `close` | `close` | BTC 收盤價 |

## 設計目的

此檔案的目的在於將「會隨資料來源改變的欄位名稱」集中管理。當資料來源改變，或原始 CSV 欄位名稱不同時，只需要修改 `column_mapping.py` 中的對應關係，前處理與分析程式即可繼續使用統一後的欄位名稱。

這樣可以降低資料來源變動對核心分析邏輯的影響，提升專案的可維護性與擴充性。

## ######################################################################################

## standardize.py

`standardize.py` 是資料前處理流程中的標準化模組，負責將原始資料清理並轉換成專案後續分析可使用的統一格式。

本檔案主要處理三種資料：

- Reddit 貼文資料
- Reddit 留言資料
- BTC 價格資料

程式會依照 `column_mapping.py` 中定義的欄位對應表，將不同來源的欄位名稱轉換成專案統一欄位，並進行缺失值處理、無效資料移除、時間格式轉換、數值欄位轉換、重複資料刪除與排序。

## 主要功能

- 統一原始資料欄位名稱
- 移除無效作者與無效留言內容
- 處理空值與資料型態
- 將 Reddit 的 Unix timestamp 轉換為 datetime
- 從留言資料的 `link_id` 產生可合併用的 `post_id`
- 將 BTC 價格時間轉換為 UTC datetime
- 移除重複資料
- 依時間排序資料

## 無效資料定義

### INVALID_AUTHORS

以下作者會被視為無效並移除：

- `[deleted]`
- `[removed]`
- `nan`
- `None`
- 空字串

這些資料無法對應到有效帳號，因此不適合用於帳號行為分析。

### INVALID_BODIES

以下留言內容會被視為無效並移除：

- `[deleted]`
- `[removed]`
- 空字串

這些留言缺乏有效文字內容，因此不適合用於文本分析或留言行為分析。

## 函式說明

### standardize_posts(raw_df)

清理並標準化 Reddit 貼文資料。

處理內容包含：

1. 依照 `POST_SOURCE_COLUMNS` 重新命名欄位
2. 清理作者欄位，移除無效作者
3. 清理貼文標題，移除沒有標題的貼文
4. 清理貼文內文
5. 將留言數 `num_comments` 轉為整數
6. 將 `created_utc` 轉為數值
7. 新增 `created_datetime` 時間欄位
8. 依 `post_id` 移除重複貼文
9. 依貼文建立時間排序

輸出結果為標準化後的貼文資料表。

### standardize_comments(raw_df)

清理並標準化 Reddit 留言資料。

處理內容包含：

1. 依照 `COMMENT_SOURCE_COLUMNS` 重新命名欄位
2. 清理作者欄位，移除無效作者
3. 清理留言內容，移除 `[deleted]`、`[removed]` 與空留言
4. 將 `created_utc` 轉為數值
5. 新增 `created_datetime` 時間欄位
6. 清理 `link_id` 與 `parent_id`
7. 從 `link_id` 移除 `t3_` 前綴，產生 `post_id`
8. 依 `comment_id` 移除重複留言
9. 依留言建立時間排序

輸出結果為標準化後的留言資料表。

### standardize_btc(raw_df)

清理並標準化 BTC 價格資料。

處理內容包含：

1. 依照 `BTC_SOURCE_COLUMNS` 重新命名欄位
2. 將 `timestamp` 轉換為 UTC datetime
3. 將 `close` 轉換為數值
4. 移除時間或價格缺失的資料
5. 依時間排序

輸出結果為標準化後的 BTC 價格資料表。

## 在專案中的角色

`standardize.py` 屬於資料前處理流程的核心模組。它本身不負責最終帳號分析或可疑分數計算，而是先將原始資料整理成乾淨、欄位一致、時間格式一致的資料，供後續模組使用。

後續流程例如：

- `merge_data.py`
- `behavior_analysis.py`
- `market_alignment.py`
- `account_prediction_performance.py`

都會依賴標準化後的資料進行分析。

## 與其他檔案的關係

| 檔案 | 關係 |
|---|---|
| `column_mapping.py` | 提供原始欄位與標準欄位的對應關係 |
| `config.py` | 提供資料輸入與輸出路徑 |
| `merge_data.py` | 使用標準化後的 posts 與 comments 進行合併 |
| `behavior_analysis.py` | 使用清理後資料計算帳號行為特徵 |
| `market_alignment.py` | 使用標準化後的 BTC 價格資料與留言時間進行市場對齊 |

## 設計目的

本檔案的目的在於將資料清理與格式統一集中處理，降低後續分析程式對原始資料格式的依賴。  
當資料來源改變時，只要透過 `column_mapping.py` 調整欄位對應，並由 `standardize.py` 統一輸出標準格式，後續核心分析流程就可以維持相對穩定。

## ######################################################################################

## io_utils.py

`io_utils.py` 是專案中的檔案輸入與輸出工具模組，負責提供共用的資料夾建立、CSV 讀取與 CSV 儲存功能。

此檔案不包含資料分析邏輯，而是將常用的 I/O 操作集中管理，讓其他程式可以重複使用相同函式，減少程式碼重複並提升可維護性。

## 主要功能

- 建立專案需要的輸出與處理資料夾
- 讀取 CSV 檔案
- 將 DataFrame 儲存為 CSV 檔案

## 函式說明

### ensure_dirs()

建立專案執行時需要的資料夾，包含：

- `data/processed/`
- `output/`
- `output/tables/`
- `output/figures/`

此函式使用 `Path.mkdir(parents=True, exist_ok=True)`，因此當資料夾不存在時會自動建立；若資料夾已存在，則不會產生錯誤。

### read_csv(path)

讀取指定路徑的 CSV 檔案，並回傳 pandas DataFrame。

此函式是對 `pandas.read_csv()` 的簡單封裝，讓專案內的 CSV 讀取方式保持一致。

### save_csv(df, path)

將 pandas DataFrame 儲存為 CSV 檔案。

輸出設定包含：

- `index=False`：不輸出 DataFrame 索引欄位
- `encoding="utf-8-sig"`：使用 UTF-8 with BOM 編碼，降低中文在 Excel 中顯示亂碼的可能性

## 在專案中的角色

`io_utils.py` 屬於共用工具模組，主要提供其他程式使用。例如：

- `preprocess.py` 可使用 `read_csv()` 讀取原始資料，並使用 `save_csv()` 儲存清理後資料
- `merge_data.py` 可使用 `ensure_dirs()` 確認輸出資料夾存在，再儲存合併後資料
- 分析程式可使用 `save_csv()` 輸出分析結果表格

## 設計目的

將檔案讀寫與資料夾建立集中管理，可以避免每支程式重複撰寫相同的 I/O 程式碼。  
當未來需要調整讀檔、存檔或編碼設定時，只需要修改 `io_utils.py`，其他程式即可共同套用新的設定。

## ######################################################################################

## preprocess.py

`preprocess.py` 是資料前處理流程的主控程式，負責讀取 Reddit 原始貼文與留言資料，呼叫標準化函式進行清理，並輸出清理後的資料檔案。

本程式本身不直接撰寫詳細清理邏輯，而是整合 `config.py`、`io_utils.py` 與 `standardize.py`：

- `config.py`：提供原始資料與輸出資料的路徑
- `io_utils.py`：提供建立資料夾、讀取 CSV、儲存 CSV 的工具函式
- `standardize.py`：提供 Reddit 貼文與留言資料的標準化清理函式

## 主要功能

- 建立必要的資料夾
- 讀取 Reddit 原始貼文資料
- 讀取 Reddit 原始留言資料
- 呼叫 `standardize_posts()` 清理貼文資料
- 呼叫 `standardize_comments()` 清理留言資料
- 輸出清理後的 posts 與 comments CSV
- 顯示前處理前後的資料大小與資料預覽

## 輸入檔案

| 檔案 | 說明 |
|---|---|
| `data/raw/reddit_posts_2025.csv` | Reddit 原始貼文資料 |
| `data/raw/reddit_comments_2025.csv` | Reddit 原始留言資料 |

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `data/processed/cleaned_posts_2025.csv` | 清理後的 Reddit 貼文資料 |
| `data/processed/cleaned_comments_2025.csv` | 清理後的 Reddit 留言資料 |

## ######################################################################################