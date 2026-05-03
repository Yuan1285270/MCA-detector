## ######################################################################################

## main.py

`main.py` 是資料讀取與初步檢查程式，主要用於確認 Reddit 原始貼文資料與留言資料是否能成功讀取。

程式會讀取以下兩個 CSV 檔案：

- `data/raw/reddit_posts_2025.csv`
- `data/raw/reddit_comments_2025.csv`

讀取後會分別輸出：

- 資料筆數與欄位數
- 欄位名稱
- 前 5 筆資料內容

## ######################################################################################

# config.py

`config.py` 是專案的路徑設定檔，負責集中管理資料輸入、資料處理與分析輸出的相關路徑。

本檔案使用 Python 的 `pathlib.Path` 建立跨平台的路徑物件，讓其他程式可以透過匯入變數的方式使用固定路徑，避免在不同程式中重複撰寫路徑字串。

## 主要功能

- 定義專案根目錄
- 定義資料資料夾位置
- 定義原始資料檔案路徑
- 定義前處理後資料檔案路徑
- 定義分析結果輸出路徑

## 路徑分類

### Base Paths

此區塊定義專案中常用的基礎資料夾路徑：

- `PROJECT_ROOT`：專案根目錄
- `DATA_DIR`：資料資料夾
- `RAW_DIR`：原始資料資料夾
- `PROCESSED_DIR`：前處理後資料資料夾
- `OUTPUT_DIR`：分析結果輸出資料夾
- `TABLE_DIR`：輸出表格資料夾
- `FIGURE_DIR`：輸出圖表資料夾

### Raw Files

此區塊定義原始資料檔案路徑：

- `RAW_POSTS_PATH`：Reddit 貼文原始資料
- `RAW_COMMENTS_PATH`：Reddit 留言原始資料
- `RAW_BTC_PATH`：BTC 每小時收盤價資料

### Processed Files

此區塊定義前處理後的資料檔案路徑：

- `CLEAN_POSTS_PATH`：清理後的 Reddit 貼文資料
- `CLEAN_COMMENTS_PATH`：清理後的 Reddit 留言資料
- `MERGED_REDDIT_PATH`：合併後的 Reddit 貼文與留言資料

### Output Tables

此區塊定義分析結果表格的輸出路徑：

- `BEHAVIOR_FEATURES_PATH`：帳號行為特徵表
- `SUSPICIOUS_BEHAVIOR_PATH`：可疑帳號行為分析結果
- `DIRECTIONAL_COMMENTS_PATH`：帶有市場報酬資訊的方向性留言資料
- `ACCOUNT_PERFORMANCE_PATH`：帳號預測表現分析結果
- `TOP_PREDICTIVE_PATH`：預測能力較高的帳號排名
- `TOP_CONTRARIAN_PATH`：反向指標帳號排名

## ######################################################################################

## merge_data.py

`merge_data.py` 負責將清理後的 Reddit 貼文資料與留言資料進行合併，產生後續分析所需的完整 Reddit 資料表。

本程式會讀取：

- `data/processed/cleaned_posts_2025.csv`
- `data/processed/cleaned_comments_2025.csv`

並透過 `post_id` 將每一則留言對應回原始貼文，補上該留言所屬貼文的作者、建立時間、標題、內文與留言數等資訊。

### 主要流程

1. 建立必要的輸出資料夾
2. 讀取清理後的貼文與留言資料
3. 從貼文資料中選取合併所需欄位
4. 將貼文欄位重新命名，加上 `post_` 前綴，避免與留言欄位混淆
5. 使用 `post_id` 進行 left merge
6. 檢查成功對應貼文與未對應貼文的留言數量
7. 移除沒有成功對應到貼文的留言
8. 依照留言時間與留言 ID 排序
9. 輸出合併後資料至 `data/processed/merged_reddit_2025.csv`

### 輸入檔案

| 檔案 | 說明 |
|---|---|
| `cleaned_posts_2025.csv` | 清理後的 Reddit 貼文資料 |
| `cleaned_comments_2025.csv` | 清理後的 Reddit 留言資料 |

### 輸出檔案

| 檔案 | 說明 |
|---|---|
| `merged_reddit_2025.csv` | 合併後的 Reddit 留言與貼文資料 |

### 合併邏輯

本程式以留言資料為主表，使用 `post_id` 將貼文資料合併進留言資料中。

採用 `left merge` 的原因是希望保留所有留言資料，並檢查每則留言是否能成功對應到原始貼文。若某些留言無法找到對應貼文，程式會在合併檢查階段統計數量，並在後續步驟中將這些缺少貼文資訊的留言移除。

### 欄位命名設計

為避免合併後欄位名稱混淆，貼文資料中的欄位會加上 `post_` 前綴，例如：

- `author` → `post_author`
- `created_utc` → `post_created_utc`
- `created_datetime` → `post_created_datetime`
- `title` → `post_title`
- `selftext` → `post_selftext`
- `num_comments` → `post_num_comments`

這樣可以清楚區分留言本身的資訊與留言所屬貼文的資訊。

### 在專案中的角色

`merge_data.py` 屬於資料前處理流程的一部分，主要負責整合清理後的 Reddit 貼文與留言資料。它不是核心分析模型，而是為後續的帳號行為分析、文本分析或市場對齊分析提供更完整的資料基礎。

## ######################################################################################