## behavior_analysis.py

`behavior_analysis.py` 是帳號行為分析的核心程式，負責根據 Reddit 使用者的貼文與留言紀錄建立帳號層級的行為特徵，並透過異常偵測模型找出行為模式較可疑的帳號。

本程式會分析帳號的基本活躍程度、貼文與留言比例、時間間隔、短時間爆發行為、夜間活動比例與週末活動比例，最後使用 `IsolationForest` 進行異常偵測。

## 主要功能

- 讀取清理後的 Reddit 貼文資料
- 讀取合併後的 Reddit 留言資料
- 建立留言行為特徵
- 建立貼文行為特徵
- 合併貼文與留言特徵
- 建立帳號行為特徵表
- 使用 Isolation Forest 偵測可疑帳號
- 輸出可疑帳號排序表
- 繪製行為特徵分布圖

## 輸入檔案

| 檔案 | 說明 |
|---|---|
| `data/processed/cleaned_posts_2025.csv` | 清理後的 Reddit 貼文資料 |
| `data/processed/merged_reddit_2025.csv` | 合併後的 Reddit 留言與貼文資料 |

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `output/tables/behavior_features.csv` | 所有帳號的行為特徵表 |
| `output/tables/suspicious_accounts_behavior.csv` | 被模型判定為可疑的帳號清單 |

## 輸出圖表

程式會將圖表輸出到 `output/figures/`，包含：

| 圖表 | 說明 |
|---|---|
| `comment_count_distribution.png` | 每個帳號留言總數分布 |
| `post_count_distribution.png` | 每個帳號貼文總數分布 |
| `comment_post_ratio_distribution.png` | 留言/貼文比例分布 |
| `anomaly_score_distribution.png` | 所有帳號異常分數分布 |