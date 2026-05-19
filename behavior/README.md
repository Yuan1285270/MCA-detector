# Behavior Module

這個資料夾是 legacy module，負責建立 account-level behavior features，並用 Isolation Forest 做基本異常偵測。

目前正式 pipeline 不再從這個資料夾作為主入口；需要的行為欄位已整合到新版 feature matrix / MCA scoring 流程。這個資料夾保留作為早期行為分析與可重現性參考。

它的定位是：

```text
raw / cleaned posts + comments -> behavior feature table -> anomaly labels
```

這裡的 anomaly score 不是 bot verdict，也不是 coordinated group evidence。它只能表示某個帳號的活動型態在資料集中比較異常，例如高頻、爆發式、夜間活動或短時間大量活動。

## Main Features

主要行為欄位包含：

```text
comment_count
post_count
comments_per_day
posts_per_day
unique_posts_commented
avg_post_num_comments
mean_comment_interval_seconds
std_comment_interval_seconds
mean_post_interval_seconds
std_post_interval_seconds
burst_ratio
night_activity_ratio
weekend_activity_ratio
comment_post_ratio
```

## Outputs

預設輸出：

```text
output/tables/behavior_features.csv
output/tables/suspicious_accounts_behavior.csv
```

`behavior_features.csv` 可被 account feature matrix 或 MCA scoring 使用。`suspicious_accounts_behavior.csv` 是行為異常帳號清單，只適合作為人工 review cue。

## Interpretation

正式報告建議講法：

```text
Behavior anomaly helps identify abnormal individual accounts.
It does not prove that an account belongs to a coordinated group.
Group-level claims still require graph expansion and temporal verification.
```
