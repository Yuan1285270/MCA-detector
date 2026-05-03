# Reddit Bitcoin 2025 Crawler

## 蒐集範圍
- **Subreddit**：`Bitcoin`
- **關鍵字**：`bitcoin`
- **時間範圍**：2025-01-01 ~ 2026-01-01


- **資料來源**：

```text
Posts API:
https://api.pullpush.io/reddit/search/submission/

Comments API:
https://api.pullpush.io/reddit/search/comment/
```

---

## 輸出資料

### 1. posts

欄位：
- `post_id`
- `author`
- `created_utc`
- `title`
- `num_comments`

---

### 2. comments

欄位：
- `comment_id`
- `post_id`
- `author`
- `body`
- `created_utc`

---

## 專案資料夾結構

```bash
D:/RedditData_2025/
├─ reddit_full_collector.py
├─ posts_2025.csv
├─ comments_2025.csv
└─ README.md
```