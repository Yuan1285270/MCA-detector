# Arctic Shift Reddit Crawler - Q1 2025


## 蒐集範圍
- **Subreddit**：`CryptoCurrency`
- **時間範圍**：2025-01-01 ~ 2025-04-01
- **資料來源**：
  - Posts API：`/api/posts/search`
  - Comments API：`/api/comments/search`

---

## 輸出資料

### 1. posts

欄位：
- `post_id`
- `author`
- `created_utc`
- `title`
- `selftext`
- `num_comments`

### 2. comments

欄位：
- `comment_id`
- `link_id`
- `parent_id`
- `author`
- `body`
- `created_utc`

---

## 專案資料夾結構
```bash
/opt/arcticshift-q1/
├─ arctic_q1_collector.py
├─ run.sh
├─ output/
│  ├─ reddit_posts_CryptoCurrency_2025_Q1.csv
│  └─ reddit_comments_CryptoCurrency_2025_Q1.csv
├─ state/
│  └─ checkpoint_CryptoCurrency_2025_Q1.json
└─ logs/
   ├─ collector.log
   ├─ stdout.log
   └─ stderr.log