# -*- coding: utf-8 -*-
import requests
import pandas as pd
import time
import os
from datetime import datetime

SAVE_DIR = r"D:\RedditData_2025"
POST_FILE = os.path.join(SAVE_DIR, "posts_2025.csv")
COMMENT_FILE = os.path.join(SAVE_DIR, "comments_2025.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now()}] {msg}")

# =========================
# 抓貼文
# =========================
def fetch_posts(year, month):
    start = int(datetime(year, month, 1).timestamp())
    end = int(datetime(year, month+1, 1).timestamp()) if month < 12 else int(datetime(year+1,1,1).timestamp())

    after = start
    all_posts = []

    while after < end:
        params = {
            "subreddit": "Bitcoin",
            "q": "bitcoin",
            "after": after,
            "before": end,
            "size": 100,
            "sort": "asc",
            "sort_type": "created_utc"
        }

        try:
            r = requests.get("https://api.pullpush.io/reddit/search/submission/", params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])

            if not data:
                break

            for p in data:
                all_posts.append({
                    "post_id": p["id"],
                    "author": p["author"],
                    "created_utc": p["created_utc"],
                    "title": p["title"],
                    "num_comments": p["num_comments"]
                })

            after = data[-1]["created_utc"] + 1
            log(f"posts: {len(all_posts)}")
            time.sleep(1)

        except Exception as e:
            log(f"error: {e}")
            time.sleep(10)

    return all_posts

# =========================
# 抓留言
# =========================
def fetch_comments(post_id):
    comments = []
    after = 0

    while True:
        params = {
            "link_id": f"t3_{post_id}",
            "size": 100,
            "after": after,
            "sort": "asc"
        }

        try:
            r = requests.get("https://api.pullpush.io/reddit/search/comment/", params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])

            if not data:
                break

            for c in data:
                comments.append({
                    "comment_id": c["id"],
                    "post_id": post_id,
                    "author": c["author"],
                    "body": c["body"],
                    "created_utc": c["created_utc"]
                })

            after = data[-1]["created_utc"] + 1
            time.sleep(0.5)

        except Exception as e:
            log(f"comment error: {e}")
            time.sleep(10)

    return comments

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    all_posts = []
    all_comments = []

    for m in range(1, 13):
        log(f"開始抓 {m} 月")

        posts = fetch_posts(2025, m)
        all_posts.extend(posts)

        for p in posts:
            cmts = fetch_comments(p["post_id"])
            all_comments.extend(cmts)
            log(f"{p['post_id']} comments: {len(cmts)}")

    pd.DataFrame(all_posts).to_csv(POST_FILE, index=False)
    pd.DataFrame(all_comments).to_csv(COMMENT_FILE, index=False)

    log("完成")