import os
import csv
import json
import time
import random
import requests
from datetime import datetime, timedelta, timezone

BASE_POSTS = "https://arctic-shift.photon-reddit.com/api/posts/search"
BASE_COMMENTS = "https://arctic-shift.photon-reddit.com/api/comments/search"

SUBREDDIT = "CryptoCurrency"
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)

ROOT_DIR = "/opt/arcticshift-q1"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
STATE_DIR = os.path.join(ROOT_DIR, "state")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

POSTS_CSV = os.path.join(OUTPUT_DIR, "reddit_posts_CryptoCurrency_2025_Q1.csv")
COMMENTS_CSV = os.path.join(OUTPUT_DIR, "reddit_comments_CryptoCurrency_2025_Q1.csv")
CHECKPOINT_FILE = os.path.join(STATE_DIR, "checkpoint_CryptoCurrency_2025_Q1.json")
RUN_LOG = os.path.join(LOG_DIR, "collector.log")

POST_FIELDS = ["post_id", "author", "created_utc", "title", "selftext", "num_comments"]
COMMENT_FIELDS = ["comment_id", "link_id", "parent_id", "author", "body", "created_utc"]

TIMEOUT = 60
SLEEP_MIN = 8
SLEEP_MAX = 15
DAY_SLEEP_MIN = 20
DAY_SLEEP_MAX = 40
BACKOFF_START = 60
BACKOFF_MAX = 1800
LIMIT = 100


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def ensure_csv(path, fields):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {
            "current_day": START_DATE.strftime("%Y-%m-%d"),
            "phase": "posts",
            "pending_post_ids": [],
            "comment_post_index": 0,
        }

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def dt_to_ts(dt):
    return int(dt.timestamp())


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def sleep_random(a, b):
    sec = random.uniform(a, b)
    log(f"sleep {sec:.1f}s")
    time.sleep(sec)


def safe_get(url, params):
    backoff = BACKOFF_START

    while True:
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                log(f"HTTP 429, backoff {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            else:
                log(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(min(backoff, 300))
                backoff = min(backoff * 2, BACKOFF_MAX)
        except requests.RequestException as e:
            log(f"Request exception: {repr(e)}, backoff {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)


def append_posts(rows):
    if not rows:
        return
    with open(POSTS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=POST_FIELDS)
        writer.writerows(rows)


def append_comments(rows):
    if not rows:
        return
    with open(COMMENTS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMMENT_FIELDS)
        writer.writerows(rows)


def normalize_post(post):
    return {
        "post_id": post.get("id"),
        "author": post.get("author"),
        "created_utc": post.get("created_utc"),
        "title": post.get("title"),
        "selftext": post.get("selftext"),
        "num_comments": post.get("num_comments"),
    }


def normalize_comment(comment):
    return {
        "comment_id": comment.get("id"),
        "link_id": comment.get("link_id"),
        "parent_id": comment.get("parent_id"),
        "author": comment.get("author"),
        "body": comment.get("body"),
        "created_utc": comment.get("created_utc"),
    }


def fetch_posts_for_day(day_dt):
    after_ts = dt_to_ts(day_dt)
    before_ts = dt_to_ts(day_dt + timedelta(days=1))
    all_posts = []
    seen_ids = set()
    cursor = None

    while True:
        params = {
            "subreddit": SUBREDDIT,
            "after": after_ts if cursor is None else cursor,
            "before": before_ts,
            "limit": LIMIT,
            "sort": "asc",
        }

        r = safe_get(BASE_POSTS, params)
        data = r.json().get("data", [])
        log(f"posts fetched = {len(data)}, after={params['after']}, before={before_ts}")

        if not data:
            break

        new_rows = []
        for post in data:
            pid = post.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            all_posts.append(post)
            new_rows.append(normalize_post(post))

        if new_rows:
            append_posts(new_rows)
            log(f"新增 posts {len(new_rows)} 筆")

        last_created = data[-1].get("created_utc")
        if not last_created:
            break

        cursor = int(last_created) + 1
        sleep_random(SLEEP_MIN, SLEEP_MAX)

    return all_posts


def fetch_comments_for_post(post_id):
    seen_ids = set()
    cursor = None

    while True:
        params = {
            "link_id": f"t3_{post_id}",
            "limit": LIMIT,
            "sort": "asc",
        }
        if cursor is not None:
            params["after"] = cursor

        r = safe_get(BASE_COMMENTS, params)
        data = r.json().get("data", [])
        log(f"comments fetched = {len(data)} for post_id={post_id}")

        if not data:
            break

        new_rows = []
        for comment in data:
            cid = comment.get("id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            new_rows.append(normalize_comment(comment))

        if new_rows:
            append_comments(new_rows)
            log(f"新增 comments {len(new_rows)} 筆 for post_id={post_id}")

        last_created = data[-1].get("created_utc")
        if not last_created:
            break

        cursor = int(last_created) + 1
        sleep_random(SLEEP_MIN, SLEEP_MAX)


def main():
    ensure_dirs()
    ensure_csv(POSTS_CSV, POST_FIELDS)
    ensure_csv(COMMENTS_CSV, COMMENT_FIELDS)

    state = load_checkpoint()

    log("=== Arctic Shift Q1 collector 啟動 ===")
    log(f"SUBREDDIT={SUBREDDIT}")
    log(f"從 {state['current_day']} 接續，phase={state['phase']}")

    while True:
        current_day = parse_date(state["current_day"])

        if current_day >= END_DATE:
            log("=== 全部完成，停止 ===")
            break

        day_str = current_day.strftime("%Y-%m-%d")

        if state["phase"] == "posts":
            log(f"處理日期 {day_str}")
            posts = fetch_posts_for_day(current_day)
            post_ids = [p.get("id") for p in posts if p.get("id")]

            state["pending_post_ids"] = post_ids
            state["comment_post_index"] = 0
            state["phase"] = "comments"
            save_checkpoint(state)

            if not post_ids:
                log(f"完成日期 {day_str}")
                next_day = current_day + timedelta(days=1)
                state["current_day"] = next_day.strftime("%Y-%m-%d")
                state["phase"] = "posts"
                save_checkpoint(state)
                sleep_random(DAY_SLEEP_MIN, DAY_SLEEP_MAX)
                continue

        if state["phase"] == "comments":
            post_ids = state.get("pending_post_ids", [])
            idx = state.get("comment_post_index", 0)

            while idx < len(post_ids):
                post_id = post_ids[idx]
                log(f"抓 comments: post {idx+1}/{len(post_ids)} post_id={post_id}")
                fetch_comments_for_post(post_id)
                idx += 1
                state["comment_post_index"] = idx
                save_checkpoint(state)
                sleep_random(SLEEP_MIN, SLEEP_MAX)

            log(f"完成日期 {day_str}")
            next_day = current_day + timedelta(days=1)
            state["current_day"] = next_day.strftime("%Y-%m-%d")
            state["phase"] = "posts"
            state["pending_post_ids"] = []
            state["comment_post_index"] = 0
            save_checkpoint(state)
            sleep_random(DAY_SLEEP_MIN, DAY_SLEEP_MAX)


if __name__ == "__main__":
    main()
