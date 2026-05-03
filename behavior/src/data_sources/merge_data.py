import pandas as pd

from config import CLEAN_POSTS_PATH, CLEAN_COMMENTS_PATH, MERGED_REDDIT_PATH
from src.variable_modules.io_utils import ensure_dirs, read_csv, save_csv


def main():
    ensure_dirs()

    print("讀取清理後資料中...")
    posts_df = read_csv(CLEAN_POSTS_PATH)
    comments_df = read_csv(CLEAN_COMMENTS_PATH)

    print("\n=== 原始資料大小 ===")
    print("Posts:", posts_df.shape)
    print("Comments:", comments_df.shape)

    post_cols = [
        "post_id",
        "author",
        "created_utc",
        "created_datetime",
        "title",
        "selftext",
        "num_comments"
    ]

    posts_for_merge = posts_df[post_cols].copy()

    posts_for_merge = posts_for_merge.rename(columns={
        "author": "post_author",
        "created_utc": "post_created_utc",
        "created_datetime": "post_created_datetime",
        "title": "post_title",
        "selftext": "post_selftext",
        "num_comments": "post_num_comments"
    })

    merged_df = comments_df.merge(
        posts_for_merge,
        on="post_id",
        how="left"
    )

    print("\n=== Merge 後資料大小 ===")
    print(merged_df.shape)

    missing_post_count = merged_df["post_title"].isna().sum()
    matched_count = len(merged_df) - missing_post_count

    print("\n=== Merge 檢查 ===")
    print("成功對到 post 的 comments 數:", matched_count)
    print("沒對到 post 的 comments 數:", missing_post_count)

    merged_df = merged_df[merged_df["post_title"].notna()].copy()

    print("\n=== 過濾後 merged 資料大小 ===")
    print(merged_df.shape)

    merged_df = merged_df.sort_values(
        ["created_datetime", "comment_id"]
    ).reset_index(drop=True)

    save_csv(merged_df, MERGED_REDDIT_PATH)

    print("\n=== Done ===")
    print(f"已輸出: {MERGED_REDDIT_PATH}")

    print("\n=== Merged Data Info ===")
    print(merged_df.head())
    print("\n欄位名稱:")
    print(list(merged_df.columns))


if __name__ == "__main__":
    main()