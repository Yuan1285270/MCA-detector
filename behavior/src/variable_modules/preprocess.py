import pandas as pd

from src.data_sources.config import RAW_POSTS_PATH, RAW_COMMENTS_PATH, CLEAN_POSTS_PATH, CLEAN_COMMENTS_PATH
from io_utils import ensure_dirs, read_csv, save_csv
from standardize import standardize_posts, standardize_comments


def main():
    ensure_dirs()

    print("讀取 raw 資料中...")
    posts_df = read_csv(RAW_POSTS_PATH)
    comments_df = read_csv(RAW_COMMENTS_PATH)

    print("Raw Posts:", posts_df.shape)
    print("Raw Comments:", comments_df.shape)

    print("\n標準化 Posts 中...")
    clean_posts = standardize_posts(posts_df)
    print("Clean Posts:", clean_posts.shape)

    print("\n標準化 Comments 中...")
    clean_comments = standardize_comments(comments_df)
    print("Clean Comments:", clean_comments.shape)

    save_csv(clean_posts, CLEAN_POSTS_PATH)
    save_csv(clean_comments, CLEAN_COMMENTS_PATH)

    print("\n=== Done ===")
    print(f"已輸出: {CLEAN_POSTS_PATH}")
    print(f"已輸出: {CLEAN_COMMENTS_PATH}")

    print("\n=== Cleaned Posts Info ===")
    print(clean_posts.head())

    print("\n=== Cleaned Comments Info ===")
    print(clean_comments.head())


if __name__ == "__main__":
    main()