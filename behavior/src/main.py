import pandas as pd

posts_path = "data/raw/reddit_posts_2025.csv"
comments_path = "data/raw/reddit_comments_2025.csv"

posts_df = pd.read_csv(posts_path)
comments_df = pd.read_csv(comments_path)

print("=== Posts Data ===")
print(posts_df.shape)
print(posts_df.columns)
print(posts_df.head())

print("\n=== Comments Data ===")
print(comments_df.shape)
print(comments_df.columns)
print(comments_df.head())