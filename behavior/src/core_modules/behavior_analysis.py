'''
帳號行為分析
1.基本活躍特徵
2.時間間隔特徵
3.爆發行為特徵
4.時段特徵
5.異常偵測
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# =========================
# 路徑設定
# =========================
from src.data_sources.config import (
    CLEAN_POSTS_PATH,
    MERGED_REDDIT_PATH,
    BEHAVIOR_FEATURES_PATH,
    SUSPICIOUS_BEHAVIOR_PATH,
    FIGURE_DIR,
)
from src.variable_modules.io_utils import ensure_dirs, read_csv, save_csv

def load_posts():
    df = read_csv(CLEAN_POSTS_PATH)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df[df["created_datetime"].notna()].copy()

    df["author"] = df["author"].astype(str).str.strip()
    invalid_authors = ["[deleted]", "[removed]", "nan", "None", ""]
    df = df[~df["author"].isin(invalid_authors)].copy()

    return df


def load_comments():
    df = read_csv(MERGED_REDDIT_PATH)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    df = df[df["created_datetime"].notna()].copy()

    df["author"] = df["author"].astype(str).str.strip()
    invalid_authors = ["[deleted]", "[removed]", "nan", "None", ""]
    df = df[~df["author"].isin(invalid_authors)].copy()

    return df


def build_comment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["author", "created_datetime"]).reset_index(drop=True)

    df["date"] = df["created_datetime"].dt.date
    df["hour"] = df["created_datetime"].dt.hour
    df["weekday"] = df["created_datetime"].dt.weekday

    # 基本留言活躍特徵
    user_stats = df.groupby("author").agg(
        comment_count=("comment_id", "count"),
        comment_start_time=("created_datetime", "min"),
        comment_end_time=("created_datetime", "max"),
        unique_posts_commented=("post_id", "nunique")
    ).reset_index()

    user_stats["comment_active_days"] = (
        (user_stats["comment_end_time"] - user_stats["comment_start_time"]).dt.days + 1
    ).clip(lower=1)

    user_stats["comments_per_day"] = (
        user_stats["comment_count"] / user_stats["comment_active_days"]
    )

    # 留言間隔特徵
    df["comment_time_diff_seconds"] = (
        df.groupby("author")["created_datetime"].diff().dt.total_seconds()
    )

    interval_stats = df.groupby("author")["comment_time_diff_seconds"].agg(
        mean_comment_interval_seconds="mean",
        std_comment_interval_seconds="std",
        min_comment_interval_seconds="min"
    ).reset_index()

    # 每小時爆發行為
    df["hour_bucket"] = df["created_datetime"].dt.floor("h")

    hourly_counts = (
        df.groupby(["author", "hour_bucket"])
        .size()
        .reset_index(name="comments_in_hour")
    )

    burst_stats = hourly_counts.groupby("author")["comments_in_hour"].agg(
        max_comments_per_hour="max",
        mean_comments_per_hour="mean"
    ).reset_index()

    burst_stats["burst_ratio"] = (
        burst_stats["max_comments_per_hour"] /
        (burst_stats["mean_comments_per_hour"] + 1e-6)
    )

    # 時段特徵
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    time_stats = df.groupby("author").agg(
        night_activity_ratio=("is_night", "mean"),
        weekend_activity_ratio=("is_weekend", "mean")
    ).reset_index()

    features = user_stats.merge(interval_stats, on="author", how="left")
    features = features.merge(burst_stats, on="author", how="left")
    features = features.merge(time_stats, on="author", how="left")

    return features


def build_post_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["author", "created_datetime"]).reset_index(drop=True)

    df["date"] = df["created_datetime"].dt.date

    post_stats = df.groupby("author").agg(
        post_count=("post_id", "count"),
        post_start_time=("created_datetime", "min"),
        post_end_time=("created_datetime", "max"),
        avg_post_num_comments=("num_comments", "mean")
    ).reset_index()

    post_stats["post_active_days"] = (
        (post_stats["post_end_time"] - post_stats["post_start_time"]).dt.days + 1
    ).clip(lower=1)

    post_stats["posts_per_day"] = (
        post_stats["post_count"] / post_stats["post_active_days"]
    )

    df["post_time_diff_seconds"] = (
        df.groupby("author")["created_datetime"].diff().dt.total_seconds()
    )

    post_interval_stats = df.groupby("author")["post_time_diff_seconds"].agg(
        mean_post_interval_seconds="mean",
        std_post_interval_seconds="std",
        min_post_interval_seconds="min"
    ).reset_index()

    post_features = post_stats.merge(post_interval_stats, on="author", how="left")
    return post_features


def merge_behavior_features(comment_features: pd.DataFrame, post_features: pd.DataFrame) -> pd.DataFrame:
    features = comment_features.merge(post_features, on="author", how="outer")

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    # 整合特徵
    features["total_activity_count"] = (
        features["comment_count"] + features["post_count"]
    )

    features["comment_post_ratio"] = (
        features["comment_count"] / (features["post_count"] + 1e-6)
    )

    # 保守一點，也做一個總活躍天數
    features["active_days"] = features[["comment_active_days", "post_active_days"]].max(axis=1)

    features = features.rename(columns={"author": "user_id"})
    return features


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    model_features = [
        "comment_count",
        "post_count",
        "comments_per_day",
        "posts_per_day",
        "unique_posts_commented",
        "avg_post_num_comments",
        "mean_comment_interval_seconds",
        "std_comment_interval_seconds",
        "mean_post_interval_seconds",
        "std_post_interval_seconds",
        "burst_ratio",
        "night_activity_ratio",
        "weekend_activity_ratio",
        "comment_post_ratio"
    ]

    X = features[model_features].copy()

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    features["anomaly_label"] = model.fit_predict(X)
    features["anomaly_score"] = model.decision_function(X)

    return features


def save_outputs(features: pd.DataFrame):
    save_csv(features, BEHAVIOR_FEATURES_PATH)

    suspicious = features[features["anomaly_label"] == -1].copy()
    suspicious = suspicious.sort_values("anomaly_score")
    save_csv(suspicious, SUSPICIOUS_BEHAVIOR_PATH)

    print("\n已輸出：")
    print(BEHAVIOR_FEATURES_PATH)
    print(SUSPICIOUS_BEHAVIOR_PATH)

    print("\n可疑帳號前 10 筆：")
    print(suspicious[[
        "user_id",
        "comment_count",
        "post_count",
        "total_activity_count",
        "comments_per_day",
        "posts_per_day",
        "comment_post_ratio",
        "burst_ratio",
        "avg_post_num_comments",
        "anomaly_score"
    ]].head(10))

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

def plot_figures(features: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(features["comment_count"], bins=50)
    plt.title("每個帳號的留言總數分布")
    plt.xlabel("留言總數")
    plt.ylabel("使用者人數")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "comment_count_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(features["post_count"], bins=50)
    plt.title("每個帳號的貼文總數分布")
    plt.xlabel("貼文總數")
    plt.ylabel("使用者人數")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "post_count_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(features["comment_post_ratio"], bins=50)
    plt.title("留言/貼文比例分布")
    plt.xlabel("留言/貼文比例")
    plt.ylabel("使用者人數")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "comment_post_ratio_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(features["anomaly_score"], bins=50)
    plt.title("所有帳號的異常分數分布")
    plt.xlabel("異常分數")
    plt.ylabel("使用者人數")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "anomaly_score_distribution.png"))
    plt.close()

    print("\n圖表已輸出到 output/figures/")


def main():
    ensure_dirs()

    print("讀取 posts / comments 資料中...")
    posts_df = load_posts()
    comments_df = load_comments()

    print("Posts 資料大小:", posts_df.shape)
    print("Comments 資料大小:", comments_df.shape)

    print("\n建立 comment 特徵中...")
    comment_features = build_comment_features(comments_df)
    print("Comment 特徵表大小:", comment_features.shape)

    print("\n建立 post 特徵中...")
    post_features = build_post_features(posts_df)
    print("Post 特徵表大小:", post_features.shape)

    print("\n合併行為特徵中...")
    features = merge_behavior_features(comment_features, post_features)
    print("合併後特徵表大小:", features.shape)
    print(features.head())

    print("\n進行異常偵測中...")
    features = detect_anomalies(features)

    suspicious_count = (features["anomaly_label"] == -1).sum()
    print("可疑帳號數量:", suspicious_count)

    save_outputs(features)
    plot_figures(features)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()