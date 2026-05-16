#!/usr/bin/env python3
"""Build account-level feature matrices from analyzed posts and comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POSTS_PATH = Path("llm/Export/reddit_posts_analyzed.csv.gz")
DEFAULT_COMMENTS_PATH = Path("llm/Export/reddit_comments_analyzed.csv.gz")
DEFAULT_OUTPUT_DIR = Path("Archive/export_working_files")

INVALID_AUTHORS = {"", "nan", "none", "[deleted]", "[removed]", "deleted", "removed"}
COMMENT_LABELS = ["mixed", "neutral", "oppositional", "supportive", "unclear"]
RHETORIC_TAGS = [
    "analytical_neutral",
    "authority_claim",
    "bandwagon",
    "call_to_action",
    "emotional_amplification",
    "fear",
    "overconfidence",
    "urgency",
    "us_vs_them",
]
BEHAVIOR_MODEL_COLUMNS = [
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
    "comment_post_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build account feature matrix from formal LLM exports.")
    parser.add_argument("--posts-path", type=Path, default=DEFAULT_POSTS_PATH)
    parser.add_argument("--comments-path", type=Path, default=DEFAULT_COMMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cluster-count", type=int, default=8)
    parser.add_argument("--outlier-contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def normalize_author(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def valid_author_mask(series: pd.Series) -> pd.Series:
    normalized = normalize_author(series)
    return normalized.notna() & ~normalized.str.lower().isin(INVALID_AUTHORS)


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        parsed = value
    elif pd.isna(value) or str(value).strip() == "":
        parsed = []
    else:
        try:
            parsed = json.loads(str(value))
        except Exception:
            parsed = []
    if not isinstance(parsed, list):
        return []
    return [str(tag).strip() for tag in parsed if str(tag).strip() in RHETORIC_TAGS]


def std(series: pd.Series) -> float:
    return float(series.std(ddof=0))


def build_comment_behavior(comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame(columns=["user_id"])

    df = comments.copy()
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df.loc[df["created_datetime"].notna()].copy()
    df = df.sort_values(["source_author", "created_datetime"], kind="stable")
    df["hour"] = df["created_datetime"].dt.hour
    df["weekday"] = df["created_datetime"].dt.weekday
    df["hour_bucket"] = df["created_datetime"].dt.floor("h")

    user_stats = (
        df.groupby("source_author", as_index=False)
        .agg(
            comment_count=("comment_id", "count"),
            comment_start_time=("created_datetime", "min"),
            comment_end_time=("created_datetime", "max"),
            unique_posts_commented=("post_id", "nunique"),
        )
        .rename(columns={"source_author": "user_id"})
    )
    user_stats["comment_active_days"] = (
        (user_stats["comment_end_time"] - user_stats["comment_start_time"]).dt.days + 1
    ).clip(lower=1)
    user_stats["comments_per_day"] = user_stats["comment_count"] / user_stats["comment_active_days"]

    df["comment_time_diff_seconds"] = df.groupby("source_author")["created_datetime"].diff().dt.total_seconds()
    interval = (
        df.groupby("source_author")["comment_time_diff_seconds"]
        .agg(
            mean_comment_interval_seconds="mean",
            std_comment_interval_seconds=lambda s: s.std(ddof=0),
            min_comment_interval_seconds="min",
        )
        .reset_index()
        .rename(columns={"source_author": "user_id"})
    )
    hourly = df.groupby(["source_author", "hour_bucket"]).size().reset_index(name="comments_in_hour")
    burst = (
        hourly.groupby("source_author")["comments_in_hour"]
        .agg(max_comments_per_hour="max", mean_comments_per_hour="mean")
        .reset_index()
        .rename(columns={"source_author": "user_id"})
    )
    burst["burst_ratio"] = burst["max_comments_per_hour"] / (burst["mean_comments_per_hour"] + 1e-6)

    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(float)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(float)
    time_stats = (
        df.groupby("source_author", as_index=False)
        .agg(night_activity_ratio=("is_night", "mean"), weekend_activity_ratio=("is_weekend", "mean"))
        .rename(columns={"source_author": "user_id"})
    )

    return user_stats.merge(interval, on="user_id", how="left").merge(burst, on="user_id", how="left").merge(
        time_stats, on="user_id", how="left"
    )


def build_post_behavior(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=["user_id"])

    df = posts.copy()
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df.loc[df["created_datetime"].notna()].copy()
    df = df.sort_values(["author", "created_datetime"], kind="stable")

    post_stats = (
        df.groupby("author", as_index=False)
        .agg(
            post_count=("post_id", "count"),
            post_start_time=("created_datetime", "min"),
            post_end_time=("created_datetime", "max"),
            avg_post_num_comments=("num_comments", "mean"),
        )
        .rename(columns={"author": "user_id"})
    )
    post_stats["post_active_days"] = (
        (post_stats["post_end_time"] - post_stats["post_start_time"]).dt.days + 1
    ).clip(lower=1)
    post_stats["posts_per_day"] = post_stats["post_count"] / post_stats["post_active_days"]

    df["post_time_diff_seconds"] = df.groupby("author")["created_datetime"].diff().dt.total_seconds()
    interval = (
        df.groupby("author")["post_time_diff_seconds"]
        .agg(
            mean_post_interval_seconds="mean",
            std_post_interval_seconds=lambda s: s.std(ddof=0),
            min_post_interval_seconds="min",
        )
        .reset_index()
        .rename(columns={"author": "user_id"})
    )
    return post_stats.merge(interval, on="user_id", how="left")


def build_behavior_features(comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    comment_features = build_comment_behavior(comments)
    post_features = build_post_behavior(posts)
    features = comment_features.merge(post_features, on="user_id", how="outer")

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)
    features["total_activity_count"] = features["comment_count"] + features["post_count"]
    features["comment_post_ratio"] = features["comment_count"] / (features["post_count"] + 1e-6)
    features["active_days"] = features[["comment_active_days", "post_active_days"]].max(axis=1)
    return features


def add_anomaly_and_clusters(
    features: pd.DataFrame,
    *,
    cluster_count: int,
    contamination: float,
    random_state: int,
) -> pd.DataFrame:
    output = features.copy()
    for col in BEHAVIOR_MODEL_COLUMNS:
        if col not in output.columns:
            output[col] = 0.0
        output[col] = pd.to_numeric(output[col], errors="coerce").fillna(0.0)

    if len(output) < 2:
        output["anomaly_label"] = 1
        output["anomaly_score"] = 0.0
        output["is_extreme_outlier"] = False
        output["cluster_kmeans"] = 0
        return output

    matrix = output[BEHAVIOR_MODEL_COLUMNS].to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    stddev = matrix.std(axis=0, keepdims=True)
    stddev[stddev == 0] = 1.0
    scaled = (matrix - mean) / stddev

    distance = np.linalg.norm(scaled, axis=1)
    anomaly_cutoff = float(np.quantile(distance, 1.0 - contamination))
    output["anomaly_label"] = np.where(distance >= anomaly_cutoff, -1, 1)
    # Match sklearn IsolationForest direction: lower values are more anomalous.
    output["anomaly_score"] = anomaly_cutoff - distance
    outlier_cutoff = output["anomaly_score"].quantile(0.01)
    output["is_extreme_outlier"] = output["anomaly_score"] <= outlier_cutoff

    k = max(1, min(cluster_count, len(output)))
    output["cluster_kmeans"] = run_kmeans(scaled, k, random_state=random_state)
    return output


def run_kmeans(matrix: np.ndarray, k: int, *, random_state: int, max_iter: int = 100) -> np.ndarray:
    if k <= 1:
        return np.zeros(len(matrix), dtype=int)
    rng = np.random.default_rng(random_state)
    initial = rng.choice(len(matrix), size=k, replace=False)
    centroids = matrix[initial].copy()
    labels = np.zeros(len(matrix), dtype=int)

    for _ in range(max_iter):
        distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = matrix[labels == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = matrix[rng.integers(0, len(matrix))]
            else:
                centroids[cluster_id] = members.mean(axis=0)
    return labels


def build_comment_llm_features(comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame(columns=["user_id"])

    df = comments.copy()
    df["feedback_score"] = pd.to_numeric(df["feedback_score"], errors="coerce").fillna(0.0)
    df["edge_weight"] = pd.to_numeric(df["edge_weight"], errors="coerce").fillna(df["feedback_score"] / 100.0)
    df["analysis_char_len"] = pd.to_numeric(df["analysis_char_len"], errors="coerce").fillna(0.0)
    df["feedback_label"] = df["feedback_label"].astype("string").str.strip().str.lower().fillna("unclear")
    df.loc[~df["feedback_label"].isin(COMMENT_LABELS), "feedback_label"] = "unclear"

    stats = (
        df.groupby("source_author", as_index=False)
        .agg(
            comment_text_count=("comment_id", "count"),
            avg_comment_feedback_score=("feedback_score", "mean"),
            std_comment_feedback_score=("feedback_score", std),
            max_comment_feedback_score=("feedback_score", "max"),
            min_comment_feedback_score=("feedback_score", "min"),
            avg_comment_edge_weight=("edge_weight", "mean"),
            std_comment_edge_weight=("edge_weight", std),
            avg_comment_text_len=("analysis_char_len", "mean"),
            std_comment_text_len=("analysis_char_len", std),
            max_comment_text_len=("analysis_char_len", "max"),
            unique_target_authors=("target_author", "nunique"),
            unique_commented_posts=("post_id", "nunique"),
        )
        .rename(columns={"source_author": "user_id"})
    )
    labels = (
        df.pivot_table(index="source_author", columns="feedback_label", values="comment_id", aggfunc="count", fill_value=0)
        .reset_index()
        .rename(columns={"source_author": "user_id"})
        .rename_axis(None, axis=1)
    )
    for label in COMMENT_LABELS:
        if label not in labels.columns:
            labels[label] = 0
        labels = labels.rename(columns={label: f"comment_label_{label}_count"})
    count_cols = [f"comment_label_{label}_count" for label in COMMENT_LABELS]
    labels["comment_label_total"] = labels[count_cols].sum(axis=1)
    denominator = labels["comment_label_total"].replace(0, np.nan)
    for label in COMMENT_LABELS:
        labels[f"comment_label_{label}_ratio"] = (
            labels[f"comment_label_{label}_count"] / denominator
        ).fillna(0.0)
    return stats.merge(labels, on="user_id", how="outer")


def build_post_llm_features(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=["user_id"])

    df = posts.copy()
    numeric = [
        "sentiment_score",
        "manipulative_rhetoric_score",
        "analysis_char_len",
        "num_comments",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    stats = (
        df.groupby("author", as_index=False)
        .agg(
            analyzed_post_count=("post_id", "count"),
            avg_post_sentiment_score=("sentiment_score", "mean"),
            std_post_sentiment_score=("sentiment_score", std),
            max_post_sentiment_score=("sentiment_score", "max"),
            min_post_sentiment_score=("sentiment_score", "min"),
            avg_manipulative_rhetoric_score=("manipulative_rhetoric_score", "mean"),
            std_manipulative_rhetoric_score=("manipulative_rhetoric_score", std),
            max_manipulative_rhetoric_score=("manipulative_rhetoric_score", "max"),
            avg_post_text_len=("analysis_char_len", "mean"),
            std_post_text_len=("analysis_char_len", std),
            avg_post_num_comments_y=("num_comments", "mean"),
            max_post_num_comments=("num_comments", "max"),
        )
        .rename(columns={"author": "user_id"})
    )

    tag_rows = []
    for row in df[["author", "rhetoric_tags"]].itertuples(index=False):
        tags = parse_tags(row.rhetoric_tags)
        record = {"user_id": row.author}
        for tag in RHETORIC_TAGS:
            record[f"rhetoric_tag_{tag}_count"] = int(tag in tags)
        tag_rows.append(record)
    tag_df = pd.DataFrame(tag_rows)
    if tag_df.empty:
        tag_summary = pd.DataFrame(columns=["user_id"])
    else:
        count_cols = [f"rhetoric_tag_{tag}_count" for tag in RHETORIC_TAGS]
        tag_summary = tag_df.groupby("user_id", as_index=False)[count_cols].sum()
        tag_summary["rhetoric_tag_total"] = tag_summary[count_cols].sum(axis=1)
        denominator = tag_summary["rhetoric_tag_total"].replace(0, np.nan)
        for tag in RHETORIC_TAGS:
            tag_summary[f"rhetoric_tag_{tag}_ratio"] = (
                tag_summary[f"rhetoric_tag_{tag}_count"] / denominator
            ).fillna(0.0)
    return stats.merge(tag_summary, on="user_id", how="outer")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comments = pd.read_csv(args.comments_path, low_memory=False)
    posts = pd.read_csv(args.posts_path, low_memory=False)

    comments["source_author"] = normalize_author(comments["source_author"])
    comments["target_author"] = normalize_author(comments["target_author"])
    posts["author"] = normalize_author(posts["author"])
    comments = comments.loc[valid_author_mask(comments["source_author"])].copy()
    posts = posts.loc[valid_author_mask(posts["author"])].copy()

    behavior = build_behavior_features(comments, posts)
    behavior = add_anomaly_and_clusters(
        behavior,
        cluster_count=args.cluster_count,
        contamination=args.outlier_contamination,
        random_state=args.random_state,
    )
    behavior.to_csv(args.output_dir / "behavior_features.csv", index=False)

    comment_llm = build_comment_llm_features(comments)
    post_llm = build_post_llm_features(posts)
    feature_matrix = behavior.merge(comment_llm, on="user_id", how="outer").merge(
        post_llm, on="user_id", how="outer"
    )

    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0.0)
    feature_matrix = feature_matrix.sort_values("user_id").reset_index(drop=True)

    base_path = args.output_dir / "account_feature_matrix.csv"
    clustered_path = args.output_dir / "account_feature_matrix_with_clusters.csv"
    base_cols = [col for col in feature_matrix.columns if col not in {"is_extreme_outlier", "cluster_kmeans"}]
    feature_matrix[base_cols].to_csv(base_path, index=False)
    feature_matrix.to_csv(clustered_path, index=False)

    print("Account feature matrices written.")
    print(f"Accounts: {len(feature_matrix):,}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
