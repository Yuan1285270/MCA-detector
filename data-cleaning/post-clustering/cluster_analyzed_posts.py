from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
POSTS_PATH = PROJECT_DIR / "Export" / "reddit_posts_analyzed.csv"
COMMENTS_PATH = PROJECT_DIR / "Export" / "reddit_comments_analyzed.csv.gz.csv"
OUTPUT_DIR = BASE_DIR / "output"
FEATURE_MATRIX_PATH = OUTPUT_DIR / "post_feature_matrix.csv"
CLUSTERED_PATH = OUTPUT_DIR / "post_clusters.csv"
SUMMARY_PATH = OUTPUT_DIR / "cluster_summary.csv"
SUSPICIOUS_PATH = OUTPUT_DIR / "suspicious_clusters.csv"
CONFIG_PATH = OUTPUT_DIR / "cluster_config.json"

N_CLUSTERS = 8
TEXT_HASH_DIM = 64
MAX_ITER = 40
RANDOM_SEED = 42
TOP_TERMS = 12
TOP_SUSPICIOUS_CLUSTERS = 3

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
TAG_NAMES = [
    "urgency",
    "fear",
    "overconfidence",
    "authority_claim",
    "bandwagon",
    "us_vs_them",
    "call_to_action",
    "emotional_amplification",
    "analytical_neutral",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def stable_hash(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(tag) for tag in value]
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(tag) for tag in parsed]


def text_ratio(text: pd.Series, pattern: str) -> pd.Series:
    return text.fillna("").astype(str).str.count(pattern) / text.fillna("").astype(str).str.len().clip(lower=1)


def zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def hashed_text_features(texts: pd.Series, dim: int) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts.fillna("")):
        counts: dict[int, int] = {}
        for token in tokenize(str(text)):
            index = stable_hash(token, dim)
            counts[index] = counts.get(index, 0) + 1
        if not counts:
            continue
        norm = math.sqrt(sum(value * value for value in counts.values()))
        for index, value in counts.items():
            matrix[row_idx, index] = value / norm
    return matrix


def run_kmeans(matrix: np.ndarray, n_clusters: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(matrix), size=n_clusters, replace=False)
    centroids = matrix[indices].copy()
    labels = np.zeros(len(matrix), dtype=np.int32)

    for _ in range(max_iter):
        distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for cluster_id in range(n_clusters):
            members = matrix[labels == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = matrix[rng.integers(0, len(matrix))]
            else:
                centroids[cluster_id] = members.mean(axis=0)

    return labels, centroids


def comment_feedback_features(comments: pd.DataFrame) -> pd.DataFrame:
    label_counts = (
        comments.pivot_table(
            index="post_id",
            columns="feedback_label",
            values="comment_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in ["supportive", "oppositional", "neutral", "mixed", "unclear"]:
        if label not in label_counts.columns:
            label_counts[label] = 0

    score = pd.to_numeric(comments["feedback_score"], errors="coerce")
    score_features = (
        comments.assign(feedback_score_numeric=score)
        .groupby("post_id", as_index=False)
        .agg(
            analyzed_comment_count=("comment_id", "count"),
            avg_feedback_score=("feedback_score_numeric", "mean"),
            min_feedback_score=("feedback_score_numeric", "min"),
            max_feedback_score=("feedback_score_numeric", "max"),
        )
    )
    features = label_counts.merge(score_features, on="post_id", how="outer")
    total = features["analyzed_comment_count"].replace(0, np.nan)
    for label in ["supportive", "oppositional", "neutral", "mixed", "unclear"]:
        features[f"{label}_comment_ratio"] = (features[label] / total).fillna(0.0)
    features = features.rename(
        columns={
            "supportive": "supportive_comment_count",
            "oppositional": "oppositional_comment_count",
            "neutral": "neutral_comment_count",
            "mixed": "mixed_comment_count",
            "unclear": "unclear_comment_count",
        }
    )
    return features


def build_feature_frame(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    feedback = comment_feedback_features(comments)
    df = df.merge(feedback, on="post_id", how="left")

    feedback_columns = [column for column in df.columns if column.endswith("_comment_count")]
    feedback_columns += [
        "analyzed_comment_count",
        "avg_feedback_score",
        "min_feedback_score",
        "max_feedback_score",
        "supportive_comment_ratio",
        "oppositional_comment_ratio",
        "neutral_comment_ratio",
        "mixed_comment_ratio",
        "unclear_comment_ratio",
    ]
    for column in feedback_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    text = df["analysis_text"].fillna("").astype(str)
    df["word_count"] = text.map(lambda value: len(tokenize(value))).astype(float)
    df["uppercase_ratio"] = text_ratio(text, r"[A-Z]")
    df["digit_ratio"] = text_ratio(text, r"\d")
    df["exclamation_ratio"] = text_ratio(text, r"!")
    df["question_ratio"] = text_ratio(text, r"\?")
    df["url_ratio"] = text_ratio(text, r"https?://|www\.")
    df["dollar_ratio"] = text_ratio(text, r"\$")

    parsed_tags = df["rhetoric_tags"].map(parse_tags)
    for tag in TAG_NAMES:
        df[f"tag_{tag}"] = parsed_tags.map(lambda tags, tag=tag: float(tag in tags))

    df["high_manipulation_flag"] = (
        pd.to_numeric(df["manipulative_rhetoric_score"], errors="coerce").fillna(0) >= 60
    ).astype(float)
    df["oppositional_pressure"] = df["oppositional_comment_ratio"] * df["analyzed_comment_count"].clip(lower=1).map(np.log1p)
    df["supportive_pressure"] = df["supportive_comment_ratio"] * df["analyzed_comment_count"].clip(lower=1).map(np.log1p)
    df["controversy_score"] = (
        df["oppositional_comment_ratio"] * 0.65
        + df["mixed_comment_ratio"] * 0.25
        + (df["avg_feedback_score"].abs().rsub(100) / 100) * 0.10
    )

    return df


def numeric_feature_columns() -> list[str]:
    return [
        "sentiment_score",
        "manipulative_rhetoric_score",
        "high_manipulation_flag",
        "analysis_char_len",
        "word_count",
        "num_comments",
        "uppercase_ratio",
        "digit_ratio",
        "exclamation_ratio",
        "question_ratio",
        "url_ratio",
        "dollar_ratio",
        "analyzed_comment_count",
        "avg_feedback_score",
        "min_feedback_score",
        "max_feedback_score",
        "supportive_comment_ratio",
        "oppositional_comment_ratio",
        "neutral_comment_ratio",
        "mixed_comment_ratio",
        "unclear_comment_ratio",
        "oppositional_pressure",
        "supportive_pressure",
        "controversy_score",
    ] + [f"tag_{tag}" for tag in TAG_NAMES]


def top_terms_for_cluster(texts: pd.Series, labels: np.ndarray, cluster_id: int, top_n: int) -> str:
    counts: dict[str, int] = {}
    stop_terms = {"the", "and", "for", "you", "that", "this", "with", "are", "but", "have"}
    for text in texts[labels == cluster_id]:
        for token in tokenize(str(text)):
            if len(token) < 3 or token in stop_terms:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(term for term, _ in ranked[:top_n])


def build_cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for cluster_id in sorted(set(labels.tolist())):
        cluster_df = df[df["cluster"] == cluster_id]
        rows.append(
            {
                "cluster": cluster_id,
                "size": int(len(cluster_df)),
                "share": round(len(cluster_df) / len(df), 4),
                "avg_sentiment_score": round(cluster_df["sentiment_score"].mean(), 2),
                "avg_manipulative_rhetoric_score": round(cluster_df["manipulative_rhetoric_score"].mean(), 2),
                "high_manipulation_ratio": round(cluster_df["high_manipulation_flag"].mean(), 4),
                "avg_num_comments": round(cluster_df["num_comments"].mean(), 2),
                "avg_analyzed_comment_count": round(cluster_df["analyzed_comment_count"].mean(), 2),
                "avg_feedback_score": round(cluster_df["avg_feedback_score"].mean(), 2),
                "avg_supportive_comment_ratio": round(cluster_df["supportive_comment_ratio"].mean(), 4),
                "avg_oppositional_comment_ratio": round(cluster_df["oppositional_comment_ratio"].mean(), 4),
                "avg_mixed_comment_ratio": round(cluster_df["mixed_comment_ratio"].mean(), 4),
                "avg_controversy_score": round(cluster_df["controversy_score"].mean(), 4),
                "avg_tag_call_to_action": round(cluster_df["tag_call_to_action"].mean(), 4),
                "avg_tag_urgency": round(cluster_df["tag_urgency"].mean(), 4),
                "avg_tag_overconfidence": round(cluster_df["tag_overconfidence"].mean(), 4),
                "avg_tag_fear": round(cluster_df["tag_fear"].mean(), 4),
                "top_terms": top_terms_for_cluster(df["analysis_text"], labels, cluster_id, TOP_TERMS),
                "sample_titles": " || ".join(cluster_df["title"].fillna("").astype(str).head(3).tolist()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["suspicion_score"] = (
        summary["avg_manipulative_rhetoric_score"] * 0.55
        + summary["high_manipulation_ratio"] * 25
        + summary["avg_tag_call_to_action"] * 12
        + summary["avg_tag_urgency"] * 10
        + summary["avg_tag_overconfidence"] * 9
        + summary["avg_tag_fear"] * 8
        + summary["avg_oppositional_comment_ratio"] * 10
        + summary["avg_controversy_score"] * 8
    ).round(3)
    return summary.sort_values(["suspicion_score", "size"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posts = pd.read_csv(POSTS_PATH, low_memory=False)
    comments = pd.read_csv(COMMENTS_PATH, low_memory=False)

    features = build_feature_frame(posts, comments)
    numeric_columns = numeric_feature_columns()
    numeric = features[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_matrix = zscore(numeric.to_numpy(dtype=np.float32))
    text_matrix = hashed_text_features(features["analysis_text"], TEXT_HASH_DIM)
    matrix = np.concatenate([numeric_matrix, text_matrix], axis=1)

    labels, _ = run_kmeans(matrix, N_CLUSTERS, MAX_ITER, RANDOM_SEED)
    clustered = features.copy()
    clustered["cluster"] = labels

    summary = build_cluster_summary(clustered, labels)
    suspicious = summary.head(TOP_SUSPICIOUS_CLUSTERS).copy()

    rank_map = {row.cluster: idx + 1 for idx, row in summary.iterrows()}
    score_map = dict(zip(summary["cluster"], summary["suspicion_score"]))
    clustered["cluster_rank_by_suspicion"] = clustered["cluster"].map(rank_map)
    clustered["cluster_suspicion_score"] = clustered["cluster"].map(score_map)

    feature_output_columns = ["post_id", "author", "title"] + numeric_columns
    features[feature_output_columns].to_csv(FEATURE_MATRIX_PATH, index=False, encoding="utf-8-sig")
    clustered.to_csv(CLUSTERED_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    suspicious.to_csv(SUSPICIOUS_PATH, index=False, encoding="utf-8-sig")

    config = {
        "posts_path": str(POSTS_PATH),
        "comments_path": str(COMMENTS_PATH),
        "n_clusters": N_CLUSTERS,
        "text_hash_dim": TEXT_HASH_DIM,
        "max_iter": MAX_ITER,
        "random_seed": RANDOM_SEED,
        "numeric_feature_columns": numeric_columns,
        "top_suspicious_clusters_saved": TOP_SUSPICIOUS_CLUSTERS,
        "notes": "Formal clustering based on final Gemini post scores and comment feedback aggregates.",
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        summary[
            [
                "cluster",
                "size",
                "suspicion_score",
                "avg_manipulative_rhetoric_score",
                "avg_oppositional_comment_ratio",
                "avg_feedback_score",
                "top_terms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
