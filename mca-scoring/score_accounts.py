#!/usr/bin/env python3
"""Compute account-level MCA review-priority scores.

The MCA score is an interpretable ranking aid, not a final account verdict.
Group evidence and seed expansion live in `coordination-expansion/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRAPH_DIR = Path("adjacency/output")
DEFAULT_FEATURES_PATH = Path("Archive/export_working_files/account_feature_matrix.csv")
DEFAULT_OUTPUT_DIR = Path("mca-scoring/output")

NONNEUTRAL_TAG_COUNT_COLUMNS = [
    "rhetoric_tag_authority_claim_count",
    "rhetoric_tag_bandwagon_count",
    "rhetoric_tag_call_to_action_count",
    "rhetoric_tag_emotional_amplification_count",
    "rhetoric_tag_fear_count",
    "rhetoric_tag_overconfidence_count",
    "rhetoric_tag_urgency_count",
    "rhetoric_tag_us_vs_them_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MCA account scores from existing artifacts.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--primary-weights",
        nargs=4,
        type=float,
        default=[0.30, 0.35, 0.15, 0.20],
        metavar=("M", "C", "R", "A"),
        help="Signal weights for manipulative, coordinative, reach, automation.",
    )
    parser.add_argument(
        "--alt-weights",
        nargs=4,
        type=float,
        default=[0.40, 0.40, 0.10, 0.10],
        metavar=("M", "C", "R", "A"),
        help="Comparison signal weights.",
    )
    return parser.parse_args()


def normalize_user_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def positive_percentile(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    output = pd.Series(0.0, index=values.index)
    mask = values > 0
    if mask.any():
        output.loc[mask] = values.loc[mask].rank(method="average", pct=True)
    return output


def read_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, low_memory=False)
    features["user_id"] = normalize_user_id(features["user_id"])
    for col in [
        "avg_manipulative_rhetoric_score",
        "comment_label_oppositional_ratio",
        "anomaly_score",
        "anomaly_label",
        "analyzed_post_count",
        *NONNEUTRAL_TAG_COUNT_COLUMNS,
    ]:
        if col not in features.columns:
            features[col] = 0.0
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    return features


def undirected_strength(path: Path, weight_col: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["source_author", "target_author", weight_col], low_memory=False)
    left = df[["source_author", weight_col]].rename(
        columns={"source_author": "user_id", weight_col: "weight"}
    )
    right = df[["target_author", weight_col]].rename(
        columns={"target_author": "user_id", weight_col: "weight"}
    )
    both = pd.concat([left, right], ignore_index=True)
    both["user_id"] = normalize_user_id(both["user_id"])
    both["weight"] = pd.to_numeric(both["weight"], errors="coerce").fillna(0.0)
    return both.groupby("user_id")["weight"].sum()


def build_scores(graph_dir: Path, features_path: Path) -> pd.DataFrame:
    features = read_features(features_path)
    scores = pd.DataFrame({"user_id": features["user_id"]})

    nonneutral_count = features[NONNEUTRAL_TAG_COUNT_COLUMNS].sum(axis=1)
    nonneutral_ratio = (
        nonneutral_count / features["analyzed_post_count"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)

    scores["avg_rhetorical_score_norm"] = positive_percentile(
        features["avg_manipulative_rhetoric_score"]
    )
    scores["non_neutral_post_ratio_norm"] = positive_percentile(nonneutral_ratio)
    scores["oppositional_stance_norm"] = positive_percentile(
        features["comment_label_oppositional_ratio"]
    )
    scores["manipulative_signal"] = (
        0.40 * scores["avg_rhetorical_score_norm"]
        + 0.35 * scores["non_neutral_post_ratio_norm"]
        + 0.25 * scores["oppositional_stance_norm"]
    )

    edge_stats = pd.read_csv(
        graph_dir / "all_interaction_edge_stats.csv",
        usecols=["source_author", "target_author", "weight_count"],
        low_memory=False,
    )
    for col in ("source_author", "target_author"):
        edge_stats[col] = normalize_user_id(edge_stats[col])
    edge_stats["weight_count"] = pd.to_numeric(
        edge_stats["weight_count"], errors="coerce"
    ).fillna(0.0)

    outgoing = edge_stats.groupby("source_author")["weight_count"].sum()
    incoming = edge_stats.groupby("target_author")["weight_count"].sum()
    unique_targets = edge_stats.groupby("source_author")["target_author"].nunique()
    unique_commenters = edge_stats.groupby("target_author")["source_author"].nunique()

    scores["outgoing_volume_raw"] = scores["user_id"].map(outgoing).fillna(0.0)
    scores["incoming_attention_raw"] = scores["user_id"].map(incoming).fillna(0.0)
    scores["interaction_breadth_raw"] = (
        scores["user_id"].map(unique_targets).fillna(0.0)
        + scores["user_id"].map(unique_commenters).fillna(0.0)
    )
    scores["outgoing_volume_norm"] = positive_percentile(np.log1p(scores["outgoing_volume_raw"]))
    scores["incoming_attention_norm"] = positive_percentile(
        np.log1p(scores["incoming_attention_raw"])
    )
    scores["interaction_breadth_norm"] = positive_percentile(
        np.log1p(scores["interaction_breadth_raw"])
    )
    scores["interaction_reach_signal"] = (
        0.35 * scores["outgoing_volume_norm"]
        + 0.25 * scores["incoming_attention_norm"]
        + 0.40 * scores["interaction_breadth_norm"]
    )

    multi = graph_dir / "multi-graph"
    co_target = undirected_strength(multi / "edges_co_target.csv", "weight_co_target")
    co_negative = undirected_strength(
        multi / "edges_co_negative_target.csv", "weight_co_negative_target"
    )
    trigger = pd.read_csv(
        multi / "edges_trigger_response.csv",
        usecols=["source_author", "target_author", "weight_trigger_response"],
        low_memory=False,
    )
    for col in ("source_author", "target_author"):
        trigger[col] = normalize_user_id(trigger[col])
    trigger["weight_trigger_response"] = pd.to_numeric(
        trigger["weight_trigger_response"], errors="coerce"
    ).fillna(0.0)
    trigger_out = trigger.groupby("source_author")["weight_trigger_response"].sum()
    trigger_in = trigger.groupby("target_author")["weight_trigger_response"].sum()

    scores["co_target_raw"] = scores["user_id"].map(co_target).fillna(0.0)
    scores["co_negative_target_raw"] = scores["user_id"].map(co_negative).fillna(0.0)
    scores["trigger_out_raw"] = scores["user_id"].map(trigger_out).fillna(0.0)
    scores["trigger_in_raw"] = scores["user_id"].map(trigger_in).fillna(0.0)
    scores["co_target_norm"] = positive_percentile(np.log1p(scores["co_target_raw"]))
    scores["co_negative_target_norm"] = positive_percentile(
        np.log1p(scores["co_negative_target_raw"])
    )
    scores["trigger_out_norm"] = positive_percentile(np.log1p(scores["trigger_out_raw"]))
    scores["trigger_in_norm"] = positive_percentile(np.log1p(scores["trigger_in_raw"]))
    scores["trigger_frequency_norm"] = (
        0.5 * scores["trigger_out_norm"] + 0.5 * scores["trigger_in_norm"]
    )
    scores["coordinative_signal"] = (
        0.30 * scores["co_target_norm"]
        + 0.35 * scores["co_negative_target_norm"]
        + 0.35 * scores["trigger_frequency_norm"]
    )

    anomaly = -features["anomaly_score"]
    anomaly = anomaly.where(features["anomaly_score"] < 0, 0.0)
    scores["automatic_behavior_signal"] = positive_percentile(anomaly)
    scores["anomaly_score_raw"] = features["anomaly_score"]
    scores["anomaly_label"] = features["anomaly_label"]

    return scores


def apply_weights(scores: pd.DataFrame, weights: list[float], column: str) -> None:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    wm, wc, wr, wa = [weight / total for weight in weights]
    scores[column] = (
        wm * scores["manipulative_signal"]
        + wc * scores["coordinative_signal"]
        + wr * scores["interaction_reach_signal"]
        + wa * scores["automatic_behavior_signal"]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = build_scores(args.graph_dir, args.features_path)
    apply_weights(scores, args.primary_weights, "mca_score_primary")
    apply_weights(scores, args.alt_weights, "mca_score_alt")

    scores.to_csv(args.output_dir / "account_mca_scores.csv", index=False)
    scores.sort_values("mca_score_primary", ascending=False).head(args.top_n).to_csv(
        args.output_dir / "top_accounts_primary.csv", index=False
    )
    scores.sort_values("mca_score_alt", ascending=False).head(args.top_n).to_csv(
        args.output_dir / "top_accounts_alt.csv", index=False
    )

    print(f"MCA scores written to {args.output_dir}")
    print(f"Accounts: {len(scores):,}")


if __name__ == "__main__":
    main()
