#!/usr/bin/env python3
"""Build a validation table for seed expansion candidates.

This joins Stage 1 expansion membership with MCA scores, full-population
behavior clusters, and Stage 2 temporal synchrony evidence. It is a review
table, not a final bot verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED_DIR = Path("coordination-expansion/output/seeds")
DEFAULT_MCA_PATH = Path("mca-scoring/output/account_mca_scores.csv")
DEFAULT_CLUSTER_PATH = Path("Archive/export_working_files/account_feature_matrix_with_clusters.csv")
DEFAULT_STAGE2_PATH = Path(
    "coordination-expansion/output/stage2-verification/stage2_verification_evidence.csv"
)
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/candidate-validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate validation tables.")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--mca-path", type=Path, default=DEFAULT_MCA_PATH)
    parser.add_argument("--cluster-path", type=Path, default=DEFAULT_CLUSTER_PATH)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-mca-threshold", type=float, default=0.50)
    return parser.parse_args()


def normalize_user_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_expansion_members(seed_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(seed_dir / "seed_expansion_summary.csv", usecols=["seed"])
    rows: list[dict[str, object]] = []
    for seed in summary["seed"].astype(str):
        path = seed_dir / seed / "tiered_expansion_members.csv"
        if not path.exists():
            continue
        members = pd.read_csv(path)
        members = members.loc[members["include"].eq(True)].copy()
        for row in members.itertuples(index=False):
            rows.append(
                {
                    "seed_group": seed,
                    "account": str(row.candidate),
                    "tier": int(row.tier),
                    "include_reason": row.include_reason,
                    "co_negative_weight_to_seed": float(row.co_negative_weight),
                    "tag_similarity_weight_to_seed": float(row.tag_similarity_weight),
                    "trigger_response_weight_to_seed": float(row.trigger_response_weight),
                    "co_target_weight_to_seed": float(row.co_target_weight),
                    "two_hop_link_count": int(row.two_hop_link_count),
                    "two_hop_connectors": row.two_hop_connectors,
                }
            )
    return pd.DataFrame(rows)


def load_mca(path: Path) -> pd.DataFrame:
    cols = [
        "user_id",
        "mca_score_primary",
        "manipulative_signal",
        "coordinative_signal",
        "interaction_reach_signal",
        "automatic_behavior_signal",
    ]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["user_id"] = normalize_user_id(df["user_id"])
    return df


def load_clusters(path: Path) -> pd.DataFrame:
    cols = [
        "user_id",
        "anomaly_label",
        "anomaly_score",
        "is_extreme_outlier",
        "cluster_kmeans",
        "comment_count",
        "comment_active_days",
        "comments_per_day",
        "post_count",
        "post_active_days",
        "posts_per_day",
        "active_days",
        "burst_ratio",
        "night_activity_ratio",
        "weekend_activity_ratio",
        "comment_label_oppositional_ratio",
        "comment_label_supportive_ratio",
        "avg_manipulative_rhetoric_score",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in cols, low_memory=False)
    df["user_id"] = normalize_user_id(df["user_id"])
    return df.rename(columns={"cluster_kmeans": "full_cluster_kmeans"})


def load_temporal_account_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["seed_group", "account"])
    pairs = pd.read_csv(path)
    if pairs.empty:
        return pd.DataFrame(columns=["seed_group", "account"])

    rows = []
    for side in ["account_a", "account_b"]:
        other = "account_b" if side == "account_a" else "account_a"
        temp = pairs[
            [
                "group_seed",
                side,
                other,
                "same_post_count",
                "within_5min_count",
                "within_30min_count",
                "median_delay_minutes",
                "verification_label",
            ]
        ].copy()
        temp = temp.rename(
            columns={
                "group_seed": "seed_group",
                side: "account",
                other: "temporal_partner",
            }
        )
        rows.append(temp)
    account_pairs = pd.concat(rows, ignore_index=True)

    label_rank = {
        "no_temporal_sync": 0,
        "weak_temporal_overlap": 1,
        "moderate_temporal_sync": 2,
        "strong_temporal_sync": 3,
    }
    account_pairs["temporal_rank"] = account_pairs["verification_label"].map(label_rank).fillna(0)
    account_pairs["median_delay_minutes"] = pd.to_numeric(
        account_pairs["median_delay_minutes"], errors="coerce"
    )

    grouped = (
        account_pairs.groupby(["seed_group", "account"])
        .agg(
            temporal_pair_count=("temporal_partner", "count"),
            temporal_same_post_pairs=("same_post_count", lambda s: int((s > 0).sum())),
            temporal_within_5min_events=("within_5min_count", "sum"),
            temporal_within_30min_events=("within_30min_count", "sum"),
            best_temporal_rank=("temporal_rank", "max"),
            min_median_delay_minutes=("median_delay_minutes", "min"),
        )
        .reset_index()
    )
    reverse_label = {value: key for key, value in label_rank.items()}
    grouped["best_temporal_label"] = grouped["best_temporal_rank"].map(reverse_label)
    return grouped


def assign_review_priority(row: pd.Series, high_mca_threshold: float) -> str:
    high_mca = row.get("mca_score_primary", 0.0) >= high_mca_threshold
    temporal_rank = row.get("best_temporal_rank", 0.0)
    extreme = bool(row.get("is_extreme_outlier", False))
    if high_mca and temporal_rank >= 2:
        return "high_confidence_temporal_candidate"
    if high_mca and extreme:
        return "high_confidence_extreme_outlier"
    if high_mca:
        return "high_mca_review_candidate"
    if temporal_rank >= 2:
        return "temporal_only_review_candidate"
    return "low_priority_context_member"


def build_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Candidate Validation Report",
        "",
        "This table joins seed expansion membership with MCA scores, full-population clusters, and Stage 2 temporal synchrony evidence.",
        "",
        "## Priority Summary",
        "",
    ]
    counts = table["review_priority"].value_counts().rename_axis("review_priority").reset_index(name="count")
    lines.append("| review_priority | count |")
    lines.append("|---|---:|")
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.review_priority} | {int(row.count)} |")

    lines.extend(["", "## High Priority Accounts", ""])
    high = table.loc[
        table["review_priority"].isin(
            [
                "high_confidence_temporal_candidate",
                "high_confidence_extreme_outlier",
                "high_mca_review_candidate",
                "temporal_only_review_candidate",
            ]
        )
    ].copy()
    high = high.sort_values(
        ["review_priority", "mca_score_primary", "temporal_within_30min_events"],
        ascending=[True, False, False],
    )
    if high.empty:
        lines.append("No high-priority accounts found.")
    else:
        for row in high.itertuples(index=False):
            lines.append(
                f"- {row.account} ({row.seed_group}) | {row.review_priority} | "
                f"MCA={row.mca_score_primary:.3f} | cluster={row.full_cluster_kmeans} | "
                f"extreme={row.is_extreme_outlier} | temporal={row.best_temporal_label} | "
                f"<30min={int(row.temporal_within_30min_events)}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expansion = load_expansion_members(args.seed_dir)
    mca = load_mca(args.mca_path)
    clusters = load_clusters(args.cluster_path)
    temporal = load_temporal_account_summary(args.stage2_path)

    table = expansion.merge(mca, left_on="account", right_on="user_id", how="left").drop(
        columns=["user_id"]
    )
    table = table.merge(clusters, left_on="account", right_on="user_id", how="left").drop(
        columns=["user_id"]
    )
    table = table.merge(temporal, on=["seed_group", "account"], how="left")

    numeric_fill_cols = [
        "temporal_pair_count",
        "temporal_same_post_pairs",
        "temporal_within_5min_events",
        "temporal_within_30min_events",
        "best_temporal_rank",
    ]
    for col in numeric_fill_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    table["best_temporal_label"] = table["best_temporal_label"].fillna("no_temporal_sync")
    table["mca_score_primary"] = pd.to_numeric(
        table["mca_score_primary"], errors="coerce"
    ).fillna(0.0)
    table["is_high_mca"] = table["mca_score_primary"] >= args.high_mca_threshold
    table["review_priority"] = table.apply(
        assign_review_priority, axis=1, high_mca_threshold=args.high_mca_threshold
    )

    output_path = args.output_dir / "candidate_validation_table.csv"
    report_path = args.output_dir / "candidate_validation_report.md"
    table.to_csv(output_path, index=False)
    report_path.write_text(build_markdown(table), encoding="utf-8")

    print(f"Candidate validation written to {args.output_dir}")
    print(table["review_priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
