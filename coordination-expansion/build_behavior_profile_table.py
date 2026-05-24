#!/usr/bin/env python3
"""Build local behavior profiles for validation candidates.

This script uses local feature-matrix behavior fields. It does not infer
cross-subreddit behavior because the local raw exports do not include a
subreddit column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_VALIDATION_TABLE = Path(
    "coordination-expansion/output/candidate-validation/candidate_validation_table.csv"
)
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/behavior-profile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local behavior profile table.")
    parser.add_argument("--validation-table", type=Path, default=DEFAULT_VALIDATION_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def behavior_label(row: pd.Series) -> str:
    total_activity = row.get("comment_count", 0.0) + row.get("post_count", 0.0)
    active_days = row.get("active_days", 0.0)
    comments_per_day = row.get("comments_per_day", 0.0)
    posts_per_day = row.get("posts_per_day", 0.0)
    burst_ratio = row.get("burst_ratio", 0.0)
    extreme = bool(row.get("is_extreme_outlier", False))

    if extreme:
        return "extreme_outlier_behavior"
    if total_activity < 10:
        return "low_activity_unknown"
    if active_days <= 14 and total_activity >= 50:
        return "short_window_high_activity"
    if comments_per_day >= 5 or posts_per_day >= 2:
        return "high_frequency_activity"
    if burst_ratio >= 5 and total_activity >= 25:
        return "bursty_activity"
    return "normal_range_activity"


def behavior_reason(row: pd.Series) -> str:
    parts = [
        f"comments={row.comment_count:.0f}",
        f"posts={row.post_count:.0f}",
        f"active_days={row.active_days:.0f}",
        f"comments_per_day={row.comments_per_day:.2f}",
        f"posts_per_day={row.posts_per_day:.2f}",
        f"burst_ratio={row.burst_ratio:.2f}",
        f"extreme_outlier={row.is_extreme_outlier}",
    ]
    return "; ".join(parts)


def build_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Behavior Profile Report",
        "",
        "This report summarizes local behavior features for candidate expansion accounts.",
        "",
        "Important limitation: the local raw exports do not include a `subreddit` column, so this report cannot reproduce Pullpush-style subreddit distribution checks.",
        "",
        "## Label Summary",
        "",
        "| behavior_profile | count |",
        "|---|---:|",
    ]
    counts = table["behavior_profile"].value_counts().rename_axis("behavior_profile").reset_index(name="count")
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.behavior_profile} | {int(row.count)} |")

    lines.extend(["", "## Notable Accounts", ""])
    notable = table.loc[
        table["behavior_profile"].isin(
            [
                "extreme_outlier_behavior",
                "short_window_high_activity",
                "high_frequency_activity",
                "bursty_activity",
                "low_activity_unknown",
            ]
        )
    ].copy()
    notable = notable.sort_values(
        ["behavior_profile", "mca_score_primary", "comments_per_day"],
        ascending=[True, False, False],
    )
    if notable.empty:
        lines.append("No notable behavior profiles found.")
    else:
        for row in notable.itertuples(index=False):
            lines.append(
                f"- {row.account} ({row.seed_group}) | {row.behavior_profile} | "
                f"MCA={row.mca_score_primary:.3f} | comments={row.comment_count:.0f} | "
                f"posts={row.post_count:.0f} | active_days={row.active_days:.0f} | "
                f"comments/day={row.comments_per_day:.2f} | posts/day={row.posts_per_day:.2f} | "
                f"priority={row.review_priority}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.validation_table)
    numeric_cols = [
        "comment_count",
        "post_count",
        "active_days",
        "comments_per_day",
        "posts_per_day",
        "burst_ratio",
        "night_activity_ratio",
        "weekend_activity_ratio",
        "mca_score_primary",
    ]
    for col in numeric_cols:
        if col not in table.columns:
            table[col] = 0.0
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    if "is_extreme_outlier" not in table.columns:
        table["is_extreme_outlier"] = False
    table["is_extreme_outlier"] = table["is_extreme_outlier"].fillna(False).astype(bool)

    keep_cols = [
        "seed_group",
        "account",
        "tier",
        "review_priority",
        "mca_score_primary",
        "is_high_mca",
        "is_extreme_outlier",
        "full_cluster_kmeans",
        "best_temporal_label",
        "comment_count",
        "post_count",
        "active_days",
        "comments_per_day",
        "posts_per_day",
        "burst_ratio",
        "night_activity_ratio",
        "weekend_activity_ratio",
        "anomaly_label",
        "anomaly_score",
    ]
    output = table[[col for col in keep_cols if col in table.columns]].copy()
    output["behavior_profile"] = output.apply(behavior_label, axis=1)
    output["behavior_reason"] = output.apply(behavior_reason, axis=1)

    output_path = args.output_dir / "behavior_profile_table.csv"
    report_path = args.output_dir / "behavior_profile_report.md"
    output.to_csv(output_path, index=False)
    report_path.write_text(build_markdown(output), encoding="utf-8")

    print(f"Behavior profile written to {args.output_dir}")
    print(output["behavior_profile"].value_counts().to_string())


if __name__ == "__main__":
    main()
