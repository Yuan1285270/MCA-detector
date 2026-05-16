#!/usr/bin/env python3
"""Build final seed-group review summary.

This aggregates account-level review priority into group-level ordering. The
output is still a review aid, not a final verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SEED_SUMMARY = Path("coordination-expansion/output/seeds/seed_expansion_summary.csv")
DEFAULT_VALIDATION_TABLE = Path(
    "coordination-expansion/output/candidate-validation/candidate_validation_table.csv"
)
DEFAULT_STAGE2_SUMMARY = Path("coordination-expansion/output/stage2-verification/stage2_group_summary.csv")
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/final-summary")

P1_PRIORITIES = {
    "high_confidence_temporal_candidate",
    "high_confidence_extreme_outlier",
}
P2_PRIORITIES = {
    "high_mca_review_candidate",
    "temporal_only_review_candidate",
}
P3_PRIORITIES = {
    "low_priority_context_member",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final group-level review summary.")
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--validation-table", type=Path, default=DEFAULT_VALIDATION_TABLE)
    parser.add_argument("--stage2-summary", type=Path, default=DEFAULT_STAGE2_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def group_priority_label(row: pd.Series) -> str:
    if row["p1_count"] >= 2:
        return "G1_multiple_high_priority_accounts"
    if row["p1_count"] == 1 and row["p2_count"] >= 1:
        return "G1_high_priority_plus_support"
    if row["p1_count"] == 1:
        return "G2_single_high_priority_account"
    if row["p2_count"] >= 2:
        return "G2_multiple_review_candidates"
    if row["p2_count"] == 1:
        return "G3_single_review_candidate"
    return "G4_context_group"


def group_interpretation(row: pd.Series) -> str:
    if row["p1_count"] >= 2 and row["strong_temporal_sync"] > 0:
        return "candidate group with multiple high-priority accounts and temporal evidence"
    if row["p1_count"] >= 2:
        return "candidate group with multiple high-priority accounts"
    if row["p1_count"] == 1 and row["p2_count"] >= 1:
        return "candidate group with one high-priority account plus supporting review candidates"
    if row["p1_count"] == 1:
        return "candidate group with one high-priority account"
    if row["p2_count"] > 0:
        return "candidate group with review candidates but no P1 account"
    return "context group retained for structure, low current review priority"


def build_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Final Group Summary",
        "",
        "This report aggregates account-level review priority into seed-group ordering.",
        "",
        "Priority definitions:",
        "",
        "- `P1`: MCA plus temporal evidence, or MCA plus extreme outlier",
        "- `P2`: high MCA only, or temporal evidence only",
        "- `P3`: context member retained by expansion, with weak validation evidence",
        "",
        "Group labels are review priority labels, not final bot verdicts.",
        "",
        "## Ranked Groups",
        "",
        "| rank | group_seed | group_priority | members | P1 | P2 | P3 | strong temporal pairs | moderate temporal pairs | interpretation |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.group_rank)} | {row.seed_group} | {row.group_priority} | "
            f"{int(row.member_count)} | {int(row.p1_count)} | {int(row.p2_count)} | "
            f"{int(row.p3_count)} | {int(row.strong_temporal_sync)} | "
            f"{int(row.moderate_temporal_sync)} | {row.interpretation} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_summary = pd.read_csv(args.seed_summary).rename(columns={"seed": "seed_group"})
    validation = pd.read_csv(args.validation_table)
    stage2 = pd.read_csv(args.stage2_summary) if args.stage2_summary.exists() else pd.DataFrame()
    if not stage2.empty:
        stage2 = stage2.rename(columns={"group_seed": "seed_group"})

    validation["priority_bucket"] = "P3"
    validation.loc[validation["review_priority"].isin(P1_PRIORITIES), "priority_bucket"] = "P1"
    validation.loc[validation["review_priority"].isin(P2_PRIORITIES), "priority_bucket"] = "P2"

    grouped = (
        validation.groupby("seed_group")
        .agg(
            member_count=("account", "nunique"),
            p1_count=("priority_bucket", lambda values: int((values == "P1").sum())),
            p2_count=("priority_bucket", lambda values: int((values == "P2").sum())),
            p3_count=("priority_bucket", lambda values: int((values == "P3").sum())),
            high_mca_count=("is_high_mca", "sum"),
            extreme_outlier_count=("is_extreme_outlier", "sum"),
            temporal_5min_events=("temporal_within_5min_events", "sum"),
            temporal_30min_events=("temporal_within_30min_events", "sum"),
            avg_mca_score=("mca_score_primary", "mean"),
            max_mca_score=("mca_score_primary", "max"),
            top_accounts=(
                "account",
                lambda accounts: "; ".join(accounts.astype(str).head(12)),
            ),
        )
        .reset_index()
    )

    if not stage2.empty:
        grouped = grouped.merge(stage2, on="seed_group", how="left")
    for col in ["strong_temporal_sync", "moderate_temporal_sync", "weak_temporal_overlap", "no_temporal_sync"]:
        if col not in grouped.columns:
            grouped[col] = 0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0).astype(int)

    grouped = grouped.merge(
        seed_summary[
            [
                "seed_group",
                "tier1_co_negative_count",
                "tier4_two_hop_count",
                "internal_coordination_edge_count",
                "shared_negative_target_count",
                "avg_rhetorical_score",
                "avg_oppositional_stance_ratio",
                "automation_anomaly_fraction",
            ]
        ],
        on="seed_group",
        how="left",
    )

    grouped["group_priority"] = grouped.apply(group_priority_label, axis=1)
    grouped["interpretation"] = grouped.apply(group_interpretation, axis=1)
    grouped = grouped.sort_values(
        [
            "p1_count",
            "p2_count",
            "strong_temporal_sync",
            "moderate_temporal_sync",
            "extreme_outlier_count",
            "high_mca_count",
            "internal_coordination_edge_count",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    grouped.insert(0, "group_rank", range(1, len(grouped) + 1))

    output_path = args.output_dir / "final_group_summary.csv"
    report_path = args.output_dir / "final_group_summary_report.md"
    grouped.to_csv(output_path, index=False)
    report_path.write_text(build_markdown(grouped), encoding="utf-8")

    print(f"Final group summary written to {args.output_dir}")
    print(
        grouped[
            [
                "group_rank",
                "seed_group",
                "group_priority",
                "member_count",
                "p1_count",
                "p2_count",
                "p3_count",
                "strong_temporal_sync",
                "moderate_temporal_sync",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
