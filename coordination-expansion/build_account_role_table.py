#!/usr/bin/env python3
"""Assign lightweight account roles inside candidate groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_VALIDATION_TABLE = Path(
    "coordination-expansion/output/candidate-validation/candidate_validation_table.csv"
)
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/account-roles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight account role table.")
    parser.add_argument("--validation-table", type=Path, default=DEFAULT_VALIDATION_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attack-threshold", type=float, default=0.50)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    return parser.parse_args()


def role_label(row: pd.Series, attack_threshold: float, support_threshold: float) -> str:
    if int(row.get("tier", 99)) == 0:
        return "leader_instigator"
    if row.get("comment_label_oppositional_ratio", 0.0) >= attack_threshold:
        return "comment_attacker"
    if row.get("comment_label_supportive_ratio", 0.0) >= support_threshold:
        return "comment_supporter"
    return "context_member"


def role_label_zh(label: str) -> str:
    return {
        "leader_instigator": "帶頭起鬨",
        "comment_attacker": "留言攻擊者",
        "comment_supporter": "留言支持者",
        "context_member": "背景成員",
    }[label]


def role_reason(row: pd.Series) -> str:
    label = row.role_label
    if label == "leader_instigator":
        return "seed account for this candidate group"
    if label == "comment_attacker":
        return f"oppositional comment ratio={row.comment_label_oppositional_ratio:.3f}"
    if label == "comment_supporter":
        return f"supportive comment ratio={row.comment_label_supportive_ratio:.3f}"
    return "no dominant attack/support role from current evidence"


def build_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Account Role Report",
        "",
        "Lightweight role labels inside candidate groups. These are descriptive review aids, not final actor claims.",
        "",
        "Roles:",
        "",
        "- `leader_instigator` / 帶頭起鬨: seed account",
        "- `comment_attacker` / 留言攻擊者: high oppositional comment ratio",
        "- `comment_supporter` / 留言支持者: high supportive comment ratio",
        "- `context_member` / 背景成員: retained by expansion, no dominant attack/support role",
        "",
        "## Role Counts",
        "",
        "| seed_group | leader | attacker | supporter | context |",
        "|---|---:|---:|---:|---:|",
    ]
    pivot = (
        table.pivot_table(
            index="seed_group",
            columns="role_label",
            values="account",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["leader_instigator", "comment_attacker", "comment_supporter", "context_member"]:
        if col not in pivot.columns:
            pivot[col] = 0
    for row in pivot.itertuples(index=False):
        lines.append(
            f"| {row.seed_group} | {int(row.leader_instigator)} | "
            f"{int(row.comment_attacker)} | {int(row.comment_supporter)} | "
            f"{int(row.context_member)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.validation_table)
    for col in ["comment_label_oppositional_ratio", "comment_label_supportive_ratio"]:
        if col not in table.columns:
            table[col] = 0.0
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)

    role_cols = [
        "seed_group",
        "account",
        "tier",
        "review_priority",
        "mca_score_primary",
        "co_negative_weight_to_seed",
        "comment_label_oppositional_ratio",
        "comment_label_supportive_ratio",
        "best_temporal_label",
        "temporal_within_30min_events",
    ]
    output = table[[col for col in role_cols if col in table.columns]].copy()
    output["role_label"] = output.apply(
        role_label,
        axis=1,
        attack_threshold=args.attack_threshold,
        support_threshold=args.support_threshold,
    )
    output["role_label_zh"] = output["role_label"].map(role_label_zh)
    output["role_reason"] = output.apply(role_reason, axis=1)

    output_path = args.output_dir / "account_role_table.csv"
    report_path = args.output_dir / "account_role_report.md"
    output.to_csv(output_path, index=False)
    report_path.write_text(build_markdown(output), encoding="utf-8")

    print(f"Account roles written to {args.output_dir}")
    print(output["role_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
