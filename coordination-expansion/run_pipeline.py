#!/usr/bin/env python3
"""Run the coordination discovery and validation pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_SEEDS = PROJECT_ROOT / "coordination-expansion/output/selected_seeds.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCA -> expansion -> validation pipeline.")
    parser.add_argument("--top-n-seeds", type=int, default=20)
    parser.add_argument("--top-n-accounts", type=int, default=100)
    parser.add_argument("--tier-max-members", type=int, default=100)
    parser.add_argument(
        "--skip-mca",
        action="store_true",
        help="Reuse existing mca-scoring/output files instead of recomputing MCA scores.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_selected_seeds(path: Path) -> list[str]:
    selected = pd.read_csv(path)
    if "seed" not in selected.columns:
        raise ValueError(f"Selected seeds file missing `seed` column: {path}")
    seeds = selected["seed"].astype(str).dropna().tolist()
    if not seeds:
        raise ValueError(f"No seeds selected in {path}")
    return seeds


def print_final_outputs() -> None:
    outputs = [
        "coordination-expansion/output/selected_seeds.csv",
        "coordination-expansion/output/seeds/seed_expansion_summary.csv",
        "coordination-expansion/output/stage2-verification/stage2_group_summary.csv",
        "coordination-expansion/output/candidate-validation/candidate_validation_table.csv",
        "coordination-expansion/output/final-summary/final_group_summary.csv",
        "coordination-expansion/output/behavior-profile/behavior_profile_table.csv",
        "coordination-expansion/output/account-roles/account_role_table.csv",
    ]
    print("\n=== Outputs ===")
    for output in outputs:
        print(output)

    summary_path = PROJECT_ROOT / "coordination-expansion/output/final-summary/final_group_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        cols = [
            "group_rank",
            "seed_group",
            "group_priority",
            "member_count",
            "p1_count",
            "p2_count",
            "p3_count",
        ]
        existing = [col for col in cols if col in summary.columns]
        print("\n=== Final Group Ranking ===")
        print(summary[existing].to_string(index=False))


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_mca:
        run_step(
            "MCA scoring",
            [
                python,
                "mca-scoring/score_accounts.py",
                "--top-n",
                str(args.top_n_accounts),
            ],
        )

    run_step(
        "Seed selection",
        [
            python,
            "coordination-expansion/select_seeds.py",
            "--top-n",
            str(args.top_n_seeds),
        ],
    )
    seeds = load_selected_seeds(DEFAULT_SELECTED_SEEDS)

    run_step(
        "Seed expansion",
        [
            python,
            "coordination-expansion/discover_coordination_groups.py",
            "--seeds",
            *seeds,
            "--tier-max-members",
            str(args.tier_max_members),
        ],
    )
    run_step(
        "Temporal verification",
        [python, "coordination-expansion/stage2_temporal_verification.py"],
    )
    run_step(
        "Candidate validation",
        [python, "coordination-expansion/build_candidate_validation_table.py"],
    )
    run_step(
        "Final group summary",
        [python, "coordination-expansion/build_final_group_summary.py"],
    )
    run_step(
        "Behavior profile",
        [python, "coordination-expansion/build_behavior_profile_table.py"],
    )
    run_step(
        "Account roles",
        [python, "coordination-expansion/build_account_role_table.py"],
    )
    print_final_outputs()


if __name__ == "__main__":
    main()
