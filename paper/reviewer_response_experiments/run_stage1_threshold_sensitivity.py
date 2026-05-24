#!/usr/bin/env python3
"""Run a small co-negative threshold sensitivity check.

This script reruns Stage 1 and Stage 2 into temporary experiment folders. It
does not modify the main coordination-expansion outputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
RUN_ROOT = EXP_ROOT / "threshold_runs"
OUT = EXP_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS_PATH = ROOT / "coordination-expansion/output/selected_seeds.csv"


def load_seeds() -> list[str]:
    selected = pd.read_csv(SEEDS_PATH)
    if "seed" not in selected.columns:
        raise ValueError(f"Missing seed column in {SEEDS_PATH}")
    seeds = selected["seed"].astype(str).dropna().tolist()
    if not seeds:
        raise ValueError("No selected seeds found")
    return seeds


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_threshold(threshold: float, run_dir: Path) -> dict[str, object]:
    seed_summary = pd.read_csv(run_dir / "seeds/seed_expansion_summary.csv")
    stage2 = pd.read_csv(run_dir / "stage2-verification/stage2_verification_evidence.csv")

    candidate_accounts: set[str] = set()
    membership_rows = 0
    for seed in seed_summary["seed"].astype(str):
        path = run_dir / "seeds" / seed / "tiered_expansion_members.csv"
        members = pd.read_csv(path)
        included = members.loc[members["include"].eq(True), "candidate"].astype(str)
        membership_rows += int(len(included))
        candidate_accounts.update(included.tolist())

    return {
        "co_negative_threshold": threshold,
        "groups": int(seed_summary["seed"].nunique()),
        "membership_rows": membership_rows,
        "candidate_accounts": len(candidate_accounts),
        "pair_rows": int(len(stage2)),
        "strong_pairs": int((stage2["verification_label"] == "strong_temporal_sync").sum()),
        "moderate_pairs": int((stage2["verification_label"] == "moderate_temporal_sync").sum()),
        "robust_pairs": int((stage2["temporal_confidence"] == "robust").sum()),
        "moderate_review_pairs": int((stage2["temporal_confidence"] == "moderate_review").sum()),
    }


def main() -> None:
    seeds = load_seeds()
    rows = []
    for threshold in [0.10, 0.20, 0.30]:
        tag = f"t{int(threshold * 100):03d}"
        run_dir = RUN_ROOT / tag
        stage2_dir = run_dir / "stage2-verification"
        run_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)

        run_step(
            f"Stage 1 threshold {threshold:.2f}",
            [
                sys.executable,
                "coordination-expansion/discover_coordination_groups.py",
                "--output-dir",
                str(run_dir),
                "--seeds",
                *seeds,
                "--tier-co-negative-threshold",
                str(threshold),
                "--tier-max-members",
                "100",
            ],
        )
        run_step(
            f"Stage 2 threshold {threshold:.2f}",
            [
                sys.executable,
                "coordination-expansion/stage2_temporal_verification.py",
                "--seed-dir",
                str(run_dir / "seeds"),
                "--output-dir",
                str(stage2_dir),
            ],
        )
        rows.append(summarize_threshold(threshold, run_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "stage1_co_negative_threshold_sensitivity.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT / 'stage1_co_negative_threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
