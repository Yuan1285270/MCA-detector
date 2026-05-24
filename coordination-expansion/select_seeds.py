#!/usr/bin/env python3
"""Select seed accounts for coordination expansion.

The first reproducible policy is intentionally simple: use MCA primary top-N
accounts as seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MCA_TOP_ACCOUNTS = Path("mca-scoring/output/top_accounts_primary.csv")
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select seed accounts from MCA ranking.")
    parser.add_argument("--mca-top-accounts", type=Path, default=DEFAULT_MCA_TOP_ACCOUNTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(args.mca_top_accounts)
    required = {"user_id", "mca_score_primary"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"MCA top accounts file missing columns: {sorted(missing)}")

    selected = ranking.head(args.top_n).copy()
    selected = selected[["user_id", "mca_score_primary"]].rename(columns={"user_id": "seed"})
    selected.insert(0, "selected_rank", range(1, len(selected) + 1))
    selected["selection_method"] = f"mca_primary_top_{args.top_n}"
    selected["selection_reason"] = "MCA primary ranking"

    output_path = args.output_dir / "selected_seeds.csv"
    selected.to_csv(output_path, index=False)

    print(f"Selected seeds written to {output_path}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
