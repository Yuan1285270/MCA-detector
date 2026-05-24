#!/usr/bin/env python3
"""Run the back half: MCA scoring, expansion, verification, and summaries."""

from __future__ import annotations

import argparse
import sys

from run_full_pipeline import run_back_half


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the back half of the MCA detector pipeline.")
    parser.add_argument("--top-n-seeds", type=int, default=20)
    parser.add_argument("--top-n-accounts", type=int, default=100)
    parser.add_argument("--tier-max-members", type=int, default=100)
    parser.add_argument(
        "--skip-mca",
        action="store_true",
        help="Reuse existing mca-scoring/output files instead of recomputing MCA scores.",
    )
    return parser.parse_args()


def main() -> None:
    run_back_half(parse_args(), sys.executable)


if __name__ == "__main__":
    main()
