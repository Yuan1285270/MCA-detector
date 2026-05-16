#!/usr/bin/env python3
"""Run the front half: raw data to analyzed exports, features, and graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_full_pipeline import RAW_COMMENTS_DEST, RAW_POSTS_DEST, run_front_half


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the front half of the MCA detector pipeline.")
    parser.add_argument("--raw-posts", type=Path, default=RAW_POSTS_DEST)
    parser.add_argument("--raw-comments", type=Path, default=RAW_COMMENTS_DEST)
    parser.add_argument("--skip-cleaning", action="store_true")
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "gemini", "none"],
        default="ollama",
        help="LLM backend for post/comment analysis. `none` uses existing llm/Export files.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Deprecated alias for --llm-provider none.",
    )
    parser.add_argument(
        "--skip-adjacency",
        action="store_true",
        help="Use existing adjacency/output graph files.",
    )
    parser.add_argument("--post-start-row", type=int, default=0)
    parser.add_argument("--post-end-row", type=int, default=0)
    parser.add_argument("--comment-start-row", type=int, default=0)
    parser.add_argument("--comment-end-row", type=int, default=0)
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=1,
        help="Number of parallel LLM batch workers per task. 1 means serial execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_workers < 1:
        raise ValueError("--llm-workers must be >= 1")
    run_front_half(args, sys.executable)


if __name__ == "__main__":
    main()
