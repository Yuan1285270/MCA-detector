#!/usr/bin/env python3
"""Run the full MCA detector pipeline from raw Reddit posts/comments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_POSTS_DEST = PROJECT_ROOT / "llm/data-cleaning/source_data/reddit_posts_2025.csv"
RAW_COMMENTS_DEST = PROJECT_ROOT / "llm/data-cleaning/source_data/reddit_comments_2025.csv"
PROCESSED_POSTS = PROJECT_ROOT / "llm/data-cleaning/processed_data/processed_data.csv"
PROCESSED_COMMENTS = PROJECT_ROOT / "llm/data-cleaning/processed_data/processed_comments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw comments/posts through the full MCA pipeline.")
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
    parser.add_argument(
        "--post-end-row",
        type=int,
        default=0,
        help="Exclusive post row end for Gemini. 0 means all cleaned posts.",
    )
    parser.add_argument("--comment-start-row", type=int, default=0)
    parser.add_argument(
        "--comment-end-row",
        type=int,
        default=0,
        help="Exclusive comment row end after post/comment merge. 0 means all analyzable comments.",
    )
    parser.add_argument("--top-n-seeds", type=int, default=20)
    parser.add_argument("--top-n-accounts", type=int, default=100)
    parser.add_argument("--tier-max-members", type=int, default=100)
    return parser.parse_args()


def run_step(name: str, command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def copy_if_needed(source: Path, dest: Path) -> None:
    source = source.expanduser().resolve()
    dest = dest.resolve()
    if source == dest:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"Copied {source} -> {dest}")


def cleaned_post_count() -> int:
    return int(sum(1 for _ in open(PROCESSED_POSTS, "rb")) - 1)


def analyzable_comment_count() -> int:
    comments = pd.read_csv(PROCESSED_COMMENTS, usecols=["post_id", "parent_type"], low_memory=False)
    posts = pd.read_csv(PROCESSED_POSTS, usecols=["post_id"], low_memory=False)
    valid_posts = set(posts["post_id"].astype(str))
    comments = comments.loc[
        comments["parent_type"].astype(str).eq("post")
        & comments["post_id"].astype(str).isin(valid_posts)
    ]
    return int(len(comments))


def require_existing(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_cleaning:
        copy_if_needed(args.raw_posts, RAW_POSTS_DEST)
        copy_if_needed(args.raw_comments, RAW_COMMENTS_DEST)
        run_step("Clean posts", [python, "llm/data-cleaning/preprocess_posts.py"])
        run_step("Clean comments", [python, "llm/data-cleaning/preprocess_comments.py"])
    else:
        require_existing(PROCESSED_POSTS, "Missing cleaned posts")
        require_existing(PROCESSED_COMMENTS, "Missing cleaned comments")

    llm_provider = "none" if args.skip_llm else args.llm_provider

    if llm_provider != "none":
        post_end = args.post_end_row or cleaned_post_count()
        comment_end = args.comment_end_row or analyzable_comment_count()
        if llm_provider == "gemini":
            run_step(
                "Gemini post analysis",
                [
                    python,
                    "analyze_posts_with_gemini.py",
                    "--start-row",
                    str(args.post_start_row),
                    "--end-row",
                    str(post_end),
                ],
                cwd=PROJECT_ROOT / "llm/gemini-cloud",
            )
            run_step(
                "Gemini comment feedback analysis",
                [
                    python,
                    "analyze_comment_feedback_with_gemini.py",
                    "--start-row",
                    str(args.comment_start_row),
                    "--end-row",
                    str(comment_end),
                ],
                cwd=PROJECT_ROOT / "llm/gemini-cloud",
            )
        else:
            run_step(
                "Ollama post analysis",
                [
                    python,
                    "analyze_with_ollama.py",
                    "--task",
                    "posts",
                    "--start-row",
                    str(args.post_start_row),
                    "--end-row",
                    str(post_end),
                ],
                cwd=PROJECT_ROOT / "llm/ollama-local",
            )
            run_step(
                "Ollama comment feedback analysis",
                [
                    python,
                    "analyze_with_ollama.py",
                    "--task",
                    "comments",
                    "--start-row",
                    str(args.comment_start_row),
                    "--end-row",
                    str(comment_end),
                ],
                cwd=PROJECT_ROOT / "llm/ollama-local",
            )
        run_step(
            "Build formal LLM exports",
            [python, "llm/gemini-cloud/build_llm_exports.py", "--provider", llm_provider],
        )
    else:
        require_existing(PROJECT_ROOT / "llm/Export/reddit_posts_analyzed.csv.gz", "Missing analyzed posts export")
        require_existing(PROJECT_ROOT / "llm/Export/reddit_comments_analyzed.csv.gz", "Missing analyzed comments export")

    run_step("Build account feature matrix", [python, "mca-scoring/build_account_feature_matrix.py"])

    if not args.skip_adjacency:
        run_step("Build adjacency graphs", [python, "adjacency/build_adjacency_matrices.py"])
    else:
        require_existing(PROJECT_ROOT / "adjacency/output/all_interaction_edge_stats.csv", "Missing adjacency output")

    run_step(
        "MCA + expansion + validation",
        [
            python,
            "coordination-expansion/run_pipeline.py",
            "--top-n-seeds",
            str(args.top_n_seeds),
            "--top-n-accounts",
            str(args.top_n_accounts),
            "--tier-max-members",
            str(args.tier_max_members),
        ],
    )

    print("\n=== Full Pipeline Complete ===")
    print("Final account roles: coordination-expansion/output/account-roles/account_role_table.csv")
    print("Final group summary: coordination-expansion/output/final-summary/final_group_summary.csv")


if __name__ == "__main__":
    main()
