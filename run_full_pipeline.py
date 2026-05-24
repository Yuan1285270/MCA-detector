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
    parser.add_argument(
        "--skip-mca",
        action="store_true",
        help="Forwarded to the back-half runner; reuse existing MCA output.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=1,
        help="Number of parallel LLM batch workers per task. 1 means serial execution.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def run_parallel_steps(name: str, steps: list[tuple[str, list[str], Path]]) -> None:
    print(f"\n=== {name} ===")
    if not steps:
        print("No work to run.")
        return

    processes: list[tuple[str, subprocess.Popen]] = []
    for step_name, command, cwd in steps:
        print(f"[start] {step_name}: {' '.join(command)}")
        processes.append((step_name, subprocess.Popen(command, cwd=cwd)))

    failures: list[tuple[str, int]] = []
    for step_name, process in processes:
        return_code = process.wait()
        if return_code == 0:
            print(f"[done] {step_name}")
        else:
            print(f"[failed] {step_name}: exit code {return_code}")
            failures.append((step_name, return_code))

    if failures:
        failed_text = ", ".join(f"{step}={code}" for step, code in failures)
        raise RuntimeError(f"Parallel step failed: {failed_text}")


def row_ranges(start: int, end: int, workers: int) -> list[tuple[int, int]]:
    if start < 0 or end <= start:
        raise ValueError(f"Expected 0 <= start < end, got start={start}, end={end}")
    worker_count = max(1, min(workers, end - start))
    total = end - start
    base = total // worker_count
    remainder = total % worker_count

    ranges = []
    cursor = start
    for index in range(worker_count):
        size = base + (1 if index < remainder else 0)
        next_cursor = cursor + size
        ranges.append((cursor, next_cursor))
        cursor = next_cursor
    return ranges


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


def build_llm_steps(
    provider: str,
    *,
    python: str,
    task: str,
    start: int,
    end: int,
    workers: int,
) -> list[tuple[str, list[str], Path]]:
    ranges = row_ranges(start, end, workers)
    steps: list[tuple[str, list[str], Path]] = []
    for batch_start, batch_end in ranges:
        if provider == "gemini":
            if task == "posts":
                script = "analyze_posts_with_gemini.py"
                label = "Gemini post analysis"
            else:
                script = "analyze_comment_feedback_with_gemini.py"
                label = "Gemini comment feedback analysis"
            command = [
                python,
                script,
                "--start-row",
                str(batch_start),
                "--end-row",
                str(batch_end),
            ]
            cwd = PROJECT_ROOT / "llm/gemini-cloud"
        else:
            label = f"Ollama {task[:-1] if task.endswith('s') else task} analysis"
            command = [
                python,
                "analyze_with_ollama.py",
                "--task",
                task,
                "--start-row",
                str(batch_start),
                "--end-row",
                str(batch_end),
            ]
            cwd = PROJECT_ROOT / "llm/ollama-local"
        steps.append((f"{label} [{batch_start}, {batch_end})", command, cwd))
    return steps


def run_llm_analysis(
    provider: str,
    *,
    python: str,
    post_start: int,
    post_end: int,
    comment_start: int,
    comment_end: int,
    workers: int,
) -> None:
    post_steps = build_llm_steps(
        provider,
        python=python,
        task="posts",
        start=post_start,
        end=post_end,
        workers=workers,
    )
    comment_steps = build_llm_steps(
        provider,
        python=python,
        task="comments",
        start=comment_start,
        end=comment_end,
        workers=workers,
    )

    if workers <= 1:
        for step_name, command, cwd in [*post_steps, *comment_steps]:
            run_step(step_name, command, cwd=cwd)
        return

    run_parallel_steps(f"{provider.title()} post analysis ({workers} workers)", post_steps)
    run_parallel_steps(f"{provider.title()} comment feedback analysis ({workers} workers)", comment_steps)


def run_front_half(args: argparse.Namespace, python: str) -> None:
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
        run_llm_analysis(
            llm_provider,
            python=python,
            post_start=args.post_start_row,
            post_end=post_end,
            comment_start=args.comment_start_row,
            comment_end=comment_end,
            workers=args.llm_workers,
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

    print("\n=== Front Half Complete ===")
    print("Analyzed posts: llm/Export/reddit_posts_analyzed.csv.gz")
    print("Analyzed comments: llm/Export/reddit_comments_analyzed.csv.gz")
    print("Account features: Archive/export_working_files/account_feature_matrix.csv")
    print("Adjacency graphs: adjacency/output/")


def run_back_half(args: argparse.Namespace, python: str) -> None:
    command = [
        python,
        "coordination-expansion/run_pipeline.py",
        "--top-n-seeds",
        str(args.top_n_seeds),
        "--top-n-accounts",
        str(args.top_n_accounts),
        "--tier-max-members",
        str(args.tier_max_members),
    ]
    if getattr(args, "skip_mca", False):
        command.append("--skip-mca")
    run_step("MCA + expansion + validation", command)


def main() -> None:
    args = parse_args()
    python = sys.executable
    if args.llm_workers < 1:
        raise ValueError("--llm-workers must be >= 1")

    run_front_half(args, python)
    run_back_half(args, python)

    print("\n=== Full Pipeline Complete ===")
    print("Final account roles: coordination-expansion/output/account-roles/account_role_table.csv")
    print("Final group summary: coordination-expansion/output/final-summary/final_group_summary.csv")


if __name__ == "__main__":
    main()
