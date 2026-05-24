#!/usr/bin/env python3
"""Merge Gemini batch outputs into the formal analyzed export files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POST_PATTERNS = [
    "llm/gemini-cloud/output/post_analysis_*.csv",
    "llm/gemini-cloud/output/final_post_*.csv",
]
DEFAULT_COMMENT_PATTERNS = [
    "llm/gemini-cloud/output/comments/comment_feedback_*.csv",
    "llm/gemini-cloud/output/comments_genai_builder/comment_feedback_genai_builder_*.csv",
]
OLLAMA_POST_PATTERNS = ["llm/ollama-local/output/posts/post_analysis_*.csv"]
OLLAMA_COMMENT_PATTERNS = ["llm/ollama-local/output/comments/comment_feedback_*.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build llm/Export analyzed CSV files from batch outputs.")
    parser.add_argument("--provider", choices=["gemini", "ollama", "all"], default="gemini")
    parser.add_argument("--posts-pattern", action="append", default=None)
    parser.add_argument("--comments-pattern", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("llm/Export"))
    parser.add_argument("--no-gzip", action="store_true")
    return parser.parse_args()


def find_batch_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(PROJECT_ROOT.glob(pattern))
    return sorted(set(files))


def merge_batches(files: list[Path], id_col: str) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(f"No batch files found for {id_col}.")

    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_batch_file"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    if id_col not in merged.columns:
        raise ValueError(f"Merged batches are missing `{id_col}`.")

    sort_cols = [col for col in ["__row_id", "created_utc", id_col] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable")
    merged = merged.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    return merged


def write_export(df: pd.DataFrame, path: Path, *, gzip: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if gzip:
        df.to_csv(path.with_suffix(path.suffix + ".gz"), index=False, compression="gzip")


def main() -> None:
    args = parse_args()
    if args.provider == "gemini":
        default_post_patterns = DEFAULT_POST_PATTERNS
        default_comment_patterns = DEFAULT_COMMENT_PATTERNS
    elif args.provider == "ollama":
        default_post_patterns = OLLAMA_POST_PATTERNS
        default_comment_patterns = OLLAMA_COMMENT_PATTERNS
    else:
        default_post_patterns = DEFAULT_POST_PATTERNS + OLLAMA_POST_PATTERNS
        default_comment_patterns = DEFAULT_COMMENT_PATTERNS + OLLAMA_COMMENT_PATTERNS

    post_patterns = args.posts_pattern or default_post_patterns
    comment_patterns = args.comments_pattern or default_comment_patterns

    post_files = find_batch_files(post_patterns)
    comment_files = find_batch_files(comment_patterns)

    print("Merging post analysis batches:")
    for path in post_files:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    posts = merge_batches(post_files, "post_id")

    print("Merging comment feedback batches:")
    for path in comment_files:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    comments = merge_batches(comment_files, "comment_id")

    output_dir = PROJECT_ROOT / args.output_dir
    write_export(posts, output_dir / "reddit_posts_analyzed.csv", gzip=not args.no_gzip)
    write_export(comments, output_dir / "reddit_comments_analyzed.csv", gzip=not args.no_gzip)

    print("LLM analyzed exports written.")
    print(f"Posts: {len(posts):,} rows")
    print(f"Comments: {len(comments):,} rows")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
