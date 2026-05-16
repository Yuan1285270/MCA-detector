#!/usr/bin/env python3
"""Stage 2 temporal synchrony verification for seed expansion groups.

Stage 1 discovers candidate coordination groups. This script checks whether
members of each group appear in the same Reddit thread within short time
windows, producing pair-level evidence for second-stage review.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED_DIR = Path("coordination-expansion/output/seeds")
DEFAULT_COMMENTS_PATH = Path("Archive/export_working_files/comment_feedback_all_merged.csv")
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/stage2-verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute temporal synchrony evidence by seed group.")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--comments-path", type=Path, default=DEFAULT_COMMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[],
        help="Optional seed names. Defaults to seed_expansion_summary.csv order.",
    )
    parser.add_argument("--window-strong-minutes", type=float, default=5.0)
    parser.add_argument("--window-moderate-minutes", type=float, default=30.0)
    parser.add_argument(
        "--max-comments-per-author",
        type=int,
        default=100,
        help="Keep only each account's most recent N comments. Use 0 to keep all local comments.",
    )
    return parser.parse_args()


def normalize_user_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_seed_names(seed_dir: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    summary_path = seed_dir / "seed_expansion_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, usecols=["seed"])
        return summary["seed"].astype(str).tolist()
    return sorted(path.name for path in seed_dir.iterdir() if path.is_dir())


def load_group_members(seed_dir: Path, seed: str) -> list[str]:
    path = seed_dir / seed / "tiered_expansion_members.csv"
    members = pd.read_csv(path)
    members = members.loc[members["include"].eq(True), "candidate"]
    return members.astype(str).dropna().drop_duplicates().tolist()


def load_comments(path: Path, authors: set[str], max_comments_per_author: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if {"comment_id", "post_id", "author", "created_utc"}.issubset(header.columns):
        comments = pd.read_csv(
            path,
            usecols=["comment_id", "post_id", "author", "created_utc"],
            low_memory=False,
        )
    elif {"comment_id", "link_id", "author", "created_utc"}.issubset(header.columns):
        comments = pd.read_csv(
            path,
            usecols=["comment_id", "link_id", "author", "created_utc"],
            low_memory=False,
        )
        comments = comments.rename(columns={"link_id": "post_id"})
        comments["post_id"] = comments["post_id"].astype("string").str.replace(
            r"^t3_", "", regex=True
        )
    else:
        raise ValueError(
            "Comments file must contain either comment_id/post_id/author/created_utc "
            "or comment_id/link_id/author/created_utc."
        )
    comments["author"] = normalize_user_id(comments["author"])
    comments = comments.loc[comments["author"].isin(authors)].copy()
    comments["post_id"] = comments["post_id"].astype("string").str.strip()
    comments["created_utc"] = pd.to_numeric(comments["created_utc"], errors="coerce")
    comments = comments.dropna(subset=["author", "post_id", "created_utc"])
    comments = comments.drop_duplicates(subset=["comment_id"])
    comments = comments.sort_values(["author", "created_utc"], ascending=[True, False])
    if max_comments_per_author > 0:
        comments = comments.groupby("author", group_keys=False).head(max_comments_per_author)
    return comments.sort_values(["author", "post_id", "created_utc"])


def load_co_negative_weights(seed_dir: Path, seed: str) -> dict[tuple[str, str], float]:
    path = seed_dir / seed / "internal_coordination_edges.csv"
    if not path.exists():
        return {}
    edges = pd.read_csv(path)
    if "weight_co_negative_target" not in edges.columns:
        return {}
    edges = edges.loc[edges["layer"].astype(str).eq("co_negative_target")].copy()
    weights: dict[tuple[str, str], float] = {}
    for row in edges.itertuples(index=False):
        source = str(row.source_author)
        target = str(row.target_author)
        value = getattr(row, "weight_co_negative_target")
        if pd.isna(value):
            continue
        weights[tuple(sorted((source, target)))] = max(
            weights.get(tuple(sorted((source, target))), 0.0),
            float(value),
        )
    return weights


def account_post_times(comments: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for (author, post_id), group in comments.groupby(["author", "post_id"], sort=False):
        output.setdefault(str(author), {})[str(post_id)] = group["created_utc"].to_numpy(dtype=float)
    return output


def pair_temporal_metrics(
    left_posts: dict[str, np.ndarray],
    right_posts: dict[str, np.ndarray],
    *,
    strong_window: float,
    moderate_window: float,
) -> dict[str, object]:
    shared_posts = sorted(set(left_posts) & set(right_posts))
    all_min_delays: list[float] = []
    within_strong = 0
    within_moderate = 0

    for post_id in shared_posts:
        left_times = left_posts[post_id]
        right_times = right_posts[post_id]
        diffs = np.abs(left_times[:, None] - right_times[None, :]) / 60.0
        all_min_delays.append(float(np.min(diffs)))
        within_strong += int((diffs <= strong_window).sum())
        within_moderate += int((diffs <= moderate_window).sum())

    median_delay = float(np.median(all_min_delays)) if all_min_delays else np.nan
    min_delay = float(np.min(all_min_delays)) if all_min_delays else np.nan
    return {
        "same_post_count": len(shared_posts),
        "within_5min_count": within_strong,
        "within_30min_count": within_moderate,
        "median_delay_minutes": median_delay,
        "min_delay_minutes": min_delay,
        "shared_link_ids": "; ".join(shared_posts[:20]),
    }


def label_pair(metrics: dict[str, object]) -> str:
    within_5 = int(metrics["within_5min_count"])
    within_30 = int(metrics["within_30min_count"])
    same_post = int(metrics["same_post_count"])
    if within_5 > 0:
        return "strong_temporal_sync"
    if within_30 >= 2:
        return "moderate_temporal_sync"
    if same_post > 0:
        return "weak_temporal_overlap"
    return "no_temporal_sync"


def build_markdown(pair_rows: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Stage 2 Temporal Synchrony Verification",
        "",
        "This report verifies Stage 1 candidate groups by checking whether group members appear in the same Reddit thread within short time windows.",
        "",
        "Labels:",
        "",
        "- `strong_temporal_sync`: at least one co-comment event within 5 minutes",
        "- `moderate_temporal_sync`: at least two co-comment events within 30 minutes",
        "- `weak_temporal_overlap`: same thread overlap, but no short-window synchrony",
        "- `no_temporal_sync`: no same-thread overlap in the local comments file",
        "",
        "## Group Summary",
        "",
        "| group_seed | pairs | strong | moderate | weak | no sync |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.group_seed} | {int(row.total_pairs)} | {int(row.strong_temporal_sync)} | "
            f"{int(row.moderate_temporal_sync)} | {int(row.weak_temporal_overlap)} | "
            f"{int(row.no_temporal_sync)} |"
        )

    lines.extend(["", "## Key Pairs", ""])
    key_pairs = pair_rows.loc[
        pair_rows["verification_label"].isin(["strong_temporal_sync", "moderate_temporal_sync"])
    ].copy()
    if key_pairs.empty:
        lines.append("No strong or moderate temporal synchrony pairs found.")
    else:
        key_pairs = key_pairs.sort_values(
            ["group_seed", "verification_label", "within_5min_count", "within_30min_count"],
            ascending=[True, True, False, False],
        )
        for row in key_pairs.itertuples(index=False):
            median = "" if pd.isna(row.median_delay_minutes) else f"{row.median_delay_minutes:.1f}"
            lines.append(
                f"- {row.group_seed}: {row.account_a} <-> {row.account_b} | "
                f"{row.verification_label} | same_post={int(row.same_post_count)} | "
                f"<5min={int(row.within_5min_count)} | <30min={int(row.within_30min_count)} | "
                f"median_delay={median}min | co_neg={row.co_negative_weight:.3f}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seed_names(args.seed_dir, args.seeds)
    group_members = {seed: load_group_members(args.seed_dir, seed) for seed in seeds}
    all_authors = {member for members in group_members.values() for member in members}
    comments = load_comments(args.comments_path, all_authors, args.max_comments_per_author)
    post_times = account_post_times(comments)

    rows: list[dict[str, object]] = []
    for seed, members in group_members.items():
        co_neg_weights = load_co_negative_weights(args.seed_dir, seed)
        for account_a, account_b in itertools.combinations(sorted(members), 2):
            metrics = pair_temporal_metrics(
                post_times.get(account_a, {}),
                post_times.get(account_b, {}),
                strong_window=args.window_strong_minutes,
                moderate_window=args.window_moderate_minutes,
            )
            label = label_pair(metrics)
            rows.append(
                {
                    "group_seed": seed,
                    "account_a": account_a,
                    "account_b": account_b,
                    "co_negative_weight": co_neg_weights.get(tuple(sorted((account_a, account_b))), 0.0),
                    **metrics,
                    "text_fingerprint_distance": np.nan,
                    "account_lifecycle_overlap": np.nan,
                    "verification_label": label,
                }
            )

    pair_rows = pd.DataFrame(rows)
    label_counts = (
        pair_rows.pivot_table(
            index="group_seed",
            columns="verification_label",
            values="account_a",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in [
        "strong_temporal_sync",
        "moderate_temporal_sync",
        "weak_temporal_overlap",
        "no_temporal_sync",
    ]:
        if col not in label_counts.columns:
            label_counts[col] = 0
    label_counts["total_pairs"] = label_counts[
        [
            "strong_temporal_sync",
            "moderate_temporal_sync",
            "weak_temporal_overlap",
            "no_temporal_sync",
        ]
    ].sum(axis=1)
    summary = label_counts[
        [
            "group_seed",
            "total_pairs",
            "strong_temporal_sync",
            "moderate_temporal_sync",
            "weak_temporal_overlap",
            "no_temporal_sync",
        ]
    ]

    pair_path = args.output_dir / "stage2_verification_evidence.csv"
    summary_path = args.output_dir / "stage2_group_summary.csv"
    report_path = args.output_dir / "stage2_temporal_verification_report.md"
    pair_rows.to_csv(pair_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_markdown(pair_rows, summary), encoding="utf-8")

    print(f"Stage 2 temporal verification written to {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
