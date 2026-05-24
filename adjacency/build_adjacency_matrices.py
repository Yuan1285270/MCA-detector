#!/usr/bin/env python3
"""Build account-level adjacency graph artifacts from analyzed Reddit data.

The output is intentionally sparse: edge-list CSV files plus compressed COO-style
NPZ matrices. A dense account-by-account matrix would be too large for the full
dataset and is not needed for graph algorithms.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INVALID_AUTHORS = {"", "nan", "none", "[deleted]", "[removed]", "deleted", "removed"}
COMMENT_LABELS = ("supportive", "oppositional", "neutral", "mixed", "unclear")

# Tag similarity intentionally excludes analytical_neutral: the graph should
# connect accounts with similar mobilization rhetoric, not merely similar lack
# of rhetoric.
TAG_RATIO_COLUMNS = (
    "rhetoric_tag_authority_claim_ratio",
    "rhetoric_tag_bandwagon_ratio",
    "rhetoric_tag_call_to_action_ratio",
    "rhetoric_tag_emotional_amplification_ratio",
    "rhetoric_tag_fear_ratio",
    "rhetoric_tag_overconfidence_ratio",
    "rhetoric_tag_urgency_ratio",
    "rhetoric_tag_us_vs_them_ratio",
)
TAG_COUNT_COLUMNS = tuple(col.replace("_ratio", "_count") for col in TAG_RATIO_COLUMNS)


@dataclass(frozen=True)
class BuildConfig:
    comments_path: str
    posts_path: str
    account_features_path: str
    output_dir: str
    include_self_loops: bool
    skip_tag_similarity: bool
    tag_min_analyzed_posts: int
    tag_min_nonneutral_tags: int
    tag_top_k: int
    tag_threshold: float
    co_target_top_k: int
    co_target_threshold: float
    co_negative_target_threshold: float
    co_target_min_shared_targets: int
    co_target_max_sources_per_target: int
    restrict_tag_nodes_to_interaction_graph: bool
    include_manipulation_intensity: bool


def parse_args() -> argparse.Namespace:
    """Define CLI options so the module can be rerun with different thresholds."""
    parser = argparse.ArgumentParser(
        description="Build single-graph and multi-graph account adjacency artifacts."
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("llm/Export/reddit_comments_analyzed.csv.gz"),
        help="Analyzed comment feedback CSV or CSV.GZ.",
    )
    parser.add_argument(
        "--posts-path",
        type=Path,
        default=Path("llm/Export/reddit_posts_analyzed.csv.gz"),
        help="Analyzed posts CSV or CSV.GZ, used for trigger-response frequency.",
    )
    parser.add_argument(
        "--account-features-path",
        type=Path,
        default=Path("Archive/export_working_files/account_feature_matrix.csv"),
        help="Account-level feature matrix used for tag similarity.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("adjacency/output"),
        help="Directory for generated graph artifacts.",
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Keep source_author == target_author edges. Default removes them.",
    )
    parser.add_argument(
        "--skip-tag-similarity",
        action="store_true",
        help="Skip the content/tag similarity layer.",
    )
    parser.add_argument(
        "--tag-min-analyzed-posts",
        type=int,
        default=2,
        help="Minimum analyzed posts required for an account to enter tag similarity.",
    )
    parser.add_argument(
        "--tag-min-nonneutral-tags",
        type=int,
        default=2,
        help="Minimum non-neutral rhetoric tag count required for tag similarity.",
    )
    parser.add_argument(
        "--tag-top-k",
        type=int,
        default=10,
        help="Maximum tag-similarity neighbors retained per account.",
    )
    parser.add_argument(
        "--tag-threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity retained for tag-similarity edges.",
    )
    parser.add_argument(
        "--co-target-top-k",
        type=int,
        default=25,
        help="Maximum co-target neighbors retained per account.",
    )
    parser.add_argument(
        "--co-target-threshold",
        type=float,
        default=0.15,
        help="Minimum cosine similarity retained for shared-target engagement edges.",
    )
    parser.add_argument(
        "--co-negative-target-threshold",
        type=float,
        default=0.20,
        help="Minimum cosine similarity retained for shared negative-target edges.",
    )
    parser.add_argument(
        "--co-target-min-shared-targets",
        type=int,
        default=2,
        help="Minimum number of shared targets required for a co-target edge.",
    )
    parser.add_argument(
        "--co-target-max-sources-per-target",
        type=int,
        default=200,
        help="Per target, only keep the strongest source accounts before projection.",
    )
    parser.add_argument(
        "--allow-tag-only-nodes",
        action="store_true",
        help="Allow tag-similarity nodes outside the interaction graph.",
    )
    parser.add_argument(
        "--tag-without-manipulation-intensity",
        action="store_true",
        help="Use tag ratios only, without avg_manipulative_rhetoric_score / 100.",
    )
    return parser.parse_args()


def normalize_author(series: pd.Series) -> pd.Series:
    """Normalize author identifiers before filtering and grouping."""
    return series.astype("string").str.strip()


def valid_author_mask(series: pd.Series) -> pd.Series:
    """Return rows with usable account ids, excluding deleted/removed users."""
    normalized = normalize_author(series)
    return normalized.notna() & ~normalized.str.lower().isin(INVALID_AUTHORS)


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create the output layout for root, single-graph, and multi-graph files."""
    dirs = {
        "root": output_dir,
        "single": output_dir / "single-graph",
        "multi": output_dir / "multi-graph",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_and_clean_comments(comments_path: Path, include_self_loops: bool) -> tuple[pd.DataFrame, dict]:
    """Load comment feedback and apply graph-level cleaning rules."""
    usecols = [
        "post_id",
        "source_author",
        "target_author",
        "created_utc",
        "feedback_label",
        "feedback_score",
        "edge_weight",
    ]
    raw = pd.read_csv(comments_path, usecols=usecols, low_memory=False)
    original_rows = len(raw)

    raw["source_author"] = normalize_author(raw["source_author"])
    raw["target_author"] = normalize_author(raw["target_author"])
    raw["post_id"] = raw["post_id"].astype("string").str.strip()
    raw["created_utc"] = pd.to_numeric(raw["created_utc"], errors="coerce")
    mask = valid_author_mask(raw["source_author"]) & valid_author_mask(raw["target_author"])
    invalid_author_rows = int((~mask).sum())
    df = raw.loc[mask].copy()

    # A self-loop means an account commented on its own post. The default graph
    # focuses on account-to-account relations, so these are removed unless the
    # caller explicitly asks to keep them.
    self_loop_rows = int((df["source_author"] == df["target_author"]).sum())
    if not include_self_loops:
        df = df.loc[df["source_author"] != df["target_author"]].copy()

    # Older intermediate files may have either feedback_score or edge_weight.
    # Reconstruct one from the other so the graph formulas stay consistent.
    df["feedback_score"] = pd.to_numeric(df["feedback_score"], errors="coerce")
    df["edge_weight"] = pd.to_numeric(df["edge_weight"], errors="coerce")
    missing_score = df["feedback_score"].isna()
    if missing_score.any():
        df.loc[missing_score, "feedback_score"] = df.loc[missing_score, "edge_weight"] * 100
    missing_weight = df["edge_weight"].isna()
    if missing_weight.any():
        df.loc[missing_weight, "edge_weight"] = df.loc[missing_weight, "feedback_score"] / 100
    df["feedback_score"] = df["feedback_score"].fillna(0.0)
    df["edge_weight"] = df["edge_weight"].fillna(df["feedback_score"] / 100)
    df["feedback_label"] = (
        df["feedback_label"].astype("string").str.strip().str.lower().fillna("unclear")
    )
    df.loc[~df["feedback_label"].isin(COMMENT_LABELS), "feedback_label"] = "unclear"

    summary = {
        "original_comment_rows": original_rows,
        "invalid_author_rows_removed": invalid_author_rows,
        "self_loop_rows_found": self_loop_rows,
        "self_loop_rows_removed": 0 if include_self_loops else self_loop_rows,
        "clean_comment_rows": len(df),
        "feedback_label_counts": {
            label: int(count) for label, count in df["feedback_label"].value_counts().sort_index().items()
        },
    }
    return df, summary


def load_and_clean_posts(posts_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load analyzed posts so post-author frequency can be used as a denominator."""
    usecols = ["post_id", "author", "created_utc"]
    raw = pd.read_csv(posts_path, usecols=usecols, low_memory=False)
    original_rows = len(raw)

    raw["post_id"] = raw["post_id"].astype("string").str.strip()
    raw["author"] = normalize_author(raw["author"])
    raw["created_utc"] = pd.to_numeric(raw["created_utc"], errors="coerce")
    mask = (
        raw["post_id"].notna()
        & raw["post_id"].ne("")
        & valid_author_mask(raw["author"])
        & raw["created_utc"].notna()
    )
    posts = raw.loc[mask].copy().rename(
        columns={"author": "post_author", "created_utc": "post_created_utc"}
    )
    posts = posts.drop_duplicates(subset=["post_id"], keep="first")
    summary = {
        "original_post_rows": original_rows,
        "clean_post_rows": int(len(posts)),
        "post_author_count": int(posts["post_author"].nunique()),
    }
    return posts, summary


def build_interaction_edges(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregate comment rows into directed account-pair edge statistics."""
    group_cols = ["source_author", "target_author"]

    # One edge represents all direct comments from source_author to target_author.
    edge_stats = (
        comments.groupby(group_cols, sort=True)
        .agg(
            n_comments=("feedback_score", "size"),
            mean_feedback_score=("feedback_score", "mean"),
            std_feedback_score=("feedback_score", "std"),
            mean_edge_weight=("edge_weight", "mean"),
            min_feedback_score=("feedback_score", "min"),
            max_feedback_score=("feedback_score", "max"),
        )
        .reset_index()
    )
    edge_stats["std_feedback_score"] = edge_stats["std_feedback_score"].fillna(0.0)

    label_counts = (
        comments.pivot_table(
            index=group_cols,
            columns="feedback_label",
            values="feedback_score",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in COMMENT_LABELS:
        if label not in label_counts.columns:
            label_counts[label] = 0
    label_total = label_counts[list(COMMENT_LABELS)].sum(axis=1).replace(0, np.nan)
    for label in COMMENT_LABELS:
        label_counts[f"{label}_ratio"] = label_counts[label] / label_total
    label_counts = label_counts.rename(
        columns={label: f"{label}_count" for label in COMMENT_LABELS}
    )

    edge_stats = edge_stats.merge(label_counts, on=group_cols, how="left")
    count_cols = [f"{label}_count" for label in COMMENT_LABELS]
    ratio_cols = [f"{label}_ratio" for label in COMMENT_LABELS]
    edge_stats[count_cols] = edge_stats[count_cols].fillna(0).astype(int)
    edge_stats[ratio_cols] = edge_stats[ratio_cols].fillna(0.0)

    # Single/signed graph weight: stance direction from mean feedback, repeated
    # interaction strength from log frequency.
    edge_stats["weight_count"] = edge_stats["n_comments"].astype(float)
    edge_stats["weight_target_engagement_profile"] = np.log1p(edge_stats["n_comments"])
    edge_stats["weight_signed"] = edge_stats["mean_edge_weight"] * np.log1p(
        edge_stats["n_comments"]
    )

    # Positive and negative layers separate reinforcement from opposition so
    # later analysis does not treat all edges as the same social relation.
    edge_stats["weight_positive"] = edge_stats["weight_signed"].clip(lower=0.0)
    edge_stats["weight_negative"] = (-edge_stats["weight_signed"]).clip(lower=0.0)
    edge_stats["weight_target_negative_profile"] = np.log1p(edge_stats["oppositional_count"])

    # Zaman-inspired adjustment: low-degree interactions carry less evidence,
    # while high-activity sources and high-attention targets retain more weight.
    out_degree = comments.groupby("source_author").size()
    in_degree = comments.groupby("target_author").size()
    alpha_out = float(np.percentile(out_degree.to_numpy(dtype=float), 99))
    alpha_in = float(np.percentile(in_degree.to_numpy(dtype=float), 99))
    edge_stats["source_out_degree"] = edge_stats["source_author"].map(out_degree).astype(float)
    edge_stats["target_in_degree"] = edge_stats["target_author"].map(in_degree).astype(float)
    exponent = (
        alpha_out / edge_stats["source_out_degree"].replace(0, np.nan)
        + alpha_in / edge_stats["target_in_degree"].replace(0, np.nan)
        - 2
    ).fillna(50.0)
    edge_stats["degree_adjustment_denominator"] = 1 + np.exp(np.clip(exponent, -50, 50))
    edge_stats["weight_degree_adjusted"] = (
        edge_stats["n_comments"] / edge_stats["degree_adjustment_denominator"]
    )

    summary = {
        "directed_pair_count": len(edge_stats),
        "source_node_count": int(comments["source_author"].nunique()),
        "target_node_count": int(comments["target_author"].nunique()),
        "interaction_node_count": int(
            len(set(comments["source_author"]) | set(comments["target_author"]))
        ),
        "degree_adjustment_alpha_out_99p": alpha_out,
        "degree_adjustment_alpha_in_99p": alpha_in,
    }
    return edge_stats, summary


def build_node_index(edge_stats: pd.DataFrame) -> pd.DataFrame:
    """Create a stable account-to-node-id mapping shared by all graph layers."""
    nodes = sorted(set(edge_stats["source_author"]) | set(edge_stats["target_author"]))
    return pd.DataFrame({"node_id": np.arange(len(nodes), dtype=np.int64), "user_id": nodes})


def build_trigger_response_edges(comments: pd.DataFrame, posts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build post-author -> responder edges for repeated response frequency.

    The direction is intentionally reversed from comment interaction edges:
    source_author is the post author whose posts act as triggers, and
    target_author is the account that repeatedly responds.
    """
    post_counts = posts.groupby("post_author").size().rename("a_total_posts")
    post_times = posts.set_index("post_id")["post_created_utc"]

    working = comments.loc[
        comments["post_id"].notna()
        & comments["post_id"].ne("")
        & comments["created_utc"].notna()
    ].copy()
    working["post_created_utc"] = working["post_id"].map(post_times)
    working = working.loc[working["target_author"].isin(post_counts.index)].copy()

    if working.empty:
        empty = pd.DataFrame(
            columns=[
                "source_author",
                "target_author",
                "a_total_posts",
                "posts_with_b_comment",
                "b_total_comments_on_a_posts",
                "response_coverage",
                "comments_per_triggered_post",
                "median_response_delay_minutes",
                "p90_response_delay_minutes",
                "weight_response_coverage",
                "weight_trigger_response",
            ]
        )
        return empty, {"trigger_response_edge_count": 0}

    per_post_pair = (
        working.groupby(["target_author", "source_author", "post_id"], sort=True)
        .agg(
            first_comment_utc=("created_utc", "min"),
            comments_on_post=("created_utc", "size"),
            post_created_utc=("post_created_utc", "first"),
        )
        .reset_index()
    )
    per_post_pair["response_delay_minutes"] = (
        (per_post_pair["first_comment_utc"] - per_post_pair["post_created_utc"]) / 60.0
    )
    per_post_pair.loc[per_post_pair["response_delay_minutes"] < 0, "response_delay_minutes"] = np.nan

    edge_stats = (
        per_post_pair.groupby(["target_author", "source_author"], sort=True)
        .agg(
            posts_with_b_comment=("post_id", "nunique"),
            b_total_comments_on_a_posts=("comments_on_post", "sum"),
            median_response_delay_minutes=("response_delay_minutes", "median"),
            p90_response_delay_minutes=("response_delay_minutes", lambda value: value.quantile(0.9)),
        )
        .reset_index()
        .rename(columns={"target_author": "source_author", "source_author": "target_author"})
    )
    edge_stats["a_total_posts"] = edge_stats["source_author"].map(post_counts).astype(float)
    edge_stats["response_coverage"] = (
        edge_stats["posts_with_b_comment"] / edge_stats["a_total_posts"].replace(0, np.nan)
    ).fillna(0.0)
    edge_stats["comments_per_triggered_post"] = (
        edge_stats["b_total_comments_on_a_posts"]
        / edge_stats["posts_with_b_comment"].replace(0, np.nan)
    ).fillna(0.0)
    edge_stats["weight_response_coverage"] = edge_stats["response_coverage"]
    edge_stats["weight_trigger_response"] = (
        edge_stats["response_coverage"]
        * np.log1p(edge_stats["posts_with_b_comment"])
        * np.log1p(edge_stats["b_total_comments_on_a_posts"])
    )

    label_counts = (
        working.pivot_table(
            index=["target_author", "source_author"],
            columns="feedback_label",
            values="feedback_score",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"target_author": "source_author", "source_author": "target_author"})
    )
    for label in COMMENT_LABELS:
        if label not in label_counts.columns:
            label_counts[label] = 0
    label_counts = label_counts.rename(
        columns={label: f"{label}_count" for label in COMMENT_LABELS}
    )
    edge_stats = edge_stats.merge(label_counts, on=["source_author", "target_author"], how="left")
    count_cols = [f"{label}_count" for label in COMMENT_LABELS]
    edge_stats[count_cols] = edge_stats[count_cols].fillna(0).astype(int)
    label_total = edge_stats[count_cols].sum(axis=1).replace(0, np.nan)
    for label in COMMENT_LABELS:
        edge_stats[f"{label}_ratio"] = (edge_stats[f"{label}_count"] / label_total).fillna(0.0)

    edge_stats = edge_stats.sort_values(
        ["weight_trigger_response", "response_coverage", "posts_with_b_comment"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary = {
        "trigger_response_edge_count": int(len(edge_stats)),
        "trigger_response_post_author_count": int(edge_stats["source_author"].nunique()),
        "trigger_response_responder_count": int(edge_stats["target_author"].nunique()),
    }
    return edge_stats, summary


def build_co_target_edges(
    edge_stats: pd.DataFrame,
    *,
    profile_weight_col: str,
    output_weight_col: str,
    threshold: float,
    top_k: int,
    min_shared_targets: int,
    max_sources_per_target: int,
) -> tuple[pd.DataFrame, dict]:
    """Project commenter->target profiles into an undirected co-target graph.

    Two source accounts are connected when they repeatedly point at the same
    target authors. Cosine similarity keeps the projection from simply ranking
    the most active commenters highest.
    """
    profile = edge_stats.loc[
        edge_stats[profile_weight_col] > 0,
        ["source_author", "target_author", profile_weight_col],
    ].copy()
    if profile.empty:
        empty = pd.DataFrame(
            columns=[
                "source_author",
                "target_author",
                "shared_target_count",
                output_weight_col,
            ]
        )
        return empty, {
            "edge_count": 0,
            "candidate_profile_edge_count": 0,
            "candidate_source_count": 0,
            "candidate_target_count": 0,
            "threshold": threshold,
            "top_k": top_k,
            "min_shared_targets": min_shared_targets,
            "max_sources_per_target": max_sources_per_target,
        }

    profile = profile.sort_values(
        ["target_author", profile_weight_col, "source_author"],
        ascending=[True, False, True],
    )
    if max_sources_per_target > 0:
        profile = profile.groupby("target_author", sort=False).head(max_sources_per_target).copy()

    source_norm = (
        profile.groupby("source_author")[profile_weight_col]
        .apply(lambda values: float(np.sqrt(np.square(values).sum())))
        .to_dict()
    )
    pair_dot: dict[tuple[str, str], float] = {}
    pair_shared: dict[tuple[str, str], int] = {}

    for _, target_edges in profile.groupby("target_author", sort=False):
        if len(target_edges) < 2:
            continue
        sources = target_edges["source_author"].astype(str).to_numpy()
        weights = target_edges[profile_weight_col].to_numpy(dtype=np.float64)
        order = np.argsort(sources)
        sources = sources[order]
        weights = weights[order]
        for left_idx in range(len(sources) - 1):
            left = sources[left_idx]
            left_weight = weights[left_idx]
            for right_idx in range(left_idx + 1, len(sources)):
                key = (left, sources[right_idx])
                pair_dot[key] = pair_dot.get(key, 0.0) + float(left_weight * weights[right_idx])
                pair_shared[key] = pair_shared.get(key, 0) + 1

    records = []
    for (source, target), dot_value in pair_dot.items():
        shared_target_count = pair_shared[(source, target)]
        if shared_target_count < min_shared_targets:
            continue
        denominator = source_norm.get(source, 0.0) * source_norm.get(target, 0.0)
        if denominator <= 0:
            continue
        similarity = float(np.clip(dot_value / denominator, 0.0, 1.0))
        if similarity >= threshold:
            records.append(
                {
                    "source_author": source,
                    "target_author": target,
                    "shared_target_count": shared_target_count,
                    output_weight_col: similarity,
                }
            )

    co_edges = pd.DataFrame(records)
    if co_edges.empty:
        co_edges = pd.DataFrame(
            columns=[
                "source_author",
                "target_author",
                "shared_target_count",
                output_weight_col,
            ]
        )
    else:
        co_edges = co_edges.sort_values(
            [output_weight_col, "shared_target_count", "source_author", "target_author"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        if top_k > 0:
            left_ranked = co_edges[["source_author", "target_author", output_weight_col]].rename(
                columns={"source_author": "account", "target_author": "neighbor"}
            )
            right_ranked = co_edges[["target_author", "source_author", output_weight_col]].rename(
                columns={"target_author": "account", "source_author": "neighbor"}
            )
            ranked = (
                pd.concat([left_ranked, right_ranked], ignore_index=True)
                .sort_values(["account", output_weight_col, "neighbor"], ascending=[True, False, True])
                .groupby("account", sort=False)
                .head(top_k)
            )
            keep_pairs = {
                tuple(sorted((str(row.account), str(row.neighbor))))
                for row in ranked.itertuples(index=False)
            }
            co_edges["_pair"] = [
                tuple(sorted((str(source), str(target))))
                for source, target in zip(co_edges["source_author"], co_edges["target_author"], strict=True)
            ]
            co_edges = co_edges.loc[co_edges["_pair"].isin(keep_pairs)].drop(columns="_pair")
            co_edges = co_edges.sort_values(
                [output_weight_col, "shared_target_count", "source_author", "target_author"],
                ascending=[False, False, True, True],
            ).reset_index(drop=True)

    summary = {
        "edge_count": int(len(co_edges)),
        "candidate_profile_edge_count": int(len(profile)),
        "candidate_source_count": int(profile["source_author"].nunique()),
        "candidate_target_count": int(profile["target_author"].nunique()),
        "threshold": threshold,
        "top_k": top_k,
        "min_shared_targets": min_shared_targets,
        "max_sources_per_target": max_sources_per_target,
    }
    return co_edges, summary


def save_sparse_npz(
    path: Path,
    edges: pd.DataFrame,
    node_lookup: dict[str, int],
    weight_col: str,
    source_col: str = "source_author",
    target_col: str = "target_author",
    mirror: bool = False,
) -> dict:
    """Save an edge list as compressed COO arrays instead of a dense matrix."""
    if edges.empty:
        row = np.array([], dtype=np.int64)
        col = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float64)
    else:
        row = edges[source_col].map(node_lookup).to_numpy(dtype=np.int64)
        col = edges[target_col].map(node_lookup).to_numpy(dtype=np.int64)
        data = edges[weight_col].to_numpy(dtype=np.float64)
        if mirror:
            # Symmetric graph layers store each edge once in CSV and both
            # directions in the sparse matrix.
            row = np.concatenate([row, col])
            col = np.concatenate([col, row[: len(col)]])
            data = np.concatenate([data, data])

    shape = np.array([len(node_lookup), len(node_lookup)], dtype=np.int64)
    np.savez_compressed(path, row=row, col=col, data=data, shape=shape)
    return {"matrix_path": str(path), "nnz": int(len(data)), "shape": shape.tolist()}


def write_edge_artifacts(
    edge_stats: pd.DataFrame,
    trigger_response_edges: pd.DataFrame,
    co_target_edges: pd.DataFrame,
    co_negative_target_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict:
    """Write all interaction graph layers and their sparse matrix versions."""
    node_lookup = dict(zip(nodes["user_id"], nodes["node_id"], strict=True))
    nodes.to_csv(dirs["root"] / "nodes.csv", index=False)

    # Each tuple defines one graph layer: output folder, edge CSV, matrix file,
    # weight column, and whether the matrix should be mirrored.
    graph_specs = [
        (
            "single_signed",
            dirs["single"],
            "edges_single_signed.csv",
            "matrix_single_signed.npz",
            "weight_signed",
            edge_stats,
            False,
        ),
        (
            "count",
            dirs["multi"],
            "edges_count.csv",
            "matrix_count.npz",
            "weight_count",
            edge_stats,
            False,
        ),
        (
            "signed",
            dirs["multi"],
            "edges_signed.csv",
            "matrix_signed.npz",
            "weight_signed",
            edge_stats,
            False,
        ),
        (
            "positive",
            dirs["multi"],
            "edges_positive.csv",
            "matrix_positive.npz",
            "weight_positive",
            edge_stats.loc[edge_stats["weight_positive"] > 0].copy(),
            False,
        ),
        (
            "negative",
            dirs["multi"],
            "edges_negative.csv",
            "matrix_negative.npz",
            "weight_negative",
            edge_stats.loc[edge_stats["weight_negative"] > 0].copy(),
            False,
        ),
        (
            "degree_adjusted",
            dirs["multi"],
            "edges_degree_adjusted.csv",
            "matrix_degree_adjusted.npz",
            "weight_degree_adjusted",
            edge_stats,
            False,
        ),
        (
            "trigger_response",
            dirs["multi"],
            "edges_trigger_response.csv",
            "matrix_trigger_response.npz",
            "weight_trigger_response",
            trigger_response_edges,
            False,
        ),
        (
            "co_target",
            dirs["multi"],
            "edges_co_target.csv",
            "matrix_co_target.npz",
            "weight_co_target",
            co_target_edges,
            True,
        ),
        (
            "co_negative_target",
            dirs["multi"],
            "edges_co_negative_target.csv",
            "matrix_co_negative_target.npz",
            "weight_co_negative_target",
            co_negative_target_edges,
            True,
        ),
    ]

    summary = {}
    for name, directory, edge_name, matrix_name, weight_col, graph_edges, mirror in graph_specs:
        output_edges = graph_edges.loc[graph_edges[weight_col] != 0].copy()
        output_edges.to_csv(directory / edge_name, index=False)
        matrix_summary = save_sparse_npz(
            directory / matrix_name,
            output_edges,
            node_lookup,
            weight_col=weight_col,
            mirror=mirror,
        )
        summary[name] = {
            "edge_list_path": str(directory / edge_name),
            "edge_count": int(len(output_edges)),
            **matrix_summary,
        }

    edge_stats.to_csv(dirs["root"] / "all_interaction_edge_stats.csv", index=False)
    summary["all_interaction_edge_stats_path"] = str(dirs["root"] / "all_interaction_edge_stats.csv")
    summary["nodes_path"] = str(dirs["root"] / "nodes.csv")
    summary["node_count"] = int(len(nodes))
    return summary


def tag_feature_columns(include_manipulation_intensity: bool) -> list[str]:
    """Return the account-level feature vector used for cosine similarity."""
    columns = list(TAG_RATIO_COLUMNS)
    if include_manipulation_intensity:
        columns.append("normalized_avg_manipulative_rhetoric_score")
    return columns


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    """Fail early when an expected input column is unavailable."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def build_tag_similarity_edges(
    account_features_path: Path,
    nodes: pd.DataFrame,
    dirs: dict[str, Path],
    *,
    min_analyzed_posts: int,
    min_nonneutral_tags: int,
    top_k: int,
    threshold: float,
    restrict_to_interaction_graph: bool,
    include_manipulation_intensity: bool,
) -> dict:
    """Build an undirected content-similarity graph from account rhetoric profiles."""
    required = ["user_id", "analyzed_post_count", *TAG_RATIO_COLUMNS, *TAG_COUNT_COLUMNS]
    if include_manipulation_intensity:
        required.append("avg_manipulative_rhetoric_score")
    features = pd.read_csv(account_features_path, low_memory=False)
    require_columns(features, required, account_features_path)

    features["user_id"] = normalize_author(features["user_id"])
    features = features.loc[valid_author_mask(features["user_id"])].copy()
    features["analyzed_post_count"] = pd.to_numeric(
        features["analyzed_post_count"], errors="coerce"
    ).fillna(0)
    for column in (*TAG_RATIO_COLUMNS, *TAG_COUNT_COLUMNS):
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    features["nonneutral_tag_total"] = features[list(TAG_COUNT_COLUMNS)].sum(axis=1)

    if include_manipulation_intensity:
        features["normalized_avg_manipulative_rhetoric_score"] = (
            pd.to_numeric(features["avg_manipulative_rhetoric_score"], errors="coerce").fillna(0.0)
            / 100.0
        )

    interaction_nodes = set(nodes["user_id"])
    before_restrict = len(features)
    if restrict_to_interaction_graph:
        features = features.loc[features["user_id"].isin(interaction_nodes)].copy()

    # The tag layer is intentionally conservative: one analyzed post is too thin
    # to claim a stable account-level rhetoric profile.
    candidate_mask = (
        (features["analyzed_post_count"] >= min_analyzed_posts)
        & (features["nonneutral_tag_total"] >= min_nonneutral_tags)
    )
    candidates = features.loc[candidate_mask].copy().sort_values("user_id").reset_index(drop=True)
    feature_cols = tag_feature_columns(include_manipulation_intensity)
    matrix = candidates[feature_cols].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    candidates = candidates.loc[norms > 0].reset_index(drop=True)
    matrix = matrix[norms > 0]
    norms = norms[norms > 0]

    edge_records: list[dict] = []
    if len(candidates) > 1:
        # Cosine similarity compares rhetoric profile shape, not raw posting
        # volume. Top-k keeps the graph sparse even when many accounts are close.
        normalized = matrix / norms[:, None]
        similarity = normalized @ normalized.T
        for i in range(len(candidates)):
            row = similarity[i]
            candidate_idx = np.where(row >= threshold)[0]
            candidate_idx = candidate_idx[candidate_idx != i]
            if len(candidate_idx) == 0:
                continue
            ordered = sorted(
                candidate_idx,
                key=lambda j: (-float(row[j]), str(candidates.at[j, "user_id"])),
            )[:top_k]
            for j in ordered:
                left = str(candidates.at[i, "user_id"])
                right = str(candidates.at[j, "user_id"])
                source, target = sorted((left, right))
                edge_records.append(
                    {
                        "source_author": source,
                        "target_author": target,
                        "weight_tag_similarity": float(row[j]),
                    }
                )

    tag_edges = pd.DataFrame(edge_records)
    if tag_edges.empty:
        tag_edges = pd.DataFrame(
            columns=["source_author", "target_author", "weight_tag_similarity"]
        )
    else:
        # Multiple accounts can select each other as top-k neighbors; collapse
        # duplicates and keep the strongest observed similarity.
        tag_edges = (
            tag_edges.groupby(["source_author", "target_author"], as_index=False)
            .agg(weight_tag_similarity=("weight_tag_similarity", "max"))
            .sort_values(["source_author", "target_author"])
            .reset_index(drop=True)
        )

    metadata_cols = [
        "user_id",
        "analyzed_post_count",
        "nonneutral_tag_total",
        "avg_manipulative_rhetoric_score",
        *TAG_RATIO_COLUMNS,
    ]
    metadata_cols = [col for col in metadata_cols if col in candidates.columns]
    candidate_metadata = candidates[metadata_cols].copy()
    candidate_metadata.to_csv(dirs["multi"] / "tag_similarity_candidate_nodes.csv", index=False)
    tag_edges.to_csv(dirs["multi"] / "edges_tag_similarity.csv", index=False)

    node_lookup = dict(zip(nodes["user_id"], nodes["node_id"], strict=True))
    if not restrict_to_interaction_graph:
        # Expand node index only when the caller explicitly allows tag-only nodes.
        all_nodes = sorted(set(nodes["user_id"]) | set(tag_edges["source_author"]) | set(tag_edges["target_author"]))
        node_lookup = {node: idx for idx, node in enumerate(all_nodes)}

    matrix_summary = save_sparse_npz(
        dirs["multi"] / "matrix_tag_similarity.npz",
        tag_edges,
        node_lookup,
        weight_col="weight_tag_similarity",
        mirror=True,
    )
    return {
        "edge_list_path": str(dirs["multi"] / "edges_tag_similarity.csv"),
        "candidate_nodes_path": str(dirs["multi"] / "tag_similarity_candidate_nodes.csv"),
        "feature_columns": feature_cols,
        "account_feature_rows": int(before_restrict),
        "rows_after_interaction_node_restriction": int(len(features)),
        "candidate_node_count": int(len(candidates)),
        "edge_count": int(len(tag_edges)),
        "undirected_matrix": True,
        **matrix_summary,
    }


def write_json(path: Path, payload: dict) -> None:
    """Write a human-readable summary for audit and reproduction."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full adjacency build pipeline."""
    args = parse_args()
    config = BuildConfig(
        comments_path=str(args.comments_path),
        posts_path=str(args.posts_path),
        account_features_path=str(args.account_features_path),
        output_dir=str(args.output_dir),
        include_self_loops=args.include_self_loops,
        skip_tag_similarity=args.skip_tag_similarity,
        tag_min_analyzed_posts=args.tag_min_analyzed_posts,
        tag_min_nonneutral_tags=args.tag_min_nonneutral_tags,
        tag_top_k=args.tag_top_k,
        tag_threshold=args.tag_threshold,
        co_target_top_k=args.co_target_top_k,
        co_target_threshold=args.co_target_threshold,
        co_negative_target_threshold=args.co_negative_target_threshold,
        co_target_min_shared_targets=args.co_target_min_shared_targets,
        co_target_max_sources_per_target=args.co_target_max_sources_per_target,
        restrict_tag_nodes_to_interaction_graph=not args.allow_tag_only_nodes,
        include_manipulation_intensity=not args.tag_without_manipulation_intensity,
    )

    dirs = ensure_output_dirs(args.output_dir)
    comments, clean_summary = load_and_clean_comments(
        args.comments_path, include_self_loops=args.include_self_loops
    )
    posts, post_summary = load_and_clean_posts(args.posts_path)
    edge_stats, graph_summary = build_interaction_edges(comments)
    trigger_response_edges, trigger_response_summary = build_trigger_response_edges(comments, posts)
    co_target_edges, co_target_summary = build_co_target_edges(
        edge_stats,
        profile_weight_col="weight_target_engagement_profile",
        output_weight_col="weight_co_target",
        threshold=args.co_target_threshold,
        top_k=args.co_target_top_k,
        min_shared_targets=args.co_target_min_shared_targets,
        max_sources_per_target=args.co_target_max_sources_per_target,
    )
    co_negative_target_edges, co_negative_target_summary = build_co_target_edges(
        edge_stats,
        profile_weight_col="weight_target_negative_profile",
        output_weight_col="weight_co_negative_target",
        threshold=args.co_negative_target_threshold,
        top_k=args.co_target_top_k,
        min_shared_targets=args.co_target_min_shared_targets,
        max_sources_per_target=args.co_target_max_sources_per_target,
    )
    nodes = build_node_index(edge_stats)
    artifact_summary = write_edge_artifacts(
        edge_stats,
        trigger_response_edges,
        co_target_edges,
        co_negative_target_edges,
        nodes,
        dirs,
    )

    tag_summary = None
    if not args.skip_tag_similarity:
        tag_summary = build_tag_similarity_edges(
            args.account_features_path,
            nodes,
            dirs,
            min_analyzed_posts=args.tag_min_analyzed_posts,
            min_nonneutral_tags=args.tag_min_nonneutral_tags,
            top_k=args.tag_top_k,
            threshold=args.tag_threshold,
            restrict_to_interaction_graph=not args.allow_tag_only_nodes,
            include_manipulation_intensity=not args.tag_without_manipulation_intensity,
        )
        artifact_summary["tag_similarity"] = tag_summary

    summary = {
        "config": asdict(config),
        "cleaning": clean_summary,
        "posts": post_summary,
        "interaction_graph": graph_summary,
        "trigger_response_graph": trigger_response_summary,
        "co_target_graph": co_target_summary,
        "co_negative_target_graph": co_negative_target_summary,
        "artifacts": artifact_summary,
    }
    write_json(dirs["root"] / "summary.json", summary)

    print("Adjacency artifacts generated.")
    print(f"Output: {args.output_dir}")
    print(f"Clean rows: {clean_summary['clean_comment_rows']:,}")
    print(f"Nodes: {artifact_summary['node_count']:,}")
    print(f"Directed pairs: {graph_summary['directed_pair_count']:,}")
    print(
        "Trigger response: "
        f"{trigger_response_summary['trigger_response_edge_count']:,} directed frequency edges"
    )
    print(
        "Co-target: "
        f"{co_target_summary['edge_count']:,} shared-target edges; "
        f"{co_negative_target_summary['edge_count']:,} shared negative-target edges"
    )
    if tag_summary:
        print(
            "Tag similarity: "
            f"{tag_summary['candidate_node_count']:,} candidate nodes, "
            f"{tag_summary['edge_count']:,} undirected edges"
        )


if __name__ == "__main__":
    main()
