#!/usr/bin/env python3
"""Discover coordination groups and expand seed accounts from graph artifacts.

This script is intentionally evidence-oriented. It does not assign a final MCA
score; instead it turns adjacency layers into reviewable group tables.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRAPH_DIR = Path("adjacency/output")
DEFAULT_FEATURES_PATH = Path("Archive/export_working_files/account_feature_matrix.csv")
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output")

LAYER_FILES = {
    "co_negative_target": ("multi-graph/edges_co_negative_target.csv", "weight_co_negative_target"),
    "co_target": ("multi-graph/edges_co_target.csv", "weight_co_target"),
    "tag_similarity": ("multi-graph/edges_tag_similarity.csv", "weight_tag_similarity"),
}

NONNEUTRAL_TAG_COUNT_COLUMNS = [
    "rhetoric_tag_authority_claim_count",
    "rhetoric_tag_bandwagon_count",
    "rhetoric_tag_call_to_action_count",
    "rhetoric_tag_emotional_amplification_count",
    "rhetoric_tag_fear_count",
    "rhetoric_tag_overconfidence_count",
    "rhetoric_tag_urgency_count",
    "rhetoric_tag_us_vs_them_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seed expansion and group-discovery evidence tables."
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--group-layer",
        choices=["co_negative_target", "co_target"],
        default="co_negative_target",
        help="Undirected projection layer used to discover groups.",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.30,
        help="Minimum edge weight retained before connected-component discovery.",
    )
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=50,
        help="Exclude giant components from the review table.",
    )
    parser.add_argument("--top-groups", type=int, default=50)
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[],
        help="Optional seed accounts to expand into local coordination neighborhoods.",
    )
    parser.add_argument("--seed-top-neighbors", type=int, default=15)
    parser.add_argument("--tier-co-negative-threshold", type=float, default=0.20)
    parser.add_argument("--tier-tag-threshold", type=float, default=0.90)
    parser.add_argument("--tier-trigger-threshold", type=float, default=0.50)
    parser.add_argument("--tier-co-target-threshold", type=float, default=0.30)
    parser.add_argument(
        "--tier-two-hop-min-links",
        type=int,
        default=2,
        help="For 2-hop co-negative expansion, require links to this many accepted members.",
    )
    parser.add_argument(
        "--tier-max-members",
        type=int,
        default=100,
        help="Maximum tiered seed-expansion members retained per seed.",
    )
    return parser.parse_args()


def normalize_user_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "seed"


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "groups": output_dir / "groups",
        "seeds": output_dir / "seeds",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_layer(graph_dir: Path, layer: str) -> tuple[pd.DataFrame, str]:
    rel_path, weight_col = LAYER_FILES[layer]
    path = graph_dir / rel_path
    df = pd.read_csv(path, low_memory=False)
    for col in ("source_author", "target_author"):
        df[col] = normalize_user_id(df[col])
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    return df, weight_col


def read_negative_edges(graph_dir: Path) -> pd.DataFrame:
    path = graph_dir / "multi-graph" / "edges_negative.csv"
    cols = ["source_author", "target_author", "weight_negative", "oppositional_count"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    for col in ("source_author", "target_author"):
        df[col] = normalize_user_id(df[col])
    df["weight_negative"] = pd.to_numeric(df["weight_negative"], errors="coerce").fillna(0.0)
    df["oppositional_count"] = pd.to_numeric(
        df["oppositional_count"], errors="coerce"
    ).fillna(0.0)
    return df


def read_count_edges(graph_dir: Path) -> pd.DataFrame:
    path = graph_dir / "multi-graph" / "edges_count.csv"
    cols = ["source_author", "target_author", "n_comments", "weight_count"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    for col in ("source_author", "target_author"):
        df[col] = normalize_user_id(df[col])
    df["n_comments"] = pd.to_numeric(df["n_comments"], errors="coerce").fillna(0.0)
    df["weight_count"] = pd.to_numeric(df["weight_count"], errors="coerce").fillna(0.0)
    return df


def read_trigger_edges(graph_dir: Path) -> pd.DataFrame:
    path = graph_dir / "multi-graph" / "edges_trigger_response.csv"
    cols = [
        "source_author",
        "target_author",
        "weight_trigger_response",
        "response_coverage",
        "posts_with_b_comment",
        "b_total_comments_on_a_posts",
        "median_response_delay_minutes",
    ]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    for col in ("source_author", "target_author"):
        df[col] = normalize_user_id(df[col])
    numeric_cols = [col for col in cols if col not in {"source_author", "target_author"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def read_account_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, low_memory=False)
    features["user_id"] = normalize_user_id(features["user_id"])
    for col in [
        "avg_manipulative_rhetoric_score",
        "comment_label_oppositional_ratio",
        "anomaly_score",
        "anomaly_label",
        "comment_count",
        "analyzed_post_count",
        *NONNEUTRAL_TAG_COUNT_COLUMNS,
    ]:
        if col not in features.columns:
            features[col] = 0.0
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    nonneutral_count = features[NONNEUTRAL_TAG_COUNT_COLUMNS].sum(axis=1)
    features["non_neutral_post_ratio"] = (
        nonneutral_count / features["analyzed_post_count"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    return features.set_index("user_id", drop=False)


def connected_components(edges: pd.DataFrame) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges[["source_author", "target_author"]].itertuples(index=False):
        source = str(row.source_author)
        target = str(row.target_author)
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def summarize_members(features: pd.DataFrame, members: list[str]) -> dict:
    subset = features.reindex(members)
    anomaly_label = pd.to_numeric(subset["anomaly_label"], errors="coerce").fillna(0.0)
    return {
        "avg_rhetorical_score": float(subset["avg_manipulative_rhetoric_score"].fillna(0.0).mean()),
        "avg_non_neutral_post_ratio": float(subset["non_neutral_post_ratio"].fillna(0.0).mean()),
        "avg_oppositional_stance_ratio": float(
            subset["comment_label_oppositional_ratio"].fillna(0.0).mean()
        ),
        "automation_anomaly_fraction": float((anomaly_label == -1).mean()),
        "avg_anomaly_score": float(subset["anomaly_score"].fillna(0.0).mean()),
        "total_comment_count": float(subset["comment_count"].fillna(0.0).sum()),
        "total_analyzed_post_count": float(subset["analyzed_post_count"].fillna(0.0).sum()),
    }


def shared_targets_for_members(
    members: list[str],
    target_edges: pd.DataFrame,
    *,
    weight_col: str,
    min_members: int = 2,
    top_n: int = 10,
) -> pd.DataFrame:
    subset = target_edges.loc[target_edges["source_author"].isin(members)].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["target_author", "member_count", "edge_count", "target_weight_sum", "top_members"]
        )

    grouped = (
        subset.groupby("target_author")
        .agg(
            member_count=("source_author", "nunique"),
            edge_count=("source_author", "size"),
            target_weight_sum=(weight_col, "sum"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["member_count"] >= min_members].copy()
    if grouped.empty:
        return grouped.assign(top_members=pd.Series(dtype=str))

    top_members = []
    for target in grouped["target_author"]:
        members_for_target = (
            subset.loc[subset["target_author"].eq(target)]
            .sort_values(weight_col, ascending=False)["source_author"]
            .astype(str)
            .head(8)
            .tolist()
        )
        top_members.append(", ".join(members_for_target))
    grouped["top_members"] = top_members
    return grouped.sort_values(["member_count", "target_weight_sum"], ascending=False).head(top_n)


def build_group_discovery(
    *,
    layer_edges: pd.DataFrame,
    layer: str,
    weight_col: str,
    threshold: float,
    min_group_size: int,
    max_group_size: int,
    top_groups: int,
    target_edges: pd.DataFrame,
    target_weight_col: str,
    features: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = layer_edges.loc[layer_edges[weight_col] >= threshold].copy()
    components = connected_components(filtered)

    overview_rows = []
    member_rows = []
    internal_edge_frames = []
    shared_target_frames = []
    skipped_rows = []

    candidate_components = [
        component
        for component in components
        if min_group_size <= len(component) <= max_group_size
    ]

    for component in components:
        if len(component) > max_group_size:
            skipped_rows.append(
                {
                    "layer": layer,
                    "threshold": threshold,
                    "component_size": len(component),
                    "reason": "larger_than_max_group_size",
                }
            )

    sortable = []
    for component in candidate_components:
        members = set(component)
        internal = filtered.loc[
            filtered["source_author"].isin(members) & filtered["target_author"].isin(members)
        ].copy()
        if internal.empty:
            continue
        shared_targets = shared_targets_for_members(
            component,
            target_edges,
            weight_col=target_weight_col,
            min_members=2,
            top_n=10,
        )
        sortable.append(
            (
                float(internal[weight_col].sum()),
                int(len(shared_targets)),
                len(component),
                component,
                internal,
                shared_targets,
            )
        )

    sortable.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    sortable = sortable[:top_groups]

    for idx, (internal_sum, shared_count, size, component, internal, shared_targets) in enumerate(
        sortable, start=1
    ):
        group_id = f"{layer}_t{int(threshold * 100):03d}_{idx:03d}"
        density = 0.0
        if size > 1:
            density = 2 * len(internal) / (size * (size - 1))
        member_summary = summarize_members(features, component)
        overview_rows.append(
            {
                "group_id": group_id,
                "layer": layer,
                "threshold": threshold,
                "member_count": size,
                "internal_edge_count": int(len(internal)),
                "internal_density": density,
                "internal_weight_sum": internal_sum,
                "internal_weight_mean": float(internal[weight_col].mean()),
                "internal_weight_max": float(internal[weight_col].max()),
                "shared_target_count": shared_count,
                "top_shared_targets": "; ".join(
                    shared_targets["target_author"].astype(str).head(5).tolist()
                ),
                **member_summary,
            }
        )
        for member in component:
            member_feature = features.reindex([member]).iloc[0]
            member_rows.append(
                {
                    "group_id": group_id,
                    "user_id": member,
                    "avg_manipulative_rhetoric_score": member_feature[
                        "avg_manipulative_rhetoric_score"
                    ],
                    "non_neutral_post_ratio": member_feature["non_neutral_post_ratio"],
                    "oppositional_stance_ratio": member_feature[
                        "comment_label_oppositional_ratio"
                    ],
                    "anomaly_label": member_feature["anomaly_label"],
                    "anomaly_score": member_feature["anomaly_score"],
                    "comment_count": member_feature["comment_count"],
                    "analyzed_post_count": member_feature["analyzed_post_count"],
                }
            )
        internal = internal.assign(group_id=group_id, layer=layer)
        internal_edge_frames.append(internal)
        if not shared_targets.empty:
            shared_targets = shared_targets.assign(group_id=group_id, layer=layer)
            shared_target_frames.append(shared_targets)

    summary = pd.DataFrame(overview_rows)
    members = pd.DataFrame(member_rows)
    internal_edges = (
        pd.concat(internal_edge_frames, ignore_index=True)
        if internal_edge_frames
        else pd.DataFrame()
    )
    shared_targets = (
        pd.concat(shared_target_frames, ignore_index=True)
        if shared_target_frames
        else pd.DataFrame()
    )
    skipped = pd.DataFrame(skipped_rows)

    summary.to_csv(output_dir / "groups" / "group_summary.csv", index=False)
    members.to_csv(output_dir / "groups" / "group_members.csv", index=False)
    internal_edges.to_csv(output_dir / "groups" / "group_internal_edges.csv", index=False)
    shared_targets.to_csv(output_dir / "groups" / "group_shared_targets.csv", index=False)
    skipped.to_csv(output_dir / "groups" / "skipped_large_components.csv", index=False)
    return summary, members, internal_edges, shared_targets


def undirected_neighbors(df: pd.DataFrame, seed: str, weight_col: str, relation: str, top_n: int) -> pd.DataFrame:
    left = df.loc[df["source_author"].eq(seed), ["target_author", weight_col]].rename(
        columns={"target_author": "neighbor", weight_col: "weight"}
    )
    right = df.loc[df["target_author"].eq(seed), ["source_author", weight_col]].rename(
        columns={"source_author": "neighbor", weight_col: "weight"}
    )
    neighbors = pd.concat([left, right], ignore_index=True)
    if neighbors.empty:
        return pd.DataFrame(columns=["relation", "neighbor", "weight"])
    neighbors["relation"] = relation
    return neighbors.sort_values(["weight", "neighbor"], ascending=[False, True]).head(top_n)


def trigger_neighbors(trigger_edges: pd.DataFrame, seed: str, top_n: int) -> pd.DataFrame:
    incoming = trigger_edges.loc[trigger_edges["target_author"].eq(seed)].copy()
    incoming = incoming.sort_values("weight_trigger_response", ascending=False).head(top_n)
    incoming = incoming.rename(columns={"source_author": "neighbor"})
    incoming["relation"] = "trigger_in_post_author_to_seed"

    outgoing = trigger_edges.loc[trigger_edges["source_author"].eq(seed)].copy()
    outgoing = outgoing.sort_values("weight_trigger_response", ascending=False).head(top_n)
    outgoing = outgoing.rename(columns={"target_author": "neighbor"})
    outgoing["relation"] = "trigger_out_seed_to_responder"

    cols = [
        "relation",
        "neighbor",
        "weight_trigger_response",
        "response_coverage",
        "posts_with_b_comment",
        "b_total_comments_on_a_posts",
        "median_response_delay_minutes",
    ]
    return pd.concat([incoming[cols], outgoing[cols]], ignore_index=True)


def undirected_weight_map(df: pd.DataFrame, weight_col: str) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for row in df[["source_author", "target_author", weight_col]].itertuples(index=False):
        source = str(row.source_author)
        target = str(row.target_author)
        key = tuple(sorted((source, target)))
        weights[key] = max(weights.get(key, 0.0), float(getattr(row, weight_col)))
    return weights


def trigger_weight_map(trigger_edges: pd.DataFrame) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for row in trigger_edges[
        ["source_author", "target_author", "weight_trigger_response"]
    ].itertuples(index=False):
        source = str(row.source_author)
        target = str(row.target_author)
        key = tuple(sorted((source, target)))
        weights[key] = max(weights.get(key, 0.0), float(row.weight_trigger_response))
    return weights


def pair_weight(weights: dict[tuple[str, str], float], left: str, right: str) -> float:
    return weights.get(tuple(sorted((str(left), str(right)))), 0.0)


def tiered_seed_expansion(
    *,
    seed: str,
    co_negative: pd.DataFrame,
    co_target: pd.DataFrame,
    tag_similarity: pd.DataFrame,
    trigger_edges: pd.DataFrame,
    co_negative_threshold: float,
    tag_threshold: float,
    trigger_threshold: float,
    co_target_threshold: float,
    two_hop_min_links: int,
    max_members: int,
) -> pd.DataFrame:
    """Expand a seed with explicit evidence tiers instead of weighted sums."""
    co_neg_weights = undirected_weight_map(co_negative, "weight_co_negative_target")
    co_target_weights = undirected_weight_map(co_target, "weight_co_target")
    tag_weights = undirected_weight_map(tag_similarity, "weight_tag_similarity")
    trigger_weights = trigger_weight_map(trigger_edges)

    direct_candidates = set()
    for weights in (co_neg_weights, co_target_weights, tag_weights, trigger_weights):
        for left, right in weights:
            if left == seed:
                direct_candidates.add(right)
            elif right == seed:
                direct_candidates.add(left)

    accepted: dict[str, dict] = {
        seed: {
            "seed": seed,
            "candidate": seed,
            "tier": 0,
            "include": True,
            "include_reason": "seed",
            "co_negative_weight": 0.0,
            "tag_similarity_weight": 0.0,
            "trigger_response_weight": 0.0,
            "co_target_weight": 0.0,
            "two_hop_link_count": 0,
            "two_hop_connectors": "",
        }
    }
    evaluated_rows = []

    for candidate in sorted(direct_candidates):
        co_neg = pair_weight(co_neg_weights, seed, candidate)
        tag = pair_weight(tag_weights, seed, candidate)
        trigger = pair_weight(trigger_weights, seed, candidate)
        co_t = pair_weight(co_target_weights, seed, candidate)

        include = False
        tier = 99
        reason = "not_included"
        if co_neg >= co_negative_threshold:
            include = True
            tier = 1
            reason = "tier1_co_negative_direct"
        elif tag >= tag_threshold and (
            co_neg > 0 or co_t >= co_target_threshold or trigger >= trigger_threshold
        ):
            include = True
            tier = 2
            reason = "tier2_tag_similarity_with_structure"
        elif trigger >= trigger_threshold and (co_neg > 0 or tag >= tag_threshold):
            include = True
            tier = 3
            reason = "tier3_trigger_with_co_negative_or_tag"
        elif co_t >= co_target_threshold:
            reason = "support_only_co_target"
        elif tag >= tag_threshold:
            reason = "support_only_tag_similarity"
        elif trigger >= trigger_threshold:
            reason = "support_only_trigger_response"

        row = {
            "seed": seed,
            "candidate": candidate,
            "tier": tier,
            "include": include,
            "include_reason": reason,
            "co_negative_weight": co_neg,
            "tag_similarity_weight": tag,
            "trigger_response_weight": trigger,
            "co_target_weight": co_t,
            "two_hop_link_count": 0,
            "two_hop_connectors": "",
        }
        evaluated_rows.append(row)
        if include:
            accepted[candidate] = row

    # Conservative 2-hop expansion: walk only through co-negative edges, and
    # require the new candidate to link back to multiple already accepted users.
    second_hop_candidates: set[str] = set()
    accepted_direct = set(accepted)
    for left, right in co_neg_weights:
        if left in accepted_direct and right not in accepted_direct:
            second_hop_candidates.add(right)
        elif right in accepted_direct and left not in accepted_direct:
            second_hop_candidates.add(left)

    for candidate in sorted(second_hop_candidates):
        connectors = [
            member
            for member in accepted_direct
            if member != candidate and pair_weight(co_neg_weights, member, candidate) >= co_negative_threshold
        ]
        if len(connectors) < two_hop_min_links:
            continue
        co_neg_to_seed = pair_weight(co_neg_weights, seed, candidate)
        tag = pair_weight(tag_weights, seed, candidate)
        trigger = pair_weight(trigger_weights, seed, candidate)
        co_t = pair_weight(co_target_weights, seed, candidate)
        row = {
            "seed": seed,
            "candidate": candidate,
            "tier": 4,
            "include": True,
            "include_reason": "tier4_two_hop_co_negative_multi_link",
            "co_negative_weight": co_neg_to_seed,
            "tag_similarity_weight": tag,
            "trigger_response_weight": trigger,
            "co_target_weight": co_t,
            "two_hop_link_count": len(connectors),
            "two_hop_connectors": ", ".join(sorted(connectors)[:12]),
        }
        evaluated_rows.append(row)
        accepted[candidate] = row

    accepted_rows = [accepted[key] for key in accepted if key != seed]
    accepted_rows = sorted(
        accepted_rows,
        key=lambda row: (
            row["tier"],
            -row["co_negative_weight"],
            -row["tag_similarity_weight"],
            -row["trigger_response_weight"],
            -row["co_target_weight"],
            row["candidate"],
        ),
    )
    accepted_rows = [accepted[seed], *accepted_rows[: max(0, max_members - 1)]]

    output = pd.DataFrame(accepted_rows)
    if output.empty:
        return pd.DataFrame(
            columns=[
                "seed",
                "candidate",
                "tier",
                "include",
                "include_reason",
                "co_negative_weight",
                "tag_similarity_weight",
                "trigger_response_weight",
                "co_target_weight",
                "two_hop_link_count",
                "two_hop_connectors",
            ]
        )
    return output


def shared_targets_with_seed(
    seed: str,
    neighbors: list[str],
    target_edges: pd.DataFrame,
    weight_col: str,
) -> pd.DataFrame:
    seed_targets = set(
        target_edges.loc[target_edges["source_author"].eq(seed), "target_author"].astype(str)
    )
    rows = []
    for neighbor in neighbors:
        neighbor_edges = target_edges.loc[target_edges["source_author"].eq(neighbor)].copy()
        shared = neighbor_edges.loc[neighbor_edges["target_author"].astype(str).isin(seed_targets)]
        rows.append(
            {
                "seed": seed,
                "neighbor": neighbor,
                "shared_target_count": int(shared["target_author"].nunique()),
                "shared_target_weight_sum": float(shared[weight_col].sum()) if not shared.empty else 0.0,
                "shared_targets": ", ".join(
                    shared.sort_values(weight_col, ascending=False)["target_author"]
                    .astype(str)
                    .drop_duplicates()
                    .head(12)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["shared_target_count", "shared_target_weight_sum"], ascending=False
    )


def build_seed_expansions(
    *,
    seeds: list[str],
    co_target: pd.DataFrame,
    co_negative: pd.DataFrame,
    tag_similarity: pd.DataFrame,
    trigger_edges: pd.DataFrame,
    target_edges: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    top_neighbors: int,
    co_negative_threshold: float,
    tag_threshold: float,
    trigger_threshold: float,
    co_target_threshold: float,
    two_hop_min_links: int,
    max_members: int,
) -> pd.DataFrame:
    summary_rows = []
    for raw_seed in seeds:
        seed = str(raw_seed).strip()
        if not seed:
            continue
        filename_seed = safe_filename(seed)
        seed_dir = output_dir / "seeds" / filename_seed
        seed_dir.mkdir(parents=True, exist_ok=True)

        co_target_neighbors = undirected_neighbors(
            co_target, seed, "weight_co_target", "co_target", top_neighbors
        )
        co_negative_neighbors = undirected_neighbors(
            co_negative,
            seed,
            "weight_co_negative_target",
            "co_negative_target",
            top_neighbors,
        )
        trigger = trigger_neighbors(trigger_edges, seed, top_neighbors)
        tag_neighbors = undirected_neighbors(
            tag_similarity,
            seed,
            "weight_tag_similarity",
            "tag_similarity",
            top_neighbors,
        )

        coordination_neighbors = pd.concat(
            [co_target_neighbors, co_negative_neighbors, tag_neighbors],
            ignore_index=True,
            sort=False,
        )
        coordination_neighbors.to_csv(seed_dir / "coordination_neighbors.csv", index=False)
        trigger.to_csv(seed_dir / "trigger_neighbors.csv", index=False)

        tiered_members = tiered_seed_expansion(
            seed=seed,
            co_negative=co_negative,
            co_target=co_target,
            tag_similarity=tag_similarity,
            trigger_edges=trigger_edges,
            co_negative_threshold=co_negative_threshold,
            tag_threshold=tag_threshold,
            trigger_threshold=trigger_threshold,
            co_target_threshold=co_target_threshold,
            two_hop_min_links=two_hop_min_links,
            max_members=max_members,
        )
        tiered_members.to_csv(seed_dir / "tiered_expansion_members.csv", index=False)

        candidate_members = {seed}
        candidate_members.update(tiered_members["candidate"].dropna().astype(str).tolist())
        candidate_members_list = sorted(candidate_members)

        internal_co_target = co_target.loc[
            co_target["source_author"].isin(candidate_members)
            & co_target["target_author"].isin(candidate_members)
        ].copy()
        internal_co_target["layer"] = "co_target"
        internal_co_negative = co_negative.loc[
            co_negative["source_author"].isin(candidate_members)
            & co_negative["target_author"].isin(candidate_members)
        ].copy()
        internal_co_negative["layer"] = "co_negative_target"
        internal_edges = pd.concat([internal_co_target, internal_co_negative], ignore_index=True, sort=False)
        internal_edges.to_csv(seed_dir / "internal_coordination_edges.csv", index=False)

        shared_negative = shared_targets_with_seed(
            seed,
            coordination_neighbors["neighbor"].dropna().astype(str).drop_duplicates().tolist(),
            target_edges,
            "weight_negative",
        )
        shared_negative.to_csv(seed_dir / "shared_negative_targets_with_seed.csv", index=False)

        member_features = features.reindex(candidate_members_list).copy()
        member_features = member_features[
            [
                "user_id",
                "avg_manipulative_rhetoric_score",
                "non_neutral_post_ratio",
                "comment_label_oppositional_ratio",
                "anomaly_label",
                "anomaly_score",
                "comment_count",
                "analyzed_post_count",
            ]
        ]
        member_features.to_csv(seed_dir / "candidate_member_features.csv", index=False)

        seed_summary = {
            "seed": seed,
            "candidate_member_count": len(candidate_members),
            "tiered_expansion_member_count": int(len(tiered_members)),
            "tier1_co_negative_count": int(
                tiered_members["include_reason"].eq("tier1_co_negative_direct").sum()
            ),
            "tier2_tag_with_structure_count": int(
                tiered_members["include_reason"].eq("tier2_tag_similarity_with_structure").sum()
            ),
            "tier3_trigger_with_support_count": int(
                tiered_members["include_reason"].eq("tier3_trigger_with_co_negative_or_tag").sum()
            ),
            "tier4_two_hop_count": int(
                tiered_members["include_reason"].eq("tier4_two_hop_co_negative_multi_link").sum()
            ),
            "co_target_neighbor_count": int(len(co_target_neighbors)),
            "co_negative_neighbor_count": int(len(co_negative_neighbors)),
            "tag_similarity_neighbor_count": int(len(tag_neighbors)),
            "internal_coordination_edge_count": int(len(internal_edges)),
            "shared_negative_target_count": int(
                (shared_negative["shared_target_count"] > 0).sum()
            )
            if not shared_negative.empty
            else 0,
            **summarize_members(features, candidate_members_list),
        }
        summary_rows.append(seed_summary)
        (seed_dir / "summary.json").write_text(
            json.dumps(seed_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "seeds" / "seed_expansion_summary.csv", index=False)
    return summary


def write_readme(output_dir: Path) -> None:
    text = """# Coordination Group Discovery Output

This folder is generated by `coordination-expansion/discover_coordination_groups.py`.

Useful files:

- `groups/group_summary.csv`: reviewable small groups from the selected coordination layer.
- `groups/group_members.csv`: group membership with account-level evidence fields.
- `groups/group_internal_edges.csv`: internal co-target or co-negative-target edges.
- `groups/group_shared_targets.csv`: targets shared by at least two group members.
- `groups/skipped_large_components.csv`: giant components excluded from review tables.
- `seeds/<seed>/`: seed expansion evidence tables for each requested account.

The output is designed for manual review. It is not a final MCA score.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)

    group_edges, group_weight_col = read_layer(args.graph_dir, args.group_layer)
    co_target, _ = read_layer(args.graph_dir, "co_target")
    co_negative, _ = read_layer(args.graph_dir, "co_negative_target")
    tag_similarity, _ = read_layer(args.graph_dir, "tag_similarity")
    negative_edges = read_negative_edges(args.graph_dir)
    count_edges = read_count_edges(args.graph_dir)
    trigger_edges = read_trigger_edges(args.graph_dir)
    features = read_account_features(args.features_path)

    if args.group_layer == "co_negative_target":
        target_edges = negative_edges
        target_weight_col = "weight_negative"
    else:
        target_edges = count_edges
        target_weight_col = "weight_count"

    group_summary, _, _, _ = build_group_discovery(
        layer_edges=group_edges,
        layer=args.group_layer,
        weight_col=group_weight_col,
        threshold=args.group_threshold,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        top_groups=args.top_groups,
        target_edges=target_edges,
        target_weight_col=target_weight_col,
        features=features,
        output_dir=args.output_dir,
    )

    seed_summary = build_seed_expansions(
        seeds=args.seeds,
        co_target=co_target,
        co_negative=co_negative,
        tag_similarity=tag_similarity,
        trigger_edges=trigger_edges,
        target_edges=negative_edges,
        features=features,
        output_dir=args.output_dir,
        top_neighbors=args.seed_top_neighbors,
        co_negative_threshold=args.tier_co_negative_threshold,
        tag_threshold=args.tier_tag_threshold,
        trigger_threshold=args.tier_trigger_threshold,
        co_target_threshold=args.tier_co_target_threshold,
        two_hop_min_links=args.tier_two_hop_min_links,
        max_members=args.tier_max_members,
    )
    write_readme(args.output_dir)

    print(f"Coordination discovery written to {args.output_dir}")
    print(f"Groups: {len(group_summary):,} reviewable components")
    if args.seeds:
        print(f"Seed expansions: {len(seed_summary):,}")


if __name__ == "__main__":
    main()
