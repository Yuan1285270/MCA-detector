#!/usr/bin/env python3
"""Build lightweight reviewer-response experiment tables.

These experiments are intended to support the paper narrative, not to claim a
fully supervised benchmark. They collect existing artifacts and add small
sanity checks around weight sensitivity, signal sparsity, stage ablation, and
temporal random baselines.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SCORES_PATH = ROOT / "mca-scoring/output/account_mca_scores.csv"
FEATURES_PATH = ROOT / "Archive/export_working_files/account_feature_matrix.csv"
COMMENTS_PATH = ROOT / "Archive/export_working_files/comment_feedback_all_merged.csv"
FINAL_GROUPS_PATH = ROOT / "coordination-expansion/output/final-summary/final_group_summary.csv"
CANDIDATES_PATH = ROOT / "coordination-expansion/output/candidate-validation/candidate_validation_table.csv"
STAGE2_PATH = ROOT / "coordination-expansion/output/stage2-verification/stage2_verification_evidence.csv"
STAGE2_SUMMARY_PATH = ROOT / "coordination-expansion/output/stage2-verification/stage2_group_summary.csv"

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

WEIGHT_VARIANTS = {
    "primary_30_35_15_20": (0.30, 0.35, 0.15, 0.20),
    "alt_40_40_10_10": (0.40, 0.40, 0.10, 0.10),
    "no_manipulative_00_50_20_30": (0.00, 0.50, 0.20, 0.30),
    "low_manipulative_10_45_20_25": (0.10, 0.45, 0.20, 0.25),
    "coordination_heavy_20_50_10_20": (0.20, 0.50, 0.10, 0.20),
    "automation_heavy_20_25_15_40": (0.20, 0.25, 0.15, 0.40),
}


def normalize_user_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def apply_weights(scores: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.Series:
    total = sum(weights)
    wm, wc, wr, wa = [w / total for w in weights]
    return (
        wm * scores["manipulative_signal"]
        + wc * scores["coordinative_signal"]
        + wr * scores["interaction_reach_signal"]
        + wa * scores["automatic_behavior_signal"]
    )


def rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False).astype(int)


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation using pandas ranks, avoiding scipy."""
    joined = pd.concat([left, right], axis=1).dropna()
    if joined.empty:
        return float("nan")
    ranked_left = joined.iloc[:, 0].rank(method="average")
    ranked_right = joined.iloc[:, 1].rank(method="average")
    return float(ranked_left.corr(ranked_right, method="pearson"))


def label_pair(metrics: dict[str, object]) -> str:
    within_5 = int(metrics["within_5min_count"])
    within_30 = int(metrics["within_30min_count"])
    same_post = int(metrics["same_post_count"])
    if within_5 > 0:
        return "strong_temporal_sync"
    if within_30 >= 2 or (same_post >= 3 and within_30 > 0):
        return "moderate_temporal_sync"
    if same_post > 0:
        return "weak_temporal_overlap"
    return "no_temporal_sync"


def temporal_confidence(metrics: dict[str, object]) -> str:
    same_post = int(metrics["same_post_count"])
    within_5 = int(metrics["within_5min_count"])
    within_30 = int(metrics["within_30min_count"])
    median_delay = metrics["median_delay_minutes"]
    median = float(median_delay) if pd.notna(median_delay) else np.inf
    if same_post == 0:
        return "none"
    if within_5 >= 2 or (same_post >= 3 and within_30 >= 2 and median <= 90):
        return "robust"
    if same_post == 1 and within_5 == 1 and within_30 == 1:
        return "fragile_single_event"
    if median > 120 and (within_5 > 0 or within_30 > 0):
        return "fragile_long_median"
    if within_5 > 0 or within_30 > 0:
        return "moderate_review"
    return "weak_context"


def pair_temporal_metrics(
    left_posts: dict[str, np.ndarray],
    right_posts: dict[str, np.ndarray],
    *,
    strong_window: float = 5.0,
    moderate_window: float = 30.0,
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
    return {
        "same_post_count": len(shared_posts),
        "within_5min_count": within_strong,
        "within_30min_count": within_moderate,
        "median_delay_minutes": float(np.median(all_min_delays)) if all_min_delays else np.nan,
        "min_delay_minutes": float(np.min(all_min_delays)) if all_min_delays else np.nan,
    }


def build_mca_weight_sensitivity() -> pd.DataFrame:
    scores = pd.read_csv(SCORES_PATH, low_memory=False)
    scores["user_id"] = normalize_user_id(scores["user_id"])
    scores = scores.set_index("user_id", drop=False)
    primary = scores["mca_score_primary"]
    primary_top = {
        n: set(primary.sort_values(ascending=False).head(n).index.astype(str))
        for n in (20, 50, 100)
    }
    key_accounts = ["BtcKing1111", "harvested", "Odd-Following-247", "JG87919"]

    rows = []
    top20_rows = []
    for name, weights in WEIGHT_VARIANTS.items():
        variant = apply_weights(scores, weights)
        ranks = rank_series(variant)
        top_sets = {n: set(variant.sort_values(ascending=False).head(n).index.astype(str)) for n in (20, 50, 100)}
        row = {
            "variant": name,
            "w_manipulative": weights[0],
            "w_coordinative": weights[1],
            "w_reach": weights[2],
            "w_automation": weights[3],
            "spearman_vs_primary_all_accounts": spearman_without_scipy(primary, variant),
        }
        for n in (20, 50, 100):
            overlap = len(primary_top[n] & top_sets[n])
            union = len(primary_top[n] | top_sets[n])
            row[f"top{n}_overlap_count"] = overlap
            row[f"top{n}_jaccard"] = overlap / union if union else 0.0
        for account in key_accounts:
            row[f"rank_{account}"] = int(ranks.loc[account]) if account in ranks.index else np.nan
        rows.append(row)
        for rank, account in enumerate(variant.sort_values(ascending=False).head(20).index.astype(str), start=1):
            top20_rows.append({"variant": name, "rank": rank, "account": account})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "mca_weight_sensitivity.csv", index=False)
    pd.DataFrame(top20_rows).to_csv(OUT / "mca_weight_top20_accounts.csv", index=False)
    return result


def build_manipulative_sparsity() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    scores = pd.read_csv(SCORES_PATH, low_memory=False)
    features["user_id"] = normalize_user_id(features["user_id"])
    scores["user_id"] = normalize_user_id(scores["user_id"])
    scores = scores.set_index("user_id", drop=False)
    features = features.set_index("user_id", drop=False)

    tag_cols = [col for col in NONNEUTRAL_TAG_COUNT_COLUMNS if col in features.columns]
    features["nonneutral_rhetoric_tag_count"] = features[tag_cols].sum(axis=1) if tag_cols else 0

    primary_top20 = set(scores.sort_values("mca_score_primary", ascending=False).head(20).index.astype(str))
    primary_top100 = set(scores.sort_values("mca_score_primary", ascending=False).head(100).index.astype(str))
    candidate_accounts = set(pd.read_csv(CANDIDATES_PATH, usecols=["account"])["account"].astype(str))

    cohorts = {
        "all_feature_accounts": features.index,
        "mca_top20": [a for a in primary_top20 if a in features.index],
        "mca_top100": [a for a in primary_top100 if a in features.index],
        "stage1_candidate_accounts": [a for a in candidate_accounts if a in features.index],
    }

    rows = []
    for name, idx in cohorts.items():
        subset = features.loc[list(idx)].copy()
        if subset.empty:
            continue
        row = {
            "cohort": name,
            "accounts": len(subset),
            "analyzed_post_count_gt0_rate": float((subset["analyzed_post_count"].fillna(0) > 0).mean()),
            "avg_rhetorical_score_gt0_rate": float((subset["avg_manipulative_rhetoric_score"].fillna(0) > 0).mean()),
            "nonneutral_rhetoric_tag_gt0_rate": float((subset["nonneutral_rhetoric_tag_count"].fillna(0) > 0).mean()),
            "oppositional_comment_ratio_gt0_rate": float((subset["comment_label_oppositional_ratio"].fillna(0) > 0).mean()),
            "mean_avg_rhetorical_score": float(subset["avg_manipulative_rhetoric_score"].fillna(0).mean()),
            "mean_nonneutral_tag_count": float(subset["nonneutral_rhetoric_tag_count"].fillna(0).mean()),
            "mean_oppositional_comment_ratio": float(subset["comment_label_oppositional_ratio"].fillna(0).mean()),
        }
        rows.append(row)
    sparsity = pd.DataFrame(rows)
    sparsity.to_csv(OUT / "manipulative_signal_sparsity.csv", index=False)

    signal_rows = []
    for cohort, idx in {
        "all_scored_accounts": scores.index,
        "mca_top20": list(primary_top20),
        "mca_top100": list(primary_top100),
    }.items():
        subset = scores.loc[list(idx)].copy()
        for col in [
            "manipulative_signal",
            "coordinative_signal",
            "interaction_reach_signal",
            "automatic_behavior_signal",
            "mca_score_primary",
        ]:
            signal_rows.append(
                {
                    "cohort": cohort,
                    "signal": col,
                    "mean": float(subset[col].mean()),
                    "median": float(subset[col].median()),
                    "nonzero_rate": float((subset[col] > 0).mean()),
                    "p90": float(subset[col].quantile(0.90)),
                    "p95": float(subset[col].quantile(0.95)),
                }
            )
    signal_stats = pd.DataFrame(signal_rows)
    signal_stats.to_csv(OUT / "mca_signal_distribution.csv", index=False)
    return sparsity, signal_stats


def build_ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_groups = pd.read_csv(FINAL_GROUPS_PATH, low_memory=False)
    candidates = pd.read_csv(CANDIDATES_PATH, low_memory=False)
    stage2 = pd.read_csv(STAGE2_PATH, low_memory=False)

    unique_candidates = candidates["account"].astype(str).nunique()
    stage_summary = pd.DataFrame(
        [
            {
                "setting": "MCA only",
                "account_or_group_scope": "20 seed accounts",
                "main_output": "ranked seed list",
                "evidence_available": "account-level score only",
                "main_limitation": "does not show group membership or synchronized behavior",
                "groups": 0,
                "unique_accounts": 20,
                "pair_evidence_rows": 0,
                "strong_pairs": 0,
                "moderate_pairs": 0,
                "robust_pairs": 0,
            },
            {
                "setting": "MCA + Stage 1 expansion",
                "account_or_group_scope": "candidate coordination groups",
                "main_output": "seed neighborhoods and shared targets",
                "evidence_available": "co-negative, support layers, tiers, shared negative targets",
                "main_limitation": "may capture ideological alignment without timing evidence",
                "groups": int(final_groups["seed_group"].nunique()),
                "unique_accounts": int(unique_candidates),
                "pair_evidence_rows": 0,
                "strong_pairs": 0,
                "moderate_pairs": 0,
                "robust_pairs": 0,
            },
            {
                "setting": "MCA + Stage 1 + Stage 2",
                "account_or_group_scope": "candidate groups with pair evidence",
                "main_output": "review-priority groups and pair-level temporal evidence",
                "evidence_available": "shared targets plus same-thread short-window timing",
                "main_limitation": "still review-oriented; temporal coincidence remains possible",
                "groups": int(final_groups["seed_group"].nunique()),
                "unique_accounts": int(unique_candidates),
                "pair_evidence_rows": int(len(stage2)),
                "strong_pairs": int((stage2["verification_label"] == "strong_temporal_sync").sum()),
                "moderate_pairs": int((stage2["verification_label"] == "moderate_temporal_sync").sum()),
                "robust_pairs": int((stage2["temporal_confidence"] == "robust").sum()),
            },
        ]
    )
    stage_summary.to_csv(OUT / "pipeline_stage_ablation_summary.csv", index=False)

    tier_rows = []
    tier_counts = candidates["tier"].value_counts(dropna=False).to_dict()
    for tier in sorted(tier_counts):
        tier_subset = candidates.loc[candidates["tier"].eq(tier)]
        tier_rows.append(
            {
                "tier": int(tier),
                "membership_rows": int(len(tier_subset)),
                "unique_accounts": int(tier_subset["account"].astype(str).nunique()),
                "high_mca_rows": int(tier_subset["is_high_mca"].fillna(False).astype(bool).sum()),
                "extreme_outlier_rows": int(tier_subset["is_extreme_outlier"].fillna(False).astype(bool).sum()),
            }
        )
    tier_summary = pd.DataFrame(tier_rows)
    tier_summary.to_csv(OUT / "stage1_tier_membership_summary.csv", index=False)

    tier_map = candidates.set_index(["seed_group", "account"])["tier"].to_dict()
    stage2 = stage2.copy()
    stage2["tier_a"] = [
        tier_map.get((row.group_seed, row.account_a), np.nan) for row in stage2.itertuples(index=False)
    ]
    stage2["tier_b"] = [
        tier_map.get((row.group_seed, row.account_b), np.nan) for row in stage2.itertuples(index=False)
    ]
    stage2["involves_tier4"] = stage2["tier_a"].eq(4) | stage2["tier_b"].eq(4)
    tier4_ablation = pd.DataFrame(
        [
            {
                "setting": "with_tier4",
                "membership_rows": int(len(candidates)),
                "unique_accounts": int(candidates["account"].astype(str).nunique()),
                "stage2_pairs": int(len(stage2)),
                "strong_pairs": int((stage2["verification_label"] == "strong_temporal_sync").sum()),
                "moderate_pairs": int((stage2["verification_label"] == "moderate_temporal_sync").sum()),
                "robust_pairs": int((stage2["temporal_confidence"] == "robust").sum()),
            },
            {
                "setting": "without_tier4_posthoc",
                "membership_rows": int((candidates["tier"] != 4).sum()),
                "unique_accounts": int(candidates.loc[candidates["tier"] != 4, "account"].astype(str).nunique()),
                "stage2_pairs": int((~stage2["involves_tier4"]).sum()),
                "strong_pairs": int(((stage2["verification_label"] == "strong_temporal_sync") & ~stage2["involves_tier4"]).sum()),
                "moderate_pairs": int(((stage2["verification_label"] == "moderate_temporal_sync") & ~stage2["involves_tier4"]).sum()),
                "robust_pairs": int(((stage2["temporal_confidence"] == "robust") & ~stage2["involves_tier4"]).sum()),
            },
            {
                "setting": "tier4_contribution_only",
                "membership_rows": int((candidates["tier"] == 4).sum()),
                "unique_accounts": int(candidates.loc[candidates["tier"] == 4, "account"].astype(str).nunique()),
                "stage2_pairs": int(stage2["involves_tier4"].sum()),
                "strong_pairs": int(((stage2["verification_label"] == "strong_temporal_sync") & stage2["involves_tier4"]).sum()),
                "moderate_pairs": int(((stage2["verification_label"] == "moderate_temporal_sync") & stage2["involves_tier4"]).sum()),
                "robust_pairs": int(((stage2["temporal_confidence"] == "robust") & stage2["involves_tier4"]).sum()),
            },
        ]
    )
    tier4_ablation.to_csv(OUT / "tier4_two_hop_posthoc_ablation.csv", index=False)

    case_groups = final_groups.loc[final_groups["seed_group"].isin(["harvested", "Odd-Following-247", "BtcKing1111"])]
    case_cols = [
        "seed_group",
        "member_count",
        "tier1_co_negative_count",
        "tier4_two_hop_count",
        "internal_coordination_edge_count",
        "shared_negative_target_count",
        "strong_temporal_sync",
        "moderate_temporal_sync",
        "robust_temporal_pairs",
        "weak_temporal_overlap",
        "group_priority",
    ]
    case_groups[case_cols].to_csv(OUT / "case_study_stage1_stage2_comparison.csv", index=False)
    return stage_summary, tier4_ablation, case_groups[case_cols]


def load_comments_for_baseline(max_comments_per_author: int = 100) -> pd.DataFrame:
    header = pd.read_csv(COMMENTS_PATH, nrows=0)
    if {"comment_id", "post_id", "author", "created_utc"}.issubset(header.columns):
        comments = pd.read_csv(
            COMMENTS_PATH,
            usecols=["comment_id", "post_id", "author", "created_utc"],
            low_memory=False,
        )
    elif {"comment_id", "link_id", "author", "created_utc"}.issubset(header.columns):
        comments = pd.read_csv(
            COMMENTS_PATH,
            usecols=["comment_id", "link_id", "author", "created_utc"],
            low_memory=False,
        )
        comments = comments.rename(columns={"link_id": "post_id"})
        comments["post_id"] = comments["post_id"].astype("string").str.replace(r"^t3_", "", regex=True)
    else:
        raise ValueError("Comment file must contain comment_id/post_id/author/created_utc or comment_id/link_id/author/created_utc.")
    comments["author"] = normalize_user_id(comments["author"])
    comments["post_id"] = comments["post_id"].astype("string").str.strip()
    comments["created_utc"] = pd.to_numeric(comments["created_utc"], errors="coerce")
    comments = comments.dropna(subset=["author", "post_id", "created_utc"])
    comments = comments.drop_duplicates(subset=["comment_id"])
    comments = comments.sort_values(["author", "created_utc"], ascending=[True, False])
    if max_comments_per_author > 0:
        comments = comments.groupby("author", group_keys=False).head(max_comments_per_author)
    return comments.sort_values(["author", "post_id", "created_utc"])


def account_post_times(comments: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for (author, post_id), group in comments.groupby(["author", "post_id"], sort=False):
        output.setdefault(str(author), {})[str(post_id)] = group["created_utc"].to_numpy(dtype=float)
    return output


def summarize_pair_table(df: pd.DataFrame, label: str) -> dict[str, object]:
    total = len(df)
    return {
        "pair_set": label,
        "pairs": int(total),
        "same_post_pairs": int((df["same_post_count"] > 0).sum()),
        "same_post_pair_rate": float((df["same_post_count"] > 0).mean()) if total else 0.0,
        "strong_pairs": int((df["verification_label"] == "strong_temporal_sync").sum()),
        "strong_pair_rate": float((df["verification_label"] == "strong_temporal_sync").mean()) if total else 0.0,
        "moderate_pairs": int((df["verification_label"] == "moderate_temporal_sync").sum()),
        "moderate_pair_rate": float((df["verification_label"] == "moderate_temporal_sync").mean()) if total else 0.0,
        "robust_pairs": int((df["temporal_confidence"] == "robust").sum()),
        "robust_pair_rate": float((df["temporal_confidence"] == "robust").mean()) if total else 0.0,
        "within_5min_events": int(df["within_5min_count"].sum()),
        "within_30min_events": int(df["within_30min_count"].sum()),
        "mean_same_post_count": float(df["same_post_count"].mean()) if total else 0.0,
    }


def build_temporal_random_baseline(sample_pairs: int = 5000) -> pd.DataFrame:
    stage2 = pd.read_csv(STAGE2_PATH, low_memory=False)
    candidates = pd.read_csv(CANDIDATES_PATH, low_memory=False)
    comments = load_comments_for_baseline(max_comments_per_author=100)
    author_counts = comments.groupby("author").size()
    eligible_authors = sorted(author_counts.loc[author_counts >= 5].index.astype(str).tolist())
    post_times = account_post_times(comments.loc[comments["author"].isin(eligible_authors)].copy())

    rng = random.Random(42)
    sampled: set[tuple[str, str]] = set()
    attempts = 0
    while len(sampled) < sample_pairs and attempts < sample_pairs * 50:
        a, b = rng.sample(eligible_authors, 2)
        sampled.add(tuple(sorted((a, b))))
        attempts += 1

    rows = []
    for a, b in sorted(sampled):
        metrics = pair_temporal_metrics(post_times.get(a, {}), post_times.get(b, {}))
        rows.append(
            {
                "account_a": a,
                "account_b": b,
                **metrics,
                "verification_label": label_pair(metrics),
                "temporal_confidence": temporal_confidence(metrics),
            }
        )
    random_pairs = pd.DataFrame(rows)
    random_pairs.to_csv(OUT / "temporal_random_baseline_pairs_sample.csv", index=False)

    # Activity-controlled baseline: sample non-candidate authors from the same
    # capped comment-count bins as each candidate pair. This is still a
    # lightweight sanity check, but it reduces the strongest activity-volume
    # confounder in the simple random baseline.
    candidate_accounts = set(candidates["account"].astype(str))
    candidate_pairs = stage2[["account_a", "account_b"]].astype(str).drop_duplicates()
    bins = [0, 1, 3, 5, 10, 20, 50, 100, np.inf]

    def count_bin(author: str) -> str:
        count = int(author_counts.get(author, 0))
        return str(pd.cut(pd.Series([count]), bins=bins, include_lowest=True).iloc[0])

    eligible_pool = [author for author in eligible_authors if author not in candidate_accounts]
    pool_by_bin: dict[str, list[str]] = {}
    for author in eligible_pool:
        pool_by_bin.setdefault(count_bin(author), []).append(author)

    fallback_pool = eligible_pool or eligible_authors
    controlled_rows = []
    controlled_sampled: set[tuple[str, str]] = set()
    candidate_pair_rows = candidate_pairs.to_dict("records")
    attempts = 0
    while len(controlled_rows) < len(candidate_pair_rows) and attempts < len(candidate_pair_rows) * 100:
        pair = candidate_pair_rows[len(controlled_rows) % len(candidate_pair_rows)]
        bins_for_pair = [count_bin(pair["account_a"]), count_bin(pair["account_b"])]
        sampled_authors = []
        for bin_label in bins_for_pair:
            pool = pool_by_bin.get(bin_label) or fallback_pool
            sampled_authors.append(rng.choice(pool))
        a, b = sampled_authors
        if a == b:
            attempts += 1
            continue
        key = tuple(sorted((a, b)))
        if key in controlled_sampled:
            attempts += 1
            continue
        controlled_sampled.add(key)
        metrics = pair_temporal_metrics(post_times.get(a, {}), post_times.get(b, {}))
        controlled_rows.append(
            {
                "account_a": a,
                "account_b": b,
                "matched_bin_a": bins_for_pair[0],
                "matched_bin_b": bins_for_pair[1],
                **metrics,
                "verification_label": label_pair(metrics),
                "temporal_confidence": temporal_confidence(metrics),
            }
        )
        attempts += 1

    controlled_pairs = pd.DataFrame(controlled_rows)
    controlled_pairs.to_csv(OUT / "temporal_activity_controlled_pairs_sample.csv", index=False)

    summary = pd.DataFrame(
        [
            summarize_pair_table(stage2, "candidate_group_pairs"),
            summarize_pair_table(random_pairs, "random_active_pairs_n5000"),
            summarize_pair_table(controlled_pairs, "activity_controlled_random_pairs"),
        ]
    )
    summary["random_seed"] = 42
    summary["min_comments_per_random_author"] = [np.nan, 5, 5]
    summary["max_comments_per_author"] = 100
    summary["eligible_random_authors"] = [np.nan, len(eligible_authors), len(eligible_pool)]
    summary.to_csv(OUT / "temporal_random_baseline_summary.csv", index=False)
    return summary


def build_signal_pruning_summary() -> pd.DataFrame:
    rows = [
        {
            "signal": "text_fingerprint_distance",
            "status": "removed_from_formal_stage2",
            "reason": "TF-IDF/cosine style text distance behaved like topic similarity in a single-topic Bitcoin community, not reliable shared-operator evidence.",
            "paper_use": "Threats to validity and signal pruning table.",
        },
        {
            "signal": "account_lifecycle_overlap",
            "status": "removed_from_formal_stage2",
            "reason": "Lifecycle and activation-window overlap did not separate harvested-type positives from independent same-topic users.",
            "paper_use": "Threats to validity and signal pruning table.",
        },
        {
            "signal": "temporal_synchrony",
            "status": "kept_as_formal_stage2",
            "reason": "Same-thread short-window co-presence is the clearest review signal for candidate group verification in the current data.",
            "paper_use": "Core Stage 2 evidence.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_pruning_summary.csv", index=False)
    return df


def write_markdown_summary(
    sensitivity: pd.DataFrame,
    sparsity: pd.DataFrame,
    stage_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    signal_pruning: pd.DataFrame,
) -> None:
    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        shown = df if max_rows is None else df.head(max_rows)
        cols = list(shown.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in shown.itertuples(index=False):
            lines.append("| " + " | ".join(format_value(v) for v in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Reviewer-Response Experiment Pack",
        "",
        "These lightweight experiments summarize existing pipeline artifacts and small sanity checks for the paper. They are not a supervised benchmark.",
        "",
        "## 1. MCA Weight Sensitivity",
        "",
        md_table(sensitivity.round(4)),
        "",
        "Paper-ready takeaway: the MCA score should be framed as seed prioritization. Sensitivity results can be used to discuss whether top seeds remain stable when heuristic weights change.",
        "",
        "## 2. Manipulative Signal Sparsity",
        "",
        md_table(sparsity.round(4)),
        "",
        "Paper-ready takeaway: LLM-derived manipulative-content features are sparse in the current dataset. This justifies treating MCA as a seed ranking score and relying on graph/temporal evidence downstream.",
        "",
        "## 3. Ablation-Style Pipeline Comparison",
        "",
        md_table(stage_summary),
        "",
        "Paper-ready takeaway: MCA-only ranks accounts, Stage 1 creates candidate groups, and Stage 2 adds temporal pair evidence. This supports the evidence-separation argument.",
        "",
        "## 3b. Stage 1 Co-Negative Threshold Sensitivity",
        "",
        "Run `run_stage1_threshold_sensitivity.py` to regenerate this table. Current output: `stage1_co_negative_threshold_sensitivity.csv`.",
        "",
        "## 4. Temporal Random Baseline",
        "",
        md_table(temporal_summary.round(6)),
        "",
        "Paper-ready takeaway: random active account pairs provide a null baseline for same-thread short-window co-presence. Use this cautiously: it is a lightweight baseline, not a full statistical significance test.",
        "",
        "## 5. Signal Pruning",
        "",
        md_table(signal_pruning),
        "",
        "Generated outputs are in this folder's `outputs/` directory.",
    ]
    (OUT / "reviewer_response_experiment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    sensitivity = build_mca_weight_sensitivity()
    sparsity, _ = build_manipulative_sparsity()
    stage_summary, _, _ = build_ablation_tables()
    temporal_summary = build_temporal_random_baseline(sample_pairs=5000)
    signal_pruning = build_signal_pruning_summary()
    write_markdown_summary(sensitivity, sparsity, stage_summary, temporal_summary, signal_pruning)
    print(f"Reviewer-response experiment outputs written to {OUT}")


if __name__ == "__main__":
    main()
