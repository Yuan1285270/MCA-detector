#!/usr/bin/env python3
"""Build a compact browser data bundle for the MCA demo site.

The site intentionally consumes the same pipeline outputs used in the report.
This script avoids requiring a backend server during demo time.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "MCA-demo-site"
OUTPUT_DIR = ROOT / "coordination-expansion" / "output"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def num(value: object, default: float = 0.0) -> float:
    if value in (None, "", "nan", "NaN"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    return int(round(num(value, default)))


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_accounts(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def priority_rank(priority: str) -> int:
    order = {
        "high_confidence_temporal_candidate": 0,
        "high_confidence_extreme_outlier": 1,
        "high_mca_review_candidate": 2,
        "temporal_only_review_candidate": 3,
        "low_priority_context_member": 4,
    }
    return order.get(priority, 9)


def temporal_rank(label: str) -> int:
    order = {
        "strong_temporal_sync": 0,
        "moderate_temporal_sync": 1,
        "weak_temporal_overlap": 2,
        "no_temporal_sync": 3,
    }
    return order.get(label, 9)


def confidence_rank(label: str) -> int:
    order = {
        "robust": 0,
        "moderate_review": 1,
        "fragile_single_event": 2,
        "fragile_long_median": 3,
        "weak_context": 4,
        "none": 5,
    }
    return order.get(label, 9)


def compact_group(row: dict[str, str]) -> dict[str, object]:
    return {
        "rank": integer(row.get("group_rank")),
        "seed": row.get("seed_group", ""),
        "memberCount": integer(row.get("member_count")),
        "p1": integer(row.get("p1_count")),
        "p2": integer(row.get("p2_count")),
        "p3": integer(row.get("p3_count")),
        "highMca": integer(row.get("high_mca_count")),
        "extremeOutliers": integer(row.get("extreme_outlier_count")),
        "reliableTemporalPairs": integer(row.get("reliable_temporal_pair_count")),
        "fragileTemporalPairs": integer(row.get("fragile_temporal_pair_count")),
        "robustTemporalPairs": integer(row.get("robust_temporal_pairs")),
        "moderateReviewTemporalPairs": integer(row.get("moderate_review_temporal_pairs")),
        "strongTemporalSync": integer(row.get("strong_temporal_sync")),
        "moderateTemporalSync": integer(row.get("moderate_temporal_sync")),
        "weakTemporalOverlap": integer(row.get("weak_temporal_overlap")),
        "noTemporalSync": integer(row.get("no_temporal_sync")),
        "tier1CoNegative": integer(row.get("tier1_co_negative_count")),
        "tier4TwoHop": integer(row.get("tier4_two_hop_count")),
        "internalEdges": integer(row.get("internal_coordination_edge_count")),
        "sharedNegativeTargets": integer(row.get("shared_negative_target_count")),
        "avgMca": num(row.get("avg_mca_score")),
        "maxMca": num(row.get("max_mca_score")),
        "avgRhetorical": num(row.get("avg_rhetorical_score")),
        "avgOppositional": num(row.get("avg_oppositional_stance_ratio")),
        "automationFraction": num(row.get("automation_anomaly_fraction")),
        "priority": row.get("group_priority", ""),
        "interpretation": row.get("interpretation", ""),
        "topAccounts": split_accounts(row.get("top_accounts", "")),
    }


def compact_account(row: dict[str, str], role_row: dict[str, str] | None, behavior_row: dict[str, str] | None) -> dict[str, object]:
    return {
        "account": row.get("account", ""),
        "tier": integer(row.get("tier")),
        "includeReason": row.get("include_reason", ""),
        "reviewPriority": row.get("review_priority", ""),
        "mca": num(row.get("mca_score_primary")),
        "manipulative": num(row.get("manipulative_signal")),
        "coordinative": num(row.get("coordinative_signal")),
        "interactionReach": num(row.get("interaction_reach_signal")),
        "automatic": num(row.get("automatic_behavior_signal")),
        "coNegativeToSeed": num(row.get("co_negative_weight_to_seed")),
        "tagSimilarityToSeed": num(row.get("tag_similarity_weight_to_seed")),
        "triggerToSeed": num(row.get("trigger_response_weight_to_seed")),
        "coTargetToSeed": num(row.get("co_target_weight_to_seed")),
        "commentCount": num(row.get("comment_count")),
        "postCount": num(row.get("post_count")),
        "activeDays": num(row.get("active_days")),
        "commentsPerDay": num(row.get("comments_per_day")),
        "burstRatio": num(row.get("burst_ratio")),
        "nightRatio": num(row.get("night_activity_ratio")),
        "oppositionalRatio": num(row.get("comment_label_oppositional_ratio")),
        "supportiveRatio": num(row.get("comment_label_supportive_ratio")),
        "avgRhetoric": num(row.get("avg_manipulative_rhetoric_score")),
        "anomalyScore": num(row.get("anomaly_score")),
        "isExtremeOutlier": truthy(row.get("is_extreme_outlier")),
        "cluster": row.get("full_cluster_kmeans", ""),
        "bestTemporalLabel": row.get("best_temporal_label", ""),
        "bestTemporalConfidence": row.get("best_temporal_confidence", ""),
        "reliableTemporalPairs": integer(row.get("reliable_temporal_pair_count")),
        "fragileTemporalPairs": integer(row.get("fragile_temporal_pair_count")),
        "minMedianDelay": num(row.get("min_median_delay_minutes")),
        "role": (role_row or {}).get("role_label", ""),
        "roleZh": (role_row or {}).get("role_label_zh", ""),
        "roleReason": (role_row or {}).get("role_reason", ""),
        "behaviorProfile": (behavior_row or {}).get("behavior_profile", ""),
        "behaviorReason": (behavior_row or {}).get("behavior_reason", ""),
    }


def compact_pair(row: dict[str, str]) -> dict[str, object]:
    return {
        "a": row.get("account_a", ""),
        "b": row.get("account_b", ""),
        "coNegative": num(row.get("co_negative_weight")),
        "samePost": integer(row.get("same_post_count")),
        "within5": integer(row.get("within_5min_count")),
        "within30": integer(row.get("within_30min_count")),
        "medianDelay": num(row.get("median_delay_minutes"), None),
        "minDelay": num(row.get("min_delay_minutes"), None),
        "activationOverlap": num(row.get("account_lifecycle_overlap"), None),
        "label": row.get("verification_label", ""),
        "confidence": row.get("temporal_confidence", ""),
    }


def compact_edge(row: dict[str, str]) -> dict[str, object]:
    source = row.get("source_author", "")
    target = row.get("target_author", "")
    layer = row.get("layer", "")
    weight = num(row.get("weight_co_negative_target") or row.get("weight_co_target"))
    if not source or not target:
        return {}
    return {
        "source": source,
        "target": target,
        "layer": layer,
        "weight": weight,
        "sharedTargets": integer(row.get("shared_target_count")),
    }


def compact_shared_target(row: dict[str, str]) -> dict[str, object]:
    return {
        "neighbor": row.get("neighbor", ""),
        "count": integer(row.get("shared_target_count")),
        "weight": num(row.get("shared_target_weight_sum")),
        "targets": row.get("shared_targets", ""),
    }


def main() -> None:
    final_groups = read_csv(OUTPUT_DIR / "final-summary" / "final_group_summary.csv")
    candidates = read_csv(OUTPUT_DIR / "candidate-validation" / "candidate_validation_table.csv")
    pairs = read_csv(OUTPUT_DIR / "stage2-verification" / "stage2_verification_evidence.csv")
    roles = read_csv(OUTPUT_DIR / "account-roles" / "account_role_table.csv")
    behavior = read_csv(OUTPUT_DIR / "behavior-profile" / "behavior_profile_table.csv")

    role_lookup = {(r.get("seed_group", ""), r.get("account", "")): r for r in roles}
    behavior_lookup = {(r.get("seed_group", ""), r.get("account", "")): r for r in behavior}

    groups = [compact_group(row) for row in final_groups]
    top_group_seeds = [str(group["seed"]) for group in groups[:20]]

    accounts_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        seed = row.get("seed_group", "")
        if seed not in top_group_seeds:
            continue
        key = (seed, row.get("account", ""))
        accounts_by_group[seed].append(compact_account(row, role_lookup.get(key), behavior_lookup.get(key)))

    for seed, accounts in accounts_by_group.items():
        accounts.sort(key=lambda item: (priority_rank(str(item["reviewPriority"])), -float(item["mca"]), int(item["tier"])))

    pairs_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        seed = row.get("group_seed", "")
        if seed not in top_group_seeds:
            continue
        pair = compact_pair(row)
        pairs_by_group[seed].append(pair)

    for seed, group_pairs in pairs_by_group.items():
        group_pairs.sort(
            key=lambda item: (
                temporal_rank(str(item["label"])),
                confidence_rank(str(item["confidence"])),
                -int(item["within30"]),
                -int(item["samePost"]),
            )
        )
        pairs_by_group[seed] = group_pairs[:80]

    edges_by_group: dict[str, list[dict[str, object]]] = {}
    targets_by_group: dict[str, list[dict[str, object]]] = {}
    for seed in top_group_seeds:
        seed_dir = OUTPUT_DIR / "seeds" / seed
        edges = []
        for row in read_csv(seed_dir / "internal_coordination_edges.csv"):
            edge = compact_edge(row)
            if edge:
                edges.append(edge)
        edges.sort(key=lambda item: -float(item["weight"]))
        edges_by_group[seed] = edges[:120]

        targets = [compact_shared_target(row) for row in read_csv(seed_dir / "shared_negative_targets_with_seed.csv")]
        targets.sort(key=lambda item: -float(item["weight"]))
        targets_by_group[seed] = targets[:12]

    abnormal_accounts: dict[str, dict[str, object]] = {}
    for seed, accounts in accounts_by_group.items():
        for account in accounts:
            priority = str(account["reviewPriority"])
            if account["isExtremeOutlier"] or priority == "high_confidence_extreme_outlier":
                key = str(account["account"])
                current = abnormal_accounts.get(key)
                if current is None or float(account["mca"]) > float(current["mca"]):
                    account_copy = dict(account)
                    account_copy["seedGroup"] = seed
                    abnormal_accounts[key] = account_copy

    abnormal_list = sorted(
        abnormal_accounts.values(),
        key=lambda item: (
            0 if item["reviewPriority"] == "high_confidence_extreme_outlier" else 1,
            -float(item["mca"]),
            -abs(float(item["anomalyScore"])),
        ),
    )[:40]

    data = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "source": "coordination-expansion/output",
            "groupCount": len(groups),
            "candidateAccountRows": len(candidates),
            "pairEvidenceRows": len(pairs),
        },
        "groups": groups,
        "accountsByGroup": accounts_by_group,
        "pairsByGroup": pairs_by_group,
        "edgesByGroup": edges_by_group,
        "sharedTargetsByGroup": targets_by_group,
        "abnormalAccounts": abnormal_list,
    }

    out_path = SITE_DIR / "data" / "demo-data.js"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "window.MCA_DEMO_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
