# Coordination Expansion Module

這個資料夾負責把 `adjacency/` 產生的 graph artifacts 轉成可人工 review 的群體證據。

它不生 adjacency matrix，也不算最終 MCA score。定位是：

```text
seed account / graph layer -> candidate group -> evidence tables
```

## Run

```bash
.venv/bin/python coordination-expansion/discover_coordination_groups.py \
  --seeds harvested JG87919 XBThodler
```

預設輸出：

```text
coordination-expansion/output/
```

## What It Does

### Group Discovery

預設從 `A_co_negative_target` 找小型 connected components：

```text
weight_co_negative_target >= 0.30
3 <= component size <= 50
```

輸出：

```text
groups/group_summary.csv
groups/group_members.csv
groups/group_internal_edges.csv
groups/group_shared_targets.csv
groups/skipped_large_components.csv
```

### Tiered Seed Expansion

從 seed account 往外擴張，但不使用加權總分。每個候選人要有明確納入理由：

```text
Tier 1: co_negative_target >= 0.20
Tier 2: tag_similarity >= 0.90 + one structural signal
Tier 3: trigger_response >= 0.50 + co_negative or tag_similarity support
Tier 4: 2-hop co_negative expansion, requiring links to >= 2 accepted members
```

`co_target` 是輔助信號，不單獨納入。

Seed 輸出：

```text
seeds/<seed>/tiered_expansion_members.csv
seeds/<seed>/coordination_neighbors.csv
seeds/<seed>/trigger_neighbors.csv
seeds/<seed>/internal_coordination_edges.csv
seeds/<seed>/shared_negative_targets_with_seed.csv
seeds/<seed>/candidate_member_features.csv
seeds/<seed>/summary.json
```

## Interpretation

輸出是 review evidence，不是最終判決。最有用的是看：

```text
這群人是否共同反對同一批 target？
這群內部是否有 co-negative / co-target structure？
這群是否也有 manipulative rhetoric 或 automation anomaly evidence？
```
