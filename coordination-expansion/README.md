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

### Stage 2 Temporal Verification

Stage 1 找到的是 candidate coordination groups；Stage 2 檢查群內帳號是否在同一篇 post 下短時間一起出現。

```bash
.venv/bin/python coordination-expansion/stage2_temporal_verification.py
```

預設輸出：

```text
stage2-verification/stage2_verification_evidence.csv
stage2-verification/stage2_group_summary.csv
stage2-verification/stage2_temporal_verification_report.md
```

pair-level labels:

```text
strong_temporal_sync   = at least one co-comment event within 5 minutes
moderate_temporal_sync = at least two co-comment events within 30 minutes
weak_temporal_overlap  = same thread overlap without short-window synchrony
no_temporal_sync       = no same-thread overlap in the local comments file
```

預設每個帳號只取最近 100 則 local comments，資料來自：

```text
Archive/export_working_files/comment_feedback_all_merged.csv.bak
```

如果要掃完整 local comments，可以加：

```bash
.venv/bin/python coordination-expansion/stage2_temporal_verification.py --max-comments-per-author 0
```

如果使用 Pullpush 或不同 group member list，結果可能和 local scan 不完全一致。

### Candidate Validation

把 Stage 1 expansion、MCA score、新版 full-population cluster、Stage 2 temporal evidence 接成一張 review table：

```bash
.venv/bin/python coordination-expansion/build_candidate_validation_table.py
```

預設讀取新版 full-cluster feature matrix：

```text
Archive/export_working_files/account_feature_matrix_with_clusters.csv
```

這份檔案是 full population `68,256` accounts 的 cluster 版本，包含：

```text
cluster_kmeans
is_extreme_outlier
anomaly_label
anomaly_score
behavior features
```

輸出：

```text
candidate-validation/candidate_validation_table.csv
candidate-validation/candidate_validation_report.md
```

目前 `review_priority` 是用來排序人工 review，不是最終 bot verdict：

```text
high_confidence_temporal_candidate
high_confidence_extreme_outlier
high_mca_review_candidate
temporal_only_review_candidate
low_priority_context_member
```

## Interpretation

輸出是 review evidence，不是最終判決。最有用的是看：

```text
這群人是否共同反對同一批 target？
這群內部是否有 co-negative / co-target structure？
這群是否也有 manipulative rhetoric 或 automation anomaly evidence？
這群是否有 temporal synchrony evidence？
```
