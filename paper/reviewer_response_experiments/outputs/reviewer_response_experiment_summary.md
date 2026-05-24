# Reviewer-Response Experiment Pack

These lightweight experiments summarize existing pipeline artifacts and small sanity checks for the paper. They are not a supervised benchmark.

## 1. MCA Weight Sensitivity

| variant | w_manipulative | w_coordinative | w_reach | w_automation | spearman_vs_primary_all_accounts | top20_overlap_count | top20_jaccard | top50_overlap_count | top50_jaccard | top100_overlap_count | top100_jaccard | rank_BtcKing1111 | rank_harvested | rank_Odd-Following-247 | rank_JG87919 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_30_35_15_20 | 0.3 | 0.35 | 0.15 | 0.2 | 1.0 | 20 | 1.0 | 50 | 1.0 | 100 | 1.0 | 1 | 14 | 18 | 40 |
| alt_40_40_10_10 | 0.4 | 0.4 | 0.1 | 0.1 | 0.983 | 17 | 0.7391 | 37 | 0.5873 | 71 | 0.5504 | 3 | 15 | 21 | 10 |
| no_manipulative_00_50_20_30 | 0.0 | 0.5 | 0.2 | 0.3 | 0.8211 | 15 | 0.6 | 29 | 0.4085 | 58 | 0.4085 | 5 | 11 | 17 | 118 |
| low_manipulative_10_45_20_25 | 0.1 | 0.45 | 0.2 | 0.25 | 0.911 | 17 | 0.7391 | 37 | 0.5873 | 71 | 0.5504 | 2 | 11 | 19 | 66 |
| coordination_heavy_20_50_10_20 | 0.2 | 0.5 | 0.1 | 0.2 | 0.9861 | 18 | 0.8182 | 40 | 0.6667 | 77 | 0.626 | 1 | 11 | 18 | 28 |
| automation_heavy_20_25_15_40 | 0.2 | 0.25 | 0.15 | 0.4 | 0.9929 | 15 | 0.6 | 38 | 0.6129 | 79 | 0.6529 | 3 | 19 | 25 | 374 |

Paper-ready takeaway: the MCA score should be framed as seed prioritization. Sensitivity results can be used to discuss whether top seeds remain stable when heuristic weights change.

## 2. Manipulative Signal Sparsity

| cohort | accounts | analyzed_post_count_gt0_rate | avg_rhetorical_score_gt0_rate | nonneutral_rhetoric_tag_gt0_rate | oppositional_comment_ratio_gt0_rate | mean_avg_rhetorical_score | mean_nonneutral_tag_count | mean_oppositional_comment_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_feature_accounts | 49767 | 0.238 | 0.1169 | 0.1027 | 0.3919 | 3.3868 | 0.32 | 0.2593 |
| mca_top20 | 20 | 1.0 | 1.0 | 1.0 | 1.0 | 37.504 | 10.4 | 0.3769 |
| mca_top100 | 100 | 1.0 | 1.0 | 1.0 | 0.99 | 35.4361 | 16.18 | 0.3135 |
| stage1_candidate_accounts | 172 | 0.2849 | 0.1977 | 0.1919 | 1.0 | 6.6545 | 1.4477 | 0.5159 |

Paper-ready takeaway: LLM-derived manipulative-content features are sparse in the current dataset. This justifies treating MCA as a seed ranking score and relying on graph/temporal evidence downstream.

## 3. Ablation-Style Pipeline Comparison

| setting | account_or_group_scope | main_output | evidence_available | main_limitation | groups | unique_accounts | pair_evidence_rows | strong_pairs | moderate_pairs | robust_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCA only | 20 seed accounts | ranked seed list | account-level score only | does not show group membership or synchronized behavior | 0 | 20 | 0 | 0 | 0 | 0 |
| MCA + Stage 1 expansion | candidate coordination groups | seed neighborhoods and shared targets | co-negative, support layers, tiers, shared negative targets | may capture ideological alignment without timing evidence | 20 | 172 | 0 | 0 | 0 | 0 |
| MCA + Stage 1 + Stage 2 | candidate groups with pair evidence | review-priority groups and pair-level temporal evidence | shared targets plus same-thread short-window timing | still review-oriented; temporal coincidence remains possible | 20 | 172 | 928 | 13 | 8 | 3 |

Paper-ready takeaway: MCA-only ranks accounts, Stage 1 creates candidate groups, and Stage 2 adds temporal pair evidence. This supports the evidence-separation argument.

## 3b. Stage 1 Co-Negative Threshold Sensitivity

Run `run_stage1_threshold_sensitivity.py` to regenerate this table. Current output: `stage1_co_negative_threshold_sensitivity.csv`.

## 4. Temporal Random Baseline

| pair_set | pairs | same_post_pairs | same_post_pair_rate | strong_pairs | strong_pair_rate | moderate_pairs | moderate_pair_rate | robust_pairs | robust_pair_rate | within_5min_events | within_30min_events | mean_same_post_count | random_seed | min_comments_per_random_author | max_comments_per_author | eligible_random_authors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_group_pairs | 928 | 424 | 0.456897 | 13 | 0.014009 | 8 | 0.008621 | 3 | 0.003233 | 14 | 73 | 0.692888 | 42 |  | 100 |  |
| random_active_pairs_n5000 | 5000 | 425 | 0.085 | 15 | 0.003 | 0 | 0.0 | 1 | 0.0002 | 16 | 63 | 0.096 | 42 | 5.0 | 100 | 6385.0 |
| activity_controlled_random_pairs | 928 | 110 | 0.118534 | 7 | 0.007543 | 4 | 0.00431 | 2 | 0.002155 | 7 | 23 | 0.154095 | 42 | 5.0 | 100 | 6255.0 |

Paper-ready takeaway: random active account pairs provide a null baseline for same-thread short-window co-presence. Use this cautiously: it is a lightweight baseline, not a full statistical significance test.

## 5. Signal Pruning

| signal | status | reason | paper_use |
| --- | --- | --- | --- |
| text_fingerprint_distance | removed_from_formal_stage2 | TF-IDF/cosine style text distance behaved like topic similarity in a single-topic Bitcoin community, not reliable shared-operator evidence. | Threats to validity and signal pruning table. |
| account_lifecycle_overlap | removed_from_formal_stage2 | Lifecycle and activation-window overlap did not separate harvested-type positives from independent same-topic users. | Threats to validity and signal pruning table. |
| temporal_synchrony | kept_as_formal_stage2 | Same-thread short-window co-presence is the clearest review signal for candidate group verification in the current data. | Core Stage 2 evidence. |

Generated outputs are in this folder's `outputs/` directory.
