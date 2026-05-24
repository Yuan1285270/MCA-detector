# Paper-Ready Notes for Reviewer-Response Experiments

These notes summarize the lightweight experiments in `outputs/` and can be
merged into the English paper. They are phrased conservatively: the results
support review-oriented evidence separation, not supervised accuracy claims.

## Recommended Insert: Sensitivity and Ablation Checks

To evaluate whether the pipeline depends entirely on one heuristic setting, we
performed lightweight sensitivity and ablation-style checks. First, we
recomputed MCA rankings under several weight variants. The original
`30/35/15/20` setting was compared with the earlier `40/40/10/10` alternative,
a no-manipulative variant, a low-manipulative variant, a coordination-heavy
variant, and an automation-heavy variant. The score was most stable under
moderate variants: the earlier `40/40/10/10` setting had a Spearman correlation
of 0.983 with the primary ranking and retained 17 of the top 20 accounts. The
coordination-heavy setting had a Spearman correlation of 0.986 and retained 18
of the top 20 accounts. Removing manipulative content entirely reduced
stability, but still retained 15 of the top 20 accounts. This supports the
interpretation of MCA as a seed-prioritization score rather than a final
classifier.

Second, we compared three pipeline stages. MCA-only produces 20 seed accounts
but no group evidence. Adding Stage 1 expansion produces 20 candidate groups
and 172 unique candidate accounts, but this evidence may still reflect shared
ideology. Adding Stage 2 temporal verification produces 928 account-pair
evidence rows, including 13 strong temporal pairs, 8 moderate temporal pairs,
and 3 robust temporal pairs. This supports the design choice of separating
account ranking, group discovery, and temporal verification.

## Recommended Insert: Temporal Null Baseline

We also sampled 5,000 random active account pairs from the local comment file,
requiring each sampled author to have at least five comments and using the same
per-author 100-comment cap as Stage 2. Candidate-group pairs were more likely
to share the same post than random active pairs (45.7% versus 8.5%). Strong
5-minute temporal sync was also more frequent among candidate pairs (1.40%
versus 0.30%). More importantly, moderate temporal sync was observed in 0.86%
of candidate pairs but 0% of random active pairs, and robust temporal evidence
was observed in 0.32% of candidate pairs but 0.02% of random active pairs.
This random baseline is not a full statistical significance test, but it
indicates that Stage 2 is not merely measuring arbitrary co-presence among
active Reddit users.

## Recommended Insert: LLM-Derived Signal Sparsity

The current dataset also shows that LLM-derived manipulative-content features
are sparse at the full-account level. Among all 49,767 feature accounts, only
23.8% have at least one analyzed post, 11.7% have a nonzero average
manipulative rhetoric score, and 10.3% have nonneutral rhetoric tags. This
means the manipulative-content component is informative for accounts with
analyzed posts, especially MCA top-ranked accounts, but it is not uniformly
available across the whole population. Therefore, MCA should be interpreted as
a seed-ranking score, while final review priority should rely on downstream
graph and temporal evidence.

## Compact Table: Weight Sensitivity

| Variant | Spearman vs primary | Top-20 overlap | BtcKing rank | harvested rank | Odd rank |
|---|---:|---:|---:|---:|---:|
| Primary 30/35/15/20 | 1.000 | 20/20 | 1 | 14 | 18 |
| Alternative 40/40/10/10 | 0.983 | 17/20 | 3 | 15 | 21 |
| No manipulative 0/50/20/30 | 0.821 | 15/20 | 5 | 11 | 17 |
| Low manipulative 10/45/20/25 | 0.911 | 17/20 | 2 | 11 | 19 |
| Coordination-heavy 20/50/10/20 | 0.986 | 18/20 | 1 | 11 | 18 |
| Automation-heavy 20/25/15/40 | 0.993 | 15/20 | 3 | 19 | 25 |

## Compact Table: Temporal Baseline

| Pair set | Pairs | Same-post rate | Strong rate | Moderate rate | Robust rate |
|---|---:|---:|---:|---:|---:|
| Candidate group pairs | 928 | 45.7% | 1.40% | 0.86% | 0.32% |
| Random active pairs | 5,000 | 8.5% | 0.30% | 0.00% | 0.02% |

## Compact Table: Stage Ablation

| Setting | Output | Groups | Unique accounts | Pair evidence | Strong | Moderate | Robust |
|---|---|---:|---:|---:|---:|---:|---:|
| MCA only | seed ranking | 0 | 20 | 0 | 0 | 0 | 0 |
| MCA + Stage 1 | candidate groups | 20 | 172 | 0 | 0 | 0 | 0 |
| MCA + Stage 1 + Stage 2 | temporal evidence | 20 | 172 | 928 | 13 | 8 | 3 |

## Suggested Threats-to-Validity Text

LLM-derived manipulative-content features are sparse in the current dataset.
Although top-ranked MCA accounts generally contain analyzed post-level
rhetoric signals, most accounts in the full population have no analyzed posts
or no nonneutral rhetoric tags. This limits the role of the manipulative
signal as a population-wide classifier. For this reason, MCA is used only for
seed prioritization, and the system relies on graph expansion and temporal
verification for group-level evidence.
