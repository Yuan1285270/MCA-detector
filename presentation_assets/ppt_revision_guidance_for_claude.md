# MCA Detector PPT Revision Guidance for Claude

This document is the source-of-truth revision brief for updating the current deck:

`/Users/yuan/Desktop/Desktop - Yuan的MacBook Pro/Capstone/AI Detective/MCA_Detector_v2_11.pptx`

Please revise the PPT using the current project logic, not the older "MCA score ranking only" story.

## Current Core Narrative

The project is no longer just an account ranking system.

The correct narrative is:

```text
LLM content / stance analysis
-> account feature matrix + multi-layer graphs
-> MCA score selects suspicious seed accounts
-> Stage 1 seed expansion finds candidate coordination groups
-> Stage 2 temporal verification checks whether the group shows synchronized action
-> output review tables and demo website
```

Important wording:

- Do not claim we identify confirmed bots, paid operators, or cyber troops.
- Use "suspicious coordination candidates", "review priority", and "evidence for manual review".
- MCA score is seed selection / review priority, not a final verdict.
- Co-negative target graph is candidate discovery evidence, not proof of same operator.
- Temporal verification is the second-stage filter that helps distinguish shared ideology from stronger coordinated action evidence.

## Existing Deck Problems

The current deck still reads as:

```text
LLM -> behavior matrix -> adjacency graph -> MCA score -> top accounts
```

This is outdated.

It needs to become:

```text
LLM -> feature matrix + multi-graph -> MCA seed ranking
-> Stage 1 expansion -> Stage 2 temporal verification -> group investigation
```

Specific outdated parts:

- Top 20 account ranking is overemphasized.
- `behavior/` appears as a main independent module. It is now legacy; useful behavior features are folded into the account feature matrix / MCA scoring flow.
- Tag similarity and text similarity are overemphasized.
- Degree-adjusted graph is too prominent.
- Role / group analysis is described as future work, but we now have lightweight role tables.
- Results should focus on candidate groups, not only accounts.

## Required New Assets

Use these SVG assets directly in the PPT. They were generated from current pipeline outputs.

Directory:

`/Users/yuan/SocialMedia_LLM/presentation_assets/ppt_graph_explanations`

Files:

1. `slide_04_updated_architecture.svg`
2. `slide_19_multigraph_edge_density.svg`
3. `slide_20_count_vs_conegative_harvested.svg`
4. `slide_21_conegative_threshold_harvested.svg`
5. `slide_36_mca_seed_expansion_pipeline.svg`
6. `slide_47_temporal_confidence_harvested_vs_jg.svg`

## Suggested Revised Deck Structure

The deck can be long. Prioritize clarity over slide count. A 40-60 page designer/examiner version is acceptable.

### Chapter 1 — Problem Definition

#### Slide 1 — Title

Keep title page, but adjust subtitle:

```text
MCA Detector
Suspicious Social Coordination Detection Pipeline
以 Reddit Bitcoin 討論為例
```

Avoid saying "detecting cyber troops" directly.

#### Slide 2 — Research Problem

Keep the current idea, but update the key line:

```text
We are not trying to prove real-world identity.
We are trying to reduce a large account population into reviewable suspicious coordination groups.
```

Use:

```text
Manipulative + Coordinated + Reviewable Evidence
```

not:

```text
Manipulative + Coordinated + Influential enough to matter
```

#### Slide 3 — Core Design Challenge

Add a new slide:

```text
Coordination != Cyber troops
```

Explain:

- People with the same ideology may attack the same targets.
- High activity does not equal manipulation.
- Bot-like behavior does not equal coordinated group behavior.
- Therefore, the system must separate candidate discovery from verification.

### Chapter 2 — Updated System Pipeline

#### Slide 4 — Replace Old System Flow

Use:

`slide_04_updated_architecture.svg`

Main message:

```text
The demo website is not the analysis method.
The analysis pipeline generates evidence tables; the website only presents them.
```

Speaker note:

```text
The old deck showed the system as a flat sequence. The new architecture separates analysis from presentation and separates seed ranking from group verification.
```

#### Slide 5 — Data Source

Keep mostly the same.

Clarify:

- Input: Reddit posts, comments, authors, timestamps, post-comment relations, text.
- LLM outputs: post rhetoric features and comment stance/feedback features.
- These do not directly label an account as malicious.

#### Slide 6 — LLM Content Analysis

Keep, but update provider wording:

```text
LLM backend can be Gemini or Ollama.
Current pipeline supports choosing provider; future analysis should use Ollama for local reproducibility.
```

Important:

Do not imply Gemini is the only method.

### Chapter 3 — Account Feature Matrix

#### Slide 7 — Rename Behavior Feature Matrix

Old title:

```text
行為特徵矩陣
```

New title:

```text
Account Feature Matrix
LLM features + behavior features + anomaly features
```

Explain:

```text
This is not the old independent behavior/ module.
The current feature matrix combines:
1. LLM post rhetoric features
2. LLM comment stance features
3. activity / behavior features
4. anomaly score
```

Mention behavior/ is legacy if needed:

```text
Early behavior analysis existed as a separate module; the formal pipeline now folds those features into the account feature matrix.
```

### Chapter 4 — Why Multi-Graph, Not One Graph

This section is very important. The deck must explicitly explain why count, signed, and frequency graphs are not enough.

#### Slide 8 — Why Build Graphs?

Message:

```text
Account-level features describe individuals.
Graphs describe relationships between accounts.
Coordination is a relationship-level pattern.
```

#### Slide 9 — Count Graph

Explain:

```text
Count graph asks: how many times did A comment on B's posts?
```

Example:

```text
A -> B has 20 comments.
```

Why not enough:

- A may simply be highly active.
- B may be a popular account.
- Count does not tell support vs attack.
- Count does not tell whether multiple accounts share targets.

Conclusion:

```text
Count graph measures interaction volume, not coordination intent.
```

#### Slide 10 — Signed Graph

Explain:

```text
Signed graph asks: is A generally supportive or oppositional toward B?
```

Why not enough:

- It is pair-level.
- It does not show whether A, C, and E attack the same targets.
- It cannot reveal group-level shared target structure.

Conclusion:

```text
Signed graph captures stance direction, but not shared target structure.
```

#### Slide 11 — Trigger / Frequency Graph

Explain:

```text
Trigger-response asks whether B often comments after A posts.
```

Why not enough:

- Popular posts naturally attract many responses.
- Highly active users create dense response edges.
- Frequency can show attention, but not necessarily coordinated action.

Conclusion:

```text
Trigger-response is useful as support, but too dense/noisy to be the main expansion edge.
```

#### Slide 12 — Multi-Graph Edge Density

Use:

`slide_19_multigraph_edge_density.svg`

This uses real edge counts:

```text
Count graph: 149,265 edges
Trigger-response graph: 149,265 edges
Signed graph: 130,447 edges
Co-target graph: 39,469 edges
Tag similarity graph: 7,475 edges
Co-negative graph: 3,736 edges
```

Message:

```text
Dense graphs are useful for context, but sparse graphs carry stronger evidence.
Co-negative is sparse enough to be the Stage 1 expansion backbone.
```

#### Slide 13 — Count / Co-target vs Co-negative Example

Use:

`slide_20_count_vs_conegative_harvested.svg`

Explain with actual `harvested` example:

```text
Co-target finds many neighbors in the same discussion space.
Co-negative narrows to accounts sharing negative target structure.
```

Use the key comparison:

```text
harvested has 15 co-target neighbors
but only 7 direct Tier 1 co-negative expansion members
```

Conclusion:

```text
Co-target asks "same discussion space?"
Co-negative asks "same attack targets?"
```

#### Slide 14 — Why Co-negative Works

Use:

`slide_21_conegative_threshold_harvested.svg`

Explain:

```text
co_negative_target compares two accounts' negative-target vectors.
It is not enough that both attacked the same person once.
The score is high when their sets of attacked targets overlap in structure.
```

Actual examples:

```text
harvested - rtublin: 0.385
harvested - NectarineDirect936: 0.335
harvested - CoolPineapple6969: 0.218
```

Threshold:

```text
Tier 1 includes direct neighbors with co_negative_target >= 0.20.
```

Important limitation:

```text
Co-negative can still capture independent users with the same ideology.
Therefore it is candidate discovery, not final proof.
```

### Chapter 5 — MCA Score

#### Slide 15 — Why MCA Exists

Message:

```text
We cannot manually inspect every account.
MCA score gives an interpretable seed ranking.
```

#### Slide 16 — MCA Four Signals

Update weights:

```text
Manipulative signal: 30%
Coordinative signal: 35%
Interaction reach: 15%
Automatic behavior: 20%
```

Do not use old 40/35/15/10 weights.

#### Slide 17 — Manipulative Signal

Current simplified design:

```text
avg_rhetorical_score
manipulative / non-neutral post ratio
oppositional stance
```

Avoid overcomplicating with old rhetoric volume formulas.

#### Slide 18 — Coordinative Signal

Explain:

```text
co-target
co-negative-target
trigger-response frequency
```

Tag similarity is not a major scoring signal in the final story. It can be described as exploratory/supporting.

#### Slide 19 — Interaction Reach

Explain:

```text
outgoing volume
incoming attention
interaction breadth
```

#### Slide 20 — Automatic Behavior

Explain:

```text
Uses anomaly score from account feature matrix.
It indicates unusual behavior, not bot identity.
```

### Chapter 6 — Why Not MCA Alone?

#### Slide 21 — MCA Alone Is Not Enough

Use:

`slide_36_mca_seed_expansion_pipeline.svg`

Key message:

```text
MCA ranks suspicious accounts.
Seed expansion asks whether suspicion becomes group structure.
Temporal verification asks whether group structure shows synchronized action.
```

Example explanation:

```text
A high-MCA account may simply be active, aggressive, or anomalous.
That does not prove it belongs to a coordinated group.
```

Conclusion:

```text
MCA = entry point.
Expansion + temporal evidence = investigation pipeline.
```

### Chapter 7 — Stage 1 Seed Expansion

#### Slide 22 — Stage 1 Purpose

Explain:

```text
Input: seed accounts from MCA ranking
Output: candidate coordination groups
```

Stage 1 does not confirm same operator.

#### Slide 23 — Expansion Rules

Show this table:

```text
Tier 0: seed account
Tier 1: co_negative_target >= 0.20
Tier 2: tag_similarity >= 0.90 + structural support
Tier 3: trigger_response >= 0.50 + co-negative or tag support
Tier 4: 2-hop co-negative, requiring links to at least 2 accepted members
```

#### Slide 24 — What if an Account Has No Tags?

Explain clearly:

```text
If an account has no posts, tag similarity may be 0.
This does not remove the account from expansion.
It simply cannot enter through the tag-based Tier 2 path.
It can still enter through Tier 1 co-negative, Tier 3 trigger with support, or Tier 4 two-hop co-negative.
```

Key sentence:

```text
Tag similarity is optional support, not a required signal.
```

#### Slide 25 — 2-hop / 二階擴張

Use Chinese:

```text
2-hop = 二階擴張 / 朋友的朋友擴張
```

Explain:

```text
Seed A connects to B.
B connects to C.
C may be an outer-ring member.
```

Why not too loose:

```text
Tier 4 requires C to connect back to at least 2 already accepted members through co-negative edges.
```

### Chapter 8 — Stage 2 Temporal Verification

#### Slide 26 — Why Stage 2 Exists

Message:

```text
Stage 1 finds shared opposition structure.
But shared opposition can be organic ideology.
Stage 2 checks whether members also show synchronized action.
```

#### Slide 27 — Stage 2 Method

Explain:

```text
For each seed group:
1. take included members
2. compare every pair of accounts
3. find posts where both commented
4. calculate time difference between their comments
5. assign temporal label and confidence
```

#### Slide 28 — Temporal Labels

Use:

```text
strong_temporal_sync:
at least one same-post event within 5 minutes

moderate_temporal_sync:
at least two same-post events within 30 minutes
or >=3 shared posts with at least one within-30-minute event

weak_temporal_overlap:
same-post overlap, but no short-window synchrony

no_temporal_sync:
no shared post
```

#### Slide 29 — Why Strong Alone Is Too Weak

Explain:

```text
One 5-minute event can happen in a popular thread.
Therefore strong_temporal_sync is an event label, not a final confidence label.
```

#### Slide 30 — Temporal Confidence

Use:

```text
robust:
repeated or multi-post short-window synchrony

moderate_review:
useful timing evidence, still needs review

fragile_single_event:
only one short-window event

fragile_long_median:
has short-window event, but typical delay is long

weak_context / none:
background only
```

#### Slide 31 — Temporal Confidence Example

Use:

`slide_47_temporal_confidence_harvested_vs_jg.svg`

Explain:

```text
Both harvested and JG87919 have co-negative structure.
But harvested has robust temporal evidence.
JG87919 has weaker / fragile timing evidence.
This is why Stage 2 is needed.
```

### Chapter 9 — Output Tables

#### Slide 32 — Candidate Validation Table

Explain:

This table joins:

```text
Stage 1 membership
MCA score
behavior / anomaly features
cluster metadata
Stage 2 temporal evidence
```

Main output:

```text
review_priority
```

Priority categories:

```text
high_confidence_temporal_candidate
high_confidence_extreme_outlier
high_mca_review_candidate
temporal_only_review_candidate
low_priority_context_member
```

#### Slide 33 — Final Group Summary

Explain:

```text
Aggregates account-level review priority into group-level ranking.
```

Group labels:

```text
G1_multiple_high_priority_accounts
G1_high_priority_plus_support
G2_single_high_priority_account
G2_multiple_review_candidates
G3_single_review_candidate
G4_context_group
```

#### Slide 34 — Behavior Profile Table

Explain:

```text
Adds descriptive behavior labels for manual review.
```

Labels:

```text
extreme_outlier_behavior
short_window_high_activity
high_frequency_activity
bursty_activity
low_activity_unknown
normal_range_activity
```

Emphasize:

```text
Behavior profile is not a bot verdict.
```

#### Slide 35 — Account Role Table

Explain:

Lightweight group roles:

```text
leader_instigator / 帶頭起鬨 = seed account
comment_attacker / 留言攻擊者 = high oppositional ratio
comment_supporter / 留言支持者 = high supportive ratio
context_member / 背景成員 = retained by expansion, no dominant role
```

Emphasize:

```text
These are descriptive review labels, not real-world actor identities.
```

### Chapter 10 — Results

#### Slide 36 — Overall Results

Use current numbers:

```text
20 seed groups
172 unique candidate accounts
13 strong temporal sync pairs
8 moderate temporal sync pairs
3 robust confidence pairs
```

#### Slide 37 — Top Groups

Replace old Top 20 Account table.

Use group-level table:

```text
lol_camis
BtcKing1111
tzacPACO
harvested
iPurchaseBitcoin
```

Columns:

```text
members
P1
P2
strong temporal
robust temporal
shared negative targets
```

#### Slide 38 — Case Study: harvested

Explain:

```text
8 members
7 Tier 1 co-negative members
11 shared negative targets
1 strong temporal pair
1 robust temporal pair
```

Key pair:

```text
NectarineDirect936 <-> harvested
same_post_count = 17
within_5min_count = 2
within_30min_count = 4
temporal_confidence = robust
```

#### Slide 39 — Case Study: JG87919

Explain:

```text
JG87919-type case shows why Stage 1 is not enough.
It has co-negative structure, but timing evidence is fragile / weaker.
This may represent shared ideology rather than synchronized action.
```

Do not say "JG87919 is definitely normal".
Say:

```text
It is a negative/control-style case showing why temporal confidence matters.
```

### Chapter 11 — Signal Pruning / Design Decisions

#### Slide 40 — Signals We Tested but Downgraded

Use this table:

```text
Text fingerprint:
Downgraded because TF-IDF captures topic similarity in Bitcoin community.

Lifecycle overlap:
Downgraded because harvested and JG87919-type cases overlap heavily.

Tag similarity:
Kept as optional support only because accounts without posts have no tags.

Degree-adjusted graph:
Kept for EDA / visualization, not core decision.

Cluster:
Kept as metadata / review context, not final evidence.
```

#### Slide 41 — Final Design Logic

Use:

```text
MCA score = seed selection
Co-negative = candidate group discovery
Temporal synchrony + confidence = second-stage verification
Validation tables = manual review support
```

### Chapter 12 — Demo Website and Limitations

#### Slide 42 — Demo Website

Explain:

```text
Client mode:
high-level risk overview, group review, account detail

Demo/research mode:
method details, evidence tables, graph view
```

Important:

```text
The website reads generated tables.
It does not recompute the pipeline.
```

#### Slide 43 — Limitations

Use:

```text
No large-scale ground truth
Temporal sync may still be affected by popular posts
Co-negative detects shared opposition, not same operator
Outputs are review candidates
```

#### Slide 44 — Future Work

Use:

```text
PTT extension with explicit push/boo stance labels
More manually labeled validation cases
Cross-platform action traces
Threshold stability tests
```

#### Slide 45 — Final Takeaway

Use:

```text
MCA Detector turns account-level suspicion into group-level evidence.
It does not replace human judgment; it reduces the search space and explains why each candidate group deserves review.
```

## Slides to Delete or Downgrade

Delete or rewrite:

- Current text similarity slide: text fingerprint is no longer formal evidence.
- Current Top 20 accounts result slide: replace with group-level output.
- XBThodler-only interaction graph slide: replace with harvested / JG87919 case studies.

Downgrade to appendix or brief mention:

- Degree-adjusted graph
- Tag similarity as primary evidence
- Cluster as primary evidence
- Old behavior module

## Required Tone

Use cautious language.

Good:

```text
candidate group
review priority
coordination evidence
shared opposition structure
temporal verification
manual review
```

Avoid:

```text
confirmed cyber troop
bot detected
same operator proven
bad actor confirmed
```

## One-Sentence Thesis for the Deck

```text
MCA Detector uses LLM-derived content and stance features to select suspicious seed accounts, expands them through co-negative target graphs into candidate coordination groups, and verifies those groups with temporal synchrony evidence for human review.
```

