# A Social Coordination Behavior Detection System Combining Manipulative Content Analysis and Graph Expansion

Students: Cong-Yuan Lin, Jun-Hao Chang, Jia-An Lai  
Advisor: Jun-Hong Li  
Affiliation: Department of Computer Science and Information Engineering, Feng Chia University

## Abstract

Manipulative information operations on social platforms do not always appear as isolated automated accounts. Suspicious accounts may coordinate through similar targets, repeated interactions, or synchronized timing, while ordinary highly active users may also share the same ideology and attack the same targets. Therefore, relying only on an account-level suspiciousness score can easily produce false positives. This paper proposes a reproducible and review-oriented pipeline for detecting suspicious social coordination. The system first uses large language models to analyze cryptocurrency-related Reddit posts and comment stances, then constructs account-level feature matrices and multi-layer adjacency graphs. In the first stage, a Manipulative Coordination Account (MCA) score is used to select high-risk seed accounts, and a co-negative-target graph is used to expand those seeds into candidate coordination groups. In the second stage, the system does not treat shared negative targets as direct evidence of cyber-troop behavior. Instead, it verifies whether group members appear in the same discussion thread within short time windows, using temporal synchrony as stronger coordination evidence. In the current Reddit cryptocurrency dataset, the pipeline selected 20 seed groups and 172 unique candidate accounts, identifying 13 strong temporal synchrony pairs and 8 moderate temporal synchrony pairs. Case analysis shows that the `harvested` group contains both shared negative targets and robust temporal synchrony, while a `JG87919`-type case contains shared negative targets but lacks effective temporal synchrony. These results suggest that the proposed pipeline can rank suspicious coordination groups for human review without overclaiming that every candidate is a confirmed bot or real-world operator.

**Keywords:** social media analysis, coordinated behavior detection, manipulative rhetoric, graph expansion, temporal synchrony, Reddit

## 1. Introduction

Social media has become an important space for public discussion, financial narratives, and investment sentiment formation. At the same time, it can be exploited by automated accounts, coordinated accounts, or manipulative content campaigns to amplify specific narratives. Prior studies on social bot detection have shown that bots may influence users through content generation, network interaction, temporal behavior, and emotional expression [1]. However, whether an account looks automated is not the same as whether it belongs to a coordinated operation. In cryptocurrency communities, ordinary users can also be highly active, emotionally intense, and jointly critical of the same opponents. As a result, account-level scores or shared attack targets alone can misclassify organic ideological alignment as manipulation.

This paper focuses on a narrower and more practical problem: not to prove that an account is a real-world cyber troop, but to generate reviewable suspicious coordination candidates. The goal is to identify candidate groups, provide interpretable evidence, and allow analysts to inspect the relationship between account suspiciousness, shared targets, and short-window synchronized behavior.

The contributions of this work are:

1. We propose a two-stage suspicious coordination pipeline that separates account-level MCA scoring, graph-based expansion, and temporal verification.
2. We use the co-negative-target graph for candidate group discovery, rather than treating shared targets as a final verdict.
3. We use short-window co-appearance in the same thread as a verification signal to distinguish ideological alignment from stronger coordinated action evidence.
4. We produce review-oriented outputs, including group summaries, pair evidence tables, candidate validation tables, and simple account role labels, while keeping the demo website separate from the analysis pipeline.

## 2. Related Work

Early social bot detection research mainly focused on account-level classification. Ferrara et al. described the rise of social bots and discussed how bots attempt to imitate human behavior and influence political, economic, and public discussions [1]. Varol et al. proposed a human-bot interaction detection framework using a large set of features, including metadata, content, sentiment, network, and temporal signals [2]. These approaches are useful for estimating whether a single account is automated, but they are less directly suited to explaining whether multiple accounts act together.

Another line of research emphasizes group behavior and behavioral sequences. Cresci et al. proposed Social Fingerprinting, which detects groups of spambots through DNA-inspired behavioral modeling [3]. Mazza et al. introduced RTbust, which exploits temporal patterns in retweet time series to detect suspicious botnets on Twitter [4]. These studies show that temporal and group-level behavior can reveal coordination that is not visible from single-account features alone.

Recent coordinated behavior detection research more directly addresses shared actions. Pacheco et al. proposed building coordination networks from shared behavioral traces and demonstrated how identity, image, hashtag sequence, retweet behavior, and temporal patterns can uncover coordinated networks [5]. Graham et al. introduced the Coordination Network Toolkit, a framework for representing coordinated behavior with weighted directed multigraphs and analyzing online influence operations, astroturfing, and activism [6]. In the Reddit context, Saeed et al. proposed TROLLMAGNIFIER and observed that accounts linked to state-sponsored troll behavior can show loose coordination, similar topics, and temporal synchronization [7].

This work also builds on stance detection. The SemEval-2016 stance detection task formulated the problem of identifying whether a text is in favor of, against, or neutral toward a target [8]. We extend this idea by converting comment feedback toward post authors into signed account-to-account edges and then projecting those edges into shared negative target relations.

## 3. System Architecture

The system is divided into two parts: the analysis pipeline and the presentation layer. The analysis pipeline produces reviewable evidence from raw Reddit posts and comments. The demo website only reads the generated output tables and presents them in a client-facing or demo-facing interface. It does not compute scores, discover groups, verify temporal synchrony, or make final decisions.

```text
raw posts/comments
  -> LLM post/comment analysis
  -> account feature matrix + adjacency graphs
  -> MCA seed ranking
  -> coordination expansion
  -> temporal verification
  -> group summary + account roles
```

### 3.1 LLM-Based Content and Stance Analysis

The system first analyzes posts to produce sentiment and manipulative rhetoric features, including `sentiment_score`, `manipulative_rhetoric_score`, and `rhetoric_tags`. It then analyzes comments to determine the stance of a comment toward the original post, producing `feedback_label`, `feedback_score`, and `edge_weight`.

Each direct comment can therefore be converted into an account-to-account edge:

```text
comment_author -> post_author
```

Oppositional comments form negative interactions, while supportive comments form positive interactions. This representation allows the system to build signed account graphs instead of treating all comments as identical interactions.

### 3.2 Account Features and MCA Score

The MCA score is not a final verdict. It is used for seed selection and review priority. The score currently includes four signal groups:

| Signal | Main features | Meaning |
|---|---|---|
| Manipulative | average rhetoric score, non-neutral post ratio, oppositional stance ratio | Whether the account often uses manipulative or attack-oriented language |
| Coordinative | co-target, co-negative-target, trigger-response frequency | Whether the account points to similar targets or follows stable response patterns |
| Interaction reach | outgoing volume, incoming attention, interaction breadth | How broad and visible the account's interactions are |
| Automatic behavior | Isolation Forest anomaly score | Whether the account's activity pattern is abnormal |

The current primary weights are 0.30 for manipulative signals, 0.35 for coordinative signals, 0.15 for interaction reach, and 0.20 for automatic behavior. Since large-scale ground truth is not available, these weights are treated as interpretable initial review weights rather than supervised optimal weights.

### 3.3 Adjacency Graph Construction

The system constructs multiple account-level adjacency graphs, including count, signed interaction, positive interaction, negative interaction, trigger-response, co-target, and co-negative-target graphs. The main graph used for candidate group discovery is `A_co_negative_target`:

```text
A_co_negative_target[i,j] = cosine_similarity(negative_target_profile_i, negative_target_profile_j)
```

This graph asks whether two accounts often oppose or attack the same post authors. However, shared negative targets are only a necessary clue, not sufficient evidence of manipulation. Ordinary users with the same ideology may naturally criticize the same controversial authors. Therefore, this graph is used for candidate discovery, not as a final verdict.

## 4. Candidate Expansion and Verification

### 4.1 Stage 1: Seed Expansion

The first stage selects seed accounts from the MCA top ranking and expands each seed through graph layers. The expansion strategy is tiered rather than a black-box weighted sum:

| Tier | Inclusion condition | Interpretation |
|---|---|---|
| Tier 1 | co-negative-target >= 0.20 | Strong overlap in shared negative targets |
| Tier 2 | tag similarity >= 0.90 plus structural support | Similar rhetoric profile, but requiring graph evidence |
| Tier 3 | trigger-response >= 0.50 plus co-negative or tag support | Stable response pattern with structural support |
| Tier 4 | 2-hop co-negative expansion with links to at least two accepted members | Peripheral candidate members |

The `co_target` graph is used only as supporting context and does not independently include a candidate. This design keeps each membership decision interpretable.

### 4.2 Stage 2: Temporal Verification

The second stage checks whether accounts within a candidate group appear in the same Reddit thread within short time windows. The current labels are:

```text
strong_temporal_sync   = at least one same-thread comment event within 5 minutes
moderate_temporal_sync = at least two same-thread comment events within 30 minutes
weak_temporal_overlap  = same-thread overlap without short-window synchrony
no_temporal_sync       = no same-thread overlap
```

To reduce the risk of overinterpreting a single coincidence in a popular thread, the system also assigns temporal confidence:

```text
robust          = repeated or multi-post short-window synchrony
moderate_review = useful timing evidence that still requires manual review
fragile         = single-event or long-median-delay evidence used only as context
none            = no usable synchrony evidence
```

Earlier versions tested `text_fingerprint_distance` and `account_lifecycle_overlap`, but these signals were removed from formal evidence. In a single-topic cryptocurrency community, TF-IDF text similarity often captures topic similarity rather than shared operators. Similarly, account lifecycle overlap did not separate known positive and negative cases. The formal Stage 2 evidence therefore focuses on temporal synchrony and temporal confidence.

## 5. Experiments and Case Studies

### 5.1 Dataset and Outputs

The current experiment uses Reddit cryptocurrency-related posts and comments. The pipeline selected 20 seed accounts from MCA ranking and expanded them into 20 candidate seed groups. The resulting output contains 178 group membership rows and 172 unique candidate accounts. Stage 2 temporal verification identified 13 strong temporal synchrony pairs and 8 moderate temporal synchrony pairs. Among them, 3 pairs reached robust confidence and 44 pairs were labeled as moderate-review temporal evidence.

The top five candidate groups are shown below. The ranking represents review priority, not a final cyber-troop verdict.

| Rank | Seed group | Members | P1 | P2 | Strong sync | Moderate sync | Robust pairs | Shared negative targets |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | lol_camis | 23 | 2 | 3 | 4 | 0 | 0 | 25 |
| 2 | BtcKing1111 | 11 | 2 | 3 | 2 | 1 | 0 | 13 |
| 3 | tzacPACO | 6 | 2 | 1 | 0 | 3 | 1 | 8 |
| 4 | harvested | 8 | 2 | 0 | 1 | 0 | 1 | 11 |
| 5 | iPurchaseBitcoin | 7 | 2 | 0 | 1 | 0 | 1 | 6 |

### 5.2 Case Study: The `harvested` Group

The `harvested` group contains 8 members. In Stage 1, it includes 7 Tier 1 co-negative direct members, 9 internal coordination edges, and 11 shared negative targets. In Stage 2, `NectarineDirect936` and `harvested` appeared together in 17 shared posts, including 2 events within 5 minutes and 4 events within 30 minutes. This pair was labeled `strong_temporal_sync / robust`.

This case shows why the two-stage design is useful: shared negative targets identify the candidate group, while repeated short-window co-appearance raises the review priority.

### 5.3 Negative Control: A `JG87919`-Type Case

The `JG87919`-type case illustrates why Stage 1 should not be treated as a final judgment. Its seed expansion produced 15 members, 34 internal coordination edges, and 20 shared negative targets. At first glance, the group appears highly coordinated by target overlap. However, manual inspection and temporal analysis suggest that it is more likely a group of independent users with similar ideology rather than a shared operator group. This supports the core design decision: co-negative-target expansion is useful for candidate discovery, but temporal verification or manual review is needed before interpreting the group as high-risk coordination.

### 5.4 Signal Pruning

The project tested text fingerprint and account lifecycle overlap as additional verification signals, but both were removed from formal evidence. Text fingerprint was too sensitive to shared cryptocurrency vocabulary, while lifecycle overlap could not distinguish same-operator-like accounts from independent users in the same community. Removing these noisy signals made the formal pipeline clearer: Stage 1 discovers candidate coordination groups, and Stage 2 verifies short-window temporal synchrony.

## 6. Discussion

The main strength of the proposed method is interpretability. MCA ranking identifies suspicious entry points, the co-negative-target graph expands those seeds into candidate groups, and temporal synchrony provides a second-stage verification signal. This avoids presenting a single black-box suspiciousness score as a final verdict.

There are still limitations. First, the current project does not have large-scale ground truth, so the output should be interpreted as review candidates rather than supervised accuracy. Second, temporal synchrony can still be affected by popular threads and highly active users. The temporal confidence layer mitigates this problem but does not remove the need for human review. Third, the current dataset is centered on Reddit cryptocurrency discussions. Applying the system to other platforms would require redefining action traces such as reposts, link sharing, hashtags, likes, or reply chains.

The demo website is intentionally separated from the project pipeline. It provides a readable interface for clients, instructors, or non-technical audiences, but it does not define the research method. The source of truth remains the reproducible pipeline and its output tables.

## 7. Conclusion

This paper presents a suspicious social coordination detection pipeline that combines LLM-based content analysis, MCA seed ranking, graph-based seed expansion, and temporal synchrony verification. Compared with simple account ranking, the proposed method shifts the question from "which account looks suspicious" to "which accounts appear to act together." Preliminary results show that the system can reduce a large account population into reviewable candidate groups and further prioritize them using temporal evidence. Future work will expand cross-platform action traces, build a larger human-labeled validation set, and evaluate the stability of temporal thresholds across different communities and platforms.

## References

[1] E. Ferrara, O. Varol, C. Davis, F. Menczer, and A. Flammini, “The Rise of Social Bots,” *Communications of the ACM*, vol. 59, no. 7, pp. 96-104, 2016.

[2] O. Varol, E. Ferrara, C. Davis, F. Menczer, and A. Flammini, “Online Human-Bot Interactions: Detection, Estimation, and Characterization,” *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 11, no. 1, pp. 280-289, 2017.

[3] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi, “Social Fingerprinting: Detection of Spambot Groups Through DNA-Inspired Behavioral Modeling,” *IEEE Transactions on Dependable and Secure Computing*, vol. 15, no. 4, pp. 561-576, 2018.

[4] M. Mazza, S. Cresci, M. Avvenuti, W. Quattrociocchi, and M. Tesconi, “RTbust: Exploiting Temporal Patterns for Botnet Detection on Twitter,” arXiv:1902.04506, 2019.

[5] D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, and F. Menczer, “Uncovering Coordinated Networks on Social Media: Methods and Case Studies,” *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 15, no. 1, pp. 455-466, 2021.

[6] T. Graham, S. Hames, and E. Alpert, “The Coordination Network Toolkit: A Framework for Detecting and Analysing Coordinated Behaviour on Social Media,” *Journal of Computational Social Science*, vol. 7, pp. 1139-1160, 2024.

[7] M. H. Saeed, S. Ali, J. Blackburn, E. De Cristofaro, S. Zannettou, and G. Stringhini, “TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit,” arXiv:2112.00443, 2021.

[8] S. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, and C. Cherry, “SemEval-2016 Task 6: Detecting Stance in Tweets,” *Proceedings of the 10th International Workshop on Semantic Evaluation*, pp. 31-41, 2016.
