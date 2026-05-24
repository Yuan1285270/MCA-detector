# Literature Audit for MCA Detector Paper

This note verifies every cited work currently used in
`paper/mca_detector_ieee_expanded_en.tex`. The goal is to prevent overclaiming
and to keep the Related Work section defensible.

## 1. Ferrara et al., 2016 -- The Rise of Social Bots

- Source checked: arXiv record and CACM bibliographic information.
- Verified content: The paper reviews modern social bots, their risks to
  online ecosystems, and detection approaches based on content, network,
  sentiment, and temporal activity features.
- Current use in paper: Used to motivate early account-level social bot
  detection and the risk of social bots in political, economic, and public
  discussions.
- Verdict: Safe. The citation supports the current sentence.
- Source: https://arxiv.org/abs/1407.5225

## 2. Varol et al., 2017 -- Online Human-Bot Interactions

- Source checked: arXiv record and ICWSM bibliographic information.
- Verified content: The paper presents a Twitter bot-detection framework using
  over one thousand features, including friends, tweet content, sentiment,
  network patterns, and activity time series. It estimates active Twitter bot
  prevalence and characterizes interactions among humans and bots.
- Current use in paper: Used to represent account-level bot detection with
  metadata, content, sentiment, network, friend, and temporal features.
- Verdict: Safe. The citation supports the current sentence.
- Source: https://arxiv.org/abs/1703.03107

## 3. Cresci et al., 2018 -- Social Fingerprinting

- Source checked: arXiv record and DTU/IEEE bibliographic information.
- Verified content: The paper models account behavior as digital DNA sequences
  and uses collective behavioral similarity to detect spambot groups in both
  supervised and unsupervised settings.
- Current use in paper: Used to motivate group-level behavior modeling rather
  than only individual account classification.
- Verdict: Safe. The citation supports the current sentence.
- Source: https://arxiv.org/abs/1703.04482

## 4. Mazza et al., 2019 -- RTbust

- Source checked: arXiv record.
- Verified content: The paper studies retweeting behavior on Twitter, converts
  retweet time series into latent vectors using an LSTM autoencoder, clusters
  accounts, and identifies botnets characterized by suspicious retweet temporal
  patterns.
- Current use in paper: Used to motivate temporal group evidence for botnet
  detection.
- Verdict: Safe, as long as we describe it specifically as Twitter retweet
  temporal-pattern botnet detection.
- Source: https://arxiv.org/abs/1902.04506

## 5. Pacheco et al., 2021 -- Uncovering Coordinated Networks

- Source checked: local PDF, arXiv record, and ICWSM bibliographic metadata.
- Local PDF:
  `Archive/Research Sources/Graph/Pacheco2021_Uncovering_Coordinated_Networks.pdf`
- Verified content: The paper proposes a general unsupervised network-based
  method that constructs coordination networks from arbitrary shared behavioral
  traces and evaluates the method through five social-media case studies.
- Current use in paper: Used to support the idea of building coordination
  networks from shared behavioral traces.
- Verdict: Safe. This is one of the closest methodological references for our
  multi-layer graph framing.
- Sources: https://arxiv.org/abs/2001.05658 and
  https://ojs.aaai.org/index.php/ICWSM/article/view/18075

## 6. Graham et al., 2024 -- Coordination Network Toolkit

- Source checked: local PDF and Journal of Computational Social Science
  bibliographic record.
- Local PDF:
  `Archive/Research Sources/Graph/Graham2024_Coordination_Network_Toolkit.pdf`
- Verified content: The paper introduces an open-source toolkit and
  methodological framework for constructing coordination networks across
  multiple social-media behaviors using weighted, directed multigraphs.
- Current use in paper: Used to support multi-behavior coordination-network
  analysis.
- Verdict: Safe after wording correction. The paper text was updated from
  "weighted directed multi-behavior networks" to the more precise "weighted,
  directed multigraphs over multiple behavior traces."
- Source: https://doi.org/10.1007/s42001-024-00260-z

## 7. Mohammad et al., 2016 -- SemEval-2016 Task 6

- Source checked: ACL Anthology.
- Verified content: The shared task defines stance detection in tweets: given a
  tweet and target, systems infer whether the tweet is in favor, against, or
  neither with respect to the target.
- Current use in paper: Used to justify target-aware stance analysis rather
  than plain sentiment analysis.
- Verdict: Safe. The citation supports the current sentence.
- Source: https://aclanthology.org/S16-1003/

## 8. Saeed et al., 2022 -- TROLLMAGNIFIER

- Source checked: local PDF, user-provided PDF, and arXiv record.
- Local PDF:
  `Archive/Research Sources/Graph/Saeed2022_TROLLMAGNIFIER_Reddit_Troll_Detection.pdf`
- Verified content: The paper uses 335 known Russian-sponsored troll accounts
  identified by Reddit, trains a detection system, identifies 1,248 potential
  troll accounts, and corroborates results with indicators such as account
  status/creation patterns, topic similarity, and temporal synchronization. The
  paper reports that 66% of detected accounts show signs of being instrumented
  by malicious actors.
- Current use in paper: Used as the closest Reddit-specific supervised troll
  detection comparison.
- Verdict: Safe after wording correction. The paper text was updated from
  "supervised seed data" and "interaction features around known troll accounts"
  to "labeled training data" and "learns the behavior of known troll accounts."
- Source: https://arxiv.org/abs/2112.00443

## Overall conclusion

The Related Work citations are defensible after two wording corrections:

1. Graham et al. should be described as weighted, directed multigraph-based
   coordination analysis.
2. TROLLMAGNIFIER should be described as supervised/labeled Reddit troll
   detection, not as the same kind of seed-expansion pipeline as our project.

No cited work currently needs to be removed.

## Local archive coverage

The local `Archive/Research Sources` folder contains related notes and several
useful full papers, but it does not contain the full PDF for every citation
currently used in the IEEE paper.

| Citation in paper | Local archive status | Note |
|---|---|---|
| Ferrara et al. 2016 | Mentioned in `Archive/Research Sources/LLM/bot-detection-literature-review-final.md` and `rhetorical_research.txt` | No full Ferrara PDF found in archive. |
| Varol et al. 2017 | Mentioned in `Archive/Research Sources/LLM/rhetorical_research.txt` | No full Varol PDF found in archive. |
| Cresci et al. 2018 | Mentioned in `Archive/Research Sources/LLM/bot-detection-literature-review-final.md` | No full Social Fingerprinting PDF found in archive. The note cites the WWW 2017 version; our paper cites the IEEE TDSC 2018 journal version. |
| Mazza et al. 2019 | Mentioned in `Archive/Research Sources/LLM/bot-detection-literature-review-final.md` | No full RTbust PDF found in archive. |
| Pacheco et al. 2021 | Full PDF now saved as `Archive/Research Sources/Graph/Pacheco2021_Uncovering_Coordinated_Networks.pdf` | External links: arXiv `https://arxiv.org/abs/2001.05658`; ICWSM/AAAI `https://ojs.aaai.org/index.php/ICWSM/article/view/18075`. |
| Graham et al. 2024 | Full PDF now saved as `Archive/Research Sources/Graph/Graham2024_Coordination_Network_Toolkit.pdf` | External link: Springer DOI `https://doi.org/10.1007/s42001-024-00260-z`. |
| Mohammad et al. 2016 | Not found in archive | Important: `Archive/Research Sources/LLM/2020.semeval-1.186.pdf` is SemEval-2020 propaganda-technique detection, not SemEval-2016 stance detection. |
| Saeed et al. 2022 | Full PDF now saved as `Archive/Research Sources/Graph/Saeed2022_TROLLMAGNIFIER_Reddit_Troll_Detection.pdf` | External link: arXiv `https://arxiv.org/abs/2112.00443`. |

Additional archive papers that are relevant but not currently cited include
Weber and Neumann's coordinating-community papers, Iannucci et al.'s temporal
multiplex coordination work, and Alizadeh et al.'s content-based influence
operations paper. These can be added if the paper needs a stronger coordination
or influence-operations related-work section, but adding them will cost space.
