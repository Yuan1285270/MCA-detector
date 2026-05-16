# MCA-detector

Collaborative capstone repository for detecting suspicious cryptocurrency social media accounts that may coordinate to steer public opinion.

The project does not claim that an account is a confirmed bot, paid operator, or real-world cyber troop. It produces evidence-backed review candidates based on manipulative rhetoric, coordination proxies, interaction reach, and automation signals.

## Modules

- `behavior/`
  - Behavior analysis module.
  - See `behavior/README.md` for details.

- `llm/`
  - LLM-based Reddit post/comment analysis pipeline.
  - Includes data cleaning, Gemini / Vertex AI batch analysis, local Ollama experiments, and exploratory post clustering.
  - See `llm/README.md` for details.

- `adjacency/`
  - Account-level graph construction module.
  - Builds single-graph and multi-graph sparse adjacency artifacts from LLM-analyzed Reddit comment feedback and account rhetoric features.
  - See `adjacency/README.md` for details.

- `mca-scoring/`
  - Account-level MCA review-priority scoring module.
  - Combines manipulative, coordinative, interaction reach, and automation signals.
  - See `mca-scoring/README.md` for details.

- `coordination-expansion/`
  - Seed expansion and group discovery module.
  - Turns graph layers into reviewable coordination evidence tables.
  - See `coordination-expansion/README.md` for details.

## Branch Workflow

- `main`: shared stable baseline
- `feature/llm-data-pipeline`: LLM data pipeline work
- `terry-behavior`: behavior analysis work

Feature branches should be merged back into `main` after review.
