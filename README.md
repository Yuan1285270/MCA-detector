# MCA-detector

Collaborative capstone repository for detecting suspicious cryptocurrency social media activity.

## Modules

- `behavior/`
  - Behavior analysis module.
  - See `behavior/README.md` for details.

- `llm/`
  - LLM-based Reddit post/comment analysis pipeline.
  - Includes data cleaning, Gemini / Vertex AI batch analysis, local Ollama experiments, and exploratory post clustering.
  - See `llm/README.md` for details.

## Branch Workflow

- `main`: shared stable baseline
- `feature/llm-data-pipeline`: LLM data pipeline work
- `terry-behavior`: behavior analysis work

Feature branches should be merged back into `main` after review.

