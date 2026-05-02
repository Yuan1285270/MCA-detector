# Data Cleaning

This folder owns the shared data preparation step before model analysis.

## Flow

```text
source_data/reddit_posts_2025.csv
    -> preprocess_posts.py
    -> processed_data/processed_data.csv

source_data/reddit_comments_2025.csv
    + processed_data/processed_data.csv
    -> preprocess_comments.py
    -> processed_data/processed_comments.csv
```

The processed CSV files are shared inputs for both:

- `../gemini-cloud/`
- `../ollama-local/`

## Exploratory Analysis

- `post-clustering/` contains post-level clustering based on final analyzed outputs.
- It is exploratory and not the formal account-level suspicious group pipeline.

