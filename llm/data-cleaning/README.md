# Data Cleaning

This folder owns the shared data preparation step before model analysis.

## Flow

```text
source_data/reddit_posts_2025.csv
    -> preprocess_posts.py
    -> remove highly repetitive functional-bot-like posts
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

## Repetitive Account Filter

`preprocess_posts.py` checks whether an account repeatedly posts highly similar text.

Current rule:

- compare up to the latest 10 posts per account
- require at least 5 posts for an account to be checked
- flag accounts whose average pairwise normalized text similarity is at least 0.70
- write candidate records to `temp/bot_candidates/`
- exclude those posts from `processed_data/processed_data.csv`

This is a conservative preprocessing filter for highly repetitive or template-like accounts. It should not be presented as standalone proof that an account is a bot.
