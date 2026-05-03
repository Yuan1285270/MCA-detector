# Post Clustering

This folder contains the post clustering pipeline for the final capstone outputs.

It builds a post-level feature matrix from:

1. Final Gemini post analysis scores and rhetoric tags
2. Comment feedback labels aggregated per post
3. Lightweight text and metadata features
4. K-Means clusters ranked by a review-oriented suspicion score

## Files

- `cluster_analyzed_posts.py`: builds final post features, runs K-Means, and writes cluster-level outputs
- `visualize_post_clusters.py`: writes an HTML visualization of the final clusters
- `../../../Archive/legacy_scripts/clean_posts.py`: legacy raw-post cleaning script kept for reproducibility, not used by the final clustering step
- `output/`: generated CSV and JSON outputs

## Default data flow

Inputs:
- `../../Export/reddit_posts_analyzed.csv`
- `../../Export/reddit_comments_analyzed.csv.gz.csv`

Outputs:
- `output/post_feature_matrix.csv`
- `output/post_clusters.csv`
- `output/cluster_summary.csv`
- `output/suspicious_clusters.csv`
- `output/cluster_config.json`
- `output/cluster_visualization.html`

## Run

```bash
../../../.venv/bin/python cluster_analyzed_posts.py
../../../.venv/bin/python visualize_post_clusters.py
```

## Notes

- The clustering step uses only packages already available in the repo environment: `pandas` and `numpy`
- Suspicious clusters are heuristic review targets, not final labels
- The final clustering should be regenerated whenever the files in `../../Export/` change
