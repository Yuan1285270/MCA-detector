from pathlib import Path

# =========================
# Base Paths
# =========================
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

# =========================
# Raw Files
# =========================
RAW_POSTS_PATH = RAW_DIR / "reddit_posts_2025.csv"
RAW_COMMENTS_PATH = RAW_DIR / "reddit_comments_2025.csv"
RAW_BTC_PATH = RAW_DIR / "BTCUSDT_1h_close_2025.csv"

# =========================
# Processed Files
# =========================
CLEAN_POSTS_PATH = PROCESSED_DIR / "cleaned_posts_2025.csv"
CLEAN_COMMENTS_PATH = PROCESSED_DIR / "cleaned_comments_2025.csv"
MERGED_REDDIT_PATH = PROCESSED_DIR / "merged_reddit_2025.csv"

# =========================
# Output Tables
# =========================
BEHAVIOR_FEATURES_PATH = TABLE_DIR / "behavior_features.csv"
SUSPICIOUS_BEHAVIOR_PATH = TABLE_DIR / "suspicious_accounts_behavior.csv"

DIRECTIONAL_COMMENTS_PATH = TABLE_DIR / "directional_comments_with_returns.csv"
ACCOUNT_PERFORMANCE_PATH = TABLE_DIR / "account_prediction_performance.csv"
TOP_PREDICTIVE_PATH = TABLE_DIR / "top_predictive_accounts.csv"
TOP_CONTRARIAN_PATH = TABLE_DIR / "top_contrarian_accounts.csv"