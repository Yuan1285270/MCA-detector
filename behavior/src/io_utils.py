import pandas as pd
from pathlib import Path
from config import OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, PROCESSED_DIR

def ensure_dirs():
    for path in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, PROCESSED_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)

def read_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")