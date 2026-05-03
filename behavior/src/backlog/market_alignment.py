import os
import pandas as pd
import numpy as np

# =========================
# 路徑設定
# =========================
from config import MERGED_REDDIT_PATH, RAW_BTC_PATH, DIRECTIONAL_COMMENTS_PATH
from io_utils import ensure_dirs, read_csv, save_csv
from standardize import standardize_btc

OUTPUT_DIR = "output"
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

DIRECTIONAL_COMMENTS_PATH = os.path.join(TABLE_DIR, "directional_comments_with_returns.csv")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def load_comments():
    df = read_csv(MERGED_REDDIT_PATH)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[df["created_datetime"].notna()].copy()

    df["author"] = df["author"].astype(str).str.strip()
    invalid_authors = ["[deleted]", "[removed]", "nan", "None", ""]
    df = df[~df["author"].isin(invalid_authors)].copy()

    df["body"] = df["body"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["body"] != ""].copy()

    invalid_bodies = ["[deleted]", "[removed]"]
    df = df[~df["body"].isin(invalid_bodies)].copy()

    return df


def load_btc():
    btc = read_csv(RAW_BTC_PATH)
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce", utc=True)
    btc = btc[btc["timestamp"].notna()].copy()

    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    btc = btc[btc["close"].notna()].copy()

    btc = btc.sort_values("timestamp").reset_index(drop=True)
    return btc


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def label_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bullish_keywords = [
        "bullish", "going up", "will go up", "will rise", "buy now",
        "to the moon", "moon", "pump", "price target", "going to hit",
        "breakout", "higher", "new high", "100k", "200k", "long btc"
    ]

    bearish_keywords = [
        "bearish", "going down", "will drop", "will crash", "sell now",
        "dump", "lower", "correction", "pullback", "new low", "short btc"
    ]

    df["is_bullish"] = df["body"].apply(lambda x: contains_any(x, bullish_keywords)).astype(int)
    df["is_bearish"] = df["body"].apply(lambda x: contains_any(x, bearish_keywords)).astype(int)

    # 互斥化：同時命中的先排除，避免方向不清
    df["direction"] = np.select(
        [
            (df["is_bullish"] == 1) & (df["is_bearish"] == 0),
            (df["is_bearish"] == 1) & (df["is_bullish"] == 0),
        ],
        [
            "bullish",
            "bearish",
        ],
        default="neutral"
    )

    df = df[df["direction"].isin(["bullish", "bearish"])].copy()
    return df


def align_comments_with_btc(comments: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    comments = comments.copy()
    btc = btc.copy()

    # 先把留言時間 floor 到小時
    comments["comment_hour"] = comments["created_datetime"].dt.floor("h")

    # 對齊當下價格
    merged = comments.merge(
        btc.rename(columns={"timestamp": "comment_hour", "close": "price_t0"}),
        on="comment_hour",
        how="left"
    )

    # 建立未來時間點
    horizons = {
        "1h": 1,
        "24h": 24,
        "72h": 72,
        "168h": 168
    }

    for label, hours in horizons.items():
        future_col = f"future_time_{label}"
        price_col = f"price_t_{label}"

        merged[future_col] = merged["comment_hour"] + pd.to_timedelta(hours, unit="h")

        merged = merged.merge(
            btc.rename(columns={"timestamp": future_col, "close": price_col}),
            on=future_col,
            how="left"
        )

        merged[f"return_{label}"] = (
            (merged[price_col] - merged["price_t0"]) / merged["price_t0"]
        )

    return merged


def label_prediction_result(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    df = df.copy()

    horizons = ["1h", "24h", "72h", "168h"]

    for h in horizons:
        ret_col = f"return_{h}"
        result_col = f"result_{h}"

        conditions = [
            (df["direction"] == "bullish") & (df[ret_col] > threshold),
            (df["direction"] == "bearish") & (df[ret_col] < -threshold),
            (df["direction"] == "bullish") & (df[ret_col] < -threshold),
            (df["direction"] == "bearish") & (df[ret_col] > threshold),
        ]
        choices = [
            "success",
            "success",
            "contrarian",
            "contrarian",
        ]

        df[result_col] = np.select(conditions, choices, default="no_signal")

    return df


def main():
    ensure_dirs()

    print("讀取留言資料中...")
    comments = load_comments()
    print("留言資料大小:", comments.shape)

    print("\n讀取 BTC 資料中...")
    btc = load_btc()
    print("BTC 資料大小:", btc.shape)

    print("\n標記方向性留言中...")
    directional = label_direction(comments)
    print("方向性留言數量:", directional.shape)

    print("\n進行市場價格對齊中...")
    aligned = align_comments_with_btc(directional, btc)
    print("對齊後資料大小:", aligned.shape)

    print("\n標記預測結果中...")
    aligned = label_prediction_result(aligned, threshold=0.01)

    save_csv(aligned, DIRECTIONAL_COMMENTS_PATH)

    print("\n已輸出：")
    print(DIRECTIONAL_COMMENTS_PATH)

    print("\n欄位預覽：")
    print(aligned[[
        "author", "created_datetime", "direction",
        "price_t0", "price_t_1h", "price_t_24h",
        "return_1h", "return_24h",
        "result_1h", "result_24h"
    ]].head(10))

    print("\n=== Done ===")


if __name__ == "__main__":
    main()