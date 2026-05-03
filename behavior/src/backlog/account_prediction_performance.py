import os
import pandas as pd
import numpy as np

# =========================
# 路徑設定
# =========================
from config import (
    DIRECTIONAL_COMMENTS_PATH,
    ACCOUNT_PERFORMANCE_PATH,
    TOP_PREDICTIVE_PATH,
    TOP_CONTRARIAN_PATH,
)
from io_utils import ensure_dirs, read_csv, save_csv

OUTPUT_DIR = "output"
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

ACCOUNT_PERFORMANCE_PATH = os.path.join(TABLE_DIR, "account_prediction_performance.csv")
TOP_PREDICTIVE_PATH = os.path.join(TABLE_DIR, "top_predictive_accounts.csv")
TOP_CONTRARIAN_PATH = os.path.join(TABLE_DIR, "top_contrarian_accounts.csv")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def load_data():
    df = read_csv(DIRECTIONAL_COMMENTS_PATH)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[df["created_datetime"].notna()].copy()
    return df


def add_result_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    horizons = ["1h", "24h", "72h", "168h"]

    for h in horizons:
        result_col = f"result_{h}"

        df[f"success_{h}_flag"] = (df[result_col] == "success").astype(int)
        df[f"contrarian_{h}_flag"] = (df[result_col] == "contrarian").astype(int)
        df[f"no_signal_{h}_flag"] = (df[result_col] == "no_signal").astype(int)

    return df


def build_account_performance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    grouped = df.groupby("author").agg(
        directional_comment_count=("comment_id", "count"),
        bullish_count=("direction", lambda x: (x == "bullish").sum()),
        bearish_count=("direction", lambda x: (x == "bearish").sum()),
        first_comment_time=("created_datetime", "min"),
        last_comment_time=("created_datetime", "max")
    ).reset_index()

    grouped["active_days_in_directional_comments"] = (
        (grouped["last_comment_time"] - grouped["first_comment_time"]).dt.days + 1
    ).clip(lower=1)

    grouped["directional_comments_per_day"] = (
        grouped["directional_comment_count"] / grouped["active_days_in_directional_comments"]
    )

    horizons = ["1h", "24h", "72h", "168h"]

    for h in horizons:
        temp = df.groupby("author").agg(
            **{
                f"success_{h}_count": (f"success_{h}_flag", "sum"),
                f"contrarian_{h}_count": (f"contrarian_{h}_flag", "sum"),
                f"no_signal_{h}_count": (f"no_signal_{h}_flag", "sum"),
                f"avg_return_{h}": (f"return_{h}", "mean"),
                f"median_return_{h}": (f"return_{h}", "median"),
            }
        ).reset_index()

        grouped = grouped.merge(temp, on="author", how="left")

        valid_base = (
            grouped[f"success_{h}_count"] +
            grouped[f"contrarian_{h}_count"]
        )

        grouped[f"valid_signal_{h}_count"] = valid_base

        grouped[f"success_{h}_rate"] = np.where(
            valid_base > 0,
            grouped[f"success_{h}_count"] / valid_base,
            0
        )

        grouped[f"contrarian_{h}_rate"] = np.where(
            valid_base > 0,
            grouped[f"contrarian_{h}_count"] / valid_base,
            0
        )

        grouped[f"no_signal_{h}_rate"] = np.where(
            grouped["directional_comment_count"] > 0,
            grouped[f"no_signal_{h}_count"] / grouped["directional_comment_count"],
            0
        )

    grouped = grouped.rename(columns={"author": "user_id"})
    return grouped


def build_rankings(account_perf: pd.DataFrame):
    df = account_perf.copy()

    # 只看樣本數夠的帳號，避免 1 留言 100% 準確這種幻覺
    min_directional_comments = 5
    min_valid_signals = 3

    predictive_24h = df[
        (df["directional_comment_count"] >= min_directional_comments) &
        (df["valid_signal_24h_count"] >= min_valid_signals)
    ].copy()

    predictive_24h["predictive_score_24h"] = (
        predictive_24h["success_24h_rate"] * 4 +
        predictive_24h["valid_signal_24h_count"] * 0.1 +
        predictive_24h["directional_comment_count"] * 0.05
    )

    predictive_24h = predictive_24h.sort_values(
        ["predictive_score_24h", "success_24h_rate", "valid_signal_24h_count"],
        ascending=[False, False, False]
    )

    contrarian_24h = df[
        (df["directional_comment_count"] >= min_directional_comments) &
        (df["valid_signal_24h_count"] >= min_valid_signals)
    ].copy()

    contrarian_24h["contrarian_score_24h"] = (
        contrarian_24h["contrarian_24h_rate"] * 4 +
        contrarian_24h["valid_signal_24h_count"] * 0.1 +
        contrarian_24h["directional_comment_count"] * 0.05
    )

    contrarian_24h = contrarian_24h.sort_values(
        ["contrarian_score_24h", "contrarian_24h_rate", "valid_signal_24h_count"],
        ascending=[False, False, False]
    )

    return predictive_24h, contrarian_24h


def save_outputs(account_perf: pd.DataFrame, predictive: pd.DataFrame, contrarian: pd.DataFrame):
    save_csv(account_perf, ACCOUNT_PERFORMANCE_PATH)
    save_csv(predictive, TOP_PREDICTIVE_PATH)
    save_csv(contrarian, TOP_CONTRARIAN_PATH)

    print("\n已輸出：")
    print(ACCOUNT_PERFORMANCE_PATH)
    print(TOP_PREDICTIVE_PATH)
    print(TOP_CONTRARIAN_PATH)


def main():
    ensure_dirs()

    print("讀取逐留言預測結果中...")
    df = load_data()
    print("資料大小:", df.shape)

    print("\n建立結果旗標中...")
    df = add_result_flags(df)

    print("\n聚合帳號層級表現中...")
    account_perf = build_account_performance(df)
    print("帳號表現表大小:", account_perf.shape)

    print("\n建立排名中...")
    predictive, contrarian = build_rankings(account_perf)

    print("可納入 predictive ranking 的帳號數:", len(predictive))
    print("可納入 contrarian ranking 的帳號數:", len(contrarian))

    save_outputs(account_perf, predictive, contrarian)

    print("\n=== Top 10 Predictive Accounts (24h) ===")
    print(predictive[[
        "user_id",
        "directional_comment_count",
        "bullish_count",
        "bearish_count",
        "valid_signal_24h_count",
        "success_24h_count",
        "contrarian_24h_count",
        "success_24h_rate",
        "avg_return_24h",
        "predictive_score_24h"
    ]].head(10))

    print("\n=== Top 10 Contrarian Accounts (24h) ===")
    print(contrarian[[
        "user_id",
        "directional_comment_count",
        "bullish_count",
        "bearish_count",
        "valid_signal_24h_count",
        "success_24h_count",
        "contrarian_24h_count",
        "contrarian_24h_rate",
        "avg_return_24h",
        "contrarian_score_24h"
    ]].head(10))

    print("\n=== Done ===")


if __name__ == "__main__":
    main()