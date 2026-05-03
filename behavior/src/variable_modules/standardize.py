import pandas as pd
from column_mapping import POST_SOURCE_COLUMNS, COMMENT_SOURCE_COLUMNS, BTC_SOURCE_COLUMNS

INVALID_AUTHORS = ["[deleted]", "[removed]", "nan", "None", ""]
INVALID_BODIES = ["[deleted]", "[removed]", ""]


def standardize_posts(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = df.rename(columns={
        POST_SOURCE_COLUMNS["post_id"]: "post_id",
        POST_SOURCE_COLUMNS["author"]: "author",
        POST_SOURCE_COLUMNS["created_utc"]: "created_utc",
        POST_SOURCE_COLUMNS["title"]: "title",
        POST_SOURCE_COLUMNS["selftext"]: "selftext",
        POST_SOURCE_COLUMNS["num_comments"]: "num_comments",
    })

    df["author"] = df["author"].astype(str).str.strip()
    df = df[~df["author"].isin(INVALID_AUTHORS)]

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""]

    df["selftext"] = df["selftext"].fillna("").astype(str).str.strip()
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).astype(int)

    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df[df["created_utc"].notna()]
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["created_datetime"].notna()]

    df = df.drop_duplicates(subset=["post_id"]).sort_values("created_datetime").reset_index(drop=True)
    return df


def standardize_comments(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = df.rename(columns={
        COMMENT_SOURCE_COLUMNS["comment_id"]: "comment_id",
        COMMENT_SOURCE_COLUMNS["link_id"]: "link_id",
        COMMENT_SOURCE_COLUMNS["parent_id"]: "parent_id",
        COMMENT_SOURCE_COLUMNS["author"]: "author",
        COMMENT_SOURCE_COLUMNS["body"]: "body",
        COMMENT_SOURCE_COLUMNS["created_utc"]: "created_utc",
    })

    df["author"] = df["author"].astype(str).str.strip()
    df = df[~df["author"].isin(INVALID_AUTHORS)]

    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df = df[~df["body"].isin(INVALID_BODIES)]

    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df[df["created_utc"].notna()]
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["created_datetime"].notna()]

    df["link_id"] = df["link_id"].fillna("").astype(str).str.strip()
    df["parent_id"] = df["parent_id"].fillna("").astype(str).str.strip()
    df["post_id"] = df["link_id"].str.replace("t3_", "", regex=False)

    df = df.drop_duplicates(subset=["comment_id"]).sort_values("created_datetime").reset_index(drop=True)
    return df


def standardize_btc(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = df.rename(columns={
        BTC_SOURCE_COLUMNS["timestamp"]: "timestamp",
        BTC_SOURCE_COLUMNS["close"]: "close",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["timestamp"].notna()]

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"].notna()]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df