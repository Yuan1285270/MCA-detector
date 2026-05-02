from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd


INPUT_PATH = "source_data/reddit_posts_2025.csv"
OUTPUT_PATH = "processed_data/processed_data.csv"
BOT_OUTPUT_DIR = "temp/bot_candidates"

DROP_SELF_TEXT_VALUES = {"", "[removed]", "[deleted]", "nan", "none"}
MIN_TEXT_LENGTH = 20
BOT_IGNORE_ACCOUNT_VALUES = {"", "[removed]", "[deleted]", "nan", "none"}
BOT_COMPARE_LATEST_N = 10
BOT_MIN_POSTS_PER_ACCOUNT = 5
BOT_SIMILARITY_THRESHOLD = 0.70

BOT_POSTS_OUTPUT_PATH = "functional_bot_posts.csv"
BOT_SUMMARY_OUTPUT_PATH = "functional_bot_account_summary.csv"
BOT_RECENT_POSTS_OUTPUT_PATH = "functional_bot_recent_check_posts.csv"


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_body_text(text: object) -> str:
    cleaned = normalize_text(text)
    if cleaned.lower() in DROP_SELF_TEXT_VALUES:
        return ""
    return cleaned


def clean_title_text(text: object) -> str:
    cleaned = normalize_text(text)
    if cleaned.lower() in {"[removed]", "[deleted]", "nan", "none"}:
        return ""
    return cleaned


def build_analysis_text(title: str, body: str) -> str:
    if title and body:
        if title.lower() == body.lower():
            return body
        return f"{title}\n\n{body}"

    return body or title


def similarity_text(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def pairwise_similarity_stats(texts: list[str]) -> tuple[float, float, float]:
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = SequenceMatcher(None, texts[i], texts[j]).ratio()
            scores.append(score)

    if not scores:
        return 0.0, 0.0, 0.0

    return sum(scores) / len(scores), min(scores), max(scores)


def identify_functional_bot_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["account_id"] = working["author"].apply(normalize_text)
    working = working[~working["account_id"].str.lower().isin(BOT_IGNORE_ACCOUNT_VALUES)].copy()

    working["title_clean"] = working["title"].apply(clean_title_text)
    working["selftext_clean"] = working["selftext"].apply(clean_body_text)
    working["analysis_text"] = working.apply(
        lambda row: build_analysis_text(row["title_clean"], row["selftext_clean"]),
        axis=1,
    )
    working["analysis_text"] = working["analysis_text"].apply(normalize_text)
    working["similarity_text"] = working["analysis_text"].apply(similarity_text)

    flagged_accounts = []
    flagged_account_ids = set()
    latest_post_ids_by_account: dict[str, set[str]] = {}
    latest_post_rank_by_account: dict[str, dict[str, int]] = {}

    for account_id, group in working.groupby("account_id", sort=False):
        if len(group) < BOT_MIN_POSTS_PER_ACCOUNT:
            continue

        latest_posts = group.sort_values("created_utc", ascending=False, kind="stable").head(BOT_COMPARE_LATEST_N).copy()
        if len(latest_posts) < BOT_MIN_POSTS_PER_ACCOUNT:
            continue

        texts = latest_posts["similarity_text"].tolist()
        avg_similarity, min_similarity, max_similarity = pairwise_similarity_stats(texts)

        if avg_similarity < BOT_SIMILARITY_THRESHOLD:
            continue

        flagged_account_ids.add(account_id)
        latest_post_ids = latest_posts["post_id"].astype(str).tolist()
        latest_post_ids_by_account[account_id] = set(latest_post_ids)
        latest_post_rank_by_account[account_id] = {
            post_id: rank for rank, post_id in enumerate(latest_post_ids, start=1)
        }
        flagged_accounts.append(
            {
                "author": account_id,
                "total_posts_by_account": int(len(group)),
                "compared_post_count": int(len(latest_posts)),
                "avg_similarity": round(avg_similarity, 6),
                "min_similarity": round(min_similarity, 6),
                "max_similarity": round(max_similarity, 6),
                "latest_post_ids": " || ".join(latest_posts["post_id"].astype(str).tolist()),
                "latest_titles": " || ".join(latest_posts["title"].fillna("").astype(str).tolist()),
            }
        )

    summary_df = pd.DataFrame(flagged_accounts)

    if not flagged_account_ids:
        return working.iloc[0:0].copy(), summary_df

    flagged_posts = working[working["account_id"].isin(flagged_account_ids)].copy()

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["avg_similarity", "total_posts_by_account", "author"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    summary_map = {
        row["author"]: row
        for _, row in summary_df.iterrows()
    }
    flagged_posts["avg_similarity"] = flagged_posts.apply(
        lambda row: summary_map[row["account_id"]]["avg_similarity"],
        axis=1,
    )
    flagged_posts["total_posts_by_account"] = flagged_posts.apply(
        lambda row: summary_map[row["account_id"]]["total_posts_by_account"],
        axis=1,
    )
    flagged_posts["compared_post_count"] = flagged_posts.apply(
        lambda row: summary_map[row["account_id"]]["compared_post_count"],
        axis=1,
    )
    flagged_posts["similarity_threshold"] = BOT_SIMILARITY_THRESHOLD
    flagged_posts["used_for_similarity_check"] = flagged_posts.apply(
        lambda row: str(row["post_id"]) in latest_post_ids_by_account[row["account_id"]],
        axis=1,
    )
    flagged_posts["similarity_check_rank"] = flagged_posts.apply(
        lambda row: latest_post_rank_by_account[row["account_id"]].get(str(row["post_id"])),
        axis=1,
    )
    flagged_posts["similarity_check_rank"] = flagged_posts["similarity_check_rank"].astype("Int64")

    flagged_posts = flagged_posts.sort_values(
        ["account_id", "created_utc"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    return flagged_posts, summary_df


def main() -> None:
    input_path = Path(__file__).resolve().parent / INPUT_PATH
    output_path = Path(__file__).resolve().parent / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bot_output_dir = Path(__file__).resolve().parent / BOT_OUTPUT_DIR
    bot_output_dir.mkdir(parents=True, exist_ok=True)
    bot_posts_output_path = bot_output_dir / BOT_POSTS_OUTPUT_PATH
    bot_summary_output_path = bot_output_dir / BOT_SUMMARY_OUTPUT_PATH
    bot_recent_posts_output_path = bot_output_dir / BOT_RECENT_POSTS_OUTPUT_PATH

    df = pd.read_csv(input_path)

    required_columns = {"post_id", "author", "created_utc", "title", "selftext", "num_comments"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    functional_bot_posts, functional_bot_summary = identify_functional_bot_posts(df)
    bot_output_columns = [
        "author",
        "post_id",
        "created_utc",
        "title",
        "selftext",
        "analysis_text",
        "avg_similarity",
        "total_posts_by_account",
        "compared_post_count",
        "used_for_similarity_check",
        "similarity_check_rank",
        "num_comments",
    ]
    bot_recent_posts_columns = [
        "author",
        "similarity_check_rank",
        "post_id",
        "created_utc",
        "title",
        "selftext",
        "analysis_text",
        "avg_similarity",
        "total_posts_by_account",
        "compared_post_count",
        "num_comments",
    ]
    bot_summary_columns = [
        "author",
        "total_posts_by_account",
        "compared_post_count",
        "avg_similarity",
        "min_similarity",
        "max_similarity",
        "latest_post_ids",
        "latest_titles",
    ]

    if functional_bot_posts.empty:
        pd.DataFrame(columns=bot_output_columns).to_csv(
            bot_posts_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        functional_bot_posts[bot_output_columns].to_csv(
            bot_posts_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    if functional_bot_posts.empty:
        pd.DataFrame(columns=bot_recent_posts_columns).to_csv(
            bot_recent_posts_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        recent_posts_df = functional_bot_posts[functional_bot_posts["used_for_similarity_check"]].copy()
        recent_posts_df["similarity_check_rank"] = recent_posts_df["similarity_check_rank"].astype(int)
        recent_posts_df = recent_posts_df.sort_values(
            ["account_id", "similarity_check_rank"],
            ascending=[True, True],
            kind="stable",
        )
        recent_posts_df[bot_recent_posts_columns].to_csv(
            bot_recent_posts_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    if functional_bot_summary.empty:
        pd.DataFrame(columns=bot_summary_columns).to_csv(
            bot_summary_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        functional_bot_summary[bot_summary_columns].to_csv(
            bot_summary_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    if functional_bot_posts.empty:
        filtered_df = df.copy()
    else:
        excluded_post_ids = set(functional_bot_posts["post_id"].astype(str))
        filtered_df = df[~df["post_id"].astype(str).isin(excluded_post_ids)].copy()

    cleaned = filtered_df.copy()
    cleaned["title_clean"] = cleaned["title"].apply(clean_title_text)
    cleaned["selftext_clean"] = cleaned["selftext"].apply(clean_body_text)
    cleaned["analysis_text"] = cleaned.apply(
        lambda row: build_analysis_text(row["title_clean"], row["selftext_clean"]),
        axis=1,
    )
    cleaned["analysis_text"] = cleaned["analysis_text"].apply(normalize_text)
    cleaned["analysis_char_len"] = cleaned["analysis_text"].str.len()

    cleaned = cleaned[cleaned["analysis_text"] != ""].copy()
    cleaned = cleaned[cleaned["analysis_char_len"] >= MIN_TEXT_LENGTH].copy()

    # Keep the first occurrence for duplicated post ids.
    cleaned = cleaned.drop_duplicates(subset=["post_id"], keep="first")

    # Remove exact duplicate analysis texts after normalization.
    cleaned = cleaned.drop_duplicates(subset=["analysis_text"], keep="first")

    cleaned = cleaned.sort_values("created_utc", kind="stable").reset_index(drop=True)

    output_columns = [
        "post_id",
        "author",
        "created_utc",
        "title",
        "selftext",
        "analysis_text",
        "analysis_char_len",
        "num_comments",
    ]

    cleaned[output_columns].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned data to {output_path}")
    print(f"Rows: {len(cleaned)}")
    print(f"Functional bot posts saved to {bot_posts_output_path}")
    print(f"Functional bot recent comparison posts saved to {bot_recent_posts_output_path}")
    print(f"Functional bot account summary saved to {bot_summary_output_path}")
    print(f"Functional bot posts removed before main cleaning: {len(functional_bot_posts)}")


if __name__ == "__main__":
    main()
