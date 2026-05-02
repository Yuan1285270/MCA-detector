from pathlib import Path
import re

import pandas as pd


INPUT_PATH = "source_data/reddit_comments_2025.csv"
PROCESSED_POSTS_PATH = "processed_data/processed_data.csv"
OUTPUT_PATH = "processed_data/processed_comments.csv"
SUMMARY_OUTPUT_DIR = "temp/comment_cleaning"
SUMMARY_OUTPUT_PATH = "comment_cleaning_summary.csv"
COMMENT_REPLIES_OUTPUT_DIR = "temp/comment_replies"
COMMENT_REPLIES_OUTPUT_PATH = "comment_replies.csv"

DROP_BODY_VALUES = {"", "[removed]", "[deleted]", "nan", "none"}


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_body_text(text: object) -> str:
    cleaned = normalize_text(text)
    if cleaned.lower() in DROP_BODY_VALUES:
        return ""
    return cleaned


def strip_reddit_prefix(value: object, prefix: str) -> str:
    cleaned = normalize_text(value)
    if cleaned.startswith(prefix):
        return cleaned[len(prefix):]
    return cleaned


def parent_type(parent_id: object) -> str:
    parent = normalize_text(parent_id)
    if parent.startswith("t3_"):
        return "post"
    if parent.startswith("t1_"):
        return "comment"
    return "unknown"


def parent_comment_id(parent_id: object) -> str:
    parent = normalize_text(parent_id)
    if parent.startswith("t1_"):
        return parent[3:]
    return ""


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_PATH
    processed_posts_path = base_dir / PROCESSED_POSTS_PATH
    output_path = base_dir / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_dir = base_dir / SUMMARY_OUTPUT_DIR
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_path = summary_output_dir / SUMMARY_OUTPUT_PATH
    comment_replies_output_dir = base_dir / COMMENT_REPLIES_OUTPUT_DIR
    comment_replies_output_dir.mkdir(parents=True, exist_ok=True)
    comment_replies_output_path = comment_replies_output_dir / COMMENT_REPLIES_OUTPUT_PATH

    comments = pd.read_csv(input_path)
    processed_posts = pd.read_csv(processed_posts_path, usecols=["post_id"])

    required_columns = {"comment_id", "link_id", "parent_id", "author", "body", "created_utc"}
    missing = required_columns - set(comments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid_post_ids = set(processed_posts["post_id"].astype(str))

    working = comments.copy()
    original_rows = len(working)

    working["post_id"] = working["link_id"].apply(lambda value: strip_reddit_prefix(value, "t3_"))
    working["parent_type"] = working["parent_id"].apply(parent_type)
    working["parent_comment_id"] = working["parent_id"].apply(parent_comment_id)
    working["analysis_text"] = working["body"].apply(clean_body_text)
    working["analysis_text"] = working["analysis_text"].apply(normalize_text)
    working["analysis_char_len"] = working["analysis_text"].str.len()

    linked_to_removed_post = ~working["post_id"].astype(str).isin(valid_post_ids)
    invalid_body = working["analysis_text"] == ""
    cleaned = working[~linked_to_removed_post].copy()
    after_post_filter_rows = len(cleaned)

    cleaned = cleaned[cleaned["analysis_text"] != ""].copy()
    after_body_filter_rows = len(cleaned)

    cleaned = cleaned.drop_duplicates(subset=["comment_id"], keep="first")
    after_comment_id_dedup_rows = len(cleaned)

    comment_replies = cleaned[cleaned["parent_type"] == "comment"].copy()
    cleaned = cleaned[cleaned["parent_type"] == "post"].copy()
    after_direct_comment_filter_rows = len(cleaned)

    cleaned = cleaned.sort_values(["created_utc", "comment_id"], kind="stable").reset_index(drop=True)
    comment_replies = comment_replies.sort_values(
        ["created_utc", "comment_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_columns = [
        "comment_id",
        "post_id",
        "parent_id",
        "parent_type",
        "parent_comment_id",
        "author",
        "created_utc",
        "body",
        "analysis_text",
        "analysis_char_len",
    ]
    cleaned[output_columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    comment_replies[output_columns].to_csv(
        comment_replies_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = [
        {"metric": "original_rows", "value": original_rows},
        {"metric": "linked_to_removed_or_unprocessed_posts", "value": int(linked_to_removed_post.sum())},
        {"metric": "invalid_body_rows", "value": int(invalid_body.sum())},
        {"metric": "after_post_filter_rows", "value": after_post_filter_rows},
        {"metric": "after_body_filter_rows", "value": after_body_filter_rows},
        {
            "metric": "duplicate_comment_id_rows_removed_after_filters",
            "value": after_body_filter_rows - after_comment_id_dedup_rows,
        },
        {"metric": "comment_to_comment_replies_saved_aside", "value": len(comment_replies)},
        {"metric": "final_direct_comment_to_post_rows", "value": after_direct_comment_filter_rows},
        {"metric": "unique_linked_posts_final", "value": cleaned["post_id"].nunique()},
        {"metric": "unique_authors_final", "value": cleaned["author"].nunique(dropna=True)},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_output_path, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned comments to {output_path}")
    print(f"Rows: {len(cleaned)}")
    print(f"Comment-to-comment replies saved aside to {comment_replies_output_path}")
    print(f"Comment-to-comment reply rows: {len(comment_replies)}")
    print(f"Cleaning summary saved to {summary_output_path}")
    print(f"Original comment rows: {original_rows}")
    print(f"Comments linked to removed/unprocessed posts: {int(linked_to_removed_post.sum())}")
    print(f"Invalid comment bodies: {int(invalid_body.sum())}")
    print(f"Duplicate comment ids removed after filters: {after_body_filter_rows - after_comment_id_dedup_rows}")


if __name__ == "__main__":
    main()
