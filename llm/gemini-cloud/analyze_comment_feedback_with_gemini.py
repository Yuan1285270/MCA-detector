import argparse
from pathlib import Path
import json
import signal
import time
from typing import Any, Dict

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


# === Settings ===
PROJECT_ID = "capstone-492007"
LOCATION = "us-central1"
COMMENTS_CSV_PATH = "../data-cleaning/processed_data/processed_comments.csv"
POSTS_CSV_PATH = "../data-cleaning/processed_data/processed_data.csv"
OUTPUT_DIR = "output/comments"

# Batch range. END_ROW is exclusive.
DEFAULT_START_ROW = 0
DEFAULT_END_ROW = 167140
START_ROW = DEFAULT_START_ROW
END_ROW = DEFAULT_END_ROW
BATCH_NAME = f"{START_ROW}_{END_ROW}"
FULL_OUTPUT_PATH = f"{OUTPUT_DIR}/comment_feedback_{BATCH_NAME}.csv"

MODEL_NAME = "gemini-2.5-flash"
POST_MAX_CHARS = 1200
COMMENT_MAX_CHARS = 1200
SAVE_EVERY = 10
SLEEP_SECONDS = 0.5
RETRY_LIMIT = 5
REQUEST_TIMEOUT_SECONDS = 90

ALLOWED_LABELS = ["supportive", "oppositional", "neutral", "mixed", "unclear"]


vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(model_name=MODEL_NAME)


class RequestTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame) -> None:
    raise RequestTimeoutError(f"Gemini request timed out after {REQUEST_TIMEOUT_SECONDS}s")


SYSTEM_PROMPT = """
You are analyzing Reddit comment feedback toward an original post.

Your task is NOT general sentiment analysis.
Your task is to judge whether the comment gives supportive, oppositional, neutral,
mixed, or unclear feedback toward the original post.

Definitions:
- supportive: the comment supports, agrees with, constructively answers, encourages, validates, or adds helpful evidence to the post.
- oppositional: the comment disagrees with, criticizes, mocks, attacks, warns against, rejects, or undermines the post.
- neutral: the comment is factual, unrelated, purely informational, or does not clearly evaluate the post.
- mixed: the comment contains both supportive and critical feedback toward the post.
- unclear: the stance toward the post cannot be determined from the text.

Important:
- Negative emotion is not always negative feedback. A comment can be angry about the same issue while supporting the post.
- Positive emotion is not always positive feedback. A comment can be excited while disagreeing with the post.
- Sarcasm should be labeled only if the stance is clear.
- If the comment is replying inside a thread, still judge its feedback toward the original post using the available context.
- Be conservative when the relationship is ambiguous.

Return JSON only:
{
  "feedback_label": "supportive|oppositional|neutral|mixed|unclear",
  "feedback_score": <integer from -100 to 100>,
  "feedback_reason": "<brief evidence-based explanation>"
}

Score guide:
- -100: strongly hostile/rejecting toward the post
- -50: clearly negative/critical feedback
- 0: neutral, unrelated, or unclear stance
- 50: clearly positive/supportive feedback
- 100: strongly supportive/validating toward the post
"""


def clean_json_text(result_text: str) -> str:
    cleaned = (result_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        value = int(round(float(value)))
        return max(low, min(high, value))
    except Exception:
        return default


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in ALLOWED_LABELS:
        return label
    return "unclear"


def error_result(message: str) -> Dict[str, Any]:
    return {
        "feedback_label": "unclear",
        "feedback_score": 0,
        "feedback_reason": f"error: {message}",
    }


def parse_result(result_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(clean_json_text(result_text))
        return {
            "feedback_label": normalize_label(parsed.get("feedback_label", "unclear")),
            "feedback_score": clamp_int(parsed.get("feedback_score", 0), -100, 100, 0),
            "feedback_reason": str(parsed.get("feedback_reason", "")),
        }
    except Exception as e:
        return error_result(f"JSON parse failed: {e}")


def analyze_feedback(post_title: str, post_text: str, comment_text: str) -> str:
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        temperature=0.0,
    )
    prompt = f"""
{SYSTEM_PROMPT}

Original post title:
{post_title}

Original post text:
{post_text[:POST_MAX_CHARS]}

Comment text:
{comment_text[:COMMENT_MAX_CHARS]}
"""
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    )
    return response.text


def analyze_feedback_with_timeout(post_title: str, post_text: str, comment_text: str) -> str:
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(REQUEST_TIMEOUT_SECONDS)
    try:
        return analyze_feedback(post_title, post_text, comment_text)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def analyze_with_retry(post_title: str, post_text: str, comment_text: str) -> str:
    last_err = None
    for attempt in range(RETRY_LIMIT):
        try:
            return analyze_feedback_with_timeout(post_title, post_text, comment_text)
        except Exception as e:
            last_err = e
            wait_time = min(2 ** attempt, 30)
            print(f"   retry {attempt + 1}/{RETRY_LIMIT} after error: {e}")
            print(f"   sleeping {wait_time}s before retry...")
            time.sleep(wait_time)
    raise last_err


def initialize_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__row_id"] = df.index
    df["feedback_raw_response"] = ""
    df["feedback_label"] = ""
    df["feedback_score"] = pd.Series([None] * len(df), dtype="object")
    df["feedback_reason"] = ""
    df["edge_weight"] = pd.Series([None] * len(df), dtype="object")
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "__row_id" not in df.columns:
        df["__row_id"] = df.index
    if "feedback_raw_response" not in df.columns:
        df["feedback_raw_response"] = ""
    if "feedback_label" not in df.columns:
        df["feedback_label"] = ""
    if "feedback_score" not in df.columns:
        df["feedback_score"] = pd.Series([None] * len(df), dtype="object")
    if "feedback_reason" not in df.columns:
        df["feedback_reason"] = ""
    if "edge_weight" not in df.columns:
        df["edge_weight"] = pd.Series([None] * len(df), dtype="object")
    return df


def save_outputs(df: pd.DataFrame) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(FULL_OUTPUT_PATH, index=False, encoding="utf-8-sig")


def load_source_batch() -> pd.DataFrame:
    comments = pd.read_csv(COMMENTS_CSV_PATH)
    posts = pd.read_csv(
        POSTS_CSV_PATH,
        usecols=["post_id", "author", "title", "analysis_text"],
    )
    posts = posts.rename(
        columns={
            "author": "post_author",
            "title": "post_title",
            "analysis_text": "post_analysis_text",
        }
    )
    merged = comments.merge(posts, on="post_id", how="inner")
    merged = merged[merged["parent_type"] == "post"].copy()
    merged["source_author"] = merged["author"]
    merged["target_author"] = merged["post_author"]
    merged["edge_type"] = "comment_to_post"
    return merged.iloc[START_ROW:END_ROW].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze comment-to-post feedback with Gemini."
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=DEFAULT_START_ROW,
        help="Inclusive starting row after comment/post merge.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=DEFAULT_END_ROW,
        help="Exclusive ending row after comment/post merge.",
    )
    return parser.parse_args()


def load_or_initialize_batch_output() -> pd.DataFrame:
    batch_source_df = load_source_batch()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if Path(FULL_OUTPUT_PATH).exists():
        print(f"Detected existing output, resuming: {FULL_OUTPUT_PATH}")
        output_df = pd.read_csv(FULL_OUTPUT_PATH)
        output_df = ensure_columns(output_df)

        if len(output_df) < len(batch_source_df):
            missing = initialize_analysis_columns(batch_source_df.iloc[len(output_df):].copy())
            output_df = pd.concat([output_df, missing], ignore_index=True)
        elif len(output_df) > len(batch_source_df):
            output_df = output_df.iloc[:len(batch_source_df)].copy()

        return output_df

    print(f"Creating new comment feedback batch: {FULL_OUTPUT_PATH}")
    output_df = initialize_analysis_columns(batch_source_df)
    save_outputs(output_df)
    return output_df


def is_done(row) -> bool:
    raw_value = row.get("feedback_raw_response", "")
    raw = "" if pd.isna(raw_value) else str(raw_value).strip()
    score = row.get("feedback_score", None)
    return raw != "" or pd.notna(score)



def main() -> None:
    global START_ROW, END_ROW, BATCH_NAME, FULL_OUTPUT_PATH

    args = parse_args()
    START_ROW = args.start_row
    END_ROW = args.end_row
    if START_ROW < 0 or END_ROW <= START_ROW:
        raise ValueError("Expected 0 <= start-row < end-row")
    BATCH_NAME = f"{START_ROW}_{END_ROW}"
    FULL_OUTPUT_PATH = f"{OUTPUT_DIR}/comment_feedback_{BATCH_NAME}.csv"

    print(f"Starting comment feedback analysis (model={MODEL_NAME}, region={LOCATION})")
    print(f"Batch range: START_ROW={START_ROW}, END_ROW={END_ROW}")

    df = load_or_initialize_batch_output()
    df = ensure_columns(df)
    save_outputs(df)

    pending_indices = [i for i, row in df.iterrows() if not is_done(row)]
    print(f"Batch rows: {len(df)}")
    print(f"Pending rows: {len(pending_indices)}")

    processed_since_save = 0

    for i in pending_indices:
        actual_row = START_ROW + i
        post_title = str(df.at[i, "post_title"] or "").strip()
        post_text = str(df.at[i, "post_analysis_text"] or "").strip()
        comment_text = str(df.at[i, "analysis_text"] or "").strip()

        print(f"[batch_row={i}, actual_row={actual_row}] analyzing comment feedback...")
        start = time.time()

        try:
            result = analyze_with_retry(post_title, post_text, comment_text)
            parsed = parse_result(result)
        except Exception as e:
            parsed = error_result(str(e))
            result = json.dumps(parsed, ensure_ascii=False)
            print(f"   row failed but batch will continue: {e}")

        elapsed = time.time() - start
        print(
            f"   label={parsed['feedback_label']} "
            f"score={parsed['feedback_score']} time={elapsed:.2f}s"
        )

        df.at[i, "feedback_raw_response"] = clean_json_text(result)
        df.at[i, "feedback_label"] = parsed["feedback_label"]
        df.at[i, "feedback_score"] = parsed["feedback_score"]
        df.at[i, "feedback_reason"] = parsed["feedback_reason"]
        df.at[i, "edge_weight"] = parsed["feedback_score"] / 100

        processed_since_save += 1
        time.sleep(SLEEP_SECONDS)

        if processed_since_save >= SAVE_EVERY:
            print(f"autosave every {SAVE_EVERY} rows")
            save_outputs(df)
            processed_since_save = 0

    if processed_since_save > 0:
        print("final save")
        save_outputs(df)

    print("Comment feedback analysis complete")
    print(f"Output: {FULL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
