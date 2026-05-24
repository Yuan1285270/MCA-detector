#!/usr/bin/env python3
"""Analyze cleaned posts or comments with a local Ollama model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ALLOWED_TAGS = [
    "urgency",
    "fear",
    "overconfidence",
    "authority_claim",
    "bandwagon",
    "us_vs_them",
    "call_to_action",
    "emotional_amplification",
    "analytical_neutral",
]
ALLOWED_LABELS = ["supportive", "oppositional", "neutral", "mixed", "unclear"]

POST_SYSTEM_PROMPT = """
You are a text-only analyst for cryptocurrency social media content.

Analyze only the given input text. Do not infer whether the author is a bot.

Return JSON only:
{
  "sentiment_score": <integer -100 to 100>,
  "sentiment_reason": "<brief evidence-based explanation>",
  "manipulative_rhetoric_score": <integer 0 to 100>,
  "manipulative_rhetoric_reason": "<brief evidence-based explanation>",
  "rhetoric_tags": ["urgency|fear|overconfidence|authority_claim|bandwagon|us_vs_them|call_to_action|emotional_amplification|analytical_neutral"]
}

Manipulative rhetoric means language that pressures, induces, amplifies fear/FOMO,
claims privileged information, creates artificial urgency, or pushes direct action
without sufficient evidence. Be conservative.
""".strip()

COMMENT_SYSTEM_PROMPT = """
You are analyzing Reddit comment feedback toward an original post.

Your task is not general sentiment analysis. Judge whether the comment gives
supportive, oppositional, neutral, mixed, or unclear feedback toward the original post.

Return JSON only:
{
  "feedback_label": "supportive|oppositional|neutral|mixed|unclear",
  "feedback_score": <integer from -100 to 100>,
  "feedback_reason": "<brief evidence-based explanation>"
}

Negative emotion is not always negative feedback; positive emotion is not always
supportive. Be conservative when the relationship is ambiguous.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cleaned Reddit data with Ollama.")
    parser.add_argument("--task", choices=["posts", "comments"], required=True)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=0, help="Exclusive end row. 0 means all rows.")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except Exception:
        return default


def normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["analytical_neutral"]
    cleaned = [str(tag).strip() for tag in value if str(tag).strip() in ALLOWED_TAGS]
    return cleaned or ["analytical_neutral"]


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else "unclear"


def parse_post_result(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(clean_json_text(text))
        return {
            "sentiment_score": clamp_int(parsed.get("sentiment_score", 0), -100, 100, 0),
            "sentiment_reason": str(parsed.get("sentiment_reason", "")),
            "manipulative_rhetoric_score": clamp_int(
                parsed.get("manipulative_rhetoric_score", 0), 0, 100, 0
            ),
            "manipulative_rhetoric_reason": str(parsed.get("manipulative_rhetoric_reason", "")),
            "rhetoric_tags": normalize_tags(parsed.get("rhetoric_tags", [])),
        }
    except Exception as exc:
        return {
            "sentiment_score": 0,
            "sentiment_reason": f"error: {exc}",
            "manipulative_rhetoric_score": 0,
            "manipulative_rhetoric_reason": "error",
            "rhetoric_tags": ["analytical_neutral"],
        }


def parse_comment_result(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(clean_json_text(text))
        return {
            "feedback_label": normalize_label(parsed.get("feedback_label", "unclear")),
            "feedback_score": clamp_int(parsed.get("feedback_score", 0), -100, 100, 0),
            "feedback_reason": str(parsed.get("feedback_reason", "")),
        }
    except Exception as exc:
        return {
            "feedback_label": "unclear",
            "feedback_score": 0,
            "feedback_reason": f"error: {exc}",
        }


def call_ollama(system_prompt: str, user_prompt: str, args: argparse.Namespace) -> str:
    last_error = None
    for attempt in range(args.retry_limit):
        try:
            response = requests.post(
                args.ollama_url,
                json={
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                    "keep_alive": "10m",
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"   retry {attempt + 1}/{args.retry_limit} after error: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"Ollama request failed after {args.retry_limit} attempts: {last_error}")


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def value_present(value: Any) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def load_or_resume(source_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return source_df.copy()

    existing = pd.read_csv(output_path, low_memory=False)
    if len(existing) < len(source_df):
        missing = source_df.iloc[len(existing):].copy()
        return pd.concat([existing, missing], ignore_index=True)
    if len(existing) > len(source_df):
        return existing.iloc[: len(source_df)].copy()
    return existing


def load_posts(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv("../data-cleaning/processed_data/processed_data.csv")
    end = args.end_row or len(df)
    return df.iloc[args.start_row:end].copy().reset_index(drop=True)


def load_comments(args: argparse.Namespace) -> pd.DataFrame:
    comments = pd.read_csv("../data-cleaning/processed_data/processed_comments.csv")
    posts = pd.read_csv(
        "../data-cleaning/processed_data/processed_data.csv",
        usecols=["post_id", "author", "title", "analysis_text"],
    ).rename(
        columns={
            "author": "post_author",
            "title": "post_title",
            "analysis_text": "post_analysis_text",
        }
    )
    merged = comments.merge(posts, on="post_id", how="inner")
    merged = merged.loc[merged["parent_type"].eq("post")].copy()
    merged["source_author"] = merged["author"]
    merged["target_author"] = merged["post_author"]
    merged["edge_type"] = "comment_to_post"
    end = args.end_row or len(merged)
    return merged.iloc[args.start_row:end].copy().reset_index(drop=True)


def analyze_posts(df: pd.DataFrame, args: argparse.Namespace, output_path: Path) -> None:
    df = load_or_resume(df, output_path)
    for col in [
        "rag_analysis",
        "sentiment_score",
        "sentiment_reason",
        "manipulative_rhetoric_score",
        "manipulative_rhetoric_reason",
        "rhetoric_tags",
    ]:
        if col not in df.columns:
            df[col] = "" if col.endswith("reason") or col == "rag_analysis" else None

    sentiment_na = pd.to_numeric(df["sentiment_score"], errors="coerce").isna()
    rhetoric_na = pd.to_numeric(df["manipulative_rhetoric_score"], errors="coerce").isna()
    has_text = df["analysis_text"].fillna("").astype(str).str.strip().ne("")
    pending_idx = df.index[sentiment_na & rhetoric_na & has_text].tolist()
    print(f"Post rows: {len(df):,}; pending: {len(pending_idx):,}")

    processed_since_save = 0
    for idx in pending_idx:
        text = str(df.at[idx, "analysis_text"]).strip()[: args.max_chars]
        if not text or text.lower() in {"nan", "[removed]", "[deleted]"}:
            parsed = parse_post_result("{}")
            raw = json.dumps(parsed)
        else:
            print(f"[post batch_row={idx}] analyzing...")
            try:
                raw = call_ollama(POST_SYSTEM_PROMPT, f"Input text:\n{text}", args)
                parsed = parse_post_result(raw)
            except Exception as exc:
                parsed = parse_post_result("{}")
                parsed["sentiment_reason"] = f"error: {exc}"
                parsed["manipulative_rhetoric_reason"] = "error"
                raw = json.dumps(parsed, ensure_ascii=False)
        df.at[idx, "rag_analysis"] = clean_json_text(raw)
        for key, value in parsed.items():
            if key == "rhetoric_tags":
                df.at[idx, key] = json.dumps(value, ensure_ascii=False)
            else:
                df.at[idx, key] = value
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            save_output(df, output_path)
            processed_since_save = 0
        time.sleep(args.sleep_seconds)
    save_output(df, output_path)


def analyze_comments(df: pd.DataFrame, args: argparse.Namespace, output_path: Path) -> None:
    df = load_or_resume(df, output_path)
    for col in ["feedback_raw_response", "feedback_label", "feedback_score", "feedback_reason", "edge_weight"]:
        if col not in df.columns:
            df[col] = ""

    score_na = pd.to_numeric(df["feedback_score"], errors="coerce").isna()
    has_text = df["analysis_text"].fillna("").astype(str).str.strip().ne("")
    pending_idx = df.index[score_na & has_text].tolist()
    print(f"Comment rows: {len(df):,}; pending: {len(pending_idx):,}")

    processed_since_save = 0
    for idx in pending_idx:
        post_title = str(df.at[idx, "post_title"]).strip()
        post_text = str(df.at[idx, "post_analysis_text"]).strip()[: args.max_chars]
        comment_text = str(df.at[idx, "analysis_text"]).strip()[: args.max_chars]
        print(f"[comment batch_row={idx}] analyzing...")
        try:
            raw = call_ollama(
                COMMENT_SYSTEM_PROMPT,
                f"Original post title:\n{post_title}\n\nOriginal post text:\n{post_text}\n\nComment text:\n{comment_text}",
                args,
            )
            parsed = parse_comment_result(raw)
        except Exception as exc:
            parsed = parse_comment_result("{}")
            parsed["feedback_reason"] = f"error: {exc}"
            raw = json.dumps(parsed, ensure_ascii=False)
        df.at[idx, "feedback_raw_response"] = clean_json_text(raw)
        df.at[idx, "feedback_label"] = parsed["feedback_label"]
        df.at[idx, "feedback_score"] = parsed["feedback_score"]
        df.at[idx, "feedback_reason"] = parsed["feedback_reason"]
        df.at[idx, "edge_weight"] = parsed["feedback_score"] / 100.0
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            save_output(df, output_path)
            processed_since_save = 0
        time.sleep(args.sleep_seconds)
    save_output(df, output_path)


def main() -> None:
    args = parse_args()
    batch_name = f"{args.start_row}_{args.end_row or 'all'}"
    if args.task == "posts":
        output_path = args.output_dir / "posts" / f"post_analysis_{batch_name}.csv"
        df = load_posts(args)
        analyze_posts(df, args, output_path)
    else:
        output_path = args.output_dir / "comments" / f"comment_feedback_{batch_name}.csv"
        df = load_comments(args)
        analyze_comments(df, args, output_path)
    print(f"Ollama analysis complete: {output_path}")


if __name__ == "__main__":
    main()
