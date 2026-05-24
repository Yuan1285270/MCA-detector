import argparse
from pathlib import Path
import json
import time
from typing import Any, Dict

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool
from vertexai.preview import rag

# === 設定區 ===
PROJECT_ID = "capstone-492007"
LOCATION = "us-central1"
CSV_PATH = "../data-cleaning/processed_data/processed_data.csv"
OUTPUT_DIR = "output"

# === VM 長跑建議：分批執行 ===
START_ROW = 0
END_ROW = 5000  # 不含 END_ROW，可改成 10000、15000 ...
BATCH_NAME = f"{START_ROW}_{END_ROW}"
FULL_OUTPUT_PATH = f"{OUTPUT_DIR}/post_analysis_{BATCH_NAME}.csv"

# 你的 RAG Corpus 完整路徑
RAG_CORPUS_PATH = "projects/capstone-492007/locations/asia-east1/ragCorpora/4611686018427387904"

MODEL_NAME = "gemini-2.5-flash"
TEXT_COLUMN = "analysis_text"
MAX_CHARS = 3000
SAVE_EVERY = 10
SLEEP_SECONDS = 1.5
RETRY_LIMIT = 5

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

vertexai.init(project=PROJECT_ID, location=LOCATION)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=RAG_CORPUS_PATH,
                )
            ],
            similarity_top_k=3,
        ),
    )
)

model = GenerativeModel(
    model_name=MODEL_NAME,
    tools=[rag_retrieval_tool]
)

SYSTEM_PROMPT = """
You are a text-only analyst for cryptocurrency social media content.

Your task is to evaluate ONLY the given text at the post/comment level.

IMPORTANT CONSTRAINTS:
- Do NOT infer whether the author is a bot, fake account, or coordinated actor.
- Do NOT perform fact-checking beyond what is explicitly written.
- Do NOT assume malicious intent without textual evidence.
- Only analyze linguistic and rhetorical features present in the text.
- You must analyze ONLY the given input text.
- Do NOT use retrieved examples as direct labels.
- If retrieved content is similar, use it only as background reference, not as the answer.

TASK:
Return the following:
1. sentiment_score (-100 to 100)
2. manipulative_rhetoric_score (0 to 100)
3. rhetoric_tags (list)

DEFINITION:
Sentiment:
- Measures emotional polarity and intensity.
- Negative: fear, panic, pessimism.
- Positive: excitement, optimism, bullish tone.
- Neutral: analytical or mixed tone.
- Strong sentiment does NOT equal manipulation.

Manipulative rhetoric means language that pressures, induces, amplifies fear/FOMO,
claims privileged information, creates artificial urgency, or pushes direct action
without sufficient evidence.

Allowed rhetoric tags and guidance:
- urgency: creates time pressure ("last chance", "before it's too late")
- fear: warns of regret, danger, collapse, or loss
- overconfidence: certainty without evidence ("guaranteed", "easy 100x")
- authority_claim: insider, source, expert, or privileged-knowledge framing
- bandwagon: social proof or crowd-following language
- us_vs_them: oppositional or conspiratorial framing
- call_to_action: direct push to act ("buy now", "sell immediately")
- emotional_amplification: exaggerated emotional style, emphatic wording, sensational framing, or arousal-amplifying language
- analytical_neutral: balanced, uncertain, explanatory, or data-driven tone

You MUST ONLY choose tags from this list:
[urgency, fear, overconfidence, authority_claim, bandwagon, us_vs_them, call_to_action, emotional_amplification, analytical_neutral]
Do NOT invent new tags.

SCORING GUIDE:
Manipulation score calibration:
- 0-20: no manipulation
- 21-40: mild rhetoric
- 41-60: moderate persuasion
- 61-80: strong manipulation
- 81-100: aggressive / coercive manipulation

Important calibration notes:
- Bullish or bearish opinion alone is NOT manipulation.
- Meme, humor, slang, or emojis are NOT manipulation unless they amplify arousal, pressure behavior, or reinforce manipulative framing.
- Analytical discussion, uncertainty, skepticism, and risk warnings should stay low.
- If evidence is weak, assign a conservative score and prefer analytical_neutral.

SELF-CHECK:
Re-evaluate your score before finalizing.
- Is the score justified ONLY by the text itself?
- If not, reduce the score.

OUTPUT FORMAT (JSON ONLY):
{
  "sentiment_score": <integer>,
  "sentiment_reason": "<brief evidence-based explanation>",
  "manipulative_rhetoric_score": <integer>,
  "manipulative_rhetoric_reason": "<brief explanation>",
  "rhetoric_tags": ["<tag1>", "<tag2>"]
}
"""


def analyze_gemini_with_rag(text: str) -> str:
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        temperature=0.0,
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\nInput text to analyze:\n{text}"
    response = model.generate_content(
        full_prompt,
        generation_config=generation_config,
    )
    return response.text



def analyze_with_retry(text: str, retries: int = RETRY_LIMIT) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            return analyze_gemini_with_rag(text)
        except Exception as e:
            last_err = e
            wait_time = min(2 ** attempt, 30)
            print(f"   retry {attempt + 1}/{retries} after error: {e}")
            print(f"   sleeping {wait_time}s before retry...")
            time.sleep(wait_time)
    raise last_err



def error_result(message: str) -> Dict[str, Any]:
    return {
        "sentiment_score": 0,
        "sentiment_reason": f"error: {message}",
        "manipulative_rhetoric_score": 0,
        "manipulative_rhetoric_reason": "error",
        "rhetoric_tags": ["analytical_neutral"],
    }


def clean_json_text(result_text: str) -> str:
    s = (result_text or "").strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```", "").strip()
    return s


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        value = int(round(float(value)))
        return max(low, min(high, value))
    except Exception:
        return default



def normalize_tags(tags) -> list:
    if not isinstance(tags, list):
        return ["analytical_neutral"]
    cleaned = [str(tag).strip() for tag in tags if str(tag).strip() in ALLOWED_TAGS]
    if not cleaned:
        return ["analytical_neutral"]
    seen = set()
    unique = []
    for tag in cleaned:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def parse_saved_tags(value: Any) -> list:
    if isinstance(value, list):
        return [] if not value else normalize_tags(value)
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list) and not parsed:
            return []
        return normalize_tags(parsed)
    except Exception:
        return []



def parse_result(result_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(clean_json_text(result_text))
        return {
            "sentiment_score": clamp_int(parsed.get("sentiment_score", 0), -100, 100, 0),
            "sentiment_reason": str(parsed.get("sentiment_reason", "")),
            "manipulative_rhetoric_score": clamp_int(
                parsed.get("manipulative_rhetoric_score", 0), 0, 100, 0
            ),
            "manipulative_rhetoric_reason": str(parsed.get("manipulative_rhetoric_reason", "")),
            "rhetoric_tags": normalize_tags(parsed.get("rhetoric_tags", [])),
        }
    except Exception as e:
        return error_result(f"JSON parse failed: {e}")



def save_outputs(df: pd.DataFrame) -> None:
    df_to_save = df.copy()
    df_to_save["rhetoric_tags"] = df_to_save["rhetoric_tags"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
    )
    df_to_save.to_csv(FULL_OUTPUT_PATH, index=False, encoding="utf-8-sig")



def initialize_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__row_id"] = df.index
    df["rag_analysis"] = ""
    df["sentiment_score"] = pd.Series([None] * len(df), dtype="object")
    df["sentiment_reason"] = ""
    df["manipulative_rhetoric_score"] = pd.Series([None] * len(df), dtype="object")
    df["manipulative_rhetoric_reason"] = ""
    df["rhetoric_tags"] = pd.Series([[] for _ in range(len(df))], dtype="object")
    return df



def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "__row_id" not in df.columns:
        df["__row_id"] = df.index
    if "rag_analysis" not in df.columns:
        df["rag_analysis"] = ""
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = pd.Series([None] * len(df), dtype="object")
    if "sentiment_reason" not in df.columns:
        df["sentiment_reason"] = ""
    if "manipulative_rhetoric_score" not in df.columns:
        df["manipulative_rhetoric_score"] = pd.Series([None] * len(df), dtype="object")
    if "manipulative_rhetoric_reason" not in df.columns:
        df["manipulative_rhetoric_reason"] = ""
    if "rhetoric_tags" not in df.columns:
        df["rhetoric_tags"] = pd.Series([[] for _ in range(len(df))], dtype="object")
    df["rhetoric_tags"] = df["rhetoric_tags"].apply(parse_saved_tags)

    pending_mask = (
        df["rag_analysis"].fillna("").astype(str).str.strip().eq("")
        & df["sentiment_score"].isna()
        & df["manipulative_rhetoric_score"].isna()
    )
    df.loc[pending_mask, "rhetoric_tags"] = pd.Series(
        [[] for _ in range(int(pending_mask.sum()))],
        index=df.index[pending_mask],
        dtype="object",
    )
    return df



def load_or_initialize_batch_output() -> pd.DataFrame:
    source_df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in source_df.columns:
        raise ValueError(f"找不到欄位 {TEXT_COLUMN}")

    batch_source_df = source_df.iloc[START_ROW:END_ROW].copy()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if Path(FULL_OUTPUT_PATH).exists():
        print(f"📂 偵測到既有批次輸出，將從中斷處續跑：{FULL_OUTPUT_PATH}")
        output_df = pd.read_csv(FULL_OUTPUT_PATH)
        output_df = ensure_columns(output_df)

        if len(output_df) < len(batch_source_df):
            missing = initialize_analysis_columns(batch_source_df.iloc[len(output_df):].copy())
            output_df = pd.concat([output_df, missing], ignore_index=True)
        elif len(output_df) > len(batch_source_df):
            output_df = output_df.iloc[:len(batch_source_df)].copy()

        return output_df

    print(f"🆕 建立新批次輸出，來源檔案：{CSV_PATH}")
    batch_df = initialize_analysis_columns(batch_source_df)
    save_outputs(batch_df)
    return batch_df



def is_done(row) -> bool:
    text = str(row.get(TEXT_COLUMN, "")).strip().lower()
    if text in ["nan", "", "[removed]", "[deleted]"]:
        return True
    analysis = str(row.get("rag_analysis", "")).strip()
    score = row.get("manipulative_rhetoric_score", None)
    return analysis != "" or pd.notna(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cleaned Reddit posts with Gemini.")
    parser.add_argument("--start-row", type=int, default=START_ROW)
    parser.add_argument("--end-row", type=int, default=END_ROW)
    return parser.parse_args()


def main() -> None:
    global START_ROW, END_ROW, BATCH_NAME, FULL_OUTPUT_PATH

    args = parse_args()
    START_ROW = args.start_row
    END_ROW = args.end_row
    if START_ROW < 0 or END_ROW <= START_ROW:
        raise ValueError("Expected 0 <= start-row < end-row")
    BATCH_NAME = f"{START_ROW}_{END_ROW}"
    FULL_OUTPUT_PATH = f"{OUTPUT_DIR}/post_analysis_{BATCH_NAME}.csv"

    print(f"🚀 開始執行 RAG 分析 (model={MODEL_NAME}, region={LOCATION})")
    print(f"📦 本次批次區間：START_ROW={START_ROW}, END_ROW={END_ROW}")

    sample_df = load_or_initialize_batch_output()
    if TEXT_COLUMN not in sample_df.columns:
        raise ValueError(f"找不到欄位 {TEXT_COLUMN}")

    sample_df = ensure_columns(sample_df)
    save_outputs(sample_df)

    pending_indices = [i for i, row in sample_df.iterrows() if not is_done(row)]
    print(f"📌 批次總列數: {len(sample_df)}")
    print(f"📌 待處理列數: {len(pending_indices)}")

    processed_since_save = 0

    for i in pending_indices:
        actual_row = START_ROW + i
        text = str(sample_df.at[i, TEXT_COLUMN]).strip()

        if text.lower() in ["nan", "", "[removed]", "[deleted]"]:
            print(f"Row batch_row={i}, actual_row={actual_row} -> skipped")
            sample_df.at[i, "rag_analysis"] = ""
            sample_df.at[i, "sentiment_score"] = None
            sample_df.at[i, "sentiment_reason"] = ""
            sample_df.at[i, "manipulative_rhetoric_score"] = None
            sample_df.at[i, "manipulative_rhetoric_reason"] = ""
            sample_df.at[i, "rhetoric_tags"] = []
            processed_since_save += 1
        else:
            text = text[:MAX_CHARS]
            print(f"[batch_row={i}, actual_row={actual_row}] analyzing...")
            start = time.time()

            try:
                result = analyze_with_retry(text)
                parsed = parse_result(result)
            except Exception as e:
                print(f"❌ Fatal error at batch_row={i}, actual_row={actual_row}: {e}")
                save_outputs(sample_df)
                raise

            elapsed = time.time() - start
            print(
                f"   score={parsed['manipulative_rhetoric_score']} "
                f"tags={parsed['rhetoric_tags']} time={elapsed:.2f}s"
            )

            sample_df.at[i, "rag_analysis"] = clean_json_text(result)
            sample_df.at[i, "sentiment_score"] = parsed["sentiment_score"]
            sample_df.at[i, "sentiment_reason"] = parsed["sentiment_reason"]
            sample_df.at[i, "manipulative_rhetoric_score"] = parsed["manipulative_rhetoric_score"]
            sample_df.at[i, "manipulative_rhetoric_reason"] = parsed["manipulative_rhetoric_reason"]
            sample_df.at[i, "rhetoric_tags"] = parsed["rhetoric_tags"]

            processed_since_save += 1
            time.sleep(SLEEP_SECONDS)

        if processed_since_save >= SAVE_EVERY:
            print(f"💾 autosave (every {SAVE_EVERY} rows)")
            save_outputs(sample_df)
            processed_since_save = 0

    if processed_since_save > 0:
        print("💾 final save")
        save_outputs(sample_df)

    print("✅ 分析完成！")
    print(f"完整輸出：{FULL_OUTPUT_PATH}")
    print(f"本批次對應原始資料區間：[{START_ROW}, {END_ROW})")


if __name__ == "__main__":
    main()
