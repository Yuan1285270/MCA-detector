import pandas as pd
import requests
import time
import json

from config import *
from ollama_rag_utils import *

SYSTEM_PROMPT = """
You are an expert text analysis model.

Your task is to analyze the input text in two dimensions:

1. Sentiment score (-100 to 100)
2. Manipulative rhetoric score (0 to 100)

Output JSON only:
{
  "score_1": int,
  "reason_1": "...",
  "score_2": int,
  "reason_2": "..."
}
""".strip()


def analyze(text, rag_index):
    refs = retrieve(text, rag_index, TOP_K)

    ref_text = "\n\n".join(refs)

    user_prompt = f"""
Reference:
{ref_text}

Text:
{text}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0},
        "keep_alive": "10m"
    }

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()

    return r.json()["message"]["content"]


def safe_json(reason):
    return json.dumps({
        "score_1": 0,
        "reason_1": reason,
        "score_2": 0,
        "reason_2": reason
    })


def main():
    df = pd.read_csv(CSV_PATH)

    if TEXT_COLUMN not in df.columns:
        raise ValueError("欄位錯誤")

    df = df.head(LIMIT)

    print("Loading knowledge base...")
    chunks = load_knowledge_chunks(KNOWLEDGE_DIR)

    print("Building index...")
    index = build_index(chunks)

    results = []

    for i, row in df.iterrows():
        text = str(row[TEXT_COLUMN]).strip()

        if text in ["", "nan", "[deleted]", "[removed]"]:
            results.append("")
            continue

        text = text[:MAX_CHARS]

        print(f"Row {i}")

        start = time.time()
        try:
            res = analyze(text, index)
        except Exception as e:
            res = safe_json(str(e))

        print(res)
        print("time:", time.time() - start)

        results.append(res)

    df["analysis"] = results
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("DONE")


if __name__ == "__main__":
    main()
