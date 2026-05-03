from pathlib import Path

import requests
import numpy as np
from pypdf import PdfReader
from config import *

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        page_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for c in page_chunks:
            chunks.append(f"[Page {page_idx + 1}]\n{c}")

    return chunks


def load_knowledge_chunks(knowledge_dir):
    chunks = []
    pdf_paths = sorted(Path(knowledge_dir).glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {knowledge_dir}")

    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name}...")
        chunks.extend(load_pdf_chunks(pdf_path))

    return chunks


def get_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "input": text
    }

    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()

    data = r.json()

    if "embeddings" in data:
        return np.array(data["embeddings"][0])

    if "embedding" in data:
        return np.array(data["embedding"])

    raise ValueError("embedding API error")


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def build_index(chunks):
    index = []
    for i, c in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        emb = get_embedding(c)
        index.append({"text": c, "emb": emb})
    return index


def retrieve(text, index, top_k=3):
    q_emb = get_embedding(text)

    scored = []
    for item in index:
        sim = cosine_sim(q_emb, item["emb"])
        scored.append((sim, item["text"]))

    scored.sort(reverse=True)
    return [x[1] for x in scored[:top_k]]
