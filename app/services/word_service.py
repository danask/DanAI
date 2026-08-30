"""
Business logic for vocabulary / word-memory management:
- Embedding-based similarity search (RAG helpers)
- Parsing & indexing an Obsidian vault into sentence records
- Sentence registration, mistake analysis, and review-quiz generation
"""
import glob
import os
import random
import re
from typing import List

import httpx
import numpy as np

from app.config import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    OBSIDIAN_VAULT_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_URL,
)
from app.database import file_meta_collection, sentence_collection
from app.utils import current_timestamp_ms


# =========================================================
# Embedding / RAG helper functions
# =========================================================
async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


async def retrieve_similar_sentences(query: str, top_k: int = 3):
    query_embedding = await get_embedding(query)
    docs = list(sentence_collection.find({"embedding": {"$exists": True}}))
    scored = [
        (cosine_similarity(query_embedding, d["embedding"]), d) for d in docs
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# =========================================================
# Obsidian vault parsing / indexing
# =========================================================
def strip_markdown_noise(line: str) -> str:
    """Strips Markdown syntax elements from a line."""
    line = re.sub(r'^#+\s*', '', line)            # headers
    line = re.sub(r'^[-*]\s*', '', line)           # bullets
    line = re.sub(r'==(.+?)==', r'\1', line)       # ==highlight==
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)   # **bold**
    line = re.sub(r'\[\[(.+?)\]\]', r'\1', line)   # [[wiki link]]
    return line.strip()


def parse_markdown_file(filepath: str):
    """Parses a single markdown file and returns a list of (text, is_correct, correction)."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = strip_markdown_noise(raw_line)
            if not line or line.startswith("---"):  # skip frontmatter / blank lines
                continue
            if "=>" in line:
                wrong, correct = line.split("=>", 1)
                wrong, correct = wrong.strip(), correct.strip()
                if wrong:
                    entries.append((wrong, False, correct))
            elif len(line.split()) >= 3:  # exclude overly short lines (titles, etc.)
                entries.append((line, True, None))
    return entries


def get_vault_markdown_files(vault_path: str):
    """Returns paths of all .md files in the vault (hidden/config folders excluded)."""
    pattern = os.path.join(vault_path, "**", "*.md")
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if ".obsidian" not in f and "/.trash/" not in f]


async def index_obsidian_vault(vault_path: str = OBSIDIAN_VAULT_PATH):
    """Scans the whole vault and re-indexes only the files that changed."""
    if not os.path.isdir(vault_path):
        print(f">>> Obsidian vault path not found: {vault_path}")
        return 0

    added, updated_files = 0, 0

    for filepath in get_vault_markdown_files(vault_path):
        mtime = os.path.getmtime(filepath)
        meta = file_meta_collection.find_one({"path": filepath})

        # mtime unchanged -> no change, skip
        if meta and meta.get("mtime") == mtime:
            continue

        # File changed: drop existing sentences from this source file, then re-insert
        sentence_collection.delete_many({"source_file": filepath})

        entries = parse_markdown_file(filepath)
        for text, is_correct, correction in entries:
            embedding = await get_embedding(text)
            sentence_collection.insert_one({
                "text": text,
                "is_correct": is_correct,
                "correction": correction,
                "note": None,
                "embedding": embedding,
                "timestamp": current_timestamp_ms(),  # Unix epoch ms (UTC)
                "source_file": filepath,
            })
            added += 1

        file_meta_collection.update_one(
            {"path": filepath},
            {"$set": {"mtime": mtime}},
            upsert=True,
        )
        updated_files += 1

    print(f">>> Obsidian indexing complete: {updated_files} file(s) updated, {added} sentence(s) added")
    return added


# =========================================================
# Sentence registration / analysis / lookup
# =========================================================
async def add_sentence(entry: dict) -> dict:
    """Registers a single sentence added directly from the app (separate from Obsidian indexing)."""
    embedding = await get_embedding(entry["text"])
    doc = dict(entry)
    doc["embedding"] = embedding
    doc["timestamp"] = current_timestamp_ms()  # Unix epoch ms (UTC)
    doc["source_file"] = None
    result = sentence_collection.insert_one(doc)
    return {"id": str(result.inserted_id)}


async def analyze_sentence(sentence: str, top_k: int) -> dict:
    similar = await retrieve_similar_sentences(sentence, top_k)

    context_lines = []
    for score, doc in similar:
        line = f'- "{doc["text"]}"'
        if not doc.get("is_correct") and doc.get("correction"):
            line += f' (과거 오답 → 올바른 표현: "{doc["correction"]}")'
        context_lines.append(line)
    context_text = "\n".join(context_lines) or "관련된 과거 문장 없음"

    prompt = (
        "당신은 영어 문법/어휘 교정 전문가입니다.\n"
        f"사용자가 과거에 공부한 비슷한 문장들 (참고용):\n{context_text}\n\n"
        f'분석할 문장: "{sentence}"\n\n'
        "다음을 수행하세요:\n"
        "1. 문법/어휘 오류 확인\n"
        "2. 오류가 있다면 올바른 문장으로 교정\n"
        "3. 왜 틀렸는지 한국어로 간단히 설명\n"
        "4. 비슷한 패턴의 연습 문장 2개 제시"
    )

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 8192},
    }

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(OLLAMA_BASE_URL, json=payload)
        response.raise_for_status()
        result_text = response.json().get("response", "")

    return {
        "sentence": sentence,
        "retrieved_context": [d["text"] for _, d in similar],
        "analysis": result_text,
    }


def get_mistakes() -> dict:
    docs = list(sentence_collection.find({"is_correct": False}))
    for d in docs:
        d["_id"] = str(d["_id"])
        d.pop("embedding", None)  # exclude embedding from the response
    return {"mistakes": docs}


async def review_mistakes(count: int = 1) -> dict:
    """Randomly samples stored sentences (including memorized expressions) to build a practice quiz."""
    all_sentences = list(sentence_collection.find({}))
    if not all_sentences:
        return {"message": "아직 저장된 문장이 없어요.", "quiz": None, "source_sentences": []}

    sample = random.sample(all_sentences, min(count, len(all_sentences)))
    context = "\n".join(f'- "{m["text"]}"' for m in sample)

    prompt = (
        "아래는 사용자가 예전에 공부하며 정리해둔 영어 문장들입니다.\n"
        f"{context}\n\n"
        "각 문장에 대해 다음 순서로 연습 문제를 만들어주세요:\n"
        "1. 문장을 한국어로 뜻으로 먼저 보여준다"
        "(예: '나의 삶의 의미는 나를 기독교로 이끄는 것이다')\n"
        "2. 답이 되는 영어 문장을 다음 줄에 표시한다.\n"
        "3. 그리고 그 표현이 왜 중요한지/어떤 뜻인지 한국어로 간단히 설명해주세요.\n"
        "4. 같은 표현을 활용한 새로운 예문 1개를 사용자가 직접 만들어보도록 제안하세요."
    )

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 8192},
    }

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(OLLAMA_BASE_URL, json=payload)
        response.raise_for_status()
        result = response.json().get("response", "")

    return {
        "quiz": result,
        "source_sentences": [m["text"] for m in sample],
    }
