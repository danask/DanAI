import glob
import os
import random
import re
import secrets
from datetime import datetime
from typing import List, Optional
 
import httpx
import numpy as np
import pymongo
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pymongo import MongoClient

# MongoDB connection setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["conversation_logs"]
collection = db["logs"]
sentence_collection = db["sentences"]      # 영어 학습 문장 저장소
file_meta_collection = db["file_meta"]     # Obsidian 파일별 mtime 추적용(증분 색인)

app = FastAPI(title="Local Ollama Agent API", version="1.0.0")

# Static 및 Templates 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/api/generate"
)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:14b")

# 임베딩 / RAG 관련 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_EMBED_URL = OLLAMA_BASE_URL.replace("/api/generate", "/api/embeddings")

OBSIDIAN_VAULT_PATH = os.getenv(
    "OBSIDIAN_VAULT_PATH",
    "/Users/danielahn/Documents/Note/Obsidian/Dan_Dev"
)


# =========================================================
# Pydantic 모델
# =========================================================

class AgentRequest(BaseModel):
    prompt: str
    task_type: str = "general"  # Options: general, code_review, summarize


class AgentResponse(BaseModel):
    task_type: str
    model_used: str
    result: str
    conversation_id: str

 
class SentenceEntry(BaseModel):
    text: str
    is_correct: bool = True
    correction: Optional[str] = None   # 틀렸다면 올바른 문장
    note: Optional[str] = None
 
 
class AnalyzeRequest(BaseModel):
    sentence: str
    top_k: int = 3
 

# =========================================================
# 기존 에이전트 (Ollama 호출) 관련 함수
# =========================================================

def build_agent_prompt(task_type: str, user_prompt: str) -> str:
    """Builds an internal agent system prompt according to the task type."""
    if task_type == "code_review":
        system_instruction = (
            "You are an expert code reviewer. Analyze the code provided by the user, "
            "identify potential bugs or performance bottlenecks, and suggest improvements."
        )
    elif task_type == "summarize":
        system_instruction = (
            "You are a conciseness expert. Summarize the following input into key bullet points "
            "without losing core context."
        )
    elif task_type == "english":
        system_instruction = (
            "You are an English grammar and writing tutor. Analyze the user's sentence(s), "
            "point out grammar or word-choice errors, provide corrected versions, "
            "and briefly explain the mistakes in Korean."
        )        
    else:
        system_instruction = (
            "Your name is DanAI. You speak Korean."            
            "You are a helpful AI software development agent running on a local environment."
        )

    return f"System: {system_instruction}\nUser Request: {user_prompt}"


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent_task(request: AgentRequest):
    conversation_id = secrets.token_hex(16)
    formatted_prompt = build_agent_prompt(request.task_type, request.prompt)

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {"num_ctx": 16384},
    }

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            response = await http_client.post(OLLAMA_BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "")

            # Save the conversation log to MongoDB
            log_entry = {
                "conversation_id": conversation_id,
                "task_type": request.task_type,
                "model_used": DEFAULT_MODEL,
                "prompt": request.prompt,
                "response": result,
                "timestamp": datetime.now().isoformat(),
            }
            collection.insert_one(log_entry)

            return AgentResponse(
                task_type=request.task_type,
                model_used=DEFAULT_MODEL,
                result=result,
                conversation_id=conversation_id,
            )
        except httpx.HTTPError as err:
            raise HTTPException(
                status_code=500, detail=f"Ollama connection error: {str(err)}"
            )

# =========================================================
# RAG 관련 헬퍼 함수 (임베딩 / 유사도 검색)
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
# Obsidian vault 파싱 / 색인
# =========================================================
def strip_markdown_noise(line: str) -> str:
    """마크다운 문법 요소 제거"""
    line = re.sub(r'^#+\s*', '', line)            # 헤더
    line = re.sub(r'^[-*]\s*', '', line)           # 불릿
    line = re.sub(r'==(.+?)==', r'\1', line)       # ==하이라이트==
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)   # **볼드**
    line = re.sub(r'\[\[(.+?)\]\]', r'\1', line)   # [[위키링크]]
    return line.strip()
 
 
def parse_markdown_file(filepath: str):
    """마크다운 파일 하나를 파싱해서 (text, is_correct, correction) 리스트 반환"""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = strip_markdown_noise(raw_line)
            if not line or line.startswith("---"):  # frontmatter, 빈 줄 skip
                continue
            if "=>" in line:
                wrong, correct = line.split("=>", 1)
                wrong, correct = wrong.strip(), correct.strip()
                if wrong:
                    entries.append((wrong, False, correct))
            elif len(line.split()) >= 3:  # 너무 짧은 줄(제목 등)은 제외
                entries.append((line, True, None))
    return entries
 
 
def get_vault_markdown_files(vault_path: str):
    """vault 내 모든 .md 파일 경로 (숨김/설정 폴더 제외)"""
    pattern = os.path.join(vault_path, "**", "*.md")
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if ".obsidian" not in f and "/.trash/" not in f]
 
 
async def index_obsidian_vault(vault_path: str = OBSIDIAN_VAULT_PATH):
    """vault 전체를 스캔해서, 변경된 파일만 재색인"""
    if not os.path.isdir(vault_path):
        print(f">>> Obsidian vault 경로를 찾을 수 없음: {vault_path}")
        return 0
 
    added, updated_files = 0, 0
 
    for filepath in get_vault_markdown_files(vault_path):
        mtime = os.path.getmtime(filepath)
        meta = file_meta_collection.find_one({"path": filepath})
 
        # mtime이 그대로면 변경 없음 -> 스킵
        if meta and meta.get("mtime") == mtime:
            continue
 
        # 변경됐으면 기존 이 파일 소스의 문장들 제거 후 재삽입
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
                "timestamp": datetime.now().isoformat(),
                "source_file": filepath,
            })
            added += 1
 
        file_meta_collection.update_one(
            {"path": filepath},
            {"$set": {"mtime": mtime}},
            upsert=True,
        )
        updated_files += 1
 
    print(f">>> Obsidian 색인 완료: 파일 {updated_files}개 갱신, 문장 {added}개 추가")
    return added
 
 
@app.on_event("startup")
async def startup_indexing():
    try:
        await index_obsidian_vault()
    except Exception as e:
        print(f">>> Obsidian 색인 중 오류: {e}")
 
 
@app.post("/sentences/reindex")
async def reindex_sentences():
    added = await index_obsidian_vault()
    return {"added": added}
 
 # =========================================================
# 문장 등록 / 분석 / 조회 엔드포인트
# =========================================================
@app.post("/sentences/add")
async def add_sentence(entry: SentenceEntry):
    """앱에서 직접 문장을 하나씩 등록할 때 사용 (Obsidian 색인과는 별개)"""
    embedding = await get_embedding(entry.text)
    doc = entry.model_dump()
    doc["embedding"] = embedding
    doc["timestamp"] = datetime.now().isoformat()
    doc["source_file"] = None
    result = sentence_collection.insert_one(doc)
    return {"id": str(result.inserted_id)}
 
 
@app.post("/analyze")
async def analyze_sentence(request: AnalyzeRequest):
    similar = await retrieve_similar_sentences(request.sentence, request.top_k)
 
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
        f'분석할 문장: "{request.sentence}"\n\n'
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
        "sentence": request.sentence,
        "retrieved_context": [d["text"] for _, d in similar],
        "analysis": result_text,
    }
 
 
@app.get("/sentences/mistakes")
async def get_mistakes():
    docs = list(sentence_collection.find({"is_correct": False}))
    for d in docs:
        d["_id"] = str(d["_id"])
        d.pop("embedding", None)  # 응답에서 임베딩은 제외
    return {"mistakes": docs}
 


 
@app.get("/review")
async def review_mistakes(count: int = 1):
    """저장된 문장(암기용 표현 포함) 중 무작위로 뽑아 연습 퀴즈를 만들어주는 엔드포인트"""
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
 
# =========================================================
# 헬스체크 / 로그 / 루트 페이지
# =========================================================

@app.get("/health")
async def health_check():
    try:
        mongo_client.admin.command('ping')
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
        
    return {
        "status": "ok", 
        "engine": "Ollama Local",
        "mongodb": db_status
    }


@app.get("/logs")
async def get_logs():
    logs_cursor = collection.find().sort("timestamp", pymongo.DESCENDING)

    logs_list = []
    for log in logs_cursor:
        log["_id"] = str(log["_id"])
        logs_list.append(log)

    return {"logs": logs_list}


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        request,          # 첫 번째 인자로 request
        "index.html",
        {}                 # context에는 request 넣을 필요 없음
    )