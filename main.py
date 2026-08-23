import os
import secrets
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
import pymongo

# MongoDB connection setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["conversation_logs"]
collection = db["logs"]

app = FastAPI(title="Local Ollama Agent API", version="1.0.0")

# Docker 컨테이너 환경에서는 호스트의 Ollama 접근 시 host.docker.internal을 바라보아야 할 수 있습니다.
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/api/generate"
)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:14b")


class AgentRequest(BaseModel):
    prompt: str
    task_type: str = "general"  # Options: general, code_review, summarize


class AgentResponse(BaseModel):
    task_type: str
    model_used: str
    result: str
    conversation_id: str


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
    else:
        system_instruction = (
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


@app.get("/health")
async def health_check():
    try:
        # ping 명령어로 몽고DB 응답 확인
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
    # timestamp 기준 최신순 정렬 및 _id 필드는 문자열로 변환하여 리스트 생성
    logs_cursor = collection.find().sort("timestamp", pymongo.DESCENDING)

    logs_list = []
    for log in logs_cursor:
        log["_id"] = str(log["_id"])  # ObjectId를 JSON 호환 string으로 변환
        logs_list.append(log)

    return {"logs": logs_list}
