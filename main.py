import os
import secrets
from datetime import datetime

import httpx
import pymongo
from fastapi import FastAPI, HTTPException, Request  # Request 추가
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

app = FastAPI(title="Local Ollama Agent API", version="1.0.0")

# Static 및 Templates 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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