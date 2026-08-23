import os
import secrets
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DanAI Local Chat</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; display: flex; justify-content: center; }
            .chat-container { width: 100%; max-width: 800px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); display: flex; flex-direction: column; height: 90vh; }
            .chat-header { padding: 16px 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
            .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
            .message { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
            .user-msg { align-self: flex-end; background-color: #007aff; color: white; border-bottom-right-radius: 2px; }
            .agent-msg { align-self: flex-start; background-color: #e9e9eb; color: #333; border-bottom-left-radius: 2px; }
            .input-area { padding: 16px; border-top: 1px solid #eee; display: flex; gap: 10px; }
            select, input, button { padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }
            input[type="text"] { flex: 1; }
            button { background-color: #007aff; color: white; border: none; cursor: pointer; font-weight: bold; }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h2>DanAI Agent Chat</h2>
                <select id="taskType">
                    <option value="general">General Agent</option>
                    <option value="code_review">Code Reviewer</option>
                    <option value="summarize">Summarizer</option>
                </select>
            </div>
            <div class="chat-box" id="chatBox"></div>
            <div class="input-area">
                <input type="text" id="promptInput" placeholder="메시지를 입력하세요..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button id="sendBtn" onclick="sendMessage()">전송</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('promptInput');
                const taskType = document.getElementById('taskType').value;
                const sendBtn = document.getElementById('sendBtn');
                const prompt = input.value.trim();

                if (!prompt) return;

                appendMessage(prompt, 'user-msg');
                input.value = '';
                input.disabled = true;
                sendBtn.disabled = true;

                const loadingId = appendMessage('생성 중...', 'agent-msg');

                try {
                    const res = await fetch('/agent/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt, task_type: taskType })
                    });
                    const data = await res.json();
                    
                    document.getElementById(loadingId).innerText = data.result || '응답을 받지 못했습니다.';
                } catch (err) {
                    document.getElementById(loadingId).innerText = '오류가 발생했습니다: ' + err.message;
                } finally {
                    input.disabled = false;
                    sendBtn.disabled = false;
                    input.focus();
                }
            }

            function appendMessage(text, className) {
                const chatBox = document.getElementById('chatBox');
                const msgDiv = document.createElement('div');
                const msgId = 'msg-' + Date.now();
                msgDiv.id = msgId;
                msgDiv.className = 'message ' + className;
                msgDiv.innerText = text;
                chatBox.appendChild(msgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                return msgId;
            }
        </script>
    </body>
    </html>
    """