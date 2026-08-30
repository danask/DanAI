"""
Chatbot & LLM interaction APIs.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.services import chat_service

router = APIRouter(tags=["chat"])


class AgentRequest(BaseModel):
    prompt: str
    task_type: str = "general"  # Options: general, code_review, summarize


class AgentResponse(BaseModel):
    task_type: str
    model_used: str
    result: str
    conversation_id: str


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent_task(request: AgentRequest):
    result = await chat_service.run_agent_task(request.task_type, request.prompt)
    return AgentResponse(**result)


@router.get("/logs")
async def get_logs():
    return chat_service.get_logs()
