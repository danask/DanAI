"""
Business logic for the chatbot / LLM agent endpoints.
"""
import secrets

import httpx
import pymongo
from fastapi import HTTPException

from app.config import DEFAULT_MODEL, OLLAMA_BASE_URL
from app.database import collection
from app.utils import current_timestamp_ms


def build_agent_prompt(task_type: str, user_prompt: str) -> str:
    """Builds an internal agent system prompt according to the task type."""
    if task_type == "code_review":
        system_instruction = (
            "Your name is DanAI. You speak Korean."            
            "You are an expert code reviewer. Analyze the code provided by the user, "
            "identify potential bugs or performance bottlenecks, and suggest improvements."
        )
    elif task_type == "summarize":
        system_instruction = (
            "Your name is DanAI. You speak Korean."            
            "You are a conciseness expert. Summarize the following input into key bullet points "
            "without losing core context."
        )
    elif task_type == "english":
        system_instruction = (
            "Your name is DanAI. You speak Korean."            
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


async def run_agent_task(task_type: str, prompt: str) -> dict:
    """Sends the prompt to Ollama and logs the conversation to MongoDB."""
    conversation_id = secrets.token_hex(16)
    formatted_prompt = build_agent_prompt(task_type, prompt)

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
                "task_type": task_type,
                "model_used": DEFAULT_MODEL,
                "prompt": prompt,
                "response": result,
                "timestamp": current_timestamp_ms(),  # Unix epoch ms (UTC)
            }
            collection.insert_one(log_entry)

            return {
                "task_type": task_type,
                "model_used": DEFAULT_MODEL,
                "result": result,
                "conversation_id": conversation_id,
            }
        except httpx.HTTPError as err:
            raise HTTPException(
                status_code=500, detail=f"Ollama connection error: {str(err)}"
            )


def get_logs() -> dict:
    """Returns all stored agent conversation logs, most recent first."""
    logs_cursor = collection.find().sort("timestamp", pymongo.DESCENDING)

    logs_list = []
    for log in logs_cursor:
        log["_id"] = str(log["_id"])
        logs_list.append(log)

    return {"logs": logs_list}
