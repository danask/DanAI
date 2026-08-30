"""
Application entry point.

Creates the FastAPI app, wires up the lifespan (Obsidian vault indexing on
startup + the 6-hour news scheduler), mounts the static frontend assets,
and includes the domain routers (chat, news, word-memory).
"""
import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import logger
from app.database import mongo_client
from app.routers import chat, news, word_memory
from app.services.news_service import fetch_and_summarize_news_job
from app.services.word_service import index_obsidian_vault

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    try:
        await index_obsidian_vault()
    except Exception as e:
        logger.error(f"Error during Obsidian indexing: {e}")

    # Register and start the 6-hour interval news scheduler
    scheduler.add_job(
        fetch_and_summarize_news_job,
        trigger="interval",
        hours=6,
        id="news_cron_job",
        replace_existing=True,
    )
    scheduler.start()

    # Immediately run one background news collection on server startup (async)
    asyncio.create_task(fetch_and_summarize_news_job())

    yield

    # Shutdown event
    scheduler.shutdown()


app = FastAPI(title="Local Ollama Agent API", version="1.0.0", lifespan=lifespan)

# Static frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Domain routers
app.include_router(chat.router)
app.include_router(news.router)
app.include_router(word_memory.router)


@app.get("/health")
async def health_check():
    try:
        mongo_client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"

    return {
        "status": "ok",
        "engine": "Ollama Local",
        "mongodb": db_status,
    }


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
