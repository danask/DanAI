"""
News aggregation & summarization APIs.
"""
from fastapi import APIRouter, Query

from app.services import news_service

router = APIRouter(prefix="/news", tags=["news"])


@router.get("")
async def get_news_list(limit: int = Query(default=5, ge=1, le=20)):
    """Returns the collected news list stored in the DB, most recent first."""
    return news_service.get_news_list(limit)


@router.get("/latest")
async def get_latest_news():
    """Returns the single most recent stored news summary."""
    return news_service.get_latest_news()


@router.post("/canada-summary")
async def trigger_canada_news_summary():
    """Manually triggers immediate news collection and summarization."""
    return await news_service.trigger_canada_news_summary()
