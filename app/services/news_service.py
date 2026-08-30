"""
Business logic for Canadian / international news aggregation and summarization.
"""
import re

import feedparser
import httpx
import pymongo
from fastapi import HTTPException
from pymongo import DESCENDING

from app.config import CANADA_NEWS_RSS_URLS, NEWS_MODEL, OLLAMA_BASE_URL, logger
from app.database import db, news_collection
from app.utils import current_timestamp_ms

# =========================================================
# APScheduler job & collection logic
# =========================================================
async def fetch_and_summarize_news_job():
    """Background job that runs every 6 hours to collect and summarize news."""
    logger.info("==================================================")
    logger.info("[CRON_TASK] Starting scheduled 6-hour news collection & summarization")
    logger.info("==================================================")

    try:
        raw_news_entries = []

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for url in CANADA_NEWS_RSS_URLS:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.text)
                    for item in feed.entries[:5]:
                        raw_news_entries.append(item)
                except Exception as net_err:
                    logger.warning(f"[CRON_TASK] {url} collection failed: {str(net_err)}")

        if not raw_news_entries:
            logger.error("[CRON_TASK] Failed to fetch articles from all RSS feeds.")
            return

        raw_news_text = ""
        for idx, entry in enumerate(raw_news_entries[:10], 1):
            title = entry.get("title", "No Title")
            summary_raw = entry.get("summary", entry.get("description", "No Summary"))
            clean_summary = re.sub(r'<[^>]+>', '', summary_raw)[:1000].strip()
            raw_news_text += f"{idx}. Title: {title}\nSummary: {clean_summary}\n\n"

        system_instruction = (
            "You are a professional Canadian and international news analyst.\n"
            "Summarize each provided news item into a detailed Korean report.\n\n"
            "STRICT RULES:\n"
            "0. Header Information: If weather and currency exchange rate data are provided in the input, start the report with a brief summary of 'Today's Greater Vancouver Weather' and 'CAD/KRW Exchange Rate'.\n"            
            "1. Length Constraint: Each summarized news item MUST be detailed and comprehensive (at least 100-150 Korean characters per item).\n"
            "2. Detail Level: Do not write simple 1-sentence summaries. Include core background, main events, figures involved, and implications/current situation for every single news item.\n"
            "3. Proper Nouns & Names: Keep ALL proper nouns, place names, people's names, organization names, and abbreviations in their ORIGINAL English form (or write English in parentheses alongside the Korean translation, e.g., 'David Eby', 'Langley Township', 'Angusford-Mission').\n"
            "4. Format: Present each news item with a bold title and 3-4 informative bullet points or structured detailed paragraphs."
        )

        formatted_prompt = f"System: {system_instruction}\n\n[News Feed Data]:\n{raw_news_text}"

        payload = {
            "model": NEWS_MODEL,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,        # expanded context window
                "num_predict": 8192,    # expanded generation limit for a minimum output length
                "temperature": 0.2
            },
        }

        async with httpx.AsyncClient(timeout=None) as http_client:
            response = await http_client.post(OLLAMA_BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            summary_result = data.get("response", "")

            # Save to the news_summaries MongoDB collection
            doc = {
                "model_used": NEWS_MODEL,
                "summary": summary_result,
                "articles_count": len(raw_news_entries[:20]),
                "created_at": current_timestamp_ms(),  # Unix epoch ms (UTC)
            }
            news_collection.insert_one(doc)
            logger.info("[CRON_TASK] News summary saved to the DB")

    except Exception as e:
        logger.error(f"[CRON_TASK] Error during automatic news collection: {str(e)}", exc_info=True)


# =========================================================
# News lookup / collection endpoint logic
# =========================================================
def get_news_list(limit: int) -> dict:
    """Returns the collected news stored in the DB, most recent first."""
    cursor = news_collection.find().sort("created_at", pymongo.DESCENDING).limit(limit)

    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)

    return {"count": len(results), "news": results}


def get_latest_news() -> dict:
    """Returns the single most recent news summary stored in the DB."""
    # Sort descending by created_at to fetch the most recent document
    latest_news = db.news_summaries.find_one(
        sort=[("created_at", DESCENDING)]
    )

    # Fallback to descending _id for legacy documents that lack created_at
    if not latest_news:
        latest_news = db.news_summaries.find_one(
            sort=[("_id", DESCENDING)]
        )

    if not latest_news:
        raise HTTPException(status_code=404, detail="저장된 뉴스 데이터가 없습니다.")

    # Convert the MongoDB ObjectId to avoid JSON serialization errors
    latest_news["_id"] = str(latest_news["_id"])

    return latest_news


async def trigger_canada_news_summary() -> dict:
    """Manually triggers an immediate news collection and save to the DB."""
    await fetch_and_summarize_news_job()
    return get_latest_news()
