"""
Vocabulary / Word memory management APIs.
"""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services import word_service

router = APIRouter(tags=["word-memory"])


class SentenceEntry(BaseModel):
    text: str
    is_correct: bool = True
    correction: Optional[str] = None   # correct sentence, when this entry is a mistake
    note: Optional[str] = None


class AnalyzeRequest(BaseModel):
    sentence: str
    top_k: int = 3


@router.post("/sentences/reindex")
async def reindex_sentences():
    added = await word_service.index_obsidian_vault()
    return {"added": added}


@router.post("/sentences/add")
async def add_sentence(entry: SentenceEntry):
    """Registers a single sentence directly from the app (separate from Obsidian indexing)."""
    return await word_service.add_sentence(entry.model_dump())


@router.post("/analyze")
async def analyze_sentence(request: AnalyzeRequest):
    return await word_service.analyze_sentence(request.sentence, request.top_k)


@router.get("/sentences/mistakes")
async def get_mistakes():
    return word_service.get_mistakes()


@router.get("/review")
async def review_mistakes(count: int = 1):
    """Randomly samples stored sentences and generates a practice quiz."""
    return await word_service.review_mistakes(count)
