"""
MongoDB client and collection setup, shared across the whole app.
"""
from pymongo import MongoClient

from app.config import MONGODB_URI

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["conversation_logs"]

collection = db["logs"]
sentence_collection = db["sentences"]      # English study sentences store
file_meta_collection = db["file_meta"]     # Tracks Obsidian file mtimes (incremental indexing)
news_collection = db["news_summaries"]     # Collected / summarized news store
