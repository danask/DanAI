# DanAI

A local FastAPI + MongoDB agent that talks to a local [Ollama](https://ollama.com) model,
provides an English vocabulary/mistake review workflow backed by an Obsidian vault,
and periodically collects and summarizes Canadian/international news.

This repository has been refactored from a single 737-line `main.py` into a
standard FastAPI package layout, split by domain.

## Project Structure

```
DanAI/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point and router inclusion
│   ├── config.py             # Environment configuration
│   ├── database.py           # MongoDB client setup
│   ├── routers/               # Endpoints split into 3 core domains
│   │   ├── news.py            # News aggregation & summarization APIs
│   │   ├── chat.py            # Chatbot & LLM interaction APIs
│   │   └── word_memory.py     # Vocabulary / word memory management APIs
│   └── services/               # Domain-specific business logic
│       ├── news_service.py
│       ├── chat_service.py
│       └── word_service.py
├── static/                      # Frontend assets
│   ├── index.html
│   ├── app.js
│   └── style.css
├── .env
├── requirements.txt
└── README.md
```

## Running the Server

Install dependencies, then start the app with Uvicorn from the project root:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

This runs the server in development mode with live reloading. The app expects
a running local Ollama instance (`OLLAMA_BASE_URL`) and a running MongoDB
instance (`MONGODB_URI`) — see `.env` for defaults.

## Requirements

- Python 3.8+
- A running MongoDB instance
- A running local Ollama instance with the configured models pulled

## API Overview

| Domain       | Endpoints                                                              |
|--------------|--------------------------------------------------------------------------|
| Chat         | `POST /agent/run`, `GET /logs`                                          |
| Word memory  | `POST /sentences/reindex`, `POST /sentences/add`, `POST /analyze`, `GET /sentences/mistakes`, `GET /review` |
| News         | `GET /news`, `GET /news/latest`, `POST /news/canada-summary`            |
| Misc         | `GET /health`, `GET /` (serves `static/index.html`)                     |

## Notes on the Refactor

- Korean **code comments and docstrings** were translated to English for
  readability.
- Korean **prompt text sent to the LLM** and **user-facing API/UI strings**
  were intentionally left unchanged, since translating them would change the
  app's actual behavior or the Korean-language product experience.
- Commented-out code (e.g. the older `/news/canada-summary` implementation in
  `app/services/news_service.py`) was preserved as-is, not deleted.
- The original `main.py` created the `FastAPI()` app twice and registered a
  now-unreachable `@app.on_event("startup")` handler (leftover from
  incremental edits). This has been consolidated into a single `lifespan`
  handler in `app/main.py` with the same startup/shutdown behavior.
