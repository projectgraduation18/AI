# UniLMS AI Microservice

RAG-based AI tutoring engine using FAISS + Gemini.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your GEMINI_API_KEY
```

## Usage

```bash
# 1. Place PDFs in data/{course_id}/
mkdir -p data/CS101
cp your_lectures/*.pdf data/CS101/

# 2. Build vector index
python ingest.py CS101

# 3. Start the server
uvicorn api:app --reload --port 8000
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with AI (with history) |
| `/api/chat/stream` | POST | Streaming chat (SSE) |
| `/api/ingest` | POST | Index new materials |
| `/api/courses` | GET | List indexed courses |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |
