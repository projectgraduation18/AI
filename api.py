"""
UniLMS AI Microservice API
- POST /api/chat          → Regular chat (with history + course_id)
- POST /api/chat/stream   → Streaming chat (SSE)
- POST /api/ingest        → Trigger document ingestion for a course
- GET  /api/courses       → List indexed courses
- GET  /health            → Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from generator import StudyGenerator
from ingest import ingest_from_url

# ── App Setup ──
app = FastAPI(
    title="UniLMS AI Engine",
    description="RAG-based AI tutoring service for UniLMS",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_bot = StudyGenerator()


# ── Request/Response Models ──

class ChatHistoryItem(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    message: str
    course_id: str = "default"
    history: list[ChatHistoryItem] = []

class ChatResponse(BaseModel):
    status: str
    response: str
    sources: list[str] = []

class IngestRequest(BaseModel):
    course_id: str
    # Optional: for remote file ingestion from .NET backend
    file_url: str | None = None
    file_name: str | None = None
    material_id: str | None = None


# ── Endpoints ──

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Regular chat endpoint.
    Receives message + course_id + chat history, returns full AI response.
    Called by the .NET backend.
    """
    try:
        history = [{"role": h.role, "content": h.content} for h in request.history]

        reply = ai_bot.chat(
            user_message=request.message,
            course_id=request.course_id,
            history=history
        )

        # Extract source names from retriever results
        results = ai_bot.retriever.retrieve(request.message, course_id=request.course_id)
        sources = list(set(
            f"{r['metadata'].get('source', '')} - p.{r['metadata'].get('page', '?')}"
            for r in results
        ))

        return ChatResponse(status="success", response=reply, sources=sources)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns the AI response token-by-token for real-time display.
    """
    history = [{"role": h.role, "content": h.content} for h in request.history]

    async def event_generator():
        try:
            for chunk in ai_bot.chat_stream(
                user_message=request.message,
                course_id=request.course_id,
                history=history
            ):
                # SSE format: data: <text>\n\n
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            # Signal end of stream
            yield "data: [DONE]\n\n"

        except FileNotFoundError as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        except Exception as e:
            yield f"data: [ERROR] AI Error: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/ingest")
def ingest_documents(request: IngestRequest):
    """
    Ingest a PDF into the vector database.
    Called by the .NET backend after admin uploads a file to Supabase Storage.

    Flow: Backend uploads PDF to Supabase → sends us the URL → we download,
    chunk, embed, and store in FAISS.
    """
    if not request.file_url or not request.file_name:
        raise HTTPException(
            status_code=400,
            detail="file_url and file_name are required"
        )

    try:
        result = ingest_from_url(
            course_id=request.course_id,
            file_url=request.file_url,
            file_name=request.file_name
        )

        # Reload the course index in the retriever cache
        try:
            ai_bot.retriever.reload_course(request.course_id)
        except FileNotFoundError:
            pass  # First file for this course, index will load on first query

        return {
            "status": result["status"],
            "message": f"Ingested {result.get('chunks', 0)} chunks for course '{request.course_id}'",
            "course_id": request.course_id,
            "chunks_created": result.get("chunks", 0),
            "material_id": request.material_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.get("/api/courses")
def list_courses():
    """List all courses that have been indexed."""
    courses = ai_bot.retriever.get_available_courses()
    return {"courses": courses, "count": len(courses)}


# ══════════════════════════════════════════════════════
# Academic Advisor Endpoints
# ══════════════════════════════════════════════════════

from advisor import generate_recommendation, generate_recommendation_stream

@app.post("/api/advisor/recommend")
def advisor_recommend(data: dict):
    """
    Generate AI-powered course recommendations.
    Called by .NET backend with student profile + records + available courses.
    """
    try:
        result = generate_recommendation(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advisor error: {str(e)}")


@app.post("/api/advisor/recommend/stream")
async def advisor_recommend_stream(data: dict):
    """Streaming version of course recommendations."""
    async def event_generator():
        try:
            for chunk in generate_recommendation_stream(data):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "UniLMS AI Engine",
        "version": "2.0.0"
    }


@app.get("/")
def home():
    return {
        "message": "UniLMS AI Engine is running!",
        "docs": "/docs",
        "endpoints": [
            "POST /api/chat",
            "POST /api/chat/stream",
            "POST /api/ingest",
            "GET /api/courses",
            "GET /health"
        ]
    }
