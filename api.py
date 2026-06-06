"""
UniLMS AI Microservice API
- POST /api/chat               → Regular chat (history + course_id)
- POST /api/chat/stream        → Streaming chat (SSE)
- POST /api/ingest             → Trigger document ingestion for a course
- POST /api/summarize          → تلخيص مادة/ملف
- POST /api/quiz               → أسئلة بـ 3 مستويات (وزياده عند الطلب)
- POST /api/advisor/recommend  → توصيات مواد
- GET  /api/courses            → List indexed courses
- GET  /health                 → Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from generator import StudyGenerator
from ingest import ingest_from_url
from study_tools import summarize_material, generate_quiz

# ── App Setup ──
app = FastAPI(
    title="UniLMS AI Engine",
    description="RAG-based AI tutoring service for UniLMS",
    version="2.1.0",
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
    file_url: str | None = None
    file_name: str | None = None
    material_id: str | None = None

class SummarizeRequest(BaseModel):
    course_id: str
    source: str | None = None  # اسم ملف معيّن، أو None = الكورس كله

class QuizRequest(BaseModel):
    course_id: str
    source: str | None = None
    questions_per_level: int = 3
    previous_questions: list[str] = []  # للـ "عايز زياده" عشان ميكررش


# ── Chat Endpoints ──
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Regular chat. Returns full AI response + sources (retrieve مرة واحدة بس)."""
    try:
        history = [{"role": h.role, "content": h.content} for h in request.history]
        result = ai_bot.chat(
            user_message=request.message,
            course_id=request.course_id,
            history=history,
        )
        return ChatResponse(status="success", response=result["reply"], sources=result["sources"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat using SSE (token-by-token)."""
    history = [{"role": h.role, "content": h.content} for h in request.history]

    async def event_generator():
        try:
            for chunk in ai_bot.chat_stream(
                user_message=request.message,
                course_id=request.course_id,
                history=history,
            ):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)
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
            "X-Accel-Buffering": "no",
        },
    )


# ── Ingestion ──
@app.post("/api/ingest")
def ingest_documents(request: IngestRequest):
    """Download PDF → chunk → embed → store in FAISS."""
    if not request.file_url or not request.file_name:
        raise HTTPException(status_code=400, detail="file_url and file_name are required")

    try:
        result = ingest_from_url(
            course_id=request.course_id,
            file_url=request.file_url,
            file_name=request.file_name,
        )
        try:
            ai_bot.retriever.reload_course(request.course_id)
        except FileNotFoundError:
            pass

        return {
            "status": result["status"],
            "message": f"Ingested {result.get('chunks', 0)} chunks for course '{request.course_id}'",
            "course_id": request.course_id,
            "chunks_created": result.get("chunks", 0),
            "material_id": request.material_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


# ── Summarize & Quiz ──
@app.post("/api/summarize")
def summarize(request: SummarizeRequest):
    """
    يلخّص ملف معيّن (source=اسم الملف زي اللي اتبعت في /api/ingest) أو الكورس كله.
    الباك اند بيناديها بعد ما اليوزر يضيف PDF.
    """
    try:
        return summarize_material(ai_bot.retriever, request.course_id, request.source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarize error: {str(e)}")


@app.post("/api/quiz")
def quiz(request: QuizRequest):
    """
    أسئلة MCQ بـ 3 مستويات (easy/medium/hard) من المادة.
    للـ "عايز زياده": ابعت نفس الطلب + previous_questions = نصوص الأسئلة اللي فاتت.
    """
    try:
        return generate_quiz(
            ai_bot.retriever,
            request.course_id,
            request.source,
            questions_per_level=request.questions_per_level,
            previous_questions=request.previous_questions,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz error: {str(e)}")


@app.get("/api/courses")
def list_courses():
    """List all indexed courses."""
    courses = ai_bot.retriever.get_available_courses()
    return {"courses": courses, "count": len(courses)}


# ══════════════════════════════════════════════════════
# Academic Advisor Endpoints
# ══════════════════════════════════════════════════════
from advisor import generate_recommendation, generate_recommendation_stream

@app.post("/api/advisor/recommend")
def advisor_recommend(data: dict):
    """AI-powered course recommendations."""
    try:
        return generate_recommendation(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advisor error: {str(e)}")


@app.post("/api/advisor/recommend/stream")
async def advisor_recommend_stream(data: dict):
    """Streaming version of recommendations."""
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "UniLMS AI Engine", "version": "2.1.0"}


@app.get("/")
def home():
    return {
        "message": "UniLMS AI Engine is running!",
        "docs": "/docs",
        "endpoints": [
            "POST /api/chat",
            "POST /api/chat/stream",
            "POST /api/ingest",
            "POST /api/summarize",
            "POST /api/quiz",
            "POST /api/advisor/recommend",
            "GET /api/courses",
            "GET /health",
        ],
    }
