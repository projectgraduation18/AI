---
title: UniLMS AI Engine
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# UniLMS AI Microservice

RAG-based AI tutoring engine using FAISS + BGE-M3 + Gemini.

محرك ذكاء اصطناعي لطلاب الجامعة: شات على مواد الكورس، تلخيص، توليد أسئلة بـ 3 مستويات،
ومستشار أكاديمي لترشيح المواد.

## Endpoints

| Endpoint | Method | الوصف |
|----------|--------|-------|
| `/api/chat` | POST | شات على مواد الكورس (مع history) |
| `/api/chat/stream` | POST | شات Streaming (SSE) |
| `/api/ingest` | POST | فهرسة PDF جديد |
| `/api/summarize` | POST | تلخيص مادة/ملف |
| `/api/quiz` | POST | أسئلة MCQ بـ 3 مستويات |
| `/api/advisor/recommend` | POST | توصية بالمواد (مستشار أكاديمي) |
| `/api/advisor/recommend/stream` | POST | نفسه Streaming (SSE) |
| `/api/courses` | GET | الكورسات المفهرسة |
| `/health` | GET | فحص صحة السيرفر |
| `/docs` | GET | Swagger UI |

## Configuration

محتاج المتغير ده كـ Secret في إعدادات الـ Space:

- `GEMINI_API_KEY` — مفتاح Gemini API

## Notes

- موديل الـ embeddings: `BAAI/bge-m3` (بينزّل أول تشغيل ~2.27GB، فالإقلاع الأول بياخد وقت).
- الفهرس الجاهز موجود في `database/` فمش محتاج إعادة فهرسة عند الإقلاع.
