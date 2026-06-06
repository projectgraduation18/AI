"""
Study Tools — تلخيص المادة + توليد أسئلة بـ 3 مستويات صعوبة.
- بيقرأ نص المادة من فهرس الكورس (مفيش إعادة تحميل للـ PDF).
- التلخيص بيرجّع Markdown، الأسئلة بترجّع JSON منظّم جاهز للفرونت/الباك.
- "عايز زياده" = استدعِ generate_quiz تاني وابعت previous_questions عشان ميكررش.
"""

import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_MATERIAL_CHARS

client = genai.Client(api_key=GEMINI_API_KEY)


SUMMARY_PROMPT = """أنت معيد بتلخّص مادة لطالب قبل الامتحان.
لخّص المحتوى ده تلخيص مفيد ومنظّم ومختصر:
- ابدأ بفكرة المادة في سطرين.
- بعدها أهم المفاهيم/النقاط في bullet points واضحة.
- أي تعريفات أو قوانين مهمة اذكرها بالظبط.
- اقفل بـ "خلاصة سريعة" 3-4 نقاط.
عربي + المصطلحات التقنية إنجليزي. من غير حشو.

المحتوى:
─────────────
{content}
─────────────
"""

QUIZ_PROMPT = """أنت معيد بتعمل امتحان MCQ لطالب من المحتوى ده.
اعمل أسئلة اختيار من متعدد بـ 3 مستويات: easy / medium / hard.
كل مستوى فيه {n} أسئلة. كل سؤال 4 اختيارات (A,B,C,D)، إجابة صحيحة واحدة، وشرح قصير.
الأسئلة لازم تكون من المحتوى ده بس، متنوعة، ومتكررش نفس الفكرة.
{avoid}
رجّع JSON بس من غير أي كلام تاني، بالشكل ده بالظبط:
{{
  "questions": [
    {{
      "level": "easy",
      "question": "نص السؤال",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "شرح قصير ليه دي الإجابة الصحيحة"
    }}
  ]
}}

المحتوى:
─────────────
{content}
─────────────
"""


def _safe_json(text: str) -> dict:
    """Parse JSON even لو الموديل لفّه بـ ```json fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def summarize_material(retriever, course_id: str, source: str | None = None) -> dict:
    """يلخّص ملف معيّن (source=اسم الملف) أو الكورس كله (source=None)."""
    content = retriever.get_material_text(course_id, source=source, max_chars=MAX_MATERIAL_CHARS)
    if not content.strip():
        return {"status": "error",
                "summary": "مفيش محتوى متفهرس للمادة دي. اتأكد إن الـ PDF اتعمله ingest الأول."}
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=SUMMARY_PROMPT.format(content=content),
        )
        return {"status": "success", "summary": resp.text, "source": source or "كل المادة"}
    except Exception as e:
        return {"status": "error", "summary": f"حصل خطأ: {e}"}


def generate_quiz(retriever, course_id: str, source: str | None = None,
                  questions_per_level: int = 3,
                  previous_questions: list[str] | None = None) -> dict:
    """
    أسئلة بـ 3 مستويات صعوبة.
    لطلب "زياده": ابعت previous_questions (نصوص الأسئلة اللي اتسألت) عشان ميكررهاش.
    """
    content = retriever.get_material_text(course_id, source=source, max_chars=MAX_MATERIAL_CHARS)
    if not content.strip():
        return {"status": "error", "questions": [],
                "message": "مفيش محتوى متفهرس للمادة دي. اعمل ingest الأول."}

    avoid = ""
    if previous_questions:
        joined = "\n".join(f"- {q}" for q in previous_questions[-40:])
        avoid = f"الأسئلة دي اتسألت قبل كده — اعمل أسئلة جديدة مختلفة عنها تماماً:\n{joined}\n"

    prompt = QUIZ_PROMPT.format(n=questions_per_level, avoid=avoid, content=content)
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = _safe_json(resp.text)
        questions = data.get("questions", [])
        return {"status": "success", "questions": questions,
                "count": len(questions), "source": source or "كل المادة"}
    except Exception as e:
        return {"status": "error", "questions": [], "message": f"حصل خطأ: {e}"}
