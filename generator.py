"""
AI Response Generator
- Softer prompt: بيجاوب ويفيد بدل ما يرفض، من غير ما يخترع أرقام
- chat() بترجّع reply + sources مع بعض (مفيش retrieve مرتين)
- Regular + streaming
"""

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CONTEXT_CHARS
from retriever import StudyRetriever

client = genai.Client(api_key=GEMINI_API_KEY)

# ── System Prompt ──
SYSTEM_PROMPT = """أنت "UniBot" — مساعد أكاديمي ذكي لطلاب الجامعة، بتشرح زي معيد صبور.

═══════════════════════════════════════
📌 قواعد الإجابة:
═══════════════════════════════════════
1. دوّر كويس جداً في كل المعلومات المرفقة (Context) — المعلومة ممكن تكون جوه
   جدول أو بصياغة مختلفة عن سؤال الطالب. لو لقيتها حتى جزئياً، جاوب كامل وطبيعي.
2. لو معلومة محددة (كود مادة، عدد ساعات، متطلب سابق، شرط GPA، رقم مادة) مش
   موجودة في الـ Context — متخترعش رقم أو كود من دماغك أبداً. بدل ما ترفض:
     • اديله أحسن إرشاد عام مفيد من معرفتك الأكاديمية،
     • ووضّح بسطر إن دي قاعدة عامة محتاجة يتأكد منها من اللائحة الرسمية،
     • واسأله سؤال توضيحي لو محتاج.
3. ممنوع تماماً ترد بـ "معنديش المعلومة دي" وتقف. دايماً اطلع للطالب بحاجة تنفعه.
4. استخدم العربي مع الاحتفاظ بالمصطلحات التقنية بالإنجليزي.

═══════════════════════════════════════
🎯 الأسلوب (مهم):
═══════════════════════════════════════
- مختصر ومفيد — من غير رغي ولا حشو ولا مقدمات طويلة.
- لو السؤال بسيط رد بسيط؛ لو محتاج تفصيل فصّل بنقاط واضحة ومرتّبة.
- لو فيه خطوات، رقّمها. لو فيه مقارنة، اعمل جدول صغير.
- في الشرح الطويل بس، اقفل بـ "خلاصة" مختصرة.

═══════════════════════════════════════
🧠 توليد أسئلة (لو اتطلب منك):
═══════════════════════════════════════
- 3 مستويات: سهل / متوسط / صعب، كل مستوى 3-4 أسئلة MCQ.
- 4 اختيارات لكل سؤال + الإجابة الصحيحة + شرح قصير.
- الأسئلة من الـ Context بس ومتنوعة.

═══════════════════════════════════════
⚠️ ممنوعات:
═══════════════════════════════════════
- متقولش "بناءً على السياق المتاح" أو "حسب المعلومات المقدمة" — رد طبيعي كأنك فاهم المادة.
- متكررش السؤال في إجابتك.
- متدّيش إجابة من كلمة/جملة واحدة لما السؤال محتاج تفصيل."""


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into structured context (grouped by source)."""
    if not results:
        return ""

    sources = {}
    for r in results:
        source = r.get("metadata", {}).get("source", "unknown")
        page = r.get("metadata", {}).get("page", "?")
        sources.setdefault(source, []).append({
            "page": page,
            "content": r["content"],
            "score": r.get("score", 0),
        })

    parts, total_chars = [], 0
    for source_name, chunks in sources.items():
        for chunk in chunks:
            entry = f"[المصدر: {source_name} | صفحة {chunk['page']}]\n{chunk['content']}"
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def format_chat_history(history: list[dict]) -> str:
    """Format chat history into a readable conversation log."""
    if not history:
        return ""
    lines = ["═══ سجل المحادثة السابقة ═══"]
    for msg in history:
        role = "🧑 الطالب" if msg["role"] == "user" else "🤖 المساعد"
        content = msg["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_prompt(user_message, context, history_text, source_names):
    return f"""{SYSTEM_PROMPT}

{history_text}

═══════════════════════════════════════
📚 المعلومات المتاحة من المنهج (Context):
═══════════════════════════════════════
{context}

═══════════════════════════════════════
❓ سؤال الطالب الآن:
═══════════════════════════════════════
{user_message}

(المصادر المستخدمة: {', '.join(source_names)})
"""


class StudyGenerator:
    def __init__(self):
        self.retriever = StudyRetriever()

    def chat(self, user_message: str, course_id: str = "default",
             history: list[dict] = None) -> dict:
        """
        Generate an AI response using RAG.
        Returns: {"reply": str, "sources": list[str]}
        """
        results = self.retriever.retrieve(user_message, course_id=course_id)

        sources = sorted({
            f"{r['metadata'].get('source', '')} - p.{r['metadata'].get('page', '?')}"
            for r in results
        })

        if not results:
            return {
                "reply": ("لسه مفيش مواد متفهرسة للمادة دي عشان أرد منها. "
                          "بس قولي سؤالك بالظبط وأحاول أساعدك بإرشاد عام."),
                "sources": [],
            }

        context = format_context(results)
        history_text = format_chat_history(history or [])
        source_names = sorted({r.get("metadata", {}).get("source", "") for r in results})
        prompt = _build_prompt(user_message, context, history_text, source_names)

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
            )
            return {"reply": response.text, "sources": sources}
        except Exception as e:
            return {"reply": f"حصل خطأ في الاتصال بالذكاء الاصطناعي: {e}", "sources": sources}

    def chat_stream(self, user_message: str, course_id: str = "default",
                    history: list[dict] = None):
        """Same as chat() but yields text chunks for streaming."""
        results = self.retriever.retrieve(user_message, course_id=course_id)

        if not results:
            yield ("لسه مفيش مواد متفهرسة للمادة دي عشان أرد منها. "
                   "بس قولي سؤالك بالظبط وأحاول أساعدك بإرشاد عام.")
            return

        context = format_context(results)
        history_text = format_chat_history(history or [])
        source_names = sorted({r.get("metadata", {}).get("source", "") for r in results})
        prompt = _build_prompt(user_message, context, history_text, source_names)

        try:
            response = client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"حصل خطأ في الاتصال بالذكاء الاصطناعي: {e}"
