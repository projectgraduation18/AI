"""
AI Response Generator
- Integrates chat history for conversational context
- Uses a detailed system prompt that encourages thorough responses
- Supports both regular and streaming responses
- Formats retrieved context with source attribution
"""

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CONTEXT_CHARS
from retriever import StudyRetriever

client = genai.Client(api_key=GEMINI_API_KEY)

# ── System Prompt ──
# This is the most important part of the RAG pipeline.
# A weak prompt = short/generic answers. A detailed prompt = rich/useful answers.

SYSTEM_PROMPT = """أنت "UniBot" — مساعد أكاديمي ذكي متخصص لطلاب الجامعة.
أنت لست مجرد محرك بحث، بل أنت مُعلّم صبور يشرح ويناقش ويساعد الطالب يفهم فعلاً.

═══════════════════════════════════════
📌 قواعد أساسية (لا تخالفها أبداً):
═══════════════════════════════════════
1. أجب فقط من المعلومات المتاحة (Context) المرفقة أدناه.
2. إذا السؤال خارج نطاق الـ Context تماماً، قل ذلك بوضوح واقترح صياغة أفضل للسؤال.
3. لا تؤلف أو تفترض أي معلومة غير موجودة في الـ Context.
4. استخدم اللغة العربية مع الاحتفاظ بالمصطلحات التقنية بالإنجليزية.

═══════════════════════════════════════
🎯 أسلوب الإجابة (هذا هو الأهم):
═══════════════════════════════════════

عند الشرح:
- ابدأ بملخص سريع (جملة أو جملتين) ثم توسع في التفاصيل.
- استخدم أمثلة عملية وتشبيهات بسيطة من الحياة اليومية لتبسيط المفاهيم.
- اربط المفاهيم ببعضها: "ده بيرتبط بمفهوم X اللي اتكلمنا عنه...".
- لو في خطوات، رقّمها بوضوح.
- استخدم formatting واضح (عناوين فرعية، نقاط، أمثلة كود لو مناسب).
- في نهاية الشرح، اعمل ملخص مختصر للنقاط الرئيسية.

عند توليد أسئلة (Quiz):
- اصنع أسئلة MCQ بثلاث مستويات: سهل، متوسط، صعب.
- كل مستوى 3-4 أسئلة على الأقل.
- اكتب 4 اختيارات لكل سؤال (A, B, C, D).
- بعد كل الأسئلة، اكتب الإجابات الصحيحة مع شرح مختصر لكل إجابة.
- الأسئلة تكون من الـ Context فقط ومتنوعة.

عند الإجابة على أسئلة عن القوانين واللوائح:
- اذكر البند أو المادة بالتحديد لو متاحة.
- اشرح بلغة بسيطة وواضحة.
- لو في استثناءات أو حالات خاصة، اذكرها.

═══════════════════════════════════════
💬 المحادثة والسياق:
═══════════════════════════════════════
- لو الطالب أشار لشيء اتكلمتوا عنه قبل كده (في سجل المحادثة)، ارجع له وابني عليه.
- لو قال "اشرحلي أكتر" أو "وضحلي النقطة دي"، ارجع للموضوع اللي كنتوا بتتكلموا فيه ووسع الشرح.
- خليك طبيعي ومتفاعل — كأنك معيد (TA) بيشرح في سكشن.
- لو الطالب شكرك أو قالك كلام عادي، رد عليه بشكل طبيعي ولطيف.

═══════════════════════════════════════
⚠️ ممنوعات:
═══════════════════════════════════════
- لا تقل "بناءً على السياق المتاح" أو "حسب المعلومات المقدمة" — رد بشكل طبيعي كأنك فاهم المادة.
- لا تكرر السؤال في إجابتك.
- لا تعطي إجابات من كلمة واحدة أو جملة واحدة — وسّع وادي تفاصيل."""


def format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into structured context for the LLM.
    Groups by source file and adds source attribution.
    """
    if not results:
        return ""

    # Group chunks by source
    sources = {}
    for r in results:
        source = r.get("metadata", {}).get("source", "unknown")
        page = r.get("metadata", {}).get("page", "?")
        key = source
        if key not in sources:
            sources[key] = []
        sources[key].append({
            "page": page,
            "content": r["content"],
            "score": r.get("score", 0)
        })

    # Build formatted context
    parts = []
    total_chars = 0
    for source_name, chunks in sources.items():
        for chunk in chunks:
            entry = f"[المصدر: {source_name} | صفحة {chunk['page']}]\n{chunk['content']}"
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def format_chat_history(history: list[dict]) -> str:
    """
    Format chat history into a readable conversation log for the LLM.
    Expected format: [{"role": "user"|"assistant", "content": "..."}]
    """
    if not history:
        return ""

    lines = ["═══ سجل المحادثة السابقة ═══"]
    for msg in history:
        role = "🧑 الطالب" if msg["role"] == "user" else "🤖 المساعد"
        # Truncate long messages in history to save tokens
        content = msg["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


class StudyGenerator:
    def __init__(self):
        self.retriever = StudyRetriever()

    def chat(self, user_message: str, course_id: str = "default",
             history: list[dict] = None) -> str:
        """
        Generate an AI response using RAG.

        Args:
            user_message: The student's question
            course_id: Which course to search in
            history: Recent chat history for context
        """
        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(user_message, course_id=course_id)

        if not results:
            return ("عذراً، لم أجد معلومات متعلقة بسؤالك في مواد هذه المادة. "
                    "حاول إعادة صياغة السؤال أو تحديد الموضوع بشكل أدق.")

        # 2. Format context and history
        context = format_context(results)
        history_text = format_chat_history(history or [])

        # 3. Build the full prompt
        source_names = list(set(
            r.get("metadata", {}).get("source", "") for r in results
        ))

        prompt = f"""{SYSTEM_PROMPT}

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

        # 4. Call Gemini
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"حدث خطأ في الاتصال بالذكاء الاصطناعي: {e}"

    def chat_stream(self, user_message: str, course_id: str = "default",
                    history: list[dict] = None):
        """
        Same as chat() but yields response chunks for streaming.
        Returns a generator that yields text pieces.
        """
        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(user_message, course_id=course_id)

        if not results:
            yield ("عذراً، لم أجد معلومات متعلقة بسؤالك في مواد هذه المادة. "
                   "حاول إعادة صياغة السؤال أو تحديد الموضوع بشكل أدق.")
            return

        # 2. Format context and history
        context = format_context(results)
        history_text = format_chat_history(history or [])

        source_names = list(set(
            r.get("metadata", {}).get("source", "") for r in results
        ))

        prompt = f"""{SYSTEM_PROMPT}

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

        # 3. Call Gemini with streaming
        try:
            response = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"حدث خطأ في الاتصال بالذكاء الاصطناعي: {e}"
