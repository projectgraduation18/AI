"""
Academic Advisor AI Module
- Receives student profile + records + available courses
- Generates personalized course recommendations
- Can also answer general academic advising questions
"""

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

ADVISOR_PROMPT = """أنت مستشار أكاديمي خبير في كلية الحاسبات والمعلومات.
مهمتك مساعدة الطلاب في اختيار المواد المناسبة للتسجيل بناءً على وضعهم الأكاديمي.

═══════════════════════════════════════
📌 قواعد أساسية:
═══════════════════════════════════════
1. رشّح المواد فقط من قائمة "المواد المتاحة" المرفقة — لا تخترع مواد.
2. تأكد إن كل مادة ترشحها، الطالب فعلاً مستوفي متطلباتها (prerequisites).
3. راعي الـ GPA — لو الـ GPA منخفض (أقل من 2.0)، رشح مواد أقل وركز على اللي راسب فيها.
4. لو الطالب راسب في مادة، رشحها تاني كأولوية عالية.
5. حاول توازن بين عدد الساعات (16-18 ساعة عادةً مناسبة).

═══════════════════════════════════════
🎯 أسلوب الرد:
═══════════════════════════════════════
- ابدأ بملخص سريع لوضع الطالب (سنة كام، عدد المواد اللي عداها، الـ GPA).
- رشّح المواد في جدول واضح فيه: اسم المادة، عدد الساعات، السبب، الأولوية.
- في النهاية، اكتب نصائح مخصصة بناءً على وضعه.
- لو الطالب كتب طلب معين (زي "عايز مواد سهلة" أو "عايز أرفع الـ GPA")، خد ده في الاعتبار.
- تكلم بالعربي بأسلوب ودود زي معيد بيساعد طالب.

═══════════════════════════════════════
⚠️ حالات خاصة:
═══════════════════════════════════════
- لو الطالب مش مسجل أي مواد لسه (سنة أولى جديد)، رشحله مواد السنة الأولى الفصل الأول.
- لو الطالب مسجل كل مواده ونجح فيها، قوله تقدر تسجل مواد السنة اللي بعدها.
- لو مفيش مواد متاحة (عدى كل حاجة)، هنّيه إنه قرب يتخرج!
"""


def generate_recommendation(data: dict) -> dict:
    """
    Generate course recommendations based on student data.

    Args:
        data: {
            "profile": { "currentYear", "department", "gpa", ... },
            "passedCourses": [{ "courseCode", "courseNameAr", "grade", ... }],
            "failedCourses": [{ "courseCode", "courseNameAr", ... }],
            "availableCourses": [{ "code", "nameAr", "creditHours", "prerequisites", ... }],
            "studentMessage": "optional message"
        }
    """
    profile = data.get("profile", {})
    passed = data.get("passedCourses", [])
    failed = data.get("failedCourses", [])
    available = data.get("availableCourses", [])
    message = data.get("studentMessage", "")

    # Build structured context
    profile_text = f"""
══ بيانات الطالب ══
• السنة الدراسية: {profile.get('currentYear', '?')}
• القسم: {profile.get('department', 'عام')}
• البرنامج: {profile.get('program', 'عام')}
• الـ GPA: {profile.get('gpa', '?')}
• ساعات معتمدة مكتملة: {profile.get('totalCreditHoursCompleted', 0)}
• مواد نجح فيها: {profile.get('passedCoursesCount', 0)}
• مواد رسب فيها: {profile.get('failedCoursesCount', 0)}
"""

    passed_text = "══ المواد اللي نجح فيها ══\n"
    if passed:
        for c in passed:
            passed_text += f"  ✅ {c['courseNameAr']} ({c['courseCode']}) — التقدير: {c.get('grade', 'N/A')}\n"
    else:
        passed_text += "  (لم يسجل أي مواد بعد)\n"

    failed_text = "══ المواد اللي رسب فيها ══\n"
    if failed:
        for c in failed:
            failed_text += f"  ❌ {c['courseNameAr']} ({c['courseCode']})\n"
    else:
        failed_text += "  (لا يوجد)\n"

    available_text = "══ المواد المتاحة للتسجيل (المتطلبات مستوفاة) ══\n"
    if available:
        for c in available:
            prereqs = ", ".join(c.get("prerequisites", [])) or "لا يوجد"
            available_text += (
                f"  • {c['nameAr']} ({c['code']}) — "
                f"{c['creditHours']} ساعات — "
                f"سنة {c['year']} فصل {c['semester']} — "
                f"متطلبات: {prereqs}\n"
            )
    else:
        available_text += "  (لا توجد مواد متاحة حالياً)\n"

    student_request = ""
    if message:
        student_request = f"\n══ طلب الطالب ══\n{message}\n"

    full_prompt = f"""{ADVISOR_PROMPT}

{profile_text}
{passed_text}
{failed_text}
{available_text}
{student_request}

بناءً على كل المعلومات أعلاه، رشّح أفضل المواد للطالب مع الأسباب والأولويات.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        return {
            "status": "success",
            "response": response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "response": f"حدث خطأ: {str(e)}"
        }


def generate_recommendation_stream(data: dict):
    """Same as above but yields chunks for streaming."""
    profile = data.get("profile", {})
    passed = data.get("passedCourses", [])
    failed = data.get("failedCourses", [])
    available = data.get("availableCourses", [])
    message = data.get("studentMessage", "")

    profile_text = f"""
══ بيانات الطالب ══
• السنة: {profile.get('currentYear', '?')} | القسم: {profile.get('department', 'عام')}
• GPA: {profile.get('gpa', '?')} | ساعات مكتملة: {profile.get('totalCreditHoursCompleted', 0)}
• نجح: {profile.get('passedCoursesCount', 0)} | رسب: {profile.get('failedCoursesCount', 0)}
"""

    passed_list = "\n".join(
        f"  ✅ {c['courseNameAr']} ({c['courseCode']})" for c in passed
    ) if passed else "  (لا يوجد)"

    failed_list = "\n".join(
        f"  ❌ {c['courseNameAr']} ({c['courseCode']})" for c in failed
    ) if failed else "  (لا يوجد)"

    available_list = "\n".join(
        f"  • {c['nameAr']} ({c['code']}) — {c['creditHours']}س — متطلبات: {', '.join(c.get('prerequisites', [])) or 'لا يوجد'}"
        for c in available
    ) if available else "  (لا يوجد)"

    full_prompt = f"""{ADVISOR_PROMPT}

{profile_text}
══ نجح فيها ══
{passed_list}

══ رسب فيها ══
{failed_list}

══ المتاحة للتسجيل ══
{available_list}

{f'══ طلب الطالب ══{chr(10)}{message}' if message else ''}

رشّح أفضل المواد مع الأسباب والأولويات.
"""

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"حدث خطأ: {str(e)}"
