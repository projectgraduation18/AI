"""
اختبار الـ endpoints بعد تشغيل السيرفر.
شغّل السيرفر الأول:  uvicorn api:app --port 8000
بعدين:              python test_api.py
"""

import requests
import json

BASE = "http://127.0.0.1:8000"
COURSE = "regulations"   # نفس اسم الكورس (فولدر data/regulations)


def show(title, resp):
    print("=" * 60)
    print(title)
    print("-" * 60)
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:2000])
    except Exception:
        print(resp.text[:2000])
    print()


# 1) Health
show("HEALTH", requests.get(f"{BASE}/health"))

# 2) Courses
show("COURSES", requests.get(f"{BASE}/api/courses"))

# 3) Chat (مع history)
chat_body = {
    "message": "ما هي شروط التخرج من الكلية؟",
    "course_id": COURSE,
    "history": [
        {"role": "user", "content": "انا طالب سنة تالتة"},
        {"role": "assistant", "content": "تمام، اتفضل اسأل."},
    ],
}
show("CHAT", requests.post(f"{BASE}/api/chat", json=chat_body))

# 4) Summarize (الكورس كله — أو حط source باسم ملف معيّن)
show("SUMMARIZE", requests.post(
    f"{BASE}/api/summarize",
    json={"course_id": COURSE, "source": None},
))

# 5) Quiz — 3 مستويات
quiz_resp = requests.post(
    f"{BASE}/api/quiz",
    json={"course_id": COURSE, "questions_per_level": 2},
)
show("QUIZ", quiz_resp)

# 6) Quiz "عايز زياده" — بنبعت الأسئلة اللي فاتت عشان ميكررش
prev = [q.get("question", "") for q in quiz_resp.json().get("questions", [])]
show("QUIZ MORE", requests.post(
    f"{BASE}/api/quiz",
    json={"course_id": COURSE, "questions_per_level": 2, "previous_questions": prev},
))

print("✅ خلصت. لو الـ CHAT رجّع إجابة مفيدة و الـ QUIZ رجّع JSON بأسئلة، يبقى تمام.")
