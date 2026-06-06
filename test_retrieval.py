"""
اختبار سريع للـ retrieval على اللائحة — من غير ما تشغّل سيرفر.
بيوريك الـ chunks اللي اترجّعت لكل سؤال + الـ scores، عشان تتأكد إن
الاستخراج (الجداول) والبحث شغالين قبل ما تجرّب الموديل نفسه.

Usage:  python test_retrieval.py layha
"""

import sys
from retriever import StudyRetriever

course_id = sys.argv[1] if len(sys.argv) > 1 else "layha"

# غيّر الأسئلة دي بأسئلة فعلية من اللائحة بتاعتك
questions = [
    "ما هي شروط التخرج من الكلية؟",
    "كام ساعة معتمدة لازمة للتخرج؟",
    "ايه شروط الإنذار الأكاديمي؟",
    "ما هي المتطلبات السابقة لمادة قواعد البيانات؟",
]

r = StudyRetriever()
print(f"📚 الكورسات المتاحة: {r.get_available_courses()}")
try:
    print(f"📄 ملفات '{course_id}': {r.get_course_sources(course_id)}\n")
except FileNotFoundError:
    print(f"\n❌ الكورس '{course_id}' مش متفهرس. شغّل الأول: python ingest.py {course_id}")
    sys.exit(1)

for q in questions:
    print("=" * 60)
    print(f"❓ {q}")
    results = r.retrieve(q, course_id=course_id)
    if not results:
        print("  ⚠️ مفيش نتايج")
        continue
    for i, res in enumerate(results[:3], 1):
        src = res["metadata"].get("source", "?")
        pg = res["metadata"].get("page", "?")
        print(f"\n  [{i}] score={res['score']:.3f} | {src} p.{pg}")
        snippet = res["content"][:300].replace("\n", " ")
        print(f"      {snippet}...")
    print()
