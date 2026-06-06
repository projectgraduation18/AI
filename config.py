import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "database")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ── Models ──
# BGE-M3 = أفضل موديل embeddings للعربي في RAG حالياً (cosine + reranking).
# ⚠️ تقيل (~2GB أول تشغيل). لو الرامات قليلة (HF/Render فري) بدّلها بـ:
#     "intfloat/multilingual-e5-base"  ← أخف بكتير، بس محتاج بادئات query:/passage:
EMBEDDING_MODEL = "BAAI/bge-m3"
GEMINI_MODEL = "gemini-2.5-flash"

# reranker اختياري — بيحسّن ترتيب النتايج جداً للوائح، بس بيزوّد الـ latency.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = False   # خليها True بعد ما تثبّت FlagEmbedding (شوف requirements.txt)

# ── API Keys ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── RAG Settings ──
RETRIEVAL_TOP_K = 30        # كام chunk نجيب من FAISS قبل الفلترة/الـ rerank
RERANK_TOP_N = 8            # كام نسيب في الآخر يتبعت للموديل
SIMILARITY_THRESHOLD = 0.3  # أقل cosine مقبول (0..1) — الأعلى أحسن (مش مفعّل افتراضياً)
CHUNK_SIZE = 1200           # حجم الـ chunk بالحروف
CHUNK_OVERLAP = 300         # التداخل بين الـ chunks لربط نهايات الصفحات

# ── Chat Settings ──
MAX_HISTORY_MESSAGES = 10   # آخر N رسالة تتبعت كسياق
MAX_CONTEXT_CHARS = 12000   # أقصى حروف سياق تتبعت للموديل

# ── Summarize / Quiz ──
MAX_MATERIAL_CHARS = 20000  # أقصى نص من المادة نبعته للتلخيص/توليد الأسئلة
