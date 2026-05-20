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
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ── API Keys ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── RAG Settings ──
RETRIEVAL_TOP_K = 20        # How many chunks to retrieve from FAISS
RERANK_TOP_N = 8            # How many to keep after relevance filtering
SIMILARITY_THRESHOLD = 1.5  # Max L2 distance (lower = more similar)
CHUNK_MAX_CHARS = 1500      # Max characters per chunk (for splitting long pages)
CHUNK_OVERLAP_CHARS = 200   # Overlap between chunks for context continuity

# ── Chat Settings ──
MAX_HISTORY_MESSAGES = 10   # Last N messages to include as context (5 exchanges)
MAX_CONTEXT_CHARS = 12000   # Max total characters sent as context to LLM
