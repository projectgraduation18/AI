"""
RAG Retriever
- Loads FAISS index per course_id (context isolation)
- Cosine similarity (normalized embeddings + IndexFlatIP): الأعلى = الأقرب
- Optional cross-encoder reranking (bge-reranker-v2-m3)
- Helper to pull full material text for summarize/quiz
"""

import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    DB_DIR, EMBEDDING_MODEL,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    USE_RERANKER, RERANKER_MODEL,
)

# ── Lazy reranker (يتحمّل بس لو USE_RERANKER=True) ──
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANKER and RERANKER_MODEL:
        from FlagEmbedding import FlagReranker
        print("🔄 Loading reranker...")
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
        print("✅ Reranker loaded!")
    return _reranker


class StudyRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._index_cache: dict[str, faiss.Index] = {}
        self._metadata_cache: dict[str, list[dict]] = {}

    def _load_course_index(self, course_id: str):
        """Load FAISS index and metadata for a course (cached)."""
        if course_id in self._index_cache:
            return

        course_db_dir = os.path.join(DB_DIR, course_id)
        index_path = os.path.join(course_db_dir, "index.faiss")
        metadata_path = os.path.join(course_db_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Database for course '{course_id}' not found. "
                f"Run: python ingest.py {course_id}"
            )

        self._index_cache[course_id] = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata_cache[course_id] = json.load(f)

    def reload_course(self, course_id: str):
        """Force reload after new materials are ingested."""
        self._index_cache.pop(course_id, None)
        self._metadata_cache.pop(course_id, None)
        self._load_course_index(course_id)

    def retrieve(self, query: str, course_id: str = "default",
                 k: int = RETRIEVAL_TOP_K,
                 top_n: int = RERANK_TOP_N) -> list[dict]:
        """
        1. Encode query (normalized)
        2. Cosine search in FAISS (IP) — الأعلى أحسن
        3. Dedup near-duplicates
        4. Optional rerank
        5. Return top-n
        """
        self._load_course_index(course_id)
        index = self._index_cache[course_id]
        metadata = self._metadata_cache[course_id]

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        actual_k = min(k, index.ntotal)
        scores, indices = index.search(query_embedding, actual_k)  # IP: higher = better

        results = []
        seen_content = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = metadata[idx]
            content = chunk["content"]
            content_key = content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            results.append({
                "content": content,
                "metadata": chunk.get("metadata", {}),
                "score": float(score),
            })

        # ── Optional cross-encoder rerank ──
        reranker = get_reranker()
        if reranker and results:
            pairs = [[query, r["content"]] for r in results]
            rscores = reranker.compute_score(pairs, normalize=True)
            if not isinstance(rscores, list):
                rscores = [rscores]
            for r, s in zip(results, rscores):
                r["score"] = float(s)

        # cosine / rerank: الأعلى أحسن
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    def get_material_text(self, course_id: str, source: str | None = None,
                          max_chars: int = 20000) -> str:
        """
        يجمّع نص مادة (أو ملف واحد جواها) من الـ metadata بالترتيب.
        بيستخدمه التلخيص وتوليد الأسئلة من غير ما يحمّل الـ PDF تاني.
        """
        self._load_course_index(course_id)
        meta = self._metadata_cache[course_id]

        chunks = [
            c for c in meta
            if source is None or c.get("metadata", {}).get("source") == source
        ]
        chunks.sort(key=lambda c: (
            c.get("metadata", {}).get("source", ""),
            c.get("metadata", {}).get("chunk_index", 0),
        ))

        out, total, seen = [], 0, set()
        for c in chunks:
            content = c["content"]
            key = content[:80]
            if key in seen:   # تجنّب تكرار الأوفرلاب
                continue
            seen.add(key)
            if total + len(content) > max_chars:
                break
            out.append(content)
            total += len(content)
        return "\n".join(out)

    def get_course_sources(self, course_id: str) -> list[str]:
        """أسماء الملفات المتفهرسة جوه كورس."""
        self._load_course_index(course_id)
        meta = self._metadata_cache[course_id]
        return sorted({c.get("metadata", {}).get("source", "") for c in meta if c.get("metadata")})

    def get_available_courses(self) -> list[str]:
        """List all indexed courses."""
        if not os.path.exists(DB_DIR):
            return []
        return [
            d for d in os.listdir(DB_DIR)
            if os.path.isdir(os.path.join(DB_DIR, d))
            and os.path.exists(os.path.join(DB_DIR, d, "index.faiss"))
        ]
