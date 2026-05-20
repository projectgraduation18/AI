"""
RAG Retriever
- Loads FAISS index per course_id (context isolation)
- Filters by similarity threshold (drops irrelevant chunks)
- Returns top-N most relevant chunks with scores
"""

import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    DB_DIR, EMBEDDING_MODEL,
    RETRIEVAL_TOP_K, RERANK_TOP_N, SIMILARITY_THRESHOLD
)


class StudyRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # Cache loaded indices to avoid reloading on every request
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
        """Force reload a course index (after new materials are ingested)."""
        self._index_cache.pop(course_id, None)
        self._metadata_cache.pop(course_id, None)
        self._load_course_index(course_id)

    def retrieve(self, query: str, course_id: str = "default",
                 k: int = RETRIEVAL_TOP_K,
                 top_n: int = RERANK_TOP_N) -> list[dict]:
        """
        Retrieve relevant chunks for a query within a specific course.

        Steps:
        1. Encode query
        2. Search FAISS for top-k candidates
        3. Filter by similarity threshold
        4. Return top-n most relevant chunks

        Returns list of dicts with: content, metadata, score
        """
        self._load_course_index(course_id)

        index = self._index_cache[course_id]
        metadata = self._metadata_cache[course_id]

        # Encode query
        query_embedding = self.model.encode([query]).astype("float32")

        # Search - get more than we need, then filter
        actual_k = min(k, index.ntotal)
        distances, indices = index.search(query_embedding, actual_k)

        results = []
        seen_content = set()  # Deduplicate similar chunks

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            # Filter by similarity threshold (L2 distance: lower = better)
           # if dist > SIMILARITY_THRESHOLD:
            #    continue

            chunk = metadata[idx]
            content = chunk["content"]

            # Skip near-duplicate content
            content_key = content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            results.append({
                "content": content,
                "metadata": chunk.get("metadata", {}),
                "score": float(dist)
            })

        # Sort by score (lower L2 distance = more relevant)
        results.sort(key=lambda x: x["score"])

        return results[:top_n]

    def get_available_courses(self) -> list[str]:
        """List all courses that have been indexed."""
        if not os.path.exists(DB_DIR):
            return []
        return [
            d for d in os.listdir(DB_DIR)
            if os.path.isdir(os.path.join(DB_DIR, d))
            and os.path.exists(os.path.join(DB_DIR, d, "index.faiss"))
        ]
