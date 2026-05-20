"""
Document Ingestion Pipeline
- Downloads PDFs from URLs (sent by .NET backend)
- Extracts text from PDFs
- Splits into overlapping chunks (not just raw pages)
- Generates embeddings and saves to FAISS index
- Supports per-course indexing (context isolation)
"""

import os
import re
import fitz  # PyMuPDF
import faiss
import json
import hashlib
import tempfile
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from config import (
    DATA_DIR, DB_DIR, EMBEDDING_MODEL,
    CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
)


# ══════════════════════════════════════════
# Step 1: Download PDF from URL
# ══════════════════════════════════════════

def download_pdf(file_url: str, save_dir: str, file_name: str) -> str:
    """
    Download a PDF from a public URL (Supabase Storage).
    Returns the local file path.
    """
    # ====== السطور اللي هنضيفها ======
    if file_url.startswith("file://"):
        print(f"  ✅ Using local file: {file_name}")
        return file_url[7:] # عشان نشيل كلمة file:// وناخد المسار العادي
    # =================================

    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, file_name)

    print(f"  ⬇️  Downloading: {file_name}...")
    response = requests.get(file_url, timeout=120)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"  ✅ Downloaded: {len(response.content) / 1024:.1f} KB")
    return local_path

# ══════════════════════════════════════════
# Step 2: Extract text from PDF
# ══════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc.load_page(page_num).get_text("text")
            clean = re.sub(r'\s+', ' ', text).strip()
            if len(clean) > 30:  # Skip near-empty pages
                pages.append({
                    "text": clean,
                    "page": page_num + 1
                })
        doc.close()
    except Exception as e:
        print(f"  ❌ Error reading {pdf_path}: {e}")
    return pages


# ══════════════════════════════════════════
# Step 3: Split into overlapping chunks
# ══════════════════════════════════════════

def split_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS,
                      overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split long text into overlapping chunks.
    WHY overlap? So we don't lose context at page boundaries.
    WHY sentence boundary? So we don't cut in the middle of a word.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        # Try to break at sentence boundary
        if end < len(text):
            break_point = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('\n', start, end),
                text.rfind('، ', start, end),  # Arabic comma
                text.rfind('. ', start, end),   # Arabic period
            )
            if break_point > start + (max_chars // 2):
                end = break_point + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 30]


# ══════════════════════════════════════════
# Step 4: Compute hash for deduplication
# ══════════════════════════════════════════

def compute_file_hash(filepath: str) -> str:
    """SHA-256 hash to prevent re-processing the same file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """SHA-256 hash from raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ══════════════════════════════════════════
# Step 5: Generate embeddings and save
# ══════════════════════════════════════════

# Cache the model globally so we don't reload it on every request
_model = None

def get_model():
    global _model
    if _model is None:
        print("🔄 Loading embedding model (first time only)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model loaded!")
    return _model


def save_to_faiss(chunks: list[dict], course_id: str):
    """
    Generate embeddings for chunks and save to FAISS index.
    If an index already exists for this course, append to it.
    """
    course_db_dir = os.path.join(DB_DIR, course_id)
    os.makedirs(course_db_dir, exist_ok=True)

    index_path = os.path.join(course_db_dir, "index.faiss")
    metadata_path = os.path.join(course_db_dir, "metadata.json")

    # Load existing data if any
    existing_chunks = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    # Combine old + new
    all_chunks = existing_chunks + chunks
    print(f"  📊 Total chunks: {len(existing_chunks)} existing + {len(chunks)} new = {len(all_chunks)}")

    # Generate embeddings for ALL chunks (FAISS needs full rebuild)
    model = get_model()
    all_texts = [c["content"] for c in all_chunks]
    print(f"  🚀 Encoding {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64)

    # Save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_path)

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Saved! Index has {index.ntotal} vectors")


# ══════════════════════════════════════════
# Main Pipeline: Called by the API endpoint
# ══════════════════════════════════════════

def ingest_from_url(course_id: str, file_url: str, file_name: str) -> dict:
    """
    Full pipeline: Download → Extract → Chunk → Embed → Save.
    This is what the /api/ingest endpoint calls.

    Args:
        course_id: Which course this material belongs to
        file_url: Public URL of the PDF (from Supabase Storage)
        file_name: Original filename

    Returns:
        dict with status and chunk count
    """
    print(f"\n{'='*50}")
    print(f"📚 Ingesting: {file_name} for course {course_id}")
    print(f"{'='*50}")

    course_data_dir = os.path.join(DATA_DIR, course_id)

    # Step 1: Download
    local_path = download_pdf(file_url, course_data_dir, file_name)

    # Step 2: Check for duplicate
    file_hash = compute_file_hash(local_path)
    hashes_path = os.path.join(DB_DIR, course_id, "file_hashes.json")
    existing_hashes = {}
    if os.path.exists(hashes_path):
        with open(hashes_path, "r") as f:
            existing_hashes = json.load(f)

    if file_hash in existing_hashes:
        print(f"  ⚡ Skipping: file already indexed ({existing_hashes[file_hash]})")
        return {"status": "skipped", "reason": "duplicate", "chunks": 0}

    # Step 3: Extract text
    pages = extract_text_from_pdf(local_path)
    if not pages:
        return {"status": "error", "reason": "no text extracted", "chunks": 0}

    print(f"  📄 Extracted {len(pages)} pages with text")

    # Step 4: Split into chunks
    new_chunks = []
    for page_data in pages:
        page_chunks = split_into_chunks(page_data["text"])
        for i, chunk_text in enumerate(page_chunks):
            new_chunks.append({
                "id": f"{file_name}_p{page_data['page']}_c{i}",
                "content": chunk_text,
                "metadata": {
                    "source": file_name,
                    "page": page_data["page"],
                    "chunk_index": i,
                    "course_id": course_id
                }
            })

    print(f"  ✂️  Created {len(new_chunks)} chunks")

    # Step 5: Embed and save
    save_to_faiss(new_chunks, course_id)

    # Step 6: Record hash
    os.makedirs(os.path.dirname(hashes_path), exist_ok=True)
    existing_hashes[file_hash] = file_name
    with open(hashes_path, "w") as f:
        json.dump(existing_hashes, f, indent=2)

    print(f"  🎉 Done! {file_name} fully ingested.\n")

    return {"status": "success", "chunks": len(new_chunks)}


# ══════════════════════════════════════════
# Manual: Build from local folder
# ══════════════════════════════════════════

def build_from_local(course_id: str = "default"):
    """
    Process all PDFs in data/{course_id}/ folder.
    Use this for initial bulk ingestion or testing.

    Usage: python ingest.py CS101
    """
    course_data_dir = os.path.join(DATA_DIR, course_id)
    if not os.path.exists(course_data_dir):
        os.makedirs(course_data_dir, exist_ok=True)
        print(f"📁 Created {course_data_dir} — put PDFs there and run again.")
        return

    pdf_files = [f for f in os.listdir(course_data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs in {course_data_dir}")
        return

    print(f"📚 Found {len(pdf_files)} PDFs for course: {course_id}\n")

    for filename in pdf_files:
        filepath = os.path.join(course_data_dir, filename)
        # Use file:// URL to reuse the same pipeline
        ingest_from_url(
            course_id=course_id,
            file_url=f"file://{os.path.abspath(filepath)}",
            file_name=filename
        )


if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "default"
    build_from_local(course_id=cid)
