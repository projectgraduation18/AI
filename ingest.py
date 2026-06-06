"""
Document Ingestion Pipeline
- Downloads PDFs from URLs (sent by .NET backend)
- Extracts text as MARKDOWN via pymupdf4llm → بيحافظ على الجداول | a | b | (أهم تعديل)
- Splits into overlapping chunks (RecursiveCharacterTextSplitter)
- Generates NORMALIZED embeddings and saves to FAISS (IndexFlatIP = cosine)
- Per-course indexing (context isolation)
"""

import os
import re
import faiss
import json
import hashlib
import numpy as np
import requests
import pymupdf4llm  # markdown extraction — بيحافظ على الجداول
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    DATA_DIR, DB_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# ══════════════════════════════════════════
# Step 1: Download PDF from URL
# ══════════════════════════════════════════
def download_pdf(file_url: str, save_dir: str, file_name: str) -> str:
    """Download a PDF from a public URL (Supabase Storage). Returns local path."""
    if file_url.startswith("file://"):
        print(f"  ✅ Using local file: {file_name}")
        return file_url[7:]  # نشيل file:// وناخد المسار العادي

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
# Step 2: Extract text from PDF (Markdown — keeps tables)
# ══════════════════════════════════════════
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract each page as Markdown via pymupdf4llm.
    الجداول بتطلع بشكل | عمود | عمود | بدل ما تتهرس — ده اللي بيخلّي اللائحة تتلقط صح.
    """
    pages = []
    try:
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        for i, p in enumerate(md_pages):
            text = (p.get("text") or "").strip()
            # ⚠️ مفيش re.sub(r'\s+', ' ') خالص — ده اللي كان بيبوّظ الجداول
            text = re.sub(r'\n{3,}', '\n\n', text)  # نظّف الأسطر الفاضية الزيادة بس
            if len(text) > 30:  # Skip near-empty pages
                pages.append({"text": text, "page": i + 1})
    except Exception as e:
        print(f"  ❌ Error reading {pdf_path}: {e}")
    return pages

# ══════════════════════════════════════════
# Step 3: Compute hash for deduplication
# ══════════════════════════════════════════
def compute_file_hash(filepath: str) -> str:
    """SHA-256 hash to prevent re-processing the same file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()

def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ══════════════════════════════════════════
# Step 4: Generate embeddings and save (cosine)
# ══════════════════════════════════════════
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
    Generate NORMALIZED embeddings and save to FAISS IndexFlatIP (= cosine).
    لو فيه index قديم للكورس، بيبنيه من جديد بكل الـ chunks (قديم + جديد).
    """
    course_db_dir = os.path.join(DB_DIR, course_id)
    os.makedirs(course_db_dir, exist_ok=True)
    index_path = os.path.join(course_db_dir, "index.faiss")
    metadata_path = os.path.join(course_db_dir, "metadata.json")

    existing_chunks = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    all_chunks = existing_chunks + chunks
    print(f"  📊 Total chunks: {len(existing_chunks)} existing + {len(chunks)} new = {len(all_chunks)}")

    model = get_model()
    all_texts = [c["content"] for c in all_chunks]
    print(f"  🚀 Encoding {len(all_texts)} chunks...")
    embeddings = model.encode(
        all_texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # ← مهم: عشان IndexFlatIP يبقى cosine
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # ← IP بدل L2: الأعلى = الأقرب
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved! Index has {index.ntotal} vectors (dim={dimension})")

# ══════════════════════════════════════════
# Main Pipeline: Called by the API endpoint
# ══════════════════════════════════════════
def ingest_from_url(course_id: str, file_url: str, file_name: str) -> dict:
    """Full pipeline: Download → Extract → Chunk → Embed → Save."""
    print(f"\n{'='*50}")
    print(f"📚 Ingesting: {file_name} for course {course_id}")
    print(f"{'='*50}")

    course_data_dir = os.path.join(DATA_DIR, course_id)

    # Step 1: Download
    local_path = download_pdf(file_url, course_data_dir, file_name)

    # Step 2: Duplicate check
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

    # Step 4: Split into chunks (مع زرع وسم رقم الصفحة لتتبّع المصدر)
    full_text = ""
    for page_data in pages:
        full_text += f" [Page_{page_data['page']}] " + page_data["text"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "، ", ". ", " ", ""],
    )

    final_chunks = text_splitter.split_text(full_text)
    new_chunks = []
    for i, chunk_text in enumerate(final_chunks):
        page_match = re.findall(r'\[Page_(\d+)\]', chunk_text)
        detected_page = int(page_match[-1]) if page_match else pages[0]["page"]
        clean_chunk_text = re.sub(r'\[Page_\d+\]', '', chunk_text).strip()

        new_chunks.append({
            "id": f"{file_name}_c{i}",
            "content": clean_chunk_text,
            "metadata": {
                "source": file_name,
                "page": detected_page,
                "chunk_index": i,
                "course_id": course_id,
            },
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
    """Process all PDFs in data/{course_id}/. Usage: python ingest.py CS101"""
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
        ingest_from_url(
            course_id=course_id,
            file_url=f"file://{os.path.abspath(filepath)}",
            file_name=filename,
        )

if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "default"
    build_from_local(course_id=cid)
