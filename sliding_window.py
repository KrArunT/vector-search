#!/usr/bin/env python3
"""
vector_search_mongodb_with_chunking.py

MongoDB + FAISS Vector Search over JSONL payloads with Rich panels, colored JSON output,
and a sliding-window chunking strategy for large documents.

Features:
  - Load JSONL, split each document into overlapping text chunks
  - Embed each chunk and upsert into MongoDB with metadata (parent_id, chunk)
  - Build (or load) a FAISS index on all chunk embeddings, with dimension mismatch handling
  - Run a FAISS k-NN query locally
  - Fetch the top-k chunks from MongoDB and display each in its own Rich Panel

Requirements:
    pip install pymongo requests rich faiss-cpu numpy
    MongoDB >= 4.x
"""
import os
import json
import pickle
import requests
import numpy as np
import faiss
from pymongo import MongoClient
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

# ─── CONFIG ────────────────────────────────────────────────────────────────
JSONL_FILE     = "data.jsonl"
OLLAMA_API     = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL   = "snowflake-arctic-embed:22m"
MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "vector_search_db"
MONGO_COLL     = "docs"
VECTOR_FIELD   = "vector"
INDEX_PATH     = "faiss.index"
IDMAP_PATH     = "id_map.pkl"
TOP_K          = 3
CHUNK_SIZE     = 500    # words per chunk
CHUNK_OVERLAP  = 50     # overlap between chunks (words)

console = Console()

def get_embedding(text: str) -> np.ndarray:
    """Fetch a single embedding from Ollama."""
    resp = requests.post(
        OLLAMA_API,
        json={"model": OLLAMA_MODEL, "prompt": text}
    )
    resp.raise_for_status()
    emb = resp.json().get("embedding")
    return np.array(emb, dtype=np.float32)


def load_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    total = len(words)
    while start < total:
        end = min(start + chunk_size, total)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def upsert_documents(coll, docs: list[dict]):
    console.print(f"[blue]Upserting {len(docs)} documents (with chunking) into MongoDB...[/]")
    for d in docs:
        parent_id = str(d.get("id") or json.dumps(d)[:8])
        flat_text = json.dumps(d, ensure_ascii=False)
        chunks = chunk_text(flat_text)
        for i, chunk in enumerate(chunks):
            vec = get_embedding(chunk)
            record = {
                **d,
                VECTOR_FIELD: vec.tolist(),
                "chunk": chunk,
                "parent_id": parent_id
            }
            doc_id = f"{parent_id}_chunk{i}"
            coll.replace_one({"_id": doc_id}, record, upsert=True)
    console.print("[blue]Upsert complete![/]")


def build_or_load_faiss(coll) -> (faiss.Index, list[str]):
    # Determine target embedding dimension from MongoDB sample
    sample = coll.find_one({VECTOR_FIELD: {"$exists": True}}, {VECTOR_FIELD: 1})
    if sample is None:
        console.print("[bold red]No embedded chunks found in MongoDB[/]")
        exit(1)
    target_dim = len(sample[VECTOR_FIELD])

    # Try loading existing index and id_map
    if os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH):
        console.print("[green]Loading FAISS index from disk...[/]")
        index = faiss.read_index(INDEX_PATH)
        with open(IDMAP_PATH, 'rb') as f:
            id_map = pickle.load(f)
        # Check dimension match
        if index.d != target_dim:
            console.print(f"[yellow]Index dimension ({index.d}) != target dim ({target_dim}), rebuilding...[/]")
            os.remove(INDEX_PATH)
            os.remove(IDMAP_PATH)
        else:
            return index, id_map

    # Build new FAISS index
    console.print("[green]Building new FAISS index...[/]")
    cursor = coll.find({VECTOR_FIELD: {"$exists": True}}, {VECTOR_FIELD: 1})
    vectors = []
    id_map = []
    for doc in cursor:
        vectors.append(doc[VECTOR_FIELD])
        id_map.append(doc["_id"])

    np_vecs = np.array(vectors, dtype=np.float32)
    # Normalize for cosine similarity
    faiss.normalize_L2(np_vecs)
    index = faiss.IndexFlatIP(target_dim)
    index.add(np_vecs)

    # Persist index and id_map
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, 'wb') as f:
        pickle.dump(id_map, f)

    console.print(f"[green]FAISS index built ({len(id_map)} chunks, dim={target_dim})[/]")
    return index, id_map


def search_and_display(coll, index: faiss.Index, id_map: list[str], query: str):
    console.print(f"\n[bold]Searching for:[/] [cyan]{query}[/]\n")
    q_vec = get_embedding(query).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    D, I = index.search(q_vec, TOP_K)
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        doc_id = id_map[idx]
        # Fetch full doc and remove raw vector field
        doc = coll.find_one({"_id": doc_id}) or {}
        doc.pop(VECTOR_FIELD, None)
        panel_title = (
            f"Result {rank} (chunk_id={doc_id}, parent_id={doc.get('parent_id')}, "
            f"score={dist:.4f})"
        )
        console.print(Panel(JSON.from_data(doc), title=panel_title, expand=False))


if __name__ == "__main__":
    # 1. Connect to MongoDB
    client = MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_COLL]

    # 2. Load docs & upsert chunked vectors
    docs = load_jsonl(JSONL_FILE)
    if not docs:
        console.print(f"[bold red]No documents found in {JSONL_FILE}[/]")
        exit(1)
    upsert_documents(coll, docs)

    # 3. Build or load FAISS index + id_map
    index, id_map = build_or_load_faiss(coll)

    # 4. Search & display
    query_text = "Nginx Workload"
    search_and_display(coll, index, id_map, query_text)
