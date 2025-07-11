#!/usr/bin/env python3
"""
vector_search_mongodb_with_chunking.py

MongoDB + FAISS Vector Search over JSONL payloads + RAG Chatbot

Features:
  - Load JSONL, split each document into overlapping text chunks
  - Clean and preprocess text (normalize whitespace, remove control chars)
  - Embed each chunk via Ollama and upsert into MongoDB with metadata (parent_id, chunk)
  - Build (or load) a FAISS index on all chunk embeddings, with dimension mismatch handling
  - Run a FAISS k-NN query locally or launch an interactive RAG chat
  - For RAG chat: retrieve top-k relevant chunks as context, then invoke Ollama chat model

Requirements:
    pip install pymongo requests rich faiss-cpu numpy
    MongoDB >= 4.x
"""
import os
import re
import json
import pickle
import argparse
import logging
from typing import List, Tuple

import requests
import numpy as np
import faiss
from pymongo import MongoClient
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.progress import track

# ─── LOGGER & CONSOLE ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_embedding(api_url: str, model: str, text: str) -> np.ndarray:
    """Fetch a single embedding from Ollama API."""
    resp = requests.post(api_url, json={"model": model, "prompt": text})
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if emb is None:
        raise ValueError("No embedding returned from API")
    return np.array(emb, dtype=np.float32)


def load_jsonl(path: str) -> List[dict]:
    """Read JSONL file into list of documents."""
    docs: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: List[str] = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + size]
        if not chunk:
            break
        chunks.append(clean_text(" ".join(chunk)))
    return chunks


def upsert_documents(
    coll,
    docs: List[dict],
    api_url: str,
    embed_model: str,
    vector_field: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Embed and upsert each chunk of every document into MongoDB."""
    console.print(f"[blue]Upserting {len(docs)} documents with chunking...[/]")
    for doc in track(docs, description="Processing docs..."):
        parent_id = str(doc.get("id") or hash(json.dumps(doc, ensure_ascii=False)))
        raw = json.dumps(doc, ensure_ascii=False)
        text = clean_text(raw)
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            vec = get_embedding(api_url, embed_model, chunk)
            record = {**doc}
            record[vector_field] = vec.tolist()
            record.update({"parent_id": parent_id, "chunk": chunk})
            doc_id = f"{parent_id}_chunk{idx}"
            coll.replace_one({"_id": doc_id}, record, upsert=True)
    console.print("[green]Upsert complete![/]")


def build_or_load_faiss(
    coll,
    vector_field: str,
    index_path: str,
    idmap_path: str,
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Load existing FAISS index or build a new one from MongoDB embeddings."""
    sample = coll.find_one({vector_field: {"$exists": True}}, {vector_field: 1})
    if sample is None:
        logger.error("No vectors found in MongoDB collection")
        exit(1)
    dim = len(sample[vector_field])

    if os.path.exists(index_path) and os.path.exists(idmap_path):
        console.print("[green]Loading FAISS index from disk...[/]")
        index = faiss.read_index(index_path)
        with open(idmap_path, 'rb') as f:
            id_map = pickle.load(f)
        if index.d != dim:
            console.print(f"[yellow]Dimension mismatch: index.d={index.d} != {dim}, rebuilding...[/]")
            os.remove(index_path)
            os.remove(idmap_path)
        else:
            return index, id_map

    console.print("[green]Building new FAISS index...[/]")
    cursor = coll.find({vector_field: {"$exists": True}}, {vector_field: 1})
    vectors, id_map = [], []
    for doc in cursor:
        vectors.append(doc[vector_field])
        id_map.append(doc['_id'])

    arr = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    faiss.write_index(index, index_path)
    with open(idmap_path, 'wb') as f:
        pickle.dump(id_map, f)

    console.print(f"[green]Index built ({len(id_map)} vectors, dim={dim})[/]")
    return index, id_map


def rag_chat(
    coll,
    index: faiss.IndexFlatIP,
    id_map: List[str],
    api_url: str,
    chat_model: str,
    embed_model: str,
    top_k: int,
    vector_field: str,
) -> None:
    """Launch interactive RAG chat: retrieve context, then call Ollama chat."""
    console.print("[bold green]Starting RAG chat (empty input to quit)[/]\n")
    while True:
        query = input("You: ").strip()
        if not query:
            break
        # retrieve top-k documents
        qv = get_embedding(api_url, embed_model, query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
        dists, idxs = index.search(qv, top_k)
        contexts = []
        for idx in idxs[0]:
            chunk_doc = coll.find_one({"_id": id_map[idx]}) or {}
            contexts.append(chunk_doc.get("chunk", ""))
        context_str = "\n---\n".join(contexts)
        # build chat payload
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer user queries."},
            {"role": "user", "content": f"Context:\n{context_str}\nQuestion: {query}"}
        ]
        resp = requests.post(
            api_url.replace("/api/embeddings", "/api/chat/completions"),
            json={"model": chat_model, "messages": messages}
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        console.print(f"[bold blue]Assistant:[/] {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="MongoDB + FAISS vector search + RAG Chatbot.")
    parser.add_argument("--jsonl", default="data.jsonl", help="Path to JSONL file")
    parser.add_argument("--api-url", default="http://localhost:11434/api/embeddings")
    parser.add_argument("--embed-model", default="qwen3:0.6b", help="Ollama embedding model")
    parser.add_argument("--chat-model", default="qwen3:0.6b", help="Ollama chat model")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="vector_search_db")
    parser.add_argument("--coll", default="docs")
    parser.add_argument("--index", default="faiss.index")
    parser.add_argument("--idmap", default="id_map.pkl")
    parser.add_argument("--vector-field", default="vector", help="Field name for embedding vectors in MongoDB")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--query", default="Nginx Workload", help="Single search query (ignored in chat mode)")
    parser.add_argument("--chat", action="store_true", help="Launch interactive RAG chat instead of one-shot search")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    coll = client[args.db][args.coll]

    docs = load_jsonl(args.jsonl)
    if not docs:
        logger.error(f"No documents found in {args.jsonl}")
        return

    upsert_documents(
        coll,
        docs,
        args.api_url,
        args.embed_model,
        args.vector_field,
        args.chunk_size,
        args.overlap,
    )

    index, id_map = build_or_load_faiss(
        coll,
        args.vector_field,
        args.index,
        args.idmap,
    )

    if args.chat:
        rag_chat(
            coll,
            index,
            id_map,
            args.api_url,
            args.chat_model,
            args.embed_model,
            args.top_k,
            args.vector_field,
        )
    else:
        # one-shot search + display
        console.print(f"[bold]Searching for:[/] [cyan]{args.query}[/]\n")
        from rich import print as rprint
        from rich import pretty
        rprint = console.print
        # reuse search display
        qv = get_embedding(args.api_url, args.embed_model, args.query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
        dists, idxs = index.search(qv, args.top_k)
        for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0]), 1):
            doc_id = id_map[idx]
            doc = coll.find_one({"_id": doc_id}, {'_id':0, args.vector_field:0}) or {}
            title = f"Result {rank} (id={doc_id}, score={dist:.4f})"
            console.print(Panel(JSON.from_data(doc), title=title, expand=False))

if __name__ == "__main__":
    main()
