#!/usr/bin/env python3
"""
vector_search_rich.py

Enhanced FAISS + MongoDB vector search over JSONL payloads
with Rich panels for displaying documents and query results vertically.

Features:
  - Load or build FAISS index & ID map
  - Optional MongoDB upsert
  - Display each document in its own Rich Panel
  - Display k-NN query results in Rich Panels

Requirements:
    pip install faiss-cpu numpy requests pymongo rich
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

# 1. CONFIGURATION
JSONL_FILE     = "data.jsonl"        # Path to your JSON Lines file
OLLAMA_API     = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL   = "nomic-embed-text"
FAISS_INDEX_FN = "vector_index.faiss"
ID_MAP_FN      = "id_map.pkl"
USE_MONGO      = True                 # Set False to skip MongoDB steps
MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "vector_search_db"
MONGO_COLL     = "docs"

console = Console()

# 2. HELPERS
def get_embedding(text: str) -> np.ndarray:
    resp = requests.post(
        OLLAMA_API,
        json={"model": OLLAMA_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    return np.array(resp.json().get("embedding"), dtype=np.float32)


def normalize(vects: np.ndarray):
    faiss.normalize_L2(vects)


def load_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def display_docs_vertical(docs: list[dict], header: str):
    if not docs:
        console.print(f"[bold red]No documents to display for {header}[/]")
        return
    console.print(f"[bold underline]{header}[/]\n")
    for idx, doc in enumerate(docs, start=1):
        lines = []
        for key, value in doc.items():
            lines.append(f"[b]{key}[/]: {value}")
        body = "\n".join(lines)
        console.print(Panel(body, title=f"Document {idx}"))


def main():
    if USE_MONGO:
        mongo = MongoClient(MONGO_URI)
        coll  = mongo[MONGO_DB][MONGO_COLL]

    # Load or build index & IDs
    if os.path.exists(FAISS_INDEX_FN) and os.path.exists(ID_MAP_FN):
        console.print("[green]Loading existing FAISS index and ID map...[/]")
        index = faiss.read_index(FAISS_INDEX_FN)
        with open(ID_MAP_FN, 'rb') as f:
            ids = pickle.load(f)
    else:
        console.print("[yellow]Building new index from JSONL file...[/]")
        docs = load_jsonl(JSONL_FILE)
        if not docs:
            console.print(f"[bold red]No documents found in {JSONL_FILE}[/]")
            return

        display_docs_vertical(docs, header="Loaded Documents")

        texts = [json.dumps(d, ensure_ascii=False) for d in docs]
        vecs  = np.vstack([get_embedding(t) for t in texts])
        ids   = [str(d.get('id', idx)) for idx, d in enumerate(docs)]

        normalize(vecs)
        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        faiss.write_index(index, FAISS_INDEX_FN)
        with open(ID_MAP_FN, 'wb') as f:
            pickle.dump(ids, f)
        console.print(f"[green]Saved new FAISS index and ID map.[/]")

        if USE_MONGO:
            console.print("[blue]Upserting docs into MongoDB...[/]")
            for d, vec in zip(docs, vecs):
                coll.replace_one({"_id": str(d.get('id', None))}, {**d, "vector": vec.tolist()}, upsert=True)
            console.print(f"[blue]Upserted {len(docs)} documents.[/]")

    # Prepare query
    query = "Nginx Workload"
    q_emb  = get_embedding(json.dumps(query)).reshape(1, -1).astype(np.float32)
    normalize(q_emb)

    # k-NN search
    k = 3
    console.print(f"\n[bold]Top {k} results for query:[/] [cyan]{query}[/]\n")
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        doc_id = ids[idx]
        entry = {"rank": rank, "id": doc_id, "score": f"{score:.4f}"}
        if USE_MONGO:
            doc = coll.find_one({"_id": doc_id}, {"vector": 0})
            entry.update({k: str(v) for k, v in doc.items()})
        results.append(entry)

    display_docs_vertical(results, header="Search Results")

if __name__ == "__main__":
    main()
