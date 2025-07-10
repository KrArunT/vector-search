#!/usr/bin/env python3
"""
vector_search.py

End-to-end FAISS + MongoDB vector search over JSONL payloads,
comparing embeddings from multiple models and recommending the best one.

Features:
  - If saved FAISS indices and ID-maps exist, load them; otherwise build from JSONL
  - Embed documents and queries with each model
  - Build & save separate FAISS indices for each model
  - (Optional) Upsert each document + vectors into MongoDB
  - Perform a sample k-NN query, display separate rich tables per model
  - Recommend the model with the highest average top-k similarity

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
from rich.table import Table

console = Console()

# 1. CONFIGURATION
JSONL_FILE      = "data.jsonl"
OLLAMA_API      = "http://localhost:11434/api/embeddings"
MODELS          = [
    "nomic-embed-text",
    "snowflake-arctic-embed:22m",
    "granite-embedding:30m",
    "all-minilm:22m",
    "all-minilm:33m",
    "snowflake-arctic-embed:33m",
    "snowflake-arctic-embed",
]
FAISS_INDEX_FNS = {model: f"vector_index_{model.replace(':','_')}.faiss" for model in MODELS}
ID_MAP_FNS      = {model: f"id_map_{model.replace(':','_')}.pkl" for model in MODELS}
USE_MONGO       = True
MONGO_URI       = "mongodb://localhost:27017"
MONGO_DB        = "vector_search_db"
MONGO_COLL      = "docs"

# 2. HELPERS
def get_embedding(text: str, model: str) -> np.ndarray:
    resp = requests.post(
        OLLAMA_API,
        json={"model": model, "prompt": text},
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def normalize(v: np.ndarray):
    faiss.normalize_L2(v)
    return v


def load_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def build_or_load_index(model: str, docs: list[dict]):
    idx_file = FAISS_INDEX_FNS[model]
    id_file  = ID_MAP_FNS[model]
    if os.path.exists(idx_file) and os.path.exists(id_file):
        index = faiss.read_index(idx_file)
        with open(id_file, 'rb') as f:
            ids = pickle.load(f)
        console.log(f"Loaded index for [bold]{model}[/] from disk.")
    else:
        console.log(f"Building index for [bold]{model}[/]...")
        texts = [json.dumps(d, ensure_ascii=False) for d in docs]
        vecs  = np.vstack([get_embedding(t, model) for t in texts])
        normalize(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, idx_file)
        ids = [str(d.get('id', i)) for i, d in enumerate(docs)]
        with open(id_file, 'wb') as f:
            pickle.dump(ids, f)
        console.log(f"Saved index for [bold]{model}[/] to disk.")
    return index, ids


def main():
    docs = load_jsonl(JSONL_FILE)
    if not docs:
        console.print(f"[red]No documents found in {JSONL_FILE}[/]")
        return

    # Build/load indices
    indices, id_maps = {}, {}
    for model in MODELS:
        idx, ids = build_or_load_index(model, docs)
        indices[model] = idx
        id_maps[model] = ids

    # Upsert vectors into MongoDB (optional)
    if USE_MONGO:
        client = MongoClient(MONGO_URI)
        coll = client[MONGO_DB][MONGO_COLL]
        for model in MODELS:
            texts = [json.dumps(d, ensure_ascii=False) for d in docs]
            vecs = normalize(np.vstack([get_embedding(t, model) for t in texts]))
            for i, d in enumerate(docs):
                coll.replace_one(
                    {"_id": str(d.get('id', i))},
                    {**d, f"vector_{model.replace(':','_')}": vecs[i].tolist()},
                    upsert=True
                )
        console.log(f"Upserted documents with vectors into MongoDB.{MONGO_DB}.{MONGO_COLL}")

    # Sample query
    query = {"action": "purchase", "user": "alice"}
    q_text = json.dumps(query)

    # Execute and display separate tables per model
    k = 3
    for model in MODELS:
        console.rule(f"Results for [bold]{model}[/]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", justify="right")
        table.add_column("ID", justify="center")
        table.add_column("Score", justify="right")

        q_emb = normalize(get_embedding(q_text, model).reshape(1, -1))
        D, I = indices[model].search(q_emb, k)

        for rank in range(k):
            idx = I[0][rank]
            score = D[0][rank]
            table.add_row(str(rank+1), id_maps[model][idx], f"{score:.4f}")

        console.print(table)
        avg_score = float(D.mean())
        console.print(f"Average top-{k} score for [bold]{model}[/]: {avg_score:.4f}\n")

    # Recommend best model
    # Compute averages again
    avg_scores = {}
    for model in MODELS:
        q_emb = normalize(get_embedding(q_text, model).reshape(1, -1))
        D, _ = indices[model].search(q_emb, k)
        avg_scores[model] = float(D.mean())
    best = max(avg_scores, key=avg_scores.get)
    console.print(f":star: [bold green]Recommended model:[/] {best} (avg top-{k} score = {avg_scores[best]:.4f})")

if __name__ == "__main__":
    main()
