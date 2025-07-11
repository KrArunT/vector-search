#!/usr/bin/env python3
"""
vector_search_rich.py

Enhanced FAISS + MongoDB vector search over JSONL payloads
with JSON output for displaying documents and query results.

Features:
  - Load or build FAISS index & ID map
  - Optional MongoDB upsert
  - Display documents and search results as pretty-printed JSON

Requirements:
    pip install faiss-cpu numpy requests pymongo
"""

import os
import json
import pickle
import requests
import numpy as np
import faiss
from pymongo import MongoClient

# 1. CONFIGURATION
JSONL_FILE     = "data.jsonl"
OLLAMA_API     = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL   = "nomic-embed-text"
FAISS_INDEX_FN = "vector_index.faiss"
ID_MAP_FN      = "id_map.pkl"
USE_MONGO      = True
MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "vector_search_db"
MONGO_COLL     = "docs"


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


def print_json(data):
    """Pretty-print JSON data to stdout."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    if USE_MONGO:
        mongo = MongoClient(MONGO_URI)
        coll  = mongo[MONGO_DB][MONGO_COLL]

    if os.path.exists(FAISS_INDEX_FN) and os.path.exists(ID_MAP_FN):
        print("Loading existing FAISS index and ID map...")
        index = faiss.read_index(FAISS_INDEX_FN)
        with open(ID_MAP_FN, 'rb') as f:
            ids = pickle.load(f)
    else:
        print("Building new FAISS index from JSONL file...")
        docs = load_jsonl(JSONL_FILE)
        if not docs:
            raise ValueError(f"No documents found in {JSONL_FILE}")

        print_json(docs)
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
        print(f"Saved FAISS index and ID map.")

        if USE_MONGO:
            print("Upserting documents into MongoDB...")
            for d, vec in zip(docs, vecs):
                coll.replace_one(
                    {"_id": str(d.get('id', None))},
                    {**d, "vector": vec.tolist()},
                    upsert=True
                )
            print(f"Upserted {len(docs)} documents.")

    # Prepare and embed sample query
    query = "Nginx Workload"
    q_emb = get_embedding(json.dumps(query)).reshape(1, -1).astype(np.float32)
    normalize(q_emb)

    # k-NN search
    k = 3
    print(f"\nQuerying FAISS for top {k} results for: {query}\n")
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        # Ensure score is JSON serializable
        score_val = float(score)
        doc_id = ids[idx]
        entry = {"rank": rank, "id": doc_id, "score": score_val}
        if USE_MONGO:
            doc = coll.find_one({"_id": doc_id}, {"vector": 0}) or {}
            # All doc values should already be JSON serializable
            entry.update(doc)
        results.append(entry)

    print_json(results)

if __name__ == "__main__":
    main()
