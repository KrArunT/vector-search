#!/usr/bin/env python3
"""
vector_search.py

End-to-end FAISS + MongoDB vector search over JSONL payloads.

Features:
  - If a saved FAISS index and ID-map exist, load them instead of rebuilding
  - Otherwise, read JSON Lines (`.jsonl`) file, embed each record, build & save the index
  - (Optional) Upsert each document + vector into MongoDB
  - Demonstrate a sample k-NN query

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
JSONL_FILE     = "data.jsonl"        # Path to your JSON Lines file
OLLAMA_API     = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL   = "nomic-embed-text"
FAISS_INDEX_FN = "vector_index.faiss"
ID_MAP_FN      = "id_map.pkl"
USE_MONGO      = True                # Set False to skip MongoDB steps
MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "vector_search_db"
MONGO_COLL     = "docs"

# 2. HELPERS
def get_embedding(text: str) -> np.ndarray:
    """Call Ollama to get a float32 embedding for the given text."""
    resp = requests.post(
        OLLAMA_API,
        json={"model": OLLAMA_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    return np.array(resp.json().get("embedding"), dtype=np.float32)


def normalize(vects: np.ndarray):
    """In-place L2 normalization for cosine similarity via inner-product."""
    faiss.normalize_L2(vects)


def load_jsonl(path: str) -> list[dict]:
    """Read a JSON Lines file and return a list of dicts."""
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def main():
    # --- Optionally connect to MongoDB ---
    if USE_MONGO:
        mongo = MongoClient(MONGO_URI)
        coll  = mongo[MONGO_DB][MONGO_COLL]

    # --- Load or build FAISS index & ID map ---
    if os.path.exists(FAISS_INDEX_FN) and os.path.exists(ID_MAP_FN):
        print("Loading existing FAISS index and ID map...")
        index = faiss.read_index(FAISS_INDEX_FN)
        with open(ID_MAP_FN, 'rb') as f:
            ids = pickle.load(f)
    else:
        print("No existing index found; building a new one from JSONL file...")
        # Read documents from JSONL
        docs = load_jsonl(JSONL_FILE)
        if not docs:
            raise ValueError(f"No documents found in {JSONL_FILE}")

        # Serialize and embed
        texts = [json.dumps(d, ensure_ascii=False) for d in docs]
        vecs  = np.vstack([get_embedding(t) for t in texts])
        ids   = [str(d.get('id', idx)) for idx, d in enumerate(docs)]

        # Normalize and build FAISS index
        normalize(vecs)
        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        # Persist index and ID map
        faiss.write_index(index, FAISS_INDEX_FN)
        with open(ID_MAP_FN, 'wb') as f:
            pickle.dump(ids, f)
        print(f"Saved new FAISS index to '{FAISS_INDEX_FN}' and ID map to '{ID_MAP_FN}'.")

        # Upsert into MongoDB if enabled
        if USE_MONGO:
            print("Upserting documents into MongoDB...")
            for d, vec in zip(docs, vecs):
                coll.replace_one(
                    {"_id": str(d.get('id', None))},
                    {**d, "vector": vec.tolist()},
                    upsert=True
                )
            print(f"Upserted {len(docs)} docs into {MONGO_DB}.{MONGO_COLL}")

    # --- Prepare & embed a sample query JSON ---
    # query = {"action": "purchase", "user": "alice"}
    query = "Nginx Workload"
    q_text = json.dumps(query)
    q_emb  = get_embedding(q_text).reshape(1, -1).astype(np.float32)
    normalize(q_emb)

    # --- Perform a k-NN search ---
    k = 3
    print(f"\nQuerying FAISS for top {k} results for {query}...")
    distances, indices = index.search(q_emb, k)

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        print(idx, score)
        doc_id = ids[idx]
        print(f"{rank}. id={doc_id}\tscore={score:.4f}")
        if USE_MONGO:
            doc = coll.find_one({"_id": doc_id}, {"vector": 0})
            print("   ->", doc)

if __name__ == "__main__":
    main()
