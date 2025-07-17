import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
import ollama
from pymongo import MongoClient
from bson import ObjectId
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree
from rich import box
import random
import string

# Configuration
JSONL_PATH = "data.jsonl"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "snowflake-arctic-embed:22m"
LLM_MODEL = "qwen3:0.6"
INDEX_PATH = "faiss.index"
REFS_PATH = "index_refs.json"
CHUNK_SIZE = 500  # characters
CHUNK_STRIDE = 250  # sliding window stride
TOP_K = 5

console = Console()

# 0. Generate sample JSONL data for testing
def generate_sample_data(path: str, num_docs: int = 10):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(num_docs):
            words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3,8))) for _ in range(50)]
            doc = {"id": f"doc{i}", "source": f"source{i}", "content": " ".join(words)}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    console.print(f"[green]Generated {num_docs} sample docs in {path}[/]")

# 1. Flatten JSON

def flatten_json(obj: Any, prefix: str = '') -> Dict[str, Any]:
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            items.update(flatten_json(v, key))
    else:
        items[prefix] = obj
    return items

# 2. Load & chunk docs (includes raw)

def load_and_chunk(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks = []
    with Progress(
        TextColumn("{task.description}"), BarColumn(), "{task.completed}/{task.total} lines", TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Chunking data", total=len(lines))
        for line in lines:
            raw = json.loads(line)
            flat = flatten_json(raw)
            text = json.dumps(flat, ensure_ascii=False)
            for i in range(0, len(text), CHUNK_STRIDE):
                seg = text[i:i + CHUNK_SIZE]
                chunks.append({
                    'chunk': seg,
                    'source': raw.get('id') or raw.get('source'),
                    'raw': raw
                })
                if i + CHUNK_SIZE >= len(text):
                    break
            progress.advance(task)
    return chunks

# 3. Embed texts via Ollama SDK

def embed_texts(texts: List[str]) -> np.ndarray:
    embs = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Embedding chunks", total=len(texts))
        for text in texts:
            try:
                resp = ollama.embed(model=EMBED_MODEL, input=text)
                vec = resp.get('embeddings') or resp.get('embedding')
                if not vec:
                    console.print(f"[red]Empty embedding for segment[/]")
                else:
                    if isinstance(vec[0], list): vec = vec[0]
                    embs.append(vec)
            except Exception as e:
                console.print(f"[red]Embed error:[/] {e}")
            progress.advance(task)
    arr = np.array(embs, dtype='float32') if embs else np.empty((0,0), dtype='float32')
    if arr.ndim > 2: arr = arr.reshape(arr.shape[0], -1)
    return arr

# 4. Build FAISS index + save refs

def build_index(embs: np.ndarray, refs: List[str]) -> None:
    if embs.ndim != 2 or embs.size == 0:
        raise ValueError("Embeddings must be 2D non-empty")
    dim = embs.shape[1]
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
        task = p.add_task("Building FAISS index")
        idx = faiss.IndexFlatL2(dim)
        idx.add(embs)
        p.advance(task)
    faiss.write_index(idx, INDEX_PATH)
    with open(REFS_PATH, 'w') as f:
        json.dump(refs, f)

# 5. Insert into MongoDB

def insert_into_mongo(chunks: List[Dict[str, Any]], embs: np.ndarray) -> List[str]:
    ids = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as p:
        task = p.add_task("Inserting into MongoDB", total=len(chunks))
        client = MongoClient(MONGO_URI)
        col = client[DB_NAME][COLLECTION_NAME]
        for ch, vec in zip(chunks, embs):
            record = {
                'chunk': ch['chunk'],
                'source': ch['source'],
                'raw': ch['raw'],
                'embedding': vec.tolist()
            }
            res = col.insert_one(record)
            ids.append(str(res.inserted_id))
            p.advance(task)
        client.close()
    return ids

# 6a. Semantic search with scores and raw

def semantic_search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(REFS_PATH)):
        console.print("[red]Index not found. Please ingest first.[/]")
        return []
    q_emb = embed_texts([query])
    if q_emb.ndim != 2 or q_emb.size == 0:
        return []
    idx = faiss.read_index(INDEX_PATH)
    distances, indices = idx.search(q_emb, k)
    refs = json.load(open(REFS_PATH))
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    results = []
    for dist, pos in zip(distances[0], indices[0]):
        doc = col.find_one({'_id': ObjectId(refs[pos])})
        if doc:
            results.append({'source': doc['source'], 'chunk': doc['chunk'], 'raw': doc['raw'], 'score': float(dist)})
    client.close()
    return results

# 6b. Keyword search with raw

def keyword_search(term: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(REFS_PATH)):
        console.print("[red]Index not found. Please ingest first.[/]")
        return []
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    docs = list(col.find({'chunk': {'$regex': term, '$options': 'i'}}).limit(k))
    client.close()
    return [{'source': d['source'], 'chunk': d['chunk'], 'raw': d['raw']} for d in docs]

# 7. LLM chat

def chat_with_llm(question: str, contexts: List[str]) -> str:
    prompt = "Use the following documents to answer the question:\n"
    for i, c in enumerate(contexts): prompt += f"[Doc {i+1}]: {c}\n"
    prompt += f"\nQuestion: {question}"
    try:
        out = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return out.get('response', '').strip()
    except Exception as e:
        return f"[Error] {e}"

# 8. CLI

def main():
    console.print(Panel("[bold cyan]RAG App with Ollama & FAISS[/]", expand=False))
    while True:
        console.print("\n[1] Gen Sample  [2] Ingest  [3] Semantic  [4] Keyword  [5] Ask  [0] Exit")
        opt = IntPrompt.ask("Option")
        if opt == 1:
            n = int(Prompt.ask("# of sample docs", default="10"))
            generate_sample_data(JSONL_PATH, n)
        elif opt == 2:
            console.print("[blue]Ingesting...[/]")
            chunks = load_and_chunk(JSONL_PATH)
            texts = [c['chunk'] for c in chunks]
            embs = embed_texts(texts)
            refs = insert_into_mongo(chunks, embs)
            if os.path.exists(INDEX_PATH) and os.path.exists(REFS_PATH):
                console.print("[yellow]Index exists, skipping re-index.[/]")
            else:
                console.print("[blue]Building index...[/]")
                build_index(embs, refs)
            console.print("[green]Ingest complete![/]")
        elif opt == 3:
            q = Prompt.ask("Query")
            results = semantic_search(q)
            for r in results:
                tree = Tree(f"[bold]{r['source']}[/] (score: {r['score']:.4f})")
                tree.add(f"[italic]Chunk:[/] {r['chunk']}")
                tree.add(f"[italic]Raw:[/] {json.dumps(r['raw'], indent=2)}")
                console.print(tree)
        elif opt == 4:
            t = Prompt.ask("Term")
            docs = keyword_search(t)
            for d in docs:
                tree = Tree(f"[bold]{d['source']}[/]")
                tree.add(f"[italic]Chunk:[/] {d['chunk']}")
                tree.add(f"[italic]Raw:[/] {json.dumps(d['raw'], indent=2)}")
                console.print(tree)
        elif opt == 5:
            q = Prompt.ask("Question")
            ctxs = [d['chunk'] for d in semantic_search(q)]
            ans = chat_with_llm(q, ctxs)
            console.print(Panel(ans, title="Answer", style="green"))
        elif opt == 0:
            break
        else:
            console.print("[red]Invalid option[/]")

if __name__ == '__main__':
    main()
