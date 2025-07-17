import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
import ollama
from pymongo import MongoClient
from bson import ObjectId
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from textual.app import App, ComposeResult
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

# Configuration
JSONL_PATH = "data.jsonl"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "snowflake-arctic-embed:22m"
LLM_MODEL = "qwen3:0.6b"
INDEX_PATH = "faiss.index"
REFS_PATH = "index_refs.json"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_STRIDE = 250  # sliding window stride
TOP_K = 5

console = Console()

# -- Utility functions --

def generate_sample_data(path: str, num_docs: int = 10) -> None:
    """
    Generate sample JSONL data with `num_docs` documents.
    Each document has id, source, and random content.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(num_docs):
            words = [
                "".join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=np.random.randint(3,8)))
                for _ in range(50)
            ]
            doc = {"id": f"doc{i}", "source": f"source{i}", "content": " ".join(words)}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    console.print(f"[green]Generated {num_docs} sample docs at {path}[/]")


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten JSON object to a flat dict."""
    items: Dict[str, Any] = {}
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


def load_and_chunk(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL from `path`, flatten each document, and chunk the flattened text.
    Returns a list of dicts: {'chunk', 'source', 'raw'}
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks: List[Dict[str, Any]] = []
    with Progress(
        TextColumn("{task.description}"), BarColumn(), "{task.completed}/{task.total}", TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Chunking data", total=len(lines))
        for line in lines:
            raw = json.loads(line)
            flat = flatten_json(raw)
            text = json.dumps(flat, ensure_ascii=False)
            for start in range(0, len(text), CHUNK_STRIDE):
                seg = text[start:start + CHUNK_SIZE]
                chunks.append({
                    'chunk': seg,
                    'source': raw.get('id') or raw.get('source'),
                    'raw': raw
                })
                if start + CHUNK_SIZE >= len(text):
                    break
            progress.advance(task)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using Ollama embed model.
    Returns a 2D numpy array of shape (n_texts, dim).
    """
    embeddings: List[List[float]] = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Embedding chunks", total=len(texts))
        for txt in texts:
            try:
                resp = ollama.embed(model=EMBED_MODEL, input=txt)
                vec = resp.get('embeddings') or resp.get('embedding') or []
                if isinstance(vec, list) and vec and isinstance(vec[0], list):
                    vec = vec[0]
                if vec:
                    embeddings.append(vec)
            except Exception as e:
                console.print(f"[red]Embed error:[/] {e}")
            progress.advance(task)
    if not embeddings:
        return np.empty((0, 0), dtype='float32')
    arr = np.array(embeddings, dtype='float32')
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def embed_query(text: str) -> np.ndarray:
    """Embed a single query text without spinner."""
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=text)
        vec = resp.get('embeddings') or resp.get('embedding') or []
        if isinstance(vec, list) and vec and isinstance(vec[0], list):
            vec = vec[0]
        return np.array(vec, dtype='float32') if vec else np.empty((0,), dtype='float32')
    except Exception as e:
        console.print(f"[red]Query embed error:[/] {e}")
        return np.empty((0,), dtype='float32')


def insert_into_mongo(chunks: List[Dict[str, Any]], embs: np.ndarray) -> List[str]:
    """
    Insert chunk records into MongoDB. Returns list of inserted ObjectId strings.
    Each record includes 'chunk', 'source', 'raw', and 'embedding'.
    """
    ids: List[str] = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Inserting into MongoDB", total=len(chunks))
        client = MongoClient(MONGO_URI)
        col = client[DB_NAME][COLLECTION_NAME]
        for ch, vec in zip(chunks, embs):
            rec = {
                'chunk': ch['chunk'],
                'source': ch['source'],
                'raw': ch['raw'],
                'embedding': vec.tolist()
            }
            res = col.insert_one(rec)
            ids.append(str(res.inserted_id))
            progress.advance(task)
        client.close()
    return ids


def build_index(embs: np.ndarray, refs: List[str]) -> None:
    """
    Build and save FAISS index from embeddings and reference list.
    """
    if embs.ndim != 2 or embs.size == 0:
        raise ValueError("Embeddings must be 2D and non-empty")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, INDEX_PATH)
    with open(REFS_PATH, 'w') as f:
        json.dump(refs, f)


def semantic_search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Perform semantic search, return list of {'source','chunk','raw','score'}."""
    if not os.path.exists(INDEX_PATH):
        console.print("[red]Index not found. Ingest first.[/]")
        return []
    vec = embed_query(query)
    if vec.size == 0:
        return []
    idx = faiss.read_index(INDEX_PATH)
    distances, indices = idx.search(np.array([vec]), k)
    refs = json.load(open(REFS_PATH))
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    results: List[Dict[str, Any]] = []
    for dist, pos in zip(distances[0], indices[0]):
        doc = col.find_one({'_id': ObjectId(refs[pos])})
        if doc:
            results.append({
                'source': doc['source'],
                'chunk': doc['chunk'],
                'raw': doc.get('raw'),
                'score': float(dist)
            })
    client.close()
    return results


def keyword_search(term: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Perform keyword regex search, return list of {'source','chunk','raw'}."""
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    docs = list(col.find({'chunk': {'$regex': term, '$options': 'i'}}).limit(k))
    client.close()
    return [{
        'source': d.get('source'),
        'chunk': d.get('chunk'),
        'raw': d.get('raw')
    } for d in docs]

# -- Textual App for Results --
class ResultsTreeApp(App):
    CSS = """
    Tree {
        height: 100%;
        width: 100%;
    }
    """

    def __init__(self, results: List[Dict[str, Any]]):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        tree = Tree("Results")
        for r in self.results:
            label = f"{r['source']} (score: {r.get('score', 0):.4f})"
            branch: TreeNode = tree.root.add(label)
            branch.add(f"Chunk: {r['chunk']}")
            raw_branch = branch.add("Raw JSON")
            for line in json.dumps(r.get('raw', {}), indent=2).splitlines():
                raw_branch.add(line)
        yield tree

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        self.log(f"Expanded: {event.node.label}")

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
        self.log(f"Collapsed: {event.node.label}")

# -- CLI Entry Point --
if __name__ == '__main__':
    console.print(Panel("[bold cyan]RAG App with Ollama & FAISS[/]", expand=False))
    while True:
        console.print("[1] Gen Sample  [2] Ingest  [3] Semantic  [4] Keyword  [5] Ask  [0] Exit")
        opt = IntPrompt.ask("Option")
        if opt == 1:
            n = int(Prompt.ask("# of sample docs", default="10"))
            generate_sample_data(JSONL_PATH, n)
        elif opt == 2:
            console.print("[blue]Ingesting...[/]")
            chunks = load_and_chunk(JSONL_PATH)
            embs = embed_texts([c['chunk'] for c in chunks])
            refs = insert_into_mongo(chunks, embs)
            if not os.path.exists(INDEX_PATH):
                console.print("[blue]Building index...[/]")
                build_index(embs, refs)
            console.print("[green]Ingest complete![/]")
        elif opt in (3, 4):
            if opt == 3:
                q = Prompt.ask("Query")
                results = semantic_search(q)
            else:
                t = Prompt.ask("Term")
                docs = keyword_search(t)
                results = [{**d, 'score': 0.0} for d in docs]
            ResultsTreeApp(results).run()
        elif opt == 5:
            q = Prompt.ask("Question")
            ctxs = [d['chunk'] for d in semantic_search(q)]
            prompt = "\n".join(ctxs) + f"\nQuestion: {q}"
            out = ollama.generate(model=LLM_MODEL, prompt=prompt)
            resp = out.get('response', '').strip()
            console.print(Panel(resp, title="Answer", style="green"))
        elif opt == 0:
            break
        else:
            console.print("[red]Invalid option[/]")
