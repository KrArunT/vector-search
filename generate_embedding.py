import json
import requests

# 1. Your JSON data
payload = {
    "user_id": 12345,
    "action": "purchase",
    "items": [
        {"sku": "A100", "qty": 2},
        {"sku": "B205", "qty": 1}
    ],
    "total": 149.99
}

# 2. Serialize to a single string
text_to_embed = json.dumps(payload)

# 3. Call the correct Ollama embeddings endpoint
response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "nomic-embed-text",
        "prompt": text_to_embed
    }
)
response.raise_for_status()

# 4. Pull out your embedding
data = response.json()
embedding = data["embedding"]

print(f"Embedding vector (length {len(embedding)}):")
print(embedding)
