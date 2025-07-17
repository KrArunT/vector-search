import json
import requests
import re
from typing import Dict
from pydantic import BaseModel, ValidationError

from ollama import chat  # Ollama Python package (must be installed & configured)

# Ollama model info
MODEL_NAME = "qwen3:4b"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 400

# Pydantic model for 20 questions JSON structure
class QuestionSet(BaseModel):
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str
    question_6: str
    question_7: str
    question_8: str
    question_9: str
    question_10: str
    question_11: str
    question_12: str
    question_13: str
    question_14: str
    question_15: str
    question_16: str
    question_17: str
    question_18: str
    question_19: str
    question_20: str

LABEL_TEMPLATE = """
You are an AI assistant tasked with generating a single, realistic question-answer pair based on a given document. The question should be something a user might naturally ask when seeking information contained in the document.

Given: {chunk}

Instructions:
1. Analyze the key topics, facts, and concepts in the given document, choose one to focus on.
2. Generate twenty similar questions that a user might ask to find the information in this document that does NOT contain any company name.
3. Use natural language and occasionally include typos or colloquialisms to mimic real user behavior in the question.
4. Ensure the question is semantically related to the document content WITHOUT directly copying phrases.
5. Make sure that all of the questions are similar to eachother. I.E. All asking about a similar topic/requesting the same information.

Output Format:
Return a JSON object with the following structure:
{{
  "question_1": "Generated question text",
  "question_2": "Generated question text",
  ...
}}

Be creative, think like a curious user, and generate your 20 similar questions that would naturally lead to the given document in a semantic search. Ensure your response is a valid JSON object containing only the questions.
"""

def flatten_json(y):
    out = []

    def recurse(t):
        if isinstance(t, dict):
            for k, v in t.items():
                out.append(str(k))
                recurse(v)
        elif isinstance(t, list):
            for item in t:
                recurse(item)
        else:
            text = str(t)
            text = re.sub(r"\s+", " ", text)
            out.append(text)

    recurse(y)
    return " ".join(out)

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap
    return chunks

def generate_positives(chunk_text):
    prompt = LABEL_TEMPLATE.format(chunk=chunk_text)

    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format=QuestionSet.model_json_schema(),
        options={"temperature": 0.8, "max_tokens": 1000},
        think=False
    )
    content = response.message.content
    try:
        # Validate & parse using Pydantic
        questions = QuestionSet.model_validate_json(content)
        return questions.dict()
    except ValidationError as e:
        print("Pydantic validation failed:", e)
        print("Raw content was:", content)
        raise

def generate_negative(chunk_text):
    prompt = f"""You are an AI assistant tasked with generating a sentence unrelated in topic and meaning to the following document chunk:

{chunk_text}

Generate a sentence unrelated in topic and meaning:"""

    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.8, "max_tokens": 100},
    )
    return response.message.content.strip()

def main():
    input_file = "data.jsonl"
    output_file = "triplet_dataset.jsonl"

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {line_num}")
                continue

            flat_text = flatten_json(json_obj)
            if not flat_text.strip():
                print(f"Skipping empty flattened text at line {line_num}")
                continue

            chunks = chunk_text(flat_text)

            for chunk_idx, chunk in enumerate(chunks, start=1):
                try:
                    positives = generate_positives(chunk)
                except Exception as e:
                    print(f"Error generating positives for line {line_num} chunk {chunk_idx}: {e}")
                    continue

                try:
                    negative = generate_negative(chunk)
                except Exception as e:
                    print(f"Error generating negative for line {line_num} chunk {chunk_idx}: {e}")
                    continue

                triplet = {
                    "anchor": chunk,
                    "positive": positives,
                    "negative": negative,
                }
                fout.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                print(f"Processed line {line_num} chunk {chunk_idx}")

if __name__ == "__main__":
    main()
