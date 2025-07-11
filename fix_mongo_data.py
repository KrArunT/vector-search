#!/usr/bin/env python3
import json

INPUT_FILE  = "data.jsonl"
OUTPUT_FILE = "data_cleaned.jsonl"

def clean_record(record: dict, default_id=None) -> dict:
    # Remove any existing '_id' field
    record.pop("_id", None)
    # Optionally, ensure there's an 'id' field for your own use:
    if 'id' not in record and default_id is not None:
        record['id'] = default_id
    return record

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {idx+1}: {e}")
                continue

            cleaned = clean_record(obj, default_id=str(idx))
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    print(f"Cleaned data written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
