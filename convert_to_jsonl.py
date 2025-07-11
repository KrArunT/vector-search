import json
import sys

def convert(input_file, output_file='data.jsonl'):
    """
    Load a JSON file containing a list of JSON objects and write them to a JSON Lines file.

    :param input_file: Path to the input JSON file (must contain a top-level list of objects).
    :param output_file: Path to the output JSONL file (default: data.jsonl).
    """
    # Read the list of objects from the JSON file
    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    # Ensure the data is a list
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects in the JSON file, but got {type(data).__name__}.")

    # Write each object as a separate line in the JSONL file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for obj in data:
            json_line = json.dumps(obj, ensure_ascii=False)
            f_out.write(json_line + '\n')

    print(f"Converted {len(data)} objects from '{input_file}' to JSON Lines format in '{output_file}'.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_jsonl.py <input.json> [output.jsonl]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data.jsonl'
    try:
        convert(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
