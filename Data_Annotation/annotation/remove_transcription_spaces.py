#!/usr/bin/env python3
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = SCRIPT_DIR / "results.json"


def clean_transcription_spaces(node):
    updated_count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "transcription" and isinstance(value, str):
                cleaned = re.sub(r"[ \u3000]+", "", value)
                if cleaned != value:
                    node[key] = cleaned
                    updated_count += 1
            else:
                updated_count += clean_transcription_spaces(value)
    elif isinstance(node, list):
        for item in node:
            updated_count += clean_transcription_spaces(item)
    return updated_count


def main():
    json_path = DEFAULT_JSON_PATH
    data = json.loads(json_path.read_text(encoding="utf-8"))
    updated_count = clean_transcription_spaces(data)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated: {json_path}")
    print(f"Transcription fields changed: {updated_count}")


if __name__ == "__main__":
    main()
