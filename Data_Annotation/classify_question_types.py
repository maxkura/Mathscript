import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from openai import OpenAI
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VALID_QUESTION_TYPES = {
    "MultipleChoice",
    "FillBlank",
    "ConstructedResponse",
}
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "classification_results.json"


def list_image_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root_dir}")

    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=lambda path: str(path).lower())
    return files


def guess_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def image_to_data_url(image_path: Path) -> str:
    mime = guess_mime_type(image_path)
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_prompt() -> str:
    return """
You are an image classification engine for Chinese exam question images.

Task:
- Look at the whole image.
- Classify the image into exactly one question type.

Allowed QuestionType values:
- MultipleChoice
- FillBlank
- ConstructedResponse

Definitions:
- MultipleChoice: the image mainly contains choice questions or selected options such as A/B/C/D.
- FillBlank: the image mainly contains blanks to fill, short-answer blanks, or direct answer slots.
- ConstructedResponse: the image mainly contains worked solutions, proof questions, calculation steps, or long-form answers.

Rules:
- Output exactly one JSON object.
- Do not output markdown.
- Do not explain your reasoning.
- If multiple question styles appear, choose the main or dominant one in the image.
- The QuestionType value must be one of the three allowed values only.

Output format:
{
  "QuestionType": "MultipleChoice | FillBlank | ConstructedResponse"
}
""".strip()


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    return ""


def call_vision_model_with_retry(
    client: OpenAI,
    model: str,
    image_data_url: str,
    prompt_text: str,
    max_retries: int = 6,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=200,
            )

            if response.choices and response.choices[0].message:
                return extract_message_text(response.choices[0].message.content)

            return ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    return ""


def try_parse_json(text: str) -> Optional[Any]:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = stripped[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return None


def normalize_question_type(value: Any) -> str:
    if not isinstance(value, str):
        return "Unclassified"

    normalized = value.strip()
    if normalized in VALID_QUESTION_TYPES:
        return normalized

    alias_map = {
        "multiplechoice": "MultipleChoice",
        "multiple_choice": "MultipleChoice",
        "choice": "MultipleChoice",
        "fillblank": "FillBlank",
        "fill_blank": "FillBlank",
        "blank": "FillBlank",
        "constructedresponse": "ConstructedResponse",
        "constructed_response": "ConstructedResponse",
        "essay": "ConstructedResponse",
        "solution": "ConstructedResponse",
    }

    return alias_map.get(normalized.lower(), "Unclassified")


def parse_question_type(raw_output: str) -> str:
    parsed = try_parse_json(raw_output)
    if isinstance(parsed, dict):
        return normalize_question_type(parsed.get("QuestionType"))
    return "Unclassified"


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def get_recorded_images(output_path: Path) -> Set[str]:
    recorded: Set[str] = set()
    for record in load_existing_results(output_path):
        img_path = record.get("ImgReal")
        if isinstance(img_path, str) and img_path:
            recorded.add(Path(img_path).resolve().as_posix())
    return recorded


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def append_result(output_path: Path, record: Dict[str, Any]) -> None:
    results = load_existing_results(output_path)
    results.append(record)
    atomic_write_json(output_path, results)


def build_client(api_key: str, base_url: str, timeout: int) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=timeout),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify whole exam-question images into question types.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Image root directory.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the consolidated classification JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DASHSCOPE_MODEL", "qwen-vl-max"),
        help="Vision model name.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="Compatible OpenAI base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="API key. Defaults to DASHSCOPE_API_KEY.",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Max retry count.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "Missing API key. Set DASHSCOPE_API_KEY or pass --api-key explicitly."
        )

    client = build_client(args.api_key, args.base_url, args.timeout)
    prompt_text = build_prompt()
    output_path = Path(args.output).resolve()

    image_files = list_image_files(args.dataset)
    recorded_images = get_recorded_images(output_path)

    pending_files = []
    for image_path in image_files:
        normalized_path = image_path.resolve().as_posix()
        if normalized_path not in recorded_images:
            pending_files.append(image_path)

    total = len(image_files)
    done = total - len(pending_files)

    print("\n===== Classification Progress =====")
    print(f"Total images: {total}")
    print(f"Already processed: {done}")
    print(f"Pending: {len(pending_files)}")
    print("==================================\n")

    for image_path in tqdm(pending_files, desc="Classifying images"):
        error_message = ""
        question_type = "Unclassified"

        try:
            image_data_url = image_to_data_url(image_path)
            raw_output = call_vision_model_with_retry(
                client=client,
                model=args.model,
                image_data_url=image_data_url,
                prompt_text=prompt_text,
                max_retries=args.max_retries,
            )
            question_type = parse_question_type(raw_output)
            if question_type == "Unclassified":
                error_message = "Model output could not be parsed into a valid QuestionType."
        except Exception as exc:
            error_message = str(exc)

        record = {
            "ImageName": image_path.name,
            "ImgReal": image_path.resolve().as_posix(),
            "QuestionType": question_type,
            "status": "success" if not error_message else "error",
            "error": error_message,
        }
        append_result(output_path, record)

    print(f"\nClassification finished. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
