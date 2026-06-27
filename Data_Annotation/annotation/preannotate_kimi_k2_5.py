#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parent.parent
DEFAULT_IMAGE_DIR = BASE_DIR / "images"
DEFAULT_OUTPUT = BASE_DIR / "results.json"
DEFAULT_PROMPT_FILE = BENCHMARK_DIR / "prompt" / "preannotation_prompt.txt"
DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_MAX_PARSE_RETRIES = 3
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_API_KEY_ENV = "ANNOTATION_API_KEY"
DEFAULT_BASE_URL_ENV = "ANNOTATION_BASE_URL"
MANAGED_FIELDS = {
    "filename",
    "idx",
    "QuestionType",
    "transcription",
    "final_answer",
    "raw_output",
    "preannotation_status",
    "error_message",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the Kimi vision model to pre-annotate images into results.json."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the shared results.json file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to the editable prompt text file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Defaults to kimi-k2.5 when omitted.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum number of API retries for each image.",
    )
    parser.add_argument(
        "--max-parse-retries",
        type=int,
        default=DEFAULT_MAX_PARSE_RETRIES,
        help="Maximum number of additional requests when model output cannot be parsed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing records with the same filename.",
    )
    return parser.parse_args()


def resolve_model(cli_model: Optional[str]) -> str:
    if cli_model and cli_model.strip():
        return cli_model.strip()
    env_model = os.getenv("ANNOTATION_MODEL", "").strip()
    if env_model:
        return env_model
    return DEFAULT_MODEL


def load_prompt(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt_text


def resolve_api_config() -> Tuple[str, str]:
    base_url = os.getenv(DEFAULT_BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    api_key = os.getenv(DEFAULT_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"API key is empty. Set {DEFAULT_API_KEY_ENV} before running pre-annotation."
        )
    return base_url, api_key


def list_image_files(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {image_dir}")

    files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def guess_mime_type(image_path: Path) -> str:
    if image_path.suffix.lower() == ".png":
        return "image/png"
    return "image/jpeg"


def image_to_data_url(image_path: Path) -> str:
    mime_type = guess_mime_type(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def call_vision_model_with_retry(
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_data_url: str,
    max_retries: int,
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
                # Moonshot exposes provider-specific request fields like `thinking`
                # in the raw JSON body. The installed OpenAI SDK does not accept a
                # direct `thinking=` kwarg here, so pass it via extra_body.
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=4000,
            )
            if response.choices:
                return extract_message_text(response.choices[0].message.content)
            return str(response)
        except Exception as exc:
            if isinstance(exc, TypeError):
                raise RuntimeError(f"Non-retryable client-side request error: {exc}") from exc
            if getattr(exc, "status_code", None) == 400:
                raise RuntimeError(
                    f"Non-retryable API request error for model {model}: {exc}"
                ) from exc
            if attempt == max_retries - 1:
                raise
            sleep_seconds = 2 ** attempt
            print(
                f"API error on attempt {attempt + 1}/{max_retries}: {exc}. "
                f"Retrying in {sleep_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state.")


def format_json_error(prefix: str, exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{prefix} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    return f"{prefix}: {exc}"


def try_parse_json(text: str) -> Tuple[Optional[Any], str]:
    if not text or not text.strip():
        return None, "Model returned an empty response."

    snippet = text.strip()
    reasons: List[str] = []

    try:
        return json.loads(snippet), ""
    except Exception as exc:
        reasons.append(format_json_error("Top-level JSON decode failed", exc))

    if "```" in snippet:
        code_blocks = snippet.split("```")
        code_block_errors: List[str] = []
        for block_index, block in enumerate(code_blocks, start=1):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate), ""
            except Exception as exc:
                code_block_errors.append(
                    format_json_error(
                        f"Code block {block_index} JSON decode failed",
                        exc,
                    )
                )
        if code_block_errors:
            reasons.extend(code_block_errors)

    start = snippet.find("{")
    end = snippet.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = snippet[start : end + 1]
        try:
            return json.loads(candidate), ""
        except Exception as exc:
            reasons.append(format_json_error("Extracted object JSON decode failed", exc))
    else:
        reasons.append("No JSON object delimiters were found in the model output.")

    return None, " | ".join(reasons)


def parse_single_record(raw_output: str) -> Tuple[Optional[Dict[str, Any]], str]:
    parsed, error_message = try_parse_json(raw_output)
    if parsed is None:
        return None, error_message
    if not isinstance(parsed, dict):
        return None, f"Parsed JSON root is {type(parsed).__name__}, expected an object."
    return parsed, ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def create_success_record(filename: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": filename,
        "idx": 1,
        "QuestionType": normalize_text(parsed.get("QuestionType")),
        "transcription": normalize_text(parsed.get("transcription")),
        "final_answer": normalize_text(parsed.get("final_answer")),
        "preannotation_status": "success",
        "error_message": "",
    }


def create_parse_failed_record(filename: str, error_message: str) -> Dict[str, Any]:
    return {
        "filename": filename,
        "idx": 1,
        "QuestionType": "",
        "transcription": "",
        "final_answer": "",
        "preannotation_status": "parse_failed",
        "error_message": normalize_text(error_message),
    }


def create_api_error_record(filename: str, error_message: str) -> Dict[str, Any]:
    return {
        "filename": filename,
        "idx": 1,
        "QuestionType": "",
        "transcription": "",
        "final_answer": "",
        "preannotation_status": "api_error",
        "error_message": normalize_text(error_message),
    }


def load_results(output_path: Path) -> List[Any]:
    if not output_path.exists():
        return []

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"results.json parse failed: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("results.json top-level value must be a JSON array.")

    return data


def build_results_index(results: List[Any]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    duplicates = set()

    for position, record in enumerate(results):
        if not isinstance(record, dict):
            continue
        filename = record.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            continue
        normalized = filename.strip()
        if normalized in index:
            duplicates.add(normalized)
            continue
        index[normalized] = position

    if duplicates:
        repeated = ", ".join(sorted(duplicates))
        raise ValueError(f"results.json contains duplicate filename values: {repeated}")

    return index


def atomic_write_json(path: Path, data: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def merge_existing_extra_fields(existing: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        return record
    merged = {
        key: value for key, value in existing.items() if key not in MANAGED_FIELDS
    }
    merged.update(record)
    return merged


def upsert_record(results: List[Any], index_map: Dict[str, int], record: Dict[str, Any]) -> None:
    filename = record["filename"]
    existing_index = index_map.get(filename)
    if existing_index is None:
        results.append(record)
        index_map[filename] = len(results) - 1
        return

    results[existing_index] = merge_existing_extra_fields(results[existing_index], record)


def should_process_image(
    filename: str,
    results: List[Any],
    index_map: Dict[str, int],
    overwrite: bool,
) -> bool:
    if overwrite:
        return True

    existing_index = index_map.get(filename)
    if existing_index is None:
        return True

    existing_record = results[existing_index]
    if not isinstance(existing_record, dict):
        return True

    return existing_record.get("preannotation_status") == "parse_failed"


def annotate_image(
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_path: Path,
    api_max_retries: int,
    parse_max_retries: int,
) -> Dict[str, Any]:
    filename = image_path.name
    image_data_url = image_to_data_url(image_path)
    total_attempts = 1 + max(0, parse_max_retries)
    last_error_message = ""

    for parse_attempt in range(total_attempts):
        raw_output = call_vision_model_with_retry(
            client=client,
            model=model,
            prompt_text=prompt_text,
            image_data_url=image_data_url,
            max_retries=api_max_retries,
        )
        parsed, parse_error_message = parse_single_record(raw_output)
        if parsed is not None:
            return create_success_record(filename, parsed)

        last_error_message = (
            f"Model output could not be parsed as JSON "
            f"(attempt {parse_attempt + 1}/{total_attempts}): {parse_error_message}"
        )
        print(
            f"Parse failed for {filename} on attempt {parse_attempt + 1}/{total_attempts}.",
            file=sys.stderr,
        )
        print(f"Parse reason: {parse_error_message}", file=sys.stderr)
        print("Raw model output begin >>>", file=sys.stderr)
        print(raw_output if raw_output else "<empty response>", file=sys.stderr)
        print("<<< Raw model output end", file=sys.stderr)
        if parse_attempt < total_attempts - 1:
            print(
                f"Re-requesting {filename} after parse failure "
                f"({parse_attempt + 1}/{total_attempts})...",
                file=sys.stderr,
            )

    return create_parse_failed_record(filename, last_error_message)


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=300),
    )


def main() -> int:
    args = parse_args()

    try:
        model = resolve_model(args.model)
        prompt_text = load_prompt(args.prompt_file)
        image_files = list_image_files(args.image_dir)
        results = load_results(args.output)
        results_index = build_results_index(results)
        base_url, api_key = resolve_api_config()
        client = build_client(base_url, api_key)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    if not image_files:
        print(f"No images found in {args.image_dir}")
        return 0

    skipped = 0
    processed = 0
    parse_failed = 0
    api_failed = 0

    print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.output}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Model: {model}")
    print(f"Found {len(image_files)} image(s)")

    progress = tqdm(image_files, desc="Pre-annotating images")
    for image_path in progress:
        filename = image_path.name

        if not should_process_image(
            filename=filename,
            results=results,
            index_map=results_index,
            overwrite=args.overwrite,
        ):
            skipped += 1
            continue

        try:
            progress.set_postfix_str(filename)
            progress.refresh()
            record = annotate_image(
                client=client,
                model=model,
                prompt_text=prompt_text,
                image_path=image_path,
                api_max_retries=args.max_retries,
                parse_max_retries=args.max_parse_retries,
            )
            if record["preannotation_status"] == "parse_failed":
                parse_failed += 1
            else:
                processed += 1
        except Exception as exc:
            record = create_api_error_record(filename, str(exc))
            api_failed += 1

        upsert_record(results, results_index, record)
        atomic_write_json(args.output, results)

    print("Completed pre-annotation.")
    print(f"Successful: {processed}")
    print(f"Parse failed: {parse_failed}")
    print(f"API failed: {api_failed}")
    print(f"Skipped existing: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
