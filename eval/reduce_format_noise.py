#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = BASE_DIR / "output" / "predict"
DEFAULT_PROMPT_FILE = BASE_DIR / "prompt" / "predict_result_cleanup_prompt.txt"
DEFAULT_OUTPUT_NAME = "result_less_format_noise.json"
DEFAULT_MODEL_NAME = "kimi-k2.5"
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_API_KEY_ENV = "KIMI_API_KEY"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 6
DEFAULT_MAX_PARSE_RETRIES = 3
DEFAULT_MAX_TOKENS = 4000
PROMPT_PLACEHOLDER = "{input_json_object}"

CleanupResult = Tuple[Optional[str], str]
CleanupCallable = Callable[[Dict[str, Any]], CleanupResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce transcription format noise in output/predict/*/results.json by "
            "calling kimi-k2.5 record by record."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target predict model directory name. Omit to scan every model under --input-root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory that contains model subdirectories with results.json.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output filename to write under each model directory.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt template path. Must contain the {input_json_object} placeholder.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Kimi-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=f"Kimi-compatible API key. Defaults to ${DEFAULT_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum API retry count for each request.",
    )
    parser.add_argument(
        "--max-parse-retries",
        type=int,
        default=DEFAULT_MAX_PARSE_RETRIES,
        help="Maximum additional requests when the model response cannot be parsed.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens per cleanup request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args()


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    if PROMPT_PLACEHOLDER not in prompt_text:
        raise ValueError(
            f"Prompt file must contain the {PROMPT_PLACEHOLDER} placeholder: {prompt_file}"
        )
    return prompt_text


def load_results(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"results.json not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"results.json parse failed: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("results.json top-level value must be a JSON array.")

    return data


def atomic_write_json(path: Path, data: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def should_cleanup_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("predict_status") == "success"
        and isinstance(record.get("transcription"), str)
        and record.get("transcription") != ""
    )


def build_cleanup_input_object(record: Dict[str, Any]) -> Dict[str, str]:
    payload = {"transcription": record.get("transcription", "")}
    raw_output = record.get("raw_output")
    if isinstance(raw_output, str) and raw_output:
        payload["raw_output"] = raw_output
    return payload


def build_cleanup_prompt(prompt_template: str, record: Dict[str, Any]) -> str:
    input_payload = build_cleanup_input_object(record)
    payload_text = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return prompt_template.replace(PROMPT_PLACEHOLDER, payload_text)


def build_parse_retry_prompt(base_prompt: str, parse_error: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "IMPORTANT RETRY INSTRUCTION\n"
        "Your previous response was not parseable as exactly one JSON object with "
        'a string field named "transcription".\n'
        f"Parse error: {parse_error}\n"
        'Return JSON only, in the form {"transcription": "..."}.\n'
    )


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or (
                    "text" in item and item.get("type") in (None, "")
                ):
                    parts.append(str(item.get("text", "")))
                continue

            item_type = getattr(item, "type", None)
            if item_type == "text":
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def build_empty_response_error(choice: Any) -> str:
    message = getattr(choice, "message", None)
    finish_reason = getattr(choice, "finish_reason", None) or "unknown"
    reasoning_content = extract_message_text(getattr(message, "reasoning_content", None))
    refusal = getattr(message, "refusal", None)
    reasoning_state = "present" if reasoning_content.strip() else "absent"
    refusal_state = "present" if refusal else "absent"
    return (
        "Model returned an empty final response "
        f"(finish_reason={finish_reason}, reasoning_content={reasoning_state}, "
        f"refusal={refusal_state})."
    )


def build_client(base_url: str, api_key: str) -> Tuple[OpenAI, httpx.Client]:
    http_client = httpx.Client(timeout=DEFAULT_TIMEOUT)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )
    return client, http_client


def resolve_api_key(cli_api_key: Optional[str]) -> str:
    if cli_api_key and cli_api_key.strip():
        return cli_api_key.strip()
    env_api_key = os.getenv(DEFAULT_API_KEY_ENV, "").strip()
    if env_api_key:
        return env_api_key
    raise ValueError(
        f"API key is empty. Set {DEFAULT_API_KEY_ENV} or pass --api-key explicitly."
    )


def call_kimi_with_retry(
    client: OpenAI,
    prompt_text: str,
    max_retries: int,
    max_tokens: int,
) -> str:
    request_kwargs: Dict[str, Any] = {
        "model": DEFAULT_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "extra_body": {"thinking": {"type": "disabled"}},
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise RuntimeError("Model response contained no choices.")

            choice = response.choices[0]
            raw_text = extract_message_text(choice.message.content)
            if raw_text.strip():
                return raw_text

            raise RuntimeError(build_empty_response_error(choice))
        except Exception as exc:
            if isinstance(exc, TypeError):
                raise RuntimeError(f"Non-retryable client-side request error: {exc}") from exc
            if getattr(exc, "status_code", None) == 400:
                raise RuntimeError(f"Non-retryable API request error: {exc}") from exc
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

    object_start = snippet.find("{")
    object_end = snippet.rfind("}")
    if object_start != -1 and object_end != -1 and object_start < object_end:
        candidate = snippet[object_start : object_end + 1]
        try:
            return json.loads(candidate), ""
        except Exception as exc:
            reasons.append(format_json_error("Extracted object JSON decode failed", exc))
    else:
        reasons.append("No JSON object delimiters were found in the model output.")

    array_start = snippet.find("[")
    array_end = snippet.rfind("]")
    if array_start != -1 and array_end != -1 and array_start < array_end:
        candidate = snippet[array_start : array_end + 1]
        try:
            return json.loads(candidate), ""
        except Exception as exc:
            reasons.append(format_json_error("Extracted array JSON decode failed", exc))
    else:
        reasons.append("No JSON array delimiters were found in the model output.")

    return None, " | ".join(reasons)


def parse_cleanup_output(raw_output: str) -> CleanupResult:
    parsed, error_message = try_parse_json(raw_output)
    if parsed is None:
        return None, error_message

    parsed_object: Any = parsed
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed_object = parsed[0]

    if not isinstance(parsed_object, dict):
        return None, f"Parsed JSON root is {type(parsed).__name__}, expected object."

    if "transcription" not in parsed_object:
        return None, 'Parsed JSON object does not contain the "transcription" field.'

    transcription = parsed_object.get("transcription")
    if not isinstance(transcription, str):
        return None, 'Parsed "transcription" must be a JSON string.'

    return transcription, ""


def cleanup_transcription_with_kimi(
    client: OpenAI,
    prompt_template: str,
    record: Dict[str, Any],
    max_retries: int,
    max_parse_retries: int,
    max_tokens: int,
) -> CleanupResult:
    base_prompt = build_cleanup_prompt(prompt_template, record)
    retry_prompt = base_prompt
    last_error = ""

    for parse_attempt in range(max_parse_retries + 1):
        try:
            raw_output = call_kimi_with_retry(
                client=client,
                prompt_text=retry_prompt,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return None, str(exc)

        cleaned_transcription, parse_error = parse_cleanup_output(raw_output)
        if cleaned_transcription is not None:
            return cleaned_transcription, ""

        last_error = parse_error
        if parse_attempt == max_parse_retries:
            break
        retry_prompt = build_parse_retry_prompt(base_prompt, parse_error)

    return None, last_error or "Model response could not be parsed."


def discover_model_directories(input_root: Path, model_name: Optional[str]) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    if model_name:
        model_dir = input_root / model_name
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not (model_dir / "results.json").exists():
            raise FileNotFoundError(f"results.json not found for model: {model_dir}")
        return [model_dir]

    model_dirs = sorted(
        [
            path
            for path in input_root.iterdir()
            if path.is_dir() and (path / "results.json").exists()
        ],
        key=lambda path: path.name,
    )
    return model_dirs


def process_model_directory(
    model_dir: Path,
    output_name: str,
    overwrite: bool,
    cleanup_func: CleanupCallable,
    show_progress: bool = True,
) -> Dict[str, Any]:
    input_path = model_dir / "results.json"
    output_path = model_dir / output_name

    if output_path.exists() and not overwrite:
        return {
            "model": model_dir.name,
            "status": "skipped_existing",
            "output_path": output_path,
            "total_records": 0,
            "eligible_records": 0,
            "cleaned_records": 0,
            "failed_records": 0,
            "passthrough_records": 0,
        }

    results = load_results(input_path)
    output_records = [dict(record) if isinstance(record, dict) else record for record in results]
    eligible_indexes = [index for index, record in enumerate(results) if should_cleanup_record(record)]

    cleaned_records = 0
    failed_records = 0

    # Create the output file before cleanup starts so partial progress survives interruption.
    atomic_write_json(output_path, output_records)
    iterator = tqdm(
        eligible_indexes,
        desc=f"Reducing format noise for {model_dir.name}",
        disable=not show_progress,
    )

    for record_index in iterator:
        record = results[record_index]
        cleaned_transcription, error_message = cleanup_func(record)
        if cleaned_transcription is None:
            failed_records += 1
            img_real = record.get("ImgReal", "")
            idx_value = record.get("idx", "")
            print(
                f"Cleanup failed for {model_dir.name} record idx={idx_value} "
                f"ImgReal={img_real}: {error_message}",
                file=sys.stderr,
            )
            atomic_write_json(output_path, output_records)
            continue
        output_records[record_index]["transcription"] = cleaned_transcription
        cleaned_records += 1
        atomic_write_json(output_path, output_records)
    return {
        "model": model_dir.name,
        "status": "processed",
        "output_path": output_path,
        "total_records": len(results),
        "eligible_records": len(eligible_indexes),
        "cleaned_records": cleaned_records,
        "failed_records": failed_records,
        "passthrough_records": len(results) - len(eligible_indexes),
    }


def main() -> int:
    args = parse_args()

    if args.max_retries < 1:
        print("--max-retries must be at least 1.", file=sys.stderr)
        return 1
    if args.max_parse_retries < 0:
        print("--max-parse-retries must be at least 0.", file=sys.stderr)
        return 1
    if args.max_tokens < 1:
        print("--max-tokens must be at least 1.", file=sys.stderr)
        return 1
    if not args.output_name.strip():
        print("--output-name must not be empty.", file=sys.stderr)
        return 1

    try:
        prompt_template = load_prompt_template(args.prompt_file)
        model_dirs = discover_model_directories(args.input_root, args.model)
        resolved_api_key = resolve_api_key(args.api_key)
        client, http_client = build_client(args.base_url, resolved_api_key)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    if not model_dirs:
        print(f"No model directories with results.json found under {args.input_root}")
        http_client.close()
        return 0

    print(f"Input root: {args.input_root}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Cleanup model: {DEFAULT_MODEL_NAME}")
    print(f"Output filename: {args.output_name}")
    print(f"Target model directories: {', '.join(path.name for path in model_dirs)}")

    total_models = 0
    skipped_models = 0
    cleaned_records = 0
    failed_records = 0
    eligible_records = 0

    try:
        for model_dir in model_dirs:
            summary = process_model_directory(
                model_dir=model_dir,
                output_name=args.output_name,
                overwrite=args.overwrite,
                cleanup_func=lambda record: cleanup_transcription_with_kimi(
                    client=client,
                    prompt_template=prompt_template,
                    record=record,
                    max_retries=args.max_retries,
                    max_parse_retries=args.max_parse_retries,
                    max_tokens=args.max_tokens,
                ),
            )
            total_models += 1

            if summary["status"] == "skipped_existing":
                skipped_models += 1
                print(
                    f"Skipped {summary['model']} because {summary['output_path']} already exists."
                )
                continue

            eligible_records += summary["eligible_records"]
            cleaned_records += summary["cleaned_records"]
            failed_records += summary["failed_records"]
            print(
                f"Processed {summary['model']}: eligible={summary['eligible_records']}, "
                f"cleaned={summary['cleaned_records']}, failed={summary['failed_records']}, "
                f"output={summary['output_path']}"
            )

        print("Completed transcription format-noise reduction.")
        print(f"Models considered: {total_models}")
        print(f"Models skipped because output exists: {skipped_models}")
        print(f"Eligible records: {eligible_records}")
        print(f"Cleaned records: {cleaned_records}")
        print(f"Failed records kept unchanged: {failed_records}")
        return 0
    finally:
        http_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
