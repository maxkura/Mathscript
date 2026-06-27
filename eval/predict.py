#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = BASE_DIR / "Data_Annotation" / "annotation" / "images"
DEFAULT_PROMPT_FILE = BASE_DIR / "prompt" / "preannotation_prompt.txt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output" / "predict"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 6
DEFAULT_MAX_TOKENS = 4000


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    base_url: str
    api_key_env: str
    request_model_name: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    max_tokens_param: str = "max_tokens"


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        model_name="gpt-4o",
        base_url="https://az.gptplus5.com/v1",
        api_key_env="OPENAI_COMPAT_API_KEY",
    ),
    "grok-4-0709": ModelConfig(
        model_name="grok-4-0709",
        base_url="https://az.gptplus5.com/v1",
        api_key_env="OPENAI_COMPAT_API_KEY",
    ),
    "claude-sonnet-4-5-20250929": ModelConfig(
        model_name="claude-sonnet-4-5-20250929",
        base_url="https://az.gptplus5.com/v1",
        api_key_env="OPENAI_COMPAT_API_KEY",
    ),
    "gemini-2.5-flash": ModelConfig(
        model_name="gemini-2.5-flash",
        base_url="https://az.gptplus5.com/v1",
        api_key_env="OPENAI_COMPAT_API_KEY",
    ),
    "kimi-k2.5": ModelConfig(
        model_name="kimi-k2.5",
        base_url="https://az.gptplus5.com/v1",
        api_key_env="KIMI_API_KEY",
        extra_body={"thinking": {"disabled": True}},
    ),
    "qwen3-vl-plus": ModelConfig(
        model_name="qwen3-vl-plus",
        base_url="https://apis.iflow.cn/v1",
        api_key_env="IFLOW_API_KEY",
    ),
    "qwen3.5-flash": ModelConfig(
        model_name="qwen3.5-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        # Disable DashScope hybrid thinking so the model returns stable final JSON.
        extra_body={"enable_thinking": False},
    ),
    "qwen3.5-plus": ModelConfig(
        model_name="qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        # Disable DashScope hybrid thinking so the model returns stable final JSON.
        extra_body={"enable_thinking": False},
    ),
    "mimo-v2-omni": ModelConfig(
        model_name="mimo-v2-omni",
        base_url="https://api.xiaomimimo.com/v1",
        api_key_env="MIMO_API_KEY",
        # MiMo enables chain-of-thought by default and counts reasoning tokens
        # against max_completion_tokens, which can starve the final JSON output.
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens_param="max_completion_tokens",
    ),
    "doubao-1-5-vision-pro-32K": ModelConfig(
        model_name="doubao-1-5-vision-pro-32K",
        request_model_name="ep-20260329112311-vh8tn",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="DOUBAO_API_KEY",
    ),
    "doubao-seed-1.6-vision": ModelConfig(
        model_name="doubao-seed-1.6-vision",
        request_model_name="ep-20260329113610-g4gbc",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="DOUBAO_API_KEY",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call a vision model API and store flattened raw predict results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model name.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for model-specific outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output file path.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override the default API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override the default API key.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum API retry count per image.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum completion token budget per image. For providers with reasoning "
            "support, this budget may include both reasoning tokens and visible output."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess images even if they already have successful records.",
    )
    return parser.parse_args()


def resolve_output_path(output_root: Path, output: Optional[Path], model: str) -> Path:
    if output is not None:
        return output
    return output_root / model / "results.json"


def resolve_model_config(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> Tuple[ModelConfig, str]:
    normalized_model = model.strip()
    if normalized_model not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model}'. Available models: {available}")

    base_config = MODEL_REGISTRY[normalized_model]
    resolved_base_url = base_url.strip() if base_url and base_url.strip() else base_config.base_url
    env_api_key = os.getenv(base_config.api_key_env, "").strip()
    resolved_api_key = api_key.strip() if api_key and api_key.strip() else env_api_key

    if not resolved_base_url:
        raise ValueError(f"Base URL is empty for model '{normalized_model}'.")
    if not resolved_api_key:
        raise ValueError(
            f"API key is empty for model '{normalized_model}'. Set {base_config.api_key_env} "
            "or pass --api-key explicitly."
        )

    config = ModelConfig(
        model_name=base_config.model_name,
        request_model_name=base_config.request_model_name,
        base_url=resolved_base_url,
        api_key_env=base_config.api_key_env,
        extra_body=base_config.extra_body,
        response_format=base_config.response_format,
        max_tokens_param=base_config.max_tokens_param,
    )
    return config, resolved_api_key


def load_prompt(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt_text


def list_image_files(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a directory: {image_dir}")

    files = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.relative_to(image_dir).as_posix().lower())


def guess_mime_type(image_path: Path) -> str:
    if image_path.suffix.lower() == ".png":
        return "image/png"
    return "image/jpeg"


def image_to_data_url(image_path: Path) -> str:
    mime_type = guess_mime_type(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def relative_image_path(image_dir: Path, image_path: Path) -> str:
    return image_path.relative_to(image_dir).as_posix()


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


def build_client(config: ModelConfig, api_key: str) -> Tuple[OpenAI, httpx.Client]:
    http_client = httpx.Client(timeout=DEFAULT_TIMEOUT)
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        http_client=http_client,
    )
    return client, http_client


def call_vision_model_with_retry(
    client: OpenAI,
    config: ModelConfig,
    prompt_text: str,
    image_data_url: str,
    max_retries: int,
    max_tokens: int,
) -> str:
    request_kwargs: Dict[str, Any] = {
        "model": config.request_model_name or config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": 0,
        config.max_tokens_param: max_tokens,
    }
    if config.extra_body is not None:
        request_kwargs["extra_body"] = config.extra_body
    if config.response_format is not None:
        request_kwargs["response_format"] = config.response_format

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
                raise RuntimeError(
                    f"Non-retryable API request error for model {config.model_name}: {exc}"
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

    return None, " | ".join(reasons)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_idx(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def parse_model_output(raw_output: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    parsed, error_message = try_parse_json(raw_output)
    if parsed is None:
        return None, error_message

    if isinstance(parsed, dict):
        parsed_items = [parsed]
    elif isinstance(parsed, list):
        parsed_items = parsed
    else:
        return None, f"Parsed JSON root is {type(parsed).__name__}, expected array or object."

    if not parsed_items:
        return None, "Parsed JSON array is empty."

    normalized_items: List[Dict[str, Any]] = []
    for item_index, item in enumerate(parsed_items, start=1):
        if not isinstance(item, dict):
            return None, f"Element {item_index} in parsed JSON array is not an object."
        normalized_items.append(
            {
                "idx": normalize_idx(item.get("idx")),
                "QuestionType": normalize_text(item.get("QuestionType")),
                "transcription": normalize_text(item.get("transcription")),
                "final_answer": normalize_text(item.get("final_answer")),
            }
        )

    return normalized_items, ""


def create_success_records(
    img_rel: str,
    parsed_items: List[Dict[str, Any]],
    raw_output: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "ImgReal": img_rel,
            "idx": item.get("idx"),
            "QuestionType": item.get("QuestionType", ""),
            "transcription": item.get("transcription", ""),
            "final_answer": item.get("final_answer", ""),
            "raw_output": raw_output,
            "predict_status": "success",
            "error_message": "",
        }
        for item in parsed_items
    ]


def create_parse_failed_record(
    img_rel: str,
    raw_output: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "ImgReal": img_rel,
        "idx": None,
        "QuestionType": "",
        "transcription": "",
        "final_answer": "",
        "raw_output": raw_output,
        "predict_status": "parse_failed",
        "error_message": normalize_text(error_message),
    }


def create_api_error_record(img_rel: str, error_message: str) -> Dict[str, Any]:
    return {
        "ImgReal": img_rel,
        "idx": None,
        "QuestionType": "",
        "transcription": "",
        "final_answer": "",
        "raw_output": f"ERROR: {normalize_text(error_message)}",
        "predict_status": "api_error",
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


def atomic_write_json(path: Path, data: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def group_results_by_image(
    results: List[Any],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ungrouped: List[Any] = []

    for record in results:
        if not isinstance(record, dict):
            ungrouped.append(record)
            continue
        img_rel = record.get("ImgReal")
        if not isinstance(img_rel, str) or not img_rel.strip():
            ungrouped.append(record)
            continue
        normalized_img_rel = img_rel.strip().replace("\\", "/")
        grouped.setdefault(normalized_img_rel, []).append(record)

    return grouped, ungrouped


def flatten_grouped_results(
    grouped_results: Dict[str, List[Dict[str, Any]]],
    ungrouped_results: List[Any],
) -> List[Any]:
    flattened: List[Any] = []
    for records in grouped_results.values():
        flattened.extend(records)
    flattened.extend(ungrouped_results)
    return flattened


def get_processed_images(grouped_results: Dict[str, List[Dict[str, Any]]]) -> set[str]:
    processed: set[str] = set()
    for img_rel, records in grouped_results.items():
        for record in records:
            if record.get("predict_status") == "success":
                processed.add(img_rel)
                break
    return processed


def main() -> int:
    args = parse_args()

    if args.max_retries < 1:
        print("--max-retries must be at least 1.", file=sys.stderr)
        return 1
    if args.max_tokens < 1:
        print("--max-tokens must be at least 1.", file=sys.stderr)
        return 1

    try:
        config, resolved_api_key = resolve_model_config(args.model, args.base_url, args.api_key)
        output_path = resolve_output_path(args.output_root, args.output, config.model_name)
        prompt_text = load_prompt(args.prompt_file)
        image_files = list_image_files(args.image_dir)
        existing_results = load_results(output_path)
        grouped_results, ungrouped_results = group_results_by_image(existing_results)
        processed_images = get_processed_images(grouped_results)
        client, http_client = build_client(config, resolved_api_key)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    try:
        if not image_files:
            print(f"No images found in {args.image_dir}")
            return 0

        pending_images: List[Path] = []
        for image_path in image_files:
            img_rel = relative_image_path(args.image_dir, image_path)
            if not args.overwrite and img_rel in processed_images:
                continue
            pending_images.append(image_path)

        skipped_existing = len(image_files) - len(pending_images)
        successful_images = 0
        parse_failed_images = 0
        api_failed_images = 0

        print(f"Image directory: {args.image_dir}")
        print(f"Output file: {output_path}")
        print(f"Prompt file: {args.prompt_file}")
        print(f"Model: {config.model_name}")
        print(f"Found {len(image_files)} image(s)")
        print(f"Skipped existing successful images: {skipped_existing}")
        print(f"Pending images: {len(pending_images)}")

        progress = tqdm(pending_images, desc="Generating raw predict results")
        for image_path in progress:
            img_rel = relative_image_path(args.image_dir, image_path)
            progress.set_postfix_str(img_rel)
            progress.refresh()

            try:
                raw_output = call_vision_model_with_retry(
                    client=client,
                    config=config,
                    prompt_text=prompt_text,
                    image_data_url=image_to_data_url(image_path),
                    max_retries=args.max_retries,
                    max_tokens=args.max_tokens,
                )
                parsed_items, parse_error_message = parse_model_output(raw_output)
                if parsed_items is None:
                    new_records = [
                        create_parse_failed_record(
                            img_rel=img_rel,
                            raw_output=raw_output,
                            error_message=parse_error_message,
                        )
                    ]
                    parse_failed_images += 1
                    print(
                        f"Parse failed for {img_rel}: {parse_error_message}",
                        file=sys.stderr,
                    )
                else:
                    new_records = create_success_records(
                        img_rel=img_rel,
                        parsed_items=parsed_items,
                        raw_output=raw_output,
                    )
                    successful_images += 1
            except Exception as exc:
                new_records = [create_api_error_record(img_rel=img_rel, error_message=str(exc))]
                api_failed_images += 1
                print(f"API failed for {img_rel}: {exc}", file=sys.stderr)

            grouped_results[img_rel] = new_records
            atomic_write_json(
                output_path,
                flatten_grouped_results(grouped_results, ungrouped_results),
            )

        print("Completed raw predict generation.")
        print(f"Successful images: {successful_images}")
        print(f"Parse failed images: {parse_failed_images}")
        print(f"API failed images: {api_failed_images}")
        print(f"Skipped existing successful images: {skipped_existing}")
        return 0
    finally:
        http_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
