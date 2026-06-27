#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiment.controlled_reading_order import (
    atomic_write_json,
    load_json_array,
    normalize_filename,
    normalize_idx,
    normalize_text,
    split_ocr_text_into_units,
)
from run_ocr_smoke_test import (
    MODEL_ID,
    PROMPTS,
    decode_generated_text,
    dtype_name,
    load_model_and_processor,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL once over a manifest and save OCR lines as text units."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON array manifest path.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON array path.")
    parser.add_argument(
        "--task",
        choices=sorted(PROMPTS.keys()),
        default="ocr",
        help="Element-level recognition task prompt.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. auto prefers CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
        help="Torch dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run samples that already succeeded.")
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    manifest = load_json_array(path)
    normalized: List[Dict[str, Any]] = []
    for item_index, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {item_index} must be a JSON object.")
        filename = normalize_filename(item.get("filename"))
        idx = normalize_idx(item.get("idx"))
        image_path = normalize_text(item.get("image_path"))
        if not filename or idx is None or not image_path:
            raise ValueError(f"Manifest item {item_index} is missing required fields.")
        normalized.append(item)
    return normalized


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records = load_json_array(path)
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        filename = normalize_filename(record.get("filename"))
        if filename:
            index[filename] = record
    return index


def serialize_results(manifest: List[Dict[str, Any]], result_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [result_index[item["filename"]] for item in manifest if item["filename"] in result_index]


def build_success_record(
    *,
    sample: Dict[str, Any],
    generated_text: str,
    task: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    cuda_available: bool,
    cuda_device_name: str | None,
) -> Dict[str, Any]:
    return {
        "subset": normalize_text(sample.get("subset")),
        "filename": normalize_filename(sample.get("filename")),
        "idx": normalize_idx(sample.get("idx")),
        "QuestionType": normalize_text(sample.get("question_type") or sample.get("QuestionType")),
        "image_path": normalize_text(sample.get("image_path")),
        "ocr_text": generated_text,
        "ocr_units": split_ocr_text_into_units(generated_text),
        "ocr_status": "success",
        "error_message": "",
        "model_id": MODEL_ID,
        "task": task,
        "prompt": PROMPTS[task],
        "device": device,
        "dtype": dtype,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "max_new_tokens": max_new_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_error_record(
    *,
    sample: Dict[str, Any],
    error_message: str,
    task: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    return {
        "subset": normalize_text(sample.get("subset")),
        "filename": normalize_filename(sample.get("filename")),
        "idx": normalize_idx(sample.get("idx")),
        "QuestionType": normalize_text(sample.get("question_type") or sample.get("QuestionType")),
        "image_path": normalize_text(sample.get("image_path")),
        "ocr_text": "",
        "ocr_units": [],
        "ocr_status": "error",
        "error_message": normalize_text(error_message),
        "model_id": MODEL_ID,
        "task": task,
        "prompt": PROMPTS[task],
        "device": device,
        "dtype": dtype,
        "cuda_available": None,
        "cuda_device_name": None,
        "max_new_tokens": max_new_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()

    try:
        manifest = load_manifest(args.manifest)
        existing_index = load_existing_results(args.output_json)
        device = resolve_device(args.device)
        torch_dtype = resolve_dtype(args.dtype, device)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    try:
        import torch

        model, processor = load_model_and_processor(MODEL_ID, device, torch_dtype)
        cuda_available = torch.cuda.is_available()
        cuda_device_name = torch.cuda.get_device_name(0) if device == "cuda" else None

        for sample in manifest:
            filename = normalize_filename(sample.get("filename"))
            existing = existing_index.get(filename)
            if existing is not None and not args.overwrite and existing.get("ocr_status") == "success":
                continue

            try:
                image_path = Path(normalize_text(sample.get("image_path"))).expanduser().resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"Input image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPTS[args.task]},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                generated_text = decode_generated_text(processor, generated, inputs)
                if not generated_text.strip():
                    raise RuntimeError("Model returned empty generated_text.")

                existing_index[filename] = build_success_record(
                    sample=sample,
                    generated_text=generated_text,
                    task=args.task,
                    device=device,
                    dtype=dtype_name(torch_dtype),
                    max_new_tokens=args.max_new_tokens,
                    cuda_available=cuda_available,
                    cuda_device_name=cuda_device_name,
                )
            except Exception as exc:
                existing_index[filename] = build_error_record(
                    sample=sample,
                    error_message=str(exc),
                    task=args.task,
                    device=device,
                    dtype=dtype_name(torch_dtype),
                    max_new_tokens=args.max_new_tokens,
                )
                print(f"OCR failed for {filename}: {exc}", file=sys.stderr)

            atomic_write_json(args.output_json, serialize_results(manifest, existing_index))
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            print(
                "CUDA OOM while loading or generating with PaddleOCR-VL. "
                "Re-run with: --device cpu --dtype float32",
                file=sys.stderr,
            )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {len(manifest)} manifest sample(s).")
    print(f"Saved JSON: {args.output_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
