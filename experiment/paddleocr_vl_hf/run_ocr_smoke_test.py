#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


# HF Hub can default to the Xet backend, which was unstable in this environment
# for the first large model download. Force plain HTTP downloads for repeatability.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parent.parent
DEFAULT_IMAGE = BENCHMARK_DIR / "data" / "test" / "clean" / "row_C_6.jpg"
DEFAULT_OUTPUT_JSON = BASE_DIR / "output" / "row_C_6.result.json"
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local smoke test against Hugging Face PaddleOCR-VL v1."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Path to the input image.")
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
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Where to save the JSON result.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return requested_device


def resolve_dtype(requested_dtype: str, device: str):
    import torch

    if requested_dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[requested_dtype]

    if device == "cpu" and dtype == torch.float16:
        raise RuntimeError("CPU inference does not support --dtype float16 in this script.")

    return dtype


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_model_and_processor(model_id: str, device: str, torch_dtype):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    install_create_causal_mask_compat(model)
    return model, processor


def install_create_causal_mask_compat(model) -> None:
    module = sys.modules.get(model.__class__.__module__)
    if module is None or not hasattr(module, "create_causal_mask"):
        return

    original = module.create_causal_mask
    parameters = inspect.signature(original).parameters
    if "inputs_embeds" in parameters:
        return

    def compatible_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" in parameters:
            kwargs["input_embeds"] = kwargs["inputs_embeds"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        return original(*args, **filtered_kwargs)

    module.create_causal_mask = compatible_create_causal_mask


def decode_generated_text(processor, generated, inputs):
    input_ids = inputs.get("input_ids")
    if input_ids is not None and generated.shape[1] > input_ids.shape[1]:
        generated = generated[:, input_ids.shape[1] :]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def run_inference(args: argparse.Namespace) -> dict:
    import torch

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

    image = Image.open(image_path).convert("RGB")
    model, processor = load_model_and_processor(MODEL_ID, device, torch_dtype)

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
    cuda_device_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    return {
        "model_id": MODEL_ID,
        "image_path": str(image_path),
        "task": args.task,
        "prompt": PROMPTS[args.task],
        "device": device,
        "dtype": dtype_name(torch_dtype),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": cuda_device_name,
        "max_new_tokens": args.max_new_tokens,
        "generated_text": generated_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_result(payload: dict, output_json: str) -> Path:
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output_path


def main() -> int:
    args = parse_args()

    try:
        payload = run_inference(args)
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

    if not payload["generated_text"]:
        print("ERROR: Model returned empty generated_text.", file=sys.stderr)
        return 1

    output_path = save_result(payload, args.output_json)
    print(f"Model: {payload['model_id']}")
    print(f"Image: {payload['image_path']}")
    print(f"Task: {payload['task']}")
    print(f"Device: {payload['device']}")
    print(f"Dtype: {payload['dtype']}")
    if payload["cuda_device_name"]:
        print(f"CUDA Device: {payload['cuda_device_name']}")
    print("Generated Text:")
    print(payload["generated_text"])
    print(f"Saved JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
