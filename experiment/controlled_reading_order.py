#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metric import aggregate_reading_order_metrics, compute_reading_order_metrics
from eval.predict_convert import convert_results


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LINEAR_SOURCE = BENCHMARK_DIR / "data" / "test" / "linear" / "results_new_yes.json"
DEFAULT_UNLINEAR_SOURCE = BENCHMARK_DIR / "data" / "test" / "unlinear" / "results_new_no.json"
DEFAULT_OUTPUT_ROOT = BENCHMARK_DIR / "output" / "experiment" / "ocr_controlled_reading_order"
DEFAULT_OCR_RUNNER = Path(__file__).resolve().parent / "paddleocr_vl_hf" / "run_ocr_batch.py"
DEFAULT_OCR_PYTHON = (
    BENCHMARK_DIR / "experiment" / ".conda" / "paddleocr_vl_hf" / "bin" / "python"
)
DEFAULT_OCR_DEVICE = "auto"
DEFAULT_OCR_DTYPE = "auto"
DEFAULT_OCR_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_RETRIES = 6
DEFAULT_MAX_TOKENS = 4000
DEFAULT_REQUEST_TIMEOUT = 300.0
BASELINE_METHOD_NAME = "spatial_ocr_baseline"
CONSTRUCTED_RESPONSE = "ConstructedResponse"
Q_MARKER_PATTERN = re.compile(r"【q(\d{3})】", re.IGNORECASE)
GOOD_MATH_PATTERN = re.compile(r"\[\[MATH:(.*?)\]\]")
FLEXIBLE_MATH_PATTERN = re.compile(r"(?:\[\[MATH:|\[MATH:)(.*?)\]\]")
SPACE_PATTERN = re.compile(r"[ \u3000]+")
LEADING_ENUM_PATTERN = re.compile(
    "^(?:(?:\\(?[0-9]+\\)?|\\(?[IVXivx]+\\)?|[①②③④⑤⑥⑦⑧⑨⑩]|[\\u4e00\\u4e8c\\u4e09\\u56db\\u4e94\\u516d\\u4e03\\u516b\\u4e5d\\u5341]+)"
    "(?:[.:)]|[\\u3001\\uFF1A\\uFF09])?)+"
)
TRAILING_ENUM_PATTERN = re.compile(
    "(?:(?:\\(?[0-9]+\\)?|\\(?[IVXivx]+\\)?|[①②③④⑤⑥⑦⑧⑨⑩]|[\\u4e00\\u4e8c\\u4e09\\u56db\\u4e94\\u516d\\u4e03\\u516b\\u4e5d\\u5341]+)"
    "(?:[.:)]|[\\u3001\\uFF1A\\uFF09])?)+$"
)
QUESTION_TITLE_PREFIX_PATTERN = re.compile(
    r"^\d+(?:[.)]|[\u3001\u9898\uFF09])?(?:(?:[\(\uFF08][^()\uFF08\uFF09]{0,16}[\)\uFF09]))?"
)


@dataclass(frozen=True)
class SampleSpec:
    subset: str
    filename: str
    idx: int
    question_type: str
    image_path: str


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    manifest_path: Path
    gt_path: Path
    gt_error_report_path: Path
    ocr_results_path: Path
    raw_root: Path
    predict_root: Path
    metric_root: Path
    summary_json_path: Path
    summary_csv_path: Path


def get_model_registry() -> Dict[str, Any]:
    from eval.predict import MODEL_REGISTRY

    return MODEL_REGISTRY


def build_text_client(config: Any, request_timeout: float) -> Tuple[Any, Any]:
    import httpx
    from openai import OpenAI

    api_key = os.getenv(config.api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"API key is empty for model '{config.model_name}'. "
            f"Set {config.api_key_env} before running reorder experiments."
        )
    http_client = httpx.Client(timeout=request_timeout)
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        http_client=http_client,
    )
    return client, http_client


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 41-sample OCR-controlled reading-order experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_gt_parser = subparsers.add_parser("build-gt", help="Build the 41-sample GT and manifest.")
    add_common_io_args(build_gt_parser)
    add_source_args(build_gt_parser)

    run_ocr_parser = subparsers.add_parser("run-ocr", help="Run PaddleOCR-VL on the 41-sample manifest.")
    add_common_io_args(run_ocr_parser)
    run_ocr_parser.add_argument("--ocr-runner", type=Path, default=DEFAULT_OCR_RUNNER)
    run_ocr_parser.add_argument("--ocr-python", type=Path, default=DEFAULT_OCR_PYTHON)
    run_ocr_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEFAULT_OCR_DEVICE)
    run_ocr_parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default=DEFAULT_OCR_DTYPE,
    )
    run_ocr_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_OCR_MAX_NEW_TOKENS)
    run_ocr_parser.add_argument("--overwrite", action="store_true")

    run_reorder_parser = subparsers.add_parser(
        "run-reorder",
        help="Build the baseline and run text-only reordering for selected or all models.",
    )
    add_common_io_args(run_reorder_parser)
    run_reorder_parser.add_argument("--model", nargs="+", default=None, help="Model(s) to run.")
    run_reorder_parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    run_reorder_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_reorder_parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    run_reorder_parser.add_argument("--overwrite", action="store_true")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw reorder outputs into standardized predict.json files.",
    )
    add_common_io_args(convert_parser)
    convert_parser.add_argument("--method", nargs="+", default=None, help="Method(s) to convert.")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate converted predictions with reading-order metrics only.",
    )
    add_common_io_args(eval_parser)
    eval_parser.add_argument("--method", nargs="+", default=None, help="Method(s) to evaluate.")

    run_all_parser = subparsers.add_parser("run-all", help="Run GT -> OCR -> reorder -> convert -> eval.")
    add_common_io_args(run_all_parser)
    add_source_args(run_all_parser)
    run_all_parser.add_argument("--ocr-runner", type=Path, default=DEFAULT_OCR_RUNNER)
    run_all_parser.add_argument("--ocr-python", type=Path, default=DEFAULT_OCR_PYTHON)
    run_all_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEFAULT_OCR_DEVICE)
    run_all_parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default=DEFAULT_OCR_DTYPE,
    )
    run_all_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_OCR_MAX_NEW_TOKENS)
    run_all_parser.add_argument("--model", nargs="+", default=None, help="Model(s) to run.")
    run_all_parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    run_all_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_all_parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    run_all_parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args(argv)


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="default")


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--linear-source", type=Path, default=DEFAULT_LINEAR_SOURCE)
    parser.add_argument("--unlinear-source", type=Path, default=DEFAULT_UNLINEAR_SOURCE)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_experiment_paths(args.output_dir, args.run_name)

    if args.command == "build-gt":
        build_gt_and_manifest(
            linear_source=args.linear_source,
            unlinear_source=args.unlinear_source,
            paths=paths,
        )
        print(f"Built GT and manifest under {paths.root}")
        return 0

    if args.command == "run-ocr":
        ensure_manifest_exists(paths)
        run_ocr_step(
            paths=paths,
            ocr_runner=args.ocr_runner,
            ocr_python=args.ocr_python,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
        print(f"OCR results saved to {display_path(paths.ocr_results_path)}")
        return 0

    if args.command == "run-reorder":
        ensure_manifest_exists(paths)
        ensure_ocr_results_ready(paths)
        methods = resolve_requested_models(args.model)
        run_reorder_step(
            paths=paths,
            model_names=methods,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            overwrite=args.overwrite,
        )
        print(f"Raw reorder outputs saved under {display_path(paths.raw_root)}")
        return 0

    if args.command == "convert":
        ensure_gt_exists(paths)
        methods = discover_methods(paths.raw_root, args.method)
        convert_step(paths=paths, methods=methods)
        print(f"Converted predict files saved under {display_path(paths.predict_root)}")
        return 0

    if args.command == "eval":
        ensure_gt_exists(paths)
        methods = discover_methods(paths.predict_root, args.method)
        evaluate_step(paths=paths, methods=methods)
        print(f"Evaluation summary saved to {display_path(paths.summary_json_path)}")
        return 0

    if args.command == "run-all":
        build_gt_and_manifest(
            linear_source=args.linear_source,
            unlinear_source=args.unlinear_source,
            paths=paths,
        )
        run_ocr_step(
            paths=paths,
            ocr_runner=args.ocr_runner,
            ocr_python=args.ocr_python,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
        methods = resolve_requested_models(args.model)
        run_reorder_step(
            paths=paths,
            model_names=methods,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            overwrite=args.overwrite,
        )
        convert_step(paths=paths, methods=[BASELINE_METHOD_NAME, *methods])
        evaluate_step(paths=paths, methods=[BASELINE_METHOD_NAME, *methods])
        print(f"Finished full experiment run under {display_path(paths.root)}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def resolve_experiment_paths(output_dir: Optional[Path], run_name: str) -> ExperimentPaths:
    root = output_dir if output_dir is not None else (DEFAULT_OUTPUT_ROOT / run_name)
    root = root.resolve()
    return ExperimentPaths(
        root=root,
        manifest_path=root / "sample_manifest.json",
        gt_path=root / "gt" / "extracted_gt.json",
        gt_error_report_path=root / "gt" / "extraction_errors.json",
        ocr_results_path=root / "ocr" / "ocr_results.json",
        raw_root=root / "raw",
        predict_root=root / "predict",
        metric_root=root / "metric",
        summary_json_path=root / "summary.json",
        summary_csv_path=root / "summary.csv",
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BENCHMARK_DIR))
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def atomic_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def load_json_array(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {display_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{display_path(path)} must contain a JSON array.")
    return payload


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


def normalize_filename(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().replace("\\", "/")


def split_ocr_text_into_units(text: str) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        units.append({"unit_id": len(units) + 1, "text": line})
    return units


def clean_transcription_spaces(node: Any) -> int:
    updated_count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "transcription" and isinstance(value, str):
                cleaned = SPACE_PATTERN.sub("", value)
                if cleaned != value:
                    node[key] = cleaned
                    updated_count += 1
            else:
                updated_count += clean_transcription_spaces(value)
    elif isinstance(node, list):
        for item in node:
            updated_count += clean_transcription_spaces(item)
    return updated_count


def resolve_image_path(filename: str) -> Path:
    search_paths = (
        BENCHMARK_DIR / "Data_Annotation" / "annotation" / "images" / filename,
        BENCHMARK_DIR / "Data_Annotation" / "split_images" / "ConstructedResponse" / filename,
        BENCHMARK_DIR / "Data_Annotation" / "image_after_processed" / "ConstructedResponse" / filename,
    )
    for candidate in search_paths:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve image path for {filename!r}.")


def load_source_records(source_path: Path, subset: str) -> List[Dict[str, Any]]:
    records = load_json_array(source_path)
    normalized_records: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"{display_path(source_path)} record {record_index} must be a JSON object."
            )
        filename = normalize_filename(record.get("filename"))
        idx = normalize_idx(record.get("idx"))
        question_type = normalize_text(record.get("QuestionType"))
        if not filename:
            raise ValueError(f"{display_path(source_path)} record {record_index} is missing filename.")
        if idx is None:
            raise ValueError(
                f"{display_path(source_path)} record {record_index} has an invalid idx value."
            )
        if question_type != CONSTRUCTED_RESPONSE:
            raise ValueError(
                f"{display_path(source_path)} record {record_index} must be {CONSTRUCTED_RESPONSE!r}, "
                f"got {question_type!r}."
            )
        merged = dict(record)
        merged["subset"] = subset
        normalized_records.append(merged)
    return normalized_records


def build_sample_specs(
    linear_source: Path,
    unlinear_source: Path,
) -> Tuple[List[SampleSpec], List[Dict[str, Any]]]:
    source_records = [
        *load_source_records(linear_source, "linear"),
        *load_source_records(unlinear_source, "unlinear"),
    ]
    seen_filenames: set[str] = set()
    sample_specs: List[SampleSpec] = []
    for record in source_records:
        filename = normalize_filename(record["filename"])
        if filename in seen_filenames:
            raise ValueError(f"Duplicate sample filename detected: {filename}")
        seen_filenames.add(filename)
        sample_specs.append(
            SampleSpec(
                subset=normalize_text(record["subset"]),
                filename=filename,
                idx=int(record["idx"]),
                question_type=normalize_text(record["QuestionType"]),
                image_path=str(resolve_image_path(filename)),
            )
        )
    return sample_specs, source_records


def build_gt_and_manifest(
    *,
    linear_source: Path,
    unlinear_source: Path,
    paths: ExperimentPaths,
) -> None:
    sample_specs, source_records = build_sample_specs(linear_source, unlinear_source)
    gt_records, error_report = extract_gt_records(source_records)
    atomic_write_json(paths.manifest_path, [asdict(spec) for spec in sample_specs])
    atomic_write_json(paths.gt_path, gt_records)
    atomic_write_json(paths.gt_error_report_path, error_report)


def ensure_manifest_exists(paths: ExperimentPaths) -> None:
    if not paths.manifest_path.exists():
        raise FileNotFoundError(
            f"Sample manifest not found at {display_path(paths.manifest_path)}. Run build-gt first."
        )


def ensure_gt_exists(paths: ExperimentPaths) -> None:
    if not paths.gt_path.exists():
        raise FileNotFoundError(
            f"Experiment GT not found at {display_path(paths.gt_path)}. Run build-gt first."
        )


def ensure_ocr_results_ready(paths: ExperimentPaths) -> None:
    if not paths.ocr_results_path.exists():
        raise FileNotFoundError(
            f"OCR results not found at {display_path(paths.ocr_results_path)}. Run run-ocr first."
        )
    ocr_records = load_json_array(paths.ocr_results_path)
    manifest = load_sample_specs(paths.manifest_path)
    expected = {spec.filename for spec in manifest}
    success = {
        normalize_filename(record.get("filename"))
        for record in ocr_records
        if isinstance(record, dict) and record.get("ocr_status") == "success"
    }
    missing = sorted(expected - success)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"OCR results are incomplete. Missing success records for: {joined}"
        )


def load_sample_specs(path: Path) -> List[SampleSpec]:
    payload = load_json_array(path)
    sample_specs: List[SampleSpec] = []
    for item_index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {item_index} must be a JSON object.")
        idx = normalize_idx(item.get("idx"))
        if idx is None:
            raise ValueError(f"Manifest item {item_index} has invalid idx.")
        filename = normalize_filename(item.get("filename"))
        question_type = normalize_text(item.get("question_type") or item.get("QuestionType"))
        subset = normalize_text(item.get("subset"))
        image_path = normalize_text(item.get("image_path"))
        if not filename or not subset or not image_path or not question_type:
            raise ValueError(f"Manifest item {item_index} is missing required fields.")
        sample_specs.append(
            SampleSpec(
                subset=subset,
                filename=filename,
                idx=idx,
                question_type=question_type,
                image_path=image_path,
            )
        )
    return sample_specs


def contains_unk_marker(value: Any) -> bool:
    return isinstance(value, str) and "[UNK]" in value


def normalize_inline_prefix(prefix: str) -> str:
    text = prefix.strip()
    if not text:
        return ""
    text = QUESTION_TITLE_PREFIX_PATTERN.sub("", text, count=1).strip()
    while True:
        updated = LEADING_ENUM_PATTERN.sub("", text, count=1).strip()
        if updated == text:
            break
        text = updated
    while True:
        updated = TRAILING_ENUM_PATTERN.sub("", text, count=1).strip()
        if updated == text:
            break
        text = updated
    return text.strip()


def normalize_pre_question_line(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    text = QUESTION_TITLE_PREFIX_PATTERN.sub("", text, count=1).strip()
    semantic_prefix = ""
    prefix_match = re.match(
        "^((?:Solution|Proof|\\u89e3[:\\uFF1A]?|\\u8bc1\\u660e[:\\uFF1A]?))(.*)$",
        text,
        re.IGNORECASE,
    )
    if prefix_match is not None:
        semantic_prefix = prefix_match.group(1)
        text = prefix_match.group(2).strip()
    text = LEADING_ENUM_PATTERN.sub("", text, count=1).strip()
    return f"{semantic_prefix}{text}" if semantic_prefix else text


def is_leading_title_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if "【q" in text.lower() or "MATH:" in text:
        return False
    compact = SPACE_PATTERN.sub("", text)
    compact_lower = compact.lower()
    if ("constructedresponse" in compact_lower and "points" in compact_lower) or (
        "\u89e3\u7b54\u9898" in compact and "\u5206" in compact
    ):
        return True
    if re.fullmatch(r"\d+(?:[.]|[\u3001\u9898])?", compact):
        return True
    if re.fullmatch(
        r"\d+(?:[.]|[\u3001\u9898])?(?:[\(\uFF08][^()\uFF08\uFF09]{0,16}[\)\uFF09])?",
        compact,
    ):
        return True
    return False


def build_gt_output_record(
    record: Dict[str, Any],
    *,
    transcription: Any,
    formula_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "filename": normalize_filename(record.get("filename")),
        "idx": normalize_idx(record.get("idx")),
        "QuestionType": normalize_text(record.get("QuestionType")),
        "transcription": transcription,
        "formula_list": formula_list,
        "final_answer": normalize_text(record.get("final_answer")),
    }


def build_gt_error_item(record: Any, reason: str, details: str) -> Dict[str, Any]:
    filename = ""
    if isinstance(record, dict):
        filename = normalize_filename(record.get("filename"))
    return {
        "filename": filename,
        "reason": reason,
        "details": details,
        "record": record,
    }


def split_gt_transcription(transcription: str) -> List[Dict[str, Any]]:
    fragments: List[Tuple[str, str]] = []
    current_question_id: Optional[str] = None

    for raw_line in transcription.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matches = list(Q_MARKER_PATTERN.finditer(line))
        if not matches:
            if current_question_id is None:
                if is_leading_title_line(line):
                    continue
                current_question_id = "Q001"
                normalized_line = normalize_pre_question_line(line)
                if normalized_line:
                    fragments.append((current_question_id, normalized_line))
                continue
            fragments.append((current_question_id, line))
            continue

        leading_prefix = line[: matches[0].start()]
        prefix_for_first_tag = ""
        if current_question_id is None:
            if leading_prefix.strip() and not is_leading_title_line(leading_prefix):
                prefix_for_first_tag = normalize_inline_prefix(leading_prefix)
        else:
            normalized_prefix = normalize_inline_prefix(leading_prefix)
            if normalized_prefix:
                fragments.append((current_question_id, normalized_prefix))

        for match_index, match in enumerate(matches):
            current_question_id = f"Q{match.group(1)}"
            start = match.end()
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(line)
            segment = line[start:end].strip()
            if match_index == 0 and prefix_for_first_tag and segment:
                segment = f"{prefix_for_first_tag}{segment}"
            if segment:
                fragments.append((current_question_id, segment))

    return [
        {
            "seq": sequence,
            "question_id": question_id,
            "content": content,
        }
        for sequence, (question_id, content) in enumerate(fragments, start=1)
    ]


def extract_formula_list(
    transcription_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned_items: List[Dict[str, Any]] = []
    formula_list: List[Dict[str, Any]] = []
    formula_sequence = 1

    for item in transcription_items:
        content = normalize_text(item.get("content"))
        for match in FLEXIBLE_MATH_PATTERN.finditer(content):
            formula_list.append(
                {
                    "formula_seq": formula_sequence,
                    "seq": item["seq"],
                    "question_id": item["question_id"],
                    "formula": match.group(1),
                }
            )
            formula_sequence += 1
        cleaned_items.append(
            {
                **item,
                "content": FLEXIBLE_MATH_PATTERN.sub(r"\1", content),
            }
        )

    return cleaned_items, formula_list


def extract_gt_record(
    original_record: Any,
    cleaned_record: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    if not isinstance(cleaned_record, dict):
        return None, "invalid_record_type", "Record must be a JSON object."
    filename = normalize_filename(cleaned_record.get("filename"))
    if not filename:
        return None, "missing_filename", "Record filename must be a non-empty string."
    transcription = cleaned_record.get("transcription")
    if not isinstance(transcription, str):
        return None, "invalid_transcription", "Record transcription must be a string."

    question_type = normalize_text(cleaned_record.get("QuestionType"))
    if question_type != CONSTRUCTED_RESPONSE:
        return None, "invalid_question_type", f"Expected {CONSTRUCTED_RESPONSE!r}, got {question_type!r}."

    transcription_items = split_gt_transcription(transcription)
    if not transcription_items:
        if contains_unk_marker(original_record.get("transcription")) or contains_unk_marker(
            original_record.get("final_answer")
        ):
            return (
                build_gt_output_record(
                    cleaned_record,
                    transcription=normalize_text(original_record.get("transcription")),
                    formula_list=[],
                ),
                None,
                None,
            )
        return None, "no_extractable_segments", "No GT transcription segments were extracted."

    cleaned_items, formula_list = extract_formula_list(transcription_items)
    return (
        build_gt_output_record(
            cleaned_record,
            transcription=cleaned_items,
            formula_list=formula_list,
        ),
        None,
        None,
    )


def extract_gt_records(source_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    original_records = json.loads(json.dumps(source_records, ensure_ascii=False))
    cleaned_records = json.loads(json.dumps(source_records, ensure_ascii=False))
    clean_transcription_spaces(cleaned_records)

    gt_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for original_record, cleaned_record in zip(original_records, cleaned_records):
        output_record, reason, details = extract_gt_record(original_record, cleaned_record)
        if output_record is not None:
            gt_records.append(output_record)
            continue
        errors.append(
            build_gt_error_item(
                original_record,
                reason or "unknown_error",
                details or "Unknown GT extraction failure.",
            )
        )

    error_report = {
        "summary": {
            "input_records": len(source_records),
            "extracted_records": len(gt_records),
            "error_records": len(errors),
        },
        "errors": errors,
    }
    return gt_records, error_report


def run_ocr_step(
    *,
    paths: ExperimentPaths,
    ocr_runner: Path,
    ocr_python: Path,
    device: str,
    dtype: str,
    max_new_tokens: int,
    overwrite: bool,
) -> None:
    if not ocr_python.exists():
        raise FileNotFoundError(f"OCR Python not found: {ocr_python}")
    if not ocr_runner.exists():
        raise FileNotFoundError(f"OCR runner not found: {ocr_runner}")

    command = [
        str(ocr_python.resolve()),
        str(ocr_runner.resolve()),
        "--manifest",
        str(paths.manifest_path),
        "--output-json",
        str(paths.ocr_results_path),
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def load_results_index(path: Path, filename_key: str = "filename") -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records = load_json_array(path)
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        filename = normalize_filename(record.get(filename_key) or record.get("ImgReal"))
        if filename:
            index[filename] = record
    return index


def serialize_results_from_manifest(
    sample_specs: Sequence[SampleSpec],
    result_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [result_index[spec.filename] for spec in sample_specs if spec.filename in result_index]


def load_ocr_success_records(paths: ExperimentPaths) -> List[Dict[str, Any]]:
    ocr_records = load_json_array(paths.ocr_results_path)
    success_records = [
        record
        for record in ocr_records
        if isinstance(record, dict) and record.get("ocr_status") == "success"
    ]
    success_index = {normalize_filename(record.get("filename")): record for record in success_records}
    sample_specs = load_sample_specs(paths.manifest_path)
    missing = [spec.filename for spec in sample_specs if spec.filename not in success_index]
    if missing:
        raise RuntimeError(f"Missing OCR success records for: {', '.join(missing)}")
    return [success_index[spec.filename] for spec in sample_specs]


def build_ordered_transcription(ocr_units: Sequence[Dict[str, Any]], ordered_unit_ids: Sequence[int]) -> str:
    unit_lookup = {
        normalize_idx(unit.get("unit_id")): normalize_text(unit.get("text"))
        for unit in ocr_units
        if normalize_idx(unit.get("unit_id")) is not None
    }
    return "\n".join(unit_lookup[unit_id] for unit_id in ordered_unit_ids)


def build_raw_result_record(
    *,
    sample_record: Dict[str, Any],
    ordered_unit_ids: Sequence[int],
    raw_output: str,
    predict_status: str,
    error_message: str,
) -> Dict[str, Any]:
    ocr_units = sample_record.get("ocr_units") or []
    transcription = ""
    if predict_status == "success":
        transcription = build_ordered_transcription(ocr_units, ordered_unit_ids)
    return {
        "ImgReal": normalize_filename(sample_record.get("filename")),
        "filename": normalize_filename(sample_record.get("filename")),
        "subset": normalize_text(sample_record.get("subset")),
        "idx": normalize_idx(sample_record.get("idx")),
        "QuestionType": CONSTRUCTED_RESPONSE,
        "ordered_unit_ids": list(ordered_unit_ids),
        "transcription": transcription,
        "final_answer": "",
        "raw_output": raw_output,
        "predict_status": predict_status,
        "error_message": error_message,
    }


def build_baseline_results(sample_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_records: List[Dict[str, Any]] = []
    for sample_record in sample_records:
        unit_ids = [
            normalize_idx(unit.get("unit_id"))
            for unit in sample_record.get("ocr_units", [])
            if normalize_idx(unit.get("unit_id")) is not None
        ]
        raw_output = json.dumps({"ordered_unit_ids": unit_ids}, ensure_ascii=False)
        baseline_records.append(
            build_raw_result_record(
                sample_record=sample_record,
                ordered_unit_ids=unit_ids,
                raw_output=raw_output,
                predict_status="success",
                error_message="",
            )
        )
    return baseline_records


def build_reorder_prompt(sample_record: Dict[str, Any]) -> str:
    lines = [
        "You are restoring the reading order of OCR lines from a handwritten high-school math solution.",
        "Reorder by semantic solution flow, not by physical OCR order.",
        "Rules:",
        "- Use every unit exactly once.",
        "- Do not rewrite, merge, split, delete, or add content.",
        "- Preserve the original text of each OCR unit.",
        "- Return JSON only in the form {\"ordered_unit_ids\":[...]}",
        f"filename: {normalize_filename(sample_record.get('filename'))}",
        f"idx: {normalize_idx(sample_record.get('idx'))}",
        "ocr_units:",
    ]
    for unit in sample_record.get("ocr_units", []):
        unit_id = normalize_idx(unit.get("unit_id"))
        text = normalize_text(unit.get("text"))
        lines.append(f"{unit_id}: {text}")
    return "\n".join(lines)


def build_repair_prompt(
    *,
    sample_record: Dict[str, Any],
    invalid_response: str,
    validation_error: str,
) -> str:
    expected_ids = [
        normalize_idx(unit.get("unit_id"))
        for unit in sample_record.get("ocr_units", [])
        if normalize_idx(unit.get("unit_id")) is not None
    ]
    lines = [
        "Your previous response was invalid.",
        f"Validation error: {validation_error}",
        f"Expected unit ids: {expected_ids}",
        "Return JSON only in the form {\"ordered_unit_ids\":[...]}",
        "Do not add any explanation.",
        "Previous response:",
        invalid_response,
    ]
    return "\n".join(lines)


def call_text_model_with_retry(
    *,
    client: Any,
    config: Any,
    prompt_text: str,
    max_retries: int,
    max_tokens: int,
) -> str:
    from eval.predict import build_empty_response_error, extract_message_text

    request_kwargs: Dict[str, Any] = {
        "model": config.request_model_name or config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
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
                f"API error for {config.model_name} on attempt {attempt + 1}/{max_retries}: {exc}. "
                f"Retrying in {sleep_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state.")


def normalize_order_item(value: Any) -> Optional[int]:
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


def parse_ordered_unit_ids(raw_output: str, expected_unit_ids: Sequence[int]) -> Tuple[Optional[List[int]], str]:
    from eval.predict import try_parse_json

    parsed, error_message = try_parse_json(raw_output)
    if parsed is None:
        return None, error_message

    payload: Any
    if isinstance(parsed, dict):
        if "ordered_unit_ids" not in parsed:
            return None, "Parsed JSON is missing 'ordered_unit_ids'."
        payload = parsed.get("ordered_unit_ids")
    elif isinstance(parsed, list):
        payload = parsed
    else:
        return None, f"Parsed JSON root is {type(parsed).__name__}, expected object or array."

    if not isinstance(payload, list):
        return None, "'ordered_unit_ids' must be a JSON array."

    normalized_ids: List[int] = []
    for item in payload:
        normalized = normalize_order_item(item)
        if normalized is None:
            return None, f"Invalid ordered_unit_ids element: {item!r}"
        normalized_ids.append(normalized)

    expected_list = list(expected_unit_ids)
    if len(normalized_ids) != len(expected_list):
        return (
            None,
            f"ordered_unit_ids length {len(normalized_ids)} does not match expected {len(expected_list)}.",
        )

    expected_set = set(expected_list)
    actual_set = set(normalized_ids)
    if len(actual_set) != len(normalized_ids):
        duplicates = sorted(
            unit_id
            for unit_id, count in Counter(normalized_ids).items()
            if count > 1
        )
        return None, f"ordered_unit_ids contains duplicates: {duplicates}"
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        return None, f"ordered_unit_ids mismatch. missing={missing}, extra={extra}"
    return normalized_ids, ""


def run_reorder_step(
    *,
    paths: ExperimentPaths,
    model_names: Sequence[str],
    max_retries: int,
    max_tokens: int,
    request_timeout: float,
    overwrite: bool,
) -> None:
    sample_specs = load_sample_specs(paths.manifest_path)
    sample_records = load_ocr_success_records(paths)
    sample_record_index = {
        normalize_filename(record.get("filename")): record for record in sample_records
    }
    ordered_records = [sample_record_index[spec.filename] for spec in sample_specs]

    baseline_path = paths.raw_root / BASELINE_METHOD_NAME / "results.json"
    if overwrite or not baseline_path.exists():
        atomic_write_json(baseline_path, build_baseline_results(ordered_records))

    for model_name in model_names:
        run_model_reorder(
            sample_specs=sample_specs,
            sample_records=ordered_records,
            output_path=paths.raw_root / model_name / "results.json",
            model_name=model_name,
            max_retries=max_retries,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            overwrite=overwrite,
        )


def run_model_reorder(
    *,
    sample_specs: Sequence[SampleSpec],
    sample_records: Sequence[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    max_retries: int,
    max_tokens: int,
    request_timeout: float,
    overwrite: bool,
) -> None:
    from eval.predict import resolve_model_config

    existing_index = load_results_index(output_path)
    config, _resolved_api_key = resolve_model_config(model_name, None, None)
    client, http_client = build_text_client(config, request_timeout)

    try:
        for spec, sample_record in zip(sample_specs, sample_records):
            existing = existing_index.get(spec.filename)
            if (
                existing is not None
                and not overwrite
                and normalize_text(existing.get("predict_status")) == "success"
            ):
                continue

            expected_unit_ids = [
                normalize_idx(unit.get("unit_id"))
                for unit in sample_record.get("ocr_units", [])
                if normalize_idx(unit.get("unit_id")) is not None
            ]
            if not expected_unit_ids:
                existing_index[spec.filename] = build_raw_result_record(
                    sample_record=sample_record,
                    ordered_unit_ids=[],
                    raw_output="",
                    predict_status="parse_failed",
                    error_message="OCR produced no non-empty units.",
                )
                atomic_write_json(output_path, serialize_results_from_manifest(sample_specs, existing_index))
                continue

            try:
                initial_output = call_text_model_with_retry(
                    client=client,
                    config=config,
                    prompt_text=build_reorder_prompt(sample_record),
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                )
                ordered_unit_ids, validation_error = parse_ordered_unit_ids(
                    initial_output,
                    expected_unit_ids,
                )
                raw_output = initial_output
                if ordered_unit_ids is None:
                    repair_output = call_text_model_with_retry(
                        client=client,
                        config=config,
                        prompt_text=build_repair_prompt(
                            sample_record=sample_record,
                            invalid_response=initial_output,
                            validation_error=validation_error,
                        ),
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                    )
                    ordered_unit_ids, validation_error = parse_ordered_unit_ids(
                        repair_output,
                        expected_unit_ids,
                    )
                    raw_output = json.dumps(
                        {
                            "initial_response": initial_output,
                            "repair_response": repair_output,
                        },
                        ensure_ascii=False,
                    )

                if ordered_unit_ids is None:
                    record = build_raw_result_record(
                        sample_record=sample_record,
                        ordered_unit_ids=[],
                        raw_output=raw_output,
                        predict_status="parse_failed",
                        error_message=validation_error,
                    )
                else:
                    record = build_raw_result_record(
                        sample_record=sample_record,
                        ordered_unit_ids=ordered_unit_ids,
                        raw_output=raw_output,
                        predict_status="success",
                        error_message="",
                    )
            except Exception as exc:
                record = build_raw_result_record(
                    sample_record=sample_record,
                    ordered_unit_ids=[],
                    raw_output=f"ERROR: {normalize_text(exc)}",
                    predict_status="api_error",
                    error_message=normalize_text(exc),
                )

            existing_index[spec.filename] = record
            atomic_write_json(output_path, serialize_results_from_manifest(sample_specs, existing_index))
    finally:
        http_client.close()


def resolve_requested_models(model_names: Optional[Sequence[str]]) -> List[str]:
    model_registry = get_model_registry()
    if model_names is None:
        return list(model_registry)
    resolved: List[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        candidate = normalize_text(model_name).strip()
        if not candidate:
            continue
        if candidate not in model_registry:
            available = ", ".join(sorted(model_registry))
            raise ValueError(f"Unsupported model {candidate!r}. Available models: {available}")
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    if not resolved:
        raise ValueError("No valid model names were provided.")
    return resolved


def discover_methods(root: Path, requested_methods: Optional[Sequence[str]]) -> List[str]:
    if requested_methods is not None:
        return [normalize_text(name).strip() for name in requested_methods if normalize_text(name).strip()]
    methods: List[str] = []
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {display_path(root)}")
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if (child / "results.json").exists() or (child / "predict.json").exists():
            methods.append(child.name)
    if not methods:
        raise FileNotFoundError(f"No methods discovered under {display_path(root)}")
    return methods


def convert_step(*, paths: ExperimentPaths, methods: Sequence[str]) -> None:
    gt_records = load_json_array(paths.gt_path)
    sample_specs = load_sample_specs(paths.manifest_path)
    for method in methods:
        input_path = paths.raw_root / method / "results.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Raw results not found for {method}: {display_path(input_path)}")
        raw_records = load_json_array(input_path)
        converted_records, error_report = convert_results(raw_records, gt_records, 0.50)
        output_dir = paths.predict_root / method
        atomic_write_json(output_dir / "predict.json", converted_records)
        atomic_write_json(output_dir / "predict_conversion_errors.json", error_report)
        if len(converted_records) != len(sample_specs):
            print(
                f"Warning: {method} converted {len(converted_records)}/{len(sample_specs)} samples.",
                file=sys.stderr,
            )


def build_record_index(records: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        filename = normalize_filename(record.get("filename"))
        if filename:
            index[filename] = record
    return index


def summarize_subset_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [row.get("reading_order") for row in rows]
    aggregate = aggregate_reading_order_metrics(metrics)
    status_counts = Counter(
        row["reading_order"]["status"]
        for row in rows
        if isinstance(row.get("reading_order"), dict) and row["reading_order"].get("status")
    )
    converted_count = sum(1 for row in rows if row.get("converted"))
    return {
        "sample_count": len(rows),
        "converted_count": converted_count,
        "applicable_samples": aggregate["applicable_samples"],
        "ros_scored_samples": aggregate["ros_scored_samples"],
        "avg_mcr": aggregate["overall_mcr"],
        "avg_bcs": aggregate["overall_bcs"],
        "avg_sqa": aggregate["overall_sqa"],
        "avg_iqa": aggregate["overall_iqa"],
        "avg_ros": aggregate["overall_ros"],
        "ros_coverage": aggregate["ros_coverage"],
        "status_counts": dict(status_counts),
    }


def evaluate_step(*, paths: ExperimentPaths, methods: Sequence[str]) -> None:
    gt_records = load_json_array(paths.gt_path)
    gt_index = build_record_index(gt_records)
    sample_specs = load_sample_specs(paths.manifest_path)

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        predict_path = paths.predict_root / method / "predict.json"
        raw_path = paths.raw_root / method / "results.json"
        conversion_error_path = paths.predict_root / method / "predict_conversion_errors.json"
        if not predict_path.exists():
            raise FileNotFoundError(
                f"Converted predict.json not found for {method}: {display_path(predict_path)}"
            )
        predict_index = build_record_index(load_json_array(predict_path))
        raw_index = load_results_index(raw_path)
        error_report = (
            json.loads(conversion_error_path.read_text(encoding="utf-8"))
            if conversion_error_path.exists()
            else {"errors": []}
        )
        conversion_errors = group_conversion_errors(error_report.get("errors", []))

        sample_rows: List[Dict[str, Any]] = []
        for spec in sample_specs:
            gt_record = gt_index.get(spec.filename)
            predict_record = predict_index.get(spec.filename)
            raw_record = raw_index.get(spec.filename)
            reading_order = None
            if gt_record is not None and predict_record is not None:
                reading_order = compute_reading_order_metrics(gt_record, predict_record)
            sample_rows.append(
                {
                    "filename": spec.filename,
                    "subset": spec.subset,
                    "idx": spec.idx,
                    "converted": predict_record is not None,
                    "raw_predict_status": normalize_text(raw_record.get("predict_status")) if raw_record else "",
                    "conversion_errors": conversion_errors.get(spec.filename, []),
                    "reading_order": reading_order,
                }
            )

        overall_metrics = {
            "method": method,
            "subsets": {
                "linear": summarize_subset_rows([row for row in sample_rows if row["subset"] == "linear"]),
                "unlinear": summarize_subset_rows(
                    [row for row in sample_rows if row["subset"] == "unlinear"]
                ),
                "all": summarize_subset_rows(sample_rows),
            },
        }

        output_dir = paths.metric_root / method
        atomic_write_json(output_dir / "sample_metrics.json", sample_rows)
        atomic_write_json(output_dir / "overall_metrics.json", overall_metrics)

        for subset_name, subset_metrics in overall_metrics["subsets"].items():
            summary_rows.append(
                {
                    "method": method,
                    "subset": subset_name,
                    "sample_count": subset_metrics["sample_count"],
                    "converted_count": subset_metrics["converted_count"],
                    "avg_bcs": subset_metrics["avg_bcs"],
                    "avg_mcr": subset_metrics["avg_mcr"],
                    "avg_sqa": subset_metrics["avg_sqa"],
                    "avg_iqa": subset_metrics["avg_iqa"],
                    "avg_ros": subset_metrics["avg_ros"],
                    "ros_coverage": subset_metrics["ros_coverage"],
                }
            )

    atomic_write_json(paths.summary_json_path, summary_rows)
    atomic_write_csv(
        paths.summary_csv_path,
        summary_rows,
        (
            "method",
            "subset",
            "sample_count",
            "converted_count",
            "avg_bcs",
            "avg_mcr",
            "avg_sqa",
            "avg_iqa",
            "avg_ros",
            "ros_coverage",
        ),
    )


def group_conversion_errors(error_items: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in error_items:
        if not isinstance(item, dict):
            continue
        filename = normalize_filename(item.get("filename"))
        if not filename:
            continue
        grouped.setdefault(filename, []).append(item)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
