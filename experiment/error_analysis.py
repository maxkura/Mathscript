#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from openai import OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from eval.metric import FormulaParseError, compute_formula_metrics, compute_stem_metrics
    from eval.predict import MODEL_REGISTRY, ModelConfig
except ModuleNotFoundError:
    from metric import FormulaParseError, compute_formula_metrics, compute_stem_metrics
    from predict import MODEL_REGISTRY, ModelConfig


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GT_PATH = BASE_DIR / "data" / "GT" / "extracted_gt.json"
DEFAULT_PREDICT_ROOT = BASE_DIR / "output" / "predict"
DEFAULT_IMAGE_DIR = BASE_DIR / "Data_Annotation" / "annotation" / "images"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output" / "experiment" / "error_analysis"
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "error_analysis_prompt.txt"

TARGET_MODELS = ("gpt-4o", "kimi-k2.5", "qwen3-vl-plus")
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_BUCKET_SIZE = 10
DEFAULT_CALIBRATION_SIZE = 6
DEFAULT_RANDOM_SEED = 20260329
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 6
DEFAULT_MAX_TOKENS = 2500

BUCKET_QUOTAS = {
    "formula_high_risk": 10,
    "text_high_risk": 10,
    "mixed_medium_risk": 10,
    "high_confidence_control": 10,
}
BUCKET_DISPLAY_NAMES = {
    "formula_high_risk": "High formula-structure risk",
    "text_high_risk": "High text-order/noise risk",
    "mixed_medium_risk": "Mixed medium-risk samples",
    "high_confidence_control": "High-confidence control samples",
}
BUCKET_ORDER = list(BUCKET_QUOTAS)

LONG_TABLE_FIELDS = [
    "image_id",
    "model_name",
    "sample_bucket",
    "coarse_label_primary",
    "coarse_label_secondary",
    "formula_slot_id",
    "fine_label_primary",
    "fine_label_secondary",
    "auto_confidence",
    "human_final",
    "notes",
]

COARSE_LABELS = {
    "Reading-order/segmentation error",
    "Body-text recognition error",
    "Formula miss",
    "Formula false positive",
    "Formatting/parsing/conversion error",
    "Truncation/empty output/refusal",
    "Mixed error",
}
FINE_LABELS = {
    "Missed formula recognition",
    "Incorrect formula recognition",
    "Symbol confusion",
    "Spatial-relation error",
    "Scope error",
    "Operator semantics/precedence error",
    "Formula attached to the wrong step/line",
    "Source-image ambiguity/illegible handwriting",
}
DEFAULT_FORMULA_DENSE_THRESHOLD = 6
DEFAULT_HIGH_LOW_MATCH_THRESHOLD = 0.40
DEFAULT_MEDIUM_LOW_MATCH_THRESHOLD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, auto-label, and summarize OCR error-analysis experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build the sampling pool and skeleton files.")
    prepare_parser.add_argument("--gt", type=Path, default=DEFAULT_GT_PATH)
    prepare_parser.add_argument("--predict-root", type=Path, default=DEFAULT_PREDICT_ROOT)
    prepare_parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=None)
    prepare_parser.add_argument("--run-name", type=str, default="default")
    prepare_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    prepare_parser.add_argument("--bucket-size", type=int, default=DEFAULT_BUCKET_SIZE)
    prepare_parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    prepare_parser.add_argument("--models", nargs="+", default=list(TARGET_MODELS))

    autolabel_parser = subparsers.add_parser("autolabel", help="Call a reviewer model for sampled items.")
    autolabel_parser.add_argument("--output-dir", type=Path, default=None)
    autolabel_parser.add_argument("--run-name", type=str, default="default")
    autolabel_parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    autolabel_parser.add_argument("--reviewer-model", type=str, default="gpt-4o")
    autolabel_parser.add_argument("--base-url", type=str, default=None)
    autolabel_parser.add_argument("--api-key", type=str, default=None)
    autolabel_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    autolabel_parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    autolabel_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    autolabel_parser.add_argument("--limit", type=int, default=None)
    autolabel_parser.add_argument("--overwrite", action="store_true")
    autolabel_parser.add_argument("--dry-run", action="store_true")

    summarize_parser = subparsers.add_parser("summarize", help="Generate a markdown summary from the long table.")
    summarize_parser.add_argument("--output-dir", type=Path, default=None)
    summarize_parser.add_argument("--run-name", type=str, default="default")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.run_name)

    if args.command == "prepare":
        run_prepare(args, output_dir)
        return 0
    if args.command == "autolabel":
        run_autolabel(args, output_dir)
        return 0
    if args.command == "summarize":
        run_summarize(output_dir)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


def resolve_output_dir(output_dir: Optional[Path], run_name: str) -> Path:
    if output_dir is not None:
        return output_dir
    return DEFAULT_OUTPUT_ROOT / run_name


def run_prepare(args: argparse.Namespace, output_dir: Path) -> None:
    ensure_directory(output_dir)
    models = tuple(args.models)
    data_bundle = build_data_bundle(
        gt_path=args.gt,
        predict_root=args.predict_root,
        image_dir=args.image_dir,
        models=models,
    )

    shared_filenames = compute_shared_filenames(data_bundle["gt_index"], data_bundle["predict_indices"], models)
    shared_contexts = [build_shared_context(data_bundle, filename, models) for filename in shared_filenames]
    structured_formula_contexts, excluded_contexts = partition_contexts(shared_contexts, models)

    image_features, per_model_feature_rows = build_feature_sets(structured_formula_contexts, models)
    selected_buckets = select_bucketed_samples(
        image_features=image_features,
        sample_size=args.sample_size,
        bucket_size=args.bucket_size,
    )
    calibration_images = select_calibration_images(selected_buckets, args.calibration_size)
    long_rows, autolabel_requests = build_output_rows_and_requests(
        structured_formula_contexts=structured_formula_contexts,
        selected_buckets=selected_buckets,
        calibration_images=calibration_images,
        image_dir=args.image_dir,
        models=models,
    )

    shared_pool_payload = {
        "models": list(models),
        "counts": {
            "gt_total": len(data_bundle["gt_records"]),
            "gt_constructed": len(data_bundle["gt_index"]),
            "shared_constructed": len(shared_contexts),
            "structured_formula_pool": len(structured_formula_contexts),
            "excluded_passthrough_shared": len(excluded_contexts),
        },
        "shared_constructed_filenames": [ctx["image_id"] for ctx in shared_contexts],
        "structured_formula_filenames": [ctx["image_id"] for ctx in structured_formula_contexts],
        "excluded_passthrough_filenames": [ctx["image_id"] for ctx in excluded_contexts],
    }

    manifest_rows = build_sampling_manifest(selected_buckets, calibration_images, image_features)
    excluded_rows = build_excluded_rows(excluded_contexts, models)

    write_json(output_dir / "shared_pool.json", shared_pool_payload)
    write_csv(output_dir / "sample_features.csv", per_model_feature_rows)
    write_csv(output_dir / "sampling_manifest.csv", manifest_rows)
    write_csv(output_dir / "excluded_passthrough_samples.csv", excluded_rows)
    write_csv(output_dir / "error_analysis_long.csv", long_rows, fieldnames=LONG_TABLE_FIELDS)
    write_jsonl(output_dir / "autolabel_requests.jsonl", autolabel_requests)
    write_calibration_notes(output_dir / "calibration_notes.md", calibration_images, selected_buckets)

    print(f"Output directory: {output_dir}")
    print(f"Shared constructed pool: {len(shared_contexts)}")
    print(f"Structured formula pool: {len(structured_formula_contexts)}")
    print(f"Excluded passthrough shared samples: {len(excluded_contexts)}")
    print(f"Selected sampled images: {sum(len(items) for items in selected_buckets.values())}")
    print(f"Calibration images: {len(calibration_images)}")


def run_autolabel(args: argparse.Namespace, output_dir: Path) -> None:
    requests_path = output_dir / "autolabel_requests.jsonl"
    long_table_path = output_dir / "error_analysis_long.csv"
    if not requests_path.exists():
        raise FileNotFoundError(f"Autolabel request file not found: {requests_path}")
    if not long_table_path.exists():
        raise FileNotFoundError(f"Long-table file not found: {long_table_path}")

    requests = load_jsonl(requests_path)
    if args.limit is not None:
        requests = requests[: args.limit]

    long_rows = load_csv(long_table_path)
    long_row_index = build_long_row_index(long_rows)
    existing_results_path = output_dir / "autolabel_results.jsonl"
    existing_results = load_jsonl(existing_results_path) if existing_results_path.exists() else []
    existing_index = {
        autolabel_key(item["image_id"], item["model_name"]): item
        for item in existing_results
        if isinstance(item, dict) and "image_id" in item and "model_name" in item
    }

    prompt_text = load_text(args.prompt_file)
    config, resolved_api_key = resolve_model_config(
        args.reviewer_model,
        args.base_url,
        args.api_key,
    )
    client, http_client = (None, None)
    if not args.dry_run:
        client, http_client = build_openai_client(
            config,
            api_key=resolved_api_key,
            timeout=args.timeout,
        )

    appended_results: List[Dict[str, Any]] = []
    try:
        for request in requests:
            key = autolabel_key(request["image_id"], request["model_name"])
            if not args.overwrite and key in existing_index:
                continue

            if args.dry_run:
                parsed = build_dry_run_response(request)
                raw_output = json.dumps(parsed, ensure_ascii=False, indent=2)
            else:
                raw_output = call_reviewer_with_retry(
                    client=client,
                    config=config,
                    prompt_text=prompt_text,
                    request=request,
                    max_retries=args.max_retries,
                    max_tokens=args.max_tokens,
                )
                parsed = parse_autolabel_response(raw_output, request)

            result_item = {
                "image_id": request["image_id"],
                "model_name": request["model_name"],
                "sample_bucket": request["sample_bucket"],
                "reviewer_model": config.model_name,
                "status": "success",
                "raw_output": raw_output,
                "parsed": parsed,
            }
            appended_results.append(result_item)
            existing_index[key] = result_item
            merge_autolabel_result_into_rows(long_rows, long_row_index, request, parsed)
            write_jsonl(existing_results_path, list(existing_index.values()))
            write_csv(long_table_path, long_rows, fieldnames=LONG_TABLE_FIELDS)
    finally:
        if http_client is not None:
            http_client.close()

    print(f"Output directory: {output_dir}")
    print(f"Processed autolabel requests: {len(appended_results)}")
    print(f"Autolabel results file: {existing_results_path}")


def run_summarize(output_dir: Path) -> None:
    long_table_path = output_dir / "error_analysis_long.csv"
    manifest_path = output_dir / "sampling_manifest.csv"
    if not long_table_path.exists():
        raise FileNotFoundError(f"Long-table file not found: {long_table_path}")

    long_rows = load_csv(long_table_path)
    manifest_rows = load_csv(manifest_path) if manifest_path.exists() else []
    summary_text = build_summary_markdown(long_rows, manifest_rows)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Summary written to: {summary_path}")


def build_data_bundle(
    gt_path: Path,
    predict_root: Path,
    image_dir: Path,
    models: Sequence[str],
) -> Dict[str, Any]:
    gt_records = load_json(gt_path, label="GT")
    gt_index = build_filename_index(
        [record for record in gt_records if record.get("QuestionType") == "ConstructedResponse"],
        label="GT constructed",
        key_field="filename",
    )

    predict_indices: Dict[str, Dict[str, Dict[str, Any]]] = {}
    results_indices: Dict[str, Dict[str, Dict[str, Any]]] = {}
    conversion_error_indices: Dict[str, Dict[str, Dict[str, Any]]] = {}
    conversion_error_summaries: Dict[str, Dict[str, Any]] = {}

    for model in models:
        predict_records = load_json(predict_root / model / "predict.json", label=f"predict ({model})")
        predict_indices[model] = build_filename_index(
            [record for record in predict_records if record.get("QuestionType") == "ConstructedResponse"],
            label=f"predict ({model}) constructed",
            key_field="filename",
        )
        results_records = load_json(predict_root / model / "results.json", label=f"results ({model})")
        results_indices[model] = build_results_index(results_records, label=f"results ({model})")

        conversion_payload = load_json(
            predict_root / model / "predict_conversion_errors.json",
            label=f"predict_conversion_errors ({model})",
        )
        conversion_error_summaries[model] = dict(conversion_payload.get("summary") or {})
        conversion_error_indices[model] = build_conversion_error_index(conversion_payload.get("errors") or [])

    image_index = build_image_index(image_dir)
    return {
        "gt_records": gt_records,
        "gt_index": gt_index,
        "predict_indices": predict_indices,
        "results_indices": results_indices,
        "conversion_error_indices": conversion_error_indices,
        "conversion_error_summaries": conversion_error_summaries,
        "image_index": image_index,
    }


def load_json(path: Path, label: str) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} JSON parse failed for {path}: {exc}") from exc
    return data


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def build_filename_index(records: Sequence[Dict[str, Any]], label: str, key_field: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = record.get(key_field)
        if not isinstance(key, str) or not key.strip():
            continue
        normalized = key.strip()
        if normalized in index:
            duplicates.append(normalized)
            continue
        index[normalized] = record
    if duplicates:
        repeated = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate {label} {key_field} values: {repeated}")
    return index


def build_results_index(records: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = record.get("ImgReal")
        if not isinstance(key, str) or not key.strip():
            continue
        normalized = key.strip()
        if normalized in index:
            duplicates.append(normalized)
            continue
        index[normalized] = record
    if duplicates:
        repeated = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate {label} ImgReal values: {repeated}")
    return index


def build_conversion_error_index(errors: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in errors:
        if not isinstance(record, dict):
            continue
        key = (
            record.get("filename")
            or record.get("ImgReal")
            or record.get("image_id")
            or record.get("image")
        )
        if isinstance(key, str) and key.strip():
            index[key.strip()] = record
    return index


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    index: Dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file():
            index[path.name] = path
    return index


def compute_shared_filenames(
    gt_index: Dict[str, Dict[str, Any]],
    predict_indices: Dict[str, Dict[str, Dict[str, Any]]],
    models: Sequence[str],
) -> List[str]:
    shared = set(gt_index)
    for model in models:
        shared &= set(predict_indices[model])
    return sorted(shared)


def build_shared_context(data_bundle: Dict[str, Any], filename: str, models: Sequence[str]) -> Dict[str, Any]:
    gt_record = data_bundle["gt_index"][filename]
    predict_records = {model: data_bundle["predict_indices"][model][filename] for model in models}
    results_records = {model: data_bundle["results_indices"][model].get(filename) for model in models}
    conversion_records = {model: data_bundle["conversion_error_indices"][model].get(filename) for model in models}
    return {
        "image_id": filename,
        "gt_record": gt_record,
        "predict_records": predict_records,
        "results_records": results_records,
        "conversion_error_records": conversion_records,
        "image_path": data_bundle["image_index"].get(filename),
    }


def partition_contexts(
    contexts: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    structured_formula_contexts: List[Dict[str, Any]] = []
    excluded_contexts: List[Dict[str, Any]] = []
    for context in contexts:
        formula_related = bool(context["gt_record"].get("formula_list")) or any(
            bool(context["predict_records"][model].get("formula_list")) for model in models
        )
        all_structured = all(
            isinstance(context["predict_records"][model].get("transcription"), list) for model in models
        )
        if formula_related and all_structured:
            structured_formula_contexts.append(context)
            continue
        if any(
            context["predict_records"][model].get("conversion_status") == "unknown_gt_passthrough"
            for model in models
        ):
            excluded_contexts.append(context)
    return structured_formula_contexts, excluded_contexts


def build_feature_sets(
    structured_formula_contexts: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    image_features: Dict[str, Dict[str, Any]] = {}
    per_model_rows: List[Dict[str, Any]] = []

    for context in structured_formula_contexts:
        image_id = context["image_id"]
        gt_record = context["gt_record"]
        per_model: Dict[str, Dict[str, Any]] = {}
        extra_counts: Dict[str, int] = {}
        missing_counts: Dict[str, int] = {}
        rule_payloads: Dict[str, Dict[str, Any]] = {}
        formula_slot_count = 0

        for model in models:
            feature = build_model_feature(context, model)
            per_model[model] = feature
            extra_counts[model] = feature["extra_predict_formula_count"]
            missing_counts[model] = feature["missing_predict_formula_count"]
            formula_slot_count = max(formula_slot_count, feature["slot_count"])
            rule_payloads[model] = build_rule_suggestion(feature)
            per_model_rows.append(
                {
                    "image_id": image_id,
                    "model_name": model,
                    "gt_formula_count": feature["gt_formula_count"],
                    "pred_formula_count": feature["pred_formula_count"],
                    "formula_gap": feature["formula_gap"],
                    "extra_predict_formula_count": feature["extra_predict_formula_count"],
                    "missing_predict_formula_count": feature["missing_predict_formula_count"],
                    "pred_line_count": feature["pred_line_count"],
                    "matched_line_count": feature["matched_line_count"],
                    "unmatched_ratio": format_float(feature["unmatched_ratio"]),
                    "low_match_ratio": format_float(feature["low_match_ratio"]),
                    "header_noise_count": feature["header_noise_count"],
                    "order_inversion_count": feature["order_inversion_count"],
                    "mean_matched_score": format_float(feature["mean_matched_score"]),
                    "empty_prediction_flag": stringify_bool(feature["empty_prediction_flag"]),
                    "conversion_error_flag": stringify_bool(feature["conversion_error_flag"]),
                    "structured_record": stringify_bool(feature["structured_record"]),
                    "conversion_status": feature["conversion_status"],
                    "rule_primary_label": rule_payloads[model]["coarse_label_primary"],
                    "rule_secondary_labels": labels_to_cell(rule_payloads[model]["coarse_label_secondary"]),
                    "rule_risk_score": format_float(rule_payloads[model]["risk_score"]),
                    "formula_metrics_source": feature["formula_metrics_source"],
                    "formula_metrics_error": feature["formula_metrics_error"],
                }
            )

        image_features[image_id] = {
            "image_id": image_id,
            "per_model": per_model,
            "rule_payloads": rule_payloads,
            "gt_formula_count": len(gt_record.get("formula_list") or []),
            "max_formula_gap": max(feature["formula_gap"] for feature in per_model.values()),
            "max_formula_count": max(
                [len(gt_record.get("formula_list") or [])]
                + [feature["pred_formula_count"] for feature in per_model.values()]
            ),
            "max_unmatched_ratio": max(feature["unmatched_ratio"] for feature in per_model.values()),
            "max_low_match_ratio": max(feature["low_match_ratio"] for feature in per_model.values()),
            "max_header_noise_count": max(feature["header_noise_count"] for feature in per_model.values()),
            "max_order_inversion_count": max(feature["order_inversion_count"] for feature in per_model.values()),
            "mean_matched_score": average([feature["mean_matched_score"] for feature in per_model.values()]),
            "mean_rule_risk_score": average(
                [payload["risk_score"] for payload in rule_payloads.values()]
            ),
            "score_variance": spread([feature["mean_matched_score"] for feature in per_model.values()]),
            "formula_gap_variance": spread([feature["formula_gap"] for feature in per_model.values()]),
            "all_formula_gap_le_1": all(feature["formula_gap"] <= 1 for feature in per_model.values()),
            "all_unmatched_ratio_le_0_10": all(
                feature["unmatched_ratio"] <= 0.10 for feature in per_model.values()
            ),
            "all_order_inversion_zero": all(
                feature["order_inversion_count"] == 0 for feature in per_model.values()
            ),
            "any_empty_prediction": any(
                feature["empty_prediction_flag"] for feature in per_model.values()
            ),
            "any_conversion_error": any(
                feature["conversion_error_flag"] for feature in per_model.values()
            ),
            "any_extra_predict_formula": any(count > 0 for count in extra_counts.values()),
            "any_missing_predict_formula": any(count > 0 for count in missing_counts.values()),
            "formula_slot_count": formula_slot_count,
            "selection_reason": "",
            "selection_rank_values": {},
        }

    return image_features, per_model_rows


def build_model_feature(context: Dict[str, Any], model: str) -> Dict[str, Any]:
    gt_record = context["gt_record"]
    predict_record = context["predict_records"][model]
    result_record = context["results_records"][model] or {}
    conversion_error = context["conversion_error_records"][model]

    transcription = predict_record.get("transcription")
    structured = isinstance(transcription, list)
    lines = [item for item in transcription if isinstance(item, dict)] if structured else []
    pred_line_count = len(lines)
    matched_lines = [
        item
        for item in lines
        if item.get("match_status") == "matched"
    ]
    matched_line_count = len(matched_lines)
    unmatched_line_count = sum(
        1 for item in lines if item.get("match_status") != "matched"
    )
    unmatched_ratio = unmatched_line_count / pred_line_count if pred_line_count else 1.0
    low_match_ratio = (
        sum(1 for item in matched_lines if safe_float(item.get("match_score")) < 0.90) / matched_line_count
        if matched_line_count
        else 0.0
    )
    header_noise_count = compute_header_noise_count(lines)
    order_inversion_count = compute_order_inversion_count(matched_lines)

    gt_formula_count = len(gt_record.get("formula_list") or [])
    pred_formula_count = len(predict_record.get("formula_list") or [])

    empty_prediction_flag = is_empty_prediction(predict_record, result_record)
    conversion_error_flag = conversion_error is not None
    mean_matched_score = average([safe_float(item.get("match_score")) for item in matched_lines])

    slot_items, formula_metrics_source, formula_metrics_error = build_formula_slots(gt_record, predict_record)
    extra_predict_formula_count = sum(1 for slot in slot_items if slot.get("error") == "extra_predict_formula")
    missing_predict_formula_count = sum(
        1 for slot in slot_items if slot.get("error") == "missing_predict_formula"
    )
    stem_metrics = compute_stem_metrics(gt_record, predict_record) if structured else None

    return {
        "image_id": context["image_id"],
        "model_name": model,
        "structured_record": structured,
        "conversion_status": predict_record.get("conversion_status", ""),
        "pred_line_count": pred_line_count,
        "matched_line_count": matched_line_count,
        "unmatched_ratio": unmatched_ratio,
        "low_match_ratio": low_match_ratio,
        "header_noise_count": header_noise_count,
        "order_inversion_count": order_inversion_count,
        "gt_formula_count": gt_formula_count,
        "pred_formula_count": pred_formula_count,
        "formula_gap": abs(pred_formula_count - gt_formula_count),
        "extra_predict_formula_count": extra_predict_formula_count,
        "missing_predict_formula_count": missing_predict_formula_count,
        "empty_prediction_flag": empty_prediction_flag,
        "conversion_error_flag": conversion_error_flag,
        "mean_matched_score": mean_matched_score,
        "slot_count": len(slot_items),
        "slot_items": slot_items,
        "formula_metrics_source": formula_metrics_source,
        "formula_metrics_error": formula_metrics_error,
        "stem_acc": None if stem_metrics is None else stem_metrics.get("stem_acc"),
        "result_predict_status": result_record.get("predict_status", ""),
    }


def build_formula_slots(
    gt_record: Dict[str, Any],
    predict_record: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str, str]:
    try:
        metrics = compute_formula_metrics(gt_record, predict_record)
    except FormulaParseError as exc:
        return fallback_formula_slots(gt_record, predict_record), "fallback", str(exc)
    if metrics is None:
        return [], "metric", ""
    return list(metrics.get("slots") or []), "metric", ""


def fallback_formula_slots(gt_record: Dict[str, Any], predict_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    gt_groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in normalize_gt_formula_items(gt_record.get("formula_list")):
        gt_groups[(item.get("question_id"), item["seq"])].append(item)

    predict_groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    unmatched_predict: List[Dict[str, Any]] = []
    for item in normalize_predict_formula_items(predict_record.get("formula_list")):
        if item.get("question_id") is None or item.get("gt_seq") is None:
            unmatched_predict.append(item)
            continue
        predict_groups[(item.get("question_id"), item["gt_seq"])].append(item)

    for formulas in gt_groups.values():
        formulas.sort(key=lambda entry: entry["formula_seq"])
    for formulas in predict_groups.values():
        formulas.sort(key=lambda entry: entry["formula_seq"])
    unmatched_predict.sort(key=lambda entry: entry["formula_seq"])

    slot_items: List[Dict[str, Any]] = []
    all_keys = sorted(set(gt_groups) | set(predict_groups), key=lambda item: (str(item[0]), item[1]))
    for group_key in all_keys:
        gt_group = gt_groups.get(group_key, [])
        predict_group = predict_groups.get(group_key, [])
        slot_count = max(len(gt_group), len(predict_group))
        for slot_index in range(slot_count):
            gt_formula = gt_group[slot_index] if slot_index < len(gt_group) else None
            predict_formula = predict_group[slot_index] if slot_index < len(predict_group) else None

            question_id = (
                gt_formula.get("question_id")
                if gt_formula is not None
                else predict_formula.get("question_id")
            )
            gt_seq = (
                gt_formula.get("seq")
                if gt_formula is not None
                else predict_formula.get("gt_seq")
            )

            if gt_formula is None:
                slot_items.append(
                    {
                        "question_id": question_id,
                        "gt_seq": gt_seq,
                        "gt_formula_seq": None,
                        "predict_formula_seq": predict_formula["formula_seq"],
                        "gt_formula": None,
                        "predict_formula": predict_formula.get("formula"),
                        "slt_teds": None,
                        "opt_teds": None,
                        "error": "extra_predict_formula",
                    }
                )
                continue

            if predict_formula is None:
                slot_items.append(
                    {
                        "question_id": question_id,
                        "gt_seq": gt_seq,
                        "gt_formula_seq": gt_formula["formula_seq"],
                        "predict_formula_seq": None,
                        "gt_formula": gt_formula.get("formula"),
                        "predict_formula": None,
                        "slt_teds": None,
                        "opt_teds": None,
                        "error": "missing_predict_formula",
                    }
                )
                continue

            slot_items.append(
                {
                    "question_id": question_id,
                    "gt_seq": gt_seq,
                    "gt_formula_seq": gt_formula["formula_seq"],
                    "predict_formula_seq": predict_formula["formula_seq"],
                    "gt_formula": gt_formula.get("formula"),
                    "predict_formula": predict_formula.get("formula"),
                    "slt_teds": None,
                    "opt_teds": None,
                    "error": "formula_metrics_unavailable",
                }
            )

    for predict_formula in unmatched_predict:
        slot_items.append(
            {
                "question_id": predict_formula.get("question_id"),
                "gt_seq": predict_formula.get("gt_seq"),
                "gt_formula_seq": None,
                "predict_formula_seq": predict_formula["formula_seq"],
                "gt_formula": None,
                "predict_formula": predict_formula.get("formula"),
                "slt_teds": None,
                "opt_teds": None,
                "error": "extra_predict_formula",
            }
        )

    return slot_items


def normalize_gt_formula_items(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict):
            continue
        formula_seq = safe_int(item.get("formula_seq"))
        seq = safe_int(item.get("seq"))
        if formula_seq is None or seq is None:
            continue
        normalized.append(
            {
                "formula_seq": formula_seq,
                "seq": seq,
                "question_id": item.get("question_id"),
                "formula": item.get("formula"),
            }
        )
    return normalized


def normalize_predict_formula_items(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict):
            continue
        formula_seq = safe_int(item.get("formula_seq"))
        seq = safe_int(item.get("seq"))
        if formula_seq is None or seq is None:
            continue
        normalized.append(
            {
                "formula_seq": formula_seq,
                "seq": seq,
                "question_id": item.get("question_id"),
                "gt_seq": safe_int(item.get("gt_seq")),
                "formula": item.get("formula"),
            }
        )
    return normalized


def compute_header_noise_count(lines: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for item in lines:
        if item.get("question_id") is None and item.get("match_status") == "unmatched":
            count += 1
            continue
        break
    return count


def compute_order_inversion_count(matched_lines: Sequence[Dict[str, Any]]) -> int:
    count = 0
    previous_gt_seq: Optional[int] = None
    for item in matched_lines:
        gt_seq = safe_int(item.get("gt_seq"))
        if gt_seq is None:
            continue
        if previous_gt_seq is not None and gt_seq < previous_gt_seq:
            count += 1
        previous_gt_seq = gt_seq
    return count


def is_empty_prediction(predict_record: Dict[str, Any], result_record: Dict[str, Any]) -> bool:
    transcription = predict_record.get("transcription")
    if isinstance(transcription, list):
        has_text = any(str(item.get("content", "")).strip() for item in transcription if isinstance(item, dict))
    else:
        has_text = bool(str(transcription or "").strip())
    has_formula = bool(predict_record.get("formula_list"))
    has_final_answer = bool(str(predict_record.get("final_answer") or "").strip())
    if result_record.get("predict_status") not in ("", "success"):
        return True
    return not (has_text or has_formula or has_final_answer)


def build_rule_suggestion(feature: Dict[str, Any]) -> Dict[str, Any]:
    triggers: List[str] = []
    if feature["conversion_error_flag"]:
        triggers.append("Formatting/parsing/conversion error")
    if feature["empty_prediction_flag"]:
        triggers.append("Truncation/empty output/refusal")
    if (
        feature["unmatched_ratio"] >= 0.30
        or feature["order_inversion_count"] >= 2
        or feature["header_noise_count"] >= 2
    ):
        triggers.append("Reading-order/segmentation error")
    if feature["gt_formula_count"] > feature["pred_formula_count"]:
        triggers.append("Formula miss")
    if feature["extra_predict_formula_count"] > 0 and feature["pred_formula_count"] >= feature["gt_formula_count"]:
        triggers.append("Formula false positive")
    if feature["low_match_ratio"] >= DEFAULT_MEDIUM_LOW_MATCH_THRESHOLD:
        triggers.append("Body-text recognition error")

    unique_triggers = deduplicate_preserve_order(triggers)
    if len(unique_triggers) >= 3:
        primary = "Mixed error"
        secondary = unique_triggers[:2]
    elif unique_triggers:
        primary = unique_triggers[0]
        secondary = unique_triggers[1:2]
    else:
        primary = "Body-text recognition error"
        secondary = []

    risk_score = min(
        1.0,
        (
            min(feature["unmatched_ratio"], 1.0) * 0.35
            + min(feature["low_match_ratio"], 1.0) * 0.20
            + min(feature["formula_gap"], 3) / 3 * 0.20
            + min(feature["order_inversion_count"], 3) / 3 * 0.10
            + min(feature["header_noise_count"], 3) / 3 * 0.05
            + (0.05 if feature["empty_prediction_flag"] else 0.0)
            + (0.05 if feature["conversion_error_flag"] else 0.0)
        ),
    )
    return {
        "coarse_label_primary": primary,
        "coarse_label_secondary": secondary,
        "risk_score": risk_score,
        "must_review": bool(
            feature["empty_prediction_flag"]
            or feature["conversion_error_flag"]
            or len(unique_triggers) >= 2
            or feature["extra_predict_formula_count"] > 0
        ),
    }


def select_bucketed_samples(
    image_features: Dict[str, Dict[str, Any]],
    sample_size: int,
    bucket_size: int,
) -> Dict[str, List[str]]:
    if sample_size != bucket_size * len(BUCKET_ORDER):
        raise ValueError("sample-size must equal bucket-size multiplied by 4.")

    candidate_lists = {
        "formula_high_risk": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_formula_high_risk(item)
            ],
            image_features,
            key_builder=rank_formula_high_risk,
        ),
        "text_high_risk": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_text_high_risk(item)
            ],
            image_features,
            key_builder=rank_text_high_risk,
        ),
        "mixed_medium_risk": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_mixed_medium_risk(item)
            ],
            image_features,
            key_builder=rank_mixed_medium_risk,
        ),
        "high_confidence_control": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_high_confidence_control(item)
            ],
            image_features,
            key_builder=rank_high_confidence_control,
        ),
        "high_confidence_control_relaxed": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_relaxed_high_confidence(item)
            ],
            image_features,
            key_builder=rank_relaxed_high_confidence,
        ),
        "high_confidence_control_relaxed_secondary": sort_candidates(
            [
                image_id
                for image_id, item in image_features.items()
                if qualifies_relaxed_high_confidence_secondary(item)
            ],
            image_features,
            key_builder=rank_relaxed_high_confidence_secondary,
        ),
    }

    selected: Dict[str, List[str]] = {bucket: [] for bucket in BUCKET_ORDER}
    used: set[str] = set()

    for bucket in ("formula_high_risk", "text_high_risk", "mixed_medium_risk"):
        take_candidates(
            selected=selected[bucket],
            candidates=candidate_lists[bucket],
            used=used,
            quota=bucket_size,
        )

    take_candidates(
        selected=selected["high_confidence_control"],
        candidates=candidate_lists["high_confidence_control"],
        used=used,
        quota=bucket_size,
    )
    if len(selected["high_confidence_control"]) < bucket_size:
        take_candidates(
            selected=selected["high_confidence_control"],
            candidates=candidate_lists["high_confidence_control_relaxed"],
            used=used,
            quota=bucket_size,
        )
    if len(selected["high_confidence_control"]) < bucket_size:
        take_candidates(
            selected=selected["high_confidence_control"],
            candidates=candidate_lists["high_confidence_control_relaxed_secondary"],
            used=used,
            quota=bucket_size,
        )

    if len(selected["formula_high_risk"]) < bucket_size:
        take_candidates(
            selected=selected["formula_high_risk"],
            candidates=candidate_lists["mixed_medium_risk"],
            used=used,
            quota=bucket_size,
        )
    if len(selected["text_high_risk"]) < bucket_size:
        take_candidates(
            selected=selected["text_high_risk"],
            candidates=candidate_lists["mixed_medium_risk"],
            used=used,
            quota=bucket_size,
        )
    if len(selected["mixed_medium_risk"]) < bucket_size:
        take_candidates(
            selected=selected["mixed_medium_risk"],
            candidates=candidate_lists["formula_high_risk"] + candidate_lists["text_high_risk"],
            used=used,
            quota=bucket_size,
        )

    if len(selected["high_confidence_control"]) < bucket_size:
        raise ValueError("Not enough high-confidence-like samples to fill bucket D.")

    for bucket in BUCKET_ORDER:
        if len(selected[bucket]) != bucket_size:
            raise ValueError(f"Bucket {bucket} ended with {len(selected[bucket])} samples, expected {bucket_size}.")
        for rank, image_id in enumerate(selected[bucket], start=1):
            image_features[image_id]["selection_reason"] = BUCKET_DISPLAY_NAMES[bucket]
            image_features[image_id]["selection_rank_values"] = {"bucket": bucket, "rank": rank}

    return selected


def qualifies_formula_high_risk(item: Dict[str, Any]) -> bool:
    gt_formula_count = item["gt_formula_count"]
    per_model = list(item["per_model"].values())
    return (
        any(gt_formula_count > 0 and feature["pred_formula_count"] == 0 for feature in per_model)
        or any(feature["formula_gap"] >= 2 for feature in per_model)
        or any(feature["pred_formula_count"] > gt_formula_count for feature in per_model)
        or item["max_formula_count"] >= DEFAULT_FORMULA_DENSE_THRESHOLD
    )


def rank_formula_high_risk(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -item["max_formula_gap"],
        -item["max_formula_count"],
        item["image_id"],
    )


def qualifies_text_high_risk(item: Dict[str, Any]) -> bool:
    per_model = list(item["per_model"].values())
    return (
        any(feature["unmatched_ratio"] >= 0.30 for feature in per_model)
        or any(feature["header_noise_count"] >= 2 for feature in per_model)
        or any(feature["order_inversion_count"] >= 2 for feature in per_model)
        or any(feature["low_match_ratio"] >= DEFAULT_HIGH_LOW_MATCH_THRESHOLD for feature in per_model)
    )


def rank_text_high_risk(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -item["max_unmatched_ratio"],
        -item["max_order_inversion_count"],
        -item["max_header_noise_count"],
        item["image_id"],
    )


def qualifies_mixed_medium_risk(item: Dict[str, Any]) -> bool:
    if qualifies_formula_high_risk(item) or qualifies_text_high_risk(item):
        return False
    per_model = list(item["per_model"].values())
    formula_risk = any(feature["formula_gap"] == 1 for feature in per_model)
    text_risk = any(
        0.10 <= feature["unmatched_ratio"] < 0.30
        or DEFAULT_MEDIUM_LOW_MATCH_THRESHOLD <= feature["low_match_ratio"] < DEFAULT_HIGH_LOW_MATCH_THRESHOLD
        for feature in per_model
    )
    return formula_risk and text_risk


def rank_mixed_medium_risk(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -int(any(feature["formula_gap"] == 1 for feature in item["per_model"].values())),
        -int(
            any(
                0.10 <= feature["unmatched_ratio"] < 0.30
                or DEFAULT_MEDIUM_LOW_MATCH_THRESHOLD
                <= feature["low_match_ratio"]
                < DEFAULT_HIGH_LOW_MATCH_THRESHOLD
                for feature in item["per_model"].values()
            )
        ),
        -item["formula_gap_variance"],
        item["image_id"],
    )


def qualifies_high_confidence_control(item: Dict[str, Any]) -> bool:
    return (
        item["all_unmatched_ratio_le_0_10"]
        and item["all_order_inversion_zero"]
        and item["all_formula_gap_le_1"]
        and not item["any_empty_prediction"]
        and not item["any_conversion_error"]
    )


def rank_high_confidence_control(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -item["mean_matched_score"],
        item["score_variance"],
        item["image_id"],
    )


def qualifies_relaxed_high_confidence(item: Dict[str, Any]) -> bool:
    return (
        not item["any_empty_prediction"]
        and not item["any_conversion_error"]
        and item["all_order_inversion_zero"]
        and item["all_formula_gap_le_1"]
    )


def rank_relaxed_high_confidence(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item["max_unmatched_ratio"],
        -item["mean_matched_score"],
        item["score_variance"],
        item["image_id"],
    )


def qualifies_relaxed_high_confidence_secondary(item: Dict[str, Any]) -> bool:
    return (
        not item["any_empty_prediction"]
        and not item["any_conversion_error"]
        and item["max_formula_gap"] <= 1
        and item["max_order_inversion_count"] <= 1
    )


def rank_relaxed_high_confidence_secondary(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item["max_order_inversion_count"],
        item["max_formula_gap"],
        item["max_unmatched_ratio"],
        -item["mean_matched_score"],
        item["image_id"],
    )


def sort_candidates(
    candidates: Sequence[str],
    image_features: Dict[str, Dict[str, Any]],
    key_builder,
) -> List[str]:
    return sorted(candidates, key=lambda image_id: key_builder(image_features[image_id]))


def take_candidates(selected: List[str], candidates: Sequence[str], used: set[str], quota: int) -> None:
    for image_id in candidates:
        if len(selected) >= quota:
            return
        if image_id in used:
            continue
        selected.append(image_id)
        used.add(image_id)


def select_calibration_images(selected_buckets: Dict[str, List[str]], calibration_size: int) -> List[str]:
    preferred = [
        ("formula_high_risk", 2),
        ("text_high_risk", 2),
        ("mixed_medium_risk", 1),
        ("high_confidence_control", 1),
    ]
    selected: List[str] = []
    for bucket, count in preferred:
        selected.extend(selected_buckets[bucket][:count])
    if len(selected) != calibration_size:
        raise ValueError(f"Calibration subset expected {calibration_size} images, got {len(selected)}.")
    return selected


def build_sampling_manifest(
    selected_buckets: Dict[str, List[str]],
    calibration_images: Sequence[str],
    image_features: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        for rank, image_id in enumerate(selected_buckets[bucket], start=1):
            feature = image_features[image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "sample_bucket": bucket,
                    "bucket_display_name": BUCKET_DISPLAY_NAMES[bucket],
                    "bucket_rank": rank,
                    "in_calibration_set": stringify_bool(image_id in calibration_images),
                    "selection_reason": feature["selection_reason"],
                    "max_formula_gap": feature["max_formula_gap"],
                    "max_formula_count": feature["max_formula_count"],
                    "max_unmatched_ratio": format_float(feature["max_unmatched_ratio"]),
                    "max_low_match_ratio": format_float(feature["max_low_match_ratio"]),
                    "max_header_noise_count": feature["max_header_noise_count"],
                    "max_order_inversion_count": feature["max_order_inversion_count"],
                    "mean_matched_score": format_float(feature["mean_matched_score"]),
                }
            )
    return rows


def build_excluded_rows(
    excluded_contexts: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for context in excluded_contexts:
        row: Dict[str, Any] = {
            "image_id": context["image_id"],
            "exclusion_reason": "unknown_gt_passthrough",
            "gt_formula_count": len(context["gt_record"].get("formula_list") or []),
        }
        for model in models:
            predict_record = context["predict_records"][model]
            result_record = context["results_records"][model] or {}
            row[f"{model}_transcription_type"] = type(predict_record.get("transcription")).__name__
            row[f"{model}_conversion_status"] = predict_record.get("conversion_status", "")
            row[f"{model}_predict_status"] = result_record.get("predict_status", "")
        rows.append(row)
    return rows


def build_output_rows_and_requests(
    structured_formula_contexts: Sequence[Dict[str, Any]],
    selected_buckets: Dict[str, List[str]],
    calibration_images: Sequence[str],
    image_dir: Path,
    models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_lookup = {
        image_id: bucket
        for bucket, image_ids in selected_buckets.items()
        for image_id in image_ids
    }
    calibration_lookup = set(calibration_images)
    context_lookup = {context["image_id"]: context for context in structured_formula_contexts}

    long_rows: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    for image_id in [item for bucket in BUCKET_ORDER for item in selected_buckets[bucket]]:
        context = context_lookup[image_id]
        bucket = selected_lookup[image_id]
        gt_record = context["gt_record"]
        for model in models:
            feature = build_model_feature(context, model)
            rule_payload = build_rule_suggestion(feature)
            coarse_row = empty_long_row(
                image_id=image_id,
                model_name=model,
                sample_bucket=bucket,
            )
            long_rows.append(coarse_row)

            candidate_slots: List[Dict[str, Any]] = []
            for slot in feature["slot_items"]:
                slot_id = build_formula_slot_id(image_id, slot)
                candidate_slots.append(
                    {
                        "formula_slot_id": slot_id,
                        "question_id": normalize_optional_str(slot.get("question_id")),
                        "line_seq": slot.get("gt_seq"),
                        "gt_formula_seq": slot.get("gt_formula_seq"),
                        "predict_formula_seq": slot.get("predict_formula_seq"),
                        "gt_formula": slot.get("gt_formula"),
                        "predict_formula": slot.get("predict_formula"),
                        "slot_error": slot.get("error"),
                    }
                )
                long_rows.append(
                    empty_long_row(
                        image_id=image_id,
                        model_name=model,
                        sample_bucket=bucket,
                        formula_slot_id=slot_id,
                    )
                )

            request = {
                "image_id": image_id,
                "model_name": model,
                "sample_bucket": bucket,
                "in_calibration_set": image_id in calibration_lookup,
                "image_path": str((context["image_path"] or (image_dir / image_id)).resolve()),
                "gt_record": {
                    "filename": gt_record.get("filename"),
                    "transcription": gt_record.get("transcription"),
                    "formula_list": gt_record.get("formula_list"),
                    "final_answer": gt_record.get("final_answer"),
                },
                "predict_record": {
                    "filename": context["predict_records"][model].get("filename"),
                    "transcription": context["predict_records"][model].get("transcription"),
                    "formula_list": context["predict_records"][model].get("formula_list"),
                    "final_answer": context["predict_records"][model].get("final_answer"),
                },
                "rule_features": {
                    "pred_line_count": feature["pred_line_count"],
                    "matched_line_count": feature["matched_line_count"],
                    "unmatched_ratio": round(feature["unmatched_ratio"], 6),
                    "low_match_ratio": round(feature["low_match_ratio"], 6),
                    "header_noise_count": feature["header_noise_count"],
                    "order_inversion_count": feature["order_inversion_count"],
                    "gt_formula_count": feature["gt_formula_count"],
                    "pred_formula_count": feature["pred_formula_count"],
                    "formula_gap": feature["formula_gap"],
                    "empty_prediction_flag": feature["empty_prediction_flag"],
                    "conversion_error_flag": feature["conversion_error_flag"],
                    "extra_predict_formula_count": feature["extra_predict_formula_count"],
                    "missing_predict_formula_count": feature["missing_predict_formula_count"],
                    "rule_primary_label": rule_payload["coarse_label_primary"],
                    "rule_secondary_labels": rule_payload["coarse_label_secondary"],
                    "rule_risk_score": round(rule_payload["risk_score"], 6),
                },
                "candidate_formula_slots": candidate_slots,
            }
            requests.append(request)

    return long_rows, requests


def empty_long_row(
    image_id: str,
    model_name: str,
    sample_bucket: str,
    formula_slot_id: str = "",
) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "model_name": model_name,
        "sample_bucket": sample_bucket,
        "coarse_label_primary": "",
        "coarse_label_secondary": "",
        "formula_slot_id": formula_slot_id,
        "fine_label_primary": "",
        "fine_label_secondary": "",
        "auto_confidence": "",
        "human_final": "",
        "notes": "",
    }


def build_formula_slot_id(image_id: str, slot: Dict[str, Any]) -> str:
    question_id = normalize_optional_str(slot.get("question_id"))
    line_seq = normalize_optional_int(slot.get("gt_seq"))
    if slot.get("gt_formula_seq") is not None:
        return f"{image_id}::{question_id}::{line_seq}::{slot['gt_formula_seq']}"
    return (
        f"{image_id}::pred_extra::{question_id}::{line_seq}::"
        f"{normalize_optional_int(slot.get('predict_formula_seq'))}"
    )


def build_openai_client(
    config: ModelConfig,
    *,
    api_key: str,
    timeout: int,
) -> Tuple[OpenAI, httpx.Client]:
    http_client = httpx.Client(timeout=timeout)
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        http_client=http_client,
    )
    return client, http_client


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


def call_reviewer_with_retry(
    client: OpenAI,
    config: ModelConfig,
    prompt_text: str,
    request: Dict[str, Any],
    max_retries: int,
    max_tokens: int,
) -> str:
    image_path = Path(request["image_path"])
    image_data_url = image_to_data_url(image_path)
    request_payload = {
        "image_id": request["image_id"],
        "model_name": request["model_name"],
        "sample_bucket": request["sample_bucket"],
        "gt_record": request["gt_record"],
        "predict_record": request["predict_record"],
        "rule_features": request["rule_features"],
        "candidate_formula_slots": request["candidate_formula_slots"],
    }
    request_kwargs: Dict[str, Any] = {
        "model": config.request_model_name or config.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "text",
                        "text": "Judge strictly from the input JSON below and return a JSON object only:\n"
                        + json.dumps(request_payload, ensure_ascii=False, indent=2),
                    },
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
            raw_text = extract_message_text(response.choices[0].message.content)
            if raw_text.strip():
                return raw_text
            raise RuntimeError("Model returned an empty final response.")
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable retry state.")


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or ("text" in item and item.get("type") in (None, "")):
                    parts.append(str(item.get("text", "")))
                continue
            if getattr(item, "type", None) == "text":
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


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
        for block_index, block in enumerate(snippet.split("```"), start=1):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate), ""
            except Exception as exc:
                reasons.append(format_json_error(f"Code block {block_index} JSON decode failed", exc))

    start_object = snippet.find("{")
    end_object = snippet.rfind("}")
    if start_object != -1 and end_object != -1 and start_object < end_object:
        candidate = snippet[start_object : end_object + 1]
        try:
            return json.loads(candidate), ""
        except Exception as exc:
            reasons.append(format_json_error("Extracted object JSON decode failed", exc))
    else:
        reasons.append("No JSON object delimiters were found in the model output.")

    return None, " | ".join(reasons)


def parse_autolabel_response(raw_output: str, request: Dict[str, Any]) -> Dict[str, Any]:
    parsed, error_message = try_parse_json(raw_output)
    if parsed is None:
        raise ValueError(error_message)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON root is {type(parsed).__name__}, expected an object.")

    primary = parsed.get("coarse_label_primary")
    if primary not in COARSE_LABELS:
        raise ValueError(f"Invalid coarse_label_primary: {primary!r}")

    secondary = normalize_label_list(parsed.get("coarse_label_secondary"))
    if len(secondary) > 2:
        raise ValueError("coarse_label_secondary must contain at most 2 labels.")
    for label in secondary:
        if label not in COARSE_LABELS:
            raise ValueError(f"Invalid coarse secondary label: {label!r}")

    auto_confidence = safe_float(parsed.get("auto_confidence"))
    if auto_confidence is None or not (0.0 <= auto_confidence <= 1.0):
        raise ValueError(f"auto_confidence must be a float in [0, 1], got {parsed.get('auto_confidence')!r}")

    candidate_lookup = {
        slot["formula_slot_id"]
        for slot in request.get("candidate_formula_slots") or []
        if isinstance(slot, dict) and slot.get("formula_slot_id")
    }

    fine_candidates = parsed.get("fine_candidates") or []
    if not isinstance(fine_candidates, list):
        raise ValueError("fine_candidates must be a list.")
    normalized_candidates = []
    for item in fine_candidates:
        if not isinstance(item, dict):
            raise ValueError("Each fine candidate must be an object.")
        slot_id = item.get("formula_slot_id")
        if slot_id not in candidate_lookup:
            raise ValueError(f"Unknown formula_slot_id in fine_candidates: {slot_id!r}")
        fine_primary = item.get("fine_label_primary")
        if fine_primary not in FINE_LABELS:
            raise ValueError(f"Invalid fine_label_primary: {fine_primary!r}")
        fine_secondary = normalize_label_list(item.get("fine_label_secondary"))
        if len(fine_secondary) > 2:
            raise ValueError("fine_label_secondary must contain at most 2 labels.")
        for label in fine_secondary:
            if label not in FINE_LABELS:
                raise ValueError(f"Invalid fine secondary label: {label!r}")
        normalized_candidates.append(
            {
                "formula_slot_id": slot_id,
                "fine_label_primary": fine_primary,
                "fine_label_secondary": fine_secondary,
            }
        )

    return {
        "coarse_label_primary": primary,
        "coarse_label_secondary": secondary,
        "fine_candidates": normalized_candidates,
        "auto_confidence": auto_confidence,
        "notes": str(parsed.get("notes", "") or "").strip(),
    }


def normalize_label_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    if not isinstance(value, list):
        raise ValueError(f"Expected list or string for label list, got {type(value).__name__}.")
    normalized: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Label list items must be non-empty strings.")
        normalized.append(item.strip())
    return normalized


def build_dry_run_response(request: Dict[str, Any]) -> Dict[str, Any]:
    primary = request["rule_features"]["rule_primary_label"]
    secondary = request["rule_features"]["rule_secondary_labels"]
    fine_candidates: List[Dict[str, Any]] = []
    for slot in request.get("candidate_formula_slots") or []:
        if slot.get("slot_error") == "missing_predict_formula":
            fine_candidates.append(
                {
                    "formula_slot_id": slot["formula_slot_id"],
                    "fine_label_primary": "Missed formula recognition",
                    "fine_label_secondary": [],
                }
            )
            break
        if slot.get("slot_error") == "extra_predict_formula":
            fine_candidates.append(
                {
                    "formula_slot_id": slot["formula_slot_id"],
                    "fine_label_primary": "Incorrect formula recognition",
                    "fine_label_secondary": [],
                }
            )
            break
    return {
        "coarse_label_primary": primary,
        "coarse_label_secondary": secondary[:2],
        "fine_candidates": fine_candidates,
        "auto_confidence": max(0.30, 1.0 - request["rule_features"]["rule_risk_score"]),
        "notes": "dry-run placeholder result",
    }


def autolabel_key(image_id: str, model_name: str) -> str:
    return f"{image_id}::{model_name}"


def build_long_row_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    index: Dict[Tuple[str, str, str], int] = {}
    for position, row in enumerate(rows):
        key = (row["image_id"], row["model_name"], row.get("formula_slot_id", ""))
        index[key] = position
    return index


def merge_autolabel_result_into_rows(
    rows: List[Dict[str, Any]],
    row_index: Dict[Tuple[str, str, str], int],
    request: Dict[str, Any],
    parsed: Dict[str, Any],
) -> None:
    coarse_key = (request["image_id"], request["model_name"], "")
    coarse_row = rows[row_index[coarse_key]]
    coarse_row["coarse_label_primary"] = parsed["coarse_label_primary"]
    coarse_row["coarse_label_secondary"] = labels_to_cell(parsed["coarse_label_secondary"])
    coarse_row["auto_confidence"] = format_float(parsed["auto_confidence"])
    coarse_row["notes"] = parsed["notes"]

    for slot in request.get("candidate_formula_slots") or []:
        key = (request["image_id"], request["model_name"], slot["formula_slot_id"])
        if key not in row_index:
            continue
        row = rows[row_index[key]]
        row["fine_label_primary"] = ""
        row["fine_label_secondary"] = ""

    for candidate in parsed.get("fine_candidates") or []:
        key = (request["image_id"], request["model_name"], candidate["formula_slot_id"])
        if key not in row_index:
            continue
        row = rows[row_index[key]]
        row["fine_label_primary"] = candidate["fine_label_primary"]
        row["fine_label_secondary"] = labels_to_cell(candidate["fine_label_secondary"])
        if not row["auto_confidence"]:
            row["auto_confidence"] = format_float(parsed["auto_confidence"])


def build_summary_markdown(
    long_rows: Sequence[Dict[str, Any]],
    manifest_rows: Sequence[Dict[str, Any]],
) -> str:
    coarse_rows = [row for row in long_rows if not row.get("formula_slot_id")]
    fine_rows = [row for row in long_rows if row.get("formula_slot_id")]

    lines: List[str] = []
    lines.append("# OCR Error Analysis Auto Summary")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- Coarse-level rows: {len(coarse_rows)}")
    lines.append(f"- Fine-level formula-slot rows: {len(fine_rows)}")
    lines.append(
        f"- Rows with coarse auto labels: {sum(1 for row in coarse_rows if row.get('coarse_label_primary'))}"
    )
    lines.append(
        f"- Rows with fine auto labels: {sum(1 for row in fine_rows if row.get('fine_label_primary'))}"
    )
    lines.append(
        f"- Rows still in auto-only state: {sum(1 for row in long_rows if not row.get('human_final'))}"
    )
    lines.append("")

    lines.append("## 2. Coarse Label Distribution Across Models")
    lines.append("")
    lines.extend(build_markdown_table_for_labels(coarse_rows, "coarse_label_primary"))
    lines.append("")

    lines.append("## 3. Fine Formula Label Distribution")
    lines.append("")
    lines.extend(build_markdown_table_for_labels(fine_rows, "fine_label_primary"))
    lines.append("")

    lines.append("## 4. Shared Hard-Case Candidates")
    lines.append("")
    lines.extend(build_shared_case_lines(coarse_rows, fine_rows))
    lines.append("")

    lines.append("## 5. Representative Case Candidates")
    lines.append("")
    lines.extend(build_representative_case_lines(coarse_rows, manifest_rows))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_markdown_table_for_labels(rows: Sequence[Dict[str, Any]], field: str) -> List[str]:
    counts_by_model: Dict[str, Counter[str]] = defaultdict(Counter)
    models = sorted({row["model_name"] for row in rows})
    labels = sorted({row[field] for row in rows if row.get(field)})
    if not labels:
        return ["No labels have been filled yet."]

    for row in rows:
        label = row.get(field)
        if label:
            counts_by_model[row["model_name"]][label] += 1

    output = [
        "| Label | " + " | ".join(models) + " |",
        "| --- | " + " | ".join("---:" for _ in models) + " |",
    ]
    for label in labels:
        cells = [str(counts_by_model[model][label]) for model in models]
        output.append("| " + " | ".join([label] + cells) + " |")
    return output


def build_shared_case_lines(
    coarse_rows: Sequence[Dict[str, Any]],
    fine_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    grouped_coarse: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in coarse_rows:
        if row.get("coarse_label_primary"):
            grouped_coarse[row["image_id"]].append(row)

    coarse_candidates = []
    for image_id, rows in grouped_coarse.items():
        label_counts = Counter(row["coarse_label_primary"] for row in rows if row.get("coarse_label_primary"))
        for label, count in label_counts.items():
            if count >= 2:
                coarse_candidates.append((image_id, label, count))
    if coarse_candidates:
        lines.append("Shared coarse-level hard cases:")
        for image_id, label, count in coarse_candidates[:10]:
            lines.append(f"- `{image_id}` is labeled `{label}` by `{count}` models.")

    grouped_fine: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in fine_rows:
        if row.get("fine_label_primary"):
            grouped_fine[row["formula_slot_id"]].append(row)
    fine_candidates = []
    for slot_id, rows in grouped_fine.items():
        label_counts = Counter(row["fine_label_primary"] for row in rows if row.get("fine_label_primary"))
        for label, count in label_counts.items():
            if count >= 2:
                fine_candidates.append((slot_id, label, count))
    if fine_candidates:
        lines.append("Shared fine-level hard cases:")
        for slot_id, label, count in fine_candidates[:10]:
            lines.append(f"- `{slot_id}` is labeled `{label}` by `{count}` models.")

    if not lines:
        return ["No shared hard cases were found with the same label from at least two models."]
    return lines


def build_representative_case_lines(
    coarse_rows: Sequence[Dict[str, Any]],
    manifest_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    manifest_by_image = {row["image_id"]: row for row in manifest_rows if row.get("image_id")}
    lines: List[str] = []
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in coarse_rows:
        if row.get("coarse_label_primary"):
            by_model[row["model_name"]].append(row)

    if not by_model:
        return ["No representative case candidates are available."]

    for model_name in sorted(by_model):
        rows = sorted(
            by_model[model_name],
            key=lambda row: (
                safe_float(row.get("auto_confidence")) if row.get("auto_confidence") else 1.0,
                row["image_id"],
            ),
        )
        lines.append(f"- `{model_name}` candidates:")
        for row in rows[:3]:
            manifest = manifest_by_image.get(row["image_id"], {})
            bucket = manifest.get("sample_bucket", row.get("sample_bucket", ""))
            lines.append(
                f"  `{row['image_id']}` / `{bucket}` / `{row['coarse_label_primary']}` / "
                f"auto_confidence={row.get('auto_confidence') or 'NA'}"
            )
    return lines


def write_calibration_notes(
    path: Path,
    calibration_images: Sequence[str],
    selected_buckets: Dict[str, List[str]],
) -> None:
    lines = ["# Calibration Notes", ""]
    lines.append("## Calibration Samples")
    lines.append("")
    for image_id in calibration_images:
        bucket = next(bucket for bucket, items in selected_buckets.items() if image_id in items)
        lines.append(f"- `{image_id}` / `{bucket}`")
    lines.append("")
    lines.append("## Label Boundary Notes")
    lines.append("")
    lines.append("- To be completed.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Any]:
    if not path.exists():
        return []
    items: List[Any] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        items.append(json.loads(stripped))
    return items


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, items: Sequence[Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_directory(path.parent)
    resolved_fieldnames = list(fieldnames or infer_fieldnames(rows))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in resolved_fieldnames})


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def average(values: Iterable[Optional[float]]) -> float:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def spread(values: Iterable[Any]) -> float:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return max(filtered) - min(filtered)


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_optional_str(value: Any) -> str:
    if value is None or value == "":
        return "NULL"
    return str(value)


def normalize_optional_int(value: Any) -> str:
    number = safe_int(value)
    if number is None:
        return "NULL"
    return str(number)


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def labels_to_cell(labels: Sequence[str]) -> str:
    return " | ".join(label for label in labels if label)


def stringify_bool(value: bool) -> str:
    return "true" if value else "false"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".") if "." in f"{value:.6f}" else f"{value:.6f}"


def format_json_error(prefix: str, exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{prefix} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    return f"{prefix}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
