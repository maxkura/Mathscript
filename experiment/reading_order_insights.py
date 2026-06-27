#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "experiment" / "reading_order_insights"
DEFAULT_METRIC_ROOT = BASE_DIR / "output" / "metric_threshold_0_5"
DEFAULT_PREDICT_ROOT = BASE_DIR / "output" / "predict"
DEFAULT_PROTOCOL_ROOT = BASE_DIR / "output" / "protocol_debias"
DEFAULT_GT_PATH = BASE_DIR / "data" / "GT" / "extracted_gt.json"
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "row_image"

MODELS = (
    "kimi-k2.5",
    "gemini-2.5-flash",
    "qwen3.5-plus",
    "qwen3-vl-plus",
    "claude-sonnet-4-5-20250929",
    "qwen3.5-flash",
    "doubao-1-5-vision-pro-32K",
    "gpt-4o",
    "grok-4-0709",
)

CLASSIC_SAMPLES = {
    "row_C_220.jpg": "Rare true cross-subquestion misorder: multiple models read the (1)/(2) boundary too early.",
    "row_C_366.jpg": "Shared hard sample: coordinate-geometry dense formulas plus title/header intrusion, mostly low-evidence.",
    "row_C_169.jpg": "Kimi-clean discriminative sample: structure is readable, but some models lose line anchoring when Unicode math variants appear.",
    "row_C_120.jpg": "Block-equation sample: systems of equations are easy to fragment into token-level lines.",
    "row_C_419.jpg": "Mid-density two-part proof: good content reading, but some models leak one Q002 line into the Q001 block.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reading-order behavior across benchmark models.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT_PATH)
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--predict-root", type=Path, default=DEFAULT_PREDICT_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = (denom_x * denom_y) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def image_path_from_filename(image_dir: Path, filename: str) -> Path:
    return image_dir / filename.replace("row_C_", "row_")


def load_image_sizes(image_dir: Path, filenames: Iterable[str]) -> Dict[str, Tuple[int, int]]:
    sizes: Dict[str, Tuple[int, int]] = {}
    for filename in filenames:
        path = image_path_from_filename(image_dir, filename)
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                sizes[filename] = image.size
        except OSError:
            continue
    return sizes


def build_gt_features(gt_records: List[Dict[str, Any]], image_sizes: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
    features: Dict[str, Dict[str, Any]] = {}
    for record in gt_records:
        if record.get("QuestionType") != "ConstructedResponse":
            continue
        filename = record.get("filename")
        transcription = record.get("transcription")
        if not isinstance(filename, str) or not isinstance(transcription, list):
            continue

        question_ids = [item.get("question_id") for item in transcription]
        distinct_question_ids: List[Any] = []
        seen_questions: set[Any] = set()
        subquestion_switches = 0
        previous_question = None
        per_question_lines: Counter[Any] = Counter()
        for question_id in question_ids:
            if previous_question is not None and question_id != previous_question:
                subquestion_switches += 1
            previous_question = question_id
            if question_id not in seen_questions:
                distinct_question_ids.append(question_id)
                seen_questions.add(question_id)
            per_question_lines[question_id] += 1

        width, height = image_sizes.get(filename, (None, None))
        formula_count = len(record.get("formula_list") or [])
        gt_line_count = len(transcription)
        features[filename] = {
            "filename": filename,
            "gt_line_count": gt_line_count,
            "subquestion_count": len(distinct_question_ids),
            "subquestion_switches": subquestion_switches,
            "max_lines_per_subquestion": max(per_question_lines.values()) if per_question_lines else 0,
            "min_lines_per_subquestion": min(per_question_lines.values()) if per_question_lines else 0,
            "formula_count": formula_count,
            "formula_per_line": formula_count / gt_line_count if gt_line_count else 0.0,
            "width": width,
            "height": height,
            "aspect_ratio": (width / height) if width and height else None,
            "gt_boundaries": build_gt_boundaries(transcription),
        }
    return features


def build_gt_boundaries(transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    boundaries: List[Dict[str, Any]] = []
    previous_question = object()
    for item in transcription:
        question_id = item.get("question_id")
        if question_id == previous_question:
            continue
        boundaries.append(
            {
                "seq": item.get("seq"),
                "question_id": question_id,
                "content": str(item.get("content", ""))[:80],
            }
        )
        previous_question = question_id
    return boundaries


def compute_prediction_features(prediction_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default = {
        "header_noise": 0,
        "pred_line_count": 0,
        "matched_line_count": 0,
        "order_inversion_count": 0,
        "first_items": [],
    }
    if not prediction_record:
        return default

    transcription = prediction_record.get("transcription")
    if not isinstance(transcription, list):
        return default

    items = sorted(transcription, key=lambda item: item.get("seq", 10**9))
    header_noise = 0
    matched_line_count = 0
    order_inversion_count = 0
    first_items: List[Dict[str, Any]] = []
    seen_match = False
    previous_gt_seq: Optional[int] = None
    for item in items:
        if (
            item.get("match_status") == "unmatched"
            and item.get("question_id") is None
            and not seen_match
        ):
            header_noise += 1
        if item.get("match_status") == "matched" and isinstance(item.get("gt_seq"), int):
            seen_match = True
            matched_line_count += 1
            gt_seq = item["gt_seq"]
            if previous_gt_seq is not None and gt_seq < previous_gt_seq:
                order_inversion_count += 1
            previous_gt_seq = gt_seq
        if len(first_items) < 12:
            first_items.append(
                {
                    "seq": item.get("seq"),
                    "question_id": item.get("question_id"),
                    "gt_seq": item.get("gt_seq"),
                    "match_status": item.get("match_status"),
                    "content": str(item.get("content", ""))[:80],
                }
            )

    return {
        "header_noise": header_noise,
        "pred_line_count": len(items),
        "matched_line_count": matched_line_count,
        "order_inversion_count": order_inversion_count,
        "first_items": first_items,
    }


def load_prediction_index(predict_root: Path, model: str) -> Dict[str, Dict[str, Any]]:
    path = predict_root / model / "predict.json"
    if not path.exists():
        return {}
    records = load_json(path)
    return {
        record.get("filename"): record
        for record in records
        if isinstance(record, dict)
        and record.get("QuestionType") == "ConstructedResponse"
        and isinstance(record.get("filename"), str)
    }


def build_model_rows(metric_root: Path, predict_root: Path, gt_features: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    sample_rows_by_model: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for model in MODELS:
        sample_metrics_path = metric_root / model / "sample_metrics.json"
        if not sample_metrics_path.exists():
            continue
        sample_metrics = load_json(sample_metrics_path)
        prediction_index = load_prediction_index(predict_root, model)

        for sample in sample_metrics:
            if sample.get("question_type") != "ConstructedResponse":
                continue
            filename = sample.get("filename")
            if not isinstance(filename, str) or filename not in gt_features:
                continue
            reading_order = sample.get("reading_order")
            if not isinstance(reading_order, dict):
                continue

            pred_features = compute_prediction_features(prediction_index.get(filename))
            pair_row = {
                "model": model,
                "filename": filename,
                "status": reading_order.get("status"),
                "ros": reading_order.get("ros"),
                "mcr": reading_order.get("mcr"),
                "bcs": reading_order.get("bcs"),
                "sqa": reading_order.get("sqa"),
                "iqa": reading_order.get("iqa"),
                **pred_features,
                **gt_features[filename],
            }
            pair_rows.append(pair_row)
            sample_rows_by_model[model][filename] = pair_row

    return pair_rows, sample_rows_by_model


def summarize_models(pair_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        rows_by_model[row["model"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for model in MODELS:
        rows = rows_by_model.get(model, [])
        statuses = Counter(row["status"] for row in rows)
        scored_rows = [row for row in rows if row.get("ros") is not None]
        summary = {
            "model": model,
            "applicable_samples": len(rows),
            "scored_samples": len(scored_rows),
            "ros_coverage": ratio(len(scored_rows), len(rows)),
            "ros": average(row.get("ros") for row in scored_rows),
            "mcr": average(row.get("mcr") for row in rows),
            "bcs": average(row.get("bcs") for row in rows),
            "sqa": average(row.get("sqa") for row in rows),
            "iqa": average(row.get("iqa") for row in rows),
            "status_counts": dict(statuses),
            "clean_count": statuses.get("CLEAN", 0),
            "fragmented_count": statuses.get("FRAGMENTED", 0),
            "low_evidence_count": statuses.get("LOW_EVIDENCE", 0),
            "minor_inner_disorder_count": statuses.get("MINOR_INNER_DISORDER", 0),
            "misordered_count": statuses.get("MISORDERED", 0),
            "avg_header_noise_low_evidence": average(
                row.get("header_noise") for row in rows if row.get("status") == "LOW_EVIDENCE"
            ),
            "avg_header_noise_fragmented": average(
                row.get("header_noise") for row in rows if row.get("status") == "FRAGMENTED"
            ),
            "avg_header_noise_clean": average(
                row.get("header_noise") for row in rows if row.get("status") == "CLEAN"
            ),
        }
        summaries.append(summary)
    return summaries


def load_protocol_overall(protocol_root: Path, subdir: str, model: str) -> Optional[Dict[str, Any]]:
    path = protocol_root / subdir / model / "overall_metrics.json"
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    return payload.get("overall")


def summarize_protocol(metric_root: Path, protocol_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        raw_path = metric_root / model / "overall_metrics.json"
        if not raw_path.exists():
            continue
        raw = load_json(raw_path).get("overall", {})
        strict = load_protocol_overall(protocol_root, "strict_metric", model) or {}
        oracle = load_protocol_overall(protocol_root, "oracle_metric", model) or {}
        rows.append(
            {
                "model": model,
                "raw_ros": raw.get("ros"),
                "raw_ros_coverage": raw.get("ros_coverage"),
                "strict_ros": strict.get("ros"),
                "strict_ros_coverage": strict.get("ros_coverage"),
                "oracle_ros": oracle.get("ros"),
                "oracle_ros_coverage": oracle.get("ros_coverage"),
                "strict_ros_gain": subtract(strict.get("ros"), raw.get("ros")),
                "oracle_ros_gain": subtract(oracle.get("ros"), raw.get("ros")),
                "strict_coverage_gain": subtract(strict.get("ros_coverage"), raw.get("ros_coverage")),
                "oracle_coverage_gain": subtract(oracle.get("ros_coverage"), raw.get("ros_coverage")),
            }
        )
    return rows


def subtract(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def compute_pairwise_correlations(pair_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    feature_names = (
        "gt_line_count",
        "subquestion_count",
        "formula_count",
        "header_noise",
        "order_inversion_count",
        "pred_line_count",
        "height",
        "width",
    )
    results: Dict[str, Dict[str, Optional[float]]] = {}

    results["ros_scored_only"] = {}
    for feature_name in feature_names:
        xs: List[float] = []
        ys: List[float] = []
        for row in pair_rows:
            feature_value = row.get(feature_name)
            ros = row.get("ros")
            if feature_value is None or ros is None:
                continue
            xs.append(float(feature_value))
            ys.append(float(ros))
        results["ros_scored_only"][feature_name] = pearson(xs, ys)

    for target_status in ("LOW_EVIDENCE", "FRAGMENTED", "CLEAN"):
        result_key = f"is_{target_status.lower()}"
        results[result_key] = {}
        for feature_name in feature_names:
            xs = []
            ys = []
            for row in pair_rows:
                feature_value = row.get(feature_name)
                if feature_value is None:
                    continue
                xs.append(float(feature_value))
                ys.append(1.0 if row.get("status") == target_status else 0.0)
            results[result_key][feature_name] = pearson(xs, ys)

    return results


def aggregate_samples(pair_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sample_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        sample_buckets[row["filename"]].append(row)

    aggregated_rows: List[Dict[str, Any]] = []
    for filename, rows in sample_buckets.items():
        scored_rows = [row for row in rows if row.get("ros") is not None]
        aggregated_rows.append(
            {
                "filename": filename,
                "image_path": str(image_path_from_filename(DEFAULT_IMAGE_DIR, filename)),
                "applicable_models": len(rows),
                "scored_models": len(scored_rows),
                "avg_ros": average(row.get("ros") for row in scored_rows),
                "avg_mcr": average(row.get("mcr") for row in rows),
                "avg_bcs": average(row.get("bcs") for row in rows),
                "avg_sqa": average(row.get("sqa") for row in rows),
                "avg_iqa": average(row.get("iqa") for row in rows),
                "low_evidence_models": sum(1 for row in rows if row.get("status") == "LOW_EVIDENCE"),
                "fragmented_models": sum(1 for row in rows if row.get("status") == "FRAGMENTED"),
                "clean_models": sum(1 for row in rows if row.get("status") == "CLEAN"),
                "minor_models": sum(1 for row in rows if row.get("status") == "MINOR_INNER_DISORDER"),
                "gt_line_count": rows[0].get("gt_line_count"),
                "subquestion_count": rows[0].get("subquestion_count"),
                "formula_count": rows[0].get("formula_count"),
                "aspect_ratio": rows[0].get("aspect_ratio"),
                "avg_header_noise": average(row.get("header_noise") for row in rows),
                "gt_boundaries": rows[0].get("gt_boundaries"),
            }
        )

    aggregated_rows.sort(
        key=lambda row: (
            -row["low_evidence_models"],
            -row["fragmented_models"],
            row["avg_ros"] if row["avg_ros"] is not None else -1.0,
        )
    )
    return aggregated_rows


def select_classic_sample_details(
    sample_rows_by_model: Dict[str, Dict[str, Any]],
    sample_aggregates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregate_index = {row["filename"]: row for row in sample_aggregates}
    selected: List[Dict[str, Any]] = []
    for filename, note in CLASSIC_SAMPLES.items():
        if filename not in aggregate_index:
            continue
        per_model: Dict[str, Any] = {}
        for model in MODELS:
            row = sample_rows_by_model.get(model, {}).get(filename)
            if row is None:
                continue
            per_model[model] = {
                "status": row.get("status"),
                "ros": row.get("ros"),
                "mcr": row.get("mcr"),
                "bcs": row.get("bcs"),
                "sqa": row.get("sqa"),
                "iqa": row.get("iqa"),
                "header_noise": row.get("header_noise"),
                "order_inversion_count": row.get("order_inversion_count"),
                "first_items": row.get("first_items"),
            }
        selected.append(
            {
                "filename": filename,
                "note": note,
                **aggregate_index[filename],
                "per_model": per_model,
            }
        )
    return selected


def select_extreme_tables(sample_aggregates: List[Dict[str, Any]], sample_rows_by_model: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    shared_hard = sample_aggregates[:10]
    low_sqa = sorted(
        [row for row in sample_aggregates if row.get("avg_sqa") is not None and row["avg_sqa"] < 0.95],
        key=lambda row: row["avg_sqa"],
    )[:10]

    kimi_advantage: List[Dict[str, Any]] = []
    for row in sample_aggregates:
        kimi_row = sample_rows_by_model.get("kimi-k2.5", {}).get(row["filename"])
        if not kimi_row or kimi_row.get("status") != "CLEAN":
            continue
        other_rows = [
            sample_rows_by_model.get(model, {}).get(row["filename"])
            for model in MODELS
            if model != "kimi-k2.5"
        ]
        low_evidence_others = sum(1 for other_row in other_rows if other_row and other_row.get("status") == "LOW_EVIDENCE")
        clean_others = sum(1 for other_row in other_rows if other_row and other_row.get("status") == "CLEAN")
        if low_evidence_others < 4:
            continue
        kimi_advantage.append(
            {
                **row,
                "kimi_ros": kimi_row.get("ros"),
                "kimi_status": kimi_row.get("status"),
                "other_low_evidence": low_evidence_others,
                "other_clean": clean_others,
            }
        )
    kimi_advantage.sort(key=lambda row: (-row["other_low_evidence"], row["avg_ros"] if row["avg_ros"] is not None else -1.0))
    return {
        "shared_hard": shared_hard,
        "low_sqa": low_sqa,
        "kimi_advantage": kimi_advantage[:15],
    }


def build_high_level_findings(model_summaries: List[Dict[str, Any]], protocol_rows: List[Dict[str, Any]], pair_correlations: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
    avg_sqa = average(summary.get("sqa") for summary in model_summaries)
    avg_iqa = average(summary.get("iqa") for summary in model_summaries)
    total_misordered = sum(summary.get("misordered_count", 0) for summary in model_summaries)
    low_evidence_models = sorted(model_summaries, key=lambda summary: summary.get("low_evidence_count", 0), reverse=True)
    best_coverage = max(model_summaries, key=lambda summary: summary.get("ros_coverage") or -1.0)
    best_ros = max(model_summaries, key=lambda summary: summary.get("ros") or -1.0)
    largest_strict_delta = max(protocol_rows, key=lambda row: abs(row.get("strict_ros_gain") or 0.0))
    largest_oracle_delta = max(protocol_rows, key=lambda row: abs(row.get("oracle_ros_gain") or 0.0))
    findings = [
        (
            "True ordering inversions are rare: across models, the average "
            f"SQA is {format_float(avg_sqa)} and the average IQA is {format_float(avg_iqa)}, "
            f"with {total_misordered} explicit `MISORDERED` cases overall."
        ),
        (
            "This benchmark mainly separates models by whether they can produce enough "
            "continuous matched evidence, rather than by simple pairwise ordering alone. "
            f"The highest ROS belongs to {best_ros['model']} "
            f"(ROS={format_float(best_ros['ros'])}), and the best coverage also belongs to "
            f"{best_coverage['model']} (coverage={format_float(best_coverage['ros_coverage'])})."
        ),
        (
            "Long derivations and dense formulas remain the main semantic bottlenecks: "
            f"corr(ROS, gt_line_count)={format_float(pair_correlations['ros_scored_only']['gt_line_count'])}, "
            f"and corr(ROS, formula_count)={format_float(pair_correlations['ros_scored_only']['formula_count'])}."
        ),
        (
            "`header/title intrusion` is the most stable LOW_EVIDENCE trigger: "
            f"corr(is_LOW_EVIDENCE, header_noise)={format_float(pair_correlations['is_low_evidence']['header_noise'])}, "
            "well above geometric factors such as image width and height."
        ),
        (
            "Protocol debiasing changes reading-order scores only slightly overall: "
            f"the largest Strict ROS shift is {largest_strict_delta['model']} "
            f"{format_signed_float(largest_strict_delta.get('strict_ros_gain'))}, and "
            f"the largest Oracle ROS shift is {largest_oracle_delta['model']} "
            f"{format_signed_float(largest_oracle_delta.get('oracle_ros_gain'))}."
        ),
        (
            f"The two models with the most LOW_EVIDENCE cases are {low_evidence_models[0]['model']} "
            f"({low_evidence_models[0]['low_evidence_count']}) and {low_evidence_models[1]['model']} "
            f"({low_evidence_models[1]['low_evidence_count']})."
        ),
    ]
    return findings


def build_report(
    model_summaries: List[Dict[str, Any]],
    protocol_rows: List[Dict[str, Any]],
    pair_correlations: Dict[str, Dict[str, Optional[float]]],
    extreme_tables: Dict[str, List[Dict[str, Any]]],
    classic_samples: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Reading-Order Capability Report")
    lines.append("")
    lines.append("## 1. Key Findings")
    lines.append("")
    for finding in build_high_level_findings(model_summaries, protocol_rows, pair_correlations):
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## 2. Model Overview")
    lines.append("")
    lines.append("| Model | ROS | Coverage | MCR | BCS | SQA | IQA | CLEAN | FRAGMENTED | LOW_EVIDENCE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in model_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary["model"],
                    format_float(summary.get("ros")),
                    format_float(summary.get("ros_coverage")),
                    format_float(summary.get("mcr")),
                    format_float(summary.get("bcs")),
                    format_float(summary.get("sqa")),
                    format_float(summary.get("iqa")),
                    str(summary.get("clean_count", 0)),
                    str(summary.get("fragmented_count", 0)),
                    str(summary.get("low_evidence_count", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 3. Protocol-Debias Comparison")
    lines.append("")
    lines.append("| Model | Raw ROS | Strict ROS | Oracle ROS | Strict dROS | Oracle dROS | Raw Cov | Strict Cov | Oracle Cov |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in protocol_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    format_float(row.get("raw_ros")),
                    format_float(row.get("strict_ros")),
                    format_float(row.get("oracle_ros")),
                    format_signed_float(row.get("strict_ros_gain")),
                    format_signed_float(row.get("oracle_ros_gain")),
                    format_float(row.get("raw_ros_coverage")),
                    format_float(row.get("strict_ros_coverage")),
                    format_float(row.get("oracle_ros_coverage")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Strict and Oracle debiasing barely change the reading-order scoring logic itself. This behaves more like a control study asking whether reading-order gaps are mainly caused by formatting noise, and the answer is largely no.")
    lines.append("")
    lines.append("## 4. Factor Correlations")
    lines.append("")
    lines.append("| Target | gt_line_count | subquestion_count | formula_count | header_noise | order_inversion_count | pred_line_count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    correlation_rows = [
        ("ROS(scored)", pair_correlations["ros_scored_only"]),
        ("is_LOW_EVIDENCE", pair_correlations["is_low_evidence"]),
        ("is_FRAGMENTED", pair_correlations["is_fragmented"]),
        ("is_CLEAN", pair_correlations["is_clean"]),
    ]
    for label, values in correlation_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_float(values.get("gt_line_count")),
                    format_float(values.get("subquestion_count")),
                    format_float(values.get("formula_count")),
                    format_float(values.get("header_noise")),
                    format_float(values.get("order_inversion_count")),
                    format_float(values.get("pred_line_count")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `gt_line_count` and `formula_count` mainly push samples from `CLEAN` to `FRAGMENTED`, suggesting that continuous coverage of long derivations is the real difficulty.")
    lines.append("- `header_noise` is most strongly correlated with `LOW_EVIDENCE`, which suggests many models read problem headers or titles before reliably switching to the answer body.")
    lines.append("- `order_inversion_count` is not strongly correlated, which implies most failures are not large-scale reversals but local fragmentation, missed alignments, or block drift.")
    lines.append("")
    lines.append("## 5. Classic Samples")
    lines.append("")
    for sample in classic_samples:
        lines.append(f"### {sample['filename']}")
        lines.append("")
        lines.append(f"- Source image: `{sample['image_path']}`")
        lines.append(f"- Observation: {sample['note']}")
        lines.append(
            f"- Aggregate metrics: avg_ros={format_float(sample.get('avg_ros'))}, "
            f"avg_sqa={format_float(sample.get('avg_sqa'))}, avg_iqa={format_float(sample.get('avg_iqa'))}, "
            f"low_evidence_models={sample.get('low_evidence_models')}, fragmented_models={sample.get('fragmented_models')}, "
            f"clean_models={sample.get('clean_models')}"
        )
        lines.append(f"- GT boundaries: {json.dumps(sample.get('gt_boundaries', []), ensure_ascii=False)}")
        lines.append("- Per-model status:")
        for model in MODELS:
            metrics = sample["per_model"].get(model)
            if not metrics:
                continue
            lines.append(
                f"  - {model}: status={metrics.get('status')}, "
                f"ros={format_float(metrics.get('ros'))}, "
                f"mcr={format_float(metrics.get('mcr'))}, "
                f"bcs={format_float(metrics.get('bcs'))}, "
                f"sqa={format_float(metrics.get('sqa'))}, "
                f"iqa={format_float(metrics.get('iqa'))}, "
                f"header_noise={metrics.get('header_noise')}, "
                f"order_inversion_count={metrics.get('order_inversion_count')}"
            )
    lines.append("")
    lines.append("## 6. Automatically Selected Candidate Samples")
    lines.append("")
    lines.append("### 6.1 Shared Hard Cases")
    lines.append("")
    lines.append("| Sample | avg_ros | low_evidence | fragmented | clean | gt_lines | subquestions | formulas | avg_header_noise |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in extreme_tables["shared_hard"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["filename"],
                    format_float(row.get("avg_ros")),
                    str(row.get("low_evidence_models")),
                    str(row.get("fragmented_models")),
                    str(row.get("clean_models")),
                    str(row.get("gt_line_count")),
                    str(row.get("subquestion_count")),
                    str(row.get("formula_count")),
                    format_float(row.get("avg_header_noise")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("### 6.2 Samples with Real Cross-Subquestion Order Risk")
    lines.append("")
    lines.append("| Sample | avg_sqa | avg_iqa | low_evidence | fragmented | gt_lines | subquestions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in extreme_tables["low_sqa"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["filename"],
                    format_float(row.get("avg_sqa")),
                    format_float(row.get("avg_iqa")),
                    str(row.get("low_evidence_models")),
                    str(row.get("fragmented_models")),
                    str(row.get("gt_line_count")),
                    str(row.get("subquestion_count")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("### 6.3 Kimi-Separating Samples")
    lines.append("")
    lines.append("| Sample | Kimi ROS | others LOW_EVIDENCE | others CLEAN | gt_lines | subquestions | formulas |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in extreme_tables["kimi_advantage"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["filename"],
                    format_float(row.get("kimi_ros")),
                    str(row.get("other_low_evidence")),
                    str(row.get("other_clean")),
                    str(row.get("gt_line_count")),
                    str(row.get("subquestion_count")),
                    str(row.get("formula_count")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 7. Suggestions for Metric Design")
    lines.append("")
    lines.append("- Do not reduce reading-order ability to pairwise order accuracy alone. The main failure mode in this benchmark is `evidence sufficiency + continuity`, not simple reversals.")
    lines.append("- If you propose a new metric, consider three explicit layers: entering the answer body, stability at subquestion boundaries, and within-block coverage continuity. The first and third separate models more clearly than pure ordering.")
    lines.append("- In the paper, separate `protocol noise` from `visual reading failure`. The small Strict/Oracle ROS shifts suggest that reading-order experiments should not gain points mainly by fixing JSON or Unicode surface forms.")
    lines.append("- Classic samples should prioritize three phenomena: `header/title intrusion`, overly early switching at `(1)/(2)` boundaries, and fragmented block equations or vertically arranged formula blocks. These patterns explain the largest real capability gaps.")
    lines.append("- If you want a metric closer to human reading behavior, score pre-body misreads separately and isolate header overshoot from ordinary unmatched lines.")
    lines.append("")
    lines.append(f"Report output directory: `{output_dir}`")
    lines.append("")
    return "\n".join(lines)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def format_signed_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:+.3f}"


def main() -> int:
    args = parse_args()
    gt_records = load_json(args.gt)
    gt_filenames = [
        record.get("filename")
        for record in gt_records
        if isinstance(record, dict) and record.get("QuestionType") == "ConstructedResponse"
    ]
    image_sizes = load_image_sizes(args.image_dir, [filename for filename in gt_filenames if isinstance(filename, str)])
    gt_features = build_gt_features(gt_records, image_sizes)
    pair_rows, sample_rows_by_model = build_model_rows(args.metric_root, args.predict_root, gt_features)
    model_summaries = summarize_models(pair_rows)
    protocol_rows = summarize_protocol(args.metric_root, args.protocol_root)
    pair_correlations = compute_pairwise_correlations(pair_rows)
    sample_aggregates = aggregate_samples(pair_rows)
    classic_samples = select_classic_sample_details(sample_rows_by_model, sample_aggregates)
    extreme_tables = select_extreme_tables(sample_aggregates, sample_rows_by_model)

    summary = {
        "model_summaries": model_summaries,
        "protocol_rows": protocol_rows,
        "pair_correlations": pair_correlations,
        "classic_samples": classic_samples,
        "extreme_tables": extreme_tables,
    }
    report = build_report(
        model_summaries=model_summaries,
        protocol_rows=protocol_rows,
        pair_correlations=pair_correlations,
        extreme_tables=extreme_tables,
        classic_samples=classic_samples,
        output_dir=args.output_dir,
    )
    write_json(args.output_dir / "summary.json", summary)
    write_text(args.output_dir / "report_zh.md", report)
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'report_zh.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
