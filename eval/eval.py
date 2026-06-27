#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .metric import (
        FormulaParseError,
        aggregate_formula_metrics,
        aggregate_reading_order_metrics,
        aggregate_refusal_metrics,
        aggregate_stem_metrics,
        classify_refusal_sample,
        compute_composite_score,
        compute_formula_metrics,
        compute_reading_order_metrics,
        compute_stem_metrics,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from metric import (  # type: ignore
        FormulaParseError,
        aggregate_formula_metrics,
        aggregate_reading_order_metrics,
        aggregate_refusal_metrics,
        aggregate_stem_metrics,
        classify_refusal_sample,
        compute_composite_score,
        compute_formula_metrics,
        compute_reading_order_metrics,
        compute_stem_metrics,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normalized benchmark predict.json files.")
    parser.add_argument("--gt", default="data/GT/extracted_gt.json", help="Ground-truth JSON array path.")
    parser.add_argument(
        "--predict-root",
        default="output/predict",
        help="Root directory containing per-model predict.json files.",
    )
    parser.add_argument(
        "--output-root",
        default="output/metric",
        help="Root directory where evaluation JSON outputs are written.",
    )
    parser.add_argument("--model", default=None, help="Evaluate only one model under predict-root.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.50,
        help=(
            "Stem line match threshold for non-structured transcription fallback. "
            "Structured inputs reuse declared gt_seq/match_status."
        ),
    )
    parser.add_argument("--w-slt", type=float, default=0.4, help="Formula SLT weight.")
    parser.add_argument("--w-opt", type=float, default=0.6, help="Formula OPT weight.")
    parser.add_argument("--alpha-stem", type=float, default=0.40, help="Composite stem weight.")
    parser.add_argument("--alpha-formula", type=float, default=0.35, help="Composite formula weight.")
    parser.add_argument("--alpha-ros", type=float, default=0.20, help="Composite reading-order weight.")
    parser.add_argument("--alpha-refusal", type=float, default=0.05, help="Composite refusal weight.")
    return parser.parse_args(argv)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_json_array(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{display_path(path)} must contain a top-level JSON array.")
    return data


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def validate_weights(args: argparse.Namespace, tolerance: float = 1e-9) -> None:
    if abs((args.w_slt + args.w_opt) - 1.0) > tolerance:
        raise ValueError("--w-slt + --w-opt must equal 1.")
    total_alpha = args.alpha_stem + args.alpha_formula + args.alpha_ros + args.alpha_refusal
    if abs(total_alpha - 1.0) > tolerance:
        raise ValueError(
            "--alpha-stem + --alpha-formula + --alpha-ros + --alpha-refusal must equal 1."
        )


def _normalize_filename(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    return text


def _normalize_idx(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _record_filename(record: Any, source_name: str, record_index: int) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"{source_name} record {record_index} must be a JSON object.")
    filename = _normalize_filename(record.get("filename"))
    if not filename:
        raise ValueError(f"{source_name} record {record_index} must contain a valid filename.")
    return filename


def build_record_index(records: Sequence[Any], source_name: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record_index, record in enumerate(records, start=1):
        filename = _record_filename(record, source_name, record_index)
        if filename in index:
            raise ValueError(f"duplicate {source_name} filename: {filename!r}")
        index[filename] = record
    return index


def discover_models(predict_root: Path, model_name: Optional[str] = None) -> List[Tuple[str, Path]]:
    if model_name:
        predict_path = predict_root / model_name / "predict.json"
        if not predict_path.exists():
            raise FileNotFoundError(
                f"predict.json for model {model_name!r} was not found under {display_path(predict_root)}."
            )
        return [(model_name, predict_path)]

    models: List[Tuple[str, Path]] = []
    for child in sorted(predict_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        predict_path = child / "predict.json"
        if predict_path.exists():
            models.append((child.name, predict_path))

    if not models:
        raise FileNotFoundError(f"no models with predict.json were found under {display_path(predict_root)}.")
    return models


def make_sample_metric(
    *,
    filename: str,
    idx: Optional[int],
    question_type: Any,
    pair_status: str,
    stem: Optional[Dict[str, Any]],
    formula: Optional[Dict[str, Any]],
    reading_order: Optional[Dict[str, Any]],
    refusal: Optional[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "idx": idx,
        "question_type": question_type,
        "pair_status": pair_status,
        "stem": stem,
        "formula": formula,
        "reading_order": reading_order,
        "refusal": refusal,
        "errors": errors,
    }


def fatal_error_report(
    *,
    model_name: str,
    message: str,
    predict_total: int = 0,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "summary": {
            "predict_total": predict_total,
            "paired_total": 0,
            "missing_predict_count": 0,
            "extra_predict_count": 0,
            "sample_error_count": 0,
            "fatal_error_count": 1,
            "sample_coverage": 0.0,
        },
        "errors": [
            {
                "reason": "fatal_error",
                "message": message,
            }
        ],
    }


def _sample_coverage(paired_total: int, gt_total: int) -> float:
    if gt_total == 0:
        return 0.0
    return paired_total / gt_total


def _model_output_dir(output_root: Path, model_name: str) -> Path:
    return output_root / model_name


def evaluate_model(
    *,
    model_name: str,
    predict_path: Path,
    gt_records: Sequence[Dict[str, Any]],
    gt_index: Dict[str, Dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    model_output_dir = _model_output_dir(output_root, model_name)
    predict_total = 0

    try:
        predict_records = load_json_array(predict_path)
        predict_total = len(predict_records)
        predict_index = build_record_index(predict_records, f"predict ({model_name})")

        sample_metrics: List[Dict[str, Any]] = []
        model_errors: List[Dict[str, Any]] = []
        stem_metrics: List[Optional[Dict[str, Any]]] = []
        formula_metrics: List[Optional[Dict[str, Any]]] = []
        reading_order_metrics: List[Optional[Dict[str, Any]]] = []
        refusal_metrics: List[Dict[str, Any]] = []
        paired_total = 0
        missing_predict_count = 0

        for gt_record_index, gt_record in enumerate(gt_records, start=1):
            filename = _record_filename(gt_record, "GT", gt_record_index)
            idx = _normalize_idx(gt_record.get("idx"))
            predict_record = predict_index.get(filename)

            if predict_record is None:
                missing_predict_count += 1
                error_item = {
                    "reason": "missing_predict",
                    "filename": filename,
                    "idx": idx,
                }
                model_errors.append(error_item)
                sample_metrics.append(
                    make_sample_metric(
                        filename=filename,
                        idx=idx,
                        question_type=gt_record.get("QuestionType"),
                        pair_status="missing_predict",
                        stem=None,
                        formula=None,
                        reading_order=None,
                        refusal=None,
                        errors=[error_item],
                    )
                )
                continue

            paired_total += 1
            refusal = classify_refusal_sample(gt_record, predict_record)
            refusal_metrics.append(refusal)
            sample_errors: List[Dict[str, Any]] = []

            if refusal["gt_label"] or refusal["pred_label"]:
                sample_metrics.append(
                    make_sample_metric(
                        filename=filename,
                        idx=idx,
                        question_type=gt_record.get("QuestionType"),
                        pair_status="paired",
                        stem=None,
                        formula=None,
                        reading_order=None,
                        refusal=refusal,
                        errors=sample_errors,
                    )
                )
                continue

            stem = compute_stem_metrics(gt_record, predict_record, args.match_threshold)
            formula = compute_formula_metrics(gt_record, predict_record, args.w_slt, args.w_opt)
            # Reading-order continuity is measured on the matched subsequence so
            # unmatched noise does not create artificial BCS breaks.
            reading_order = compute_reading_order_metrics(gt_record, predict_record)

            stem_metrics.append(stem)
            formula_metrics.append(formula)
            reading_order_metrics.append(reading_order)

            if formula is not None:
                for formula_error in formula["errors"]:
                    error_item = {
                        "reason": formula_error["reason"],
                        "filename": filename,
                        "idx": idx,
                        "question_id": formula_error["question_id"],
                        "gt_seq": formula_error["gt_seq"],
                        "predict_formula_seq": formula_error["predict_formula_seq"],
                    }
                    sample_errors.append(error_item)
                    model_errors.append(error_item)

            sample_metrics.append(
                make_sample_metric(
                    filename=filename,
                    idx=idx,
                    question_type=gt_record.get("QuestionType"),
                    pair_status="paired",
                    stem=stem,
                    formula=formula,
                    reading_order=reading_order,
                    refusal=refusal,
                    errors=sample_errors,
                )
            )

        extra_predict_errors = [
            {
                "reason": "extra_predict",
                "filename": filename,
                "idx": _normalize_idx(record.get("idx")),
            }
            for filename, record in sorted(predict_index.items())
            if filename not in gt_index
        ]
        model_errors.extend(extra_predict_errors)

        counts = {
            "gt_total": len(gt_records),
            "predict_total": predict_total,
            "paired_total": paired_total,
            "missing_predict_count": missing_predict_count,
            "extra_predict_count": len(extra_predict_errors),
            "sample_coverage": _sample_coverage(paired_total, len(gt_records)),
        }

        stem_aggregate = aggregate_stem_metrics(stem_metrics)
        formula_aggregate = aggregate_formula_metrics(formula_metrics)
        reading_order_aggregate = aggregate_reading_order_metrics(reading_order_metrics)
        refusal_aggregate = aggregate_refusal_metrics(refusal_metrics)

        composite = compute_composite_score(
            {
                "stem_acc": stem_aggregate["overall_stem_acc"],
                "matched_formula_score": formula_aggregate["matched_formula_score"],
                "ros": reading_order_aggregate["overall_ros"],
                "refusal_f1": refusal_aggregate["overall_refusal_f1"],
            },
            alpha_stem=args.alpha_stem,
            alpha_formula=args.alpha_formula,
            alpha_ros=args.alpha_ros,
            alpha_refusal=args.alpha_refusal,
        )

        # stem_acc now measures matched-content similarity only; coverage remains available via sample fields.
        overall = {
            "stem_acc": stem_aggregate["overall_stem_acc"],
            "slt_teds": formula_aggregate["overall_slt_teds"],
            "opt_teds": formula_aggregate["overall_opt_teds"],
            "formula_score": formula_aggregate["overall_formula_score"],
            "formula_slot_count": formula_aggregate["formula_slot_count"],
            "formula_partial_slot_count": formula_aggregate["formula_partial_slot_count"],
            "formula_opaque_slot_count": formula_aggregate["formula_opaque_slot_count"],
            "matched_slt_teds": formula_aggregate["matched_slt_teds"],
            "matched_opt_teds": formula_aggregate["matched_opt_teds"],
            "matched_formula_score": formula_aggregate["matched_formula_score"],
            "matched_formula_slot_count": formula_aggregate["matched_formula_slot_count"],
            "full_parse_slt_teds": formula_aggregate["full_parse_slt_teds"],
            "full_parse_opt_teds": formula_aggregate["full_parse_opt_teds"],
            "full_parse_formula_score": formula_aggregate["full_parse_formula_score"],
            "full_parse_formula_slot_count": formula_aggregate["full_parse_formula_slot_count"],
            "mcr": reading_order_aggregate["overall_mcr"],
            "bcs": reading_order_aggregate["overall_bcs"],
            "sqa": reading_order_aggregate["overall_sqa"],
            "iqa": reading_order_aggregate["overall_iqa"],
            "ros": reading_order_aggregate["overall_ros"],
            "ros_coverage": reading_order_aggregate["ros_coverage"],
            "refusal_precision": refusal_aggregate["overall_refusal_precision"],
            "refusal_recall": refusal_aggregate["overall_refusal_recall"],
            "refusal_f1": refusal_aggregate["overall_refusal_f1"],
            "hallucination_rate": refusal_aggregate["overall_hallucination_rate"],
            "composite_score": composite["composite_score"],
            "w_total_stem": args.alpha_stem,
            "w_total_formula": args.alpha_formula,
            "w_total_ros": args.alpha_ros,
            "w_total_refusal": args.alpha_refusal,
        }

        overall_metrics = {
            "model": model_name,
            "counts": counts,
            "overall": overall,
        }
        error_report = {
            "model": model_name,
            "summary": {
                "predict_total": predict_total,
                "paired_total": paired_total,
                "missing_predict_count": missing_predict_count,
                "extra_predict_count": len(extra_predict_errors),
                "sample_error_count": len(model_errors),
                "fatal_error_count": 0,
                "sample_coverage": counts["sample_coverage"],
            },
            "errors": model_errors,
        }

        atomic_write_json(model_output_dir / "sample_metrics.json", sample_metrics)
        atomic_write_json(model_output_dir / "overall_metrics.json", overall_metrics)
        atomic_write_json(model_output_dir / "eval_errors.json", error_report)

        return {
            "model": model_name,
            "overall": overall,
            "counts": {
                "sample_coverage": counts["sample_coverage"],
                "missing_predict_count": counts["missing_predict_count"],
            },
        }

    except (FileNotFoundError, json.JSONDecodeError, ValueError, FormulaParseError) as exc:
        error_report = fatal_error_report(
            model_name=model_name,
            message=str(exc),
            predict_total=predict_total,
        )
        atomic_write_json(model_output_dir / "eval_errors.json", error_report)
        return None


def leaderboard_sort_key(item: Dict[str, Any]) -> Tuple[int, float, str]:
    composite_score = item["overall"].get("composite_score")
    if composite_score is None:
        return (1, 0.0, item["model"])
    return (0, -composite_score, item["model"])


def make_leaderboard_overall(overall: Dict[str, Any]) -> Dict[str, Any]:
    leaderboard_overall = dict(overall)
    leaderboard_overall["slt_teds"] = overall.get("matched_slt_teds")
    leaderboard_overall["opt_teds"] = overall.get("matched_opt_teds")
    leaderboard_overall["formula_score"] = overall.get("matched_formula_score")
    return leaderboard_overall


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        validate_weights(args)

        gt_path = resolve_path(args.gt)
        predict_root = resolve_path(args.predict_root)
        output_root = resolve_path(args.output_root)

        if not gt_path.exists():
            raise FileNotFoundError(f"GT file does not exist: {display_path(gt_path)}")
        if not predict_root.exists():
            raise FileNotFoundError(f"predict root does not exist: {display_path(predict_root)}")
        if not predict_root.is_dir():
            raise ValueError(f"predict root is not a directory: {display_path(predict_root)}")

        gt_records = load_json_array(gt_path)
        gt_index = build_record_index(gt_records, "GT")
        models = discover_models(predict_root, args.model)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    leaderboard_entries: List[Dict[str, Any]] = []
    for model_name, predict_path in models:
        leaderboard_item = evaluate_model(
            model_name=model_name,
            predict_path=predict_path,
            gt_records=gt_records,
            gt_index=gt_index,
            output_root=output_root,
            args=args,
        )
        if leaderboard_item is not None:
            leaderboard_item = dict(leaderboard_item)
            leaderboard_item["overall"] = make_leaderboard_overall(leaderboard_item["overall"])
            leaderboard_entries.append(leaderboard_item)

    leaderboard_entries.sort(key=leaderboard_sort_key)
    atomic_write_json(output_root / "leaderboard_summary.json", leaderboard_entries)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
