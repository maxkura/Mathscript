#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_INPUTS: Sequence[Tuple[str, str]] = (
    ("0", "output/metric_threshold_0/leaderboard_summary.json"),
    ("0.3", "output/metric_threshold_0_3/leaderboard_summary.json"),
    ("0.5", "output/metric_threshold_0_5/leaderboard_summary.json"),
    ("0.7", "output/metric_threshold_0_7/leaderboard_summary.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-threshold and combined leaderboard tables from eval outputs."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("THRESHOLD", "LEADERBOARD_JSON"),
        help="Override inputs as repeated pairs of threshold and leaderboard_summary.json path.",
    )
    parser.add_argument(
        "--tables-markdown-out",
        default="output/leaderboard_threshold_tables.md",
        help="Markdown path for all threshold tables and the combined table.",
    )
    parser.add_argument(
        "--comparison-markdown-out",
        default="output/leaderboard_threshold_comparison.md",
        help="Markdown path for the combined threshold comparison table.",
    )
    parser.add_argument(
        "--comparison-csv-out",
        default="output/leaderboard_threshold_comparison.csv",
        help="CSV path for the combined threshold comparison table.",
    )
    parser.add_argument(
        "--per-threshold-csv-dir",
        default="output/leaderboard_threshold_tables_csv",
        help="Directory for per-threshold leaderboard CSV files.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def build_threshold_rows(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, entry in enumerate(entries, start=1):
        overall = entry["overall"]
        counts = entry.get("counts", {})
        rows.append(
            {
                "rank": rank,
                "model": entry["model"],
                "composite_score": fmt_float(overall.get("composite_score")),
                "stem_acc": fmt_float(overall.get("stem_acc")),
                "formula_score": fmt_float(overall.get("formula_score")),
                "ros": fmt_float(overall.get("ros")),
                "refusal_f1": fmt_float(overall.get("refusal_f1")),
                "sample_coverage": fmt_float(counts.get("sample_coverage")),
                "missing_predict_count": counts.get("missing_predict_count", ""),
            }
        )
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def build_comparison_rows(
    threshold_entries: Sequence[Tuple[str, Sequence[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = {}

    for threshold, entries in threshold_entries:
        for rank, entry in enumerate(entries, start=1):
            model_row = by_model.setdefault("model::" + entry["model"], {"model": entry["model"]})
            model_row[f"t{threshold}_rank"] = rank
            model_row[f"t{threshold}_score"] = float(entry["overall"]["composite_score"])

    rows: List[Dict[str, Any]] = []
    for row in by_model.values():
        score_values = [value for key, value in row.items() if key.endswith("_score")]
        rank_values = [value for key, value in row.items() if key.endswith("_rank")]
        row["avg_rank"] = sum(rank_values) / len(rank_values) if rank_values else None
        row["avg_score"] = sum(score_values) / len(score_values) if score_values else None
        rows.append(row)

    def sort_key(item: Dict[str, Any]) -> Tuple[float, float, str]:
        avg_rank = float(item["avg_rank"]) if item["avg_rank"] is not None else float("inf")
        avg_score = float(item["avg_score"]) if item["avg_score"] is not None else float("-inf")
        return (avg_rank, -avg_score, item["model"])

    rows.sort(key=sort_key)
    return rows


def build_tables_markdown(
    threshold_rows: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
    comparison_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[str],
) -> str:
    sections: List[str] = ["# Match Threshold Eval Tables"]

    headers = [
        "Rank",
        "Model",
        "Composite",
        "Stem Acc",
        "Formula Score",
        "ROS",
        "Refusal F1",
        "Coverage",
        "Missing Predict",
    ]

    for threshold, rows in threshold_rows:
        sections.append(f"## Match Threshold = {threshold}")
        sections.append(
            markdown_table(
                headers,
                [
                    [
                        row["rank"],
                        row["model"],
                        row["composite_score"],
                        row["stem_acc"],
                        row["formula_score"],
                        row["ros"],
                        row["refusal_f1"],
                        row["sample_coverage"],
                        row["missing_predict_count"],
                    ]
                    for row in rows
                ],
            )
        )

    comparison_headers = ["Model"]
    for threshold in thresholds:
        comparison_headers.extend([f"t={threshold} Rank", f"t={threshold} Score"])
    comparison_headers.extend(["Avg Rank", "Avg Score"])

    sections.append("## Combined Comparison")
    sections.append(
        markdown_table(
            comparison_headers,
            [
                [
                    row["model"],
                    *[
                        row.get(f"t{threshold}_rank", "")
                        if key_index % 2 == 0
                        else fmt_float(row.get(f"t{threshold}_score"))
                        for threshold in thresholds
                        for key_index in range(2)
                    ],
                    fmt_float(row.get("avg_rank")),
                    fmt_float(row.get("avg_score")),
                ]
                for row in comparison_rows
            ],
        )
    )
    sections.append("")
    return "\n\n".join(sections)


def build_comparison_markdown(comparison_rows: Sequence[Dict[str, Any]], thresholds: Sequence[str]) -> str:
    headers = ["Model"]
    for threshold in thresholds:
        headers.extend([f"t={threshold} Rank", f"t={threshold} Score"])
    headers.extend(["Avg Rank", "Avg Score"])

    rows = []
    for row in comparison_rows:
        table_row: List[Any] = [row["model"]]
        for threshold in thresholds:
            table_row.append(row.get(f"t{threshold}_rank", ""))
            table_row.append(fmt_float(row.get(f"t{threshold}_score")))
        table_row.append(fmt_float(row.get("avg_rank")))
        table_row.append(fmt_float(row.get("avg_score")))
        rows.append(table_row)

    return "\n".join(
        [
            "# Leaderboard Comparison Across Match Thresholds",
            "",
            markdown_table(headers, rows),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    inputs = args.input if args.input else list(DEFAULT_INPUTS)

    threshold_entries: List[Tuple[str, Sequence[Dict[str, Any]]]] = []
    threshold_rows: List[Tuple[str, Sequence[Dict[str, Any]]]] = []
    thresholds: List[str] = []

    per_threshold_csv_dir = resolve_path(args.per_threshold_csv_dir)
    per_threshold_csv_dir.mkdir(parents=True, exist_ok=True)

    for threshold, leaderboard_path_value in inputs:
        leaderboard_path = resolve_path(leaderboard_path_value)
        entries = load_json(leaderboard_path)
        if not isinstance(entries, list):
            raise ValueError(f"{leaderboard_path} must contain a top-level JSON array.")

        rows = build_threshold_rows(entries)
        threshold_entries.append((threshold, entries))
        threshold_rows.append((threshold, rows))
        thresholds.append(threshold)

        csv_name = f"leaderboard_threshold_{threshold.replace('.', '_')}.csv"
        write_csv(
            per_threshold_csv_dir / csv_name,
            [
                "rank",
                "model",
                "composite_score",
                "stem_acc",
                "formula_score",
                "ros",
                "refusal_f1",
                "sample_coverage",
                "missing_predict_count",
            ],
            rows,
        )

    comparison_rows = build_comparison_rows(threshold_entries)
    comparison_csv_rows: List[Dict[str, Any]] = []
    comparison_fieldnames = ["model"]
    for threshold in thresholds:
        comparison_fieldnames.extend([f"t{threshold}_rank", f"t{threshold}_score"])
    comparison_fieldnames.extend(["avg_rank", "avg_score"])

    for row in comparison_rows:
        csv_row: Dict[str, Any] = {"model": row["model"]}
        for threshold in thresholds:
            csv_row[f"t{threshold}_rank"] = row.get(f"t{threshold}_rank", "")
            csv_row[f"t{threshold}_score"] = fmt_float(row.get(f"t{threshold}_score"))
        csv_row["avg_rank"] = fmt_float(row.get("avg_rank"))
        csv_row["avg_score"] = fmt_float(row.get("avg_score"))
        comparison_csv_rows.append(csv_row)

    write_text(
        resolve_path(args.tables_markdown_out),
        build_tables_markdown(threshold_rows, comparison_rows, thresholds),
    )
    write_text(
        resolve_path(args.comparison_markdown_out),
        build_comparison_markdown(comparison_rows, thresholds),
    )
    write_csv(resolve_path(args.comparison_csv_out), comparison_fieldnames, comparison_csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
