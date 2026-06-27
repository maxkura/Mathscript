#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parent.parent
COLUMNS = [
    "model",
    "stem_acc",
    "slt_teds",
    "opt_teds",
    "formula_score",
    "bcs",
    "sqa",
    "iqa",
    "ros",
    "ros_coverage",
    "refusal_f1",
    "composite_score",
]


def format_value(value):
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return value


def extract_rows(data):
    if not isinstance(data, list):
        raise ValueError("leaderboard_summary.json must contain a list of records")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        overall = item.get("overall", {})
        row = {"model": item.get("model", "")}
        for column in COLUMNS[1:]:
            row[column] = format_value(overall.get(column, ""))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract selected leaderboard fields from JSON and save them as CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(BASE_DIR / "leaderboard_summary.json"),
        help="Path to leaderboard_summary.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(BASE_DIR / "leaderboard_summary.csv"),
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = extract_rows(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written to: {output_path}")
    print(f"Rows exported: {len(rows)}")


if __name__ == "__main__":
    main()
