#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from extraction_utils import atomic_write_json, extract_results, load_json_array
from remove_transcription_spaces import clean_transcription_spaces


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "data" / "GT"
DEFAULT_INPUT = BASE_DIR / "results.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "extracted_gt.json"
DEFAULT_ERROR_REPORT = DEFAULT_OUTPUT_DIR / "extraction_errors.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured GT records from annotation/results.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source results.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the extracted GT output file.",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=DEFAULT_ERROR_REPORT,
        help="Path to the extraction error report file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        original_records = load_json_array(args.input)
        cleaned_records = copy.deepcopy(original_records)
        clean_transcription_spaces(cleaned_records)
        extracted_records, error_report = extract_results(original_records, cleaned_records)
        atomic_write_json(args.output, extracted_records)
        atomic_write_json(args.error_report, error_report)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    summary = error_report["summary"]
    print(f"Input file: {args.input}")
    print(f"GT output: {args.output}")
    print(f"Error report: {args.error_report}")
    print(f"Input records: {summary['input_records']}")
    print(f"Processed success records: {summary['processed_success_records']}")
    print(f"Extracted records: {summary['extracted_records']}")
    print(f"Structured extracted records: {summary['structured_extracted_records']}")
    print(f"Passthrough records: {summary['passthrough_records']}")
    print(
        "Passthrough non-constructed records: "
        f"{summary['passthrough_non_constructed_records']}"
    )
    print(f"Passthrough UNK records: {summary['passthrough_unk_records']}")
    print(f"Error records: {summary['error_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
