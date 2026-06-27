#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CONSTRUCTED_RESPONSE = "ConstructedResponse"
PASSTHROUGH_QUESTION_TYPES = ("MultipleChoice", "FillBlank")
VALID_QUESTION_TYPES = PASSTHROUGH_QUESTION_TYPES + (CONSTRUCTED_RESPONSE,)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
MATH_PATTERN = re.compile(r"\[\[MATH:(.*?)\]\]")
SPACE_PATTERN = re.compile(r"[ \u3000]+")
INLINE_ESCAPED_NEWLINE_PATTERN = re.compile(r"(?<!\\)\\n")
INLINE_ESCAPED_CARRIAGE_PATTERN = re.compile(r"(?<!\\)\\r")
INLINE_ESCAPED_TAB_PATTERN = re.compile(r"(?<!\\)\\t")
DOUBLE_ESCAPED_NEWLINE_PATTERN = re.compile(r"\\\\n")
DOUBLE_ESCAPED_CARRIAGE_PATTERN = re.compile(r"\\\\r")
DOUBLE_ESCAPED_TAB_PATTERN = re.compile(r"\\\\t")
DEFAULT_OUTPUT_ROOT = Path("output") / "predict"
DEFAULT_DENOISED_INPUT_NAME = "result_less_format_noise.json"
DEFAULT_RAW_INPUT_NAME = "results.json"
DEFAULT_GT_PATH = Path("data") / "GT" / "extracted_gt.json"
DEFAULT_MATCH_THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert predict results into standardized predict records."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name used to derive default input/output paths in single-model mode.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Batch-convert every model directory under output-root that contains "
            f"{DEFAULT_DENOISED_INPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Predict input path in single-model mode. Defaults to "
            f"output/predict/<model>/{DEFAULT_DENOISED_INPUT_NAME} and falls back to "
            f"output/predict/<model>/{DEFAULT_RAW_INPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for predict inputs/outputs. Default: output/predict",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Formal predict output path. Defaults to output/predict/<model>/predict.json.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_PATH,
        help="GT file path. Default: data/GT/extracted_gt.json",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=None,
        help=(
            "Conversion error report path. Defaults to "
            "output/predict/<model>/predict_conversion_errors.json."
        ),
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Minimum similarity score for a valid GT/predict line match.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return BENCHMARK_DIR / path


def resolve_output_path(output_root: Path, output: Optional[Path], model: str) -> Path:
    if output is not None:
        return resolve_path(output)
    return resolve_path(output_root / model / "predict.json")


def resolve_default_input_path(output_root: Path, model: str) -> Path:
    model_dir = resolve_path(output_root / model)
    denoised_input_path = model_dir / DEFAULT_DENOISED_INPUT_NAME
    if denoised_input_path.exists():
        return denoised_input_path
    return model_dir / DEFAULT_RAW_INPUT_NAME


def resolve_input_path(output_root: Path, input_path: Optional[Path], model: str) -> Path:
    if input_path is not None:
        return resolve_path(input_path)
    return resolve_default_input_path(output_root, model)


def resolve_error_report_path(
    output_root: Path,
    error_report: Optional[Path],
    model: str,
) -> Path:
    if error_report is not None:
        return resolve_path(error_report)
    return resolve_path(output_root / model / "predict_conversion_errors.json")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BENCHMARK_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def discover_batch_model_directories(output_root: Path) -> Tuple[List[Path], List[Path]]:
    resolved_root = resolve_path(output_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Predict root not found: {display_path(resolved_root)}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Predict root is not a directory: {display_path(resolved_root)}")

    discovered_dirs = sorted(
        [path for path in resolved_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    process_dirs = [
        path for path in discovered_dirs if (path / DEFAULT_DENOISED_INPUT_NAME).is_file()
    ]
    skipped_dirs = [
        path for path in discovered_dirs if not (path / DEFAULT_DENOISED_INPUT_NAME).is_file()
    ]
    return process_dirs, skipped_dirs


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


def strip_code_fences(raw: str) -> str:
    stripped = raw.strip()
    return CODE_FENCE_PATTERN.sub("", stripped).strip()


def contains_unk_marker(value: Any) -> bool:
    return isinstance(value, str) and "[UNK]" in value


def get_record_filename(record: Dict[str, Any]) -> str:
    for key in ("ImgReal", "filename"):
        value = normalize_filename(record.get(key))
        if value:
            return value
    return ""


def load_json_array(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {display_path(path)}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"JSON parse failed for {display_path(path)}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Input JSON top-level value must be a JSON array: {display_path(path)}"
        )

    return data


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


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


def _jsonish_quote_start(raw: str, key: str) -> int:
    marker = f'"{key}"'
    key_index = raw.find(marker)
    if key_index < 0:
        return -1
    colon_index = raw.find(":", key_index + len(marker))
    if colon_index < 0:
        return -1
    return raw.find('"', colon_index + 1)


def _extract_jsonish_quoted_value(raw: str, key: str) -> Optional[str]:
    quote_index = _jsonish_quote_start(raw, key)
    if quote_index < 0:
        return None

    cursor = quote_index + 1
    pieces: List[str] = []
    escaped = False
    while cursor < len(raw):
        char = raw[cursor]
        if escaped:
            pieces.append("\\" + char)
            escaped = False
            cursor += 1
            continue
        if char == "\\":
            escaped = True
            cursor += 1
            continue
        if char == '"':
            lookahead = raw[cursor + 1 : cursor + 96]
            if re.match(r"\s*(?:,|\}|$)", lookahead):
                return "".join(pieces)
            pieces.append(char)
            cursor += 1
            continue
        pieces.append(char)
        cursor += 1

    return "".join(pieces) or None


def _extract_jsonish_scalar_value(raw: str, key: str) -> Optional[str]:
    marker = f'"{key}"'
    key_index = raw.find(marker)
    if key_index < 0:
        return None
    colon_index = raw.find(":", key_index + len(marker))
    if colon_index < 0:
        return None
    cursor = colon_index + 1
    while cursor < len(raw) and raw[cursor].isspace():
        cursor += 1
    end = cursor
    while end < len(raw) and raw[end] not in ",}\n\r":
        end += 1
    return raw[cursor:end].strip() or None


def _decode_jsonish_string(value: str) -> str:
    text = value
    text = DOUBLE_ESCAPED_NEWLINE_PATTERN.sub("\n", text)
    text = DOUBLE_ESCAPED_CARRIAGE_PATTERN.sub("\r", text)
    text = DOUBLE_ESCAPED_TAB_PATTERN.sub("\t", text)
    text = INLINE_ESCAPED_NEWLINE_PATTERN.sub("\n", text)
    text = INLINE_ESCAPED_CARRIAGE_PATTERN.sub("\r", text)
    text = INLINE_ESCAPED_TAB_PATTERN.sub("\t", text)
    text = text.replace('\\"', '"')
    text = text.replace("\\\\", "\\")
    return text


def extract_transcription_from_raw_output(raw_output: Any) -> Optional[str]:
    cleaned = strip_code_fences(normalize_text(raw_output).strip())
    if not cleaned:
        return None
    for key in ("transcription", "trans"):
        value = _extract_jsonish_quoted_value(cleaned, key)
        if value:
            return _decode_jsonish_string(value)
    return None


def salvage_record_from_raw_output(
    record: Dict[str, Any],
    gt_record: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    raw_output = normalize_text(record.get("raw_output")).strip()
    if not raw_output:
        return None

    cleaned = strip_code_fences(raw_output)
    transcription = None
    for key in ("transcription", "trans"):
        transcription = _extract_jsonish_quoted_value(cleaned, key)
        if transcription:
            break
    if not transcription:
        return None

    salvaged = copy.deepcopy(record)
    final_answer = _extract_jsonish_quoted_value(cleaned, "final_answer")
    question_type = _extract_jsonish_quoted_value(cleaned, "QuestionType")
    idx_raw = _extract_jsonish_scalar_value(cleaned, "idx")

    salvaged["predict_status"] = "success"
    salvaged["error_message"] = ""
    salvaged["transcription"] = _decode_jsonish_string(transcription)
    salvaged["final_answer"] = _decode_jsonish_string(final_answer or "")
    salvaged["QuestionType"] = (
        question_type
        or normalize_text(salvaged.get("QuestionType"))
        or normalize_text((gt_record or {}).get("QuestionType"))
    )
    salvaged["idx"] = (
        normalize_idx((gt_record or {}).get("idx"))
        or normalize_idx(idx_raw)
        or normalize_idx(salvaged.get("idx"))
    )
    if gt_record is not None:
        filename = normalize_filename(gt_record.get("filename"))
        salvaged["filename"] = filename
        salvaged["ImgReal"] = filename
    return salvaged


def normalize_question_type(question_type: Any, gt_record: Optional[Dict[str, Any]]) -> str:
    normalized = normalize_text(question_type).strip()
    if normalized in VALID_QUESTION_TYPES:
        return normalized
    if normalized in {"", "[UNK]"} and gt_record is not None:
        gt_question_type = normalize_text(gt_record.get("QuestionType")).strip()
        if gt_question_type in VALID_QUESTION_TYPES:
            return gt_question_type
    return normalized


def normalize_line_for_matching(value: Any) -> str:
    normalized = normalize_text(value)
    normalized = SPACE_PATTERN.sub("", normalized)
    normalized = MATH_PATTERN.sub(r"\1", normalized)
    return normalized.strip()


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current_row = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[column_index - 1] + 1
            delete_cost = previous_row[column_index] + 1
            replace_cost = previous_row[column_index - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    return previous_row[-1]


def line_similarity(predict_text: Any, gt_text: Any) -> float:
    normalized_predict = normalize_line_for_matching(predict_text)
    normalized_gt = normalize_line_for_matching(gt_text)
    denominator = max(len(normalized_predict), len(normalized_gt))
    if denominator == 0:
        return 1.0
    distance = levenshtein_distance(normalized_predict, normalized_gt)
    return 1.0 - (distance / denominator)


def hungarian_minimize(cost_matrix: List[List[float]]) -> List[int]:
    size = len(cost_matrix)
    if size == 0:
        return []

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for row in range(1, size + 1):
        p[0] = row
        column = 0
        min_values = [float("inf")] * (size + 1)
        used = [False] * (size + 1)

        while True:
            used[column] = True
            matched_row = p[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = (
                    cost_matrix[matched_row - 1][candidate_column - 1]
                    - u[matched_row]
                    - v[candidate_column]
                )
                if current < min_values[candidate_column]:
                    min_values[candidate_column] = current
                    way[candidate_column] = column
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    min_values[candidate_column] -= delta

            column = next_column
            if p[column] == 0:
                break

        while True:
            previous_column = way[column]
            p[column] = p[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def normalize_gt_transcription_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("GT transcription must be a list for ConstructedResponse records.")

    normalized_items: List[Dict[str, Any]] = []
    seen_sequences: set[int] = set()

    for item_index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"GT transcription item {item_index} must be a JSON object.")
        sequence = normalize_idx(item.get("seq"))
        if sequence is None:
            raise ValueError(f"GT transcription item {item_index} has an invalid seq.")
        if sequence in seen_sequences:
            raise ValueError(f"GT transcription contains duplicated seq {sequence}.")
        seen_sequences.add(sequence)
        normalized_items.append(
            {
                "seq": sequence,
                "question_id": item.get("question_id"),
                "content": normalize_text(item.get("content")),
            }
        )

    return normalized_items


def split_predict_transcription(transcription: str) -> List[Dict[str, Any]]:
    lines = [line for line in transcription.split("\n") if line]
    return [
        {
            "seq": sequence,
            "question_id": None,
            "gt_seq": None,
            "match_status": "unmatched",
            "match_score": None,
            "content": content,
        }
        for sequence, content in enumerate(lines, start=1)
    ]


def compute_similarity_matrix(
    gt_items: List[Dict[str, Any]],
    predict_items: List[Dict[str, Any]],
) -> List[List[float]]:
    return [
        [
            line_similarity(predict_item["content"], gt_item["content"])
            for predict_item in predict_items
        ]
        for gt_item in gt_items
    ]


def compute_best_predict_scores(
    similarity_matrix: List[List[float]],
    predict_count: int,
) -> List[Optional[float]]:
    if predict_count == 0:
        return []
    if not similarity_matrix:
        return [None] * predict_count
    return [
        max(row[predict_index] for row in similarity_matrix)
        for predict_index in range(predict_count)
    ]


def compute_best_matches(
    gt_items: List[Dict[str, Any]],
    predict_items: List[Dict[str, Any]],
    threshold: float,
    similarity_matrix: Optional[List[List[float]]] = None,
) -> Dict[int, Tuple[int, float]]:
    gt_count = len(gt_items)
    predict_count = len(predict_items)
    if gt_count == 0 or predict_count == 0:
        return {}

    size = max(gt_count, predict_count)
    weights = [[0.0 for _ in range(size)] for _ in range(size)]

    if similarity_matrix is None:
        similarity_matrix = compute_similarity_matrix(gt_items, predict_items)

    for gt_index in range(gt_count):
        for predict_index in range(predict_count):
            score = similarity_matrix[gt_index][predict_index]
            if score >= threshold:
                weights[gt_index][predict_index] = score

    cost_matrix = [
        [1.0 - weights[row][column] for column in range(size)]
        for row in range(size)
    ]
    assignment = hungarian_minimize(cost_matrix)

    matches: Dict[int, Tuple[int, float]] = {}
    for gt_index in range(gt_count):
        predict_index = assignment[gt_index]
        if predict_index < 0 or predict_index >= predict_count:
            continue
        score = weights[gt_index][predict_index]
        if score > 0.0:
            matches[predict_index] = (gt_index, score)
    return matches


def align_predict_items(
    gt_items: List[Dict[str, Any]],
    predict_items: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    similarity_matrix = compute_similarity_matrix(gt_items, predict_items)
    best_scores = compute_best_predict_scores(similarity_matrix, len(predict_items))
    matches = compute_best_matches(
        gt_items,
        predict_items,
        threshold,
        similarity_matrix=similarity_matrix,
    )
    aligned_items: List[Dict[str, Any]] = []

    for predict_index, item in enumerate(predict_items):
        matched = matches.get(predict_index)
        if matched is None:
            aligned_items.append(
                {
                    **item,
                    "question_id": None,
                    "gt_seq": None,
                    "match_status": "unmatched",
                    "match_score": best_scores[predict_index],
                }
            )
            continue

        gt_index, score = matched
        gt_item = gt_items[gt_index]
        aligned_items.append(
            {
                **item,
                "question_id": gt_item.get("question_id"),
                "gt_seq": gt_item["seq"],
                "match_status": "matched",
                "match_score": score,
            }
        )

    return aligned_items


def extract_formula_list(
    transcription_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned_items: List[Dict[str, Any]] = []
    formula_list: List[Dict[str, Any]] = []
    formula_sequence = 1

    for item in transcription_items:
        content = normalize_text(item.get("content"))
        for match in MATH_PATTERN.finditer(content):
            formula_list.append(
                {
                    "formula_seq": formula_sequence,
                    "seq": item.get("seq"),
                    "question_id": item.get("question_id"),
                    "gt_seq": item.get("gt_seq"),
                    "formula": match.group(1),
                }
            )
            formula_sequence += 1

        cleaned_items.append(
            {
                **item,
                "content": MATH_PATTERN.sub(r"\1", content),
            }
        )

    return cleaned_items, formula_list


def build_output_record(
    *,
    filename: str,
    idx: Optional[int],
    question_type: str,
    transcription: Any,
    formula_list: List[Dict[str, Any]],
    final_answer: str,
    conversion_status: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "filename": filename,
        "idx": idx,
        "QuestionType": question_type,
        "transcription": transcription,
        "formula_list": formula_list,
        "final_answer": final_answer,
    }
    if conversion_status is not None:
        record["conversion_status"] = conversion_status
    return record


def build_error_item(record: Any, reason: str, details: str) -> Dict[str, Any]:
    filename = ""
    idx: Optional[int] = None
    predict_status = ""
    if isinstance(record, dict):
        filename = get_record_filename(record)
        idx = normalize_idx(record.get("idx"))
        predict_status = normalize_text(record.get("predict_status"))

    return {
        "filename": filename,
        "idx": idx,
        "predict_status": predict_status,
        "reason": reason,
        "details": details,
        "record": record,
    }


def build_gt_index(
    gt_records: List[Any],
) -> Tuple[Dict[str, Dict[str, Any]], set[str]]:
    gt_index: Dict[str, Dict[str, Any]] = {}
    duplicate_keys: set[str] = set()

    for record_index, record in enumerate(gt_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"GT record {record_index} must be a JSON object.")
        filename = get_record_filename(record)
        if not filename:
            raise ValueError(f"GT record {record_index} must contain a valid filename.")
        if filename in gt_index:
            duplicate_keys.add(filename)
            continue
        gt_index[filename] = record

    return gt_index, duplicate_keys


def resolve_gt_record(
    record: Dict[str, Any],
    gt_index: Dict[str, Dict[str, Any]],
    duplicate_keys: set[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    filename = get_record_filename(record)
    if not filename:
        return None, "missing_filename", "Predict record filename is missing or empty."

    if filename in duplicate_keys:
        return (
            None,
            "duplicate_gt_key",
            f"GT contains duplicated records for filename {filename!r}.",
        )

    gt_record = gt_index.get(filename)
    if gt_record is None:
        return (
            None,
            "missing_gt_record",
            f"No GT record was found for filename {filename!r}.",
        )

    return gt_record, None, None


def convert_record(
    record: Dict[str, Any],
    gt_index: Dict[str, Dict[str, Any]],
    duplicate_keys: set[str],
    match_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    gt_record, reason, details = resolve_gt_record(record, gt_index, duplicate_keys)
    if gt_record is None:
        return None, reason, details, None

    transcription_from_raw_output = extract_transcription_from_raw_output(record.get("raw_output"))
    if not normalize_text(record.get("transcription")).strip() and transcription_from_raw_output:
        repaired_record = copy.deepcopy(record)
        repaired_record["transcription"] = transcription_from_raw_output
        if not normalize_text(repaired_record.get("final_answer")).strip():
            cleaned = strip_code_fences(normalize_text(repaired_record.get("raw_output")).strip())
            final_answer = _extract_jsonish_quoted_value(cleaned, "final_answer")
            if final_answer is not None:
                repaired_record["final_answer"] = _decode_jsonish_string(final_answer)
        record = repaired_record

    filename = get_record_filename(record)
    idx = normalize_idx(record.get("idx"))
    question_type = normalize_question_type(record.get("QuestionType"), gt_record)
    final_answer = normalize_text(record.get("final_answer"))

    if question_type in PASSTHROUGH_QUESTION_TYPES:
        return (
            build_output_record(
                filename=filename,
                idx=idx,
                question_type=question_type,
                transcription=normalize_text(record.get("transcription")),
                formula_list=[],
                final_answer=final_answer,
            ),
            None,
            None,
            "passthrough",
        )

    if question_type != CONSTRUCTED_RESPONSE:
        return (
            None,
            "invalid_question_type",
            (
                f"Predict record QuestionType must be one of {VALID_QUESTION_TYPES}, "
                f"got {question_type!r}."
            ),
            None,
        )

    transcription = record.get("transcription")
    if not isinstance(transcription, str):
        return (
            None,
            "invalid_transcription",
            "ConstructedResponse transcription must be a string.",
            None,
        )

    cleaned_record = copy.deepcopy(record)
    clean_transcription_spaces(cleaned_record)
    predict_items = split_predict_transcription(normalize_text(cleaned_record.get("transcription")))
    if not predict_items:
        return (
            None,
            "no_predict_segments",
            "No non-empty predict transcription segments remain after preprocessing.",
            None,
        )

    try:
        gt_items = normalize_gt_transcription_items(gt_record.get("transcription"))
    except ValueError as exc:
        if contains_unk_marker(gt_record.get("transcription")) or contains_unk_marker(
            gt_record.get("final_answer")
        ):
            return (
                build_output_record(
                    filename=filename,
                    idx=idx,
                    question_type=question_type,
                    transcription=normalize_text(record.get("transcription")),
                    formula_list=[],
                    final_answer=final_answer,
                    conversion_status="unknown_gt_passthrough",
                ),
                None,
                None,
                "unknown_gt_passthrough",
            )
        return None, "invalid_gt_transcription", str(exc), None

    aligned_items = align_predict_items(gt_items, predict_items, match_threshold)
    cleaned_items, formula_list = extract_formula_list(aligned_items)
    return (
        build_output_record(
            filename=filename,
            idx=idx,
            question_type=question_type,
            transcription=cleaned_items,
            formula_list=formula_list,
            final_answer=final_answer,
        ),
        None,
        None,
        "structured",
    )


def convert_results(
    raw_records: List[Any],
    gt_records: List[Any],
    match_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_index, duplicate_keys = build_gt_index(gt_records)

    converted_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    success_records = 0
    skipped_non_success_records = 0
    salvaged_non_success_records = 0
    recovered_question_type_records = 0
    structured_converted_records = 0
    passthrough_records = 0
    unknown_gt_passthrough_records = 0

    for record in raw_records:
        if not isinstance(record, dict):
            errors.append(
                build_error_item(
                    record,
                    "invalid_record_type",
                    "Predict record must be a JSON object.",
                )
            )
            continue

        predict_status = normalize_text(record.get("predict_status"))
        if predict_status == "success":
            success_records += 1
        else:
            gt_record: Optional[Dict[str, Any]] = None
            filename = get_record_filename(record)
            if filename and filename not in duplicate_keys:
                gt_record = gt_index.get(filename)
            salvaged_record = salvage_record_from_raw_output(record, gt_record)
            if salvaged_record is None:
                skipped_non_success_records += 1
                continue
            record = salvaged_record
            success_records += 1
            salvaged_non_success_records += 1

        original_question_type = normalize_text(record.get("QuestionType")).strip()
        gt_record_for_question_type: Optional[Dict[str, Any]] = None
        filename = get_record_filename(record)
        if filename and filename not in duplicate_keys:
            gt_record_for_question_type = gt_index.get(filename)
        normalized_question_type = normalize_question_type(
            record.get("QuestionType"),
            gt_record_for_question_type,
        )
        if normalized_question_type != original_question_type and normalized_question_type in VALID_QUESTION_TYPES:
            recovered_question_type_records += 1

        converted_record, reason, details, conversion_kind = convert_record(
            record,
            gt_index,
            duplicate_keys,
            match_threshold,
        )
        if converted_record is not None:
            converted_records.append(converted_record)
            if conversion_kind == "structured":
                structured_converted_records += 1
            elif conversion_kind == "passthrough":
                passthrough_records += 1
            elif conversion_kind == "unknown_gt_passthrough":
                passthrough_records += 1
                unknown_gt_passthrough_records += 1
            continue

        errors.append(
            build_error_item(
                record,
                reason or "unknown_error",
                details or "Unknown conversion failure.",
            )
        )

    error_report = {
        "summary": {
            "input_records": len(raw_records),
            "success_records": success_records,
            "skipped_non_success_records": skipped_non_success_records,
            "salvaged_non_success_records": salvaged_non_success_records,
            "converted_records": len(converted_records),
            "structured_converted_records": structured_converted_records,
            "passthrough_records": passthrough_records,
            "unknown_gt_passthrough_records": unknown_gt_passthrough_records,
            "recovered_question_type_records": recovered_question_type_records,
            "error_records": len(errors),
        },
        "errors": errors,
    }
    return converted_records, error_report


def validate_args(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    if not 0.0 <= args.match_threshold <= 1.0:
        return "--match-threshold must be between 0 and 1.", None

    if args.all_models:
        conflicting_options: List[str] = []
        if args.model is not None:
            conflicting_options.append("--model")
        if args.input is not None:
            conflicting_options.append("--input")
        if args.output is not None:
            conflicting_options.append("--output")
        if args.error_report is not None:
            conflicting_options.append("--error-report")
        if conflicting_options:
            joined = ", ".join(conflicting_options)
            return f"{joined} cannot be used with --all-models.", None
        return None, None

    if args.model is None:
        return "Either --model or --all-models must be provided.", None

    model = args.model.strip()
    if not model:
        return "--model must be a non-empty string.", None
    return None, model


def process_model_conversion(
    *,
    model: str,
    input_path: Path,
    output_path: Path,
    gt_path: Path,
    error_report_path: Path,
    gt_records: List[Any],
    match_threshold: float,
) -> Dict[str, Any]:
    raw_records = load_json_array(input_path)
    converted_records, error_report = convert_results(
        raw_records,
        gt_records,
        match_threshold,
    )
    atomic_write_json(output_path, converted_records)
    atomic_write_json(error_report_path, error_report)
    return {
        "model": model,
        "input_path": input_path,
        "gt_path": gt_path,
        "output_path": output_path,
        "error_report_path": error_report_path,
        "summary": error_report["summary"],
    }


def print_conversion_summary(result: Dict[str, Any], include_model_header: bool) -> None:
    summary = result["summary"]
    if include_model_header:
        print(f"Model: {result['model']}")
    print(f"Predict input: {display_path(result['input_path'])}")
    print(f"GT input: {display_path(result['gt_path'])}")
    print(f"Predict output: {display_path(result['output_path'])}")
    print(f"Error report: {display_path(result['error_report_path'])}")
    print(f"Input records: {summary['input_records']}")
    print(f"Success records: {summary['success_records']}")
    print(f"Skipped non-success records: {summary['skipped_non_success_records']}")
    print(f"Salvaged non-success records: {summary.get('salvaged_non_success_records', 0)}")
    print(f"Converted records: {summary['converted_records']}")
    print(f"Structured converted records: {summary['structured_converted_records']}")
    print(f"Passthrough records: {summary['passthrough_records']}")
    print(
        "Unknown GT passthrough records: "
        f"{summary['unknown_gt_passthrough_records']}"
    )
    print(
        "Recovered question type records: "
        f"{summary.get('recovered_question_type_records', 0)}"
    )
    print(f"Error records: {summary['error_records']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    error_message, model = validate_args(args)
    if error_message is not None:
        print(error_message, file=sys.stderr)
        return 1

    gt_path = resolve_path(args.gt)

    try:
        gt_records = load_json_array(gt_path)
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    if args.all_models:
        output_root = resolve_path(args.output_root)
        try:
            model_dirs, skipped_dirs = discover_batch_model_directories(output_root)
        except Exception as exc:
            print(f"Initialization failed: {exc}", file=sys.stderr)
            return 1

        if not model_dirs:
            print(
                f"No model directories with {DEFAULT_DENOISED_INPUT_NAME} found under "
                f"{display_path(output_root)}"
            )
            return 0

        print(f"Predict root: {display_path(output_root)}")
        print(f"GT input: {display_path(gt_path)}")
        print(f"Batch input filename: {DEFAULT_DENOISED_INPUT_NAME}")
        print(f"Models to process: {', '.join(path.name for path in model_dirs)}")
        if skipped_dirs:
            print(
                "Models skipped without "
                f"{DEFAULT_DENOISED_INPUT_NAME}: {', '.join(path.name for path in skipped_dirs)}"
            )

        processed_models = 0
        failed_models: List[str] = []

        for index, model_dir in enumerate(model_dirs):
            if index > 0:
                print()

            model_name = model_dir.name
            input_path = model_dir / DEFAULT_DENOISED_INPUT_NAME
            output_path = resolve_output_path(output_root, None, model_name)
            error_report_path = resolve_error_report_path(output_root, None, model_name)

            try:
                result = process_model_conversion(
                    model=model_name,
                    input_path=input_path,
                    output_path=output_path,
                    gt_path=gt_path,
                    error_report_path=error_report_path,
                    gt_records=gt_records,
                    match_threshold=args.match_threshold,
                )
            except Exception as exc:
                failed_models.append(model_name)
                print(
                    f"Model {model_name} failed: {exc}",
                    file=sys.stderr,
                )
                continue

            processed_models += 1
            print_conversion_summary(result, include_model_header=True)

        print()
        print("Completed batch predict conversion.")
        print(f"Model directories discovered: {len(model_dirs) + len(skipped_dirs)}")
        print(f"Models processed: {processed_models}")
        print(
            "Models skipped without "
            f"{DEFAULT_DENOISED_INPUT_NAME}: {len(skipped_dirs)}"
        )
        print(f"Models failed: {len(failed_models)}")
        if failed_models:
            print(f"Failed model names: {', '.join(failed_models)}")
            return 1
        return 0

    assert model is not None
    output_root = args.output_root
    input_path = resolve_input_path(output_root, args.input, model)
    output_path = resolve_output_path(output_root, args.output, model)
    error_report_path = resolve_error_report_path(output_root, args.error_report, model)

    try:
        result = process_model_conversion(
            model=model,
            input_path=input_path,
            output_path=output_path,
            gt_path=gt_path,
            error_report_path=error_report_path,
            gt_records=gt_records,
            match_threshold=args.match_threshold,
        )
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    print_conversion_summary(result, include_model_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
