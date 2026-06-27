#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


Q_PATTERN = re.compile(r"【q(\d{3})】")
MATH_PATTERN = re.compile(r"\[\[MATH:(.*?)\]\]")
CONSTRUCTED_RESPONSE = "ConstructedResponse"
PASSTHROUGH_QUESTION_TYPES = ("MultipleChoice", "FillBlank")
VALID_QUESTION_TYPES = PASSTHROUGH_QUESTION_TYPES + (CONSTRUCTED_RESPONSE,)


def load_json_array(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Input JSON top-level value must be a JSON array.")

    return data


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _string_or_empty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def build_error_item(
    record: Any,
    reason: str,
    details: str,
    *,
    filename: str = "",
    preannotation_status: str = "",
) -> Dict[str, Any]:
    if isinstance(record, dict):
        if not filename:
            filename = _string_or_empty(record.get("filename"))
        if not preannotation_status:
            preannotation_status = _string_or_empty(record.get("preannotation_status"))

    return {
        "filename": filename,
        "preannotation_status": preannotation_status,
        "reason": reason,
        "details": details,
        "record": record,
    }


def contains_unk_marker(value: Any) -> bool:
    return isinstance(value, str) and "[UNK]" in value


def build_output_record(
    record: Dict[str, Any],
    *,
    transcription: Any,
    formula_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "filename": normalize_text(record.get("filename")).strip(),
        "idx": record.get("idx", 1),
        "QuestionType": normalize_text(record.get("QuestionType")),
        "transcription": transcription,
        "formula_list": formula_list,
        "final_answer": normalize_text(record.get("final_answer")),
    }


def build_passthrough_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return build_output_record(
        record,
        transcription=record.get("transcription", ""),
        formula_list=[],
    )


def split_transcription(transcription: str) -> List[Dict[str, Any]]:
    fragments: List[Tuple[str, str]] = []
    current_question_id: str | None = None

    for line in transcription.split("\n"):
        matches = list(Q_PATTERN.finditer(line))
        if not matches:
            if current_question_id is None:
                continue
            if line:
                fragments.append((current_question_id, line))
            continue

        for index, match in enumerate(matches):
            current_question_id = f"Q{match.group(1)}"
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            content = line[start:end]
            if content:
                fragments.append((current_question_id, content))

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
    formula_seq = 1

    for item in transcription_items:
        content = item["content"]
        for match in MATH_PATTERN.finditer(content):
            formula_list.append(
                {
                    "formula_seq": formula_seq,
                    "seq": item["seq"],
                    "question_id": item["question_id"],
                    "formula": match.group(1),
                }
            )
            formula_seq += 1

        cleaned_items.append(
            {
                **item,
                "content": MATH_PATTERN.sub(r"\1", content),
            }
        )

    return cleaned_items, formula_list


def extract_record(
    original_record: Any,
    cleaned_record: Any,
) -> Tuple[Dict[str, Any] | None, str | None, str | None, str | None]:
    if not isinstance(cleaned_record, dict):
        return None, "invalid_record_type", "Record must be a JSON object.", None

    status = _string_or_empty(cleaned_record.get("preannotation_status"))
    if status != "success":
        return (
            None,
            "unsupported_preannotation_status",
            f"Only records with preannotation_status='success' are supported, got {status!r}.",
            None,
        )

    filename = cleaned_record.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        return None, "missing_filename", "Record filename must be a non-empty string.", None

    transcription = cleaned_record.get("transcription")
    if not isinstance(transcription, str):
        return None, "invalid_transcription", "Record transcription must be a string.", None

    question_type = normalize_text(cleaned_record.get("QuestionType"))
    passthrough_source = original_record if isinstance(original_record, dict) else cleaned_record

    if question_type in PASSTHROUGH_QUESTION_TYPES:
        return (
            build_passthrough_record(passthrough_source),
            None,
            None,
            "passthrough_non_constructed",
        )

    if question_type != CONSTRUCTED_RESPONSE:
        return (
            None,
            "invalid_question_type",
            f"Record QuestionType must be one of {VALID_QUESTION_TYPES}, got {question_type!r}.",
            None,
        )

    transcription_items = split_transcription(transcription)
    if not transcription_items:
        if contains_unk_marker(passthrough_source.get("transcription")) or contains_unk_marker(
            passthrough_source.get("final_answer")
        ):
            return build_passthrough_record(passthrough_source), None, None, "passthrough_unk"
        return (
            None,
            "no_extractable_segments",
            "No transcription segments were extracted after applying q-marker rules.",
            None,
        )

    cleaned_items, formula_list = extract_formula_list(transcription_items)
    output_record = build_output_record(
        cleaned_record,
        transcription=cleaned_items,
        formula_list=formula_list,
    )
    return output_record, None, None, "structured"


def extract_results(
    original_records: List[Any],
    cleaned_records: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    extracted_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed_success_records = 0
    structured_extracted_records = 0
    passthrough_non_constructed_records = 0
    passthrough_unk_records = 0

    for original_record, cleaned_record in zip(original_records, cleaned_records):
        if isinstance(cleaned_record, dict) and cleaned_record.get("preannotation_status") == "success":
            processed_success_records += 1

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            cleaned_record,
        )
        if extracted_record is not None:
            extracted_records.append(extracted_record)
            if extraction_kind == "structured":
                structured_extracted_records += 1
            elif extraction_kind == "passthrough_non_constructed":
                passthrough_non_constructed_records += 1
            elif extraction_kind == "passthrough_unk":
                passthrough_unk_records += 1
            continue

        errors.append(
            build_error_item(
                original_record,
                reason or "unknown_error",
                details or "Unknown extraction failure.",
            )
        )

    passthrough_records = passthrough_non_constructed_records + passthrough_unk_records
    error_report = {
        "summary": {
            "input_records": len(original_records),
            "processed_success_records": processed_success_records,
            "extracted_records": len(extracted_records),
            "structured_extracted_records": structured_extracted_records,
            "passthrough_records": passthrough_records,
            "passthrough_non_constructed_records": passthrough_non_constructed_records,
            "passthrough_unk_records": passthrough_unk_records,
            "error_records": len(errors),
        },
        "errors": errors,
    }
    return extracted_records, error_report
