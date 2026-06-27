#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from . import eval as eval_module
    from .metric import UNICODE_FORMULA_REPLACEMENTS
    from .predict_convert import convert_results
except ImportError:  # pragma: no cover - script execution fallback
    import eval as eval_module  # type: ignore
    from metric import UNICODE_FORMULA_REPLACEMENTS  # type: ignore
    from predict_convert import convert_results  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR.parent
DEFAULT_SOURCE_ROOT = Path("output") / "predict"
DEFAULT_OUTPUT_ROOT = Path("output") / "protocol_debias"
DEFAULT_GT_PATH = Path("data") / "GT" / "extracted_gt.json"

SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎", "0123456789+-=()")
SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾", "0123456789+-=()")
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
BROKEN_MATH_BRACKET_PATTERN = re.compile(r"\[\[MATH\](.*?)(?:\]\]|\Z)", re.DOTALL)
INLINE_ESCAPED_NEWLINE_PATTERN = re.compile(r"(?<!\\)\\n")
INLINE_ESCAPED_CARRIAGE_PATTERN = re.compile(r"(?<!\\)\\r")
INLINE_ESCAPED_TAB_PATTERN = re.compile(r"(?<!\\)\\t")
DOUBLE_ESCAPED_NEWLINE_PATTERN = re.compile(r"\\\\n")
DOUBLE_ESCAPED_CARRIAGE_PATTERN = re.compile(r"\\\\r")
DOUBLE_ESCAPED_TAB_PATTERN = re.compile(r"\\\\t")
PARALLEL_ASCII_PATTERN = re.compile(r"(?<=[A-Za-z0-9}\)])//(?=[A-Za-z0-9\\{(])")
MATH_WRAPPER_PATTERN = re.compile(r"\[\[MATH:(.*?)\]\]")
RELATION_PATTERN = re.compile(
    r"([A-Za-z0-9\\{}_^()\[\],.+\-*/|]+"
    r"(?:=|<|>|\\le|\\ge|\\ne|\\neq|\\perp|\\parallel|\\subset|\\supset|\\subseteq|\\supseteq|"
    r"\\cap|\\cup|\\in|\\notin|\\Rightarrow|\\rightarrow|\\to)"
    r"[A-Za-z0-9\\{}_^()\[\],.+\-*/|]*)"
)
POINT_PATTERN = re.compile(
    r"([A-Za-z](?:_\{?\d+\}?)?\([^()\n]{1,80}\))"
)
COMMAND_PATTERN = re.compile(
    r"(\\(?:frac|sqrt|vec|overrightarrow|sin|cos|tan|log|ln|left|right|begin|end)[A-Za-z0-9\\{}_^()\[\],.+\-*/|=<>]*)"
)
GEOMETRY_TOKEN_PATTERN = re.compile(
    r"([A-Z](?:_\{?\d+\}?|[A-Z]){1,})"
)
SINGLE_VARIABLE_PATTERN = re.compile(
    "(?:(?<=\\u53d6)|(?<=\\u5f97)|(?<=\\u4e3a)|(?<=\\u8bbe)|(?<=[(,\\uFF0C\\u3001]))([xyzabckmnt])(?=(?:\\u53d6|\\u5f97|\\u4e3a|,|\\uFF0C|\\u3001|\\u65f6|$))"
)
JSON_KEY_PATTERN_TEMPLATE = '"{key}"'
SKIP_GEOMETRY_PREFIXES = (
    "\u5e73\u9762",
    "\u76f4\u7ebf",
    "\u7ebf\u6bb5",
    "\u5c04\u7ebf",
    "\u70b9",
    "\u5411\u91cf",
)
PASSTHROUGH_TYPES = {"MultipleChoice", "FillBlank"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build protocol-debiased predict files and run the official evaluation flow."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing per-model raw result files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for protocol-debiased predict and metric files.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_PATH,
        help="Ground-truth JSON path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Restrict recomputation to a single model directory.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.50,
        help="Line match threshold passed to the existing conversion/eval pipeline.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (BENCHMARK_DIR / path).resolve()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BENCHMARK_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_array(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{display_path(path)} must contain a top-level JSON array.")
    return data


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def build_gt_index(gt_records: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    gt_index: Dict[str, Dict[str, Any]] = {}
    for record_index, record in enumerate(gt_records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"GT record {record_index} must be a JSON object.")
        filename = normalize_filename(record.get("filename"))
        if not filename:
            raise ValueError(f"GT record {record_index} is missing a valid filename.")
        gt_index[filename] = record
    return gt_index


def normalize_filename(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().replace("\\", "/")


def normalize_idx(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def discover_model_dirs(source_root: Path, model_name: Optional[str]) -> List[Path]:
    if model_name:
        model_dir = source_root / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {display_path(model_dir)}")
        return [model_dir]

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {display_path(source_root)}")
    return sorted(
        [path for path in source_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )


def strip_code_fences(raw: str) -> str:
    stripped = raw.strip()
    return CODE_FENCE_PATTERN.sub("", stripped).strip()


def _jsonish_quote_start(raw: str, key: str) -> int:
    marker = f'"{key}"'
    key_index = raw.find(marker)
    if key_index < 0:
        return -1
    colon_index = raw.find(":", key_index + len(marker))
    if colon_index < 0:
        return -1
    quote_index = raw.find('"', colon_index + 1)
    return quote_index


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


def salvage_record_from_raw_output(
    record: Dict[str, Any],
    gt_record: Optional[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], Dict[str, int]]]:
    raw_output = normalize_text(record.get("raw_output")).strip()
    if not raw_output:
        return None

    cleaned = strip_code_fences(raw_output)
    transcription = _extract_jsonish_quoted_value(cleaned, "transcription")
    if not transcription:
        return None

    final_answer = _extract_jsonish_quoted_value(cleaned, "final_answer")
    question_type = _extract_jsonish_quoted_value(cleaned, "QuestionType")
    idx_raw = _extract_jsonish_scalar_value(cleaned, "idx")

    salvaged = copy.deepcopy(record)
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
        salvaged["filename"] = normalize_filename(gt_record.get("filename"))

    return salvaged, {"parse_salvaged": 1}


def normalize_subscript_runs(text: str) -> str:
    result: List[str] = []
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char in "₀₁₂₃₄₅₆₇₈₉":
            start = cursor
            while cursor < len(text) and text[cursor] in "₀₁₂₃₄₅₆₇₈₉":
                cursor += 1
            digits = text[start:cursor].translate(SUBSCRIPT_TRANSLATION)
            result.append(f"_{{{digits}}}")
            continue
        if char in "⁰¹²³⁴⁵⁶⁷⁸⁹":
            start = cursor
            while cursor < len(text) and text[cursor] in "⁰¹²³⁴⁵⁶⁷⁸⁹":
                cursor += 1
            digits = text[start:cursor].translate(SUPERSCRIPT_TRANSLATION)
            result.append(f"^{{{digits}}}")
            continue
        result.append(char)
        cursor += 1
    return "".join(result)


def normalize_mathish_text(text: str) -> Tuple[str, Dict[str, int]]:
    diagnostics: Dict[str, int] = {}
    normalized = normalize_text(text)
    if not normalized:
        return "", diagnostics

    replaced_literal_newlines = 0
    if "\\n" in normalized:
        replaced_literal_newlines = normalized.count("\\n")
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if replaced_literal_newlines:
        diagnostics["literal_newline_expanded"] = replaced_literal_newlines

    replaced_tabs = normalized.count("\\t")
    if replaced_tabs:
        normalized = normalized.replace("\\t", "\t")
        diagnostics["literal_tab_expanded"] = replaced_tabs

    broken_math = len(BROKEN_MATH_BRACKET_PATTERN.findall(normalized))
    if broken_math:
        normalized = BROKEN_MATH_BRACKET_PATTERN.sub(r"[[MATH:\1]]", normalized)
        diagnostics["broken_math_tag_fixed"] = broken_math

    before_parallel = normalized
    normalized = PARALLEL_ASCII_PATTERN.sub(r"\\parallel", normalized)
    if before_parallel != normalized:
        diagnostics["parallel_ascii_normalized"] = 1

    before_subscript = normalized
    normalized = normalize_subscript_runs(normalized)
    if before_subscript != normalized:
        diagnostics["subscript_normalized"] = 1

    unicode_replacements = 0
    for source, target in UNICODE_FORMULA_REPLACEMENTS.items():
        if source in normalized:
            unicode_replacements += normalized.count(source)
            normalized = normalized.replace(source, target)
    if unicode_replacements:
        diagnostics["unicode_formula_replaced"] = unicode_replacements

    normalized = normalized.replace("∥", r"\parallel")
    normalized = normalized.replace("⊥", r"\perp")
    normalized = normalized.replace("∩", r"\cap")
    normalized = normalized.replace("∪", r"\cup")
    normalized = normalized.replace("∈", r"\in")
    normalized = normalized.replace("∉", r"\notin")
    normalized = normalized.replace("≤", r"\le")
    normalized = normalized.replace("≥", r"\ge")
    normalized = normalized.replace("≠", r"\ne")

    return normalized, diagnostics


def normalize_record_for_protocol_debias(
    record: Dict[str, Any],
    gt_record: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    diagnostics: Counter[str] = Counter()
    normalized = copy.deepcopy(record)

    if normalize_text(normalized.get("predict_status")) != "success":
        salvaged = salvage_record_from_raw_output(normalized, gt_record)
        if salvaged is not None:
            normalized, salvage_diag = salvaged
            diagnostics.update(salvage_diag)

    if gt_record is not None:
        normalized["filename"] = normalize_filename(gt_record.get("filename"))
        normalized["ImgReal"] = normalize_filename(gt_record.get("filename"))
        normalized["idx"] = normalize_idx(gt_record.get("idx")) or normalize_idx(normalized.get("idx"))
        if not normalize_text(normalized.get("QuestionType")):
            normalized["QuestionType"] = normalize_text(gt_record.get("QuestionType"))

    transcription, transcription_diag = normalize_mathish_text(normalized.get("transcription"))
    if transcription:
        normalized["transcription"] = transcription
    diagnostics.update(transcription_diag)

    final_answer, final_answer_diag = normalize_mathish_text(normalized.get("final_answer"))
    if normalize_text(normalized.get("final_answer")) or final_answer:
        normalized["final_answer"] = final_answer
    diagnostics.update({f"final_{key}": value for key, value in final_answer_diag.items()})

    return normalized, dict(diagnostics)


def _preceding_text(text: str, start: int) -> str:
    return text[max(0, start - 4) : start]


def extract_formula_candidates_from_line(text: str) -> List[str]:
    content = normalize_text(text)
    if not content:
        return []

    explicit = [match.group(1).strip() for match in MATH_WRAPPER_PATTERN.finditer(content) if match.group(1).strip()]
    if explicit:
        return explicit

    candidates: List[Tuple[int, int, str]] = []

    def add_candidate(start: int, end: int, candidate: str) -> None:
        cleaned = candidate.strip(" ,\uFF0C\u3001\u3002\uFF1B;:")
        if not cleaned:
            return
        if any(ch in cleaned for ch in "\u89e3\u8bc1\u660e\u8bbe\u5f53\u53d6\u53c8\u2235\u2234"):
            return
        candidates.append((start, end, cleaned))

    for match in RELATION_PATTERN.finditer(content):
        add_candidate(match.start(1), match.end(1), match.group(1))

    for match in POINT_PATTERN.finditer(content):
        add_candidate(match.start(1), match.end(1), match.group(1))

    for match in COMMAND_PATTERN.finditer(content):
        add_candidate(match.start(1), match.end(1), match.group(1))

    for match in GEOMETRY_TOKEN_PATTERN.finditer(content):
        candidate = match.group(1)
        if len(candidate) < 2:
            continue
        if any(_preceding_text(content, match.start(1)).endswith(prefix) for prefix in SKIP_GEOMETRY_PREFIXES):
            continue
        add_candidate(match.start(1), match.end(1), candidate)

    for match in SINGLE_VARIABLE_PATTERN.finditer(content):
        add_candidate(match.start(1), match.end(1), match.group(1))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1]))
    merged: List[Tuple[int, int, str]] = []
    seen: set[str] = set()
    for start, end, candidate in candidates:
        if candidate in seen:
            continue
        if merged and start >= merged[-1][0] and end <= merged[-1][1]:
            continue
        seen.add(candidate)
        merged.append((start, end, candidate))
    return [candidate for _, _, candidate in merged]


def rebuild_formula_list_with_heuristics(
    predict_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if predict_record.get("QuestionType") in PASSTHROUGH_TYPES:
        return predict_record, {}

    transcription = predict_record.get("transcription")
    if not isinstance(transcription, list):
        return predict_record, {}

    diagnostics: Counter[str] = Counter()
    existing_by_seq: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for formula in predict_record.get("formula_list", []):
        seq = normalize_idx(formula.get("seq"))
        if seq is not None:
            existing_by_seq[seq].append(formula)

    rebuilt: List[Dict[str, Any]] = []
    formula_sequence = 1
    for item in transcription:
        if not isinstance(item, dict):
            continue
        seq = normalize_idx(item.get("seq"))
        if seq is None:
            continue

        formulas_for_line = existing_by_seq.get(seq, [])
        if formulas_for_line:
            for formula in formulas_for_line:
                rebuilt.append({**formula, "formula_seq": formula_sequence})
                formula_sequence += 1
            continue

        candidates = extract_formula_candidates_from_line(normalize_text(item.get("content")))
        if candidates:
            diagnostics["heuristic_formula_lines"] += 1
            diagnostics["heuristic_formula_slots"] += len(candidates)
        for candidate in candidates:
            rebuilt.append(
                {
                    "formula_seq": formula_sequence,
                    "seq": seq,
                    "question_id": item.get("question_id"),
                    "gt_seq": item.get("gt_seq"),
                    "formula": candidate,
                }
            )
            formula_sequence += 1

    updated = copy.deepcopy(predict_record)
    updated["formula_list"] = rebuilt
    return updated, dict(diagnostics)


def process_model(
    *,
    model_dir: Path,
    gt_records: List[Any],
    gt_index: Dict[str, Dict[str, Any]],
    output_root: Path,
    match_threshold: float,
) -> Dict[str, Any]:
    model_name = model_dir.name
    input_path = model_dir / "result_less_format_noise.json"
    if not input_path.exists():
        input_path = model_dir / "results.json"
    raw_records = load_json_array(input_path)

    normalized_raw_records: List[Dict[str, Any]] = []
    normalization_diagnostics: Counter[str] = Counter()
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        filename = normalize_filename(record.get("ImgReal") or record.get("filename"))
        gt_record = gt_index.get(filename)
        normalized, diag = normalize_record_for_protocol_debias(record, gt_record)
        normalized_raw_records.append(normalized)
        normalization_diagnostics.update(diag)

    predict_records, conversion_error_report = convert_results(
        normalized_raw_records,
        gt_records,
        match_threshold,
    )

    predict_output: List[Dict[str, Any]] = []
    formula_recovery_diagnostics: Counter[str] = Counter()
    for record in predict_records:
        updated, diag = rebuild_formula_list_with_heuristics(record)
        predict_output.append(updated)
        formula_recovery_diagnostics.update(diag)

    model_output_dir = output_root / "predict" / model_name
    atomic_write_json(model_output_dir / "normalized_results.json", normalized_raw_records)
    atomic_write_json(model_output_dir / "predict.json", predict_output)
    atomic_write_json(model_output_dir / "predict_conversion_errors.json", conversion_error_report)

    diagnostics = {
        "model": model_name,
        "input_path": display_path(input_path),
        "input_records": len(raw_records),
        "normalization": dict(normalization_diagnostics),
        "formula_recovery": dict(formula_recovery_diagnostics),
        "conversion_summary": conversion_error_report.get("summary", {}),
    }
    atomic_write_json(output_root / "diagnostics" / f"{model_name}.json", diagnostics)
    return diagnostics


def run_eval(predict_root: Path, metric_root: Path, gt_path: Path, match_threshold: float) -> None:
    exit_code = eval_module.main(
        [
            "--predict-root",
            str(predict_root),
            "--output-root",
            str(metric_root),
            "--gt",
            str(gt_path),
            "--match-threshold",
            str(match_threshold),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Evaluation failed for {display_path(predict_root)} -> {display_path(metric_root)}."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        source_root = resolve_path(args.source_root)
        output_root = resolve_path(args.output_root)
        gt_path = resolve_path(args.gt)

        gt_records = load_json_array(gt_path)
        gt_index = build_gt_index(gt_records)
        model_dirs = discover_model_dirs(source_root, args.model)

        diagnostics: List[Dict[str, Any]] = []
        for model_dir in model_dirs:
            diagnostics.append(
                process_model(
                    model_dir=model_dir,
                    gt_records=gt_records,
                    gt_index=gt_index,
                    output_root=output_root,
                    match_threshold=args.match_threshold,
                )
            )

        atomic_write_json(output_root / "diagnostics" / "summary.json", diagnostics)

        run_eval(output_root / "predict", output_root / "metric", gt_path, args.match_threshold)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
