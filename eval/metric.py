#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CONSTRUCTED_RESPONSE = "ConstructedResponse"
MATCHED_STATUS = "matched"
MATH_WRAPPER_PATTERN = re.compile(r"\[\[MATH:(.*?)\]\]", re.DOTALL)
SPACE_PATTERN = re.compile(r"[ \t\u3000]+")

RELATION_COMMANDS = {
    "=": "Eq",
    "<": "Lt",
    ">": "Gt",
    r"\le": "Le",
    r"\ge": "Ge",
    r"\ne": "Ne",
    r"\neq": "Ne",
    r"\perp": "Perp",
    r"\parallel": "Parallel",
    r"\to": "To",
    r"\rightarrow": "To",
    r"\Rightarrow": "Implies",
}
FUNCTION_COMMANDS = {
    r"\sin": "Sin",
    r"\cos": "Cos",
    r"\tan": "Tan",
    r"\log": "Log",
    r"\ln": "Log",
}
SPACING_COMMANDS = {
    r"\,",
    r"\:",
    r"\;",
    r"\!",
    r"\quad",
    r"\qquad",
    r"\left",
    r"\right",
}
MULTIPLICATION_TOKENS = {"*", r"\cdot", r"\times"}
PREFIX_MARKER_LABELS = {
    "=": "EqPrefix",
    r"\Rightarrow": "ImpliesPrefix",
    r"\rightarrow": "ToPrefix",
    r"\to": "ToPrefix",
}
POSTFIX_RELATION_LABELS = {
    r"\perp": "PerpPartial",
    r"\parallel": "ParallelPartial",
}
SET_BINARY_COMMANDS = {
    r"\cup": "Union",
    r"\cap": "Intersect",
}
FUNCTION_SYMBOL_LABELS = {
    r"\sin": "Sin",
    r"\cos": "Cos",
    r"\tan": "Tan",
    r"\log": "Log",
    r"\ln": "Log",
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "log": "Log",
    "ln": "Log",
    r"\vec": "Vec",
    r"\overrightarrow": "Overrightarrow",
}
UNICODE_FORMULA_REPLACEMENTS = {
    "−": "-",
    "×": r"\times",
    "÷": r"\div",
    "・": r"\cdot",
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\ne",
    "≈": r"\approx",
    "⊥": r"\perp",
    "∥": r"\parallel",
    "∠": r"\angle",
    "∩": r"\cap",
    "∪": r"\cup",
    "∈": r"\in",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "∉": r"\notin",
    "⊄": r"\not\subset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∞": r"\infty",
    "∅": r"\emptyset",
}
ASCII_FUNCTION_NORMALIZATION = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "log": r"\log",
    "ln": r"\ln",
}
COMMAND_NORMALIZATION = {
    r"\leq": r"\le",
    r"\geq": r"\ge",
    r"\neq": r"\ne",
}
RELATION_OPERATOR_VALUES = {
    "=",
    "<",
    ">",
    ":",
    r"\le",
    r"\ge",
    r"\ne",
    r"\neq",
    r"\perp",
    r"\parallel",
    r"\to",
    r"\rightarrow",
    r"\Rightarrow",
    r"\in",
    r"\notin",
    r"\subset",
    r"\supset",
    r"\subseteq",
    r"\supseteq",
}
STRUCTURE_COMMANDS = {
    r"\frac",
    r"\sqrt",
    r"\div",
    r"\angle",
    r"\approx",
    r"\infty",
    r"\emptyset",
}
KNOWN_FORMULA_COMMANDS = tuple(
    sorted(
        {
            command
            for command in (
                set(COMMAND_NORMALIZATION)
                | set(COMMAND_NORMALIZATION.values())
                | set(RELATION_COMMANDS)
                | set(FUNCTION_COMMANDS)
                | set(SPACING_COMMANDS)
                | set(MULTIPLICATION_TOKENS)
                | set(POSTFIX_RELATION_LABELS)
                | set(SET_BINARY_COMMANDS)
                | set(FUNCTION_SYMBOL_LABELS)
                | set(RELATION_OPERATOR_VALUES)
                | STRUCTURE_COMMANDS
            )
            if command.startswith("\\")
        },
        key=len,
        reverse=True,
    )
)


class FormulaParseError(ValueError):
    """Raised when a formula is structurally invalid for evaluation."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


@dataclass
class _TreeNode:
    kind: str
    label: str
    relation: Optional[str] = None
    children: List["_TreeNode"] = field(default_factory=list)


@dataclass
class _FormulaAst:
    kind: str
    label: str = ""
    children: List["_FormulaAst"] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _strip_math_wrapper(text: str) -> str:
    return MATH_WRAPPER_PATTERN.sub(r"\1", text)


def _normalize_stem_line(text: str) -> str:
    normalized = _strip_math_wrapper(_normalize_text(text))
    normalized = SPACE_PATTERN.sub("", normalized)
    return normalized.strip()


def _split_transcription_lines(value: Any) -> List[Tuple[int, str, str]]:
    if isinstance(value, list):
        items = _normalize_transcription_items(value)
        source_lines: List[Tuple[int, str, str]] = []
        line_number = 1
        for item in items:
            for line in _normalize_text(item["content"]).split("\n"):
                normalized = _normalize_stem_line(line)
                if not normalized:
                    continue
                source_lines.append((line_number, line, normalized))
                line_number += 1
        return source_lines

    if isinstance(value, str):
        source_lines = []
        line_number = 1
        for line in _normalize_text(value).split("\n"):
            normalized = _normalize_stem_line(line)
            if not normalized:
                continue
            source_lines.append((line_number, line, normalized))
            line_number += 1
        return source_lines

    raise ValueError("transcription must be a string or a list.")


def _normalize_transcription_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("transcription must be a list.")

    normalized: List[Dict[str, Any]] = []
    seen_sequences: set[int] = set()
    for item_index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"transcription item {item_index} must be a JSON object.")
        sequence = _normalize_idx(item.get("seq"))
        if sequence is None:
            raise ValueError(f"transcription item {item_index} has an invalid seq.")
        if sequence in seen_sequences:
            raise ValueError(f"transcription contains duplicated seq {sequence}.")
        seen_sequences.add(sequence)
        normalized.append(
            {
                "seq": sequence,
                "question_id": item.get("question_id"),
                "gt_seq": _normalize_idx(item.get("gt_seq")),
                "match_status": _normalize_text(item.get("match_status")),
                "content": _normalize_text(item.get("content")),
            }
        )

    normalized.sort(key=lambda item: item["seq"])
    return normalized


def _normalize_structured_stem_items(value: Any) -> List[Dict[str, Any]]:
    items = _normalize_transcription_items(value)
    return [
        {
            **item,
            "normalized_content": _normalize_stem_line(item["content"]),
        }
        for item in items
    ]


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _line_similarity(left: str, right: str) -> float:
    max_length = max(len(left), len(right))
    if max_length == 0:
        return 1.0
    distance = _levenshtein_distance(left, right)
    return max(0.0, 1.0 - (distance / max_length))


def _stem_accuracy_from_scores(weighted_score_sum: float, matched_gt_chars: int) -> Optional[float]:
    if matched_gt_chars == 0:
        return None
    return weighted_score_sum / matched_gt_chars


def _hungarian_minimize(cost_matrix: Sequence[Sequence[float]]) -> List[int]:
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
        min_values = [math.inf] * (size + 1)
        used = [False] * (size + 1)

        while True:
            used[column] = True
            matched_row = p[column]
            delta = math.inf
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


def _best_line_matches(
    gt_lines: Sequence[Tuple[int, str, str]],
    predict_lines: Sequence[Tuple[int, str, str]],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    gt_count = len(gt_lines)
    predict_count = len(predict_lines)
    if gt_count == 0 or predict_count == 0:
        return []

    size = max(gt_count, predict_count)
    weights = [[0.0 for _ in range(size)] for _ in range(size)]
    for gt_index, (_, _, gt_line) in enumerate(gt_lines):
        for predict_index, (_, _, predict_line) in enumerate(predict_lines):
            score = _line_similarity(gt_line, predict_line)
            if score >= threshold:
                weights[gt_index][predict_index] = score

    cost_matrix = [
        [1.0 - weights[row][column] for column in range(size)]
        for row in range(size)
    ]
    assignment = _hungarian_minimize(cost_matrix)

    matches: List[Tuple[int, int, float]] = []
    for gt_index in range(gt_count):
        predict_index = assignment[gt_index]
        if predict_index < 0 or predict_index >= predict_count:
            continue
        score = weights[gt_index][predict_index]
        if score > 0.0:
            matches.append((gt_index, predict_index, score))
    return matches


def _compute_structured_stem_metrics(
    gt_transcription: Any,
    predict_transcription: Any,
) -> Optional[Dict[str, Any]]:
    gt_items = _normalize_structured_stem_items(gt_transcription)
    if not gt_items:
        return None

    weighted_gt_chars = sum(len(item["normalized_content"]) for item in gt_items)
    if weighted_gt_chars == 0:
        return None

    gt_by_seq = {item["seq"]: item for item in gt_items}
    predict_items = _normalize_structured_stem_items(predict_transcription)

    predict_by_gt_seq: Dict[int, Dict[str, Any]] = {}
    matched_predict_sequences: set[int] = set()
    for item in predict_items:
        if item.get("match_status") != MATCHED_STATUS:
            continue

        gt_seq = item.get("gt_seq")
        if gt_seq is None:
            raise ValueError(f"matched transcription item seq {item['seq']} must contain a valid gt_seq.")
        if gt_seq not in gt_by_seq:
            raise ValueError(
                f"matched transcription item seq {item['seq']} points to unknown gt_seq {gt_seq}."
            )
        if gt_seq in predict_by_gt_seq:
            previous_seq = predict_by_gt_seq[gt_seq]["seq"]
            raise ValueError(
                f"predict transcription contains duplicated matched gt_seq {gt_seq} "
                f"(seq {previous_seq} and {item['seq']})."
            )

        predict_by_gt_seq[gt_seq] = item
        matched_predict_sequences.add(item["seq"])

    weighted_score_sum = 0.0
    matched_gt_chars = 0
    match_items: List[Dict[str, Any]] = []
    for gt_item in gt_items:
        predict_item = predict_by_gt_seq.get(gt_item["seq"])
        if predict_item is None:
            continue

        gt_chars = len(gt_item["normalized_content"])
        score = _line_similarity(gt_item["normalized_content"], predict_item["normalized_content"])
        matched_gt_chars += gt_chars
        weighted_score_sum += gt_chars * score
        match_items.append(
            {
                "gt_line_index": gt_item["seq"],
                "predict_line_index": predict_item["seq"],
                "score": score,
                "gt_line": gt_item["content"],
                "predict_line": predict_item["content"],
            }
        )

    matched_line_count = len(match_items)
    return {
        # Normalize only by matched GT chars so stem_acc reflects matched-content quality.
        "stem_acc": _stem_accuracy_from_scores(weighted_score_sum, matched_gt_chars),
        "weighted_score_sum": weighted_score_sum,
        "weighted_gt_chars": weighted_gt_chars,
        "matched_gt_chars": matched_gt_chars,
        "gt_line_count": len(gt_items),
        "predict_line_count": len(predict_items),
        "matched_line_count": matched_line_count,
        "unmatched_gt_line_count": len(gt_items) - matched_line_count,
        "unmatched_predict_line_count": len(predict_items) - len(matched_predict_sequences),
        "matches": match_items,
    }


def _tree_size(node: Optional[_TreeNode]) -> int:
    if node is None:
        return 0
    return 1 + sum(_tree_size(child) for child in node.children)


def _tree_signature(node: _TreeNode) -> Tuple[str, str, Optional[str]]:
    return (node.kind, node.label, node.relation)


def _tree_edit_distance(left: Optional[_TreeNode], right: Optional[_TreeNode]) -> int:
    if left is None:
        return _tree_size(right)
    if right is None:
        return _tree_size(left)

    root_cost = 0 if _tree_signature(left) == _tree_signature(right) else 1
    left_children = left.children
    right_children = right.children
    left_count = len(left_children)
    right_count = len(right_children)

    dp = [[0 for _ in range(right_count + 1)] for _ in range(left_count + 1)]
    for left_index in range(1, left_count + 1):
        dp[left_index][0] = dp[left_index - 1][0] + _tree_size(left_children[left_index - 1])
    for right_index in range(1, right_count + 1):
        dp[0][right_index] = dp[0][right_index - 1] + _tree_size(right_children[right_index - 1])

    for left_index in range(1, left_count + 1):
        for right_index in range(1, right_count + 1):
            delete_cost = dp[left_index - 1][right_index] + _tree_size(left_children[left_index - 1])
            insert_cost = dp[left_index][right_index - 1] + _tree_size(right_children[right_index - 1])
            replace_cost = dp[left_index - 1][right_index - 1] + _tree_edit_distance(
                left_children[left_index - 1],
                right_children[right_index - 1],
            )
            dp[left_index][right_index] = min(delete_cost, insert_cost, replace_cost)

    return root_cost + dp[left_count][right_count]


def _tree_teds(left: _TreeNode, right: _TreeNode) -> float:
    left_size = _tree_size(left)
    right_size = _tree_size(right)
    denominator = left_size + right_size
    if denominator == 0:
        return 1.0
    distance = _tree_edit_distance(left, right)
    return max(0.0, 1.0 - (distance / denominator))


def _clone_with_relation(node: _TreeNode, relation: str) -> _TreeNode:
    return _TreeNode(
        kind=node.kind,
        label=node.label,
        relation=relation,
        children=node.children,
    )


def _match_known_command_prefix(value: str) -> str:
    for command in KNOWN_FORMULA_COMMANDS:
        if value.startswith(command):
            return command
    return value


def _normalize_formula_text(formula: Any) -> str:
    if formula is None:
        return ""
    text = _strip_math_wrapper(_normalize_text(formula)).strip()
    if not text:
        return ""

    text = text.replace(r"\left.", "").replace(r"\right.", "")
    text = text.replace("<=", r"\le").replace(">=", r"\ge").replace("!=", r"\ne")
    for raw, replacement in COMMAND_NORMALIZATION.items():
        text = text.replace(raw, replacement)
    for raw, replacement in UNICODE_FORMULA_REPLACEMENTS.items():
        text = text.replace(raw, replacement)

    text = (
        text.replace(r"\{", "{")
        .replace(r"\}", "}")
        .replace(r"\[", "[")
        .replace(r"\]", "]")
        .replace(r"\(", "(")
        .replace(r"\)", ")")
    )
    for command in SPACING_COMMANDS:
        text = text.replace(command, "")
    for bare_name, command in ASCII_FUNCTION_NORMALIZATION.items():
        text = re.sub(
            rf"(?<![A-Za-z\\]){re.escape(bare_name)}(?=\s*(?:[(\[{{<\\A-Za-zα-ωΑ-Ω]))",
            lambda _: command,
            text,
        )

    return SPACE_PATTERN.sub("", text).strip()


def _tokenize_formula(formula: str) -> List[_Token]:
    tokens: List[_Token] = []
    index = 0
    while index < len(formula):
        char = formula[index]
        if char.isspace():
            index += 1
            continue
        if char == "\\":
            end_index = index + 1
            if end_index < len(formula) and formula[end_index].isalpha():
                while end_index < len(formula) and formula[end_index].isalpha():
                    end_index += 1
                raw_value = formula[index:end_index]
                value = _match_known_command_prefix(raw_value)
                end_index = index + len(value)
            elif end_index < len(formula):
                end_index += 1
                value = formula[index:end_index]
            else:
                value = formula[index:end_index]
            if value not in SPACING_COMMANDS:
                tokens.append(_Token("COMMAND", value))
            index = end_index
            continue
        if char in "{}()[]":
            kind = {
                "{": "LBRACE",
                "}": "RBRACE",
                "(": "LPAREN",
                ")": "RPAREN",
                "[": "LBRACKET",
                "]": "RBRACKET",
            }[char]
            tokens.append(_Token(kind, char))
            index += 1
            continue
        if char in "^_+-=*/<>,:":
            tokens.append(_Token("OP", char))
            index += 1
            continue
        if char.isdigit():
            end_index = index + 1
            while end_index < len(formula) and (formula[end_index].isdigit() or formula[end_index] == "."):
                end_index += 1
            tokens.append(_Token("NUMBER", formula[index:end_index]))
            index = end_index
            continue
        if char.isalpha():
            end_index = index + 1
            while end_index < len(formula) and formula[end_index].isalnum():
                end_index += 1
            tokens.append(_Token("SYMBOL", formula[index:end_index]))
            index = end_index
            continue
        tokens.append(_Token("SYMBOL", char))
        index += 1
    return tokens


class _TokenStream:
    def __init__(self, tokens: Sequence[_Token]):
        self._tokens = list(tokens)
        self._index = 0

    def peek(self) -> Optional[_Token]:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def peek_ahead(self, offset: int = 1) -> Optional[_Token]:
        index = self._index + offset
        if index >= len(self._tokens):
            return None
        return self._tokens[index]

    def pop(self) -> _Token:
        token = self.peek()
        if token is None:
            raise FormulaParseError("unexpected end of formula.")
        self._index += 1
        return token

    def match(self, *values: str) -> Optional[_Token]:
        token = self.peek()
        if token is None:
            return None
        if token.value in values:
            self._index += 1
            return token
        return None

    def at_end(self) -> bool:
        return self.peek() is None


def _token_label(token: _Token) -> str:
    return token.value


def _tree_label(text: str, limit: int = 48) -> str:
    normalized = text or "EMPTY"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def _closing_kind(open_kind: str) -> str:
    return {
        "LBRACE": "RBRACE",
        "LPAREN": "RPAREN",
        "LBRACKET": "RBRACKET",
    }[open_kind]


def _closing_value(open_kind: str) -> str:
    return {
        "LBRACE": "}",
        "LPAREN": ")",
        "LBRACKET": "]",
    }[open_kind]


def _is_relation_token(token: Optional[_Token]) -> bool:
    if token is None:
        return False
    if token.kind == "OP":
        return token.value in {"=", "<", ">", ":"}
    if token.kind == "COMMAND":
        return token.value in RELATION_OPERATOR_VALUES
    return False


def _is_primary_start(token: Optional[_Token]) -> bool:
    if token is None:
        return False
    if token.kind in {"NUMBER", "SYMBOL", "COMMAND", "LBRACE", "LPAREN", "LBRACKET"}:
        return True
    if token.kind == "OP":
        return token.value in {"+", "-", "<"}
    return False


def _is_multiplicative_boundary(token: Optional[_Token]) -> bool:
    if token is None:
        return True
    if token.kind in {"RBRACE", "RPAREN", "RBRACKET"}:
        return True
    if token.kind == "OP" and token.value == ",":
        return True
    if _is_relation_token(token):
        return True
    if token.kind == "COMMAND" and token.value in SET_BINARY_COMMANDS:
        return True
    return False


def _is_function_like(node: _FormulaAst) -> bool:
    if node.kind == "FunctionSymbol":
        return True
    if node.kind == "Attachment" and node.children:
        return _is_function_like(node.children[0])
    return False


def _coerce_interval(node: _FormulaAst) -> _FormulaAst:
    if node.kind != "TupleOrList":
        return node
    if node.meta.get("from_call"):
        return node
    if len(node.children) != 2:
        return node
    opener = node.meta.get("open")
    closer = node.meta.get("close")
    if opener not in {"(", "["}:
        return node
    if closer not in {")", "]", "?"}:
        return node
    return _FormulaAst(
        kind="Interval",
        label=f"{opener}{closer}",
        children=node.children,
        meta=dict(node.meta),
    )


def _unwrap_argument_node(node: _FormulaAst) -> _FormulaAst:
    if node.kind == "DelimitedGroup" and node.meta.get("closed") and len(node.children) == 1:
        return node.children[0]
    return node


class _FormulaParser:
    def __init__(self, tokens: Sequence[_Token]):
        self.stream = _TokenStream(tokens)
        self.recovery_count = 0

    def _register_recovery(self) -> None:
        self.recovery_count += 1

    def _opaque(self, label: str) -> _FormulaAst:
        self._register_recovery()
        return _FormulaAst(kind="OpaqueSegment", label=label)

    def _opaque_token(self, token: _Token) -> _FormulaAst:
        return self._opaque(_token_label(token))

    def parse_root(self) -> _FormulaAst:
        if self.stream.at_end():
            return self._opaque("EMPTY")

        node = self._parse_formula()
        leftovers: List[_Token] = []
        while not self.stream.at_end():
            leftovers.append(self.stream.pop())
        if leftovers:
            opaque = self._opaque("".join(token.value for token in leftovers))
            node = self._merge_sequence(node, opaque)
        return node

    def _merge_sequence(self, left: _FormulaAst, right: _FormulaAst) -> _FormulaAst:
        if left.kind == "Sequence":
            return _FormulaAst(kind="Sequence", label="SEQ", children=left.children + [right])
        return _FormulaAst(kind="Sequence", label="SEQ", children=[left, right])

    def _parse_formula(self) -> _FormulaAst:
        prefixes: List[str] = []
        while True:
            token = self.stream.peek()
            if token is None:
                break
            if token.kind == "OP" and token.value in PREFIX_MARKER_LABELS:
                prefixes.append(self.stream.pop().value)
                continue
            if token.kind == "COMMAND" and token.value in PREFIX_MARKER_LABELS:
                prefixes.append(self.stream.pop().value)
                continue
            break

        if self.stream.at_end():
            node = self._opaque("EMPTY")
        else:
            node = self._parse_relation_chain()

        for prefix in reversed(prefixes):
            node = _FormulaAst(
                kind="PrefixMarker",
                label=PREFIX_MARKER_LABELS.get(prefix, prefix),
                children=[node],
                meta={"token": prefix},
            )
        return node

    def _parse_relation_chain(self) -> _FormulaAst:
        left = self._parse_set_binary()
        operators: List[str] = []
        operands = [left]
        while _is_relation_token(self.stream.peek()):
            operator = self.stream.pop().value
            operators.append(operator)
            if self.stream.peek() is None or _is_multiplicative_boundary(self.stream.peek()):
                right = self._opaque("MISSING")
            else:
                right = self._parse_set_binary()
            if operator in {r"\in", r"\notin", r"\subset", r"\supset", r"\subseteq", r"\supseteq"}:
                right = _coerce_interval(right)
            operands.append(right)

        if not operators:
            return left
        return _FormulaAst(
            kind="RelationChain",
            label="RelSeq",
            children=operands,
            meta={"operators": operators},
        )

    def _parse_set_binary(self) -> _FormulaAst:
        left = self._parse_additive()
        while True:
            token = self.stream.peek()
            if token is None or token.kind != "COMMAND" or token.value not in SET_BINARY_COMMANDS:
                break
            operator = self.stream.pop().value
            if self.stream.peek() is None:
                right = self._opaque("MISSING")
            else:
                right = self._parse_additive()
            left = _FormulaAst(
                kind="BinaryOp",
                label=SET_BINARY_COMMANDS[operator],
                children=[_coerce_interval(left), _coerce_interval(right)],
                meta={"token": operator},
            )
        return left

    def _parse_additive(self) -> _FormulaAst:
        left = self._parse_multiplicative()
        while True:
            token = self.stream.peek()
            if token is None or token.kind != "OP" or token.value not in {"+", "-"}:
                break
            operator = self.stream.pop().value
            right = self._parse_multiplicative() if self.stream.peek() is not None else self._opaque("MISSING")
            left = _FormulaAst(
                kind="BinaryOp",
                label="Add" if operator == "+" else "Sub",
                children=[left, right],
                meta={"token": operator},
            )
        return left

    def _parse_multiplicative(self) -> _FormulaAst:
        left = self._parse_postfix()
        while True:
            token = self.stream.peek()
            if _is_multiplicative_boundary(token):
                break

            explicit = False
            operator_label = "Mul"
            if token is not None and token.kind == "OP" and token.value in {"*", "/"}:
                explicit = True
                operator_label = "Mul" if token.value == "*" else "Div"
            elif token is not None and token.kind == "COMMAND" and token.value in MULTIPLICATION_TOKENS | {r"\div"}:
                explicit = True
                operator_label = "Div" if token.value == r"\div" else "Mul"

            if explicit:
                self.stream.pop()
                right = self._parse_postfix() if self.stream.peek() is not None else self._opaque("MISSING")
                left = _FormulaAst(
                    kind="BinaryOp",
                    label=operator_label,
                    children=[left, right],
                )
                continue

            if not _is_primary_start(token):
                break
            right = self._parse_postfix()
            if _is_function_like(left):
                left = _FormulaAst(
                    kind="FunctionCall",
                    label="Call",
                    children=[left, right],
                    meta={"implicit": True},
                )
            else:
                left = _FormulaAst(
                    kind="BinaryOp",
                    label="Mul",
                    children=[left, right],
                    meta={"implicit": True},
                )
        return left

    def _parse_postfix(self) -> _FormulaAst:
        base = self._parse_primary()
        while True:
            token = self.stream.peek()
            if token is None:
                break
            if token.kind == "OP" and token.value in {"^", "_"}:
                operator = self.stream.pop().value
                argument = self._parse_attachment_argument()
                base = _FormulaAst(
                    kind="Attachment",
                    label="Pow" if operator == "^" else "Subscript",
                    children=[base, argument],
                    meta={"token": operator},
                )
                continue
            if token.kind in {"LPAREN", "LBRACKET", "LBRACE"} and self._can_follow_as_call(base):
                group = self._parse_delimited_group(self.stream.pop())
                group.meta["from_call"] = True
                base = _FormulaAst(kind="FunctionCall", label="Call", children=[base, group])
                continue
            if self._is_postfix_partial_relation(token):
                operator = self.stream.pop().value
                base = _FormulaAst(
                    kind="PostfixRelation",
                    label=POSTFIX_RELATION_LABELS.get(operator, operator),
                    children=[base],
                    meta={"token": operator},
                )
                continue
            break
        return base

    def _can_follow_as_call(self, node: _FormulaAst) -> bool:
        return node.kind in {"Atom", "FunctionSymbol", "Attachment", "FunctionCall"}

    def _is_postfix_partial_relation(self, token: _Token) -> bool:
        if token.kind != "COMMAND" or token.value not in POSTFIX_RELATION_LABELS:
            return False
        next_token = self.stream.peek_ahead()
        return _is_multiplicative_boundary(next_token)

    def _parse_attachment_argument(self) -> _FormulaAst:
        if self.stream.peek() is None:
            return self._opaque("MISSING")
        return _unwrap_argument_node(self._parse_primary())

    def _parse_argument(self) -> _FormulaAst:
        if self.stream.peek() is None:
            return self._opaque("MISSING")
        return _unwrap_argument_node(self._parse_primary())

    def _parse_primary(self) -> _FormulaAst:
        token = self.stream.peek()
        if token is None:
            return self._opaque("EMPTY")

        if token.kind == "OP" and token.value in {"+", "-"}:
            operator = self.stream.pop().value
            child = self._parse_postfix() if self.stream.peek() is not None else self._opaque("MISSING")
            if operator == "+":
                return child
            return _FormulaAst(kind="UnaryOp", label="Neg", children=[child])

        if token.kind == "OP" and token.value == "<":
            return self._parse_angle_group()

        token = self.stream.pop()
        if token.kind == "NUMBER":
            return _FormulaAst(kind="Atom", label=token.value, meta={"atom_kind": "number"})
        if token.kind == "SYMBOL":
            if token.value in FUNCTION_SYMBOL_LABELS:
                return _FormulaAst(
                    kind="FunctionSymbol",
                    label=FUNCTION_SYMBOL_LABELS[token.value],
                    meta={"token": token.value},
                )
            return _FormulaAst(kind="Atom", label=token.value, meta={"atom_kind": "symbol"})
        if token.kind == "COMMAND":
            if token.value == r"\frac":
                numerator = self._parse_argument()
                denominator = self._parse_argument()
                return _FormulaAst(kind="BinaryOp", label="Frac", children=[numerator, denominator])
            if token.value == r"\sqrt":
                children: List[_FormulaAst] = []
                if self.stream.peek() is not None and self.stream.peek().kind == "LBRACKET":
                    children.append(self._parse_delimited_group(self.stream.pop()))
                children.append(self._parse_argument())
                return _FormulaAst(kind="FunctionCall", label="Sqrt", children=children)
            if token.value in FUNCTION_SYMBOL_LABELS:
                return _FormulaAst(
                    kind="FunctionSymbol",
                    label=FUNCTION_SYMBOL_LABELS[token.value],
                    meta={"token": token.value},
                )
            return _FormulaAst(kind="Atom", label=token.value, meta={"atom_kind": "command"})
        if token.kind in {"LBRACE", "LPAREN", "LBRACKET"}:
            return self._parse_delimited_group(token)
        return self._opaque_token(token)

    def _parse_angle_group(self) -> _FormulaAst:
        self.stream.pop()
        closed = False
        children: List[_FormulaAst] = []
        while True:
            token = self.stream.peek()
            if token is None:
                self._register_recovery()
                break
            if token.kind == "OP" and token.value == ">":
                self.stream.pop()
                closed = True
                break
            children.append(self._parse_formula())
        return _FormulaAst(
            kind="AngleGroup",
            label="<>",
            children=children,
            meta={"open": "<", "close": ">" if closed else "?", "closed": closed},
        )

    def _parse_delimited_group(self, opening_token: _Token) -> _FormulaAst:
        opener = opening_token.value
        closing_kind = _closing_kind(opening_token.kind)
        closer = _closing_value(opening_token.kind)
        children: List[_FormulaAst] = []
        saw_comma = False
        expect_value = True
        closed = False

        while True:
            token = self.stream.peek()
            if token is None:
                self._register_recovery()
                break
            if token.kind == closing_kind:
                self.stream.pop()
                closed = True
                break
            if token.kind == "OP" and token.value == ",":
                saw_comma = True
                self.stream.pop()
                if expect_value:
                    children.append(self._opaque("MISSING"))
                expect_value = True
                continue
            children.append(self._parse_formula())
            expect_value = False

        if saw_comma and expect_value and children:
            children.append(self._opaque("MISSING"))

        if opening_token.kind == "LBRACE" and not closed:
            kind = "SystemOrPiece"
        elif saw_comma:
            kind = "TupleOrList"
        else:
            kind = "DelimitedGroup"
        return _FormulaAst(
            kind=kind,
            label=f"{opener}{closer if closed else '?'}",
            children=children,
            meta={"open": opener, "close": closer if closed else "?", "closed": closed},
        )


def _count_ast_recoveries(node: _FormulaAst) -> int:
    count = 1 if node.kind == "OpaqueSegment" else 0
    if node.meta.get("closed") is False:
        count += 1
    return count + sum(_count_ast_recoveries(child) for child in node.children)


def _ast_parse_status(node: _FormulaAst) -> str:
    if node.kind == "OpaqueSegment":
        return "opaque"
    child_statuses = {_ast_parse_status(child) for child in node.children}
    if node.meta.get("closed") is False or "opaque" in child_statuses or "partial" in child_statuses:
        return "partial"
    return "full"


def _opt_leaf_label(node: _FormulaAst) -> str:
    if node.meta.get("atom_kind") == "number" or re.fullmatch(r"\d+(?:\.\d+)?", node.label):
        return f"Num:{node.label}"
    if re.fullmatch(r"[A-Za-zα-ωΑ-Ω]+", node.label):
        return f"Var:{node.label}"
    if node.label.startswith("\\"):
        return f"Cmd:{node.label}"
    return f"Sym:{node.label}"


def _sequence_tree(kind: str, label: str, children: List[_TreeNode]) -> _TreeNode:
    if not children:
        return _TreeNode(kind="leaf", label="EMPTY")
    if len(children) == 1:
        return children[0]
    return _TreeNode(
        kind=kind,
        label=label,
        children=[_clone_with_relation(child, "Right") for child in children],
    )


def _ast_to_slt(node: _FormulaAst) -> _TreeNode:
    if node.kind == "Atom":
        return _TreeNode(kind="leaf", label=node.label)
    if node.kind == "FunctionSymbol":
        return _TreeNode(kind="container", label=node.label)
    if node.kind == "OpaqueSegment":
        return _TreeNode(
            kind="container",
            label="Opaque",
            children=[_TreeNode(kind="leaf", label=_tree_label(node.label), relation="Inside")],
        )
    if node.kind == "UnaryOp":
        child = _ast_to_slt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
        return _TreeNode(kind="container", label="PrefixOp", children=[_clone_with_relation(child, "Inside")])
    if node.kind == "BinaryOp":
        if node.label == "Frac":
            numerator = _ast_to_slt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
            denominator = (
                _ast_to_slt(node.children[1])
                if len(node.children) > 1
                else _TreeNode(kind="leaf", label="EMPTY")
            )
            return _TreeNode(
                kind="container",
                label=r"\frac",
                children=[
                    _clone_with_relation(numerator, "Above"),
                    _clone_with_relation(denominator, "Below"),
                ],
            )
        children = [_ast_to_slt(child) for child in node.children]
        return _TreeNode(
            kind="container",
            label="InlineOp",
            children=[_clone_with_relation(child, "Right") for child in children],
        )
    if node.kind == "Attachment":
        base = _ast_to_slt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
        argument = _ast_to_slt(node.children[1]) if len(node.children) > 1 else _TreeNode(kind="leaf", label="EMPTY")
        relation = "Above" if node.label == "Pow" else ("Below" if _tree_size(argument) == 1 else "Sub")
        return _TreeNode(
            kind=base.kind,
            label=base.label,
            relation=base.relation,
            children=base.children + [_clone_with_relation(argument, relation)],
        )
    if node.kind == "FunctionCall":
        if node.label == "Sqrt":
            children: List[_TreeNode] = []
            if len(node.children) == 2:
                children.append(_clone_with_relation(_ast_to_slt(node.children[0]), "Above"))
                children.append(_clone_with_relation(_ast_to_slt(node.children[1]), "Inside"))
            elif node.children:
                children.append(_clone_with_relation(_ast_to_slt(node.children[0]), "Inside"))
            return _TreeNode(kind="container", label=r"\sqrt", children=children)

        rendered_children = [_ast_to_slt(child) for child in node.children]
        return _TreeNode(
            kind="container",
            label="InlineCall",
            children=[_clone_with_relation(child, "Right") for child in rendered_children],
        )
    if node.kind == "RelationChain":
        children: List[_TreeNode] = []
        operators = list(node.meta.get("operators") or [])
        for index, operand in enumerate(node.children):
            children.append(_ast_to_slt(operand))
            if index < len(operators):
                children.append(_TreeNode(kind="leaf", label="REL"))
        return _sequence_tree("container", "RelSeq", children)
    if node.kind == "PrefixMarker":
        child = _ast_to_slt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
        return _TreeNode(kind="container", label="PrefixMark", children=[_clone_with_relation(child, "Right")])
    if node.kind == "PostfixRelation":
        child = _ast_to_slt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
        return _TreeNode(kind="container", label="PostfixRel", children=[_clone_with_relation(child, "Left")])
    if node.kind == "Interval":
        children = [_ast_to_slt(child) for child in node.children]
        return _TreeNode(
            kind="container",
            label=f"Interval:{node.meta.get('open', '?')}{node.meta.get('close', '?')}",
            children=[_clone_with_relation(child, "Right") for child in children],
        )
    if node.kind in {"TupleOrList", "DelimitedGroup", "AngleGroup", "SystemOrPiece", "Sequence"}:
        children = [_ast_to_slt(child) for child in node.children]
        return _TreeNode(
            kind="container",
            label=f"{node.kind}:{node.label}",
            children=[_clone_with_relation(child, "Right") for child in children],
        )
    return _TreeNode(kind="leaf", label=node.label or node.kind)


def _ast_to_opt(node: _FormulaAst) -> _TreeNode:
    if node.kind in {"Atom", "FunctionSymbol"}:
        return _TreeNode(kind="leaf", label=_opt_leaf_label(node))
    if node.kind == "OpaqueSegment":
        return _TreeNode(kind="leaf", label=f"Opaque:{_tree_label(node.label)}")
    if node.kind == "UnaryOp":
        child = _ast_to_opt(node.children[0]) if node.children else _TreeNode(kind="leaf", label="EMPTY")
        return _TreeNode(kind="op", label=node.label, children=[child])
    if node.kind == "BinaryOp":
        return _TreeNode(kind="op", label=node.label, children=[_ast_to_opt(child) for child in node.children])
    if node.kind == "Attachment":
        return _TreeNode(kind="op", label=node.label, children=[_ast_to_opt(child) for child in node.children])
    if node.kind == "FunctionCall":
        label = node.label
        if label == "Call" and node.children:
            head = node.children[0]
            if head.kind == "FunctionSymbol":
                label = head.label
                children = [_ast_to_opt(child) for child in node.children[1:]]
            else:
                children = [_ast_to_opt(child) for child in node.children]
        else:
            children = [_ast_to_opt(child) for child in node.children]
        return _TreeNode(kind="op", label=label, children=children)
    if node.kind == "RelationChain":
        children: List[_TreeNode] = []
        operators = list(node.meta.get("operators") or [])
        for index, operand in enumerate(node.children):
            children.append(_ast_to_opt(operand))
            if index < len(operators):
                children.append(_TreeNode(kind="leaf", label=f"Rel:{operators[index]}"))
        return _TreeNode(kind="op", label="RelChain", children=children)
    if node.kind == "PrefixMarker":
        children = [_ast_to_opt(child) for child in node.children]
        return _TreeNode(kind="op", label=node.label, children=children)
    if node.kind == "PostfixRelation":
        children = [_ast_to_opt(child) for child in node.children]
        return _TreeNode(kind="op", label=node.label, children=children)
    if node.kind == "Interval":
        return _TreeNode(
            kind="op",
            label=f"Interval:{node.meta.get('open', '?')}{node.meta.get('close', '?')}",
            children=[_ast_to_opt(child) for child in node.children],
        )
    if node.kind in {"TupleOrList", "DelimitedGroup", "AngleGroup", "SystemOrPiece", "Sequence"}:
        return _TreeNode(kind="op", label=node.kind, children=[_ast_to_opt(child) for child in node.children])
    return _TreeNode(kind="leaf", label=_tree_label(node.label or node.kind))


def _build_formula_trees(formula: Any) -> Tuple[_TreeNode, _TreeNode, Dict[str, Any]]:
    normalized = _normalize_formula_text(formula)
    if not normalized:
        ast = _FormulaAst(kind="OpaqueSegment", label="EMPTY")
    else:
        parser = _FormulaParser(_tokenize_formula(normalized))
        ast = parser.parse_root()

    parse_status = _ast_parse_status(ast)
    recovery_count = _count_ast_recoveries(ast)
    return (
        _ast_to_slt(ast),
        _ast_to_opt(ast),
        {
            "parse_status": parse_status,
            "recovery_count": recovery_count,
            "normalized_is_empty": not normalized,
        },
    )


def _normalize_formula_items(
    value: Any,
    *,
    require_alignment: bool,
) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("formula_list must be a list.")

    normalized: List[Dict[str, Any]] = []
    for item_index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"formula_list item {item_index} must be a JSON object.")
        formula_seq = _normalize_idx(item.get("formula_seq"))
        seq = _normalize_idx(item.get("seq"))
        if formula_seq is None or seq is None:
            raise ValueError(f"formula_list item {item_index} has an invalid formula_seq or seq.")
        question_id = item.get("question_id")
        gt_seq = _normalize_idx(item.get("gt_seq"))
        if require_alignment and ("gt_seq" in item) and item.get("gt_seq") is not None and gt_seq is None:
            raise ValueError(f"formula_list item {item_index} has an invalid gt_seq.")
        normalized.append(
            {
                "formula_seq": formula_seq,
                "seq": seq,
                "question_id": question_id,
                "gt_seq": gt_seq,
                "formula": item.get("formula"),
            }
        )
    return normalized


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _pairwise_order_accuracy(expected: Sequence[Any], observed: Sequence[Any]) -> float:
    if len(expected) < 2 or len(observed) < 2:
        return 1.0
    observed_positions = {value: index for index, value in enumerate(observed)}
    comparable = 0
    correct = 0
    for left_index in range(len(expected)):
        for right_index in range(left_index + 1, len(expected)):
            left_value = expected[left_index]
            right_value = expected[right_index]
            if left_value not in observed_positions or right_value not in observed_positions:
                continue
            comparable += 1
            if observed_positions[left_value] < observed_positions[right_value]:
                correct += 1
    if comparable == 0:
        return 1.0
    return correct / comparable


def _contains_unk_fragment(value: Any) -> bool:
    if isinstance(value, str):
        return "[UNK]" in value
    if isinstance(value, dict):
        return any(_contains_unk_fragment(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unk_fragment(item) for item in value)
    return False


def _record_has_unk(record: Dict[str, Any]) -> bool:
    return _contains_unk_fragment(record.get("transcription")) or _contains_unk_fragment(
        record.get("final_answer")
    )


def compute_stem_metrics(
    gt_record: Dict[str, Any],
    predict_record: Dict[str, Any],
    match_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    if _record_has_unk(gt_record) or _record_has_unk(predict_record):
        return None

    gt_transcription = gt_record.get("transcription")
    predict_transcription = predict_record.get("transcription")

    if isinstance(gt_transcription, list) and isinstance(predict_transcription, list):
        return _compute_structured_stem_metrics(gt_transcription, predict_transcription)

    gt_lines = _split_transcription_lines(gt_transcription)
    if not gt_lines:
        return None

    weighted_gt_chars = sum(len(normalized) for _, _, normalized in gt_lines)
    if weighted_gt_chars == 0:
        return None

    predict_lines = _split_transcription_lines(predict_transcription)
    matches = _best_line_matches(gt_lines, predict_lines, match_threshold)

    weighted_score_sum = 0.0
    matched_gt_chars = 0
    match_items: List[Dict[str, Any]] = []
    for gt_index, predict_index, score in matches:
        gt_line = gt_lines[gt_index]
        predict_line = predict_lines[predict_index]
        gt_chars = len(gt_line[2])
        matched_gt_chars += gt_chars
        weighted_score_sum += gt_chars * score
        match_items.append(
            {
                "gt_line_index": gt_line[0],
                "predict_line_index": predict_line[0],
                "score": score,
                "gt_line": gt_line[1],
                "predict_line": predict_line[1],
            }
        )

    matched_gt_indices = {gt_index for gt_index, _, _ in matches}
    matched_predict_indices = {predict_index for _, predict_index, _ in matches}
    return {
        "stem_acc": _stem_accuracy_from_scores(weighted_score_sum, matched_gt_chars),
        "weighted_score_sum": weighted_score_sum,
        "weighted_gt_chars": weighted_gt_chars,
        "matched_gt_chars": matched_gt_chars,
        "gt_line_count": len(gt_lines),
        "predict_line_count": len(predict_lines),
        "matched_line_count": len(matches),
        "unmatched_gt_line_count": len(gt_lines) - len(matched_gt_indices),
        "unmatched_predict_line_count": len(predict_lines) - len(matched_predict_indices),
        "matches": match_items,
    }


def compute_formula_metrics(
    gt_record: Dict[str, Any],
    predict_record: Dict[str, Any],
    w_slt: float = 0.4,
    w_opt: float = 0.6,
) -> Optional[Dict[str, Any]]:
    if _record_has_unk(gt_record) or _record_has_unk(predict_record):
        return None

    gt_formulas = _normalize_formula_items(gt_record.get("formula_list"), require_alignment=False)
    predict_formulas = _normalize_formula_items(predict_record.get("formula_list"), require_alignment=True)

    if not gt_formulas and not predict_formulas:
        return None

    gt_groups: Dict[Tuple[Any, int], List[Dict[str, Any]]] = defaultdict(list)
    for formula in gt_formulas:
        gt_groups[(formula.get("question_id"), formula["seq"])].append(formula)

    predict_groups: Dict[Tuple[Any, int], List[Dict[str, Any]]] = defaultdict(list)
    unmatched_predict: List[Dict[str, Any]] = []
    for formula in predict_formulas:
        if formula.get("question_id") is None or formula.get("gt_seq") is None:
            unmatched_predict.append(formula)
            continue
        predict_groups[(formula.get("question_id"), formula["gt_seq"])].append(formula)

    for formulas in gt_groups.values():
        formulas.sort(key=lambda item: item["formula_seq"])
    for formulas in predict_groups.values():
        formulas.sort(key=lambda item: item["formula_seq"])
    unmatched_predict.sort(key=lambda item: item["formula_seq"])

    slot_items: List[Dict[str, Any]] = []
    formula_errors: List[Dict[str, Any]] = []
    total_slt = 0.0
    total_opt = 0.0
    partial_slot_count = 0
    opaque_slot_count = 0

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

            gt_slt: Optional[_TreeNode] = None
            gt_opt: Optional[_TreeNode] = None
            gt_meta: Optional[Dict[str, Any]] = None
            if gt_formula is not None:
                gt_slt, gt_opt, gt_meta = _build_formula_trees(gt_formula.get("formula"))

            predict_slt: Optional[_TreeNode] = None
            predict_opt: Optional[_TreeNode] = None
            predict_meta: Optional[Dict[str, Any]] = None
            if predict_formula is not None:
                predict_slt, predict_opt, predict_meta = _build_formula_trees(predict_formula.get("formula"))

            gt_parse_status = None if gt_meta is None else gt_meta.get("parse_status")
            predict_parse_status = None if predict_meta is None else predict_meta.get("parse_status")
            if "opaque" in {gt_parse_status, predict_parse_status}:
                slot_parse_status = "opaque"
                opaque_slot_count += 1
            elif "partial" in {gt_parse_status, predict_parse_status}:
                slot_parse_status = "partial"
                partial_slot_count += 1
            else:
                slot_parse_status = "full"

            if gt_formula is None:
                slot_items.append(
                    {
                        "question_id": question_id,
                        "gt_seq": gt_seq,
                        "gt_formula_seq": None,
                        "predict_formula_seq": predict_formula["formula_seq"],
                        "gt_formula": None,
                        "predict_formula": predict_formula.get("formula"),
                        "slt_teds": 0.0,
                        "opt_teds": 0.0,
                        "gt_parse_status": None,
                        "predict_parse_status": predict_parse_status,
                        "slot_parse_status": slot_parse_status,
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
                        "slt_teds": 0.0,
                        "opt_teds": 0.0,
                        "gt_parse_status": gt_parse_status,
                        "predict_parse_status": None,
                        "slot_parse_status": slot_parse_status,
                        "error": "missing_predict_formula",
                    }
                )
                continue

            if predict_meta is not None and predict_meta.get("normalized_is_empty"):
                slot_slt = 0.0
                slot_opt = 0.0
                slot_error = "formula_parse_failed"
                formula_errors.append(
                    {
                        "reason": "formula_parse_failed",
                        "question_id": question_id,
                        "gt_seq": gt_seq,
                        "predict_formula_seq": predict_formula["formula_seq"],
                    }
                )
            else:
                slot_slt = _tree_teds(predict_slt, gt_slt)
                slot_opt = _tree_teds(predict_opt, gt_opt)
                slot_error = None

            total_slt += slot_slt
            total_opt += slot_opt
            slot_items.append(
                {
                    "question_id": question_id,
                    "gt_seq": gt_seq,
                    "gt_formula_seq": gt_formula["formula_seq"],
                    "predict_formula_seq": predict_formula["formula_seq"],
                    "gt_formula": gt_formula.get("formula"),
                    "predict_formula": predict_formula.get("formula"),
                    "slt_teds": slot_slt,
                    "opt_teds": slot_opt,
                    "gt_parse_status": gt_parse_status,
                    "predict_parse_status": predict_parse_status,
                    "slot_parse_status": slot_parse_status,
                    "error": slot_error,
                }
            )

    for predict_formula in unmatched_predict:
        _, _, predict_meta = _build_formula_trees(predict_formula.get("formula"))
        predict_parse_status = predict_meta.get("parse_status")
        if predict_parse_status == "opaque":
            opaque_slot_count += 1
        elif predict_parse_status == "partial":
            partial_slot_count += 1
        slot_items.append(
            {
                "question_id": predict_formula.get("question_id"),
                "gt_seq": predict_formula.get("gt_seq"),
                "gt_formula_seq": None,
                "predict_formula_seq": predict_formula["formula_seq"],
                "gt_formula": None,
                "predict_formula": predict_formula.get("formula"),
                "slt_teds": 0.0,
                "opt_teds": 0.0,
                "gt_parse_status": None,
                "predict_parse_status": predict_parse_status,
                "slot_parse_status": predict_parse_status,
                "error": "extra_predict_formula",
            }
        )

    slot_count = len(slot_items)
    if slot_count == 0:
        return None

    sample_slt = total_slt / slot_count
    sample_opt = total_opt / slot_count
    return {
        "slot_count": slot_count,
        "slt_teds": sample_slt,
        "opt_teds": sample_opt,
        "formula_score": (w_slt * sample_slt) + (w_opt * sample_opt),
        "w_slt": w_slt,
        "w_opt": w_opt,
        "slots": slot_items,
        "errors": formula_errors,
        "partial_slot_count": partial_slot_count,
        "opaque_slot_count": opaque_slot_count,
    }


def compute_reading_order_metrics(
    gt_record: Dict[str, Any],
    predict_record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if _record_has_unk(gt_record) or _record_has_unk(predict_record):
        return None

    if gt_record.get("QuestionType") != CONSTRUCTED_RESPONSE:
        return None

    gt_transcription = gt_record.get("transcription")
    predict_transcription = predict_record.get("transcription")
    if not isinstance(gt_transcription, list) or not isinstance(predict_transcription, list):
        return None

    gt_items = _normalize_transcription_items(gt_transcription)
    predict_items = _normalize_transcription_items(predict_transcription)
    if not predict_items:
        return None

    matched_items = [
        item
        for item in predict_items
        if item.get("match_status") == MATCHED_STATUS and item.get("gt_seq") is not None
    ]
    matched_line_count = len(matched_items)
    predict_line_count = len(predict_items)
    mcr = matched_line_count / predict_line_count

    evidence_flags: List[str] = []

    gt_counts: Dict[Any, int] = defaultdict(int)
    for item in gt_items:
        gt_counts[item.get("question_id")] += 1

    continuity_scores = []
    continuity_weights = []
    for question_id, count in gt_counts.items():
        if count < 2:
            continue
        positions = [
            index
            for index, item in enumerate(matched_items)
            if item.get("question_id") == question_id
        ]
        if not positions:
            continuity_scores.append(0.0)
            continuity_weights.append(count)
            continue
        segments = 1
        for position_index in range(1, len(positions)):
            if positions[position_index] != positions[position_index - 1] + 1:
                segments += 1
        breaks = segments - 1
        continuity_scores.append(1.0 / (1 + breaks))
        continuity_weights.append(count)

    if continuity_weights:
        bcs = sum(score * weight for score, weight in zip(continuity_scores, continuity_weights)) / sum(
            continuity_weights
        )
    else:
        bcs = 1.0
        evidence_flags.append("no_continuity_evidence")

    gt_first_order: List[Any] = []
    seen_gt_questions: set[Any] = set()
    for item in gt_items:
        question_id = item.get("question_id")
        if question_id not in seen_gt_questions:
            seen_gt_questions.add(question_id)
            gt_first_order.append(question_id)

    predict_first_order: List[Any] = []
    seen_predict_questions: set[Any] = set()
    for item in matched_items:
        question_id = item.get("question_id")
        if question_id not in seen_predict_questions:
            seen_predict_questions.add(question_id)
            predict_first_order.append(question_id)

    shared_questions = [question_id for question_id in gt_first_order if question_id in seen_predict_questions]
    if len(shared_questions) < 2:
        sqa = 1.0
        evidence_flags.append("no_subquestion_order_evidence")
    else:
        predict_shared_order = [question_id for question_id in predict_first_order if question_id in set(shared_questions)]
        sqa = _pairwise_order_accuracy(shared_questions, predict_shared_order)

    comparable_pairs = 0
    correct_pairs = 0
    predict_by_question: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for item in matched_items:
        predict_by_question[item.get("question_id")].append(item)
    for items in predict_by_question.values():
        items.sort(key=lambda item: item["seq"])
        gt_sequences = [item["gt_seq"] for item in items if item.get("gt_seq") is not None]
        for left_index in range(len(gt_sequences)):
            for right_index in range(left_index + 1, len(gt_sequences)):
                comparable_pairs += 1
                if gt_sequences[left_index] < gt_sequences[right_index]:
                    correct_pairs += 1
    if comparable_pairs == 0:
        iqa = 1.0
        evidence_flags.append("no_intra_question_order_evidence")
    else:
        iqa = correct_pairs / comparable_pairs

    ros: Optional[float]
    if mcr < 0.6:
        status = "LOW_EVIDENCE"
        ros = None
    elif bcs < 1.0:
        status = "FRAGMENTED"
        ros = 0.6 * ((0.5 * bcs) + (0.3 * sqa) + (0.2 * iqa))
    elif sqa < 0.9:
        status = "MISORDERED"
        ros = 0.6 + (0.2 * sqa) + (0.1 * iqa)
    else:
        status = "CLEAN" if iqa >= 0.9 else "MINOR_INNER_DISORDER"
        ros = 0.9 + (0.1 * iqa)

    if ros is not None:
        ros = min(1.0, ros)

    return {
        "mcr": mcr,
        "bcs": bcs,
        "sqa": sqa,
        "iqa": iqa,
        "ros": ros,
        "status": status,
        "matched_line_count": matched_line_count,
        "predict_line_count": predict_line_count,
        "evidence_flags": evidence_flags,
    }


def classify_refusal_sample(
    gt_record: Dict[str, Any],
    predict_record: Dict[str, Any],
) -> Dict[str, Any]:
    gt_label = _record_has_unk(gt_record)
    pred_label = _record_has_unk(predict_record)
    if gt_label and pred_label:
        confusion = "TP"
    elif pred_label:
        confusion = "FP"
    elif gt_label:
        confusion = "FN"
    else:
        confusion = "TN"
    return {
        "gt_label": gt_label,
        "pred_label": pred_label,
        "confusion": confusion,
    }


def aggregate_stem_metrics(metrics: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    applicable = [metric for metric in metrics if metric is not None]
    total_matched_chars = sum(metric.get("matched_gt_chars", 0) for metric in applicable)
    total_score = sum(metric["weighted_score_sum"] for metric in applicable)
    return {
        "overall_stem_acc": _stem_accuracy_from_scores(total_score, total_matched_chars),
        "applicable_samples": len(applicable),
    }


def aggregate_formula_metrics(metrics: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    applicable = [metric for metric in metrics if metric is not None]
    total_slots = sum(metric["slot_count"] for metric in applicable)
    total_slt = sum(metric["slot_count"] * metric["slt_teds"] for metric in applicable)
    total_opt = sum(metric["slot_count"] * metric["opt_teds"] for metric in applicable)
    total_partial_slots = sum(metric.get("partial_slot_count", 0) for metric in applicable)
    total_opaque_slots = sum(metric.get("opaque_slot_count", 0) for metric in applicable)
    w_slt = applicable[0]["w_slt"] if applicable else 0.4
    w_opt = applicable[0]["w_opt"] if applicable else 0.6
    overall_slt = None if total_slots == 0 else total_slt / total_slots
    overall_opt = None if total_slots == 0 else total_opt / total_slots
    if total_slots == 0:
        overall_formula = None
    else:
        overall_formula = (w_slt * overall_slt) + (w_opt * overall_opt)

    def _summarize_subset(prefix: str, predicate: Any) -> Dict[str, Any]:
        subset_slots = 0
        subset_slt = 0.0
        subset_opt = 0.0
        for metric in applicable:
            for slot in metric.get("slots", []):
                if not predicate(slot):
                    continue
                subset_slots += 1
                subset_slt += slot["slt_teds"]
                subset_opt += slot["opt_teds"]

        if subset_slots == 0:
            subset_slt_avg = None
            subset_opt_avg = None
            subset_formula = None
        else:
            subset_slt_avg = subset_slt / subset_slots
            subset_opt_avg = subset_opt / subset_slots
            subset_formula = (w_slt * subset_slt_avg) + (w_opt * subset_opt_avg)

        return {
            f"{prefix}_slt_teds": subset_slt_avg,
            f"{prefix}_opt_teds": subset_opt_avg,
            f"{prefix}_formula_score": subset_formula,
            f"{prefix}_formula_slot_count": subset_slots,
        }

    return {
        "overall_slt_teds": overall_slt,
        "overall_opt_teds": overall_opt,
        "overall_formula_score": overall_formula,
        "formula_slot_count": total_slots,
        "formula_partial_slot_count": total_partial_slots,
        "formula_opaque_slot_count": total_opaque_slots,
        **_summarize_subset("matched", lambda slot: slot.get("error") is None),
        **_summarize_subset(
            "full_parse",
            lambda slot: slot.get("error") is None and slot.get("slot_parse_status") == "full",
        ),
        "applicable_samples": len(applicable),
    }


def aggregate_reading_order_metrics(metrics: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    applicable = [metric for metric in metrics if metric is not None]
    scored = [metric for metric in applicable if metric.get("ros") is not None]

    def _average(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    applicable_count = len(applicable)
    scored_count = len(scored)
    return {
        "overall_mcr": _average([metric["mcr"] for metric in applicable]),
        "overall_bcs": _average([metric["bcs"] for metric in applicable]),
        "overall_sqa": _average([metric["sqa"] for metric in applicable]),
        "overall_iqa": _average([metric["iqa"] for metric in applicable]),
        "overall_ros": _average([metric["ros"] for metric in scored]),
        "ros_coverage": None if applicable_count == 0 else scored_count / applicable_count,
        "applicable_samples": applicable_count,
        "ros_scored_samples": scored_count,
    }


def aggregate_refusal_metrics(metrics: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for metric in metrics:
        confusion = metric.get("confusion")
        if confusion == "TP":
            tp += 1
        elif confusion == "FP":
            fp += 1
        elif confusion == "FN":
            fn += 1
        elif confusion == "TN":
            tn += 1

    predicted_positive = tp + fp
    actual_positive = tp + fn
    if predicted_positive == 0 and actual_positive == 0:
        precision = None
        recall = None
        f1 = None
    else:
        precision = 0.0 if predicted_positive == 0 else tp / predicted_positive
        recall = 0.0 if actual_positive == 0 else tp / actual_positive
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

    return {
        "overall_refusal_precision": precision,
        "overall_refusal_recall": recall,
        "overall_refusal_f1": f1,
        "overall_hallucination_rate": _safe_divide(fn, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_composite_score(
    overall_metrics: Dict[str, Optional[float]],
    alpha_stem: float = 0.40,
    alpha_formula: float = 0.35,
    alpha_ros: float = 0.20,
    alpha_refusal: float = 0.05,
) -> Dict[str, Any]:
    configured_weights = {
        "stem": alpha_stem,
        "formula": alpha_formula,
        "ros": alpha_ros,
        "refusal": alpha_refusal,
    }
    values = {
        "stem": overall_metrics.get("stem_acc"),
        "formula": overall_metrics.get("matched_formula_score"),
        "ros": overall_metrics.get("ros"),
        "refusal": overall_metrics.get("refusal_f1"),
    }
    available = {
        name: configured_weights[name]
        for name, value in values.items()
        if value is not None
    }
    if not available:
        return {
            "composite_score": None,
            "effective_weights": {},
            "configured_weights": configured_weights,
        }

    total_weight = sum(available.values())
    effective_weights = {
        name: weight / total_weight
        for name, weight in available.items()
    }
    composite_score = sum(
        effective_weights[name] * values[name]
        for name in effective_weights
    )
    return {
        "composite_score": composite_score,
        "effective_weights": effective_weights,
        "configured_weights": configured_weights,
    }


__all__ = [
    "FormulaParseError",
    "compute_stem_metrics",
    "compute_formula_metrics",
    "compute_reading_order_metrics",
    "classify_refusal_sample",
    "aggregate_stem_metrics",
    "aggregate_formula_metrics",
    "aggregate_reading_order_metrics",
    "aggregate_refusal_metrics",
    "compute_composite_score",
]
