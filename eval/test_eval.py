#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from eval.eval import main
    from eval.metric import (
        FormulaParseError,
        _build_formula_trees,
        _tokenize_formula,
        _tree_teds,
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
except ModuleNotFoundError:
    from eval import main
    from metric import (
        FormulaParseError,
        _build_formula_trees,
        _tokenize_formula,
        _tree_teds,
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


def make_gt_record(**overrides):
    record = {
        "filename": "sample.jpg",
        "idx": 1,
        "QuestionType": "ConstructedResponse",
        "transcription": [
            {"seq": 1, "question_id": "Q001", "content": "First step"},
            {"seq": 2, "question_id": "Q001", "content": "x=1"},
            {"seq": 3, "question_id": "Q002", "content": "Question two"},
        ],
        "formula_list": [
            {
                "formula_seq": 1,
                "seq": 2,
                "question_id": "Q001",
                "formula": "x=1",
            }
        ],
        "final_answer": "",
    }
    record.update(overrides)
    return record


def make_predict_record(**overrides):
    record = {
        "filename": "sample.jpg",
        "idx": 1,
        "QuestionType": "ConstructedResponse",
        "transcription": [
            {
                "seq": 1,
                "question_id": "Q001",
                "gt_seq": 1,
                "match_status": "matched",
                "match_score": 1.0,
                "content": "First step",
            },
            {
                "seq": 2,
                "question_id": "Q001",
                "gt_seq": 2,
                "match_status": "matched",
                "match_score": 1.0,
                "content": "x=1",
            },
            {
                "seq": 3,
                "question_id": "Q002",
                "gt_seq": 3,
                "match_status": "matched",
                "match_score": 1.0,
                "content": "Question two",
            },
        ],
        "formula_list": [
            {
                "formula_seq": 1,
                "seq": 2,
                "question_id": "Q001",
                "gt_seq": 2,
                "formula": "x=1",
            }
        ],
        "final_answer": "",
    }
    record.update(overrides)
    return record


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MetricTests(unittest.TestCase):
    def test_stem_metrics_support_global_reordering(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "A"},
                {"seq": 2, "question_id": "Q001", "content": "B"},
                {"seq": 3, "question_id": "Q002", "content": "C"},
            ]
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q002",
                    "gt_seq": 3,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "C",
                },
                {
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A",
                },
                {
                    "seq": 3,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B",
                },
            ]
        )

        metrics = compute_stem_metrics(gt_record, predict_record, 0.70)

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["stem_acc"], 1.0)
        self.assertEqual(metrics["matched_line_count"], 3)

    def test_stem_metrics_use_declared_gt_seq_instead_of_rematching_content(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "A"},
                {"seq": 2, "question_id": "Q001", "content": "B"},
            ],
            formula_list=[],
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B",
                },
                {
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A",
                },
            ],
            formula_list=[],
        )

        metrics = compute_stem_metrics(gt_record, predict_record, 0.70)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["matched_line_count"], 2)
        self.assertAlmostEqual(metrics["stem_acc"], 0.0)
        self.assertEqual(metrics["matches"][0]["gt_line_index"], 1)
        self.assertEqual(metrics["matches"][0]["predict_line_index"], 1)

    def test_stem_metrics_penalize_missing_lines_and_noise(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "ABCDE"},
                {"seq": 2, "question_id": "Q001", "content": "FGHIJ"},
            ],
            formula_list=[],
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "ABCDE",
                },
                {
                    "seq": 2,
                    "question_id": None,
                    "gt_seq": None,
                    "match_status": "unmatched",
                    "match_score": 0.0,
                    "content": "2026 Grade 11 Midterm Math Exam",
                },
            ],
            formula_list=[],
        )

        metrics = compute_stem_metrics(gt_record, predict_record, 0.70)

        self.assertAlmostEqual(metrics["stem_acc"], 1.0)
        self.assertEqual(metrics["weighted_gt_chars"], 10)
        self.assertEqual(metrics["matched_gt_chars"], 5)
        self.assertEqual(metrics["matched_line_count"], 1)
        self.assertEqual(metrics["unmatched_predict_line_count"], 1)

    def test_stem_metrics_accept_string_passthrough_inputs(self):
        gt_record = make_gt_record(
            QuestionType="MultipleChoice",
            transcription="Choose A\nChoose B",
            formula_list=[],
        )
        predict_record = make_predict_record(
            QuestionType="MultipleChoice",
            transcription="Choose B\nChoose A",
            formula_list=[],
        )

        metrics = compute_stem_metrics(gt_record, predict_record, 0.70)

        self.assertAlmostEqual(metrics["stem_acc"], 1.0)
        self.assertEqual(metrics["gt_line_count"], 2)

    def test_stem_metrics_raise_for_invalid_matched_gt_seq(self):
        gt_record = make_gt_record(
            transcription=[{"seq": 1, "question_id": "Q001", "content": "ABCD"}],
            formula_list=[],
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": None,
                    "match_status": "matched",
                    "match_score": 0.75,
                    "content": "ABCE",
                }
            ],
            formula_list=[],
        )

        with self.assertRaisesRegex(ValueError, "gt_seq"):
            compute_stem_metrics(gt_record, predict_record, 0.95)

    def test_stem_metrics_raise_for_duplicated_matched_gt_seq(self):
        gt_record = make_gt_record(
            transcription=[{"seq": 1, "question_id": "Q001", "content": "ABCD"}],
            formula_list=[],
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "ABCD",
                },
                {
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "ABCD",
                },
            ],
            formula_list=[],
        )

        with self.assertRaisesRegex(ValueError, "duplicated matched gt_seq"):
            compute_stem_metrics(gt_record, predict_record, 0.95)

    def test_stem_metrics_threshold_filter_marks_low_similarity_unmatched_for_passthrough(self):
        gt_record = make_gt_record(
            QuestionType="MultipleChoice",
            transcription="ABCD",
            formula_list=[],
        )
        predict_record = make_predict_record(
            QuestionType="MultipleChoice",
            transcription="ABCE",
            formula_list=[],
        )

        metrics = compute_stem_metrics(gt_record, predict_record, 0.95)

        self.assertEqual(metrics["matched_line_count"], 0)
        self.assertEqual(metrics["unmatched_gt_line_count"], 1)
        self.assertEqual(metrics["unmatched_predict_line_count"], 1)
        self.assertIsNone(metrics["stem_acc"])
        self.assertEqual(metrics["matched_gt_chars"], 0)

    def test_aggregate_stem_metrics_use_only_matched_gt_chars_in_denominator(self):
        aggregate = aggregate_stem_metrics(
            [
                {
                    "stem_acc": 1.0,
                    "weighted_score_sum": 5.0,
                    "weighted_gt_chars": 10,
                    "matched_gt_chars": 5,
                },
                {
                    "stem_acc": 0.5,
                    "weighted_score_sum": 3.0,
                    "weighted_gt_chars": 20,
                    "matched_gt_chars": 6,
                },
                {
                    "stem_acc": None,
                    "weighted_score_sum": 0.0,
                    "weighted_gt_chars": 8,
                    "matched_gt_chars": 0,
                },
            ]
        )

        self.assertAlmostEqual(aggregate["overall_stem_acc"], 8 / 11)
        self.assertEqual(aggregate["applicable_samples"], 3)

    def test_stem_metrics_return_none_when_either_sample_is_unk(self):
        metrics = compute_stem_metrics(
            make_gt_record(formula_list=[]),
            make_predict_record(final_answer="[UNK]", formula_list=[]),
            0.70,
        )

        self.assertIsNone(metrics)

    def test_formula_metrics_handle_exact_match_and_missing_slot(self):
        gt_record = make_gt_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "formula": "x=1",
                },
                {
                    "formula_seq": 2,
                    "seq": 2,
                    "question_id": "Q001",
                    "formula": "y=2",
                },
            ]
        )
        predict_record = make_predict_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "formula": "x=1",
                }
            ]
        )

        metrics = compute_formula_metrics(gt_record, predict_record, 0.4, 0.6)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["slot_count"], 2)
        self.assertAlmostEqual(metrics["slt_teds"], 0.5)
        self.assertAlmostEqual(metrics["opt_teds"], 0.5)
        self.assertAlmostEqual(metrics["formula_score"], 0.5)

    def test_formula_metrics_record_predict_parse_failure_without_crashing(self):
        gt_record = make_gt_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "formula": "x=1",
                }
            ]
        )
        predict_record = make_predict_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "formula": None,
                }
            ]
        )

        metrics = compute_formula_metrics(gt_record, predict_record, 0.4, 0.6)

        self.assertEqual(metrics["slot_count"], 1)
        self.assertEqual(metrics["slt_teds"], 0.0)
        self.assertEqual(metrics["opt_teds"], 0.0)
        self.assertEqual(metrics["errors"][0]["reason"], "formula_parse_failed")

    def test_formula_metrics_add_extra_predict_slot(self):
        gt_record = make_gt_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "formula": "x=1",
                }
            ]
        )
        predict_record = make_predict_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "formula": "x=1",
                },
                {
                    "formula_seq": 2,
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "formula": "y=2",
                },
            ]
        )

        metrics = compute_formula_metrics(gt_record, predict_record, 0.4, 0.6)

        self.assertEqual(metrics["slot_count"], 2)
        self.assertAlmostEqual(metrics["formula_score"], 0.5)
        self.assertEqual(metrics["slots"][1]["error"], "extra_predict_formula")

    def test_formula_metrics_return_none_when_either_sample_is_unk(self):
        metrics = compute_formula_metrics(
            make_gt_record(),
            make_predict_record(final_answer="[UNK]"),
            0.4,
            0.6,
        )

        self.assertIsNone(metrics)

    def test_formula_metrics_skip_samples_without_any_formula_slots(self):
        gt_record = make_gt_record(formula_list=[])
        predict_record = make_predict_record(formula_list=[])

        metrics = compute_formula_metrics(gt_record, predict_record, 0.4, 0.6)

        self.assertIsNone(metrics)

    def test_formula_metrics_degrade_invalid_gt_formula_without_crashing(self):
        gt_record = make_gt_record(
            formula_list=[
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "formula": None,
                }
            ]
        )

        metrics = compute_formula_metrics(gt_record, make_predict_record(), 0.4, 0.6)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["slot_count"], 1)
        self.assertEqual(metrics["opaque_slot_count"], 1)
        self.assertEqual(metrics["slots"][0]["gt_parse_status"], "opaque")
        self.assertEqual(metrics["slots"][0]["slot_parse_status"], "opaque")
        self.assertGreaterEqual(metrics["formula_score"], 0.0)
        self.assertLessEqual(metrics["formula_score"], 1.0)

    def test_formula_metrics_parse_prompt_shapes_and_record_status(self):
        samples = [
            (r"E(x,y,0)", "full"),
            (r"OP\perp", "full"),
            (r"OB=\sqrt{1+1}=\sqrt{2}", "full"),
            (r"a\in[0,1)", "partial"),
            (r"\Rightarrow1<x<\frac{1}{a}", "full"),
            (r"\left\{x_{1}=-\frac{\sqrt{2}}{2}-y", "partial"),
        ]

        for formula, expected_status in samples:
            with self.subTest(formula=formula):
                metrics = compute_formula_metrics(
                    make_gt_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "formula": formula,
                            }
                        ]
                    ),
                    make_predict_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "gt_seq": 2,
                                "formula": formula,
                            }
                        ]
                    ),
                )

                slot = metrics["slots"][0]
                self.assertAlmostEqual(metrics["formula_score"], 1.0)
                self.assertEqual(slot["gt_parse_status"], expected_status)
                self.assertEqual(slot["predict_parse_status"], expected_status)
                self.assertEqual(slot["slot_parse_status"], expected_status)

    def test_formula_metrics_cover_gt_high_frequency_shapes(self):
        samples = [
            r"\overrightarrow{AP}=(-\frac{\sqrt{2}}{2},\frac{\sqrt{2}}{2},1)",
            r"(-\infty,0)\cup[1,+\infty)",
            r"\cos^{2}B",
        ]

        for formula in samples:
            with self.subTest(formula=formula):
                metrics = compute_formula_metrics(
                    make_gt_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "formula": formula,
                            }
                        ]
                    ),
                    make_predict_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "gt_seq": 2,
                                "formula": formula,
                            }
                        ]
                    ),
                )

                self.assertAlmostEqual(metrics["formula_score"], 1.0)
                self.assertIn(metrics["slots"][0]["slot_parse_status"], {"full", "partial"})

    def test_formula_tokenizer_splits_known_command_prefixes(self):
        self.assertEqual(
            [token.value for token in _tokenize_formula(r"BC\perpPA")],
            ["BC", r"\perp", "PA"],
        )
        self.assertEqual(
            [token.value for token in _tokenize_formula(r"AD\capOB=O")],
            ["AD", r"\cap", "OB", "=", "O"],
        )
        self.assertEqual(
            [token.value for token in _tokenize_formula(r"\overrightarrow{AP}=\frac{1}{2}")],
            [r"\overrightarrow", "{", "AP", "}", "=", r"\frac", "{", "1", "}", "{", "2", "}"],
        )

    def test_formula_metrics_normalize_equivalent_relation_commands(self):
        samples = [
            (r"x\le1", r"x\leq1"),
            (r"x\ge1", r"x\geq1"),
        ]

        for gt_formula, predict_formula in samples:
            with self.subTest(gt_formula=gt_formula, predict_formula=predict_formula):
                metrics = compute_formula_metrics(
                    make_gt_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "formula": gt_formula,
                            }
                        ]
                    ),
                    make_predict_record(
                        formula_list=[
                            {
                                "formula_seq": 1,
                                "seq": 2,
                                "question_id": "Q001",
                                "gt_seq": 2,
                                "formula": predict_formula,
                            }
                        ]
                    ),
                )

                self.assertAlmostEqual(metrics["slt_teds"], 1.0)
                self.assertAlmostEqual(metrics["opt_teds"], 1.0)
                self.assertAlmostEqual(metrics["formula_score"], 1.0)

    def test_formula_parser_recognizes_relation_and_set_structure(self):
        _, perp_opt, perp_meta = _build_formula_trees(r"BC\perpPA")
        _, set_opt, set_meta = _build_formula_trees(r"AD\capOB=O")

        self.assertEqual(perp_meta["parse_status"], "full")
        self.assertEqual(perp_opt.kind, "op")
        self.assertEqual(perp_opt.label, "RelChain")
        self.assertEqual([child.label for child in perp_opt.children], ["Var:BC", r"Rel:\perp", "Var:PA"])

        self.assertEqual(set_meta["parse_status"], "full")
        self.assertEqual(set_opt.kind, "op")
        self.assertEqual(set_opt.label, "RelChain")
        self.assertEqual(set_opt.children[0].label, "Intersect")
        self.assertEqual(set_opt.children[1].label, "Rel:=")
        self.assertEqual(set_opt.children[2].label, "Var:O")

    def test_formula_opt_teds_is_more_sensitive_to_attachment_structure(self):
        gt_slt, gt_opt, _ = _build_formula_trees(r"x_{1}=\frac{1}{a}")
        predict_slt, predict_opt, _ = _build_formula_trees(r"x=\frac{1}{a}")

        self.assertGreater(_tree_teds(predict_slt, gt_slt), _tree_teds(predict_opt, gt_opt))

    def test_formula_slt_is_less_sensitive_to_relation_identity_than_opt(self):
        gt_slt, gt_opt, _ = _build_formula_trees(r"a\le1")
        predict_slt, predict_opt, _ = _build_formula_trees(r"a\ge1")

        self.assertAlmostEqual(_tree_teds(predict_slt, gt_slt), 1.0)
        self.assertGreater(_tree_teds(predict_slt, gt_slt), _tree_teds(predict_opt, gt_opt))

    def test_formula_slt_is_less_sensitive_to_inline_operator_identity_than_opt(self):
        gt_slt, gt_opt, _ = _build_formula_trees(r"A\cupB")
        predict_slt, predict_opt, _ = _build_formula_trees(r"A\capB")

        self.assertAlmostEqual(_tree_teds(predict_slt, gt_slt), 1.0)
        self.assertGreater(_tree_teds(predict_slt, gt_slt), _tree_teds(predict_opt, gt_opt))

    def test_formula_metrics_aggregate_reports_partial_and_opaque_slots(self):
        full_metric = compute_formula_metrics(
            make_gt_record(),
            make_predict_record(),
        )
        partial_metric = compute_formula_metrics(
            make_gt_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": r"a\in[0,1)",
                    }
                ]
            ),
            make_predict_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "gt_seq": 2,
                        "formula": r"a\in[0,1)",
                    }
                ]
            ),
        )
        opaque_metric = compute_formula_metrics(
            make_gt_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": None,
                    }
                ]
            ),
            make_predict_record(),
        )

        aggregate = aggregate_formula_metrics([full_metric, partial_metric, opaque_metric])

        self.assertEqual(aggregate["formula_slot_count"], 3)
        self.assertEqual(aggregate["formula_partial_slot_count"], 1)
        self.assertEqual(aggregate["formula_opaque_slot_count"], 1)

    def test_formula_metrics_aggregate_reports_matched_and_full_parse_subsets(self):
        full_metric = compute_formula_metrics(
            make_gt_record(),
            make_predict_record(),
        )
        partial_metric = compute_formula_metrics(
            make_gt_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": r"a\in[0,1)",
                    }
                ]
            ),
            make_predict_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "gt_seq": 2,
                        "formula": r"a\in[0,1)",
                    }
                ]
            ),
        )
        missing_metric = compute_formula_metrics(
            make_gt_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": "y=2",
                    }
                ]
            ),
            make_predict_record(formula_list=[]),
        )

        aggregate = aggregate_formula_metrics([full_metric, partial_metric, missing_metric])

        self.assertEqual(aggregate["matched_formula_slot_count"], 2)
        self.assertAlmostEqual(aggregate["matched_slt_teds"], 1.0)
        self.assertAlmostEqual(aggregate["matched_opt_teds"], 1.0)
        self.assertAlmostEqual(aggregate["matched_formula_score"], 1.0)
        self.assertEqual(aggregate["full_parse_formula_slot_count"], 1)
        self.assertAlmostEqual(aggregate["full_parse_slt_teds"], 1.0)
        self.assertAlmostEqual(aggregate["full_parse_opt_teds"], 1.0)
        self.assertAlmostEqual(aggregate["full_parse_formula_score"], 1.0)

    def test_formula_metrics_aggregate_reports_empty_subset_as_null(self):
        missing_metric = compute_formula_metrics(
            make_gt_record(
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": "y=2",
                    }
                ]
            ),
            make_predict_record(formula_list=[]),
        )

        aggregate = aggregate_formula_metrics([missing_metric])

        self.assertEqual(aggregate["matched_formula_slot_count"], 0)
        self.assertIsNone(aggregate["matched_slt_teds"])
        self.assertIsNone(aggregate["matched_opt_teds"])
        self.assertIsNone(aggregate["matched_formula_score"])
        self.assertEqual(aggregate["full_parse_formula_slot_count"], 0)
        self.assertIsNone(aggregate["full_parse_slt_teds"])
        self.assertIsNone(aggregate["full_parse_opt_teds"])
        self.assertIsNone(aggregate["full_parse_formula_score"])

    def test_reading_order_low_evidence(self):
        gt_record = make_gt_record()
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "First step",
                },
                {
                    "seq": 2,
                    "question_id": None,
                    "gt_seq": None,
                    "match_status": "unmatched",
                    "match_score": 0.0,
                    "content": "Noise 1",
                },
                {
                    "seq": 3,
                    "question_id": None,
                    "gt_seq": None,
                    "match_status": "unmatched",
                    "match_score": 0.0,
                    "content": "Noise 2",
                },
            ]
        )

        metrics = compute_reading_order_metrics(gt_record, predict_record)

        self.assertEqual(metrics["status"], "LOW_EVIDENCE")
        self.assertIsNone(metrics["ros"])
        self.assertAlmostEqual(metrics["mcr"], 1 / 3)

    def test_reading_order_fragmented(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "A1"},
                {"seq": 2, "question_id": "Q001", "content": "A2"},
                {"seq": 3, "question_id": "Q002", "content": "B1"},
                {"seq": 4, "question_id": "Q002", "content": "B2"},
            ]
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A1",
                },
                {
                    "seq": 2,
                    "question_id": "Q002",
                    "gt_seq": 3,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B1",
                },
                {
                    "seq": 3,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A2",
                },
                {
                    "seq": 4,
                    "question_id": "Q002",
                    "gt_seq": 4,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B2",
                },
            ]
        )

        metrics = compute_reading_order_metrics(gt_record, predict_record)

        self.assertEqual(metrics["status"], "FRAGMENTED")
        self.assertLess(metrics["bcs"], 1.0)
        self.assertLess(metrics["ros"], 0.6)

    def test_reading_order_ignores_unmatched_breaks_for_bcs(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "A1"},
                {"seq": 2, "question_id": "Q001", "content": "A2"},
            ],
            formula_list=[],
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A1",
                },
                {
                    "seq": 2,
                    "question_id": None,
                    "gt_seq": None,
                    "match_status": "unmatched",
                    "match_score": 0.0,
                    "content": "Noise",
                },
                {
                    "seq": 3,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A2",
                },
            ],
            formula_list=[],
        )

        metrics = compute_reading_order_metrics(gt_record, predict_record)

        self.assertEqual(metrics["bcs"], 1.0)
        self.assertEqual(metrics["status"], "CLEAN")
        self.assertAlmostEqual(metrics["mcr"], 2 / 3)

    def test_reading_order_misordered(self):
        gt_record = make_gt_record(
            transcription=[
                {"seq": 1, "question_id": "Q001", "content": "A1"},
                {"seq": 2, "question_id": "Q001", "content": "A2"},
                {"seq": 3, "question_id": "Q002", "content": "B1"},
                {"seq": 4, "question_id": "Q002", "content": "B2"},
            ]
        )
        predict_record = make_predict_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q002",
                    "gt_seq": 3,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B1",
                },
                {
                    "seq": 2,
                    "question_id": "Q002",
                    "gt_seq": 4,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "B2",
                },
                {
                    "seq": 3,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A1",
                },
                {
                    "seq": 4,
                    "question_id": "Q001",
                    "gt_seq": 2,
                    "match_status": "matched",
                    "match_score": 1.0,
                    "content": "A2",
                },
            ]
        )

        metrics = compute_reading_order_metrics(gt_record, predict_record)

        self.assertEqual(metrics["status"], "MISORDERED")
        self.assertEqual(metrics["bcs"], 1.0)
        self.assertLess(metrics["sqa"], 0.9)
        self.assertGreaterEqual(metrics["ros"], 0.6)

    def test_reading_order_clean_and_minor_inner_disorder_and_coverage(self):
        clean_metrics = compute_reading_order_metrics(make_gt_record(), make_predict_record())
        minor_metrics = compute_reading_order_metrics(
            make_gt_record(
                transcription=[
                    {"seq": 1, "question_id": "Q001", "content": "A1"},
                    {"seq": 2, "question_id": "Q001", "content": "A2"},
                    {"seq": 3, "question_id": "Q001", "content": "A3"},
                ],
                formula_list=[],
            ),
            make_predict_record(
                transcription=[
                    {
                        "seq": 1,
                        "question_id": "Q001",
                        "gt_seq": 1,
                        "match_status": "matched",
                        "match_score": 1.0,
                        "content": "A1",
                    },
                    {
                        "seq": 2,
                        "question_id": "Q001",
                        "gt_seq": 3,
                        "match_status": "matched",
                        "match_score": 1.0,
                        "content": "A3",
                    },
                    {
                        "seq": 3,
                        "question_id": "Q001",
                        "gt_seq": 2,
                        "match_status": "matched",
                        "match_score": 1.0,
                        "content": "A2",
                    },
                ],
                formula_list=[],
            ),
        )
        low_evidence_metrics = compute_reading_order_metrics(
            make_gt_record(formula_list=[]),
            make_predict_record(
                transcription=[
                    {
                        "seq": 1,
                        "question_id": "Q001",
                        "gt_seq": 1,
                        "match_status": "matched",
                        "match_score": 1.0,
                        "content": "First step",
                    },
                    {
                        "seq": 2,
                        "question_id": None,
                        "gt_seq": None,
                        "match_status": "unmatched",
                        "match_score": 0.0,
                        "content": "Noise 1",
                    },
                    {
                        "seq": 3,
                        "question_id": None,
                        "gt_seq": None,
                        "match_status": "unmatched",
                        "match_score": 0.0,
                        "content": "Noise 2",
                    },
                ],
                formula_list=[],
            ),
        )

        aggregate = aggregate_reading_order_metrics(
            [clean_metrics, minor_metrics, low_evidence_metrics]
        )

        self.assertEqual(clean_metrics["status"], "CLEAN")
        self.assertEqual(minor_metrics["status"], "MINOR_INNER_DISORDER")
        self.assertLess(minor_metrics["iqa"], 0.9)
        self.assertAlmostEqual(aggregate["ros_coverage"], 2 / 3)

    def test_reading_order_metrics_return_none_when_either_sample_is_unk(self):
        metrics = compute_reading_order_metrics(
            make_gt_record(),
            make_predict_record(final_answer="[UNK]"),
        )

        self.assertIsNone(metrics)

    def test_refusal_metrics_cover_tp_fp_fn_tn(self):
        tp = classify_refusal_sample(
            make_gt_record(transcription="[UNK]", final_answer="[UNK]"),
            make_predict_record(transcription="[UNK]", final_answer="[UNK]"),
        )
        fp = classify_refusal_sample(
            make_gt_record(QuestionType="FillBlank", transcription="1", formula_list=[]),
            make_predict_record(QuestionType="FillBlank", transcription="[UNK]", formula_list=[]),
        )
        fn = classify_refusal_sample(
            make_gt_record(transcription="[UNK]", final_answer="[UNK]"),
            make_predict_record(transcription="Concrete content", final_answer="Concrete content"),
        )
        tn = classify_refusal_sample(
            make_gt_record(QuestionType="FillBlank", transcription="1", formula_list=[]),
            make_predict_record(QuestionType="FillBlank", transcription="1", formula_list=[]),
        )

        aggregate = aggregate_refusal_metrics([tp, fp, fn, tn])

        self.assertEqual(tp["confusion"], "TP")
        self.assertEqual(fp["confusion"], "FP")
        self.assertEqual(fn["confusion"], "FN")
        self.assertEqual(tn["confusion"], "TN")
        self.assertAlmostEqual(aggregate["overall_refusal_precision"], 0.5)
        self.assertAlmostEqual(aggregate["overall_refusal_recall"], 0.5)
        self.assertAlmostEqual(aggregate["overall_refusal_f1"], 0.5)
        self.assertAlmostEqual(aggregate["overall_hallucination_rate"], 0.5)

    def test_composite_score_renormalizes_non_null_modules(self):
        composite = compute_composite_score(
            {
                "stem_acc": 0.8,
                "matched_formula_score": None,
                "ros": 0.6,
                "refusal_f1": None,
            },
            alpha_stem=0.4,
            alpha_formula=0.3,
            alpha_ros=0.2,
            alpha_refusal=0.1,
        )

        self.assertAlmostEqual(composite["composite_score"], (2 / 3) * 0.8 + (1 / 3) * 0.6)
        self.assertAlmostEqual(composite["effective_weights"]["stem"], 2 / 3)
        self.assertAlmostEqual(composite["effective_weights"]["ros"], 1 / 3)

    def test_composite_score_uses_matched_formula_score(self):
        composite = compute_composite_score(
            {
                "stem_acc": 0.8,
                "formula_score": 0.1,
                "matched_formula_score": 0.9,
                "ros": 0.6,
                "refusal_f1": 0.4,
            },
            alpha_stem=0.2,
            alpha_formula=0.4,
            alpha_ros=0.3,
            alpha_refusal=0.1,
        )

        expected = (0.2 * 0.8) + (0.4 * 0.9) + (0.3 * 0.6) + (0.1 * 0.4)
        self.assertAlmostEqual(composite["composite_score"], expected)

    def test_aggregate_refusal_metrics_fn_only_returns_zero_scores(self):
        fn_a = classify_refusal_sample(
            make_gt_record(transcription="[UNK]", final_answer="[UNK]"),
            make_predict_record(transcription="Concrete content", final_answer="Concrete content"),
        )
        fn_b = classify_refusal_sample(
            make_gt_record(transcription="[UNK]", final_answer="[UNK]"),
            make_predict_record(transcription="Still answered", final_answer="Still answered"),
        )

        aggregate = aggregate_refusal_metrics([fn_a, fn_b])

        self.assertEqual(fn_a["confusion"], "FN")
        self.assertEqual(fn_b["confusion"], "FN")
        self.assertEqual(aggregate["overall_refusal_precision"], 0.0)
        self.assertEqual(aggregate["overall_refusal_recall"], 0.0)
        self.assertEqual(aggregate["overall_refusal_f1"], 0.0)
        self.assertEqual(aggregate["overall_hallucination_rate"], 1.0)


class EvalIntegrationTests(unittest.TestCase):
    def test_main_batch_eval_writes_outputs_and_reports_missing_and_extra(self):
        gt_records = [
            make_gt_record(),
            make_gt_record(
                filename="unknown.jpg",
                idx=2,
                QuestionType="ConstructedResponse",
                transcription="[UNK]",
                formula_list=[],
                final_answer="[UNK]",
            ),
            make_gt_record(
                filename="choice.jpg",
                idx=3,
                QuestionType="MultipleChoice",
                transcription="A",
                formula_list=[],
            ),
        ]
        model_a_records = [
            make_predict_record(),
            make_predict_record(
                filename="unknown.jpg",
                idx=999,
                QuestionType="ConstructedResponse",
                transcription="[UNK]",
                formula_list=[],
                final_answer="[UNK]",
            ),
            make_predict_record(
                filename="extra.jpg",
                idx=99,
                QuestionType="FillBlank",
                transcription="Extra sample",
                formula_list=[],
                final_answer="Extra sample",
            ),
        ]
        model_b_records = [
            make_predict_record(),
            make_predict_record(
                filename="unknown.jpg",
                idx=2,
                QuestionType="ConstructedResponse",
                transcription="Continue generating content",
                formula_list=[],
                final_answer="Continue generating content",
            ),
            make_predict_record(
                filename="choice.jpg",
                idx=3,
                QuestionType="MultipleChoice",
                transcription="A",
                formula_list=[],
                final_answer="A",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "model-a" / "predict.json", model_a_records)
            write_json(predict_root / "model-b" / "predict.json", model_b_records)

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)

            sample_metrics = json.loads(
                (output_root / "model-a" / "sample_metrics.json").read_text(encoding="utf-8")
            )
            overall_metrics = json.loads(
                (output_root / "model-a" / "overall_metrics.json").read_text(encoding="utf-8")
            )
            error_report = json.loads(
                (output_root / "model-a" / "eval_errors.json").read_text(encoding="utf-8")
            )
            leaderboard = json.loads(
                (output_root / "leaderboard_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(sample_metrics), 3)
            self.assertEqual(overall_metrics["counts"]["paired_total"], 2)
            self.assertEqual(overall_metrics["counts"]["missing_predict_count"], 1)
            self.assertEqual(overall_metrics["counts"]["extra_predict_count"], 1)
            self.assertAlmostEqual(overall_metrics["counts"]["sample_coverage"], 2 / 3)
            self.assertEqual(sample_metrics[1]["idx"], 2)
            self.assertEqual(error_report["summary"]["sample_error_count"], 2)
            self.assertEqual(sample_metrics[2]["pair_status"], "missing_predict")
            self.assertIsNone(sample_metrics[1]["stem"])
            self.assertIsNone(sample_metrics[1]["formula"])
            self.assertIsNone(sample_metrics[1]["reading_order"])
            self.assertEqual(sample_metrics[1]["refusal"]["confusion"], "TP")
            self.assertAlmostEqual(overall_metrics["overall"]["stem_acc"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["formula_score"], 1.0)
            self.assertEqual(overall_metrics["overall"]["formula_slot_count"], 1)
            self.assertEqual(overall_metrics["overall"]["formula_partial_slot_count"], 0)
            self.assertEqual(overall_metrics["overall"]["formula_opaque_slot_count"], 0)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_slt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_opt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_formula_score"], 1.0)
            self.assertEqual(overall_metrics["overall"]["matched_formula_slot_count"], 1)
            self.assertAlmostEqual(overall_metrics["overall"]["full_parse_slt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["full_parse_opt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["full_parse_formula_score"], 1.0)
            self.assertEqual(overall_metrics["overall"]["full_parse_formula_slot_count"], 1)
            self.assertAlmostEqual(overall_metrics["overall"]["ros"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["refusal_f1"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["composite_score"], 1.0)
            self.assertEqual(error_report["summary"]["missing_predict_count"], 1)
            self.assertEqual(error_report["summary"]["extra_predict_count"], 1)
            self.assertEqual(
                list(sample_metrics[0].keys()),
                [
                    "filename",
                    "idx",
                    "question_type",
                    "pair_status",
                    "stem",
                    "formula",
                    "reading_order",
                    "refusal",
                    "errors",
                ],
            )
            self.assertEqual(
                list(overall_metrics["overall"].keys()),
                [
                    "stem_acc",
                    "slt_teds",
                    "opt_teds",
                    "formula_score",
                    "formula_slot_count",
                    "formula_partial_slot_count",
                    "formula_opaque_slot_count",
                    "matched_slt_teds",
                    "matched_opt_teds",
                    "matched_formula_score",
                    "matched_formula_slot_count",
                    "full_parse_slt_teds",
                    "full_parse_opt_teds",
                    "full_parse_formula_score",
                    "full_parse_formula_slot_count",
                    "mcr",
                    "bcs",
                    "sqa",
                    "iqa",
                    "ros",
                    "ros_coverage",
                    "refusal_precision",
                    "refusal_recall",
                    "refusal_f1",
                    "hallucination_rate",
                    "composite_score",
                    "w_total_stem",
                    "w_total_formula",
                    "w_total_ros",
                    "w_total_refusal",
                ],
            )
            self.assertAlmostEqual(leaderboard[0]["overall"]["matched_formula_score"], 1.0)
            self.assertEqual(leaderboard[0]["overall"]["full_parse_formula_slot_count"], 1)
            self.assertEqual(leaderboard[0]["model"], "model-a")
            self.assertEqual(leaderboard[1]["model"], "model-b")

    def test_main_predict_unk_samples_only_affect_refusal_metrics(self):
        gt_records = [
            make_gt_record(filename="normal.jpg", idx=1),
            make_gt_record(filename="predict-unk.jpg", idx=2),
        ]
        predict_records = [
            make_predict_record(filename="normal.jpg", idx=1),
            make_predict_record(
                filename="predict-unk.jpg",
                idx=2,
                QuestionType="ConstructedResponse",
                transcription="[UNK]",
                final_answer="[UNK]",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "solo" / "predict.json", predict_records)

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)

            sample_metrics = json.loads(
                (output_root / "solo" / "sample_metrics.json").read_text(encoding="utf-8")
            )
            overall_metrics = json.loads(
                (output_root / "solo" / "overall_metrics.json").read_text(encoding="utf-8")
            )

            self.assertEqual(sample_metrics[1]["filename"], "predict-unk.jpg")
            self.assertIsNone(sample_metrics[1]["stem"])
            self.assertIsNone(sample_metrics[1]["formula"])
            self.assertIsNone(sample_metrics[1]["reading_order"])
            self.assertEqual(sample_metrics[1]["refusal"]["confusion"], "FP")
            self.assertAlmostEqual(overall_metrics["overall"]["stem_acc"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["formula_score"], 1.0)
            self.assertEqual(overall_metrics["overall"]["formula_slot_count"], 1)
            self.assertEqual(overall_metrics["overall"]["matched_formula_slot_count"], 1)
            self.assertEqual(overall_metrics["overall"]["full_parse_formula_slot_count"], 1)
            self.assertAlmostEqual(overall_metrics["overall"]["ros"], 1.0)

    def test_main_single_model_filter_and_null_module_renormalization(self):
        gt_records = [
            make_gt_record(
                filename="single.jpg",
                idx=7,
                QuestionType="MultipleChoice",
                transcription="A",
                formula_list=[],
                final_answer="A",
            )
        ]
        predict_records = [
            make_predict_record(
                filename="single.jpg",
                idx=7,
                QuestionType="MultipleChoice",
                transcription="A",
                formula_list=[],
                final_answer="A",
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "solo" / "predict.json", predict_records)
            write_json(predict_root / "other" / "predict.json", [make_predict_record()])

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                    "--model",
                    "solo",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "solo" / "overall_metrics.json").exists())
            self.assertFalse((output_root / "other" / "overall_metrics.json").exists())

            overall_metrics = json.loads(
                (output_root / "solo" / "overall_metrics.json").read_text(encoding="utf-8")
            )
            leaderboard = json.loads(
                (output_root / "leaderboard_summary.json").read_text(encoding="utf-8")
            )

            self.assertAlmostEqual(overall_metrics["overall"]["stem_acc"], 1.0)
            self.assertIsNone(overall_metrics["overall"]["formula_score"])
            self.assertEqual(overall_metrics["overall"]["matched_formula_slot_count"], 0)
            self.assertIsNone(overall_metrics["overall"]["matched_formula_score"])
            self.assertEqual(overall_metrics["overall"]["full_parse_formula_slot_count"], 0)
            self.assertIsNone(overall_metrics["overall"]["full_parse_formula_score"])
            self.assertIsNone(overall_metrics["overall"]["ros"])
            self.assertIsNone(overall_metrics["overall"]["refusal_f1"])
            self.assertAlmostEqual(overall_metrics["overall"]["composite_score"], 1.0)
            self.assertEqual(len(leaderboard), 1)
            self.assertEqual(leaderboard[0]["model"], "solo")

    def test_main_leaderboard_formula_fields_use_matched_subset_metrics(self):
        gt_records = [
            make_gt_record(
                filename="formula-gap.jpg",
                idx=1,
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": "x=1",
                    },
                    {
                        "formula_seq": 2,
                        "seq": 2,
                        "question_id": "Q001",
                        "formula": "y=2",
                    },
                ],
            )
        ]
        predict_records = [
            make_predict_record(
                filename="formula-gap.jpg",
                idx=1,
                formula_list=[
                    {
                        "formula_seq": 1,
                        "seq": 2,
                        "question_id": "Q001",
                        "gt_seq": 2,
                        "formula": "x=1",
                    }
                ],
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "solo" / "predict.json", predict_records)

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)

            overall_metrics = json.loads(
                (output_root / "solo" / "overall_metrics.json").read_text(encoding="utf-8")
            )
            leaderboard = json.loads(
                (output_root / "leaderboard_summary.json").read_text(encoding="utf-8")
            )

            self.assertAlmostEqual(overall_metrics["overall"]["slt_teds"], 0.5)
            self.assertAlmostEqual(overall_metrics["overall"]["opt_teds"], 0.5)
            self.assertAlmostEqual(overall_metrics["overall"]["formula_score"], 0.5)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_slt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_opt_teds"], 1.0)
            self.assertAlmostEqual(overall_metrics["overall"]["matched_formula_score"], 1.0)
            self.assertAlmostEqual(leaderboard[0]["overall"]["slt_teds"], 1.0)
            self.assertAlmostEqual(leaderboard[0]["overall"]["opt_teds"], 1.0)
            self.assertAlmostEqual(leaderboard[0]["overall"]["formula_score"], 1.0)
            self.assertAlmostEqual(leaderboard[0]["overall"]["matched_formula_score"], 1.0)
            self.assertAlmostEqual(leaderboard[0]["overall"]["composite_score"], 1.0)

    def test_main_global_fatal_duplicate_gt_writes_no_outputs(self):
        gt_records = [make_gt_record(), make_gt_record()]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "model-a" / "predict.json", [make_predict_record()])

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse((output_root / "leaderboard_summary.json").exists())
            self.assertFalse((output_root / "model-a" / "eval_errors.json").exists())

    def test_main_model_fatal_duplicate_predict_only_writes_error_report(self):
        gt_records = [make_gt_record(filename="dup.jpg", idx=1, formula_list=[])]
        good_predict_records = [
            make_predict_record(filename="dup.jpg", idx=1, formula_list=[]),
        ]
        bad_predict_records = [
            make_predict_record(filename="dup.jpg", idx=1, formula_list=[]),
            make_predict_record(filename="dup.jpg", idx=2, formula_list=[]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gt_path = temp_path / "gt.json"
            predict_root = temp_path / "predict"
            output_root = temp_path / "metric"

            write_json(gt_path, gt_records)
            write_json(predict_root / "good" / "predict.json", good_predict_records)
            write_json(predict_root / "bad" / "predict.json", bad_predict_records)

            exit_code = main(
                [
                    "--gt",
                    str(gt_path),
                    "--predict-root",
                    str(predict_root),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse((output_root / "bad" / "sample_metrics.json").exists())
            self.assertFalse((output_root / "bad" / "overall_metrics.json").exists())
            self.assertTrue((output_root / "bad" / "eval_errors.json").exists())
            self.assertTrue((output_root / "good" / "overall_metrics.json").exists())

            fatal_report = json.loads(
                (output_root / "bad" / "eval_errors.json").read_text(encoding="utf-8")
            )
            leaderboard = json.loads(
                (output_root / "leaderboard_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(fatal_report["summary"]["fatal_error_count"], 1)
            self.assertEqual(fatal_report["summary"]["sample_error_count"], 0)
            self.assertEqual(len(leaderboard), 1)
            self.assertEqual(leaderboard[0]["model"], "good")


if __name__ == "__main__":
    unittest.main()
