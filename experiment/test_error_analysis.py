#!/usr/bin/env python3

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "error_analysis.py"
SPEC = importlib.util.spec_from_file_location("error_analysis", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
error_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(error_analysis)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_gt_record(filename: str, idx: int, formula_text: str = "x=1") -> dict:
    return {
        "filename": filename,
        "idx": idx,
        "QuestionType": "ConstructedResponse",
        "transcription": [
            {"seq": 1, "question_id": "Q001", "content": "First step"},
            {"seq": 2, "question_id": "Q001", "content": formula_text},
        ],
        "formula_list": [
            {
                "formula_seq": 1,
                "seq": 2,
                "question_id": "Q001",
                "formula": formula_text,
            }
        ],
        "final_answer": "",
    }


def make_predict_record(filename: str, idx: int, formula_text: str = "x=1") -> dict:
    return {
        "filename": filename,
        "idx": idx,
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
                "content": formula_text,
            },
        ],
        "formula_list": [
            {
                "formula_seq": 1,
                "seq": 2,
                "question_id": "Q001",
                "gt_seq": 2,
                "formula": formula_text,
            }
        ],
        "final_answer": "",
    }


def make_results_record(filename: str) -> dict:
    return {
        "ImgReal": filename,
        "QuestionType": "ConstructedResponse",
        "idx": 19,
        "predict_status": "success",
        "raw_output": "{}",
        "transcription": "First step\nx=1",
        "final_answer": "",
        "error_message": "",
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_partition_contexts_excludes_unknown_gt_passthrough(self):
        context_ok = {
            "image_id": "row_C_keep.jpg",
            "gt_record": make_gt_record("row_C_keep.jpg", 19),
            "predict_records": {
                model: make_predict_record("row_C_keep.jpg", 19)
                for model in error_analysis.TARGET_MODELS
            },
            "results_records": {model: make_results_record("row_C_keep.jpg") for model in error_analysis.TARGET_MODELS},
            "conversion_error_records": {model: None for model in error_analysis.TARGET_MODELS},
            "image_path": Path("/tmp/row_C_keep.jpg"),
        }
        context_passthrough = {
            "image_id": "row_C_skip.jpg",
            "gt_record": {
                "filename": "row_C_skip.jpg",
                "idx": 19,
                "QuestionType": "ConstructedResponse",
                "transcription": [{"seq": 1, "question_id": "Q001", "content": "Text"}],
                "formula_list": [],
                "final_answer": "",
            },
            "predict_records": {
                model: {
                    "filename": "row_C_skip.jpg",
                    "idx": 19,
                    "QuestionType": "ConstructedResponse",
                    "transcription": "Plain-text passthrough",
                    "formula_list": [],
                    "conversion_status": "unknown_gt_passthrough",
                    "final_answer": "",
                }
                for model in error_analysis.TARGET_MODELS
            },
            "results_records": {model: make_results_record("row_C_skip.jpg") for model in error_analysis.TARGET_MODELS},
            "conversion_error_records": {model: None for model in error_analysis.TARGET_MODELS},
            "image_path": Path("/tmp/row_C_skip.jpg"),
        }

        structured, excluded = error_analysis.partition_contexts(
            [context_ok, context_passthrough],
            error_analysis.TARGET_MODELS,
        )

        self.assertEqual([item["image_id"] for item in structured], ["row_C_keep.jpg"])
        self.assertEqual([item["image_id"] for item in excluded], ["row_C_skip.jpg"])

    def test_build_formula_slot_id_uses_line_seq_and_pred_extra_prefix(self):
        anchored = {
            "question_id": "Q001",
            "gt_seq": 6,
            "gt_formula_seq": 1,
            "predict_formula_seq": 1,
        }
        pred_extra = {
            "question_id": None,
            "gt_seq": None,
            "gt_formula_seq": None,
            "predict_formula_seq": 3,
        }

        self.assertEqual(
            error_analysis.build_formula_slot_id("row_C_1.jpg", anchored),
            "row_C_1.jpg::Q001::6::1",
        )
        self.assertEqual(
            error_analysis.build_formula_slot_id("row_C_1.jpg", pred_extra),
            "row_C_1.jpg::pred_extra::NULL::NULL::3",
        )

    def test_parse_autolabel_response_validates_known_slot_ids(self):
        request = {
            "candidate_formula_slots": [
                {"formula_slot_id": "row_C_1.jpg::Q001::6::1"},
            ]
        }
        raw = json.dumps(
            {
                "coarse_label_primary": "Formula miss",
                "coarse_label_secondary": ["Reading-order/segmentation error"],
                "fine_candidates": [
                    {
                        "formula_slot_id": "row_C_1.jpg::Q001::6::1",
                        "fine_label_primary": "Missed formula recognition",
                        "fine_label_secondary": [],
                    }
                ],
                "auto_confidence": 0.8,
                "notes": "Test",
            },
            ensure_ascii=False,
        )

        parsed = error_analysis.parse_autolabel_response(raw, request)

        self.assertEqual(parsed["coarse_label_primary"], "Formula miss")
        self.assertEqual(parsed["fine_candidates"][0]["formula_slot_id"], "row_C_1.jpg::Q001::6::1")

    def test_merge_autolabel_result_updates_coarse_and_fine_rows(self):
        rows = [
            error_analysis.empty_long_row("row_C_1.jpg", "gpt-4o", "formula_high_risk"),
            error_analysis.empty_long_row(
                "row_C_1.jpg",
                "gpt-4o",
                "formula_high_risk",
                formula_slot_id="row_C_1.jpg::Q001::6::1",
            ),
        ]
        row_index = error_analysis.build_long_row_index(rows)
        request = {
            "image_id": "row_C_1.jpg",
            "model_name": "gpt-4o",
            "candidate_formula_slots": [
                {"formula_slot_id": "row_C_1.jpg::Q001::6::1"},
            ],
        }
        parsed = {
            "coarse_label_primary": "Formula miss",
            "coarse_label_secondary": ["Reading-order/segmentation error"],
            "fine_candidates": [
                {
                    "formula_slot_id": "row_C_1.jpg::Q001::6::1",
                    "fine_label_primary": "Missed formula recognition",
                    "fine_label_secondary": ["Scope error"],
                }
            ],
            "auto_confidence": 0.82,
            "notes": "Test note",
        }

        error_analysis.merge_autolabel_result_into_rows(rows, row_index, request, parsed)

        self.assertEqual(rows[0]["coarse_label_primary"], "Formula miss")
        self.assertEqual(rows[0]["coarse_label_secondary"], "Reading-order/segmentation error")
        self.assertEqual(rows[1]["fine_label_primary"], "Missed formula recognition")
        self.assertEqual(rows[1]["fine_label_secondary"], "Scope error")

    def test_prepare_smoke_outputs_expected_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gt_path = root / "data" / "GT" / "extracted_gt.json"
            image_dir = root / "Data_Annotation" / "annotation" / "images"
            predict_root = root / "output" / "predict"
            output_dir = root / "output" / "experiment" / "error_analysis" / "demo"
            image_dir.mkdir(parents=True)

            gt_records = [
                make_gt_record("row_C_keep.jpg", 19),
                {
                    "filename": "row_C_skip.jpg",
                    "idx": 19,
                    "QuestionType": "ConstructedResponse",
                    "transcription": [{"seq": 1, "question_id": "Q001", "content": "Text"}],
                    "formula_list": [],
                    "final_answer": "",
                },
            ]
            write_json(gt_path, gt_records)
            (image_dir / "row_C_keep.jpg").write_bytes(b"fake")
            (image_dir / "row_C_skip.jpg").write_bytes(b"fake")

            for model in error_analysis.TARGET_MODELS:
                model_dir = predict_root / model
                model_dir.mkdir(parents=True)
                write_json(
                    model_dir / "predict.json",
                    [
                        make_predict_record("row_C_keep.jpg", 19),
                        {
                            "filename": "row_C_skip.jpg",
                            "idx": 19,
                            "QuestionType": "ConstructedResponse",
                            "transcription": "passthrough",
                            "formula_list": [],
                            "conversion_status": "unknown_gt_passthrough",
                            "final_answer": "",
                        },
                    ],
                )
                write_json(
                    model_dir / "results.json",
                    [make_results_record("row_C_keep.jpg"), make_results_record("row_C_skip.jpg")],
                )
                write_json(
                    model_dir / "predict_conversion_errors.json",
                    {"summary": {"error_records": 0}, "errors": []},
                )

            args = type(
                "Args",
                (),
                {
                    "gt": gt_path,
                    "predict_root": predict_root,
                    "image_dir": image_dir,
                    "sample_size": 4,
                    "bucket_size": 1,
                    "calibration_size": 4,
                    "seed": 1,
                    "models": list(error_analysis.TARGET_MODELS),
                },
            )()

            original_quotas = dict(error_analysis.BUCKET_QUOTAS)
            try:
                error_analysis.BUCKET_QUOTAS.update(
                    {
                        "formula_high_risk": 1,
                        "text_high_risk": 1,
                        "mixed_medium_risk": 1,
                        "high_confidence_control": 1,
                    }
                )
                with self.assertRaises(ValueError):
                    error_analysis.run_prepare(args, output_dir)
            finally:
                error_analysis.BUCKET_QUOTAS.clear()
                error_analysis.BUCKET_QUOTAS.update(original_quotas)


if __name__ == "__main__":
    unittest.main()
