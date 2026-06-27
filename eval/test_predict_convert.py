#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from predict_convert import convert_results, main


def make_gt_record(**overrides):
    record = {
        "filename": "sample.jpg",
        "idx": 1,
        "QuestionType": "ConstructedResponse",
        "transcription": [
            {
                "seq": 1,
                "question_id": "Q001",
                "content": "∴x+1",
            },
            {
                "seq": 2,
                "question_id": "Q001",
                "content": "FindA∩B",
            },
        ],
        "formula_list": [],
        "final_answer": "",
    }
    record.update(overrides)
    return record


def make_raw_record(**overrides):
    record = {
        "ImgReal": "sample.jpg",
        "idx": 1,
        "QuestionType": "ConstructedResponse",
        "transcription": "FindA∩B\n∴ [[MATH:x + 1]]\n2026Grade11MidtermMathExam",
        "final_answer": "",
        "raw_output": "{}",
        "predict_status": "success",
        "error_message": "",
    }
    record.update(overrides)
    return record


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = main(argv)
    return exit_code, stdout.getvalue(), stderr.getvalue()


class PredictConvertTests(unittest.TestCase):
    def test_constructed_response_alignment_formula_extraction_and_noise(self):
        converted_records, error_report = convert_results(
            [make_raw_record()],
            [make_gt_record()],
            0.70,
        )

        self.assertEqual(error_report["summary"]["converted_records"], 1)
        self.assertEqual(error_report["summary"]["structured_converted_records"], 1)
        self.assertEqual(error_report["summary"]["error_records"], 0)

        record = converted_records[0]
        self.assertEqual(record["filename"], "sample.jpg")
        self.assertEqual(record["idx"], 1)
        self.assertEqual(record["QuestionType"], "ConstructedResponse")

        transcription = record["transcription"]
        self.assertEqual(transcription[0]["seq"], 1)
        self.assertEqual(transcription[0]["question_id"], "Q001")
        self.assertEqual(transcription[0]["gt_seq"], 2)
        self.assertEqual(transcription[0]["match_status"], "matched")
        self.assertEqual(transcription[0]["match_score"], 1.0)
        self.assertEqual(transcription[0]["content"], "FindA∩B")

        self.assertEqual(transcription[1]["seq"], 2)
        self.assertEqual(transcription[1]["question_id"], "Q001")
        self.assertEqual(transcription[1]["gt_seq"], 1)
        self.assertEqual(transcription[1]["match_status"], "matched")
        self.assertEqual(transcription[1]["match_score"], 1.0)
        self.assertEqual(transcription[1]["content"], "∴x+1")

        self.assertEqual(transcription[2]["seq"], 3)
        self.assertIsNone(transcription[2]["question_id"])
        self.assertIsNone(transcription[2]["gt_seq"])
        self.assertEqual(transcription[2]["match_status"], "unmatched")
        self.assertAlmostEqual(transcription[2]["match_score"], 1 / 26)
        self.assertEqual(transcription[2]["content"], "2026Grade11MidtermMathExam")

        self.assertEqual(
            record["formula_list"],
            [
                {
                    "formula_seq": 1,
                    "seq": 2,
                    "question_id": "Q001",
                    "gt_seq": 1,
                    "formula": "x+1",
                }
            ],
        )

    def test_ocr_tolerant_matching_keeps_score(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "FindA∩B",
                }
            ]
        )
        raw_record = make_raw_record(transcription="FindAnB")

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(error_report["summary"]["error_records"], 0)
        match_score = converted_records[0]["transcription"][0]["match_score"]
        self.assertAlmostEqual(match_score, 6 / 7)

    def test_unmatched_line_keeps_best_raw_score_below_threshold(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "ABCDE",
                }
            ]
        )
        raw_record = make_raw_record(transcription="ABXDE")

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.90)

        self.assertEqual(error_report["summary"]["error_records"], 0)
        transcription_item = converted_records[0]["transcription"][0]
        self.assertEqual(transcription_item["match_status"], "unmatched")
        self.assertIsNone(transcription_item["question_id"])
        self.assertIsNone(transcription_item["gt_seq"])
        self.assertAlmostEqual(transcription_item["match_score"], 0.8)

    def test_unmatched_line_keeps_best_raw_score_when_losing_one_to_one_competition(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "ABCDE",
                },
                {
                    "seq": 2,
                    "question_id": "Q002",
                    "content": "ZZZZZ",
                },
            ]
        )
        raw_record = make_raw_record(transcription="ABCDE\nABXDE")

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(error_report["summary"]["error_records"], 0)
        transcription = converted_records[0]["transcription"]
        self.assertEqual(transcription[0]["match_status"], "matched")
        self.assertEqual(transcription[0]["gt_seq"], 1)
        self.assertEqual(transcription[0]["match_score"], 1.0)
        self.assertEqual(transcription[1]["match_status"], "unmatched")
        self.assertIsNone(transcription[1]["question_id"])
        self.assertIsNone(transcription[1]["gt_seq"])
        self.assertAlmostEqual(transcription[1]["match_score"], 0.8)

    def test_empty_gt_transcription_keeps_unmatched_score_null(self):
        gt_record = make_gt_record(transcription=[])
        raw_record = make_raw_record(transcription="ABCDE")

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(error_report["summary"]["error_records"], 0)
        transcription_item = converted_records[0]["transcription"][0]
        self.assertEqual(transcription_item["match_status"], "unmatched")
        self.assertIsNone(transcription_item["question_id"])
        self.assertIsNone(transcription_item["gt_seq"])
        self.assertIsNone(transcription_item["match_score"])

    def test_passthrough_question_types_keep_raw_transcription(self):
        gt_record = make_gt_record(
            filename="choice.jpg",
            idx=2,
            QuestionType="MultipleChoice",
            transcription="A",
        )
        raw_record = make_raw_record(
            ImgReal="choice.jpg",
            idx=2,
            QuestionType="MultipleChoice",
            transcription=" D ",
            final_answer=" D ",
        )

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(error_report["summary"]["passthrough_records"], 1)
        self.assertEqual(converted_records[0]["QuestionType"], "MultipleChoice")
        self.assertEqual(converted_records[0]["transcription"], " D ")
        self.assertEqual(converted_records[0]["formula_list"], [])
        self.assertEqual(converted_records[0]["final_answer"], " D ")

    def test_unknown_gt_passthrough_keeps_raw_constructed_response(self):
        gt_record = make_gt_record(
            transcription="[UNK]",
            final_answer="[UNK]",
        )
        raw_record = make_raw_record(
            transcription="Firststep\n[[MATH:x + 1]]\nSecondstep",
            final_answer="x=1",
        )

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(error_report["summary"]["passthrough_records"], 1)
        self.assertEqual(error_report["summary"]["unknown_gt_passthrough_records"], 1)
        self.assertEqual(len(converted_records), 1)
        self.assertEqual(
            converted_records[0]["conversion_status"],
            "unknown_gt_passthrough",
        )
        self.assertEqual(
            converted_records[0]["transcription"],
            "Firststep\n[[MATH:x + 1]]\nSecondstep",
        )
        self.assertEqual(converted_records[0]["formula_list"], [])
        self.assertEqual(converted_records[0]["final_answer"], "x=1")

    def test_non_unk_string_gt_transcription_still_errors(self):
        gt_record = make_gt_record(transcription="not-structured")

        converted_records, error_report = convert_results(
            [make_raw_record(transcription="Firststep\nSecondstep")],
            [gt_record],
            0.70,
        )

        self.assertEqual(converted_records, [])
        self.assertEqual(error_report["summary"]["error_records"], 1)
        self.assertEqual(error_report["errors"][0]["reason"], "invalid_gt_transcription")

    def test_non_success_records_are_skipped_without_error(self):
        converted_records, error_report = convert_results(
            [
                make_raw_record(),
                make_raw_record(
                    ImgReal="skipped.jpg",
                    idx=None,
                    predict_status="parse_failed",
                    transcription="",
                ),
            ],
            [make_gt_record()],
            0.70,
        )

        self.assertEqual(len(converted_records), 1)
        self.assertEqual(error_report["summary"]["success_records"], 1)
        self.assertEqual(error_report["summary"]["skipped_non_success_records"], 1)
        self.assertEqual(error_report["summary"]["error_records"], 0)

    def test_parse_failed_record_can_be_salvaged_from_raw_output(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "Firststep",
                }
            ]
        )
        raw_record = make_raw_record(
            predict_status="parse_failed",
            idx=None,
            QuestionType="",
            transcription="",
            final_answer="",
            raw_output=(
                "```json\n"
                "{\n"
                '  "idx": 1,\n'
                '  "QuestionType": "ConstructedResponse",\n'
                '  "transcription": "Firststep\\n[[MATH:x\\perp y]]",\n'
                '  "final_answer": ""\n'
                "}\n"
                "```"
            ),
        )

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(len(converted_records), 1)
        self.assertEqual(error_report["summary"]["success_records"], 1)
        self.assertEqual(error_report["summary"]["skipped_non_success_records"], 0)
        self.assertEqual(error_report["summary"]["salvaged_non_success_records"], 1)
        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(converted_records[0]["filename"], "sample.jpg")
        self.assertEqual(converted_records[0]["QuestionType"], "ConstructedResponse")
        self.assertEqual(converted_records[0]["transcription"][0]["content"], "Firststep")

    def test_unk_question_type_falls_back_to_gt_question_type(self):
        gt_record = make_gt_record(
            filename="choice.jpg",
            idx=2,
            QuestionType="FillBlank",
            transcription="Answer",
            final_answer="Answer",
        )
        raw_record = make_raw_record(
            ImgReal="choice.jpg",
            idx=2,
            QuestionType="[UNK]",
            transcription="Answer",
            final_answer="Answer",
        )

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(len(converted_records), 1)
        self.assertEqual(error_report["summary"]["recovered_question_type_records"], 1)
        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(converted_records[0]["QuestionType"], "FillBlank")
        self.assertEqual(converted_records[0]["transcription"], "Answer")

    def test_empty_transcription_can_fall_back_to_raw_output_trans_field(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "ProofBegins",
                }
            ]
        )
        raw_record = make_raw_record(
            transcription="",
            final_answer="",
            raw_output=(
                "{\n"
                '  "idx": 1,\n'
                '  "QuestionType": "ConstructedResponse",\n'
                '  "trans": "ProofBegins\\nSecondstep",\n'
                '  "final_answer": ""\n'
                "}"
            ),
        )

        converted_records, error_report = convert_results([raw_record], [gt_record], 0.70)

        self.assertEqual(len(converted_records), 1)
        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(converted_records[0]["QuestionType"], "ConstructedResponse")
        self.assertEqual(converted_records[0]["transcription"][0]["content"], "ProofBegins")

    def test_missing_gt_record_is_reported(self):
        converted_records, error_report = convert_results(
            [make_raw_record(ImgReal="missing.jpg", idx=99)],
            [make_gt_record()],
            0.70,
        )

        self.assertEqual(converted_records, [])
        self.assertEqual(error_report["summary"]["error_records"], 1)
        self.assertEqual(error_report["errors"][0]["reason"], "missing_gt_record")

    def test_gt_record_without_idx_is_allowed(self):
        converted_records, error_report = convert_results(
            [make_raw_record(idx=1)],
            [make_gt_record(idx=None)],
            0.70,
        )

        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(len(converted_records), 1)
        self.assertEqual(converted_records[0]["filename"], "sample.jpg")
        self.assertEqual(converted_records[0]["idx"], 1)

    def test_predict_record_without_idx_is_allowed(self):
        converted_records, error_report = convert_results(
            [make_raw_record(idx=None)],
            [make_gt_record(idx=7)],
            0.70,
        )

        self.assertEqual(error_report["summary"]["error_records"], 0)
        self.assertEqual(len(converted_records), 1)
        self.assertIsNone(converted_records[0]["idx"])

    def test_duplicate_gt_key_is_reported(self):
        gt_record = make_gt_record()
        converted_records, error_report = convert_results(
            [make_raw_record(idx=None)],
            [gt_record, make_gt_record(idx=2)],
            0.70,
        )

        self.assertEqual(converted_records, [])
        self.assertEqual(error_report["summary"]["error_records"], 1)
        self.assertEqual(error_report["errors"][0]["reason"], "duplicate_gt_key")

    def test_empty_constructed_response_after_cleaning_is_error(self):
        converted_records, error_report = convert_results(
            [make_raw_record(transcription=" \n　\n")],
            [make_gt_record()],
            0.70,
        )

        self.assertEqual(converted_records, [])
        self.assertEqual(error_report["summary"]["error_records"], 1)
        self.assertEqual(error_report["errors"][0]["reason"], "no_predict_segments")

    def test_main_writes_predict_and_error_report_files(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "FindA∩B",
                }
            ]
        )
        raw_record = make_raw_record(transcription="FindA∩B")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "results.json"
            gt_path = temp_path / "extracted_gt.json"
            output_path = temp_path / "predict.json"
            error_report_path = temp_path / "predict_conversion_errors.json"

            write_json(input_path, [raw_record])
            write_json(gt_path, [gt_record])

            exit_code, _stdout, _stderr = run_main(
                [
                    "--model",
                    "demo-model",
                    "--input",
                    str(input_path),
                    "--gt",
                    str(gt_path),
                    "--output",
                    str(output_path),
                    "--error-report",
                    str(error_report_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(error_report_path.exists())

            output_data = json.loads(output_path.read_text(encoding="utf-8"))
            error_report = json.loads(error_report_path.read_text(encoding="utf-8"))

            self.assertEqual(len(output_data), 1)
            self.assertEqual(output_data[0]["filename"], "sample.jpg")
            self.assertEqual(
                output_data[0]["transcription"][0]["match_status"],
                "matched",
            )
            self.assertEqual(error_report["summary"]["converted_records"], 1)
            self.assertEqual(error_report["summary"]["error_records"], 0)

    def test_main_prefers_result_less_format_noise_input_when_present(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "FromDenoisedInput",
                }
            ]
        )
        denoised_record = make_raw_record(transcription="FromDenoisedInput")
        raw_record = make_raw_record(transcription="FromRawInput")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output" / "predict"
            model_dir = output_root / "demo-model"
            gt_path = temp_path / "extracted_gt.json"

            write_json(model_dir / "result_less_format_noise.json", [denoised_record])
            write_json(model_dir / "results.json", [raw_record])
            write_json(gt_path, [gt_record])

            exit_code, _stdout, _stderr = run_main(
                [
                    "--model",
                    "demo-model",
                    "--output-root",
                    str(output_root),
                    "--gt",
                    str(gt_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            output_data = json.loads((model_dir / "predict.json").read_text(encoding="utf-8"))
            self.assertEqual(output_data[0]["transcription"][0]["content"], "FromDenoisedInput")

    def test_main_falls_back_to_results_when_denoised_input_missing(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "FromRawInput",
                }
            ]
        )
        raw_record = make_raw_record(transcription="FromRawInput")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output" / "predict"
            model_dir = output_root / "demo-model"
            gt_path = temp_path / "extracted_gt.json"

            write_json(model_dir / "results.json", [raw_record])
            write_json(gt_path, [gt_record])

            exit_code, _stdout, _stderr = run_main(
                [
                    "--model",
                    "demo-model",
                    "--output-root",
                    str(output_root),
                    "--gt",
                    str(gt_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            output_data = json.loads((model_dir / "predict.json").read_text(encoding="utf-8"))
            self.assertEqual(output_data[0]["transcription"][0]["content"], "FromRawInput")

    def test_all_models_processes_only_dirs_with_denoised_input(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "Batch success",
                }
            ]
        )
        denoised_record = make_raw_record(transcription="Batch success")
        raw_only_record = make_raw_record(transcription="Raw result only")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output" / "predict"
            gt_path = temp_path / "extracted_gt.json"
            processed_dir = output_root / "model-a"
            skipped_dir = output_root / "model-b"

            write_json(processed_dir / "result_less_format_noise.json", [denoised_record])
            write_json(skipped_dir / "results.json", [raw_only_record])
            write_json(gt_path, [gt_record])

            exit_code, stdout, _stderr = run_main(
                [
                    "--all-models",
                    "--output-root",
                    str(output_root),
                    "--gt",
                    str(gt_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("Models to process: model-a", stdout)
            self.assertIn("Models skipped without result_less_format_noise.json: model-b", stdout)
            self.assertTrue((processed_dir / "predict.json").exists())
            self.assertTrue((processed_dir / "predict_conversion_errors.json").exists())
            self.assertFalse((skipped_dir / "predict.json").exists())
            self.assertFalse((skipped_dir / "predict_conversion_errors.json").exists())

    def test_all_models_returns_nonzero_when_one_model_fails(self):
        gt_record = make_gt_record(
            transcription=[
                {
                    "seq": 1,
                    "question_id": "Q001",
                    "content": "Batch success",
                }
            ]
        )
        good_record = make_raw_record(transcription="Batch success")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output" / "predict"
            gt_path = temp_path / "extracted_gt.json"
            good_dir = output_root / "good-model"
            bad_dir = output_root / "bad-model"

            write_json(good_dir / "result_less_format_noise.json", [good_record])
            (bad_dir).mkdir(parents=True, exist_ok=True)
            (bad_dir / "result_less_format_noise.json").write_text(
                "{not-valid-json}\n",
                encoding="utf-8",
            )
            write_json(gt_path, [gt_record])

            exit_code, stdout, stderr = run_main(
                [
                    "--all-models",
                    "--output-root",
                    str(output_root),
                    "--gt",
                    str(gt_path),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("Model: good-model", stdout)
            self.assertIn("Models failed: 1", stdout)
            self.assertIn("Model bad-model failed:", stderr)
            self.assertTrue((good_dir / "predict.json").exists())
            self.assertTrue((good_dir / "predict_conversion_errors.json").exists())
            self.assertFalse((bad_dir / "predict.json").exists())
            self.assertFalse((bad_dir / "predict_conversion_errors.json").exists())

    def test_all_models_conflicting_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output" / "predict"
            gt_path = temp_path / "extracted_gt.json"
            write_json(gt_path, [make_gt_record()])

            conflict_cases = [
                ["--all-models", "--model", "demo-model"],
                ["--all-models", "--input", str(temp_path / "input.json")],
                ["--all-models", "--output", str(temp_path / "predict.json")],
                ["--all-models", "--error-report", str(temp_path / "errors.json")],
            ]

            for conflict_args in conflict_cases:
                exit_code, _stdout, stderr = run_main(
                    conflict_args
                    + [
                        "--output-root",
                        str(output_root),
                        "--gt",
                        str(gt_path),
                    ]
                )
                self.assertEqual(exit_code, 1)
                self.assertIn("cannot be used with --all-models", stderr)

    def test_missing_model_and_all_models_is_error(self):
        exit_code, _stdout, stderr = run_main([])
        self.assertEqual(exit_code, 1)
        self.assertIn("Either --model or --all-models must be provided.", stderr)


if __name__ == "__main__":
    unittest.main()
