#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment.controlled_reading_order import (
    BASELINE_METHOD_NAME,
    CONSTRUCTED_RESPONSE,
    ExperimentPaths,
    atomic_write_json,
    build_gt_and_manifest,
    convert_step,
    evaluate_step,
    extract_formula_list,
    extract_gt_records,
    load_json_array,
    parse_ordered_unit_ids,
    resolve_experiment_paths,
    run_reorder_step,
    split_gt_transcription,
)


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
LINEAR_SOURCE = BENCHMARK_DIR / "data" / "test" / "linear" / "results_new_yes.json"
UNLINEAR_SOURCE = BENCHMARK_DIR / "data" / "test" / "unlinear" / "results_new_no.json"


def _record_by_filename(records, filename: str):
    for record in records:
        if isinstance(record, dict) and record.get("filename") == filename:
            return record
    raise AssertionError(f"Record not found for {filename}")


class GtExtractionUnitTests(unittest.TestCase):
    def test_no_q_tags_defaults_to_q001(self) -> None:
        items = split_gt_transcription("ConstructedResponse (10 points)\nSolution: First step\nSecond step")
        self.assertEqual(
            items,
            [
                {"seq": 1, "question_id": "Q001", "content": "Solution: First step"},
                {"seq": 2, "question_id": "Q001", "content": "Second step"},
            ],
        )

    def test_pre_q_lines_then_q002(self) -> None:
        items = split_gt_transcription("Solution: Prepare first\nThen substitute\n【q002】Reach the conclusion")
        self.assertEqual(
            items,
            [
                {"seq": 1, "question_id": "Q001", "content": "Solution: Prepare first"},
                {"seq": 2, "question_id": "Q001", "content": "Then substitute"},
                {"seq": 3, "question_id": "Q002", "content": "Reach the conclusion"},
            ],
        )

    def test_q_number_gaps_are_preserved(self) -> None:
        items = split_gt_transcription("【q001】Alpha\n【q002】Beta\n【q004】Gamma")
        self.assertEqual([item["question_id"] for item in items], ["Q001", "Q002", "Q004"])

    def test_semantic_prefix_is_preserved_before_q_marker(self) -> None:
        items = split_gt_transcription("Solution:【q001】Main text")
        self.assertEqual(
            items,
            [{"seq": 1, "question_id": "Q001", "content": "Solution:Main text"}],
        )

    def test_empty_q_marker_line_updates_question_without_emitting_empty_line(self) -> None:
        items = split_gt_transcription("Solution: Prepare first\n(2)【q002】\nContinue writing")
        self.assertEqual(
            items,
            [
                {"seq": 1, "question_id": "Q001", "content": "Solution: Prepare first"},
                {"seq": 2, "question_id": "Q002", "content": "Continue writing"},
            ],
        )

    def test_bad_math_wrapper_is_extracted_and_unwrapped(self) -> None:
        cleaned_items, formula_list = extract_formula_list(
            [
                {"seq": 1, "question_id": "Q001", "content": "[MATH:x^2+1]]"},
                {"seq": 2, "question_id": "Q001", "content": "[[MATH:y=2]]"},
            ]
        )
        self.assertEqual(
            [item["content"] for item in cleaned_items],
            ["x^2+1", "y=2"],
        )
        self.assertEqual(
            [item["formula"] for item in formula_list],
            ["x^2+1", "y=2"],
        )


class RealSampleRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source_records = load_json_array(LINEAR_SOURCE) + load_json_array(UNLINEAR_SOURCE)
        cls.gt_records, cls.error_report = extract_gt_records(source_records)

    def test_no_real_sample_extraction_errors(self) -> None:
        self.assertEqual(self.error_report["summary"]["input_records"], 41)
        self.assertEqual(self.error_report["summary"]["extracted_records"], 41)
        self.assertEqual(self.error_report["summary"]["error_records"], 0)

    def test_row_c_12_structure(self) -> None:
        record = _record_by_filename(self.gt_records, "row_C_12.jpg")
        self.assertEqual(record["transcription"][0]["question_id"], "Q001")
        self.assertTrue(record["transcription"][0]["content"].startswith("\u89e3"))
        self.assertIn("Q002", [item["question_id"] for item in record["transcription"]])

    def test_row_c_151_keeps_q004_gap(self) -> None:
        record = _record_by_filename(self.gt_records, "row_C_151.jpg")
        self.assertIn("Q004", [item["question_id"] for item in record["transcription"]])

    def test_row_c_224_retains_solution_prefix(self) -> None:
        record = _record_by_filename(self.gt_records, "row_C_224.jpg")
        self.assertTrue(record["transcription"][0]["content"].startswith("\u89e3"))
        self.assertIn("Q002", [item["question_id"] for item in record["transcription"]])

    def test_row_c_309_stays_single_question(self) -> None:
        record = _record_by_filename(self.gt_records, "row_C_309.jpg")
        self.assertEqual({item["question_id"] for item in record["transcription"]}, {"Q001"})

    def test_row_c_352_splits_into_multiple_questions(self) -> None:
        record = _record_by_filename(self.gt_records, "row_C_352.jpg")
        question_ids = [item["question_id"] for item in record["transcription"]]
        self.assertIn("Q001", question_ids)
        self.assertIn("Q002", question_ids)


class ReorderValidationTests(unittest.TestCase):
    def test_parse_valid_permutation(self) -> None:
        ordered, error = parse_ordered_unit_ids('{"ordered_unit_ids":[2,1,3]}', [1, 2, 3])
        self.assertEqual(ordered, [2, 1, 3])
        self.assertEqual(error, "")

    def test_parse_duplicate_ids_fails(self) -> None:
        ordered, error = parse_ordered_unit_ids('{"ordered_unit_ids":[1,1,3]}', [1, 2, 3])
        self.assertIsNone(ordered)
        self.assertIn("duplicates", error)

    def test_parse_missing_ids_fails(self) -> None:
        ordered, error = parse_ordered_unit_ids('{"ordered_unit_ids":[1,2]}', [1, 2, 3])
        self.assertIsNone(ordered)
        self.assertIn("length", error)

    def test_parse_extra_ids_fails(self) -> None:
        ordered, error = parse_ordered_unit_ids('{"ordered_unit_ids":[1,2,3,4]}', [1, 2, 3])
        self.assertIsNone(ordered)
        self.assertIn("length", error)

    def test_parse_non_json_fails(self) -> None:
        ordered, error = parse_ordered_unit_ids("not json", [1, 2, 3])
        self.assertIsNone(ordered)
        self.assertTrue(error)


class PipelineSmokeTests(unittest.TestCase):
    def _write_minimal_sources(self, root: Path) -> tuple[Path, Path]:
        linear_source_records = [
            _record_by_filename(load_json_array(LINEAR_SOURCE), "row_C_12.jpg"),
        ]
        unlinear_source_records = [
            _record_by_filename(load_json_array(UNLINEAR_SOURCE), "row_C_309.jpg"),
        ]
        linear_path = root / "linear_subset.json"
        unlinear_path = root / "unlinear_subset.json"
        atomic_write_json(linear_path, linear_source_records)
        atomic_write_json(unlinear_path, unlinear_source_records)
        return linear_path, unlinear_path

    def _build_success_ocr_records(self, paths: ExperimentPaths) -> None:
        manifest = load_json_array(paths.manifest_path)
        gt_records = load_json_array(paths.gt_path)
        gt_by_filename = {record["filename"]: record for record in gt_records}

        ocr_records = []
        for sample in manifest:
            gt_record = gt_by_filename[sample["filename"]]
            lines = [item["content"] for item in gt_record["transcription"]]
            ocr_records.append(
                {
                    "subset": sample["subset"],
                    "filename": sample["filename"],
                    "idx": sample["idx"],
                    "QuestionType": CONSTRUCTED_RESPONSE,
                    "image_path": sample["image_path"],
                    "ocr_text": "\n".join(lines),
                    "ocr_units": [
                        {"unit_id": unit_index, "text": line}
                        for unit_index, line in enumerate(lines, start=1)
                    ],
                    "ocr_status": "success",
                    "error_message": "",
                }
            )
        atomic_write_json(paths.ocr_results_path, ocr_records)

    def test_baseline_convert_and_eval_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            linear_path, unlinear_path = self._write_minimal_sources(temp_root)
            paths = resolve_experiment_paths(temp_root / "run", "ignored")

            build_gt_and_manifest(
                linear_source=linear_path,
                unlinear_source=unlinear_path,
                paths=paths,
            )
            self._build_success_ocr_records(paths)

            run_reorder_step(
                paths=paths,
                model_names=[],
                max_retries=1,
                max_tokens=64,
                request_timeout=30.0,
                overwrite=True,
            )
            convert_step(paths=paths, methods=[BASELINE_METHOD_NAME])
            evaluate_step(paths=paths, methods=[BASELINE_METHOD_NAME])

            summary_rows = load_json_array(paths.summary_json_path)
            self.assertEqual(len(summary_rows), 3)
            for row in summary_rows:
                self.assertEqual(row["method"], BASELINE_METHOD_NAME)
                self.assertEqual(row["converted_count"], row["sample_count"])
                self.assertEqual(row["avg_bcs"], 1.0)


if __name__ == "__main__":
    unittest.main()
