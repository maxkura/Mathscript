#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reduce_format_noise import (
    discover_model_directories,
    parse_cleanup_output,
    process_model_directory,
    should_cleanup_record,
)


def make_record(**overrides):
    record = {
        "ImgReal": "sample.jpg",
        "idx": 1,
        "QuestionType": "ConstructedResponse",
        "transcription": "Original transcription",
        "final_answer": "",
        "raw_output": '{"transcription":"Original transcription"}',
        "predict_status": "success",
        "error_message": "",
    }
    record.update(overrides)
    return record


class ReduceFormatNoiseTests(unittest.TestCase):
    def test_should_cleanup_record_requires_success_and_non_empty_transcription(self):
        self.assertTrue(should_cleanup_record(make_record()))
        self.assertFalse(should_cleanup_record(make_record(predict_status="parse_failed")))
        self.assertFalse(should_cleanup_record(make_record(transcription="")))
        self.assertFalse(should_cleanup_record({"predict_status": "success"}))
        self.assertFalse(should_cleanup_record("not-a-record"))

    def test_parse_cleanup_output_accepts_direct_object_fenced_json_and_wrapped_text(self):
        direct_output = '{"transcription":"Cleaned text"}'
        fenced_output = '```json\n{"transcription":"Cleaned text"}\n```'
        wrapped_output = 'Here is the result\n{"transcription":"Cleaned text"}\nPlease review it'

        self.assertEqual(parse_cleanup_output(direct_output), ("Cleaned text", ""))
        self.assertEqual(parse_cleanup_output(fenced_output), ("Cleaned text", ""))
        self.assertEqual(parse_cleanup_output(wrapped_output), ("Cleaned text", ""))

    def test_process_model_directory_only_rewrites_transcription(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "gpt-4o"
            model_dir.mkdir(parents=True)
            input_records = [
                make_record(transcription="Needs cleanup 1", final_answer="A"),
                make_record(
                    ImgReal="sample-2.jpg",
                    idx=2,
                    transcription="",
                    final_answer="B",
                ),
                make_record(
                    ImgReal="sample-3.jpg",
                    idx=3,
                    transcription="Failed record",
                    predict_status="parse_failed",
                    error_message="bad json",
                ),
            ]
            (model_dir / "results.json").write_text(
                json.dumps(input_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            def cleanup_func(record):
                return (f"Cleaned:{record['transcription']}", "")

            summary = process_model_directory(
                model_dir=model_dir,
                output_name="result_less_format_noise.json",
                overwrite=False,
                cleanup_func=cleanup_func,
                show_progress=False,
            )

            output_records = json.loads(
                (model_dir / "result_less_format_noise.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "processed")
            self.assertEqual(summary["eligible_records"], 1)
            self.assertEqual(summary["cleaned_records"], 1)
            self.assertEqual(summary["failed_records"], 0)

            self.assertEqual(output_records[0]["transcription"], "Cleaned:Needs cleanup 1")
            self.assertEqual(output_records[0]["final_answer"], "A")
            self.assertEqual(output_records[0]["QuestionType"], "ConstructedResponse")
            self.assertEqual(output_records[1], input_records[1])
            self.assertEqual(output_records[2], input_records[2])

    def test_process_model_directory_keeps_original_transcription_when_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "qwen3-vl-plus"
            model_dir.mkdir(parents=True)
            input_records = [
                make_record(ImgReal="a.jpg", idx=1, transcription="Keep original"),
                make_record(ImgReal="b.jpg", idx=2, transcription="Can clean"),
            ]
            (model_dir / "results.json").write_text(
                json.dumps(input_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            def cleanup_func(record):
                if record["idx"] == 1:
                    return (None, "parse failed")
                return ("Cleanup success", "")

            summary = process_model_directory(
                model_dir=model_dir,
                output_name="result_less_format_noise.json",
                overwrite=False,
                cleanup_func=cleanup_func,
                show_progress=False,
            )

            output_records = json.loads(
                (model_dir / "result_less_format_noise.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["eligible_records"], 2)
            self.assertEqual(summary["cleaned_records"], 1)
            self.assertEqual(summary["failed_records"], 1)
            self.assertEqual(output_records[0]["transcription"], "Keep original")
            self.assertEqual(output_records[1]["transcription"], "Cleanup success")

    def test_discover_model_directories_scans_all_or_one_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir)
            model_a = input_root / "gpt-4o"
            model_b = input_root / "kimi-k2.5"
            model_a.mkdir()
            model_b.mkdir()
            (model_a / "results.json").write_text("[]", encoding="utf-8")
            (model_b / "results.json").write_text("[]", encoding="utf-8")
            (input_root / "ignore-me").mkdir()

            scanned = discover_model_directories(input_root, None)
            single = discover_model_directories(input_root, "kimi-k2.5")

            self.assertEqual([path.name for path in scanned], ["gpt-4o", "kimi-k2.5"])
            self.assertEqual([path.name for path in single], ["kimi-k2.5"])

    def test_process_model_directory_skips_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "claude-sonnet-4-5-20250929"
            model_dir.mkdir(parents=True)
            (model_dir / "results.json").write_text("[]", encoding="utf-8")
            output_path = model_dir / "result_less_format_noise.json"
            output_path.write_text('[{"kept": true}]', encoding="utf-8")

            summary = process_model_directory(
                model_dir=model_dir,
                output_name="result_less_format_noise.json",
                overwrite=False,
                cleanup_func=lambda record: ("Should not run", ""),
                show_progress=False,
            )

            self.assertEqual(summary["status"], "skipped_existing")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [{"kept": True}])

    def test_process_model_directory_overwrite_reprocesses_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "gemini-2.5-flash"
            model_dir.mkdir(parents=True)
            input_records = [make_record(transcription="Old content")]
            (model_dir / "results.json").write_text(
                json.dumps(input_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_path = model_dir / "result_less_format_noise.json"
            output_path.write_text('[{"transcription":"Should not keep"}]', encoding="utf-8")

            summary = process_model_directory(
                model_dir=model_dir,
                output_name="result_less_format_noise.json",
                overwrite=True,
                cleanup_func=lambda record: ("New content", ""),
                show_progress=False,
            )

            self.assertEqual(summary["status"], "processed")
            output_records = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output_records[0]["transcription"], "New content")
            self.assertEqual(output_records[0]["final_answer"], "")

    def test_process_model_directory_writes_progress_after_each_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "kimi-k2.5"
            model_dir.mkdir(parents=True)
            input_records = [
                make_record(ImgReal="a.jpg", idx=1, transcription="First item"),
                make_record(ImgReal="b.jpg", idx=2, transcription="Second item"),
            ]
            (model_dir / "results.json").write_text(
                json.dumps(input_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_path = model_dir / "result_less_format_noise.json"
            seen_snapshots = []

            def cleanup_func(record):
                if record["idx"] == 2:
                    snapshot = json.loads(output_path.read_text(encoding="utf-8"))
                    seen_snapshots.append(snapshot)
                return (f"Cleaned:{record['transcription']}", "")

            summary = process_model_directory(
                model_dir=model_dir,
                output_name="result_less_format_noise.json",
                overwrite=False,
                cleanup_func=cleanup_func,
                show_progress=False,
            )

            self.assertEqual(summary["cleaned_records"], 2)
            self.assertEqual(len(seen_snapshots), 1)
            self.assertEqual(seen_snapshots[0][0]["transcription"], "Cleaned:First item")
            self.assertEqual(seen_snapshots[0][1]["transcription"], "Second item")


if __name__ == "__main__":
    unittest.main()
