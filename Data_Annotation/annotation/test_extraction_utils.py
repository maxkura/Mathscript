#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from extraction_utils import extract_record, extract_results
from remove_transcription_spaces import clean_transcription_spaces


def make_record(**overrides):
    record = {
        "preannotation_status": "success",
        "error_message": "",
        "filename": "sample.jpg",
        "idx": 15,
        "QuestionType": "ConstructedResponse",
        "sub_question_count": 1,
        "transcription": "15.\n(1)【q001】 A [[MATH:x + 1]]",
        "final_answer": "answer",
    }
    record.update(overrides)
    return record


def make_cleaned_record(record):
    cleaned_record = copy.deepcopy(record)
    clean_transcription_spaces(cleaned_record)
    return cleaned_record


class ExtractRecordTests(unittest.TestCase):
    def test_constructed_response_with_q_markers_is_structured(self):
        original_record = make_record(
            transcription="15.\n(1)【q001】 A [[MATH:x + 1]]\n(2)【q002】 B",
            final_answer=" final ",
        )

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            make_cleaned_record(original_record),
        )

        self.assertIsNone(reason)
        self.assertIsNone(details)
        self.assertEqual(extraction_kind, "structured")
        self.assertEqual(extracted_record["QuestionType"], "ConstructedResponse")
        self.assertIsInstance(extracted_record["transcription"], list)
        self.assertEqual(extracted_record["transcription"][0]["question_id"], "Q001")
        self.assertEqual(extracted_record["transcription"][0]["content"], "Ax+1")
        self.assertEqual(extracted_record["transcription"][1]["question_id"], "Q002")
        self.assertEqual(extracted_record["transcription"][1]["content"], "B")
        self.assertEqual(
            extracted_record["formula_list"],
            [
                {
                    "formula_seq": 1,
                    "seq": 1,
                    "question_id": "Q001",
                    "formula": "x+1",
                }
            ],
        )
        self.assertEqual(extracted_record["final_answer"], " final ")

    def test_non_constructed_types_are_passthrough(self):
        for question_type in ("MultipleChoice", "FillBlank"):
            with self.subTest(question_type=question_type):
                original_record = make_record(
                    QuestionType=question_type,
                    transcription="  raw text  ",
                    final_answer="  raw answer  ",
                )

                extracted_record, reason, details, extraction_kind = extract_record(
                    original_record,
                    make_cleaned_record(original_record),
                )

                self.assertIsNone(reason)
                self.assertIsNone(details)
                self.assertEqual(extraction_kind, "passthrough_non_constructed")
                self.assertEqual(extracted_record["QuestionType"], question_type)
                self.assertEqual(extracted_record["transcription"], "  raw text  ")
                self.assertEqual(extracted_record["final_answer"], "  raw answer  ")
                self.assertEqual(extracted_record["formula_list"], [])

    def test_constructed_response_with_unk_in_final_answer_is_passthrough(self):
        original_record = make_record(
            transcription="Unable to extract the full problem",
            final_answer="Result is [UNK]",
        )

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            make_cleaned_record(original_record),
        )

        self.assertIsNone(reason)
        self.assertIsNone(details)
        self.assertEqual(extraction_kind, "passthrough_unk")
        self.assertEqual(extracted_record["transcription"], "Unable to extract the full problem")
        self.assertEqual(extracted_record["final_answer"], "Result is [UNK]")
        self.assertEqual(extracted_record["formula_list"], [])

    def test_malformed_unk_variant_is_passthrough(self):
        original_record = make_record(
            transcription=" [[UNK] ",
            final_answer="",
        )

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            make_cleaned_record(original_record),
        )

        self.assertIsNone(reason)
        self.assertIsNone(details)
        self.assertEqual(extraction_kind, "passthrough_unk")
        self.assertEqual(extracted_record["transcription"], " [[UNK] ")
        self.assertEqual(extracted_record["formula_list"], [])

    def test_constructed_response_without_q_marker_and_without_unk_is_error(self):
        original_record = make_record(
            transcription="Only full-problem text, no sub-question markers",
            final_answer="Regular answer",
        )

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            make_cleaned_record(original_record),
        )

        self.assertIsNone(extracted_record)
        self.assertEqual(reason, "no_extractable_segments")
        self.assertIn("No transcription segments", details)
        self.assertIsNone(extraction_kind)

    def test_invalid_question_type_is_error(self):
        original_record = make_record(
            QuestionType="Essay",
            transcription="[UNK]",
            final_answer="[UNK]",
        )

        extracted_record, reason, details, extraction_kind = extract_record(
            original_record,
            make_cleaned_record(original_record),
        )

        self.assertIsNone(extracted_record)
        self.assertEqual(reason, "invalid_question_type")
        self.assertIn("Essay", details)
        self.assertIsNone(extraction_kind)


class ExtractResultsTests(unittest.TestCase):
    def test_extract_results_summary_tracks_structured_passthrough_and_errors(self):
        original_records = [
            make_record(filename="structured.jpg"),
            make_record(
                filename="multiple_choice.jpg",
                QuestionType="MultipleChoice",
                transcription=" D ",
                final_answer=" D ",
            ),
            make_record(
                filename="fill_blank.jpg",
                QuestionType="FillBlank",
                transcription=" x = 1 ",
                final_answer=" x = 1 ",
            ),
            make_record(
                filename="unk.jpg",
                transcription="[[UNK]",
                final_answer="",
            ),
            make_record(
                filename="invalid_type.jpg",
                QuestionType="",
                transcription="[UNK]",
                final_answer="[UNK]",
            ),
            make_record(
                filename="failed.jpg",
                preannotation_status="failed",
            ),
        ]
        cleaned_records = [make_cleaned_record(record) for record in original_records]

        extracted_records, error_report = extract_results(original_records, cleaned_records)
        summary = error_report["summary"]

        self.assertEqual(summary["input_records"], 6)
        self.assertEqual(summary["processed_success_records"], 5)
        self.assertEqual(summary["extracted_records"], 4)
        self.assertEqual(summary["structured_extracted_records"], 1)
        self.assertEqual(summary["passthrough_records"], 3)
        self.assertEqual(summary["passthrough_non_constructed_records"], 2)
        self.assertEqual(summary["passthrough_unk_records"], 1)
        self.assertEqual(summary["error_records"], 2)
        self.assertEqual(
            [record["filename"] for record in extracted_records],
            [
                "structured.jpg",
                "multiple_choice.jpg",
                "fill_blank.jpg",
                "unk.jpg",
            ],
        )
        self.assertEqual(
            [error["reason"] for error in error_report["errors"]],
            [
                "invalid_question_type",
                "unsupported_preannotation_status",
            ],
        )


if __name__ == "__main__":
    unittest.main()
