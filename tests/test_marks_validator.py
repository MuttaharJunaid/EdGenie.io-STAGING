"""
tests/test_marks_validator.py

Unit tests for MarkHierarchyValidator:
- Trailing [N] extraction from leaf text
- [Total: N] removal from question stems
- Bottom-up mark hierarchy reconciliation (sub-parts -> part -> question)
- Discrepancy detection and correction
"""

import pytest
from pipeline.extractor.marks_validator import MarkHierarchyValidator


def test_clean_text_and_extract_leaf_mark():
    # Trailing bracket extraction
    text = "Calculate the acceleration of the car. [2]"
    clean_text, mark = MarkHierarchyValidator.clean_text_and_extract_leaf_mark(text, 1)
    assert clean_text == "Calculate the acceleration of the car."
    assert mark == 2

    # Bracket with whitespace
    text2 = "State Newton's first law of motion.   [ 1 ] "
    clean_text2, mark2 = MarkHierarchyValidator.clean_text_and_extract_leaf_mark(text2, 1)
    assert clean_text2 == "State Newton's first law of motion."
    assert mark2 == 1

    # [Total: 10] removal
    text3 = "A student measures the resistance. [Total: 8]"
    clean_text3, mark3 = MarkHierarchyValidator.clean_text_and_extract_leaf_mark(text3, 5)
    assert clean_text3 == "A student measures the resistance."


def test_reconcile_question_marks_with_subparts():
    # LLM hallucinated wrong marks: part has mark=1, but sub-parts have marks 1 and 2
    raw_question = {
        "QuestionNumber": "Jun/11/2023/Q1",
        "Question_text": "An experiment is carried out to investigate motion. [Total: 6]",
        "Marks": 1,  # WRONG: LLM assigned 1
        "Parts": [
            {
                "part_label": "a",
                "part_text": "Define velocity. [1]",
                "marks": 1,  # Correct leaf mark
                "sub_parts": [],
            },
            {
                "part_label": "b",
                "part_text": "In the experiment:",
                "marks": 1,  # WRONG: LLM defaulted to 1 instead of 1 + 2 = 3
                "sub_parts": [
                    {
                        "sub_part_label": "(i)",
                        "sub_part_text": "Calculate the initial kinetic energy. [1]",
                        "marks": 1,
                    },
                    {
                        "sub_part_label": "(ii)",
                        "sub_part_text": "Determine the work done against friction. [2]",
                        "marks": 2,
                    },
                ],
            },
            {
                "part_label": "c",
                "part_text": "State one source of experimental error. [2]",
                "marks": 1,  # WRONG: Text says [2] but LLM output 1
                "sub_parts": [],
            },
        ],
    }

    reconciled, audit_logs = MarkHierarchyValidator.reconcile_question_marks(raw_question)

    # Check that trailing brackets were stripped
    assert reconciled["Parts"][0]["part_text"] == "Define velocity."
    assert reconciled["Parts"][0]["marks"] == 1

    assert reconciled["Parts"][1]["sub_parts"][0]["sub_part_text"] == "Calculate the initial kinetic energy."
    assert reconciled["Parts"][1]["sub_parts"][0]["marks"] == 1

    assert reconciled["Parts"][1]["sub_parts"][1]["sub_part_text"] == "Determine the work done against friction."
    assert reconciled["Parts"][1]["sub_parts"][1]["marks"] == 2

    # Part (b) reconciled sum: 1 + 2 = 3
    assert reconciled["Parts"][1]["marks"] == 3

    # Part (c) leaf mark parsed from [2]:
    assert reconciled["Parts"][2]["part_text"] == "State one source of experimental error."
    assert reconciled["Parts"][2]["marks"] == 2

    # Question total marks reconciled sum: 1 + 3 + 2 = 6
    assert reconciled["Marks"] == 6
    assert reconciled["Question_text"] == "An experiment is carried out to investigate motion."
    assert len(audit_logs) > 0


def test_reconcile_and_validate_records_batch():
    records = [
        {
            "QuestionNumber": "Jun/11/2023/Q1",
            "Question_text": "Intro text",
            "Marks": 1,
            "Parts": [
                {"part_label": "a", "part_text": "part a", "marks": 2, "sub_parts": []},
                {"part_label": "b", "part_text": "part b", "marks": 3, "sub_parts": []},
            ],
        },
        {
            "QuestionNumber": "Jun/11/2023/Q2",
            "Question_text": "Single part question. [4]",
            "Marks": 1,
            "Parts": [],
        },
    ]

    reconciled, stats = MarkHierarchyValidator.reconcile_and_validate_records(records)
    assert stats["questions_count"] == 2
    # Q1: 2 + 3 = 5
    assert reconciled[0]["Marks"] == 5
    # Q2: extracted [4] = 4
    assert reconciled[1]["Marks"] == 4
    assert stats["total_paper_marks"] == 9
    assert stats["discrepancies_fixed"] >= 1
