"""
tests/test_agentic_extractor_fallback.py

Unit tests for:
- AgenticExtractor JSON repair heuristics
- Gemini full-PDF fallback to page-by-page extraction
- ClassifierAgent priority inversion (OpenRouter 1st, Gemini fallback)
"""

from unittest.mock import patch

from pipeline.extractor.agentic_extractor import AgenticExtractor
from pipeline.extractor.classifier_agent import ClassifierAgent
from pipeline.extractor.paper_type_detector import PaperCategory


def test_attempt_json_repair_unclosed_string():
    # Truncated string
    broken_json = '{"questions": [{"QuestionNumber": "Q1", "Question_text": "Calculate the force'
    repaired = AgenticExtractor._attempt_json_repair(broken_json)
    assert repaired is not None
    import json
    data = json.loads(repaired)
    assert "questions" in data
    assert data["questions"][0]["QuestionNumber"] == "Q1"


def test_attempt_json_repair_unclosed_brackets():
    broken_json = '{"questions": [{"QuestionNumber": "Q1", "Marks": 3}'
    repaired = AgenticExtractor._attempt_json_repair(broken_json)
    assert repaired is not None
    import json
    data = json.loads(repaired)
    assert data["questions"][0]["Marks"] == 3


def test_agentic_extractor_full_pdf_fallback_to_pages(tmp_path):
    # Mock full-pdf failure and verify page-by-page fallback is triggered
    extractor = AgenticExtractor()
    extractor._gemini_quota_exhausted = False
    extractor._gemini_cooldown_until = None

    # Create dummy pdf
    dummy_pdf = tmp_path / "dummy.pdf"
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Question 1\nCalculate speed. [3]")
    doc.save(str(dummy_pdf))
    doc.close()

    meta = {"session": "jun", "year": 2023, "variant": "11"}

    with patch.object(extractor, "_extract_with_gemini_full_pdf", side_effect=ValueError("MAX_TOKENS truncated")):
        with patch.object(
            extractor,
            "_extract_page_with_gemini",
            return_value=[
                {
                    "QuestionNumber": "Jun/11/2023/Q1",
                    "Question_text": "Calculate speed.",
                    "Marks": 3,
                    "Parts": [],
                }
            ],
        ) as mock_page_extract:
            results = extractor.extract(dummy_pdf, PaperCategory.ESSAY_QP, meta)
            assert mock_page_extract.called
            assert len(results) == 1
            assert results[0]["QuestionNumber"] == "Jun/11/2023/Q1"
            assert results[0]["Marks"] == 3


def test_classifier_agent_openrouter_priority_with_gemini_fallback():
    # Verify OpenRouter is called 1st, and if it fails, Gemini is called 2nd
    classifier = ClassifierAgent()
    classifier.openrouter_api_key = "fake_or_key"
    classifier.google_api_key = "fake_google_key"

    records = [{"QuestionNumber": "Q1", "Question_text": "test"}]
    syllabus_text = "Unit 1: Test"

    with patch.object(
        classifier,
        "_classify_with_openrouter",
        side_effect=RuntimeError("OpenRouter 402 Insufficient credits"),
    ) as mock_or:
        with patch.object(
            classifier,
            "_classify_with_gemini",
            return_value={
                "Q1": {
                    "Unit": "1",
                    "Topic": "1.1",
                    "Topic_name": "Test",
                    "Sub_topic": ["Test sub"],
                    "Difficulty_score": 5,
                    "Difficulty_label": "medium",
                    "Command_words": ["calculate"],
                }
            },
        ) as mock_gemini:
            res = classifier.classify(records, syllabus_text, PaperCategory.ESSAY_QP, {})
            # Both were called in sequence
            assert mock_or.called
            assert mock_gemini.called
            assert "Q1" in res
            assert res["Q1"]["Topic"] == "1.1"
