"""
Stage 3 — Cambridge Visual Extraction Unit & Integration Tests
Run from project root:  .venv/bin/pytest tests/test_visuals.py -v
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pymupdf as fitz
import pytest

from pipeline.visuals.extractor import CambridgeVisualExtractor, Line, DEFAULT_CAPTION_PATTERN
from pipeline.visuals.cli import _patch_question_json


class TestCambridgeVisualExtractorComponents:
    """Unit tests for individual components of CambridgeVisualExtractor."""

    @pytest.fixture
    def extractor(self):
        return CambridgeVisualExtractor()

    def test_caption_pattern_matching(self, extractor):
        """Captions must match isolated lines like 'Fig. 1.1', 'Figure 10.2'."""
        assert extractor.caption_pattern.match("Fig. 1.1")
        assert extractor.caption_pattern.match("Fig. 10.2")
        assert extractor.caption_pattern.match("Figure 3.4")
        assert extractor.caption_pattern.match("Fig.1.1")

        # Must not match inline mentions in prose
        assert not extractor.caption_pattern.match("As shown in Fig. 1.1, calculate the force.")
        assert not extractor.caption_pattern.match("See Fig. 2.1 below.")
        assert not extractor.caption_pattern.match("Table 1.1")

    def test_get_question_anchors(self, extractor):
        """Question anchors must be bold digits at the left margin."""
        lines = [
            Line(text="5054/41/M/J/10", bbox=fitz.Rect(50, 20, 200, 35), bold=False, width=150), # header
            Line(text="1", bbox=fitz.Rect(40, 100, 50, 120), bold=True, width=10),               # Q1 anchor
            Line(text="A student measures the period of a pendulum.", bbox=fitz.Rect(70, 100, 400, 120), bold=False, width=330),
            Line(text="2", bbox=fitz.Rect(40, 400, 50, 420), bold=True, width=10),               # Q2 anchor
        ]
        anchors = extractor.get_question_anchors(lines)
        assert anchors == [(1, 100), (2, 400)]

    def test_get_captions(self, extractor):
        """get_captions extracts caption label, question number, and sub-index."""
        lines = [
            Line(text="Fig. 1.1", bbox=fitz.Rect(200, 250, 250, 265), bold=True, width=50),
            Line(text="Fig. 1.2", bbox=fitz.Rect(200, 500, 250, 515), bold=True, width=50),
        ]
        caps = extractor.get_captions(lines)
        assert len(caps) == 2
        assert caps[0]["q_from_caption"] == 1
        assert caps[0]["sub"] == 1
        assert caps[0]["label"] == "Fig. 1.1"
        assert caps[1]["sub"] == 2

    def test_upper_bound_detection(self, extractor):
        """Upper bound walks upward to the nearest preceding prose sentence."""
        lines = [
            Line(text="The apparatus is shown below.", bbox=fitz.Rect(70, 100, 400, 120), bold=False, width=330),
            Line(text="Fig. 1.1", bbox=fitz.Rect(200, 300, 250, 315), bold=True, width=50),
        ]
        cap = {"bbox": lines[1].bbox, "label": "Fig. 1.1"}
        upper = extractor.find_upper_bound(cap, lines)
        assert upper == lines[0].bbox.y1

    def test_upper_bound_wrapped_sentence_chaining(self, extractor):
        """Upper bound excludes short wrapped tails belonging to a paragraph."""
        lines = [
            # Line 1: Long main sentence
            Line(text="A student investigates the rate of cooling of water as shown", bbox=fitz.Rect(70, 100, 400, 115), bold=False, width=330),
            # Line 2: Short wrapped tail ("in Fig. 1.1.")
            Line(text="in Fig. 1.1.", bbox=fitz.Rect(70, 120, 130, 135), bold=False, width=60),
            # Caption
            Line(text="Fig. 1.1", bbox=fitz.Rect(200, 300, 250, 315), bold=True, width=50),
        ]
        cap = {"bbox": lines[2].bbox, "label": "Fig. 1.1"}
        upper = extractor.find_upper_bound(cap, lines)
        # Should return the bottom of the wrapped tail (line 2), excluding both lines from figure
        assert upper == lines[1].bbox.y1


class TestVisualExtractionIntegration:
    """Integration test on sample Cambridge exam paper."""

    SAMPLE_PDF = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_41.pdf")

    def test_extract_on_real_pdf(self, tmp_path):
        """Extract figures from real exam paper and verify manifest output."""
        if not self.SAMPLE_PDF.exists():
            pytest.skip(f"Sample PDF {self.SAMPLE_PDF} not present in storage.")

        output_dir = tmp_path / "5054_s10_qp_41_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(self.SAMPLE_PDF, output_dir)

        assert len(manifest) >= 5
        manifest_file = output_dir / "manifest.json"
        assert manifest_file.exists()

        # Check entry structure and files on disk
        for entry in manifest:
            assert "page" in entry
            assert "question" in entry
            assert "bbox" in entry
            assert "filename" in entry
            img_file = output_dir / entry["filename"]
            assert img_file.exists()
            assert img_file.stat().st_size > 0


class TestJsonPatching:
    """Tests for Stage 2 JSON patching with Stage 3 visuals."""

    def test_patch_question_json(self, tmp_path):
        """_patch_question_json must match questions by question number."""
        json_file = tmp_path / "paper.json"
        records = [
            {"QuestionNumber": "5054/s10/qp/41/Q1", "text": "Question 1 text", "image": None},
            {"QuestionNumber": "5054/s10/qp/41/Q2", "text": "Question 2 text", "image": None},
            {"QuestionNumber": "5054/s10/qp/41/Q3", "text": "Question 3 text", "image": None},
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        manifest = [
            {"question": 1, "relative_path": "visuals/p2_q1_Fig1_1.png"},
            {"question": 1, "relative_path": "visuals/p2_q1_Fig1_2.png"},
            {"question": 2, "relative_path": "visuals/p6_q2_Fig2_1.png"},
        ]

        _patch_question_json(json_file, manifest)

        with open(json_file, "r", encoding="utf-8") as f:
            updated = json.load(f)

        assert updated[0]["image"] == [
            "visuals/p2_q1_Fig1_1.png",
            "visuals/p2_q1_Fig1_2.png",
        ]
        assert updated[1]["image"] == "visuals/p6_q2_Fig2_1.png"
        assert updated[2]["image"] is None
