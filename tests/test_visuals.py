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
        """Captions must match isolated lines like 'Fig. 1.1', 'Figure 10.2', 'Table 7.1', 'Fig. 9.1 (not to scale)'."""
        assert extractor.caption_pattern.match("Fig. 1.1")
        assert extractor.caption_pattern.match("Fig. 10.2")
        assert extractor.caption_pattern.match("Figure 3.4")
        assert extractor.caption_pattern.match("Fig.1.1")
        assert extractor.caption_pattern.match("Table 7.1")
        assert extractor.caption_pattern.match("Table 9.1")
        assert extractor.caption_pattern.match("Fig. 9.1 (not to scale)")
        assert extractor.caption_pattern.match("Fig. 1.1 (not to scale)")

        # Must not match inline mentions in prose
        assert not extractor.caption_pattern.match("As shown in Fig. 1.1, calculate the force.")
        assert not extractor.caption_pattern.match("See Fig. 2.1 below.")
        assert not extractor.caption_pattern.match("The values shown in Table 7.1.")

    def test_get_caption_bounds_fig_and_table(self, extractor):
        """Fig captions scan upward for upper bound; Table captions scan downward for lower bound."""
        lines = [
            Line(text="Question text introducing the figure.", bbox=fitz.Rect(70, 100, 400, 120), bold=False, width=330),
            Line(text="Fig. 1.1", bbox=fitz.Rect(200, 250, 250, 265), bold=True, width=50),
            Line(text="Table 1.1", bbox=fitz.Rect(200, 300, 250, 315), bold=True, width=50),
            Line(text="(a) State the relationship shown in Table 1.1.", bbox=fitz.Rect(70, 450, 400, 470), bold=False, width=330),
        ]
        caps = extractor.get_captions(lines)
        assert len(caps) == 2
        assert caps[0]["kind"] == "fig"
        assert caps[1]["kind"] == "table"

        # Fig bounds: upper is preceding prose y1 (120), lower is caption y1 (265)
        fig_bounds = extractor.get_caption_bounds(caps[0], lines)
        assert fig_bounds == (120, 265)

        # Table bounds: upper is caption y0 (300), lower is following question part (a) y0 (450)
        table_bounds = extractor.get_caption_bounds(caps[1], lines)
        assert table_bounds == (300, 450)

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
            assert entry.get("target_type") in ("question", "part", "sub_part")
            has_q = "_question_" in entry["filename"]
            has_sub = "_sub_part_" in entry["filename"]
            has_part = ("_part_" in entry["filename"]) and not has_sub
            assert sum([has_q, has_part, has_sub]) == 1
            if entry["target_type"] == "question":
                assert has_q
            elif entry["target_type"] == "part":
                assert has_part
                assert entry["target_label"] == entry["part"]
            elif entry["target_type"] == "sub_part":
                assert has_sub
                assert entry["target_label"] == entry["part"]

            img_file = output_dir / entry["filename"]
            assert img_file.exists()
            assert img_file.stat().st_size > 0

    def test_is_mcq_paper_detection(self):
        """_is_mcq_paper must correctly distinguish real MCQ papers from uncaptioned written papers."""
        extractor = CambridgeVisualExtractor()

        math_pdf = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
        if math_pdf.exists():
            doc_math = fitz.open(str(math_pdf))
            assert not extractor._is_mcq_paper(doc_math, math_pdf), "4024 written paper must NOT be classified as MCQ!"
            doc_math.close()

        mcq_pdf = Path("pipeline/storage/5054/2011/nov/5054_w11_qp_11.pdf")
        if mcq_pdf.exists():
            doc_mcq = fitz.open(str(mcq_pdf))
            assert extractor._is_mcq_paper(doc_mcq, mcq_pdf), "5054/11 MCQ paper must be classified as MCQ!"
            doc_mcq.close()

        essay_pdf = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_22.pdf")
        if essay_pdf.exists():
            doc_essay = fitz.open(str(essay_pdf))
            assert not extractor._is_mcq_paper(doc_essay, essay_pdf), "5054/22 theory paper must NOT be classified as MCQ!"
            doc_essay.close()

    def test_extract_mcq_paper_real_pdf(self, tmp_path):
        """MCQ extraction must recover labeled diagrams and filter pure answer tables."""
        mcq_pdf = Path("pipeline/storage/5054/2011/nov/5054_w11_qp_11.pdf")
        if not mcq_pdf.exists():
            mcq_pdf = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_11.pdf")
        if not mcq_pdf.exists():
            pytest.skip("No sample MCQ PDF available in storage.")

        output_dir = tmp_path / "mcq_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(mcq_pdf, output_dir)

        questions = {m["question"] for m in manifest}
        assert len(manifest) >= 15
        # Must extract annotated diagrams like Q4, Q6, Q40
        for q in [4, 6, 40]:
            if q in questions:
                assert q in questions
        for m in manifest:
            assert "_question_" in m["filename"]
            assert m["target_type"] == "question"

    def test_extract_maths_paper_real_pdf(self, tmp_path):
        """Maths uncaptioned written paper must extract sparse bearing and coordinate diagrams."""
        math_pdf = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
        if not math_pdf.exists():
            pytest.skip("4024_s10_qp_11.pdf not found in storage.")

        output_dir = tmp_path / "math_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(math_pdf, output_dir)

        questions = {m["question"] for m in manifest}
        # Q5 (bearing diagram with scattered dots), Q9, Q18, Q20 must all be extracted
        for expected_q in [5, 9, 18, 20]:
            assert expected_q in questions, f"Q{expected_q} was expected in extracted Maths diagrams!"



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
            {"question": 1, "target_type": "question", "relative_path": "visuals/p2_q1_question_Fig1_1.png"},
            {"question": 1, "target_type": "question", "relative_path": "visuals/p2_q1_question_Fig1_2.png"},
            {"question": 2, "target_type": "question", "relative_path": "visuals/p6_q2_question_Fig2_1.png"},
        ]

        _patch_question_json(json_file, manifest)

        with open(json_file, "r", encoding="utf-8") as f:
            updated = json.load(f)

        assert updated[0]["image"] == [
            "visuals/p2_q1_question_Fig1_1.png",
            "visuals/p2_q1_question_Fig1_2.png",
        ]
        assert updated[1]["image"] == "visuals/p6_q2_question_Fig2_1.png"
        assert updated[2]["image"] is None

    def test_patch_question_json_nested_subparts(self, tmp_path):
        """_patch_question_json must patch parent parts and nested subparts recursively."""
        json_file = tmp_path / "paper_nested.json"
        records = [
            {
                "QuestionNumber": "5054/s10/qp/22/Q10",
                "image": None,
                "Parts": [
                    {
                        "part_label": "a",
                        "image": None,
                        "sub_parts": []
                    },
                    {
                        "part_label": "b",
                        "image": None,
                        "sub_parts": [
                            {"sub_part_label": "b(i)", "image": None, "sub_parts": []},
                            {"sub_part_label": "b(ii)", "image": None, "sub_parts": []},
                        ]
                    }
                ]
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        manifest = [
            {"question": 10, "target_type": "part", "target_label": "b", "part": "b", "relative_path": "visuals/p14_q10_part_b_Fig_10_1.png"},
            {"question": 10, "target_type": "sub_part", "target_label": "b(ii)", "sub_part": "b(ii)", "relative_path": "visuals/p15_q10_sub_part_b(ii)_Fig_10_2.png"},
        ]

        _patch_question_json(json_file, manifest)

        with open(json_file, "r", encoding="utf-8") as f:
            updated = json.load(f)

        q10 = updated[0]
        # Part b should have Fig_10_1
        assert q10["Parts"][1]["image"] == "visuals/p14_q10_part_b_Fig_10_1.png"
        # Subpart b(i) has no image
        assert q10["Parts"][1]["sub_parts"][0]["image"] is None
        # Subpart b(ii) has Fig_10_2
        assert q10["Parts"][1]["sub_parts"][1]["image"] == "visuals/p15_q10_sub_part_b(ii)_Fig_10_2.png"


class TestTableVisualExtraction:
    """Tests for table extraction in MCQ and essay papers."""

    def test_mcq_table_questions_extracted(self, tmp_path):
        """MCQ option and question tables (e.g. Q7, Q14, Q21, Q28, Q38 in 5054_s10_qp_11) must be extracted."""
        mcq_pdf = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_11.pdf")
        if not mcq_pdf.exists():
            pytest.skip("5054_s10_qp_11.pdf not found in storage.")

        output_dir = tmp_path / "mcq_tables"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(mcq_pdf, output_dir, force_mcq=True)

        extracted_qs = {m["question"] for m in manifest}
        # Table questions that were previously dropped by density threshold
        table_qs = [7, 14, 15, 21, 25, 28, 32, 34, 35, 38]
        for q in table_qs:
            assert q in extracted_qs, f"MCQ table question Q{q} was not extracted!"

    def test_essay_uncaptioned_tables_extracted(self, tmp_path):
        """Uncaptioned data tables in structured papers (e.g. 4024_s10_qp_21) must be extracted."""
        math_pdf = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_21.pdf")
        if not math_pdf.exists():
            pytest.skip("4024_s10_qp_21.pdf not found in storage.")

        output_dir = tmp_path / "math_tables"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(math_pdf, output_dir, force_mcq=False)

        assert len(manifest) >= 5
        # Ensure filenames follow strict discriminator naming rule
        for m in manifest:
            fname = m["filename"]
            assert any(tag in fname for tag in ["_question_", "_part_", "_sub_part_"]), f"Invalid tag in {fname}"

    def test_extract_maths_paper_w10_all_visuals_and_part_naming(self, tmp_path):
        """All diagrams, shapes, graphs, and tables in 4024_w10_qp_21 must be extracted with part-specific filenames."""
        math_pdf = Path("pipeline/storage/4024/2010/nov/4024_w10_qp_21.pdf")
        if not math_pdf.exists():
            pytest.skip("4024_w10_qp_21.pdf not found in storage.")

        output_dir = tmp_path / "w10_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(math_pdf, output_dir, force_mcq=False)

        assert len(manifest) >= 11, f"Expected at least 11 visuals, got {len(manifest)}"

        questions = {m["question"] for m in manifest}
        for expected_q in [2, 3, 4, 6, 7, 8, 9, 10]:
            assert expected_q in questions, f"Q{expected_q} was expected in extracted visuals!"

        # Verify specific part-level naming
        part_entries = [m for m in manifest if m.get("target_type") in ("part", "sub_part")]
        assert len(part_entries) >= 10, "Expected at least 10 part-specific visual entries"
        for m in part_entries:
            fname = m["filename"]
            plbl = m["target_label"]
            assert f"_part_{plbl}_" in fname or f"_sub_part_{plbl}_" in fname, f"Filename {fname} missing part tag for {plbl}"
            assert m["part"] == plbl

    def test_extract_maths_paper_s10_q3_q5_q12_associations(self, tmp_path):
        """4024_s10_qp_21 Q3 stem, Q5 parts a/b, and Q12 parts a/b and subpart b(ii) associations & clean crop."""
        math_pdf = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_21.pdf")
        if not math_pdf.exists():
            pytest.skip("4024_s10_qp_21.pdf not found in storage.")

        output_dir = tmp_path / "s10_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(math_pdf, output_dir, force_mcq=False)

        # 1. Q3 must be associated with question text (stem)
        q3_entries = [m for m in manifest if m["question"] == 3]
        assert len(q3_entries) == 1
        assert q3_entries[0]["target_type"] == "question"
        assert q3_entries[0]["target_label"] == "Q3"
        assert "_question_" in q3_entries[0]["filename"]

        # 2. Q5 must have 1st image -> part a, 2nd image -> part b
        q5_entries = [m for m in manifest if m["question"] == 5]
        assert len(q5_entries) == 2
        assert q5_entries[0]["target_type"] == "part" and q5_entries[0]["target_label"] == "a"
        assert "_part_a_" in q5_entries[0]["filename"]
        assert q5_entries[1]["target_type"] == "part" and q5_entries[1]["target_label"] == "b"
        assert "_part_b_" in q5_entries[1]["filename"]

        # 3. Q12 must have part a, part b, and subpart b(ii) images
        q12_entries = [m for m in manifest if m["question"] == 12]
        assert len(q12_entries) == 3
        assert q12_entries[0]["target_type"] == "part" and q12_entries[0]["target_label"] == "a"
        assert "_part_a_" in q12_entries[0]["filename"]
        assert q12_entries[1]["target_type"] == "part" and q12_entries[1]["target_label"] == "b"
        assert "_part_b_" in q12_entries[1]["filename"]
        assert q12_entries[2]["target_type"] == "sub_part" and q12_entries[2]["target_label"] == "b(ii)"
        assert "_sub_part_b(ii)_" in q12_entries[2]["filename"]

        # 4. Verify Q12 3rd image clean snipping:
        # x0 >= 350 (no left prose bleed) and y1 >= 710 (captures Diagram II caption)
        q12_d3_bbox = q12_entries[2]["bbox"]
        assert q12_d3_bbox[0] >= 350.0, f"Left margin bled into prose text: x0={q12_d3_bbox[0]}"
        assert q12_d3_bbox[3] >= 710.0, f"Diagram II caption cut off: y1={q12_d3_bbox[3]}"

        # 5. Q2 table must be associated with question text (stem), not part_c
        q2_stem_entries = [m for m in manifest if m["question"] == 2 and m["target_type"] == "question"]
        assert len(q2_stem_entries) == 1, "Q2 shop table must be classified as question stem!"
        assert "_question_" in q2_stem_entries[0]["filename"]

        # 6. Q7 1st table must be associated with question text (stem), not part_b
        q7_stem_entries = [m for m in manifest if m["question"] == 7 and m["target_type"] == "question"]
        assert len(q7_stem_entries) == 1, "Q7 time table must be classified as question stem!"
        assert "_question_" in q7_stem_entries[0]["filename"]

        # Q7 2nd table is under part b
        q7_part_b_entries = [m for m in manifest if m["question"] == 7 and m["target_type"] == "part" and m["target_label"] == "b"]
        assert len(q7_part_b_entries) == 1, "Q7 cumulative frequency table must be classified as part b!"

    def test_extract_w10_qp_23_circuit_and_no_text_bleed(self, tmp_path):
        """Verify 5054_w10_qp_23 extracts Fig 10.1 and 10.2 without text bleed, and snips Q11(c)(iii) circuit."""
        pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_qp_23.pdf")
        if not pdf_path.exists():
            pytest.skip("5054_w10_qp_23.pdf not found in storage.")

        output_dir = tmp_path / "w10_23_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Fig. 10.1 (Page 13, Q10): y0 must be >= 110.0 (no bleed from 'force on the bus...')
        # and x1 must be < 500.0 (no PapaCambridge watermark in top-right)
        fig10_1 = [m for m in manifest if m.get("caption") == "Fig. 10.1"]
        assert len(fig10_1) == 1
        bbox10_1 = fig10_1[0]["bbox"]
        assert bbox10_1[1] >= 110.0, f"Fig 10.1 bled into top text: y0={bbox10_1[1]}"
        assert bbox10_1[2] < 500.0, f"Fig 10.1 captured top-right watermark: x1={bbox10_1[2]}"

        # 2. Fig. 10.2 (Page 14, Q10 c(i)): y0 must be >= 370.0 (no bleed from 'into the stationary bus.')
        fig10_2 = [m for m in manifest if m.get("caption") == "Fig. 10.2"]
        assert len(fig10_2) == 1
        bbox10_2 = fig10_2[0]["bbox"]
        assert bbox10_2[1] >= 370.0, f"Fig 10.2 bled into top text: y0={bbox10_2[1]}"
        assert bbox10_2[3] <= 629.0, f"Fig 10.2 bled into bottom mark [3]: y1={bbox10_2[3]}"

        # 3. Uncaptioned Circuit Visual on Page 16 (Q11 c(iii))
        q11_circuit = [m for m in manifest if m["page"] == 16 and m["question"] == 11]
        assert len(q11_circuit) == 1, "Expected uncaptioned circuit visual on page 16 to be extracted!"
        entry = q11_circuit[0]
        assert entry["target_type"] == "sub_part"
        assert entry["target_label"] == "c(iii)"
        assert "fig" not in entry["filename"].lower(), f"Uncaptioned visual filename shouldn't contain fig: {entry['filename']}"
        assert "_sub_part_c(iii)_" in entry["filename"]

    def test_get_question_anchors_rejects_units_and_enforces_monotonicity(self):
        """Verify get_question_anchors rejects units like '1 cm' and non-monotonic numbers."""
        extractor = CambridgeVisualExtractor()
        lines = [
            Line(text="1 cm", bbox=fitz.Rect(30, 368, 52, 379), bold=False, width=22),          # Diagram scale (axis)
            Line(text="2 mm", bbox=fitz.Rect(45, 200, 70, 215), bold=False, width=25),          # Measurement
            Line(text="10 kg", bbox=fitz.Rect(45, 250, 70, 265), bold=False, width=25),         # Measurement
            Line(text="1", bbox=fitz.Rect(45, 100, 55, 115), bold=True, width=10),              # Valid Q1
            Line(text="9 (a)", bbox=fitz.Rect(45, 300, 80, 315), bold=True, width=35),          # Valid Q9(a)
            Line(text="10 An atom of phosphorus...", bbox=fitz.Rect(45, 400, 300, 415), bold=True, width=255), # Valid Q10
        ]
        # Without min_question constraint:
        anchors = extractor.get_question_anchors(lines, min_question=1)
        assert (1, 100) in anchors
        assert (9, 300) in anchors
        assert (10, 400) in anchors
        assert not any(a[0] == 1 and a[1] == 368 for a in anchors), "'1 cm' at y0=368 must NOT be an anchor!"

        # With min_question constraint (e.g. at Question 9):
        anchors_q9 = extractor.get_question_anchors(lines, min_question=9)
        assert (9, 300) in anchors_q9
        assert (10, 400) in anchors_q9
        assert not any(a[0] < 9 for a in anchors_q9), "Anchors < 9 must be excluded when min_question=9"

    def test_validate_and_sanitize_manifest_reconciles_caption_mismatch(self, tmp_path):
        """Verify _validate_and_sanitize_manifest corrects question number and renames file when caption conflicts."""
        extractor = CambridgeVisualExtractor()
        dummy_file = tmp_path / "p13_q1_part_b_Fig_9_2.png"
        dummy_file.write_bytes(b"dummy image data")

        manifest = [
            {
                "page": 13,
                "question": 1,
                "target_type": "part",
                "target_label": "b",
                "part": "b",
                "caption": "Fig. 9.2",
                "bbox": [43.6, 160.5, 529.6, 449.7],
                "filename": "p13_q1_part_b_Fig_9_2.png",
                "file": "p13_q1_part_b_Fig_9_2.png",
                "relative_path": f"{tmp_path.name}/p13_q1_part_b_Fig_9_2.png",
            }
        ]

        sanitized = extractor._validate_and_sanitize_manifest(manifest, tmp_path)
        assert len(sanitized) == 1
        entry = sanitized[0]
        assert entry["question"] == 9, "Caption Fig. 9.2 must force question == 9"
        assert entry["filename"] == "p13_q9_part_b_Fig_9_2.png"
        assert (tmp_path / "p13_q9_part_b_Fig_9_2.png").exists(), "On-disk file must be renamed"
        assert not dummy_file.exists(), "Old file with wrong question must no longer exist"

    def test_5054_w10_qp_21_fig_9_2_naming_and_patching(self, tmp_path):
        """End-to-end regression: 5054_w10_qp_21 Fig 9.2 must be named for Q9 part b and patched to Q9 in JSON."""
        pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_qp_21.pdf")
        if not pdf_path.exists():
            pytest.skip("5054_w10_qp_21.pdf not found in storage.")

        output_dir = tmp_path / "w10_21_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Fig 9.2 check
        fig9_2 = [m for m in manifest if m.get("caption") == "Fig. 9.2"]
        assert len(fig9_2) == 1
        entry = fig9_2[0]
        assert entry["question"] == 9, f"Fig. 9.2 must be question 9, got {entry['question']}"
        assert entry["target_label"] == "b"
        assert entry["filename"] == "p13_q9_part_b_Fig_9_2.png"
        assert (output_dir / "p13_q9_part_b_Fig_9_2.png").exists()

    def test_5054_w10_qp_31_p6_circuit_and_p11_graph_grid(self, tmp_path):
        """Regression test: 5054_w10_qp_31 Page 6 circuit must be Q3 part a, and Page 11 full-page graph grid must be Q4 part e."""
        pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_qp_31.pdf")
        if not pdf_path.exists():
            pytest.skip("5054_w10_qp_31.pdf not found in storage.")

        output_dir = tmp_path / "w10_31_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Page 6 circuit diagram check: must be Q3 part a, NOT Q20
        p6_entries = [m for m in manifest if m.get("page") == 6]
        assert len(p6_entries) == 1, f"Expected 1 visual on page 6, found: {p6_entries}"
        p6 = p6_entries[0]
        assert p6["question"] == 3, f"Page 6 visual must belong to Q3, got Q{p6['question']}"
        assert p6["target_type"] == "part"
        assert p6["target_label"] == "a"
        assert p6["filename"] == "p6_q3_part_a_diagram_1.png"
        assert (output_dir / "p6_q3_part_a_diagram_1.png").exists()

        # Verify no visual in entire paper is associated with non-existent Question 20
        q20_entries = [m for m in manifest if m.get("question") == 20]
        assert len(q20_entries) == 0, f"No visuals should be mapped to Q20: {q20_entries}"

        # 2. Page 11 millimeter graph grid check: must be extracted and associated with Q4 part e
        p11_entries = [m for m in manifest if m.get("page") == 11]
        assert len(p11_entries) == 1, f"Expected 1 visual on page 11 (graph grid), found: {p11_entries}"
        p11 = p11_entries[0]
        assert p11["question"] == 4, f"Page 11 grid must belong to Q4, got Q{p11['question']}"
        assert p11["target_type"] == "part"
        assert p11["target_label"] == "e"
        assert p11["filename"] == "p11_q4_part_e_diagram_1.png"
        assert (output_dir / "p11_q4_part_e_diagram_1.png").exists()







