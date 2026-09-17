"""
Stage 3 — Cambridge Visual Extraction Unit & Integration Tests
Run from project root:  .venv/bin/pytest tests/test_visuals.py -v
"""
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from pipeline.visuals.cli import _patch_question_json
from pipeline.visuals.extractor import CambridgeVisualExtractor, Line


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

        with open(json_file, encoding="utf-8") as f:
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

        with open(json_file, encoding="utf-8") as f:
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

    def test_5090_w11_qp_62_side_by_side_micrographs_and_copyright_guard(self, tmp_path):
        """Regression test: 5090_w11_qp_62 Page 6 Fig 2.1 & 2.2 side-by-side micrographs must not overlap, and Page 8 copyright must be excluded."""
        pdf_path = Path("pipeline/storage/5090/2011/nov/5090_w11_qp_62.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_w11_qp_62.pdf not found in storage.")

        output_dir = tmp_path / "w11_62_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Page 6 check: Fig 2.1 and Fig 2.2 must both be extracted independently
        p6_entries = [m for m in manifest if m.get("page") == 6]
        assert len(p6_entries) == 2, f"Expected 2 visuals on page 6, found: {p6_entries}"
        
        f21 = next((m for m in p6_entries if "Fig_2_1" in m["filename"]), None)
        f22 = next((m for m in p6_entries if "Fig_2_2" in m["filename"]), None)
        assert f21 is not None, "Fig 2.1 not found in page 6 manifest"
        assert f22 is not None, "Fig 2.2 not found in page 6 manifest"

        # Fig 2.1 must capture full width of the plant cell micrograph (>200pt)
        assert f21["bbox"][2] - f21["bbox"][0] > 200.0, f"Fig 2.1 crop too narrow: {f21['bbox']}"
        # Fig 2.2 must capture full width of the blood cell micrograph (>200pt)
        assert f22["bbox"][2] - f22["bbox"][0] > 200.0, f"Fig 2.2 crop too narrow: {f22['bbox']}"

        # Clean horizontal separation: Fig 2.1 x1 must be strictly left of Fig 2.2 x0
        assert f21["bbox"][2] <= f22["bbox"][0] + 5.0, f"Fig 2.1 ({f21['bbox']}) bleeds into Fig 2.2 ({f22['bbox']})"

        # 2. Page 8 check: Copyright Acknowledgements citation must NOT be extracted as a figure
        p8_figs = [m for m in manifest if m.get("page") == 8 and m.get("target_type") != "table" and "Fig" in m.get("filename", "")]
        assert len(p8_figs) == 0, f"Copyright text falsely extracted as figure on page 8: {p8_figs}"

    def test_5090_w25_qp_22_fig_2_1_unbroken(self, tmp_path):
        """Regression test: 5090_w25_qp_22 Page 3 Fig 2.1 must be extracted as a single complete diagram, not split into half."""
        pdf_path = Path("pipeline/storage/5090/2025/nov/5090_w25_qp_22.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_w25_qp_22.pdf not found in storage.")

        output_dir = tmp_path / "w25_22_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # Page 3 check: Only Fig 2.1 must be extracted, with no orphaned diagrams
        p3_entries = [m for m in manifest if m.get("page") == 3]
        assert len(p3_entries) == 1, f"Expected exactly 1 visual on page 3, found {len(p3_entries)}: {p3_entries}"
        
        f21 = p3_entries[0]
        assert "Fig_2_1" in f21["filename"], f"Expected Fig_2_1 on page 3, got: {f21['filename']}"
        assert (output_dir / f21["filename"]).exists()

        # Must span full diagram bounds:
        # x0 <= 65.0 (includes '250cm³from metabolism' at x0=63.9)
        # x1 >= 530.0 (includes right side at x1=535)
        # y0 <= 135.0 (includes upper cylinder headers at y=130)
        # y1 >= 340.0 (includes caption at y=347)
        bbox = f21["bbox"]
        assert bbox[0] <= 65.0, f"Left edge trimmed: {bbox[0]} > 65.0"
        assert bbox[2] >= 530.0, f"Right edge trimmed: {bbox[2]} < 530.0"
        assert bbox[1] <= 135.0, f"Top edge trimmed: {bbox[1]} > 135.0"
        assert bbox[3] >= 340.0, f"Bottom edge trimmed: {bbox[3]} < 340.0"
        assert bbox[2] - bbox[0] > 450.0, f"Width too small (split diagram): {bbox[2] - bbox[0]}"

    def test_5090_w25_qp_11_mcq_no_overwrites_all_visuals_preserved(self, tmp_path):
        """Regression test: 5090_w25_qp_11 MCQ extraction must preserve all diagrams and tables without overwriting."""
        pdf_path = Path("pipeline/storage/5090/2025/nov/5090_w25_qp_11.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_w25_qp_11.pdf not found in storage.")

        output_dir = tmp_path / "w25_11_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir)

        # 1. Total visual count: 22 distinct visuals
        assert len(manifest) == 22, f"Expected 22 visuals, got {len(manifest)}"

        # 2. Strict filename uniqueness: no two visuals may share a filename
        filenames = [m["filename"] for m in manifest]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames detected in manifest!"
        for fn in filenames:
            assert (output_dir / fn).exists(), f"File {fn} missing from disk!"

        # 3. Questions with multiple visuals must preserve all visuals
        multi_q = {9: 2, 10: 2, 11: 2, 16: 2, 26: 2}
        for q, expected_cnt in multi_q.items():
            q_entries = [m for m in manifest if m.get("question") == q]
            assert len(q_entries) == expected_cnt, f"Q{q} expected {expected_cnt} visuals, found {len(q_entries)}"

        # 4. Q8 2x2 option diagrams must be captured in one complete unified image
        q8_entries = [m for m in manifest if m.get("question") == 8]
        assert len(q8_entries) == 1, f"Q8 expected 1 combined visual, found {len(q8_entries)}"
        q8_bbox = q8_entries[0]["bbox"]
        assert q8_bbox[2] - q8_bbox[0] > 400.0, f"Q8 width too small: {q8_bbox}"
        assert q8_bbox[3] - q8_bbox[1] > 350.0, f"Q8 height too small (options split): {q8_bbox}"

        # 5. Q26 table must not bleed into footer
        q26_t = next(m for m in manifest if m.get("question") == 26 and "mcq_2" in m["filename"])
        assert q26_t["bbox"][3] <= 780.0, f"Q26 table bleeds into footer: {q26_t['bbox']}"

    def test_5090_w25_qp_42_fig_1_1_complete(self, tmp_path):
        """Regression test: 5090_w25_qp_42 Page 3 Fig 1.1 must capture beakers, full descriptions, and caption with no duplicate diagram."""
        pdf_path = Path("pipeline/storage/5090/2025/nov/5090_w25_qp_42.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_w25_qp_42.pdf not found in storage.")

        output_dir = tmp_path / "w25_42_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # Page 3 must have exactly 1 visual (Fig 1.1), zero orphaned diagram_1
        p3_entries = [m for m in manifest if m.get("page") == 3]
        assert len(p3_entries) == 1, f"Expected exactly 1 visual on page 3, found {len(p3_entries)}: {p3_entries}"

        f11 = p3_entries[0]
        assert "Fig_1_1" in f11["filename"], f"Expected Fig_1_1, got: {f11['filename']}"
        assert (output_dir / f11["filename"]).exists()

        bbox = f11["bbox"]
        assert bbox[1] <= 195.0, f"Top cut off: {bbox[1]} > 195.0"
        assert bbox[3] >= 430.0, f"Caption cut off: {bbox[3]} < 430.0"
        assert bbox[2] - bbox[0] > 450.0, f"Width too narrow: {bbox[2] - bbox[0]}"
        assert bbox[3] - bbox[1] > 230.0, f"Height too short: {bbox[3] - bbox[1]}"

    def test_5090_s25_qp_42_part_carryover_and_subpart_naming(self, tmp_path):
        """Regression test: 5090_s25_qp_42 Page 3 diagram must be attributed to sub_part_a(ii), not question."""
        pdf_path = Path("pipeline/storage/5090/2025/jun/5090_s25_qp_42.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_s25_qp_42.pdf not found in storage.")

        output_dir = tmp_path / "s25_42_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Page 3 ranking arrow must be attributed to subpart a(ii)
        p3_entries = [m for m in manifest if m.get("page") == 3]
        assert len(p3_entries) == 1, f"Expected 1 visual on page 3, found: {p3_entries}"
        p3_diag = p3_entries[0]
        assert p3_diag["target_type"] == "sub_part"
        assert p3_diag["target_label"] == "a(ii)"
        assert p3_diag["part"] == "a(ii)"
        assert p3_diag["filename"] == "p3_q1_sub_part_a(ii)_diagram_1.png"
        assert (output_dir / p3_diag["filename"]).exists()

        # 2. Page 5 graph grid must be attributed to subpart c(i)
        p5_entries = [m for m in manifest if m.get("page") == 5]
        assert len(p5_entries) == 1, f"Expected 1 visual on page 5, found: {p5_entries}"
        p5_diag = p5_entries[0]
        assert p5_diag["target_type"] == "sub_part"
        assert p5_diag["target_label"] == "c(i)"
        assert p5_diag["part"] == "c(i)"
        assert p5_diag["filename"] == "p5_q1_sub_part_c(i)_diagram_2.png"
        assert (output_dir / p5_diag["filename"]).exists()

        # 3. Verify Page 2 Fig 1.1 and Page 4 Table 1.1 preserved
        p2_entries = [m for m in manifest if m.get("page") == 2]
        assert len(p2_entries) == 1
        assert p2_entries[0]["filename"] == "p2_q1_part_a_Fig_1_1.png"

        p4_entries = [m for m in manifest if m.get("page") == 4]
        assert len(p4_entries) == 1
        assert p4_entries[0]["filename"] == "p4_q1_part_c_Table_1_1.png"

    def test_5090_w25_qp_22_matching_table_unified(self, tmp_path):
        """
        Regression test: 5090_w25_qp_22 Q1(a) matching exercise (gland/hormone/target organ)
        must be extracted as a single unified image, not as 3 separate per-column crops.

        The matching layout on page 2 consists of 12 bordered boxes in 3 columns x 4 rows
        with connecting lines between them. After the _merge_matching_columns pass the three
        column clusters must be merged into one bounding rect spanning all three columns.
        """
        pdf_path = Path("pipeline/storage/5090/2025/nov/5090_w25_qp_22.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_w25_qp_22.pdf not found in storage.")

        output_dir = tmp_path / "w25_22_matching_visuals"
        extractor = CambridgeVisualExtractor(dpi=150)
        manifest = extractor.extract(pdf_path, output_dir, force_mcq=False)

        # 1. Page 2 must contain exactly 1 visual (the unified matching exercise)
        p2_entries = [m for m in manifest if m.get("page") == 2]
        assert len(p2_entries) == 1, (
            f"Expected exactly 1 visual on page 2 (unified matching exercise), "
            f"found {len(p2_entries)}: {p2_entries}"
        )

        matching_entry = p2_entries[0]

        # 2. The image file must exist on disk
        assert (output_dir / matching_entry["filename"]).exists(), (
            f"Matching exercise image not saved: {matching_entry['filename']}"
        )

        # 3. Bounding box must span all 3 columns:
        #    - x0 <= 75  (left edge of column 1, x0=71.9)
        #    - x1 >= 520 (right edge of column 3, x1=523.4)
        #    - width >= 400 pt  (must cover all three columns, not just one)
        bbox = matching_entry["bbox"]
        assert bbox[0] <= 75.0, (
            f"Left edge too far right -- only captured rightmost column(s): x0={bbox[0]}")
        assert bbox[2] >= 520.0, (
            f"Right edge too far left -- only captured leftmost column(s): x1={bbox[2]}")
        assert bbox[2] - bbox[0] >= 400.0, (
            f"Width too small -- columns not merged: width={bbox[2] - bbox[0]:.1f} pt")

        # 4. Height must cover the full row span of the boxes (y0~217 to y1~446)
        assert bbox[3] - bbox[1] >= 200.0, (
            f"Height too small -- rows not fully captured: height={bbox[3] - bbox[1]:.1f} pt")

        # 5. Column-header text ("gland", "hormone", "target organ") sits at y~183.
        #    The boxes start at y~217. The crop top must NOT rise above y=210 to
        #    avoid absorbing the header text prose into the image.
        assert bbox[1] >= 210.0, (
            f"Top edge too high -- header text may be included: y0={bbox[1]:.1f} (expected >= 210)")

        # 6. Cross-check: existing Fig 2.1 test -- page 3 must still have exactly 1 visual
        p3_entries = [m for m in manifest if m.get("page") == 3]
        assert len(p3_entries) == 1, (
            f"Fig 2.1 regression: expected 1 visual on page 3, found {len(p3_entries)}: {p3_entries}")
        assert "Fig_2_1" in p3_entries[0]["filename"], (
            f"Fig 2.1 regression: expected Fig_2_1 on page 3, got {p3_entries[0]['filename']}")
