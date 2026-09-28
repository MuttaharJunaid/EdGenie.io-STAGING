from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.extractor.paper_type_detector import PaperCategory, PaperTypeDetector
from pipeline.extractor.syllabus_loader import SyllabusLoader


def test_paper_type_detection():
    # Detect by filename
    assert PaperTypeDetector.detect_by_filename("7707_s23_qp_12.pdf") == PaperCategory.MCQ_QP
    assert PaperTypeDetector.detect_by_filename("7707_w22_ms_11.pdf") == PaperCategory.MCQ_MS
    assert PaperTypeDetector.detect_by_filename("7707_s23_qp_22.pdf") == PaperCategory.ESSAY_QP
    assert PaperTypeDetector.detect_by_filename("7707_w22_ms_32.pdf") == PaperCategory.ESSAY_MS

def test_syllabus_loader(tmp_path):
    # Test 5-column layout
    csv_file = tmp_path / "syllabus.csv"
    csv_file.write_text("""unit_number,unit_name,topic_number,topic_name,sub_topic
1,Financial Accounting,1.2,Source Documents,"Invoices, Credit notes"
2,Financial Statements,2.1,Income Statement,"Gross Profit, Net Profit"
""")

    units = SyllabusLoader.load_syllabus(csv_file)
    assert len(units) == 2
    assert units[0].unit_number == "1"
    assert units[0].topic_name == "Source Documents"
    assert units[1].unit_number == "2"
    assert "Income Statement" in units[1].topic_name

    # Test 3-column layout
    csv_file_3col = tmp_path / "syllabus_3col.csv"
    csv_file_3col.write_text("""Unit,Topic,Subtopic
UNIT 1: Cells,1.1 Cell structure and function,Examine plant cells
UNIT 2: Classification,2.1 Concept system,Species definition
""")
    units_3col = SyllabusLoader.load_syllabus(csv_file_3col)
    assert len(units_3col) == 2
    assert units_3col[0].unit_number == "1"
    assert units_3col[0].unit_name == "Cells"
    assert units_3col[0].topic_number == "1.1"
    # Test resolve_syllabus_path
    assert SyllabusLoader.resolve_syllabus_path(csv_file) == csv_file.resolve()
    assert SyllabusLoader.resolve_syllabus_path("pipeline/syllabi/PHYSICS_5054.csv") is not None
    assert SyllabusLoader.resolve_syllabus_path(None, subject_code="5054") is not None
    assert SyllabusLoader.resolve_syllabus_path("non_existent_file_xyz_123.csv") is None
    assert units_3col[0].sub_topic == "Examine plant cells"

    formatted = SyllabusLoader.format_syllabus_for_prompt(units)
    assert "Unit 1" in formatted
    assert "Source Documents" in formatted


def test_process_dir_sorting_and_er_gt_exclusion(tmp_path):
    # Create test PDFs: qp_12, qp_11, _er_11, _gt_11
    (tmp_path / "5054_s23_qp_12.pdf").write_bytes(b"%PDF-1.4 12")
    (tmp_path / "5054_s23_qp_11.pdf").write_bytes(b"%PDF-1.4 11")
    (tmp_path / "5054_s23_er_11.pdf").write_bytes(b"%PDF-1.4 er")
    (tmp_path / "5054_s23_gt_11.pdf").write_bytes(b"%PDF-1.4 gt")
    (tmp_path / "5054_s23_ir_31.pdf").write_bytes(b"%PDF-1.4 ir")

    all_pdfs = list(tmp_path.glob("**/*.pdf"))
    pdf_files = [
        p for p in all_pdfs
        if not any(sfx in p.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ir.", "_ci_", "_ci."])
    ]
    pdf_files.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))

    # Verify _er, _gt, _ir are excluded
    filenames = [p.name for p in pdf_files]
    assert "5054_s23_er_11.pdf" not in filenames
    assert "5054_s23_gt_11.pdf" not in filenames
    assert "5054_s23_ir_31.pdf" not in filenames

    # Verify natural order sorting (qp_11 comes before qp_12)
    assert filenames == ["5054_s23_qp_11.pdf", "5054_s23_qp_12.pdf"]


def test_classifier_agent_merge():
    """ClassifierAgent.merge() correctly fills classification fields into skeleton records."""
    from pipeline.extractor.classifier_agent import ClassifierAgent
    agent = ClassifierAgent()

    records = [
        {
            "QuestionNumber": "Jun/11/2023/Q1",
            "Question_text": "Define supply.",
            "Unit": "",
            "Topic": "",
            "Topic_name": "",
            "Sub_topic": [],
            "Difficulty_score": 0,
            "Difficulty_label": "",
            "Command_words": [],
            "Marks": 2,
            "Parts": [],
            "image": None,
        }
    ]

    classifications = {
        "Jun/11/2023/Q1": {
            "question_number": "Jun/11/2023/Q1",
            "Unit": "3",
            "Topic": "3.1",
            "Topic_name": "Supply and Demand",
            "Sub_topic": ["Law of Supply"],
            "Difficulty_score": 3,
            "Difficulty_label": "easy",
            "Command_words": ["define"],
        }
    }

    merged = agent.merge(records, classifications)
    assert merged[0]["Unit"] == "3"
    assert merged[0]["Topic"] == "3.1"
    assert merged[0]["Topic_name"] == "Supply and Demand"
    assert merged[0]["Sub_topic"] == ["Law of Supply"]
    assert merged[0]["Difficulty_score"] == 3
    assert merged[0]["Difficulty_label"] == "easy"
    assert merged[0]["Command_words"] == ["define"]
    assert merged[0]["Question_text"] == "Define supply."
    assert merged[0]["Marks"] == 2


def test_classifier_agent_merge_ms_no_op():
    """merge() is a no-op for MS records — AnswerNumber key, no classification fields."""
    from pipeline.extractor.classifier_agent import ClassifierAgent
    agent = ClassifierAgent()

    records = [
        {"AnswerNumber": "Jun/11/2023/Q1", "Answer_text": "Some answer", "Parts": [], "image": None}
    ]
    merged = agent.merge(records, {})
    assert merged[0]["Answer_text"] == "Some answer"


def test_answer_prompt_cleanup_and_preservation():
    """Verify that trailing answer prompts (height =, V2 =, Answer (a) ...) are stripped while mid-sentence = are preserved."""
    from pipeline.extractor.text_utils import clean_answer_prompts

    # Standalone answer prompts should be stripped completely
    assert clean_answer_prompts("height =") == ""
    assert clean_answer_prompts("Answer (a) AEˆD =") == ""
    assert clean_answer_prompts("V2 =") == ""
    assert clean_answer_prompts("water:") == ""

    # Trailing prompts at end of sentences should be stripped
    assert clean_answer_prompts("Calculate the height of the cylinder. height =") == "Calculate the height of the cylinder."
    assert clean_answer_prompts("Calculate V2. V2 =") == "Calculate V2."
    assert clean_answer_prompts("AEˆD, Answer (a) AEˆD =") == "AEˆD,"

    # Genuine mathematical expressions with = must NOT be stripped
    expr1 = "It is given that k = 2 lm 3 n . + Express m in terms of k, l and n."
    assert clean_answer_prompts(expr1) == expr1

    expr2 = "Given that f(t) = t, find t."
    assert clean_answer_prompts(expr2) == expr2

    expr3 = "Using congruent triangles, show that BD = AC."
    assert clean_answer_prompts(expr3) == expr3

    expr4 = "t2 – 4t + 3 = 0."
    assert clean_answer_prompts(expr4) == expr4



def test_cli_paper_type_override(tmp_path):
    """Verify that paper_type parameter overrides category on the Essay/MCQ axis for both QP and MS."""
    import pymupdf as fitz
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.cli import extract_single

    # Build a minimal synthetic QP PDF in tmp_path (avoids dependency on gitignored storage/)
    pdf_path = tmp_path / "4024_s10_qp_11.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1  Calculate the force. [3]")
    doc.save(str(pdf_path))
    doc.close()

    with patch.object(AgenticExtractor, "extract", return_value=[{"QuestionNumber": "Q1", "Question_text": "test", "Marks": 1}]):
        # Override to essay
        res_essay = extract_single(pdf_path, force=True, paper_type="essay")
        assert res_essay is not None
        assert res_essay[2] == PaperCategory.ESSAY_QP

        # Override to mcq
        res_mcq = extract_single(pdf_path, force=True, paper_type="mcq")
        assert res_mcq is not None
        assert res_mcq[2] == PaperCategory.MCQ_QP


def test_math_signs_and_exponents_normalization():
    """Verify clean_math_text normalises powers, units, and mathematical notation."""
    from pipeline.extractor.text_utils import clean_math_text

    # 1. Powers of 10 and scientific notation
    assert clean_math_text("5.98 × 1021") == "5.98 × 10²¹"
    assert clean_math_text("3.0 * 108 m/s") == "3.0 × 10⁸ m/s"
    assert clean_math_text("10-4") == "10⁻⁴"
    assert clean_math_text("10–3") == "10⁻³"
    # Plain numbers containing 10 or 100 must NOT be corrupted
    assert clean_math_text("0 ≤ t < 100 120") == "0 ≤ t < 100 120"
    assert clean_math_text("80 100 120") == "80 100 120"
    assert clean_math_text("1000") == "1000"

    # 2. Units with powers
    assert clean_math_text("500 cm3") == "500 cm³"
    assert clean_math_text("10 m2") == "10 m²"
    assert clean_math_text("volume in km3") == "volume in km³"

    # 3. Algebraic powers
    assert clean_math_text("4x2 – 25y2") == "4x² – 25y²"
    assert clean_math_text("t2 – 4t + 3 = 0") == "t² – 4t + 3 = 0"
    assert clean_math_text("n3 > 1") == "n³ > 1"

    # 4. Collapse spaces before superscripts
    assert clean_math_text("5t ² metres") == "5t² metres"



def test_agentic_extractor_multi_page_stitching():
    """Verify that AgenticExtractor correctly merges essay questions and subparts split across pages."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # Page 1 contains Q1 (complete) and Q2 (stem and part a)
    page1 = [
        {
            "QuestionNumber": "Q1",
            "Question_text": "Define speed.",
            "Marks": 2,
            "Parts": [{"part_label": "a", "part_text": "Define speed.", "marks": 2}],
        },
        {
            "QuestionNumber": "Q2",
            "Question_text": "A stone falls freely under gravity.",
            "Marks": 2,
            "Parts": [{"part_label": "a", "part_text": "Calculate its acceleration.", "marks": 2}],
        },
    ]

    # Page 2 continues Q2 (stem continuation, part a continuation, part b with sub-parts b(i) and b(ii)) and introduces Q3
    page2 = [
        {
            "QuestionNumber": "Q2",
            "Question_text": "Air resistance is negligible.",
            "Marks": 3,
            "Parts": [
                {
                    "part_label": "a",
                    "part_text": "Take acceleration of free fall g = 9.8 m/s².",
                    "marks": 2,
                    "sub_parts": [],
                },
                {
                    "part_label": "b",
                    "part_text": "The stone hits the ground.",
                    "marks": 3,
                    "sub_parts": [
                        {"sub_part_label": "b(i)", "sub_part_text": "State the velocity.", "marks": 1},
                        {"sub_part_label": "b(ii)", "sub_part_text": "Calculate energy.", "marks": 2},
                    ],
                }
            ],
        },
        {
            "QuestionNumber": "Q3",
            "Question_text": "State Ohm's Law.",
            "Marks": 1,
            "Parts": [{"part_label": "a", "part_text": "State Ohm's Law.", "marks": 1}],
        },
    ]

    stitched = extractor._stitch_essay_qp([page1, page2], meta)

    assert len(stitched) == 3
    q1 = stitched[0]
    q2 = stitched[1]
    q3 = stitched[2]

    assert q1["QuestionNumber"] == "Jun/11/2010/Q1"
    assert q2["QuestionNumber"] == "Jun/11/2010/Q2"
    assert q3["QuestionNumber"] == "Jun/11/2010/Q3"

    # Q2 stem should be stitched with \n\n
    assert q2["Question_text"] == "A stone falls freely under gravity.\n\nAir resistance is negligible."

    # Q2 should have both part a and part b, and part a should have stitched text with \n\n
    assert len(q2["Parts"]) == 2
    assert q2["Parts"][0]["part_label"] == "a"
    assert q2["Parts"][0]["part_text"] == "Calculate its acceleration.\n\nTake acceleration of free fall g = 9.8 m/s²."
    assert q2["Parts"][1]["part_label"] == "b"
    assert len(q2["Parts"][1]["sub_parts"]) == 2
    assert q2["Marks"] == 5  # 2 (part a) + 3 (part b: 1+2)


def test_agentic_extractor_mcq_stitching():
    """Verify that AgenticExtractor correctly assembles MCQ questions across pages."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "12", "year": "2023"}

    page1 = [
        {
            "QuestionNumber": "1",
            "Question_text": "Which quantity is a vector?",
            "Options": {"A": "mass", "B": "force", "C": "time", "D": "speed"},
            "Marks": 1,
        }
    ]
    page2 = [
        {
            "QuestionNumber": "2",
            "Question_text": "A car moves at 10 m/s.",
            "Options": {"A": "10", "B": "20", "C": "30", "D": "40"},
            "Marks": 1,
        }
    ]

    stitched = extractor._stitch_mcq_qp([page1, page2], meta)
    assert len(stitched) == 2
    assert stitched[0]["QuestionNumber"] == "Jun/12/2023/Q1"
    assert stitched[0]["Options"]["B"] == "force"
    assert stitched[1]["QuestionNumber"] == "Jun/12/2023/Q2"


def test_agentic_extractor_page_extraction_error_handling():
    """Verify that when Gemini page extraction encounters an unrecoverable error, it raises."""
    import fitz

    from pipeline.extractor.agentic_extractor import AgenticExtractor

    extractor = AgenticExtractor()
    mock_doc = fitz.open()
    mock_doc.new_page()

    # _extract_page_with_gemini is the actual method that raises;
    # _extract_page was a dead wrapper that has been removed.
    schema, prompt = extractor._get_page_prompt(
        PaperCategory.ESSAY_QP, 0, 1,
        meta={"session": "Jun", "variant": "11", "year": "2010"},
    )
    with patch.object(extractor, "_extract_page_with_gemini", side_effect=Exception("Gemini quota exceeded")):
        with pytest.raises(Exception, match="Gemini quota exceeded"):
            extractor._extract_page_with_gemini(
                doc=mock_doc,
                pno=0,
                schema=schema,
                prompt=prompt,
            )


def test_agentic_extractor_prompt_math_and_subparts():
    """Verify that AgenticExtractor includes explicit math notation rules and sub_parts structure in prompts."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory
    from pipeline.extractor.schemas import EssayQuestionsContainer, MCQQuestionsContainer

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # Check Essay QP prompt
    schema_essay, prompt_essay = extractor._get_page_prompt(PaperCategory.ESSAY_QP, 0, 10, meta)
    assert schema_essay == EssayQuestionsContainer
    # Sub-parts instructions
    assert "sub_parts" in prompt_essay
    assert "sub_part_label" in prompt_essay
    assert "sub_part_text" in prompt_essay
    # Math instructions
    assert "Exponentials:" in prompt_essay
    assert "Fractions:" in prompt_essay
    assert "Multiplications:" in prompt_essay
    assert "Fractions as Exponents:" in prompt_essay
    assert "Vector Signs:" in prompt_essay
    assert "Congruency Signs:" in prompt_essay
    assert "Equivalence / Identity Signs:" in prompt_essay

    # Check MCQ QP prompt
    schema_mcq, prompt_mcq = extractor._get_page_prompt(PaperCategory.MCQ_QP, 0, 10, meta)
    assert schema_mcq == MCQQuestionsContainer
    assert "Multiplications:" in prompt_mcq
    assert "Vector Signs:" in prompt_mcq

    # Check clean_math_text equivalence and congruency normalization
    from pipeline.extractor.text_utils import clean_math_text
    assert clean_math_text(r"f(x) \equiv 2x") == "f(x) ≡ 2x"
    assert clean_math_text(r"\triangle ABC \cong \triangle DEF") == r"\triangle ABC ≅ \triangle DEF"


def test_normalize_subpart_label():
    """Verify that _normalize_subpart_label strictly normalizes sub-part labels to (i), (ii), (iii)."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    assert AgenticExtractor._normalize_subpart_label("ai") == "(i)"
    assert AgenticExtractor._normalize_subpart_label("aii") == "(ii)"
    assert AgenticExtractor._normalize_subpart_label("aiii") == "(iii)"
    assert AgenticExtractor._normalize_subpart_label("a(i)") == "(i)"
    assert AgenticExtractor._normalize_subpart_label("b(i)") == "(i)"
    assert AgenticExtractor._normalize_subpart_label("b(ii)") == "(ii)"
    assert AgenticExtractor._normalize_subpart_label("(i)") == "(i)"
    assert AgenticExtractor._normalize_subpart_label("i") == "(i)"
    assert AgenticExtractor._normalize_subpart_label("(1)") == "(1)"
    assert AgenticExtractor._normalize_subpart_label("(iv)") == "(iv)"
    assert AgenticExtractor._normalize_subpart_label("(v)") == "(v)"


def test_agentic_extractor_full_pdf_and_page_prompts_sanity_check():
    """Verify that both full-PDF and page-level prompts enforce (i), (ii), (iii) and include SANITY CHECK."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # 1. Full-PDF prompt
    _, full_prompt = extractor._get_full_pdf_prompt(PaperCategory.ESSAY_QP, meta)
    assert "POST-EXTRACTION SANITY CHECK:" in full_prompt
    assert "Displacement Check:" in full_prompt
    assert "NEVER use \"ai\", \"aii\", \"aiii\"" in full_prompt
    assert "(i)" in full_prompt and "(ii)" in full_prompt

    # 2. Page-level prompt (used by Qwen fallback)
    _, page_prompt = extractor._get_page_prompt(PaperCategory.ESSAY_QP, 0, 10, meta)
    assert "POST-EXTRACTION SANITY CHECK:" in page_prompt
    assert "Displacement Check:" in page_prompt
    assert "NEVER use \"ai\", \"aii\", \"aiii\"" in page_prompt


def test_prompt_text_formatting_and_stem_boundary_instructions():
    """Verify that prompts enforce paragraph breaks, indentation, pre-(a) stem boundary, and diagram label exclusion rules."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "41", "year": "2025"}

    # 1. Essay QP Prompt
    _, essay_qp_prompt = extractor._get_full_pdf_prompt(PaperCategory.ESSAY_QP, meta)
    assert "CRITICAL TEXT FORMATTING, NEWLINES, INDENTATION & DIAGRAM LABEL RULES:" in essay_qp_prompt
    assert "PARAGRAPH BREAKS (\\n\\n):" in essay_qp_prompt
    assert "Every distinct paragraph printed in the examination paper MUST be separated by a double newline" in essay_qp_prompt
    assert "INDENTATION & LISTS:" in essay_qp_prompt
    assert "Preserve hierarchical indentation using 2 or 4 leading spaces" in essay_qp_prompt
    assert "STRICT QUESTION STEM vs. PART BOUNDARY:" in essay_qp_prompt
    assert "ALL text, scenario setups, experimental procedures, apparatus descriptions, and instructions appearing BEFORE the printed label '(a)' belong STRICTLY and EXCLUSIVELY to \"Question_text\"" in essay_qp_prompt
    assert "NEVER place pre-'(a)' text into \"part_text\" of part (a)" in essay_qp_prompt
    assert "EXCLUDE DIAGRAM & TABLE LABELS / CALLOUTS:" in essay_qp_prompt
    assert "Any text label, pointer arrow text, component name, axis callout, or annotation shown NEAR, INSIDE, or ATTACHED to a diagram" in essay_qp_prompt
    assert "DO NOT extract labels, callouts, or annotations shown near or inside diagrams" in essay_qp_prompt
    assert "Newlines & Indentation Check:" in essay_qp_prompt
    assert "Question Stem vs Part (a) Boundary Check:" in essay_qp_prompt
    assert "Diagram & Table Labels Check:" in essay_qp_prompt

    # 2. MCQ QP Prompt
    _, mcq_qp_prompt = extractor._get_page_prompt(PaperCategory.MCQ_QP, 0, 5, meta)
    assert "CRITICAL TEXT FORMATTING, NEWLINES, INDENTATION & DIAGRAM LABEL RULES:" in mcq_qp_prompt
    assert "PARAGRAPH BREAKS (\\n\\n):" in mcq_qp_prompt
    assert "DO NOT extract labels, callouts, or annotations shown near or inside diagrams and tables" in mcq_qp_prompt

    # 3. Essay MS Prompt
    _, essay_ms_prompt = extractor._get_full_pdf_prompt(PaperCategory.ESSAY_MS, meta)
    assert "CRITICAL TEXT FORMATTING, NEWLINES, INDENTATION & DIAGRAM LABEL RULES:" in essay_ms_prompt


def test_stitch_essay_ms_merges_answers_and_deduplicates_parts():
    """Verify that _stitch_essay_ms merges Answer_text with \\n\\n and deduplicates part answers."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "41", "year": "2025"}

    page1 = [
        {
            "AnswerNumber": "Q1",
            "Answer_text": "General criteria for Q1.",
            "Parts": [
                {"part_label": "a", "part_answer": "Points 1 and 2: A is correct"},
                {"part_label": "b", "part_answer": "Repeat test for accuracy"},
            ],
        }
    ]
    page2 = [
        {
            "AnswerNumber": "Q1",
            "Answer_text": "Additional general guidance.",
            "Parts": [
                {"part_label": "a", "part_answer": "Point 3: B is correct"},
                {"part_label": "c", "part_answer": "Keep in dark"},
            ],
        }
    ]

    stitched = extractor._stitch_essay_ms([page1, page2], meta)
    assert len(stitched) == 1
    q1 = stitched[0]

    # Answer_text should be merged with \n\n
    assert q1["Answer_text"] == "General criteria for Q1.\n\nAdditional general guidance."

    # Parts should be deduplicated by part_label
    assert len(q1["Parts"]) == 3
    assert q1["Parts"][0]["part_label"] == "a"
    assert q1["Parts"][0]["part_answer"] == "Points 1 and 2: A is correct\n\nPoint 3: B is correct"
    assert q1["Parts"][1]["part_label"] == "b"
    assert q1["Parts"][2]["part_label"] == "c"


def test_agentic_extractor_gemini_full_pdf_primary_and_page_fallback(tmp_path):
    """Verify that AgenticExtractor.extract uses Gemini full-PDF as primary and falls back to Gemini page-by-page."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # Create dummy PDF with text
    import fitz
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Question 1: Calculate force [2]")
    pdf_file = tmp_path / "test_paper.pdf"
    doc.save(str(pdf_file))
    doc.close()

    # 1. Primary: Gemini succeeds
    gemini_records = [
        {
            "QuestionNumber": "Q1",
            "Question_text": "Calculate the speed of light.",
            "Marks": 2,
            "Parts": [
                {
                    "part_label": "a",
                    "part_text": "Calculate speed.",
                    "marks": 2,
                    "sub_parts": [
                        {"sub_part_label": "ai", "sub_part_text": "Step 1", "marks": 1},
                        {"sub_part_label": "aii", "sub_part_text": "Step 2", "marks": 1},
                    ],
                }
            ],
        }
    ]

    with patch.object(extractor, "_extract_with_gemini_full_pdf", return_value=gemini_records) as mock_gemini:
        results = extractor.extract(pdf_file, PaperCategory.ESSAY_QP, meta)
        assert mock_gemini.called
        assert len(results) == 1
        assert results[0]["QuestionNumber"] == "Jun/11/2010/Q1"
        # Verify sub-parts normalized from ai, aii to (i), (ii)
        assert results[0]["Parts"][0]["sub_parts"][0]["sub_part_label"] == "(i)"
        assert results[0]["Parts"][0]["sub_parts"][1]["sub_part_label"] == "(ii)"

    # 2. Fallback: Gemini full PDF fails, falls back to Gemini page-by-page
    page_records = [
        {
            "QuestionNumber": "Q1",
            "Question_text": "Extracted via page-by-page fallback.",
            "Marks": 1,
            "Parts": [
                {
                    "part_label": "a",
                    "part_text": "Part a.",
                    "marks": 1,
                    "sub_parts": [
                        {"sub_part_label": "b(i)", "sub_part_text": "Sub part i", "marks": 1},
                    ],
                }
            ],
        }
    ]

    with patch.object(extractor, "_extract_with_gemini_full_pdf", side_effect=Exception("Gemini quota exceeded")), \
         patch.object(extractor, "_extract_page_with_gemini", return_value=page_records) as mock_page:

        fallback_results = extractor.extract(pdf_file, PaperCategory.ESSAY_QP, meta)
        assert mock_page.called
        assert len(fallback_results) == 1
        assert fallback_results[0]["Question_text"] == "Extracted via page-by-page fallback."
        assert fallback_results[0]["Parts"][0]["sub_parts"][0]["sub_part_label"] == "(i)"


def test_question_paper_null_topics_mcq_and_essay():
    """Verify that Unit, Topic, Topic_name, and Sub_topic are strictly null in QP prompts and output records."""
    import json

    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # 1. Prompts specify Unit, Topic, Topic_name, and Sub_topic as null
    _, mcq_prompt = extractor._get_full_pdf_prompt(PaperCategory.MCQ_QP, meta)
    assert "- Unit: null" in mcq_prompt
    assert "- Topic: null" in mcq_prompt
    assert "- Topic_name: null" in mcq_prompt
    assert "- Sub_topic: null" in mcq_prompt

    _, essay_prompt = extractor._get_full_pdf_prompt(PaperCategory.ESSAY_QP, meta)
    assert "Leave Unit, Topic, Topic_name, and Sub_topic strictly as null" in essay_prompt

    # 2. Post-processing sets these fields to None (serializing to null in JSON)
    raw_mcq = [{
        "QuestionNumber": "Q1",
        "Question_text": "Sample MCQ",
        "Options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "Unit": "Should be wiped",
        "Topic": "Should be wiped",
    }]
    mcq_res = extractor._postprocess_mcq_qp(raw_mcq, meta)
    assert mcq_res[0]["Unit"] is None
    assert mcq_res[0]["Topic"] is None
    assert mcq_res[0]["Topic_name"] is None
    assert mcq_res[0]["Sub_topic"] is None

    mcq_json = json.dumps(mcq_res[0])
    assert '"Unit": null' in mcq_json
    assert '"Topic": null' in mcq_json
    assert '"Topic_name": null' in mcq_json
    assert '"Sub_topic": null' in mcq_json

    raw_essay = [{
        "QuestionNumber": "Q1",
        "Question_text": "Sample Essay",
        "Marks": 2,
        "Unit": "Should be wiped",
        "Topic": "Should be wiped",
        "Parts": []
    }]
    essay_res = extractor._postprocess_essay_qp(raw_essay, meta)
    assert essay_res[0]["Unit"] is None
    assert essay_res[0]["Topic"] is None
    assert essay_res[0]["Topic_name"] is None
    assert essay_res[0]["Sub_topic"] is None

    essay_json = json.dumps(essay_res[0])
    assert '"Unit": null' in essay_json
    assert '"Topic": null' in essay_json
    assert '"Topic_name": null' in essay_json
    assert '"Sub_topic": null' in essay_json


def test_mark_scheme_empty_answer_text_and_gibberish_stripping():
    """Verify that MS prompts instruct LLM to avoid gibberish, and postprocessor strips gibberish Answer_text when parts exist."""
    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    meta = {"session": "Jun", "variant": "11", "year": "2010"}

    # 1. Prompts warn against gibberish Answer_text
    _, ms_prompt = extractor._get_full_pdf_prompt(PaperCategory.ESSAY_MS, meta)
    assert 'set "Answer_text": ""' in ms_prompt
    assert "DO NOT put gibberish, placeholder text" in ms_prompt

    _, ms_page_prompt = extractor._get_page_prompt(PaperCategory.ESSAY_MS, 0, 5, meta)
    assert 'set "Answer_text": ""' in ms_page_prompt
    assert "DO NOT put gibberish, placeholder text" in ms_page_prompt

    # 2. Post-processing cleans placeholder / gibberish when parts exist
    test_cases = [
        ("...", ""),
        ("---", ""),
        ("N/A", ""),
        ("None", ""),
        ("null", ""),
        ("see parts", ""),
        ("part a", ""),
        ("part (b)", ""),
        ("Answer (a)", ""),
        ("x = 5", ""),  # Duplicates part_answer exactly
        ("General guidance for whole question", "General guidance for whole question"),  # Genuine text preserved
    ]

    for given_stem, expected_stem in test_cases:
        raw_answers = [{
            "AnswerNumber": "Q1",
            "Answer_text": given_stem,
            "Parts": [
                {"part_label": "a", "part_answer": "x = 5"},
                {"part_label": "b", "part_answer": "y = 10"},
            ]
        }]
        res = extractor._postprocess_essay_ms(raw_answers, meta)
        assert res[0]["Answer_text"] == expected_stem, f"Failed on input: {given_stem}"

    # 3. If there are NO parts, Answer_text is kept as is
    no_parts_raw = [{
        "AnswerNumber": "Q2",
        "Answer_text": "Complete solution without parts",
        "Parts": []
    }]
    res_no_parts = extractor._postprocess_essay_ms(no_parts_raw, meta)
    assert res_no_parts[0]["Answer_text"] == "Complete solution without parts"


def test_prune_pdf_bytes_removes_blank_pages(tmp_path):
    """Verify that _prune_pdf_bytes strips blank pages while preserving content pages."""
    import fitz

    from pipeline.extractor.agentic_extractor import AgenticExtractor

    pdf_file = tmp_path / "sample_with_blanks.pdf"
    doc = fitz.open()

    # Page 1: Cover page
    p1 = doc.new_page()
    p1.insert_text((50, 50), "CAMBRIDGE EXAMINATION - MATHEMATICS")

    # Page 2: Question 1
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Question 1: Calculate x [2]")

    # Page 3: Official BLANK PAGE
    p3 = doc.new_page()
    p3.insert_text((50, 50), "BLANK PAGE\n© UCLES 2010")

    # Page 4: Purely empty page
    _ = doc.new_page()

    # Page 5: Question 2
    p5 = doc.new_page()
    p5.insert_text((50, 50), "Question 2: Solve y [3]")

    doc.save(str(pdf_file))
    doc.close()

    pruned_bytes = AgenticExtractor._prune_pdf_bytes(pdf_file)
    pruned_doc = fitz.open(stream=pruned_bytes, filetype="pdf")

    # Should keep pages 1, 2, and 5 (total 3 pages), pruning pages 3 and 4
    assert len(pruned_doc) == 3
    all_text = " ".join([page.get_text() for page in pruned_doc])
    assert "Question 1" in all_text
    assert "Question 2" in all_text
    assert "BLANK PAGE" not in all_text
    pruned_doc.close()


def test_classifier_agent_prompt_prefix_caching_structure():
    """Verify that static syllabus and instructions come before dynamic question data in ClassifierAgent prompt."""
    from pipeline.extractor.classifier_agent import ClassifierAgent
    from pipeline.extractor.paper_type_detector import PaperCategory

    agent = ClassifierAgent()
    syllabus = "UNIT 1: Number\n1.1 Primes and factors"
    records = [{"QuestionNumber": "Q1", "Question_text": "Find prime factors of 24."}]
    meta = {"year": 2023, "session": "jun", "variant": "11", "subject_code": "4024"}

    prompt = agent._build_prompt(records, syllabus, PaperCategory.ESSAY_QP, meta)

    # Syllabus and static instructions must appear BEFORE dynamic questions JSON and metadata
    syllabus_pos = prompt.find("OFFICIAL SYLLABUS:")
    instructions_pos = prompt.find("CLASSIFICATION INSTRUCTIONS:")
    dynamic_pos = prompt.find("DYNAMIC PAPER DATA & QUESTIONS TO CLASSIFY:")
    json_pos = prompt.find("PRE-EXTRACTED QUESTION JSON:")

    assert syllabus_pos != -1
    assert instructions_pos != -1
    assert dynamic_pos != -1
    assert json_pos != -1

    assert syllabus_pos < dynamic_pos
    assert instructions_pos < dynamic_pos
    assert dynamic_pos < json_pos


def test_gemini_429_backoff_and_retry(tmp_path):
    """Verify that AgenticExtractor retries on 429 rate limit with backoff and succeeds on subsequent attempt."""
    import json
    from unittest.mock import MagicMock, patch

    from pipeline.extractor.agentic_extractor import AgenticExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    extractor = AgenticExtractor()
    mock_client = MagicMock()
    mock_models = MagicMock()
    mock_client.models = mock_models
    extractor._get_gemini_client = MagicMock(return_value=mock_client)

    # Create dummy PDF
    import fitz
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Question 1: Test [1]")
    pdf_file = tmp_path / "test_429.pdf"
    doc.save(str(pdf_file))
    doc.close()

    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"questions": [{"QuestionNumber": "Q1", "Question_text": "Test", "Marks": 1}]})

    # Attempt 1: 429 Resource Exhausted, Attempt 2: Success
    mock_models.generate_content.side_effect = [
        Exception("429 Resource Exhausted: Please retry after 0.1s"),
        mock_resp,
    ]

    with patch("time.sleep", return_value=None):
        results = extractor._extract_with_gemini_full_pdf(
            pdf_file, PaperCategory.ESSAY_QP, {"session": "jun", "variant": "11", "year": "2023"}
        )

    assert mock_models.generate_content.call_count == 2
    assert len(results) == 1
    assert results[0]["QuestionNumber"] == "Q1"








