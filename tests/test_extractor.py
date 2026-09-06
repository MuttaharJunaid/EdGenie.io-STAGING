import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pipeline.extractor.schemas import SyllabusUnit
from pipeline.extractor.paper_type_detector import PaperTypeDetector, PaperCategory
from pipeline.extractor.syllabus_loader import SyllabusLoader
from pipeline.extractor.question_parser import QuestionParser
from pipeline.extractor.text_extractor import ExtractedPage, TextBlock
from pipeline.extractor.topic_classifier import TopicClassifierAgent, TopicClassificationResult
from pipeline.extractor.difficulty_analyzer import DifficultyAnalyzerAgent, DifficultyAnalysisResult

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

def test_question_parser_mcq():
    # Mock text blocks representing a simple MCQ page
    blocks = [
        TextBlock(text="1", bbox=[10,100,200,120], font_name="bold", font_size=12, is_bold=True, is_italic=False, page_num=1),
        TextBlock(text="What is a credit note used for?", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="A", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="To correct an overcharge", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="B", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="To record cash sales", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="C", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="To request a payment", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="D", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="None of the above", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
    ]
    page = ExtractedPage(page_num=1, width=500, height=700, blocks=blocks, raw_text="")
    
    questions = QuestionParser.parse_mcq_qp([page], "7707", 2023, "jun", "12")
    assert len(questions) == 1
    assert questions[0]["QuestionNumber"] == "Jun/12/2023/Q1"
    assert questions[0]["Question_text"] == "What is a credit note used for?"
    assert questions[0]["Options"]["A"] == "To correct an overcharge"
    assert questions[0]["Options"]["B"] == "To record cash sales"
    assert questions[0]["Marks"] == 1

def test_question_parser_essay():
    # Mock text blocks representing a simple Essay page
    blocks = [
        TextBlock(text="1", bbox=[10,100,200,120], font_name="bold", font_size=12, is_bold=True, is_italic=False, page_num=1),
        TextBlock(text="Alex runs a wholesale business.", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="(a)", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="Define the term 'working capital'. [2]", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="(b)", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
        TextBlock(text="State two advantages of maintaining a general journal. [4]", bbox=[10,100,200,120], font_name="normal", font_size=10, is_bold=False, is_italic=False, page_num=1),
    ]
    page = ExtractedPage(page_num=1, width=500, height=700, blocks=blocks, raw_text="")
    
    questions = QuestionParser.parse_essay_qp([page], "7707", 2023, "jun", "22")
    assert len(questions) == 1
    assert questions[0]["QuestionNumber"] == "Jun/22/2023/Q1"
    assert "Alex runs a wholesale business." in questions[0]["Question_text"]
    assert len(questions[0]["Parts"]) == 2
    assert questions[0]["Parts"][0]["part_label"] == "(a)"
    assert "Define the term" in questions[0]["Parts"][0]["part_text"]
    assert questions[0]["Parts"][0]["marks"] == 2
    assert questions[0]["Parts"][1]["part_label"] == "(b)"
    assert questions[0]["Parts"][1]["marks"] == 4
    assert questions[0]["Marks"] == 6

@patch("pipeline.extractor.topic_classifier.ChatGoogleGenerativeAI")
def test_topic_classifier_agent(mock_gemini_cls):
    mock_llm = MagicMock()
    mock_gemini_cls.return_value = mock_llm
    
    mock_result = TopicClassificationResult(
        unit="1",
        topic="1.2",
        topic_name="Source Documents",
        sub_topics=["Credit Notes"],
        confidence=0.95,
        reasoning="Test reasoning"
    )
    
    structured_mock = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock
    structured_mock.invoke.return_value = mock_result
    structured_mock.return_value = mock_result
    
    # Mock the chain invocation if treated as raw RunnableSequence
    with patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=mock_result):
        # We enforce GOOGLE_API_KEY for testing the initialization
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key", "EDGENIE_LLM_PROVIDER": "google"}):
            agent = TopicClassifierAgent()
            res = agent.classify_question("Test question", "Test syllabus")
        
        assert res.unit == "1"
        assert res.topic == "1.2"
        assert res.topic_name == "Source Documents"
        assert "Credit Notes" in res.sub_topics


def test_openrouter_fallback(tmp_path):
    from pipeline.extractor.agentic_parser import AgenticPDFParser, PaperCategory

    pdf_file = tmp_path / "test_qp.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test content")

    parser = AgenticPDFParser(provider="google")

    with patch.object(parser, "_parse_with_gemini", side_effect=Exception("Gemini quota exceeded")), \
         patch.object(parser, "_parse_with_openrouter", return_value=[{"QuestionNumber": "Q1"}]) as mock_openrouter, \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_openrouter_key"}):

        res = parser.parse_pdf(pdf_file, PaperCategory.MCQ_QP, "syllabus text", {"year": 2023, "session": "jun"})

        assert len(res) == 1
        assert res[0]["QuestionNumber"] == "Q1"
        assert mock_openrouter.called


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


def test_pymupdf_extractor_imports():
    """Smoke test: PyMuPDFExtractor can be imported and instantiated."""
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    extractor = PyMuPDFExtractor()
    assert extractor is not None


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


def test_pymupdf_extractor_fmt_qnum():
    """_fmt_qnum produces the correct QuestionNumber format."""
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    ext = PyMuPDFExtractor()
    result = ext._fmt_qnum(3, {"session": "nov", "variant": "41", "year": 2018})
    assert result == "Nov/41/2018/Q3"


def test_pymupdf_extractor_real_mcq_ms():
    """Extract real MCQ MS PDF without errors."""
    from pathlib import Path
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_ms_11.pdf")
    if not pdf_path.exists():
        pytest.skip("Test storage file not present")

    ext = PyMuPDFExtractor()
    meta = {"session": "nov", "variant": "11", "year": 2010}
    results = ext.extract(pdf_path, PaperCategory.MCQ_MS, meta)

    assert len(results) == 40
    assert results[0]["AnswerNumber"] == "Nov/11/2010/Q1"
    assert results[0]["Answer"] in ["A", "B", "C", "D"]
    assert results[39]["AnswerNumber"] == "Nov/11/2010/Q40"


def test_pymupdf_extractor_real_essay_ms():
    """Extract real Essay MS PDF without errors."""
    from pathlib import Path
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_ms_21.pdf")
    if not pdf_path.exists():
        pytest.skip("Test storage file not present")

    ext = PyMuPDFExtractor()
    meta = {"session": "nov", "variant": "21", "year": 2010}
    results = ext.extract(pdf_path, PaperCategory.ESSAY_MS, meta)

    assert len(results) >= 8
    assert results[0]["AnswerNumber"] == "Nov/21/2010/Q1"
    assert "Parts" in results[0]
    assert len(results[0]["Parts"]) > 0


def test_pymupdf_extractor_real_mcq_qp():
    """Extract real MCQ QP PDF and verify option parsing and stem integrity."""
    from pathlib import Path
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    pdf_path = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_11.pdf")
    if not pdf_path.exists():
        pytest.skip("Test storage file not present")

    ext = PyMuPDFExtractor()
    meta = {"session": "jun", "variant": "11", "year": 2010, "subject_code": "5054"}
    results = ext.extract(pdf_path, PaperCategory.MCQ_QP, meta)

    assert len(results) == 40
    # Q1 options order & values
    q1 = results[0]
    assert q1["QuestionNumber"] == "Jun/11/2010/Q1"
    assert q1["Options"]["A"] == "mw"
    assert q1["Options"]["B"] == "mW"
    assert q1["Options"]["C"] == "Mw"
    assert q1["Options"]["D"] == "MW"

    # Q7 is a table option question: options must be empty, and stem preserved cleanly
    q7 = results[6]
    assert "passenger" in q7["Question_text"].lower()
    assert q7["Options"]["A"] == ""
    assert q7["Options"]["B"] == ""
    assert q7["Options"]["C"] == ""
    assert q7["Options"]["D"] == ""

    # Q8 numbers not stripped from option contents
    q8 = results[7]
    assert q8["Options"]["A"] == "1.5 kg"
    assert q8["Options"]["C"] == "30 kg"

    # Q33 simple text options are properly extracted
    q33 = results[32]
    assert "motor" in q33["Question_text"].lower()
    assert "to change the current direction" in q33["Options"]["A"]
    assert "horizontal position" in q33["Options"]["A"]
    assert "vertical position" in q33["Options"]["B"]


def test_pymupdf_extractor_real_essay_qp_answer_prompt_cleanup():
    """Extract real Essay QP PDF and verify answer prompts like 'change =' are omitted while prose with '=' is kept."""
    from pathlib import Path
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    pdf_path = Path("pipeline/storage/5054/2011/jun/5054_s11_qp_22.pdf")
    if not pdf_path.exists():
        pytest.skip("Test storage file not present")

    ext = PyMuPDFExtractor()
    meta = {"session": "jun", "variant": "22", "year": 2011, "subject_code": "5054"}
    results = ext.extract(pdf_path, PaperCategory.ESSAY_QP, meta)

    q11 = next((q for q in results if "Q11" in q["QuestionNumber"]), None)
    assert q11 is not None

    parts_by_label = {p["part_label"]: p for p in q11["Parts"]}
    assert "a" in parts_by_label
    assert "b" in parts_by_label
    assert "c" in parts_by_label

    part_a = parts_by_label["a"]
    subparts_a = {sp["sub_part_label"]: sp for sp in part_a["sub_parts"]}
    assert "a(i)" in subparts_a
    assert "a(ii)" in subparts_a
    assert "a(iii)" in subparts_a

    part_b = parts_by_label["b"]
    subparts_b = {sp["sub_part_label"]: sp for sp in part_b["sub_parts"]}
    assert "b(i)" in subparts_b
    assert "b(ii)" in subparts_b
    assert "b(iii)" in subparts_b

    # Preamble for part a is preserved in part_text
    assert "In an experiment to measure the specific heat capacity" in part_a["part_text"]

    # Part a(i): "change =" must be omitted, but "t = 0" and "t = 100 s" preserved
    p_ai = subparts_a["a(i)"]
    assert "change =" not in p_ai["sub_part_text"]
    assert "t = 0 and t = 100 s" in p_ai["sub_part_text"]
    assert "t = 100 s and t = 200 s" in p_ai["sub_part_text"]

    # Part b(ii): "energy =" omitted, marks and kinetic energy prose preserved
    p_bii = subparts_b["b(ii)"]
    assert "energy =" not in p_bii["sub_part_text"]
    assert p_bii["marks"] == 3
    assert "kinetic energy of the bullet" in p_bii["sub_part_text"]

    # Part b(iii): "water:" and "bullet:" answer prompts omitted
    p_biii = subparts_b["b(iii)"]
    assert "water:" not in p_biii["sub_part_text"]
    assert "bullet:" not in p_biii["sub_part_text"]
    assert "addition of energy has caused" in p_biii["sub_part_text"]


def test_nested_subparts_essay_qp_structure():
    """Verify that Essay QP builds true nested tree: Parts have sub_parts, and sub_parts can have sub_parts."""
    from pathlib import Path
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    from pipeline.extractor.paper_type_detector import PaperCategory

    pdf_path = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_22.pdf")
    if not pdf_path.exists():
        pytest.skip("Test storage file not present")

    ext = PyMuPDFExtractor()
    meta = {"session": "jun", "variant": "22", "year": 2010, "subject_code": "5054"}
    results = ext.extract(pdf_path, PaperCategory.ESSAY_QP, meta)

    # ── Test Q10 ──────────────────────────────────────────
    q10 = next((q for q in results if "Q10" in q["QuestionNumber"]), None)
    assert q10 is not None
    assert q10["Marks"] == 15

    parts_q10 = {p["part_label"]: p for p in q10["Parts"]}
    assert "a" in parts_q10
    assert "b" in parts_q10

    # Part a has no subparts: sub_parts list is empty, marks = 5
    part_a = parts_q10["a"]
    assert part_a["marks"] == 5
    assert len(part_a["sub_parts"]) == 0
    assert "Describe a method for measuring the speed of sound in air" in part_a["part_text"]

    # Part b has preamble text, marks = 10 (roll-up sum of children), and 5 sub-parts (b(i) to b(v))
    part_b = parts_q10["b"]
    assert part_b["marks"] == 10
    assert "Ultrasound is used in quality control to detect cracks" in part_b["part_text"]
    assert len(part_b["sub_parts"]) == 5

    sublabels_b = [sp["sub_part_label"] for sp in part_b["sub_parts"]]
    assert sublabels_b == ["b(i)", "b(ii)", "b(iii)", "b(iv)", "b(v)"]
    assert part_b["sub_parts"][0]["sub_part_text"] == "State what is meant by ultrasound."
    assert part_b["sub_parts"][0]["marks"] == 2

    # ── Test Q11 (Multi-level depth: parts -> subparts -> sub-subparts) ──
    q11 = next((q for q in results if "Q11" in q["QuestionNumber"]), None)
    assert q11 is not None

    parts_q11 = {p["part_label"]: p for p in q11["Parts"]}
    part_c = parts_q11["c"]
    assert "Fig. 11.2 is a circuit used to monitor changes in room temperature" in part_c["part_text"]
    assert len(part_c["sub_parts"]) == 2

    # c(i) has sub-subparts c(i)(1) and c(i)(2)
    ci = part_c["sub_parts"][0]
    assert ci["sub_part_label"] == "c(i)"
    assert len(ci["sub_parts"]) == 2
    assert ci["sub_parts"][0]["sub_part_label"] == "c(i)(1)"
    assert "the reading on the ammeter" in ci["sub_parts"][0]["sub_part_text"]
    assert ci["sub_parts"][0]["marks"] == 2
    assert ci["sub_parts"][1]["sub_part_label"] == "c(i)(2)"
    assert "the reading on the voltmeter" in ci["sub_parts"][1]["sub_part_text"]
    assert ci["sub_parts"][1]["marks"] == 2
    # c(i) rolled up marks = 2 + 2 = 4
    assert ci["marks"] == 4


def test_answer_prompt_cleanup_and_preservation():
    """Verify that trailing answer prompts (height =, V2 =, Answer (a) ...) are stripped while mid-sentence = are preserved."""
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    ext = PyMuPDFExtractor()

    # Standalone answer prompts should be stripped completely
    assert ext._clean_answer_prompts("height =") == ""
    assert ext._clean_answer_prompts("Answer (a) AEˆD =") == ""
    assert ext._clean_answer_prompts("V2 =") == ""
    assert ext._clean_answer_prompts("water:") == ""

    # Trailing prompts at end of sentences should be stripped
    assert ext._clean_answer_prompts("Calculate the height of the cylinder. height =") == "Calculate the height of the cylinder."
    assert ext._clean_answer_prompts("Calculate V2. V2 =") == "Calculate V2."
    assert ext._clean_answer_prompts("AEˆD, Answer (a) AEˆD =") == "AEˆD,"

    # Genuine mathematical expressions with = must NOT be stripped
    expr1 = "It is given that k = 2 lm 3 n . + Express m in terms of k, l and n."
    assert ext._clean_answer_prompts(expr1) == expr1

    expr2 = "Given that f(t) = t, find t."
    assert ext._clean_answer_prompts(expr2) == expr2

    expr3 = "Using congruent triangles, show that BD = AC."
    assert ext._clean_answer_prompts(expr3) == expr3

    expr4 = "t2 – 4t + 3 = 0."
    assert ext._clean_answer_prompts(expr4) == expr4


def test_cli_paper_type_override(tmp_path):
    """Verify that paper_type parameter overrides category on the Essay/MCQ axis for both QP and MS."""
    from pipeline.extractor.cli import extract_single
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    # Test with real 4024 paper (normally detected as ESSAY_QP)
    pdf_path = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
    if pdf_path.exists():
        with patch.object(AgenticExtractor, "extract", return_value=[{"QuestionNumber": "Q1", "Question_text": "test", "Marks": 1}]):
            # Override to essay
            res_essay = extract_single(pdf_path, force=True, paper_type="essay")
            assert res_essay is not None
            assert res_essay[2] == PaperCategory.ESSAY_QP

            # Override to mcq
            res_mcq = extract_single(pdf_path, force=True, paper_type="mcq")
            assert res_mcq is not None
            assert res_mcq[2] == PaperCategory.MCQ_QP


def test_non_zero_marks_guarantee():
    """Verify that no question, part, or sub-part has marks: 0 in single-part or multi-part questions."""
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    ext = PyMuPDFExtractor()

    # 1. Test 4024_s10_qp_11 (Maths paper with single-part questions Q4, Q7, Q11, Q14)
    pdf_4024 = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
    if pdf_4024.exists():
        res_4024 = ext.extract(pdf_4024, PaperCategory.ESSAY_QP, {"session": "Jun", "variant": "11", "year": "2010"})
        for q in res_4024:
            assert q["Marks"] > 0, f"Question {q['QuestionNumber']} has 0 marks"
            for p in q.get("Parts", []):
                assert p["marks"] > 0, f"Part {p.get('part_label')} in {q['QuestionNumber']} has 0 marks"
                for sp in p.get("sub_parts", []):
                    assert sp["marks"] > 0, f"Subpart {sp.get('sub_part_label')} in {q['QuestionNumber']} has 0 marks"

        q_map_4024 = {int(q["QuestionNumber"].split("/Q")[-1]): q for q in res_4024}
        assert q_map_4024[4]["Marks"] == 2
        assert q_map_4024[7]["Marks"] == 2
        assert q_map_4024[11]["Marks"] == 3
        assert q_map_4024[14]["Marks"] == 3

    # 2. Test 5054_s10_qp_22 (Physics essay paper with figure band marks)
    pdf_5054 = Path("pipeline/storage/5054/2010/jun/5054_s10_qp_22.pdf")
    if pdf_5054.exists():
        res_5054 = ext.extract(pdf_5054, PaperCategory.ESSAY_QP, {"session": "Jun", "variant": "22", "year": "2010"})
        for q in res_5054:
            assert q["Marks"] > 0, f"Question {q['QuestionNumber']} has 0 marks"
            for p in q.get("Parts", []):
                assert p["marks"] > 0, f"Part {p.get('part_label')} in {q['QuestionNumber']} has 0 marks"
                for sp in p.get("sub_parts", []):
                    assert sp["marks"] > 0, f"Subpart {sp.get('sub_part_label')} in {q['QuestionNumber']} has 0 marks"

        q_map_5054 = {int(q["QuestionNumber"].split("/Q")[-1]): q for q in res_5054}
        # Q7 part b (line near footer) has 2 marks
        q7_parts = {p["part_label"]: p for p in q_map_5054[7]["Parts"]}
        assert q7_parts["b"]["marks"] == 2
        assert q_map_5054[7]["Marks"] == 7

        # Q9 e(iii) (diagram band) has 3 marks
        q9_parts = {p["part_label"]: p for p in q_map_5054[9]["Parts"]}
        q9_e_sub = {sp["sub_part_label"]: sp for sp in q9_parts["e"]["sub_parts"]}
        assert q9_e_sub["e(iii)"]["marks"] == 3
        assert q9_parts["e"]["marks"] == 8
        assert q_map_5054[9]["Marks"] == 15

        # Q11 b(ii) (diagram band) has 2 marks
        q11_parts = {p["part_label"]: p for p in q_map_5054[11]["Parts"]}
        q11_b_sub = {sp["sub_part_label"]: sp for sp in q11_parts["b"]["sub_parts"]}
        assert q11_b_sub["b(ii)"]["marks"] == 2
        assert q_map_5054[11]["Marks"] == 15


def test_math_signs_and_exponents_normalization():
    """Verify clean_math_text and layout-based math normalization."""
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    ext = PyMuPDFExtractor()

    # 1. Powers of 10 and scientific notation
    assert ext.clean_math_text("5.98 × 1021") == "5.98 × 10²¹"
    assert ext.clean_math_text("3.0 * 108 m/s") == "3.0 × 10⁸ m/s"
    assert ext.clean_math_text("10-4") == "10⁻⁴"
    assert ext.clean_math_text("10–3") == "10⁻³"
    # Plain numbers containing 10 or 100 must NOT be corrupted
    assert ext.clean_math_text("0 ≤ t < 100 120") == "0 ≤ t < 100 120"
    assert ext.clean_math_text("80 100 120") == "80 100 120"
    assert ext.clean_math_text("1000") == "1000"

    # 2. Units with powers
    assert ext.clean_math_text("500 cm3") == "500 cm³"
    assert ext.clean_math_text("10 m2") == "10 m²"
    assert ext.clean_math_text("volume in km3") == "volume in km³"

    # 3. Algebraic powers
    assert ext.clean_math_text("4x2 – 25y2") == "4x² – 25y²"
    assert ext.clean_math_text("t2 – 4t + 3 = 0") == "t² – 4t + 3 = 0"
    assert ext.clean_math_text("n3 > 1") == "n³ > 1"

    # 4. Collapse spaces before superscripts
    assert ext.clean_math_text("5t ² metres") == "5t² metres"

    # 5. Integration check on real PDF 4024_s10_qp_11
    pdf_4024 = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
    if pdf_4024.exists():
        res = ext.extract(pdf_4024, PaperCategory.ESSAY_QP, {"session": "Jun", "variant": "11", "year": "2010"})
        q_map = {int(q["QuestionNumber"].split("/Q")[-1]): q for q in res}

        # Q6: inequality sign < instead of \x02
        q6_text = q_map[6]["Parts"][0]["part_text"]
        assert "<" in q6_text
        assert "\x02" not in q6_text

        # Q7: inequality sign > instead of \x03, and n³, n²
        q7_stem = q_map[7]["Question_text"]
        assert "n > 1" in q7_stem
        assert "n³ > 1" in q7_stem
        assert "n²" in q7_stem
        assert "\x03" not in q7_stem

        # Q12: inequality sign ≤ instead of \x04
        q12_b = q_map[12]["Parts"][1]["part_text"]
        assert "y ≤ x + 5" in q12_b
        assert "\x04" not in q12_b

        # Q14: 0 ≤ t < 10 and 100 ≤ t < 120
        q14_stem = q_map[14]["Question_text"]
        assert "0 ≤ t < 10" in q14_stem
        assert "100 ≤ t < 120" in q14_stem
        assert "\x05" not in q14_stem

        # Q17: y is inversely proportional to x²
        assert "x²" in q_map[17]["Question_text"]

        # Q19: Earth mass 5.98 × 10²¹ and km³
        q19_stem = q_map[19]["Question_text"]
        assert "5.98 × 10²¹" in q19_stem
        assert "1.08 × 10¹²" in q19_stem
        assert "km³" in q19_stem


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

    # Page 2 continues Q2 (part b with sub-parts b(i) and b(ii)) and introduces Q3
    page2 = [
        {
            "QuestionNumber": "Q2",
            "Question_text": "",
            "Marks": 3,
            "Parts": [
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

    # Q2 should have both part a and part b
    assert len(q2["Parts"]) == 2
    assert q2["Parts"][0]["part_label"] == "a"
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


def test_agentic_extractor_qwen_fallback():
    """Verify that when Gemini extraction fails, AgenticExtractor falls back to Qwen 2.5 VL."""
    import fitz
    from pipeline.extractor.agentic_extractor import AgenticExtractor

    extractor = AgenticExtractor()
    mock_doc = fitz.open()
    mock_doc.new_page()

    expected_qwen_result = [
        {
            "QuestionNumber": "Q1",
            "Question_text": "Extracted via Qwen 2.5 VL fallback.",
            "Marks": 2,
            "Parts": [{"part_label": "a", "part_text": "Fallback part.", "marks": 2}],
        }
    ]

    with patch.object(extractor, "_extract_page_with_gemini", side_effect=Exception("Gemini quota exceeded")), \
         patch.object(extractor, "_extract_page_with_qwen", return_value=expected_qwen_result) as mock_qwen:

        result = extractor._extract_page(
            doc=mock_doc,
            pno=0,
            total_pages=1,
            category=PaperCategory.ESSAY_QP,
            meta={"session": "Jun", "variant": "11", "year": "2010"},
        )

        assert mock_qwen.called
        assert len(result) == 1
        assert result[0]["QuestionNumber"] == "Q1"
        assert result[0]["Question_text"] == "Extracted via Qwen 2.5 VL fallback."


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
    from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
    ext = PyMuPDFExtractor()
    assert ext.clean_math_text(r"f(x) \equiv 2x") == "f(x) ≡ 2x"
    assert ext.clean_math_text(r"\triangle ABC \cong \triangle DEF") == r"\triangle ABC ≅ \triangle DEF"


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


def test_agentic_extractor_gemini_full_pdf_primary_and_qwen_fallback(tmp_path):
    """Verify that AgenticExtractor.extract uses Gemini full-PDF as primary and falls back to Qwen page-by-page."""
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

    # 2. Fallback: Gemini fails, falls back to Qwen page-by-page
    qwen_records = [
        {
            "QuestionNumber": "Q1",
            "Question_text": "Extracted via Qwen fallback.",
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
         patch.object(extractor, "_extract_page_with_qwen", return_value=qwen_records) as mock_qwen:

        fallback_results = extractor.extract(pdf_file, PaperCategory.ESSAY_QP, meta)
        assert mock_qwen.called
        assert len(fallback_results) == 1
        assert fallback_results[0]["Question_text"] == "Extracted via Qwen fallback."
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







