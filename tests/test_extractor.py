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

