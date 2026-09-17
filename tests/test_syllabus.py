"""
tests/test_syllabus.py

Unit tests for Stage 0 syllabus crawler and builder.
"""

from unittest.mock import MagicMock, patch

from pipeline.syllabus.builder import SyllabusBuilder, SyllabusRow
from pipeline.syllabus.crawler import CambridgeSyllabusCrawler


def test_crawler_level_urls():
    crawler = CambridgeSyllabusCrawler()
    assert "cambridge-o-level" in crawler.get_subjects_url("o-level")
    assert "cambridge-international-as-and-a-levels" in crawler.get_subjects_url("a-level")
    assert "cambridge-igcse" in crawler.get_subjects_url("igcse")


def test_syllabus_save_csv(tmp_path):
    builder = SyllabusBuilder()
    rows = [
        SyllabusRow(unit="UNIT 1: Cells", topic="1.1 Cell structure and function", subtopic="Examine plant cells"),
        SyllabusRow(unit="UNIT 1: Cells", topic="1.1 Cell structure and function", subtopic="Identify cell wall"),
    ]
    csv_path = builder.save_csv(rows, "Biology", "5090", tmp_path)
    assert csv_path.exists()
    assert csv_path.name == "BIOLOGY_5090.csv"

    content = csv_path.read_text()
    assert "Unit,Topic,Subtopic" in content
    assert "UNIT 1: Cells" in content
    assert "Examine plant cells" in content


@patch("pipeline.syllabus.builder.genai.Client")
def test_syllabus_extract_from_pdf_mock(mock_client_cls, tmp_path):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.text = '{"rows": [{"unit": "UNIT 1: General Physics", "topic": "1.1 Length and time", "subtopic": "Use measuring cylinders"}]}'
    mock_client.models.generate_content.return_value = mock_response

    test_pdf = tmp_path / "test_syllabus.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 dummy syllabus content")

    builder = SyllabusBuilder()
    rows = builder.extract_from_pdf(test_pdf, "Physics", "5054", "o-level")

    assert len(rows) == 1
    assert rows[0].unit == "UNIT 1: General Physics"
    assert rows[0].topic == "1.1 Length and time"
    assert rows[0].subtopic == "Use measuring cylinders"
