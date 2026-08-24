import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pipeline.scraper.models import ScrapeTarget, PaperFileInfo, PaperDocumentType
from pipeline.scraper.url_builder import URLBuilder
from pipeline.scraper.scraper import PapaCambridgeScraper
from pipeline.scraper.downloader import PDFDownloader, HTTPStatusError

def test_url_builder():
    # Test post-2018 target (may-june format)
    target_post2018 = ScrapeTarget(
        level="o-level",
        subject_name="accounting",
        subject_code="7707",
        year=2023,
        session="jun"
    )
    expected_page_url_2023 = "https://pastpapers.papacambridge.com/papers/caie/o-level-accounting-7707-2023-may-june"
    assert URLBuilder.build_page_url(target_post2018) == expected_page_url_2023

    # Test pre-2018 target (jun format)
    target_pre2018 = ScrapeTarget(
        level="o-level",
        subject_name="accounting",
        subject_code="7707",
        year=2017,
        session="jun"
    )
    expected_page_url_2017 = "https://pastpapers.papacambridge.com/papers/caie/o-level-accounting-7707-2017-jun"
    assert URLBuilder.build_page_url(target_pre2018) == expected_page_url_2017

    # Test building Direct PDF Download URL
    filename = "7707_s23_qp_12.pdf"
    expected_download_url = (
        "https://pastpapers.papacambridge.com/download_file.php?files="
        "https%3A%2F%2Fpastpapers.papacambridge.com%2Fdirectories%2FCAIE%2F"
        "CAIE-pastpapers%2Fupload%2F7707_s23_qp_12.pdf"
    )
    assert URLBuilder.build_download_url(filename) == expected_download_url

def test_get_candidate_urls():
    target = ScrapeTarget(
        level="o-level",
        subject_name="accounting",
        subject_code="7707",
        year=2023,
        session="jun"
    )
    candidates = URLBuilder.get_candidate_urls(target)
    assert candidates[0] == "https://pastpapers.papacambridge.com/papers/caie/o-level-accounting-7707-2023-may-june"


def test_pdf_classification():
    target = ScrapeTarget(
        level="o-level",
        subject_name="accounting",
        subject_code="7707",
        year=2023,
        session="jun"
    )
    scraper = PapaCambridgeScraper(target)

    # Standard Question Paper
    info1 = scraper.classify_pdf("7707_s23_qp_12.pdf", "http://test.com/7707_s23_qp_12.pdf")
    assert info1.document_type == PaperDocumentType.QUESTION_PAPER
    assert info1.paper_number == 1
    assert info1.variant == "12"

    # Standard Mark Scheme
    info2 = scraper.classify_pdf("7707_w23_ms_22.pdf", "http://test.com/7707_w23_ms_22.pdf")
    assert info2.document_type == PaperDocumentType.MARK_SCHEME
    assert info2.paper_number == 2
    assert info2.variant == "22"

    # Examiner Report
    info3 = scraper.classify_pdf("7707_s23_er.pdf", "http://test.com/7707_s23_er.pdf")
    assert info3.document_type == PaperDocumentType.EXAMINER_REPORT
    assert info3.paper_number is None
    assert info3.variant is None

    # Unknown or non-matching format fallback
    info4 = scraper.classify_pdf("7707_some_weird_filename_qp.pdf", "http://test.com/file.pdf")
    assert info4.document_type == PaperDocumentType.QUESTION_PAPER

def test_url_extraction_static():
    target = ScrapeTarget(
        level="o-level",
        subject_name="accounting",
        subject_code="7707",
        year=2023,
        session="jun"
    )
    scraper = PapaCambridgeScraper(target)

    # Mock HTML containing both download_file.php links and raw pdf links
    mock_html = """
    <html>
        <body>
            <!-- Valid link 1 -->
            <a href="download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/7707_s23_qp_12.pdf">Download</a>
            <!-- Direct PDF link -->
            <a href="/directories/CAIE/CAIE-pastpapers/upload/7707_s23_ms_12.pdf">Direct</a>
            <!-- Different subject (should be skipped) -->
            <a href="download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_s23_qp_11.pdf">Other</a>
        </body>
    </html>
    """
    
    urls = scraper._extract_pdf_urls_static(mock_html)
    assert len(urls) == 2
    assert any("7707_s23_qp_12.pdf" in u for u in urls)
    assert any("7707_s23_ms_12.pdf" in u for u in urls)
    assert not any("9706" in u for u in urls)

@patch("pipeline.scraper.downloader.requests.Session")
def test_downloader(mock_session_cls, tmp_path):
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    downloader = PDFDownloader()
    dest = tmp_path / "test.pdf"

    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Length": "10"}
    mock_response.iter_content.return_value = [b"0123456789"]
    mock_session.get.return_value = mock_response

    # First download (fresh)
    path, size, sha = downloader.download("http://example.com/test.pdf", dest, force=False)
    assert path == dest
    assert size == 10
    assert sha == "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882" # sha256 of "0123456789"
    assert dest.exists()

    # Second download (cached / skip network)
    mock_session.reset_mock()
    path, size, sha = downloader.download("http://example.com/test.pdf", dest, force=False)
    mock_session.get.assert_not_called()
    assert size == 10

    # Third download (forced)
    downloader.download("http://example.com/test.pdf", dest, force=True)
    mock_session.get.assert_called_once()
