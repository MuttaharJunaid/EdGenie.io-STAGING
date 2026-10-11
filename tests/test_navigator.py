import pytest
from unittest.mock import MagicMock
from pipeline.scraper.navigator import (
    PapaCambridgeNavigator,
    normalize_level_url,
    parse_session_element_text,
)
from pipeline.scraper.models import ScrapeTarget

def test_normalize_level_url():
    assert normalize_level_url("o-level") == "https://pastpapers.papacambridge.com/papers/caie/o-level?theme=lightTheme"
    assert normalize_level_url("O-Level") == "https://pastpapers.papacambridge.com/papers/caie/o-level?theme=lightTheme"
    assert normalize_level_url("as-and-a-level") == "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme"
    assert normalize_level_url("a-level") == "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme"
    assert normalize_level_url("as-and-a2-level") == "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme"


def test_parse_session_element_text():
    assert parse_session_element_text("2026-May-June") == (2026, "jun")
    assert parse_session_element_text("2025-Oct-Nov") == (2025, "nov")
    assert parse_session_element_text("2024-March") == (2024, "march")
    assert parse_session_element_text("2022-Feb-March") == (2022, "march")
    assert parse_session_element_text("2017 Nov") == (2017, "nov")
    assert parse_session_element_text("2015 Jun") == (2015, "jun")
    assert parse_session_element_text("Topical Past Paper") == (None, None)
    assert parse_session_element_text("Solved Past Papers") == (None, None)


def test_extract_download_links_from_session_page():
    sample_html = """
    <div>
        <a href="download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_w25_gt.pdf" download="" class="badge badge-info">
            Download File
        </a>
        <a href="download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_w25_ms_11.pdf" download="" class="badge badge-info">
            Download File
        </a>
        <a href="download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_s22_ms_11.pdf" download="" class="badge badge-info">
            Download File
        </a>
    </div>
    """
    nav = PapaCambridgeNavigator()
    # With prefix filter
    links = nav.extract_download_links_from_session_page(sample_html, expected_prefix="9706_w25")
    assert len(links) == 2
    assert "https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_w25_gt.pdf" in links
    assert "https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_w25_ms_11.pdf" in links
    assert "https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9706_s22_ms_11.pdf" not in links


def test_navigator_navigation_hierarchy():
    mock_session = MagicMock()
    
    # Mock Level 1 page
    level1_html = """
    <html>
        <body>
            <a href="papers/caie/as-and-a-level-accounting-9706">
                <span class="wraptext" id="Accounting - 9706\n"> Accounting - 9706\n</span>
            </a>
            <a href="papers/caie/as-and-a-level-chemistry-9701">
                <span class="wraptext" id="Chemistry - 9701\n"> Chemistry - 9701\n</span>
            </a>
        </body>
    </html>
    """
    
    # Mock Level 2 subject page
    level2_html = """
    <html>
        <body>
            <a href="papers/caie/as-and-a-level-accounting-9706-2026-may-june">
                <span class="wraptext" id="2026-May-June"> 2026-May-June</span>
            </a>
            <a href="papers/caie/as-and-a-level-accounting-9706-2025-oct-nov">
                <span class="wraptext" id="2025-Oct-Nov"> 2025-Oct-Nov</span>
            </a>
        </body>
    </html>
    """
    
    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "as-and-a-level?theme=lightTheme" in url:
            resp.text = level1_html
        elif "as-and-a-level-accounting-9706" in url:
            resp.text = level2_html
        else:
            resp.text = "<html></html>"
        return resp
        
    mock_session.get.side_effect = mock_get
    
    nav = PapaCambridgeNavigator(session=mock_session)
    
    target = ScrapeTarget(
        level="as-and-a-level",
        subject_name="accounting",
        subject_code="9706",
        year=2026,
        session="jun",
    )
    
    session_url = nav.resolve_session_page_url(target)
    assert session_url == "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level-accounting-9706-2026-may-june"
    
    # Verify cached subject URL avoids re-fetching Level 1 page
    target_nov = ScrapeTarget(
        level="as-and-a-level",
        subject_name="accounting",
        subject_code="9706",
        year=2025,
        session="nov",
    )
    session_url_nov = nav.resolve_session_page_url(target_nov)
    assert session_url_nov == "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level-accounting-9706-2025-oct-nov"
