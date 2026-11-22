"""
tests/test_watermark_remover.py

Unit tests for watermark removal dispatching and scraping pipeline integration:
- Method 3 watermark removal for papers >= 2026.
- Legacy watermark & examiner margin removal for papers 2010–2018.
- Non-eligible papers (e.g. 2019, 2021) left untouched.
- Scraper integration updating size_bytes and sha256 in-place.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.scraper.models import ScrapeTarget
from pipeline.scraper.scraper import PapaCambridgeScraper
from pipeline.scraper.watermark_remover import (
    clean_paper_watermark_if_eligible,
    detect_and_remove_legacy_watermarks,
    extract_year_from_filename,
    remove_legacy_watermarks,
)


def test_extract_year_from_filename():
    assert extract_year_from_filename("5054_s15_qp_21.pdf") == 2015
    assert extract_year_from_filename("7707_w26_qp_11.pdf") == 2026
    assert extract_year_from_filename("0580_m22_ms_22.pdf") == 2022
    assert extract_year_from_filename("5054_s10_qp_23.pdf") == 2010
    assert extract_year_from_filename("5054_w18_qp_11.pdf") == 2018
    assert extract_year_from_filename("invalid_filename.pdf") is None


@patch("pipeline.scraper.watermark_remover.remove_watermark_method3")
@patch("pipeline.scraper.watermark_remover.remove_legacy_watermarks")
def test_clean_paper_watermark_if_eligible_dispatch(mock_legacy, mock_m3, tmp_path: Path):
    mock_m3.return_value = True
    mock_legacy.return_value = True

    pdf_2026 = tmp_path / "7707_s26_qp_21.pdf"
    pdf_2015 = tmp_path / "5054_s15_qp_11.pdf"
    pdf_2010 = tmp_path / "5054_s10_qp_23.pdf"
    pdf_2018 = tmp_path / "5054_w18_qp_12.pdf"
    pdf_2021 = tmp_path / "5054_s21_qp_11.pdf"
    pdf_2019 = tmp_path / "5054_s19_qp_11.pdf"

    # Year 2026 -> Method 3
    res_2026 = clean_paper_watermark_if_eligible(pdf_2026, year=2026)
    assert res_2026 is True
    mock_m3.assert_called_once_with(pdf_2026)
    mock_legacy.assert_not_called()

    mock_m3.reset_mock()
    mock_legacy.reset_mock()

    # Year 2015 -> Legacy cleanup
    res_2015 = clean_paper_watermark_if_eligible(pdf_2015, year=2015)
    assert res_2015 is True
    mock_legacy.assert_called_once_with(pdf_2015)
    mock_m3.assert_not_called()

    mock_m3.reset_mock()
    mock_legacy.reset_mock()

    # Boundaries: 2010 and 2018 -> Legacy cleanup
    assert clean_paper_watermark_if_eligible(pdf_2010, year=2010) is True
    mock_legacy.assert_called_with(pdf_2010)

    mock_legacy.reset_mock()
    assert clean_paper_watermark_if_eligible(pdf_2018, year=2018) is True
    mock_legacy.assert_called_with(pdf_2018)

    mock_legacy.reset_mock()

    # Excluded years: 2019, 2021 -> Not eligible
    assert clean_paper_watermark_if_eligible(pdf_2021, year=2021) is False
    assert clean_paper_watermark_if_eligible(pdf_2019, year=2019) is False
    mock_legacy.assert_not_called()
    mock_m3.assert_not_called()


@patch("pipeline.scraper.watermark_remover.remove_watermark_method3")
@patch("pipeline.scraper.watermark_remover.remove_legacy_watermarks")
def test_clean_paper_watermark_inferred_from_filename(mock_legacy, mock_m3, tmp_path: Path):
    mock_m3.return_value = True
    mock_legacy.return_value = True

    # Year omitted: inferred from filename
    pdf_legacy = tmp_path / "5054_s12_qp_11.pdf"
    assert clean_paper_watermark_if_eligible(pdf_legacy) is True
    mock_legacy.assert_called_once_with(pdf_legacy)

    pdf_m3 = tmp_path / "7707_w26_qp_22.pdf"
    assert clean_paper_watermark_if_eligible(pdf_m3) is True
    mock_m3.assert_called_once_with(pdf_m3)


def test_detect_and_remove_legacy_watermarks_nonexistent(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.pdf"
    stats = detect_and_remove_legacy_watermarks(nonexistent)
    assert stats["modified"] is False


@patch("pipeline.scraper.scraper.clean_paper_watermark_if_eligible")
def test_scraper_invokes_cleanup_and_updates_file_info(mock_cleanup, tmp_path: Path, monkeypatch):
    from pipeline.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    target = ScrapeTarget(
        level="o-level",
        subject_name="physics",
        subject_code="5054",
        year=2015,
        session="jun",
    )

    mock_downloader = MagicMock()
    fake_pdf = target.storage_dir / "5054_s15_qp_21.pdf"
    fake_pdf.parent.mkdir(parents=True, exist_ok=True)
    fake_pdf.write_bytes(b"%PDF-1.4 initial content")

    mock_downloader.download.return_value = (fake_pdf, 24, "fakehash123")
    mock_downloader._calculate_sha256.return_value = "newhash456"

    # Simulate cleanup modifying the file
    def fake_clean(path, year=None):
        path.write_bytes(b"%PDF-1.4 cleaned content with watermarks removed")
        return True

    mock_cleanup.side_effect = fake_clean

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html>fake page</html>"
    mock_session.get.return_value = mock_resp

    scraper = PapaCambridgeScraper(target=target, downloader=mock_downloader, session=mock_session)

    with patch.object(scraper, "_extract_pdf_urls_static", return_value=["https://example.com/5054_s15_qp_21.pdf"]):
        downloaded = scraper.scrape(dry_run=False)

    assert len(downloaded) == 1
    assert downloaded[0] == fake_pdf
    mock_cleanup.assert_called_once_with(fake_pdf, year=2015)

    # Check that scrape_manifest.json reflects the new sha256 and size
    manifest_path = target.storage_dir / "scrape_manifest.json"
    assert manifest_path.exists()
    import json
    data = json.loads(manifest_path.read_text())
    assert len(data["files"]) == 1
    file_info = data["files"][0]
    assert file_info["sha256"] == "newhash456"
    assert file_info["size_bytes"] == len(b"%PDF-1.4 cleaned content with watermarks removed")


def test_legacy_cleanup_preserves_spot_colors_and_diagram_labels(tmp_path: Path):
    """
    Verifies that legacy cleanup eliminates the right-hand margin line and examiner text
    while strictly preserving spot color operators (e.g. /Separation /Black 1 scn)
    and not corrupting diagram labels into white text (1 G).
    """
    import pymupdf as fitz

    test_pdf = tmp_path / "5054_w13_qp_42_synth.pdf"
    doc = fitz.open()
    doc.new_page()  # Page 1: cover
    p2 = doc.new_page()  # Page 2: question page

    # Draw right margin vertical line
    p2.draw_line(fitz.Point(527, 70), fitz.Point(527, 750), width=1.0)
    # Insert examiner header
    p2.insert_text(fitz.Point(530, 80), "For Examiner's Use")

    # Add custom stream with spot color text
    c_xref = p2.get_contents()[0]
    orig_stream = doc.xref_stream(c_xref)
    custom_stream = orig_stream + b"\n/CS1 cs 1 scn BT 11 0 0 11 300 700 Tm (diagram label) Tj ET\n"
    doc.update_stream(c_xref, custom_stream)

    doc.save(test_pdf)
    doc.close()

    stats = detect_and_remove_legacy_watermarks(test_pdf)
    assert stats["modified"] is True
    assert stats["examiner_margins_removed"] >= 1

    doc_cleaned = fitz.open(test_pdf)
    p2_cleaned = doc_cleaned[1]

    # Examiner text should be stripped/blanked
    assert "examiner" not in p2_cleaned.get_text().lower()

    # Content stream should keep 1 scn and NOT rewrite spot color to 1 G
    all_streams = b"".join(doc_cleaned.xref_stream(x) for x in p2_cleaned.get_contents())
    assert b"1 scn" in all_streams
    assert b"1 G[(diagram" not in all_streams and b"1 G (diagram" not in all_streams

    doc_cleaned.close()


