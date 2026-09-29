"""
tests/test_watermark_remover.py

Unit tests for Method 3 watermark removal module and scraper pipeline integration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf as fitz
import pytest

from pipeline.scraper.models import ScrapeTarget
from pipeline.scraper.scraper import PapaCambridgeScraper
from pipeline.scraper.watermark_remover import (
    clean_paper_watermark_if_eligible,
    extract_year_from_filename,
    remove_watermark_method3,
)


def test_extract_year_from_filename():
    """Verify year extraction across standard Cambridge naming patterns."""
    assert extract_year_from_filename("5054_s26_qp_11.pdf") == 2026
    assert extract_year_from_filename("4024_w23_ms_12.pdf") == 2023
    assert extract_year_from_filename("7707_m26_er.pdf") == 2026
    assert extract_year_from_filename("9709_s19_qp_32.pdf") == 2019
    assert extract_year_from_filename("sample_doc.pdf") is None


def test_clean_paper_watermark_eligibility(tmp_path):
    """Verify that only papers of year >= 2026 are cleaned."""
    # 2023 paper should be skipped
    p2023 = tmp_path / "5054_s23_qp_11.pdf"
    p2023.write_bytes(b"%PDF-1.4 dummy")
    assert clean_paper_watermark_if_eligible(p2023) is False

    # 2026 paper: eligible for watermark cleaning
    p2026 = tmp_path / "5054_s26_qp_11.pdf"
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Question 1")
    doc.save(str(p2026))
    doc.close()

    # Even though eligible, an already clean doc returns False without modifying
    assert clean_paper_watermark_if_eligible(p2026) is False


def test_remove_watermark_method3_synthetic(tmp_path):
    """Verify that Method 3 blanks watermark XObjects and strips inline payload."""
    pdf_path = tmp_path / "test_wm.pdf"

    # Create synthetic PDF with a Form XObject and an inline watermark stream
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1 Calculate the speed. [3]")

    # Append mock stream containing PapaCambridge watermark signature
    payload = (
        b"\nq\n0.055 0.227 0.361 rg\n/gRLs0-0 gs\nq\n"
        b"n 1.7 0 m 1.7 16.5 l f*\nQ\n"
        b"36 15 m 559.28 15 l S\n"
        b"BT (PapaCambridge) Tj ET\n"
        b"BT (\\267  papacambridge\\056com) Tj ET\nQ\n"
    )
    xref = page.get_contents()[0]
    existing = doc.xref_stream(xref)
    doc.update_stream(xref, existing + payload)
    doc.save(str(pdf_path))
    doc.close()

    # Verify watermark is initially present
    doc_check = fitz.open(pdf_path)
    assert b"/gRLs0-0" in doc_check[0].read_contents()
    doc_check.close()

    # Invoke Method 3
    cleaned = remove_watermark_method3(pdf_path)
    assert cleaned is True

    # Verify watermark payload is stripped
    doc_after = fitz.open(pdf_path)
    raw_after = doc_after[0].read_contents()
    assert b"/gRLs0-0" not in raw_after
    assert b"papacambridge" not in raw_after
    # Verify original question text is intact
    assert "Calculate the speed" in doc_after[0].get_text()
    doc_after.close()

    # Running again on already clean file is idempotent
    assert remove_watermark_method3(pdf_path) is False


def test_scraper_invokes_watermark_remover_for_2026(tmp_path):
    """Verify that scraper calls Method 3 watermark removal when year >= 2026."""
    target = ScrapeTarget(
        level="o-level",
        subject_name="physics",
        subject_code="5054",
        year=2026,
        session="jun",
        storage_dir=tmp_path,
    )
    scraper = PapaCambridgeScraper(target)

    # Mock PDF download
    dummy_pdf = tmp_path / "5054_s26_qp_11.pdf"
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Question 1")
    doc.save(str(dummy_pdf))
    doc.close()

    with patch.object(scraper, "_extract_pdf_urls_static", return_value=["http://example.com/5054_s26_qp_11.pdf"]):
        with patch.object(
            scraper.downloader,
            "download",
            return_value=(dummy_pdf, dummy_pdf.stat().st_size, "dummy_hash"),
        ):
            with patch("pipeline.scraper.scraper.remove_watermark_method3", return_value=True) as mock_wm:
                downloaded = scraper.scrape(force=True)
                assert len(downloaded) == 1
                assert mock_wm.called
                assert mock_wm.call_args[0][0] == target.storage_dir / "5054_s26_qp_11.pdf"


def test_scraper_skips_watermark_remover_for_pre_2026(tmp_path):
    """Verify that scraper does NOT call watermark removal for papers prior to 2026."""
    target = ScrapeTarget(
        level="o-level",
        subject_name="physics",
        subject_code="5054",
        year=2023,
        session="jun",
        storage_dir=tmp_path,
    )
    scraper = PapaCambridgeScraper(target)

    dummy_pdf = tmp_path / "5054_s23_qp_11.pdf"
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Question 1")
    doc.save(str(dummy_pdf))
    doc.close()

    with patch.object(scraper, "_extract_pdf_urls_static", return_value=["http://example.com/5054_s23_qp_11.pdf"]):
        with patch.object(
            scraper.downloader,
            "download",
            return_value=(dummy_pdf, dummy_pdf.stat().st_size, "dummy_hash"),
        ):
            with patch("pipeline.scraper.scraper.remove_watermark_method3") as mock_wm:
                downloaded = scraper.scrape(force=True)
                assert len(downloaded) == 1
                assert not mock_wm.called
