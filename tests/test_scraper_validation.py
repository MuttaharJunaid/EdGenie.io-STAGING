"""
tests/test_scraper_validation.py

Unit tests for post-scraping session directory & prefix validation:
- Verifies expected prefixes (e.g., subject_s15 for jun 2015, subject_w15 for nov 2015).
- Deletes only session directories that contain mismatched file prefixes.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipeline.scraper.cli import app
from pipeline.scraper.validator import (
    get_expected_prefix,
    validate_scraped_sessions,
    validate_session_directory,
)

runner = CliRunner()


@pytest.mark.parametrize(
    "subject,year,session,expected",
    [
        ("5054", 2015, "jun", "5054_s15"),
        ("5054", 2015, "nov", "5054_w15"),
        ("7707", 2020, "nov", "7707_w20"),
        ("7707", 2025, "jun", "7707_s25"),
        ("0580", 2022, "march", "0580_m22"),
        ("2281", 2018, "may-june", "2281_s18"),
        ("2281", 2018, "oct-nov", "2281_w18"),
    ],
)
def test_get_expected_prefix(subject: str, year: int, session: str, expected: str):
    prefix = get_expected_prefix(subject, year, session)
    assert prefix == expected


def test_validate_session_directory_clean(tmp_path: Path):
    session_dir = tmp_path / "5054" / "2015" / "jun"
    session_dir.mkdir(parents=True)

    # Valid PDFs
    (session_dir / "5054_s15_qp_11.pdf").touch()
    (session_dir / "5054_s15_ms_11.pdf").touch()
    (session_dir / "5054_s15_er.pdf").touch()
    # Non-PDF manifest file should be ignored during check
    (session_dir / "scrape_manifest.json").touch()

    is_valid, invalid = validate_session_directory(session_dir, "5054_s15")
    assert is_valid is True
    assert invalid == []


def test_validate_session_directory_mismatched(tmp_path: Path):
    session_dir = tmp_path / "7115" / "2016" / "nov"
    session_dir.mkdir(parents=True)

    # Valid nov files
    (session_dir / "7115_w16_qp_11.pdf").touch()
    (session_dir / "7115_w16_ms_11.pdf").touch()
    # Leaked summer file
    (session_dir / "7115_s16_er.pdf").touch()

    is_valid, invalid = validate_session_directory(session_dir, "7115_w16")
    assert is_valid is False
    assert invalid == ["7115_s16_er.pdf"]


def test_validate_scraped_sessions_deletes_only_invalid_dir(tmp_path: Path):
    storage = tmp_path / "storage"
    subj = "5054"

    # Create 2015 jun (VALID)
    jun_dir = storage / subj / "2015" / "jun"
    jun_dir.mkdir(parents=True)
    (jun_dir / "5054_s15_qp_11.pdf").touch()
    (jun_dir / "5054_s15_ms_11.pdf").touch()

    # Create 2015 nov (INVALID: has 5054_s15 file in nov directory)
    nov_dir = storage / subj / "2015" / "nov"
    nov_dir.mkdir(parents=True)
    (nov_dir / "5054_w15_qp_11.pdf").touch()
    (nov_dir / "5054_s15_er.pdf").touch()  # Mismatch!

    res = validate_scraped_sessions(
        subject_code=subj,
        years=[2015],
        sessions=["jun", "nov"],
        storage_dir=storage,
    )

    # jun must still exist
    assert jun_dir.exists()
    assert (jun_dir / "5054_s15_qp_11.pdf").exists()

    # nov directory must have been deleted
    assert not nov_dir.exists()

    # Results report
    assert res["validated_sessions"] == 1
    assert str(nov_dir) in res["deleted_sessions"]


def test_cli_validate_command(tmp_path: Path):
    storage = tmp_path / "storage"
    subj = "5054"

    jun_dir = storage / subj / "2015" / "jun"
    jun_dir.mkdir(parents=True)
    (jun_dir / "5054_s15_qp_11.pdf").touch()

    res = runner.invoke(
        app,
        [
            "validate",
            "--code", "5054",
            "--year", "2015",
            "--session", "jun",
        ],
    )
    assert res.exit_code == 0


def test_url_builder_candidate_urls_2024_nov_repeated_slug():
    from pipeline.scraper.models import ScrapeTarget
    from pipeline.scraper.url_builder import URLBuilder

    target = ScrapeTarget(
        level="o-level",
        subject_name="islamiyat",
        subject_code="2058",
        year=2024,
        session="nov",
    )
    urls = URLBuilder.get_candidate_urls(target)
    # Check that repeated-slug without hyphen before year is present
    expected_anomaly = "https://pastpapers.papacambridge.com/papers/caie/o-level-islamiyat-2058-islamiyat-20582024-oct-nov"
    assert expected_anomaly in urls


def test_url_builder_candidate_urls_2020_jun_session_before_year():
    from pipeline.scraper.models import ScrapeTarget
    from pipeline.scraper.url_builder import URLBuilder

    target = ScrapeTarget(
        level="o-level",
        subject_name="islamiyat",
        subject_code="2058",
        year=2020,
        session="jun",
    )
    urls = URLBuilder.get_candidate_urls(target)
    # Check that session before year (may-june-2020) is present
    expected_inverted = "https://pastpapers.papacambridge.com/papers/caie/o-level-islamiyat-2058-may-june-2020"
    assert expected_inverted in urls


def test_url_builder_candidate_urls_4024_maths_alias():
    from pipeline.scraper.models import ScrapeTarget
    from pipeline.scraper.url_builder import URLBuilder

    target = ScrapeTarget(
        level="o-level",
        subject_name="mathematics",
        subject_code="4024",
        year=2024,
        session="nov",
    )
    urls = URLBuilder.get_candidate_urls(target)
    expected_4024 = "https://pastpapers.papacambridge.com/papers/caie/o-level-mathematics-d-calculator-version-4024-mathematics-d-40242024-oct-nov"
    assert expected_4024 in urls


def test_url_builder_candidate_urls_nov_duplicate_year_session():
    from pipeline.scraper.models import ScrapeTarget
    from pipeline.scraper.url_builder import URLBuilder

    target = ScrapeTarget(
        level="o-level",
        subject_name="chemistry",
        subject_code="5070",
        year=2025,
        session="nov",
    )
    urls = URLBuilder.get_candidate_urls(target)
    expected_url = "https://pastpapers.papacambridge.com/papers/caie/o-level-chemistry-5070-2025-oct-nov2025-oct-nov"
    assert expected_url in urls


