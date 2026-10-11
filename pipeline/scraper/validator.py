"""
pipeline/scraper/validator.py

Post-scraping session directory validation for Cambridge exam papers.

Enforces CAIE directory and filename prefix consistency:
  - May/June:   <subject_code>_s<yy> (e.g., 5054_s15)
  - Oct/Nov:    <subject_code>_w<yy> (e.g., 5054_w15)
  - Feb/March:  <subject_code>_m<yy> (e.g., 0580_m15)

If any PDF in a session directory fails this prefix check, that specific
session directory (and only that session) is deleted to preserve data integrity.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from pipeline.config import settings

logger = logging.getLogger(__name__)


def get_expected_prefix(subject_code: str, year: int, session: str) -> str:
    """
    Returns the expected CAIE filename prefix for a subject, year, and session.
    Example: (5054, 2015, "jun") -> "5054_s15"
             (5054, 2015, "nov") -> "5054_w15"
             (0580, 2022, "march") -> "0580_m22"
    """
    s_clean = session.lower().strip()
    if s_clean in ["jun", "j", "s", "june", "may-june", "may-jun", "may"]:
        letter = "s"
    elif s_clean in ["nov", "n", "w", "november", "oct-nov", "oct-november", "oct"]:
        letter = "w"
    elif s_clean in ["march", "mar", "m", "feb-march", "february-march"]:
        letter = "m"
    else:
        letter = s_clean[0]

    yy = f"{year % 100:02d}"
    return f"{subject_code.lower()}_{letter}{yy}"


def validate_session_directory(
    session_dir: Path,
    expected_prefix: str,
) -> tuple[bool, list[str]]:
    """
    Validates all PDF files in session_dir against expected_prefix.
    Returns:
      (is_valid, list_of_invalid_filenames)
    If any PDF does not start with expected_prefix (case-insensitive),
    is_valid is False.
    """
    if not session_dir.exists() or not session_dir.is_dir():
        return True, []

    pdfs = list(session_dir.glob("*.pdf"))
    if not pdfs:
        return True, []

    prefix_lower = expected_prefix.lower()
    invalid_files = [
        p.name for p in pdfs if not p.name.lower().startswith(prefix_lower)
    ]
    return len(invalid_files) == 0, invalid_files


def validate_scraped_sessions(
    subject_code: str,
    years: list[int],
    sessions: list[str],
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Validates all session directories for the processed years and sessions.
    If a session directory contains any PDF with an invalid prefix, that
    specific session directory is deleted.

    Returns a summary dictionary with counts and details.
    """
    base_dir = (storage_dir or settings.storage_dir) / str(subject_code).strip()
    results: dict[str, Any] = {
        "validated_sessions": 0,
        "deleted_sessions": [],
        "details": [],
    }

    if not base_dir.exists():
        return results

    for y in sorted(years):
        for s in sessions:
            session_dir = base_dir / str(y) / s
            if not session_dir.exists() or not session_dir.is_dir():
                continue

            expected_prefix = get_expected_prefix(subject_code, y, s)
            is_valid, invalid_files = validate_session_directory(session_dir, expected_prefix)

            if not is_valid:
                logger.warning(
                    "Session validation failed for %s/%s/%s: %d file(s) did not match prefix '%s': %s",
                    subject_code,
                    y,
                    s,
                    len(invalid_files),
                    expected_prefix,
                    invalid_files,
                )
                logger.warning("Deleting invalid session directory: %s", session_dir)
                try:
                    shutil.rmtree(session_dir)
                    results["deleted_sessions"].append(str(session_dir))
                    results["details"].append({
                        "year": y,
                        "session": s,
                        "expected_prefix": expected_prefix,
                        "status": "deleted",
                        "invalid_files": invalid_files,
                    })

                    # Clean up parent year directory if now empty
                    year_dir = session_dir.parent
                    if year_dir.exists() and not any(year_dir.iterdir()):
                        year_dir.rmdir()
                except Exception as exc:
                    logger.error("Failed to delete directory %s: %s", session_dir, exc)
            else:
                results["validated_sessions"] += 1
                results["details"].append({
                    "year": y,
                    "session": s,
                    "expected_prefix": expected_prefix,
                    "status": "valid",
                    "invalid_files": [],
                })

    return results
