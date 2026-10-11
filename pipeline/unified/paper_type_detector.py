"""
pipeline/unified/paper_type_detector.py

Detects whether an exam paper or mark scheme is MCQ or Essay-based,
and parses CAIE filename metadata (subject, year, session, variant).
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import pymupdf as fitz

logger = logging.getLogger(__name__)


class PaperCategory(str, Enum):
    MCQ_QP = "mcq_qp"
    MCQ_MS = "mcq_ms"
    ESSAY_QP = "essay_qp"
    ESSAY_MS = "essay_ms"
    OTHER = "other"


def parse_caie_filename(filename: str) -> dict[str, Any]:
    """
    Parses standard CAIE exam PDF filenames:
    e.g. '7707_s25_qp_21.pdf', '5054_w23_ms_12.pdf', '0580_m22_qp_42.pdf'
    Returns dict with subject_code, year, session, variant, paper_number, doc_type.
    """
    name = Path(filename).name
    stem = Path(filename).stem.lower()

    subject_code = "unknown"
    year = 2026
    session = "jun"
    variant = "11"
    paper_number = 1
    doc_type = "qp"

    # Pattern: <subject>_<session_letter><2-digit year>_<type>_<variant>
    match = re.search(
        r"^([0-9]{4})_([a-z])([0-9]{2})_([a-z]{2})(?:_([0-9]{1,2}))?",
        stem,
    )
    if match:
        subject_code = match.group(1)
        sess_letter = match.group(2)
        if sess_letter == "s":
            session = "jun"
        elif sess_letter == "w":
            session = "nov"
        elif sess_letter == "m":
            session = "mar"
        else:
            session = sess_letter

        year = 2000 + int(match.group(3))
        doc_type = match.group(4)
        var_part = match.group(5)
        if var_part:
            variant = var_part
            if len(var_part) == 2 and var_part.isdigit():
                paper_number = int(var_part[0])
            elif len(var_part) == 1 and var_part.isdigit():
                paper_number = int(var_part)
    else:
        # Fallback numbers and keywords
        numbers = re.findall(r"\d+", stem)
        if len(numbers) >= 1:
            subject_code = numbers[0]
        if len(numbers) >= 2:
            yr_val = int(numbers[1])
            year = 2000 + yr_val if yr_val < 100 else yr_val

        if "may" in stem or "jun" in stem or "_s" in stem:
            session = "jun"
        elif "oct" in stem or "nov" in stem or "_w" in stem:
            session = "nov"

        if "_ms_" in stem or "ms" in stem:
            doc_type = "ms"
        elif "_qp_" in stem or "qp" in stem:
            doc_type = "qp"

        if len(numbers) >= 3 and len(numbers[-1]) == 2:
            variant = numbers[-1]
            paper_number = int(variant[0])

    return {
        "subject_code": subject_code,
        "year": year,
        "session": session,
        "variant": variant,
        "paper_number": paper_number,
        "doc_type": doc_type,
    }


class PaperTypeDetector:
    """Detects whether an exam paper or mark scheme is MCQ or Essay-based."""

    @staticmethod
    def detect_by_filename(filename: str) -> PaperCategory:
        """Heuristic check based on standard CAIE paper numbers (e.g. Paper 1 is MCQ)."""
        lower_name = filename.lower()

        is_ms = "_ms_" in lower_name or "ms" in lower_name
        is_qp = "_qp_" in lower_name or "qp" in lower_name

        match = re.search(r"_(qp|ms)_([0-9]{1,2})", lower_name)
        if match:
            paper_num_str = match.group(2)
            if paper_num_str.startswith("1"):
                return PaperCategory.MCQ_MS if is_ms else PaperCategory.MCQ_QP
            else:
                return PaperCategory.ESSAY_MS if is_ms else PaperCategory.ESSAY_QP

        if is_ms:
            if "mcq" in lower_name or "paper_1" in lower_name or "p1" in lower_name:
                return PaperCategory.MCQ_MS
            return PaperCategory.ESSAY_MS

        if is_qp:
            if "mcq" in lower_name or "paper_1" in lower_name or "p1" in lower_name:
                return PaperCategory.MCQ_QP
            return PaperCategory.ESSAY_QP

        return PaperCategory.OTHER

    @staticmethod
    def detect_by_content(pdf_path: str | Path) -> PaperCategory:
        """Scan PDF content to detect MCQ vs Essay clues (e.g. A, B, C, D choices)."""
        try:
            doc = fitz.open(str(pdf_path))
            num_pages_to_check = min(len(doc), 3)
            text_sample = ""
            for i in range(num_pages_to_check):
                text_sample += doc[i].get_text()

            lower_text = text_sample.lower()
            is_ms = "mark scheme" in lower_text or "ms" in Path(pdf_path).name.lower()

            if is_ms:
                mcq_ms_hits = len(re.findall(r"\b\d{1,2}\s+[a-d]\b", lower_text))
                if mcq_ms_hits > 10:
                    return PaperCategory.MCQ_MS
                return PaperCategory.ESSAY_MS

            mcq_keywords = [
                "four possible answers",
                "multiple choice",
                "soft clean eraser",
                "soft pencil (type b or hb is recommended)",
                "read the instructions on the answer sheet",
            ]

            mcq_hits = sum(1 for kw in mcq_keywords if kw in lower_text)
            if mcq_hits >= 2:
                return PaperCategory.MCQ_QP

            return PaperCategory.ESSAY_QP
        except Exception as e:
            logger.error(f"Error scanning content of {pdf_path}: {e}")
            return PaperCategory.OTHER

    @classmethod
    def detect(cls, pdf_path: str | Path) -> PaperCategory:
        """Detect paper type checking both content (primary) and filename fallback."""
        category = cls.detect_by_content(pdf_path)
        if category == PaperCategory.OTHER:
            filename_category = cls.detect_by_filename(Path(pdf_path).name)
            if filename_category != PaperCategory.OTHER:
                return filename_category
        return category
