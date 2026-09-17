import logging
import re
from enum import Enum

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class PaperCategory(str, Enum):
    MCQ_QP = "mcq_qp"
    MCQ_MS = "mcq_ms"
    ESSAY_QP = "essay_qp"
    ESSAY_MS = "essay_ms"
    OTHER = "other"

class PaperTypeDetector:
    """Detects whether an exam paper or mark scheme is MCQ or Essay-based."""

    @staticmethod
    def detect_by_filename(filename: str) -> PaperCategory:
        """Heuristic check based on standard CAIE paper numbers (e.g. Paper 1 is MCQ)."""
        lower_name = filename.lower()

        # Check if it's a mark scheme
        is_ms = "_ms_" in lower_name or "ms" in lower_name
        is_qp = "_qp_" in lower_name or "qp" in lower_name

        # Extract paper code/number
        # E.g. 7707_s23_qp_12.pdf -> paper number is 1 (first digit of 12)
        match = re.search(r"_(qp|ms)_([0-9]{1,2})", lower_name)
        if match:
            paper_num_str = match.group(2)
            # Typically: paper 1 is MCQ, papers 2, 3, 4 are Essay/Theory
            # e.g., 11, 12, 13 are Paper 1 (MCQ)
            if paper_num_str.startswith("1"):
                return PaperCategory.MCQ_MS if is_ms else PaperCategory.MCQ_QP
            else:
                return PaperCategory.ESSAY_MS if is_ms else PaperCategory.ESSAY_QP

        # Fallback if paper number is not clear but keywords match
        if is_ms:
            # Check if MCQ keywords are in the name
            if "mcq" in lower_name or "paper_1" in lower_name or "p1" in lower_name:
                return PaperCategory.MCQ_MS
            return PaperCategory.ESSAY_MS

        if is_qp:
            if "mcq" in lower_name or "paper_1" in lower_name or "p1" in lower_name:
                return PaperCategory.MCQ_QP
            return PaperCategory.ESSAY_QP

        return PaperCategory.OTHER

    @staticmethod
    def detect_by_content(pdf_path) -> PaperCategory:
        """Scan PDF content to detect MCQ vs Essay clues (e.g. A, B, C, D choices)."""
        try:
            doc = fitz.open(pdf_path)
            # We only need to check the first 2-3 pages
            num_pages_to_check = min(len(doc), 3)
            text_sample = ""
            for i in range(num_pages_to_check):
                text_sample += doc[i].get_text()

            lower_text = text_sample.lower()

            # Mark Scheme Heuristics
            # If the filename or content shows it's a mark scheme
            is_ms = "mark scheme" in lower_text or "ms" in pdf_path.name.lower()

            if is_ms:
                # MCQ Mark Schemes usually consist of a table/list of question numbers (1-30 or 1-40)
                # and single letters: 1 A, 2 B, 3 C, etc.
                # Essay Mark Schemes contain detailed criteria, columns, figures, calculations.
                # If we see repeating patterns of single letter answers
                # e.g. "1 B", "2 A", "3 D"
                mcq_ms_hits = len(re.findall(r"\b\d{1,2}\s+[a-d]\b", lower_text))
                if mcq_ms_hits > 10:
                    return PaperCategory.MCQ_MS
                return PaperCategory.ESSAY_MS

            # Question Paper Heuristics
            # MCQ QPs usually contain instructions about soft pencil, four possible answers, etc.
            mcq_keywords = [
                "four possible answers",
                "multiple choice",
                "soft clean eraser",
                "soft pencil (type b or HB is recommended)",
                "read the instructions on the answer sheet"
            ]

            mcq_hits = sum(1 for kw in mcq_keywords if kw in lower_text)
            if mcq_hits >= 2:
                return PaperCategory.MCQ_QP

            return PaperCategory.ESSAY_QP
        except Exception as e:
            logger.error(f"Error scanning content of {pdf_path}: {e}")
            return PaperCategory.OTHER

    @classmethod
    def detect(cls, pdf_path) -> PaperCategory:
        """Detect paper type checking both content (primary) and filename fallback."""
        category = cls.detect_by_content(pdf_path)
        if category == PaperCategory.OTHER:
            # Let's cross-verify with filename
            filename_category = cls.detect_by_filename(pdf_path.name)
            if filename_category != PaperCategory.OTHER:
                return filename_category
        return category
