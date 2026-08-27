"""
pipeline/extractor/pymupdf_extractor.py

Duplicate Stage 2 — layout-based text extractor.

Uses the same PyMuPDF primitives as Stage 3 (CambridgeVisualExtractor) to
pull question prose directly from the PDF without any LLM call.  The
output is a schema-skeleton list — all classification fields (Unit, Topic,
Difficulty, …) are left empty; a subsequent ClassifierAgent pass fills them.

Supported paper categories
--------------------------
  MCQ_QP   — stem + A/B/C/D options per question
  MCQ_MS   — answer-key letter per question (layout scan)
  ESSAY_QP — stem + (a)/(b)/… sub-parts + [N] marks
  ESSAY_MS — raw answer text + sub-part answers per question

What is excluded (mirrors cambridge_text_extractor.py):
  • Header/footer boilerplate, paper codes, "Turn over", "BLANK PAGE" pages
  • "For Examiner's Use" sidebar (x0 > CONTENT_MAX_X)
  • Text inside figure caption bands (same logic as Stage 3)
  • Fig. N.M caption lines themselves
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List

import pymupdf as fitz

from pipeline.visuals.extractor import (
    CambridgeVisualExtractor,
    DEFAULT_CAPTION_PATTERN,
    HEADER_Y,
    CONTENT_MAX_X,
    MCQ_Q_PATTERN,
    MCQ_Q_SOLO,
    MCQ_ANSWER_LABELS,
)
from pipeline.extractor.paper_type_detector import PaperCategory

logger = logging.getLogger(__name__)

FOOTER_Y = 780

BOILERPLATE_PATTERNS = [
    re.compile(r"^\W?\s*UCLES", re.IGNORECASE),
    re.compile(r"^\d{3,4}/\d{1,2}/[A-Z]/[A-Z]/\d{2}$"),
    re.compile(r"^\[?Turn over\]?", re.IGNORECASE),
    re.compile(r"^BLANK PAGE$", re.IGNORECASE),
    re.compile(r"^This document consists", re.IGNORECASE),
    re.compile(r"^Permission to reproduce", re.IGNORECASE),
    re.compile(r"^University of Cambridge", re.IGNORECASE),
    re.compile(r"^Cambridge Assessment", re.IGNORECASE),
    re.compile(r"^DC\s*\(", re.IGNORECASE),
    re.compile(r"^[\*\s\d]{6,}$"),
]

DOT_LEADER = re.compile(r"\.{4,}")
MARKS_PATTERN = re.compile(r"\[(\d+)\]")
PART_LABEL = re.compile(r"^\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\)\s*", re.IGNORECASE)
OPTION_INLINE = re.compile(r"^([ABCD])\s+\S")   # "A Some text"
OPTION_SOLO   = re.compile(r"^([ABCD])$")         # bare "A"


class PyMuPDFExtractor:
    """
    Layout-based text extractor for the duplicate Stage 2 strategy.
    Call extract(pdf_path, category, meta_context) → list of schema-skeleton dicts.
    """

    def __init__(self):
        self._cve = CambridgeVisualExtractor()

    # ── Boilerplate filter ─────────────────────────────────────────────────

    def _is_boilerplate(self, line) -> bool:
        if line.bbox.y0 <= HEADER_Y or line.bbox.y0 >= FOOTER_Y:
            return True
        if line.bbox.x0 > CONTENT_MAX_X:
            return True
        t = line.text.strip()
        return any(p.match(t) for p in BOILERPLATE_PATTERNS)

    # ── Shared: build per-page data ────────────────────────────────────────

    def _build_page_data(self, doc: fitz.Document, mcq: bool = False) -> List:
        page_data = []
        for pno in range(len(doc)):
            page = doc[pno]
            lines = self._cve.get_lines(page)
            if any(ln.text.strip().upper() == "BLANK PAGE" for ln in lines):
                page_data.append(None)
                continue
            if mcq:
                raw_anchors = self._cve.get_mcq_question_anchors(lines)
            else:
                raw_anchors = self._cve.get_question_anchors(lines)
            anchors = [(a[0], a[1]) for a in raw_anchors]
            captions = self._cve.get_captions(lines)
            bands = []
            for cap in captions:
                upper = self._cve.find_upper_bound(cap, lines)
                lower = cap["bbox"].y1
                bands.append((upper, lower))
            page_data.append({"lines": lines, "anchors": anchors, "bands": bands})
        return page_data

    # ── QuestionNumber formatter ───────────────────────────────────────────

    def _fmt_qnum(self, qnum: int, meta: Dict[str, Any]) -> str:
        session = str(meta.get("session", "jun")).capitalize()
        variant = str(meta.get("variant", "11"))
        year = str(meta.get("year", 2023))
        return f"{session}/{variant}/{year}/Q{qnum}"

    # ── Public entry point ─────────────────────────────────────────────────

    def extract(self, pdf_path: Path, category: PaperCategory, meta_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return schema-skeleton records (classification fields empty)."""
        pdf_path = Path(pdf_path)
        if category == PaperCategory.MCQ_QP:
            return self._extract_mcq_qp(pdf_path, meta_context)
        elif category == PaperCategory.MCQ_MS:
            return self._extract_mcq_ms(pdf_path, meta_context)
        elif category == PaperCategory.ESSAY_QP:
            return self._extract_essay_qp(pdf_path, meta_context)
        elif category == PaperCategory.ESSAY_MS:
            return self._extract_essay_ms(pdf_path, meta_context)
        else:
            raise ValueError(f"PyMuPDFExtractor: unsupported category {category}")

    # ── Essay QP ──────────────────────────────────────────────────────────

    def _extract_essay_qp(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = fitz.open(str(pdf_path))
        page_data = self._build_page_data(doc, mcq=False)

        raw: Dict[int, List[str]] = {}
        order: List[int] = []
        current_q = None

        for pd in page_data:
            if pd is None:
                continue
            for ln in pd["lines"]:
                if self._is_boilerplate(ln):
                    continue
                anchor_here = next(
                    (qn for qn, y0 in pd["anchors"] if abs(y0 - ln.bbox.y0) < 0.6), None
                )
                if anchor_here is not None:
                    current_q = anchor_here
                    if ln.text.strip().isdigit():
                        continue
                if current_q is None:
                    continue
                if DEFAULT_CAPTION_PATTERN.match(ln.text.strip()):
                    continue
                if any(u - 1 <= ln.bbox.y0 and ln.bbox.y1 <= lo + 1 for (u, lo) in pd["bands"]):
                    continue
                text = DOT_LEADER.sub("", ln.text).strip()
                if text:
                    raw.setdefault(current_q, []).append(text)
                    if current_q not in order:
                        order.append(current_q)

        results = []
        for qnum in order:
            stem_lines: List[str] = []
            parts: List[Dict] = []
            current_part: Dict = None
            total_marks = 0

            for line in raw[qnum]:
                part_m = PART_LABEL.match(line)
                if part_m:
                    if current_part:
                        parts.append(current_part)
                    marks_m = MARKS_PATTERN.search(line)
                    m_val = int(marks_m.group(1)) if marks_m else 0
                    total_marks += m_val
                    part_text = PART_LABEL.sub("", line).strip()
                    part_text = MARKS_PATTERN.sub("", part_text).strip()
                    current_part = {
                        "part_label": f"({part_m.group(1)})",
                        "part_text": part_text,
                        "marks": m_val,
                        "image": None,
                    }
                else:
                    marks_m = MARKS_PATTERN.search(line)
                    if marks_m and current_part is not None:
                        m_val = int(marks_m.group(1))
                        total_marks += m_val
                        current_part["marks"] = m_val
                        tail = MARKS_PATTERN.sub("", line).strip()
                        if tail:
                            current_part["part_text"] = (current_part["part_text"] + " " + tail).strip()
                    elif current_part is not None:
                        current_part["part_text"] = (current_part["part_text"] + " " + line).strip()
                    else:
                        stem_lines.append(line)

            if current_part:
                parts.append(current_part)

            results.append({
                "QuestionNumber": self._fmt_qnum(qnum, meta),
                "Question_text": " ".join(stem_lines),
                "Unit": "",
                "Topic": "",
                "Topic_name": "",
                "Sub_topic": [],
                "Difficulty_score": 0,
                "Difficulty_label": "",
                "Command_words": [],
                "Marks": total_marks,
                "image": None,
                "Parts": parts,
            })

        return results

    # ── Essay MS ──────────────────────────────────────────────────────────

    def _extract_essay_ms(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = fitz.open(str(pdf_path))
        page_data = self._build_page_data(doc, mcq=False)

        raw: Dict[int, List[str]] = {}
        order: List[int] = []
        current_q = None

        for pd in page_data:
            if pd is None:
                continue
            for ln in pd["lines"]:
                if self._is_boilerplate(ln):
                    continue
                anchor_here = next(
                    (qn for qn, y0 in pd["anchors"] if abs(y0 - ln.bbox.y0) < 0.6), None
                )
                if anchor_here is not None:
                    current_q = anchor_here
                    if ln.text.strip().isdigit():
                        continue
                if current_q is None:
                    continue
                if DEFAULT_CAPTION_PATTERN.match(ln.text.strip()):
                    continue
                text = DOT_LEADER.sub("", ln.text).strip()
                if text:
                    raw.setdefault(current_q, []).append(text)
                    if current_q not in order:
                        order.append(current_q)

        results = []
        for qnum in order:
            answer_lines: List[str] = []
            parts: List[Dict] = []
            current_part: Dict = None

            for line in raw[qnum]:
                part_m = PART_LABEL.match(line)
                if part_m:
                    if current_part:
                        parts.append(current_part)
                    part_ans = PART_LABEL.sub("", line).strip()
                    current_part = {
                        "part_label": f"({part_m.group(1)})",
                        "part_answer": part_ans,
                        "image": None,
                    }
                else:
                    if current_part is not None:
                        current_part["part_answer"] = (current_part["part_answer"] + " " + line).strip()
                    else:
                        answer_lines.append(line)

            if current_part:
                parts.append(current_part)

            results.append({
                "AnswerNumber": self._fmt_qnum(qnum, meta),
                "Answer_text": " ".join(answer_lines),
                "image": None,
                "Parts": parts,
            })

        return results

    # ── MCQ QP ────────────────────────────────────────────────────────────

    def _extract_mcq_qp(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = fitz.open(str(pdf_path))
        page_data = self._build_page_data(doc, mcq=True)

        raw: Dict[int, List] = {}
        order: List[int] = []
        current_q = None

        for pd in page_data:
            if pd is None:
                continue
            for ln in pd["lines"]:
                if self._is_boilerplate(ln):
                    continue
                anchor_here = next(
                    (qn for qn, y0 in pd["anchors"] if abs(y0 - ln.bbox.y0) < 0.6), None
                )
                if anchor_here is not None:
                    current_q = anchor_here
                    if MCQ_Q_SOLO.match(ln.text.strip()):
                        continue
                if current_q is None:
                    continue
                if DEFAULT_CAPTION_PATTERN.match(ln.text.strip()):
                    continue
                if any(u - 1 <= ln.bbox.y0 and ln.bbox.y1 <= lo + 1 for (u, lo) in pd["bands"]):
                    continue
                t = ln.text.strip()
                if t:
                    raw.setdefault(current_q, []).append(ln)
                    if current_q not in order:
                        order.append(current_q)

        results = []
        for qnum in order:
            lns = raw[qnum]
            stem_parts: List[str] = []
            options: Dict[str, str] = {"A": "", "B": "", "C": "", "D": ""}
            current_opt: str = None

            for ln in lns:
                t = ln.text.strip()
                # Strip leading question-number prefix (e.g. "5 Two men..." → "Two men...")
                t_body = re.sub(r"^\d{1,2}\s+", "", t) if MCQ_Q_PATTERN.match(t) else t

                opt_m = OPTION_INLINE.match(t_body) or OPTION_SOLO.match(t_body)
                if opt_m and opt_m.group(1) in MCQ_ANSWER_LABELS:
                    current_opt = opt_m.group(1)
                    opt_text = OPTION_INLINE.sub("", t_body).strip() or OPTION_SOLO.sub("", t_body).strip()
                    options[current_opt] = opt_text
                elif current_opt is not None:
                    options[current_opt] = (options[current_opt] + " " + t_body).strip()
                else:
                    stem_parts.append(t_body)

            results.append({
                "QuestionNumber": self._fmt_qnum(qnum, meta),
                "Question_text": " ".join(stem_parts),
                "Unit": "",
                "Topic": "",
                "Topic_name": "",
                "Sub_topic": [],
                "Difficulty_score": 0,
                "Difficulty_label": "",
                "Command_words": [],
                "Options": options,
                "Marks": 1,
                "image": None,
            })

        return results

    # ── MCQ MS ────────────────────────────────────────────────────────────

    def _extract_mcq_ms(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for 'N  X' answer-key lines (digit + single letter A-D)."""
        doc = fitz.open(str(pdf_path))
        answer_pattern = re.compile(r"^\s*(\d{1,2})\s+([ABCD])\s*$")
        seen: set = set()
        results: List[Dict] = []

        for pno in range(len(doc)):
            page = doc[pno]
            lines = self._cve.get_lines(page)
            for ln in lines:
                if self._is_boilerplate(ln):
                    continue
                m = answer_pattern.match(ln.text.strip())
                if m:
                    qnum = int(m.group(1))
                    if qnum not in seen:
                        seen.add(qnum)
                        results.append({
                            "AnswerNumber": self._fmt_qnum(qnum, meta),
                            "Answer": m.group(2),
                            "image": None,
                        })

        results.sort(key=lambda r: int(r["AnswerNumber"].split("/Q")[-1]))
        return results
