"""
pipeline/extractor/pymupdf_extractor.py

Duplicate Stage 2 — layout-based text extractor.

Uses PyMuPDF layout analysis to pull question prose and mark schemes directly
from PDFs without any LLM calls. Output is a schema-skeleton list — classification
fields (Unit, Topic, Difficulty, …) are populated in a subsequent ClassifierAgent pass.

Supported paper categories
--------------------------
  MCQ_QP   — Question stem + Options {A, B, C, D} per question
  MCQ_MS   — Answer-key letter (A/B/C/D) per question (handles single-line & multi-column tables)
  ESSAY_QP — Question stem + sub-parts (a)/(b)/... + [N] marks
  ESSAY_MS — Answer text + sub-part answers (handles modern table mark schemes & legacy layouts)
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    re.compile(r"Permission to reproduce", re.IGNORECASE),
    re.compile(r"Every reasonable effort has been made", re.IGNORECASE),
    re.compile(r"University of Cambridge International Examinations", re.IGNORECASE),
    re.compile(r"University of Cambridge", re.IGNORECASE),
    re.compile(r"Cambridge Assessment", re.IGNORECASE),
    re.compile(r"^DC\s*\(", re.IGNORECASE),
    re.compile(r"^[\*\s\d]{6,}$"),
]

DOT_LEADER = re.compile(r"\.{4,}")
MARKS_PATTERN = re.compile(r"\[(\d+)\]")
PART_LABEL = re.compile(r"^\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\)\s*", re.IGNORECASE)
OPTION_INLINE = re.compile(r"^([ABCD])\s+(.*)$")
OPTION_SOLO = re.compile(r"^([ABCD])$")

# Mark scheme specific regexes
MODERN_Q_STRICT = re.compile(r"^\s*([1-9]\d?)\s*\(([a-g])\)(?:\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\))?\s*$", re.IGNORECASE)
MODERN_Q_INLINE = re.compile(r"^\s*([1-9]\d?)\s*\(([a-g])\)(?:\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\))?\s+(.*)$", re.IGNORECASE)
MARK_NOTATION = re.compile(r"\b[BCAM]\d\b", re.IGNORECASE)


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
        return any(p.search(t) for p in BOILERPLATE_PATTERNS)

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
                upper, lower = self._cve.get_caption_bounds(cap, lines, page_top=40.0, page_bottom=page.rect.y1 - 20)
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

        doc.close()
        return results

    # ── Essay MS ──────────────────────────────────────────────────────────

    def _detect_ms_type(self, doc: fitz.Document) -> str:
        """Returns 'modern' or 'legacy' based on occurrence of modern question table structure."""
        modern_count = 0
        for pno in range(len(doc)):
            page = doc[pno]
            blocks = page.get_text("blocks")
            for b in blocks:
                for line in b[4].split("\n"):
                    t = line.strip()
                    if MODERN_Q_STRICT.match(t) or MODERN_Q_INLINE.match(t):
                        modern_count += 1
        return "modern" if modern_count >= 3 else "legacy"

    def _extract_essay_ms(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = fitz.open(str(pdf_path))
        ms_type = self._detect_ms_type(doc)

        if ms_type == "modern":
            # Modern table format with question labels like "1(a)", "1(b)(i)"
            questions = {}
            order = []
            current_q = None
            current_part_label = None

            for pno in range(len(doc)):
                page = doc[pno]
                blocks = page.get_text("blocks")
                for b in blocks:
                    lines = [l.strip() for l in b[4].split("\n") if l.strip()]
                    for line in lines:
                        if any(h in line for h in [
                            "Cambridge O Level", "Cambridge IGCSE", "Cambridge International",
                            "PUBLISHED", "May/June", "October/November", "Question", "Answer",
                            "Marks", "Page "
                        ]) and len(line) < 45:
                            continue

                        m_strict = MODERN_Q_STRICT.match(line)
                        m_inline = MODERN_Q_INLINE.match(line)

                        if m_strict:
                            qnum = int(m_strict.group(1))
                            sub1 = m_strict.group(2)
                            sub2 = m_strict.group(3)
                            part_label = f"({sub1})" if not sub2 else f"({sub1})({sub2})"
                            current_q = qnum
                            current_part_label = part_label
                            if current_q not in order:
                                order.append(current_q)
                            questions.setdefault(current_q, {"answer_text": [], "parts": {}})
                        elif m_inline:
                            qnum = int(m_inline.group(1))
                            sub1 = m_inline.group(2)
                            sub2 = m_inline.group(3)
                            part_label = f"({sub1})" if not sub2 else f"({sub1})({sub2})"
                            rem = m_inline.group(4).strip()
                            rem = MARK_NOTATION.sub("", rem).strip()
                            current_q = qnum
                            current_part_label = part_label
                            if current_q not in order:
                                order.append(current_q)
                            questions.setdefault(current_q, {"answer_text": [], "parts": {}})
                            if rem:
                                pdict = questions[current_q]["parts"]
                                pdict[current_part_label] = (pdict.get(current_part_label, "") + " " + rem).strip()
                        elif current_q is not None and current_part_label is not None:
                            clean_line = MARK_NOTATION.sub("", line).strip()
                            if clean_line and clean_line not in ["B1", "C1", "A1", "M1", "B2", "A2"]:
                                pdict = questions[current_q]["parts"]
                                pdict[current_part_label] = (pdict.get(current_part_label, "") + " " + clean_line).strip()

            results = []
            for qnum in order:
                q_data = questions[qnum]
                parts_list = []
                for plabel, ptext in q_data["parts"].items():
                    parts_list.append({
                        "part_label": plabel,
                        "part_answer": ptext,
                        "image": None,
                    })
                results.append({
                    "AnswerNumber": self._fmt_qnum(qnum, meta),
                    "Answer_text": " ".join(q_data["answer_text"]),
                    "image": None,
                    "Parts": parts_list,
                })
            doc.close()
            return results

        else:
            # Legacy margin layout (e.g. 2010-2015 mark schemes)
            questions = {}
            current_q = None

            for pno in range(len(doc)):
                page = doc[pno]
                lines = self._cve.get_lines(page)
                if any(ln.text.strip().upper() == "BLANK PAGE" for ln in lines):
                    continue
                non_bp = [ln for ln in lines if not self._is_boilerplate(ln)]
                if not non_bp:
                    continue

                rows = {}
                for ln in non_bp:
                    y_key = round(ln.bbox.y0 / 3.0) * 3.0
                    rows.setdefault(y_key, []).append(ln)

                for y in sorted(rows.keys()):
                    row_lines = sorted(rows[y], key=lambda l: l.bbox.x0)
                    q_anchor = None
                    content_lines = []
                    for ln in row_lines:
                        t = ln.text.strip()
                        if ln.bbox.x0 < 65 and re.match(r"^\d{1,2}$", t):
                            q_val = int(t)
                            if 1 <= q_val <= 30:
                                q_anchor = q_val
                        elif not re.match(r"^\s*[BCAM]\d\s*$", t, re.IGNORECASE):
                            content_lines.append(ln)

                    if q_anchor is not None:
                        current_q = q_anchor

                    if current_q is not None:
                        row_text = " ".join(ln.text.strip() for ln in content_lines).strip()
                        row_text = DOT_LEADER.sub("", row_text).strip()
                        if row_text and row_text not in ["Section A", "Section B"]:
                            questions.setdefault(current_q, []).append(row_text)

            results = []
            for qnum in sorted(questions.keys()):
                raw_lines = questions[qnum]
                answer_lines = []
                parts = []
                current_part = None

                for line in raw_lines:
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

            doc.close()
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
            current_opt: Optional[str] = None

            for i, ln in enumerate(lns):
                t = ln.text.strip()
                # Only strip leading question-number prefix on the first line of the question
                if i == 0 and re.match(rf"^{qnum}\s+", t):
                    t_body = re.sub(rf"^{qnum}\s+", "", t)
                else:
                    t_body = t

                solo_m = OPTION_SOLO.match(t_body)
                opt_m = OPTION_INLINE.match(t_body)

                # Disambiguate Option A from question stem starting with "A ..." (e.g. "A passenger is sitting...")
                is_real_option_a = True
                if opt_m and opt_m.group(1) == "A" and current_opt is None:
                    rest = opt_m.group(2).strip()
                    if len(rest.split()) > 3 and not re.match(r"^(is|are|has|have|can|must|decreases|increases|unchanged)\b", rest):
                        is_real_option_a = False

                if solo_m and solo_m.group(1) in MCQ_ANSWER_LABELS:
                    current_opt = solo_m.group(1)
                    options[current_opt] = ""
                elif opt_m and opt_m.group(1) in MCQ_ANSWER_LABELS and is_real_option_a:
                    current_opt = opt_m.group(1)
                    options[current_opt] = opt_m.group(2).strip()
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

        doc.close()
        return results

    # ── MCQ MS ────────────────────────────────────────────────────────────

    def _extract_mcq_ms(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts MCQ answer key letters (A-D) per question.
        Supports:
          1. Direct lines ("1 C", "2 D")
          2. Multi-column tables where Question Number and Key are separate columns
        """
        doc = fitz.open(str(pdf_path))
        answer_pattern = re.compile(r"^\s*(\d{1,2})\s+([ABCD])\s*$")
        answers: Dict[int, str] = {}

        for pno in range(len(doc)):
            page = doc[pno]
            lines = self._cve.get_lines(page)
            non_bp = [ln for ln in lines if not self._is_boilerplate(ln)]

            # 1. Direct line matches
            for ln in non_bp:
                m = answer_pattern.match(ln.text.strip())
                if m:
                    qnum = int(m.group(1))
                    if 1 <= qnum <= 50:
                        answers[qnum] = m.group(2)

            # 2. Multi-column table layout
            num_candidates = []
            key_candidates = []
            for ln in non_bp:
                t = ln.text.strip()
                if re.match(r"^\d{1,2}$", t):
                    q = int(t)
                    if 1 <= q <= 50:
                        num_candidates.append((q, ln.bbox))
                elif re.match(r"^[ABCD]$", t):
                    key_candidates.append((t, ln.bbox))

            for q, n_bbox in num_candidates:
                if q in answers:
                    continue
                matches = []
                for k, k_bbox in key_candidates:
                    y_diff = abs(k_bbox.y0 - n_bbox.y0)
                    x_diff = k_bbox.x0 - n_bbox.x0
                    if y_diff <= 10 and 0 < x_diff < 160:
                        matches.append((y_diff, x_diff, k))
                if matches:
                    matches.sort(key=lambda m: (m[0], m[1]))
                    answers[q] = matches[0][2]

        results = []
        for qnum in sorted(answers.keys()):
            results.append({
                "AnswerNumber": self._fmt_qnum(qnum, meta),
                "Answer": answers[qnum],
                "image": None,
            })

        doc.close()
        return results
