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
from typing import Any, Dict, List, Optional, Set, Tuple

import pymupdf as fitz

from pipeline.visuals.extractor import (
    CambridgeVisualExtractor,
    DEFAULT_CAPTION_PATTERN,
    HEADER_Y,
    CONTENT_MAX_X,
    MCQ_Q_PATTERN,
    MCQ_Q_SOLO,
    MCQ_ANSWER_LABELS,
    MCQ_MIN_CLUSTER_AREA,
    MCQ_MIN_CLUSTER_DIM,
    rect_area,
    SUPERSCRIPT_MAP,
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

DOT_LEADER = re.compile(r"(\.{3,}|_{3,}|…{2,})")
MARKS_PATTERN = re.compile(r"\[(\d+)\]")
PART_LABEL = re.compile(r"^\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\)\s*", re.IGNORECASE)

# Hierarchical part detection for essay QP
# Matches (in order of precedence):
#   1) Compound: (b)(i) text  or  (b) (i) text
#   2) Letter only: (a) text
#   3) Roman only: (i) text
# Also accepts optional leading question number: "11 (a) ..."
PART_START_RE = re.compile(
    r"^\s*"
    r"(?:\d{1,2}\s+)?"                              # optional leading question number
    r"(?:"
        r"\(([a-g])\)\s*\(([ivx]+)\)"               # group 1+2: compound letter + roman
        r"|"
        r"\(([a-g])\)"                               # group 3: letter only
        r"|"
        r"\(([ivx]+)\)"                              # group 4: roman only
    r")\s*"
    r"(.*)$",                                        # group 5: remainder text
    re.IGNORECASE,
)

# Numbered sub-subparts inside roman subparts, e.g. "1.  the reading on the ammeter," or "(1) ..."
SUB_NUM_RE = re.compile(r"^\s*(?:([1-9])\.|\(([1-9])\))\s+([a-zA-Z].*)$", re.IGNORECASE)


# Detect standalone answer prompt lines in PDF, e.g. "change = ....", "speed = .... [2]", "height = .... [1]", "Answer (a) ...", "water: ...."
ANSWER_PROMPT_LINE = re.compile(
    r"^\s*(?:"
    r"Answer\b(?!\s+(?:all|the|four|three|two|one|any|questions?|each|on)\b).*"
    r"|"
    r"(?:[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*|.*?:\s*)(?:\.{2,}|_{2,}|…{2,}).*"
    r"|"
    r"(?:[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*|.*?:\s*)\s*\[\d+\]\s*"
    r")\s*$",
    re.IGNORECASE,
)

# Strip trailing answer prompt strings like "change =", "height =", "V2 =", "water:", "Answer (a) ..." from question prose
TRAILING_ANSWER_PROMPT = re.compile(
    r"(?:^|(?<=[.,;])\s*|\s+)"
    r"(?:"
        r"Answer\b(?!\s+(?:all|the|four|three|two|one|any|questions?|each|on)\b).*"
        r"|"
        r"(?:"
            r"(?:average\s+|total\s+|initial\s+|final\s+)?"
            r"[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*"
            r"(?:[a-zA-Z/°ΩΩ%–\^]+(?:\s*[0-9–\^]+)*(?:\s*/\s*[a-zA-Z0-9–\^]+)*)?"
        r")"
        r"|"
        r"[a-zA-Z0-9\s\-]{1,25}:\s*"
    r")\s*$",
    re.IGNORECASE,
)

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
        if line.bbox.y0 <= HEADER_Y:
            return True
        t = line.text.strip()
        if any(p.search(t) for p in BOILERPLATE_PATTERNS):
            return True
        if line.bbox.y0 >= FOOTER_Y:
            if MARKS_PATTERN.search(t):
                return False
            return True
        if line.bbox.x0 > CONTENT_MAX_X:
            return True
        return False

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

            q_to_table_y0: Dict[int, float] = {}
            table_rects: List[fitz.Rect] = []
            diag_boxes: List[fitz.Rect] = []

            if mcq and anchors:
                page_bottom = page.rect.y1 - 20
                # 1. Detect tables containing MCQ options A, B, C, D
                try:
                    tabs = page.find_tables()
                    for t in tabs.tables:
                        extracted = t.extract()
                        first_col = [row[0] for row in extracted if row and row[0] is not None]
                        first_col_str = " ".join(str(c) for c in first_col)
                        if all(opt in first_col_str for opt in ["A", "B", "C", "D"]):
                            t_rect = fitz.Rect(t.bbox)
                            table_rects.append(t_rect)
                            for i, (qnum, q_y0) in enumerate(anchors):
                                span_bottom = anchors[i + 1][1] if i + 1 < len(anchors) else page_bottom
                                if max(q_y0, t_rect.y0) < min(span_bottom, t_rect.y1):
                                    q_to_table_y0[qnum] = min(q_to_table_y0.get(qnum, 9999.0), t_rect.y0)
                                    break
                except Exception as e:
                    logger.debug(f"find_tables failed on page {pno}: {e}")

                # 2. Detect diagram clusters (drawings excluding option tables)
                try:
                    raw_rects = [
                        d["rect"] for d in page.get_drawings()
                        if max(d["rect"].width, d["rect"].height) > self._cve.min_stroke_len
                        and not (min(d["rect"].width, d["rect"].height) < 2.0 and max(d["rect"].width, d["rect"].height) > 300)
                    ]
                    all_clusters = [
                        r for r in self._cve.cluster_rects(raw_rects)
                        if rect_area(r) >= MCQ_MIN_CLUSTER_AREA
                        or (max(r.width, r.height) >= MCQ_MIN_CLUSTER_DIM and min(r.width, r.height) >= 5.0)
                    ]
                    diag_clusters = [c for c in all_clusters if not any(c.intersects(tr) for tr in table_rects)]
                    for c in diag_clusters:
                        nearby_boxes = [
                            ln.bbox for ln in lines
                            if c.y0 - 25 <= ln.bbox.y0 and ln.bbox.y1 <= c.y1 + 30
                            and c.x0 - 50 <= ln.bbox.x0 and ln.bbox.x1 <= c.x1 + 50
                            and not ln.text.strip().endswith(("?", "."))
                            and len(ln.text.strip()) < 40
                        ]
                        all_b = [c] + nearby_boxes
                        diag_boxes.append(fitz.Rect(
                            min(b.x0 for b in all_b),
                            min(b.y0 for b in all_b),
                            max(b.x1 for b in all_b),
                            max(b.y1 for b in all_b),
                        ))
                except Exception as e:
                    logger.debug(f"diagram clustering failed on page {pno}: {e}")

            page_data.append({
                "lines": lines,
                "anchors": anchors,
                "bands": bands,
                "q_to_table_y0": q_to_table_y0,
                "diag_boxes": diag_boxes,
            })
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

    @staticmethod
    def _clean_answer_prompts(text: str) -> str:
        """Strip trailing answer prompt strings like 'change =', 'height =', 'V2 =', 'water:', 'Answer (a) ...' from question prose."""
        prev = None
        cur = text.strip()
        while prev != cur:
            prev = cur
            cur = TRAILING_ANSWER_PROMPT.sub("", cur).strip()
        # If the remaining text is purely a standalone answer prompt or variable prompt, clear it
        if re.match(r"^(?:Answer\b.*|[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*|[a-zA-Z0-9\s\-]{1,25}:\s*)$", cur, re.IGNORECASE):
            cur = ""
        return cur

    @staticmethod
    def clean_math_text(text: str) -> str:
        """Normalize mathematical notation, exponents, units, and symbols."""
        if not text:
            return ""
        # 1. Powers of 10 in scientific notation: 1021 -> 10²¹, 10-4 -> 10⁻⁴
        def _sup_10(m):
            prefix = m.group(1)
            p = m.group(2)
            sup = "".join(SUPERSCRIPT_MAP.get(c, c) for c in p)
            return f"{prefix}{sup}"

        text = re.sub(r"([×\*]\s*10)([–\-]?\d{1,4})\b", _sup_10, text)
        text = re.sub(r"\b(10)[–\-]([1-9]\d{0,3})\b", lambda m: f"{m.group(1)}⁻{''.join(SUPERSCRIPT_MAP.get(c, c) for c in m.group(2))}", text)

        # 2. Units with powers: cm3 -> cm³, m2 -> m², km3 -> km³
        def _sup_unit(m):
            u = m.group(1)
            p = SUPERSCRIPT_MAP.get(m.group(2), m.group(2))
            return f"{u}{p}"

        text = re.sub(r"\b(cm|mm|km|dm|m|s|kg|g|ms|min|h|N|J|W|Pa|mol)([23])\b", _sup_unit, text)

        # 3. Common algebraic powers: 4x2 -> 4x², 25y2 -> 25y², 5a2 -> 5a², t2 -> t², n3 -> n³, n2 -> n²
        def _sup_var(m):
            prefix = m.group(1)
            var = m.group(2)
            p = SUPERSCRIPT_MAP.get(m.group(3), m.group(3))
            return f"{prefix}{var}{p}"

        text = re.sub(r"([0-9\(\)\s\+\-–=]|^)([a-zA-Z])([23])(?=[\s\+\-–=,\.\/\)]|$)", _sup_var, text)

        # 4. Multiplication formatting: ' 5.98 * 10 ' or ' 5.98 x 10 ' -> ' 5.98 × 10 '
        text = re.sub(r"(\d)\s*[xX]\s*(10[⁰¹²³⁴⁵⁶⁷⁸⁹\d])", r"\1 × \2", text)
        text = re.sub(r"(\d)\s*\*\s*(\d)", r"\1 × \2", text)

        # 5. Collapse spaces before superscripts: 't ²' -> 't²'
        text = re.sub(r"([a-zA-Z0-9])\s+([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ])", r"\1\2", text)

        # 6. Equivalence & congruency symbols normalization
        text = text.replace(r"\equiv", "≡").replace("&equiv;", "≡")
        text = text.replace(r"\cong", "≅").replace("&cong;", "≅")

        return text

    # ── Essay QP ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_hierarchical_parts(qnum: int, raw_lines: List[str]) -> Tuple[str, List[Dict], int]:
        """
        Parse flat raw_lines into a recursive tree structure with arbitrary depth.

        Structure:
          - Top-level part:
            {
              "part_label": "b",
              "part_text": "...",          # preamble context preceding sub-parts
              "marks": 10,                 # aggregate roll-up sum of child marks
              "image": None,
              "sub_parts": [ ... ]
            }
          - Nested sub-part (e.g. b(i), b(ii)):
            {
              "sub_part_label": "b(i)",
              "sub_part_text": "...",
              "marks": 2,
              "image": None,
              "sub_parts": [ ... ]
            }
          - Deeper nested sub-sub-part (e.g. c(i)(1)):
            {
              "sub_part_label": "c(i)(1)",
              "sub_part_text": "...",
              "marks": 2,
              "image": None,
              "sub_parts": []
            }
        """
        stem_lines: List[str] = []
        parts: List[Dict] = []
        current_part: Optional[Dict] = None
        current_subpart: Optional[Dict] = None
        current_subsubpart: Optional[Dict] = None
        stem_marks: int = 0

        for raw_line in raw_lines:
            line = raw_line.strip()
            # Strip standalone question-number prefix (e.g. "11 " before "(a)")
            line = re.sub(rf"^\s*{qnum}\s+(?=\()", "", line).strip()

            # Check for numbered sub-subpart first if we are inside an active roman subpart
            sub_num_m = SUB_NUM_RE.match(line) if current_subpart else None
            part_m = PART_START_RE.match(line) if not sub_num_m else None

            if part_m:
                comp_l, comp_r, ltr, rom, rem = (
                    part_m.group(1),
                    part_m.group(2),
                    part_m.group(3),
                    part_m.group(4),
                    (part_m.group(5) or "").strip(),
                )
                marks_m = MARKS_PATTERN.search(rem)
                m_val = int(marks_m.group(1)) if marks_m else 0
                rem_clean = MARKS_PATTERN.sub("", rem).strip()

                if comp_l and comp_r:
                    # Compound start: (b)(i) -> starts part 'b', starts subpart 'b(i)'
                    current_part = {
                        "part_label": comp_l.lower(),
                        "part_text": "",
                        "marks": 0,
                        "image": None,
                        "sub_parts": [],
                    }
                    parts.append(current_part)
                    current_subpart = {
                        "sub_part_label": f"{comp_l.lower()}({comp_r.lower()})",
                        "sub_part_text": rem_clean,
                        "marks": m_val,
                        "image": None,
                        "sub_parts": [],
                    }
                    current_part["sub_parts"].append(current_subpart)
                    current_subsubpart = None
                elif ltr:
                    # Letter part: (a), (b), etc.
                    current_part = {
                        "part_label": ltr.lower(),
                        "part_text": rem_clean,
                        "marks": m_val,
                        "image": None,
                        "sub_parts": [],
                    }
                    parts.append(current_part)
                    current_subpart = None
                    current_subsubpart = None
                elif rom:
                    # Roman subpart: (i), (ii), etc.
                    parent_ltr = current_part["part_label"] if current_part else "a"
                    if not current_part:
                        current_part = {
                            "part_label": parent_ltr,
                            "part_text": "",
                            "marks": 0,
                            "image": None,
                            "sub_parts": [],
                        }
                        parts.append(current_part)
                    current_subpart = {
                        "sub_part_label": f"{parent_ltr}({rom.lower()})",
                        "sub_part_text": rem_clean,
                        "marks": m_val,
                        "image": None,
                        "sub_parts": [],
                    }
                    current_part["sub_parts"].append(current_subpart)
                    current_subsubpart = None
            elif sub_num_m:
                # Numbered sub-subpart inside a roman subpart: '1.', '2.'
                num = sub_num_m.group(1) or sub_num_m.group(2)
                rem = sub_num_m.group(3).strip()
                marks_m = MARKS_PATTERN.search(rem)
                m_val = int(marks_m.group(1)) if marks_m else 0
                rem_clean = MARKS_PATTERN.sub("", rem).strip()

                current_subsubpart = {
                    "sub_part_label": f"{current_subpart['sub_part_label']}({num})",
                    "sub_part_text": rem_clean,
                    "marks": m_val,
                    "image": None,
                    "sub_parts": [],
                }
                current_subpart["sub_parts"].append(current_subsubpart)
            else:
                # Continuation prose line
                marks_m = MARKS_PATTERN.search(line)
                m_val = int(marks_m.group(1)) if marks_m else 0
                line_clean = MARKS_PATTERN.sub("", line).strip()

                if current_subsubpart is not None:
                    if m_val:
                        current_subsubpart["marks"] = max(current_subsubpart["marks"], m_val)
                    if line_clean:
                        current_subsubpart["sub_part_text"] = (
                            current_subsubpart["sub_part_text"] + " " + line_clean
                        ).strip()
                elif current_subpart is not None:
                    if m_val:
                        current_subpart["marks"] = max(current_subpart["marks"], m_val)
                    if line_clean:
                        current_subpart["sub_part_text"] = (
                            current_subpart["sub_part_text"] + " " + line_clean
                        ).strip()
                elif current_part is not None:
                    if m_val:
                        current_part["marks"] = max(current_part["marks"], m_val)
                    if line_clean:
                        current_part["part_text"] = (
                            current_part["part_text"] + " " + line_clean
                        ).strip()
                else:
                    if m_val:
                        stem_marks = max(stem_marks, m_val)
                    if line_clean:
                        stem_lines.append(line_clean)

        # Post-processing: clean answer prompts, normalize math text, and compute recursive marks roll-up
        def _clean_and_rollup(node: Dict, is_top: bool = False) -> int:
            text_key = "part_text" if is_top else "sub_part_text"
            cleaned_text = PyMuPDFExtractor._clean_answer_prompts(node[text_key])
            node[text_key] = PyMuPDFExtractor.clean_math_text(cleaned_text)
            sub_parts = node.get("sub_parts", [])
            if sub_parts:
                node["marks"] = sum(_clean_and_rollup(sp, is_top=False) for sp in sub_parts)
            else:
                if node.get("marks", 0) <= 0:
                    node["marks"] = 1
            return node["marks"]

        if parts:
            total_marks = sum(_clean_and_rollup(p, is_top=True) for p in parts)
        else:
            total_marks = max(stem_marks, 1)

        stem_cleaned = PyMuPDFExtractor._clean_answer_prompts(" ".join(stem_lines))
        stem_text = PyMuPDFExtractor.clean_math_text(stem_cleaned)
        return stem_text, parts, max(total_marks, 1)

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
                    marks_m = MARKS_PATTERN.search(ln.text)
                    if marks_m:
                        raw.setdefault(current_q, []).append(f"[{marks_m.group(1)}]")
                    continue
                if any(u - 1 <= ln.bbox.y0 and ln.bbox.y1 <= lo + 1 for (u, lo) in pd["bands"]):
                    marks_m = MARKS_PATTERN.search(ln.text)
                    if marks_m:
                        raw.setdefault(current_q, []).append(f"[{marks_m.group(1)}]")
                    continue

                t = ln.text.strip()
                # Skip standalone answer prompt lines (e.g. "change = ....", "speed = .... [2]")
                if ANSWER_PROMPT_LINE.match(t):
                    marks_m = MARKS_PATTERN.search(t)
                    if marks_m:
                        raw.setdefault(current_q, []).append(f"[{marks_m.group(1)}]")
                    continue

                text = DOT_LEADER.sub("", ln.text).strip()
                if text:
                    raw.setdefault(current_q, []).append(text)
                    if current_q not in order:
                        order.append(current_q)

        results = []
        for qnum in order:
            stem_text, parts, total_marks = self._build_hierarchical_parts(qnum, raw[qnum])
            results.append({
                "QuestionNumber": self._fmt_qnum(qnum, meta),
                "Question_text": stem_text,
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
                        "part_answer": PyMuPDFExtractor.clean_math_text(ptext),
                        "image": None,
                    })
                results.append({
                    "AnswerNumber": self._fmt_qnum(qnum, meta),
                    "Answer_text": PyMuPDFExtractor.clean_math_text(" ".join(q_data["answer_text"])),
                    "image": None,
                    "Parts": parts_list,
                })
            doc.close()
            return results

        else:
            # Legacy margin layout (e.g. 2010-2015 mark schemes).
            #
            # In older Cambridge PDFs, single-digit questions (Q1-Q9) have the number
            # as a standalone span in the left margin, then content in adjacent spans.
            # BUT two-digit questions (Q10, Q11...) are rendered as ONE fused span:
            #   e.g. "10 (a) (master cylinder creates) pressure in brake fluid..."
            # The old code only matched bare digit tokens, so Q10+ were silently dropped.
            # Fix: also detect the "NN <content>" pattern and split off the inline prefix.
            LEGACY_Q_INLINE = re.compile(r"^(\d{1,2})\s+(.+)$")

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
                    inline_prefix = None  # text fused onto the same span as the question number

                    for ln in row_lines:
                        t = ln.text.strip()
                        if re.match(r"^\s*[BCAM]\d\s*$", t, re.IGNORECASE):
                            # Mark-notation token (B1, C1, A1, M1...) — skip entirely
                            continue
                        if ln.bbox.x0 < 65:
                            if re.match(r"^\d{1,2}$", t):
                                # Standalone question number (Q1-Q9 typical layout)
                                q_val = int(t)
                                if 1 <= q_val <= 30:
                                    q_anchor = q_val
                                    continue
                            else:
                                # Fused "10 (a) content..." pattern for Q10+
                                m = LEGACY_Q_INLINE.match(t)
                                if m:
                                    q_val = int(m.group(1))
                                    if 1 <= q_val <= 30:
                                        q_anchor = q_val
                                        inline_prefix = m.group(2).strip()
                                        continue
                        content_lines.append(ln)

                    if q_anchor is not None:
                        current_q = q_anchor

                    if current_q is not None:
                        # Prepend any text that was fused onto the question-number span
                        parts_text = []
                        if inline_prefix:
                            parts_text.append(inline_prefix)
                            inline_prefix = None
                        parts_text += [ln.text.strip() for ln in content_lines]
                        row_text = " ".join(parts_text).strip()
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
                    "Answer_text": PyMuPDFExtractor.clean_math_text(" ".join(answer_lines)),
                    "image": None,
                    "Parts": [
                        {
                            "part_label": p["part_label"],
                            "part_answer": PyMuPDFExtractor.clean_math_text(p["part_answer"]),
                            "image": None,
                        }
                        for p in parts
                    ],
                })

            doc.close()
            return results

    # ── MCQ QP ────────────────────────────────────────────────────────────

    def _extract_mcq_qp(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = fitz.open(str(pdf_path))
        page_data = self._build_page_data(doc, mcq=True)

        raw: Dict[int, List] = {}
        order: List[int] = []

        for pd in page_data:
            if pd is None:
                continue
            anchors = pd["anchors"]
            page_lines = pd["lines"]
            q_to_table_y0 = pd.get("q_to_table_y0", {})
            diag_boxes = pd.get("diag_boxes", [])

            for i, (qnum, q_y0) in enumerate(anchors):
                next_y0 = anchors[i + 1][1] if i + 1 < len(anchors) else 9999.0
                qlines = [ln for ln in page_lines if q_y0 <= ln.bbox.y0 < next_y0]

                is_table_q = qnum in q_to_table_y0
                table_y0 = q_to_table_y0.get(qnum, 9999.0)
                q_diags = [b for b in diag_boxes if max(q_y0, b.y0) < min(next_y0, b.y1) and b.y1 <= table_y0 + 5]

                for ln in qlines:
                    if self._is_boilerplate(ln):
                        continue
                    if any(b.intersects(ln.bbox) or b.contains(ln.bbox) for b in q_diags):
                        continue
                    if is_table_q and ln.bbox.y0 >= table_y0 - 2.0:
                        continue
                    t = ln.text.strip()
                    if not t:
                        continue
                    raw.setdefault(qnum, []).append((ln, is_table_q))
                    if qnum not in order:
                        order.append(qnum)

        results = []
        for qnum in order:
            ln_data = raw[qnum]
            is_table_q = any(item[1] for item in ln_data)
            stem_parts: List[str] = []
            options: Dict[str, str] = {"A": "", "B": "", "C": "", "D": ""}
            current_opt: Optional[str] = None

            for i, (ln, _) in enumerate(ln_data):
                t = ln.text.strip()
                if not stem_parts and re.match(rf"^{qnum}\s+", t):
                    t_body = re.sub(rf"^{qnum}\s+", "", t)
                elif t == str(qnum):
                    continue
                else:
                    t_body = t

                if is_table_q:
                    # Table options: options remain empty strings, prose forms stem
                    if t_body:
                        stem_parts.append(t_body)
                else:
                    solo_m = OPTION_SOLO.match(t_body)
                    opt_m = OPTION_INLINE.match(t_body)

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

            cleaned_stem = PyMuPDFExtractor.clean_math_text(" ".join(stem_parts))
            cleaned_options = {
                opt: PyMuPDFExtractor.clean_math_text(val)
                for opt, val in options.items()
            }

            results.append({
                "QuestionNumber": self._fmt_qnum(qnum, meta),
                "Question_text": cleaned_stem,
                "Unit": "",
                "Topic": "",
                "Topic_name": "",
                "Sub_topic": [],
                "Difficulty_score": 0,
                "Difficulty_label": "",
                "Command_words": [],
                "Options": cleaned_options,
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
