"""
Stage 3 — Cambridge Visual Extractor

Extracts whole labelled figures (graphs, diagrams, tables, apparatus drawings)
from Cambridge/CIE-style exam PDFs, complete with axis labels, tick numbers, and
component captions, cropped tight around the actual artwork -- and tags each
one with the question it belongs to.

KEY IMPROVEMENTS:
1. Wrapped-Sentence Bleed Prevention: Chaining backward through short body lines
   prevents wrapped sentence tails from bleeding into figure crops.
2. Stroke-Length Path Preservation: Preserves thin-but-long strokes (leader lines,
   hanging threads, table gridlines) using max-dimension filtering.
3. Degenerate-Rect Union Handling: Custom rect_union and rects_touch calculations
   prevent PyMuPDF from silently dropping zero-width/height straight lines.
4. Banded Stitch Rendering: Splits large vertical gaps between figure content and
   captions into bands, and stitches them with a small visual gap to remove huge
   blank student answer spaces.
5. Strict Fallback Suppression: Overlap detection suppresses noisy partial duplicate
   crops.
"""
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pymupdf as fitz
from PIL import Image

logger = logging.getLogger(__name__)

# Matches an ISOLATED caption line, e.g. "Fig. 10.2", "Figure 1.1", "Table 7.1", "Fig. 9.1 (not to scale)", "Fig.1.1"
DEFAULT_CAPTION_PATTERN = re.compile(
    r"^\s*(Fig(?:ure)?\.?|Table)\s*(\d{1,2})\.(\d{1,2})(?:\s*\([^\)]+\))?\s*$",
    re.IGNORECASE,
)
DIAGRAM_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Diagram|Fig(?:ure)?)\s+([IVX\d]+(?:\s*\([^\)]+\))?)\s*$",
    re.IGNORECASE,
)

LEFT_MARGIN_X = 60.0       # question numbers sit left of this
HEADER_Y = 40.0            # page-number-in-header sits above this; ignore it
MAX_LINE_HEIGHT = 45.0     # guard against rotated page-tall margin text (e.g. 'DO NOT WRITE IN THIS MARGIN')
PROSE_MIN_WIDTH = 120.0    # a "real sentence" line is at least this wide
PROSE_MAX_X0 = 135.0       # allow indented question sub-part lines (which sit around x0 ≈ 115-125 pt)
PROSE_CHAIN_GAP = 20.0     # max vertical gap between lines of one paragraph
CONTENT_MAX_X = 522.0      # "For Examiner's Use" sidebar starts right of this
MIN_STROKE_LEN = 1.0       # drop sub-pixel path fragments, keep dots and thin long lines
GAP_BREAK = 40.0           # blank vertical gap (pt) that splits a figure into bands
STITCH_GAP_PX = 16         # visual gap between stitched bands in the output
PART_LABEL = re.compile(r"^\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\)\s*", re.IGNORECASE)
MARK_INLINE_PATTERN = re.compile(r"\[\s*\d+\s*\]|\[\s*Total", re.IGNORECASE)
PRINT_X0, PRINT_X1 = 35.0, 540.0
PRINT_Y0, PRINT_Y1 = 45.0, 785.0

# Hierarchical part detection — mirrors the one in pymupdf_extractor.py
_VIS_PART_RE = re.compile(
    r"^\s*"
    r"(?:\d{1,2}\s+)?"
    r"(?:"
        r"\(([a-g])\)\s*\(([ivx]+)\)"    # compound: (b)(i)
        r"|"
        r"\(([a-g])\)"                    # letter only: (a)
        r"|"
        r"\(([ivx]+)\)"                   # roman only: (i)
    r")\s*"
    r"(.*)$",
    re.IGNORECASE,
)
_VIS_SUB_NUM_RE = re.compile(r"^\s*(?:([1-9])\.|\(([1-9])\))\s+([a-zA-Z].*)$", re.IGNORECASE)
GRID_PROMPT_PATTERN = re.compile(
    r"\b(?:grid\s+opposite|grid\s+below|on\s+the\s+grid|plot\s+a\s+graph|draw\s+a\s+graph|graph\s+opposite|grid\s+on\s+page)\b",
    re.IGNORECASE,
)


# MCQ paper detection & extraction constants
MCQ_Q_PATTERN = re.compile(r"^(\d{1,2})\s+\S")       # "5 Two men..." or "40 Which..."
MCQ_Q_SOLO = re.compile(r"^(\d{1,2})$")               # bare "1", "2" … "40" alone on line
MCQ_Q_X0_MAX = 58.0        # MCQ question numbers sit at x0 ≈ 50, never beyond 58
MCQ_ANSWER_LABELS = {"A", "B", "C", "D"}  # MCQ answer option labels
MCQ_MIN_CLUSTER_AREA = 2000.0  # Minimum vector cluster area to count as a real diagram
MCQ_MIN_CLUSTER_DIM = 30.0     # Min longest dimension to count as a real diagram
MCQ_MAX_TEXT_DENSITY = 0.35    # Diagram clusters must have text coverage < 35% (tables have >= 45%)
FALLBACK_MIN_CLUSTER_AREA = 3000.0  # Min cluster area for uncaptioned fallback diagrams (e.g. Maths)
FALLBACK_MIN_CLUSTER_DIM = 20.0     # Min dimension for uncaptioned fallback diagrams

# Units of measurement and graph scale abbreviations that should never be parsed as question numbers
UNIT_TOKENS: Set[str] = {
    "cm", "mm", "m", "km", "nm", "μm",
    "metre", "metres", "meter", "meters",
    "g", "kg", "mg",
    "s", "ms", "μs", "sec", "secs", "second", "seconds",
    "min", "mins", "minute", "minutes",
    "h", "hr", "hrs", "hour", "hours",
    "n", "kn", "mn",
    "j", "kj", "mj",
    "w", "kw", "mw",
    "v", "kv", "mv",
    "a", "ma", "ka",
    "pa", "kpa", "mpa",
    "hz", "khz", "mhz",
    "c", "°c", "k",
    "rad", "deg",
    "mol", "db",
    "ohm", "ω", "kω", "mω",
    "%", "per",
}

# Mathematical font symbol mapping and superscript tables
SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '–': '⁻', '—': '⁻', '=': '⁼',
    '(': '⁽', ')': '⁾', 'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ',
}

PUA_FONT_MAP = {
    0xf0b4: '×', 0xf0b8: '÷', 0xf0b0: '°', 0xf057: 'Ω',
    0xf0b7: '•', 0xf0b1: '±', 0xf0ae: '→', 0xf0ac: '←',
    0xf0a3: '≤', 0xf0b3: '≥', 0xf0b9: '≠', 0xf071: 'θ',
    0xf070: 'π', 0xf044: 'Δ', 0xf06c: 'λ', 0xf06d: 'μ',
    0xf072: 'ρ', 0xf061: 'α', 0xf062: 'β', 0xf067: 'γ',
    0xf053: 'Σ', 0xf073: 'σ', 0xf0a5: '∞', 0xf0c5: '∅',
    0xf0ce: '∈', 0xf0cf: '∉', 0xf0cd: '⊂', 0xf0cc: '⊆',
    0xf0c7: '∩', 0xf0c8: '∪', 0xf0d6: '√', 0xf0b2: '"',
    0xf020: ' ',
}


def map_font_glyphs(text: str, font: str) -> str:
    """Map font-specific encodings and Private Use Area code points to standard Unicode mathematical symbols."""
    res = []
    for ch in text:
        code = ord(ch)
        if code in PUA_FONT_MAP:
            res.append(PUA_FONT_MAP[code])
        elif "MathematicalPi-One" in font:
            if code == 0x02 or code == 0xe03c:
                res.append('<')
            elif code == 0x03:
                res.append('>')
            elif code in (0x04, 0x05, 0xe02c):
                res.append('≤')
            else:
                res.append(ch)
        elif "MathematicalPi-Three" in font:
            if code in (0x02, 0xe031):
                res.append('(')
            elif code in (0x03, 0xe032):
                res.append(')')
            else:
                res.append(ch)
        elif "MathematicalPi-Two" in font:
            if code == 0x02:
                res.append('ℰ')
            else:
                res.append(ch)
        elif "MathematicalPi-Four" in font:
            if code == 0x02:
                res.append('∪')
            elif code == 0x03:
                res.append("'")
            else:
                res.append(ch)
        elif 0xf8ea <= code <= 0xf8f8:
            continue
        elif code < 32 and ch not in '\n\r\t':
            if code == 0x02:
                res.append('<')
            elif code == 0x03:
                res.append('>')
            elif code in (0x04, 0x05):
                res.append('≤')
            else:
                res.append(' ')
        else:
            res.append(ch)
    return ''.join(res)


def rect_area(r: fitz.Rect) -> float:
    """Safe area computation returning 0.0 for empty/inverted rects."""
    return max(0.0, r.width) * max(0.0, r.height)


@dataclass
class Line:
    """Represents an extracted text line with font and layout metadata."""
    text: str
    bbox: fitz.Rect
    bold: bool
    width: float


def rect_union(a: fitz.Rect, b: fitz.Rect) -> fitz.Rect:
    """Bounding union computed manually without PyMuPDF zero-area dropping."""
    return fitz.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def rects_touch(
    a: fitz.Rect,
    b: fitz.Rect,
    pad: float = 0.0,
    pad_x: Optional[float] = None,
    pad_y: Optional[float] = None,
) -> bool:
    """Padded overlap test computed manually without PyMuPDF zero-area dropping."""
    px = pad_x if pad_x is not None else pad
    py = pad_y if pad_y is not None else pad
    return not (
        b.x1 < a.x0 - px
        or b.x0 > a.x1 + px
        or b.y1 < a.y0 - py
        or b.y0 > a.y1 + py
    )



class CambridgeVisualExtractor:
    """
    Extracts whole labelled figures from Cambridge/CIE-style exam PDFs using
    layout-anchored, caption-guided multi-band rendering.
    """

    def __init__(
        self,
        dpi: int = 300,
        min_vector_area: float = 300.0,
        pad: float = 4.0,
        left_margin_x: float = LEFT_MARGIN_X,
        header_y: float = HEADER_Y,
        prose_min_width: float = PROSE_MIN_WIDTH,
        prose_max_x0: float = PROSE_MAX_X0,
        prose_chain_gap: float = PROSE_CHAIN_GAP,
        content_max_x: float = CONTENT_MAX_X,
        min_stroke_len: float = MIN_STROKE_LEN,
        gap_break: float = GAP_BREAK,
        stitch_gap_px: int = STITCH_GAP_PX,
        caption_pattern: Optional[re.Pattern] = None,
    ):
        self.dpi = dpi
        self.min_vector_area = min_vector_area
        self.pad = pad
        self.left_margin_x = left_margin_x
        self.header_y = header_y
        self.prose_min_width = prose_min_width
        self.prose_max_x0 = prose_max_x0
        self.prose_chain_gap = prose_chain_gap
        self.content_max_x = content_max_x
        self.min_stroke_len = min_stroke_len
        self.gap_break = gap_break
        self.stitch_gap_px = stitch_gap_px
        self.caption_pattern = caption_pattern or DEFAULT_CAPTION_PATTERN

    @staticmethod
    def _is_blank_image(pix: fitz.Pixmap, white_threshold: float = 0.995) -> bool:
        """
        Return True when the rendered pixmap is essentially all white.

        A near-uniform white region has no diagram content -- it's a blank
        answer space or a margin swatch that slipped through the area filter.
        We sample the pixel data via Pillow and check what fraction of pixels
        are 'white-enough' (all three channels >= 245).
        """
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        w, h = img.size
        total = w * h
        if total == 0:
            return True
        pixels = img.getdata()
        white_count = sum(1 for r, g, b in pixels if r >= 245 and g >= 245 and b >= 245)
        return (white_count / total) >= white_threshold

    @staticmethod
    def _build_raster_xref_map(page: fitz.Page) -> Dict[Tuple[float, float, float, float], int]:
        """
        Build a bbox-tuple -> xref mapping for all embedded raster images on
        the page. PyMuPDF returns image placement info in the same order as
        get_images(), so we can zip them to pair each bbox with its xref.

        The xref lets us extract the raw compressed image bytes (before the
        page composites them on top of white), which is what we need for
        reliable watermark detection.
        """
        xref_map: Dict[Tuple[float, float, float, float], int] = {}
        for img_info in page.get_image_info(xrefs=True):
            xref = img_info.get("xref", 0)
            if xref:
                xref_map[tuple(img_info["bbox"])] = xref
        return xref_map

    @staticmethod
    def _is_papacambridge_ribbon(doc: fitz.Document, xref: int, dark_threshold: float = 0.50) -> bool:
        """
        Return True when the raw embedded image XObject looks like the
        PapaCambridge diagonal ribbon watermark.

        The ribbon is a dark (grey/black) PNG stamp that PapaCambridge stamps
        into the top-right corner of every PDF. When rendered onto a white page
        it appears 70-80% white (because most of the tile is transparent/white),
        but the raw image XObject itself is almost entirely dark pixels (the
        ribbon contains the text 'www.PapaCambridge.com' on a dark background).

        A raw-pixel dark fraction > 50% reliably identifies this ribbon while
        being safely below any threshold a real physics/chemistry diagram could hit
        (those are mostly white with a few black lines -- well under 30% dark).
        """
        try:
            bdata = doc.extract_image(xref)
            raw_img = Image.open(io.BytesIO(bdata["image"])).convert("RGB")
        except Exception:
            return False
        raw_arr = list(raw_img.getdata())
        total = len(raw_arr)
        if total == 0:
            return False
        dark_count = sum(1 for r, g, b in raw_arr if r < 180 and g < 180 and b < 180)
        return (dark_count / total) >= dark_threshold


    def get_lines(self, page: fitz.Page) -> List[Line]:
        """Extract text lines with bold detection, skipping sidebars and clustering into visual rows."""
        lines: List[Line] = []
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_parts = []
                for i, s in enumerate(spans):
                    txt = s.get("text", "")
                    if not txt:
                        continue
                    txt = map_font_glyphs(txt, s.get("font", ""))
                    if i > 0 and line_parts:
                        prev_s = spans[i - 1]
                        if prev_s.get("text", "").strip():
                            sz_ratio = s["size"] / prev_s["size"] if prev_s.get("size", 0) > 0 else 1.0
                            y_diff = prev_s["bbox"][3] - s["bbox"][3]
                            if 0.5 < sz_ratio < 0.88 and y_diff > 1.5:
                                if all(c in SUPERSCRIPT_MAP or c.isspace() for c in txt.strip()):
                                    txt = "".join(SUPERSCRIPT_MAP.get(c, c) for c in txt.strip())
                    line_parts.append(txt)
                text = "".join(line_parts)
                # Collapse space before superscript if any: e.g. '5t ²' -> '5t²'
                text = re.sub(r"([a-zA-Z0-9])\s+([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ])", r"\1\2", text)
                if not text.strip():
                    continue
                bbox = fitz.Rect(line["bbox"])
                if bbox.x0 >= self.content_max_x:
                    continue  # Examiner-use sidebar
                if (bbox.y1 - bbox.y0) > MAX_LINE_HEIGHT:
                    continue  # Rotated page-tall margin text or watermark run
                bold = any("Bold" in s.get("font", "") for s in line.get("spans", []))
                lines.append(Line(text=text, bbox=bbox, bold=bold, width=bbox.width))
        
        # Cluster lines into visual rows (chaining y0 gap <= 2.0pt) before sorting by x0
        lines.sort(key=lambda l: l.bbox.y0)
        rows: List[List[Line]] = []
        for ln in lines:
            if not rows:
                rows.append([ln])
            else:
                last_in_row = rows[-1][-1]
                row_y0 = min(item.bbox.y0 for item in rows[-1])
                if abs(ln.bbox.y0 - row_y0) <= 2.5 or (ln.bbox.y0 - last_in_row.bbox.y0 <= 2.0):
                    rows[-1].append(ln)
                else:
                    rows.append([ln])

        clustered_lines: List[Line] = []
        for r in rows:
            r.sort(key=lambda l: l.bbox.x0)
            clustered_lines.extend(r)
        return clustered_lines

    def get_question_anchors(self, lines: List[Line], min_question: int = 1) -> List[Tuple[int, float]]:
        """[(question_number, y0), ...] using left-margin layout geometry."""
        anchors: List[Tuple[int, float]] = []
        for ln in lines:
            # Cambridge question numbers sit in the margin gutter (~38 to 60 pt)
            if ln.bbox.x0 < 38.0 or ln.bbox.x0 >= self.left_margin_x or ln.bbox.y0 <= self.header_y or ln.bbox.y0 >= 780:
                continue
            stripped = ln.text.strip()
            if stripped.isdigit() and len(stripped) <= 2:
                val = int(stripped)
                if val >= min_question:
                    anchors.append((val, ln.bbox.y0))
            else:
                m = re.match(r"^(\d{1,2})\s+(\S.*)$", stripped)
                if m:
                    q_val = int(m.group(1))
                    if q_val < min_question:
                        continue
                    rest = m.group(2).strip()
                    # Question parts like "9 (a)", "10 (b)" or "11 (i)" are valid question anchors, not units
                    if rest.startswith("(") or re.match(r"^[a-z]\)", rest, re.IGNORECASE):
                        anchors.append((q_val, ln.bbox.y0))
                        continue

                    # Filter out units / axis ticks (e.g. "1 cm", "2 mm", "10 kg", "20 minutes.")
                    tokens = re.split(r"[\s/]+", rest)
                    first_tok = tokens[0].rstrip(".,;:!?")
                    # If first token is 'a' or 'A', distinguish English article ("A student", "A car") from unit Amperes
                    if first_tok.lower() == "a" and len(tokens) > 1 and tokens[1][0].isalpha():
                        anchors.append((q_val, ln.bbox.y0))
                        continue

                    if first_tok.lower() in UNIT_TOKENS:
                        continue
                    anchors.append((q_val, ln.bbox.y0))
        anchors.sort(key=lambda a: a[1])
        return anchors

    def get_captions(self, lines: List[Line]) -> List[Dict[str, Any]]:
        """Find caption lines matching 'Fig. N.M' or 'Table N.M' with optional parenthesized note."""
        caps: List[Dict[str, Any]] = []
        for ln in lines:
            m = self.caption_pattern.match(ln.text.strip())
            if m:
                tag = m.group(1).lower()
                kind = "table" if "tab" in tag else "fig"
                caps.append({
                    "q_from_caption": int(m.group(2)),
                    "sub": int(m.group(3)),
                    "bbox": ln.bbox,
                    "label": ln.text.strip(),
                    "kind": kind,
                })
        caps.sort(key=lambda c: c["bbox"].y0)
        return caps

    def cluster_rects(
        self,
        rects: List[fitz.Rect],
        pad: float = 8.0,
        pad_x: Optional[float] = None,
        pad_y: Optional[float] = None,
        max_iter: int = 12,
    ) -> List[fitz.Rect]:
        """Merge nearby bounding boxes into clusters using safe union."""
        out_rects = [fitz.Rect(r) for r in rects]
        changed = True
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1
            merged: List[fitz.Rect] = []
            used = [False] * len(out_rects)
            for i, r in enumerate(out_rects):
                if used[i]:
                    continue
                base = fitz.Rect(r)
                for j in range(i + 1, len(out_rects)):
                    if used[j]:
                        continue
                    r2 = out_rects[j]
                    if rects_touch(base, r2, pad=pad, pad_x=pad_x, pad_y=pad_y):
                        base = rect_union(base, r2)
                        used[j] = True
                        changed = True
                used[i] = True
                merged.append(base)
            out_rects = merged
        return out_rects

    def find_upper_bound(self, caption: Dict[str, Any], lines: List[Line], page_top: float = 40.0) -> float:
        """
        Walk upward from caption to nearest preceding sentence, question part, or caption.
        Chains through short wrapped-tail lines instead of stopping at narrow lines.
        """
        above = [ln for ln in lines if ln.bbox.y1 <= caption["bbox"].y0 + 0.5]
        above.sort(key=lambda l: l.bbox.y0)
        n = len(above)
        i = n - 1
        while i >= 0:
            ln = above[i]
            t = ln.text.strip()
            if self.caption_pattern.match(t):
                return ln.bbox.y1
            if ln.bbox.x0 <= self.prose_max_x0:
                if ln.width >= self.prose_min_width or t.endswith((".", ":", "?")) or _VIS_PART_RE.match(t) or PART_LABEL.match(t):
                    return ln.bbox.y1
                # Check if this short line is a wrapped tail of a real paragraph
                j = i - 1
                while j >= 0:
                    prev = above[j]
                    gap = above[j + 1].bbox.y0 - prev.bbox.y1
                    if gap > self.prose_chain_gap or prev.bbox.x0 > self.prose_max_x0:
                        break
                    prev_t = prev.text.strip()
                    if prev.width >= self.prose_min_width or prev_t.endswith((".", ":", "?")) or _VIS_PART_RE.match(prev_t):
                        return ln.bbox.y1
                    j -= 1
            i -= 1
        return page_top

    def find_lower_bound(self, caption: Dict[str, Any], lines: List[Line], page_bottom: float = 780.0) -> float:
        """
        Walk downward from table caption to nearest subsequent question part, prose line, marks, or caption.
        """
        below = [ln for ln in lines if ln.bbox.y0 >= caption["bbox"].y1 - 0.5]
        below.sort(key=lambda l: l.bbox.y0)
        for ln in below:
            t = ln.text.strip()
            if self.caption_pattern.match(t):
                return ln.bbox.y0
            if PART_LABEL.match(t) or re.match(r"^\s*\[\d+\]\s*$", t) or re.match(r"^\[Total:", t, re.I):
                return ln.bbox.y0
            if ln.bbox.x0 <= self.left_margin_x and (t.isdigit() or re.match(r"^\d{1,2}\s+\S", t)):
                return ln.bbox.y0
            if ln.bbox.x0 <= self.prose_max_x0 and ln.width >= self.prose_min_width:
                return ln.bbox.y0
        return page_bottom

    def get_caption_bounds(
        self,
        caption: Dict[str, Any],
        lines: List[Line],
        page_top: float = 40.0,
        page_bottom: float = 780.0,
    ) -> Tuple[float, float]:
        """
        Returns (upper_y, lower_y) for a caption region based on whether it is a Fig or Table.
        - Fig captions sit below their artwork (walk upward for upper bound).
        - Table captions sit above their grid (walk downward for lower bound).
        """
        kind = caption.get("kind", "fig")
        if kind == "table":
            upper = caption["bbox"].y0
            lower = self.find_lower_bound(caption, lines, page_bottom)
        else:
            upper = self.find_upper_bound(caption, lines, page_top)
            lower = caption["bbox"].y1
        return (upper, lower)

    def render_stitched(
        self,
        page: fitz.Page,
        boxes: List[fitz.Rect],
    ) -> Tuple[Image.Image, fitz.Rect]:
        """
        Render boxes as separate content bands across blank gaps, stitching
        them into one composite image without bloated answer spaces.
        """
        boxes = sorted([fitz.Rect(b) for b in boxes], key=lambda r: r.y0)
        bands = [fitz.Rect(boxes[0])]
        for b in boxes[1:]:
            if b.y0 - bands[-1].y1 > self.gap_break:
                bands.append(fitz.Rect(b))
            else:
                bands[-1] = rect_union(bands[-1], b)

        x0 = max(min(b.x0 for b in bands) - self.pad, 0)
        x1 = min(max(b.x1 for b in bands) + self.pad, page.rect.x1)

        strips = []
        for i, b in enumerate(bands):
            top_pad = 2 if i == 0 else self.pad  # avoid revealing excluded prose line
            clip = fitz.Rect(x0, b.y0 - top_pad, x1, b.y1 + self.pad) & page.rect
            pix = page.get_pixmap(clip=clip, dpi=self.dpi)
            strips.append(Image.open(io.BytesIO(pix.tobytes("png"))))

        max_w = max(s.width for s in strips)
        total_h = sum(s.height for s in strips) + self.stitch_gap_px * (len(strips) - 1)
        canvas = Image.new("RGB", (max_w, total_h), "white")
        y = 0
        for s in strips:
            canvas.paste(s, (0, y))
            y += s.height + self.stitch_gap_px
        bbox_out = fitz.Rect(x0, bands[0].y0, x1, bands[-1].y1)
        return canvas, bbox_out

    # ── MCQ Paper Support ─────────────────────────────────────────────────────

    def _is_mcq_paper(self, doc: fitz.Document, pdf_path: Optional[Path] = None) -> bool:
        """
        Return True when this is an MCQ-style paper (e.g. 5054/11, 5070/11, etc.).

        Detection heuristics:
        1. Non-QP check: If it's a mark scheme or contains 'mark scheme' -> False.
        2. Caption check: If any page contains 'Fig. N.M' / 'Table N.M' captions -> False
           (Cambridge MCQ papers never use Fig. N.M captions).
        3. Written paper indicators: If recurring bracketed marks '[1]', '[2]', '[Total:',
           or 'Answer (a)' lines are found -> False (this is a written exam, e.g. Maths 4024).
        4. Option evidence: Scans questions for recurring uppercase options {A, B, C, D}
           using MCQ_ANSWER_LABELS. Requires actual option evidence before classifying as MCQ.
        """
        # Cover page text inspection
        text_p0 = doc[0].get_text().lower() if len(doc) > 0 else ""
        text_p1 = doc[1].get_text().lower() if len(doc) > 1 else ""
        cover_text = text_p0 + " " + text_p1

        if "mark scheme" in cover_text or (pdf_path and "_ms_" in pdf_path.name.lower()):
            return False

        caption_count = 0
        mark_brackets_count = 0
        mcq_questions_with_options = 0

        pages_to_check = min(8, len(doc))
        for pno in range(pages_to_check):
            page = doc[pno]
            lines = self.get_lines(page)
            captions = self.get_captions(lines)
            if captions:
                caption_count += len(captions)

            # Check for written paper marks like [1], [2], [Total: or Answer (a) lines
            for ln in lines:
                t = ln.text.strip()
                if re.search(r"\[\s*\d+\s*\]", t) or re.search(r"\[\s*Total", t, re.I) or t.startswith("Answer ("):
                    mark_brackets_count += 1

            # Check MCQ question anchors and options
            anchors = self.get_mcq_question_anchors(lines)
            if anchors:
                page_bottom = page.rect.y1 - 20
                for i, (qnum, q_y0, q_y1) in enumerate(anchors):
                    span_bottom = anchors[i + 1][1] if i + 1 < len(anchors) else page_bottom
                    opts_found = set()
                    for ln in lines:
                        if q_y0 <= ln.bbox.y0 < span_bottom:
                            t = ln.text.strip()
                            if t in MCQ_ANSWER_LABELS:
                                opts_found.add(t)
                            elif len(t) > 2 and t[0] in MCQ_ANSWER_LABELS and t[1] in {" ", ".", ")"}:
                                opts_found.add(t[0])
                    if MCQ_ANSWER_LABELS.issubset(opts_found):
                        mcq_questions_with_options += 1

        # Captioned papers are not standard uncaptioned MCQ papers
        if caption_count > 0:
            return False

        # If paper has multiple written question mark allocations [1], [2], it's not MCQ
        if mark_brackets_count >= 3:
            return False

        # Strong evidence: multiple questions with full A, B, C, D options
        if mcq_questions_with_options >= 2:
            return True

        # Cover page explicitly says 'multiple choice' and at least 1 question has A, B, C, D
        if "multiple choice" in cover_text and mcq_questions_with_options >= 1:
            return True

        return False

    def get_mcq_question_anchors(self, lines: List[Line]) -> List[Tuple[int, float, float]]:
        """
        Return [(q_number, y0, y1), ...] for MCQ-style question starts.

        MCQ question lines sit at x0 ≈ 50 in two forms:
        - "5 Two men..." or "10 Five blocks..." — number prepended to text on same line
        - "1"  alone, with question text on the very next line at x0=71

        Both forms are detected. Duplicates (same question number) are removed,
        keeping the earliest occurrence.
        """
        seen: Set[int] = set()
        anchors: List[Tuple[int, float, float]] = []
        for ln in lines:
            if ln.bbox.x0 > MCQ_Q_X0_MAX or ln.bbox.y0 <= self.header_y:
                continue
            stripped = ln.text.strip()
            m = MCQ_Q_PATTERN.match(stripped) or MCQ_Q_SOLO.match(stripped)
            if m:
                qnum = int(m.group(1))
                if qnum not in seen:
                    seen.add(qnum)
                    anchors.append((qnum, ln.bbox.y0, ln.bbox.y1))
        anchors.sort(key=lambda a: a[1])
        return anchors

    def _extract_mcq_page(
        self,
        page: fitz.Page,
        pno: int,
        doc: fitz.Document,
        output_dir: Path,
        manifest: List[Dict[str, Any]],
        last_seen_question: Optional[int],
    ) -> Optional[int]:
        """
        Extract diagrams from one MCQ page.

        Strategy:
        - Parse MCQ question anchors (the "NN text..." or bare "N" lines at x0 ≈ 50).
        - Filter out vector grid boxes that are text-only answer tables (tables of
          prose options like A, B, C, D).
        - For each question span [q_y0 ... next_q_y0], assign remaining diagram clusters.
        - Group vertically close clusters (< 35pt gap) into visual diagram groups.
        - Render each visual diagram group as an independent image (e.g. mcq_1, mcq_2).
        - Expand crop bounding box to include all associated text labels (e.g. axis labels,
          component labels like 'resistance wire', 'brass', 'iron', 'plastic', 'air').

        Returns the last question number seen on this page.
        """
        lines = self.get_lines(page)
        mcq_anchors = self.get_mcq_question_anchors(lines)
        if mcq_anchors:
            last_seen_question = mcq_anchors[-1][0]

        # Build vector clusters on this page (excluding thin full-width page divider / footer lines)
        raw_rects = [
            d["rect"] for d in page.get_drawings()
            if max(d["rect"].width, d["rect"].height) > self.min_stroke_len
            and not (min(d["rect"].width, d["rect"].height) < 2.0 and max(d["rect"].width, d["rect"].height) > 300)
        ]
        all_clusters = [
            r for r in self.cluster_rects(raw_rects)
            if rect_area(r) >= MCQ_MIN_CLUSTER_AREA
            or (max(r.width, r.height) >= MCQ_MIN_CLUSTER_DIM and min(r.width, r.height) >= 5.0)
        ]

        if not all_clusters:
            return last_seen_question

        # All vector clusters in MCQ pages represent either diagrams or option/data tables.
        # Both must be extracted so questions with tables (where text extraction leaves
        # option text empty) have visual representations.
        filtered_clusters = all_clusters

        if not filtered_clusters:
            return last_seen_question

        # Build question spans: (q_number, span_y_top, span_y_bottom)
        page_bottom = page.rect.y1 - 20
        spans: List[Tuple[int, float, float]] = []
        for i, (qnum, q_y0, q_y1) in enumerate(mcq_anchors):
            if i + 1 < len(mcq_anchors):
                span_bottom = mcq_anchors[i + 1][1]
            else:
                span_bottom = page_bottom
            spans.append((qnum, q_y0, span_bottom))

        # Assign each cluster to question span
        assigned: Dict[int, List[fitz.Rect]] = {}
        for cluster in filtered_clusters:
            mid_y = (cluster.y0 + cluster.y1) / 2.0
            for qnum, span_top, span_bottom in spans:
                if span_top <= mid_y <= span_bottom:
                    assigned.setdefault(qnum, []).append(cluster)
                    break
            else:
                if spans and last_seen_question is not None:
                    nearest_q = min(spans, key=lambda s: abs(s[1] - cluster.y0))[0]
                    assigned.setdefault(nearest_q, []).append(cluster)

        # Process each question's clusters
        for qnum, q_clusters in sorted(assigned.items()):
            q_clusters.sort(key=lambda r: r.y0)

            # Group clusters that are close vertically (< 35pt gap) into visual diagram units
            groups: List[List[fitz.Rect]] = []
            for c in q_clusters:
                if not groups:
                    groups.append([c])
                else:
                    last_group = groups[-1]
                    max_y1 = max(r.y1 for r in last_group)
                    if c.y0 - max_y1 < 35.0:
                        last_group.append(c)
                    else:
                        groups.append([c])

            for grp in groups:
                grp_y0 = min(r.y0 for r in grp)
                grp_y1 = max(r.y1 for r in grp)
                grp_x0 = min(r.x0 for r in grp)
                grp_x1 = max(r.x1 for r in grp)

                # Find associated text lines (labels, axis titles, component names)
                label_boxes: List[fitz.Rect] = []
                for ln in lines:
                    # Exclude question headers / full prose question lines
                    if ln.bbox.x0 <= 60 and ln.text.strip().split()[0].isdigit():
                        continue
                    if ln.bbox.x0 <= 80 and ln.width > 200:
                        continue
                    # Proximity checks (generous bounds so labels, arrow annotations, and axis titles are captured without being cut tight)
                    if grp_y0 - 25 <= ln.bbox.y0 and ln.bbox.y1 <= grp_y1 + 35:
                        if ln.bbox.x0 <= grp_x1 + 90 and ln.bbox.x1 >= grp_x0 - 90:
                            label_boxes.append(ln.bbox)

                all_boxes = grp + label_boxes
                canvas, bbox_out = self.render_stitched(page, all_boxes)

                sub_idx = len([e for e in manifest if e.get("question") == qnum]) + 1
                fname = f"p{pno + 1}_q{qnum}_question_mcq_{sub_idx}.png"
                save_path = output_dir / fname
                canvas.save(str(save_path))

                entry = {
                    "page": pno + 1,
                    "question": qnum,
                    "target_type": "question",
                    "target_label": f"Q{qnum}",
                    "part": None,
                    "caption": None,
                    "bbox": [round(v, 1) for v in bbox_out],
                    "filename": fname,
                    "file": fname,
                    "relative_path": f"{output_dir.name}/{fname}",
                }
                manifest.append(entry)
                logger.info(f"  Extracted (MCQ): {fname} (Page {pno + 1}, Q{qnum})")

        return last_seen_question

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        force_mcq: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:

        """
        Extract all visual figures and diagrams from PDF into output_dir.

        Returns:
            List of manifest entries for all extracted visuals.
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        manifest: List[Dict[str, Any]] = []
        last_seen_question: Optional[int] = None
        last_seen_grid_part: Optional[Tuple[str, str, Optional[int], int]] = None

        logger.info(f"CambridgeVisualExtractor: extracting visuals from {pdf_path.name} (pages: {len(doc)})")

        # Detect MCQ papers (11/12/13) up front — they have no Fig. N.M captions
        # and use a different question-anchor scheme.
        if force_mcq is not None:
            is_mcq = force_mcq
        else:
            is_mcq = self._is_mcq_paper(doc, pdf_path)
        if is_mcq:
            logger.info("  Detected MCQ paper format — using question-span extractor")

        for pno in range(len(doc)):
            page = doc[pno]

            # MCQ papers: delegate entirely to the question-span extractor
            if is_mcq:
                last_seen_question = self._extract_mcq_page(
                    page, pno, doc, output_dir, manifest, last_seen_question
                )
                continue

            lines = self.get_lines(page)
            captions = self.get_captions(lines)
            first_cap_q = captions[0].get("q_from_caption") if captions else None
            # Skip question anchors on cover page (pno == 0) — Cambridge questions never start on the cover page
            if pno == 0:
                anchors = []
                first_cap_q = None
            else:
                anchors = self.get_question_anchors(lines, min_question=last_seen_question or 1)
                if anchors:
                    if last_seen_grid_part and anchors[0][0] > (last_seen_grid_part[2] or 0):
                        last_seen_grid_part = None
                    last_seen_question = anchors[-1][0]
                elif first_cap_q:
                    if last_seen_grid_part and first_cap_q > (last_seen_grid_part[2] or 0):
                        last_seen_grid_part = None
                    last_seen_question = max(last_seen_question or 0, first_cap_q)

            # Build question-scoped part anchors: [(q_num, part_label_str, target_type, y0), ...]
            # Scoped strictly by question boundary so parts from question N never bleed into question N+1.
            part_anchors: List[Tuple[Optional[int], str, str, float]] = []
            _curr_q: Optional[int] = anchors[0][0] if anchors else (first_cap_q or last_seen_question)
            _current_letter: Optional[str] = None
            _current_subpart: Optional[str] = None

            def _get_active_q_for_y(y: float) -> Optional[int]:
                active_q = anchors[0][0] if anchors else (first_cap_q or last_seen_question)
                for qn, qy in anchors:
                    if qy <= y:
                        active_q = qn
                    else:
                        break
                return active_q

            for ln in lines:
                active_q_for_ln = _get_active_q_for_y(ln.bbox.y0)
                if active_q_for_ln != _curr_q:
                    _curr_q = active_q_for_ln
                    _current_letter = None
                    _current_subpart = None

                sub_num_m = _VIS_SUB_NUM_RE.match(ln.text.strip()) if _current_subpart else None
                pm = _VIS_PART_RE.match(ln.text.strip()) if not sub_num_m else None
                if pm:
                    comp_l, comp_r, ltr, rom = pm.group(1), pm.group(2), pm.group(3), pm.group(4)
                    if comp_l and comp_r:
                        _current_letter = comp_l.lower()
                        _current_subpart = f"{comp_l.lower()}({comp_r.lower()})"
                        part_anchors.append((_curr_q, _current_subpart, "sub_part", ln.bbox.y0))
                    elif ltr:
                        _current_letter = ltr.lower()
                        _current_subpart = None
                        part_anchors.append((_curr_q, ltr.lower(), "part", ln.bbox.y0))
                    elif rom and _current_letter:
                        _current_subpart = f"{_current_letter}({rom.lower()})"
                        part_anchors.append((_curr_q, _current_subpart, "sub_part", ln.bbox.y0))
                    else:
                        continue
                elif sub_num_m and _current_subpart:
                    num = sub_num_m.group(1) or sub_num_m.group(2)
                    _ssp_str = f"{_current_subpart}({num})"
                    part_anchors.append((_curr_q, _ssp_str, "sub_part", ln.bbox.y0))

                if GRID_PROMPT_PATTERN.search(ln.text):
                    tgt_t = "sub_part" if _current_subpart else ("part" if _current_letter else "question")
                    tgt_lbl = _current_subpart or _current_letter or f"Q{_curr_q}"
                    last_seen_grid_part = (tgt_t, tgt_lbl, _curr_q, pno)

            def _resolve_part_target(q_target: Optional[int], target_y: float) -> Tuple[str, str]:
                """Resolve (target_type, target_label) scoped strictly to q_target's own parts."""
                best_label = f"Q{q_target}"
                best_type = "question"
                q_parts = [(plabel, ptype, py0) for pq, plabel, ptype, py0 in part_anchors if pq == q_target]
                for plabel, ptype, py0 in q_parts:
                    if py0 <= target_y + 10:
                        best_label, best_type = plabel, ptype
                    else:
                        break
                return best_type, best_label

            # Content printable limits
            PRINT_X0, PRINT_X1 = 35.0, 540.0
            PRINT_Y0, PRINT_Y1 = 45.0, 785.0

            # 0. Pre-scan for graph grid meshes (e.g. millimeter or coordinate grids).
            # Individual grid lines can be 300-700pt long and <1pt thin, which would otherwise
            # be dropped by page-spanning line or margin filters.
            grid_boxes: List[fitz.Rect] = []
            grid_line_indices: Set[int] = set()

            raw_drawings_list = page.get_drawings(extended=True)
            h_cands: List[Tuple[int, fitz.Rect]] = []
            v_cands: List[Tuple[int, fitz.Rect]] = []
            for d_idx, d in enumerate(raw_drawings_list):
                if d.get("type") in ("clip", "group"):
                    continue
                raw_r = d.get("rect")
                if not raw_r:
                    continue
                r_c = fitz.Rect(raw_r)
                if r_c.height == 0:
                    r_c = fitz.Rect(r_c.x0, r_c.y0 - 0.25, r_c.x1, r_c.y1 + 0.25)
                if r_c.width == 0:
                    r_c = fitz.Rect(r_c.x0 - 0.25, r_c.y0, r_c.x1 + 0.25, r_c.y1)
                r_c = r_c & page.rect
                if r_c.is_empty:
                    continue
                if r_c.height <= 2.0 and r_c.width >= 40.0 and r_c.y0 >= 55.0 and r_c.y1 <= PRINT_Y1 and r_c.x0 >= PRINT_X0 and r_c.x1 <= PRINT_X1:
                    h_cands.append((d_idx, r_c))
                elif r_c.width <= 2.0 and r_c.height >= 40.0 and r_c.x0 >= PRINT_X0 and r_c.x1 <= 520.0 and r_c.y0 >= 55.0 and r_c.y1 <= PRINT_Y1:
                    v_cands.append((d_idx, r_c))

            if len(h_cands) >= 6 and len(v_cands) >= 6:
                adj_h: Dict[int, List[int]] = {i: [] for i, _ in h_cands}
                adj_v: Dict[int, List[int]] = {j: [] for j, _ in v_cands}
                for i, hr in h_cands:
                    for j, vr in v_cands:
                        if hr.x0 - 2.0 <= vr.x0 <= hr.x1 + 2.0 and vr.y0 - 2.0 <= hr.y0 <= vr.y1 + 2.0:
                            adj_h[i].append(j)
                            adj_v[j].append(i)

                visited_h: Set[int] = set()
                visited_v: Set[int] = set()
                for i, _ in h_cands:
                    if i in visited_h:
                        continue
                    q_h = [i]
                    q_v: List[int] = []
                    comp_h = {i}
                    comp_v: Set[int] = set()
                    visited_h.add(i)
                    while q_h or q_v:
                        while q_h:
                            curr_h = q_h.pop()
                            for n_v in adj_h[curr_h]:
                                if n_v not in comp_v:
                                    comp_v.add(n_v)
                                    visited_v.add(n_v)
                                    q_v.append(n_v)
                        while q_v:
                            curr_v = q_v.pop()
                            for n_h in adj_v[curr_v]:
                                if n_h not in comp_h:
                                    comp_h.add(n_h)
                                    visited_h.add(n_h)
                                    q_h.append(n_h)

                    if len(comp_h) >= 6 and len(comp_v) >= 6:
                        comp_rects = [r for idx, r in h_cands if idx in comp_h] + [r for idx, r in v_cands if idx in comp_v]
                        grid_bbox = fitz.Rect(
                            min(r.x0 for r in comp_rects),
                            min(r.y0 for r in comp_rects),
                            max(r.x1 for r in comp_rects),
                            max(r.y1 for r in comp_rects),
                        )
                        grid_boxes.append(grid_bbox)
                        grid_line_indices.update(comp_h)
                        grid_line_indices.update(comp_v)

            # 1. Filter printable drawings (drop margin text, vertical stripes, crop marks).
            #    Use extended=True so PyMuPDF interleaves {"type": "clip"} entries that carry
            #    the active scissor rect. We intersect every path's raw rect with the scissor
            #    before clustering — without this, bezier control points outside the clipped
            #    region inflate rects to negative/far-out coordinates, causing them to merge
            #    with the margin strip and get silently dropped (root cause of Fig 3.1 bug).
            filtered_drawings = []
            current_scissor: Optional[fitz.Rect] = None
            for d_idx, d in enumerate(raw_drawings_list):
                if d_idx in grid_line_indices:
                    continue
                d_type = d.get("type")
                if d_type in ("clip", "group"):
                    if d_type == "clip":
                        scissor = d.get("scissor")
                        current_scissor = fitz.Rect(scissor) if scissor else None
                    continue
                raw_r = d.get("rect")
                if raw_r is None:
                    continue
                r = fitz.Rect(raw_r)
                # Give 1-D lines (e.g. horizontal/vertical graph grid lines) nominal thickness
                # so PyMuPDF's is_empty does not drop them and rect clipping works properly
                if r.height == 0:
                    r = fitz.Rect(r.x0, r.y0 - 0.25, r.x1, r.y1 + 0.25)
                if r.width == 0:
                    r = fitz.Rect(r.x0 - 0.25, r.y0, r.x1 + 0.25, r.y1)
                # Clip to the active scissor first (before intersecting with page.rect)
                if current_scissor is not None and not current_scissor.is_empty:
                    clipped_r = r & current_scissor
                    if clipped_r.is_empty:
                        # Drawing is completely outside the scissor, indicating local clip scope has ended in PDF
                        current_scissor = None
                    else:
                        r = clipped_r
                r = r & page.rect
                if r.is_empty or max(r.width, r.height) <= self.min_stroke_len:
                    continue
                # Filter full-page or near full-page canvas/background drawings
                if r.width >= 0.8 * page.rect.width and r.height >= 0.8 * page.rect.height:
                    continue
                # Left / right margin gutter (e.g. DO NOT WRITE IN THIS MARGIN sidebar)
                if r.x1 <= PRINT_X0 or r.x0 >= PRINT_X1:
                    continue
                # Rotated page-tall margin lines or watermark runs
                if r.height > 400 and r.width < 50:
                    continue
                # Header region, header boxes, or corner registration marks
                if r.y1 <= 60.0 or (r.y0 <= 60.0 and r.width > 300.0):
                    continue
                # Footer region or corner registration marks
                if r.y0 >= PRINT_Y1:
                    continue
                if (r.y1 <= PRINT_Y0 or r.y0 >= PRINT_Y1) and (r.x1 <= 65.0 or r.x0 >= 520.0):
                    continue
                # Page-spanning dividing/separator rules
                if r.width > 350.0 and r.height < 3.0:
                    continue
                filtered_drawings.append(r)

            # Include aggregated grid bounding boxes directly
            filtered_drawings.extend(grid_boxes)

            # 2. Filter embedded images
            raster_xref_map = self._build_raster_xref_map(page)
            filtered_rasters = []
            for img in page.get_image_info(xrefs=True):
                r = fitz.Rect(img["bbox"]) & page.rect
                if r.x1 <= PRINT_X0 or r.x0 >= PRINT_X1 or r.y1 <= PRINT_Y0 or r.y0 >= PRINT_Y1:
                    continue
                # Exclude PapaCambridge diagonal ribbon in top-right corner
                if r.x0 >= 380.0 and r.y0 <= 20.0 and r.y1 <= 220.0 and r.x1 >= 570.0:
                    continue
                bbox_key = tuple(img["bbox"])
                xref = img.get("xref") or raster_xref_map.get(bbox_key)
                if xref and self._is_papacambridge_ribbon(doc, xref):
                    continue
                filtered_rasters.append(r)

            all_geo = [("vector", r) for r in filtered_drawings] + [("raster", r) for r in filtered_rasters]

            claimed = [False] * len(all_geo)
            overlaps_any_caption = [False] * len(all_geo)
            claimed_lines: Set[int] = set()

            if captions:
                bounds: List[Tuple[float, float]] = []
                for cap in captions:
                    upper, lower = self.get_caption_bounds(cap, lines, page_top=PRINT_Y0, page_bottom=PRINT_Y1)
                    bounds.append((upper, lower))

                # Mark geometry overlapping any caption band to prevent duplicate fallbacks
                for gi, (kind, r) in enumerate(all_geo):
                    for (u, l) in bounds:
                        if not (r.y1 < u or r.y0 > l):
                            overlaps_any_caption[gi] = True
                            break

                for ci, cap in enumerate(captions):
                    upper, lower = bounds[ci]
                    overlapping = [
                        cj for cj, (u2, l2) in enumerate(bounds)
                        if cj != ci and not (l2 < upper or u2 > lower)
                    ]

                    def nearest_caption_idx(x_center: float) -> int:
                        best, best_d = ci, abs(captions[ci]["bbox"].x0 - x_center)
                        for cj in overlapping:
                            d = abs(captions[cj]["bbox"].x0 - x_center)
                            if d < best_d:
                                best, best_d = cj, d
                        return best

                    claimed_boxes = [fitz.Rect(cap["bbox"])]

                    for gi, (kind, r) in enumerate(all_geo):
                        if claimed[gi]:
                            continue
                        overlap_y0 = max(r.y0, upper)
                        overlap_y1 = min(r.y1, lower)
                        if overlap_y1 > overlap_y0:
                            overlap_h = overlap_y1 - overlap_y0
                            # Major vertical overlap or enclosed in band
                            if overlap_h / max(r.height, 1.0) >= 0.25 or (r.y0 >= upper - 8 and r.y1 <= lower + 8):
                                if nearest_caption_idx(r.x0 + r.width / 2) == ci:
                                    clipped = r & fitz.Rect(PRINT_X0, upper, PRINT_X1, lower)
                                    if not clipped.is_empty:
                                        claimed_boxes.append(clipped)
                                        claimed[gi] = True

                    for li, ln in enumerate(lines):
                        if li in claimed_lines:
                            continue
                        t = ln.text.strip()
                        if (
                            ln.bbox.y0 >= upper - 0.5
                            and ln.bbox.y1 <= lower + 0.5
                            and not self.caption_pattern.match(t)
                        ):
                            # Exclude margin question numbers
                            if ln.bbox.x1 <= self.left_margin_x or (ln.bbox.x0 <= 60.0 and t.split()[0].isdigit()):
                                continue
                            # Exclude examiner sidebar
                            if ln.bbox.x0 >= self.content_max_x or "Examiner" in t:
                                continue
                            # Exclude mark allocations like [2], [3], [4], [Total: 10]
                            if MARK_INLINE_PATTERN.search(t):
                                continue
                            # Exclude question part headers like (a), (b), (i)
                            if _VIS_PART_RE.match(t) or PART_LABEL.match(t):
                                continue
                            # Exclude prose paragraph lines
                            if ln.bbox.x0 <= self.prose_max_x0 and (ln.width >= self.prose_min_width or t.endswith((".", ":", "?"))):
                                continue

                            if nearest_caption_idx(ln.bbox.x0 + ln.width / 2) == ci:
                                claimed_boxes.append(ln.bbox)
                                claimed_lines.add(li)

                    canvas, bbox_out = self.render_stitched(page, claimed_boxes)

                    qnum: Optional[int] = None
                    for qn, y in anchors:
                        if y <= cap["bbox"].y0:
                            qnum = qn

                    # Authoritative caption priority: Fig. X.Y or Table X.Y ground truth
                    cap_q = cap.get("q_from_caption")
                    if cap_q is not None:
                        if qnum is not None and qnum != cap_q:
                            logger.warning(
                                f"  [Sanity Override] Caption '{cap['label']}' specifies Q{cap_q}, "
                                f"overriding conflicting layout anchor Q{qnum}."
                            )
                        qnum = cap_q
                    elif qnum is None:
                        qnum = last_seen_question

                    if qnum is not None:
                        last_seen_question = max(last_seen_question or 0, qnum)

                    # ── Part association ──────────────────────────────────────────────
                    # Strategy 1: check if any part line on this page explicitly names
                    # this figure (e.g. "On Fig. 5.2, continue the ray…" → part b)
                    fig_ref_re = re.compile(
                        r"\bFig(?:ure)?\s*" + re.escape(cap["label"]),
                        re.IGNORECASE,
                    )
                    target_type: Optional[str] = None
                    target_label: Optional[str] = None

                    for ln2 in lines:
                        pm2 = _VIS_PART_RE.match(ln2.text.strip())
                        if pm2 and fig_ref_re.search(pm2.group(5) or ""):
                            comp_l2, comp_r2, ltr2, rom2 = pm2.group(1), pm2.group(2), pm2.group(3), pm2.group(4)
                            if comp_l2 and comp_r2:
                                target_type = "sub_part"
                                target_label = f"{comp_l2.lower()}({comp_r2.lower()})"
                            elif ltr2:
                                target_type = "part"
                                target_label = ltr2.lower()
                            elif rom2:
                                target_type = "sub_part"
                                target_label = f"{_current_letter}({rom2.lower()})" if _current_letter else rom2.lower()
                            break

                    # Strategy 2: vertical containment against question's own part anchors
                    if target_type is None:
                        target_type, target_label = _resolve_part_target(qnum, cap["bbox"].y0)

                    # Build discriminator tag: exactly one of "question", "part_{label}", "sub_part_{label}"
                    if target_type == "question":
                        tag_str = "question"
                    elif target_type == "part":
                        tag_str = f"part_{target_label}"
                    else:
                        tag_str = f"sub_part_{target_label}"

                    tag = "Table" if cap.get("kind") == "table" else "Fig"
                    clean_label = f"{tag}_{cap['q_from_caption']}_{cap['sub']}"
                    fname = f"p{pno + 1}_q{qnum}_{tag_str}_{clean_label}.png"
                    save_path = output_dir / fname
                    canvas.save(str(save_path))

                    entry = {
                        "page": pno + 1,
                        "question": qnum,
                        "target_type": target_type,
                        "target_label": target_label,
                        "part": target_label if target_type in ("part", "sub_part") else None,
                        "caption": cap["label"],
                        "bbox": [round(v, 1) for v in bbox_out],
                        "filename": fname,
                        "file": fname,
                        "relative_path": f"{output_dir.name}/{fname}",
                    }
                    manifest.append(entry)
                    logger.info(f"  Extracted: {fname} (Page {pno + 1}, Q{qnum}, {cap['label']}, target={tag_str})")

            # Fallback: unclaimed geometry with substantial 2-D footprint (skip cover page pno == 0)
            if pno > 0 and last_seen_question is not None:
                raw_drawings = [
                    r for gi, (kind, r) in enumerate(all_geo)
                    if not claimed[gi] and not overlaps_any_caption[gi] and kind == "vector"
                    and not (r.width > 350 and r.height < 3.0)  # horizontal ruling lines
                    and not (r.height > 350 and r.width < 3.0)  # vertical margin lines
                ]
                # Filter isolated 1-D micro vectors (fraction bars, minus signs, stray tick marks)
                unclaimed_vectors = []
                for r in raw_drawings:
                    if min(r.width, r.height) <= 1.0 and max(r.width, r.height) < 25.0:
                        if not any(d2 != r and rects_touch(r, d2, pad=35.0) for d2 in raw_drawings):
                            continue
                    unclaimed_vectors.append(r)

                if unclaimed_vectors:
                    # Identify prose lines to prevent paragraph text from bleeding into diagram crops
                    prose_indices: Set[int] = set()
                    for i, ln in enumerate(lines):
                        if ln.bbox.x0 <= 120.0 and ln.width >= 120.0:
                            prose_indices.add(i)
                        elif ln.bbox.x0 <= 120.0 and i > 0 and (i - 1) in prose_indices and (ln.bbox.y0 - lines[i - 1].bbox.y1 <= 22.0):
                            prose_indices.add(i)

                    # Bridge sparse diagram marks with nearby short text labels (e.g. compass bearings, points, dimensions)
                    short_labels = []
                    for i, ln in enumerate(lines):
                        if i in prose_indices:
                            continue
                        t = ln.text.strip()
                        if _VIS_PART_RE.match(t) or PART_LABEL.match(t):
                            continue
                        if t.startswith(("Answer", "Question", "©", "[")) or t.endswith("]") or (ln.bbox.x0 <= 60 and t.split()[0].isdigit()):
                            continue
                        if ln.bbox.x0 <= 120.0 and (t.endswith((".", ";")) or len(t) > 35):
                            continue
                        if ln.bbox.x0 < PRINT_X0 or ln.bbox.x1 > PRINT_X1 or ln.bbox.y0 < 60.0 or ln.bbox.y1 > PRINT_Y1:
                            continue
                        if ln.width < 120:
                            short_labels.append(ln.bbox)

                    combined_vecs = list(unclaimed_vectors)
                    for lbl in short_labels:
                        if any(rects_touch(lbl, v, pad_x=55.0, pad_y=25.0) for v in unclaimed_vectors):
                            combined_vecs.append(lbl)

                    clustered_fallbacks = self.cluster_rects(combined_vecs, pad_x=55.0, pad_y=20.0)
                    for r in clustered_fallbacks:
                        # Must be strictly within printable bounds and substantial
                        if r.x0 < PRINT_X0 or r.x1 > PRINT_X1 or r.y0 < 55.0 or r.y1 > PRINT_Y1:
                            continue
                        area = rect_area(r)
                        if area < 2500.0 or min(r.width, r.height) < 25.0:
                            continue

                        # Compute vector bounding box inside cluster to reject 1D lines (e.g. isolated fraction bars)
                        v_in_cluster = [v for v in unclaimed_vectors if rects_touch(v, r, pad=2.0)]
                        if not v_in_cluster:
                            continue
                        if all(min(v.width, v.height) <= 1.0 for v in v_in_cluster) and sum(max(v.width, v.height) for v in v_in_cluster) < 100.0:
                            continue
                        v_box = v_in_cluster[0]
                        for v in v_in_cluster[1:]:
                            v_box = rect_union(v_box, v)
                        # Allow 2D vector layouts or multi-stroke assemblies (like circuit diagrams, terminals, apparatus)
                        if min(v_box.width, v_box.height) < 5.0 and len(v_in_cluster) <= 2:
                            continue

                        # Expand to include Diagram / Figure caption directly below artwork
                        r_expanded = fitz.Rect(r)
                        for ln in lines:
                            t = ln.text.strip()
                            if DIAGRAM_CAPTION_PATTERN.match(t):
                                if abs((ln.bbox.x0 + ln.bbox.x1) / 2.0 - (r.x0 + r.x1) / 2.0) < r.width * 0.6:
                                    if 0 <= ln.bbox.y0 - r.y1 <= 30.0:
                                        r_expanded = rect_union(r_expanded, ln.bbox)
                        r = r_expanded

                        qnum = None
                        for qn, y in anchors:
                            if y <= r.y0 + 10:
                                qnum = qn
                        if qnum is None:
                            qnum = last_seen_question

                        # Part association: vertical containment against question's own part anchors
                        target_type, target_label = _resolve_part_target(qnum, r.y0)

                        is_grid = any(rects_touch(r, gbox, pad=5.0) for gbox in grid_boxes)
                        if (
                            is_grid
                            and target_type == "question"
                            and last_seen_grid_part is not None
                            and last_seen_grid_part[2] == qnum
                            and (pno - last_seen_grid_part[3] <= 2)
                        ):
                            target_type, target_label = last_seen_grid_part[0], last_seen_grid_part[1]
                            last_seen_grid_part = None

                        if target_type == "question":
                            tag_str = "question"
                        elif target_type == "part":
                            tag_str = f"part_{target_label}"
                        else:
                            tag_str = f"sub_part_{target_label}"

                        clip = (r + (-self.pad, -self.pad, self.pad, self.pad)) & page.rect
                        pix = page.get_pixmap(clip=clip, dpi=self.dpi)
                        if self._is_blank_image(pix):
                            continue

                        sub_cnt = len([e for e in manifest if e.get("question") == qnum and "diagram" in e.get("filename", "")]) + 1
                        fname = f"p{pno + 1}_q{qnum}_{tag_str}_diagram_{sub_cnt}.png"
                        save_path = output_dir / fname
                        pix.save(str(save_path))

                        entry = {
                            "page": pno + 1,
                            "question": qnum,
                            "target_type": target_type,
                            "target_label": target_label,
                            "part": target_label if target_type in ("part", "sub_part") else None,
                            "caption": "Diagram",
                            "bbox": [round(v, 1) for v in clip],
                            "filename": fname,
                            "file": fname,
                            "relative_path": f"{output_dir.name}/{fname}",
                        }
                        manifest.append(entry)
                        logger.info(f"  Extracted (diagram): {fname} (Page {pno + 1}, Q{qnum}, target={tag_str})")

            # Table pass: detect uncaptioned data tables via find_tables (skip cover page pno == 0)
            if pno > 0 and last_seen_question is not None:
                try:
                    tabs = page.find_tables()
                    page_extracted = [fitz.Rect(e["bbox"]) for e in manifest if e.get("page") == pno + 1]
                    for t in tabs.tables:
                        extracted_rows = t.extract()
                        if not extracted_rows or len(extracted_rows) < 2 or len(extracted_rows[0]) < 2:
                            continue
                        first_row_str = " ".join(str(c) for c in extracted_rows[0] if c)
                        if any(w in first_row_str.lower() for w in ["examiner's use", "candidate number", "centre number"]):
                            continue
                        t_rect = fitz.Rect(t.bbox)
                        if t_rect.y0 <= PRINT_Y0 or t_rect.x0 >= CONTENT_MAX_X or t_rect.x1 <= PRINT_X0 or t_rect.y1 >= PRINT_Y1:
                            continue
                        if rect_area(t_rect) < 1500.0:
                            continue
                        # Skip if already claimed or overlapping an extracted visual
                        if any(rects_touch(t_rect, eb, pad=10.0) for eb in page_extracted):
                            continue

                        qnum = None
                        for qn, y in anchors:
                            if y <= t_rect.y0 + 10:
                                qnum = qn
                        if qnum is None:
                            qnum = last_seen_question

                        # Part association: vertical containment against question's own part anchors
                        target_type, target_label = _resolve_part_target(qnum, t_rect.y0)

                        if target_type == "question":
                            tag_str = "question"
                        elif target_type == "part":
                            tag_str = f"part_{target_label}"
                        else:
                            tag_str = f"sub_part_{target_label}"

                        clip = (t_rect + (-self.pad, -self.pad, self.pad, self.pad)) & page.rect
                        pix = page.get_pixmap(clip=clip, dpi=self.dpi)
                        if self._is_blank_image(pix):
                            continue

                        sub_cnt = len([e for e in manifest if e.get("question") == qnum]) + 1
                        fname = f"p{pno + 1}_q{qnum}_{tag_str}_Table_{sub_cnt}.png"
                        save_path = output_dir / fname
                        pix.save(str(save_path))

                        entry = {
                            "page": pno + 1,
                            "question": qnum,
                            "target_type": target_type,
                            "target_label": target_label,
                            "part": target_label if target_type in ("part", "sub_part") else None,
                            "caption": "Table",
                            "bbox": [round(v, 1) for v in clip],
                            "filename": fname,
                            "file": fname,
                            "relative_path": f"{output_dir.name}/{fname}",
                        }
                        manifest.append(entry)
                        page_extracted.append(clip)
                        logger.info(f"  Extracted (table): {fname} (Page {pno + 1}, Q{qnum})")
                except Exception as e:
                    logger.debug(f"find_tables extraction failed on page {pno}: {e}")

        doc.close()

        # Post-extraction sanity validation: enforce caption ground truth and schema filenames
        manifest = self._validate_and_sanitize_manifest(manifest, output_dir)

        # Write manifest.json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"CambridgeVisualExtractor: successfully extracted {len(manifest)} visuals "
            f"into {output_dir.name}/ (manifest: {manifest_path.name})"
        )
        return manifest

    def _validate_and_sanitize_manifest(
        self, manifest: List[Dict[str, Any]], output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Post-extraction sanity validator and reconciler:
        1. Enforces that any captioned visual (Fig. X.Y or Table X.Y) has question == X.
        2. Ensures filename strictly adheres to canonical schema:
           p{page}_q{question}_{discriminator}_{label}.png
        3. Reconciles on-disk files if any rename is required.
        """
        modified = False
        for entry in manifest:
            cap = entry.get("caption") or ""
            current_q = entry.get("question")
            m_fig = re.search(r"Fig(?:ure)?\.?\s*(\d+)\.(\d+)", cap, re.I)
            m_tab = re.search(r"Table\s*(\d+)\.(\d+)", cap, re.I)
            cap_q = int(m_fig.group(1)) if m_fig else (int(m_tab.group(1)) if m_tab else None)

            # Ground-truth override: caption question number defines question
            if cap_q is not None and current_q != cap_q:
                logger.warning(
                    f"  [Manifest Sanity] Fixing question mismatch in {entry.get('filename')}: "
                    f"question was {current_q}, but caption '{cap}' specifies Q{cap_q}."
                )
                entry["question"] = cap_q
                modified = True

            # Reconcile discriminator tag: "question", "part_{label}", or "sub_part_{label}"
            target_type = entry.get("target_type")
            target_label = entry.get("target_label")
            part = entry.get("part")

            if target_type == "question":
                tag_str = "question"
            elif target_type == "part":
                tag_str = f"part_{target_label}"
            elif target_type == "sub_part":
                tag_str = f"sub_part_{target_label}"
            elif part and part not in ("question", "stem", ""):
                clean_p = re.sub(r"^(?:sub_)?part_", "", str(part))
                tag_str = f"part_{clean_p}"
            else:
                tag_str = "question"

            # Reconcile label tag: "Fig_X_Y", "Table_X_Y", or "diagram_{N}"
            if m_fig:
                clean_label = f"Fig_{m_fig.group(1)}_{m_fig.group(2)}"
            elif m_tab:
                clean_label = f"Table_{m_tab.group(1)}_{m_tab.group(2)}"
            else:
                old_fn = entry.get("filename", "")
                m_end = re.search(r"(diagram_\d+|table_\d+|Fig_[^.]+|Table_[^.]+)", old_fn, re.I)
                clean_label = m_end.group(1) if m_end else "diagram_1"

            expected_fname = f"p{entry['page']}_q{entry['question']}_{tag_str}_{clean_label}.png"
            old_fname = entry.get("filename")

            if old_fname != expected_fname:
                logger.info(f"  [Manifest Sanity] Renaming {old_fname} -> {expected_fname}")
                old_path = output_dir / old_fname if old_fname else None
                new_path = output_dir / expected_fname
                if old_path and old_path.exists() and old_path != new_path:
                    try:
                        old_path.rename(new_path)
                    except Exception as e:
                        logger.warning(f"  Failed to rename {old_path} to {new_path}: {e}")
                entry["filename"] = expected_fname
                entry["file"] = expected_fname
                entry["relative_path"] = f"{output_dir.name}/{expected_fname}"
                modified = True

        return manifest
