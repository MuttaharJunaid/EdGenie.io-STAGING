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

LEFT_MARGIN_X = 60.0       # question numbers sit left of this
HEADER_Y = 55.0            # page-number-in-header sits above this; ignore it
MAX_LINE_HEIGHT = 45.0     # guard against rotated page-tall margin text (e.g. 'DO NOT WRITE IN THIS MARGIN')
PROSE_MIN_WIDTH = 150.0    # a "real sentence" line is at least this wide
PROSE_MAX_X0 = 110.0       # ...and starts within the normal text column
PROSE_CHAIN_GAP = 20.0     # max vertical gap between lines of one paragraph
CONTENT_MAX_X = 522.0      # "For Examiner's Use" sidebar starts right of this
MIN_STROKE_LEN = 3.0       # drop sub-pixel path fragments, keep thin long lines
GAP_BREAK = 40.0           # blank vertical gap (pt) that splits a figure into bands
STITCH_GAP_PX = 16         # visual gap between stitched bands in the output
PART_LABEL = re.compile(r"^\s*\(([a-z]|i{1,3}v?|vi{0,3}|ix|x{0,3})\)\s*", re.IGNORECASE)

# MCQ paper detection & extraction constants
MCQ_Q_PATTERN = re.compile(r"^(\d{1,2})\s+\S")       # "5 Two men..." or "40 Which..."
MCQ_Q_SOLO = re.compile(r"^(\d{1,2})$")               # bare "1", "2" … "40" alone on line
MCQ_Q_X0_MAX = 58.0        # MCQ question numbers sit at x0 ≈ 50, never beyond 58
MCQ_ANSWER_LABELS = {"A", "B", "C", "D"}  # MCQ answer option labels
MCQ_MIN_CLUSTER_AREA = 2000.0  # Minimum vector cluster area to count as a real diagram
MCQ_MIN_CLUSTER_DIM = 30.0     # Min longest dimension to count as a real diagram


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


def rects_touch(a: fitz.Rect, b: fitz.Rect, pad: float = 0.0) -> bool:
    """Padded overlap test computed manually without PyMuPDF zero-area dropping."""
    return not (
        b.x1 < a.x0 - pad
        or b.x0 > a.x1 + pad
        or b.y1 < a.y0 - pad
        or b.y0 > a.y1 + pad
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
        image_infos = page.get_image_info()
        xobjects = page.get_images(full=True)  # same order as get_image_info()
        for img_info, xobj in zip(image_infos, xobjects):
            bbox_key = tuple(img_info["bbox"])
            xref_map[bbox_key] = xobj[0]  # xobj[0] is the xref
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
                text = "".join(s.get("text", "") for s in line.get("spans", []))
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

    def get_question_anchors(self, lines: List[Line]) -> List[Tuple[int, float]]:
        """[(question_number, y0), ...] using left-margin layout geometry."""
        anchors: List[Tuple[int, float]] = []
        for ln in lines:
            if ln.bbox.x0 >= self.left_margin_x or ln.bbox.y0 <= self.header_y or ln.bbox.y0 >= 780:
                continue
            stripped = ln.text.strip()
            if stripped.isdigit() and len(stripped) <= 2 and int(stripped) >= 1:
                anchors.append((int(stripped), ln.bbox.y0))
            else:
                m = re.match(r"^(\d{1,2})\s+\S", stripped)
                if m and int(m.group(1)) >= 1:
                    anchors.append((int(m.group(1)), ln.bbox.y0))
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

    def cluster_rects(self, rects: List[fitz.Rect], pad: float = 8.0, max_iter: int = 12) -> List[fitz.Rect]:
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
                    if rects_touch(base, r2, pad):
                        base = rect_union(base, r2)
                        used[j] = True
                        changed = True
                used[i] = True
                merged.append(base)
            out_rects = merged
        return out_rects

    def find_upper_bound(self, caption: Dict[str, Any], lines: List[Line], page_top: float = 40.0) -> float:
        """
        Walk upward from caption to nearest preceding sentence or caption.
        Chains through short wrapped-tail lines instead of stopping at narrow lines.
        """
        above = [ln for ln in lines if ln.bbox.y1 <= caption["bbox"].y0 + 0.5]
        above.sort(key=lambda l: l.bbox.y0)
        n = len(above)
        i = n - 1
        while i >= 0:
            ln = above[i]
            if self.caption_pattern.match(ln.text.strip()):
                return ln.bbox.y1
            if ln.bbox.x0 <= self.prose_max_x0:
                if ln.width >= self.prose_min_width:
                    return ln.bbox.y1
                # Check if this short line is a wrapped tail of a real paragraph
                j = i - 1
                while j >= 0:
                    prev = above[j]
                    gap = above[j + 1].bbox.y0 - prev.bbox.y1
                    if gap > self.prose_chain_gap or prev.bbox.x0 > self.prose_max_x0:
                        break
                    if prev.width >= self.prose_min_width:
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
        Return True when this is an MCQ-style paper (papers 11, 12, 13).

        Detection heuristics:
        1. Filename check: if filename stem ends with _11, _12, _13, _1, _qp_11, _qp_12, _qp_13, _qp_1.
        2. Layout check: No page in the first 8 pages contains any 'Fig. N.M' caption
           AND at least 2 pages contain MCQ question anchors (using both combined
           and split-line single-digit anchor patterns).
        """
        if pdf_path:
            stem = pdf_path.stem.lower()
            if any(stem.endswith(sfx) for sfx in ["_11", "_12", "_13", "_1", "_qp_11", "_qp_12", "_qp_13", "_qp_1"]):
                return True

        caption_count = 0
        mcq_q_page_count = 0

        for pno in range(min(8, len(doc))):
            page = doc[pno]
            lines = self.get_lines(page)
            captions = self.get_captions(lines)
            if captions:
                caption_count += len(captions)
            anchors = self.get_mcq_question_anchors(lines)
            if anchors:
                mcq_q_page_count += 1

        return caption_count == 0 and mcq_q_page_count >= 2

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
            if r.get_area() >= MCQ_MIN_CLUSTER_AREA
            or (max(r.width, r.height) >= MCQ_MIN_CLUSTER_DIM and min(r.width, r.height) >= 5.0)
        ]

        if not all_clusters:
            return last_seen_question

        # Text-density filter: only extract as diagram if text coverage < 8% (density < 0.08).
        # Clusters with density >= 0.08 are answer/data tables with prose options, not diagrams.
        text_blocks = page.get_text("blocks")
        filtered_clusters = []
        for c in all_clusters:
            c_area = c.get_area()
            text_area = 0.0
            for b in text_blocks:
                b_rect = fitz.Rect(b[:4])
                if rects_touch(c, b_rect):
                    intersection = c & b_rect
                    text_area += intersection.get_area()
            density = (text_area / c_area) if c_area > 0 else 0.0
            if density < 0.08:
                filtered_clusters.append(c)

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
                fname = f"p{pno + 1}_q{qnum}_mcq_{sub_idx}.png"
                save_path = output_dir / fname
                canvas.save(str(save_path))

                entry = {
                    "page": pno + 1,
                    "question": qnum,
                    "caption": None,
                    "bbox": [round(v, 1) for v in bbox_out],
                    "filename": fname,
                    "file": fname,
                    "relative_path": f"{output_dir.name}/{fname}",
                }
                manifest.append(entry)
                logger.info(f"  Extracted (MCQ): {fname} (Page {pno + 1}, Q{qnum})")

        return last_seen_question

    def extract(self, pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:

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

        logger.info(f"CambridgeVisualExtractor: extracting visuals from {pdf_path.name} (pages: {len(doc)})")

        # Detect MCQ papers (11/12/13) up front — they have no Fig. N.M captions
        # and use a different question-anchor scheme.
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
            anchors = self.get_question_anchors(lines)
            if anchors:
                last_seen_question = anchors[-1][0]
            captions = self.get_captions(lines)

            # Content printable limits
            PRINT_X0, PRINT_X1 = 35.0, 540.0
            PRINT_Y0, PRINT_Y1 = 45.0, 785.0

            # 1. Filter printable drawings (drop margin text, vertical stripes, crop marks)
            filtered_drawings = []
            for d in page.get_drawings():
                r = fitz.Rect(d["rect"]) & page.rect
                if max(r.width, r.height) <= self.min_stroke_len:
                    continue
                # Left / right margin gutter (e.g. DO NOT WRITE IN THIS MARGIN sidebar)
                if r.x1 <= PRINT_X0 or r.x0 >= PRINT_X1:
                    continue
                # Rotated page-tall margin lines or watermark runs
                if r.height > 400 and r.width < 50:
                    continue
                # Header / footer corner registration marks
                if (r.y1 <= PRINT_Y0 or r.y0 >= PRINT_Y1) and (r.x1 <= 65.0 or r.x0 >= 520.0):
                    continue
                filtered_drawings.append(r)

            # 2. Filter embedded images
            raster_xref_map = self._build_raster_xref_map(page)
            filtered_rasters = []
            for img in page.get_image_info():
                r = fitz.Rect(img["bbox"]) & page.rect
                if r.x1 <= PRINT_X0 or r.x0 >= PRINT_X1 or r.y1 <= PRINT_Y0 or r.y0 >= PRINT_Y1:
                    continue
                bbox_key = tuple(img["bbox"])
                xref = raster_xref_map.get(bbox_key)
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
                                    clipped = r & fitz.Rect(PRINT_X0, upper - 5, PRINT_X1, lower + 5)
                                    claimed_boxes.append(clipped)
                                    claimed[gi] = True

                    for li, ln in enumerate(lines):
                        if li in claimed_lines:
                            continue
                        if (
                            ln.bbox.y0 >= upper - 2
                            and ln.bbox.y1 <= lower + 2
                            and not self.caption_pattern.match(ln.text.strip())
                        ):
                            if nearest_caption_idx(ln.bbox.x0 + ln.width / 2) == ci:
                                claimed_boxes.append(ln.bbox)
                                claimed_lines.add(li)

                    canvas, bbox_out = self.render_stitched(page, claimed_boxes)

                    qnum: Optional[int] = None
                    for qn, y in anchors:
                        if y <= cap["bbox"].y0:
                            qnum = qn
                    if qnum is None:
                        qnum = last_seen_question if last_seen_question is not None else cap["q_from_caption"]

                    tag = "Table" if cap.get("kind") == "table" else "Fig"
                    clean_label = f"{tag}_{cap['q_from_caption']}_{cap['sub']}"
                    fname = f"p{pno + 1}_q{qnum}_{clean_label}.png"
                    save_path = output_dir / fname
                    canvas.save(str(save_path))

                    entry = {
                        "page": pno + 1,
                        "question": qnum,
                        "caption": cap["label"],
                        "bbox": [round(v, 1) for v in bbox_out],
                        "filename": fname,
                        "file": fname,
                        "relative_path": f"{output_dir.name}/{fname}",
                    }
                    manifest.append(entry)
                    logger.info(f"  Extracted: {fname} (Page {pno + 1}, Q{qnum}, {cap['label']})")

            # Fallback: unclaimed geometry with substantial 2-D footprint (skip cover page pno == 0)
            if pno > 0 and last_seen_question is not None:
                unclaimed_vectors = [
                    r for gi, (kind, r) in enumerate(all_geo)
                    if not claimed[gi] and not overlaps_any_caption[gi] and kind == "vector"
                ]
                clustered_fallbacks = self.cluster_rects(unclaimed_vectors)
                for r in clustered_fallbacks:
                    # Must be strictly within printable bounds and substantial
                    if r.x0 < PRINT_X0 or r.x1 > PRINT_X1 or r.y0 < PRINT_Y0 or r.y1 > PRINT_Y1:
                        continue
                    if r.get_area() < 8000 or min(r.width, r.height) < 30.0:
                        continue

                    qnum = None
                    for qn, y in anchors:
                        if y <= r.y0:
                            qnum = qn
                    if qnum is None:
                        qnum = last_seen_question

                    clip = (r + (-self.pad, -self.pad, self.pad, self.pad)) & page.rect
                    pix = page.get_pixmap(clip=clip, dpi=self.dpi)
                    if self._is_blank_image(pix):
                        continue

                    fname = f"p{pno + 1}_q{qnum}_fallback_vector_{len(manifest) + 1}.png"
                    save_path = output_dir / fname
                    pix.save(str(save_path))

                    entry = {
                        "page": pno + 1,
                        "question": qnum,
                        "caption": None,
                        "bbox": [round(v, 1) for v in clip],
                        "filename": fname,
                        "file": fname,
                        "relative_path": f"{output_dir.name}/{fname}",
                    }
                    manifest.append(entry)
                    logger.info(f"  Extracted (fallback): {fname} (Page {pno + 1}, Q{qnum})")

        doc.close()

        # Write manifest.json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"CambridgeVisualExtractor: successfully extracted {len(manifest)} visuals "
            f"into {output_dir.name}/ (manifest: {manifest_path.name})"
        )
        return manifest
