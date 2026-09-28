"""
Boundary Snap Refiner Module

Refines candidate visual bounding boxes onto exact ink boundaries:
1. Expands candidate box by a search margin to guarantee ink coverage.
2. Performs adaptive binarization on the search window.
3. Contracts by projection profile / bounding contours to true ink extent.
4. Adds standard quiet padding (default: 2.0 pt / ~8 px at 300 DPI).
5. Failsafe: validates with cut-stroke check to ensure no strokes are severed.
   If snapping cuts strokes or degenerates, safely reverts to the candidate box.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz

from pipeline.visuals.verifier import binarize_image, count_cut_strokes

logger = logging.getLogger(__name__)


def snap_boundary_to_ink(
    page: fitz.Page,
    candidate_bbox: fitz.Rect,
    dpi: int = 150,
    search_expand_pt: float = 12.0,
    padding_pt: float = 2.0,
) -> tuple[fitz.Rect, dict[str, Any]]:
    """
    Snaps candidate_bbox to true ink boundaries within page.
    Returns (snapped_rect, snap_metadata).
    """
    orig_rect = fitz.Rect(candidate_bbox) & page.rect
    if orig_rect.is_empty or orig_rect.width < 10 or orig_rect.height < 10:
        return orig_rect, {"snapped": False, "reason": "degenerate_candidate"}

    # 1. Expand search window
    search_window = fitz.Rect(
        orig_rect.x0 - search_expand_pt,
        orig_rect.y0 - search_expand_pt,
        orig_rect.x1 + search_expand_pt,
        orig_rect.y1 + search_expand_pt,
    ) & page.rect

    # 2. Render search window
    scale = dpi / 72.0
    pix = page.get_pixmap(clip=search_window, dpi=dpi)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bin_img = binarize_image(img_data)

    # 3. Locate inner core corresponding to orig_rect
    core_x0 = int(round((orig_rect.x0 - search_window.x0) * scale))
    core_y0 = int(round((orig_rect.y0 - search_window.y0) * scale))
    core_x1 = int(round((orig_rect.x1 - search_window.x0) * scale))
    core_y1 = int(round((orig_rect.y1 - search_window.y0) * scale))

    # Mask connected components that touch the original core
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    if num_labels <= 1:
        return orig_rect, {"snapped": False, "reason": "no_ink_found"}

    # Identify components that overlap with the core
    core_labels = set()
    core_mask = np.zeros_like(bin_img, dtype=bool)
    core_mask[max(0, core_y0) : min(bin_img.shape[0], core_y1), max(0, core_x0) : min(bin_img.shape[1], core_x1)] = True

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 4:
            continue
        if np.any((labels == label) & core_mask):
            core_labels.add(label)

    if not core_labels:
        return orig_rect, {"snapped": False, "reason": "no_core_labels"}

    # Build mask of relevant ink components
    relevant_ink = np.isin(labels, list(core_labels))
    ink_coords = np.argwhere(relevant_ink)
    if ink_coords.size == 0:
        return orig_rect, {"snapped": False, "reason": "empty_relevant_ink"}

    # 4. Compute snapped bounding box
    min_y_px, min_x_px = ink_coords.min(axis=0)
    max_y_px, max_x_px = ink_coords.max(axis=0)

    snapped_x0 = search_window.x0 + (min_x_px / scale) - padding_pt
    snapped_y0 = search_window.y0 + (min_y_px / scale) - padding_pt
    snapped_x1 = search_window.x0 + (max_x_px / scale) + padding_pt
    snapped_y1 = search_window.y0 + (max_y_px / scale) + padding_pt

    snapped_rect = fitz.Rect(snapped_x0, snapped_y0, snapped_x1, snapped_y1) & page.rect

    # 5. Failsafe check: verify with cut-stroke detector
    # Snapping should NEVER introduce cut strokes
    orig_cuts = count_cut_strokes(page, orig_rect, dpi=dpi)
    snapped_cuts = count_cut_strokes(page, snapped_rect, dpi=dpi)

    if snapped_cuts > orig_cuts:
        logger.debug(f"Boundary snap cut strokes ({snapped_cuts} > {orig_cuts}). Reverting.")
        return orig_rect, {"snapped": False, "reason": "cut_stroke_regression"}

    if snapped_rect.width < 20 or snapped_rect.height < 20:
        return orig_rect, {"snapped": False, "reason": "snapped_box_too_small"}

    return snapped_rect, {
        "snapped": True,
        "orig_bbox": [orig_rect.x0, orig_rect.y0, orig_rect.x1, orig_rect.y1],
        "snapped_bbox": [snapped_rect.x0, snapped_rect.y0, snapped_rect.x1, snapped_rect.y1],
        "cut_strokes": snapped_cuts,
    }


def trace_connected_labels(
    page: fitz.Page,
    artwork_rect: fitz.Rect,
    lines: list[Any] | None = None,
    drawings: list[dict[str, Any]] | None = None,
    max_pointer_reach: float = 80.0,
) -> list[fitz.Rect]:
    """
    Topological leader-line and pointer graph traversal (Pillar 1).
    Follows vector line/curve strokes from inside or near the artwork outwards
    to identify and attach diagram labels (e.g. 'cell membrane', 'A', 'B', etc.),
    eliminating arbitrary scalar proximity margins.
    """
    if drawings is None:
        drawings = page.get_drawings(extended=True)

    art = fitz.Rect(artwork_rect)
    outside_pts: list[fitz.Point] = []

    # 1. Identify vector paths with one endpoint near/inside artwork and one endpoint outside
    for d in drawings:
        for it in d.get("items", []):
            cmd = it[0]
            pts: list[fitz.Point] = []
            if cmd == "l":
                pts = [it[1], it[2]]
            elif cmd == "c":
                pts = [it[1], it[4]]

            if len(pts) == 2:
                p1, p2 = pts
                p1_in = (art.x0 - 6.0 <= p1.x <= art.x1 + 6.0) and (art.y0 - 6.0 <= p1.y <= art.y1 + 6.0)
                p2_in = (art.x0 - 6.0 <= p2.x <= art.x1 + 6.0) and (art.y0 - 6.0 <= p2.y <= art.y1 + 6.0)
                if p1_in and not p2_in:
                    d_reach = max(art.x0 - p2.x, p2.x - art.x1, art.y0 - p2.y, p2.y - art.y1, 0.0)
                    if d_reach <= max_pointer_reach:
                        outside_pts.append(p2)
                elif p2_in and not p1_in:
                    d_reach = max(art.x0 - p1.x, p1.x - art.x1, art.y0 - p1.y, p1.y - art.y1, 0.0)
                    if d_reach <= max_pointer_reach:
                        outside_pts.append(p1)

    connected_boxes: list[fitz.Rect] = []

    # 2. Attach text words/lines near pointer endpoints
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    for pt in outside_pts:
        for w in words:
            wr = fitz.Rect(w[:4])
            d_x = max(wr.x0 - pt.x, pt.x - wr.x1, 0.0)
            d_y = max(wr.y0 - pt.y, pt.y - wr.y1, 0.0)
            dist = (d_x * d_x + d_y * d_y) ** 0.5
            if dist <= 18.0:
                # Exclude margin question anchors (e.g. '1', '2')
                w_text = w[4].strip()
                if wr.x1 <= 60.0 and w_text.isdigit():
                    continue
                # If we have line objects passed in, find full line rect
                found_line = False
                if lines:
                    for ln in lines:
                        if ln.bbox.intersects(wr):
                            if ln.bbox.x0 <= 60.0 and ln.text.strip() and ln.text.strip().split()[0].isdigit():
                                continue
                            if ln.bbox not in connected_boxes:
                                connected_boxes.append(fitz.Rect(ln.bbox))
                            found_line = True
                            break
                if not found_line and wr not in connected_boxes:
                    connected_boxes.append(wr)

    # 3. Detect aligned option labels (e.g. A, B, C, D directly above/below shapes)
    blocks = page.get_text("blocks")
    for w in words:
        w_text = w[4].strip()
        if w_text in ("A", "B", "C", "D", "(a)", "(b)", "(c)", "(d)", "(i)", "(ii)", "(iii)", "(iv)"):
            wr = fitz.Rect(w[:4])
            # Must sit horizontally within/near artwork span
            if wr.x1 >= art.x0 - 20.0 and wr.x0 <= art.x1 + 20.0:
                # Sits within 45 pt above or below, with NO intervening text blocks
                gap_above = art.y0 - wr.y1
                gap_below = wr.y0 - art.y1
                if 0.0 <= gap_above <= 45.0:
                    has_intervening = any(
                        b[1] >= wr.y1 - 2.0 and b[3] <= art.y0 + 2.0 and b[4].strip()
                        for b in blocks
                    )
                    if not has_intervening and wr not in connected_boxes:
                        connected_boxes.append(wr)
                elif 0.0 <= gap_below <= 45.0:
                    has_intervening = any(
                        b[1] >= art.y1 - 2.0 and b[3] <= wr.y0 + 2.0 and b[4].strip()
                        for b in blocks
                    )
                    if not has_intervening and wr not in connected_boxes:
                        connected_boxes.append(wr)

    return connected_boxes


def snap_to_whitespace_gutters(
    page: fitz.Page,
    candidate_bbox: fitz.Rect,
    allowed_bounds: fitz.Rect | None = None,
    dpi: int = 150,
    search_expand_pt: float = 12.0,
    padding_pt: float = 2.0,
) -> tuple[fitz.Rect, dict[str, Any]]:
    """
    Dynamic Whitespace Projection Gutter Snapping (Pillar 1).
    Snaps candidate_bbox outwards to the nearest true whitespace gutters
    (zero ink density corridors) along all 4 edges, bounded by allowed_bounds.
    Ensures zero strokes are cut and clean white margins are maintained.
    """
    orig_rect = fitz.Rect(candidate_bbox) & page.rect
    if orig_rect.is_empty or orig_rect.width < 10 or orig_rect.height < 10:
        return orig_rect, {"snapped": False, "reason": "degenerate_candidate"}

    bounds = fitz.Rect(allowed_bounds) if allowed_bounds is not None else page.rect

    search_window = fitz.Rect(
        max(bounds.x0, orig_rect.x0 - search_expand_pt),
        max(bounds.y0, orig_rect.y0 - search_expand_pt),
        min(bounds.x1, orig_rect.x1 + search_expand_pt),
        min(bounds.y1, orig_rect.y1 + search_expand_pt),
    ) & page.rect

    scale = dpi / 72.0
    pix = page.get_pixmap(clip=search_window, dpi=dpi)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bin_img = binarize_image(img_data)

    h_prof = np.sum(bin_img > 0, axis=1)  # horizontal profile (per row)
    v_prof = np.sum(bin_img > 0, axis=0)  # vertical profile (per col)

    core_y0 = max(0, int(round((orig_rect.y0 - search_window.y0) * scale)))
    core_y1 = min(bin_img.shape[0], int(round((orig_rect.y1 - search_window.y0) * scale)))
    core_x0 = max(0, int(round((orig_rect.x0 - search_window.x0) * scale)))
    core_x1 = min(bin_img.shape[1], int(round((orig_rect.x1 - search_window.x0) * scale)))

    top_px = core_y0
    for y in range(core_y0, -1, -1):
        if h_prof[y] == 0:
            top_px = y
            break

    bottom_px = core_y1
    for y in range(core_y1, len(h_prof)):
        if h_prof[y] == 0:
            bottom_px = y
            break

    left_px = core_x0
    for x in range(core_x0, -1, -1):
        if v_prof[x] == 0:
            left_px = x
            break

    right_px = core_x1
    for x in range(core_x1, len(v_prof)):
        if v_prof[x] == 0:
            right_px = x
            break

    snapped_x0 = max(bounds.x0, search_window.x0 + (left_px / scale) - padding_pt)
    snapped_y0 = max(bounds.y0, search_window.y0 + (top_px / scale) - padding_pt)
    snapped_x1 = min(bounds.x1, search_window.x0 + (right_px / scale) + padding_pt)
    snapped_y1 = min(bounds.y1, search_window.y0 + (bottom_px / scale) + padding_pt)

    snapped_rect = fitz.Rect(snapped_x0, snapped_y0, snapped_x1, snapped_y1) & page.rect

    orig_cuts = count_cut_strokes(page, orig_rect, dpi=dpi)
    snapped_cuts = count_cut_strokes(page, snapped_rect, dpi=dpi)

    if snapped_cuts > orig_cuts:
        return orig_rect, {"snapped": False, "reason": "cut_stroke_regression", "cut_strokes": orig_cuts}

    return snapped_rect, {
        "snapped": True,
        "orig_bbox": [orig_rect.x0, orig_rect.y0, orig_rect.x1, orig_rect.y1],
        "snapped_bbox": [snapped_rect.x0, snapped_rect.y0, snapped_rect.x1, snapped_rect.y1],
        "cut_strokes": snapped_cuts,
    }

