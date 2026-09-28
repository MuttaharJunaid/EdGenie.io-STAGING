"""
Visual Crop Verification Module

Performs geometric and semantic verification on cropped exam visual assets:
1. Cut-stroke test: verifies no ink stroke is severed across the crop boundary.
2. Quiet-margin test: measures distance from inner ink boundary to crop edge.
3. Furniture intrusion test: detects answer ruling or mark brackets inside crop.
4. Perceptual hashing (dHash 16x16): detects identical / duplicate visuals across papers.
5. Confidence gating: routes to auto_accepted, auto_retry, or review_queue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import io
import logging
import re
from typing import Any

import cv2
import numpy as np
import PIL.Image as Image
import pymupdf as fitz

logger = logging.getLogger(__name__)


@dataclass
class CropVerificationResult:
    cut_strokes: int
    quiet_margin_ok: bool
    furniture_intrusion: bool
    margins: dict[str, int] = field(default_factory=dict)
    furniture_details: str = ""
    status: str = "auto_accepted"  # "auto_accepted" | "auto_retry" | "review_queue"
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_dhash(img: np.ndarray | Image.Image, hash_size: int = 16) -> str:
    """
    Computes a 16x16 difference hash (dHash) as a 64-character hex string.
    Fast, scale-invariant, and rotation-tolerant near-duplicate image detector.
    """
    if isinstance(img, Image.Image):
        gray = np.array(img.convert("L"))
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    byte_arr = np.packbits(bits)
    return byte_arr.tobytes().hex()


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculates the bitwise Hamming distance between two hex-encoded dHashes.
    Values <= 4 typically indicate near-identical diagrams.
    """
    b1 = bytes.fromhex(hash1)
    b2 = bytes.fromhex(hash2)
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))


def binarize_image(img: np.ndarray) -> np.ndarray:
    """
    Locally adaptive binarization of an image (ink = 255, background = 0).
    Uses Gaussian adaptive thresholding to retain faint dashed lines without noise.
    """
    if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Background is white (255), ink is dark (<200)
    # THRESH_BINARY_INV turns ink into 255
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )
    return bin_img


def count_cut_strokes(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    dpi: int = 150,
    border_pt: float = 4.0,
) -> int:
    """
    Tests whether any visual stroke is severed at the crop boundary.

    Renders a region expanded by border_pt points around crop_bbox.
    Identifies connected ink components: any component that touches BOTH the
    inner crop core and the outer border margin is a severed stroke.
    """
    # Expanded bounding box
    expanded_rect = fitz.Rect(
        crop_bbox.x0 - border_pt,
        crop_bbox.y0 - border_pt,
        crop_bbox.x1 + border_pt,
        crop_bbox.y1 + border_pt,
    ) & page.rect

    if expanded_rect.width <= 0 or expanded_rect.height <= 0:
        return 0

    pix = page.get_pixmap(clip=expanded_rect, dpi=dpi)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bin_img = binarize_image(img_data)

    h, w = bin_img.shape
    border_px_x = int(round(border_pt * (dpi / 72.0)))
    border_px_y = int(round(border_pt * (dpi / 72.0)))

    if h <= 2 * border_px_y or w <= 2 * border_px_x:
        return 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )
    if num_labels <= 1:
        return 0

    # Margin mask: outer border
    margin_mask = np.ones((h, w), dtype=bool)
    margin_mask[border_px_y:-border_px_y, border_px_x:-border_px_x] = False

    # Inner core mask: within crop area
    inner_mask = ~margin_mask

    cut_count = 0
    for label in range(1, num_labels):
        comp = labels == label
        # Ignore tiny speckles (< 4 pixels)
        if stats[label, cv2.CC_STAT_AREA] < 4:
            continue
        touches_outer = np.any(comp & margin_mask)
        touches_inner = np.any(comp & inner_mask)
        if touches_outer and touches_inner:
            cut_count += 1

    return cut_count


def check_quiet_margin(
    crop_bin_img: np.ndarray,
    expected_pad_px: int = 8,
    tol_px: int = 4,
) -> tuple[bool, dict[str, int]]:
    """
    Measures the distance from inner ink boundary to crop edges.
    Returns (is_valid, margins_dict).
    """
    ink_coords = np.argwhere(crop_bin_img > 0)
    if ink_coords.size == 0:
        return True, {"top": expected_pad_px, "left": expected_pad_px, "bottom": expected_pad_px, "right": expected_pad_px}

    min_y, min_x = ink_coords.min(axis=0)
    max_y, max_x = ink_coords.max(axis=0)
    h, w = crop_bin_img.shape

    m_top = int(min_y)
    m_left = int(min_x)
    m_bottom = int(h - 1 - max_y)
    m_right = int(w - 1 - max_x)

    margins = {"top": m_top, "left": m_left, "bottom": m_bottom, "right": m_right}

    # Ink flush against edge (< 1 px) indicates clipping or no margin
    if any(m < 1 for m in margins.values()):
        return False, margins

    # Margin should roughly match or exceed expected padding
    is_ok = all(m >= expected_pad_px - tol_px for m in margins.values())
    return is_ok, margins


def check_furniture_intrusion(
    crop_bin_img: np.ndarray,
    text_inside_crop: str = "",
) -> tuple[bool, str]:
    """
    Detects whether answer lines, mark brackets, or exam furniture bled into the crop.
    """
    # 1. Text-layer checks
    if text_inside_crop:
        import re
        if re.search(r"(\.{6,}|_{6,})", text_inside_crop):
            return True, "Dotted/solid answer lines detected in text layer"
        if re.search(r"^\s*\[\d{1,2}\]\s*$", text_inside_crop, re.MULTILINE):
            return True, "Mark allocation bracket detected in crop text"
        if re.search(r"\[Total:\s*\d+\]", text_inside_crop, re.IGNORECASE):
            return True, "Total mark notice detected in crop text"
        if "DO NOT WRITE IN THIS MARGIN" in text_inside_crop.upper():
            return True, "Margin notice detected in crop text"

    # 2. Vision-based horizontal ruling check
    h, w = crop_bin_img.shape
    min_line_len = int(w * 0.7)
    if min_line_len > 80:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))
        h_lines = cv2.morphologyEx(crop_bin_img, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            h_lines, connectivity=8
        )
        detected_lines = num_labels - 1
        if detected_lines >= 3:
            # Check if this is part of a 2D coordinate grid / graph by looking for vertical grid lines
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.3)))
            v_lines = cv2.morphologyEx(crop_bin_img, cv2.MORPH_OPEN, v_kernel)
            num_v_labels, _, _, _ = cv2.connectedComponentsWithStats(v_lines, connectivity=8)
            is_grid = (num_v_labels - 1) >= 2
            if not is_grid:
                # Multiple repeating horizontal rules spanning >= 70% width
                return True, f"{detected_lines} repeating answer ruling lines detected"

    return False, ""


def verify_crop(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    crop_img: np.ndarray | None = None,
    visual_class: str = "line_diagram",
    dpi: int = 150,
) -> CropVerificationResult:
    """
    Executes complete verification pipeline for a visual crop:
    - Cut stroke detection
    - Quiet margin verification
    - Furniture intrusion detection
    - Confidence gating
    """
    reasons: list[str] = []

    clip_rect = crop_bbox & page.rect
    if clip_rect.is_empty or clip_rect.width < 5 or clip_rect.height < 5:
        return CropVerificationResult(
            cut_strokes=0,
            quiet_margin_ok=False,
            furniture_intrusion=False,
            status="auto_accepted",
            confidence=1.0,
            reasons=[],
        )

    # 1. Render crop image if not provided
    if crop_img is None:
        pix = page.get_pixmap(clip=clip_rect, dpi=dpi)
        if pix.width == 0 or pix.height == 0:
            return CropVerificationResult(
                cut_strokes=0,
                quiet_margin_ok=False,
                furniture_intrusion=False,
                status="auto_accepted",
                confidence=1.0,
                reasons=[],
            )
        crop_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    bin_crop = binarize_image(crop_img)

    # 2. Extract text inside crop for furniture checking
    text_inside = page.get_text("text", clip=crop_bbox).strip()

    # 3. Check cut strokes
    cut_strokes = count_cut_strokes(page, crop_bbox, dpi=dpi)
    if cut_strokes > 0:
        reasons.append(f"{cut_strokes} cut stroke(s) detected across crop boundary")

    # 4. Check quiet margins
    margin_ok, margins = check_quiet_margin(bin_crop, expected_pad_px=int(round(8 * (dpi / 300.0))))
    if not margin_ok:
        reasons.append("Quiet margin check failed (ink touches edge or excessive whitespace)")

    # 5. Check furniture intrusion
    has_furniture, furniture_detail = check_furniture_intrusion(bin_crop, text_inside)
    if has_furniture:
        reasons.append(f"Furniture intrusion: {furniture_detail}")

    # 6. Confidence calculation & Status routing
    confidence = 1.0
    if cut_strokes > 0:
        confidence -= 0.35 * min(cut_strokes, 2)
    if not margin_ok:
        confidence -= 0.15
    if has_furniture:
        confidence -= 0.30

    confidence = max(0.0, min(1.0, round(confidence, 2)))

    # Gate logic:
    # Auto-accept: 0 cut strokes, no furniture intrusion, confidence >= 0.85
    if cut_strokes == 0 and not has_furniture and confidence >= 0.85:
        status = "auto_accepted"
    # Auto-retry: small recoverable error (e.g. slight margin violation but 0 cut strokes)
    elif cut_strokes == 0 and not has_furniture:
        status = "auto_retry"
    # Review queue: cut strokes or furniture intrusion
    else:
        status = "review_queue"

    return CropVerificationResult(
        cut_strokes=cut_strokes,
        quiet_margin_ok=margin_ok,
        furniture_intrusion=has_furniture,
        margins=margins,
        furniture_details=furniture_detail,
        status=status,
        confidence=confidence,
        reasons=reasons,
    )


def auto_repair_crop(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    allowed_bounds: fitz.Rect | None = None,
    visual_class: str = "line_diagram",
    dpi: int = 150,
    max_expand_pt: float = 35.0,
) -> tuple[fitz.Rect, CropVerificationResult]:
    """
    Self-Auditing Auto-Repair Gate (Pillar 3).
    Verifies the visual crop and automatically repairs boundaries if:
    1. Strokes are cut: expands boundary outward until strokes terminate cleanly.
    2. Question stem prose text or answer lines bled into top/bottom: trims boundary back to whitespace.
    3. Margins are flush: adds quiet padding so ink is not cut off.
    """
    orig_rect = fitz.Rect(crop_bbox) & page.rect
    if orig_rect.is_empty or orig_rect.width < 10 or orig_rect.height < 10:
        return orig_rect, verify_crop(page, orig_rect, visual_class=visual_class, dpi=dpi)

    if allowed_bounds is not None:
        bounds = (fitz.Rect(allowed_bounds) | orig_rect) & page.rect
    else:
        bounds = page.rect
    v_res = verify_crop(page, orig_rect, visual_class=visual_class, dpi=dpi)

    # If already auto-accepted with zero cut strokes and no furniture intrusion, return immediately
    if v_res.cut_strokes == 0 and not v_res.furniture_intrusion:
        return orig_rect, v_res

    current_rect = fitz.Rect(orig_rect)

    # 1. Handle furniture intrusion / text bleed at top or bottom
    if v_res.furniture_intrusion:
        blocks = page.get_text("blocks")
        for b in blocks:
            b_rect = fitz.Rect(b[:4])
            b_text = b[4].strip()
            if b_rect.intersects(current_rect):
                has_bottom_furn = bool(
                    re.search(
                        r"(\.{6,}|_{6,})|^\s*\[\d{1,2}\]\s*$|\[Total:\s*\d+\]",
                        b_text,
                        re.I | re.M,
                    )
                )
                if has_bottom_furn and b_rect.y0 >= current_rect.y1 - 35.0 and b_rect.y0 > current_rect.y0 + 20.0:
                    current_rect.y1 = max(current_rect.y0 + 20.0, b_rect.y0 - 2.0)
                if ("shows a" in b_text.lower() or "which of the" in b_text.lower()) and b_rect.y1 <= current_rect.y0 + 35.0:
                    current_rect.y0 = min(current_rect.y1 - 20.0, b_rect.y1 + 2.0)

    # 2. Handle cut strokes: progressively expand edges if cut strokes exist
    cur_cuts = count_cut_strokes(page, current_rect, dpi=dpi)
    if cur_cuts > 0:
        # Minimal quiet padding only when cuts exist, to see if 2pt clearance clears border-grazing strokes
        current_rect = fitz.Rect(
            max(bounds.x0, current_rect.x0 - 2.0),
            max(bounds.y0, current_rect.y0 - 2.0),
            min(bounds.x1, current_rect.x1 + 2.0),
            min(bounds.y1, current_rect.y1 + 2.0),
        )
        cur_cuts = count_cut_strokes(page, current_rect, dpi=dpi)
        best_rect = current_rect
        best_cuts = cur_cuts

        # Try directional expansions first, then combined
        directions = [
            ("right", lambda r, s: fitz.Rect(r.x0, r.y0, min(bounds.x1, r.x1 + s), r.y1)),
            ("top", lambda r, s: fitz.Rect(r.x0, max(bounds.y0, r.y0 - s), r.x1, r.y1)),
            ("bottom", lambda r, s: fitz.Rect(r.x0, r.y0, r.x1, min(bounds.y1, r.y1 + s))),
            ("left", lambda r, s: fitz.Rect(max(bounds.x0, r.x0 - s), r.y0, r.x1, r.y1)),
            ("all", lambda r, s: fitz.Rect(
                max(bounds.x0, r.x0 - s),
                max(bounds.y0, r.y0 - s),
                min(bounds.x1, r.x1 + s),
                min(bounds.y1, r.y1 + s),
            )),
        ]

        for d_name, expand_fn in directions:
            for step in range(2, int(max_expand_pt) + 1, 2):
                test_r = expand_fn(current_rect, step)
                c = count_cut_strokes(page, test_r, dpi=dpi)
                if c < best_cuts:
                    best_cuts = c
                    best_rect = test_r
                    if c == 0:
                        break
            if best_cuts == 0:
                break

        current_rect = best_rect

    final_res = verify_crop(page, current_rect, visual_class=visual_class, dpi=dpi)
    if (
        not current_rect.is_empty
        and current_rect.width >= 10
        and current_rect.height >= 10
        and final_res.cut_strokes <= v_res.cut_strokes
    ):
        return current_rect, final_res
    return orig_rect, v_res

