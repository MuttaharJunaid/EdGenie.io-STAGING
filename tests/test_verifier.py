import pytest
from pathlib import Path
import numpy as np
import cv2
import pymupdf as fitz

from pipeline.visuals.verifier import (
    compute_dhash,
    compute_hamming_distance,
    check_quiet_margin,
    check_furniture_intrusion,
    count_cut_strokes,
    verify_crop,
)


def test_compute_dhash_and_hamming():
    # 1. Base image
    img1 = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(img1, (60, 60), 40, 255, -1)
    h1 = compute_dhash(img1)
    assert len(h1) == 64

    # 2. Identical image
    assert compute_hamming_distance(h1, h1) == 0

    # 3. Slightly shifted image (near duplicate) — 1-px shift on a 120×120 circle
    # produces ~7 changed bits in the 8×8 dhash gradient grid; standard near-duplicate
    # threshold is ≤ 10, so allow up to 8 here.
    img2 = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(img2, (61, 60), 40, 255, -1)
    h2 = compute_dhash(img2)
    dist = compute_hamming_distance(h1, h2)
    assert dist <= 8, f"Expected near-duplicate distance ≤ 8, got {dist}"

    # 4. Completely different image (e.g. horizontal bar vs circle)
    img3 = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(img3, (10, 50), (110, 70), 255, -1)
    h3 = compute_dhash(img3)
    assert compute_hamming_distance(h1, h3) > 10


def test_check_quiet_margin():
    # Case 1: Image with 8px margin all around
    img = np.zeros((100, 100), dtype=np.uint8)
    img[8:92, 8:92] = 255
    ok, margins = check_quiet_margin(img, expected_pad_px=8, tol_px=3)
    assert ok is True
    assert margins["top"] == 8
    assert margins["left"] == 8

    # Case 2: Ink flush against top edge (margin = 0)
    img_flush = np.zeros((100, 100), dtype=np.uint8)
    img_flush[0:90, 8:92] = 255
    ok_flush, margins_flush = check_quiet_margin(img_flush, expected_pad_px=8)
    assert ok_flush is False
    assert margins_flush["top"] == 0


def test_check_furniture_intrusion():
    # Clean image
    clean = np.zeros((100, 100), dtype=np.uint8)
    has_f, _ = check_furniture_intrusion(clean, text_inside_crop="Apparatus diagram")
    assert has_f is False

    # Mark bracket in text layer
    has_bracket, reason = check_furniture_intrusion(clean, text_inside_crop="Fig. 1.1\n[3]")
    assert has_bracket is True
    assert "Mark allocation bracket" in reason

    # Dotted answer lines
    has_dots, reason_dots = check_furniture_intrusion(clean, text_inside_crop="Explain why ......................")
    assert has_dots is True
    assert "answer lines" in reason_dots


def test_count_cut_strokes_on_synthetic_page():
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    
    # Draw a rectangle in center: (50, 50, 250, 250)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(80, 80, 220, 220))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    # 1. Bounding box strictly encompassing the rectangle with padding -> 0 cuts
    crop_clean = fitz.Rect(75, 75, 225, 225)
    cuts_clean = count_cut_strokes(page, crop_clean, dpi=150)
    assert cuts_clean == 0

    # 2. Bounding box cutting through the rectangle -> > 0 cuts
    crop_cut = fitz.Rect(75, 75, 150, 225)  # cuts through right side of rectangle
    cuts_cut = count_cut_strokes(page, crop_cut, dpi=150)
    assert cuts_cut > 0

    doc.close()



def test_auto_repair_crop_synthetic():
    from pipeline.visuals.verifier import auto_repair_crop

    doc = fitz.open()
    page = doc.new_page(width=400, height=400)

    # Artwork box from (100, 100) to (200, 200)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(100, 100, 200, 200))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    # Intentionally cut crop: (100, 100, 190, 200) -> cuts the right side of the rectangle
    cut_crop = fitz.Rect(100, 100, 190, 200)
    repaired, v_res = auto_repair_crop(page, cut_crop, dpi=150)

    # Repaired crop must have expanded right to at least 200
    assert repaired.x1 >= 200.0
    assert v_res.cut_strokes == 0
    doc.close()

