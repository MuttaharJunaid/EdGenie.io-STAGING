import pytest
import pymupdf as fitz
from pipeline.visuals.boundary_snap import (
    snap_boundary_to_ink,
    snap_to_whitespace_gutters,
    trace_connected_labels,
)
from pipeline.visuals.verifier import count_cut_strokes


def test_snap_boundary_synthetic():
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)

    # Draw a rectangle artwork at (100, 100, 200, 200)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(100, 100, 200, 200))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    # Provide a slightly loose candidate box (90, 90, 210, 210)
    loose_candidate = fitz.Rect(90, 90, 210, 210)
    snapped, meta = snap_boundary_to_ink(page, loose_candidate, dpi=150, padding_pt=2.0)

    assert meta["snapped"] is True
    # Snapped box should be tighter than the loose candidate (closer to 100±2, 200±2)
    assert abs(snapped.x0 - 98.0) <= 2.0
    assert abs(snapped.y0 - 98.0) <= 2.0
    assert abs(snapped.x1 - 202.0) <= 2.0
    assert abs(snapped.y1 - 202.0) <= 2.0

    # Ensure zero cut strokes on snapped box
    assert count_cut_strokes(page, snapped, dpi=150) == 0
    doc.close()


def test_snap_boundary_degenerate():
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    empty_cand = fitz.Rect(10, 10, 15, 15)  # too small
    snapped, meta = snap_boundary_to_ink(page, empty_cand)
    assert meta["snapped"] is False
    doc.close()


def test_snap_to_whitespace_gutters_synthetic():
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)

    # Draw an artwork box at (100, 100, 200, 200)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(100, 100, 200, 200))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    # Candidate box that is tight against ink (100, 100, 200, 200)
    cand = fitz.Rect(100, 100, 200, 200)
    snapped, meta = snap_to_whitespace_gutters(page, cand, dpi=150, padding_pt=3.0)

    assert meta["snapped"] is True
    assert snapped.x0 <= 100.0
    assert snapped.y0 <= 100.0
    assert snapped.x1 >= 200.0
    assert snapped.y1 >= 200.0
    assert count_cut_strokes(page, snapped, dpi=150) == 0
    doc.close()


def test_trace_connected_labels_synthetic():
    from pipeline.visuals.boundary_snap import trace_connected_labels

    doc = fitz.open()
    page = doc.new_page(width=500, height=500)

    # Artwork box at (150, 150, 300, 300)
    art_rect = fitz.Rect(150, 150, 300, 300)
    shape = page.new_shape()
    shape.draw_rect(art_rect)
    # Leader line from inside artwork (200, 160) extending outside to (200, 110)
    shape.draw_line(fitz.Point(200, 160), fitz.Point(200, 110))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()

    # Text label "cell membrane" at (160, 95, 240, 108)
    page.insert_text(fitz.Point(165, 105), "cell membrane", fontsize=10)

    labels = trace_connected_labels(page, art_rect)
    assert len(labels) >= 1
    # Check that the connected label envelops the text
    found = any(abs(r.y0 - 95.0) < 15.0 for r in labels)
    assert found is True, f"Failed to trace connected label near y=105: {labels}"
    doc.close()

