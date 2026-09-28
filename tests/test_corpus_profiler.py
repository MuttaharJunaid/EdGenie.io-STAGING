import pytest
from pathlib import Path
import pymupdf as fitz

from pipeline.visuals.corpus_profiler import (
    parse_pdf_metadata,
    compute_union_area,
    profile_page,
    profile_pdf,
)


def test_parse_pdf_metadata_variants():
    # 1. Standard question paper with variant
    m1 = parse_pdf_metadata(Path("5054_s25_qp_22.pdf"))
    assert m1["syllabus"] == "5054"
    assert m1["year"] == 2025
    assert m1["session"] == "jun"
    assert m1["doc_role"] == "qp"
    assert m1["variant"] == "22"
    assert m1["paper_number"] == 2

    # 2. Mark scheme with winter session
    m2 = parse_pdf_metadata(Path("5090_w24_ms_11.pdf"))
    assert m2["syllabus"] == "5090"
    assert m2["year"] == 2024
    assert m2["session"] == "nov"
    assert m2["doc_role"] == "ms"
    assert m2["variant"] == "11"

    # 3. Examiner report without variant
    m3 = parse_pdf_metadata(Path("1123_s10_er.pdf"))
    assert m3["syllabus"] == "1123"
    assert m3["year"] == 2010
    assert m3["session"] == "jun"
    assert m3["doc_role"] == "er"
    assert m3["variant"] is None

    # 4. Insert booklet with variant
    m4 = parse_pdf_metadata(Path("1123_w10_in_21.pdf"))
    assert m4["syllabus"] == "1123"
    assert m4["year"] == 2010
    assert m4["session"] == "nov"
    assert m4["doc_role"] == "in"
    assert m4["variant"] == "21"


def test_compute_union_area():
    # Empty
    assert compute_union_area([]) == 0.0

    # Non-overlapping
    r1 = (0.0, 0.0, 10.0, 10.0)  # area 100
    r2 = (20.0, 0.0, 30.0, 10.0)  # area 100
    assert compute_union_area([r1, r2]) == 200.0

    # Overlapping
    r3 = (5.0, 5.0, 15.0, 15.0)  # area 100, overlap with r1 is 5x5=25
    assert compute_union_area([r1, r3]) == 175.0

    # Contained inside
    r4 = (2.0, 2.0, 8.0, 8.0)  # area 36, inside r1
    assert compute_union_area([r1, r4]) == 100.0


def test_profile_pdf_real():
    sample_pdf = Path("pipeline/storage/5090/2025/nov/5090_w25_qp_22.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found")

    res = profile_pdf(sample_pdf)
    assert res["syllabus"] == "5090"
    assert res["year"] == 2025
    assert res["session"] == "nov"
    assert res["doc_role"] == "qp"
    assert res["page_count"] > 0
    assert res["provenance"] in ("born-digital", "scanned", "hybrid")
    assert len(res["pages"]) == res["page_count"]
    # Check first page structure
    p0 = res["pages"][0]
    assert "text_chars" in p0
    assert "drawing_count" in p0
    assert "route" in p0
