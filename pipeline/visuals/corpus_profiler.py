"""
Cambridge Past Paper Corpus Profiler

Inventories and profiles PDF past papers across:
1. Provenance: born-digital ("native"), scanned raster ("scanned"), hybrid ("hybrid").
2. Document role: Question Paper (qp), Mark Scheme (ms), Examiner Report (er), Insert (in), etc.
3. Visual density: Drawing vectors, raster images, coverage, estimated DPI.
4. Syllabus metadata: Subject code, year, session, variant.

Outputs a structured corpus manifest JSON file to guide extraction routing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

import pymupdf as fitz

logger = logging.getLogger(__name__)


def parse_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """
    Parses standard Cambridge past paper filename into structured metadata.
    Examples:
      - 5054_s25_qp_22.pdf -> subject 5054, year 2025, session jun, role qp, variant 22
      - 1123_w10_in_21.pdf -> subject 1123, year 2010, session nov, role in, variant 21
      - 1123_s10_er.pdf    -> subject 1123, year 2010, session jun, role er, variant None
    """
    stem = pdf_path.stem

    # Pattern: {subject}_{session}{year}_{role}_{variant} (e.g. 5054_s25_qp_22)
    match_var = re.search(
        r"(\d{4})_([swm]\d{2})_(qp|ms|er|gt|in|ci|sp|sm)_(\d{1,2})",
        stem,
        re.IGNORECASE,
    )
    if match_var:
        subj = match_var.group(1)
        sy_raw = match_var.group(2).lower()
        session_map = {"s": "jun", "w": "nov", "m": "mar"}
        session = session_map.get(sy_raw[0], "jun")
        yr_val = int(sy_raw[1:])
        year = 2000 + yr_val if yr_val < 100 else yr_val
        role = match_var.group(3).lower()
        variant = match_var.group(4)
        paper_num = int(variant[0]) if variant and variant[0].isdigit() else 1
        return {
            "paper_code": stem,
            "syllabus": subj,
            "year": year,
            "session": session,
            "doc_role": role,
            "variant": variant,
            "paper_number": paper_num,
        }

    # Pattern without variant: {subject}_{session}{year}_{role} (e.g. 1123_s10_er)
    match_novar = re.search(
        r"(\d{4})_([swm]\d{2})_(qp|ms|er|gt|in|ci|sp|sm)",
        stem,
        re.IGNORECASE,
    )
    if match_novar:
        subj = match_novar.group(1)
        sy_raw = match_novar.group(2).lower()
        session_map = {"s": "jun", "w": "nov", "m": "mar"}
        session = session_map.get(sy_raw[0], "jun")
        yr_val = int(sy_raw[1:])
        year = 2000 + yr_val if yr_val < 100 else yr_val
        role = match_novar.group(3).lower()
        return {
            "paper_code": stem,
            "syllabus": subj,
            "year": year,
            "session": session,
            "doc_role": role,
            "variant": None,
            "paper_number": 1,
        }

    # Fallback splitting by underscores
    parts = stem.split("_")
    subj = parts[0] if parts and parts[0].isdigit() else "unknown"
    return {
        "paper_code": stem,
        "syllabus": subj,
        "year": 2025,
        "session": "jun",
        "doc_role": "qp" if "_qp" in stem.lower() else "ms",
        "variant": None,
        "paper_number": 1,
    }


def compute_union_area(rects: list[tuple[float, float, float, float]]) -> float:
    """
    Computes exact union area of axis-aligned rectangles using a 1D sweep-line algorithm.
    Runs in O(N^2 log N) microsecond time for typical page image counts without external libraries.
    """
    if not rects:
        return 0.0

    x_coords = sorted({r[0] for r in rects} | {r[2] for r in rects})
    total_area = 0.0

    for i in range(len(x_coords) - 1):
        x_left, x_right = x_coords[i], x_coords[i + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        y_intervals = []
        for x0, y0, x1, y1 in rects:
            if x0 <= x_left and x1 >= x_right:
                y_intervals.append((y0, y1))

        if not y_intervals:
            continue

        y_intervals.sort()
        merged_y: list[list[float]] = []
        for y0, y1 in y_intervals:
            if not merged_y or y0 > merged_y[-1][1]:
                merged_y.append([y0, y1])
            else:
                merged_y[-1][1] = max(merged_y[-1][1], y1)

        total_dy = sum(y_end - y_start for y_start, y_end in merged_y)
        total_area += dx * total_dy

    return total_area


def profile_page(page: fitz.Page, page_index: int) -> dict[str, Any]:
    """
    Profiles an individual PDF page:
    - Text character count
    - Drawing vector count
    - Raster image count and area coverage fraction
    - Effective raster DPI estimate
    - Route determination: native, scanned, hybrid, or blank
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    text_chars = len(page.get_text().strip())
    raw_drawings = page.get_drawings()
    drawing_count = len(raw_drawings)

    raw_images = page.get_images(full=True)
    image_count = len(raw_images)

    img_rect_tuples: list[tuple[float, float, float, float]] = []
    max_dpi: float | None = None

    for img_info in raw_images:
        xref = img_info[0]
        # img_info[2] is width, img_info[3] is height
        img_w_px = float(img_info[2])
        img_h_px = float(img_info[3])

        rects = page.get_image_rects(xref)
        for r in rects:
            clipped = r & page_rect
            if not clipped.is_empty and clipped.width > 0 and clipped.height > 0:
                img_rect_tuples.append((clipped.x0, clipped.y0, clipped.x1, clipped.y1))
                # Calculate DPI: (pixels / (points / 72))
                dpi_x = img_w_px / (clipped.width / 72.0)
                dpi_y = img_h_px / (clipped.height / 72.0)
                calc_dpi = (dpi_x + dpi_y) / 2.0
                if max_dpi is None or calc_dpi > max_dpi:
                    max_dpi = round(calc_dpi, 1)

    img_coverage = (
        compute_union_area(img_rect_tuples) / page_area if page_area > 0 else 0.0
    )
    img_coverage = min(1.0, max(0.0, round(img_coverage, 3)))

    # Classification rules:
    # 1. Blank page: no text, no drawings, no images
    if text_chars == 0 and drawing_count == 0 and image_count == 0:
        route = "blank"
    # 2. Scanned: high image coverage (> 70%) and minimal selectable text (< 50 chars)
    elif img_coverage > 0.70 and text_chars < 50:
        route = "scanned"
    # 3. Hybrid: high image coverage (> 70%) with selectable text layer on top (OCR or digital overlay)
    elif img_coverage > 0.70 and text_chars >= 50:
        route = "hybrid"
    # 4. Native (born-digital): text, vector drawings, embedded illustrations
    else:
        route = "native"

    return {
        "page_index": page_index,
        "page_number": page_index + 1,
        "width_pt": round(page_rect.width, 1),
        "height_pt": round(page_rect.height, 1),
        "text_chars": text_chars,
        "drawing_count": drawing_count,
        "image_count": image_count,
        "img_coverage": img_coverage,
        "effective_dpi": max_dpi,
        "route": route,
    }


def profile_pdf(pdf_path: Path) -> dict[str, Any]:
    """
    Profiles an entire PDF document and returns metadata plus page-level breakdown.
    """
    pdf_path = Path(pdf_path).resolve()
    meta = parse_pdf_metadata(pdf_path)

    doc = fitz.open(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for idx, page in enumerate(doc):
        pages.append(profile_page(page, idx))

    doc.close()

    # Determine overall document-level provenance
    non_blank = [p for p in pages if p["route"] != "blank"]
    if not non_blank:
        doc_provenance = "blank"
    elif all(p["route"] == "native" for p in non_blank):
        doc_provenance = "born-digital"
    elif all(p["route"] == "scanned" for p in non_blank):
        doc_provenance = "scanned"
    else:
        doc_provenance = "hybrid"

    native_count = sum(1 for p in pages if p["route"] == "native")
    scanned_count = sum(1 for p in pages if p["route"] == "scanned")
    hybrid_count = sum(1 for p in pages if p["route"] == "hybrid")
    blank_count = sum(1 for p in pages if p["route"] == "blank")

    return {
        "file_path": str(pdf_path),
        "filename": pdf_path.name,
        "syllabus": meta["syllabus"],
        "year": meta["year"],
        "session": meta["session"],
        "doc_role": meta["doc_role"],
        "variant": meta["variant"],
        "paper_number": meta["paper_number"],
        "provenance": doc_provenance,
        "page_count": len(pages),
        "native_pages": native_count,
        "scanned_pages": scanned_count,
        "hybrid_pages": hybrid_count,
        "blank_pages": blank_count,
        "pages": pages,
    }


def profile_corpus(
    storage_dir: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """
    Scans storage_dir for all PDF files, profiles each, and returns aggregated statistics.
    Optionally saves the full manifest to output_json.
    """
    storage_dir = Path(storage_dir).resolve()
    pdf_files = sorted(storage_dir.rglob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDFs in {storage_dir}")

    records: list[dict[str, Any]] = []
    provenance_counts: dict[str, int] = {"born-digital": 0, "scanned": 0, "hybrid": 0, "blank": 0}
    route_counts: dict[str, int] = {"native": 0, "scanned": 0, "hybrid": 0, "blank": 0}
    subject_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}

    total_pages = 0

    for pdf in pdf_files:
        try:
            rec = profile_pdf(pdf)
            records.append(rec)

            provenance_counts[rec["provenance"]] = provenance_counts.get(rec["provenance"], 0) + 1
            subj = rec["syllabus"]
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
            role = rec["doc_role"]
            role_counts[role] = role_counts.get(role, 0) + 1

            total_pages += rec["page_count"]
            route_counts["native"] += rec["native_pages"]
            route_counts["scanned"] += rec["scanned_pages"]
            route_counts["hybrid"] += rec["hybrid_pages"]
            route_counts["blank"] += rec["blank_pages"]

        except Exception as e:
            logger.error(f"Error profiling {pdf.name}: {e}")

    total_pdfs = len(records)
    born_digital_pct = (
        round(provenance_counts["born-digital"] / total_pdfs * 100, 1)
        if total_pdfs > 0
        else 0.0
    )
    scanned_pct = (
        round(provenance_counts["scanned"] / total_pdfs * 100, 1)
        if total_pdfs > 0
        else 0.0
    )
    hybrid_pct = (
        round(provenance_counts["hybrid"] / total_pdfs * 100, 1)
        if total_pdfs > 0
        else 0.0
    )

    manifest_data: dict[str, Any] = {
        "total_pdfs": total_pdfs,
        "total_pages": total_pages,
        "born_digital_pdf_pct": born_digital_pct,
        "scanned_pdf_pct": scanned_pct,
        "hybrid_pdf_pct": hybrid_pct,
        "provenance_breakdown": provenance_counts,
        "page_route_breakdown": route_counts,
        "subject_breakdown": subject_counts,
        "doc_role_breakdown": role_counts,
        "records": records,
    }

    if output_json:
        out_path = Path(output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Saved corpus manifest to {out_path}")

    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Cambridge Past Paper Corpus Profiler")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="pipeline/storage",
        help="Root storage directory containing PDFs",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="pipeline/storage/corpus_manifest.json",
        help="Destination JSON file path",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only display summary without writing manifest JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    storage_dir = Path(args.storage_dir)
    out_json = None if args.summary_only else Path(args.output_json)

    manifest = profile_corpus(storage_dir, out_json)

    print("\n" + "=" * 65)
    print("CAMBRIDGE EXAM CORPUS INVENTORY REPORT")
    print("=" * 65)
    print(f"Total PDFs Analyzed: {manifest['total_pdfs']}")
    print(f"Total Pages Analyzed: {manifest['total_pages']}")
    print("-" * 65)
    print("DOCUMENT PROVENANCE BREAKDOWN:")
    for prov, count in manifest["provenance_breakdown"].items():
        pct = round(count / manifest["total_pdfs"] * 100, 1) if manifest["total_pdfs"] else 0
        print(f"  • {prov:15}: {count:4} PDFs ({pct:5.1f}%)")
    print("-" * 65)
    print("PAGE-LEVEL EXTRACTION ROUTE BREAKDOWN:")
    for route, count in manifest["page_route_breakdown"].items():
        pct = round(count / manifest["total_pages"] * 100, 1) if manifest["total_pages"] else 0
        print(f"  • {route:15}: {count:5} pages ({pct:5.1f}%)")
    print("-" * 65)
    print("DOCUMENT ROLES:")
    for role, count in sorted(manifest["doc_role_breakdown"].items()):
        print(f"  • {role:15}: {count:4} PDFs")
    print("-" * 65)
    print("SUBJECTS:")
    for subj, count in sorted(manifest["subject_breakdown"].items()):
        print(f"  • Syllabus {subj:8}: {count:4} PDFs")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
