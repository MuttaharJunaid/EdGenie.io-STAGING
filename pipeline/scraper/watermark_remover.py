"""
Watermark Removal Module (Method 3: Form XObject Blanking + Inline Vector & Footer Strip)

Designed specifically for Cambridge past papers from 2026+ distributed with
PapaCambridge watermarks. Performs a clean, local in-place purge:
1. Blanks all Form XObjects responsible for the 30° repeating diagonal background logo (/FormXob.pcm).
2. Strips the inline vector watermark payload (Drawing bezier curves with /gRLs0-0 gs).
3. Strips both footer text sets ('PapaCambridge' and 'papacambridge.com' with octal \\056 escapes)
   along with the footer divider rule and tracking metadata.
"""

import logging
import re
from pathlib import Path
import pymupdf as fitz

from pipeline.config import settings

logger = logging.getLogger(__name__)

# Precompiled patterns for footer text and divider lines
TEXT_BT_ET_RE = re.compile(
    rb"BT(?:(?!ET).)*?(?:papacambridge|PapaCambridge|papacambridge\\056com|Downloaded\s+from|Trace\s+ID)(?:(?!ET).)*?ET",
    re.DOTALL | re.IGNORECASE,
)
DIVIDER_LINE_RE = re.compile(rb"(?:n\s+)?36\s+15\s+m\s+559\.28\s+15\s+l\s+S", re.DOTALL)


def extract_year_from_filename(filename: str) -> int | None:
    """
    Extracts exam year from standard CAIE filenames:
    e.g. 5054_s26_qp_11.pdf -> 2026
         4024_w23_ms_12.pdf -> 2023
         7707_m26_er.pdf    -> 2026
    """
    m = re.search(r"_[a-zA-Z](\d{2})_", filename)
    if m:
        return 2000 + int(m.group(1))
    m2 = re.search(r"(?:19|20)(\d{2})", filename)
    if m2:
        return int(m2.group(0))
    return None


def remove_watermark_method3(pdf_path: Path) -> bool:
    """
    Applies Method 3 watermark removal in-place on the given PDF file.
    
    Returns:
        bool: True if watermarks were detected and removed, False otherwise.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        logger.warning(f"Watermark remover: file not found at {pdf_path}")
        return False

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Watermark remover: failed to open {pdf_path.name}: {e}")
        return False

    modified = False

    # 1. Identify and blank all watermark Form XObjects (diagonal repeating background watermark)
    xobjs_to_blank = set()
    for page in doc:
        try:
            for x in page.get_xobjects():
                xref, name = x[0], x[1]
                if "pcm" in name.lower() or "watermark" in name.lower():
                    xobjs_to_blank.add(xref)
        except Exception:
            pass

    for xref in xobjs_to_blank:
        try:
            if doc.xref_is_stream(xref) and len(doc.xref_stream(xref)) > 0:
                doc.update_stream(xref, b"")
                modified = True
        except Exception:
            pass

    # 2. Strip inline vector watermarks (/gRLs0-0 gs payload) and footer text blocks from page content streams
    for page in doc:
        page_modified = False
        try:
            for xref in page.get_contents():
                raw = doc.xref_stream(xref)
                # Primary: PapaCambridge inline vector watermark + footer block starts at /gRLs0-0
                idx = raw.find(b"/gRLs0-0")
                if idx != -1:
                    start_q = raw.rfind(b"\nq\n", 0, idx)
                    if start_q != -1:
                        cleaned = raw[:start_q] + b"\nQ\n"
                        doc.update_stream(xref, cleaned)
                        modified = True
                        page_modified = True
                        continue

                # Fallback: Strip individual text blocks and divider rule if /gRLs0-0 is absent
                if b"PapaCambridge" in raw or b"papacambridge" in raw:
                    cleaned = TEXT_BT_ET_RE.sub(b"\n", raw)
                    cleaned = DIVIDER_LINE_RE.sub(b"\n", cleaned)
                    if cleaned != raw:
                        doc.update_stream(xref, cleaned)
                        modified = True
                        page_modified = True

            if page_modified:
                page.clean_contents()
        except Exception as e:
            logger.warning(f"Watermark remover: error processing page {page.number} of {pdf_path.name}: {e}")

    if not modified:
        doc.close()
        return False

    # Save to a temporary file and atomically replace original
    temp_path = pdf_path.with_suffix(".tmp_clean.pdf")
    try:
        doc.save(temp_path, deflate=True)
        doc.close()
        temp_path.replace(pdf_path)
        logger.info(f"Method 3 watermark removal succeeded in-place for {pdf_path.name}")
        return True
    except Exception as e:
        logger.error(f"Watermark remover: failed saving cleaned PDF {pdf_path.name}: {e}")
        doc.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False


def clean_paper_watermark_if_eligible(pdf_path: Path, year: int | None = None, min_year: int | None = None) -> bool:
    """
    Checks if paper belongs to year >= min_year (default from settings: 2026),
    and if so, invokes Method 3 watermark removal in-place.
    """
    effective_min_year = min_year if min_year is not None else settings.scraper_watermark_min_year
    effective_year = year or extract_year_from_filename(pdf_path.name)

    if effective_year is not None and effective_year >= effective_min_year:
        return remove_watermark_method3(pdf_path)
    return False
