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
from typing import Any

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
    m2 = re.search(r"\b(?:19|20)(\d{2})\b", filename)
    if m2:
        return int(m2.group(0))
    from pipeline.unified.paper_type_detector import parse_caie_filename

    try:
        parsed = parse_caie_filename(filename)
        if parsed.get("subject_code") != "unknown":
            return int(parsed["year"])
    except Exception:
        pass
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
    for page in doc.pages():
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
    for page in doc.pages():
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


# Watermark Form XObject identifiers
WATERMARK_XOBJ_KEYWORDS = ("pcm", "watermark", "xtremepapers", "papacambridge", "studentbounty")


def detect_and_remove_legacy_watermarks(
    pdf_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Scans a single 2010-2018 PDF and purges:
    - Top-right watermark bands and associated peripheral watermark images (QQAPIm*).
    - Right-hand 'For Examiner's use' header text and vertical margin line on question pages.
    - Watermark Form XObjects (e.g. XtremePapers, StudentBounty, pcm).

    Args:
        pdf_path: Path to the PDF file.
        output_path: Destination path. If None, modifies pdf_path in-place atomically.
        dry_run: If True, only reports detections without modifying the file.

    Returns:
        dict with detection statistics:
            {
                "file": str,
                "modified": bool,
                "watermark_images_deleted": int,
                "examiner_margins_removed": int,
                "form_xobjects_blanked": int,
            }
    """
    pdf_path = Path(pdf_path).resolve()
    target_out = Path(output_path).resolve() if output_path else pdf_path

    stats: dict[str, Any] = {
        "file": pdf_path.name,
        "modified": False,
        "watermark_images_deleted": 0,
        "examiner_margins_removed": 0,
        "form_xobjects_blanked": 0,
    }

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Cannot open {pdf_path.name}: {e}")
        return stats

    if len(doc) == 0:
        doc.close()
        return stats

    modified = False

    # Collect all page content stream xrefs to ensure they are never blanked as Form XObjects
    page_content_xrefs: set[int] = set()
    for page in doc:
        try:
            page_content_xrefs.update(page.get_contents())
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1. Blank Watermark Form XObjects (e.g. XtremePapers, StudentBounty, pcm)
    # -------------------------------------------------------------------------
    xobjs_to_blank: set[int] = set()
    for page in doc:
        try:
            for x in page.get_xobjects():
                xref, name = x[0], x[1]
                if xref in page_content_xrefs:
                    continue
                if any(k in name.lower() for k in WATERMARK_XOBJ_KEYWORDS):
                    xobjs_to_blank.add(xref)
                else:
                    obj_def = doc.xref_object(xref)
                    if "/Private /Watermark" in obj_def:
                        xobjs_to_blank.add(xref)
                    elif "/Subtype /Form" in obj_def and doc.xref_is_stream(xref):
                        s = doc.xref_stream(xref)
                        if any(
                            kw in s
                            for kw in (
                                b"XtremePapers.com",
                                b"PapaCambridge.com",
                                b"StudentBounty.com",
                                b"papacambridge",
                            )
                        ):
                            xobjs_to_blank.add(xref)
        except Exception:
            pass

    for xref in xobjs_to_blank:
        try:
            if doc.xref_is_stream(xref) and len(doc.xref_stream(xref)) > 0:
                if not dry_run:
                    doc.update_stream(xref, b"")
                modified = True
                stats["form_xobjects_blanked"] += 1
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 2. Page-by-page cleanup
    # -------------------------------------------------------------------------
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        pw = page.rect.width
        ph = page.rect.height
        page_changed = False

        # A. Detect and strip top-right watermark band and peripheral watermark images
        wm_image_names: list[str] = []
        for img in page.get_images():
            xref, smask, w, h, bpc, cs, alt, name, filt = img[:9]
            if w <= 1 and h <= 1:
                continue
            rects = page.get_image_rects(xref)
            for r in rects:
                # Top-right diagonal band (PapaCambridge / StudentBounty)
                is_tr_band = (
                    (r.x1 >= pw - 25 and r.y0 <= 25 and r.width >= 100 and r.height >= 100)
                    or (r.x0 >= 350 and r.y0 <= 15 and r.x1 >= pw - 5 and r.y1 <= 220)
                )
                # Known QQAPIm watermark artifacts
                is_qq_watermark = (
                    name.startswith("QQAPIm")
                    and (
                        (r.x0 >= 350 and r.y0 <= 15)  # top-right band
                        or (r.y0 >= ph - 80)          # bottom footer overlay/strip
                        or (r.x0 >= pw - 70 and r.width <= 70)  # right edge strip
                    )
                )
                if is_tr_band or is_qq_watermark:
                    wm_image_names.append(name)
                    break

        if wm_image_names:
            if not dry_run:
                for c_xref in page.get_contents():
                    stream = doc.xref_stream(c_xref)
                    orig_stream = stream
                    for name in wm_image_names:
                        pattern = re.compile(rb"/" + re.escape(name.encode("latin1")) + rb"\s+Do\b")
                        stream = pattern.sub(b"", stream)
                    if stream != orig_stream:
                        doc.update_stream(c_xref, stream)
                page_changed = True
            modified = True
            stats["watermark_images_deleted"] += len(wm_image_names)

        # B. Detect and remove right-hand 'For Examiner's Use' text and vertical margin line
        # Only on question pages (page > 1)
        if page_num > 1:
            # 1. Identify continuous vertical margin line
            vlines = [
                d["rect"]
                for d in page.get_drawings()
                if (
                    d["rect"].width <= 2.5
                    and d["rect"].height >= 250.0
                    and d["rect"].x0 >= 500.0
                    and d["rect"].y0 <= 120.0
                    and d["rect"].y1 >= 700.0
                )
            ]

            # 2. Identify examiner header text in right margin
            exam_blocks = [
                fitz.Rect(b[:4])
                for b in page.get_text("blocks")
                if (
                    b[0] >= 480.0
                    and b[1] <= 140.0
                    and any(w in b[4].lower() for w in ("examiner", "use"))
                )
            ]

            if vlines or exam_blocks:
                if not dry_run:
                    # 1. Blank out examiner header text tokens directly in content streams
                    # without touching font or color specifications (/Separation /Black spot colors)
                    examiner_triple_pattern = re.compile(
                        rb"(\[\s*\(Fo\)[^\]]*?\([^\)]*?r\)\s*\]\s*T[Jj][\s\S]{0,120}?\(Examiner[^\)]*?\)[\s\S]{0,120}?\[?\s*\(\s*Use\s*\)\s*\]?\s*T[Jj])",
                        re.IGNORECASE,
                    )
                    for c_xref in page.get_contents():
                        stream = doc.xref_stream(c_xref)
                        orig_stream = stream
                        # Blank out the For...Examiner...Use sequence if present
                        stream = examiner_triple_pattern.sub(
                            lambda m: re.sub(
                                rb"\([^\)]*\)",
                                lambda sm: b"(" + b" " * (len(sm.group(0)) - 2) + b")",
                                m.group(0),
                            ),
                            stream,
                        )
                        # Also blank any standalone (Examiner...) string literals in the stream
                        stream = re.sub(
                            rb"(\()([^\)]*?Examiner[^\)]*?)(\))",
                            lambda m: m.group(1) + b" " * len(m.group(2)) + m.group(3),
                            stream,
                            flags=re.IGNORECASE,
                        )
                        # And blank any hex-encoded Examiner occurrences <...4578616d696e6572...>
                        stream = re.sub(
                            rb"(<[^>]*?(?:4578616d696e6572|6578616d696e6572)[^>]*?>)",
                            lambda m: b"<" + b"20" * ((len(m.group(0)) - 2) // 2) + b">",
                            stream,
                            flags=re.IGNORECASE,
                        )
                        if stream != orig_stream:
                            doc.update_stream(c_xref, stream)

                    # 2. Visually mask the right margin (x >= 525) to cleanly cover the vertical rule and any margin artifacts
                    # Using page.draw_rect safely appends a white mask in an isolated graphics state without rewriting existing streams
                    margin_rect = fitz.Rect(525.0, 50.0, pw, ph - 45.0)
                    page.draw_rect(margin_rect, color=(1, 1, 1), fill=(1, 1, 1))
                    page_changed = True
                modified = True
                stats["examiner_margins_removed"] += 1

    stats["modified"] = modified

    if not modified or dry_run:
        doc.close()
        return stats

    # Save cleaned document atomically with garbage collection of unreferenced watermark objects
    target_out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_out.with_suffix(".tmp_cleaned.pdf")
    try:
        doc.save(temp_path, garbage=3, deflate=True)
        doc.close()
        temp_path.replace(target_out)
        logger.info(
            f"Legacy cleanup succeeded for {target_out.name}: "
            f"purged {stats['watermark_images_deleted']} watermark image(s), "
            f"{stats['examiner_margins_removed']} examiner margin(s), "
            f"{stats['form_xobjects_blanked']} Form XObject(s)"
        )
    except Exception as e:
        logger.error(f"Failed to save cleaned PDF {target_out.name}: {e}")
        doc.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        stats["modified"] = False

    return stats


def remove_legacy_watermarks(pdf_path: Path) -> bool:
    """
    Applies legacy watermark (top-right diagonal band, peripheral artifacts)
    and right-hand examiner margin removal in-place on 2010-2018 papers.

    Returns:
        bool: True if watermarks/margins were detected and removed, False otherwise.
    """
    stats = detect_and_remove_legacy_watermarks(pdf_path)
    return stats.get("modified", False)


def clean_paper_watermark_if_eligible(
    pdf_path: Path,
    year: int | None = None,
    min_year: int | None = None,
    legacy_start_year: int | None = None,
    legacy_end_year: int | None = None,
) -> bool:
    """
    Checks paper year and invokes appropriate cleanup:
    - If year >= min_year (default 2026): invokes Method 3 watermark removal.
    - If legacy_start_year <= year <= legacy_end_year (default 2010–2018):
      invokes legacy watermark and examiner margin removal.
    """
    effective_min_year = min_year if min_year is not None else settings.scraper_watermark_min_year
    effective_legacy_start = (
        legacy_start_year if legacy_start_year is not None else settings.scraper_legacy_cleanup_start_year
    )
    effective_legacy_end = (
        legacy_end_year if legacy_end_year is not None else settings.scraper_legacy_cleanup_end_year
    )
    effective_year = year or extract_year_from_filename(pdf_path.name)

    if effective_year is not None:
        if effective_year >= effective_min_year:
            return remove_watermark_method3(pdf_path)
        if effective_legacy_start <= effective_year <= effective_legacy_end:
            return remove_legacy_watermarks(pdf_path)
    return False

