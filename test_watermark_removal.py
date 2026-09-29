"""
Benchmark script implementing all 5 local watermark removal methods on 5054_s26_qp_11.pdf.

Methods:
1. Content Stream Regex (Text-based watermarks in BT...ET blocks)
2. PDF Operator Filtering (Structural content stream filtering for XObject calls, text blocks, and divider rule)
3. Enhanced Method 3: Complete Watermark Eradication
   - Blanks all 16 FormXob.pcm background repeating XObjects
   - Strips the inline vector watermark (Drawing 49-74: raw bezier paths with /gRLs0-0 gs at shifting page positions)
   - Strips both footer text sets ('PapaCambridge' and 'papacambridge.com' with \056 escapes)
   - Strips the footer divider rule and tracking metadata
4. PyMuPDF Native Redaction (Visual text search and footer area redaction)
5. Graphics State / Transparency Manipulation (ExtGState alpha /ca 0.0 & gs block removal)
"""

import os
import re
import sys
import shutil
from pathlib import Path
import pymupdf as fitz

SRC_PDF = Path("/Users/myco/Desktop/EdGenie/pipeline/storage/5054/2026/jun/5054_s26_qp_22.pdf")
STORAGE_DIR = SRC_PDF.parent
PROJECT_ROOT = Path("/Users/myco/Desktop/EdGenie")


def run_method_1(src_path: Path, out_path: Path):
    """
    Method 1: Remove by Content Stream (Text-based watermarks)
    """
    print(f"\n--- Running Method 1: Content Stream Regex -> {out_path.name} ---")
    doc = fitz.open(src_path)
    total_stripped = 0

    bt_et_pattern = re.compile(
        rb"BT(?:(?!ET).)*?(?:papacambridge|PapaCambridge|Downloaded\s+from|Trace\s+ID)(?:(?!ET).)*?ET",
        re.DOTALL | re.IGNORECASE,
    )

    for page in doc:
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            matches = list(bt_et_pattern.finditer(stream))
            if matches:
                total_stripped += len(matches)
                cleaned = bt_et_pattern.sub(b"\n", stream)
                doc.update_stream(xref, cleaned)
        page.clean_contents()

    doc.save(out_path, deflate=True)
    doc.close()
    print(f"Method 1 Complete: Stripped {total_stripped} text blocks.")


def run_method_2(src_path: Path, out_path: Path):
    """
    Method 2: Remove by PDF Operator Filtering
    """
    print(f"\n--- Running Method 2: PDF Operator Filtering -> {out_path.name} ---")
    doc = fitz.open(src_path)

    xobj_pattern = re.compile(
        rb"q\s+[-0-9\.\s]+cm\s+/FormXob\.pcm\s+Do\s+Q\s*",
        re.DOTALL,
    )
    standalone_xobj = re.compile(rb"/FormXob\.pcm\s+Do\s*", re.DOTALL)

    text_pattern = re.compile(
        rb"BT(?:(?!ET).)*?(?:papacambridge|PapaCambridge|Downloaded\s+from|Trace\s+ID)(?:(?!ET).)*?ET",
        re.DOTALL | re.IGNORECASE,
    )

    footer_line_pattern = re.compile(rb"(?:n\s+)?36\s+15\s+m\s+559\.28\s+15\s+l\s+S", re.DOTALL)

    for page in doc:
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            cleaned = xobj_pattern.sub(b"", stream)
            cleaned = standalone_xobj.sub(b"", cleaned)
            cleaned = text_pattern.sub(b"\n", cleaned)
            cleaned = footer_line_pattern.sub(b"\n", cleaned)
            doc.update_stream(xref, cleaned)
        page.clean_contents()

    doc.save(out_path, deflate=True)
    doc.close()
    print(f"Method 2 Complete: Cleaned XObject invocations, text blocks, and divider rule.")


def run_method_3(src_path: Path, out_path: Path):
    """
    Method 3 (Enhanced): Complete Watermark Eradication
    1. Globally blanks all Form XObjects responsible for the diagonal background watermark (FormXob.pcm).
    2. Strips the inline vector watermark payload (Drawing 49-74: raw bezier paths rendered with /gRLs0-0 gs
       placed on left/center/right of pages 2-15).
    3. Strips both footer text sets ('PapaCambridge' and 'papacambridge.com' with octal \\056 escapes)
       along with the footer divider rule and tracking metadata.
    """
    print(f"\n--- Running Method 3 (Enhanced): Complete Watermark Eradication -> {out_path.name} ---")
    doc = fitz.open(src_path)

    # 1. Blank all Form XObjects (diagonal repeating background watermark)
    xobjs_to_blank = set()
    for page in doc:
        for x in page.get_xobjects():
            xref, name = x[0], x[1]
            if "pcm" in name.lower() or "watermark" in name.lower():
                xobjs_to_blank.add(xref)

    for xref in xobjs_to_blank:
        doc.update_stream(xref, b"")

    # 2. Strip the inline vector watermark & footer injection payload from each page
    for page in doc:
        for xref in page.get_contents():
            raw = doc.xref_stream(xref)
            idx = raw.find(b"/gRLs0-0")
            if idx != -1:
                start_q = raw.rfind(b"\nq\n", 0, idx)
                if start_q != -1:
                    cleaned = raw[:start_q] + b"\nQ\n"
                    doc.update_stream(xref, cleaned)
        page.clean_contents()

    doc.save(out_path, deflate=True)
    doc.close()
    print(f"Method 3 Complete: Blanked {len(xobjs_to_blank)} Form XObjects & stripped all inline vector/footer watermarks.")


def run_method_4(src_path: Path, out_path: Path):
    """
    Method 4: Remove by Redaction (PyMuPDF native)
    """
    print(f"\n--- Running Method 4: PyMuPDF Native Redaction -> {out_path.name} ---")
    doc = fitz.open(src_path)
    total_redacts = 0

    for page in doc:
        for term in ["PapaCambridge", "papacambridge.com", "Downloaded from", "Trace ID"]:
            rects = page.search_for(term)
            for r in rects:
                page.add_redact_annot(r, fill=(1, 1, 1))
                total_redacts += 1

        page_h = page.rect.height
        footer_rect = fitz.Rect(0, page_h - 32, page.rect.width, page_h)
        page.add_redact_annot(footer_rect, fill=(1, 1, 1))
        total_redacts += 1

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    doc.save(out_path, deflate=True)
    doc.close()
    print(f"Method 4 Complete: Applied {total_redacts} redaction annotations.")


def run_method_5(src_path: Path, out_path: Path):
    """
    Method 5: Remove by Graphics State / Transparency Manipulation
    """
    print(f"\n--- Running Method 5: Graphics State / Transparency Manipulation -> {out_path.name} ---")
    doc = fitz.open(src_path)
    modified_pages = 0

    for p in doc:
        try:
            page_obj = doc.xref_object(p.xref)
            if "/gRLs0" in page_obj:
                new_page_obj = re.sub(r"/ca\s+\.?\d+", "/ca 0.0", page_obj)
                doc.update_object(p.xref, new_page_obj)
        except Exception:
            pass

    gs_block_pattern = re.compile(
        rb"q\s+[-0-9\.\s]+rg\s+/gRLs0(?:-0)?\s+gs\s+.*?(?:/FormXob\.pcm\s+Do\s+Q\s*)+Q",
        re.DOTALL,
    )
    standalone_gs = re.compile(rb"/gRLs0(?:-0)?\s+gs\s*", re.DOTALL)

    for page in doc:
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            cleaned = gs_block_pattern.sub(b"", stream)
            cleaned = standalone_gs.sub(b"", cleaned)
            if cleaned != stream:
                doc.update_stream(xref, cleaned)
                modified_pages += 1
        page.clean_contents()

    doc.save(out_path, deflate=True)
    doc.close()
    print(f"Method 5 Complete: Neutralized ExtGState opacity and stripped graphics state blocks.")


def main():
    if not SRC_PDF.exists():
        print(f"Error: {SRC_PDF} does not exist!")
        sys.exit(1)

    print(f"Source PDF: {SRC_PDF} ({SRC_PDF.stat().st_size:,} bytes)")

    methods = [
        ("method1", run_method_1),
        ("method2", run_method_2),
        ("method3", run_method_3),
        ("method4", run_method_4),
        ("method5", run_method_5),
    ]

    results = []

    for name, func in methods:
        out_storage = STORAGE_DIR / f"5054_s26_qp_11_{name}.pdf"
        func(SRC_PDF, out_storage)
        
        out_root = PROJECT_ROOT / f"5054_s26_qp_11_{name}.pdf"
        shutil.copy2(out_storage, out_root)
        
        results.append((name, out_storage, out_root))

    print("\n" + "=" * 80)
    print("DETAILED VERIFICATION & WATERMARK REMOVAL BENCHMARK REPORT")
    print("=" * 80)

    for name, out_path, out_root in results:
        doc = fitz.open(out_path)
        p = doc[1]  # check page 2
        txt = p.get_text()
        has_text_wm = any(k in txt.lower() for k in ["papacambridge", "downloaded from"])
        contents = p.read_contents()
        has_xobj_call = b"/FormXob.pcm Do" in contents
        has_gs_call = b"/gRLs0" in contents
        xobjs = p.get_xobjects()
        
        xobj_stream_len = 0
        if xobjs:
            try:
                xobj_stream_len = len(doc.xref_stream(xobjs[0][0]))
            except Exception:
                pass

        total_pages = len(doc)
        page2_text_len = len(txt)
        doc.close()

        print(f"\n* [{name.upper()}]")
        print(f"    - Output Storage:                {out_path}")
        print(f"    - Output Root:                   {out_root}")
        print(f"    - Output File Size:              {out_path.stat().st_size:,} bytes")
        print(f"    - Page Count:                    {total_pages}")
        print(f"    - Page 2 Question Text Preserved: {page2_text_len} characters")
        print(f"    - Watermark Text Remaining:      {'YES (Not Removed)' if has_text_wm else 'NO (Successfully Removed)'}")
        print(f"    - XObject /FormXob Call in Stream:{'YES' if has_xobj_call else 'NO (Stripped)'}")
        print(f"    - XObject Stream Size:           {xobj_stream_len} bytes {'(Blanked to 0)' if xobj_stream_len == 0 else ''}")
        print(f"    - Graphics State /gRLs0 in Stream:{'YES' if has_gs_call else 'NO (Stripped)'}")


if __name__ == "__main__":
    main()
