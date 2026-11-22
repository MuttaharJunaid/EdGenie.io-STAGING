"""
Legacy Exam Paper Cleanup Script (2010–2018 Papers)

Performs lossless detection and removal of:
1. Top-right 'PapaCambridge' / 'StudentBounty' diagonal watermark bands (and peripheral QQAPIm watermark images)
   without affecting nearby question text, dotted answer lines, or diagrams.
2. Right-hand margin 'For Examiner's use only' text and continuous vertical margin lines
   on question pages (pages >= 2).
3. Watermark Form XObjects (e.g. XtremePapers / StudentBounty cover page artifacts).

Usage:
    python clean_legacy_watermarks.py <directory_or_pdf> [--output-dir <path>] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from pipeline.scraper.watermark_remover import detect_and_remove_legacy_watermarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legacy_cleanup")


def process_directory(
    target_path: Path,
    output_dir: Path | None = None,
    recursive: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Processes all PDFs in target_path.
    """
    target_path = Path(target_path).resolve()
    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        pdf_files = [target_path]
    elif target_path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(target_path.glob(pattern))
    else:
        logger.error(f"Path does not exist: {target_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) to inspect in {target_path}")

    results = []
    total_imgs = 0
    total_margins = 0
    total_xobjs = 0
    total_modified = 0

    for idx, p in enumerate(pdf_files, 1):
        out_p = (output_dir / p.name) if output_dir else None
        res = detect_and_remove_legacy_watermarks(p, output_path=out_p, dry_run=dry_run)
        results.append(res)

        if res["modified"]:
            total_modified += 1
            total_imgs += res["watermark_images_deleted"]
            total_margins += res["examiner_margins_removed"]
            total_xobjs += res["form_xobjects_blanked"]
            logger.info(
                f"[{idx}/{len(pdf_files)}] {p.name}: "
                f"cleaned {res['watermark_images_deleted']} watermark image(s), "
                f"{res['examiner_margins_removed']} examiner margin(s), "
                f"{res['form_xobjects_blanked']} Form XObject(s)"
            )
        else:
            logger.debug(f"[{idx}/{len(pdf_files)}] {p.name}: clean (no watermarks/margins)")

    action = "Detected (dry-run)" if dry_run else "Cleaned"
    logger.info(
        f"\n==================== SUMMARY ====================\n"
        f"Total PDFs Scanned:         {len(pdf_files)}\n"
        f"PDFs Requiring Cleanup:     {total_modified}\n"
        f"Watermark Images {action}:   {total_imgs}\n"
        f"Examiner Margins {action}:   {total_margins}\n"
        f"Form XObjects {action}:      {total_xobjs}\n"
        f"================================================="
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lossless detection and removal of PapaCambridge bands and Examiner margins (2010–2018 CAIE papers)"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Target directory (or single PDF file) to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. If omitted, cleans in-place.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories recursively",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report without modifying files",
    )

    args = parser.parse_args()

    target_dir = Path(args.directory)
    output_dir = Path(args.output_dir) if args.output_dir else None

    process_directory(
        target_path=target_dir,
        output_dir=output_dir,
        recursive=not args.no_recursive,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
