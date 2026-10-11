import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from pipeline.config import settings
from pipeline.scraper.downloader import PDFDownloader
from pipeline.scraper.models import PaperDocumentType, PaperFileInfo, ScrapeManifest, ScrapeTarget
from pipeline.scraper.url_builder import URLBuilder
from pipeline.scraper.watermark_remover import remove_watermark_method3

logger = logging.getLogger(__name__)

class PapaCambridgeScraper:
    """Downloads all PDFs from a PapaCambridge year/session page."""

    FILENAME_PATTERN = re.compile(
        r"^([0-9]{4})_([a-zA-Z])([0-9]{2})_([a-zA-Z0-9]+)(?:_([0-9a-zA-Z]+))?(?:\..*)?$"
    )

    def __init__(
        self,
        target: ScrapeTarget,
        downloader: PDFDownloader | None = None,
        session: requests.Session | None = None,
        navigator: Any | None = None,
    ):
        self.target = target
        self.session = session or requests.Session()
        if "User-Agent" not in self.session.headers:
            self.session.headers.update({
                "User-Agent": settings.scraper_user_agent
            })
        self.downloader = downloader or PDFDownloader()
        from pipeline.scraper.navigator import PapaCambridgeNavigator
        self.navigator = navigator or PapaCambridgeNavigator(self.session)

    def _extract_pdf_urls_static(self, html: str) -> list[str]:
        """Parse download links from static HTML using download button locators."""
        from pipeline.scraper.validator import get_expected_prefix
        expected_prefix = get_expected_prefix(self.target.subject_code, self.target.year, self.target.session)
        return self.navigator.extract_download_links_from_session_page(html, expected_prefix)

    def _extract_pdf_urls_playwright(self, url: str) -> list[str]:
        """Fallback: uses Playwright to render JS and extract PDF URLs."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("Playwright is not installed. Skipping JS rendering fallback.")
            return []

        logger.info("Using Playwright fallback to scrape JS-rendered content...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=settings.scraper_timeout * 1000)
                # Wait for any relevant dynamic elements to load
                page.wait_for_timeout(3000)

                # Get the fully rendered content
                content = page.content()
                browser.close()

                return self._extract_pdf_urls_static(content)
        except Exception as e:
            logger.error(f"Playwright rendering failed: {e}")
            return []

    def classify_pdf(self, filename: str, url: str) -> PaperFileInfo:
        """
        Parses filename into CAIE metadata:
        e.g., 7707_s23_qp_12.pdf -> PaperFileInfo
        """
        match = self.FILENAME_PATTERN.match(filename)

        doc_type = PaperDocumentType.OTHER
        paper_num = None
        variant = None
        subject_code = None
        year = None
        session = None

        if match:
            # Group 1: Subject code
            # Group 2: Session (s/w/etc)
            # Group 3: Year (23)
            # Group 4: Document type abbreviation (qp/ms/etc)
            # Group 5: Variant/Paper code (12)
            subject_code = match.group(1)
            session_letter = match.group(2).lower()
            if session_letter == 's':
                session = "jun"
            elif session_letter == 'w':
                session = "nov"
            else:
                session = session_letter

            try:
                year = 2000 + int(match.group(3))
            except ValueError:
                pass

            type_part = match.group(4)
            variant_part = match.group(5)

            doc_type = PaperDocumentType.from_filename_part(type_part)
            if variant_part:
                variant = variant_part
                # Try to parse paper number out of variant.
                # Standard variant is a 2-digit number (e.g. 12 -> paper 1, variant 2)
                # If variant_part is a digit sequence
                if variant_part.isdigit():
                    if len(variant_part) == 2:
                        paper_num = int(variant_part[0])
                    elif len(variant_part) == 1:
                        paper_num = int(variant_part)
        else:
            # Fallback parsing if regex doesn't match
            # Search for keyword markers in the filename
            lower_name = filename.lower()
            if "_qp_" in lower_name or "qp" in lower_name:
                doc_type = PaperDocumentType.QUESTION_PAPER
            elif "_ms_" in lower_name or "ms" in lower_name:
                doc_type = PaperDocumentType.MARK_SCHEME
            elif "_er_" in lower_name or "er" in lower_name:
                doc_type = PaperDocumentType.EXAMINER_REPORT
            elif "_in_" in lower_name or "in" in lower_name:
                doc_type = PaperDocumentType.INSERT
            elif "_gt_" in lower_name or "gt" in lower_name:
                doc_type = PaperDocumentType.GRADE_THRESHOLD
            elif "_ci_" in lower_name or "ci" in lower_name:
                doc_type = PaperDocumentType.CONFIDENTIAL_INSTRUCTIONS
            elif "_rp_" in lower_name:
                doc_type = PaperDocumentType.RECORDING_PAPER
            elif "_sf_" in lower_name:
                doc_type = PaperDocumentType.SOUND_FILE

            # Extract numbers
            numbers = re.findall(r"\d+", filename)
            if len(numbers) >= 1:
                subject_code = numbers[0]

            if len(numbers) >= 2:
                # E.g. [7707, 23] -> year is 2023
                try:
                    yr_val = int(numbers[1])
                    if yr_val < 100:
                        year = 2000 + yr_val
                    else:
                        year = yr_val
                except ValueError:
                    pass

            if "may" in lower_name or "jun" in lower_name or "_s" in lower_name:
                session = "jun"
            elif "oct" in lower_name or "nov" in lower_name or "_w" in lower_name:
                session = "nov"

            # Usually the last number after the subject code is the paper number / variant
            if len(numbers) >= 3:
                # E.g. [7707, 23, 12] or similar
                last_num = numbers[-1]
                if len(last_num) == 2 and last_num.isdigit():
                    paper_num = int(last_num[0])
                    variant = last_num

        return PaperFileInfo(
            filename=filename,
            url=url,
            document_type=doc_type,
            paper_number=paper_num,
            variant=variant,
            subject_code=subject_code,
            year=year,
            session=session
        )

    def scrape(self, force: bool = False, dry_run: bool = False) -> list[Path]:
        """
        Main function to trigger page fetching and PDF downloading.
        Returns a list of local paths to the downloaded files.
        """
        # Step 1: Hierarchical navigation strategy (authoritative route discovery)
        candidate_urls: list[str] = []
        try:
            nav_session_url = self.navigator.resolve_session_page_url(self.target)
            if nav_session_url:
                candidate_urls.append(nav_session_url)
        except Exception as e:
            logger.warning(f"Hierarchical navigation encountered an issue: {e}. Falling back to URLBuilder.")

        # Step 2: Append fallback URLs from URLBuilder
        for u in URLBuilder.get_candidate_urls(self.target):
            if u not in candidate_urls:
                candidate_urls.append(u)

        pdf_urls = []
        chosen_url = None

        for page_url in candidate_urls:
            logger.info(f"Targeting PapaCambridge candidate page: {page_url}")

            # Step 1: Fetch Level 3 candidate page HTML
            try:
                response = self.session.get(page_url, timeout=settings.scraper_timeout)
                if response.status_code == 404:
                    logger.debug(f"Candidate page returned 404: {page_url}")
                    continue
                response.raise_for_status()
                html = response.text
            except requests.RequestException as e:
                logger.debug(f"Failed to fetch static HTML for candidate {page_url}: {e}")
                html = ""

            # Step 2: Extract download URLs from static HTML
            if html:
                pdf_urls = self._extract_pdf_urls_static(html)

            # Step 3: If no URLs found, try Playwright fallback
            if not pdf_urls:
                logger.debug(f"No matching PDF URLs found in static HTML for candidate {page_url}. Trying Playwright fallback...")
                pdf_urls = self._extract_pdf_urls_playwright(page_url)

            # If we found valid PDF files matching this subject, we stop checking other candidates
            if pdf_urls:
                chosen_url = page_url
                logger.info(f"Successfully validated candidate page: {chosen_url}")
                break

        if not pdf_urls:
            logger.warning(f"No PDF files found for target: {self.target.subject_code} - {self.target.year} - {self.target.session}")
            return []

        logger.info(f"Found {len(pdf_urls)} matching PDF download links from: {chosen_url}")

        downloaded_paths = []
        manifest_files = []
        storage_dir = self.target.storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: Download each PDF
        for url in pdf_urls:
            filename = url.split("/")[-1]
            dest_path = storage_dir / filename

            # Classify file info
            file_info = self.classify_pdf(filename, url)

            if dry_run:
                logger.info(f"[Dry Run] Would download: {filename} ({file_info.document_type.name})")
                manifest_files.append(file_info)
                continue

            try:
                # Download
                _, size_bytes, sha256_hash = self.downloader.download(url, dest_path, force=force)
                file_info.size_bytes = size_bytes
                file_info.sha256 = sha256_hash

                # Automatically remove watermarks using Method 3 for papers of year >= 2026
                paper_year = file_info.year or self.target.year
                if paper_year is not None and paper_year >= settings.scraper_watermark_min_year:
                    logger.info(
                        f"Paper year {paper_year} >= {settings.scraper_watermark_min_year}: "
                        f"invoking Method 3 watermark removal on {dest_path.name}"
                    )
                    cleaned = remove_watermark_method3(dest_path)
                    if cleaned:
                        file_info.size_bytes = dest_path.stat().st_size
                        file_info.sha256 = self.downloader._calculate_sha256(dest_path)
                        logger.info(
                            f"Method 3 watermark removal applied in-place for {dest_path.name} "
                            f"(new size: {file_info.size_bytes:,} bytes)"
                        )

                downloaded_paths.append(dest_path)
                manifest_files.append(file_info)
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")

        # Step 5: Save manifest.json
        if manifest_files:
            manifest_path = storage_dir / "scrape_manifest.json"
            manifest = ScrapeManifest(
                scraped_at=datetime.now(timezone.utc),
                target=self.target,
                files=manifest_files
            )

            # Serialize with datetime handling and pydantic V2 dump
            if not dry_run:
                try:
                    with open(manifest_path, "w") as f:
                        f.write(manifest.model_dump_json(indent=2))
                    logger.info(f"Saved manifest to {manifest_path}")
                except Exception as e:
                    logger.error(f"Failed to save manifest file: {e}")

        return downloaded_paths

