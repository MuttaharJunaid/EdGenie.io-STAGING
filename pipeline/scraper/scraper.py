import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from pipeline.config import settings
from pipeline.scraper.downloader import PDFDownloader
from pipeline.scraper.models import PaperDocumentType, PaperFileInfo, ScrapeManifest, ScrapeTarget
from pipeline.scraper.url_builder import URLBuilder
from pipeline.scraper.watermark_remover import clean_paper_watermark_if_eligible
from pipeline.unified.paper_type_detector import parse_caie_filename

logger = logging.getLogger(__name__)

class PapaCambridgeScraper:
    """Downloads all PDFs from a PapaCambridge year/session page."""

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

    def classify_pdf(self, filename: str, url: str) -> PaperFileInfo:
        """
        Parses filename into CAIE metadata:
        e.g., 7707_s23_qp_12.pdf -> PaperFileInfo
        """
        meta = parse_caie_filename(filename)
        doc_type = PaperDocumentType.from_filename_part(meta.get("doc_type", "other"))

        return PaperFileInfo(
            filename=filename,
            url=url,
            document_type=doc_type,
            paper_number=meta.get("paper_number"),
            variant=meta.get("variant"),
            subject_code=meta.get("subject_code"),
            year=meta.get("year"),
            session=meta.get("session"),
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

                # Automatically clean watermarks/examiner margins if eligible (>= 2026 or 2010-2018)
                paper_year = file_info.year or self.target.year
                if paper_year is not None:
                    cleaned = clean_paper_watermark_if_eligible(dest_path, year=paper_year)
                    if cleaned:
                        file_info.size_bytes = dest_path.stat().st_size
                        file_info.sha256 = self.downloader._calculate_sha256(dest_path)
                        logger.info(
                            f"Watermark/margin cleanup applied in-place for {dest_path.name} "
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
                scraped_at=datetime.now(UTC),
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

