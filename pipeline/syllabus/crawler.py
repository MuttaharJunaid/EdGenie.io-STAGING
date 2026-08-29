"""
pipeline/syllabus/crawler.py

Automated crawler to discover, inspect, and download official Cambridge International
syllabus PDFs directly from the Cambridge International website.

Flow
----
1. Navigates to the official subjects directory corresponding to the qualification level
   (e.g., https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/).
2. Locates and clicks the subject link matching the subject name and code.
3. Searches for the "Syllabuses" section/block on the subject page and gathers all PDF links.
4. Checks candidate PDFs in reverse order (bottom-up / most recent):
   - Opens/downloads the PDF and inspects page count via PyMuPDF.
   - If the PDF has <= 3 pages (e.g., an errata/update flyer), it skips to the previous one.
   - Once a PDF with > 3 pages is found (the complete syllabus document), downloads and saves it.
5. Returns the local Path to the downloaded full syllabus PDF.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.cambridgeinternational.org"

LEVEL_SUBJECT_URLS = {
    "o-level": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/",
    "o_level": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/",
    "olevel": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/",
    "o": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/",
    "a-level": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/subjects/",
    "a_level": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/subjects/",
    "alevel": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/subjects/",
    "as-and-a-level": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/subjects/",
    "a": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-advanced/cambridge-international-as-and-a-levels/subjects/",
    "igcse": "https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-igcse/subjects/",
}


class CambridgeSyllabusCrawler:
    """Automates finding and downloading the latest full Cambridge syllabus PDF."""

    def __init__(self, download_dir: Optional[Path] = None, timeout: int = 45):
        self.download_dir = download_dir or (Path(__file__).resolve().parent.parent.parent / "pipeline" / "storage" / "syllabi_pdfs")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def get_subjects_url(self, level: str) -> str:
        """Returns the canonical subjects index URL for the qualification level."""
        clean = level.lower().strip().replace(" ", "-")
        return LEVEL_SUBJECT_URLS.get(clean, LEVEL_SUBJECT_URLS["o-level"])

    def fetch_syllabus_pdf(self, subject: str, code: str, level: str) -> Tuple[Path, int]:
        """
        Navigates Cambridge website, finds subject syllabus PDF with > 3 pages,
        downloads it locally, and returns (local_pdf_path, page_count).
        """
        subjects_url = self.get_subjects_url(level)
        code_str = str(code).strip()
        name_clean = subject.strip().lower()

        logger.info(f"Connecting to Cambridge subjects index: {subjects_url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(self.timeout * 1000)

            # 1. Open subjects index
            try:
                page.goto(subjects_url, wait_until="domcontentloaded")
            except Exception as e:
                browser.close()
                raise RuntimeError(f"Failed to load Cambridge subjects index at {subjects_url}: {e}")

            # 2. Find subject link
            all_links = page.query_selector_all("a")
            target_href = None
            target_text = None

            # Priority 1: Link text contains code
            for a in all_links:
                txt = a.inner_text().strip()
                href = a.get_attribute("href") or ""
                if code_str in txt and "/programmes-and-qualifications/" in href and not href.endswith("/subjects/"):
                    target_href = href
                    target_text = txt
                    break

            # Priority 2: href contains code
            if not target_href:
                for a in all_links:
                    txt = a.inner_text().strip()
                    href = a.get_attribute("href") or ""
                    if code_str in href and "/programmes-and-qualifications/" in href and not href.endswith("/subjects/"):
                        target_href = href
                        target_text = txt
                        break

            # Priority 3: text contains subject name
            if not target_href:
                for a in all_links:
                    txt = a.inner_text().strip().lower()
                    href = a.get_attribute("href") or ""
                    if name_clean in txt and "/programmes-and-qualifications/" in href and not href.endswith("/subjects/"):
                        target_href = href
                        target_text = txt
                        break

            if not target_href:
                browser.close()
                raise ValueError(f"Could not find subject '{subject}' (code: {code}) on Cambridge subjects index: {subjects_url}")

            logger.info(f"Found subject link: '{target_text}' -> {target_href}")

            # 3. Navigate to subject page
            full_subject_url = urljoin(BASE_URL, target_href)
            logger.info(f"Loading subject page: {full_subject_url}")
            try:
                page.goto(full_subject_url, wait_until="domcontentloaded")
            except Exception as e:
                browser.close()
                raise RuntimeError(f"Failed to load subject page {full_subject_url}: {e}")

            # 4. Search for 'Syllabuses' section and all PDF links
            pdf_links_data = page.evaluate('''() => {
                const results = [];
                // Look for .syllabus-block with heading containing Syllabuse
                const blocks = Array.from(document.querySelectorAll('.syllabus-block'));
                for (const b of blocks) {
                    const title = b.querySelector('.syllabus-block-title, h3, h2');
                    if (title && title.innerText.toLowerCase().includes('syllabuse')) {
                        const anchors = Array.from(b.querySelectorAll('a'));
                        for (const a of anchors) {
                            const href = a.getAttribute('href') || '';
                            if (href.toLowerCase().includes('.pdf')) {
                                results.push({
                                    text: a.innerText.trim(),
                                    href: href
                                });
                            }
                        }
                    }
                }
                
                // Fallback: search all links on page matching syllabus .pdf
                if (results.length === 0) {
                    const allA = Array.from(document.querySelectorAll('a'));
                    for (const a of allA) {
                        const href = a.getAttribute('href') || '';
                        const text = a.innerText.trim();
                        if (href.toLowerCase().includes('.pdf') && (text.toLowerCase().includes('syllabus') || href.toLowerCase().includes('syllabus'))) {
                            results.push({ text: text, href: href });
                        }
                    }
                }
                return results;
            }''')

            browser.close()

        if not pdf_links_data:
            raise ValueError(f"No PDF syllabus links found on Cambridge page: {full_subject_url}")

        logger.info(f"Found {len(pdf_links_data)} PDF link(s) under Syllabuses.")

        # 5. Reverse-order inspection: start from last PDF under Syllabuses,
        # skip if <= 3 pages, keep looking until finding a PDF with > 3 pages.
        for item in reversed(pdf_links_data):
            rel_href = item["href"]
            title = item["text"]
            full_pdf_url = urljoin(BASE_URL, rel_href)
            logger.info(f"Inspecting candidate syllabus PDF: '{title}' -> {full_pdf_url}")

            try:
                resp = self.http_session.get(full_pdf_url, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code} fetching {full_pdf_url}, checking previous...")
                    continue

                pdf_bytes = resp.content
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()

                logger.info(f"Candidate PDF '{title}' has {page_count} page(s).")

                if page_count > 3:
                    # Found the complete syllabus PDF!
                    pdf_filename = f"{code_str}_{Path(rel_href).name}"
                    save_path = self.download_dir / pdf_filename
                    save_path.write_bytes(pdf_bytes)
                    logger.info(f"Successfully downloaded full syllabus PDF ({page_count} pages) to {save_path}")
                    return save_path, page_count
                else:
                    logger.info(f"Skipping '{title}' ({page_count} pages <= 3). Checking previous candidate...")

            except Exception as e:
                logger.warning(f"Error inspecting {full_pdf_url}: {e}")
                continue

        raise ValueError(
            f"Could not find any syllabus PDF with > 3 pages for {subject} ({code}) on Cambridge website."
        )
