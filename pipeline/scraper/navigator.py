"""
pipeline/scraper/navigator.py

Navigational scraper for PapaCambridge past papers.

Implements a 3-tier hierarchy crawler:
1. Level 1 (Level Page):
   - O-Level: https://pastpapers.papacambridge.com/papers/caie/o-level?theme=lightTheme
   - AS & A-Level: https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme
   Locates the target subject via `<span class="wraptext" id="...">` matching subject code/name.
2. Level 2 (Subject Page):
   Saves the subject URL. Discovers all year & session links via `<span class="wraptext" id="...">`
   (e.g., `2026-May-June`, `2025-Oct-Nov`, `2024-March`).
3. Level 3 (Session Page):
   Navigates directly to the exact session URL and extracts PDF download links via
   download buttons (`<a class="badge badge-info" href="download_file.php?files=..." download="">`).
4. Re-uses the saved subject URL to navigate back and process subsequent sessions.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from pipeline.config import settings
from pipeline.scraper.models import ScrapeTarget
from pipeline.scraper.validator import get_expected_prefix

logger = logging.getLogger(__name__)


def normalize_level_url(level: str) -> str:
    """Returns the official Level 1 URL for PapaCambridge."""
    lvl = level.lower().strip()
    if "a-level" in lvl or "a2" in lvl or "as" in lvl:
        return "https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme"
    return "https://pastpapers.papacambridge.com/papers/caie/o-level?theme=lightTheme"


def parse_session_element_text(text: str) -> tuple[int | None, str | None]:
    """
    Parses a session span text into (year, session).
    Examples:
      - '2026-May-June' -> (2026, 'jun')
      - '2025-Oct-Nov'  -> (2025, 'nov')
      - '2024-March'    -> (2024, 'march')
      - '2022-Feb-March'-> (2022, 'march')
      - '2017 Nov'      -> (2017, 'nov')
      - '2015 Jun'      -> (2015, 'jun')
      - 'Topical Past Paper' -> (None, None)
    """
    t = text.strip()
    match = re.match(r"^(\d{4})[-\s]+(.*)$", t, re.IGNORECASE)
    if not match:
        return None, None

    year = int(match.group(1))
    raw_sess = match.group(2).lower().strip()

    if any(s in raw_sess for s in ["may", "june", "jun"]):
        session = "jun"
    elif any(s in raw_sess for s in ["oct", "nov", "november"]):
        session = "nov"
    elif any(s in raw_sess for s in ["mar", "march"]):
        session = "march"
    else:
        session = raw_sess

    return year, session


class PapaCambridgeNavigator:
    """
    Navigates PapaCambridge using element locators and link hierarchies.
    Caches discovered subject URLs and session maps to avoid redundant page fetches.
    """

    BASE_URL = "https://pastpapers.papacambridge.com"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        if "User-Agent" not in self.session.headers:
            self.session.headers.update({"User-Agent": settings.scraper_user_agent})

        # Cache: level -> {subject_code -> subject_url}
        self._level_subjects_cache: dict[str, dict[str, str]] = {}
        # Cache: subject_url -> {(year, session) -> session_url}
        self._subject_sessions_cache: dict[str, dict[tuple[int, str], str]] = {}
        # Cache: target_key -> saved subject URL
        self._saved_subject_urls: dict[str, str] = {}

    def get_subject_url(
        self,
        level: str | None,
        subject_code: str,
        subject_name: str | None = None,
    ) -> str | None:
        """
        Navigates to the Level 1 page and finds the subject URL strictly using
        the subject code locator on `<span class="wraptext" id="...">`.
        """
        clean_code = str(subject_code).strip()
        cache_key = f"{level}:{clean_code}" if level else f"auto:{clean_code}"

        if cache_key in self._saved_subject_urls:
            return self._saved_subject_urls[cache_key]

        # Determine level candidate search order
        if level:
            candidate_levels = [level]
            other_level = "o-level" if ("a-level" in level.lower() or "as" in level.lower()) else "as-and-a-level"
            candidate_levels.append(other_level)
        else:
            from pipeline.scraper.subject_registry import SubjectRegistry
            reg = SubjectRegistry.lookup_by_code(clean_code)
            if reg and "level" in reg:
                candidate_levels = [reg["level"], "o-level", "as-and-a-level"]
            elif clean_code.startswith(("9", "8")):
                candidate_levels = ["as-and-a-level", "o-level"]
            else:
                candidate_levels = ["o-level", "as-and-a-level"]

        seen_lvls = set()
        unique_candidate_levels = [l for l in candidate_levels if not (l in seen_lvls or seen_lvls.add(l))]

        found_url = None
        for lvl in unique_candidate_levels:
            level_url = normalize_level_url(lvl)
            logger.info(f"Navigating to Level 1 page for code '{clean_code}': {level_url}")

            try:
                resp = self.session.get(level_url, timeout=settings.scraper_timeout)
                if resp.status_code != 200:
                    continue
            except requests.RequestException as e:
                logger.error(f"Failed to fetch Level 1 page {level_url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Strictly find the subject locator on the basis of subject code
            for span in soup.find_all("span", class_="wraptext"):
                text = span.get_text(strip=True)
                span_id = span.get("id", "").strip()

                code_match = bool(
                    re.search(rf"\b{clean_code}\b", text) or re.search(rf"\b{clean_code}\b", span_id)
                )

                if code_match:
                    a_tag = span.find_parent("a")
                    if a_tag and a_tag.get("href"):
                        found_url = urljoin(self.BASE_URL, a_tag["href"].strip())
                        logger.info(f"Located subject locator '{text}' using code '{clean_code}' -> {found_url}")
                        break

            if found_url:
                break

        if found_url:
            self._saved_subject_urls[cache_key] = found_url

        return found_url

    def get_subject_sessions_map(self, subject_url: str) -> dict[tuple[int, str], str]:
        """
        Navigates to the Level 2 subject page (saved subject URL) and maps all
        available session folders to their exact session URLs.
        """
        if subject_url in self._subject_sessions_cache:
            return self._subject_sessions_cache[subject_url]

        logger.info(f"Navigating to Subject Page: {subject_url}")
        try:
            resp = self.session.get(subject_url, timeout=settings.scraper_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Subject page {subject_url}: {e}")
            return {}

        soup = BeautifulSoup(resp.text, "lxml")
        sessions_map: dict[tuple[int, str], str] = {}

        for span in soup.find_all("span", class_="wraptext"):
            txt = span.get_text(strip=True)
            y, s = parse_session_element_text(txt)
            if y is not None and s is not None:
                a_tag = span.find_parent("a")
                if a_tag and a_tag.get("href"):
                    session_url = urljoin(self.BASE_URL, a_tag["href"].strip())
                    sessions_map[(y, s)] = session_url
                    logger.debug(f"Mapped session: ({y}, '{s}') -> {session_url}")

        logger.info(f"Discovered {len(sessions_map)} session routes from subject page: {subject_url}")
        self._subject_sessions_cache[subject_url] = sessions_map
        return sessions_map

    def resolve_session_page_url(self, target: ScrapeTarget) -> str | None:
        """
        High-level resolution:
        1. Find subject URL from Level 1 page (and save it).
        2. Navigate to Subject URL to get sessions map.
        3. Match target (target.year, target.session).
        4. Return exact session URL if found.
        """
        subject_url = self.get_subject_url(
            level=target.level,
            subject_code=target.subject_code,
            subject_name=target.subject_name,
        )
        if not subject_url:
            logger.warning(
                f"Could not locate subject {target.subject_name} ({target.subject_code}) on Level 1 page."
            )
            return None

        sessions_map = self.get_subject_sessions_map(subject_url)
        s_clean = target.session.lower().strip()
        if s_clean in ["jun", "j", "s", "june", "may-june", "may-jun"]:
            normalized_s = "jun"
        elif s_clean in ["nov", "n", "w", "november", "oct-nov", "oct-november"]:
            normalized_s = "nov"
        elif s_clean in ["m", "march", "feb-march", "february-march"]:
            normalized_s = "march"
        else:
            normalized_s = s_clean

        session_url = sessions_map.get((target.year, normalized_s))
        if session_url:
            logger.info(
                f"Resolved authoritative session URL for {target.subject_code} {target.year} {target.session}: {session_url}"
            )
        else:
            logger.warning(
                f"Session ({target.year}, '{normalized_s}') not listed on subject page {subject_url}."
            )

        return session_url

    def extract_download_links_from_session_page(
        self,
        html: str,
        expected_prefix: str | None = None,
    ) -> list[str]:
        """
        Extracts download URLs from a Level 3 session page HTML using the
        download button locator:
          <a href="download_file.php?files=..." class="badge badge-info" download="">
        """
        soup = BeautifulSoup(html, "lxml")
        pdf_urls: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # 1. Official download button locator
            if "download_file.php?files=" in href:
                raw_file_url = href.split("download_file.php?files=")[-1].strip()
                if raw_file_url.endswith(".pdf"):
                    pdf_urls.add(raw_file_url)
            # 2. Badge button locator
            elif href.endswith(".pdf") and any("badge-info" in cls for cls in a.get("class", [])):
                full_url = urljoin(self.BASE_URL, href)
                pdf_urls.add(full_url)
            # 3. Direct PDF download link
            elif href.endswith(".pdf"):
                full_url = urljoin(self.BASE_URL, href)
                pdf_urls.add(full_url)

        # Filter by prefix if provided
        filtered = []
        for url in pdf_urls:
            filename = url.split("/")[-1]
            if expected_prefix:
                if filename.lower().startswith(expected_prefix.lower()):
                    filtered.append(url)
                else:
                    logger.debug(f"Skipping link not matching prefix '{expected_prefix}': {filename}")
            else:
                filtered.append(url)

        return sorted(filtered)
