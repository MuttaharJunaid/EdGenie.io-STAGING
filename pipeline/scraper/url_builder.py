from urllib.parse import quote

from pipeline.scraper.models import ScrapeTarget
from pipeline.scraper.subject_registry import SubjectRegistry


class URLBuilder:
    """Helper class to construct PapaCambridge URLs."""

    BASE_URL = "https://pastpapers.papacambridge.com"
    BASE_PAPERS_PATH = "papers/caie"
    BASE_UPLOAD_PATH = "directories/CAIE/CAIE-pastpapers/upload"

    @staticmethod
    def sanitize_segment(segment: str) -> str:
        """Sanitizes a URL segment by converting it to lowercase, stripping, and replacing spaces with dashes."""
        return segment.lower().strip().replace(" ", "-")

    @classmethod
    def build_page_url(cls, target: ScrapeTarget) -> str:
        """
        Builds the default Level 3 (year + session) PapaCambridge page URL using SubjectRegistry.
        """
        level = cls.sanitize_segment(target.level)
        subject_slug = SubjectRegistry.resolve_subject_slug(
            level=target.level,
            subject_name=target.subject_name,
            code=target.subject_code,
        )
        year = str(target.year)
        session_slug = SubjectRegistry.format_session_slug(target.session, target.year)

        return f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{subject_slug}-{year}-{session_slug}"

    @classmethod
    def get_candidate_urls(cls, target: ScrapeTarget) -> list[str]:
        """
        Generates list of potential candidate URLs for the given target.
        Resolves exact subject slug and year-aware session slug (may-june/oct-nov for 2018+).
        """
        level = cls.sanitize_segment(target.level)
        subject_name = cls.sanitize_segment(target.subject_name)
        subject_code = cls.sanitize_segment(target.subject_code)
        year = str(target.year)

        resolved_slug = SubjectRegistry.resolve_subject_slug(
            level=target.level,
            subject_name=target.subject_name,
            code=target.subject_code,
        )
        session_slug = SubjectRegistry.format_session_slug(target.session, target.year)


        urls = [
            f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{year}-{session_slug}"
        ]

        # 1. Alternative session fallback (e.g. legacy jun/nov if may-june/oct-nov fails)
        alt_session = "jun" if session_slug == "may-june" else ("nov" if session_slug == "oct-nov" else ("may-june" if session_slug == "jun" else ("oct-nov" if session_slug == "nov" else session_slug)))
        if alt_session != session_slug:
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{year}-{alt_session}")

        # 2. Session before year (PapaCambridge inverted pattern, e.g. may-june-2020)
        urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{session_slug}-{year}")
        if alt_session != session_slug:
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{alt_session}-{year}")

        # 3. Repeated slug without hyphen before year (PapaCambridge repeated-slug bug, e.g. 2024 Nov)
        urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{resolved_slug}{year}-{session_slug}")
        urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{resolved_slug}-{year}-{session_slug}")
        if alt_session != session_slug:
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{resolved_slug}{year}-{alt_session}")
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{resolved_slug}-{year}-{alt_session}")

        # 4. Special alias handling (e.g. 4024 mathematics-d-4024)
        if subject_code == "4024":
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-mathematics-d-4024{year}-{session_slug}")
            if alt_session != session_slug:
                urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-mathematics-d-4024{year}-{alt_session}")

        # 5. November session duplicate year-session anomaly (e.g. 2025-oct-nov2025-oct-nov)
        is_nov = target.session.lower().strip() in ["nov", "n", "w", "november", "oct-nov", "oct-november", "oct"] or session_slug in ["oct-nov", "nov"]
        if is_nov:
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{year}-oct-nov{year}-oct-nov")
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{year}-{session_slug}{year}-{session_slug}")
            if alt_session != session_slug:
                urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{level}-{resolved_slug}-{year}-{alt_session}{year}-{alt_session}")

        # 6. Explicit input fallback if different
        raw_session = cls.sanitize_segment(target.session)
        fallback_part = f"{level}-{subject_name}-{subject_code}-{year}-{raw_session}"
        if fallback_part not in urls[0]:
            urls.append(f"{cls.BASE_URL}/{cls.BASE_PAPERS_PATH}/{fallback_part}")

        # Deduplicate while preserving order
        seen = set()
        unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

        return unique_urls

    @classmethod
    def build_download_url(cls, filename: str) -> str:
        """
        Builds the download URL that points directly to download_file.php.
        Example:
          https://pastpapers.papacambridge.com/download_file.php?files=https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/{filename}
        """
        raw_pdf_url = f"{cls.BASE_URL}/{cls.BASE_UPLOAD_PATH}/{filename}"
        return f"{cls.BASE_URL}/download_file.php?files={quote(raw_pdf_url, safe='')}"

