import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FILE = Path(__file__).parent / "subject_registry.json"


class SubjectRegistry:
    """
    Manages mapping of CAIE O-Level and AS/A-Level subjects to their exact
    PapaCambridge URL slugs and syllabus codes.
    """

    _data: dict[str, dict[str, Any]] | None = None

    @classmethod
    def _load_data(cls) -> dict[str, dict[str, Any]]:
        if cls._data is not None:
            return cls._data

        if not REGISTRY_FILE.exists():
            logger.warning(f"Subject registry file missing at {REGISTRY_FILE}")
            cls._data = {"o-level": {}, "as-and-a-level": {}}
            return cls._data

        try:
            with open(REGISTRY_FILE, encoding="utf-8") as f:
                cls._data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading subject registry: {e}")
            cls._data = {"o-level": {}, "as-and-a-level": {}}

        return cls._data

    @classmethod
    def normalize_level(cls, level: str) -> str:
        lvl = level.lower().strip()
        if "a-level" in lvl or "as" in lvl:
            return "as-and-a-level"
        return "o-level"

    @classmethod
    def lookup_by_code(cls, code: str) -> dict[str, Any] | None:
        """Lookup subject entry across all levels (o-level, as-and-a-level) by syllabus code."""
        data = cls._load_data()
        clean_code = str(code).strip()
        for lvl in ("o-level", "as-and-a-level"):
            level_dict = data.get(lvl, {})
            if clean_code in level_dict:
                return level_dict[clean_code]
        return None

    @classmethod
    def get_entry(
        cls,
        level: str,
        code: str | None = None,
        subject_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Lookup a subject entry by syllabus code or subject name/keyword.
        """
        data = cls._load_data()
        lvl = cls.normalize_level(level)
        level_dict = data.get(lvl, {})

        if not level_dict:
            return None

        # 1. Direct code lookup (e.g. '4024', '5054', '7707', '9706')
        if code:
            clean_code = str(code).strip()
            if clean_code in level_dict:
                return level_dict[clean_code]

        # 2. Subject name / slug lookup
        if subject_name:
            clean_name = subject_name.lower().strip().replace(" ", "-")

            # Direct key match
            if clean_name in level_dict:
                return level_dict[clean_name]

            # Search by code embedded in subject_name (e.g. 'mathematics-4024' or '4024')
            code_in_name = re.search(r"(\d{4})", clean_name)
            if code_in_name and code_in_name.group(1) in level_dict:
                return level_dict[code_in_name.group(1)]

            # Partial match on subject_slug or title
            keywords = [k for k in re.split(r"[-_\s]+", clean_name) if k and k not in {"level", "o", "a", "as"}]
            if keywords:
                for key, entry in level_dict.items():
                    slug = entry.get("subject_slug", "").lower()
                    title = entry.get("title", "").lower()
                    if all(kw in slug or kw in title for kw in keywords):
                        return entry

                # Secondary fallback: match any keyword if unique
                matches = [
                    entry for key, entry in level_dict.items()
                    if any(kw in entry.get("subject_slug", "").lower() for kw in keywords)
                ]
                if len(matches) == 1:
                    return matches[0]

        return None

    @classmethod
    def resolve_subject_slug(
        cls,
        level: str,
        subject_name: str | None = None,
        code: str | None = None,
    ) -> str:
        """
        Returns the exact subject_slug for PapaCambridge URL building.

        If a match is found in the registry, returns its exact subject_slug
        (e.g., 'mathematics-d-calculator-version-4024').
        Otherwise, constructs a safe sanitized fallback string.
        """
        entry = cls.get_entry(level=level, code=code, subject_name=subject_name)
        if entry and "subject_slug" in entry:
            return entry["subject_slug"]

        # Fallback slug construction
        parts = []
        if subject_name:
            parts.append(subject_name.lower().strip().replace(" ", "-"))
        if code:
            code_str = str(code).strip()
            if not parts or not parts[0].endswith(code_str):
                parts.append(code_str)

        return "-".join(parts) if parts else "subject"

    @classmethod
    def format_session_slug(cls, session: str, year: int) -> str:
        """
        Format session slug based on the year.
        From 2018 onwards PapaCambridge URLs end with:
          - June session -> 'may-june'
          - Nov session  -> 'oct-nov'
          - March session -> 'march' (or 'feb-march')
        Before 2018:
          - June session -> 'jun'
          - Nov session  -> 'nov'
          - March session -> 'm'
        """
        s = session.lower().strip()
        is_june = s in ["jun", "j", "s", "june", "may-june", "may-jun", "may"]
        is_nov = s in ["nov", "n", "w", "november", "oct-nov", "oct-november", "oct"]
        is_march = s in ["m", "march", "feb-march", "february-march"]

        if year >= 2018:
            if is_june:
                return "may-june"
            if is_nov:
                return "oct-nov"
            if is_march:
                return "march"
        else:
            if is_june:
                return "jun"
            if is_nov:
                return "nov"
            if is_march:
                return "m"

        return s

