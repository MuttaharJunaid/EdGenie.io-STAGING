"""
pipeline/unified/syllabus_loader.py

Loads and parses the Cambridge CSV syllabi for topic classification.
"""

from __future__ import annotations

import csv
import functools
import logging
import re
from pathlib import Path

from pipeline.unified.schemas import SyllabusUnit

logger = logging.getLogger(__name__)


class SyllabusLoader:
    """Loads and parses the admin-provided CSV syllabus for topic classification."""

    @staticmethod
    def resolve_syllabus_path(syllabus_path: str | Path | None, subject_code: str = "") -> Path | None:
        """
        Resolves a syllabus path across common directory layouts and filenames:
        - Direct path as provided (e.g. 'syllabi/PHYSICS_5054.csv')
        - Project root relative path
        - 'syllabi/' directory (e.g. 'pipeline/syllabi/PHYSICS_5054.csv' -> 'syllabi/PHYSICS_5054.csv')
        - Subject code lookup in 'syllabi/' (e.g. subject_code='5054' -> 'syllabi/PHYSICS_5054.csv')
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        syllabi_dir = project_root / "syllabi"

        if syllabus_path:
            p = Path(syllabus_path)
            if p.exists() and p.is_file():
                return p.resolve()

            root_p = project_root / p
            if root_p.exists() and root_p.is_file():
                return root_p.resolve()

            name_p = syllabi_dir / p.name
            if name_p.exists() and name_p.is_file():
                return name_p.resolve()

            matches = list(syllabi_dir.glob(f"*{p.stem}*.csv"))
            if matches:
                return matches[0].resolve()

        if subject_code:
            code_matches = list(syllabi_dir.glob(f"*{subject_code}*.csv"))
            if code_matches:
                return code_matches[0].resolve()

        return None

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def load_syllabus(csv_path: Path) -> list[SyllabusUnit]:
        """Loads a syllabus CSV file using Python's standard csv library. Returns a list of SyllabusUnit objects."""
        logger.info("Loading syllabus CSV from: %s", csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Syllabus file not found at {csv_path}")

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            if not header_row:
                return []

            columns = [col.lower().strip() for col in header_row]

            is_3col = all(col in columns for col in ["unit", "topic", "subtopic"])
            is_5col = all(
                col in columns
                for col in ["unit_number", "unit_name", "topic_number", "topic_name", "sub_topic"]
            )

            if not is_3col and not is_5col:
                raise ValueError(
                    f"Syllabus CSV must contain either ['unit', 'topic', 'subtopic'] "
                    f"or ['unit_number', 'unit_name', 'topic_number', 'topic_name', 'sub_topic']. "
                    f"Got: {columns}"
                )

            col_map = {name: idx for idx, name in enumerate(columns)}
            units: list[SyllabusUnit] = []

            for row in reader:
                if not row or not any(field.strip() for field in row):
                    continue

                def _get_val(col_name: str) -> str:
                    idx = col_map.get(col_name)
                    return str(row[idx]).strip() if idx is not None and idx < len(row) else ""

                if is_5col:
                    unit_number = _get_val("unit_number")
                    unit_name = _get_val("unit_name")
                    topic_number = _get_val("topic_number")
                    topic_name = _get_val("topic_name")
                    sub_topic = _get_val("sub_topic")
                else:
                    unit_raw = _get_val("unit")
                    topic_raw = _get_val("topic")
                    sub_topic = _get_val("subtopic")

                    unit_match = re.match(r"(?i)^unit\s+(\d+(?:\.\d+)*)\s*:\s*(.*)$", unit_raw)
                    if unit_match:
                        unit_number = unit_match.group(1).strip()
                        unit_name = unit_match.group(2).strip()
                    else:
                        unit_number = unit_raw
                        unit_name = ""

                    topic_match = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", topic_raw)
                    if topic_match:
                        topic_number = topic_match.group(1).strip()
                        topic_name = topic_match.group(2).strip()
                    else:
                        topic_number = topic_raw
                        topic_name = ""

                unit = SyllabusUnit(
                    unit_number=unit_number,
                    unit_name=unit_name,
                    topic_number=topic_number,
                    topic_name=topic_name,
                    sub_topic=sub_topic,
                )
                units.append(unit)

        logger.info("Successfully loaded %d syllabus rules.", len(units))
        return units

    @staticmethod
    def format_syllabus_for_prompt(units: list[SyllabusUnit]) -> str:
        """Formats the syllabus list into a clean text block suitable for LLM prompts."""
        lines = []
        for u in units:
            lines.append(
                f"Unit {u.unit_number} ({u.unit_name}) -> Topic {u.topic_number} ({u.topic_name}) | Sub-topics: {u.sub_topic}"
            )
        return "\n".join(lines)
