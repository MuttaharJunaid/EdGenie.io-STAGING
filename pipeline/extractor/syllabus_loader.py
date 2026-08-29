import re
import functools
import pandas as pd
import logging
from pathlib import Path
from pipeline.extractor.schemas import SyllabusUnit

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
            # 1. Direct path exists
            if p.exists() and p.is_file():
                return p.resolve()

            # 2. Relative to project root
            root_p = project_root / p
            if root_p.exists() and root_p.is_file():
                return root_p.resolve()

            # 3. Just the filename inside syllabi/
            name_p = syllabi_dir / p.name
            if name_p.exists() and name_p.is_file():
                return name_p.resolve()

            # 4. Search by stem in syllabi/
            matches = list(syllabi_dir.glob(f"*{p.stem}*.csv"))
            if matches:
                return matches[0].resolve()

        # 5. Fallback: auto-discover by subject_code in syllabi/
        if subject_code:
            code_matches = list(syllabi_dir.glob(f"*{subject_code}*.csv"))
            if code_matches:
                return code_matches[0].resolve()

        return None

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def load_syllabus(csv_path: Path) -> list[SyllabusUnit]:
        """
        Loads a syllabus CSV file.
        Returns a list of SyllabusUnit objects.
        """
        logger.info(f"Loading syllabus CSV from: {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Syllabus file not found at {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Normalize headers to lowercase and strip whitespaces
        df.columns = [col.lower().strip() for col in df.columns]

        # Check column layouts
        is_3col = all(col in df.columns for col in ["unit", "topic", "subtopic"])
        is_5col = all(col in df.columns for col in ["unit_number", "unit_name", "topic_number", "topic_name", "sub_topic"])

        if not is_3col and not is_5col:
            raise ValueError(
                f"Syllabus CSV must contain either ['unit', 'topic', 'subtopic'] "
                f"or ['unit_number', 'unit_name', 'topic_number', 'topic_name', 'sub_topic']. "
                f"Got: {list(df.columns)}"
            )

        units = []
        for _, row in df.iterrows():
            if is_5col:
                unit_number = str(row["unit_number"]).strip()
                unit_name = str(row["unit_name"]).strip()
                topic_number = str(row["topic_number"]).strip()
                topic_name = str(row["topic_name"]).strip()
                sub_topic = str(row["sub_topic"]).strip()
            else:
                # Parse 3-column layout (e.g. unit="UNIT 1: Cells", topic="1.1 Cell structure...")
                unit_raw = str(row["unit"]).strip()
                topic_raw = str(row["topic"]).strip()
                sub_topic = str(row["subtopic"]).strip()

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
                sub_topic=sub_topic
            )
            units.append(unit)

        logger.info(f"Successfully loaded {len(units)} syllabus rules.")
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
