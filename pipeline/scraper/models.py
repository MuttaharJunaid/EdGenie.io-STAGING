from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.config import settings


class PaperDocumentType(str, Enum):
    QUESTION_PAPER = "qp"
    MARK_SCHEME = "ms"
    EXAMINER_REPORT = "er"
    INSERT = "in"
    GRADE_THRESHOLD = "gt"
    CONFIDENTIAL_INSTRUCTIONS = "ci"
    RECORDING_PAPER = "rp"
    SOUND_FILE = "sf"
    OTHER = "other"

    @classmethod
    def from_filename_part(cls, part: str) -> "PaperDocumentType":
        part = part.lower().strip()
        try:
            return cls(part)
        except ValueError:
            return cls.OTHER

class ScrapeTarget(BaseModel):
    """Defines what to scrape."""
    level: str = "o-level"              # "o-level" | "as-and-a-level"
    subject_name: str = ""              # optional, resolved via code
    subject_code: str                   # "7707", "5054", etc.
    year: int                           # 2023
    session: str                        # "jun" | "nov"

    @property
    def page_url(self) -> str:
        # E.g. https://pastpapers.papacambridge.com/papers/caie/o-level-accounting-7707-2023-jun
        # Special handling: PapaCambridge level could be formatted differently depending on input.
        # But we assume URLBuilder will format properly.
        # Let's keep a standard property here as a fallback or default.
        return (
            f"https://pastpapers.papacambridge.com/papers/caie/"
            f"{self.level}-{self.subject_name}-{self.subject_code}"
            f"-{self.year}-{self.session}"
        )

    @property
    def storage_dir(self) -> Path:
        return settings.storage_dir / self.subject_code / str(self.year) / self.session

class PaperFileInfo(BaseModel):
    filename: str
    url: str
    size_bytes: int | None = None
    sha256: str | None = None
    document_type: PaperDocumentType
    paper_number: int | None = None
    variant: str | None = None
    subject_code: str | None = None
    year: int | None = None
    session: str | None = None

class ScrapeManifest(BaseModel):
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    target: ScrapeTarget
    files: list[PaperFileInfo] = Field(default_factory=list)
