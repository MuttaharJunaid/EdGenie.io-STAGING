"""
pipeline/syllabus/builder.py

Extracts a structured Cambridge syllabus from an official PDF using Gemini
and writes it as a CSV matching the EdGenie syllabi/ format:

    Unit, Topic, Subtopic

The extraction prompt mirrors the exact layout that already exists in
syllabi/BIOLOGY_5090.csv, syllabi/ACCOUNTING_7707.csv etc. so that the
resulting file can be passed directly to --syllabus in edgenie-extractor.
"""

import csv
import json
import logging
import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from pipeline.syllabus.crawler import CambridgeSyllabusCrawler

logger = logging.getLogger(__name__)

# ── Output schema ──────────────────────────────────────────────────────────────

class SyllabusRow(BaseModel):
    """One row of the Cambridge syllabus CSV."""
    unit: str        # e.g. "UNIT 1: Cells"
    topic: str       # e.g. "1.1 Cell structure and function"
    subtopic: str    # e.g. "Examine animal and plant cells"


class SyllabusContainer(BaseModel):
    rows: list[SyllabusRow]


# ── Builder ────────────────────────────────────────────────────────────────────

class SyllabusBuilder:
    """
    Reads an official Cambridge syllabus PDF with Gemini and produces
    a structured CSV ready for use with edgenie-extractor --syllabus.
    """

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file before running edgenie-syllabus."
            )
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.crawler = CambridgeSyllabusCrawler()

    # ── Gemini extraction ──────────────────────────────────────────────────────

    def extract_from_pdf(
        self,
        pdf_path: Path,
        subject: str,
        code: str,
        level: str,
    ) -> list[SyllabusRow]:
        """
        Send the syllabus PDF to Gemini and return structured rows.
        Raises on API errors.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Reading syllabus PDF: {pdf_path.name}")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        prompt = f"""You are an expert at reading official Cambridge International Education syllabus documents.

SUBJECT : {subject}
CODE    : {code}
LEVEL   : {level}

Your task is to extract the COMPLETE subject content / syllabus outline from this PDF and
return it as a structured list of rows, each with three fields:

  unit     — The top-level unit, formatted exactly as "UNIT N: <Unit Name>"
              Examples:
                "UNIT 1: Cells"
                "UNIT 3: Mechanics"
                "UNIT 2: Sources and recording of data"

  topic    — The numbered topic inside that unit, formatted as "N.N <Topic Name>"
              Examples:
                "1.1 Cell structure and function"
                "3.2 Forces"
                "2.1 The double entry system of book-keeping"

  subtopic — A single, concise learning-objective or sub-topic bullet from the syllabus.
              One learning objective = one row. Keep it short (5-10 words max).
              Examples:
                "Examine animal and plant cells"
                "Define resultant force"
                "Process data double entry"

RULES:
1. Extract EVERY learning objective listed in the subject content section of the PDF.
   Do NOT summarise — if there are 200 learning objectives, return 200 rows.
2. Each learning objective gets its own row. Never combine multiple objectives into one.
3. Use the exact unit name and topic number/name as they appear in the PDF.
4. Ignore front-matter (assessment structure, grade descriptors, command words lists,
   resource lists, administrative info). Extract ONLY the content/learning objectives.
5. If a topic has no sub-bullets (it is itself a leaf learning objective), use the
   topic title as the subtopic as well.
6. Do not include page numbers, footnotes, or section headers in the output.
"""

        client = genai.Client(api_key=self.api_key)
        logger.info(f"Sending {pdf_path.name} to Gemini ({self.model}) for syllabus extraction…")

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyllabusContainer,
                temperature=0.1,
                max_output_tokens=65535,
            ),
        )

        raw = response.text
        if not raw or not raw.strip():
            raise ValueError("Gemini returned an empty response for the syllabus PDF.")

        data = json.loads(raw)
        rows = [SyllabusRow(**r) for r in data.get("rows", [])]
        logger.info(f"Extracted {len(rows)} syllabus rows from {pdf_path.name}.")
        return rows

    # ── CSV writer ─────────────────────────────────────────────────────────────

    def save_csv(
        self,
        rows: list[SyllabusRow],
        subject: str,
        code: str,
        output_dir: Path,
    ) -> Path:
        """
        Write extracted rows to a CSV in the EdGenie syllabi format.
        Returns the path of the created file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{subject.strip().upper().replace(' ', '_')}_{code}.csv"
        csv_path = output_dir / filename

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Unit", "Topic", "Subtopic"])
            for row in rows:
                writer.writerow([row.unit, row.topic, row.subtopic])

        logger.info(f"Saved syllabus CSV to: {csv_path}")
        return csv_path

    # ── Full Automated Build ───────────────────────────────────────────────────

    def build(
        self,
        subject: str,
        code: str,
        level: str,
        pdf_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> tuple[Path, Path, list[SyllabusRow]]:
        """
        Runs the full automated syllabus extraction pipeline:
        1. If pdf_path is None, automatically crawls Cambridge website to find
           and download the official syllabus PDF with > 3 pages.
        2. Sends PDF to Gemini to extract all learning objectives.
        3. Saves resulting CSV to syllabi/ directory.
        Returns (csv_path, downloaded_or_provided_pdf_path, rows).
        """
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "syllabi"

        if pdf_path is None:
            logger.info(f"Automatically finding syllabus for {subject} ({code}) on Cambridge website...")
            resolved_pdf, _pages = self.crawler.fetch_syllabus_pdf(subject, code, level)
        else:
            resolved_pdf = Path(pdf_path)

        rows = self.extract_from_pdf(resolved_pdf, subject, code, level)
        csv_path = self.save_csv(rows, subject, code, output_dir)
        return csv_path, resolved_pdf, rows
