"""
pipeline/extractor/classifier_agent.py

Duplicate Stage 2 — Classification-only Gemini pass.

Takes pre-extracted question records (schema skeletons from PyMuPDFExtractor)
and sends ONLY the JSON text + syllabus to Gemini to fill in:
  Unit, Topic, Topic_name, Sub_topic, Difficulty_score,
  Difficulty_label, Command_words

No PDF bytes are sent.  The same classification/difficulty instructions
from agentic_parser.py are reused verbatim.
"""

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel

from google import genai
from google.genai import types

from pipeline.extractor.paper_type_detector import PaperCategory

logger = logging.getLogger(__name__)


# ── Classification-only output schema ─────────────────────────────────────────

class SingleClassification(BaseModel):
    """Classification fields for one question, keyed by its QuestionNumber."""
    question_number: str
    Unit: str
    Topic: str
    Topic_name: str
    Sub_topic: List[str]
    Difficulty_score: int
    Difficulty_label: str
    Command_words: List[str]


class ClassificationContainer(BaseModel):
    classifications: List[SingleClassification]


# ── Agent ──────────────────────────────────────────────────────────────────────

class ClassifierAgent:
    """
    Sends pre-extracted question JSON + syllabus to Gemini for classification
    and difficulty labelling.  Returns a dict keyed by question_number.
    """

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def _build_prompt(
        self,
        records: List[Dict[str, Any]],
        syllabus_text: str,
        category: PaperCategory,
        meta_context: Dict[str, Any],
    ) -> str:
        session = str(meta_context.get("session", "jun")).capitalize()
        year = meta_context.get("year", 2023)
        variant = meta_context.get("variant", "11")
        subject_code = meta_context.get("subject_code", "unknown")

        is_mcq = category == PaperCategory.MCQ_QP
        paper_type = "Multiple-Choice" if is_mcq else "Essay/Structured"

        records_json = json.dumps(records, indent=2, ensure_ascii=False)

        prompt = f"""You are a senior educational examiner for O/A-Levels.

Below are pre-extracted question records from a Cambridge {paper_type} past paper.
Your ONLY task is to classify each question by topic and difficulty — do NOT re-extract or modify the question text, options, or parts.

METADATA CONTEXT:
- Year: {year}
- Session: {session}
- Variant: {variant}
- Subject Code: {subject_code}

SYLLABUS CONTEXT:
{syllabus_text}

PRE-EXTRACTED QUESTION RECORDS (JSON):
{records_json}

CLASSIFICATION INSTRUCTIONS:
1. For each record, look up the question_number (from the QuestionNumber field).
2. Map to the correct Unit number, Topic number, Topic_name, and Sub_topic(s) using the SYLLABUS exactly.
3. Estimate Difficulty_score (1–10) and Difficulty_label: 'easy' (1–3), 'medium' (4–6), 'hard' (7–8), 'expert' (9–10).
4. List Command_words found in the question text (e.g. ['explain', 'calculate', 'describe']).
5. Return ONE classification entry per question using the exact question_number string from the record.
6. Do NOT include any answer options, question text, or mark allocations in your response — only the classification fields.
"""
        return prompt

    def classify(
        self,
        records: List[Dict[str, Any]],
        syllabus_text: str,
        category: PaperCategory,
        meta_context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Send extracted records + syllabus to Gemini for classification.
        Returns a dict: {question_number: {Unit, Topic, Topic_name, Sub_topic,
                                            Difficulty_score, Difficulty_label, Command_words}}
        Raises RuntimeError if no API key or Gemini call fails.
        """
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set; cannot run classification pass.")

        # MS papers have no classification fields — skip
        if category in (PaperCategory.MCQ_MS, PaperCategory.ESSAY_MS):
            return {}

        prompt = self._build_prompt(records, syllabus_text, category, meta_context)
        client = genai.Client(api_key=self.api_key)

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClassificationContainer,
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )

            raw = response.text
            if not raw or not raw.strip():
                raise ValueError("Gemini returned an empty classification response.")

            data = json.loads(raw)
            classifications = data.get("classifications", [])
            return {c["question_number"]: c for c in classifications}

        except Exception as e:
            logger.error(f"ClassifierAgent Gemini call failed: {e}")
            raise

    def merge(
        self,
        records: List[Dict[str, Any]],
        classifications: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge classification results back into the schema skeleton records.
        Missing keys are left as-is (empty/zero).
        """
        for record in records:
            q_key = record.get("QuestionNumber") or record.get("AnswerNumber") or ""
            cl = classifications.get(q_key, {})
            if cl:
                record["Unit"] = cl.get("Unit", record.get("Unit", ""))
                record["Topic"] = cl.get("Topic", record.get("Topic", ""))
                record["Topic_name"] = cl.get("Topic_name", record.get("Topic_name", ""))
                record["Sub_topic"] = cl.get("Sub_topic", record.get("Sub_topic", []))
                record["Difficulty_score"] = cl.get("Difficulty_score", record.get("Difficulty_score", 0))
                record["Difficulty_label"] = cl.get("Difficulty_label", record.get("Difficulty_label", ""))
                record["Command_words"] = cl.get("Command_words", record.get("Command_words", []))
        return records
