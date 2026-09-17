"""
pipeline/extractor/classifier_agent.py

Duplicate Stage 2 — Classification-only Gemini pass with OpenRouter fallback.

Takes the pre-extracted JSON records (schema skeletons produced by PyMuPDF text extraction)
along with the official Cambridge syllabus, and calls the LLM to fill in ONLY:
  - Unit
  - Topic
  - Topic_name
  - Sub_topic
  - Difficulty_score
  - Difficulty_label
  - Command_words

No raw PDF bytes are sent, minimizing LLM token consumption.
"""

import json
import logging
import os
import re
import time
from typing import Any

import dotenv

dotenv.load_dotenv()

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperCategory


class QuotaExceededError(Exception):
    """Raised when LLM API resource/quota limits are exhausted (HTTP 429 / RESOURCE_EXHAUSTED)."""
    pass

logger = logging.getLogger(__name__)


# ── Classification-only output schema ─────────────────────────────────────────

class SingleClassification(BaseModel):
    """Classification fields for one question, keyed by its QuestionNumber."""
    question_number: str
    Unit: str
    Topic: str
    Topic_name: str
    Sub_topic: list[str]
    Difficulty_score: int
    Difficulty_label: str
    Command_words: list[str]


class ClassificationContainer(BaseModel):
    classifications: list[SingleClassification]


# ── Agent ──────────────────────────────────────────────────────────────────────

class ClassifierAgent:
    """
    Takes pre-extracted question JSON records from text extraction and the syllabus,
    then uses OpenRouter (primary 1st choice) with Google Gemini (fallback) to fill in
    topic classification, difficulty level, and command words for every question.
    """

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        self.provider = provider or getattr(settings, "classification_provider", "openrouter")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = (
            model_name
            or getattr(settings, "openrouter_model", "google/gemini-2.5-flash")
        )

    def _build_prompt(
        self,
        records: list[dict[str, Any]],
        syllabus_text: str,
        category: PaperCategory,
        meta_context: dict[str, Any],
    ) -> str:
        session = str(meta_context.get("session", "jun")).capitalize()
        year = meta_context.get("year", 2023)
        variant = str(meta_context.get("variant", "11"))
        subject_code = str(meta_context.get("subject_code", "unknown"))

        is_mcq = category == PaperCategory.MCQ_QP
        paper_type = "Multiple-Choice (MCQ)" if is_mcq else "Essay / Structured Theory"
        records_json = json.dumps(records, indent=2, ensure_ascii=False)

        prompt = f"""You are a senior educational examiner for O/A-Levels and a curriculum classification specialist.

YOUR ROLE & WORKFLOW:
The text extraction has already been completed and saved to JSON format. Your task is to analyze each question's text and content against the official syllabus, and produce the topic classification, difficulty level, and command words for every question.

OFFICIAL SYLLABUS:
{syllabus_text}

CLASSIFICATION INSTRUCTIONS:
1. QUESTION MAPPING:
   For every question in the input JSON, locate its `QuestionNumber` (e.g. "{session}/{variant}/{year}/Q1" or "Q1") and use it as `question_number`.

2. TOPIC CLASSIFICATION (using the provided syllabus):
   - `Unit`: The Unit number from the syllabus (e.g. "1", "2").
   - `Topic`: The Topic number from the syllabus (e.g. "1.1", "2.3").
   - `Topic_name`: The exact Topic name from the syllabus (e.g. "Physical Quantities and Units").
   - `Sub_topic`: A list of specific sub-topic learning objectives tested by this question (e.g. ["Scalars and vectors", "Speed and velocity"]).

3. DIFFICULTY ANALYSIS:
   - `Difficulty_score`: Integer from 1 to 10 based on cognitive depth, multi-step calculations, and concept complexity.
   - `Difficulty_label`: One of 'easy' (1-3), 'medium' (4-6), 'hard' (7-8), 'expert' (9-10).
   - `Command_words`: List of command words identified in the question text (e.g. ["calculate", "explain", "state", "describe", "determine"]).

4. OUTPUT CONSTRAINTS:
   - Output exactly ONE classification entry per question in the `classifications` array.
   - Do NOT omit any questions.
   - Output ONLY the requested classification fields (do not modify or re-output question text or options).

---
DYNAMIC PAPER DATA & QUESTIONS TO CLASSIFY:
- Paper Type: {paper_type}
- Subject Code: {subject_code}
- Year: {year}
- Session: {session}
- Variant: {variant}

PRE-EXTRACTED QUESTION JSON:
{records_json}
"""
        return prompt

    def classify(
        self,
        records: list[dict[str, Any]],
        syllabus_text: str,
        category: PaperCategory,
        meta_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Send extracted records + syllabus to OpenRouter (1st choice, with Gemini fallback) for classification.
        Returns a dict: {question_number: {Unit, Topic, Topic_name, Sub_topic,
                                            Difficulty_score, Difficulty_label, Command_words}}
        """
        # MS papers have no classification fields — skip
        if category in (PaperCategory.MCQ_MS, PaperCategory.ESSAY_MS):
            return {}

        prompt = self._build_prompt(records, syllabus_text, category, meta_context)

        # ── 1st Choice: OpenRouter ──────────────────────────────────────────
        if (self.provider == "openrouter" or self.openrouter_api_key) and self.openrouter_api_key:
            try:
                logger.info("Running primary classification pass via OpenRouter (1st choice)...")
                return self._classify_with_openrouter(prompt)
            except Exception as or_err:
                logger.warning(
                    f"OpenRouter classification pass failed ({or_err}). "
                    "Attempting fallback to Google Gemini..."
                )
                if self.google_api_key:
                    try:
                        return self._classify_with_gemini(prompt)
                    except Exception as gemini_err:
                        raise QuotaExceededError(
                            f"Both OpenRouter and Gemini classification failed: {or_err} | {gemini_err}"
                        ) from gemini_err
                else:
                    raise or_err

        # ── Fallback / Direct: Google Gemini ────────────────────────────────
        elif self.google_api_key:
            logger.info("Running classification pass via Google Gemini...")
            return self._classify_with_gemini(prompt)
        else:
            raise RuntimeError("No valid API key (OPENROUTER_API_KEY or GOOGLE_API_KEY) found for classification pass.")

    def _classify_with_gemini(self, prompt: str) -> dict[str, dict[str, Any]]:
        """Calls Gemini with structured output for question classification with 429 exponential backoff."""
        client = genai.Client(api_key=self.google_api_key)
        model = os.getenv("GEMINI_MODEL") or getattr(settings, "gemini_classification_model", "gemini-2.5-flash")
        logger.info(f"Classifying questions via Gemini model: {model}")

        max_retries = getattr(settings, "gemini_max_retries", 3)
        base_delay = getattr(settings, "gemini_retry_delay", 15.0)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClassificationContainer,
                        temperature=0.1,
                        max_output_tokens=65535,
                    ),
                )

                raw = response.text
                if not raw or not raw.strip():
                    raise ValueError("Gemini returned an empty classification response.")

                data = json.loads(raw)
                classifications = data.get("classifications", [])
                return {c["question_number"]: c for c in classifications}
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_429 = "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str
                if is_429:
                    if attempt < max_retries:
                        backoff = base_delay * (2 ** (attempt - 1))
                        m = re.search(r"retry(?:\s+after|\s+in)?\s*[:\s]?\s*(\d+(?:\.\d+)?)\s*s", err_str)
                        if m:
                            backoff = max(float(m.group(1)) + 1.0, backoff)
                        logger.warning(
                            f"Gemini rate limit / 429 hit during classification (attempt {attempt}/{max_retries}). "
                            f"Backing off for {backoff:.1f}s before retrying..."
                        )
                        time.sleep(backoff)
                        client = genai.Client(api_key=self.google_api_key)
                        continue
                    else:
                        logger.error(f"Gemini classification 429 persisted after {max_retries} attempts.")
                        raise e
                raise e

        if last_err is not None:
            raise last_err
        return {}

    def _classify_with_openrouter(self, prompt: str) -> dict[str, dict[str, Any]]:
        """Calls OpenRouter (1st choice) with exact same prompt and schema."""
        logger.info("Calling OpenRouter for question classification...")
        api_key = self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        model = os.getenv("OPENROUTER_MODEL", getattr(settings, "openrouter_model", "google/gemini-2.5-flash"))
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ClassificationContainer,
                temperature=0.1,
                max_tokens=8192,
            )
            parsed_response = completion.choices[0].message.parsed
            classifications = parsed_response.classifications if hasattr(parsed_response, "classifications") else []
            return {c.question_number: c.model_dump() for c in classifications}

        except Exception as e:
            logger.warning(f"OpenRouter structured parse failed ({e}). Retrying with json_object response_format...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=8192,
            )
            raw_text = response.choices[0].message.content
            if not raw_text or not raw_text.strip():
                raise ValueError("OpenRouter returned an empty classification response.")
            data = json.loads(raw_text)
            classifications = data.get("classifications", [])
            return {c["question_number"]: c for c in classifications}

    def merge(
        self,
        records: list[dict[str, Any]],
        classifications: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
