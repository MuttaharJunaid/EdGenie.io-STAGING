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
from typing import Any, Dict, List, Optional

import dotenv
dotenv.load_dotenv()

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel

from pipeline.config import settings
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
    Takes pre-extracted question JSON records from text extraction and the syllabus,
    then uses Gemini (with OpenRouter fallback) to fill in topic classification,
    difficulty level, and command words for every question.
    """

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or getattr(settings, "llm_provider", "google")
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or getattr(settings, "llm_model", None)
            or "gemini-2.5-flash"
        )
        if not self.model_name:
            self.model_name = "gemini-2.5-flash"

        self.api_key = os.getenv("GOOGLE_API_KEY")

        if self.provider == "google" and not self.api_key and os.getenv("OPENROUTER_API_KEY"):
            self.provider = "openrouter"

    def _build_prompt(
        self,
        records: List[Dict[str, Any]],
        syllabus_text: str,
        category: PaperCategory,
        meta_context: Dict[str, Any],
    ) -> str:
        session = str(meta_context.get("session", "jun")).capitalize()
        year = meta_context.get("year", 2023)
        variant = str(meta_context.get("variant", "11"))
        subject_code = str(meta_context.get("subject_code", "unknown"))

        is_mcq = category == PaperCategory.MCQ_QP
        paper_type = "Multiple-Choice (MCQ)" if is_mcq else "Essay / Structured Theory"

        # Provide clean JSON of the pre-extracted questions
        records_json = json.dumps(records, indent=2, ensure_ascii=False)

        prompt = f"""You are a senior educational examiner for O/A-Levels and a curriculum classification specialist.

You are provided with:
1. A JSON file of questions pre-extracted from a Cambridge {paper_type} past paper.
2. The official Cambridge syllabus for this subject.

YOUR ROLE & WORKFLOW:
The text extraction has already been completed and saved to the JSON file. Your work is to take in the pre-extracted JSON file and the syllabus, analyze each question's text and content, and fill in the topic classification, difficulty level, and command words for every question.

METADATA CONTEXT:
- Year: {year}
- Session: {session}
- Variant: {variant}
- Subject Code: {subject_code}

OFFICIAL SYLLABUS:
{syllabus_text}

PRE-EXTRACTED QUESTION JSON:
{records_json}

CLASSIFICATION INSTRUCTIONS:
1. QUESTION MAPPING:
   For every question in the input JSON, locate its `QuestionNumber` (e.g. "{session}/{variant}/{year}/Q1") and use it as `question_number`.

2. TOPIC CLASSIFICATION (using the provided syllabus):
   - `Unit`: The Unit number from the syllabus (e.g. "1", "2").
   - `Topic`: The Topic number from the syllabus (e.g. "1.1", "2.3").
   - `Topic_name`: The exact Topic name from the syllabus (e.g. "Physical Quantities and Units").
   - `Sub_topic`: A list of the specific sub-topic learning objectives tested by this question (e.g. ["Scalars and vectors", "Speed and velocity"]).

3. DIFFICULTY ANALYSIS:
   - `Difficulty_score`: Integer from 1 to 10 based on cognitive depth, multi-step calculations, and concept complexity.
   - `Difficulty_label`: One of 'easy' (1-3), 'medium' (4-6), 'hard' (7-8), 'expert' (9-10).
   - `Command_words`: List of command words identified in the question text (e.g. ["calculate", "explain", "state", "describe", "determine"]).

4. OUTPUT CONSTRAINTS:
   - Output exactly ONE classification entry per question in the `classifications` array.
   - Do NOT omit any questions.
   - Output ONLY the requested classification fields (do not modify or re-output question text or options).
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
        Send extracted records + syllabus to Gemini (with OpenRouter fallback) for classification.
        Returns a dict: {question_number: {Unit, Topic, Topic_name, Sub_topic,
                                            Difficulty_score, Difficulty_label, Command_words}}
        """
        # MS papers have no classification fields — skip
        if category in (PaperCategory.MCQ_MS, PaperCategory.ESSAY_MS):
            return {}

        prompt = self._build_prompt(records, syllabus_text, category, meta_context)

        # Primary attempt: Google Gemini
        if self.provider == "google" and self.api_key:
            try:
                return self._classify_with_gemini(prompt)
            except Exception as gemini_err:
                err_str = str(gemini_err).lower()
                is_quota = any(q in err_str for q in ["resource_exhausted", "429", "quota", "rate limit", "credits"])
                logger.warning(
                    f"Gemini classification pass failed ({gemini_err}). "
                    "Attempting fallback to OpenRouter..."
                )
                if os.getenv("OPENROUTER_API_KEY"):
                    try:
                        return self._classify_with_openrouter(prompt)
                    except Exception as or_err:
                        or_err_str = str(or_err).lower()
                        if is_quota or any(q in or_err_str for q in ["resource_exhausted", "429", "quota", "credits", "402"]):
                            raise QuotaExceededError(f"Gemini & OpenRouter quotas exhausted: {gemini_err} | {or_err}") from or_err
                        raise or_err
                else:
                    if is_quota:
                        raise QuotaExceededError(f"Gemini quota exceeded: {gemini_err}") from gemini_err
                    logger.error("OPENROUTER_API_KEY not set; cannot fallback to OpenRouter.")
                    raise gemini_err

        elif self.provider == "openrouter" or os.getenv("OPENROUTER_API_KEY"):
            try:
                return self._classify_with_openrouter(prompt)
            except Exception as or_err:
                or_err_str = str(or_err).lower()
                if any(q in or_err_str for q in ["resource_exhausted", "429", "quota", "credits", "402"]):
                    raise QuotaExceededError(f"OpenRouter quota exhausted: {or_err}") from or_err
                raise or_err
        else:
            raise RuntimeError("No valid API key (GOOGLE_API_KEY or OPENROUTER_API_KEY) found for classification pass.")

    def _classify_with_gemini(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Calls Gemini with structured output for question classification."""
        client = genai.Client(api_key=self.api_key)
        model = self.model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        logger.info(f"Classifying questions via Gemini model: {model}")

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

    def _classify_with_openrouter(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Fallback to OpenRouter with exact same prompt and schema."""
        logger.info("Running classification pass via OpenRouter fallback...")
        api_key = os.getenv("OPENROUTER_API_KEY")
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
