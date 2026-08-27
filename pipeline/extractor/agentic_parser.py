import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import fitz  # PyMuPDF for page image rendering if needed
import base64

from google import genai
from google.genai import types
from openai import OpenAI

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperCategory
from pipeline.extractor.schemas import (
    MCQQuestionsContainer, MCQAnswersContainer,
    EssayQuestionsContainer, EssayAnswersContainer
)

logger = logging.getLogger(__name__)

class AgenticPDFParser:
    """Agentic PDF parser that sends raw PDF bytes directly to multimodal LLMs for full parsing and classification."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.model_name = model_name or settings.llm_model

        # Default provider check based on keys
        if self.provider == "google" and not os.getenv("GOOGLE_API_KEY") and os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"

        logger.info(f"Initialized AgenticPDFParser with provider={self.provider}")

    def _get_schema_and_prompt(self, category: PaperCategory, syllabus_text: str, meta_context: Dict[str, Any]) -> tuple[Any, str]:
        """Returns the appropriate response schema container and system prompt based on paper category."""
        
        session = meta_context.get("session", "jun")
        year = meta_context.get("year", 2023)
        variant = meta_context.get("variant", "12")
        subject_code = meta_context.get("subject_code", "7707")

        q_format_ctx = f"{session.capitalize()}/{variant}/{year}/Q"

        base_prompt = f"""You are a senior educational examiner for O/A-Levels and a master PDF document parser.
Your task is to analyze the attached exam past paper document and extract all questions/answers matching the correct schema format.

METADATA CONTEXT:
- Year: {year}
- Session: {session}
- Variant: {variant}
- Subject Code: {subject_code}
- Question Number Prefix Format: {q_format_ctx} (e.g. Q1 becomes "{q_format_ctx}1")

SYLLABUS CONTEXT:
{syllabus_text}

LAYOUT AND TEXT INDENTATION CONSTRAINTS:
1. You MUST preserve the exact indentation, line breaks, columns, ledger/journal account templates, tables, and spacing shown in the original paper.
2. For financial statements, double-entry ledgers, and tabular templates, use plain-text space alignment or markdown tables. Do NOT flatten multiple lines or combine aligned credit/debit columns.
3. Every line break and spacing detail in the ledger templates is critical for practice. Preserve them exactly.
4. You must also make sure /n or any new line is not shown in the final json that will be sent as output.
"""

        if category == PaperCategory.MCQ_QP:
            schema = MCQQuestionsContainer
            prompt = base_prompt + f"""
DIRECTIONS FOR MCQ QUESTION PAPERS:
1. Extract every multiple choice question (usually 1 to 30 or 40).
2. CRITICAL CONSTRAINT FOR SEPARATION: Every single question number (from 1 to 30 or 40) MUST have its own independent object in the `questions` array.
3. Under no circumstances should you combine multiple questions into a single question object or write all questions under Q1.
   - For example, Question 1, Question 2, and Question 3 must be separate items at index 0, 1, and 2 in the output array.
   - Never combine Question 2's text or options into Question 1's options.
4. For each question, extract:
   - QuestionNumber: E.g., "{q_format_ctx}1", "{q_format_ctx}2".
   - Question_text: The complete text of the question.
   - Options: An object with fixed keys A, B, C, D each mapping to the option text string.
     e.g. {{"A": "text of option A", "B": "text of option B", "C": "text of option C", "D": "text of option D"}}
   - Marks: 1.
5. Perform classification against the SYLLABUS:
   - Map to correct Unit, Topic number, Topic name matching the syllabus exactly.
   - List tested Sub_topic(s).
6. Perform Difficulty Analysis:
   - Difficulty_score: 1 to 10.
   - Difficulty_label: 'easy' (1-3), 'medium' (4-6), 'hard' (7-8), 'expert' (9-10).
   - Command_words: Command words found (e.g. ['identify', 'calculate']).
"""
        elif category == PaperCategory.MCQ_MS:
            schema = MCQAnswersContainer
            prompt = base_prompt + f"""
DIRECTIONS FOR MCQ MARK SCHEMES:
1. Extract all question numbers and their corresponding answer keys (e.g., 1 A, 2 B, 3 D, 4 C).
2. CRITICAL CONSTRAINT FOR SEPARATION: Every single answer number MUST have its own independent object in the `answers` array.
3. For each:
   - AnswerNumber: E.g., "{q_format_ctx}1", "{q_format_ctx}2".
   - Answer: The single letter key (e.g., "A", "B", "C", "D").
"""
        elif category == PaperCategory.ESSAY_QP:
            schema = EssayQuestionsContainer
            prompt = base_prompt + f"""
DIRECTIONS FOR ESSAY QUESTION PAPERS:
1. Extract all main essay/theory questions (usually 1 to 4, 5, or 6).
2. For each main question, extract:
   - QuestionNumber: E.g., "{q_format_ctx}1", "{q_format_ctx}2".
   - Question_text: The main text introducing the scenario or ledger figures, preserving spacing and table layouts.
   - Parts: List of sub-parts (e.g. (a), (b), (c)).
     - Each part must have part_label (e.g. "(a)", "(b)"), part_text (the actual question prompt of that part, preserving spacing), and marks allocated.
   - Marks: Sum of marks of all parts.
3. Perform classification against the SYLLABUS for the main question:
   - Map to correct Unit, Topic, Topic_name, and Sub_topics.
4. Perform Difficulty Analysis:
   - Estimate Difficulty_score (1-10) and Difficulty_label.
   - Command_words: list of words found.
"""
        elif category == PaperCategory.ESSAY_MS:
            schema = EssayAnswersContainer
            prompt = base_prompt + f"""
DIRECTIONS FOR ESSAY MARK SCHEMES:
1. Extract the detailed answers and marking guidelines for each question.
2. For each:
   - AnswerNumber: E.g., "{q_format_ctx}1", "{q_format_ctx}2".
   - Answer_text: General answer guidelines or main solutions, preserving all original ledger/table indentation.
   - Parts: List of part answers matching the QP parts.
     - Each part has part_label (e.g. "(a)") and part_answer (the suggested answer text, calculations, or ledger balances, preserving all indentation).
"""
        else:
            raise ValueError(f"Unsupported paper category: {category}")

        return schema, prompt

    def parse_pdf(self, pdf_path: Path, category: PaperCategory, syllabus_text: str, meta_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sends PDF to LLM and returns structured records conforming to the target schema."""
        schema, prompt = self._get_schema_and_prompt(category, syllabus_text, meta_context)

        if self.provider == "google":
            try:
                return self._parse_with_gemini(pdf_path, schema, prompt)
            except Exception as gemini_err:
                logger.warning(
                    f"Gemini text-extraction failed for {pdf_path.name} ({gemini_err}). "
                    "Attempting OpenRouter fallback..."
                )
                openrouter_key = os.getenv("OPENROUTER_API_KEY")
                if openrouter_key:
                    return self._parse_with_openrouter(pdf_path, schema, prompt)
                else:
                    logger.error("OPENROUTER_API_KEY environment variable not set; cannot fallback to OpenRouter.")
                    raise gemini_err
        elif self.provider == "openrouter":
            return self._parse_with_openrouter(pdf_path, schema, prompt)
        elif self.provider == "openai":
            return self._parse_with_openai(pdf_path, schema, prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_with_openrouter(self, pdf_path: Path, schema: Any, prompt: str) -> List[Dict[str, Any]]:
        """Sends exact same prompt and payload to OpenRouter as fallback when Gemini fails."""
        logger.info(f"Parsing PDF via OpenRouter fallback: {pdf_path.name}")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        model = os.getenv("OPENROUTER_MODEL", getattr(settings, "openrouter_model", "google/gemini-2.5-flash"))
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        doc = fitz.open(pdf_path)
        contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}"
                }
            })

        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": contents}],
                response_format=schema,
                temperature=0.1
            )
            parsed_response = completion.choices[0].message.parsed
            if hasattr(parsed_response, "questions"):
                return [q.model_dump() for q in parsed_response.questions]
            elif hasattr(parsed_response, "answers"):
                return [a.model_dump() for a in parsed_response.answers]
            return parsed_response.model_dump()
        except Exception as e:
            logger.warning(f"OpenRouter structured parse attempt failed ({e}). Retrying with json_object response_format...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": contents}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_text = response.choices[0].message.content
            if not raw_text or not raw_text.strip():
                raise ValueError("OpenRouter returned an empty response.")
            data = json.loads(raw_text)
            if "questions" in data:
                return data["questions"]
            elif "answers" in data:
                return data["answers"]
            return data

    def _parse_with_gemini(self, pdf_path: Path, schema: Any, prompt: str) -> List[Dict[str, Any]]:
        """Uses official Google GenAI SDK to send raw PDF bytes to Gemini, with optional context caching."""
        logger.info(f"Uploading and parsing PDF with Gemini: {pdf_path.name}")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        client = genai.Client(api_key=api_key)
        model = self.model_name or "gemini-2.5-flash"

        # Load PDF bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        try:
            config_args = {
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.1,
                # MCQ papers with 30-40 questions + 4 options each + classification
                # easily exceed Gemini's default output token limit, causing
                # truncated JSON ("Unterminated string" errors). Set a generous cap.
                "max_output_tokens": 65535,
            }

            contents_arg = [
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                prompt
            ]

            # Context caching: only worthwhile for very large prompts (>32k tokens ~= 128k chars)
            try:
                if len(prompt) > 100000:
                    logger.info("Prompt size sufficient. Attempting to create Gemini Context Cache...")
                    cache = client.caches.create(
                        model=model,
                        config=types.CreateCachedContentConfig(
                            contents=[prompt],
                            ttl="600s"  # 10 minutes cache TTL
                        )
                    )
                    config_args["cached_content"] = cache.name
                    contents_arg = [
                        types.Part.from_bytes(
                            data=pdf_bytes,
                            mime_type="application/pdf"
                        )
                    ]
                    logger.info(f"Gemini context cache '{cache.name}' registered successfully.")
            except Exception as cache_err:
                logger.debug(f"Gemini prompt caching skipped (falling back to standard call): {cache_err}")

            response = client.models.generate_content(
                model=model,
                contents=contents_arg,
                config=types.GenerateContentConfig(**config_args)
            )

            # Log finish reason to help diagnose future truncations
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                logger.info(f"Gemini response finish_reason: {finish_reason}")
                if str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                    logger.warning(
                        f"Gemini finished with non-STOP reason '{finish_reason}'. "
                        "The response may be incomplete. Consider reducing paper size or using a higher token model."
                    )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise ValueError("Gemini returned an empty response. The PDF may be too large or the output limit was hit.")

            data = json.loads(raw_text)
            if "questions" in data:
                return data["questions"]
            elif "answers" in data:
                return data["answers"]
            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"Gemini returned malformed/truncated JSON: {e}. "
                "This usually means the output was cut off before the JSON was complete. "
                "Check finish_reason in logs above."
            )
            raise
        except Exception as e:
            logger.error(f"Gemini API parse failed: {e}")
            raise

    def _parse_with_openai(self, pdf_path: Path, schema: Any, prompt: str) -> List[Dict[str, Any]]:
        """Renders PDF pages as high-resolution images and sends them to OpenAI GPT-4o."""
        logger.info(f"Rendering PDF pages to images and parsing with OpenAI: {pdf_path.name}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key)
        model = self.model_name or "gpt-4o"

        # Render PDF pages to images
        doc = fitz.open(pdf_path)
        contents = [{"type": "text", "text": prompt}]

        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}"
                }
            })

        try:
            # Call openai structured completions
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "user", "content": contents}
                ],
                response_format=schema,
                temperature=0.1
            )

            parsed_response = completion.choices[0].message.parsed
            # Convert to dict representation
            if hasattr(parsed_response, "questions"):
                return [q.model_dump() for q in parsed_response.questions]
            elif hasattr(parsed_response, "answers"):
                return [a.model_dump() for a in parsed_response.answers]
            return parsed_response.model_dump()
        except Exception as e:
            logger.error(f"OpenAI API parse failed: {e}")
            raise e
