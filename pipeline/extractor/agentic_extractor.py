"""
pipeline/extractor/agentic_extractor.py

Agentic Text Extractor — page-by-page LLM-based extraction.

Extracts question text, options, hierarchical parts, and mark schemes
page-by-page to guarantee zero output truncation, using Google Gemini
as primary and Qwen 2.5 VL 7B (via OpenRouter) on fallback.
"""

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz
from google import genai
from google.genai import types
from openai import OpenAI

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperCategory
from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
from pipeline.extractor.schemas import (
    EssayAnswersContainer,
    EssayQuestionsContainer,
    MCQAnswersContainer,
    MCQQuestionsContainer,
)

logger = logging.getLogger(__name__)


class AgenticExtractor:
    """Extracts structured questions and mark schemes using chunked page-by-page LLM calls."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        self.model_name = model_name or getattr(settings, "llm_model", "gemini-3.6-flash")
        self.fallback_model = fallback_model or getattr(
            settings, "fallback_model", "qwen/qwen3-vl-8b-instruct"
        )
        self._gemini_client: Optional[genai.Client] = None
        self._openai_client: Optional[OpenAI] = None
        self._gemini_quota_exhausted: bool = False

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            self._openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._openai_client

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fmt_qnum(self, q_raw: Any, meta: Dict[str, Any]) -> str:
        """Ensure QuestionNumber/AnswerNumber is formatted as Session/Variant/Year/QN."""
        session = str(meta.get("session", "jun")).capitalize()
        variant = str(meta.get("variant", "11"))
        year = str(meta.get("year", 2023))

        # Extract digits from q_raw (e.g. "Q7", "7", "May/June/11/2010/Q7")
        q_str = str(q_raw).strip()
        last_seg = q_str.split("/")[-1]
        m = re.search(r"(?:Q|Question\s*)?(\d{1,2})\b", last_seg, re.IGNORECASE)
        if m:
            qnum = int(m.group(1))
        else:
            m_full = re.search(r"(?:Q|Question\s*)(\d{1,2})\b", q_str, re.IGNORECASE)
            qnum = int(m_full.group(1)) if m_full else 1
        return f"{session}/{variant}/{year}/Q{qnum}"

    @staticmethod
    def _normalize_subpart_label(lbl: str) -> str:
        """
        Normalize sub-part labels to strictly roman numerals in parentheses:
        e.g. 'ai' -> '(i)', 'a(i)' -> '(i)', 'aii' -> '(ii)', '(i)' -> '(i)', 'i' -> '(i)', 'b(iii)' -> '(iii)'.
        """
        if not lbl:
            return "(i)"
        s = str(lbl).strip()
        m = re.search(r"(?:[a-z]?\s*\(?|\()([ivxldcm]+|\d+)\)?$", s, re.IGNORECASE)
        if m:
            val = m.group(1).lower()
            return f"({val})"
        m2 = re.search(r"\(([ivxldcm]+|\d+)\)", s, re.IGNORECASE)
        if m2:
            return f"({m2.group(1).lower()})"
        return s

    MATH_NOTATION_INSTRUCTIONS = """
CRITICAL MATHEMATICAL NOTATION & SYMBOL RULES:
1. Exponentials: Format powers and scientific notation with superscripts (e.g., x², x³, 5t², n³, 10²¹, 10⁻⁴, s⁻¹, kg m⁻³).
2. Fractions: Transcribe fractions cleanly using division slash notation (e.g., 1/2, 3/4, (x + 1)/(x - 2)) or standard LaTeX \\frac{a}{b}. Never omit numerators or denominators.
3. Multiplications: Always use the proper mathematical multiplication symbol '×' (U+00D7), NEVER lowercase 'x' or asterisk '*' (e.g., 5.98 × 10²¹, 3 × 4).
4. Fractions as Exponents: Format fractional powers clearly with parentheses or superscripts (e.g., x^(1/2), 8^(2/3), x^{1/3}, a^(m/n)) so the base and exponent are unambiguous.
5. Vector Signs: Transcribe vectors accurately (e.g., \\vec{v}, \\mathbf{a}, \\overrightarrow{AB}, or column vectors as \\begin{pmatrix} x \\\\ y \\end{pmatrix} or (x, y)^T).
6. Congruency Signs: Use the geometric congruency symbol '≅' (U+2245) or '\\cong' (e.g., △ABC ≅ △DEF).
7. Equivalence / Identity Signs: Use the identity/equivalence symbol '≡' (U+2261) or '\\equiv' (e.g., f(x) ≡ 2x + 1).
8. Other Symbols: Preserve inequalities (≤, ≥, <, >, ≠), degrees (°), Greek letters (π, θ, α, β, Δ, Ω), square roots (√x), and set symbols (∈, ∉, ∪, ∩, ∅).
"""

    SANITY_CHECK_INSTRUCTIONS = """
POST-EXTRACTION SANITY CHECK:
Before outputting the final JSON response, perform a mandatory verification check:
1. Displacement Check: Verify that NO question text, part text, equations, tables, or diagrams are displaced, misplaced, or assigned to the wrong question/part.
2. Hierarchy Check: If a question has parts (a), (b), etc., all nested sub-parts under part (a) MUST be labeled strictly as (i), (ii), (iii), NEVER as ai, aii, aiii, or a(i).
3. Blanks & Prompts Check: Ensure answer blanks, dotted lines, and fill-in prompts (e.g., "height = ..........", "Answer: [ ]", "...................") are completely excluded from the extracted text.
4. Mark Scheme Answer_text Check: If solutions exist under individual parts (a, b, etc.) and there is NO overarching question text, ensure "Answer_text" is strictly "" (empty string). DO NOT output gibberish, placeholder characters (e.g. '...', '---', 'N/A', 'None'), or duplicate part text in "Answer_text".
5. Question Paper Topic/Unit Check: In question papers, Unit, Topic, Topic_name, and Sub_topic MUST be strictly null (do NOT classify or guess topics).
6. Marks Guarantee: Confirm that every question, part, and sub-part has marks >= 1 (never 0).
"""

    def _get_page_prompt(
        self,
        category: PaperCategory,
        pno: int,
        total_pages: int,
        meta: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """Returns the target Pydantic container schema and page extraction prompt."""
        session = str(meta.get("session", "jun")).capitalize()
        variant = str(meta.get("variant", "11"))
        year = meta.get("year", 2023)

        if category == PaperCategory.MCQ_QP:
            schema = MCQQuestionsContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine this exact exam page (Page {pno + 1} of {total_pages}) from a Multiple Choice Question (MCQ) paper.

TASK:
Extract all multiple-choice questions appearing on THIS page.
For each question:
- QuestionNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q2".
- Question_text: The complete verbatim question stem text.
- Options: An object with keys "A", "B", "C", "D" containing the text for each choice.
- Marks: 1 for every MCQ question.
- Unit: null
- Topic: null
- Topic_name: null
- Sub_topic: null
(Leave Unit, Topic, Topic_name, and Sub_topic strictly as null. Do NOT classify topics or syllabus.)

CRITICAL RULES:
1. Extract ONLY questions visible on this page. DO NOT invent or duplicate questions from other pages.
2. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", candidate details).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.MCQ_MS:
            schema = MCQAnswersContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine this exact mark scheme page (Page {pno + 1} of {total_pages}) for a Multiple Choice (MCQ) paper.

TASK:
Extract all question numbers and their corresponding correct answer letter keys (A, B, C, or D) on THIS page.
For each answer:
- AnswerNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q2".
- Answer: The single uppercase letter (A, B, C, or D).

CRITICAL RULES:
1. Extract ONLY answers visible on this page.
2. DO NOT include headers or footers.

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.ESSAY_QP:
            schema = EssayQuestionsContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine this exact exam page (Page {pno + 1} of {total_pages}) from a structured/essay theory paper.

TASK:
Extract all questions, parts, and sub-parts appearing on THIS page.

STRUCTURE OF QUESTIONS, PARTS, AND SUB-PARTS:
- A question may contain:
  1. Main preamble/stem text in "Question_text" (e.g. "A car travels along a straight road..."). If a question continues from a previous page without a new stem, leave "Question_text": "".
  2. If the question has NO parts or subparts (e.g. a single-part mathematics question):
     - Put all question text in "Question_text".
     - Set "Parts": [].
     - "Marks": integer marks for the question [N].
  3. If the question has parts (e.g. (a), (b), (c)):
     - "part_label": "a", "b", etc.
     - "part_text": Introductory text or question text for that part. If part (a) immediately divides into (i) and (ii), place any introductory sentence in "part_text" (or "" if none).
     - "marks": mark allocation integer [N] indicated in brackets (e.g. [2] -> 2). Default to 1.
     - "sub_parts": If part (a) or (b) has nested sub-parts (e.g. roman numerals (i), (ii)):
       - "sub_part_label": Strictly formatted with parentheses, e.g. "(i)", "(ii)", "(iii)", "(iv)". NEVER use "ai", "aii", "aiii", "a(i)", or "b(i)".
       - "sub_part_text": verbatim text of the sub-part.
       - "marks": integer marks for this sub-part (e.g. [1] -> 1, [2] -> 2). Leaf sub-parts must have marks >= 1.
       - "sub_parts": Any deeper nested sub-subparts (e.g. "(1)", "(a)"), else empty list [].
- "Marks": Total marks allocated to the question on this page (sum of parts/subparts).
- Leave Unit, Topic, Topic_name, and Sub_topic strictly as null (do NOT classify topics or syllabus; classification is performed in a separate stage).

CRITICAL RULES:
1. Extract ONLY questions and parts visible on this page.
2. DO NOT include answer blanks, dotted lines, or response prompts (e.g. "...................", "height = ..........", "Answer: [ ]").
3. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", "[Total: 15]").
4. Every leaf part, sub-part, and question must have marks >= 1 (never 0).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.ESSAY_MS:
            schema = EssayAnswersContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine this exact mark scheme page (Page {pno + 1} of {total_pages}) for a structured/essay theory paper.

TASK:
Extract all question solutions, marking points, and criteria visible on THIS page.
For each question:
- AnswerNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q7".
- Answer_text: General answer guidelines or main answer ONLY IF there is an overarching question solution without parts.
  CRITICAL: If the solutions are divided into parts (e.g. part (a), (b), etc.) and there is NO overarching/main text for the question as a whole, set "Answer_text": "" (empty string). DO NOT put gibberish, placeholder text (such as '...', '---', 'N/A', 'None'), duplicate part text, or table headers into "Answer_text" when text only belongs in the parts.
- Parts: List of part answers matching the question parts:
  - part_label: e.g. "a", "b", "(i)", "(ii)".
  - part_answer: suggested answer text, formulas, or marking criteria.

CRITICAL RULES:
1. Extract ONLY solutions visible on this page.
2. DO NOT include header/footer boilerplate.
3. NEVER put gibberish or placeholder characters into "Answer_text" when answers belong to individual parts.

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""
        else:
            raise ValueError(f"AgenticExtractor: unsupported category {category}")

        return schema, prompt

    def _get_full_pdf_prompt(
        self,
        category: PaperCategory,
        meta: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """Returns the target Pydantic container schema and prompt for full-PDF single-call Gemini extraction."""
        session = str(meta.get("session", "jun")).capitalize()
        variant = str(meta.get("variant", "11"))
        year = meta.get("year", 2023)

        if category == PaperCategory.MCQ_QP:
            schema = MCQQuestionsContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine the complete attached Multiple Choice Question (MCQ) past paper document.

TASK:
Extract ALL multiple-choice questions (e.g. Q1 through Q30 or Q40) across the entire paper in sequential order.
For each question:
- QuestionNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q2".
- Question_text: The complete verbatim question stem text.
- Options: An object with keys "A", "B", "C", "D" containing the text for each choice.
- Marks: 1 for every MCQ question.
- Unit: null
- Topic: null
- Topic_name: null
- Sub_topic: null
(Leave Unit, Topic, Topic_name, and Sub_topic strictly as null. Do NOT classify topics or syllabus.)

CRITICAL RULES:
1. Extract every single question in the paper. Do NOT combine questions.
2. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", candidate details).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.MCQ_MS:
            schema = MCQAnswersContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine the complete attached Multiple Choice (MCQ) mark scheme document.

TASK:
Extract all question numbers and their corresponding correct answer letter keys (A, B, C, or D) across the entire mark scheme.
For each answer:
- AnswerNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q2".
- Answer: The single uppercase letter (A, B, C, or D).

CRITICAL RULES:
1. Extract all answers in numerical order.
2. DO NOT include headers or footers.

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.ESSAY_QP:
            schema = EssayQuestionsContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine the complete attached Cambridge examination past paper document for a structured/essay theory paper.

TASK:
Extract ALL questions, parts, and sub-parts across the entire paper in sequential order (Q1 through the end of the paper).

STRUCTURE OF QUESTIONS, PARTS, AND SUB-PARTS:
- A question may contain:
  1. Main preamble/scenario text in "Question_text" (e.g. "A train travels along a straight track...").
  2. If the question has NO parts or subparts (e.g. a single-part mathematics question):
     - Put all question text in "Question_text".
     - Set "Parts": [].
     - "Marks": integer marks for the question indicated in brackets [N].
  3. If the question has parts (e.g. (a), (b), (c)):
     - "part_label": "a", "b", "c", etc. (lowercase letter without parentheses).
     - "part_text": Text for that part. If part (a) immediately divides into (i) and (ii), place any introductory sentence in "part_text" (or "" if none).
     - "marks": mark allocation integer [N] indicated in brackets (e.g. [2] -> 2). Default to 1.
     - "sub_parts": If part (a) or (b) has nested sub-parts (e.g. roman numerals (i), (ii)):
       - "sub_part_label": Strictly roman numerals in parentheses, e.g. "(i)", "(ii)", "(iii)", "(iv)". NEVER use "ai", "aii", "aiii", "a(i)", or "b(i)".
       - "sub_part_text": verbatim text of the sub-part.
       - "marks": integer marks for this sub-part (e.g. [1] -> 1, [2] -> 2). Leaf sub-parts must have marks >= 1.
       - "sub_parts": Any deeper nested sub-subparts (e.g. "(1)", "(a)"), else empty list [].
- "Marks": Total marks allocated to the question (sum of all parts and subparts).
- Leave Unit, Topic, Topic_name, and Sub_topic strictly as null (do NOT classify topics or syllabus; classification is performed in a separate stage).

CRITICAL RULES:
1. Extract ALL questions from Q1 to the end of the paper without omitting any question.
2. DO NOT include answer blanks, dotted lines, or response prompts (e.g. "...................", "height = ..........", "Answer: [ ]").
3. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", candidate instructions).
4. Every leaf part, sub-part, and question must have marks >= 1 (never 0).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""

        elif category == PaperCategory.ESSAY_MS:
            schema = EssayAnswersContainer
            prompt = f"""You are an expert Cambridge examination transcription agent.
Examine the complete attached mark scheme document for a structured/essay theory paper.

TASK:
Extract all question solutions, marking points, and criteria across the entire mark scheme.
For each question:
- AnswerNumber: E.g., "{session}/{variant}/{year}/Q1", "{session}/{variant}/{year}/Q2".
- Answer_text: General answer guidelines or main answer ONLY IF there is an overarching question solution without parts.
  CRITICAL: If the solutions are divided into parts (e.g. part (a), (b), etc.) and there is NO overarching/main text for the question as a whole, set "Answer_text": "" (empty string). DO NOT put gibberish, placeholder text (such as '...', '---', 'N/A', 'None'), duplicate part text, or table headers into "Answer_text" when text only belongs in the parts.
- Parts: List of part answers matching the question parts:
  - part_label: e.g. "a", "b", "(i)", "(ii)".
  - part_answer: suggested answer text, formulas, or marking criteria.

CRITICAL RULES:
1. Extract ALL solutions in sequential order.
2. DO NOT include header/footer boilerplate.
3. NEVER put gibberish or placeholder characters into "Answer_text" when answers belong to individual parts.

{self.MATH_NOTATION_INSTRUCTIONS}

{self.SANITY_CHECK_INSTRUCTIONS}
"""
        else:
            raise ValueError(f"AgenticExtractor: unsupported category {category}")

        return schema, prompt

    # ── Full PDF Extraction (Gemini Primary) ──────────────────────────────

    def _extract_with_gemini_full_pdf(
        self,
        pdf_path: Path,
        category: PaperCategory,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract complete PDF using Google Gemini in a single multimodal call,
        passing the full PDF bytes and receiving structured output for the entire paper.
        """
        client = self._get_gemini_client()
        schema, prompt = self._get_full_pdf_prompt(category, meta)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        last_err = None
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                        max_output_tokens=65535,
                    ),
                )
                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    return []

                data = json.loads(raw_text)
                if "questions" in data:
                    return data["questions"]
                elif "answers" in data:
                    return data["answers"]
                elif isinstance(data, list):
                    return data
                return []
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "resource_exhausted" in err_str or "429" in err_str:
                    raise e

                is_transient = any(
                    kw in err_str
                    for kw in [
                        "ssl",
                        "eof",
                        "connection",
                        "connecterror",
                        "timeout",
                        "timed out",
                        "remotedisconnected",
                        "reset",
                        "503",
                        "500",
                    ]
                )
                if is_transient and attempt < 3:
                    logger.warning(
                        f"Transient Gemini connection error during full-PDF extraction (attempt {attempt}/3): {e}. "
                        f"Retrying in {attempt * 2}s..."
                    )
                    time.sleep(attempt * 2)
                    self._gemini_client = None
                    continue
                break

        if last_err is not None:
            raise last_err
        return []

    # ── Page-Level Extraction ─────────────────────────────────────────────

    def _extract_page_with_gemini(
        self,
        doc: fitz.Document,
        pno: int,
        schema: Any,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Extract a single page using Gemini with native PDF bytes and retries."""
        # Slice single page into in-memory PDF
        sub_doc = fitz.open()
        sub_doc.insert_pdf(doc, from_page=pno, to_page=pno)
        page_pdf_bytes = sub_doc.tobytes(garbage=3, deflate=True)
        sub_doc.close()

        last_err = None
        for attempt in range(1, 4):
            try:
                client = self._get_gemini_client()
                # On attempt 3 if previous attempts failed with connection/format issues, try sending image/png
                if attempt == 3:
                    page = doc[pno]
                    pix = page.get_pixmap(dpi=150)
                    part = types.Part.from_bytes(data=pix.tobytes("png"), mime_type="image/png")
                else:
                    part = types.Part.from_bytes(data=page_pdf_bytes, mime_type="application/pdf")

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    return []

                data = json.loads(raw_text)
                if "questions" in data:
                    return data["questions"]
                elif "answers" in data:
                    return data["answers"]
                elif isinstance(data, list):
                    return data
                return []
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # If quota exhausted, don't wait — immediately fall back to Qwen
                if "resource_exhausted" in err_str or "429" in err_str:
                    raise e

                is_transient = any(
                    kw in err_str
                    for kw in [
                        "ssl",
                        "eof",
                        "connection",
                        "connecterror",
                        "timeout",
                        "timed out",
                        "remotedisconnected",
                        "reset",
                        "503",
                        "500",
                    ]
                )
                if is_transient and attempt < 3:
                    logger.warning(
                        f"Transient Gemini connection error on page {pno + 1} (attempt {attempt}/3): {e}. "
                        f"Retrying in {attempt * 2}s..."
                    )
                    time.sleep(attempt * 2)
                    self._gemini_client = None  # Reset client to recreate fresh TLS connection
                    continue
                break

        if last_err is not None:
            raise last_err
        return []

    def _extract_page_with_qwen(
        self,
        doc: fitz.Document,
        pno: int,
        schema: Any,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Fallback extraction using Qwen VL via OpenRouter with page image."""
        # Render page as PNG image
        page = doc[pno]
        pix = page.get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        base64_img = base64.b64encode(png_bytes).decode("utf-8")

        client = self._get_openai_client()

        # Candidate models if primary fallback is unavailable
        candidate_models = [self.fallback_model]
        for fallback_cand in ["qwen/qwen3-vl-8b-instruct", "qwen/qwen2.5-vl-72b-instruct"]:
            if fallback_cand not in candidate_models:
                candidate_models.append(fallback_cand)

        last_err = None
        for model in candidate_models:
            try:
                # 1. Try structured parse
                try:
                    completion = client.beta.chat.completions.parse(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                                    },
                                ],
                            }
                        ],
                        response_format=schema,
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    parsed_obj = completion.choices[0].message.parsed
                    if hasattr(parsed_obj, "questions"):
                        return [q.model_dump() for q in parsed_obj.questions]
                    elif hasattr(parsed_obj, "answers"):
                        return [a.model_dump() for a in parsed_obj.answers]
                    return []
                except Exception as parse_err:
                    if any(c in str(parse_err) for c in ["404", "No endpoints", "402"]):
                        raise parse_err
                    logger.warning(
                        f"OpenRouter structured parse with {model} failed ({parse_err}). Retrying with json_object mode..."
                    )
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt + "\nOutput strictly valid JSON."},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                                    },
                                ],
                            }
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    raw_text = response.choices[0].message.content
                    if not raw_text or not raw_text.strip():
                        return []
                    data = json.loads(raw_text)
                    if "questions" in data:
                        return data["questions"]
                    elif "answers" in data:
                        return data["answers"]
                    elif isinstance(data, list):
                        return data
                    return []
            except Exception as cand_err:
                last_err = cand_err
                logger.warning(f"OpenRouter model {model} failed on page {pno + 1}: {cand_err}")
                continue

        if last_err is not None:
            raise last_err
        return []

    def _extract_page(
        self,
        doc: fitz.Document,
        pno: int,
        total_pages: int,
        category: PaperCategory,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract a single page with Gemini, falling back to Qwen VL on failure or quota exhaustion."""
        schema, prompt = self._get_page_prompt(category, pno, total_pages, meta)

        if not self._gemini_quota_exhausted:
            try:
                logger.info(f"Extracting page {pno + 1}/{total_pages} with Gemini ({self.model_name})...")
                return self._extract_page_with_gemini(doc, pno, schema, prompt)
            except Exception as gemini_err:
                err_str = str(gemini_err).lower()
                if "resource_exhausted" in err_str or "429" in err_str:
                    logger.warning(
                        f"Gemini quota exhausted ({gemini_err}). "
                        f"Switching to Qwen ({self.fallback_model}) for page {pno + 1} and remaining pages..."
                    )
                    self._gemini_quota_exhausted = True
                else:
                    logger.warning(
                        f"Gemini page {pno + 1} extraction failed: {gemini_err}. "
                        f"Falling back to Qwen ({self.fallback_model})..."
                    )

        # Qwen VL fallback
        try:
            logger.info(f"Extracting page {pno + 1}/{total_pages} with Qwen ({self.fallback_model})...")
            return self._extract_page_with_qwen(doc, pno, schema, prompt)
        except Exception as qwen_err:
            logger.error(f"Both Gemini and Qwen fallback failed on page {pno + 1}: {qwen_err}")
            raise qwen_err

    # ── Post-Processors ───────────────────────────────────────────────────

    def _postprocess_essay_qp(
        self,
        questions: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Post-processes essay questions: normalizes QuestionNumber, cleans math and answer blanks,
        normalizes sub-part labels strictly to (i), (ii), (iii), rolls up marks, and guarantees marks >= 1.
        """
        results = []
        for raw_q in questions:
            qn_raw = raw_q.get("QuestionNumber", "")
            qnum_str = self._fmt_qnum(qn_raw, meta)
            stem = PyMuPDFExtractor.clean_math_text(
                PyMuPDFExtractor._clean_answer_prompts(raw_q.get("Question_text", "") or "")
            )

            q_obj = {
                "QuestionNumber": qnum_str,
                "Question_text": stem,
                "Unit": None,
                "Topic": None,
                "Topic_name": None,
                "Sub_topic": None,
                "Difficulty_score": raw_q.get("Difficulty_score", 0) or 0,
                "Difficulty_label": raw_q.get("Difficulty_label", "") or "",
                "Command_words": raw_q.get("Command_words", []) or [],
                "Marks": max(raw_q.get("Marks", 1), 1),
                "image": raw_q.get("image", None),
                "Parts": [],
            }

            def _clean_node(node: Dict[str, Any], is_top: bool = False) -> int:
                text_key = "part_text" if is_top else "sub_part_text"
                node[text_key] = PyMuPDFExtractor.clean_math_text(
                    PyMuPDFExtractor._clean_answer_prompts(node.get(text_key, "") or "")
                )
                if not is_top:
                    lbl = node.get("sub_part_label", "")
                    node["sub_part_label"] = self._normalize_subpart_label(lbl)

                sub_parts = node.get("sub_parts", [])
                if sub_parts:
                    node["marks"] = sum(_clean_node(sp, is_top=False) for sp in sub_parts)
                else:
                    if node.get("marks", 0) <= 0:
                        node["marks"] = 1
                return node["marks"]

            for p in raw_q.get("Parts", []):
                p_label = str(p.get("part_label", "")).strip().lower().replace("(", "").replace(")", "")
                part_node = {
                    "part_label": p_label,
                    "part_text": p.get("part_text", "") or "",
                    "marks": max(p.get("marks", 1), 1),
                    "image": p.get("image", None),
                    "sub_parts": [],
                }
                for sp in p.get("sub_parts", []):
                    sp_node = {
                        "sub_part_label": sp.get("sub_part_label", "") or "",
                        "sub_part_text": sp.get("sub_part_text", "") or "",
                        "marks": max(sp.get("marks", 1), 1),
                        "image": sp.get("image", None),
                        "sub_parts": [],
                    }
                    for ssp in sp.get("sub_parts", []):
                        ssp_node = {
                            "sub_part_label": str(ssp.get("sub_part_label", "")).strip(),
                            "sub_part_text": ssp.get("sub_part_text", "") or "",
                            "marks": max(ssp.get("marks", 1), 1),
                            "image": ssp.get("image", None),
                        }
                        sp_node["sub_parts"].append(ssp_node)
                    part_node["sub_parts"].append(sp_node)

                _clean_node(part_node, is_top=True)
                q_obj["Parts"].append(part_node)

            if q_obj["Parts"]:
                q_obj["Marks"] = sum(p["marks"] for p in q_obj["Parts"])
            else:
                q_obj["Marks"] = max(raw_q.get("Marks", 1), 1)

            results.append(q_obj)

        return results

    def _postprocess_mcq_qp(
        self,
        questions: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Post-process MCQ questions: normalizes QuestionNumber, cleans options and math."""
        results = []
        for i, raw_q in enumerate(questions, start=1):
            qn_raw = raw_q.get("QuestionNumber", "")
            qnum_str = self._fmt_qnum(qn_raw if qn_raw else i, meta)
            opts = raw_q.get("Options", {})
            if isinstance(opts, dict):
                cleaned_opts = {
                    k: PyMuPDFExtractor.clean_math_text(str(opts.get(k, "")))
                    for k in ["A", "B", "C", "D"]
                }
            else:
                cleaned_opts = {"A": "", "B": "", "C": "", "D": ""}

            results.append({
                "QuestionNumber": qnum_str,
                "Question_text": PyMuPDFExtractor.clean_math_text(
                    raw_q.get("Question_text", "") or ""
                ),
                "Unit": None,
                "Topic": None,
                "Topic_name": None,
                "Sub_topic": None,
                "Difficulty_score": raw_q.get("Difficulty_score", 0) or 0,
                "Difficulty_label": raw_q.get("Difficulty_label", "") or "",
                "Command_words": raw_q.get("Command_words", []) or [],
                "Options": cleaned_opts,
                "Marks": 1,
                "image": raw_q.get("image", None),
            })
        return results

    def _postprocess_mcq_ms(
        self,
        answers: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Post-process MCQ answers: normalizes AnswerNumber and letter key."""
        results = []
        for i, raw_a in enumerate(answers, start=1):
            an_raw = raw_a.get("AnswerNumber", "")
            ans_str = self._fmt_qnum(an_raw if an_raw else i, meta)
            ans_char = str(raw_a.get("Answer", "")).strip().upper()
            if ans_char not in ["A", "B", "C", "D"] and ans_char:
                ans_char = ans_char[0]
            results.append({
                "AnswerNumber": ans_str,
                "Answer": ans_char or "A",
                "image": raw_a.get("image", None),
            })
        return results

    def _postprocess_essay_ms(
        self,
        answers: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Post-process essay mark scheme answers."""
        results = []
        for i, raw_a in enumerate(answers, start=1):
            an_raw = raw_a.get("AnswerNumber", "")
            ans_str = self._fmt_qnum(an_raw if an_raw else i, meta)
            ans_text = PyMuPDFExtractor.clean_math_text(raw_a.get("Answer_text", "") or "")
            parts = []
            for p in raw_a.get("Parts", []):
                parts.append({
                    "part_label": str(p.get("part_label", "")).strip(),
                    "part_answer": PyMuPDFExtractor.clean_math_text(p.get("part_answer", "") or ""),
                    "image": p.get("image", None),
                })

            # If parts exist and Answer_text contains gibberish, placeholder, or repeats part text/label
            if parts and ans_text:
                stripped_ans = ans_text.strip().lower()
                part_labels = {p["part_label"].strip("() ").lower() for p in parts if p.get("part_label")}
                part_answers = {p["part_answer"].strip().lower() for p in parts if p.get("part_answer")}

                is_placeholder = bool(
                    re.match(
                        r"^(?:[\.\-–—_:\s\*#]+|n/?a|none|null|nil|see\s+(?:parts?|below)|as\s+above|refer\s+to.*|part\s*\(?[a-z\divx]+\)?.*|answers?\s*\(?[a-z\divx]*\)?.*)$",
                        stripped_ans,
                        re.IGNORECASE,
                    )
                )
                is_duplicate = (
                    stripped_ans in part_answers
                    or stripped_ans.strip("() ") in part_labels
                )

                if is_placeholder or is_duplicate:
                    ans_text = ""

            results.append({
                "AnswerNumber": ans_str,
                "Answer_text": ans_text,
                "image": raw_a.get("image", None),
                "Parts": parts,
            })
        return results

    # ── Multi-Page Stitcher (for Qwen Fallback) ───────────────────────────

    def _stitch_essay_qp(
        self,
        pages_records: List[List[Dict[str, Any]]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stitch essay questions and parts spanning across multiple pages."""
        questions_map: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []

        for page_recs in pages_records:
            for raw_q in page_recs:
                qn_str = str(raw_q.get("QuestionNumber", ""))
                m = re.search(r"(\d{1,2})", qn_str)
                qnum = int(m.group(1)) if m else (order[-1] if order else 1)

                if qnum not in questions_map:
                    order.append(qnum)
                    questions_map[qnum] = {
                        "QuestionNumber": self._fmt_qnum(qnum, meta),
                        "Question_text": raw_q.get("Question_text", "") or "",
                        "Unit": None,
                        "Topic": None,
                        "Topic_name": None,
                        "Sub_topic": None,
                        "Difficulty_score": 0,
                        "Difficulty_label": "",
                        "Command_words": [],
                        "Marks": max(raw_q.get("Marks", 1), 1),
                        "image": None,
                        "Parts": [],
                    }

                existing_q = questions_map[qnum]

                # Merge stem continuation
                new_stem = raw_q.get("Question_text", "") or ""
                if new_stem and new_stem not in existing_q["Question_text"]:
                    if not existing_q["Question_text"]:
                        existing_q["Question_text"] = new_stem
                    else:
                        existing_q["Question_text"] = f"{existing_q['Question_text']} {new_stem}".strip()

                # Merge parts
                for p in raw_q.get("Parts", []):
                    p_label = str(p.get("part_label", "")).strip().lower().replace("(", "").replace(")", "")
                    existing_part = next(
                        (ep for ep in existing_q["Parts"] if ep.get("part_label") == p_label),
                        None,
                    )
                    if existing_part is None:
                        new_part = {
                            "part_label": p_label,
                            "part_text": p.get("part_text", "") or "",
                            "marks": max(p.get("marks", 1), 1),
                            "image": None,
                            "sub_parts": list(p.get("sub_parts", [])),
                        }
                        existing_q["Parts"].append(new_part)
                    else:
                        p_text = p.get("part_text", "") or ""
                        if p_text and p_text not in existing_part.get("part_text", ""):
                            existing_part["part_text"] = f"{existing_part.get('part_text', '')} {p_text}".strip()
                        for sp in p.get("sub_parts", []):
                            sp_lbl = str(sp.get("sub_part_label", "")).strip()
                            if not any(esp.get("sub_part_label") == sp_lbl for esp in existing_part["sub_parts"]):
                                existing_part["sub_parts"].append(sp)

        raw_list = [questions_map[qn] for qn in order]
        return self._postprocess_essay_qp(raw_list, meta)

    def _stitch_mcq_qp(
        self,
        pages_records: List[List[Dict[str, Any]]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stitch MCQ questions across pages."""
        questions_map: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []

        for page_recs in pages_records:
            for raw_q in page_recs:
                qn_str = str(raw_q.get("QuestionNumber", ""))
                m = re.search(r"(\d{1,2})", qn_str)
                qnum = int(m.group(1)) if m else (len(order) + 1)

                if qnum not in questions_map:
                    order.append(qnum)
                    questions_map[qnum] = raw_q

        raw_list = [questions_map[qn] for qn in sorted(order)]
        return self._postprocess_mcq_qp(raw_list, meta)

    def _stitch_mcq_ms(
        self,
        pages_records: List[List[Dict[str, Any]]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stitch MCQ answers across pages."""
        answers_map: Dict[int, Dict[str, Any]] = {}

        for page_recs in pages_records:
            for raw_a in page_recs:
                an_str = str(raw_a.get("AnswerNumber", ""))
                m = re.search(r"(\d{1,2})", an_str)
                if m:
                    qnum = int(m.group(1))
                    answers_map[qnum] = raw_a

        raw_list = [answers_map[qn] for qn in sorted(answers_map.keys())]
        return self._postprocess_mcq_ms(raw_list, meta)

    def _stitch_essay_ms(
        self,
        pages_records: List[List[Dict[str, Any]]],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stitch essay mark scheme answers across pages."""
        answers_map: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []

        for page_recs in pages_records:
            for raw_a in page_recs:
                an_str = str(raw_a.get("AnswerNumber", ""))
                m = re.search(r"(\d{1,2})", an_str)
                qnum = int(m.group(1)) if m else (len(order) + 1)

                if qnum not in answers_map:
                    order.append(qnum)
                    answers_map[qnum] = {
                        "AnswerNumber": raw_a.get("AnswerNumber", ""),
                        "Answer_text": raw_a.get("Answer_text", "") or "",
                        "image": None,
                        "Parts": [],
                    }

                existing = answers_map[qnum]
                for p in raw_a.get("Parts", []):
                    existing["Parts"].append(p)

        raw_list = [answers_map[qn] for qn in sorted(order)]
        return self._postprocess_essay_ms(raw_list, meta)

    # ── Public Entry Point ────────────────────────────────────────────────

    def extract(
        self,
        pdf_path: Path,
        category: PaperCategory,
        meta_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for agentic text extraction.
        Primary: Sends the complete PDF to Gemini in a single multimodal call.
        Fallback: Slices page-by-page and extracts with Qwen VL via OpenRouter, then stitches.
        """
        pdf_path = Path(pdf_path)

        # ── Primary: Full PDF extraction with Gemini ───────────────────────
        if not self._gemini_quota_exhausted:
            try:
                logger.info(f"Extracting complete PDF with Gemini ({self.model_name}): {pdf_path.name}...")
                raw_records = self._extract_with_gemini_full_pdf(pdf_path, category, meta_context)
                if raw_records:
                    logger.info(f"Gemini successfully extracted {len(raw_records)} records from {pdf_path.name}.")
                    if category == PaperCategory.ESSAY_QP:
                        return self._postprocess_essay_qp(raw_records, meta_context)
                    elif category == PaperCategory.MCQ_QP:
                        return self._postprocess_mcq_qp(raw_records, meta_context)
                    elif category == PaperCategory.MCQ_MS:
                        return self._postprocess_mcq_ms(raw_records, meta_context)
                    elif category == PaperCategory.ESSAY_MS:
                        return self._postprocess_essay_ms(raw_records, meta_context)
                    return raw_records
                else:
                    logger.warning(f"Gemini returned 0 records for {pdf_path.name}. Falling back to Qwen...")
            except Exception as e:
                err_str = str(e).lower()
                if "resource_exhausted" in err_str or "429" in err_str:
                    logger.warning(f"Gemini quota exhausted ({e}). Falling back to Qwen ({self.fallback_model})...")
                    self._gemini_quota_exhausted = True
                else:
                    logger.warning(f"Gemini extraction failed ({e}). Falling back to Qwen ({self.fallback_model})...")

        # ── Fallback: Page-by-page extraction with Qwen VL ────────────────
        logger.info(f"Starting Qwen page-by-page fallback extraction for {pdf_path.name} ({self.fallback_model})...")
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages_records: List[List[Dict[str, Any]]] = []

        for pno in range(total_pages):
            page = doc[pno]
            text = page.get_text().strip()

            # Skip blank pages
            if not text or text.upper() == "BLANK PAGE":
                logger.debug(f"Skipping blank page {pno + 1}")
                continue

            # Skip cover page if it is purely candidate instructions with no questions
            if pno == 0 and ("READ THESE INSTRUCTIONS FIRST" in text or "Write your Centre number" in text):
                if not re.search(r"^\s*(?:Question\s*1|1\b[^\n]*\[\d+\])", text, re.MULTILINE):
                    logger.info(f"Skipping instruction cover page {pno + 1}")
                    continue

            schema, prompt = self._get_page_prompt(category, pno, total_pages, meta_context)
            logger.info(f"Extracting page {pno + 1}/{total_pages} with Qwen ({self.fallback_model})...")
            page_recs = self._extract_page_with_qwen(doc, pno, schema, prompt)
            if page_recs:
                pages_records.append(page_recs)

        doc.close()

        # Stitch pages according to paper category
        if category == PaperCategory.ESSAY_QP:
            return self._stitch_essay_qp(pages_records, meta_context)
        elif category == PaperCategory.MCQ_QP:
            return self._stitch_mcq_qp(pages_records, meta_context)
        elif category == PaperCategory.MCQ_MS:
            return self._stitch_mcq_ms(pages_records, meta_context)
        elif category == PaperCategory.ESSAY_MS:
            return self._stitch_essay_ms(pages_records, meta_context)
        else:
            raise ValueError(f"AgenticExtractor: unsupported category {category}")
