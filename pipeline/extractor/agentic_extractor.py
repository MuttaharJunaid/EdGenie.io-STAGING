"""
pipeline/extractor/agentic_extractor.py

Agentic Text Extractor — 100% LLM-based extraction.

Extracts question text, options, hierarchical parts, and mark schemes
using Google Gemini (full-PDF multimodal primary + page-by-page chunked fallback).
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

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperCategory
from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
from pipeline.extractor.marks_validator import MarkHierarchyValidator
from pipeline.extractor.schemas import (
    EssayAnswersContainer,
    EssayQuestionsContainer,
    MCQAnswersContainer,
    MCQQuestionsContainer,
)

logger = logging.getLogger(__name__)


class AgenticExtractor:
    """
    Extracts structured questions and mark schemes using 100% LLM analysis via Google Gemini.
    Primary: Complete PDF extracted in a single multimodal call.
    Fallback: Automatic chunked page-by-page extraction with Gemini and multi-page stitching
              if full-PDF generation truncates or hits token limits.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or getattr(settings, "llm_model", "gemini-2.5-flash")
        self._gemini_client: Optional[genai.Client] = None
        self._gemini_quota_exhausted: bool = False
        self._gemini_cooldown_until: Optional[float] = None

    @property
    def is_gemini_available(self) -> bool:
        """Returns whether Gemini is available, taking into account cooldown periods."""
        if self._gemini_cooldown_until:
            if time.time() < self._gemini_cooldown_until:
                return False
            else:
                logger.info("Gemini cooldown period expired; re-enabling Gemini.")
                self._gemini_cooldown_until = None
                self._gemini_quota_exhausted = False
        return not self._gemini_quota_exhausted

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

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
6. Marks Guarantee & Sum Consistency: Confirm that leaf marks >= 1, sum of sub-parts equals part marks, and sum of parts equals question Marks.
"""

    MARKS_INSTRUCTIONS = """
CRITICAL MARK ALLOCATION & HIERARCHY RULES:
1. Bracketed Marks [N]: In Cambridge exam papers, mark allocations are explicitly printed in square brackets at the right-hand margin (e.g., [1], [2], [3], [4]). You MUST look specifically for these bracketed numbers. NEVER guess marks, and NEVER default to 1 unless the paper explicitly states [1].
2. Strip [N] from Text: When extracting question prose, part text, or sub-part text, do NOT leave trailing mark brackets like '[1]' or '[2]' in the text field. The numeric mark goes strictly into the integer "marks" field, and the prose text must be clean.
3. Sub-parts Mark Sum Rule: If a part (e.g., part (a)) contains nested sub-parts (e.g., (i) and (ii)), each sub-part has its own explicit mark [N]. The parent part's "marks" MUST strictly equal the SUM of all its constituent sub-parts' marks:
   part.marks = sum(sub_part.marks)
4. Parts Mark Sum Rule: If a question contains parts (e.g., (a), (b), (c)), the total "Marks" of the question MUST strictly equal the SUM of all its constituent parts' marks:
   question.Marks = sum(part.marks)
5. Single-Part Questions: If a question has NO parts or sub-parts, its total "Marks" MUST match the explicit mark [N] printed at the end of that question.
6. Total Question Check: Many Cambridge papers print '[Total: X]' or '[Total: 10]' at the end of a question. Verify that the sum of all parts equals X. Never create an extra question or part for the '[Total: X]' label.
7. Leaf Marks >= 1: Every leaf part, sub-part, or question must have marks >= 1 (never 0).
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
     - "marks": mark allocation integer [N] indicated in brackets (e.g. [2] -> 2). If part divides into sub-parts, part.marks MUST equal the sum of its sub-parts' marks.
     - "sub_parts": If part (a) or (b) has nested sub-parts (e.g. roman numerals (i), (ii)):
       - "sub_part_label": Strictly formatted with parentheses, e.g. "(i)", "(ii)", "(iii)", "(iv)". NEVER use "ai", "aii", "aiii", "a(i)", or "b(i)".
       - "sub_part_text": verbatim text of the sub-part (strip trailing [N] bracket from text).
       - "marks": explicit integer marks indicated in brackets [N] for this sub-part (e.g. [1] -> 1, [2] -> 2). Leaf sub-parts must have marks >= 1.
       - "sub_parts": Any deeper nested sub-subparts (e.g. "(1)", "(a)"), else empty list [].
- "Marks": Total marks allocated to the question on this page (sum of parts/subparts).
- Leave Unit, Topic, Topic_name, and Sub_topic strictly as null (do NOT classify topics or syllabus; classification is performed in a separate stage).

CRITICAL RULES:
1. Extract ONLY questions and parts visible on this page.
2. DO NOT include answer blanks, dotted lines, or response prompts (e.g. "...................", "height = ..........", "Answer: [ ]").
3. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", "[Total: 15]").
4. Every leaf part, sub-part, and question must have marks >= 1 (never 0).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.MARKS_INSTRUCTIONS}

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
     - "marks": mark allocation integer [N] indicated in brackets (e.g. [2] -> 2). If part divides into sub-parts, part.marks MUST equal the sum of its sub-parts' marks.
     - "sub_parts": If part (a) or (b) has nested sub-parts (e.g. roman numerals (i), (ii)):
       - "sub_part_label": Strictly roman numerals in parentheses, e.g. "(i)", "(ii)", "(iii)", "(iv)". NEVER use "ai", "aii", "aiii", "a(i)", or "b(i)".
       - "sub_part_text": verbatim text of the sub-part (strip trailing [N] bracket from text).
       - "marks": explicit integer marks indicated in brackets [N] for this sub-part (e.g. [1] -> 1, [2] -> 2). Leaf sub-parts must have marks >= 1.
       - "sub_parts": Any deeper nested sub-subparts (e.g. "(1)", "(a)"), else empty list [].
- "Marks": Total marks allocated to the question (sum of all parts and subparts).
- Leave Unit, Topic, Topic_name, and Sub_topic strictly as null (do NOT classify topics or syllabus; classification is performed in a separate stage).

CRITICAL RULES:
1. Extract ALL questions from Q1 to the end of the paper without omitting any question.
2. DO NOT include answer blanks, dotted lines, or response prompts (e.g. "...................", "height = ..........", "Answer: [ ]").
3. DO NOT include header/footer boilerplate (e.g. "UCLES", "Turn over", candidate instructions).
4. Every leaf part, sub-part, and question must have marks >= 1 (never 0).

{self.MATH_NOTATION_INSTRUCTIONS}

{self.MARKS_INSTRUCTIONS}

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

    @staticmethod
    def _prune_pdf_bytes(pdf_path: Path) -> bytes:
        """
        Removes blank and non-question pages using PyMuPDF before sending
        to Gemini multimodal API to save 20-35% of token quota.
        """
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) <= 2:
                with open(pdf_path, "rb") as f:
                    return f.read()

            pruned_doc = fitz.open()
            pages_kept = 0
            pruned_count = 0
            for pno in range(len(doc)):
                page = doc[pno]
                text = page.get_text().strip()

                # Never prune cover page (pno == 0) as it contains subject metadata / title
                if pno > 0:
                    has_blank_phrase = bool(re.search(r"\bBLANK\s+PAGE\b", text, re.IGNORECASE))
                    has_marks = bool(re.search(r"\[\s*\d+\s*\]", text))
                    has_question = bool(re.search(r"\b(?:Question\s+\d+|^\d+\s+[A-Z])\b", text, re.IGNORECASE | re.MULTILINE))

                    is_purely_empty = not text and len(page.get_images()) == 0 and len(page.get_drawings()) == 0
                    is_cambridge_blank = has_blank_phrase and not has_marks and not has_question

                    if is_purely_empty or is_cambridge_blank:
                        pruned_count += 1
                        continue

                pruned_doc.insert_pdf(doc, from_page=pno, to_page=pno)
                pages_kept += 1

            if pages_kept > 0 and pruned_count > 0:
                logger.info(
                    f"Pruned {pruned_count} blank page(s) from {pdf_path.name} "
                    f"to optimize token quota (sending {pages_kept}/{len(doc)} pages)."
                )
                return pruned_doc.tobytes()
        except Exception as e:
            logger.debug(f"Failed to prune blank pages from {pdf_path.name} ({e}), using full PDF bytes.")

        with open(pdf_path, "rb") as f:
            return f.read()

    @staticmethod
    def _attempt_json_repair(raw: str) -> Optional[str]:
        """Attempts basic repair of truncated JSON responses (e.g. unclosed quotes/brackets)."""
        if not raw:
            return None
        text = raw.strip()

        # Count unescaped double quotes to see if string was cut off
        in_string = False
        escaped = False
        for ch in text:
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            escaped = False

        if in_string:
            text += '"'

        # Close open objects/arrays
        stack = []
        in_string = False
        escaped = False
        for ch in text:
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            elif not in_string:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]" and stack:
                    if stack[-1] == ch:
                        stack.pop()
            escaped = False

        while stack:
            text += stack.pop()

        return text

    def _extract_with_gemini_full_pdf(
        self,
        pdf_path: Path,
        category: PaperCategory,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract complete PDF using Google Gemini in a single multimodal call,
        passing the full PDF bytes and receiving structured output for the entire paper.
        Includes blank page pruning and exponential backoff retry on 429 rate limits.
        """
        client = self._get_gemini_client()
        schema, prompt = self._get_full_pdf_prompt(category, meta)

        pdf_bytes = self._prune_pdf_bytes(pdf_path)
        part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        max_retries = getattr(settings, "gemini_max_retries", 3)
        base_delay = getattr(settings, "gemini_retry_delay", 15.0)

        last_err = None
        for attempt in range(1, max_retries + 1):
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
                # Check for output truncation due to token limits
                if response.candidates:
                    cand = response.candidates[0]
                    finish_reason = getattr(cand, "finish_reason", None)
                    finish_str = str(finish_reason).upper()
                    if "MAX_TOKENS" in finish_str or "LENGTH" in finish_str:
                        logger.warning(
                            f"Gemini generation hit MAX_TOKENS limit ({finish_str}) on {pdf_path.name}."
                        )
                        raise ValueError(f"Gemini output truncated due to MAX_TOKENS ceiling ({finish_str})")

                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    return []

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError as jde:
                    repaired = self._attempt_json_repair(raw_text)
                    if repaired:
                        try:
                            data = json.loads(repaired)
                            logger.info("Successfully recovered truncated JSON via repair heuristic.")
                        except Exception:
                            raise jde
                    else:
                        raise jde

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
                is_429 = "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str

                if is_429:
                    if attempt < max_retries:
                        backoff = base_delay * (2 ** (attempt - 1))
                        m = re.search(r"retry(?:\s+after|\s+in)?\s*[:\s]?\s*(\d+(?:\.\d+)?)\s*s", err_str)
                        if m:
                            backoff = max(float(m.group(1)) + 1.0, backoff)
                        logger.warning(
                            f"Gemini rate limit / 429 hit during full-PDF extraction of {pdf_path.name} "
                            f"(attempt {attempt}/{max_retries}). Backing off for {backoff:.1f}s before retrying..."
                        )
                        time.sleep(backoff)
                        self._gemini_client = None  # Reset client to ensure clean session
                        client = self._get_gemini_client()
                        continue
                    else:
                        logger.error(f"Gemini 429/quota persisted after {max_retries} attempts for {pdf_path.name}.")
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
                if is_transient and attempt < max_retries:
                    logger.warning(
                        f"Transient Gemini connection error during full-PDF extraction (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {attempt * 2}s..."
                    )
                    time.sleep(attempt * 2)
                    self._gemini_client = None
                    client = self._get_gemini_client()
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
                is_429 = "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str

                if is_429:
                    max_retries = getattr(settings, "gemini_max_retries", 3)
                    base_delay = getattr(settings, "gemini_retry_delay", 15.0)
                    if attempt < max_retries:
                        backoff = base_delay * (2 ** (attempt - 1))
                        m = re.search(r"retry(?:\s+after|\s+in)?\s*[:\s]?\s*(\d+(?:\.\d+)?)\s*s", err_str)
                        if m:
                            backoff = max(float(m.group(1)) + 1.0, backoff)
                        logger.warning(
                            f"Gemini rate limit / 429 on page {pno + 1} (attempt {attempt}/{max_retries}). "
                            f"Backing off for {backoff:.1f}s before retrying..."
                        )
                        time.sleep(backoff)
                        self._gemini_client = None
                        client = self._get_gemini_client()
                        continue
                    else:
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
                    client = self._get_gemini_client()
                    continue
                break

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
        """Extract a single page with Gemini."""
        schema, prompt = self._get_page_prompt(category, pno, total_pages, meta)
        logger.info(f"Extracting page {pno + 1}/{total_pages} with Gemini ({self.model_name})...")
        return self._extract_page_with_gemini(doc, pno, schema, prompt)

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

        reconciled_results, stats = MarkHierarchyValidator.reconcile_and_validate_records(results)
        return reconciled_results

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

    # ── Multi-Page Stitcher (for Gemini Page Fallback) ─────────────────────

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
                qn_str = str(raw_q.get("QuestionNumber", "")).strip()
                last_seg = qn_str.split("/")[-1]
                m = re.search(r"(?:Q|Question\s*)?(\d{1,2})\b", last_seg, re.IGNORECASE)
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
                qn_str = str(raw_q.get("QuestionNumber", "")).strip()
                last_seg = qn_str.split("/")[-1]
                m = re.search(r"(?:Q|Question\s*)?(\d{1,2})\b", last_seg, re.IGNORECASE)
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
                an_str = str(raw_a.get("AnswerNumber", "")).strip()
                last_seg = an_str.split("/")[-1]
                m = re.search(r"(?:Q|Question\s*)?(\d{1,2})\b", last_seg, re.IGNORECASE)
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
                an_str = str(raw_a.get("AnswerNumber", "")).strip()
                last_seg = an_str.split("/")[-1]
                m = re.search(r"(?:Q|Question\s*)?(\d{1,2})\b", last_seg, re.IGNORECASE)
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
        Fallback: Slices page-by-page and extracts with Gemini, then stitches (protects against output truncation).
        """
        pdf_path = Path(pdf_path)

        # ── Primary: Full PDF extraction with Gemini ───────────────────────
        if self.is_gemini_available:
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
                    logger.warning(f"Gemini returned 0 records for {pdf_path.name}. Falling back to page-by-page Gemini extraction...")
            except Exception as e:
                logger.warning(
                    f"Gemini full-PDF extraction encountered an issue ({e}). "
                    f"Falling back to chunked page-by-page Gemini extraction to guarantee zero output truncation..."
                )

        # ── Fallback: Gemini Page-by-page extraction ──────────────────────
        logger.info(f"Starting Gemini page-by-page fallback extraction for {pdf_path.name} ({self.model_name})...")
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
            logger.info(f"Extracting page {pno + 1}/{total_pages} with Gemini ({self.model_name})...")
            page_recs = self._extract_page_with_gemini(doc, pno, schema, prompt)
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
