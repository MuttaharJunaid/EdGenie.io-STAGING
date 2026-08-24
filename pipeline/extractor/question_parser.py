import re
import logging
from typing import List, Dict, Any, Tuple
from pipeline.extractor.paper_type_detector import PaperCategory
from pipeline.extractor.text_extractor import ExtractedPage, TextBlock

logger = logging.getLogger(__name__)

class QuestionParser:
    """Parses raw text blocks from a PDF into individual questions/answers based on paper type."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Helper to sanitize text by joining hyphenated words and stripping whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    @classmethod
    def parse_mcq_qp(cls, pages: List[ExtractedPage], subject_code: str, year: int, session: str, variant: str) -> List[Dict[str, Any]]:
        """Parses MCQ Question Paper."""
        questions = []
        all_text = "\n".join([page.raw_text for page in pages])
        
        # Regex to locate questions like: 1 Which statement is...
        # Option choices like: A Statement A, B Statement B, etc.
        # Find all question starts (numbers 1 to 40)
        # We can split the text using question number headers
        # A question start is usually: a number at the beginning of a line, or a new paragraph starting with \d+\b
        # Let's search for patterns like "\n1 " or "\n 1 "
        # Or look through blocks sequentially:
        # A block starting with a digit is a strong indicator.
        
        blocks_by_page = []
        for page in pages:
            for block in page.blocks:
                # Exclude top 55px (header) and bottom 55px (footer) regions
                y0 = block.bbox[1]
                y1 = block.bbox[3]
                if y1 < 55 or y0 > page.height - 55:
                    continue
                blocks_by_page.append(block)
            
        current_q_num = 0
        current_q_text = []
        current_options = {}
        
        # We can scan the blocks to build questions
        i = 0
        while i < len(blocks_by_page):
            block = blocks_by_page[i]
            text = block.text.strip()
            
            # Match a question number (e.g. "1", "2")
            match = re.match(r"^(\d+)\s+(.*)$", text)
            # Or if it's just a number
            if not match and text.isdigit():
                q_num = int(text)
                if q_num == current_q_num + 1:
                    # Save current question if any
                    if current_q_num > 0:
                        questions.append({
                            "num": current_q_num,
                            "text": cls.clean_text(" ".join(current_q_text)),
                            "options": current_options.copy()
                        })
                    current_q_num = q_num
                    current_q_text = []
                    current_options = {}
                    i += 1
                    continue
                    
            if match:
                q_num = int(match.group(1))
                if q_num == current_q_num + 1:
                    if current_q_num > 0:
                        questions.append({
                            "num": current_q_num,
                            "text": cls.clean_text(" ".join(current_q_text)),
                            "options": current_options.copy()
                        })
                    current_q_num = q_num
                    current_q_text = [match.group(2)]
                    current_options = {}
                    i += 1
                    continue

            # Check if it's an option block: e.g. "A Option text" or "A" followed by "Option text"
            option_match = re.match(r"^([A-D])\s+(.*)$", text)
            if option_match:
                opt_letter = option_match.group(1)
                opt_text = option_match.group(2)
                current_options[opt_letter] = opt_text
            elif text in ["A", "B", "C", "D"] and i + 1 < len(blocks_by_page):
                # Next block contains the option text
                current_options[text] = blocks_by_page[i+1].text.strip()
                i += 1
            else:
                # Accumulate question text
                if current_q_num > 0 and not text.startswith("©") and "calculator" not in text.lower() and "turn over" not in text.lower():
                    current_q_text.append(text)
            i += 1

        # Append last question
        if current_q_num > 0:
            questions.append({
                "num": current_q_num,
                "text": cls.clean_text(" ".join(current_q_text)),
                "options": current_options.copy()
            })

        # Format questions to schema
        formatted_questions = []
        for q in questions:
            q_num_str = f"{session.capitalize()}/{variant}/{year}/Q{q['num']}"
            formatted_questions.append({
                "QuestionNumber": q_num_str,
                "Question_text": q["text"],
                # To be populated by Classifier & Analyzer Agents:
                "Unit": "PENDING",
                "Topic": "PENDING",
                "Topic_name": "PENDING",
                "Sub_topic": [],
                "Difficulty_score": 0,
                "Difficulty_label": "PENDING",
                "Command_words": ["identify"], 
                "Options": q["options"],
                "Marks": 1,
                "Parts": None,
                "image": None
            })
        return formatted_questions

    @classmethod
    def parse_mcq_ms(cls, pages: List[ExtractedPage], session: str, variant: str, year: int) -> List[Dict[str, Any]]:
        """Parses MCQ Mark Scheme containing answers like 1 A, 2 B, etc."""
        answers = []
        all_text = "\n".join([page.raw_text for page in pages])
        
        # Regex to find: e.g. 1 A, 2 B, etc.
        # Find pattern: \b(\d+)\s+([A-D])\b
        matches = re.findall(r"\b(\d{1,2})\s+([A-D])\b", all_text)
        
        # Avoid duplicate keys if multiple matching regions exist (e.g. header vs table body)
        seen_questions = set()
        for num_str, ans in matches:
            q_num = int(num_str)
            if q_num not in seen_questions:
                seen_questions.add(q_num)
                ans_num_str = f"{session.capitalize()}/{variant}/{year}/Q{q_num}"
                answers.append({
                    "AnswerNumber": ans_num_str,
                    "Answer": ans,
                    "image": None
                })
        
        # Sort answers by question number
        answers.sort(key=lambda x: int(re.search(r"Q(\d+)", x["AnswerNumber"]).group(1)))
        return answers

    @classmethod
    def parse_essay_qp(cls, pages: List[ExtractedPage], subject_code: str, year: int, session: str, variant: str) -> List[Dict[str, Any]]:
        """Parses Essay-based Question Paper into questions and sub-parts."""
        questions = []
        blocks_by_page = []
        for page in pages:
            for block in page.blocks:
                # Exclude top 55px (header) and bottom 55px (footer) regions
                y0 = block.bbox[1]
                y1 = block.bbox[3]
                if y1 < 55 or y0 > page.height - 55:
                    continue
                blocks_by_page.append(block)

        current_q_num = 0
        current_q_text = []
        current_parts = []
        current_part_label = ""
        current_part_text = []
        current_part_marks = 0

        # Regexes for parts: e.g. (a), (b), (i), (ii)
        part_pattern = re.compile(r"^\(([a-g])\)\s*(.*)$")
        subpart_pattern = re.compile(r"^\(([ivx]+)\)\s*(.*)$")

        i = 0
        while i < len(blocks_by_page):
            block = blocks_by_page[i]
            text = block.text.strip()

            # Detect main question header (e.g. "1" or "1 Alex runs...")
            match = re.match(r"^(\d+)\s*(.*)$", text)
            if match and match.group(1).isdigit():
                q_num = int(match.group(1))
                # Only increment if it's the next sequential question number or the first one
                if q_num == current_q_num + 1 or current_q_num == 0:
                    if current_q_num > 0:
                        # Save previous part first
                        if current_part_label:
                            current_parts.append({
                                "part_label": current_part_label,
                                "part_text": cls.clean_text(" ".join(current_part_text)),
                                "marks": current_part_marks,
                                "image": None
                            })
                        questions.append({
                            "num": current_q_num,
                            "text": cls.clean_text(" ".join(current_q_text)),
                            "parts": current_parts.copy()
                        })
                    current_q_num = q_num
                    remaining_text = match.group(2).strip()
                    current_q_text = [remaining_text] if remaining_text else []
                    current_parts = []
                    current_part_label = ""
                    current_part_text = []
                    current_part_marks = 0
                    i += 1
                    continue

            # Detect sub-part header: e.g. (a) or (b)
            part_match = part_pattern.match(text)
            if part_match:
                # Save previous part first
                if current_part_label:
                    current_parts.append({
                        "part_label": current_part_label,
                        "part_text": cls.clean_text(" ".join(current_part_text)),
                        "marks": current_part_marks,
                        "image": None
                    })
                current_part_label = f"({part_match.group(1)})"
                remaining_text = part_match.group(2).strip()
                
                # Extract marks if present in this block
                marks_match = re.search(r"\[(\d+)\s*(?:marks|mark)?\]", remaining_text, re.IGNORECASE)
                if marks_match:
                    current_part_marks = int(marks_match.group(1))
                    remaining_text = remaining_text.replace(marks_match.group(0), "").strip()
                else:
                    current_part_marks = 0
                
                current_part_text = [remaining_text] if remaining_text else []
                i += 1
                continue

            # Check for standalone marks or marks embedded in general text
            marks_match = re.search(r"\[(\d+)\s*(?:marks|mark)?\]", text, re.IGNORECASE)
            if marks_match:
                val = int(marks_match.group(1))
                if current_part_label:
                    current_part_marks = val
                clean_text = text.replace(marks_match.group(0), "").strip()
                if not clean_text or clean_text in ["[", "]", "marks", "mark"]:
                    i += 1
                    continue
                else:
                    text = clean_text

            # Accumulate text
            if current_q_num > 0:
                if current_part_label:
                    current_part_text.append(text)
                else:
                    current_q_text.append(text)
            i += 1

        # Save last question
        if current_q_num > 0:
            if current_part_label:
                current_parts.append({
                    "part_label": current_part_label,
                    "part_text": cls.clean_text(" ".join(current_part_text)),
                    "marks": current_part_marks,
                    "image": None
                })
            questions.append({
                "num": current_q_num,
                "text": cls.clean_text(" ".join(current_q_text)),
                "parts": current_parts.copy()
            })

        # Format questions to schema
        formatted_questions = []
        for q in questions:
            q_num_str = f"{session.capitalize()}/{variant}/{year}/Q{q['num']}"
            total_marks = sum(p["marks"] for p in q["parts"]) if q["parts"] else 10 # Default fallback
            formatted_questions.append({
                "QuestionNumber": q_num_str,
                "Question_text": q["text"],
                "Unit": "PENDING",
                "Topic": "PENDING",
                "Topic_name": "PENDING",
                "Sub_topic": [],
                "Difficulty_score": 0,
                "Difficulty_label": "PENDING",
                "Command_words": ["explain"],
                "Marks": total_marks,
                "image": None,
                "Parts": q["parts"]
            })
        return formatted_questions

    @classmethod
    def parse_essay_ms(cls, pages: List[ExtractedPage], session: str, variant: str, year: int) -> List[Dict[str, Any]]:
        """Parses Essay-based Mark Scheme into structured answers matching the QP parts."""
        answers = []
        
        # We can implement a simplified rule-based parser that scans page text for question regions,
        # but because Essay Mark Schemes have high layout complexity (multiple lines of math, guidelines),
        # this is normally heavily augmented by Stage 5 Validator or Stage 2 LLM Agent if layout is complex.
        # Here we do a regex split to build candidate structures which can be completed.
        all_text = "\n".join([page.raw_text for page in pages])
        
        # Split text into major question blocks: "Question 1", "Question 2" or "Q1", "Q2" at start of line
        q_blocks = re.split(r"\n\s*(?:Question|Q)\s*(\d+)\b", all_text)
        
        if len(q_blocks) > 1:
            for idx in range(1, len(q_blocks), 2):
                q_num = int(q_blocks[idx])
                block_content = q_blocks[idx+1]
                
                # Split block content into parts (a), (b), (c)
                part_blocks = re.split(r"\n\s*\(([a-g])\)\s*", block_content)
                parts_list = []
                main_answer_text = ""
                
                if len(part_blocks) > 1:
                    main_answer_text = cls.clean_text(part_blocks[0])
                    for p_idx in range(1, len(part_blocks), 2):
                        part_letter = f"({part_blocks[p_idx]})"
                        part_content = cls.clean_text(part_blocks[p_idx+1])
                        parts_list.append({
                            "part_label": part_letter,
                            "part_answer": part_content,
                            "image": None
                        })
                else:
                    main_answer_text = cls.clean_text(block_content)

                ans_num_str = f"{session.capitalize()}/{variant}/{year}/Q{q_num}"
                answers.append({
                    "AnswerNumber": ans_num_str,
                    "Answer_text": main_answer_text,
                    "image": None,
                    "Parts": parts_list
                })
        else:
            # Fallback if split fails: check if pages have simple question items
            # Create a mock shell that Stage 5 Validator or LLM agent can populate
            for q_num in range(1, 10):
                ans_num_str = f"{session.capitalize()}/{variant}/{year}/Q{q_num}"
                answers.append({
                    "AnswerNumber": ans_num_str,
                    "Answer_text": "Detailed marking criteria can be found in original PDF.",
                    "image": None,
                    "Parts": []
                })

        return answers
