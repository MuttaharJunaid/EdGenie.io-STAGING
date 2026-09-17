"""
pipeline/extractor/marks_validator.py

Validates and reconciles question, part, and sub-part mark allocations.
Enforces Cambridge exam hierarchy rules:
1. Leaf sub-parts/parts have marks >= 1, parsed from explicit [N] notation.
2. Parent part marks strictly equal the sum of child sub-part marks.
3. Total question marks strictly equal the sum of constituent part marks.
4. Trailing mark indicators [N] and [Total: N] are stripped from prose text.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches [1], [2], [10], etc. at end or within text
MARK_BRACKET_RE = re.compile(r"\[\s*(\d{1,2})\s*\]")
# Matches [Total: 10] or [Total 8] or (Total: 10 marks)
TOTAL_MARK_RE = re.compile(
    r"\[?\s*(?:Total\s*:?|Total\s+marks\s*:?)\s*(\d{1,3})\s*(?:marks?)?\s*\]?",
    re.IGNORECASE,
)


class MarkHierarchyValidator:
    """Validates and reconciles marks across questions, parts, and sub-parts."""

    @classmethod
    def clean_text_and_extract_leaf_mark(
        cls,
        text: str,
        current_mark: int,
    ) -> tuple[str, int]:
        """
        If text contains trailing bracketed mark like [2], extract 2 into mark
        and strip [2] / [Total: N] from the text.
        """
        if not text:
            return "", max(current_mark, 1)

        cleaned_text = text.strip()

        # Remove [Total: N] if present anywhere in text
        cleaned_text = TOTAL_MARK_RE.sub("", cleaned_text).strip()

        # Check for trailing [N]
        m = MARK_BRACKET_RE.search(cleaned_text)
        extracted_mark = current_mark
        if m:
            # Check if this [N] is at or near the end of the text
            end_match = list(MARK_BRACKET_RE.finditer(cleaned_text))
            if end_match:
                last_m = end_match[-1]
                # If the bracket is in the last 20 chars of the text
                if last_m.end() >= len(cleaned_text) - 15:
                    try:
                        extracted_mark = int(last_m.group(1))
                    except (ValueError, TypeError):
                        pass
                    # Strip the trailing [N]
                    cleaned_text = (
                        cleaned_text[: last_m.start()] + cleaned_text[last_m.end() :]
                    ).strip()

        extracted_mark = max(extracted_mark, 1)
        return cleaned_text, extracted_mark

    @classmethod
    def reconcile_question_marks(
        cls,
        q_record: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Reconciles marks for a single question record:
        - Reconciles sub-parts -> part marks
        - Reconciles parts -> question Marks
        - Cleans mark brackets from text
        Returns the updated record and a list of audit messages.
        """
        audit_logs: list[str] = []
        q_num = q_record.get("QuestionNumber", "Unknown")

        # Strip any [Total: N] from Question_text
        q_text = q_record.get("Question_text", "") or ""
        total_m = TOTAL_MARK_RE.search(q_text)
        expected_total_from_text = None
        if total_m:
            try:
                expected_total_from_text = int(total_m.group(1))
            except (ValueError, TypeError):
                pass
            q_text = TOTAL_MARK_RE.sub("", q_text).strip()
            q_record["Question_text"] = q_text

        parts = q_record.get("Parts", [])
        if not parts:
            # Single-part question without parts list
            cleaned_q_text, leaf_mark = cls.clean_text_and_extract_leaf_mark(
                q_text,
                q_record.get("Marks", 1),
            )
            q_record["Question_text"] = cleaned_q_text
            original_marks = q_record.get("Marks", 1)
            final_mark = leaf_mark if leaf_mark > 0 else max(original_marks, 1)
            if expected_total_from_text and expected_total_from_text > 0:
                final_mark = expected_total_from_text
            q_record["Marks"] = final_mark
            return q_record, audit_logs

        calculated_q_marks = 0

        for part in parts:
            p_label = part.get("part_label", "")
            sub_parts = part.get("sub_parts", [])

            if sub_parts:
                # Part has child sub-parts (e.g. (i), (ii))
                # Clean part_text
                p_text = part.get("part_text", "") or ""
                cleaned_p_text, _ = cls.clean_text_and_extract_leaf_mark(p_text, 1)
                part["part_text"] = cleaned_p_text

                calculated_part_marks = 0
                for sp in sub_parts:
                    sp_label = sp.get("sub_part_label", "")
                    sub_sub_parts = sp.get("sub_parts", [])

                    if sub_sub_parts:
                        # Deeply nested sub-sub-parts
                        cleaned_sp_text, _ = cls.clean_text_and_extract_leaf_mark(
                            sp.get("sub_part_text", "") or "", 1
                        )
                        sp["sub_part_text"] = cleaned_sp_text

                        calculated_sp_marks = 0
                        for ssp in sub_sub_parts:
                            cleaned_ssp_text, ssp_mark = cls.clean_text_and_extract_leaf_mark(
                                ssp.get("sub_part_text", "") or "",
                                ssp.get("marks", 1),
                            )
                            ssp["sub_part_text"] = cleaned_ssp_text
                            ssp["marks"] = max(ssp_mark, 1)
                            calculated_sp_marks += ssp["marks"]

                        if sp.get("marks") != calculated_sp_marks:
                            audit_logs.append(
                                f"{q_num} part ({p_label}) sub-part {sp_label}: "
                                f"updated marks from {sp.get('marks')} to sum of sub-sub-parts ({calculated_sp_marks})"
                            )
                        sp["marks"] = calculated_sp_marks
                    else:
                        # Leaf sub-part
                        cleaned_sp_text, sp_mark = cls.clean_text_and_extract_leaf_mark(
                            sp.get("sub_part_text", "") or "",
                            sp.get("marks", 1),
                        )
                        sp["sub_part_text"] = cleaned_sp_text
                        sp["marks"] = max(sp_mark, 1)

                    calculated_part_marks += sp["marks"]

                # Enforce part.marks = sum of sub_parts
                orig_p_marks = part.get("marks")
                if orig_p_marks != calculated_part_marks:
                    audit_logs.append(
                        f"{q_num} part ({p_label}): updated marks from {orig_p_marks} to sum of sub-parts ({calculated_part_marks})"
                    )
                part["marks"] = calculated_part_marks

            else:
                # Leaf part without sub-parts
                cleaned_p_text, p_mark = cls.clean_text_and_extract_leaf_mark(
                    part.get("part_text", "") or "",
                    part.get("marks", 1),
                )
                part["part_text"] = cleaned_p_text
                part["marks"] = max(p_mark, 1)

            calculated_q_marks += part["marks"]

        # Enforce question.Marks = sum of parts
        orig_q_marks = q_record.get("Marks")
        if orig_q_marks != calculated_q_marks:
            audit_logs.append(
                f"{q_num}: updated total question Marks from {orig_q_marks} to sum of parts ({calculated_q_marks})"
            )
        q_record["Marks"] = calculated_q_marks

        if expected_total_from_text and expected_total_from_text != calculated_q_marks:
            logger.debug(
                f"{q_num}: parsed total {calculated_q_marks} differs from explicit [Total: {expected_total_from_text}]"
            )

        return q_record, audit_logs

    @classmethod
    def reconcile_and_validate_records(
        cls,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Reconciles and validates mark consistency across all extracted records in a paper.
        Returns the updated records list and a stats dictionary.
        """
        all_audit_logs: list[str] = []
        total_paper_marks = 0
        questions_count = len(records)
        discrepancies_count = 0

        reconciled_records = []
        for record in records:
            # Only essay QP papers have hierarchical parts/subparts
            if "QuestionNumber" in record and ("Parts" in record or "Marks" in record):
                reconciled, logs = cls.reconcile_question_marks(record)
                if logs:
                    discrepancies_count += len(logs)
                    all_audit_logs.extend(logs)
                total_paper_marks += reconciled.get("Marks", 1)
                reconciled_records.append(reconciled)
            elif "QuestionNumber" in record and "Options" in record:
                # MCQ question paper: strictly 1 mark per question
                record["Marks"] = 1
                total_paper_marks += 1
                reconciled_records.append(record)
            else:
                # Mark schemes or other records
                reconciled_records.append(record)

        stats = {
            "questions_count": questions_count,
            "total_paper_marks": total_paper_marks,
            "discrepancies_fixed": discrepancies_count,
            "audit_logs": all_audit_logs,
        }

        if discrepancies_count > 0:
            logger.info(
                f"Marks validation reconciled {discrepancies_count} mark hierarchy discrepancy/discrepancies across {questions_count} question(s). Total Paper Marks: {total_paper_marks}"
            )
            for log_entry in all_audit_logs[:10]:
                logger.debug(f"  [Marks Audit] {log_entry}")

        return reconciled_records, stats
