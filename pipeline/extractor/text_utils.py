"""
pipeline/extractor/text_utils.py

Shared text-processing utilities for mathematical notation normalisation and
answer-prompt stripping.  These helpers are used by both the layout extractor
(PyMuPDFExtractor) and the agentic extractor (AgenticExtractor).
"""

import re

from pipeline.visuals.extractor import SUPERSCRIPT_MAP

# Strip trailing answer-prompt strings like "change =", "height =", "V2 =",
# "water:", "Answer (a) ..." from question prose.
TRAILING_ANSWER_PROMPT = re.compile(
    r"(?:^|(?<=[.,;])\s*|\s+)"
    r"(?:"
        r"Answer\b(?!\s+(?:all|the|four|three|two|one|any|questions?|each|on)\b).*"
        r"|"
        r"(?:"
            r"(?:average\s+|total\s+|initial\s+|final\s+)?"
            r"[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*"
            r"(?:[a-zA-Z/°ΩΩ%–\^]+(?:\s*[0-9–\^]+)*(?:\s*/\s*[a-zA-Z0-9–\^]+)*)?"
        r")"
        r"|"
        r"[a-zA-Z0-9\s\-]{1,25}:\s*"
    r")\s*$",
    re.IGNORECASE,
)

def clean_answer_prompts(text: str) -> str:
    """Strip trailing answer-prompt strings from question prose.

    Removes trailing patterns like ``change =``, ``height =``, ``V2 =``,
    ``water:``, ``Answer (a) ...`` that Cambridge papers embed as fill-in
    response cues.
    """
    prev = None
    cur = text.strip()
    while prev != cur:
        prev = cur
        cur = TRAILING_ANSWER_PROMPT.sub("", cur).strip()
    # If the remaining text is purely a standalone answer prompt or variable
    # prompt (e.g. just "Answer" or "x ="), clear it entirely.
    if re.match(
        r"^(?:Answer\b.*|[a-zA-Z0-9\s\(\)/ˆ^–—_\-]{1,35}\s*=\s*|[a-zA-Z0-9\s\-]{1,25}:\s*)$",
        cur,
        re.IGNORECASE,
    ):
        cur = ""
    return cur


def clean_math_text(text: str) -> str:
    """Normalise mathematical notation, exponents, units, and symbols."""
    if not text:
        return ""

    # 1. Powers of 10 in scientific notation: × 1021 -> × 10²¹, 10-4 -> 10⁻⁴
    def _sup_10(m):
        prefix = m.group(1)
        p = m.group(2)
        sup = "".join(SUPERSCRIPT_MAP.get(c, c) for c in p)
        return f"{prefix}{sup}"

    text = re.sub(r"([×\*]\s*10)([–\-]?\d{1,4})\b", _sup_10, text)
    text = re.sub(
        r"\b(10)[–\-]([1-9]\d{0,3})\b",
        lambda m: f"{m.group(1)}⁻{''.join(SUPERSCRIPT_MAP.get(c, c) for c in m.group(2))}",
        text,
    )

    # 2. Units with powers: cm3 -> cm³, m2 -> m², km3 -> km³
    def _sup_unit(m):
        u = m.group(1)
        p = SUPERSCRIPT_MAP.get(m.group(2), m.group(2))
        return f"{u}{p}"

    text = re.sub(
        r"\b(cm|mm|km|dm|m|s|kg|g|ms|min|h|N|J|W|Pa|mol)([23])\b",
        _sup_unit,
        text,
    )

    # 3. Common algebraic powers: 4x2 -> 4x², 25y2 -> 25y², t2 -> t², n3 -> n³
    def _sup_var(m):
        prefix = m.group(1)
        var = m.group(2)
        p = SUPERSCRIPT_MAP.get(m.group(3), m.group(3))
        return f"{prefix}{var}{p}"

    text = re.sub(
        r"([0-9\(\)\s\+\-–=]|^)([a-zA-Z])([23])(?=[\s\+\-–=,\.\/\)]|$)",
        _sup_var,
        text,
    )

    # 4. Multiplication formatting: '5.98 * 10' or '5.98 x 10' -> '5.98 × 10'
    text = re.sub(r"(\d)\s*[xX]\s*(10[⁰¹²³⁴⁵⁶⁷⁸⁹\d])", r"\1 × \2", text)
    text = re.sub(r"(\d)\s*\*\s*(\d)", r"\1 × \2", text)

    # 5. Collapse spaces before superscripts: 't ²' -> 't²'
    text = re.sub(r"([a-zA-Z0-9])\s+([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ])", r"\1\2", text)

    # 6. Equivalence & congruency symbols normalisation
    text = text.replace(r"\equiv", "≡").replace("&equiv;", "≡")
    text = text.replace(r"\cong", "≅").replace("&cong;", "≅")

    # 7. Markdown pipe tables: ensure table rows joined by '| |' are separated by newlines
    if "|" in text and re.search(r"\|[-:\s|]{3,}\|", text):
        text = re.sub(r"\|\s*\|(?!=)", "|\n|", text)

    return text

