"""
Stage 3 — Gemini Visual Grounding Extractor

Extracts figures, diagrams, tables, graphs, circuits, and apparatus from Cambridge exam PDFs
using Google Gemini's native 2D visual grounding and object detection API, with question,
part, and sub-part attribution, and renders 300 DPI pixel-perfect crops using PyMuPDF.
"""
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pymupdf as fitz
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from pipeline.config import settings
from pipeline.visuals.extractor import CambridgeVisualExtractor

logger = logging.getLogger(__name__)


# ── Pydantic Schemas for Gemini Visual Grounding ─────────────────────────────

class VisualBoundingBox(BaseModel):
    question: int = Field(
        description="The integer question number this visual belongs to (e.g. 1, 2, 7, 24)."
    )
    target_type: Literal["question", "part", "sub_part"] = Field(
        description=(
            "Hierarchical attribution: 'question' if it belongs to the main stem/preamble or the whole question; "
            "'part' if it belongs to a specific part (e.g. (a), (b)); "
            "'sub_part' if it belongs to a specific nested sub-part (e.g. (i), (ii))."
        )
    )
    target_label: str = Field(
        description=(
            "Exact identifier: 'Q1' for question stem, 'a', 'b', 'c' for part, '(i)', '(ii)', '(iii)' for sub-part."
        )
    )
    caption: Optional[str] = Field(
        default=None,
        description="Figure or table caption if explicitly printed (e.g. 'Fig. 1.1', 'Table 2.1'), or null if uncaptioned."
    )
    visual_type: Literal[
        "diagram", "graph", "table", "circuit", "apparatus", "geometry", "other"
    ] = Field(
        default="diagram",
        description="Type of visual element: diagram, graph, table, circuit, apparatus, geometry, or other."
    )
    box_2d: List[int] = Field(
        description="[ymin, xmin, ymax, xmax] 2D bounding box coordinates normalized to 0-1000 integer scale."
    )


class PageVisualsContainer(BaseModel):
    visuals: List[VisualBoundingBox] = Field(
        default_factory=list,
        description="List of all detected visual elements on this exam page."
    )


# ── Gemini Visual Extractor ──────────────────────────────────────────────────

class GeminiVisualExtractor:
    """
    Visual extraction agent powered by Google Gemini multimodal vision.
    Detects 2D bounding boxes and question/part/sub-part attribution, and uses
    PyMuPDF to crop high-resolution (300 DPI) images from the source PDF.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dpi: int = 300,
        pad: float = 6.0,
    ):
        self.model_name = model_name or getattr(settings, "visuals_model", "gemini-2.5-flash")
        self.dpi = dpi
        self.pad = pad
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for GeminiVisualExtractor.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _normalize_target(target_type: str, target_label: str, question: int) -> Tuple[str, str]:
        """Normalize target_type and target_label to ensure consistency with Stage 2 and Stage 3 schemas."""
        tt = str(target_type).strip().lower()
        tl = str(target_label).strip()

        if tt == "question":
            return "question", f"Q{question}"
        elif tt == "part":
            clean = re.sub(r"[^a-zA-Z]", "", tl).lower()
            return "part", clean or tl.lower()
        elif tt == "sub_part":
            # Normalize to (i), (ii), (iii), etc.
            m = re.search(r"\(?([ivx]+|[1-9])\)?", tl, re.IGNORECASE)
            if m:
                return "sub_part", f"({m.group(1).lower()})"
            return "sub_part", tl
        return "question", f"Q{question}"

    def _build_page_prompt(self, pno: int, total_pages: int) -> str:
        return f"""You are an expert Cambridge examination visual layout and grounding analyst.
Examine this Cambridge examination past paper page image (Page {pno + 1} of {total_pages}).

TASK:
Detect ALL visual elements appearing on this page that are required for students to understand and answer questions.
This includes:
1. Scientific and technical diagrams (apparatus setups, biological specimens, cross-sections, physics demonstrations).
2. Graphs, charts, and coordinate grids (with all axes, tick numbers, units, scales, and curves).
3. Electrical and electronic circuit diagrams.
4. Geometric figures, angle diagrams, compass bearings, 2D/3D shapes, and maps.
5. Tables of data, measurements, or properties (BOTH captioned tables like 'Table 1.1' and uncaptioned data grids).

CRITICAL REQUIREMENT — EXACT QUESTION, PART, OR SUB-PART ATTRIBUTION:
For EVERY detected visual element, you MUST determine precisely which question component it belongs to:
- "question": integer question number (e.g. 1, 2, 7, 24). Look at the nearest question number prefix or context.
- "target_type": Exactly one of:
  * "question": if the diagram illustrates the main question stem / preamble (before any parts), or if the whole question shares this diagram.
  * "part": if the diagram belongs specifically to a part (e.g. part (a), (b), (c)).
  * "sub_part": if the diagram belongs specifically to a nested sub-part (e.g. part (a)(i), (ii), (iii)).
- "target_label":
  * If target_type is "question": set to "Q{pno + 1}" style (e.g. "Q1", "Q7").
  * If target_type is "part": set to lowercase letter (e.g. "a", "b", "c").
  * If target_type is "sub_part": set strictly to roman numerals in parentheses, e.g. "(i)", "(ii)", "(iii)".
- "caption": exact caption text if explicitly printed (e.g. "Fig. 1.1", "Table 2.1"), or null if uncaptioned.
- "visual_type": "diagram", "graph", "table", "circuit", "apparatus", "geometry", or "other".
- "box_2d": [ymin, xmin, ymax, xmax] coordinates normalized to 0-1000 integer range.

BOUNDING BOX PRECISION RULES:
1. The bounding box MUST enclose the ENTIRE visual: artwork, lines, tick marks, axis labels, units, annotations, and caption.
2. Do NOT cut off arrows, leaders, numbers, or component labels.
3. Do NOT include question text prose above or below the figure.
4. Do NOT include dotted student answer lines (e.g. "..................."), mark allocations [2], or blanks ("height = ..........").
5. DO NOT extract exam headers, footers, page numbers, barcodes, or "DO NOT WRITE IN THIS MARGIN" borders.
6. If there are NO visual elements on this page (pure text or cover page), return an empty list for "visuals".
"""

    def _detect_visuals_for_page(
        self,
        page: fitz.Page,
        pno: int,
        total_pages: int,
    ) -> List[VisualBoundingBox]:
        """Send 150 DPI page rendering to Gemini and return detected visual bounding boxes."""
        client = self._get_client()
        prompt = self._build_page_prompt(pno, total_pages)

        # Render page at 150 DPI for VLM inspection
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

        last_err = None
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageVisualsContainer,
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                raw_text = response.text
                if not raw_text or not raw_text.strip():
                    return []

                data = json.loads(raw_text)
                container = PageVisualsContainer.model_validate(data)
                return container.visuals
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "resource_exhausted" in err_str or "429" in err_str:
                    logger.warning(f"Gemini quota exhausted on page {pno + 1}: {e}")
                    raise e

                is_transient = any(
                    kw in err_str
                    for kw in [
                        "ssl", "eof", "connection", "connecterror",
                        "timeout", "timed out", "remotedisconnected",
                        "reset", "503", "500",
                    ]
                )
                if is_transient and attempt < 3:
                    logger.warning(
                        f"Transient Gemini error on page {pno + 1} (attempt {attempt}/3): {e}. Retrying in {attempt * 2}s..."
                    )
                    time.sleep(attempt * 2)
                else:
                    raise e

        if last_err:
            raise last_err
        return []

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        force_mcq: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract all visual elements from PDF using Gemini visual grounding,
        with fallback to CambridgeVisualExtractor if Gemini fails or quota is exhausted.
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"GeminiVisualExtractor: analyzing {pdf_path.name} with model {self.model_name}...")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            manifest: List[Dict[str, Any]] = []

            for pno in range(total_pages):
                # Skip cover page (pno == 0) unless single-page document
                if pno == 0 and total_pages > 1:
                    continue

                page = doc[pno]
                page_w, page_h = page.rect.width, page.rect.height

                logger.info(f"  [Gemini] Grounding visuals on Page {pno + 1}/{total_pages}...")
                boxes = self._detect_visuals_for_page(page, pno, total_pages)
                if not boxes:
                    continue

                page_count = 0
                for v in boxes:
                    if not v.box_2d or len(v.box_2d) != 4:
                        continue

                    ymin, xmin, ymax, xmax = v.box_2d
                    # Ensure normalized coordinates are valid
                    if ymax <= ymin or xmax <= xmin:
                        continue

                    # Convert 0-1000 scale to PDF points
                    x0 = (xmin / 1000.0) * page_w
                    y0 = (ymin / 1000.0) * page_h
                    x1 = (xmax / 1000.0) * page_w
                    y1 = (ymax / 1000.0) * page_h

                    # Expand slightly with pad to prevent clipping labels/axes
                    clip = (fitz.Rect(x0 - self.pad, y0 - self.pad, x1 + self.pad, y1 + self.pad)) & page.rect
                    if clip.width < 20 or clip.height < 20:
                        continue

                    # Render high-res crop
                    pix = page.get_pixmap(clip=clip, dpi=self.dpi)
                    if CambridgeVisualExtractor._is_blank_image(pix):
                        continue

                    # Normalize target type and label
                    target_type, target_label = self._normalize_target(
                        v.target_type, v.target_label, v.question
                    )

                    if target_type == "question":
                        tag_str = "question"
                    elif target_type == "part":
                        tag_str = f"part_{target_label}"
                    else:
                        tag_str = f"sub_part_{target_label}"

                    page_count += 1
                    cap_tag = (v.caption or v.visual_type).replace(" ", "_").replace(".", "_")
                    cap_slug = re.sub(r"[^a-zA-Z0-9_]", "", cap_tag) or "fig"
                    fname = f"p{pno + 1}_q{v.question}_{tag_str}_{cap_slug}_{page_count}.png"

                    save_path = output_dir / fname
                    pix.save(str(save_path))

                    entry = {
                        "page": pno + 1,
                        "question": v.question,
                        "target_type": target_type,
                        "target_label": target_label,
                        "part": target_label if target_type in ("part", "sub_part") else None,
                        "caption": v.caption,
                        "visual_type": v.visual_type,
                        "bbox": [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                        "filename": fname,
                        "file": fname,
                        "relative_path": f"{output_dir.name}/{fname}",
                    }
                    manifest.append(entry)
                    logger.info(
                        f"    Extracted: {fname} (Page {pno + 1}, Q{v.question}, target={tag_str})"
                    )

            doc.close()

            # Save manifest
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            logger.info(
                f"GeminiVisualExtractor: successfully extracted {len(manifest)} visuals "
                f"into {output_dir.name}/ (manifest: {manifest_path.name})"
            )
            return manifest

        except Exception as e:
            logger.warning(
                f"GeminiVisualExtractor encountered error ({e}). "
                f"Falling back to CambridgeVisualExtractor heuristic engine..."
            )
            fallback = CambridgeVisualExtractor(dpi=self.dpi)
            return fallback.extract(pdf_path, output_dir, force_mcq=force_mcq)
