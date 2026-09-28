"""
Stage 3 — Unstructured.io API-Driven Visual Extractor

Extracts figures, diagrams, tables, and apparatus drawings from Cambridge exam PDFs
using the Unstructured.io Transform API, tags them with their question hierarchy
(question root, part, sub-part), renders high-resolution raster assets, and outputs
the canonical manifest for downstream patching and ingestion.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

import pymupdf as fitz
from unstructured_transform_client import TransformClient

from pipeline.config import settings

logger = logging.getLogger(__name__)

# Regex patterns for Cambridge question hierarchy tracking
Q_ROOT_RE = re.compile(
    r"^(?:(?:\*{0,2})Question\s+(\d{1,2})(?:\*{0,2})|(?:\*{0,2})(\d{1,2})(?:\*{0,2})(?:[\.\t\s]|$))",
    re.IGNORECASE,
)
PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([a-h])\)(?:\*{0,2})", re.IGNORECASE)
SUB_PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([ivx]+)\)(?:\*{0,2})", re.IGNORECASE)

FIG_CAP_RE = re.compile(r"(?:Fig\.|Figure)\s*(\d+)[\.\_](\d+)", re.IGNORECASE)
TABLE_CAP_RE = re.compile(r"Table\s*(\d+)[\.\_](\d+)", re.IGNORECASE)


class UnstructuredVisualExtractor:
    """
    Production Visual Extractor backed by the Unstructured.io Transform API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        server_url: str | None = None,
        dpi: int = 300,
        profile: str = "best",
        pad: float = 2.0,
        cache_dir: Path | None = None,
        extract_tables: bool = True,
    ):
        self.api_key = (
            api_key
            or os.environ.get("UNSTRUCTURED_API_KEY")
            or getattr(settings, "unstructured_api_key", None)
        )
        self.server_url = (
            server_url
            or os.environ.get("UNSTRUCTURED_SERVER_URL")
            or getattr(settings, "unstructured_server_url", "https://transform.unstructured.io")
        )
        self.dpi = dpi
        self.profile = profile
        self.pad = pad
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".unstructured_cache")
        self.extract_tables = extract_tables
        self._client: TransformClient | None = None

    def _get_client(self) -> TransformClient:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "UNSTRUCTURED_API_KEY is not set. Please provide it in .env, environment, "
                    "or pass api_key to UnstructuredVisualExtractor."
                )
            self._client = TransformClient(api_key=self.api_key, server_url=self.server_url)
        return self._client

    def _get_api_response(
        self,
        pdf_path: Path,
        output_dir: Path,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Fetch elements from Unstructured.io Transform API with local caching.
        """
        # Check cache in output_dir or global cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        local_cache = output_dir / "unstructured_response.json"
        global_cache = self.cache_dir / f"{pdf_path.stem}.json"

        cache_target = local_cache if local_cache.exists() else global_cache

        if not force_refresh and cache_target.exists():
            try:
                with open(cache_target, encoding="utf-8") as f:
                    data = json.load(f)
                elements = data.get("elements", [])
                # Ensure cached elements have valid coordinates for visual extraction
                has_coords = any(
                    (e.get("metadata") or {}).get("coordinates")
                    for e in elements
                    if e.get("type") in ("Image", "Table")
                )
                if elements and has_coords:
                    logger.info(f"Loaded {len(elements)} elements from cache '{cache_target.name}'")
                    return elements
            except Exception as e:
                logger.warning(f"Could not read cache {cache_target}: {e}. Calling API.")

        logger.info(f"Calling Unstructured.io API for '{pdf_path.name}' (profile={self.profile})...")
        client = self._get_client()

        with open(pdf_path, "rb") as f:
            result = client.parse.run(
                input=f,
                output="elements",
                profile=self.profile,
                include=["coordinates", "table_html"],
            )

        raw_elements: list[dict[str, Any]] = []
        if hasattr(result, "elements"):
            raw_elements = [
                e.model_dump() if hasattr(e, "model_dump") else (e if isinstance(e, dict) else vars(e))
                for e in result.elements
            ]
        elif isinstance(result, dict) and "elements" in result:
            raw_elements = result["elements"]

        # Cache response locally and in global cache
        cache_data = {
            "pdf_name": pdf_path.name,
            "status": str(getattr(result, "status", "completed")),
            "elements": raw_elements,
        }
        try:
            with open(global_cache, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            with open(local_cache, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached API response to '{global_cache}' and '{local_cache}'")
        except Exception as e:
            logger.warning(f"Failed writing cache: {e}")

        return raw_elements

    def _is_cover_or_admin_element(
        self,
        el: dict[str, Any],
        page_number: int,
    ) -> bool:
        """
        Check if an element is a cover page logo, header, or examiner administrative box.
        """
        if page_number == 1:
            return True

        text = (el.get("text") or "").lower()
        if any(term in text for term in ("for examiner's use", "examiner's use", "candidate number", "centre number")):
            return True
        if "cambridge" in text and any(term in text for term in ("assessment", "o level", "international education")):
            return True

        return False

    def _is_mcq_paper(self, pdf_path: Path, force_mcq: bool | None = None) -> bool:
        if force_mcq is not None:
            return force_mcq
        stem = pdf_path.stem.lower()
        # Typical Cambridge paper 1 variant (e.g. 5090_w19_qp_11 or _12 or _13)
        return bool(re.search(r"_qp_1[1-3]", stem) or re.search(r"_1[1-3]$", stem))

    def _is_blank_image(self, pix: fitz.Pixmap) -> bool:
        """Return True if image is uniformly blank/white."""
        samples = pix.samples
        if not samples:
            return True
        # Sample every 10th byte for efficiency
        return all(b >= 250 for b in samples[::10])

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        force_mcq: bool | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Extract visual figures and tables from exam PDF.
        Returns a list of manifest entry dicts.
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        is_mcq = self._is_mcq_paper(pdf_path, force_mcq)

        # 1. Fetch elements from Unstructured API / cache
        elements = self._get_api_response(pdf_path, output_dir, force_refresh=force_refresh)
        if not elements:
            logger.warning(f"No elements returned from Unstructured.io for {pdf_path.name}")
            return []

        # 2. Open PDF for rendering
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)

        manifest: list[dict[str, Any]] = []

        # Active question tracking state
        current_q: int | None = None
        current_part: str | None = None
        current_sub_part: str | None = None

        allowed_types = {"Image", "Table"} if self.extract_tables else {"Image"}

        for idx, el in enumerate(elements):
            etype = el.get("type")
            text = (el.get("text") or "").strip()
            meta = el.get("metadata") or {}
            page_num = meta.get("page_number", 0)

            # Context tracking from Title / NarrativeText elements (skip page 1)
            if etype in ("Title", "NarrativeText") and page_num > 1:
                qm = Q_ROOT_RE.match(text)
                if qm:
                    q_cand = int(qm.group(1) or qm.group(2))
                    if q_cand != current_q:
                        current_q = q_cand
                        current_part = None
                        current_sub_part = None

                pm = PART_RE.findall(text)
                sm = SUB_PART_RE.findall(text)
                if pm:
                    current_part = pm[-1].lower()
                    current_sub_part = None
                if sm:
                    current_sub_part = sm[-1].lower()

            # Process visual elements
            if etype in allowed_types:
                if self._is_cover_or_admin_element(el, page_num):
                    continue

                if not (1 <= page_num <= num_pages):
                    continue

                coords = meta.get("coordinates") or {}
                pts = coords.get("points") or []
                lw = coords.get("layout_width")
                lh = coords.get("layout_height")

                if not pts or not lw or not lh:
                    continue

                page = doc[page_num - 1]
                sx = page.rect.width / float(lw)
                sy = page.rect.height / float(lh)

                xs = [p[0] * sx for p in pts]
                ys = [p[1] * sy for p in pts]
                raw_bbox = fitz.Rect(min(xs), min(ys), max(xs), max(ys))

                if raw_bbox.width < 15 or raw_bbox.height < 15:
                    continue

                # Add light padding clamped to page boundary
                clip = (raw_bbox + (-self.pad, -self.pad, self.pad, self.pad)) & page.rect

                # Target determination
                if is_mcq:
                    target_type = "question"
                    target_label = f"Q{current_q}" if current_q else "Q1"
                    tag_str = "question"
                    part_label = None
                else:
                    if current_part and current_sub_part:
                        target_type = "sub_part"
                        target_label = f"{current_part}({current_sub_part})"
                        tag_str = f"sub_part_{target_label}"
                        part_label = target_label
                    elif current_part:
                        target_type = "part"
                        target_label = current_part
                        tag_str = f"part_{current_part}"
                        part_label = current_part
                    else:
                        target_type = "question"
                        target_label = f"Q{current_q}" if current_q else "Q1"
                        tag_str = "question"
                        part_label = None

                # Determine caption & visual suffix
                caption: str | None = None
                q_for_name = current_q if current_q is not None else 1

                # Check OCR text or adjacent element for Fig / Table caption
                surrounding_text = text
                if idx > 0 and elements[idx - 1].get("metadata", {}).get("page_number") == page_num:
                    surrounding_text = (elements[idx - 1].get("text") or "") + " " + surrounding_text
                if idx + 1 < len(elements) and elements[idx + 1].get("metadata", {}).get("page_number") == page_num:
                    surrounding_text = surrounding_text + " " + (elements[idx + 1].get("text") or "")

                if is_mcq:
                    sub_idx = len([m for m in manifest if m.get("question") == q_for_name]) + 1
                    fname = f"p{page_num}_q{q_for_name}_question_mcq_{sub_idx}.png"
                elif etype == "Image":
                    fm = FIG_CAP_RE.search(surrounding_text)
                    if fm:
                        suffix = f"Fig_{fm.group(1)}_{fm.group(2)}"
                        caption = f"Fig. {fm.group(1)}.{fm.group(2)}"
                    else:
                        sub_cnt = (
                            len(
                                [
                                    m
                                    for m in manifest
                                    if m.get("question") == q_for_name
                                    and "diagram" in m.get("filename", "").lower()
                                ]
                            )
                            + 1
                        )
                        suffix = f"diagram_{sub_cnt}"
                    fname = f"p{page_num}_q{q_for_name}_{tag_str}_{suffix}.png"
                else:  # Table
                    tm = TABLE_CAP_RE.search(surrounding_text)
                    if tm:
                        suffix = f"Table_{tm.group(1)}_{tm.group(2)}"
                        caption = f"Table {tm.group(1)}.{tm.group(2)}"
                    else:
                        sub_cnt = (
                            len(
                                [
                                    m
                                    for m in manifest
                                    if m.get("question") == q_for_name
                                    and "table" in m.get("filename", "").lower()
                                ]
                            )
                            + 1
                        )
                        suffix = f"Table_{sub_cnt}"
                        caption = "Table"
                    fname = f"p{page_num}_q{q_for_name}_{tag_str}_{suffix}.png"

                # Deduplicate filename if already present
                if any(m["filename"] == fname for m in manifest):
                    dup_idx = sum(1 for m in manifest if fname.split(".png")[0] in m["filename"]) + 1
                    base = fname.rsplit(".", 1)[0]
                    fname = f"{base}_{dup_idx}.png"

                # Render crop at high resolution
                pix = page.get_pixmap(clip=clip, dpi=self.dpi)
                if self._is_blank_image(pix):
                    continue

                save_path = output_dir / fname
                pix.save(str(save_path))

                entry = {
                    "page": page_num,
                    "question": q_for_name,
                    "target_type": target_type,
                    "target_label": target_label,
                    "part": part_label,
                    "caption": caption,
                    "bbox": [round(v, 1) for v in clip],
                    "filename": fname,
                    "file": fname,
                    "relative_path": f"{output_dir.name}/{fname}",
                    "element_id": el.get("element_id"),
                    "element_type": etype,
                }
                manifest.append(entry)
                logger.info(f"  Extracted ({etype.lower()}): {fname} (Page {page_num}, Q{q_for_name})")

        doc.close()

        # Save manifest.json in output_dir
        manifest_path = output_dir / "manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save manifest.json: {e}")

        logger.info(
            f"UnstructuredVisualExtractor: successfully extracted {len(manifest)} visuals from {pdf_path.name}"
        )
        return manifest
