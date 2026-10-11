"""
pipeline/unified/pipeline.py

Unified 3-in-1 Pipeline: Text Extraction + Visual Extraction + Syllabus Classification
via a single Unstructured.io Transform API call (client.parse.run with schema).

The document is parsed ONCE. From that single job:
  - result.extracted_data  →  structured question JSON (text + classification)
  - result.elements        →  layout elements with bounding boxes for visual snipping

PyMuPDF then handles local rendering/cropping from those bounding boxes.
No duplicate API charges. No separate classification pass.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pymupdf as fitz
from unstructured_transform_client import TransformClient

from pipeline.config import settings
from pipeline.unified.marks_validator import MarkHierarchyValidator
from pipeline.unified.paper_type_detector import PaperCategory
from pipeline.unified.schemas import (
    EssayAnswersContainer,
    EssayQuestionsContainer,
    MCQAnswersContainer,
    MCQQuestionsContainer,
    to_strict_openai_schema,
)
from pipeline.unified.syllabus_loader import SyllabusLoader
from pipeline.unified.manifest_patcher import patch_records_with_manifest


logger = logging.getLogger(__name__)

# ── Regex patterns (same as UnstructuredVisualExtractor) ─────────────────────
Q_ROOT_RE = re.compile(
    r"^(?:(?:\*{0,2})Question\s+(\d{1,2})(?:\*{0,2})|(?:\*{0,2})(\d{1,2})(?:\*{0,2})(?:[\.\t\s]|$))",
    re.IGNORECASE,
)
PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([a-h])\)(?:\*{0,2})", re.IGNORECASE)
SUB_PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([ivx]+)\)(?:\*{0,2})", re.IGNORECASE)
FIG_CAP_RE = re.compile(r"(?:Fig\.|Figure)\s*(\d+)[\.\_](\d+)", re.IGNORECASE)
TABLE_CAP_RE = re.compile(r"Table\s*(\d+)[\.\_](\d+)", re.IGNORECASE)


class UnifiedPipeline:
    """
    Runs text extraction, visual extraction, and syllabus classification in a
    single Unstructured.io Transform API job (client.parse.run with schema).

    Single credit charge. Single network round-trip.
    """

    def __init__(
        self,
        api_key: str | None = None,
        server_url: str | None = None,
        profile: str = "best",
        dpi: int = 300,
        pad: float = 2.0,
        timeout_seconds: float = 600.0,
        poll_interval: float = 5.0,
        max_poll_time: float = 600.0,
        extract_tables: bool = True,
    ) -> None:
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
        self.profile = profile
        self.dpi = dpi
        self.pad = pad
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.extract_tables = extract_tables
        self._client: TransformClient | None = None

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self) -> TransformClient:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "UNSTRUCTURED_API_KEY is not set. "
                    "Please provide it in .env or the environment."
                )
            self._client = TransformClient(
                api_key=self.api_key,
                server_url=self.server_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._client

    # ── PDF Pruning (blank page removal) ─────────────────────────────────────

    @staticmethod
    def _prune_pdf_bytes(pdf_path: Path) -> bytes | None:
        """Strip blank / BLANK PAGE pages before upload to cut cloud processing time."""
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) <= 2:
                return pdf_path.read_bytes()

            pruned = fitz.open()
            pruned_count = 0
            for pno in range(len(doc)):
                page = doc[pno]
                text = page.get_text().strip()
                if pno > 0:
                    has_blank = bool(re.search(r"\bBLANK\s+PAGE\b", text, re.IGNORECASE))
                    has_marks = bool(re.search(r"\[\s*\d+\s*\]", text))
                    has_q = bool(re.search(
                        r"\b(?:Question\s+\d+|^\d+\s+[A-Z])\b",
                        text, re.IGNORECASE | re.MULTILINE,
                    ))
                    if (not text and not page.get_images() and not page.get_drawings()) \
                            or (has_blank and not has_marks and not has_q):
                        pruned_count += 1
                        continue
                pruned.insert_pdf(doc, from_page=pno, to_page=pno)

            if pruned_count > 0 and len(pruned) > 0:
                logger.info(
                    "Pruned %d blank page(s) from %s (%d/%d pages submitted)",
                    pruned_count, pdf_path.name, len(pruned), len(doc),
                )
                return pruned.tobytes()
        except Exception as exc:
            logger.debug("PDF pruning bypassed for %s: %s", pdf_path.name, exc)
        return None

    # ── Schema + Prompt Builder ───────────────────────────────────────────────

    def _build_schema_and_prompt(
        self,
        category: PaperCategory,
        meta: dict[str, Any],
        syllabus_text: str,
    ) -> tuple[dict[str, Any], str]:
        """
        Return (strict_json_schema, prompt_string) for the given paper category.
        The syllabus_text is embedded in the prompt so classification is done
        inline by the extraction agent — no separate classifier call.
        """
        year = meta.get("year", "2026")
        session_str = meta.get("session", "s")
        session = "May/June" if session_str in ("s", "jun", "june") else "Oct/Nov"
        variant = meta.get("variant", "11")

        syllabus_block = (
            f"\n\nSYLLABUS REFERENCE (use ONLY these units/topics for classification):\n"
            f"{syllabus_text}\n"
            if syllabus_text
            else ""
        )

        if category == PaperCategory.ESSAY_QP:
            schema = to_strict_openai_schema(EssayQuestionsContainer)
            prompt = (
                f"You are extracting and classifying questions from a Cambridge O-Level "
                f"Essay Question Paper ({session} {year}, variant {variant}).\n\n"
                f"EXTRACTION RULES:\n"
                f"1. Set QuestionNumber as '{session}/{variant}/{year}/Q<N>' "
                f"(e.g. '{session}/{variant}/{year}/Q1').\n"
                f"2. Set Question_text to the full question stem preserving newlines.\n"
                f"3. Do NOT transcribe text labels inside diagrams into Question_text.\n"
                f"4. Nest parts (a), (b) etc. into Parts. Nest sub-parts (i), (ii) into sub_parts.\n"
                f"5. Set Marks to the total marks for the question.\n\n"
                f"CLASSIFICATION RULES:\n"
                f"6. Set Unit to the unit number and name from the syllabus (e.g. '1' or '1 - Cell Biology').\n"
                f"7. Set Topic to the topic number (e.g. '1.1').\n"
                f"8. Set Topic_name to the topic name string.\n"
                f"9. Set Sub_topic to a list of matching sub-topic strings from the syllabus.\n"
                f"10. Set Difficulty_score (1–10) and Difficulty_label "
                f"(Easy/Medium/Hard) based on complexity.\n"
                f"11. Set Command_words to the list of Cambridge command words used "
                f"(e.g. ['Describe', 'Explain', 'Calculate']).\n"
                f"12. Unit, Topic, Topic_name, Sub_topic MUST match the syllabus reference "
                f"below exactly — do not invent topics."
                f"{syllabus_block}"
            )

        elif category == PaperCategory.ESSAY_MS:
            schema = to_strict_openai_schema(EssayAnswersContainer)
            prompt = (
                f"You are extracting mark scheme answers from a Cambridge O-Level "
                f"Mark Scheme ({session} {year}, variant {variant}).\n\n"
                f"RULES:\n"
                f"1. Set AnswerNumber as '{session}/{variant}/{year}/Q<N>'.\n"
                f"2. Set Parts with each part_label and part_answer including mark allocations [N].\n"
                f"3. Answer_text is the overarching answer if there are no parts."
                f"{syllabus_block}"
            )

        elif category == PaperCategory.MCQ_QP:
            schema = to_strict_openai_schema(MCQQuestionsContainer)
            prompt = (
                f"You are extracting and classifying MCQ questions from a Cambridge O-Level "
                f"MCQ Paper ({session} {year}, variant {variant}).\n\n"
                f"EXTRACTION RULES:\n"
                f"1. Set QuestionNumber as '{session}/{variant}/{year}/Q<N>'.\n"
                f"2. Set Question_text to the question stem.\n"
                f"3. Put options A, B, C, D into the Options object.\n"
                f"4. Marks is always 1.\n\n"
                f"CLASSIFICATION RULES:\n"
                f"5. Set Unit, Topic, Topic_name, Sub_topic from the syllabus below.\n"
                f"6. Set Difficulty_score (1–10), Difficulty_label, and Command_words."
                f"{syllabus_block}"
            )

        elif category == PaperCategory.MCQ_MS:
            schema = to_strict_openai_schema(MCQAnswersContainer)
            prompt = (
                f"You are extracting the answer key from a Cambridge O-Level MCQ Mark Scheme "
                f"({session} {year}, variant {variant}).\n\n"
                f"RULES:\n"
                f"1. Set AnswerNumber as '{session}/{variant}/{year}/Q<N>'.\n"
                f"2. Set Answer to the correct letter (A, B, C, or D)."
            )

        else:
            raise ValueError(f"Unsupported paper category: {category}")

        return schema, prompt

    # ── API Job Submission & Polling ──────────────────────────────────────────

    def _submit_and_poll(
        self,
        file_input: Any,
        schema: dict[str, Any],
        prompt: str,
        pdf_name: str,
    ) -> Any:
        """
        Submit via client.parse.run (with schema) so parse + extract happen
        in one job. Poll until completion. Returns the raw ParseResult/JobResult.
        """
        client = self._get_client()

        logger.info("Submitting '%s' to Unstructured.io (unified parse+extract)...", pdf_name)

        res = client.parse.run(
            input=file_input,
            schema=schema,
            prompt=prompt,
            output="elements",
            include=["coordinates", "table_html"],
            profile=self.profile,
            wait_seconds=0,
        )

        # Synchronous success (unlikely but handle it)
        raw_state = getattr(res, "status", "")
        state = str(raw_state.value if hasattr(raw_state, "value") else raw_state).lower()
        if any(s in state for s in ("completed", "successful", "done")):
            return res

        # Asynchronous: JobAccepted — poll until done
        job_id = getattr(res, "id", None) or getattr(res, "job_id", None)
        if not job_id:
            raise RuntimeError(
                f"Unstructured.io returned an unexpected response for '{pdf_name}': {res}"
            )

        logger.info("Job accepted (ID: %s). Polling...", job_id)
        start = time.time()
        while time.time() - start < self.max_poll_time:
            time.sleep(self.poll_interval)
            job_status = client.jobs.get(
                job_id,
                output="elements",
                include=["coordinates", "table_html"],
            )
            raw_state = getattr(job_status, "status", "")
            state = str(raw_state.value if hasattr(raw_state, "value") else raw_state).lower()
            elapsed = time.time() - start
            logger.info("Job %s: status=%s (%.1fs elapsed)", job_id, state, elapsed)

            if any(s in state for s in ("completed", "successful", "done")):
                return job_status
            if any(s in state for s in ("failed", "error", "cancelled")):
                err = getattr(job_status, "error", state)
                raise RuntimeError(
                    f"Unstructured.io job {job_id} failed: state={state}, error={err}"
                )

        raise TimeoutError(
            f"Unstructured.io job {job_id} timed out after {self.max_poll_time}s"
        )

    # ── Result Parsing: Structured Text Records ───────────────────────────────

    @staticmethod
    def _parse_text_records(job_result: Any) -> list[dict[str, Any]]:
        """
        Extract the structured question/answer records from a completed job result.
        Handles both ParseResult (direct) and JobResult (.result attribute).
        """
        # JobResult wraps ParseResult in .result
        parse_result = getattr(job_result, "result", job_result)

        ext_data = getattr(parse_result, "extracted_data", None)
        if ext_data is None and isinstance(parse_result, dict):
            ext_data = parse_result.get("extracted_data")
        if not ext_data:
            return []

        first = ext_data[0]
        extracted_obj = getattr(first, "data", first)
        if isinstance(extracted_obj, dict) and "data" in extracted_obj:
            extracted_obj = extracted_obj["data"]

        raw: list[Any] = []
        if isinstance(extracted_obj, dict):
            for key in ("questions", "answers"):
                if key in extracted_obj and isinstance(extracted_obj[key], list):
                    raw = extracted_obj[key]
                    break
            if not raw:
                for v in extracted_obj.values():
                    if isinstance(v, list):
                        raw = v
                        break
        elif isinstance(extracted_obj, list):
            raw = extracted_obj

        records: list[dict[str, Any]] = []
        for r in raw:
            if hasattr(r, "model_dump"):
                records.append(r.model_dump())
            elif hasattr(r, "dict"):
                records.append(r.dict())
            elif isinstance(r, dict):
                records.append(r)
        return records

    # ── Result Parsing: Layout Elements ──────────────────────────────────────

    @staticmethod
    def _parse_elements(job_result: Any) -> list[dict[str, Any]]:
        """
        Extract the flat list of layout elements from the completed job result.
        Elements carry bounding-box coordinates used for PyMuPDF snipping.
        """
        parse_result = getattr(job_result, "result", job_result)
        elements_raw = getattr(parse_result, "elements", None)
        if elements_raw is None and isinstance(parse_result, dict):
            elements_raw = parse_result.get("elements", [])
        if not elements_raw:
            return []

        elements: list[dict[str, Any]] = []
        for el in elements_raw:
            if hasattr(el, "model_dump"):
                elements.append(el.model_dump())
            elif hasattr(el, "dict"):
                elements.append(el.dict())
            elif isinstance(el, dict):
                elements.append(el)
        return elements

    # ── Visual Cropping (PyMuPDF) ─────────────────────────────────────────────

    def _crop_visuals(
        self,
        pdf_path: Path,
        elements: list[dict[str, Any]],
        output_dir: Path,
        is_mcq: bool,
    ) -> list[dict[str, Any]]:
        """
        Use bounding boxes from layout elements to crop and save visual assets
        with PyMuPDF. Returns a manifest list identical to UnstructuredVisualExtractor.
        """
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        manifest: list[dict[str, Any]] = []

        current_q: int | None = None
        current_part: str | None = None
        current_sub_part: str | None = None

        allowed_types = {"Image", "Table"} if self.extract_tables else {"Image"}

        for idx, el in enumerate(elements):
            etype = el.get("type")
            text = (el.get("text") or "").strip()
            meta = el.get("metadata") or {}
            page_num = meta.get("page_number", 0)

            # Context tracking
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

            if etype not in allowed_types:
                continue

            # Skip cover page
            if page_num == 1:
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

            # Artifact filtering (barcodes, timing marks, margin artifacts)
            if self._is_unwanted_artifact(el, page_num, raw_bbox, page.rect):
                continue

            # Expand bbox to include adjacent FigureCaption
            caption_extras: list[str] = []
            for offset in (-1, 1, -2, 2):
                n_idx = idx + offset
                if 0 <= n_idx < len(elements):
                    ne = elements[n_idx]
                    if (
                        ne.get("type") == "FigureCaption"
                        and (ne.get("metadata") or {}).get("page_number") == page_num
                    ):
                        n_coords = (ne.get("metadata") or {}).get("coordinates") or {}
                        n_pts = n_coords.get("points") or []
                        n_lw = n_coords.get("layout_width")
                        n_lh = n_coords.get("layout_height")
                        if n_pts and n_lw and n_lh:
                            n_sx = page.rect.width / float(n_lw)
                            n_sy = page.rect.height / float(n_lh)
                            n_xs = [p[0] * n_sx for p in n_pts]
                            n_ys = [p[1] * n_sy for p in n_pts]
                            fc_rect = fitz.Rect(min(n_xs), min(n_ys), max(n_xs), max(n_ys))
                            if max(0.0, fc_rect.y0 - raw_bbox.y1, raw_bbox.y0 - fc_rect.y1) <= 50.0:
                                raw_bbox = raw_bbox | fc_rect
                                cap_text = (ne.get("text") or "").strip()
                                if cap_text:
                                    caption_extras.append(cap_text)

            clip = (raw_bbox + (-self.pad, -self.pad, self.pad, self.pad)) & page.rect

            # Determine question scope and filename
            q_for_name = current_q if current_q is not None else 1
            surrounding = text
            if caption_extras:
                surrounding = " ".join(caption_extras) + " " + surrounding
            if idx > 0 and (elements[idx - 1].get("metadata") or {}).get("page_number") == page_num:
                surrounding = (elements[idx - 1].get("text") or "") + " " + surrounding
            if idx + 1 < len(elements) and (elements[idx + 1].get("metadata") or {}).get("page_number") == page_num:
                surrounding = surrounding + " " + (elements[idx + 1].get("text") or "")

            if is_mcq:
                target_type = "question"
                target_label = f"Q{q_for_name}"
                tag_str = "question"
                part_label = None
                sub_idx = len([m for m in manifest if m.get("question") == q_for_name]) + 1
                fname = f"p{page_num}_q{q_for_name}_question_mcq_{sub_idx}.png"
                caption: str | None = None
            else:
                if current_part and current_sub_part:
                    target_type = "sub_part"
                    target_label = f"{current_part}({current_sub_part})"
                    tag_str = f"sub_part_{target_label}"
                    part_label: str | None = target_label
                elif current_part:
                    target_type = "part"
                    target_label = current_part
                    tag_str = f"part_{current_part}"
                    part_label = current_part
                else:
                    target_type = "question"
                    target_label = f"Q{q_for_name}"
                    tag_str = "question"
                    part_label = None

                if etype == "Image":
                    fm = FIG_CAP_RE.search(surrounding)
                    if fm:
                        suffix = f"Fig_{fm.group(1)}_{fm.group(2)}"
                        caption = f"Fig. {fm.group(1)}.{fm.group(2)}"
                    else:
                        sub_cnt = len([
                            m for m in manifest
                            if m.get("question") == q_for_name
                            and "diagram" in m.get("filename", "").lower()
                        ]) + 1
                        suffix = f"diagram_{sub_cnt}"
                        caption = caption_extras[0] if caption_extras else None
                    fname = f"p{page_num}_q{q_for_name}_{tag_str}_{suffix}.png"
                else:
                    tm = TABLE_CAP_RE.search(surrounding)
                    if tm:
                        suffix = f"Table_{tm.group(1)}_{tm.group(2)}"
                        caption = f"Table {tm.group(1)}.{tm.group(2)}"
                    else:
                        sub_cnt = len([
                            m for m in manifest
                            if m.get("question") == q_for_name
                            and "table" in m.get("filename", "").lower()
                        ]) + 1
                        suffix = f"Table_{sub_cnt}"
                        caption = caption_extras[0] if caption_extras else "Table"
                    fname = f"p{page_num}_q{q_for_name}_{tag_str}_{suffix}.png"

            # Deduplicate filename
            if any(m["filename"] == fname for m in manifest):
                dup_idx = sum(1 for m in manifest if fname.split(".png")[0] in m["filename"]) + 1
                fname = f"{fname.rsplit('.', 1)[0]}_{dup_idx}.png"

            # Render & save
            pix = page.get_pixmap(clip=clip, dpi=self.dpi)
            samples = pix.samples
            if samples and all(b >= 250 for b in samples[::10]):
                continue  # skip blank/white images

            save_path = output_dir / fname
            pix.save(str(save_path))

            entry: dict[str, Any] = {
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
            logger.info("  Visual saved: %s (page %d, Q%s)", fname, page_num, q_for_name)

        doc.close()
        return manifest

    def _is_unwanted_artifact(
        self,
        el: dict[str, Any],
        page_num: int,
        raw_bbox: fitz.Rect,
        page_rect: fitz.Rect,
    ) -> bool:
        """Filter barcodes, timing marks, fiducials, and perimeter artifacts."""
        text = (el.get("text") or "").strip().lower()
        if page_num == 1:
            return True
        if any(t in text for t in (
            "for examiner's use", "examiner's use", "candidate number",
            "centre number", "blank page",
        )):
            return True
        if "cambridge" in text and any(t in text for t in ("assessment", "o level", "international")):
            return True
        if re.search(r"\*\s*\d{6,}\s*\*|\bDFD\b", el.get("text") or "", re.IGNORECASE):
            return True

        pw, ph = page_rect.width, page_rect.height
        if pw <= 0 or ph <= 0:
            return False

        y0r, y1r = raw_bbox.y0 / ph, raw_bbox.y1 / ph
        x0r, x1r = raw_bbox.x0 / pw, raw_bbox.x1 / pw
        wr = raw_bbox.width / pw
        hr = raw_bbox.height / ph

        if y1r <= 0.090:
            return True
        if y0r <= 0.080 and hr <= 0.08:
            return True
        if y0r >= 0.90:
            return True
        if y1r >= 0.92 and hr <= 0.08:
            return True
        if ((x1r <= 0.12) or (x0r >= 0.85)) and ((y1r <= 0.10) or (y0r >= 0.88)) \
                and wr <= 0.08 and hr <= 0.08:
            return True
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        pdf_path: Path,
        category: PaperCategory,
        meta: dict[str, Any],
        output_dir: Path,
        syllabus_csv: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the unified pipeline for one PDF.

        Returns a dict with:
          "records"  – list of structured question/answer dicts
          "manifest" – list of visual manifest entry dicts
          "json_path"     – Path to the saved canonical JSON
          "manifest_path" – Path to the saved manifest.json
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Cache check ───────────────────────────────────────────────────────
        cache_text = pdf_path.parent / f"{pdf_path.stem}_unified_text.json"
        cache_elements = pdf_path.parent / f"{pdf_path.stem}_unified_elements.json"
        manifest_path = output_dir / "manifest.json"
        json_path = pdf_path.with_suffix(".json")

        records: list[dict[str, Any]] = []
        elements: list[dict[str, Any]] = []

        if not force and cache_text.exists() and cache_elements.exists():
            try:
                with open(cache_text, encoding="utf-8") as f:
                    records = json.load(f)
                with open(cache_elements, encoding="utf-8") as f:
                    elements = json.load(f)
                logger.info(
                    "Loaded %d records and %d elements from cache for '%s'",
                    len(records), len(elements), pdf_path.name,
                )
            except Exception as exc:
                logger.warning("Cache read error: %s — re-running API call", exc)
                records, elements = [], []

        if not records or not elements:
            # ── Load syllabus ─────────────────────────────────────────────────
            syllabus_text = ""
            if syllabus_csv and syllabus_csv.exists():
                try:
                    units = SyllabusLoader.load_syllabus(syllabus_csv)
                    syllabus_text = SyllabusLoader.format_syllabus_for_prompt(units)
                    logger.info("Loaded syllabus: %d units from '%s'", len(units), syllabus_csv.name)
                except Exception as exc:
                    logger.warning("Could not load syllabus '%s': %s", syllabus_csv, exc)

            schema, prompt = self._build_schema_and_prompt(category, meta, syllabus_text)

            # ── Upload pruned PDF ─────────────────────────────────────────────
            pruned = self._prune_pdf_bytes(pdf_path)
            file_input: Any = (pdf_path.name, pruned) if pruned else pdf_path

            # ── Single API call ───────────────────────────────────────────────
            job_result = self._submit_and_poll(file_input, schema, prompt, pdf_path.name)

            records = self._parse_text_records(job_result)
            elements = self._parse_elements(job_result)

            # ── Cache raw outputs ─────────────────────────────────────────────
            try:
                with open(cache_text, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                with open(cache_elements, "w", encoding="utf-8") as f:
                    json.dump(elements, f, indent=2, ensure_ascii=False)
                logger.info("Cached unified outputs for '%s'", pdf_path.name)
            except Exception as exc:
                logger.warning("Could not cache unified output: %s", exc)

        # ── Validate marks hierarchy ──────────────────────────────────────────
        if records:
            records, _ = MarkHierarchyValidator.reconcile_and_validate_records(records)

        # ── Crop visuals with PyMuPDF ─────────────────────────────────────────
        is_mcq = bool(re.search(r"_qp_1[1-3]", pdf_path.stem.lower()))
        manifest: list[dict[str, Any]] = []

        if "_qp_" in pdf_path.stem.lower() and elements:
            manifest = self._crop_visuals(pdf_path, elements, output_dir, is_mcq)
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                logger.info("Saved manifest.json (%d visuals)", len(manifest))
            except Exception as exc:
                logger.warning("Could not save manifest.json: %s", exc)

        # ── Link visuals into records ─────────────────────────────────────────
        if records and manifest:
            patch_records_with_manifest(records, manifest)

        # ── Save canonical JSON ───────────────────────────────────────────────
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            logger.info("Saved canonical JSON: %s (%d records)", json_path.name, len(records))
        except Exception as exc:
            logger.warning("Could not save canonical JSON: %s", exc)

        return {
            "records": records,
            "manifest": manifest,
            "json_path": json_path,
            "manifest_path": manifest_path,
        }
