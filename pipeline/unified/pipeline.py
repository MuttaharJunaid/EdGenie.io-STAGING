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
import requests
from unstructured_transform_client import TransformClient

from pipeline.config import settings
from pipeline.unified.manifest_patcher import patch_records_with_manifest
from pipeline.unified.marks_validator import MarkHierarchyValidator
from pipeline.unified.paper_type_detector import PaperCategory, PaperTypeDetector
from pipeline.unified.schemas import (
    EssayAnswersContainer,
    EssayQuestionsContainer,
    MCQAnswersContainer,
    MCQQuestionsContainer,
    to_strict_openai_schema,
)
from pipeline.unified.syllabus_loader import SyllabusLoader

logger = logging.getLogger(__name__)


class JobTimeoutError(TimeoutError):
    """Raised when polling an Unstructured.io job exceeds max_poll_time, carrying the job ID."""

    def __init__(self, job_id: str, message: str, elapsed: float) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.elapsed = elapsed

# ── Regex patterns (same as UnstructuredVisualExtractor) ─────────────────────
Q_ROOT_RE = re.compile(
    r"^(?:(?:\*{0,2})Question\s+(\d{1,2})(?:\*{0,2})|(?:\*{0,2})(\d{1,2})(?:\*{0,2})(?:[\.\t\s]|$))",
    re.IGNORECASE,
)
PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([a-h])\)(?:\*{0,2})", re.IGNORECASE)
SUB_PART_RE = re.compile(r"(?:^|\s)(?:\*{0,2})\(([ivx]+)\)(?:\*{0,2})", re.IGNORECASE)
FIG_CAP_RE = re.compile(r"(?:Fig\.|Figure)\s*(\d+)[\.\_](\d+)", re.IGNORECASE)
TABLE_CAP_RE = re.compile(r"Table\s*(\d+)[\.\_](\d+)", re.IGNORECASE)


def get_next_run_number(base_dir: Path | str | None = None) -> int:
    """
    Determines the next pipeline run number by inspecting existing timed_out_jobs files
    and the persistent .pipeline_run_counter file. Monotonically increments.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    pattern = re.compile(r"^timed_out_jobs(?:_run)?_(\d+)\.json$", re.IGNORECASE)
    existing_nums: list[int] = []
    if base_dir.exists():
        for p in base_dir.glob("timed_out_jobs*.json"):
            m = pattern.match(p.name)
            if m:
                existing_nums.append(int(m.group(1)))

    counter_file = base_dir / ".pipeline_run_counter"
    counter_val = 0
    if counter_file.exists():
        try:
            counter_val = int(counter_file.read_text().strip())
        except Exception:
            pass

    next_num = max(existing_nums + [counter_val], default=0) + 1
    try:
        counter_file.write_text(str(next_num))
    except Exception:
        pass
    return next_num


def resolve_timed_out_jobs_path(
    run_number: int | str | None = None,
    suffix: str | None = None,
    explicit_file: str | Path | None = None,
    auto_increment: bool = False,
    base_dir: Path | str | None = None,
) -> Path:
    """
    Resolves the target Path for a run's timed_out_jobs JSON file.

    Priority:
    1. explicit_file: user-provided path or filename
    2. suffix: user-provided custom suffix (e.g. '1' -> timed_out_jobs_1.json, 'run_1' -> timed_out_jobs_run_1.json)
    3. run_number: run number (e.g. 1 -> timed_out_jobs_run_1.json)
    4. auto_increment=True: automatically fetches next run number -> timed_out_jobs_run_<next>.json
    5. Fallback: timed_out_jobs.json
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    if explicit_file is not None:
        p = Path(explicit_file)
        return p if p.is_absolute() else base_dir / p

    if suffix is not None:
        cleaned_suffix = str(suffix).strip().lstrip("_")
        return base_dir / f"timed_out_jobs_{cleaned_suffix}.json"

    if run_number is not None:
        rn_str = str(run_number).strip().lstrip("_")
        if rn_str.lower().startswith("run_") or rn_str.lower().startswith("run"):
            return base_dir / f"timed_out_jobs_{rn_str}.json"
        return base_dir / f"timed_out_jobs_run_{rn_str}.json"

    if auto_increment:
        next_run = get_next_run_number(base_dir)
        return base_dir / f"timed_out_jobs_run_{next_run}.json"

    return base_dir / "timed_out_jobs.json"


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
        timed_out_jobs_file: Path | str | None = None,
        run_number: int | str | None = None,
        suffix: str | None = None,
        auto_increment: bool = False,
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
        self.profile = profile or getattr(settings, "unstructured_profile", "best")
        self.dpi = dpi or getattr(settings, "visual_dpi", 300)
        self.pad = pad if pad is not None else getattr(settings, "visual_pad", 2.0)
        self.timeout_seconds = timeout_seconds or getattr(settings, "unstructured_timeout_seconds", 600.0)
        self.poll_interval = poll_interval or getattr(settings, "unstructured_poll_interval", 5.0)
        self.max_poll_time = max_poll_time or getattr(settings, "unstructured_max_poll_time", 600.0)
        self.extract_tables = extract_tables if extract_tables is not None else getattr(settings, "extract_tables", True)
        self._client: TransformClient | None = None
        self.timed_out_jobs_file: Path = resolve_timed_out_jobs_path(
            run_number=run_number,
            suffix=suffix,
            explicit_file=timed_out_jobs_file,
            auto_increment=auto_increment,
        )
        self.timed_out_jobs: list[dict[str, Any]] = []
        if self.timed_out_jobs_file.exists():
            self._load_timed_out_jobs()

    def init_timed_out_jobs_file(self) -> Path:
        """Ensure the timed-out jobs file exists on disk with an initial empty JSON array."""
        if not self.timed_out_jobs_file.exists():
            try:
                self.timed_out_jobs_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.timed_out_jobs_file, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
            except Exception as exc:
                logger.warning("Could not initialize %s: %s", self.timed_out_jobs_file, exc)
        return self.timed_out_jobs_file

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

    # ── PDF Pruning (blank page removal & page mapping) ─────────────────────

    @staticmethod
    def _get_prune_plan(pdf_path: Path) -> tuple[list[int], dict[int, int]]:
        """
        Calculates which pages to keep and the mapping from pruned 1-based page numbers
        to original 1-based page numbers.
        Returns:
            (kept_pages_0_based, page_map: {pruned_page_1_based: original_page_1_based})
        """
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            if num_pages <= 2:
                doc.close()
                return list(range(num_pages)), {p: p for p in range(1, num_pages + 1)}

            kept_pages: list[int] = []
            page_map: dict[int, int] = {}
            pruned_idx = 1
            for pno in range(num_pages):
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
                        continue
                kept_pages.append(pno)
                page_map[pruned_idx] = pno + 1
                pruned_idx += 1
            doc.close()
            return kept_pages, page_map
        except Exception as exc:
            logger.debug("Failed to calculate prune plan for %s: %s", pdf_path, exc)
            return [], {}

    @classmethod
    def _prune_pdf_bytes(cls, pdf_path: Path) -> bytes | None:
        """Strip blank / BLANK PAGE pages before upload to cut cloud processing time."""
        try:
            doc = fitz.open(str(pdf_path))
            kept_pages, _ = cls._get_prune_plan(pdf_path)
            pruned_count = len(doc) - len(kept_pages)
            if pruned_count > 0 and len(kept_pages) > 0:
                pruned = fitz.open()
                for pno in kept_pages:
                    pruned.insert_pdf(doc, from_page=pno, to_page=pno)
                logger.info(
                    "Pruned %d blank page(s) from %s (%d/%d pages submitted)",
                    pruned_count, pdf_path.name, len(pruned), len(doc),
                )
                res = bytes(pruned.tobytes())
                pruned.close()
                doc.close()
                return res
            doc.close()
        except Exception as exc:
            logger.debug("PDF pruning bypassed for %s: %s", pdf_path.name, exc)
        return None

    @classmethod
    def _remap_elements_pages(
        cls,
        elements: list[dict[str, Any]],
        page_map: dict[int, int],
    ) -> list[dict[str, Any]]:
        """
        Translates metadata['page_number'] of each layout element from pruned page numbers
        to original PDF page numbers if not already remapped.
        """
        if not elements or not page_map:
            return elements

        for el in elements:
            meta = el.get("metadata")
            if not isinstance(meta, dict) or meta.get("_pages_remapped"):
                continue
            pruned_pno = meta.get("page_number")
            if isinstance(pruned_pno, int) and pruned_pno in page_map:
                meta["pruned_page_number"] = pruned_pno
                meta["page_number"] = page_map[pruned_pno]
                meta["_pages_remapped"] = True
        return elements

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
            try:
                job_status = client.jobs.get(
                    job_id,
                    output="elements",
                    include=["coordinates", "table_html"],
                )
            except Exception as poll_exc:
                logger.warning("Transient error during job %s poll: %s. Retrying...", job_id, poll_exc)
                continue

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

        elapsed = time.time() - start
        raise JobTimeoutError(
            job_id=str(job_id),
            message=f"Unstructured.io job {job_id} timed out after {elapsed:.1f}s",
            elapsed=elapsed,
        )

    # ── Timed-Out Job Tracking & Playground Endpoint Recovery ────────────────

    def _save_timed_out_jobs(self) -> None:
        """Persist timed-out jobs to timed_out_jobs.json."""
        try:
            with open(self.timed_out_jobs_file, "w", encoding="utf-8") as f:
                json.dump(self.timed_out_jobs, f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist timed_out_jobs.json: %s", exc)

    def _load_timed_out_jobs(self) -> list[dict[str, Any]]:
        """Load timed-out jobs from timed_out_jobs.json if exists."""
        if self.timed_out_jobs_file.exists():
            try:
                with open(self.timed_out_jobs_file, encoding="utf-8") as f:
                    self.timed_out_jobs = json.load(f)
            except Exception as exc:
                logger.warning("Could not load timed_out_jobs.json: %s", exc)
        return self.timed_out_jobs

    def record_timed_out_job(
        self,
        job_id: str,
        pdf_path: Path,
        category: PaperCategory,
        meta: dict[str, Any],
        output_dir: Path,
        syllabus_csv: Path | None = None,
    ) -> dict[str, Any]:
        """Record details of a timed-out job in memory and on disk."""
        record = {
            "job_id": job_id,
            "pdf_path": str(Path(pdf_path).resolve()),
            "pdf_name": Path(pdf_path).name,
            "category": category.value if hasattr(category, "value") else str(category),
            "meta": meta,
            "output_dir": str(Path(output_dir).resolve()),
            "syllabus_csv": str(Path(syllabus_csv).resolve()) if syllabus_csv else None,
            "timed_out_at": time.time(),
            "status": "timed_out",
        }
        existing = next((j for j in self.timed_out_jobs if j.get("job_id") == job_id), None)
        if existing:
            existing.update(record)
        else:
            self.timed_out_jobs.append(record)
        self._save_timed_out_jobs()
        return record

    def mark_timed_out_job_completed(self, job_id: str) -> None:
        """Mark a job as completed in the timed-out registry."""
        for j in self.timed_out_jobs:
            if j.get("job_id") == job_id:
                j["status"] = "completed"
                j["completed_at"] = time.time()
        self._save_timed_out_jobs()

    def fetch_job_result_from_endpoint(
        self,
        job_id: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> Any | None:
        """
        Fetch the result of a job using the Unstructured playground endpoint:
        https://transform.unstructured.io/playground/jobs/{job_id}?output=elements

        Falls back to the v2 API endpoint and SDK client if needed.
        """
        playground_url = f"https://transform.unstructured.io/playground/jobs/{job_id}?output=elements"
        server_base = str(self.server_url or "https://transform.unstructured.io").rstrip("/")
        api_url = f"{server_base}/api/v2/jobs/{job_id}?output=elements"

        api_key = self.api_key or os.environ.get("UNSTRUCTURED_API_KEY", "")
        headers = {
            "Accept": "application/json",
            "User-Agent": "EdGenie-Unified-Pipeline/3.1.0",
        }
        if api_key:
            headers["unstructured-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(1, max_retries + 1):
            # 1. Primary target: Playground endpoint
            try:
                logger.info(
                    "Querying playground endpoint for job %s (attempt %d/%d): %s",
                    job_id, attempt, max_retries, playground_url,
                )
                resp = requests.get(playground_url, headers=headers, timeout=timeout_seconds)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        if self._is_job_ready(data):
                            logger.info("Playground endpoint returned ready result for job %s", job_id)
                            return data
                        logger.info("Job %s on playground endpoint not ready yet (status=%s)", job_id, data.get("status"))
                    except Exception as json_err:
                        logger.debug("Failed to decode JSON from playground response: %s", json_err)
                else:
                    logger.debug("Playground endpoint status %d for job %s", resp.status_code, job_id)
            except Exception as exc:
                logger.warning("Error querying playground endpoint for job %s: %s", job_id, exc)

            # 2. Resilient fallback: API v2 endpoint
            try:
                resp = requests.get(api_url, headers=headers, timeout=timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    if self._is_job_ready(data):
                        logger.info("API v2 endpoint returned ready result for job %s", job_id)
                        return data
            except Exception as exc:
                logger.debug("Error querying API v2 endpoint for job %s: %s", job_id, exc)

            # 3. Resilient fallback: SDK client
            try:
                client = self._get_client()
                job_status = client.jobs.get(
                    job_id,
                    output="elements",
                    include=["coordinates", "table_html"],
                )
                if self._is_job_ready(job_status):
                    logger.info("SDK client returned ready result for job %s", job_id)
                    return job_status
            except Exception as exc:
                logger.debug("Error querying SDK jobs.get for job %s: %s", job_id, exc)

            if attempt < max_retries:
                time.sleep(retry_delay)

        return None

    @staticmethod
    def _is_job_ready(result: Any) -> bool:
        """Check whether a job result payload has completed parsing."""
        if result is None:
            return False

        if isinstance(result, dict):
            status = str(result.get("status") or result.get("state") or "").lower()
            if any(s in status for s in ("completed", "successful", "done")):
                return True
            inner_res = result.get("result")
            if isinstance(inner_res, dict):
                inner_status = str(inner_res.get("status") or "").lower()
                if any(s in inner_status for s in ("completed", "successful", "done")):
                    return True
                if inner_res.get("elements") or inner_res.get("extracted_data"):
                    return True
            if result.get("elements") or result.get("extracted_data"):
                return True
            return False

        if isinstance(result, list):
            return len(result) > 0

        status_val = getattr(result, "status", None)
        status_str = str(getattr(status_val, "value", status_val) or "").lower()
        if any(s in status_str for s in ("completed", "successful", "done")):
            return True
        parse_res = getattr(result, "result", None)
        if parse_res and (getattr(parse_res, "elements", None) or getattr(parse_res, "extracted_data", None)):
            return True

        return False

    def recover_timed_out_jobs(
        self,
        jobs_file: Path | str | None = None,
        max_retries_per_job: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Recover and complete the remaining work for all noted timed-out jobs
        using the playground endpoint:
        https://transform.unstructured.io/playground/jobs/{job_id}?output=elements
        """
        if jobs_file is not None:
            self.timed_out_jobs_file = Path(jobs_file)
        self._load_timed_out_jobs()

        recovered_results: list[dict[str, Any]] = []
        pending_jobs = [j for j in self.timed_out_jobs if j.get("status") == "timed_out"]
        if not pending_jobs:
            logger.info("No timed-out jobs pending recovery.")
            return []

        logger.info("Found %d timed-out job(s) to recover.", len(pending_jobs))

        for job in pending_jobs:
            job_id = job["job_id"]
            pdf_path = Path(job["pdf_path"])
            output_dir = Path(job["output_dir"])
            category_val = job.get("category")
            category = None
            if category_val:
                try:
                    category = PaperCategory(category_val)
                except Exception:
                    pass

            logger.info("Attempting recovery for job %s ('%s')...", job_id, pdf_path.name)
            job_result = self.fetch_job_result_from_endpoint(
                job_id,
                max_retries=max_retries_per_job,
            )

            if job_result is not None:
                try:
                    work_res = self.complete_pipeline_work(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        job_result=job_result,
                        category=category,
                    )
                    self.mark_timed_out_job_completed(job_id)
                    recovered_results.append({
                        "job_id": job_id,
                        "pdf_name": pdf_path.name,
                        "status": "recovered",
                        "records_count": len(work_res.get("records", [])),
                        "manifest_count": len(work_res.get("manifest", [])),
                        "json_path": str(work_res.get("json_path", "")),
                    })
                    logger.info("Successfully recovered and completed work for job %s!", job_id)
                except Exception as exc:
                    logger.error("Failed completing work for recovered job %s: %s", job_id, exc)
            else:
                logger.warning("Job %s could not be recovered from endpoint yet.", job_id)

        return recovered_results

    # ── Result Parsing: Structured Text Records ───────────────────────────────

    @staticmethod
    def _parse_text_records(job_result: Any) -> list[dict[str, Any]]:
        """
        Extract the structured question/answer records from a completed job result.
        Handles both ParseResult (direct), JobResult (.result attribute),
        and dictionary payloads from the endpoint.
        """
        parse_result = getattr(job_result, "result", None)
        if parse_result is None and isinstance(job_result, dict):
            parse_result = job_result.get("result", job_result)
        elif parse_result is None:
            parse_result = job_result

        ext_data = getattr(parse_result, "extracted_data", None)
        if ext_data is None and isinstance(parse_result, dict):
            ext_data = parse_result.get("extracted_data") or parse_result.get("data")
        if not ext_data and isinstance(job_result, dict):
            ext_data = job_result.get("extracted_data") or job_result.get("data")
        if not ext_data:
            return []

        first = ext_data[0] if isinstance(ext_data, list) and ext_data else ext_data
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
        parse_result = getattr(job_result, "result", None)
        if parse_result is None and isinstance(job_result, dict):
            parse_result = job_result.get("result", job_result)
        elif parse_result is None:
            parse_result = job_result

        elements_raw = getattr(parse_result, "elements", None)
        if elements_raw is None and isinstance(parse_result, dict):
            elements_raw = parse_result.get("elements")
        if elements_raw is None and isinstance(job_result, dict):
            elements_raw = job_result.get("elements")
        elif elements_raw is None and isinstance(job_result, list):
            elements_raw = job_result

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
        _, page_map = self._get_prune_plan(pdf_path)
        elements = self._remap_elements_pages(elements, page_map)

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
                    part_label = target_label
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

    def complete_pipeline_work(
        self,
        pdf_path: Path,
        output_dir: Path,
        job_result: Any = None,
        records: list[dict[str, Any]] | None = None,
        elements: list[dict[str, Any]] | None = None,
        category: PaperCategory | None = None,
    ) -> dict[str, Any]:
        """
        Completes the remaining processing steps after receiving the job result:
          1. Parses structured question/answer records and layout elements.
          2. Caches raw outputs to <stem>_unified_text.json and <stem>_unified_elements.json.
          3. Reconciles and validates marks hierarchy.
          4. Crops diagram and table visuals using PyMuPDF and creates manifest.json.
          5. Patches question records with cropped visual paths.
          6. Saves canonical <stem>.json.
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if category is None:
            category = PaperTypeDetector.detect(pdf_path)

        cache_text = pdf_path.parent / f"{pdf_path.stem}_unified_text.json"
        cache_elements = pdf_path.parent / f"{pdf_path.stem}_unified_elements.json"
        manifest_path = output_dir / "manifest.json"
        json_path = pdf_path.with_suffix(".json")

        if records is None or elements is None:
            if job_result is not None:
                records = self._parse_text_records(job_result)
                elements = self._parse_elements(job_result)

                # Remap elements' pruned page numbers to original PDF page numbers
                _, page_map = self._get_prune_plan(pdf_path)
                elements = self._remap_elements_pages(elements, page_map)

                # Cache raw outputs
                try:
                    with open(cache_text, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, ensure_ascii=False)
                    with open(cache_elements, "w", encoding="utf-8") as f:
                        json.dump(elements, f, indent=2, ensure_ascii=False)
                    logger.info("Cached unified outputs for '%s'", pdf_path.name)
                except Exception as exc:
                    logger.warning("Could not cache unified output: %s", exc)
            else:
                records = records or []
                elements = elements or []
        else:
            _, page_map = self._get_prune_plan(pdf_path)
            elements = self._remap_elements_pages(elements, page_map)
            # Sync cache_elements if elements were remapped from an older unmapped cache
            if cache_elements.exists() and any(el.get("metadata", {}).get("_pages_remapped") for el in elements):
                try:
                    with open(cache_elements, "w", encoding="utf-8") as f:
                        json.dump(elements, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

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
          "records"       – list of structured question/answer dicts
          "manifest"      – list of visual manifest entry dicts
          "json_path"     – Path to the saved canonical JSON
          "manifest_path" – Path to the saved manifest.json
        """
        pdf_path = Path(pdf_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Cache check ───────────────────────────────────────────────────────
        cache_text = pdf_path.parent / f"{pdf_path.stem}_unified_text.json"
        cache_elements = pdf_path.parent / f"{pdf_path.stem}_unified_elements.json"

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
                return self.complete_pipeline_work(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    records=records,
                    elements=elements,
                    category=category,
                )
            except Exception as exc:
                logger.warning("Cache read error: %s — re-running API call", exc)
                records, elements = [], []

        # ── Load syllabus ─────────────────────────────────────────────────────
        syllabus_text = ""
        if syllabus_csv and syllabus_csv.exists():
            try:
                units = SyllabusLoader.load_syllabus(syllabus_csv)
                syllabus_text = SyllabusLoader.format_syllabus_for_prompt(units)
                logger.info("Loaded syllabus: %d units from '%s'", len(units), syllabus_csv.name)
            except Exception as exc:
                logger.warning("Could not load syllabus '%s': %s", syllabus_csv, exc)

        schema, prompt = self._build_schema_and_prompt(category, meta, syllabus_text)

        # ── Upload pruned PDF ─────────────────────────────────────────────────
        pruned = self._prune_pdf_bytes(pdf_path)
        file_input: Any = (pdf_path.name, pruned) if pruned else pdf_path

        # ── Single API call with timeout resilience ───────────────────────────
        job_result = None
        try:
            job_result = self._submit_and_poll(file_input, schema, prompt, pdf_path.name)
        except JobTimeoutError as te:
            job_id = te.job_id
            logger.warning(
                "Job %s for '%s' timed out after %.1fs. Keeping note of job for endpoint recovery...",
                job_id, pdf_path.name, te.elapsed,
            )
            self.record_timed_out_job(
                job_id=job_id,
                pdf_path=pdf_path,
                category=category,
                meta=meta,
                output_dir=output_dir,
                syllabus_csv=syllabus_csv,
            )
            # Attempt recovery using playground endpoint
            logger.info(
                "Attempting recovery using endpoint https://transform.unstructured.io/playground/jobs/%s?output=elements...",
                job_id,
            )
            job_result = self.fetch_job_result_from_endpoint(job_id)
            if job_result is not None:
                logger.info("Job %s result successfully retrieved from endpoint! Completing remaining work...", job_id)
                self.mark_timed_out_job_completed(job_id)
            else:
                raise

        return self.complete_pipeline_work(
            pdf_path=pdf_path,
            output_dir=output_dir,
            job_result=job_result,
            category=category,
        )
