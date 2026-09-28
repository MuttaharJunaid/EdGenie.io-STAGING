import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.extractor.paper_type_detector import PaperCategory, PaperTypeDetector
from pipeline.ingestion.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

CDN_DOMAIN_PREFIX = os.getenv("CDN_DOMAIN_PREFIX", "https://assets.edgenie.io").rstrip("/")


class PostgresIngestor:
    """
    Stage 4 PostgresIngestor (Relational + JSONB):
    Ingests Stage 2 structured question/answer JSONs and Stage 3 visual manifests
    directly into PostgreSQL database tables:
        1. subjects
        2. papers
        3. questions (with JSONB options/parts/sub_topics/command_words)
        4. answers   (with JSONB parts)
        5. visuals   (with JSONB bbox)
        6. pipeline_runs

    Decoupled from Firebase Storage: runs purely against PostgreSQL without
    requiring cloud storage credentials.
    """

    def __init__(self, db_client: PostgresClient | None = None, cdn_prefix: str = CDN_DOMAIN_PREFIX):
        self.db = db_client or PostgresClient()
        self.cdn_prefix = cdn_prefix.rstrip("/")

    def parse_paper_filename(self, filename: str) -> dict[str, Any]:
        """
        Parses standard Cambridge filename pattern like '5054_s25_qp_22.json' or '5054_w18_ms_11.pdf'.
        Robust to prefixes/suffixes like 'mock_5054_s25_qp_22.json'.
        """
        stem = Path(filename).stem

        # 1. Primary regex match: e.g. 5054_s25_qp_22
        match = re.search(r"(\d{4})_([swm]\d{2})_(qp|ms|er|gt|ci)_(\d{1,2})", stem, re.IGNORECASE)
        if match:
            subject_code = match.group(1)
            sy_raw = match.group(2).lower()
            session_char = sy_raw[0]
            session = {"s": "jun", "w": "nov", "m": "mar"}.get(session_char, "jun")
            yr_val = int(sy_raw[1:])
            year = 2000 + yr_val if yr_val < 100 else yr_val
            paper_type = match.group(3).lower()
            variant = match.group(4)
            paper_number = int(variant[0]) if variant and variant[0].isdigit() else 1
            return {
                "paper_code": stem,
                "subject_code": subject_code,
                "year": year,
                "session": session,
                "paper_type": paper_type,
                "variant": variant,
                "paper_number": paper_number,
            }

        # 2. Fallback delimiter split
        parts = stem.split("_")
        subject_code = parts[0] if len(parts) > 0 else "5054"
        is_ms = "_ms_" in stem.lower() or stem.lower().endswith("_ms")
        paper_type = "ms" if is_ms else "qp"

        return {
            "paper_code": stem,
            "subject_code": subject_code,
            "year": 2025,
            "session": "jun",
            "paper_type": paper_type,
            "variant": "11",
            "paper_number": 1,
        }

    def ingest_paper_json(self, json_path: Path) -> dict[str, Any]:
        """
        Ingests a single Stage 2 paper JSON file and its Stage 3 visual manifest into PostgreSQL.
        Schema is auto-initialized on first call so this works on a fresh database.
        """
        # Ensure schema tables and indexes exist (idempotent via IF NOT EXISTS)
        self.db.init_schema()

        json_path = Path(json_path).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            records: list[dict[str, Any]] = json.load(f)

        if not records:
            logger.warning(f"No question/answer records found in {json_path.name}")
            return {"paper_code": json_path.stem, "status": "empty", "questions_ingested": 0, "answers_ingested": 0}

        # Parse filename & metadata
        meta = self.parse_paper_filename(json_path.name)
        sample = records[0]

        paper_code = meta["paper_code"]
        subject_code = str(sample.get("SubjectCode") or meta["subject_code"])
        year = int(sample.get("Year") or meta["year"])
        session = str(sample.get("Session") or meta["session"]).lower()
        variant = str(sample.get("Variant") or meta["variant"])
        paper_number = int(sample.get("PaperNumber") or meta["paper_number"])
        paper_type = meta["paper_type"]

        # Detect category (MCQ QP, Essay QP, MCQ MS, Essay MS)
        pdf_path = json_path.with_suffix(".pdf")
        category = PaperCategory.OTHER
        if pdf_path.exists():
            try:
                category = PaperTypeDetector.detect(pdf_path)
            except Exception:
                pass

        if category == PaperCategory.OTHER:
            if paper_type == "ms":
                category = PaperCategory.MCQ_MS if "Answer" in sample and "Options" not in sample else PaperCategory.ESSAY_MS
            else:
                category = PaperCategory.MCQ_QP if "Options" in sample else PaperCategory.ESSAY_QP

        # 1. Upsert Paper Record
        pdf_cdn_url = f"{self.cdn_prefix}/pdfs/{subject_code}/{year}/{session}/{paper_code}.pdf"
        paper_record = {
            "paper_code": paper_code,
            "subject_code": subject_code,
            "year": year,
            "session": session,
            "variant": variant,
            "paper_number": paper_number,
            "paper_type": paper_type,
            "category": category.value if isinstance(category, PaperCategory) else str(category),
            "pdf_filename": pdf_path.name if pdf_path.exists() else f"{paper_code}.pdf",
            "pdf_cdn_url": pdf_cdn_url,
            "question_count": len(records),
            "metadata": {
                "json_filename": json_path.name,
                "ingested_at": datetime.utcnow().isoformat(),
            },
        }
        self.db.upsert_paper(paper_record)

        # 2. Ingest Questions / Answers
        questions_to_upsert: list[dict[str, Any]] = []
        answers_to_upsert: list[dict[str, Any]] = []

        is_ms = paper_type == "ms" or "ms" in category.value

        for rec in records:
            qnum_str = rec.get("QuestionNumber") or rec.get("AnswerNumber") or "Q1"
            qnum_match = re.search(r"Q(\d+)", qnum_str)
            qnum = int(qnum_match.group(1)) if qnum_match else 1

            # Format formatted question string
            q_formatted = qnum_str

            # Image CDN URL handling (supports str, list of strs, or None)
            img_rel = rec.get("image")
            img_filename = None
            img_cdn_url = None
            images_list = []

            if isinstance(img_rel, list) and img_rel:
                filenames = [Path(p).name for p in img_rel if p]
                img_filename = ", ".join(filenames) if filenames else None
                cdn_urls = [f"{self.cdn_prefix}/visuals/{paper_code}/{fn}" for fn in filenames]
                img_cdn_url = ", ".join(cdn_urls) if cdn_urls else None
                images_list = [{"filename": fn, "cdn_url": url} for fn, url in zip(filenames, cdn_urls)]
            elif isinstance(img_rel, str) and img_rel.strip():
                fn = Path(img_rel.strip()).name
                img_filename = fn
                img_cdn_url = f"{self.cdn_prefix}/visuals/{paper_code}/{fn}"
                images_list = [{"filename": fn, "cdn_url": img_cdn_url}]

            if is_ms:
                # Answer record
                qp_paper_code = paper_code.replace("_ms_", "_qp_")
                ans_id = f"{paper_code}_Q{qnum}"
                question_key = f"{qp_paper_code}_Q{qnum}"

                answer_text = rec.get("Answer") or rec.get("Answer_text") or ""
                parts = rec.get("Parts")

                answers_to_upsert.append(
                    {
                        "id": ans_id,
                        "question_key": question_key,
                        "paper_code": paper_code,
                        "subject_code": subject_code,
                        "year": year,
                        "session": session,
                        "variant": variant,
                        "question_number": qnum,
                        "answer_text": answer_text,
                        "parts": parts,
                        "image_filename": img_filename,
                        "image_cdn_url": img_cdn_url,
                        "metadata": {"raw": rec, "images": images_list},
                    }
                )
            else:
                # Question record
                q_key = f"{paper_code}_Q{qnum}"
                q_text = rec.get("Question_text") or rec.get("QuestionText") or ""
                marks = int(rec.get("Marks") or 1)
                unit = str(rec.get("Unit") or "")
                topic = str(rec.get("Topic") or "")
                topic_name = str(rec.get("Topic_name") or "")
                sub_topics = rec.get("Sub_topic") or rec.get("Sub_topics") or []
                diff_score = int(rec.get("Difficulty_score") or 0)
                diff_label = str(rec.get("Difficulty_label") or "")
                command_words = rec.get("Command_words") or []
                options = rec.get("Options")
                parts = rec.get("Parts")

                questions_to_upsert.append(
                    {
                        "question_key": q_key,
                        "paper_code": paper_code,
                        "subject_code": subject_code,
                        "year": year,
                        "session": session,
                        "variant": variant,
                        "question_number": qnum,
                        "question_number_formatted": q_formatted,
                        "question_text": q_text,
                        "marks": marks,
                        "unit": unit,
                        "topic": topic,
                        "topic_name": topic_name,
                        "sub_topics": sub_topics,
                        "difficulty_score": diff_score,
                        "difficulty_label": diff_label,
                        "command_words": command_words,
                        "options": options,
                        "parts": parts,
                        "image_filename": img_filename,
                        "image_cdn_url": img_cdn_url,
                        "metadata": {"raw": rec, "images": images_list},
                    }
                )

        q_count = self.db.upsert_questions(questions_to_upsert)
        a_count = self.db.upsert_answers(answers_to_upsert)

        # 3. Ingest Visual Manifest if present
        visuals_dir = json_path.parent / json_path.stem
        visual_manifest_path = visuals_dir / "manifest.json"
        visuals_to_upsert: list[dict[str, Any]] = []

        if visual_manifest_path.exists():
            try:
                with open(visual_manifest_path, encoding="utf-8") as mf:
                    v_entries = json.load(mf)
                for v in v_entries:
                    v_fname = v.get("filename") or v.get("file")
                    if not v_fname:
                        continue
                    v_id = f"{paper_code}_{v_fname}"
                    v_cdn = f"{self.cdn_prefix}/visuals/{paper_code}/{v_fname}"
                    visuals_to_upsert.append(
                        {
                            "id": v_id,
                            "paper_code": paper_code,
                            "subject_code": subject_code,
                            "year": year,
                            "session": session,
                            "question_number": v.get("question"),
                            "caption": v.get("caption"),
                            "page": v.get("page"),
                            "bbox": v.get("bbox", []),
                            "filename": v_fname,
                            "relative_path": f"{paper_code}/{v_fname}",
                            "cdn_url": v_cdn,
                        }
                    )
            except Exception as e:
                logger.error(f"Error reading visual manifest for {paper_code}: {e}")

        v_count = self.db.upsert_visuals(visuals_to_upsert)

        logger.info(
            f"Ingested {paper_code} into PostgreSQL: {q_count} questions, "
            f"{a_count} answers, {v_count} visuals."
        )

        return {
            "paper_code": paper_code,
            "category": category.value if isinstance(category, PaperCategory) else str(category),
            "questions_ingested": q_count,
            "answers_ingested": a_count,
            "visuals_ingested": v_count,
            "pdf_cdn_url": pdf_cdn_url,
        }

    def ingest_directory(self, dir_path: Path) -> dict[str, Any]:
        """
        Recursively ingests all paper JSON files inside a directory into PostgreSQL.
        Ensures Question Papers are ingested before Mark Schemes to maintain logical consistency.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Ensure schema tables and indexes exist
        self.db.init_schema()

        started_at = datetime.utcnow()

        all_json = list(dir_path.glob("**/*.json"))
        # Exclude manifest files (manifest.json, scrape_manifest.json, etc.)
        paper_jsons = [j for j in all_json if not j.name.endswith("manifest.json")]

        # Sort so that _qp_ files are processed before _ms_ files
        def sort_key(p: Path) -> tuple[int, str]:
            # 0 for qp, 1 for ms, 2 for other
            priority = 0 if "_qp_" in p.name else (1 if "_ms_" in p.name else 2)
            return (priority, p.name)

        sorted_jsons = sorted(paper_jsons, key=sort_key)

        total_questions = 0
        total_answers = 0
        total_visuals = 0
        successful_papers = 0
        failed_papers = []

        for jf in sorted_jsons:
            try:
                res = self.ingest_paper_json(jf)
                successful_papers += 1
                total_questions += res.get("questions_ingested", 0)
                total_answers += res.get("answers_ingested", 0)
                total_visuals += res.get("visuals_ingested", 0)
            except Exception as e:
                logger.error(f"Failed to ingest {jf.name} into PostgreSQL: {e}")
                failed_papers.append({"file": jf.name, "error": str(e)})

        completed_at = datetime.utcnow()

        # Log this pipeline run in pipeline_runs table
        subject_codes = list({
            jf.name.split("_")[0] for jf in sorted_jsons
            if jf.name and jf.name[0].isdigit()
        })
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO pipeline_runs
                            (stage, subject_code, year, status, details, started_at, completed_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            "ingest_directory",
                            ",".join(subject_codes) if subject_codes else None,
                            None,
                            "completed" if not failed_papers else "completed_with_errors",
                            json.dumps({
                                "dir": str(dir_path),
                                "total_files": len(sorted_jsons),
                                "successful_papers": successful_papers,
                                "failed_papers": len(failed_papers),
                                "total_questions": total_questions,
                                "total_answers": total_answers,
                                "total_visuals": total_visuals,
                                "failures": failed_papers,
                            }),
                            started_at,
                            completed_at,
                        ),
                    )
        except Exception as e:
            logger.warning(f"Could not log pipeline run: {e}")

        return {
            "total_files": len(sorted_jsons),
            "successful_papers": successful_papers,
            "failed_papers": len(failed_papers),
            "total_questions": total_questions,
            "total_answers": total_answers,
            "total_visuals": total_visuals,
            "failures": failed_papers,
        }

