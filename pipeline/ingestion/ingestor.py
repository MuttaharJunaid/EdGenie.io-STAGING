import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.ingestion.firebase_storage import FirebaseStorageManager
from pipeline.ingestion.mongo_client import MongoDBClient

logger = logging.getLogger(__name__)


class MongoIngestor:
    """
    Stage 4 MongoIngestor:
    - Uploads PDFs & extracted visual images to Firebase Storage with prefix 'https://assets.edgenie.io'
    - Constructs document models and upserts into MongoDB collections:
        1. subjects
        2. papers
        3. questions
        4. answers
        5. visuals
        6. pipeline_runs
    """

    def __init__(
        self,
        mongo_client: Optional[MongoDBClient] = None,
        storage_manager: Optional[FirebaseStorageManager] = None,
    ):
        self.mongo = mongo_client or MongoDBClient()
        self.storage = storage_manager or FirebaseStorageManager()

    def ingest_paper_json(self, json_path: Path, force: bool = False) -> Dict[str, Any]:
        """
        Ingests a single Stage 2 paper JSON file and its Stage 3 visual manifest into MongoDB
        and syncs assets to Firebase Storage (https://assets.edgenie.io).
        """
        json_path = Path(json_path).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            records: List[Dict[str, Any]] = json.load(f)

        if not records:
            logger.warning(f"No question records found in {json_path.name}")
            return {"status": "empty", "ingested": 0}

        sample = records[0]
        paper_code = sample.get("PaperCode") or json_path.stem
        subject_code = str(sample.get("SubjectCode", paper_code.split("_")[0]))
        year = sample.get("Year", 2010)
        session = sample.get("Session", "jun")

        # 1. Sync raw PDF to Firebase Storage
        pdf_path = json_path.with_suffix(".pdf")
        pdf_cdn_url = None
        if pdf_path.exists():
            pdf_cdn_url = self.storage.upload_pdf(pdf_path)
        else:
            pdf_cdn_url = self.storage.build_cdn_url(f"pdfs/{subject_code}/{year}/{session}/{paper_code}.pdf")

        # 2. Sync visual manifest and images to Firebase Storage
        visuals_dir = json_path.parent / json_path.stem
        visual_manifest_path = visuals_dir / "manifest.json"
        visual_entries: List[Dict[str, Any]] = []

        if visual_manifest_path.exists():
            with open(visual_manifest_path, "r", encoding="utf-8") as mf:
                visual_entries = json.load(mf)

        # 3. Create MongoDB Paper Document
        paper_doc = {
            "_id": paper_code,
            "paper_code": paper_code,
            "subject_code": subject_code,
            "year": year,
            "session": session,
            "paper_number": sample.get("PaperNumber"),
            "variant": sample.get("Variant"),
            "pdf_filename": pdf_path.name if pdf_path.exists() else f"{paper_code}.pdf",
            "pdf_cdn_url": pdf_cdn_url,
            "question_count": len(records),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.mongo.upsert_document("papers", {"_id": paper_code}, paper_doc)

        # 4. Create MongoDB Question & Visual Documents
        ingested_questions = 0
        for rec in records:
            qnum_str = rec.get("QuestionNumber", "Q1")
            qnum = int(qnum_str.split("Q")[-1]) if "Q" in qnum_str else 1
            q_key = f"{paper_code}_Q{qnum}"

            # Format image CDN URL if visual exists
            img_rel = rec.get("image")
            image_cdn_url = None
            if img_rel:
                image_name = Path(img_rel).name
                local_img_path = json_path.parent / img_rel
                if local_img_path.exists():
                    image_cdn_url = self.storage.upload_visual(local_img_path, paper_code)
                else:
                    image_cdn_url = self.storage.build_cdn_url(f"visuals/{paper_code}/{image_name}")

            q_doc = {
                "_id": q_key,
                "question_key": q_key,
                "paper_code": paper_code,
                "subject_code": subject_code,
                "year": year,
                "session": session,
                "question_number": qnum,
                "question_text": rec.get("QuestionText", ""),
                "marks": rec.get("Marks"),
                "topic_number": rec.get("Topic"),
                "topic_name": rec.get("Topic_name"),
                "sub_topics": rec.get("Sub_topics", []),
                "difficulty_score": rec.get("Difficulty_score"),
                "difficulty_label": rec.get("Difficulty_label"),
                "command_words": rec.get("Command_words", []),
                "image_filename": Path(img_rel).name if img_rel else None,
                "image_cdn_url": image_cdn_url,
                "created_at": datetime.utcnow().isoformat(),
            }
            self.mongo.upsert_document("questions", {"_id": q_key}, q_doc)

            # Ingest Answer Document if present
            if "Answer" in rec or "Answer_text" in rec:
                ans_doc = {
                    "_id": f"{q_key}_ans",
                    "question_id": q_key,
                    "paper_code": paper_code,
                    "subject_code": subject_code,
                    "year": year,
                    "session": session,
                    "question_number": qnum,
                    "answer_text": rec.get("Answer", rec.get("Answer_text")),
                    "updated_at": datetime.utcnow().isoformat(),
                }
                self.mongo.upsert_document("answers", {"_id": f"{q_key}_ans"}, ans_doc)

            ingested_questions += 1

        # 5. Ingest Visual Documents
        for v in visual_entries:
            v_id = f"{paper_code}_{v.get('filename')}"
            v_img_path = visuals_dir / v.get("filename")
            v_cdn_url = (
                self.storage.upload_visual(v_img_path, paper_code)
                if v_img_path.exists()
                else self.storage.build_cdn_url(f"visuals/{paper_code}/{v.get('filename')}")
            )

            v_doc = {
                "_id": v_id,
                "paper_code": paper_code,
                "subject_code": subject_code,
                "year": year,
                "session": session,
                "question_number": v.get("question"),
                "caption": v.get("caption"),
                "page": v.get("page"),
                "bbox": v.get("bbox"),
                "filename": v.get("filename"),
                "cdn_url": v_cdn_url,
            }
            self.mongo.upsert_document("visuals", {"_id": v_id}, v_doc)

        logger.info(f"Ingested {paper_code}: {ingested_questions} questions, {len(visual_entries)} visuals into MongoDB.")
        return {
            "paper_code": paper_code,
            "questions_ingested": ingested_questions,
            "visuals_ingested": len(visual_entries),
            "pdf_cdn_url": pdf_cdn_url,
        }

    def ingest_directory(self, dir_path: Path, force: bool = False) -> Dict[str, Any]:
        """
        Ingests all paper JSON files inside a directory tree into MongoDB.
        """
        dir_path = Path(dir_path).resolve()
        json_files = list(dir_path.glob("**/*.json"))
        # Exclude manifest.json files
        json_files = [j for j in json_files if j.name != "manifest.json"]

        results = []
        for jf in json_files:
            try:
                res = self.ingest_paper_json(jf, force=force)
                results.append(res)
            except Exception as e:
                logger.error(f"Error ingesting {jf.name}: {e}")

        return {
            "total_papers": len(json_files),
            "successfully_ingested": len(results),
            "details": results,
        }
