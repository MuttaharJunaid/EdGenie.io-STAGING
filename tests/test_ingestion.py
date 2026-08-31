import json
import pytest
from pathlib import Path
from pipeline.ingestion.postgres_client import PostgresClient
from pipeline.ingestion.postgres_ingestor import PostgresIngestor
from pipeline.ingestion.firebase_storage import FirebaseStorageManager


def test_parse_paper_filename():
    ingestor = PostgresIngestor()
    meta = ingestor.parse_paper_filename("5054_s25_qp_22.json")
    assert meta["subject_code"] == "5054"
    assert meta["year"] == 2025
    assert meta["session"] == "jun"
    assert meta["paper_type"] == "qp"
    assert meta["variant"] == "22"
    assert meta["paper_number"] == 2

    meta_w18 = ingestor.parse_paper_filename("5054_w18_ms_11.json")
    assert meta_w18["subject_code"] == "5054"
    assert meta_w18["year"] == 2018
    assert meta_w18["session"] == "nov"
    assert meta_w18["paper_type"] == "ms"
    assert meta_w18["variant"] == "11"
    assert meta_w18["paper_number"] == 1


def test_firebase_storage_manager_urls():
    fsm = FirebaseStorageManager(cdn_prefix="https://assets.edgenie.io")
    url = fsm.build_cdn_url("pdfs/5054/2025/jun/5054_s25_qp_22.pdf")
    assert url == "https://assets.edgenie.io/pdfs/5054/2025/jun/5054_s25_qp_22.pdf"

    visual_url = fsm.build_cdn_url("visuals/5054_s25_qp_22/p5_q3_Fig_3_1.png")
    assert visual_url == "https://assets.edgenie.io/visuals/5054_s25_qp_22/p5_q3_Fig_3_1.png"


def test_postgres_schema_init_and_ingestion(tmp_path):
    client = PostgresClient()
    client.init_schema()

    # Create a mock QP JSON
    qp_data = [
        {
            "QuestionNumber": "Jun/22/2025/Q1",
            "Question_text": "A mock physics question on scalars and vectors.",
            "Unit": "1",
            "Topic": "1.1",
            "Topic_name": "Physical quantities",
            "Sub_topic": ["Scalars and vectors"],
            "Difficulty_score": 3,
            "Difficulty_label": "easy",
            "Command_words": ["state", "calculate"],
            "Marks": 4,
            "image": "mock_5054_s25_qp_22/p2_q1_Fig_1_1.png",
            "Parts": [
                {"part_label": "(a)", "part_text": "Define scalar.", "marks": 2, "image": None},
                {"part_label": "(b)", "part_text": "Calculate resultant.", "marks": 2, "image": None},
            ],
        }
    ]

    qp_file = tmp_path / "mock_5054_s25_qp_22.json"
    with open(qp_file, "w", encoding="utf-8") as f:
        json.dump(qp_data, f)

    # Create mock visual manifest
    vis_dir = tmp_path / "mock_5054_s25_qp_22"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_manifest = [
        {
            "page": 2,
            "question": 1,
            "caption": "Fig. 1.1",
            "bbox": [100.0, 200.0, 400.0, 500.0],
            "filename": "p2_q1_Fig_1_1.png",
            "file": "p2_q1_Fig_1_1.png",
            "relative_path": "mock_5054_s25_qp_22/p2_q1_Fig_1_1.png",
        }
    ]
    with open(vis_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(vis_manifest, f)

    # Create a mock MS JSON
    ms_data = [
        {
            "AnswerNumber": "Jun/22/2025/Q1",
            "Answer_text": "Scalar has magnitude only.",
            "image": None,
            "Parts": [
                {"part_label": "(a)", "part_answer": "magnitude only", "image": None},
                {"part_label": "(b)", "part_answer": "5.0 N", "image": None},
            ],
        }
    ]
    ms_file = tmp_path / "mock_5054_s25_ms_22.json"
    with open(ms_file, "w", encoding="utf-8") as f:
        json.dump(ms_data, f)

    ingestor = PostgresIngestor(db_client=client)

    # Ingest QP
    res_qp = ingestor.ingest_paper_json(qp_file)
    assert res_qp["questions_ingested"] == 1
    assert res_qp["visuals_ingested"] == 1

    # Ingest MS
    res_ms = ingestor.ingest_paper_json(ms_file)
    assert res_ms["answers_ingested"] == 1

    # Ingest Directory
    res_dir = ingestor.ingest_directory(tmp_path)
    assert res_dir["successful_papers"] == 2
    assert res_dir["total_questions"] == 1
    assert res_dir["total_answers"] == 1
    assert res_dir["total_visuals"] == 1
