"""
tests/test_unified_pipeline.py

Comprehensive test suite for the Unified 3-in-1 Pipeline:
Text Extraction, Visual Extraction, and Syllabus Classification.

Covers:
- Paper type detection across core subjects (Physics, Accounting, Economics, Chemistry, Biology, Maths D, Business, English).
- CAIE filename parsing for various sessions and paper numbers.
- Syllabus resolution and CSV loading for core subjects (3-column and 5-column formats).
- OpenAI strict schema generation for Essay & MCQ containers.
- Prompt building with inline syllabus injection for core subjects.
- Blank page pruning on real PDFs.
- Result parsing and async job polling.
- PyMuPDF visual element clipping, artifact filtering, and caption expansion.
- Manifest generation and record patching.
- Mark hierarchy validation and reconciliation.
- Unified CLI command verification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest
from typer.testing import CliRunner

from pipeline.unified.cli import app
from pipeline.unified.manifest_patcher import (
    is_image_compatible_with_question,
    patch_records_with_manifest,
)
from pipeline.unified.marks_validator import MarkHierarchyValidator
from pipeline.unified.paper_type_detector import (
    PaperCategory,
    PaperTypeDetector,
    parse_caie_filename,
)
from pipeline.unified.pipeline import (
    UnifiedPipeline,
    get_next_run_number,
    resolve_timed_out_jobs_path,
)
from pipeline.unified.schemas import (
    EssayAnswersContainer,
    EssayQuestionsContainer,
    MCQAnswersContainer,
    MCQQuestionsContainer,
    to_strict_openai_schema,
)
from pipeline.unified.syllabus_loader import SyllabusLoader

runner = CliRunner()


# ── 1. Paper Type Detection & CAIE Filename Parsing ─────────────────────────

@pytest.mark.parametrize(
    "filename,expected_cat",
    [
        # Physics 5054
        ("5054_s23_qp_22.pdf", PaperCategory.ESSAY_QP),
        ("5054_s23_qp_12.pdf", PaperCategory.MCQ_QP),
        ("5054_s23_ms_22.pdf", PaperCategory.ESSAY_MS),
        ("5054_s23_ms_12.pdf", PaperCategory.MCQ_MS),
        # Accounting 7707
        ("7707_s25_qp_21.pdf", PaperCategory.ESSAY_QP),
        ("7707_s25_qp_11.pdf", PaperCategory.MCQ_QP),
        ("7707_s25_ms_21.pdf", PaperCategory.ESSAY_MS),
        ("7707_s25_ms_11.pdf", PaperCategory.MCQ_MS),
        # Economics 2281
        ("2281_s23_qp_22.pdf", PaperCategory.ESSAY_QP),
        ("2281_s23_qp_12.pdf", PaperCategory.MCQ_QP),
        ("2281_w22_ms_21.pdf", PaperCategory.ESSAY_MS),
        ("2281_w22_ms_11.pdf", PaperCategory.MCQ_MS),
        # Chemistry 5070
        ("5070_s23_qp_22.pdf", PaperCategory.ESSAY_QP),
        ("5070_s23_qp_11.pdf", PaperCategory.MCQ_QP),
        # Biology 5090
        ("5090_s23_qp_21.pdf", PaperCategory.ESSAY_QP),
        ("5090_s23_qp_12.pdf", PaperCategory.MCQ_QP),
        # Mathematics D 4024
        ("4024_s23_qp_21.pdf", PaperCategory.ESSAY_QP),
        ("4024_s23_qp_12.pdf", PaperCategory.MCQ_QP),
        # Business Studies 7115
        ("7115_s26_qp_21.pdf", PaperCategory.ESSAY_QP),
        ("7115_s26_qp_11.pdf", PaperCategory.MCQ_QP),
    ],
)
def test_paper_type_detection_by_filename(filename: str, expected_cat: PaperCategory):
    detected = PaperTypeDetector.detect_by_filename(filename)
    assert detected == expected_cat


def test_parse_caie_filename_metadata():
    # May/June variant
    meta1 = parse_caie_filename("5054_s23_qp_22.pdf")
    assert meta1["subject_code"] == "5054"
    assert meta1["year"] == 2023
    assert meta1["session"] == "jun"
    assert meta1["variant"] == "22"
    assert meta1["paper_number"] == 2
    assert meta1["doc_type"] == "qp"

    # Oct/Nov variant
    meta2 = parse_caie_filename("7707_w24_ms_11.pdf")
    assert meta2["subject_code"] == "7707"
    assert meta2["year"] == 2024
    assert meta2["session"] == "nov"
    assert meta2["variant"] == "11"
    assert meta2["paper_number"] == 1
    assert meta2["doc_type"] == "ms"

    # March session
    meta3 = parse_caie_filename("0580_m22_qp_42.pdf")
    assert meta3["subject_code"] == "0580"
    assert meta3["year"] == 2022
    assert meta3["session"] == "mar"
    assert meta3["variant"] == "42"


def test_paper_type_detection_on_real_sample_pdfs():
    # If storage files exist, test content-based detection
    sample_essay = Path("pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf")
    if sample_essay.exists():
        cat = PaperTypeDetector.detect(sample_essay)
        assert cat == PaperCategory.ESSAY_QP

    sample_accounting = Path("pipeline/storage/7707/2025/jun/7707_s25_qp_21.pdf")
    if sample_accounting.exists():
        cat = PaperTypeDetector.detect(sample_accounting)
        assert cat == PaperCategory.ESSAY_QP


# ── 2. Syllabus Resolution & Loading for Core Subjects ──────────────────────

@pytest.mark.parametrize(
    "code,expected_csv_name",
    [
        ("5054", "PHYSICS_5054.csv"),
        ("7707", "ACCOUNTING_7707.csv"),
        ("2281", "ECONOMICS_2281.csv"),
        ("5070", "CHEMISTRY_5070.csv"),
        ("5090", "BIOLOGY_5090.csv"),
        ("4024", "MATHEMATICS_4024.csv"),
        ("7115", "BUSINESS_7115.csv"),
        ("1123", "ENGLISH_1123.csv"),
    ],
)
def test_syllabus_resolution_for_core_subjects(code: str, expected_csv_name: str):
    resolved = SyllabusLoader.resolve_syllabus_path(None, subject_code=code)
    assert resolved is not None
    assert resolved.name == expected_csv_name
    assert resolved.exists()


def test_syllabus_loading_and_formatting():
    csv_path = Path("syllabi/PHYSICS_5054.csv")
    if csv_path.exists():
        units = SyllabusLoader.load_syllabus(csv_path)
        assert len(units) > 0
        assert units[0].unit_number != ""

        formatted = SyllabusLoader.format_syllabus_for_prompt(units)
        assert "Unit" in formatted
        assert "Topic" in formatted
        assert "Sub-topics:" in formatted


# ── 3. Strict OpenAI Schema Generation ──────────────────────────────────────

def test_strict_openai_schema_essay_qp():
    schema = to_strict_openai_schema(EssayQuestionsContainer)
    assert schema.get("type") == "object"
    assert schema["additionalProperties"] is False
    assert "questions" in schema["required"]

    for def_val in schema.get("$defs", {}).values():
        if def_val.get("type") == "object":
            assert def_val["additionalProperties"] is False
            assert len(def_val.get("required", [])) == len(def_val.get("properties", {}))


def test_strict_openai_schema_mcq_qp():
    schema = to_strict_openai_schema(MCQQuestionsContainer)
    assert schema["additionalProperties"] is False
    assert "questions" in schema["required"]

    for def_val in schema.get("$defs", {}).values():
        if def_val.get("type") == "object":
            assert def_val["additionalProperties"] is False
            assert len(def_val.get("required", [])) == len(def_val.get("properties", {}))

    options_def = schema["$defs"]["MCQOptions"]
    assert set(options_def["required"]) == {"A", "B", "C", "D"}



def test_strict_openai_schema_mark_schemes():
    schema_essay_ms = to_strict_openai_schema(EssayAnswersContainer)
    assert schema_essay_ms["additionalProperties"] is False
    assert "answers" in schema_essay_ms["properties"]

    schema_mcq_ms = to_strict_openai_schema(MCQAnswersContainer)
    assert schema_mcq_ms["additionalProperties"] is False
    assert "answers" in schema_mcq_ms["properties"]


# ── 4. Prompt Building with Core Subject Syllabus Reference ─────────────────

def test_build_prompt_essay_qp_with_syllabus():
    pipeline = UnifiedPipeline()
    meta = {"year": 2023, "session": "jun", "variant": "22"}
    syllabus_sample = "Unit 1 (General Physics) -> Topic 1.1 (Measurement) | Sub-topics: Length"

    schema, prompt = pipeline._build_schema_and_prompt(
        PaperCategory.ESSAY_QP, meta, syllabus_sample
    )

    assert "May/June 2023, variant 22" in prompt
    assert "SYLLABUS REFERENCE" in prompt
    assert "Topic 1.1 (Measurement)" in prompt
    assert "EXTRACTION RULES" in prompt
    assert "CLASSIFICATION RULES" in prompt


def test_build_prompt_mcq_qp_with_syllabus():
    pipeline = UnifiedPipeline()
    meta = {"year": 2025, "session": "jun", "variant": "11"}
    syllabus_sample = "Unit 2 (Accounting) -> Topic 2.1 (Double Entry) | Sub-topics: Ledgers"

    schema, prompt = pipeline._build_schema_and_prompt(
        PaperCategory.MCQ_QP, meta, syllabus_sample
    )

    assert "Options" in prompt
    assert "Marks is always 1" in prompt
    assert "Topic 2.1 (Double Entry)" in prompt


def test_build_prompt_mark_schemes():
    pipeline = UnifiedPipeline()
    meta = {"year": 2024, "session": "nov", "variant": "21"}

    _, prompt_essay_ms = pipeline._build_schema_and_prompt(
        PaperCategory.ESSAY_MS, meta, ""
    )
    assert "mark scheme answers" in prompt_essay_ms.lower()

    _, prompt_mcq_ms = pipeline._build_schema_and_prompt(
        PaperCategory.MCQ_MS, meta, ""
    )
    assert "answer key" in prompt_mcq_ms.lower()


# ── 5. Blank Page Pruning on Real Exam Papers ───────────────────────────────

def test_prune_pdf_bytes_on_sample_paper():
    sample_pdf = Path("pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf")
    if sample_pdf.exists():
        pruned_bytes = UnifiedPipeline._prune_pdf_bytes(sample_pdf)
        if pruned_bytes:
            doc_orig = fitz.open(str(sample_pdf))
            doc_pruned = fitz.open(stream=pruned_bytes, filetype="pdf")
            # Page count of pruned doc must be <= original doc
            assert len(doc_pruned) <= len(doc_orig)
            # Cover page (page 0) is preserved
            assert len(doc_pruned) > 0


# ── 6. Result Parsing: Structured Records & Layout Elements ─────────────────

def test_parse_text_records_and_elements():
    pipeline = UnifiedPipeline()

    mock_job = MagicMock()
    mock_job.result.extracted_data = [
        MagicMock(
            data={
                "questions": [
                    {
                        "QuestionNumber": "May/June/22/2023/Q1",
                        "Question_text": "A student measures acceleration.",
                        "Marks": 5,
                        "Unit": "1",
                        "Topic": "1.2",
                        "Topic_name": "Kinematics",
                        "Difficulty_score": 6,
                        "Difficulty_label": "Medium",
                        "Command_words": ["Calculate"],
                    }
                ]
            }
        )
    ]
    mock_job.result.elements = [
        {
            "type": "Image",
            "metadata": {
                "page_number": 2,
                "coordinates": {
                    "points": [[20, 20], [100, 20], [100, 100], [20, 100]],
                    "layout_width": 200,
                    "layout_height": 200,
                },
            },
            "text": "Fig. 1.1",
        }
    ]

    records = pipeline._parse_text_records(mock_job)
    elements = pipeline._parse_elements(mock_job)

    assert len(records) == 1
    assert records[0]["QuestionNumber"] == "May/June/22/2023/Q1"
    assert records[0]["Topic"] == "1.2"
    assert records[0]["Marks"] == 5

    assert len(elements) == 1
    assert elements[0]["type"] == "Image"
    assert elements[0]["metadata"]["page_number"] == 2


# ── 7. Visual Elements Artifact Filtering ────────────────────────────────────

def test_artifact_filtering():
    pipeline = UnifiedPipeline()
    page_rect = fitz.Rect(0, 0, 595, 842)

    # 1. Page 1 (cover page) element -> always filtered
    el_cover = {"type": "Image", "text": "Cambridge Assessment"}
    raw_bbox = fitz.Rect(50, 50, 200, 200)
    assert pipeline._is_unwanted_artifact(el_cover, 1, raw_bbox, page_rect) is True

    # 2. Header margin artifact (y <= 0.09)
    el_header = {"type": "Image", "text": "Barcode"}
    bbox_top = fitz.Rect(50, 10, 200, 50)  # y1 = 50 / 842 = 0.059 <= 0.09
    assert pipeline._is_unwanted_artifact(el_header, 2, bbox_top, page_rect) is True

    # 3. Footer margin artifact (y >= 0.90)
    el_footer = {"type": "Image", "text": "Timing mark"}
    bbox_bottom = fitz.Rect(50, 780, 200, 830)  # y0 = 780 / 842 = 0.926 >= 0.90
    assert pipeline._is_unwanted_artifact(el_footer, 2, bbox_bottom, page_rect) is True

    # 4. Legitimate diagram in the middle of page 2
    el_diag = {"type": "Image", "text": "Apparatus"}
    bbox_valid = fitz.Rect(100, 200, 400, 500)
    assert pipeline._is_unwanted_artifact(el_diag, 2, bbox_valid, page_rect) is False


# ── 8. Manifest Linking & Record Patching ───────────────────────────────────

def test_manifest_patching():
    records = [
        {
            "QuestionNumber": "May/June/22/2023/Q1",
            "Question_text": "Look at the diagram.",
            "Parts": [
                {
                    "part_label": "a",
                    "part_text": "Identify part A.",
                    "sub_parts": [],
                }
            ],
        },
        {
            "QuestionNumber": "May/June/22/2023/Q2",
            "Question_text": "Calculate power.",
            "Parts": [],
        },
    ]

    manifest = [
        {
            "question": 1,
            "target_type": "question",
            "relative_path": "5054_s23_qp_22/p2_q1_question_diagram_1.png",
        },
        {
            "question": 1,
            "target_type": "part",
            "target_label": "a",
            "relative_path": "5054_s23_qp_22/p2_q1_part_a_diagram_1.png",
        },
        {
            "question": 2,
            "target_type": "question",
            "relative_path": "5054_s23_qp_22/p4_q2_question_diagram_1.png",
        },
    ]

    patched_count = patch_records_with_manifest(records, manifest)
    assert patched_count >= 3

    assert records[0]["image"] == "5054_s23_qp_22/p2_q1_question_diagram_1.png"
    assert records[0]["Parts"][0]["image"] == "5054_s23_qp_22/p2_q1_part_a_diagram_1.png"
    assert records[1]["image"] == "5054_s23_qp_22/p4_q2_question_diagram_1.png"


def test_image_question_compatibility_guard():
    assert is_image_compatible_with_question("p2_q1_question_diagram_1.png", 1) is True
    assert is_image_compatible_with_question("p2_q1_question_diagram_1.png", 2) is False
    assert is_image_compatible_with_question("p4_q3_part_b_Fig_3_1.png", 3) is True
    assert is_image_compatible_with_question("p4_q3_part_b_Fig_3_1.png", 4) is False


# ── 9. Mark Hierarchy Validation & Reconciliation ───────────────────────────

def test_mark_hierarchy_validation():
    # Question with leaf parts having trailing [N]
    record = {
        "QuestionNumber": "May/June/22/2023/Q1",
        "Question_text": "State the principle of conservation of energy. [Total: 8]",
        "Marks": 0,
        "Parts": [
            {
                "part_label": "a",
                "part_text": "Define kinetic energy. [2]",
                "marks": 1,
                "sub_parts": [],
            },
            {
                "part_label": "b",
                "part_text": "A ball falls through a height h.",
                "marks": 1,
                "sub_parts": [
                    {
                        "sub_part_label": "(i)",
                        "sub_part_text": "Calculate velocity. [3]",
                        "marks": 1,
                    },
                    {
                        "sub_part_label": "(ii)",
                        "sub_part_text": "Explain energy transfer. [3]",
                        "marks": 1,
                    },
                ],
            },
        ],
    }

    reconciled, stats = MarkHierarchyValidator.reconcile_and_validate_records([record])
    q = reconciled[0]

    # Part a should have 2 marks
    assert q["Parts"][0]["marks"] == 2
    # Sub-parts should have 3 marks each
    assert q["Parts"][1]["sub_parts"][0]["marks"] == 3
    assert q["Parts"][1]["sub_parts"][1]["marks"] == 3
    # Part b should be sum of sub-parts = 6
    assert q["Parts"][1]["marks"] == 6
    # Total question Marks should be 2 + 6 = 8
    assert q["Marks"] == 8
    # [Total: 8] stripped from text
    assert "[Total:" not in q["Question_text"]


def test_mcq_marks_strictly_one():
    mcq_records = [
        {"QuestionNumber": "May/June/12/2023/Q1", "Options": {"A": "1", "B": "2", "C": "3", "D": "4"}, "Marks": 0},
        {"QuestionNumber": "May/June/12/2023/Q2", "Options": {"A": "5", "B": "6", "C": "7", "D": "8"}, "Marks": 5},
    ]
    reconciled, stats = MarkHierarchyValidator.reconcile_and_validate_records(mcq_records)
    for r in reconciled:
        assert r["Marks"] == 1
    assert stats["total_paper_marks"] == 2


# ── 10. Unified CLI Subcommand Tests ────────────────────────────────────────

def test_unified_cli_help():
    res = runner.invoke(app, ["extract-visuals-text-and-classify", "--help"])
    assert res.exit_code == 0
    assert "--pdf" in res.output
    assert "--syllabus" in res.output
    assert "--dpi" in res.output
    assert "--force" in res.output


def test_unified_cli_missing_pdf_fails():
    res = runner.invoke(app, ["extract-visuals-text-and-classify", "--pdf", "non_existent_file.pdf"])
    assert res.exit_code != 0


def test_unified_cli_extract_alias_help():
    res = runner.invoke(app, ["extract", "--help"])
    assert res.exit_code == 0
    assert "--pdf" in res.output
    assert "--syllabus" in res.output
    assert "--dpi" in res.output
    assert "--force" in res.output


def test_unified_cli_extract_dir_help():
    res = runner.invoke(app, ["extract-dir", "--help"])
    assert res.exit_code == 0
    assert "--dir" in res.output
    assert "--type" in res.output
    assert "--syllabus" in res.output
    assert "--force" in res.output


def test_unified_cli_extract_dir_missing_dir():
    res = runner.invoke(app, ["extract-dir", "--dir", "non_existent_dir_12345"])
    assert res.exit_code != 0
    assert "Directory not found" in res.output


def test_unified_cli_extract_dir_empty_or_non_exam(tmp_path):
    # Directory with only examiner report and grade thresholds
    er_file = tmp_path / "5054_s23_er.pdf"
    er_file.write_bytes(b"%PDF-1.4 mock examiner report")
    gt_file = tmp_path / "5054_s23_gt.pdf"
    gt_file.write_bytes(b"%PDF-1.4 mock grade threshold")

    import os
    os.environ["UNSTRUCTURED_API_KEY"] = "mock-key"

    res = runner.invoke(app, ["extract-dir", "--dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "No exam question papers or mark schemes found" in res.output


# ── 11. Timeout Resilience & Endpoint Recovery Tests ────────────────────────

def test_job_timeout_error_properties():
    from pipeline.unified import JobTimeoutError

    err = JobTimeoutError(job_id="job-abc-123", message="Polling timed out", elapsed=120.5)
    assert err.job_id == "job-abc-123"
    assert err.elapsed == 120.5
    assert "Polling timed out" in str(err)
    assert isinstance(err, TimeoutError)


def test_fetch_job_result_from_endpoint(monkeypatch):
    pipeline = UnifiedPipeline(api_key="mock-key")
    test_job_id = "b60382ec-1f13-4a8d-9946-d7171d977e9a"

    expected_url = f"https://transform.unstructured.io/playground/jobs/{test_job_id}?output=elements"
    captured_urls: list[str] = []

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "status": "completed",
                "extracted_data": [
                    {
                        "data": {
                            "questions": [
                                {
                                    "QuestionNumber": "May/June/22/2023/Q1",
                                    "Question_text": "What is speed? [1]",
                                    "Marks": 1,
                                }
                            ]
                        }
                    }
                ],
                "elements": [],
            }

    def mock_get(url, headers, timeout):
        captured_urls.append(url)
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

    result = pipeline.fetch_job_result_from_endpoint(test_job_id)
    assert result is not None
    assert captured_urls[0] == expected_url
    assert result["status"] == "completed"
    records = pipeline._parse_text_records(result)
    assert len(records) == 1
    assert records[0]["QuestionNumber"] == "May/June/22/2023/Q1"


def test_complete_pipeline_work(tmp_path):
    pipeline = UnifiedPipeline(api_key="mock-key")
    pdf_path = tmp_path / "5054_s23_qp_22.pdf"
    # Create empty valid 1-page PDF
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    output_dir = tmp_path / "5054_s23_qp_22"

    job_result = {
        "status": "completed",
        "extracted_data": [
            {
                "data": {
                    "questions": [
                        {
                            "QuestionNumber": "May/June/22/2023/Q1",
                            "Question_text": "Explain thermal expansion. [2]",
                            "Marks": 2,
                            "Parts": [],
                        }
                    ]
                }
            }
        ],
        "elements": [],
    }

    res = pipeline.complete_pipeline_work(
        pdf_path=pdf_path,
        output_dir=output_dir,
        job_result=job_result,
        category=PaperCategory.ESSAY_QP,
    )

    assert "records" in res
    assert len(res["records"]) == 1
    assert res["records"][0]["Marks"] == 2
    assert res["json_path"].exists()
    assert (tmp_path / "5054_s23_qp_22_unified_text.json").exists()


def test_recover_timed_out_jobs(tmp_path, monkeypatch):
    jobs_file = tmp_path / "timed_out_jobs.json"
    pdf_path = tmp_path / "5054_s23_qp_22.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    output_dir = tmp_path / "5054_s23_qp_22"

    initial_jobs = [
        {
            "job_id": "job-recovered-001",
            "pdf_path": str(pdf_path),
            "output_dir": str(output_dir),
            "category": PaperCategory.ESSAY_QP.value,
            "status": "timed_out",
        }
    ]
    import json
    jobs_file.write_text(json.dumps(initial_jobs, indent=2))

    pipeline = UnifiedPipeline(api_key="mock-key")

    mock_job_result = {
        "status": "completed",
        "extracted_data": [
            {
                "data": {
                    "questions": [
                        {
                            "QuestionNumber": "May/June/22/2023/Q1",
                            "Question_text": "Define force. [1]",
                            "Marks": 1,
                        }
                    ]
                }
            }
        ],
        "elements": [],
    }

    monkeypatch.setattr(pipeline, "fetch_job_result_from_endpoint", lambda j, **kwargs: mock_job_result)

    recovered = pipeline.recover_timed_out_jobs(jobs_file=jobs_file)
    assert len(recovered) == 1
    assert recovered[0]["job_id"] == "job-recovered-001"
    assert recovered[0]["status"] == "recovered"
    assert recovered[0]["records_count"] == 1

    # Verify status in file was updated to completed
    updated_jobs = json.loads(jobs_file.read_text())
    assert updated_jobs[0]["status"] == "completed"


def test_unified_cli_recover_commands_help():
    res_rec = runner.invoke(app, ["recover-job", "--help"])
    assert res_rec.exit_code == 0
    assert "--job-id" in res_rec.output
    assert "--pdf" in res_rec.output

    res_all = runner.invoke(app, ["recover-timed-out", "--help"])
    assert res_all.exit_code == 0
    assert "--file" in res_all.output


def test_prune_plan_and_element_page_remapping(tmp_path):
    # Create a 4-page PDF where page 3 is BLANK PAGE
    pdf_path = tmp_path / "test_blank_page_doc.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Cover Page")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "1 Question 1 text [2]")

    p3 = doc.new_page()
    p3.insert_text((50, 50), "BLANK PAGE")

    p4 = doc.new_page()
    p4.insert_text((50, 50), "2 Question 2 text [3]")

    doc.save(str(pdf_path))
    doc.close()

    kept_pages, page_map = UnifiedPipeline._get_prune_plan(pdf_path)
    # Kept pages should be 0, 1, 3 (page 2 is skipped)
    assert kept_pages == [0, 1, 3]
    # Pruned page 1 -> original page 1, pruned page 2 -> original page 2, pruned page 3 -> original page 4
    assert page_map == {1: 1, 2: 2, 3: 4}

    # Verify elements remapping
    elements = [
        {"type": "Image", "metadata": {"page_number": 2}, "text": "Fig. 1.1"},
        {"type": "Image", "metadata": {"page_number": 3}, "text": "Fig. 2.1"},
    ]
    remapped = UnifiedPipeline._remap_elements_pages(elements, page_map)
    assert remapped[0]["metadata"]["page_number"] == 2
    assert remapped[0]["metadata"]["pruned_page_number"] == 2
    assert remapped[1]["metadata"]["page_number"] == 4  # Remapped to real page 4!
    assert remapped[1]["metadata"]["pruned_page_number"] == 3

    # Idempotency check
    remapped_again = UnifiedPipeline._remap_elements_pages(remapped, page_map)
    assert remapped_again[1]["metadata"]["page_number"] == 4





# ── 12. Per-Run Timed-Out Jobs File Naming ──────────────────────────────────

class TestGetNextRunNumber:
    def test_starts_at_1_when_no_files_exist(self, tmp_path):
        assert get_next_run_number(tmp_path) == 1

    def test_increments_past_existing_files(self, tmp_path):
        (tmp_path / "timed_out_jobs_run_1.json").write_text("[]")
        (tmp_path / "timed_out_jobs_run_3.json").write_text("[]")
        assert get_next_run_number(tmp_path) == 4

    def test_reads_counter_file_when_no_run_files(self, tmp_path):
        (tmp_path / ".pipeline_run_counter").write_text("5")
        assert get_next_run_number(tmp_path) == 6

    def test_takes_max_of_counter_and_glob(self, tmp_path):
        (tmp_path / ".pipeline_run_counter").write_text("2")
        (tmp_path / "timed_out_jobs_run_7.json").write_text("[]")
        assert get_next_run_number(tmp_path) == 8

    def test_writes_updated_counter_file(self, tmp_path):
        run = get_next_run_number(tmp_path)
        counter = int((tmp_path / ".pipeline_run_counter").read_text())
        assert counter == run

    def test_monotonically_increments_across_calls(self, tmp_path):
        r1 = get_next_run_number(tmp_path)
        r2 = get_next_run_number(tmp_path)
        r3 = get_next_run_number(tmp_path)
        assert r1 < r2 < r3

    def test_ignores_malformed_counter_file(self, tmp_path):
        (tmp_path / ".pipeline_run_counter").write_text("not_a_number")
        assert get_next_run_number(tmp_path) == 1

    def test_alternative_suffix_pattern(self, tmp_path):
        (tmp_path / "timed_out_jobs_2.json").write_text("[]")
        # File uses pattern timed_out_jobs_<N>.json — should be detected
        result = get_next_run_number(tmp_path)
        assert result == 3

    def test_defaults_to_cwd_when_base_dir_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = get_next_run_number(None)
        assert result == 1


class TestResolveTimedOutJobsPath:
    """Tests for resolve_timed_out_jobs_path priority chain."""

    def test_explicit_file_absolute(self, tmp_path):
        explicit = tmp_path / "custom.json"
        result = resolve_timed_out_jobs_path(explicit_file=explicit, base_dir=tmp_path)
        assert result == explicit

    def test_explicit_file_relative_prefixed_with_base_dir(self, tmp_path):
        result = resolve_timed_out_jobs_path(explicit_file="myfile.json", base_dir=tmp_path)
        assert result == tmp_path / "myfile.json"

    def test_explicit_takes_priority_over_run_number(self, tmp_path):
        explicit = tmp_path / "override.json"
        result = resolve_timed_out_jobs_path(
            run_number=5, explicit_file=explicit, base_dir=tmp_path
        )
        assert result == explicit

    def test_suffix_option(self, tmp_path):
        result = resolve_timed_out_jobs_path(suffix="run_2", base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs_run_2.json"

    def test_suffix_leading_underscore_stripped(self, tmp_path):
        result = resolve_timed_out_jobs_path(suffix="_run_3", base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs_run_3.json"

    def test_run_number_option(self, tmp_path):
        result = resolve_timed_out_jobs_path(run_number=4, base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs_run_4.json"

    def test_run_number_as_string(self, tmp_path):
        result = resolve_timed_out_jobs_path(run_number="7", base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs_run_7.json"

    def test_run_number_already_prefixed_with_run(self, tmp_path):
        result = resolve_timed_out_jobs_path(run_number="run_2", base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs_run_2.json"

    def test_auto_increment_creates_sequential_files(self, tmp_path):
        p1 = resolve_timed_out_jobs_path(auto_increment=True, base_dir=tmp_path)
        p1.write_text("[]")  # simulate the file being created
        p2 = resolve_timed_out_jobs_path(auto_increment=True, base_dir=tmp_path)
        assert p1 != p2
        assert "run_" in p1.name
        assert "run_" in p2.name

    def test_fallback_to_default_filename(self, tmp_path):
        result = resolve_timed_out_jobs_path(base_dir=tmp_path)
        assert result == tmp_path / "timed_out_jobs.json"

    def test_suffix_overrides_run_number(self, tmp_path):
        result = resolve_timed_out_jobs_path(
            run_number=1, suffix="custom_run", base_dir=tmp_path
        )
        # suffix takes priority over run_number when explicit_file is absent
        assert result == tmp_path / "timed_out_jobs_custom_run.json"


class TestUnifiedPipelineRunNumberInit:
    """Tests that UnifiedPipeline.__init__ resolves timed_out_jobs_file correctly."""

    def test_default_is_fallback_filename(self):
        p = UnifiedPipeline(api_key="mock")
        assert p.timed_out_jobs_file.name == "timed_out_jobs.json"

    def test_explicit_timed_out_jobs_file(self, tmp_path):
        custom = tmp_path / "myrun.json"
        p = UnifiedPipeline(api_key="mock", timed_out_jobs_file=custom)
        assert p.timed_out_jobs_file == custom

    def test_run_number_param(self, tmp_path):
        p = UnifiedPipeline(api_key="mock", run_number=3,
                            timed_out_jobs_file=None)
        # resolve_timed_out_jobs_path with run_number=3 defaults to cwd
        assert p.timed_out_jobs_file.name == "timed_out_jobs_run_3.json"

    def test_suffix_param(self):
        p = UnifiedPipeline(api_key="mock", suffix="run_99")
        assert p.timed_out_jobs_file.name == "timed_out_jobs_run_99.json"

    def test_auto_increment_param(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = UnifiedPipeline(api_key="mock", auto_increment=True)
        assert "run_" in p.timed_out_jobs_file.name

    def test_init_timed_out_jobs_file_creates_empty_json(self, tmp_path):
        custom = tmp_path / "new_run.json"
        p = UnifiedPipeline(api_key="mock", timed_out_jobs_file=custom)
        p.init_timed_out_jobs_file()
        import json
        assert json.loads(custom.read_text()) == []

    def test_init_timed_out_jobs_file_idempotent(self, tmp_path):
        import json
        custom = tmp_path / "existing_run.json"
        existing_data = [{"job_id": "abc", "status": "timed_out"}]
        custom.write_text(json.dumps(existing_data))
        p = UnifiedPipeline(api_key="mock", timed_out_jobs_file=custom)
        p.init_timed_out_jobs_file()  # should not overwrite
        assert json.loads(custom.read_text()) == existing_data

    def test_loads_existing_jobs_on_init(self, tmp_path):
        import json
        custom = tmp_path / "run.json"
        jobs = [{"job_id": "xyz", "status": "timed_out"}]
        custom.write_text(json.dumps(jobs))
        p = UnifiedPipeline(api_key="mock", timed_out_jobs_file=custom)
        assert len(p.timed_out_jobs) == 1
        assert p.timed_out_jobs[0]["job_id"] == "xyz"


class TestRecoverTimedOutCLIOptions:
    """Tests that recover-timed-out CLI accepts --run-number, --suffix, and --file."""

    def test_help_shows_run_number_and_suffix(self):
        res = runner.invoke(app, ["recover-timed-out", "--help"])
        assert res.exit_code == 0
        assert "--run-number" in res.output
        assert "--suffix" in res.output
        assert "--file" in res.output

    def test_no_file_found_exits_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = runner.invoke(app, ["recover-timed-out", "--file", "nonexistent_run_999.json"])
        assert res.exit_code == 0  # graceful exit, not crash
        assert "No timed-out jobs file found" in res.output

    def test_run_number_resolves_correct_file(self, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        jobs_file = tmp_path / "timed_out_jobs_run_5.json"
        jobs_file.write_text(json.dumps([]))
        res = runner.invoke(app, ["recover-timed-out", "--run-number", "5"])
        assert res.exit_code == 0
        # Should NOT say "No timed-out jobs file found"
        assert "not found" not in res.output.lower()

    def test_suffix_resolves_correct_file(self, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        jobs_file = tmp_path / "timed_out_jobs_run_7.json"
        jobs_file.write_text(json.dumps([]))
        res = runner.invoke(app, ["recover-timed-out", "--suffix", "run_7"])
        assert res.exit_code == 0
        assert "not found" not in res.output.lower()

    def test_lists_available_files_when_not_found(self, tmp_path, monkeypatch):
        import json
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timed_out_jobs_run_1.json").write_text("[]")
        (tmp_path / "timed_out_jobs_run_2.json").write_text("[]")
        res = runner.invoke(app, ["recover-timed-out", "--file", "timed_out_jobs_run_99.json"])
        assert res.exit_code == 0
        assert "timed_out_jobs_run_1.json" in res.output or "Available" in res.output
