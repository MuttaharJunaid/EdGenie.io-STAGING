"""
Unit & Integration Tests for UnstructuredVisualExtractor
Run from project root:  .venv/bin/pytest tests/test_unstructured_visuals.py -v
"""
import json
from pathlib import Path
import warnings

import pymupdf as fitz
import pytest

from pipeline.visuals.extractor import CambridgeVisualExtractor
from pipeline.visuals.unstructured_extractor import UnstructuredVisualExtractor


SAMPLE_PDF = Path("pipeline/storage/5090/2019/nov/5090_w19_qp_32.pdf")
SCRATCH_SAMPLE_JSON = Path(
    "/Users/myco/.gemini/antigravity/brain/31962e5c-2e92-410a-93af-20fdb637417c/scratch/sample_unstructured_response.json"
)


@pytest.fixture
def mock_cache_dir(tmp_path: Path):
    """Populate a temporary cache directory with the sample unstructured JSON for 5090_w19_qp_32."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if SCRATCH_SAMPLE_JSON.exists():
        with open(SCRATCH_SAMPLE_JSON, encoding="utf-8") as f:
            data = json.load(f)
        with open(cache_dir / "5090_w19_qp_32.json", "w", encoding="utf-8") as f:
            json.dump({"pdf_name": "5090_w19_qp_32.pdf", "elements": data["elements"]}, f)
    return cache_dir


class TestUnstructuredVisualExtractor:
    def test_init_defaults(self):
        extractor = UnstructuredVisualExtractor(api_key="test_key")
        assert extractor.dpi == 300
        assert extractor.profile == "best"
        assert extractor.extract_tables is True
        assert extractor.api_key == "test_key"

    def test_cover_page_and_admin_filtering(self):
        extractor = UnstructuredVisualExtractor(api_key="test_key")
        # Cover page element
        assert extractor._is_cover_or_admin_element({"text": "Cambridge O Level"}, page_number=1) is True
        # Examiner use table on page 2
        assert (
            extractor._is_cover_or_admin_element(
                {"text": "For Examiner's Use Only 1 2 Total"}, page_number=2
            )
            is True
        )
        # Regular visual element on page 2
        assert (
            extractor._is_cover_or_admin_element(
                {"text": "sharp blade cutting potato tissue"}, page_number=2
            )
            is False
        )

    def test_mcq_paper_detection(self):
        extractor = UnstructuredVisualExtractor(api_key="test_key")
        assert extractor._is_mcq_paper(Path("5090_w19_qp_11.pdf")) is True
        assert extractor._is_mcq_paper(Path("5090_w19_qp_12.pdf")) is True
        assert extractor._is_mcq_paper(Path("5090_w19_qp_22.pdf")) is False
        assert extractor._is_mcq_paper(Path("5090_w19_qp_32.pdf")) is False
        # Overrides
        assert extractor._is_mcq_paper(Path("5090_w19_qp_32.pdf"), force_mcq=True) is True
        assert extractor._is_mcq_paper(Path("5090_w19_qp_11.pdf"), force_mcq=False) is False

    @pytest.mark.skipif(not SAMPLE_PDF.exists() or not SCRATCH_SAMPLE_JSON.exists(), reason="Sample files required")
    def test_extraction_on_sample_paper(self, mock_cache_dir: Path, tmp_path: Path):
        output_dir = tmp_path / "visuals_out"
        extractor = UnstructuredVisualExtractor(
            api_key="mock_key",
            cache_dir=mock_cache_dir,
            extract_tables=True,
        )

        manifest = extractor.extract(SAMPLE_PDF, output_dir, force_refresh=False)
        assert len(manifest) == 10

        # Check filenames
        filenames = [m["filename"] for m in manifest]
        assert "p3_q1_question_diagram_1.png" in filenames
        assert "p4_q1_sub_part_a(ii)_Table_1.png" in filenames
        assert "p6_q1_part_c_diagram_2.png" in filenames
        assert "p7_q1_part_c_Table_2.png" in filenames
        assert "p7_q1_sub_part_c(i)_diagram_3.png" in filenames
        assert "p9_q2_question_diagram_1.png" in filenames
        assert "p9_q2_question_diagram_2.png" in filenames
        assert "p9_q2_question_diagram_3.png" in filenames
        assert "p10_q2_part_b_diagram_4.png" in filenames
        assert "p11_q2_part_c_diagram_5.png" in filenames

        # Verify all files were saved and are valid PNG images
        for m in manifest:
            saved_file = output_dir / m["filename"]
            assert saved_file.exists()
            assert saved_file.stat().st_size > 0

        # Verify manifest.json saved
        manifest_file = output_dir / "manifest.json"
        assert manifest_file.exists()
        with open(manifest_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 10

    def test_legacy_extractor_deprecation_warning(self):
        with pytest.deprecated_call():
            CambridgeVisualExtractor()
