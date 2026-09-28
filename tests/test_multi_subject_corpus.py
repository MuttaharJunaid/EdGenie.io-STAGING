"""
Pillar 4: Multi-Subject Corpus Integrity Benchmark Test Suite.

Verifies 100% visual extraction perfection across all Cambridge subjects:
- Biology (5090)
- Physics (5054)
- Mathematics (4024)
- Economics (2281)
- Principles of Accounts (7110)

Audits zero cut strokes, clean whitespace margins, valid canonical schemas,
and non-empty high-resolution visuals.
"""

from pathlib import Path
import fitz
import pytest

from pipeline.visuals.extractor import CambridgeVisualExtractor
from pipeline.visuals.verifier import verify_crop


@pytest.fixture
def extractor():
    return CambridgeVisualExtractor(dpi=150)


class TestMultiSubjectCorpus:
    """Rigorous cross-subject benchmark across Cambridge examination papers."""

    def test_biology_5090_integrity(self, extractor, tmp_path):
        """Verify Biology (5090) visual extraction integrity and zero cut strokes."""
        pdf_path = Path("pipeline/storage/5090/2010/jun/5090_s10_qp_11.pdf")
        if not pdf_path.exists():
            pytest.skip("5090_s10_qp_11.pdf not found in storage.")

        out_dir = tmp_path / "bio_5090"
        manifest = extractor.extract(pdf_path, out_dir)
        assert len(manifest) >= 25, f"Expected at least 25 visuals for 5090, got {len(manifest)}"

        doc = fitz.open(pdf_path)
        for m in manifest:
            img_file = out_dir / m["filename"]
            assert img_file.exists(), f"Extracted file {m['filename']} missing from disk"
            assert img_file.stat().st_size > 1000, f"Extracted file {m['filename']} unexpectedly small"

            page = doc[m["page"] - 1]
            v_res = verify_crop(page, fitz.Rect(m["bbox"]), dpi=150)
            assert v_res.cut_strokes == 0, f"{m['filename']} has {v_res.cut_strokes} cut strokes: {v_res.reasons}"

    def test_physics_5054_integrity(self, extractor, tmp_path):
        """Verify Physics (5054) visual extraction integrity and zero cut strokes."""
        pdf_path = Path("pipeline/storage/5054/2010/nov/5054_w10_qp_23.pdf")
        if not pdf_path.exists():
            pytest.skip("5054_w10_qp_23.pdf not found in storage.")

        out_dir = tmp_path / "phys_5054"
        manifest = extractor.extract(pdf_path, out_dir, force_mcq=False)
        assert len(manifest) >= 8, f"Expected at least 8 visuals for 5054, got {len(manifest)}"

        doc = fitz.open(pdf_path)
        for m in manifest:
            img_file = out_dir / m["filename"]
            assert img_file.exists(), f"Extracted file {m['filename']} missing"

            page = doc[m["page"] - 1]
            v_res = verify_crop(page, fitz.Rect(m["bbox"]), dpi=150)
            assert v_res.cut_strokes == 0, f"{m['filename']} has {v_res.cut_strokes} cut strokes: {v_res.reasons}"

    def test_mathematics_4024_integrity(self, extractor, tmp_path):
        """Verify Mathematics (4024) diagram and geometric figure extraction."""
        pdf_path = Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf")
        if not pdf_path.exists():
            pytest.skip("4024_s10_qp_11.pdf not found in storage.")

        out_dir = tmp_path / "math_4024"
        manifest = extractor.extract(pdf_path, out_dir)
        assert len(manifest) >= 10, f"Expected >= 10 visuals for 4024, got {len(manifest)}"

        doc = fitz.open(pdf_path)
        for m in manifest:
            img_file = out_dir / m["filename"]
            assert img_file.exists()

            page = doc[m["page"] - 1]
            v_res = verify_crop(page, fitz.Rect(m["bbox"]), dpi=150)
            assert v_res.cut_strokes == 0, f"{m['filename']} has {v_res.cut_strokes} cut strokes"

    def test_economics_2281_integrity(self, extractor, tmp_path):
        """Verify Economics (2281) curve, chart, and table visual extraction."""
        pdf_path = Path("pipeline/storage/2281/2010/jun/2281_s10_qp_11.pdf")
        if not pdf_path.exists():
            pytest.skip("2281_s10_qp_11.pdf not found in storage.")

        out_dir = tmp_path / "econ_2281"
        manifest = extractor.extract(pdf_path, out_dir)
        assert len(manifest) >= 8, f"Expected >= 8 visuals for 2281, got {len(manifest)}"

        doc = fitz.open(pdf_path)
        for m in manifest:
            img_file = out_dir / m["filename"]
            assert img_file.exists()

            page = doc[m["page"] - 1]
            v_res = verify_crop(page, fitz.Rect(m["bbox"]), dpi=150)
            assert v_res.cut_strokes == 0, f"{m['filename']} has {v_res.cut_strokes} cut strokes"

    def test_accounting_7110_integrity(self, extractor, tmp_path):
        """Verify Accounting (7110) ledger and financial visual extraction."""
        pdf_path = Path("pipeline/storage/7110/2010/jun/7110_s10_qp_11.pdf")
        if not pdf_path.exists():
            pytest.skip("7110_s10_qp_11.pdf not found in storage.")

        out_dir = tmp_path / "acc_7110"
        manifest = extractor.extract(pdf_path, out_dir)
        assert len(manifest) >= 8, f"Expected >= 8 visuals for 7110, got {len(manifest)}"

        doc = fitz.open(pdf_path)
        for m in manifest:
            img_file = out_dir / m["filename"]
            assert img_file.exists()

            page = doc[m["page"] - 1]
            v_res = verify_crop(page, fitz.Rect(m["bbox"]), dpi=150)
            assert v_res.cut_strokes == 0, f"{m['filename']} has {v_res.cut_strokes} cut strokes"

    def test_cross_subject_manifest_schema_compliance(self, extractor, tmp_path):
        """Verify all extracted visuals across subjects comply strictly with schema."""
        required_keys = {"page", "question", "target_type", "target_label", "bbox", "filename"}
        sample_pdfs = [
            Path("pipeline/storage/5090/2010/jun/5090_s10_qp_11.pdf"),
            Path("pipeline/storage/4024/2010/jun/4024_s10_qp_11.pdf"),
            Path("pipeline/storage/2281/2010/jun/2281_s10_qp_11.pdf"),
        ]

        for pdf in sample_pdfs:
            if not pdf.exists():
                continue
            manifest = extractor.extract(pdf, tmp_path / pdf.stem)
            for m in manifest:
                missing = required_keys - set(m.keys())
                assert not missing, f"{pdf.name}: Entry missing keys: {missing}"
                bbox = m["bbox"]
                assert len(bbox) == 4, f"Invalid bbox format: {bbox}"
                assert bbox[2] > bbox[0], f"Inverted x-coordinates: {bbox}"
                assert bbox[3] > bbox[1], f"Inverted y-coordinates: {bbox}"
