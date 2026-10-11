"""
pipeline/unified/__init__.py

EdGenie Unified 3-in-1 Pipeline:
Text Extraction + Visual Extraction + Syllabus Classification
powered by Unstructured.io Transform API & PyMuPDF.
"""

from pipeline.unified.marks_validator import MarkHierarchyValidator
from pipeline.unified.manifest_patcher import patch_records_with_manifest
from pipeline.unified.paper_type_detector import (
    PaperCategory,
    PaperTypeDetector,
    parse_caie_filename,
)
from pipeline.unified.pipeline import UnifiedPipeline
from pipeline.unified.schemas import (
    EssayAnswer,
    EssayAnswersContainer,
    EssayQuestion,
    EssayQuestionsContainer,
    MCQAnswer,
    MCQAnswersContainer,
    MCQQuestion,
    MCQQuestionsContainer,
    SyllabusUnit,
    to_strict_openai_schema,
)
from pipeline.unified.syllabus_loader import SyllabusLoader

__all__ = [
    "UnifiedPipeline",
    "PaperCategory",
    "PaperTypeDetector",
    "parse_caie_filename",
    "MarkHierarchyValidator",
    "SyllabusLoader",
    "patch_records_with_manifest",
    "to_strict_openai_schema",
    "EssayAnswer",
    "EssayAnswersContainer",
    "EssayQuestion",
    "EssayQuestionsContainer",
    "MCQAnswer",
    "MCQAnswersContainer",
    "MCQQuestion",
    "MCQQuestionsContainer",
    "SyllabusUnit",
]
