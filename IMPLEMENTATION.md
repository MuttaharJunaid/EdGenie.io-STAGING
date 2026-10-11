# EdGenie — Implementation Plan & Architecture Specification

> **Version**: 3.0.0  
> **Last Updated**: 2026-10-11  
> **Status**: Active / Production Architecture (Unified 3-in-1 Extraction & PostgreSQL Relational + JSONB Ingestion)

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Stage 1 — Scraper](#4-stage-1--scraper)
5. [Stage 2 — Unified 3-in-1 Pipeline (Text + Visuals + Classification)](#5-stage-2--unified-3-in-1-pipeline-text--visuals--classification)
6. [Stage 3 — PostgreSQL Ingestion (Relational + JSONB) & Firebase CDN](#6-stage-3--postgresql-ingestion-relational--jsonb--firebase-cdn)
7. [Validation & QA](#7-validation--qa)
8. [CLI Command Reference & Workflow](#8-cli-command-reference--workflow)
9. [JSON Output Schemas (Canonical)](#9-json-output-schemas-canonical)
10. [Change Log](#10-change-log)

---

## 1. Product Overview

### Problem

CAIE O-Level and AS/A-Level students need targeted past paper practice to achieve top grades (`A*` / `A`). Exam papers are inherently unbalanced: **common easy topics appear frequently**, while **difficult/complex topics appear rarely**. Standard topical booklets or random paper solving leave students unprepared for the rare-but-challenging questions that actually determine final grade boundaries.

### Solution

**EdGenie** is a platform designed for colleges and students to systematically master past papers. For every CAIE subject (Physics, Chemistry, Biology, Mathematics, Accounting, Business Studies, Economics, Computer Science, English, etc.), EdGenie:

1. **Maintains Official Syllabi** in structured CSVs (`syllabi/`) mapping Unit, Topic, and Sub-topics.
2. **Scrapes** past papers, mark schemes, and examiner reports from PapaCambridge across O-Level and AS/A-Level.
3. **Extracts & Classifies in a Single Cloud Pass**: Submits PDF exam papers to the **Unstructured.io Transform API** with strict OpenAI-compliant JSON schemas and inlined syllabus references, extracting:
   - Full question stems, nested sub-parts (`(a)`, `(b)(i)`), mark allocations, and options (A/B/C/D for MCQs).
   - High-precision visual layout elements and bounding boxes for diagrams, figures, and tables.
   - Exact topic, sub-topic, difficulty score (1–10), and command words classified inline.
4. **Locally Crops Visuals**: PyMuPDF renders high-resolution 300 DPI crops from layout bounding boxes with artifact filtering (cover logos, barcodes, header/footer marks), saving a `manifest.json` and patching image paths into question records.
5. **Reconciles Marks Hierarchy**: The `MarkHierarchyValidator` verifies that leaf marks match bracketed `[N]` notation, parent part marks strictly equal the sum of child sub-parts, and total question marks equal the sum of parts.
6. **Hosts Media Assets** on **Firebase Storage** served via custom CDN prefix `https://assets.edgenie.io`.
7. **Ingests** structured questions, answers, and CDN links into **PostgreSQL** using relational integrity and JSONB querying.
8. **Serves** a question bank where students filter by **Topic × Difficulty × Frequency** to focus on high-yield, challenging questions.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EdGenie Pipeline                                  │
│                                                                             │
│  ┌───────────────────────┐         ┌─────────────────────────────────────┐  │
│  │        Stage 1        │────────▶│               Stage 2               │  │
│  │   PapaCambridge       │         │       Unified 3-in-1 Pipeline       │  │
│  │      Scraper          │         │    (Unstructured.io + PyMuPDF)      │  │
│  │  (`edgenie-scraper`)  │         │  (`edgenie-unified` / `-extractor`) │  │
│  └───────────────────────┘         └──────────────────┬──────────────────┘  │
│                                                       │                     │
│                                                       ▼                     │
│                                    ┌─────────────────────────────────────┐  │
│                                    │               Stage 3               │  │
│                                    │     PostgreSQL DB Ingestion &       │  │
│                                    │        Firebase CDN Hosting         │  │
│                                    │        (`edgenie-ingestor`)         │  │
│                                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
EdGenie/
├── IMPLEMENTATION.md                    # ◀ Single source of truth for architecture
├── README.md                            # Quickstart & command reference
├── pyproject.toml                       # Dependencies & CLI entrypoints
├── syllabi/                             # Official subject syllabus CSV files
│   ├── PHYSICS_5054.csv
│   ├── ACCOUNTING_7707.csv
│   ├── ECONOMICS_2281.csv
│   ├── CHEMISTRY_5070.csv
│   ├── BIOLOGY_5090.csv
│   ├── MATHEMATICS_4024.csv
│   ├── COMPUTER_SCIENCE_2210.csv
│   ├── BUSINESS_7115.csv
│   └── ENGLISH_1123.csv
│
├── pipeline/                            # Core pipeline implementation
│   ├── __init__.py
│   ├── config.py                        # Pipeline configuration settings (PipelineSettings)
│   │
│   ├── scraper/                         # STAGE 1 — Scraper
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-scraper` CLI
│   │   ├── scraper.py                   # PapaCambridgeScraper
│   │   ├── downloader.py                # Retrying PDF downloader (tenacity)
│   │   ├── url_builder.py               # Year-aware URL builder
│   │   ├── subject_registry.py          # Subject registry & session code resolver
│   │   ├── subject_registry.json        # 58 O-Level & 115 A-Level subject dictionary
│   │   ├── watermark_remover.py         # PapaCambridge watermark detection and cleaner
│   │   └── models.py                    # ScrapeTarget and ScrapeManifest models
│   │
│   ├── unified/                         # STAGE 2 — Unified 3-in-1 Pipeline
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-unified` / `edgenie-extractor` CLI
│   │   ├── pipeline.py                  # UnifiedPipeline: single API parse + local PyMuPDF cropping
│   │   ├── schemas.py                   # Pydantic models (MCQ, Essay, Syllabus) & to_strict_openai_schema
│   │   ├── paper_type_detector.py       # PaperCategory, PaperTypeDetector & parse_caie_filename
│   │   ├── marks_validator.py           # MarkHierarchyValidator (leaf marks, part sums, totals)
│   │   ├── syllabus_loader.py           # SyllabusLoader: multi-path CSV resolution and prompt formatter
│   │   └── manifest_patcher.py          # patch_records_with_manifest & question compatibility guards
│   │
│   ├── ingestion/                       # STAGE 3 — PostgreSQL Ingestion & Firebase Storage
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-ingestor` CLI (db-init, db-paper, db-dir, full-sync)
│   │   ├── firebase_storage.py          # Firebase Storage Asset Manager with `https://assets.edgenie.io` CDN prefix
│   │   ├── postgres_client.py           # PostgreSQL client manager (6 core tables, foreign keys + GIN indexes)
│   │   └── postgres_ingestor.py         # PostgresIngestor document builder & atomic upsert service
│   │
│   └── storage/                         # Downloaded PDFs, output JSONs, and visual assets
│       └── {subject_code}/{year}/{session}/
│
└── tests/                               # Pytest test suite
    ├── test_unified_pipeline.py         # Tests for unified pipeline across core subjects
    └── test_ingestion.py                # Tests for PostgreSQL ingestion and Firebase CDN
```

---

## 4. Stage 1 — Scraper

- **Target**: PapaCambridge qualification past papers.
- **Subject Registry**: Maps 58 O-Level and 115 A-Level subjects to standardized codes and session slugs (`may-june`, `oct-nov`, `march`).
- **Watermark Cleaning**: `watermark_remover.py` cleans overlay banners on papers $\ge 2026$.
- **CLI**: `edgenie-scraper scrape --level <lvl> --subject <name> --code <code> --year-start <yr> --year-end <yr>`.

---

## 5. Stage 2 — Unified 3-in-1 Pipeline (Text + Visuals + Classification)

### 5.1 Single-Job Transform Architecture

Instead of paying twice (once for parsing visuals and again for extracting structured text), EdGenie uses Unstructured.io's `client.parse.run(..., schema=schema, prompt=prompt, output="elements", include=["coordinates", "table_html"])`:

```
Cambridge Exam PDF + Subject Syllabus CSV
    │
    ▼
[PyMuPDF Page Pruner] (removes blank pages to minimize cloud compute)
    │
    ▼
[Unstructured.io Transform API Job] (wait_seconds=0 async submission + polling)
    ├── result.extracted_data ──▶ Structured Questions & Inline Syllabus Classification
    └── result.elements       ──▶ Layout Bounding Boxes (Image, Table, Figure)
                                          │
                                          ▼
                                  [PyMuPDF Cropping Engine]
                                  - DPI: 300
                                  - Artifact Filter (cover, barcodes, timing marks)
                                  - FigureCaption Bounding Box Expansion
                                          │
                                          ▼
                                  [Manifest Patcher]
                                  Links images to question stems, parts, & subparts
                                          │
                                          ▼
                                  [Mark Hierarchy Validator]
                                  Reconciles [N] leaf marks and sums
                                          │
                                          ▼
                              Canonical Output JSON & Visual Manifest
```

### 5.2 Schemas & Strict OpenAI Outputs

Schemas in `pipeline/unified/schemas.py` are converted via `to_strict_openai_schema()`:
- `additionalProperties: False` enforced on every object in the hierarchy.
- All properties listed in `required`.
- Containers: `EssayQuestionsContainer`, `EssayAnswersContainer`, `MCQQuestionsContainer`, `MCQAnswersContainer`.

### 5.3 Inline Syllabus Classification

The `SyllabusLoader` resolves the syllabus CSV from `syllabi/` and formats units, topics, and sub-topics into a reference block embedded directly in the extraction prompt. The extraction agent classifies:
- `Unit`: Unit number and title.
- `Topic`: Topic number (e.g. `1.1`).
- `Topic_name`: Official topic name from syllabus.
- `Sub_topic`: Specific learning objectives tested.
- `Difficulty_score` (1–10) & `Difficulty_label` (`easy`, `medium`, `hard`).
- `Command_words`: Identified Cambridge command words (`["Describe", "Calculate"]`).

### 5.4 Mark Hierarchy Validation

The `MarkHierarchyValidator` enforces Cambridge examination rules:
1. Leaf sub-parts and parts have marks $\ge 1$, parsed from trailing bracketed `[N]` notation.
2. Parent part marks strictly equal the sum of constituent child sub-part marks.
3. Total question marks strictly equal the sum of constituent part marks.
4. Trailing indicators (`[Total: N]`) are stripped from question stem text.
5. MCQ questions are strictly assigned 1 mark.

---

## 6. Stage 3 — PostgreSQL Ingestion (Relational + JSONB) & Firebase CDN

### 6.1 Database Schema (6 Core Tables)

1. **`subjects`**: Subject catalog (`subject_code`, `subject_name`, `level`, `syllabus_rules_count`).
2. **`papers`**: Exam paper records (`paper_code`, `subject_code`, `year`, `session`, `variant`, `paper_number`, `paper_type`, `category`, `pdf_cdn_url`).
3. **`questions`**: Core questions table (`question_key`, `question_number`, `question_text`, `marks`, `unit`, `topic`, `topic_name`, `sub_topics` (JSONB GIN), `difficulty_score`, `difficulty_label`, `command_words` (JSONB GIN), `options` (JSONB), `parts` (JSONB GIN), `image_cdn_url`).
4. **`answers`**: Mark scheme answers (`id`, `question_key`, `paper_code`, `answer_text`, `parts` (JSONB GIN), `image_cdn_url`).
5. **`visuals`**: Visual figure manifest (`id`, `paper_code`, `question_number`, `caption`, `page`, `bbox` (JSONB), `filename`, `cdn_url`).
6. **`pipeline_runs`**: Ingestion auditing and execution metrics (`stage`, `status`, `details` (JSONB)).

### 6.2 Firebase Storage CDN

- **PDF CDN URL**: `https://assets.edgenie.io/pdfs/{subject_code}/{year}/{session}/{filename}.pdf`
- **Visual Image CDN URL**: `https://assets.edgenie.io/visuals/{paper_code}/{filename}.png`

---

## 7. Validation & QA

1. **Automated Unit & Regression Tests**: 48 test cases in `tests/test_unified_pipeline.py` verifying detection, schema generation, syllabus injection, artifact filtering, manifest patching, marks validation, and CLI commands.
2. **Completeness & Sequentiality**: Validates that all questions in a paper are sequential without dropped parts.
3. **CDN Link Verification**: Ensures that all image links resolve to valid assets on `https://assets.edgenie.io`.

---

## 8. CLI Command Reference & Workflow

### Unified Extraction Commands

```bash
# Minimal command (auto-discovers syllabus from syllabi/ by subject code)
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# With explicit syllabus CSV
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --syllabus syllabi/PHYSICS_5054.csv

# Force re-extraction (bypassing local cache)
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --force

# Custom DPI and type override
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_12.pdf \
    --type mcq --dpi 300
```

*(Note: `edgenie-extractor extract-visuals-text-and-classify` is an identical entry point.)*

### Ingestion Commands

```bash
# Initialize PostgreSQL tables, foreign keys, and indexes
edgenie-ingestor db-init

# Ingest single paper JSON and visual manifest
edgenie-ingestor db-paper --json pipeline/storage/5054/2023/jun/5054_s23_qp_22.json

# Ingest an entire subject directory tree
edgenie-ingestor db-dir --dir pipeline/storage/5054

# View current database record counts
edgenie-ingestor db-stats

# (Optional) Upload visuals and PDFs to Firebase Storage
edgenie-ingestor storage-dir --dir pipeline/storage/5054

# (Optional) Full sync (Firebase upload + PostgreSQL ingestion)
edgenie-ingestor full-sync --dir pipeline/storage/5054
```

---

## 9. JSON Output Schemas (Canonical)

### Canonical Essay Question Record

```json
{
  "QuestionNumber": "May/June/22/2023/Q1",
  "Question_text": "A student measures the acceleration of a trolley down a ramp.",
  "Unit": "1 - General Physics",
  "Topic": "1.2",
  "Topic_name": "Kinematics",
  "Sub_topic": ["Acceleration", "Speed-time graphs"],
  "Difficulty_score": 5,
  "Difficulty_label": "Medium",
  "Command_words": ["Calculate", "Describe"],
  "Marks": 7,
  "image": "5054_s23_qp_22/p2_q1_question_Fig_1_1.png",
  "Parts": [
    {
      "part_label": "a",
      "part_text": "Define acceleration.",
      "marks": 2,
      "image": null,
      "sub_parts": []
    },
    {
      "part_label": "b",
      "part_text": "Calculate the velocity of the trolley.",
      "marks": 5,
      "image": null,
      "sub_parts": [
        {
          "sub_part_label": "(i)",
          "sub_part_text": "Show that the initial kinetic energy is 1.2 J.",
          "marks": 2,
          "image": null
        },
        {
          "sub_part_label": "(ii)",
          "sub_part_text": "Calculate the final speed.",
          "marks": 3,
          "image": null
        }
      ]
    }
  ]
}
```

---

## 10. Change Log

| Version | Date | Description |
|---|---|---|
| **1.0.0** | 2026-07-13 | Initial architecture proposal. |
| **2.0.0** | 2026-08-24 | Updated Stage 1 with `SubjectRegistry` (58 O-Level / 115 A-Level subjects). |
| **2.4.0** | 2026-09-01 | PostgreSQL (Relational + JSONB) Architecture & Decoupled Ingestion. |
| **3.0.0** | 2026-10-11 | **Unified 3-in-1 Pipeline Architecture**: Unified text extraction, visual bounding box extraction, and syllabus classification into a single Unstructured.io Transform API job (`client.parse.run` with strict schema + syllabus prompt) and PyMuPDF local cropping. Eliminated separate text extractor, separate visual detector, and separate classifier agents. Removed obsolete `pipeline/extractor/`, `pipeline/visuals/`, and `pipeline/syllabus/` directories. Added 48 comprehensive unit tests for core subjects in `tests/test_unified_pipeline.py`. Consolidated all CLI extraction under `edgenie-unified` and `edgenie-extractor`. |
