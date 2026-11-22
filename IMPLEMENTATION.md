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
│   │   ├── cli.py                       # `edgenie-scraper` CLI (scrape, validate, list-papers)
│   │   ├── scraper.py                   # PapaCambridgeScraper (Navigator primary, URLBuilder fallback)
│   │   ├── navigator.py                 # PapaCambridgeNavigator (Level 1/2/3 hierarchical crawler)
│   │   ├── validator.py                 # Post-scraping session directory prefix validator
│   │   ├── downloader.py                # Retrying PDF downloader (tenacity)
│   │   ├── url_builder.py               # Year-aware URL builder with anomaly patterns
│   │   ├── subject_registry.py          # Subject registry & session code resolver
│   │   ├── subject_registry.json        # 58 O-Level & 115 A-Level subject dictionary
│   │   ├── watermark_remover.py         # PapaCambridge watermark detection and cleaner
│   │   └── models.py                    # ScrapeTarget and ScrapeManifest models
│   │
│   ├── unified/                         # STAGE 2 — Unified 3-in-1 Pipeline
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-unified` / `edgenie-extractor` CLI (extract, extract-dir)
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
    ├── test_navigator.py                # Tests for PapaCambridgeNavigator hierarchical traversal
    ├── test_scraper_validation.py       # Tests for prefix validation & URL anomaly builder
    ├── test_unified_pipeline.py         # Tests for unified single/directory pipeline
    └── test_ingestion.py                # Tests for PostgreSQL ingestion and Firebase CDN
```

---

## 4. Stage 1 — Scraper

### 4.1 Hierarchical PapaCambridge Navigation Engine (`PapaCambridgeNavigator`)

To eliminate brittleness from PapaCambridge URL variations, Stage 1 uses a 3-tier DOM navigation hierarchy:

```
Level 1: Qualification Root
  https://pastpapers.papacambridge.com/papers/caie/o-level?theme=lightTheme
  https://pastpapers.papacambridge.com/papers/caie/as-and-a-level?theme=lightTheme
    │
    ▼ (Locate <span class="wraptext" id="..."> matching \b<subject_code>\b)
Level 2: Subject Landing Page (Cached across sessions)
  e.g. .../o-level-physics-5054
    │
    ▼ (Map session links matching <span class="wraptext" id="2026-May-June">)
Level 3: Session File Downloads
  e.g. .../o-level-physics-5054-2023-may-june
    │
    ▼ (Extract direct download links <a href="download_file.php?files=..." class="badge badge-info">)
PDF Downloads (Downloader + Retries + Watermark Stripping)
```

1. **Level 1 (Qualification Root)**: Requests the light-themed root for O-Level or AS/A-Level.
2. **Subject Code Locator**: Uses only the numerical subject code (`--code` / `-c`) to match `id` or text attributes of `.wraptext` spans (e.g. `Accounting - 9706` or `Physics (5054)`).
3. **Session Persistence**: Saves the Level 2 subject URL and session links. When multiple sessions or years are scraped in sequence, the navigator reuses the cached URL without re-visiting Level 1.
4. **Level 3 (Authoritative Downloads)**: Navigates directly to the target session page and parses download badge links (`download_file.php?files=...`).

### 4.2 Anomaly-Resilient Fallback URL Builder (`URLBuilder`)

If the navigator encounters unlisted routes, the pipeline falls back to multi-candidate URL testing that accounts for known PapaCambridge URL anomalies:
- **Repeated Subject/Code Slugs** (e.g. `.../o-level-islamiyat-2058-islamiyat-20582024-oct-nov`)
- **Duplicate Session Strings** (e.g. `.../o-level-chemistry-5070-2025-oct-nov2025-oct-nov`)
- **Legacy Inverted Sessions** (e.g. `.../o-level-islamiyat-2058-may-june-2020`)
- **Syllabus Aliases** (e.g. code `4024` maps `mathematics-d` to `mathematics-syllabus-d`).

### 4.3 Post-Scraping Directory & Prefix Validation (`validate_scraped_sessions`)

To prevent corrupted sessions from polluting downstream extraction and database ingestion:
- Computes the expected CAIE filename prefix for each session:
  - May/June: `{subject_code}_s{yy}` (e.g., `5054_s15`)
  - Oct/Nov: `{subject_code}_w{yy}` (e.g., `5054_w15`)
  - Feb/March: `{subject_code}_m{yy}` (e.g., `0580_m22`)
- Automatically validates all downloaded files in every scraped session folder.
- If any session folder contains non-matching PDFs, the corrupted session directory is automatically deleted, and the failure is logged.
- Can also be invoked manually on existing directories via `edgenie-scraper validate`.

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

### 5.5 Job Timeout Resilience & Playground Endpoint Recovery

If polling of an Unstructured.io Transform job exceeds `max_poll_time` (e.g. 600s):
1. `JobTimeoutError` captures the `job_id` and elapsed polling time.
2. The job is automatically recorded into `timed_out_jobs.json` with metadata (`pdf_path`, `category`, `output_dir`).
3. The pipeline queries the playground endpoint directly:
   `https://transform.unstructured.io/playground/jobs/{job_id}?output=elements`
4. Once retrieved, `complete_pipeline_work()` executes marks reconciliation, PyMuPDF visual cropping, manifest creation, visual linking into question records, and canonical JSON output.
5. In batch mode (`extract-dir`), timed-out jobs are recorded during the run and automatically re-checked for endpoint recovery at the end of the batch pass.
6. Dedicated recovery commands (`recover-job` and `recover-timed-out`) allow manual or automated catch-up at any time without re-spending API extraction credits.

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

### 8.1 Scraper Commands (`edgenie-scraper`)

The scraper uses `--code` (`-c`) as the only mandatory parameter. Level and subject name are resolved automatically.

```bash
# ── Single Session Scraping ──────────────────────────────────────────────────
# Scrapes only the specified year and session for subject code 5054
edgenie-scraper scrape --code 5054 --year 2023 --session jun

# ── Complete Directory Scraping (Multi-Year / Multi-Session) ─────────────────
# Scrapes all specified sessions across the full year range
edgenie-scraper scrape \
    --code 5054 \
    --year-start 2015 \
    --year-end 2024 \
    --session jun \
    --session nov

# Force re-download even if files already exist on disk
edgenie-scraper scrape --code 5054 --year 2023 --session jun --force

# Dry-run mode (discovers links and reports targets without downloading)
edgenie-scraper scrape --code 5054 --year 2023 --session jun --dry-run

# ── Session Prefix Validation & Pruning ──────────────────────────────────────
# Validates CAIE filename prefixes (e.g. 5054_s23, 5054_w23) and deletes corrupted session dirs
edgenie-scraper validate --code 5054 --year-start 2015 --year-end 2024

# ── Directory Listing & Inspection ───────────────────────────────────────────
# Lists downloaded files, manifests, sizes, and paper types
edgenie-scraper list-papers --code 5054 --year 2023
```

---

### 8.2 Unified Extraction & Classification Commands (`edgenie-unified`)

#### A. Single PDF Commands (`extract` / `extract-visuals-text-and-classify`)

```bash
# Single exam paper (auto-discovers syllabus CSV from syllabi/ by subject code)
edgenie-unified extract --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# With explicit syllabus CSV and force re-extraction (bypassing local cache)
edgenie-unified extract \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --syllabus syllabi/PHYSICS_5054.csv \
    --force

# Custom DPI and paper type override
edgenie-unified extract \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_12.pdf \
    --type mcq \
    --dpi 300
```

*(Note: `edgenie-unified extract-visuals-text-and-classify` and `edgenie-extractor extract` are identical aliases.)*

#### B. Complete Directory Batch Extraction Commands (`extract-dir`)

```bash
# Batch extract all exam papers in a single session directory
edgenie-unified extract-dir --dir pipeline/storage/5054/2023/jun/

# Batch extract all exam papers across an entire subject directory tree
edgenie-unified extract-dir --dir pipeline/storage/5054/

# Force re-extraction across directory with custom DPI
edgenie-unified extract-dir --dir pipeline/storage/5054/2023/jun/ --force --dpi 300
```

> **Directory Processing Features**:
> - Recursively searches directory tree for PDF files.
> - Automatically excludes non-exam files (`_er`, `_gt`, `_ci`, `_ir`, `_sy`).
> - Processes only exam Question Papers (`_qp_`) and Mark Schemes (`_ms_`).
> - Auto-detects paper categories and resolves corresponding syllabus CSV per paper.
> - Outputs a comprehensive summary table showing extraction status, records, and visuals.

#### C. Job Timeout Recovery Commands (`recover-job`, `recover-timed-out`)

```bash
# Recover a specific timed-out job via playground endpoint and complete processing
edgenie-unified recover-job --job-id b60382ec-1f13-4a8d-9946-d7171d977e9a --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# Recover all recorded timed-out jobs from timed_out_jobs.json
edgenie-unified recover-timed-out --file timed_out_jobs.json
```

---

### 8.3 Ingestion Commands (`edgenie-ingestor`)

#### A. Single Paper / JSON Commands (`db-paper`)

```bash
# Ingest single paper JSON and visual manifest into PostgreSQL
edgenie-ingestor db-paper --json pipeline/storage/5054/2023/jun/5054_s23_qp_22.json
```

#### B. Complete Directory Commands (`db-dir`, `storage-dir`, `full-sync`)

```bash
# Initialize PostgreSQL tables, foreign keys, and GIN indexes
edgenie-ingestor db-init

# Ingest an entire subject directory tree into PostgreSQL
edgenie-ingestor db-dir --dir pipeline/storage/5054

# View current database record counts across all relational tables
edgenie-ingestor db-stats

# (Optional) Upload visuals and PDFs in a directory to Firebase Storage CDN
edgenie-ingestor storage-dir --dir pipeline/storage/5054

# (Optional) Complete full sync (Firebase CDN upload + PostgreSQL ingestion)
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
| **3.1.0** | 2026-10-12 | **Scraping Navigation Engine, Prefix Validation & Batch Directory Extraction**: Introduced hierarchical `PapaCambridgeNavigator` traversing Level 1 (qualification root), Level 2 (subject session map), and Level 3 (direct download badge links) with subject code locator matching (`--code` is now the only required scraper CLI parameter). Added multi-pattern URL anomaly builder and automated post-scraping session prefix validation (`validator.py` and `edgenie-scraper validate`) that prunes corrupted/mismatched session folders. Added `edgenie-unified extract-dir` for recursive directory-level batch extraction with automatic non-exam file exclusion (`_er`, `_gt`, etc.), alongside `extract` single PDF alias. Updated full documentation across `README.md` and `IMPLEMENTATION.md`. |
