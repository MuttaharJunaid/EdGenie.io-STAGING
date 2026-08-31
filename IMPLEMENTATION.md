# EdGenie — Implementation Plan & Architecture Specification

> **Version**: 2.4.0  
> **Last Updated**: 2026-09-01  
> **Status**: Active / Production Architecture (PostgreSQL Relational + JSONB & Decoupled Firebase Storage Asset CDN)

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Stage 0 — Syllabus Extraction & Management](#4-stage-0--syllabus-extraction--management)
5. [Stage 1 — Scraper (Non-Agentic)](#5-stage-1--scraper-non-agentic)
6. [Stage 2 — Text Extraction & Question Parser](#6-stage-2--text-extraction--question-parser)
7. [Stage 3 — Cambridge Visual Extractor (Non-Agentic)](#7-stage-3--cambridge-visual-extractor-non-agentic)
8. [Stage 4 — MongoDB Ingestion & Firebase Asset Storage](#8-stage-4--mongodb-ingestion--firebase-asset-storage)
9. [Stage 5 — Validation / QA Agent](#9-stage-5--validation--qa-agent)
10. [CLI Command Reference & End-to-End Workflow](#10-cli-command-reference--end-to-end-workflow)
11. [JSON Output Schemas (Canonical)](#11-json-output-schemas-canonical)
12. [Change Log](#12-change-log)

---

## 1. Product Overview

### Problem

CAIE O-Level and AS/A-Level students need targeted past paper practice to achieve top grades (`A*` / `A`). Exam papers are inherently unbalanced: **common easy topics appear frequently**, while **difficult/complex topics appear rarely**. Standard topical booklets or random paper solving leave students unprepared for the rare-but-challenging questions that actually determine final grade boundaries.

### Solution

**EdGenie** is a platform designed for colleges and students to systematically master past papers. For every CAIE subject (Physics, Chemistry, Biology, Mathematics, Accounting, Business Studies, Economics, etc.), EdGenie:

1. **Extracts Official Syllabi** directly from the Cambridge International website into 3-column CSV structures (`Unit,Topic,Subtopic`).
2. **Scrapes** past papers, mark schemes, and examiner reports from PapaCambridge across O-Level and AS/A-Level.
3. **Extracts** every question into canonical JSON using layout analysis, and classifies it against syllabus topics and difficulty levels via text-only LLM passes.
4. **Extracts visuals** (diagrams, graphs, tables, circuit diagrams, apparatus drawings) and binds them tightly to their parent question.
5. **Hosts Media Assets** (all PDFs and visual PNGs) on **Firebase Storage** served via custom CDN prefix `https://assets.edgenie.io`.
6. **Ingests** structured questions, answers, CDN media links, and vector embeddings into **MongoDB** (with MongoDB Vector Search).
7. **Validates** quality via automated test suites and QA checks before publishing.
8. **Serves** a question bank where students filter by **Topic × Difficulty × Frequency** to focus on high-yield, challenging questions.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EdGenie Pipeline                                  │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Stage 0  │───▶│ Stage 1  │───▶│   Stage 2    │───▶│   Stage 3    │       │
│  │ Syllabus │    │ Scraper  │    │ Text Extract │    │Visual Extract│       │
│  │ Crawler  │    │(Python)  │    │  + Classify  │    │(Non-Agentic) │       │
│  └──────────┘    └──────────┘    └──────────────┘    └──────┬───────┘       │
│                                                             │               │
│                                                             ▼               │
│                                                      ┌──────────────┐       │
│                                                      │   Stage 4    │       │
│                                                      │  MongoDB     │       │
│                                                      │ Ingestion    │       │
│                                                      └──────┬───────┘       │
│                                                             │               │
│                                                             ▼               │
│                                                      ┌──────────────┐       │
│                                                      │ Firebase     │       │
│                                                      │ Storage      │       │
│                                                      │ CDN Assets   │       │
│                                                      │(assets.edge- │       │
│                                                      │  nie.io)     │       │
│                                                      └──────┬───────┘       │
│                                                             │               │
│                                                             ▼               │
│                                                      ┌──────────────┐       │
│                                                      │   Stage 5    │       │
│                                                      │  Validation  │       │
│                                                      │  QA Agent    │       │
│                                                      └──────┬───────┘       │
│                                                             │               │
│                          ┌──────────────────────────────────┘               │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      MongoDB Database + Vector Index                  │  │
│  │  (Collections: subjects, papers, questions, answers, visuals, runs)   │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────┐     ┌────────────────┐                                 │
│  │  FastAPI        │◀───│  Next.js        │                                 │
│  │  Backend        │───▶│  Frontend       │                                 │
│  └────────────────┘     └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
EdGenie/
├── IMPLEMENTATION.md                    # ◀ Single source of truth for architecture
├── README.md                            # Quickstart & command reference
├── pyproject.toml                       # Monorepo dependencies & CLI entrypoints
├── syllabi/                             # Admin/Extracted 3-column syllabus CSVs
│   ├── PHYSICS_5054.csv
│   ├── ACCOUNTING_7707.csv
│   └── ...
├── pipeline/                            # Core pipeline implementation
│   ├── __init__.py
│   ├── config.py                        # Pipeline configuration settings
│   │
│   ├── syllabus/                        # STAGE 0 — Official Syllabus Crawler & Builder
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-syllabus` CLI (`build` & `list`)
│   │   ├── crawler.py                   # Cambridge website crawler (Playwright + PyMuPDF)
│   │   └── builder.py                   # Structured Gemini syllabus extractor & CSV builder
│   │
│   ├── scraper/                         # STAGE 1 — Scraper
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-scraper` CLI
│   │   ├── scraper.py                   # Main PapaCambridgeScraper
│   │   ├── downloader.py                # Retrying PDF downloader (tenacity)
│   │   ├── url_builder.py               # Year-aware URL builder
│   │   ├── subject_registry.py          # Subject mapping registry & session formatting
│   │   ├── subject_registry.json        # 58 O-Level & 115 A-Level subject dictionary
│   │   └── models.py                    # ScrapeTarget and ScrapeManifest Pydantic models
│   │
│   ├── extractor/                       # STAGE 2 — Text Extraction & Classification
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-extractor` CLI (`process` & `process-dir`)
│   │   ├── pymupdf_extractor.py         # PyMuPDF layout analysis extractor (MCQ/Essay QP/MS)
│   │   ├── classifier_agent.py          # Text-only Gemini/OpenRouter syllabus classifier
│   │   ├── paper_type_detector.py       # Document type classification (QP, MS, MCQ, Essay)
│   │   └── syllabus_loader.py           # Syllabus CSV loader & auto-resolution
│   │
│   ├── visuals/                         # STAGE 3 — Cambridge Visual Extractor (Non-Agentic)
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-visuals` CLI (`extract` & `extract-dir`)
│   │   └── extractor.py                 # CambridgeVisualExtractor (PyMuPDF + Pillow)
│   │
│   ├── ingestion/                       # STAGE 4 — PostgreSQL Ingestion & Firebase Storage
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-ingestor` CLI (`db-init`, `db-paper`, `db-dir`, `storage-paper`, `storage-dir`, `full-sync`)
│   │   ├── firebase_storage.py          # Firebase Storage Asset Manager with `https://assets.edgenie.io` CDN prefix
│   │   ├── postgres_client.py           # PostgreSQL client manager (6 core tables, foreign keys + GIN indexes)
│   │   └── postgres_ingestor.py         # PostgresIngestor document builder & atomic upsert service
│   │
│   ├── storage/                         # Local downloaded PDFs & output JSON/visuals
│   │   ├── syllabi_pdfs/                # Downloaded full Cambridge syllabus PDFs
│   │   └── {subject_code}/{year}/{session}/
│   │
│   └── tests/                           # Pytest unit & integration tests
│       ├── test_scraper.py
│       ├── test_extractor.py
│       ├── test_visuals.py
│       └── test_syllabus.py
```

---

## 4. Stage 0 — Syllabus Extraction & Management

### 4.1 Overview

Stage 0 automatically extracts official Cambridge International syllabus documents directly from `cambridgeinternational.org` and compiles them into structured 3-column CSV files (`Unit,Topic,Subtopic`) saved in `syllabi/`.

### 4.2 Cambridge Syllabus Crawler (`CambridgeSyllabusCrawler`)
- **Automated Discovery**: Navigates Cambridge International qualifications directory for O-Level, IGCSE, or AS/A-Level by subject name/code.
- **Section Parsing**: Locates the `Syllabuses` section on the subject page and enumerates all downloadable PDF links.
- **Flyer/Update Fi### 6.1 Architecture: Decoupled Layout Extraction & LLM Classification

Stage 2 uses a decoupled two-step hybrid pipeline designed to minimize LLM token costs, ensure deterministic layout parsing, and provide deep syllabus mapping:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Stage 2 Text Extraction                           │
│                                                                             │
│  [Step 2a: edgenie-extractor extract / extract-dir]                         │
│  ┌───────────────────────────┐         ┌─────────────────────────────────┐  │
│  │   1. Paper Type Detection │         │  2. Syllabus Resolution         │  │
│  │  (MCQ_QP, MCQ_MS, etc.)   │         │ (Auto-locates in syllabi/ dir)  │  │
│  └─────────────┬─────────────┘         └────────────────┬────────────────┘  │
│                │                                        │                   │
│                ▼                                        │                   │
│  ┌───────────────────────────┐                          │                   │
│  │ 3. PyMuPDF Layout Extract │                          │                   │
│  │  (Zero LLM Token Cost)    │                          │                   │
│  │  - Visual Row Clustering  │                          │                   │
│  │  - Table Density Filter   │                          │                   │
│  └─────────────┬─────────────┘                          │                   │
│                │                                        │                   │
│                ▼                                        │                   │
│  ┌───────────────────────────┐                          │                   │
│  │ 4. Link Visuals Manifest  │                          │                   │
│  │  (from Stage 3 manifest)  │                          │                   │
│  └─────────────┬─────────────┘                          │                   │
│                │                                        │                   │
│                ▼                                        │                   │
│  ┌───────────────────────────┐                          │                   │
│  │ 5. Save Skeleton JSON     │                          │                   │
│  │    (<stem>.json)          │                          │                   │
│  └─────────────┬─────────────┘                          │                   │
│                │                                        │                   │
│                ▼                                        ▼                   │
│  [Step 2b: edgenie-extractor classify / classify-dir]                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 6. Gemini / OpenRouter Classification Pass (ClassifierAgent)          │  │
│  │    Input: Question JSON + Syllabus CSV text                           │  │
│  │    Output: Unit, Topic, Sub_topic, Difficulty, Command_words          │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 7. Merge Classifications & Overwrite Final JSON (<stem>.json)         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Step 1: Pure Layout-Based Text Extraction (`edgenie-extractor extract / extract-dir`)
- **Zero LLM Token Usage**: Extracts full question text, options, sub-parts, and marks using PyMuPDF bounding-box geometry and line analysis.
- **Visual Row Clustering**: Clusters text lines into visual rows (chaining $y_0 \le 2.0\text{ pt}$) before sorting left-to-right by $x_0$, resolving inline option scrambling (e.g. `A mw B mW C Mw D MW`).
- **Table Density Filter**: Uses an 8% text density threshold ($\text{density} \ge 0.08$) to preserve vector-bordered tables as textual question content rather than misidentifying them as diagram artwork.
- **Boilerplate Stripping**: Removes header/footer furniture, UCLES copyright stamps, paper codes, "[Turn over" tags, and "BLANK PAGE" blocks.
- **Category Handlers**:
  - **`MCQ_QP`**: Extracts question stems and options `A, B, C, D` with disambiguated stem prefixes and preserved numeric values.
  - **`MCQ_MS`**: Multi-column table reader pairing question numbers with their corresponding answer key letters.
  - **`ESSAY_QP`**: Extracts main question stems, sub-parts `(a)/(b)/(i)/(ii)`, and marks `[N]`.
  - **`ESSAY_MS`**: Dual layout detection supporting modern table mark schemes (e.g. `1(a)`) and legacy margin mark schemes.
- **Immediate Disk Save**: Writes the extracted skeleton to disk as `<stem>.json` immediately before calling any LLM.

### 6.3 Step 2: Topic Classification & Difficulty Analysis (`edgenie-extractor classify / classify-dir`)
- **Text-Only LLM Pass**: Passes only the pre-extracted question JSON and official syllabus text to Gemini (with automatic OpenRouter fallback).
- **Enriched Fields**:
  - `Unit`: Syllabus unit number (e.g. `"1"`).
  - `Topic`: Syllabus topic number (e.g. `"1.2"`).
  - `Topic_name`: Official topic title from syllabus.
  - `Sub_topic`: List of specific learning objectives tested.
  - `Difficulty_score`: Integer from 1 to 10.
  - `Difficulty_label`: `'easy'`, `'medium'`, `'hard'`, or `'expert'`.
  - `Command_words`: List of identified command words (e.g. `["calculate", "explain"]`).
- **Targeted Update**: Only the classification and difficulty fields are merged back into `<stem>.json`.

### 6.4 Combined Extraction & Classification (`edgenie-extractor process / process-dir`)
- Provides a one-step command running both PyMuPDF layout extraction and Gemini topic classification in sequence.

### 6.5 Syllabus Auto-Resolution (`SyllabusLoader.resolve_syllabus_path`)
- Automatically resolves syllabus paths whether provided as `syllabi/PHYSICS_5054.csv`, `pipeline/syllabi/...`, filename only, or auto-discovered by subject code from the `syllabi/` directory.

### 6.6 File Exclusions & Natural Sorting
- Automatically excludes non-question/non-mark scheme PDFs: `_er_` (Examiner Reports), `_gt_` (Grade Thresholds), `_ir_` and `_ci_` (Confidential Instructions).
- `extract-dir`, `classify-dir`, and `process-dir` process files sequentially starting from the first stored file.

---

## 7. Stage 3 — Cambridge Visual Extractor (Non-Agentic)

Stage 3 (`CambridgeVisualExtractor`) extracts figures, diagrams, tables, circuit diagrams, and apparatus drawings from exam PDFs, tight-cropped around artwork with axis labels and annotations intact.

---

## 8. Stage 4 — PostgreSQL Ingestion (Relational + JSONB) & Firebase Asset Storage

### 8.1 Overview

Stage 4 decouples **Database Ingestion** from **Cloud Media Storage**:
1. **PostgreSQL Database**: Ingests all structured questions, answers, visual manifests, syllabus topics, and pipeline run logs into a production-grade relational database with JSONB payload flexibility and GIN indexes.
2. **Firebase Storage CDN**: Hosts all media assets (PDFs and extracted visual PNGs) served via custom CDN domain prefix **`https://assets.edgenie.io`**.

### 8.2 Cloud Storage Integration (Firebase Storage)

All PDFs and extracted visual PNGs in `pipeline/storage` are uploaded to Firebase Storage. Public asset links use the domain prefix `https://assets.edgenie.io`:

- **PDF CDN Link Format**:
  `https://assets.edgenie.io/pdfs/{subject_code}/{year}/{session}/{filename}.pdf`  
  *Example*: `https://assets.edgenie.io/pdfs/5054/2025/jun/5054_s25_qp_22.pdf`
- **Visual Image CDN Link Format**:
  `https://assets.edgenie.io/visuals/{paper_code}/{filename}.png`  
  *Example*: `https://assets.edgenie.io/visuals/5054_s25_qp_22/p5_q3_Fig_3_1.png`

### 8.3 PostgreSQL Database Architecture (Relational + JSONB)

PostgreSQL (with Relational Tables + JSONB columns) serves as EdGenie's core database. The `edgenie` database contains **6 core tables**:

#### 1. `subjects` Table
| Column | Type | Description |
|---|---|---|
| `subject_code` | `VARCHAR(10) PRIMARY KEY` | e.g. `'5054'` |
| `subject_name` | `VARCHAR(100) NOT NULL` | e.g. `'Physics'` |
| `level` | `VARCHAR(20) DEFAULT 'O-Level'` | `'O-Level'` or `'A-Level'` |
| `syllabus_rules_count` | `INT DEFAULT 0` | Total learning objective rules |
| `created_at` / `updated_at` | `TIMESTAMPTZ` | Timestamps |

#### 2. `papers` Table
| Column | Type | Description |
|---|---|---|
| `paper_code` | `VARCHAR(50) PRIMARY KEY` | e.g. `'5054_s25_qp_22'` |
| `subject_code` | `VARCHAR(10) REFERENCES subjects` | Foreign key with CASCADE |
| `year` | `INT NOT NULL` | e.g. `2025` |
| `session` | `VARCHAR(10) NOT NULL` | `'jun'`, `'nov'`, `'mar'` |
| `variant` | `VARCHAR(10)` | e.g. `'22'` |
| `paper_number` | `INT` | e.g. `2` |
| `paper_type` | `VARCHAR(20) NOT NULL` | `'qp'`, `'ms'`, `'er'`, `'gt'` |
| `category` | `VARCHAR(20) NOT NULL` | `'mcq_qp'`, `'essay_qp'`, `'mcq_ms'`, `'essay_ms'` |
| `pdf_filename` | `VARCHAR(255)` | e.g. `'5054_s25_qp_22.pdf'` |
| `pdf_cdn_url` | `TEXT` | `https://assets.edgenie.io/pdfs/...` |
| `question_count` | `INT DEFAULT 0` | Total questions |
| `metadata` | `JSONB DEFAULT '{}'` | Extensible metadata |

#### 3. `questions` Table (Primary Core Table)
| Column | Type | Description |
|---|---|---|
| `question_key` | `VARCHAR(100) PRIMARY KEY` | e.g. `'5054_s25_qp_22_Q1'` |
| `paper_code` | `VARCHAR(50) REFERENCES papers` | Foreign key with CASCADE |
| `subject_code` | `VARCHAR(10) REFERENCES subjects` | Foreign key |
| `year` / `session` / `variant` | `INT` / `VARCHAR` / `VARCHAR` | Partitionable metadata |
| `question_number` | `INT NOT NULL` | e.g. `1` |
| `question_number_formatted` | `VARCHAR(100)` | e.g. `'Jun/22/2025/Q1'` |
| `question_text` | `TEXT NOT NULL` | Question body / intro |
| `marks` | `INT DEFAULT 1` | Total question marks |
| `unit` / `topic` / `topic_name` | `VARCHAR` | Syllabus mappings |
| `sub_topics` | `JSONB DEFAULT '[]'` | Specific learning objective strings (Indexed with GIN) |
| `difficulty_score` | `INT DEFAULT 0` | Cognitive depth (1–10) |
| `difficulty_label` | `VARCHAR(20)` | `'easy'`, `'medium'`, `'hard'`, `'expert'` |
| `command_words` | `JSONB DEFAULT '[]'` | e.g. `["calculate", "state"]` (Indexed with GIN) |
| `options` | `JSONB` | For MCQs: `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `parts` | `JSONB` | For Essays: `[{"part_label": "(a)", "part_text": "...", "marks": 2, "image": null}]` |
| `image_filename` | `VARCHAR(255)` | Local visual image filename |
| `image_cdn_url` | `TEXT` | CDN image URL |
| `metadata` | `JSONB DEFAULT '{}'` | Raw payloads & multi-image arrays |

#### 4. `answers` Table
| Column | Type | Description |
|---|---|---|
| `id` | `VARCHAR(100) PRIMARY KEY` | e.g. `'5054_s25_ms_22_Q1'` |
| `question_key` | `VARCHAR(100)` | Logical link to `questions(question_key)` |
| `paper_code` | `VARCHAR(50) REFERENCES papers` | e.g. `'5054_s25_ms_22'` |
| `subject_code` / `year` / `session` | Relational fields | Filter metadata |
| `question_number` | `INT NOT NULL` | e.g. `1` |
| `answer_text` | `TEXT` | Full answer or MCQ correct letter |
| `parts` | `JSONB` | Essay MS parts: `[{"part_label": "(a)", "part_answer": "..."}]` (GIN Indexed) |
| `image_filename` / `image_cdn_url` | `VARCHAR` / `TEXT` | Visual image references |

#### 5. `visuals` Table
| Column | Type | Description |
|---|---|---|
| `id` | `VARCHAR(255) PRIMARY KEY` | e.g. `'5054_s25_qp_22_p5_q3_Fig_3_1.png'` |
| `paper_code` | `VARCHAR(50) REFERENCES papers` | Parent paper reference |
| `question_number` | `INT` | Associated question number |
| `caption` | `VARCHAR(100)` | e.g. `'Fig. 3.1'` or `'Table 1.1'` |
| `page` | `INT` | Page number in PDF |
| `bbox` | `JSONB` | Coordinates `[x0, y0, x1, y1]` |
| `filename` | `VARCHAR(255)` | Image filename |
| `relative_path` | `VARCHAR(255)` | Path inside paper visuals dir |
| `cdn_url` | `TEXT` | `https://assets.edgenie.io/visuals/...` |

#### 6. `pipeline_runs` Table
| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` | Run ID |
| `stage` | `VARCHAR(50)` | e.g. `'stage_4_ingestion'` |
| `subject_code` / `year` | Relational fields | Execution scope |
| `status` | `VARCHAR(20)` | `'completed'`, `'failed'` |
| `details` | `JSONB` | Execution details and metrics |
| `started_at` / `completed_at` | `TIMESTAMPTZ` | Timestamps |

---

## 9. Stage 5 — Validation / QA Agent

Stage 5 runs automated validation checks prior to publishing:
1. **Completeness Check**: Ensures all question numbers in a paper are present and sequential.
2. **Mark Scheme Link Check**: Verifies every question has a corresponding answer entry.
3. **Visual Integrity Check**: Verifies that every referenced CDN image URL starts with `https://assets.edgenie.io` and is accessible.

---

## 10. CLI Command Reference & End-to-End Workflow

### End-to-End Workflow Example (Full Subject Pipeline)

```bash
# 0. Build official syllabus CSV (Unit,Topic,Subtopic) from Cambridge International
edgenie-syllabus build --level o-level --subject physics --code 5054

# 1. Scrape O-Level Physics (5054) past papers (2010 to 2020)
edgenie-scraper scrape \
    --level o-level \
    --subject physics \
    --code 5054 \
    --year-start 2010 \
    --year-end 2020

# 2a. Extract text & question structure for all papers in the directory (0 LLM Tokens)
edgenie-extractor extract-dir --dir pipeline/storage/5054

# 2b. Run Gemini / OpenRouter AI classification pass to tag Unit, Topic, Difficulty, and Command Words
edgenie-extractor classify-dir --dir pipeline/storage/5054

# 3. Extract visual diagrams and patch question JSON files
edgenie-visuals extract-dir --dir pipeline/storage/5054

# 3. Extract visual diagrams and patch question JSON files
edgenie-visuals extract-dir --dir pipeline/storage/5054

# 4. Ingest structured JSONs into PostgreSQL database
edgenie-ingestor db-dir --dir pipeline/storage/5054

# 5. (Optional) Sync assets to Firebase Storage CDN (assets.edgenie.io)
edgenie-ingestor storage-dir --dir pipeline/storage/5054
```

### CLI Command Reference

| Stage | Command | Description |
|---|---|---|
| **Stage 0** | `edgenie-syllabus build` | Crawls Cambridge International, downloads full syllabus PDF, and builds 3-column CSV. |
| **Stage 0** | `edgenie-syllabus list` | Lists all built syllabus CSV files in `syllabi/`. |
| **Stage 1** | `edgenie-scraper scrape` | Downloads PDFs for given level, subject, code, and year range. |
| **Stage 2** | `edgenie-extractor extract` | Pure text extraction for a single PDF into JSON (0 LLM tokens). |
| **Stage 2** | `edgenie-extractor extract-dir` | Pure text extraction for all PDFs in a directory (0 LLM tokens). |
| **Stage 2** | `edgenie-extractor classify` | AI syllabus & difficulty classification on pre-extracted JSON (Gemini/OpenRouter). |
| **Stage 2** | `edgenie-extractor classify-dir` | AI syllabus & difficulty classification for all JSONs in a directory. |
| **Stage 2** | `edgenie-extractor process` | Combined pipeline: extracts layout text and classifies single PDF with Gemini. |
| **Stage 2** | `edgenie-extractor process-dir` | Combined pipeline: extracts layout text and classifies all PDFs in a directory. |
| **Stage 3** | `edgenie-visuals extract` | Extracts diagrams from a single QP PDF and patches sibling JSON. |
| **Stage 3** | `edgenie-visuals extract-dir` | Extracts diagrams from all QP PDFs in a directory. |
| **Stage 4** | `edgenie-ingestor db-init` | Initializes PostgreSQL tables, foreign key constraints, and B-Tree/GIN indexes. |
| **Stage 4** | `edgenie-ingestor db-paper` | Ingests a single paper JSON and its visual manifest into PostgreSQL. |
| **Stage 4** | `edgenie-ingestor db-dir` | Ingests all paper JSONs inside a directory tree into PostgreSQL. |
| **Stage 4** | `edgenie-ingestor db-stats` | Displays current record counts across all PostgreSQL tables. |
| **Stage 4** | `edgenie-ingestor storage-paper` | Uploads single paper PDF & visuals to Firebase Storage (`assets.edgenie.io`). |
| **Stage 4** | `edgenie-ingestor storage-dir` | Uploads local storage directory PDFs & images to Firebase Storage. |
| **Stage 4** | `edgenie-ingestor full-sync` | Automated step: performs both Firebase asset sync and PostgreSQL DB ingestion. |

---

## 11. JSON Output Schemas (Canonical)

### Single Question Record (`Stage 2 Output`)

```json
{
  "QuestionNumber": "Jun/41/2010/Q1",
  "PaperCode": "5054_s10_qp_41",
  "SubjectCode": "5054",
  "Year": 2010,
  "Session": "jun",
  "PaperNumber": 4,
  "Variant": 1,
  "QuestionText": "Fig. 1.1 shows a measuring cylinder containing water...",
  "Marks": 3,
  "Topic": "1.2",
  "Topic_name": "Measurement of Physical Quantities",
  "Sub_topics": ["Measuring cylinder", "Volume measurement"],
  "Difficulty_score": 4,
  "Difficulty_label": "medium",
  "Command_words": ["calculate", "state"],
  "image": "5054_s10_qp_41/p2_q1_Fig1_1.png"
}
```

---

## 12. Change Log

| Version | Date | Description |
|---|---|---|
| **1.0.0** | 2026-07-13 | Initial architecture proposal. |
| **1.5.0** | 2026-08-20 | Replaced legacy Playwright/pdf2svg web extractor with `CambridgeVisualExtractor`. |
| **2.0.0** | 2026-08-24 | Production Pipeline Update: Updated Stage 1 with `SubjectRegistry` (58 O-Level / 115 A-Level subjects) & year-aware session slugs (`may-june`/`oct-nov`). Updated Stage 3 with MCQ question-span division & watermark filtering. |
| **2.1.0** | 2026-08-26 | **Database & Cloud Storage Configuration**: Replaced PostgreSQL with **MongoDB** (6 core collections: `subjects`, `papers`, `questions`, `answers`, `visuals`, `pipeline_runs`). Integrated **Firebase Storage** asset hosting with custom CDN domain prefix **`https://assets.edgenie.io`** for all PDFs and visual images. Added `edgenie-ingestor` CLI commands. |
| **2.2.0** | 2026-08-30 | **Hybrid Stage 2 Text Extraction**: Fully replaced multimodal LLM text extraction with **PyMuPDF Layout Extractor** + **ClassifierAgent**. Zero LLM tokens used for text extraction; JSON saved immediately before classification; text-only Gemini pass (with OpenRouter fallback) populates syllabus topic classification and difficulty analysis. Auto-resolves syllabus paths and excludes `_er_`, `_gt_`, `_ir_`, `_ci_` files. |
| **2.3.0** | 2026-08-30 | **Decoupled Text Extraction & AI Classification**: Separated Stage 2 into distinct CLI commands (`extract`, `extract-dir`, `classify`, `classify-dir`, `process`, `process-dir`). Added visual row clustering ($y_0 \le 2.0\text{ pt}$) to fix same-row text scrambling, 8% text density filter for vector answer tables, and Option A prefix disambiguation. Synchronized all documentation across README.md and IMPLEMENTATION.md. |
| **2.4.0** | 2026-09-01 | **PostgreSQL (Relational + JSONB) Architecture & Decoupled Ingestion**: Transitioned primary database from MongoDB to **PostgreSQL (Relational + JSONB)** with `psycopg 3` driver. Decoupled database ingestion (`db-init`, `db-paper`, `db-dir`, `db-stats`) from Firebase Storage asset sync (`storage-paper`, `storage-dir`, `full-sync`). Enabled GIN indexing on `sub_topics`, `command_words`, `options`, `parts` JSONB fields. Fixed scissor-aware PDF vector clipping for Cambridge visual artwork. |
