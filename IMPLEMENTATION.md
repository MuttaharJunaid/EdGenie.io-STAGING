# EdGenie — Implementation Plan & Architecture Specification

> **Version**: 2.1.0  
> **Last Updated**: 2026-08-26  
> **Status**: Active / Production Architecture (MongoDB + Firebase Storage Asset CDN + Stages 1-4 Implemented)

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Stage 1 — Scraper (Non-Agentic)](#4-stage-1--scraper-non-agentic)
5. [Stage 2 — Text Extraction & Question Parser](#5-stage-2--text-extraction--question-parser)
6. [Stage 3 — Cambridge Visual Extractor (Non-Agentic)](#6-stage-3--cambridge-visual-extractor-non-agentic)
7. [Stage 4 — MongoDB Ingestion & Firebase Asset Storage](#7-stage-4--mongodb-ingestion--firebase-asset-storage)
8. [Stage 5 — Validation / QA Agent](#8-stage-5--validation--qa-agent)
9. [CLI Command Reference & End-to-End Workflow](#9-cli-command-reference--end-to-end-workflow)
10. [JSON Output Schemas (Canonical)](#10-json-output-schemas-canonical)
11. [Change Log](#11-change-log)

---

## 1. Product Overview

### Problem

CAIE O-Level and AS/A-Level students need targeted past paper practice to achieve top grades (`A*` / `A`). Exam papers are inherently unbalanced: **common easy topics appear frequently**, while **difficult/complex topics appear rarely**. Standard topical booklets or random paper solving leave students unprepared for the rare-but-challenging questions that actually determine final grade boundaries.

### Solution

**EdGenie** is a platform designed for colleges and students to systematically master past papers. For every CAIE subject (Physics, Chemistry, Biology, Mathematics, Accounting, Business Studies, Economics, etc.), EdGenie:

1. **Scrapes** past papers, mark schemes, and examiner reports from PapaCambridge across O-Level and AS/A-Level.
2. **Extracts** every question into canonical JSON, classifies it against syllabus topics, and estimates difficulty and historical topic frequency.
3. **Extracts visuals** (diagrams, graphs, tables, circuit diagrams, apparatus drawings) and binds them tightly to their parent question.
4. **Hosts Media Assets** (all PDFs and visual PNGs) on **Firebase Storage** served via custom CDN prefix `https://assets.edgenie.io`.
5. **Ingests** structured questions, answers, CDN media links, and vector embeddings into **MongoDB** (with MongoDB Vector Search).
6. **Validates** quality via automated test suites and QA checks before publishing.
7. **Serves** a question bank where students filter by **Topic × Difficulty × Frequency** to focus on high-yield, challenging questions.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EdGenie Pipeline                                  │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Stage 1  │───▶│   Stage 2    │───▶│   Stage 3    │───▶│   Stage 4    │  │
│  │ Scraper  │    │ Text Extract │    │Visual Extract│    │  MongoDB     │  │
│  │(Python)  │    │  + Classify  │    │(Non-Agentic) │    │ Ingestion    │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                                 │          │
│                                                                 ▼          │
│                                                          ┌──────────────┐  │
│                                                          │ Firebase     │  │
│                                                          │ Storage      │  │
│                                                          │ CDN Assets   │  │
│                                                          │(assets.edge- │  │
│                                                          │  nie.io)     │  │
│                                                          └──────┬───────┘  │
│                                                                 │          │
│                                                                 ▼          │
│                                                          ┌──────────────┐  │
│                                                          │   Stage 5    │  │
│                                                          │  Validation  │  │
│                                                          │  QA Agent    │  │
│                                                          └──────┬───────┘  │
│                                                                 │          │
│                          ┌──────────────────────────────────────┘          │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      MongoDB Database + Vector Index                  │  │
│  │  (Collections: subjects, papers, questions, answers, visuals, runs)   │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────┐     ┌────────────────┐                                 │
│  │  FastAPI        │◀───│  Next.js        │                                │
│  │  Backend        │───▶│  Frontend       │                                │
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
├── pipeline/                            # Core pipeline implementation
│   ├── __init__.py
│   ├── config.py                        # Pipeline configuration settings
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
│   ├── extractor/                       # STAGE 2 — Text & Question Parser
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-extractor` CLI (`process` & `process-dir`)
│   │   ├── text_extractor.py            # PyMuPDF text block & line extraction
│   │   ├── question_parser.py           # Structured question boundary parsing
│   │   ├── paper_type_detector.py       # Document type classification (QP, MS, MCQ, Essay)
│   │   ├── topic_classifier.py          # LangChain agent for syllabus topic mapping
│   │   ├── difficulty_analyzer.py       # Multi-signal difficulty estimator
│   │   └── syllabus_loader.py           # Admin CSV syllabus loader
│   │
│   ├── visuals/                         # STAGE 3 — Cambridge Visual Extractor (Non-Agentic)
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-visuals` CLI (`extract` & `extract-dir`)
│   │   └── extractor.py                 # CambridgeVisualExtractor (PyMuPDF + Pillow)
│   │
│   ├── ingestion/                       # STAGE 4 — MongoDB Ingestion & Firebase Storage
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-ingestor` CLI (`ingest-paper`, `ingest-dir`, `sync-storage`)
│   │   ├── firebase_storage.py          # Firebase Storage Asset Manager with `https://assets.edgenie.io` CDN prefix
│   │   ├── mongo_client.py              # MongoDB client manager (6 core collections + indexes)
│   │   └── ingestor.py                  # MongoIngestor document builder & upsert service
│   │
│   ├── storage/                         # Local downloaded PDFs & output JSON/visuals
│   │   └── {subject_code}/{year}/{session}/
│   │
│   └── tests/                           # Pytest unit & integration tests
│       ├── test_scraper.py
│       ├── test_extractor.py
│       └── test_visuals.py
```

---

## 4. Stage 1 — Scraper (Non-Agentic)

### 4.1 Overview

Stage 1 is a robust Python scraper that downloads past paper PDFs (Question Papers, Mark Schemes, Examiner Reports, Inserts) directly from PapaCambridge.

### 4.2 Subject Registry & URL Formatting

- **Subject Registry (`subject_registry.json` & `subject_registry.py`)**: Stores exact subject slugs for 58 O-Level subjects and 115 AS/A-Level subjects. Converts user input (`mathematics`, `physics`, `accounting`, `4024`, `5054`) into exact PapaCambridge slugs (e.g., `mathematics-d-calculator-version-4024`).
- **Year-Aware Session Formatting**:
  - **2018 onwards**: June session maps to `may-june`, November session maps to `oct-nov`, March session maps to `march`.  
    *Example*: `https://pastpapers.papacambridge.com/papers/caie/o-level-business-studies-7115-2018-oct-nov`
  - **Before 2018**: June session maps to `jun`, November session maps to `nov`, March session maps to `m`.  
    *Example*: `https://pastpapers.papacambridge.com/papers/caie/o-level-mathematics-d-calculator-version-4024-2017-jun`

---

## 5. Stage 2 — Text Extraction & Question Parser

Stage 2 parses downloaded PDF documents, detects paper categories, extracts structured questions with sub-parts and marks, maps answer keys (for MCQs), and classifies questions against syllabus topics and difficulty levels.

---

## 6. Stage 3 — Cambridge Visual Extractor (Non-Agentic)

Stage 3 (`CambridgeVisualExtractor`) extracts figures, diagrams, tables, circuit diagrams, and apparatus drawings from exam PDFs, tight-cropped around artwork with axis labels and annotations intact.

---

## 7. Stage 4 — MongoDB Ingestion & Firebase Asset Storage

### 7.1 Overview

Stage 4 hosts media assets on **Firebase Storage** served via custom CDN domain prefix **`https://assets.edgenie.io`** and ingests all structured questions, answers, visual manifests, and vector embeddings into **MongoDB**.

### 7.2 Cloud Storage Integration (Firebase Storage)

All PDFs and extracted visual PNGs in `pipeline/storage` are uploaded to Firebase Storage. Public asset links use the domain prefix `https://assets.edgenie.io`:

- **PDF CDN Link Format**:
  `https://assets.edgenie.io/pdfs/{subject_code}/{year}/{session}/{filename}.pdf`  
  *Example*: `https://assets.edgenie.io/pdfs/5054/2010/jun/5054_s10_qp_41.pdf`
- **Visual Image CDN Link Format**:
  `https://assets.edgenie.io/visuals/{paper_code}/{filename}.png`  
  *Example*: `https://assets.edgenie.io/visuals/5054_s10_qp_41/p2_q1_Fig1_1.png`

### 7.3 MongoDB Collections Architecture

MongoDB replaces PostgreSQL as the primary database. The `edgenie` database contains **6 core collections**:

#### 1. `subjects` Collection
```json
{
  "_id": "5054",
  "subject_code": "5054",
  "subject_name": "Physics",
  "level": "o-level",
  "syllabus_topics": [
    {
      "unit_number": "1",
      "unit_name": "General Physics",
      "topic_number": "1.1",
      "topic_name": "Physical Quantities and Units"
    }
  ],
  "updated_at": "2026-08-26T20:30:00Z"
}
```

#### 2. `papers` Collection
```json
{
  "_id": "5054_s10_qp_41",
  "paper_code": "5054_s10_qp_41",
  "subject_code": "5054",
  "year": 2010,
  "session": "jun",
  "paper_number": 4,
  "variant": 1,
  "pdf_filename": "5054_s10_qp_41.pdf",
  "pdf_cdn_url": "https://assets.edgenie.io/pdfs/5054/2010/jun/5054_s10_qp_41.pdf",
  "question_count": 8,
  "updated_at": "2026-08-26T20:30:00Z"
}
```

#### 3. `questions` Collection (Primary Collection)
```json
{
  "_id": "5054_s10_qp_41_Q1",
  "question_key": "5054_s10_qp_41_Q1",
  "paper_code": "5054_s10_qp_41",
  "subject_code": "5054",
  "year": 2010,
  "session": "jun",
  "paper_number": 4,
  "variant": 1,
  "question_number": 1,
  "question_text": "Fig. 1.1 shows a measuring cylinder containing water...",
  "marks": 3,
  "topic_number": "1.2",
  "topic_name": "Measurement of Physical Quantities",
  "sub_topics": ["Measuring cylinder", "Volume measurement"],
  "difficulty_score": 4,
  "difficulty_label": "medium",
  "command_words": ["calculate", "state"],
  "image_filename": "p2_q1_Fig1_1.png",
  "image_cdn_url": "https://assets.edgenie.io/visuals/5054_s10_qp_41/p2_q1_Fig1_1.png",
  "answer": {
    "text": "12.5 cm3 [1], reading at bottom of meniscus [1]",
    "marks": 2
  },
  "embedding": [0.012, -0.045, 0.089],
  "created_at": "2026-08-26T20:30:00Z"
}
```

#### 4. `answers` Collection
```json
{
  "_id": "5054_s10_qp_41_Q1_ans",
  "question_id": "5054_s10_qp_41_Q1",
  "paper_code": "5054_s10_ms_41",
  "subject_code": "5054",
  "year": 2010,
  "session": "jun",
  "question_number": 1,
  "answer_text": "12.5 cm3 [1], reading at bottom of meniscus [1]",
  "updated_at": "2026-08-26T20:30:00Z"
}
```

#### 5. `visuals` Collection
```json
{
  "_id": "5054_s10_qp_41_p2_q1_Fig1_1.png",
  "paper_code": "5054_s10_qp_41",
  "subject_code": "5054",
  "year": 2010,
  "session": "jun",
  "question_number": 1,
  "caption": "Fig. 1.1",
  "page": 2,
  "bbox": [170.3, 137.7, 386.6, 314.1],
  "filename": "p2_q1_Fig1_1.png",
  "cdn_url": "https://assets.edgenie.io/visuals/5054_s10_qp_41/p2_q1_Fig1_1.png"
}
```

#### 6. `pipeline_runs` Collection
```json
{
  "_id": "run_5054_2010_jun",
  "subject_code": "5054",
  "year": 2010,
  "session": "jun",
  "status": "completed",
  "processed_papers": 4,
  "ingested_questions": 40,
  "uploaded_assets": 12,
  "completed_at": "2026-08-26T20:30:00Z"
}
```

---

## 8. Stage 5 — Validation / QA Agent

Stage 5 runs automated validation checks prior to publishing:
1. **Completeness Check**: Ensures all question numbers in a paper are present and sequential.
2. **Mark Scheme Link Check**: Verifies every question has a corresponding answer entry.
3. **Visual Integrity Check**: Verifies that every referenced CDN image URL starts with `https://assets.edgenie.io` and is accessible.

---

## 9. CLI Command Reference & End-to-End Workflow

### End-to-End Workflow Example (Full Subject Pipeline)

```bash
# 1. Scrape O-Level Physics (5054) past papers (2010 to 2020)
edgenie-scraper scrape \
    --level o-level \
    --subject physics \
    --code 5054 \
    --year-start 2010 \
    --year-end 2020

# 2. Extract text & questions for all papers in the directory
edgenie-extractor process-dir --dir pipeline/storage/5054

# 3. Extract visual diagrams and patch question JSON files
edgenie-visuals extract-dir --dir pipeline/storage/5054

# 4. Sync assets to Firebase Storage (assets.edgenie.io) and ingest documents into MongoDB
edgenie-ingestor ingest-dir --dir pipeline/storage/5054
```

### CLI Command Reference

| Stage | Command | Description |
|---|---|---|
| **Stage 1** | `edgenie-scraper scrape` | Downloads PDFs for given level, subject, code, and year range. |
| **Stage 2** | `edgenie-extractor process` | Processes a single PDF into structured question JSON. |
| **Stage 2** | `edgenie-extractor process-dir` | Processes all PDFs in a directory. |
| **Stage 3** | `edgenie-visuals extract` | Extracts diagrams from a single QP PDF and patches sibling JSON. |
| **Stage 3** | `edgenie-visuals extract-dir` | Extracts diagrams from all QP PDFs in a directory. |
| **Stage 4** | `edgenie-ingestor ingest-paper` | Syncs single paper assets to Firebase Storage (`assets.edgenie.io`) & MongoDB. |
| **Stage 4** | `edgenie-ingestor ingest-dir` | Syncs full directory assets to Firebase Storage & MongoDB. |
| **Stage 4** | `edgenie-ingestor sync-storage` | Uploads local storage directory PDFs & images to Firebase Storage. |

---

## 10. JSON Output Schemas (Canonical)

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

## 11. Change Log

| Version | Date | Description |
|---|---|---|
| **1.0.0** | 2026-07-13 | Initial architecture proposal. |
| **1.5.0** | 2026-08-20 | Replaced legacy Playwright/pdf2svg web extractor with `CambridgeVisualExtractor`. |
| **2.0.0** | 2026-08-24 | Production Pipeline Update: Updated Stage 1 with `SubjectRegistry` (58 O-Level / 115 A-Level subjects) & year-aware session slugs (`may-june`/`oct-nov`). Updated Stage 3 with MCQ question-span division & watermark filtering. |
| **2.1.0** | 2026-08-26 | **Database & Cloud Storage Configuration**: Replaced PostgreSQL with **MongoDB** (6 core collections: `subjects`, `papers`, `questions`, `answers`, `visuals`, `pipeline_runs`). Integrated **Firebase Storage** asset hosting with custom CDN domain prefix **`https://assets.edgenie.io`** for all PDFs and visual images. Added `edgenie-ingestor` CLI commands. |
