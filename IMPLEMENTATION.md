# EdGenie — Implementation Plan & Architecture Specification

> **Version**: 2.0.0  
> **Last Updated**: 2026-08-24  
> **Status**: Active / Production Architecture (Stages 1, 2, and 3 Fully Implemented)

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Stage 1 — Scraper (Non-Agentic)](#4-stage-1--scraper-non-agentic)
5. [Stage 2 — Text Extraction & Question Parser](#5-stage-2--text-extraction--question-parser)
6. [Stage 3 — Cambridge Visual Extractor (Non-Agentic)](#6-stage-3--cambridge-visual-extractor-non-agentic)
7. [Stage 4 — DB Ingestion (Non-Agentic)](#7-stage-4--db-ingestion-non-agentic)
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
4. **Ingests** structured questions, answers, and visual assets into PostgreSQL with `pgvector` for vector/semantic search.
5. **Validates** quality via automated test suites and QA checks before publishing.
6. **Serves** a question bank where students filter by **Topic × Difficulty × Frequency** to focus on high-yield, challenging questions.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EdGenie Pipeline                                  │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Stage 1  │───▶│   Stage 2    │───▶│   Stage 3    │───▶│   Stage 4    │  │
│  │ Scraper  │    │ Text Extract │    │Visual Extract│    │ DB Ingestion │  │
│  │(Python)  │    │  + Classify  │    │(Non-Agentic) │    │  (Python)    │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘  │
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
│  │                     PostgreSQL + pgvector                             │  │
│  │  (Questions, Answers, Visuals, Topics, Difficulty, Frequency, etc.)  │  │
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
│   │   ├── scraper.py                   # Main PapaCambridgeScraper (requests + BeautifulSoup + Playwright fallback)
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
│   │   ├── question_parser.py           # Structured question boundary parsing & MCQ answer key mapping
│   │   ├── paper_type_detector.py       # Document type classification (QP, MS, MCQ, Essay)
│   │   ├── topic_classifier.py          # LangChain agent for syllabus topic mapping
│   │   ├── difficulty_analyzer.py       # Multi-signal difficulty estimator (Bloom's taxonomy)
│   │   └── syllabus_loader.py           # Admin CSV syllabus loader
│   │
│   ├── visuals/                         # STAGE 3 — Cambridge Visual Extractor (Non-Agentic)
│   │   ├── __init__.py
│   │   ├── cli.py                       # `edgenie-visuals` CLI (`extract` & `extract-dir`)
│   │   └── extractor.py                 # CambridgeVisualExtractor (PyMuPDF + Pillow)
│   │
│   ├── ingestion/                       # STAGE 4 — DB Ingestion (PostgreSQL + pgvector)
│   │   └── ingestor.py
│   │
│   ├── storage/                         # Downloaded PDFs & output JSON/visuals
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

- **Subject Registry (`subject_registry.json` & `subject_registry.py`)**: Stores exact subject slugs for 58 O-Level subjects and 115 AS/A-Level subjects mapped from PapaCambridge's directory index. Users can provide common keywords (e.g. `mathematics`, `physics`, `accounting`) or syllabus codes (`4024`, `5054`, `7707`, `9706`), and the registry resolves the exact PapaCambridge slug (e.g., `mathematics-d-calculator-version-4024`).
- **Year-Aware Session Formatting**:
  - **2018 onwards**: June session maps to `may-june`, November session maps to `oct-nov`, March session maps to `march`.  
    *Example*: `https://pastpapers.papacambridge.com/papers/caie/o-level-business-studies-7115-2018-oct-nov`
  - **Before 2018**: June session maps to `jun`, November session maps to `nov`, March session maps to `m`.  
    *Example*: `https://pastpapers.papacambridge.com/papers/caie/o-level-mathematics-d-calculator-version-4024-2017-jun`

### 4.3 PDF Filename Standard

PDF filenames downloaded by the scraper follow the official CAIE naming format:
```
{subject_code}_{session_code}{year_short}_{paper_type}_{variant}.pdf

Examples:
  5054_s10_qp_41.pdf    → Physics (5054), Summer 2010, Question Paper 41
  5054_s10_ms_41.pdf    → Physics (5054), Summer 2010, Mark Scheme 41
  7115_w18_qp_12.pdf    → Business Studies (7115), Winter 2018, Question Paper 12
```

---

## 5. Stage 2 — Text Extraction & Question Parser

### 5.1 Overview

Stage 2 parses downloaded PDF documents, detects paper categories, extracts structured questions with sub-parts and marks, maps answer keys (for MCQs), and classifies questions against syllabus topics and difficulty levels.

### 5.2 Key Components

1. **`PaperTypeDetector`**: Classifies PDFs into `QUESTION_PAPER_ESSAY`, `QUESTION_PAPER_MCQ`, `MARK_SCHEME_ESSAY`, `MARK_SCHEME_MCQ`, `EXAMINER_REPORT`, `INSERT`, `GRADE_THRESHOLD`.
2. **`TextExtractor`**: Extracts clean text blocks using PyMuPDF while preserving line coordinates, font sizes, and bold attributes.
3. **`QuestionParser`**:
   - Parses structured question numbers, sub-parts `(a)`, `(b)`, `(i)`, `(ii)`, and mark allocations `[3]`.
   - Parses MCQ answer keys (A, B, C, D) from mark scheme papers.
4. **`TopicClassifierAgent`**: Uses LangChain to match extracted question text against admin-supplied syllabus CSV units and topics.
5. **`DifficultyAnalyzerAgent`**: Evaluates difficulty (1–10 score & label: `easy`, `medium`, `hard`, `expert`) based on Bloom's taxonomy command words (`calculate`, `explain`, `evaluate`, `justify`), mark allocations, and question complexity.

---

## 6. Stage 3 — Cambridge Visual Extractor (Non-Agentic)

> [!IMPORTANT]
> **All Agentic / LLM vision logic (Gemini Vision `classifier.py`, `web_extractor.py`, `playwright`, `pdf2svg`) was removed.**
> Visual extraction is now **100% non-agentic, deterministic, layout-anchored, and local**, executing in <0.5s per paper using PyMuPDF (`fitz`) and Pillow (`PIL`).

### 6.1 Overview

`CambridgeVisualExtractor` (`pipeline/visuals/extractor.py`) extracts graphs, diagrams, circuit diagrams, apparatus drawings, tables, and visual options directly from exam PDFs, tight-cropped around the artwork with all axis labels, tick numbers, and component text intact.

### 6.2 Extraction Strategies

#### A. Structured Papers (Papers 21, 31, 41, etc.)
1. **Caption Anchoring**: Matches isolated caption lines using regex matching `Fig. N.M` / `Figure N.M` (e.g. `Fig. 1.1`, `Figure 2.3`).
2. **Wrapped-Sentence Bleed Prevention**: Walks backward from captions to detect preceding paragraph bounds. Short wrapped sentence tails are chained to prevent prose text from bleeding into figure crops.
3. **Degenerate-Rect Union Handling**: Uses custom manual bounding union (`rect_union`) and touch tests (`rects_touch`) to prevent PyMuPDF from silently dropping zero-width or zero-height straight vector lines.
4. **Banded Stitch Rendering**: Splits large vertical blank spaces between artwork and caption into distinct bands and stitches them with a small visual gap (`render_stitched`), eliminating empty student answer spaces.
5. **Overlap Suppression**: Suppresses duplicate/partial fallback crops when geometry falls inside a claimed caption band.

#### B. MCQ Papers (Papers 11, 12, 13, etc.)
1. **Question Anchor Detection**: Recognizes MCQ question starts at `x0 ≤ 58` in both combined (`"5 Two men..."`) and split-line (`"1"` alone on line) formats using `MCQ_Q_PATTERN` and `MCQ_Q_SOLO`.
2. **Question Span Division**: Divides pages into question spans (`q_y0 → next_q_y0`).
3. **Cluster Grouping & Multi-Image Output**: Groups vector clusters that are vertically close (<35pt gap). Distinct visual items under a question are extracted as independent sequential images (`p6_q11_mcq_1.png`, `p6_q11_mcq_2.png`).
4. **Label Bounding Box Expansion**: Expands crop bounds to include all associated diagram labels, axis text, and component annotations (e.g. `'resistance wire'`, `'brass'`, `'iron'`, `'plastic'`, `'air'`), preventing text truncation.
5. **Table Preservation**: Preserves all visual tables (data tables, option comparison matrices) while ignoring thin full-width page divider lines.

#### C. Quality & Watermark Filtering
- **PapaCambridge Watermark Filter**: Inspects raw embedded image XObjects (`_is_papacambridge_ribbon`) using image `xref` dark-pixel fraction (>50% dark) to remove diagonal PapaCambridge watermark ribbons.
- **Blank Image Filter**: Samples pixel RGB values (`_is_blank_image`) to discard near-100% white margin swatches.

### 6.3 Stage 2 JSON Auto-Linking

When running Stage 3 via `edgenie-visuals extract` or `extract-dir`, Stage 3 automatically finds the sibling Stage 2 question JSON file and patches each question record's `"image"` field with the extracted relative visual path:
```json
{
  "QuestionNumber": "Jun/41/2010/Q1",
  "image": "5054_s10_qp_41/p2_q1_Fig1_1.png"
}
```

---

## 7. Stage 4 — DB Ingestion (Non-Agentic)

Stage 4 ingests structured questions, answer keys, and visual manifests into PostgreSQL with `pgvector`.

### Database Schema Core Tables
- `subjects`: Syllabus metadata (`code`, `name`, `level`).
- `papers`: Past paper records (`year`, `session`, `paper_number`, `variant`).
- `questions`: Question text, marks, syllabus topic, sub-topics, difficulty score, difficulty label, frequency rank, and visual image paths.
- `answers`: Mark scheme answers and criteria linked via `question_id`.
- `visuals`: Metadata for extracted images (`filename`, `relative_path`, `bbox`, `page`).

---

## 8. Stage 5 — Validation / QA Agent

Stage 5 runs automated validation checks prior to publishing:
1. **Completeness Check**: Ensures all question numbers in a paper are present and sequential.
2. **Mark Scheme Link Check**: Verifies every question has a corresponding answer entry.
3. **Visual Integrity Check**: Verifies that every referenced `"image"` path exists on disk and has valid dimensions.

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
```

### CLI Command Reference

| Stage | Command | Description |
|---|---|---|
| **Stage 1** | `edgenie-scraper scrape` | Downloads PDFs for given level, subject, code, and year range. |
| **Stage 2** | `edgenie-extractor process` | Processes a single PDF into structured question JSON. |
| **Stage 2** | `edgenie-extractor process-dir` | Processes all PDFs in a directory. |
| **Stage 3** | `edgenie-visuals extract` | Extracts diagrams from a single QP PDF and patches sibling JSON. |
| **Stage 3** | `edgenie-visuals extract-dir` | Extracts diagrams from all QP PDFs in a directory. |

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

### Visual Manifest Record (`Stage 3 Output`)

```json
{
  "page": 2,
  "question": 1,
  "caption": "Fig. 1.1",
  "bbox": [170.3, 137.7, 386.6, 314.1],
  "filename": "p2_q1_Fig1_1.png",
  "file": "p2_q1_Fig1_1.png",
  "relative_path": "5054_s10_qp_41/p2_q1_Fig1_1.png"
}
```

---

## 11. Change Log

| Version | Date | Description |
|---|---|---|
| **1.0.0** | 2026-07-13 | Initial architecture proposal. |
| **1.5.0** | 2026-08-20 | Replaced legacy Playwright/pdf2svg web extractor with `CambridgeVisualExtractor`. |
| **2.0.0** | 2026-08-24 | **Production Pipeline Update**: Completely updated Stage 1 scraper with `SubjectRegistry` (58 O-Level / 115 A-Level subjects) & year-aware session slugs (`may-june`/`oct-nov`). Updated Stage 3 with MCQ question-span division, diagram multi-image splitting, text label bbox expansion, PapaCambridge watermark filtering, and automated Stage 2 JSON image linking. Removed all legacy LLM/agentic visual extraction code. |
