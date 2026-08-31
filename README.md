# EdGenie

AI-powered past paper analysis, text extraction, visual diagram segmentation, and syllabus classification platform for CAIE O-Level and AS/A-Level students.

---

## Installation & Setup

```bash
# 1. Clone the repository and navigate to project root
cd EdGenie

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Environment Variables (`.env`)

Create a `.env` file in the project root:
```ini
# Google Gemini API Key (Primary for Syllabus & Topic Classification)
GOOGLE_API_KEY=your_gemini_api_key_here

# OpenRouter API Key (Automatic fallback)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# PostgreSQL Connection (Relational + JSONB)
POSTGRES_URI=postgresql://localhost:5432/edgenie

# Firebase Storage CDN (Stage 4 Media Hosting)
CDN_DOMAIN_PREFIX=https://assets.edgenie.io
GOOGLE_APPLICATION_CREDENTIALS=firebase-key.json
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
```

---

## PostgreSQL Setup Guide (Relational + JSONB)

EdGenie uses **PostgreSQL 15+** with relational tables for strict indexing / foreign keys and **JSONB** columns for sub-topics, command words, options, and structured question parts.

### 1. Local Database Setup (macOS / Homebrew)
```bash
# Start PostgreSQL service
brew services start postgresql@15

# Create edgenie database (if not already existing)
createdb edgenie
```

### 2. Initialize EdGenie Tables & Indexes
```bash
edgenie-ingestor db-init
```
This automatically sets up the 6 relational tables (`subjects`, `papers`, `questions`, `answers`, `visuals`, `pipeline_runs`) along with GIN indexes on JSONB fields.

---

## End-to-End Subject Workflow (Start to Finish)

To run the complete pipeline — **Stage 0 (Syllabus) → Stage 1 (Scraping) → Stage 2 (Text Extraction & Classification) → Stage 3 (Visual Diagram Extraction) → Stage 4 (PostgreSQL DB Ingestion & Firebase Asset Storage)** — follow these commands.

### Example: Complete Pipeline for O-Level Physics (5054)

#### Stage 0: Build Official Syllabus CSV
```bash
# Crawl Cambridge International website, download full syllabus PDF, and build 3-column CSV (Unit,Topic,Subtopic)
edgenie-syllabus build --level o-level --subject physics --code 5054
```

#### Stage 1: Scrape Past Papers & Mark Schemes
```bash
# Scrape O-Level Physics (5054) papers from 2010 to 2020 (June & Nov sessions)
edgenie-scraper scrape \
    --level o-level \
    --subject physics \
    --code 5054 \
    --year-start 2010 \
    --year-end 2020
```

#### Stage 2: Text Extraction & Topic Classification

You can run text extraction and classification **separately** (recommended for zero-token extraction followed by batch classification) or in a **single combined step**:

##### Option A: Separated Commands (Extraction then Classification)
```bash
# Step 2a: Extract text layout, question structure, sub-parts, and marks for all PDFs (0 LLM Tokens)
edgenie-extractor extract-dir --dir pipeline/storage/5054

# Step 2b: Run Gemini / OpenRouter AI classification pass to tag Unit, Topic, Sub-topic, Difficulty, and Command Words
edgenie-extractor classify-dir --dir pipeline/storage/5054
```

##### Option B: Combined Command (Extraction + Classification in One Step)
```bash
# Runs both PyMuPDF layout extraction and Gemini topic classification in a single command
edgenie-extractor process-dir --dir pipeline/storage/5054
```

#### Stage 3: Visual Diagram & Figure Extraction
```bash
# Extract all labelled figures, graphs, tables, and diagrams, linking them to questions in JSON
edgenie-visuals extract-dir --dir pipeline/storage/5054
```

#### Stage 4: PostgreSQL Database Ingestion & Asset Sync

Stage 4 is **decoupled** — you can ingest records directly into PostgreSQL without requiring Firebase cloud credentials:

```bash
# Ingest all structured JSON questions, answers, and visual manifests into PostgreSQL
edgenie-ingestor db-dir --dir pipeline/storage/5054

# View current database counts across all tables
edgenie-ingestor db-stats

# (Optional) Upload all PDFs and visual images to Firebase Storage (assets.edgenie.io)
edgenie-ingestor storage-dir --dir pipeline/storage/5054
```

---

## Single Paper Commands

If you want to run stages on a single PDF paper (e.g. `5054_w18_qp_41.pdf`):

```bash
# Stage 2 (Option A): Pure text extraction (0 LLM tokens, saves 5054_w18_qp_41.json)
edgenie-extractor extract --pdf pipeline/storage/5054/2018/nov/5054_w18_qp_41.pdf

# Stage 2 (Option A): Gemini syllabus classification on pre-extracted JSON
edgenie-extractor classify --json pipeline/storage/5054/2018/nov/5054_w18_qp_41.json

# Stage 2 (Option B): Combined extraction + classification for a single paper
edgenie-extractor process --pdf pipeline/storage/5054/2018/nov/5054_w18_qp_41.pdf

# Stage 3: Extract visual diagrams from a single paper
edgenie-visuals extract --pdf pipeline/storage/5054/2018/nov/5054_w18_qp_41.pdf

# Stage 4: Ingest a single paper's JSON and visual manifest into PostgreSQL
edgenie-ingestor db-paper --json pipeline/storage/5054/2018/nov/5054_w18_qp_41.json

# Stage 4: (Optional) Upload a single paper's PDF and visuals to Firebase Storage
edgenie-ingestor storage-paper --pdf pipeline/storage/5054/2018/nov/5054_w18_qp_41.pdf
```

---

## Complete CLI Command Reference

### 1. Syllabus Manager (`edgenie-syllabus`)
- **Build syllabus CSV**: `edgenie-syllabus build --level <level> --subject <subject> --code <code>`
- **List available syllabi**: `edgenie-syllabus list`

### 2. Scraper (`edgenie-scraper`)
- **Scrape year range**: `edgenie-scraper scrape --level o-level --subject physics --code 5054 --year-start 2010 --year-end 2020`
- **Scrape single session**: `edgenie-scraper scrape --level o-level --subject accounting --code 7707 --year 2023 --session jun`
- **Dry-run (preview URLs without downloading)**: `edgenie-scraper scrape --level o-level --subject mathematics-d --code 4024 --year-start 2010 --year-end 2020 --dry-run`

### 3. Text Extractor & Classifier (`edgenie-extractor`)
- **Pure Text Extraction (Single PDF, 0 tokens)**: `edgenie-extractor extract --pdf <path_to_pdf>`
- **Pure Text Extraction (Directory, 0 tokens)**: `edgenie-extractor extract-dir --dir <path_to_dir>`
- **Gemini Classification (Single JSON/PDF)**: `edgenie-extractor classify --json <path_to_json> [--syllabus <path_to_csv>]`
- **Gemini Classification (Directory)**: `edgenie-extractor classify-dir --dir <path_to_dir> [--syllabus <path_to_csv>]`
- **Combined Process (Single PDF)**: `edgenie-extractor process --pdf <path_to_pdf> [--syllabus <path_to_csv>]`
- **Combined Process (Directory)**: `edgenie-extractor process-dir --dir <path_to_dir> [--syllabus <path_to_csv>]`

### 4. Visual Extractor (`edgenie-visuals`)
- **Extract single paper diagrams**: `edgenie-visuals extract --pdf <path_to_pdf>`
- **Extract directory diagrams**: `edgenie-visuals extract-dir --dir <path_to_dir>`

### 5. PostgreSQL & Firebase Storage Ingestor (`edgenie-ingestor`)
- **Initialize PostgreSQL schema**: `edgenie-ingestor db-init`
- **Ingest single paper into PostgreSQL**: `edgenie-ingestor db-paper --json <path_to_json>`
- **Ingest full directory into PostgreSQL**: `edgenie-ingestor db-dir --dir <path_to_dir>`
- **Check PostgreSQL table record counts**: `edgenie-ingestor db-stats`
- **Upload single paper assets to Firebase CDN**: `edgenie-ingestor storage-paper --pdf <path_to_pdf>`
- **Upload directory assets to Firebase CDN**: `edgenie-ingestor storage-dir --dir <path_to_dir>`
- **Combined asset upload & DB ingestion**: `edgenie-ingestor full-sync --dir <path_to_dir>`

