# EdGenie

AI-powered past paper analysis, text extraction, visual diagram segmentation, and syllabus classification platform for CAIE O-Level and AS/A-Level students.

EdGenie unifies text extraction, visual figure snipping, and official Cambridge syllabus classification into a single cloud pass powered by the **Unstructured.io Transform API** and **PyMuPDF**, paired with a **PostgreSQL (Relational + JSONB)** ingestion engine and **Firebase Storage CDN** asset pipeline.

---

## Installation & Setup

```bash
# 1. Clone the repository and navigate to project root
cd EdGenie

# 2. Create and activate a virtual environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Environment Variables (`.env`)

Create a `.env` file in the project root:

```ini
# Unstructured.io Transform API Key (Unified 3-in-1 Extraction)
UNSTRUCTURED_API_KEY=your_unstructured_api_key_here
UNSTRUCTURED_SERVER_URL=https://transform.unstructured.io

# PostgreSQL Connection (Relational + JSONB Storage)
POSTGRES_URI=postgresql://localhost:5432/edgenie

# Firebase Storage & Asset CDN (Optional Stage 3 Media Hosting)
CDN_DOMAIN_PREFIX=https://assets.edgenie.io
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
GOOGLE_APPLICATION_CREDENTIALS=firebase-key.json
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

## End-to-End Workflow

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

### Example: Complete Workflow for O-Level Physics (5054)

#### Stage 1: Scrape Past Papers & Mark Schemes

The EdGenie scraper operates on a **subject code-only** model (`--code` / `-c`), automatically resolving qualification level (O-Level vs AS/A-Level) and subject title via the built-in registry. It uses the hierarchical **`PapaCambridgeNavigator`** to locate official subject DOM containers, map active sessions, and extract direct download links, reinforced by slug anomaly handling and automatic post-scraping prefix validation.

##### A. Single Session Scraping Command
```bash
# Scrape a single session for Physics 5054 (code-only; level and subject auto-resolved)
edgenie-scraper scrape --code 5054 --year 2023 --session jun
```

##### B. Complete Multi-Year Directory Scraping Command
```bash
# Scrape full directory of papers across multiple years and sessions
edgenie-scraper scrape \
    --code 5054 \
    --year-start 2015 \
    --year-end 2024 \
    --session jun \
    --session nov
```

##### C. Directory Inspection & Validation Commands
```bash
# Validate downloaded PDF prefixes in session directories (deletes corrupted/mismatched folders)
edgenie-scraper validate --code 5054 --year-start 2015 --year-end 2024

# List all downloaded files and scrape manifests for a subject directory
edgenie-scraper list-papers --code 5054 --year 2023
```

---

#### Stage 2: Unified 3-in-1 Extraction & Classification

A single call to `edgenie-unified` submits the PDF to the Unstructured.io Transform API with strict OpenAI-compliant JSON schemas and official Cambridge syllabus reference:

1. **Text Extraction**: Extracts full question stems, nested sub-parts (`(a)`, `(b)(i)`, etc.), mark allocations, and options (A/B/C/D for MCQs).
2. **Visual Bounding Boxes**: Detects layout bounding boxes for diagrams, graphs, circuits, and tables in the same cloud pass.
3. **Syllabus Classification**: Inlines the subject syllabus CSV directly into the prompt so Unit, Topic, Sub-topic, Difficulty, and Command Words are classified simultaneously.
4. **Local PyMuPDF Cropping**: Filters cover-page logos and header/footer artifacts, crops visual assets at 300 DPI, and patches relative image paths directly into the question JSON tree.

##### A. Single PDF Commands (`extract` / `extract-visuals-text-and-classify`)
```bash
# Single exam paper (syllabus auto-discovered from syllabi/ directory)
edgenie-unified extract --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# Explicit syllabus CSV and force re-extraction (bypassing local cache)
edgenie-unified extract \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --syllabus syllabi/PHYSICS_5054.csv \
    --force

# Custom DPI or paper type override
edgenie-unified extract \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_12.pdf \
    --type mcq \
    --dpi 300
```
*(Note: `edgenie-unified extract-visuals-text-and-classify` and `edgenie-extractor extract` are identical aliases.)*

##### B. Complete Directory Batch Extraction Commands (`extract-dir`)
```bash
# Batch extract an entire session directory
edgenie-unified extract-dir --dir pipeline/storage/5054/2023/jun/

# Batch extract an entire multi-year subject directory tree
edgenie-unified extract-dir --dir pipeline/storage/5054/

# Force re-extraction across directory with custom DPI
edgenie-unified extract-dir --dir pipeline/storage/5054/2023/jun/ --force --dpi 300
```

> **Directory Processing Features**:
> - Recursively discovers all question papers (`_qp_`) and mark schemes (`_ms_`).
> - Automatically ignores non-exam files: Examiner Reports (`_er`), Grade Thresholds (`_gt`), Confidential Instructions (`_ci`), and Syllabus Guides (`_sy`).
> - Auto-detects paper category (Essay QP, MCQ QP, Essay MS, MCQ MS) and resolves syllabus CSV per subject code.
##### C. Job Timeout Recovery Commands (`recover-job` / `recover-timed-out`)
```bash
# Recover a specific timed-out job via playground endpoint and complete processing
edgenie-unified recover-job \
    --job-id b60382ec-1f13-4a8d-9946-d7171d977e9a \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# Automatically recover all pending timed-out jobs noted in timed_out_jobs.json
edgenie-unified recover-timed-out --file timed_out_jobs.json
```

---

#### Stage 3: PostgreSQL Database Ingestion & Cloud Asset Hosting

Stage 3 is decoupled — you can ingest records directly into PostgreSQL without requiring cloud credentials:

##### A. Single Paper / JSON Commands (`db-paper`)
```bash
# Ingest single paper JSON and visual manifest into PostgreSQL
edgenie-ingestor db-paper --json pipeline/storage/5054/2023/jun/5054_s23_qp_22.json
```

##### B. Complete Directory Commands (`db-dir`, `storage-dir`, `full-sync`)
```bash
# Ingest all extracted papers in a directory tree into PostgreSQL
edgenie-ingestor db-dir --dir pipeline/storage/5054

# View current database record counts across all PostgreSQL tables
edgenie-ingestor db-stats

# (Optional) Upload all PDFs and visual images in a directory to Firebase Storage CDN
edgenie-ingestor storage-dir --dir pipeline/storage/5054

# (Optional) Complete full sync (both Firebase CDN upload and PostgreSQL ingestion)
edgenie-ingestor full-sync --dir pipeline/storage/5054
```

---

## Core Subject Syllabi

Official Cambridge syllabus CSVs are pre-loaded in the `syllabi/` folder:

| Subject | Code | Syllabus File |
|---|---|---|
| Physics | 5054 | `syllabi/PHYSICS_5054.csv` |
| Chemistry | 5070 | `syllabi/CHEMISTRY_5070.csv` |
| Biology | 5090 | `syllabi/BIOLOGY_5090.csv` |
| Accounting | 7707 | `syllabi/ACCOUNTING_7707.csv` |
| Economics | 2281 | `syllabi/ECONOMICS_2281.csv` |
| Mathematics Syllabus D | 4024 | `syllabi/MATHEMATICS_4024.csv` |
| Computer Science | 2210 | `syllabi/COMPUTER_SCIENCE_2210.csv` |
| Business Studies | 7115 | `syllabi/BUSINESS_7115.csv` |
| English Language | 1123 | `syllabi/ENGLISH_1123.csv` |

---

## Output File Structure

For any processed question paper PDF (e.g. `5054_s23_qp_22.pdf`):

```
pipeline/storage/5054/2023/jun/
├── 5054_s23_qp_22.pdf                  # Raw Cambridge exam paper
├── 5054_s23_qp_22.json                 # Canonical structured question JSON (text + classification + visual links)
├── 5054_s23_qp_22_unified_text.json    # Local raw text extraction cache
├── 5054_s23_qp_22_unified_elements.json# Local layout elements cache
└── 5054_s23_qp_22/                     # Visual assets folder
    ├── manifest.json                   # Visual metadata & coordinates manifest
    ├── p2_q1_question_diagram_1.png
    ├── p4_q3_part_a_Fig_3_1.png
    └── p7_q5_part_b_Table_5_1.png
```

---

## Running Automated Tests

```bash
# Run the complete test suite (Unified pipeline + DB ingestion)
pytest tests/ -v
```
