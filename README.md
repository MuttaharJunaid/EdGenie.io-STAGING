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
```bash
# Scrape O-Level Physics (5054) papers from 2010 to 2024
edgenie-scraper scrape \
    --level o-level \
    --subject physics \
    --code 5054 \
    --year-start 2010 \
    --year-end 2024
```

#### Stage 2: Unified 3-in-1 Extraction & Classification

A single call to `edgenie-unified` (or `edgenie-extractor`) submits the PDF to the Unstructured.io Transform API with strict OpenAI-compliant JSON schemas and the official Cambridge syllabus reference:

1. **Text Extraction**: Extracts full question stems, nested sub-parts (`(a)`, `(b)(i)`, etc.), mark allocations, and options (A/B/C/D for MCQs).
2. **Visual Bounding Boxes**: Detects layout bounding boxes for diagrams, graphs, circuits, and tables in the same cloud pass.
3. **Syllabus Classification**: Inlines the subject syllabus CSV directly into the prompt so Unit, Topic, Sub-topic, Difficulty, and Command Words are classified simultaneously.
4. **Local PyMuPDF Cropping**: Filters cover-page logos and header/footer artifacts, crops visual assets at 300 DPI, and patches relative image paths directly into the question JSON tree.

```bash
# Single exam paper (syllabus auto-discovered from syllabi/ directory)
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf

# Or explicitly pass the syllabus CSV
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --syllabus syllabi/PHYSICS_5054.csv

# Force re-extraction (bypassing local cache)
edgenie-unified extract-visuals-text-and-classify \
    --pdf pipeline/storage/5054/2023/jun/5054_s23_qp_22.pdf \
    --force
```

*(Note: `edgenie-extractor extract-visuals-text-and-classify` is an identical alias.)*

#### Stage 3: PostgreSQL Database Ingestion & Cloud Asset Hosting

Stage 3 is decoupled — you can ingest records directly into PostgreSQL without requiring cloud credentials:

```bash
# Ingest single paper JSON and visual manifest into PostgreSQL
edgenie-ingestor db-paper --json pipeline/storage/5054/2023/jun/5054_s23_qp_22.json

# Ingest all extracted papers in a directory tree into PostgreSQL
edgenie-ingestor db-dir --dir pipeline/storage/5054

# View current database record counts across all PostgreSQL tables
edgenie-ingestor db-stats

# (Optional) Upload all PDFs and visual images to Firebase Storage (assets.edgenie.io)
edgenie-ingestor storage-dir --dir pipeline/storage/5054

# (Optional) Full synchronization (both Firebase upload and PostgreSQL ingestion)
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
