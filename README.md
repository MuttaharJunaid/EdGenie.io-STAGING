# EdGenie

AI-powered past paper analysis and extraction platform for CAIE O-Level and AS/A-Level students.

---

## Installation & Setup

```bash
# 1. Clone the repository and navigate to project root
cd EdGenie

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package in editable mode
pip install -e ".[dev]"
```

---

## End-to-End Subject Workflow (Start to End)

To run the complete pipeline — **Stage 1 (Scraping) → Stage 2 (Text & Question Extraction) → Stage 3 (Visual Diagram Extraction)** — follow these commands.

### Example: Complete Pipeline for O-Level Physics (5054)

#### Stage 1: Scraping Past Papers & Mark Schemes
```bash
# Scrape O-Level Physics (5054) papers from 2010 to 2020 (June & Nov sessions)
edgenie-scraper scrape \
    --level o-level \
    --subject physics \
    --code 5054 \
    --year-start 2010 \
    --year-end 2020
```

#### Stage 2: Text & Question Extraction
```bash
# Extract questions, marks, and topic classifications for all downloaded PDFs
edgenie-extractor process-dir --dir pipeline/storage/5054
```

#### Stage 3: Visual Diagram & Figure Extraction
```bash
# Extract all labelled figures, graphs, tables, and diagrams, linking them to questions
edgenie-visuals extract-dir --dir pipeline/storage/5054
```

---

## Single Paper Commands

If you want to run the extraction stages for a single PDF paper (e.g. `5054_s10_qp_41.pdf`):

```bash
# Stage 2: Extract questions & text for a single paper
edgenie-extractor process --pdf pipeline/storage/5054/2010/jun/5054_s10_qp_41.pdf

# Stage 3: Extract visual diagrams for a single paper
edgenie-visuals extract --pdf pipeline/storage/5054/2010/jun/5054_s10_qp_41.pdf
```

---

## Command Reference

### 1. Scraper (`edgenie-scraper`)
- **Scrape a single year & session**:
  ```bash
  edgenie-scraper scrape --level o-level --subject accounting --code 7707 --year 2023 --session jun
  ```
- **Scrape A-Level subject range**:
  ```bash
  edgenie-scraper scrape --level as-and-a-level --subject accounting --code 9706 --year-start 2015 --year-end 2023
  ```
- **Dry-run (preview URLs without downloading)**:
  ```bash
  edgenie-scraper scrape --level o-level --subject mathematics-d --code 4024 --year-start 2010 --year-end 2020 --dry-run
  ```

### 2. Text Extractor (`edgenie-extractor`)
- **Process single paper**: `edgenie-extractor process --pdf <path_to_pdf>`
- **Process subject directory**: `edgenie-extractor process-dir --dir <path_to_dir>`

### 3. Visual Extractor (`edgenie-visuals`)
- **Extract single paper**: `edgenie-visuals extract --pdf <path_to_pdf>`
- **Extract subject directory**: `edgenie-visuals extract-dir --dir <path_to_dir>`
