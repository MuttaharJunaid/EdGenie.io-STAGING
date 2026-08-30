import logging
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

import dotenv
dotenv.load_dotenv()

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperTypeDetector, PaperCategory
from pipeline.extractor.syllabus_loader import SyllabusLoader
from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
from pipeline.extractor.classifier_agent import ClassifierAgent
from pipeline.scraper.scraper import PapaCambridgeScraper

app = typer.Typer(help="EdGenie Question Extractor & Analyzer CLI")
console = Console()


def setup_logging(level: str):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )


def _get_meta_context(target: Path) -> Dict[str, Any]:
    scraper_helper = PapaCambridgeScraper(target=None)
    pdf_name = target.with_suffix(".pdf").name
    meta = scraper_helper.classify_pdf(pdf_name, "")

    parent_parts = target.parent.parts
    year = meta.year
    session = "jun"
    subject_code = meta.subject_code or "unknown"
    variant = meta.variant or "11"

    if len(parent_parts) >= 3:
        try:
            possible_year = int(parent_parts[-2])
            year = possible_year
            session = parent_parts[-1]
            subject_code = parent_parts[-3]
        except ValueError:
            pass

    return {
        "year": year,
        "session": session,
        "variant": variant,
        "subject_code": subject_code,
    }


def _print_results_table(pdf_name: str, parsed_records: List[Dict[str, Any]], is_qp: bool):
    table = Table(title=f"Extraction Results: {pdf_name}")
    table.add_column("Q Number", style="cyan")
    table.add_column("Marks", style="magenta")
    if is_qp:
        table.add_column("Topic Number", style="green")
        table.add_column("Topic Name", style="yellow")
        table.add_column("Difficulty", style="red")
        table.add_column("Command Words", style="blue")
    else:
        table.add_column("Answer Key / Text Preview", style="green")

    for record in parsed_records[:10]:
        if is_qp:
            table.add_row(
                record["QuestionNumber"].split("/")[-1],
                str(record.get("Marks", "-")),
                record.get("Topic", "-") or "-",
                record.get("Topic_name", "-") or "-",
                f"{record.get('Difficulty_label', '-') or '-'} ({record.get('Difficulty_score', '-')})",
                ", ".join(record.get("Command_words", [])) or "-"
            )
        else:
            preview = record.get("Answer", record.get("Answer_text", ""))
            if len(preview) > 50:
                preview = preview[:50] + "..."
            table.add_row(
                record["AnswerNumber"].split("/")[-1],
                "-",
                preview
            )
    if len(parsed_records) > 10:
        table.add_row("...", "...", "...", "...", "...", "...")

    console.print(table)


# ── Core Step 1: Pure Layout Text Extraction ──────────────────────────────────

def extract_single(
    pdf_path: Path,
    force: bool = False,
) -> Optional[Tuple[Path, List[Dict[str, Any]], PaperCategory, Dict[str, Any]]]:
    """
    Extracts text layout into structured JSON (zero LLM calls).
    Returns (output_json_path, parsed_records, category, meta_context).
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF file does not exist at '{pdf_path}'")
        return None

    console.print(f"\n[bold blue]1. Detecting Paper Type...[/bold blue]")
    category = PaperTypeDetector.detect(pdf)
    console.print(f"Detected Category: [bold yellow]{category.name}[/bold yellow] ([cyan]{category.value}[/cyan])")

    if any(sfx in pdf.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ir.", "_ci_", "_ci."]):
        console.print(f"[yellow]Skipping '{pdf.name}' — Examiner Reports (_er_), Grade Thresholds (_gt_), Confidential Instructions (_ir_/_ci_) are excluded from text extraction.[/yellow]")
        return None

    if category == PaperCategory.OTHER:
        console.print(f"[bold red]Error:[/bold red] Unable to determine CAIE document category of the PDF.")
        return None

    output_json_path = pdf.with_suffix(".json")
    if output_json_path.exists() and not force:
        console.print(f"[yellow]Output file {output_json_path.name} already exists. Use --force to overwrite.[/yellow]")
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            meta_context = _get_meta_context(pdf)
            return output_json_path, records, category, meta_context
        except Exception:
            pass

    meta_context = _get_meta_context(pdf)

    # Layout extraction
    console.print(f"\n[bold green]2. Running Layout Text Extractor (PyMuPDF)...[/bold green]")
    layout_extractor = PyMuPDFExtractor()
    with console.status("[bold green]Extracting text via layout analysis..."):
        parsed_records = layout_extractor.extract(pdf, category, meta_context)
    console.print(f"Layout extractor produced [green]{len(parsed_records)}[/green] records.")

    if not parsed_records:
        console.print(f"[bold red]Error:[/bold red] Layout extractor returned no records.")
        return None

    # Link Stage 3 Visuals Manifest if available
    visuals_manifest_path = pdf.parent / pdf.stem / "manifest.json"
    if visuals_manifest_path.exists():
        try:
            with open(visuals_manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
            q_to_files = {}
            for entry in manifest_data:
                q = entry.get("question")
                if q is not None:
                    q_to_files.setdefault(int(q), []).append(entry["relative_path"])
            patched_count = 0
            for record in parsed_records:
                q_raw = record.get("QuestionNumber") or record.get("AnswerNumber") or ""
                m = re.search(r"(?:Q|q)?(\d+)(?:\([a-z]\))?$", str(q_raw).split("/")[-1].strip())
                if m and int(m.group(1)) in q_to_files:
                    imgs = q_to_files[int(m.group(1))]
                    record["image"] = imgs if len(imgs) > 1 else imgs[0]
                    patched_count += 1
            if patched_count:
                console.print(f"  [green]Linked {patched_count} visual figures from Stage 3 manifest.[/green]")
        except Exception as e:
            logging.getLogger(__name__).debug(f"Could not auto-link visuals manifest: {e}")

    # Save initial extracted JSON skeleton
    console.print(f"\n[bold blue]3. Saving Extracted JSON ({output_json_path.name})...[/bold blue]")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2, ensure_ascii=False)
    console.print(f"[green]✓ Extracted text saved to: {output_json_path.name}[/green]")

    is_qp = category in [PaperCategory.MCQ_QP, PaperCategory.ESSAY_QP]
    _print_results_table(pdf.name, parsed_records, is_qp)
    return output_json_path, parsed_records, category, meta_context


# ── Core Step 2: Gemini / OpenRouter Classification ───────────────────────────

def classify_single(
    target_path: Path,
    syllabus_path: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Takes an extracted JSON file (or PDF path), loads syllabus, runs ClassifierAgent,
    and updates topic/difficulty fields in the JSON file.
    """
    target = Path(target_path)
    if target.suffix.lower() == ".pdf":
        pdf_path = target
        json_path = target.with_suffix(".json")
        if not json_path.exists() or force:
            console.print(f"[dim]JSON file {json_path.name} not found or force specified. Extracting text first...[/dim]")
            res = extract_single(pdf_path, force=force)
            if res is None:
                return False
    elif target.suffix.lower() == ".json":
        json_path = target
        pdf_path = target.with_suffix(".pdf")
    else:
        console.print(f"[bold red]Error:[/bold red] Target must be a .pdf or .json file: '{target_path}'")
        return False

    if not json_path.exists():
        console.print(f"[bold red]Error:[/bold red] Extracted JSON file not found at '{json_path}'. Run extraction first.")
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error reading JSON:[/bold red] {e}")
        return False

    if not records:
        console.print(f"[yellow]JSON file {json_path.name} is empty. Skipping classification.[/yellow]")
        return False

    # Check if first record is question paper
    first_record = records[0]
    if "AnswerNumber" in first_record and "QuestionNumber" not in first_record:
        console.print(f"[cyan]Mark Scheme paper detected ({json_path.name}). Mark schemes do not require syllabus topic classification. Skipping.[/cyan]")
        return True

    category = PaperCategory.MCQ_QP if "Options" in first_record else PaperCategory.ESSAY_QP
    meta_context = _get_meta_context(json_path)

    # Check if already classified
    if not force and any(r.get("Topic") for r in records):
        console.print(f"[yellow]JSON file {json_path.name} already contains topic classifications. Use --force to reclassify.[/yellow]")
        return True

    # Load Syllabus
    syllabus_text = ""
    resolved_syllabus = SyllabusLoader.resolve_syllabus_path(syllabus_path, subject_code=str(meta_context.get("subject_code", "")))
    if resolved_syllabus and resolved_syllabus.exists():
        console.print(f"\n[bold blue]Loading Syllabus ({resolved_syllabus.name})...[/bold blue]")
        syllabus_units = SyllabusLoader.load_syllabus(resolved_syllabus)
        syllabus_text = SyllabusLoader.format_syllabus_for_prompt(syllabus_units)
    else:
        console.print(f"[bold red]Error:[/bold red] No syllabus CSV found for subject {meta_context.get('subject_code')}. Specify --syllabus path.")
        return False

    has_api_keys = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api_keys:
        console.print(
            f"[bold red]Error:[/bold red] No LLM API key found. "
            "Set GOOGLE_API_KEY or OPENROUTER_API_KEY in your environment or .env file."
        )
        return False

    console.print(f"\n[bold green]Running Gemini Classification Pass ({json_path.name} + {resolved_syllabus.name})...[/bold green]")
    classifier = ClassifierAgent()
    with console.status("[bold green]Classifying questions via Gemini..."):
        classifications = classifier.classify(records, syllabus_text, category, meta_context)

    console.print(f"  [green]✓ Classified {len(classifications)} question(s).[/green]")
    if classifications:
        sample_q = next(iter(classifications.values()))
        console.print(
            f"  [dim]Sample classification response ({sample_q.get('question_number')}):\n"
            f"    Unit        : {sample_q.get('Unit')}\n"
            f"    Topic       : {sample_q.get('Topic')} ({sample_q.get('Topic_name')})\n"
            f"    Sub-topics  : {', '.join(sample_q.get('Sub_topic', []))}\n"
            f"    Difficulty  : {sample_q.get('Difficulty_label')} ({sample_q.get('Difficulty_score')}/10)\n"
            f"    Command word: {', '.join(sample_q.get('Command_words', []))}[/dim]"
        )

    updated_records = classifier.merge(records, classifications)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, indent=2, ensure_ascii=False)
    console.print(f"  [green]✓ Updated classification fields in: {json_path.name}[/green]")

    _print_results_table(json_path.name, updated_records, is_qp=True)
    return True


# ── Typer CLI Commands ────────────────────────────────────────────────────────

@app.command("extract")
def extract_cmd(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to raw PDF file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-extraction even if JSON exists"),
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logging level"),
):
    """Extracts text layout and question structure from a single PDF into JSON (0 LLM tokens)."""
    setup_logging(log_level)
    extract_single(Path(pdf_path), force=force)


@app.command("extract-dir")
def extract_dir_cmd(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing PDF files"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-extraction of existing files"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Extracts text layout and question structure for all PDFs in a directory (0 LLM tokens)."""
    setup_logging(log_level)
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory does not exist at '{directory_path}'")
        raise typer.Exit(code=1)

    all_pdfs = list(dir_path.glob("**/*.pdf"))
    pdf_files = [
        p for p in all_pdfs
        if not any(sfx in p.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ir.", "_ci_", "_ci."])
    ]
    if not pdf_files:
        console.print(f"[yellow]No valid Question Paper / Mark Scheme PDFs found in '{directory_path}' (excluded _er_, _gt_, _ir_, _ci_ files).[/yellow]")
        raise typer.Exit(code=0)

    pdf_files.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
    console.print(f"Found [bold cyan]{len(pdf_files)}[/bold cyan] PDF papers to extract.")

    count = 0
    for pdf in pdf_files:
        console.print(f"\n[bold magenta]Extracting File: {pdf.name}[/bold magenta]")
        res = extract_single(pdf, force=force)
        if res is not None:
            count += 1

    console.print(f"\n[bold green]Text Extraction Finished![/bold green] Successfully extracted {count} files.")


@app.command("classify")
def classify_cmd(
    pdf_path: Optional[str] = typer.Option(None, "--pdf", "-p", help="Path to raw PDF file"),
    json_path: Optional[str] = typer.Option(None, "--json", "-j", help="Path to extracted JSON file"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-classification"),
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logging level"),
):
    """Classifies pre-extracted question JSON against the syllabus using Gemini."""
    setup_logging(log_level)
    target = Path(json_path) if json_path else (Path(pdf_path) if pdf_path else None)
    if not target:
        console.print("[bold red]Error:[/bold red] You must provide either --pdf or --json path.")
        raise typer.Exit(code=1)

    success = classify_single(target, syllabus_path=syllabus_path, force=force)
    if not success:
        raise typer.Exit(code=1)


@app.command("classify-dir")
def classify_dir_cmd(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing PDF or JSON files"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-classification"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Classifies all pre-extracted question JSONs in a directory against the syllabus."""
    setup_logging(log_level)
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory does not exist at '{directory_path}'")
        raise typer.Exit(code=1)

    # Prefer JSON files; if none, check PDFs
    json_files = list(dir_path.glob("**/*_qp_*.json")) + list(dir_path.glob("**/*_qp.json"))
    if not json_files:
        # Check for QP PDFs
        all_pdfs = list(dir_path.glob("**/*_qp_*.pdf")) + list(dir_path.glob("**/*_qp.pdf"))
        json_files = [p for p in all_pdfs if not any(sfx in p.name.lower() for sfx in ["_er_", "_gt_", "_ir_", "_ci_"])]

    if not json_files:
        console.print(f"[yellow]No Question Paper files found in '{directory_path}' for classification.[/yellow]")
        raise typer.Exit(code=0)

    json_files.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
    console.print(f"Found [bold cyan]{len(json_files)}[/bold cyan] Question Paper files to classify.")

    count = 0
    for target in json_files:
        console.print(f"\n[bold magenta]Classifying File: {target.name}[/bold magenta]")
        if classify_single(target, syllabus_path=syllabus_path, force=force):
            count += 1

    console.print(f"\n[bold green]Classification Finished![/bold green] Successfully classified {count} files.")


@app.command("process")
def process_cmd(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to raw PDF file"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing even if output exists"),
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logging level"),
):
    """Combined pipeline: extracts layout text and classifies with Gemini in one step."""
    setup_logging(log_level)
    pdf = Path(pdf_path)
    res = extract_single(pdf, force=force)
    if res is None:
        raise typer.Exit(code=1)

    output_json, records, category, meta = res
    if category in [PaperCategory.MCQ_QP, PaperCategory.ESSAY_QP]:
        classify_single(output_json, syllabus_path=syllabus_path, force=force)


@app.command("process-dir")
def process_dir_cmd(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing PDF files"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing of existing files"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Combined pipeline: extracts layout text and classifies all PDFs in a directory."""
    setup_logging(log_level)
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory does not exist at '{directory_path}'")
        raise typer.Exit(code=1)

    all_pdfs = list(dir_path.glob("**/*.pdf"))
    pdf_files = [
        p for p in all_pdfs
        if not any(sfx in p.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ir.", "_ci_", "_ci."])
    ]
    if not pdf_files:
        console.print(f"[yellow]No valid Question Paper / Mark Scheme PDFs found in '{directory_path}'.[/yellow]")
        raise typer.Exit(code=0)

    pdf_files.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
    console.print(f"Found [bold cyan]{len(pdf_files)}[/bold cyan] PDF papers to process.")

    count = 0
    for pdf in pdf_files:
        console.print(f"\n[bold magenta]Processing File: {pdf.name}[/bold magenta]")
        res = extract_single(pdf, force=force)
        if res is not None:
            output_json, records, category, meta = res
            if category in [PaperCategory.MCQ_QP, PaperCategory.ESSAY_QP]:
                classify_single(output_json, syllabus_path=syllabus_path, force=force)
            count += 1

    console.print(f"\n[bold green]Directory Processing Finished![/bold green] Successfully processed {count} files.")


if __name__ == "__main__":
    app()
