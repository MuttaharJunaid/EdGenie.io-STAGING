import logging
import os
import re
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

import dotenv
dotenv.load_dotenv()

from pipeline.config import settings
from pipeline.extractor.paper_type_detector import PaperTypeDetector, PaperCategory
from pipeline.extractor.syllabus_loader import SyllabusLoader
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

@app.command()
def process(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to raw PDF file"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing even if output exists"),
    duplicate: bool = typer.Option(
        False, "--duplicate",
        help="Use PyMuPDF layout extractor (no LLM for text extraction). "
             "Classification/difficulty still uses Gemini. Output saved as *_dup.json."
    ),
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logging level"),
):
    """Processes a single past paper or mark scheme PDF into structured and analyzed JSON."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    pdf = Path(pdf_path)
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF file does not exist at '{pdf_path}'")
        raise typer.Exit(code=1)

    # 1. Detect Paper Type
    console.print(f"\n[bold blue]1. Detecting Paper Type...[/bold blue]")
    category = PaperTypeDetector.detect(pdf)
    console.print(f"Detected Category: [bold yellow]{category.name}[/bold yellow] ([cyan]{category.value}[/cyan])")

    if any(sfx in pdf.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ir.", "_ci_", "_ci."]):
        console.print(f"[yellow]Skipping '{pdf.name}' — Examiner Reports (_er_), Grade Thresholds (_gt_), Confidential Instructions (_ir_/_ci_) are excluded from text extraction.[/yellow]")
        return

    if category == PaperCategory.OTHER:
        console.print(f"[bold red]Error:[/bold red] Unable to determine CAIE document category of the PDF.")
        raise typer.Exit(code=1)

    # Output path: *_dup.json for duplicate strategy, *.json for standard
    output_json_path = pdf.with_name(pdf.stem + "_dup.json") if duplicate else pdf.with_suffix(".json")
    if output_json_path.exists() and not force:
        console.print(f"[yellow]Output file {output_json_path.name} already exists. Use --force to overwrite. Skipping.[/yellow]")
        return

    # 2. Parse Metadata from Filename
    # E.g. 7707_s22_qp_12.pdf -> subject_code=7707, session=summer, year=2022, variant=12
    scraper_helper = PapaCambridgeScraper(target=None) # Just call class helper
    meta = scraper_helper.classify_pdf(pdf.name, "")

    # Extract details from path structures if filename lacks them
    # E.g. storage/7707/2023/jun/filename.pdf
    parent_parts = pdf.parent.parts
    year = meta.year
    session = "jun"
    subject_code = meta.subject_code or "unknown"
    variant = meta.variant or "11"

    if len(parent_parts) >= 3:
        # Check standard storage folders: /storage/{subject_code}/{year}/{session}
        try:
            possible_year = int(parent_parts[-2])
            year = possible_year
            session = parent_parts[-1]
            subject_code = parent_parts[-3]
        except ValueError:
            pass

    meta_context = {
        "year": year,
        "session": session,
        "variant": variant,
        "subject_code": subject_code,
    }

    # 3. Load Syllabus
    syllabus_text = ""
    resolved_syllabus = SyllabusLoader.resolve_syllabus_path(syllabus_path, subject_code=str(meta_context.get("subject_code", "")))
    if resolved_syllabus and resolved_syllabus.exists():
        console.print(f"\n[bold blue]2. Loading Syllabus ({resolved_syllabus.name})...[/bold blue]")
        syllabus_units = SyllabusLoader.load_syllabus(resolved_syllabus)
        syllabus_text = SyllabusLoader.format_syllabus_for_prompt(syllabus_units)
    elif syllabus_path:
        console.print(f"[bold yellow]Warning:[/bold yellow] Syllabus CSV file not found at '{syllabus_path}' or in 'syllabi/' directory.")

    # 4. Check API keys — LLM is always needed (for classification at minimum)
    has_api_keys = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api_keys:
        console.print(
            f"[bold red]Error:[/bold red] No LLM API key found. "
            "Set GOOGLE_API_KEY or OPENAI_API_KEY in your environment or .env file."
        )
        raise typer.Exit(code=1)

    is_qp = category in [PaperCategory.MCQ_QP, PaperCategory.ESSAY_QP]

    if duplicate:
        # ── Duplicate strategy: PyMuPDF layout extraction + Gemini classification only ──
        console.print(f"\n[bold green]3. Running PyMuPDF Layout Extractor (--duplicate strategy)...[/bold green]")
        from pipeline.extractor.pymupdf_extractor import PyMuPDFExtractor
        from pipeline.extractor.classifier_agent import ClassifierAgent

        layout_extractor = PyMuPDFExtractor()
        with console.status("[bold green]Extracting text via layout analysis..."):
            parsed_records = layout_extractor.extract(pdf, category, meta_context)
        console.print(f"Layout extractor produced [green]{len(parsed_records)}[/green] skeleton records.")

        if not parsed_records:
            console.print(f"[bold red]Error:[/bold red] Layout extractor returned no records.")
            raise typer.Exit(code=1)

        # 4. Check for Stage 3 Visuals Manifest & Link Images
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
                logger.debug(f"Could not auto-link visuals manifest: {e}")

        # 5. Save extracted text JSON immediately before calling LLM
        console.print(f"\n[bold blue]4. Saving Extracted JSON ({output_json_path.name})...[/bold blue]")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_records, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Extracted text saved to: {output_json_path.name}[/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to save initial JSON output:[/bold red] {e}")
            raise typer.Exit(code=1)

        # 6. Run Gemini classification pass and update only classification fields
        if is_qp:
            if syllabus_text:
                console.print(f"\n[bold green]5. Running Gemini Classification Pass (text-only, using {output_json_path.name} + syllabus)...[/bold green]")
                classifier = ClassifierAgent()
                with console.status("[bold green]Classifying questions via Gemini..."):
                    classifications = classifier.classify(parsed_records, syllabus_text, category, meta_context)
                
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

                parsed_records = classifier.merge(parsed_records, classifications)
                # Overwrite output JSON with updated classification fields
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_records, f, indent=2, ensure_ascii=False)
                console.print(f"  [green]✓ Updated classification fields in: {output_json_path.name}[/green]")
            else:
                console.print(f"[yellow]No syllabus provided — skipping Gemini classification pass. Topic/Difficulty fields remain empty.[/yellow]")

    else:
        # ── Standard strategy: full multimodal LLM extraction + classification ──
        console.print(f"\n[bold green]3. Running AGENTIC Multimodal PDF Parser...[/bold green]")
        from pipeline.extractor.agentic_parser import AgenticPDFParser
        agentic_parser = AgenticPDFParser()
        with console.status("[bold green]Sending raw PDF to Multimodal LLM..."):
            parsed_records = agentic_parser.parse_pdf(pdf, category, syllabus_text, meta_context)
        console.print(f"Agentic parser successfully extracted [green]{len(parsed_records)}[/green] records.")

        if not parsed_records:
            console.print(f"[bold red]Error:[/bold red] Agentic parser returned no records.")
            raise typer.Exit(code=1)

        # Check for Stage 3 Visuals Manifest & Link Images
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
                logger.debug(f"Could not auto-link visuals manifest: {e}")

        # Save standard output JSON
        console.print(f"\n[bold blue]4. Saving JSON Output...[/bold blue]")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_records, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved JSON file successfully to: {output_json_path.name}[/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to save JSON output:[/bold red] {e}")
            raise typer.Exit(code=1)

    # 7. Print Results summary table
    table = Table(title=f"Extraction Results: {pdf.name}")
    table.add_column("Q Number", style="cyan")
    table.add_column("Marks", style="magenta")
    if is_qp:
        table.add_column("Topic Number", style="green")
        table.add_column("Topic Name", style="yellow")
        table.add_column("Difficulty", style="red")
        table.add_column("Command Words", style="blue")
    else:
        table.add_column("Answer Key / Text Preview", style="green")

    for record in parsed_records[:10]:  # Show first 10 items
        if is_qp:
            table.add_row(
                record["QuestionNumber"].split("/")[-1],
                str(record.get("Marks", "-")),
                record.get("Topic", "-"),
                record.get("Topic_name", "-"),
                f"{record.get('Difficulty_label', '-')} ({record.get('Difficulty_score', '-')})",
                ", ".join(record.get("Command_words", []))
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


@app.command()
def process_dir(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing PDF files"),
    syllabus_path: Optional[str] = typer.Option(None, "--syllabus", "-s", help="Path to syllabus CSV"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing of existing files"),
    duplicate: bool = typer.Option(
        False, "--duplicate",
        help="Use PyMuPDF layout extractor strategy for all papers (same as --duplicate in process)."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Processes all PDF past papers inside a directory."""
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

    # Sort sequentially starting from the first stored PDF
    pdf_files.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))

    strategy = "[bold cyan]duplicate (PyMuPDF)[/bold cyan]" if duplicate else "[bold cyan]standard (agentic LLM)[/bold cyan]"
    console.print(f"Found [bold cyan]{len(pdf_files)}[/bold cyan] PDF papers to process (excluding Examiner Reports and Grade Thresholds).")
    console.print(f"Strategy: {strategy}")

    processed_count = 0
    for pdf in pdf_files:
        console.print(f"\n[bold magenta]Processing File: {pdf.name}[/bold magenta]")
        try:
            process(
                pdf_path=str(pdf),
                syllabus_path=syllabus_path,
                force=force,
                duplicate=duplicate,
                log_level=log_level
            )
            processed_count += 1
        except Exception as e:
            console.print(f"[red]Failed processing file {pdf.name}: {e}[/red]")

    console.print(f"\n[bold green]Directory Processing Finished![/bold green] Successfully processed {processed_count} files.")

if __name__ == "__main__":
    app()
