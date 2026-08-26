import logging
import os
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

    if category == PaperCategory.OTHER:
        console.print(f"[bold red]Error:[/bold red] Unable to determine CAIE document category of the PDF.")
        raise typer.Exit(code=1)

    output_json_path = pdf.with_suffix(".json")
    if output_json_path.exists() and not force:
        console.print(f"[yellow]Output file {output_json_path.name} already exists. Use --force to overwrite. Skipping.[/yellow]")
        raise typer.Exit(code=0)

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
        # Check standard storage folders
        # /storage/{subject_code}/{year}/{session}
        try:
            possible_year = int(parent_parts[-2])
            year = possible_year
            session = parent_parts[-1]
            subject_code = parent_parts[-3]
        except ValueError:
            pass

    # 3. Load Syllabus
    syllabus_text = ""
    if syllabus_path:
        csv_path = Path(syllabus_path)
        if csv_path.exists():
            console.print(f"\n[bold blue]2. Loading Syllabus...[/bold blue]")
            syllabus_units = SyllabusLoader.load_syllabus(csv_path)
            syllabus_text = SyllabusLoader.format_syllabus_for_prompt(syllabus_units)
        else:
            console.print(f"[bold yellow]Warning:[/bold yellow] Syllabus CSV file not found at '{syllabus_path}'.")

    # 4. Check API keys — agentic parsing is the ONLY path
    has_api_keys = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not has_api_keys:
        console.print(
            f"[bold red]Error:[/bold red] No LLM API key found. "
            "Set GOOGLE_API_KEY or OPENAI_API_KEY in your environment or .env file. "
            "Agentic parsing requires an API key."
        )
        raise typer.Exit(code=1)

    # 5. Run Agentic Multimodal PDF Parser
    is_qp = category in [PaperCategory.MCQ_QP, PaperCategory.ESSAY_QP]

    console.print(f"\n[bold green]3. Running AGENTIC Multimodal PDF Parser...[/bold green]")
    meta_context = {
        "year": year,
        "session": session,
        "variant": variant,
        "subject_code": subject_code
    }

    from pipeline.extractor.agentic_parser import AgenticPDFParser
    agentic_parser = AgenticPDFParser()
    with console.status("[bold green]Sending raw PDF to Multimodal LLM..."):
        parsed_records = agentic_parser.parse_pdf(pdf, category, syllabus_text, meta_context)
    console.print(f"Agentic parser successfully extracted [green]{len(parsed_records)}[/green] records.")

    if not parsed_records:
        console.print(f"[bold red]Error:[/bold red] Agentic parser returned no records.")
        raise typer.Exit(code=1)

    # 6. Check for Stage 3 Visuals Manifest & Link Images
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

    # 7. Write to JSON
    console.print(f"\n[bold blue]4. Saving JSON Output...[/bold blue]")
    try:
        with open(output_json_path, "w") as f:
            json.dump(parsed_records, f, indent=2)
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
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Processes all PDF past papers inside a directory."""
    setup_logging(log_level)
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory does not exist at '{directory_path}'")
        raise typer.Exit(code=1)

    pdf_files = list(dir_path.glob("**/*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in '{directory_path}'[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"Found [bold cyan]{len(pdf_files)}[/bold cyan] PDF papers to process.")

    processed_count = 0
    for pdf in pdf_files:
        console.print(f"\n[bold magenta]Processing File: {pdf.name}[/bold magenta]")
        try:
            process(
                pdf_path=str(pdf),
                syllabus_path=syllabus_path,
                force=force,
                log_level=log_level
            )
            processed_count += 1
        except Exception as e:
            console.print(f"[red]Failed processing file {pdf.name}: {e}[/red]")

    console.print(f"\n[bold green]Directory Processing Finished![/bold green] Successfully processed {processed_count} files.")

if __name__ == "__main__":
    app()
