"""
pipeline/syllabus/cli.py

CLI command: edgenie-syllabus

Usage
-----
    edgenie-syllabus build \\
        --subject "Physics" \\
        --code 5054 \\
        --level o-level

Flow
----
1. Constructs the official Cambridge subject page URL and opens it in the
   default system browser so the user can find and download the syllabus PDF.
2. Prompts the user to enter the path to the downloaded PDF.
3. Sends the PDF to Gemini for structured content extraction.
4. Saves the result as  syllabi/{SUBJECT}_{CODE}.csv  in the EdGenie format.
"""

import logging
import os
import webbrowser
from pathlib import Path
from typing import Optional

import dotenv
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

dotenv.load_dotenv()

from pipeline.syllabus.builder import SyllabusBuilder

app = typer.Typer(help="EdGenie Syllabus Builder — extract Cambridge syllabuses into structured CSV")
console = Console()

LEVEL_CHOICES = ["o-level", "a-level", "igcse"]


def setup_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@app.command()
def build(
    subject: str = typer.Option(..., "--subject", "-s", help='Subject name, e.g. "Physics"'),
    code: str = typer.Option(..., "--code", "-c", help="Cambridge subject code, e.g. 5054"),
    level: str = typer.Option(
        ...,
        "--level",
        "-l",
        help="Qualification level: o-level | a-level | igcse",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help='Directory to save the CSV (default: syllabi/ in project root)',
    ),
    pdf_path: Optional[str] = typer.Option(
        None,
        "--pdf",
        "-p",
        help="Path to an already-downloaded Cambridge syllabus PDF (skips browser step)",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Build a structured CSV syllabus from an official Cambridge syllabus PDF.

    Opens the Cambridge subject page in your browser so you can download
    the PDF, then extracts its content using Gemini and saves a CSV that
    is ready to pass to  edgenie-extractor --syllabus.
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Validate level
    level_normalised = level.lower().strip()
    if level_normalised not in LEVEL_CHOICES:
        console.print(
            f"[bold red]Error:[/bold red] --level must be one of: {', '.join(LEVEL_CHOICES)}. "
            f"Got: '{level}'"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]EdGenie Syllabus Builder[/bold cyan]\n"
            f"Subject : [yellow]{subject}[/yellow]\n"
            f"Code    : [yellow]{code}[/yellow]\n"
            f"Level   : [yellow]{level_normalised}[/yellow]",
            title="Stage 0 — Syllabus Extraction",
        )
    )

    # ── Check API key ──────────────────────────────────────────────────────────
    if not os.getenv("GOOGLE_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GOOGLE_API_KEY is not set. "
            "Add it to your .env file and try again."
        )
        raise typer.Exit(code=1)

    builder = SyllabusBuilder()

    # ── Step 1: resolve PDF path ───────────────────────────────────────────────
    if pdf_path:
        resolved_pdf = Path(pdf_path)
        if not resolved_pdf.exists():
            console.print(
                f"[bold red]Error:[/bold red] PDF file not found at: {pdf_path}"
            )
            raise typer.Exit(code=1)
        console.print(f"\n[bold blue]Using provided PDF:[/bold blue] {resolved_pdf}")
    else:
        # Open Cambridge subject page so user can download the PDF
        page_url = builder.cambridge_page_url(subject, code, level_normalised)

        console.print(f"\n[bold blue]Step 1 — Download the syllabus PDF[/bold blue]")
        console.print(
            f"Opening the Cambridge subject page in your browser:\n"
            f"[link={page_url}]{page_url}[/link]\n"
        )
        console.print(
            "[dim]On that page, go to the [bold]Syllabus[/bold] tab and download the "
            "latest syllabus PDF (e.g. '2026–2028 Syllabus').[/dim]\n"
        )
        webbrowser.open(page_url)

        console.print(
            "[bold yellow]Once downloaded, enter the full path to the PDF below.[/bold yellow]"
        )
        pdf_input = typer.prompt("Path to downloaded syllabus PDF").strip().strip("'\"")
        resolved_pdf = Path(pdf_input).expanduser().resolve()

        if not resolved_pdf.exists():
            console.print(
                f"[bold red]Error:[/bold red] File not found at: {resolved_pdf}\n"
                "Please check the path and run the command again."
            )
            raise typer.Exit(code=1)

    # ── Step 2: Extract with Gemini ────────────────────────────────────────────
    console.print(f"\n[bold blue]Step 2 — Extracting syllabus content with Gemini...[/bold blue]")
    with console.status("[bold green]Sending PDF to Gemini for extraction…"):
        try:
            rows = builder.extract_from_pdf(resolved_pdf, subject, code, level_normalised)
        except Exception as e:
            console.print(f"[bold red]Gemini extraction failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    if not rows:
        console.print("[bold red]Error:[/bold red] Gemini returned no rows. The PDF may not be a valid syllabus.")
        raise typer.Exit(code=1)

    console.print(f"Extracted [bold green]{len(rows)}[/bold green] learning objectives.")

    # ── Step 3: Save CSV ───────────────────────────────────────────────────────
    console.print(f"\n[bold blue]Step 3 — Saving CSV...[/bold blue]")

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Default: syllabi/ in the project root (same as existing CSVs)
        out_dir = Path(__file__).resolve().parent.parent.parent / "syllabi"

    try:
        csv_path = builder.save_csv(rows, subject, code, out_dir)
    except Exception as e:
        console.print(f"[bold red]Failed to save CSV:[/bold red] {e}")
        raise typer.Exit(code=1)

    # ── Preview table ──────────────────────────────────────────────────────────
    table = Table(title=f"Syllabus Preview — first 15 rows of {csv_path.name}")
    table.add_column("Unit", style="cyan", no_wrap=False)
    table.add_column("Topic", style="yellow", no_wrap=False)
    table.add_column("Subtopic", style="green", no_wrap=False)

    for row in rows[:15]:
        table.add_row(row.unit, row.topic, row.subtopic)
    if len(rows) > 15:
        table.add_row("…", "…", f"… ({len(rows) - 15} more rows)")

    console.print(table)
    console.print(
        f"\n[bold green]✓ Syllabus CSV saved:[/bold green] [cyan]{csv_path}[/cyan]"
    )
    console.print(
        f"\n[dim]Use it with:[/dim]\n"
        f"  edgenie-extractor process --pdf <paper.pdf> --syllabus {csv_path}"
    )


if __name__ == "__main__":
    app()
