"""
pipeline/syllabus/cli.py

CLI command: edgenie-syllabus

Usage
-----
    edgenie-syllabus build \\
        --subject "Physics" \\
        --code 5054 \\
        --level o-level

    edgenie-syllabus list

Automated Production Flow
-------------------------
1. Navigates directly to the official Cambridge International qualifications directory
   (e.g., https://www.cambridgeinternational.org/programmes-and-qualifications/cambridge-upper-secondary/cambridge-o-level/subjects/).
2. Locates the subject of interest (e.g. Physics 5054) and navigates to its page.
3. Searches for the "Syllabuses" section, checks candidate PDFs in reverse order (bottom-up),
   and inspects page counts:
   - Skips PDFs with <= 3 pages (e.g., errata / update flyers).
   - Once a PDF with > 3 pages is found (the complete syllabus document), downloads it.
4. Sends the downloaded PDF to Gemini to extract all learning objectives into:
   Unit, Topic, Subtopic
5. Saves the final CSV into the syllabi/ directory (e.g. syllabi/PHYSICS_5054.csv).
"""

import logging
import os
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

app = typer.Typer(
    help="EdGenie Syllabus Builder — extract official Cambridge syllabuses into structured CSV",
    no_args_is_help=True,
)
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
        help="Directory to save the CSV (default: syllabi/ in project root)",
    ),
    pdf_path: Optional[str] = typer.Option(
        None,
        "--pdf",
        "-p",
        help="Optional: Path to an existing syllabus PDF (skips automated web discovery)",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Automated syllabus extraction: crawls official Cambridge website,
    downloads the latest complete syllabus (>3 pages), and uses Gemini
    to extract a structured CSV into the syllabi/ directory.
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
            f"[bold cyan]EdGenie Automated Syllabus Extractor[/bold cyan]\n"
            f"Subject : [yellow]{subject}[/yellow]\n"
            f"Code    : [yellow]{code}[/yellow]\n"
            f"Level   : [yellow]{level_normalised}[/yellow]",
            title="Stage 0 — Cambridge Syllabus Discovery & Extraction",
        )
    )

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GOOGLE_API_KEY is not set. "
            "Add it to your .env file and try again."
        )
        raise typer.Exit(code=1)

    builder = SyllabusBuilder()

    # Step 1: Discover & Download or Use Provided PDF
    if pdf_path:
        resolved_pdf = Path(pdf_path)
        if not resolved_pdf.exists():
            console.print(f"[bold red]Error:[/bold red] PDF file not found at: {pdf_path}")
            raise typer.Exit(code=1)
        console.print(f"\n[bold blue]Using provided local PDF:[/bold blue] {resolved_pdf}")
    else:
        console.print(f"\n[bold blue]1. Crawling Cambridge website for official syllabus PDF...[/bold blue]")
        with console.status("[bold green]Navigating Cambridge portal & inspecting candidate PDFs (>3 pages)..."):
            try:
                resolved_pdf, page_count = builder.crawler.fetch_syllabus_pdf(subject, code, level_normalised)
            except Exception as e:
                console.print(f"[bold red]Failed to discover syllabus PDF from Cambridge website:[/bold red] {e}")
                raise typer.Exit(code=1)

        console.print(
            f"[green]✓ Downloaded official syllabus PDF:[/green] [cyan]{resolved_pdf.name}[/cyan] "
            f"([bold]{page_count}[/bold] pages)"
        )

    # Step 2: Extract structured content via Gemini
    console.print(f"\n[bold blue]2. Extracting syllabus content with Gemini...[/bold blue]")
    with console.status("[bold green]Sending PDF to Gemini for structured extraction…"):
        try:
            rows = builder.extract_from_pdf(resolved_pdf, subject, code, level_normalised)
        except Exception as e:
            console.print(f"[bold red]Gemini extraction failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    if not rows:
        console.print("[bold red]Error:[/bold red] Gemini returned no rows. The PDF may not be a valid syllabus.")
        raise typer.Exit(code=1)

    console.print(f"[green]✓ Successfully extracted [bold]{len(rows)}[/bold] learning objectives.[/green]")

    # Step 3: Save CSV
    console.print(f"\n[bold blue]3. Saving CSV to syllabi directory...[/bold blue]")
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent.parent / "syllabi"

    try:
        csv_path = builder.save_csv(rows, subject, code, out_dir)
    except Exception as e:
        console.print(f"[bold red]Failed to save CSV:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Preview table
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
        f"\n[bold green]✓ Complete! Syllabus CSV saved to:[/bold green] [cyan]{csv_path}[/cyan]"
    )
    console.print(
        f"\n[dim]Ready for text-extraction and topic classification:[/dim]\n"
        f"  edgenie-extractor process --pdf <paper.pdf> --syllabus {csv_path}"
    )


@app.command(name="list")
def list_syllabi(
    directory: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Syllabi directory (default: syllabi/ in project root)"
    )
):
    """List all available structured syllabus CSV files."""
    if directory:
        s_dir = Path(directory)
    else:
        s_dir = Path(__file__).resolve().parent.parent.parent / "syllabi"

    if not s_dir.exists():
        console.print(f"[yellow]Syllabi directory does not exist at: {s_dir}[/yellow]")
        return

    csvs = sorted(s_dir.glob("*.csv"))
    if not csvs:
        console.print(f"[yellow]No syllabus CSV files found in: {s_dir}[/yellow]")
        return

    table = Table(title=f"Available Syllabuses ({len(csvs)} files in {s_dir.name}/)")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Full Path", style="green")

    for f in csvs:
        table.add_row(f.name, f"{f.stat().st_size:,} bytes", str(f))

    console.print(table)


if __name__ == "__main__":
    app()
