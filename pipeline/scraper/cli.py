import logging
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

import dotenv
dotenv.load_dotenv()

from pipeline.config import settings
from pipeline.scraper.models import ScrapeTarget, ScrapeManifest
from pipeline.scraper.scraper import PapaCambridgeScraper

# Initialize Typer App
app = typer.Typer(help="EdGenie Past Paper Scraper CLI")
console = Console()

def setup_logging(level: str):
    """Configures the logging behavior for the scraper pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )

@app.command()
def scrape(
    level: str = typer.Option("o-level", "--level", "-l", help="CAIE Level (o-level or as-and-a-level)"),
    subject: str = typer.Option("accounting", "--subject", "-s", help="Subject name (e.g. accounting, physics)"),
    code: str = typer.Option("7707", "--code", "-c", help="Subject code (e.g. 7707, 9706)"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year to scrape"),
    year_start: Optional[int] = typer.Option(None, "--year-start", help="Starting year for range scrape"),
    year_end: Optional[int] = typer.Option(None, "--year-end", help="Ending year for range scrape"),
    session: List[str] = typer.Option(["jun", "nov"], "--session", "-se", help="Sessions to scrape (jun/nov)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force download even if files already exist"),
    dry_run: bool = typer.Option(False, "--dry-run", "--dry", help="Only show what would be downloaded, do not execute"),
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
):
    """Scrapes past papers and mark schemes for O/A-Levels from PapaCambridge."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Validate sessions
    valid_sessions = []
    for s in session:
        s_clean = s.lower().strip()
        if s_clean in ["jun", "nov", "m", "s", "w", "j", "n", "june", "november", "march", "may-june", "may-jun", "oct-nov", "feb-march"]:
            if s_clean in ["jun", "j", "s", "june", "may-june", "may-jun"]:
                valid_sessions.append("jun")
            elif s_clean in ["nov", "n", "w", "november", "oct-nov"]:
                valid_sessions.append("nov")
            elif s_clean in ["m", "march", "feb-march"]:
                valid_sessions.append("march")
        else:
            console.print(f"[bold red]Error:[/bold red] Invalid session '{s}'. Supported: jun (may-june), nov (oct-nov), march.")
            raise typer.Exit(code=1)

    # De-duplicate sessions
    valid_sessions = list(set(valid_sessions))

    # Calculate years list
    years = []
    if year is not None:
        years.append(year)
    elif year_start is not None and year_end is not None:
        if year_start > year_end:
            console.print("[bold red]Error:[/bold red] --year-start cannot be greater than --year-end.")
            raise typer.Exit(code=1)
        years = list(range(year_start, year_end + 1))
    else:
        # Fallback to current year if nothing else is specified
        current_year = datetime.now().year if 'datetime' in globals() else 2026
        # Let's import datetime here just to be safe
        from datetime import datetime
        years.append(datetime.now().year)

    console.print(f"[bold blue]Starting EdGenie Scraper Pipeline...[/bold blue]")
    console.print(f"Subject: [yellow]{subject} ({code})[/yellow] | Level: [yellow]{level}[/yellow]")
    console.print(f"Years: [yellow]{years}[/yellow] | Sessions: [yellow]{valid_sessions}[/yellow]")
    if force:
        console.print("[bold red]Force mode enabled. Existing files will be overwritten.[/bold red]")
    if dry_run:
        console.print("[bold green]Dry Run mode enabled. No files will be downloaded.[/bold green]")

    total_downloaded = 0
    targets = []

    for y in years:
        for s in valid_sessions:
            target = ScrapeTarget(
                level=level,
                subject_name=subject,
                subject_code=code,
                year=y,
                session=s
            )
            targets.append(target)

    # Process all targets
    for target in targets:
        console.print(f"\n[bold cyan]Processing Target: {target.year} {target.session.upper()}[/bold cyan]")
        scraper = PapaCambridgeScraper(target)
        
        try:
            downloaded = scraper.scrape(force=force, dry_run=dry_run)
            total_downloaded += len(downloaded)
            if not dry_run:
                console.print(f"[green]Successfully processed {len(downloaded)} files.[/green]")
        except Exception as e:
            logger.error(f"Error scraping target {target.year} {target.session}: {e}", exc_info=True)

    console.print(f"\n[bold green]Scrape Job Finished![/bold green] Total files downloaded/identified: [bold]{total_downloaded}[/bold]")


@app.command()
def list_papers(
    code: str = typer.Option(..., "--code", "-c", help="Subject code to inspect"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter by year"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session (jun/nov)"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Logging level")
):
    """Lists downloaded files and metadata stored in scrape manifests for a subject code."""
    setup_logging(log_level)
    
    subject_dir = settings.storage_dir / code
    if not subject_dir.exists() or not subject_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Subject directory {subject_dir} does not exist. No papers downloaded yet.")
        raise typer.Exit(code=1)

    table = Table(title=f"Downloaded Papers for Subject Code: {code}")
    table.add_column("Year", style="cyan")
    table.add_column("Session", style="magenta")
    table.add_column("Filename", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Paper", style="blue")
    table.add_column("Variant", style="blue")
    table.add_column("Size (KB)", justify="right")

    # Find all scrape_manifest.json files under subject_dir
    manifest_paths = list(subject_dir.glob("**/scrape_manifest.json"))
    
    if not manifest_paths:
        console.print(f"[yellow]No scrape manifests found in {subject_dir}. Attempting to list raw files directly...[/yellow]")
        # Raw file listing if no manifests
        pdf_paths = list(subject_dir.glob("**/*.pdf"))
        for path in pdf_paths:
            # path is e.g. storage/7707/2023/jun/filename.pdf
            parts = path.relative_to(subject_dir).parts
            if len(parts) >= 3:
                y, s, filename = parts[0], parts[1], parts[2]
                if year and str(year) != y:
                    continue
                if session and session.lower() != s.lower():
                    continue
                size_kb = round(path.stat().st_size / 1024, 1)
                table.add_row(y, s, filename, "Unknown", "-", "-", str(size_kb))
    else:
        for manifest_path in manifest_paths:
            # Parse manifest
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f) if 'json' in globals() else None
                    if manifest_data is None:
                        import json
                        manifest_data = json.load(f)
                
                manifest = ScrapeManifest(**manifest_data)
                
                # Apply filters
                if year and manifest.target.year != year:
                    continue
                if session and manifest.target.session.lower() != session.lower():
                    continue

                for file_info in manifest.files:
                    size_kb = round((file_info.size_bytes or 0) / 1024, 1)
                    table.add_row(
                        str(manifest.target.year),
                        manifest.target.session.upper(),
                        file_info.filename,
                        file_info.document_type.name,
                        str(file_info.paper_number) if file_info.paper_number else "-",
                        file_info.variant if file_info.variant else "-",
                        str(size_kb)
                    )
            except Exception as e:
                console.print(f"[red]Failed to load manifest {manifest_path}: {e}[/red]")

    console.print(table)


if __name__ == "__main__":
    app()
