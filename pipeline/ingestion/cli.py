"""
Stage 4 — Ingestion & Asset Upload CLI
"""
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

import dotenv
dotenv.load_dotenv()

from pipeline.config import settings
from pipeline.ingestion.ingestor import MongoIngestor
from pipeline.ingestion.firebase_storage import FirebaseStorageManager

app = typer.Typer(help="EdGenie Stage 4 — MongoDB Ingestion & Firebase Storage Asset Sync CLI")
console = Console()


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@app.command()
def ingest_paper(
    json_path: str = typer.Option(..., "--json", "-j", help="Path to Stage 2 paper JSON file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Ingest a single paper JSON and its visuals into MongoDB & Firebase Storage."""
    setup_logging(log_level)
    j_path = Path(json_path)
    if not j_path.exists():
        console.print(f"[bold red]Error:[/bold red] JSON file not found at '{json_path}'")
        raise typer.Exit(code=1)

    ingestor = MongoIngestor()
    result = ingestor.ingest_paper_json(j_path, force=force)

    console.print(f"[bold green]Ingested Paper:[/bold green] {result.get('paper_code')}")
    console.print(f"  Questions Ingested: [yellow]{result.get('questions_ingested')}[/yellow]")
    console.print(f"  Visuals Ingested:   [cyan]{result.get('visuals_ingested')}[/cyan]")
    console.print(f"  PDF CDN URL:        [blue]{result.get('pdf_cdn_url')}[/blue]")


@app.command()
def ingest_dir(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing Stage 2 JSON files"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Ingest all paper JSON files inside a directory into MongoDB & Firebase Storage."""
    setup_logging(log_level)
    d_path = Path(directory_path)
    if not d_path.exists() or not d_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{directory_path}'")
        raise typer.Exit(code=1)

    ingestor = MongoIngestor()
    result = ingestor.ingest_directory(d_path, force=force)

    console.print(f"[bold green]Directory Ingestion Finished![/bold green]")
    console.print(f"  Total Papers: [cyan]{result.get('total_papers')}[/cyan]")
    console.print(f"  Successfully Ingested: [green]{result.get('successfully_ingested')}[/green]")


@app.command()
def sync_storage(
    storage_dir: str = typer.Option("pipeline/storage", "--dir", "-d", help="Storage directory to sync"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Sync all PDFs and visual images from local storage directory to Firebase Storage."""
    setup_logging(log_level)
    s_path = Path(storage_dir)
    if not s_path.exists():
        console.print(f"[bold red]Error:[/bold red] Storage directory not found at '{storage_dir}'")
        raise typer.Exit(code=1)

    storage_mgr = FirebaseStorageManager()
    mapping = storage_mgr.sync_storage_dir(s_path)

    console.print(f"[bold green]Storage Sync Complete![/bold green]")
    console.print(f"Uploaded/mapped [yellow]{len(mapping)}[/yellow] assets with prefix [blue]{storage_mgr.cdn_prefix}[/blue]")

    table = Table(title="Sample Asset CDN URLs")
    table.add_column("Local File", style="cyan")
    table.add_column("Asset CDN URL (assets.edgenie.io)", style="blue")

    for k, v in list(mapping.items())[:15]:
        table.add_row(k, v)

    console.print(table)


if __name__ == "__main__":
    app()
