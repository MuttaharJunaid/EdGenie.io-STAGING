"""
Stage 4 — PostgreSQL Ingestion (Relational + JSONB) & Firebase Asset Storage CLI
"""
import logging
from pathlib import Path

import dotenv
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

dotenv.load_dotenv()

from pipeline.ingestion.firebase_storage import FirebaseStorageManager
from pipeline.ingestion.postgres_client import PostgresClient
from pipeline.ingestion.postgres_ingestor import PostgresIngestor

app = typer.Typer(help="EdGenie Stage 4 — PostgreSQL (Relational + JSONB) Ingestion & Firebase Storage CLI")
console = Console()


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


# ── Database Ingestion Commands (PostgreSQL Relational + JSONB) ────────────────

@app.command(name="db-init")
def db_init(
    uri: str | None = typer.Option(None, "--uri", "-u", help="PostgreSQL connection URI (defaults to POSTGRES_URI or DATABASE_URL)"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Initialize PostgreSQL database tables, foreign keys, and B-Tree/GIN indexes."""
    setup_logging(log_level)
    client = PostgresClient(uri=uri) if uri else PostgresClient()
    try:
        client.init_schema()
        console.print("[bold green]Success:[/bold green] PostgreSQL schema initialized (6 tables, foreign keys & GIN indexes ready).")
        stats = client.get_stats()
        _print_stats_table(stats)
    except Exception as e:
        console.print(f"[bold red]Error initializing PostgreSQL schema:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="db-stats")
def db_stats(
    uri: str | None = typer.Option(None, "--uri", "-u", help="PostgreSQL connection URI"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Display current record counts across all PostgreSQL tables."""
    setup_logging(log_level)
    client = PostgresClient(uri=uri) if uri else PostgresClient()
    try:
        stats = client.get_stats()
        _print_stats_table(stats)
    except Exception as e:
        console.print(f"[bold red]Error querying PostgreSQL stats:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="db-paper")
def db_paper(
    json_path: str = typer.Option(..., "--json", "-j", help="Path to Stage 2 paper JSON file"),
    uri: str | None = typer.Option(None, "--uri", "-u", help="PostgreSQL connection URI (defaults to POSTGRES_URI or DATABASE_URL)"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Ingest a single paper JSON and its visual manifest into PostgreSQL."""
    setup_logging(log_level)
    j_path = Path(json_path)
    if not j_path.exists():
        console.print(f"[bold red]Error:[/bold red] JSON file not found at '{json_path}'")
        raise typer.Exit(code=1)

    db_client = PostgresClient(uri=uri) if uri else PostgresClient()
    ingestor = PostgresIngestor(db_client=db_client)

    try:
        result = ingestor.ingest_paper_json(j_path)
        console.print(f"[bold green]Ingested Paper into PostgreSQL:[/bold green] {result.get('paper_code')}")
        console.print(f"  Category:           [cyan]{result.get('category')}[/cyan]")
        console.print(f"  Questions Ingested: [yellow]{result.get('questions_ingested')}[/yellow]")
        console.print(f"  Answers Ingested:   [magenta]{result.get('answers_ingested')}[/magenta]")
        console.print(f"  Visuals Ingested:   [blue]{result.get('visuals_ingested')}[/blue]")
        console.print(f"  PDF CDN URL:        [dim]{result.get('pdf_cdn_url')}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error ingesting paper:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="db-dir")
def db_dir(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory containing Stage 2 JSON files"),
    uri: str | None = typer.Option(None, "--uri", "-u", help="PostgreSQL connection URI"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Ingest all paper JSON files inside a directory tree into PostgreSQL."""
    setup_logging(log_level)
    d_path = Path(directory_path)
    if not d_path.exists() or not d_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{directory_path}'")
        raise typer.Exit(code=1)

    db_client = PostgresClient(uri=uri) if uri else PostgresClient()
    ingestor = PostgresIngestor(db_client=db_client)

    try:
        result = ingestor.ingest_directory(d_path)
        console.print("\n[bold green]Directory Ingestion Finished![/bold green]")
        console.print(f"  Total JSON Papers Found: [cyan]{result.get('total_files')}[/cyan]")
        console.print(f"  Successfully Ingested:   [green]{result.get('successful_papers')}[/green]")
        console.print(f"  Failed Papers:           [red]{result.get('failed_papers')}[/red]")
        console.print(f"  Total Questions Ingested: [yellow]{result.get('total_questions')}[/yellow]")
        console.print(f"  Total Answers Ingested:   [magenta]{result.get('total_answers')}[/magenta]")
        console.print(f"  Total Visuals Ingested:   [blue]{result.get('total_visuals')}[/blue]")

        stats = db_client.get_stats()
        _print_stats_table(stats)
    except Exception as e:
        console.print(f"[bold red]Error ingesting directory:[/bold red] {e}")
        raise typer.Exit(code=1)


# ── Firebase Storage Ingestion Commands ────────────────────────────────────────

@app.command(name="storage-paper")
def storage_paper(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to PDF paper file"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Upload a single paper's PDF and visual figures to Firebase Storage."""
    setup_logging(log_level)
    p_path = Path(pdf_path)
    if not p_path.exists():
        console.print(f"[bold red]Error:[/bold red] PDF file not found at '{pdf_path}'")
        raise typer.Exit(code=1)

    storage_mgr = FirebaseStorageManager()
    pdf_url = storage_mgr.upload_pdf(p_path)
    console.print(f"[bold green]Uploaded PDF to Firebase:[/bold green] {p_path.name} -> [blue]{pdf_url}[/blue]")

    # Check for visual figures directory
    visuals_dir = p_path.parent / p_path.stem
    if visuals_dir.exists() and visuals_dir.is_dir():
        for img in visuals_dir.glob("*.png"):
            v_url = storage_mgr.upload_visual(img, p_path.stem)
            console.print(f"  Uploaded visual: {img.name} -> [blue]{v_url}[/blue]")


@app.command(name="storage-dir")
def storage_dir(
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

    console.print("[bold green]Firebase Storage Sync Complete![/bold green]")
    console.print(f"Uploaded/mapped [yellow]{len(mapping)}[/yellow] assets with CDN prefix [blue]{storage_mgr.cdn_prefix}[/blue]")

    table = Table(title="Sample Asset CDN URLs")
    table.add_column("Local File", style="cyan")
    table.add_column("Asset CDN URL (assets.edgenie.io)", style="blue")

    for k, v in list(mapping.items())[:15]:
        table.add_row(k, v)

    console.print(table)


# ── Full Automation ────────────────────────────────────────────────────────────

@app.command(name="full-sync")
def full_sync(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Storage directory to sync assets and ingest to DB"),
    uri: str | None = typer.Option(None, "--uri", "-u", help="PostgreSQL connection URI"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Perform both Firebase Storage asset upload and PostgreSQL database ingestion for a directory."""
    setup_logging(log_level)
    d_path = Path(directory_path)
    if not d_path.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{directory_path}'")
        raise typer.Exit(code=1)

    console.print("[bold cyan]Step 1: Syncing assets to Firebase Storage...[/bold cyan]")
    storage_mgr = FirebaseStorageManager()
    storage_mgr.sync_storage_dir(d_path)

    console.print("\n[bold cyan]Step 2: Ingesting structured JSONs into PostgreSQL...[/bold cyan]")
    db_client = PostgresClient(uri=uri) if uri else PostgresClient()
    ingestor = PostgresIngestor(db_client=db_client)
    res = ingestor.ingest_directory(d_path)

    console.print("[bold green]Full Sync Finished Successfully![/bold green]")
    console.print(f"  Papers Ingested:    {res.get('successful_papers')}")
    console.print(f"  Questions Ingested: {res.get('total_questions')}")
    console.print(f"  Answers Ingested:   {res.get('total_answers')}")


# Backward-compatible command aliases
@app.command(name="ingest-paper", hidden=True)
def ingest_paper_alias(json_path: str = typer.Option(..., "--json", "-j")):
    db_paper(json_path=json_path)

@app.command(name="ingest-dir", hidden=True)
def ingest_dir_alias(directory_path: str = typer.Option(..., "--dir", "-d")):
    db_dir(directory_path=directory_path)

@app.command(name="sync-storage", hidden=True)
def sync_storage_alias(storage_dir_path: str = typer.Option("pipeline/storage", "--dir", "-d")):
    storage_dir(storage_dir=storage_dir_path)


def _print_stats_table(stats: dict) -> None:
    table = Table(title="EdGenie PostgreSQL Database Record Counts")
    table.add_column("Table Name", style="cyan")
    table.add_column("Record Count", style="green", justify="right")
    for tbl, cnt in stats.items():
        table.add_row(tbl, str(cnt))
    console.print(table)


if __name__ == "__main__":
    app()
