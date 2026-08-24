"""
Stage 3 — Visual Extraction CLI

Extracts figures, diagrams, tables, and apparatus drawings from Cambridge exam PDFs
using layout and caption anchoring, tags them with their question numbers, and
patches Stage 2 question JSON files.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

import dotenv
dotenv.load_dotenv()

from pipeline.config import settings
from pipeline.visuals.extractor import CambridgeVisualExtractor

app = typer.Typer(help="EdGenie Stage 3 — Visual Extraction CLI (Cambridge Layout-Anchored)")
console = Console()


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def extract_visuals_for_pdf(
    pdf_path: Path,
    json_path: Optional[Path] = None,
    force: bool = False,
    dpi: int = 300,
    min_vector_area: float = 400.0,
) -> Optional[List[Dict[str, Any]]]:
    """
    Core extraction function for a single PDF.
    Returns list of manifest dicts or None if skipped/empty.
    """
    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found at '{pdf}'")
        return None

    # Only process question paper PDFs in Stage 3
    if "_qp_" not in pdf.stem.lower():
        console.print(
            f"[yellow]Skipping '{pdf.name}' — only question papers (_qp_) contain extractable visuals.[/yellow]"
        )
        return None

    # Sibling JSON (Stage 2 question output)
    qp_json = json_path if json_path else pdf.with_suffix(".json")

    # Output directory named after the PDF (e.g. storage/.../5054_s10_qp_41/)
    visuals_dir = pdf.parent / pdf.stem
    if visuals_dir.exists() and not force:
        console.print(
            f"[yellow]Directory '{pdf.stem}' already exists. Use --force to re-extract.[/yellow]"
        )
        return None

    console.print(f"\n[bold blue]Stage 3: Cambridge Visual Extraction — {pdf.name}[/bold blue]")

    # Run extraction
    extractor = CambridgeVisualExtractor(dpi=dpi, min_vector_area=min_vector_area)
    with console.status("[bold green]Extracting figures and diagrams..."):
        manifest = extractor.extract(pdf, visuals_dir)

    if not manifest:
        console.print("[yellow]No visual figures detected in this paper.[/yellow]")
        return []

    console.print(f"  Extracted [green]{len(manifest)}[/green] figures into [cyan]{visuals_dir.name}/[/cyan]")

    # Patch sibling Stage 2 JSON if it exists
    if qp_json.exists():
        _patch_question_json(qp_json, manifest)

    # Render summary table
    tbl = Table(title=f"Extracted Visuals — {pdf.name}")
    tbl.add_column("File", style="cyan")
    tbl.add_column("Question", style="yellow")
    tbl.add_column("Caption", style="green")
    tbl.add_column("Page", style="magenta")
    tbl.add_column("BBox [x0, y0, x1, y1]", style="blue")

    for entry in manifest[:25]:
        q_label = f"Q{entry['question']}" if entry.get("question") is not None else "-"
        cap_label = entry.get("caption") or "(fallback)"
        bbox_str = str(entry.get("bbox", []))
        tbl.add_row(
            entry["filename"],
            q_label,
            cap_label,
            str(entry.get("page", "-")),
            bbox_str,
        )

    if len(manifest) > 25:
        tbl.add_row("…", "…", "…", "…", f"({len(manifest) - 25} more)")

    console.print(tbl)
    return manifest


@app.command()
def extract(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to raw exam PDF file"),
    json_path: Optional[str] = typer.Option(
        None, "--json", "-j",
        help="Path to Stage 2 JSON (for auto-patching question image fields)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing visuals/"),
    dpi: int = typer.Option(300, "--dpi", help="Render resolution DPI (default: 300)"),
    min_vector_area: float = typer.Option(400.0, "--min-vector-area", help="Minimum vector cluster area"),
    log_level: str = typer.Option(settings.log_level, "--log-level"),
):
    """Extract figures, graphs, tables and diagrams from a single exam PDF."""
    setup_logging(log_level)
    pdf = Path(pdf_path)
    j_path = Path(json_path) if json_path else None
    result = extract_visuals_for_pdf(
        pdf_path=pdf,
        json_path=j_path,
        force=force,
        dpi=dpi,
        min_vector_area=min_vector_area,
    )
    if result is None and not force and (pdf.parent / pdf.stem).exists():
        raise typer.Exit(code=0)


@app.command()
def extract_dir(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory of PDF files"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing visuals"),
    dpi: int = typer.Option(300, "--dpi", help="Render resolution DPI"),
    min_vector_area: float = typer.Option(400.0, "--min-vector-area", help="Minimum vector cluster area"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Extract visuals from all question paper PDFs in a directory."""
    setup_logging(log_level)
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{directory_path}'")
        raise typer.Exit(code=1)

    pdfs = [p for p in dir_path.glob("**/*.pdf") if "_qp_" in p.stem.lower()]
    if not pdfs:
        console.print(f"[yellow]No QP PDFs found in '{directory_path}'[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"Found [bold cyan]{len(pdfs)}[/bold cyan] QP PDFs to process.")
    done = 0

    for pdf in pdfs:
        console.print(f"\n[bold magenta]── {pdf.name} ──[/bold magenta]")
        j_path = pdf.with_suffix(".json")

        try:
            res = extract_visuals_for_pdf(
                pdf_path=pdf,
                json_path=j_path if j_path.exists() else None,
                force=force,
                dpi=dpi,
                min_vector_area=min_vector_area,
            )
            if res is not None:
                done += 1
        except Exception as e:
            console.print(f"[red]Failed {pdf.name}: {e}[/red]")

    console.print(f"\n[bold green]Done![/bold green] Processed {done}/{len(pdfs)} PDFs.")


# ── JSON Patching Helper ──────────────────────────────────────────────────────

def _patch_question_json(json_path: Path, manifest: List[Dict[str, Any]]) -> None:
    """
    Update the Stage 2 JSON file so each question's `image` field points
    to its extracted figures matching question number.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: could not read {json_path.name} for patching: {e}[/yellow]")
        return

    if not records or not manifest:
        return

    # Group manifest relative paths by question number
    q_to_files: Dict[int, List[str]] = {}
    for entry in manifest:
        q = entry.get("question")
        if q is not None:
            q_to_files.setdefault(int(q), []).append(entry["relative_path"])

    patched = 0
    for record in records:
        q_raw = record.get("QuestionNumber") or record.get("AnswerNumber") or ""
        # Extract integer question number from strings like "5054/s10/qp/41/Q1" or "Q1" or "1"
        m = re.search(r"(?:Q|q)?(\d+)(?:\([a-z]\))?$", str(q_raw).split("/")[-1].strip())
        if m:
            q_num = int(m.group(1))
            if q_num in q_to_files:
                images = q_to_files[q_num]
                record["image"] = images if len(images) > 1 else images[0]
                patched += 1

    if patched:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        console.print(f"  Patched image fields in [green]{json_path.name}[/green] for [cyan]{patched}[/cyan] questions.")


if __name__ == "__main__":
    app()
