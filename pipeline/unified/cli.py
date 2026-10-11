"""
pipeline/unified/cli.py

CLI for the Unified 3-in-1 Unstructured.io Pipeline.

Command:
    edgenie-extractor extract-visuals-text-and-classify --pdf <path> [options]

This command:
  1. Detects paper type (ESSAY_QP, MCQ_QP, ESSAY_MS, MCQ_MS).
  2. Optionally loads the subject syllabus CSV for inline classification.
  3. Submits the PDF to Unstructured.io via client.parse.run (with schema),
     which parses, extracts structured text, and runs bounding-box layout
     analysis in ONE cloud job.
  4. Polls until the job completes.
  5. Extracts structured question/answer records from result.extracted_data.
  6. Extracts layout elements with bounding boxes from result.elements.
  7. Crops and saves visual assets (images, tables) locally with PyMuPDF.
  8. Links visual paths into question records.
  9. Saves the canonical <stem>.json and visuals/manifest.json.

No duplicate API charges. No separate classification step.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import dotenv
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

dotenv.load_dotenv()

from pipeline.unified.paper_type_detector import (
    PaperCategory,
    PaperTypeDetector,
    parse_caie_filename,
)
from pipeline.unified.pipeline import UnifiedPipeline
from pipeline.unified.syllabus_loader import SyllabusLoader

app = typer.Typer(
    help="EdGenie Unified 3-in-1 Pipeline: Text + Visuals + Classification (Unstructured.io)",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main_callback() -> None:
    """EdGenie Unified 3-in-1 Pipeline: Text + Visuals + Classification."""
    pass


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def _get_meta_context(pdf: Path) -> dict[str, Any]:
    meta = parse_caie_filename(pdf.with_suffix(".pdf").name)
    year = meta["year"]
    session = meta["session"]
    subject_code = meta["subject_code"]
    variant = meta["variant"]

    parts = pdf.parent.parts
    if len(parts) >= 3:
        try:
            year = int(parts[-2])
            session = parts[-1]
            subject_code = parts[-3]
        except ValueError:
            pass

    return {
        "year": year,
        "session": session,
        "variant": variant,
        "subject_code": subject_code,
    }


def _resolve_category(pdf: Path, paper_type: str | None) -> PaperCategory:
    category = PaperTypeDetector.detect(pdf)
    if paper_type:
        pt = paper_type.strip().lower()
        is_ms = "_ms_" in pdf.name.lower() or "ms" in pdf.name.lower()
        if pt == "essay":
            category = PaperCategory.ESSAY_MS if is_ms else PaperCategory.ESSAY_QP
        elif pt == "mcq":
            category = PaperCategory.MCQ_MS if is_ms else PaperCategory.MCQ_QP
    return category


def _resolve_syllabus(syllabus: str | None, subject_code: str) -> Path | None:
    return SyllabusLoader.resolve_syllabus_path(syllabus, subject_code)


def _print_results_table(pdf_name: str, records: list[dict[str, Any]], is_qp: bool) -> None:
    tbl = Table(title=f"Unified Extraction Results: {pdf_name}")
    tbl.add_column("Q Number", style="cyan")
    tbl.add_column("Marks", style="magenta")
    if is_qp:
        tbl.add_column("Topic", style="green")
        tbl.add_column("Topic Name", style="yellow")
        tbl.add_column("Difficulty", style="red")
        tbl.add_column("Command Words", style="blue")
    else:
        tbl.add_column("Answer Preview", style="green")

    for rec in records[:10]:
        if is_qp:
            tbl.add_row(
                (rec.get("QuestionNumber") or "-").split("/")[-1],
                str(rec.get("Marks", "-")),
                rec.get("Topic") or "-",
                rec.get("Topic_name") or "-",
                f"{rec.get('Difficulty_label') or '-'} ({rec.get('Difficulty_score', '-')})",
                ", ".join(rec.get("Command_words") or []) or "-",
            )
        else:
            preview = rec.get("Answer", rec.get("Answer_text", ""))
            if len(preview) > 50:
                preview = preview[:50] + "..."
            tbl.add_row(
                (rec.get("AnswerNumber") or "-").split("/")[-1],
                "-",
                preview,
            )
    if len(records) > 10:
        tbl.add_row("...", "...", "...", "...", "...", "...")
    console.print(tbl)


# ── CLI command ───────────────────────────────────────────────────────────────

@app.command("extract-visuals-text-and-classify")
def unified_extract_cmd(
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to the exam PDF"),
    paper_type: str | None = typer.Option(
        None, "--type", "-t",
        help="Override paper type: 'essay' or 'mcq'",
    ),
    syllabus: str | None = typer.Option(
        None, "--syllabus", "-s",
        help="Path to the subject syllabus CSV for inline classification. "
             "Auto-discovered from syllabi/ if omitted.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run even if outputs exist"),
    dpi: int = typer.Option(300, "--dpi", help="DPI for visual crops"),
    profile: str = typer.Option("best", "--profile", help="Unstructured.io processing profile"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Unified 3-in-1 pipeline: extracts text, visuals, and syllabus classification
    for a Cambridge exam PDF in a single Unstructured.io API call.

    Outputs:
      <stem>.json          — structured questions / mark-scheme records
      <stem>/manifest.json — visual asset manifest
      <stem>/<visuals>.png — cropped diagram/table images (question papers only)
    """
    _setup_logging(log_level)

    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found at '{pdf_path}'")
        raise typer.Exit(1)

    # Skip non-question and non-mark-scheme files
    if any(sfx in pdf.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ci_"]):
        console.print(
            f"[yellow]Skipping '{pdf.name}' — Examiner Reports / Grade Thresholds "
            f"are excluded from extraction.[/yellow]"
        )
        raise typer.Exit(0)

    unstructured_key = os.environ.get("UNSTRUCTURED_API_KEY")
    if not unstructured_key:
        console.print(
            "[bold red]Error:[/bold red] UNSTRUCTURED_API_KEY is not set. "
            "Add it to your .env file."
        )
        raise typer.Exit(1)

    # ── Step 1: detect paper type ─────────────────────────────────────────────
    console.print("\n[bold blue]1. Detecting Paper Type...[/bold blue]")
    category = _resolve_category(pdf, paper_type)
    if category == PaperCategory.OTHER:
        console.print("[bold red]Error:[/bold red] Could not determine paper category.")
        raise typer.Exit(1)
    console.print(
        f"   Category: [bold yellow]{category.name}[/bold yellow] "
        f"([cyan]{category.value}[/cyan])"
    )

    # ── Step 2: meta context ──────────────────────────────────────────────────
    meta = _get_meta_context(pdf)

    # ── Step 3: resolve syllabus ──────────────────────────────────────────────
    console.print("\n[bold blue]2. Resolving Syllabus CSV...[/bold blue]")
    syllabus_path = _resolve_syllabus(syllabus, meta["subject_code"])
    if syllabus_path:
        console.print(f"   Syllabus: [green]{syllabus_path.name}[/green]")
    else:
        console.print(
            "   [yellow]No syllabus CSV found — classification fields will be "
            "best-effort without topic constraints.[/yellow]"
        )

    # ── Step 4: check for existing output ─────────────────────────────────────
    json_out = pdf.with_suffix(".json")
    visuals_dir = pdf.parent / pdf.stem
    if json_out.exists() and visuals_dir.exists() and not force:
        console.print(
            f"\n[yellow]Outputs already exist for '{pdf.name}'. "
            f"Use --force to re-extract.[/yellow]"
        )
        raise typer.Exit(0)

    visuals_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 5: run the unified pipeline ─────────────────────────────────────
    console.print(
        "\n[bold green]3. Running Unified 3-in-1 Pipeline "
        "(Unstructured.io parse + extract + elements)...[/bold green]"
    )
    pipeline = UnifiedPipeline(dpi=dpi, profile=profile)

    with console.status("[bold green]Submitting to Unstructured.io and polling..."):
        result = pipeline.run(
            pdf_path=pdf,
            category=category,
            meta=meta,
            output_dir=visuals_dir,
            syllabus_csv=syllabus_path,
            force=force,
        )

    records: list[dict[str, Any]] = result["records"]
    manifest: list[dict[str, Any]] = result["manifest"]
    json_path: Path = result["json_path"]
    manifest_path_out: Path = result["manifest_path"]

    # ── Step 6: report ────────────────────────────────────────────────────────
    console.print(f"\n[bold blue]4. Results[/bold blue]")
    console.print(f"   [green]✓[/green] {len(records)} record(s) extracted and classified.")
    console.print(f"   [green]✓[/green] {len(manifest)} visual asset(s) saved to '{visuals_dir.name}/'.")
    console.print(f"   [green]✓[/green] Canonical JSON: [cyan]{json_path.name}[/cyan]")
    if manifest:
        console.print(f"   [green]✓[/green] Manifest: [cyan]{manifest_path_out.name}[/cyan]")

    is_qp = category in (PaperCategory.ESSAY_QP, PaperCategory.MCQ_QP)
    if records:
        _print_results_table(pdf.name, records, is_qp)
