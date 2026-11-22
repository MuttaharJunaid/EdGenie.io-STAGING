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

import logging
import os
from pathlib import Path
from typing import Any

import dotenv
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pipeline.unified.paper_type_detector import (
    PaperCategory,
    PaperTypeDetector,
    parse_caie_filename,
)
from pipeline.unified.pipeline import (
    JobTimeoutError,
    UnifiedPipeline,
    get_next_run_number,
    resolve_timed_out_jobs_path,
)
from pipeline.unified.syllabus_loader import SyllabusLoader

dotenv.load_dotenv()

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

def _process_single_pdf(
    pdf: Path,
    paper_type: str | None = None,
    syllabus: str | None = None,
    force: bool = False,
    dpi: int = 300,
    profile: str = "best",
    pipeline: UnifiedPipeline | None = None,
    run_number: int | str | None = None,
    suffix: str | None = None,
    timed_out_file: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executes unified extraction, visual snipping, and classification for a single PDF.
    Returns a result dict with status ('success', 'skipped', 'timed_out', or 'error') and details.
    """
    if not pdf.exists():
        return {"status": "error", "error": f"PDF not found at '{pdf}'"}

    # Skip non-question and non-mark-scheme files
    if any(sfx in pdf.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ci_", "_sy_", "_sy."]):
        return {
            "status": "skipped",
            "reason": "Examiner Reports / Grade Thresholds / Non-exam PDF excluded",
        }

    category = _resolve_category(pdf, paper_type)
    if category == PaperCategory.OTHER:
        return {"status": "error", "error": f"Could not determine paper category for '{pdf.name}'"}

    meta = _get_meta_context(pdf)
    syllabus_path = _resolve_syllabus(syllabus, meta["subject_code"])

    json_out = pdf.with_suffix(".json")
    visuals_dir = pdf.parent / pdf.stem
    if json_out.exists() and visuals_dir.exists() and not force:
        return {
            "status": "skipped",
            "reason": f"Outputs already exist for '{pdf.name}' (use --force to re-extract)",
            "category": category,
            "json_path": json_out,
            "visuals_dir": visuals_dir,
            "meta": meta,
        }

    visuals_dir.mkdir(parents=True, exist_ok=True)

    if pipeline is None:
        timed_out_path = resolve_timed_out_jobs_path(
            run_number=run_number,
            suffix=suffix,
            explicit_file=timed_out_file,
            auto_increment=(run_number is None and suffix is None and timed_out_file is None),
        )
        pipeline = UnifiedPipeline(
            dpi=dpi,
            profile=profile,
            timed_out_jobs_file=timed_out_path,
        )
        pipeline.init_timed_out_jobs_file()

    try:
        result = pipeline.run(
            pdf_path=pdf,
            category=category,
            meta=meta,
            output_dir=visuals_dir,
            syllabus_csv=syllabus_path,
            force=force,
        )
        return {
            "status": "success",
            "category": category,
            "records": result["records"],
            "manifest": result["manifest"],
            "json_path": result["json_path"],
            "manifest_path": result["manifest_path"],
            "visuals_dir": visuals_dir,
            "meta": meta,
            "syllabus_path": syllabus_path,
        }
    except JobTimeoutError as te:
        return {
            "status": "timed_out",
            "job_id": te.job_id,
            "error": f"Job {te.job_id} timed out during polling: {te}",
            "category": category,
            "meta": meta,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "category": category, "meta": meta}


# ── CLI commands ─────────────────────────────────────────────────────────────

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
    run_number: int | None = typer.Option(
        None, "--run-number", "-r",
        help="Pipeline run number for timed_out_jobs suffix (e.g. 1 -> timed_out_jobs_run_1.json)",
    ),
    suffix: str | None = typer.Option(
        None, "--suffix",
        help="Custom suffix for timed_out_jobs JSON file (e.g. 'run_1' or '1')",
    ),
    timed_out_file: str | None = typer.Option(
        None, "--timed-out-file",
        help="Custom path or filename for the timed-out jobs JSON file",
    ),
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

    unstructured_key = os.environ.get("UNSTRUCTURED_API_KEY")
    if not unstructured_key:
        console.print(
            "[bold red]Error:[/bold red] UNSTRUCTURED_API_KEY is not set. "
            "Add it to your .env file."
        )
        raise typer.Exit(1)

    timed_out_path = resolve_timed_out_jobs_path(
        run_number=run_number,
        suffix=suffix,
        explicit_file=timed_out_file,
        auto_increment=(run_number is None and suffix is None and timed_out_file is None),
    )
    pipeline = UnifiedPipeline(
        dpi=dpi,
        profile=profile,
        timed_out_jobs_file=timed_out_path,
    )
    pipeline.init_timed_out_jobs_file()

    console.print("\n[bold blue]Starting Unified 3-in-1 Extraction Pipeline...[/bold blue]")
    console.print(f"Target PDF: [cyan]{pdf.name}[/cyan]")
    console.print(f"Timed-out jobs registry: [cyan]{pipeline.timed_out_jobs_file.name}[/cyan]")

    res = _process_single_pdf(
        pdf=pdf,
        paper_type=paper_type,
        syllabus=syllabus,
        force=force,
        dpi=dpi,
        profile=profile,
        pipeline=pipeline,
    )

    if res["status"] == "skipped":
        console.print(f"[yellow]{res.get('reason', 'Skipped')}[/yellow]")
        raise typer.Exit(0)

    if res["status"] == "timed_out":
        job_id = res.get("job_id", "-")
        console.print(
            f"   [yellow]⏳ Timed out (Job ID: {job_id})[/yellow] — recorded in [cyan]{pipeline.timed_out_jobs_file.name}[/cyan] for recovery."
        )
        console.print(
            f"   [dim]Recover with:[/dim] [bold cyan]edgenie-unified recover-timed-out --file {pipeline.timed_out_jobs_file.name}[/bold cyan]"
        )
        raise typer.Exit(1)

    if res["status"] == "error":
        console.print(f"[bold red]Error:[/bold red] {res.get('error')}")
        raise typer.Exit(1)

    category: PaperCategory = res["category"]
    records: list[dict[str, Any]] = res["records"]
    manifest: list[dict[str, Any]] = res["manifest"]
    json_path: Path = res["json_path"]
    manifest_path_out: Path = res["manifest_path"]
    visuals_dir: Path = res["visuals_dir"]

    console.print("\n[bold blue]Extraction Results[/bold blue]")
    console.print(f"   [green]✓[/green] Category: [bold yellow]{category.name}[/bold yellow] ({category.value})")
    console.print(f"   [green]✓[/green] {len(records)} record(s) extracted and classified.")
    console.print(f"   [green]✓[/green] {len(manifest)} visual asset(s) saved to '{visuals_dir.name}/'.")
    console.print(f"   [green]✓[/green] Canonical JSON: [cyan]{json_path.name}[/cyan]")
    if manifest:
        console.print(f"   [green]✓[/green] Manifest: [cyan]{manifest_path_out.name}[/cyan]")

    is_qp = category in (PaperCategory.ESSAY_QP, PaperCategory.MCQ_QP)
    if records:
        _print_results_table(pdf.name, records, is_qp)


@app.command("extract")
def extract_cmd(
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
    run_number: int | None = typer.Option(
        None, "--run-number", "-r",
        help="Pipeline run number for timed_out_jobs suffix (e.g. 1 -> timed_out_jobs_run_1.json)",
    ),
    suffix: str | None = typer.Option(
        None, "--suffix",
        help="Custom suffix for timed_out_jobs JSON file (e.g. 'run_1' or '1')",
    ),
    timed_out_file: str | None = typer.Option(
        None, "--timed-out-file",
        help="Custom path or filename for the timed-out jobs JSON file",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Convenience alias for 'extract-visuals-text-and-classify' on a single PDF."""
    unified_extract_cmd(
        pdf_path=pdf_path,
        paper_type=paper_type,
        syllabus=syllabus,
        force=force,
        dpi=dpi,
        profile=profile,
        run_number=run_number,
        suffix=suffix,
        timed_out_file=timed_out_file,
        log_level=log_level,
    )


@app.command("extract-dir")
def extract_dir_cmd(
    dir_path: str = typer.Option(..., "--dir", "-d", help="Directory path containing past paper PDFs"),
    paper_type: str | None = typer.Option(
        None, "--type", "-t",
        help="Override paper type: 'essay' or 'mcq' (applies to all papers if set)",
    ),
    syllabus: str | None = typer.Option(
        None, "--syllabus", "-s",
        help="Path to subject syllabus CSV (auto-discovered from syllabi/ per subject code if omitted)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Re-extract even if outputs already exist"),
    dpi: int = typer.Option(300, "--dpi", help="DPI for visual crops"),
    profile: str = typer.Option("best", "--profile", help="Unstructured.io processing profile"),
    run_number: int | None = typer.Option(
        None, "--run-number", "-r",
        help="Pipeline run number for timed_out_jobs suffix (e.g. 1 -> timed_out_jobs_run_1.json)",
    ),
    suffix: str | None = typer.Option(
        None, "--suffix",
        help="Custom suffix for timed_out_jobs JSON file (e.g. 'run_1' or '1')",
    ),
    timed_out_file: str | None = typer.Option(
        None, "--timed-out-file",
        help="Custom path or filename for the timed-out jobs JSON file",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Batch processes an entire directory (e.g. subject folder or session folder) of exam PDFs.

    Recursively scans for exam question papers (_qp_) and mark schemes (_ms_),
    skips non-exam files (_er, _gt, _ci, etc.), auto-detects paper categories
    and syllabi, runs the 3-in-1 pipeline for each paper, and presents a summary table.
    """
    _setup_logging(log_level)

    base_dir = Path(dir_path).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{dir_path}'")
        raise typer.Exit(1)

    unstructured_key = os.environ.get("UNSTRUCTURED_API_KEY")
    if not unstructured_key:
        console.print(
            "[bold red]Error:[/bold red] UNSTRUCTURED_API_KEY is not set. "
            "Add it to your .env file."
        )
        raise typer.Exit(1)

    all_pdfs = sorted(list(base_dir.rglob("*.pdf")))
    exam_pdfs: list[Path] = []
    non_exam_count = 0

    for p in all_pdfs:
        # Exclude examiner reports, grade thresholds, confidential instructions, syllabus overviews
        if any(sfx in p.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ci_", "_sy_", "_sy."]):
            non_exam_count += 1
            continue
        if "_qp_" in p.name.lower() or "_ms_" in p.name.lower():
            exam_pdfs.append(p)
        else:
            cat = PaperTypeDetector.detect(p)
            if cat != PaperCategory.OTHER:
                exam_pdfs.append(p)
            else:
                non_exam_count += 1

    if not exam_pdfs:
        console.print(
            f"[yellow]No exam question papers or mark schemes found in '{dir_path}' "
            f"(skipped {non_exam_count} non-exam PDF(s)).[/yellow]"
        )
        raise typer.Exit(0)

    timed_out_path = resolve_timed_out_jobs_path(
        run_number=run_number,
        suffix=suffix,
        explicit_file=timed_out_file,
        auto_increment=(run_number is None and suffix is None and timed_out_file is None),
    )
    pipeline = UnifiedPipeline(
        dpi=dpi,
        profile=profile,
        timed_out_jobs_file=timed_out_path,
    )
    pipeline.init_timed_out_jobs_file()

    console.print(f"\n[bold blue]Starting Batch Directory Extraction for:[/bold blue] {base_dir}")
    console.print(f"Found [bold green]{len(exam_pdfs)}[/bold green] exam PDF(s) to process (skipped {non_exam_count} non-exam file(s)).")
    console.print(f"Timed-out jobs registry: [cyan]{pipeline.timed_out_jobs_file.name}[/cyan]")

    results_summary: list[dict[str, str]] = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, pdf in enumerate(exam_pdfs, 1):
        console.print(f"\n[bold cyan][{idx}/{len(exam_pdfs)}] Processing: {pdf.name}[/bold cyan]")
        res = _process_single_pdf(
            pdf=pdf,
            paper_type=paper_type,
            syllabus=syllabus,
            force=force,
            dpi=dpi,
            profile=profile,
            pipeline=pipeline,
        )

        status = res.get("status")
        cat_val = res.get("category")
        cat_name = getattr(cat_val, "name", "-") if cat_val is not None else "-"
        records_len = len(res.get("records", []))
        manifest_len = len(res.get("manifest", []))

        if status == "success":
            processed_count += 1
            console.print(
                f"   [green]✓ Succeeded:[/green] {records_len} record(s), "
                f"{manifest_len} visual asset(s) cropped."
            )
            results_summary.append({
                "file": pdf.name,
                "category": cat_name,
                "status": "[green]Success[/green]",
                "records": str(records_len),
                "visuals": str(manifest_len),
                "note": "Extracted & Classified",
            })
        elif status == "skipped":
            skipped_count += 1
            reason = res.get("reason", "Skipped")
            console.print(f"   [yellow]↷ Skipped:[/yellow] {reason}")
            results_summary.append({
                "file": pdf.name,
                "category": cat_name,
                "status": "[yellow]Skipped[/yellow]",
                "records": "-",
                "visuals": "-",
                "note": reason[:45],
            })
        elif status == "timed_out":
            failed_count += 1
            job_id = str(res.get("job_id") or "-")
            console.print(f"   [yellow]⏳ Timed out (Job ID: {job_id})[/yellow] — recorded in {pipeline.timed_out_jobs_file.name} for recovery.")
            results_summary.append({
                "file": pdf.name,
                "category": cat_name,
                "status": "[yellow]Timed Out[/yellow]",
                "records": "-",
                "visuals": "-",
                "note": f"Job ID: {job_id[:12]}...",
            })
        else:
            failed_count += 1
            err = res.get("error", "Unknown error")
            console.print(f"   [red]✗ Failed:[/red] {err}")
            results_summary.append({
                "file": pdf.name,
                "category": cat_name,
                "status": "[red]Failed[/red]",
                "records": "-",
                "visuals": "-",
                "note": err[:45],
            })

    # Check for timed-out jobs and attempt recovery via playground endpoint
    pending_timed_out = [j for j in pipeline.timed_out_jobs if j.get("status") == "timed_out"]
    if pending_timed_out:
        console.print(
            f"\n[bold yellow]Attempting endpoint recovery for {len(pending_timed_out)} timed-out job(s) "
            f"from '{pipeline.timed_out_jobs_file.name}' via https://transform.unstructured.io/playground/jobs/...[/bold yellow]"
        )
        recovered_list = pipeline.recover_timed_out_jobs()
        if recovered_list:
            for rec in recovered_list:
                processed_count += 1
                failed_count = max(0, failed_count - 1)
                console.print(
                    f"   [green]✓ Recovered job {rec['job_id']} for {rec['pdf_name']}:[/green] "
                    f"{rec['records_count']} record(s), {rec['manifest_count']} visual asset(s)."
                )
                for row in results_summary:
                    if row["file"] == rec["pdf_name"]:
                        row["status"] = "[green]Recovered[/green]"
                        row["records"] = str(rec["records_count"])
                        row["visuals"] = str(rec["manifest_count"])
                        row["note"] = f"Recovered from endpoint ({rec['job_id'][:8]}...)"
        remaining_pending = [j for j in pipeline.timed_out_jobs if j.get("status") == "timed_out"]
        if remaining_pending:
            console.print(
                f"\n[yellow]⏳ {len(remaining_pending)} job(s) remain pending in '{pipeline.timed_out_jobs_file.name}'.[/yellow]\n"
                f"   [dim]Recover later with:[/dim] [bold cyan]edgenie-unified recover-timed-out --file {pipeline.timed_out_jobs_file.name}[/bold cyan]"
            )

    # Summary table
    summary_tbl = Table(title=f"Batch Extraction Summary: {base_dir.name}")
    summary_tbl.add_column("PDF File", style="cyan")
    summary_tbl.add_column("Category", style="magenta")
    summary_tbl.add_column("Status")
    summary_tbl.add_column("Records", justify="right")
    summary_tbl.add_column("Visuals", justify="right")
    summary_tbl.add_column("Notes", style="dim")

    for row in results_summary:
        summary_tbl.add_row(
            row["file"],
            row["category"],
            row["status"],
            row["records"],
            row["visuals"],
            row["note"],
        )

    console.print("\n")
    console.print(summary_tbl)
    console.print(
        f"\n[bold]Batch Finished:[/bold] Total: {len(exam_pdfs)} | "
        f"[green]Processed: {processed_count}[/green] | "
        f"[yellow]Skipped: {skipped_count}[/yellow] | "
        f"[red]Failed: {failed_count}[/red]"
    )

    if failed_count > 0:
        raise typer.Exit(1)


@app.command("recover-job")
def recover_job_cmd(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Unstructured.io job ID to recover"),
    pdf_path: str = typer.Option(..., "--pdf", "-p", help="Path to original PDF"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Visuals output directory (default: <stem>/)"),
    dpi: int = typer.Option(300, "--dpi", help="DPI for visual crops"),
    profile: str = typer.Option("best", "--profile", help="Unstructured.io processing profile"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Recover and complete processing for a specific job using:
    https://transform.unstructured.io/playground/jobs/{job_id}?output=elements
    """
    _setup_logging(log_level)
    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        console.print(f"[bold red]Error:[/bold red] PDF not found at '{pdf_path}'")
        raise typer.Exit(1)

    out_dir = Path(output_dir).resolve() if output_dir else pdf.parent / pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Recovering Unstructured.io Job:[/bold blue] [cyan]{job_id}[/cyan]")
    console.print(f"Target PDF: [cyan]{pdf.name}[/cyan]")
    console.print(f"Endpoint: https://transform.unstructured.io/playground/jobs/{job_id}?output=elements")

    pipeline = UnifiedPipeline(dpi=dpi, profile=profile)
    job_result = pipeline.fetch_job_result_from_endpoint(job_id)
    if not job_result:
        console.print(f"[bold red]Error:[/bold red] Job {job_id} could not be retrieved from endpoint (not ready or not found).")
        raise typer.Exit(1)

    category = PaperTypeDetector.detect(pdf)
    work_res = pipeline.complete_pipeline_work(
        pdf_path=pdf,
        output_dir=out_dir,
        job_result=job_result,
        category=category,
    )

    records = work_res["records"]
    manifest = work_res["manifest"]
    console.print(f"\n[bold green]✓ Job {job_id} successfully recovered and processed![/bold green]")
    console.print(f"   [green]✓[/green] Category: [bold yellow]{category.name}[/bold yellow]")
    console.print(f"   [green]✓[/green] {len(records)} record(s) extracted and classified.")
    console.print(f"   [green]✓[/green] {len(manifest)} visual asset(s) cropped.")
    console.print(f"   [green]✓[/green] Canonical JSON: [cyan]{work_res['json_path'].name}[/cyan]")
    if manifest:
        console.print(f"   [green]✓[/green] Manifest: [cyan]{work_res['manifest_path'].name}[/cyan]")


@app.command("run-snipping-using-response")
def run_snipping_using_response_cmd(
    pdf: str | None = typer.Option(None, "--pdf", "-p", help="Path to a single exam PDF"),
    dir_path: str | None = typer.Option(None, "--dir", "-d", help="Directory containing exam PDFs to process"),
    dpi: int = typer.Option(300, "--dpi", help="DPI for visual crops"),
    profile: str = typer.Option("best", "--profile", help="Unstructured.io processing profile"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Run snipping/cropping using stored Unstructured.io JSON responses.

    Looks for cached responses (<stem>_unified_text.json and <stem>_unified_elements.json)
    and runs only the visual cropping/snipping step, writing visuals into the
    respective <stem>/ directories and updating manifest.json/canonical JSON.
    """
    import json

    _setup_logging(log_level)

    if not pdf and not dir_path:
        console.print("[bold red]Error:[/bold red] Provide either --pdf or --dir")
        raise typer.Exit(1)

    pdfs: list[Path] = []
    if pdf:
        p = Path(pdf).resolve()
        if not p.exists():
            console.print(f"[bold red]Error:[/bold red] PDF not found at '{pdf}'")
            raise typer.Exit(1)
        pdfs = [p]
    if dir_path:
        base_dir = Path(dir_path).resolve()
        if not base_dir.exists() or not base_dir.is_dir():
            console.print(f"[bold red]Error:[/bold red] Directory not found at '{dir_path}'")
            raise typer.Exit(1)
        all_pdfs = sorted(list(base_dir.rglob("*.pdf")))
        for p in all_pdfs:
            if any(sfx in p.name.lower() for sfx in ["_er_", "_er.", "_gt_", "_gt.", "_ir_", "_ci_", "_sy_", "_sy."]):
                continue
            if "_qp_" in p.name.lower() or "_ms_" in p.name.lower():
                pdfs.append(p)
            else:
                cat = PaperTypeDetector.detect(p)
                if cat != PaperCategory.OTHER:
                    pdfs.append(p)

    if not pdfs:
        console.print("[yellow]No exam PDFs found.[/yellow]")
        raise typer.Exit(0)

    pipeline = UnifiedPipeline(dpi=dpi, profile=profile)

    processed = 0
    failed = 0
    for idx, pdf_path in enumerate(pdfs, 1):
        pdf_path = Path(pdf_path).resolve()
        stem = pdf_path.stem
        out_dir = pdf_path.parent / stem
        cache_text = pdf_path.parent / f"{stem}_unified_text.json"
        cache_elements = pdf_path.parent / f"{stem}_unified_elements.json"

        console.print(f"\n[bold cyan][{idx}/{len(pdfs)}] Processing: {pdf_path.name}[/bold cyan]")

        if not cache_text.exists() or not cache_elements.exists():
            console.print(f"   [yellow]↷ Skipped:[/yellow] Missing stored responses ({stem}_unified_text.json / {stem}_unified_elements.json)")
            continue

        try:
            with open(cache_text, encoding="utf-8") as f:
                records = json.load(f)
            with open(cache_elements, encoding="utf-8") as f:
                elements = json.load(f)
        except Exception as e:
            console.print(f"   [red]✗ Failed:[/red] Could not load cached responses: {e}")
            failed += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            category = PaperTypeDetector.detect(pdf_path)
            work_res = pipeline.complete_pipeline_work(
                pdf_path=pdf_path,
                output_dir=out_dir,
                records=records,
                elements=elements,
                category=category,
            )
            manifest = work_res.get("manifest", [])
            console.print(f"   [green]✓ Done:[/green] {len(records)} record(s), {len(manifest)} visual asset(s) snipped into {out_dir.name}/")
            processed += 1
        except Exception as e:
            console.print(f"   [red]✗ Failed:[/red] {e}")
            failed += 1
            continue

    console.print(f"\n[bold]Snipping complete:[/bold] Processed: {processed} | Failed: {failed}")
    if failed > 0:
        raise typer.Exit(1)


@app.command("recover-timed-out")
def recover_timed_out_cmd(
    file_path: str = typer.Option(
        "timed_out_jobs.json", "--file", "-f",
        help="Path or suffix of timed_out_jobs.json (e.g. timed_out_jobs_run_1.json, timed_out_jobs.json, or run_1)",
    ),
    run_number: int | None = typer.Option(
        None, "--run-number", "-r",
        help="Pipeline run number to recover (e.g. 1 -> timed_out_jobs_run_1.json)",
    ),
    suffix: str | None = typer.Option(
        None, "--suffix",
        help="Suffix of timed_out_jobs file to recover (e.g. 'run_1' or '1')",
    ),
    dpi: int = typer.Option(300, "--dpi", help="DPI for visual crops"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Recover all pending timed-out jobs recorded in a timed-out jobs JSON file via the playground endpoint.
    """
    _setup_logging(log_level)

    jobs_path: Path | None = None
    if run_number is not None:
        cand1 = Path(f"timed_out_jobs_run_{run_number}.json").resolve()
        cand2 = Path(f"timed_out_jobs_{run_number}.json").resolve()
        if cand1.exists():
            jobs_path = cand1
        elif cand2.exists():
            jobs_path = cand2
        else:
            jobs_path = cand1
    elif suffix is not None:
        sfx = suffix.lstrip("_")
        cand1 = Path(f"timed_out_jobs_{sfx}.json").resolve()
        cand2 = Path(f"timed_out_jobs_run_{sfx}.json").resolve()
        if cand1.exists():
            jobs_path = cand1
        elif cand2.exists():
            jobs_path = cand2
        else:
            jobs_path = cand1
    else:
        p = Path(file_path).resolve()
        if p.exists():
            jobs_path = p
        else:
            # Check if file_path was passed as suffix or partial name (e.g. 'run_1' or '1')
            sfx = file_path.lstrip("_")
            cand1 = Path(f"timed_out_jobs_{sfx}.json").resolve()
            cand2 = Path(f"timed_out_jobs_run_{sfx}.json").resolve()
            cand3 = Path(f"{file_path}.json").resolve()
            if cand1.exists():
                jobs_path = cand1
            elif cand2.exists():
                jobs_path = cand2
            elif cand3.exists():
                jobs_path = cand3
            else:
                jobs_path = p

    if not jobs_path.exists():
        console.print(f"[yellow]No timed-out jobs file found at '{jobs_path.name}'.[/yellow]")
        available = sorted(Path.cwd().glob("timed_out_jobs*.json"))
        if available:
            file_names = ", ".join(f.name for f in available)
            console.print(f"[dim]Available timed-out job files: {file_names}[/dim]")
        raise typer.Exit(0)

    console.print(f"\n[bold blue]Recovering timed-out jobs from:[/bold blue] [cyan]{jobs_path.name}[/cyan]")
    pipeline = UnifiedPipeline(dpi=dpi, timed_out_jobs_file=jobs_path)
    recovered = pipeline.recover_timed_out_jobs(jobs_file=jobs_path)
    if not recovered:
        console.print(f"[yellow]No pending timed-out jobs in '{jobs_path.name}' were ready for recovery.[/yellow]")
    else:
        for rec in recovered:
            console.print(
                f"   [green]✓ Recovered job {rec['job_id']} for {rec['pdf_name']}:[/green] "
                f"{rec['records_count']} record(s), {rec['manifest_count']} visual asset(s)."
            )
        console.print(f"[bold green]Successfully recovered {len(recovered)} job(s) in '{jobs_path.name}'![/bold green]")

