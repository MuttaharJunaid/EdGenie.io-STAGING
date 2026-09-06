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
from typing import Any, Dict, List, Optional, Tuple

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
    paper_type: Optional[str] = None,
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
    force_mcq: Optional[bool] = None
    if paper_type:
        pt_norm = paper_type.strip().lower()
        if pt_norm == "mcq":
            force_mcq = True
            console.print("Paper Type overridden by user: [bold yellow]MCQ[/bold yellow]")
        elif pt_norm == "essay":
            force_mcq = False
            console.print("Paper Type overridden by user: [bold yellow]ESSAY[/bold yellow]")
        else:
            console.print(f"[yellow]Warning: unrecognized paper type override '{paper_type}'. Valid values: 'essay', 'mcq'.[/yellow]")

    with console.status("[bold green]Extracting figures and diagrams..."):
        manifest = extractor.extract(pdf, visuals_dir, force_mcq=force_mcq)

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
    paper_type: Optional[str] = typer.Argument(None, help="Paper type override: 'essay' or 'mcq' (can be passed at end of command)"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Paper type override: 'essay' or 'mcq'"),
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
        paper_type=paper_type or type,
    )
    if result is None and not force and (pdf.parent / pdf.stem).exists():
        raise typer.Exit(code=0)


@app.command()
def extract_dir(
    directory_path: str = typer.Option(..., "--dir", "-d", help="Directory of PDF files"),
    paper_type: Optional[str] = typer.Argument(None, help="Paper type override: 'essay' or 'mcq' (can be passed at end of command)"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Paper type override: 'essay' or 'mcq'"),
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
    effective_type = paper_type or type

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
                paper_type=effective_type,
            )
            if res is not None:
                done += 1
        except Exception as e:
            console.print(f"[red]Failed {pdf.name}: {e}[/red]")

    console.print(f"\n[bold green]Done![/bold green] Processed {done}/{len(pdfs)} PDFs.")


# ── JSON Patching Helper ──────────────────────────────────────────────────────

def _patch_question_json(json_path: Path, manifest: List[Dict[str, Any]]) -> None:
    """
    Update the Stage 2 JSON file so each question's `image` field (and each
    part's `image` field) points to its extracted figures by question+part.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: could not read {json_path.name} for patching: {e}[/yellow]")
        return

    if not records or not manifest:
        return

    # Group manifest relative paths by (question, part)
    # part == None or "stem" → question-level image
    q_to_stem_files: Dict[int, List[str]] = {}
    q_part_to_files: Dict[Tuple[int, str], List[str]] = {}
    for entry in manifest:
        q = entry.get("question")
        if q is None:
            continue
        q_int = int(q)
        target_type = entry.get("target_type")
        target_label = entry.get("target_label")
        part = entry.get("part") or ""
        rp = entry.get("relative_path", "")
        if target_type == "question" or (target_type is None and part in ("question", "stem", "")):
            q_to_stem_files.setdefault(q_int, []).append(rp)
        else:
            clean_part = str(target_label if (target_type in ("part", "sub_part") and target_label) else re.sub(r"^(?:sub_)?part_", "", part)).strip()
            def _add_part_file(k: Tuple[int, str], file_path: str):
                flist = q_part_to_files.setdefault(k, [])
                if file_path not in flist:
                    flist.append(file_path)

            _add_part_file((q_int, clean_part), rp)
            norm_part = clean_part.strip("() ").lower()
            if norm_part:
                _add_part_file((q_int, norm_part), rp)
                _add_part_file((q_int, f"({norm_part})"), rp)
            m_comp = re.match(r"^([a-z])\s*\(?([ivx\d]+)\)?$", clean_part, re.IGNORECASE)
            if m_comp:
                p_l, sp_r = m_comp.group(1).lower(), m_comp.group(2).lower()
                _add_part_file((q_int, f"({sp_r})"), rp)
                _add_part_file((q_int, sp_r), rp)
                _add_part_file((q_int, f"{p_l}({sp_r})"), rp)
                _add_part_file((q_int, f"{p_l}{sp_r}"), rp)

    patched = 0
    for record in records:
        q_raw = record.get("QuestionNumber") or record.get("AnswerNumber") or ""
        m = re.search(r"(?:Q|q)?(\d+)(?:\([a-z]\))?$", str(q_raw).split("/")[-1].strip())
        if not m:
            continue
        q_num = int(m.group(1))

        # Patch question-level image (stem figures)
        stem_files = q_to_stem_files.get(q_num)
        if stem_files:
            record["image"] = stem_files if len(stem_files) > 1 else stem_files[0]
            patched += 1

        # Patch part-level and nested sub-part images recursively
        def _patch_tree_images(node: Dict[str, Any], is_top: bool = False, parent_lbl: str = "") -> int:
            label_key = "part_label" if is_top else "sub_part_label"
            plabel = str(node.get(label_key, "")).strip()
            cnt = 0
            if plabel:
                candidates = [
                    plabel,
                    plabel.strip("() ").lower(),
                    f"({plabel.strip('() ').lower()})",
                ]
                if parent_lbl:
                    clean_leaf = plabel.strip("() ").lower()
                    candidates.append(f"{parent_lbl}({clean_leaf})")
                    candidates.append(f"{parent_lbl}{clean_leaf}")

                part_files = None
                for cand in candidates:
                    if cand and (q_num, cand) in q_part_to_files:
                        part_files = q_part_to_files[(q_num, cand)]
                        break
                if part_files:
                    node["image"] = part_files if len(part_files) > 1 else part_files[0]
                    cnt += 1
            curr_lbl = plabel.strip("() ").lower() if is_top else parent_lbl
            for sp in node.get("sub_parts", []):
                cnt += _patch_tree_images(sp, is_top=False, parent_lbl=curr_lbl)
            return cnt

        for part_rec in record.get("Parts", []):
            patched += _patch_tree_images(part_rec, is_top=True)


    if patched:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        console.print(f"  Patched image fields in [green]{json_path.name}[/green] ({patched} entries).")



if __name__ == "__main__":
    app()
