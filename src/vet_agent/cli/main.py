import logging
from pathlib import Path

import typer

from vet_agent.ingestion.pdf_reader import extract_pages
from vet_agent.ingestion.pipeline import run_ingestion
from vet_agent.ingestion.report import write_artifacts

app = typer.Typer(help="Vet-Agent CLI")


@app.callback()
def main() -> None:
    """Vet-Agent: agentic RAG over Plumb's Veterinary Drug Handbook."""
    # A no-op group callback keeps `ingest` registered as a named subcommand
    # (so `--help` lists it) instead of Typer collapsing a single-command app.


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="Path to the Plumb's handbook PDF"),  # noqa: B008
    toc_start: int = typer.Option(19, help="First page index (0-based) of the TOC"),  # noqa: B008
    toc_end: int = typer.Option(27, help="Last page index (0-based) of the TOC"),  # noqa: B008
    out_dir: Path = typer.Option(Path("data/ingest"), help="Output directory"),  # noqa: B008
    max_missing: int = typer.Option(  # noqa: B008
        0,
        help="Max TOC drugs allowed with no located heading before the run fails. "
        "Defaults to 0 (zero tolerance) — a medical reference must not lose drugs. "
        "Raise only for local iteration while fixing the parser.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DEBUG logs"),  # noqa: B008
) -> None:
    """Parse the PDF into monographs + chunks and write artifacts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not pdf.exists():
        typer.echo(f"Error: PDF not found at {pdf}")
        raise typer.Exit(code=1)

    typer.echo(f"Reading {pdf} ...")
    pages = extract_pages(pdf)
    monographs, chunks, report = run_ingestion(pages, toc_page_range=(toc_start, toc_end))

    # Always write artifacts first, so parse_report.json (with missing_headings) is
    # available for inspection even when the coverage gate below fails the run.
    write_artifacts(monographs, report, out_dir=out_dir)
    typer.echo(
        f"Parsed {report.drugs_parsed}/{report.toc_entries} TOC drugs, {len(chunks)} chunks, "
        f"{len(report.missing_headings)} missing headings, {len(report.anomalies)} anomalies. "
        f"Artifacts -> {out_dir}"
    )

    # Coverage gate: fail loudly if too many TOC drugs could not be located.
    if len(report.missing_headings) > max_missing:
        typer.echo(
            f"Error: {len(report.missing_headings)} TOC drugs had no located heading "
            f"(allowed: {max_missing}). See 'missing_headings' in "
            f"{out_dir / 'parse_report.json'}. Fix the parser, or pass --max-missing "
            f"to proceed during local iteration."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
