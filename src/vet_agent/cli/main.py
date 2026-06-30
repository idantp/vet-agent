import logging
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from vet_agent.config import Settings
from vet_agent.eval.benchmark import ModelScore, benchmark_model, write_scorecard
from vet_agent.eval.eval_set import load_eval_set
from vet_agent.ingestion.models import SectionType
from vet_agent.ingestion.pdf_reader import extract_pages
from vet_agent.ingestion.pipeline import run_ingestion
from vet_agent.ingestion.report import write_artifacts
from vet_agent.knowledge.embedders import MODEL_REGISTRY, get_embedder
from vet_agent.knowledge.interfaces import Reranker
from vet_agent.knowledge.loader import load_chunks, read_chunks
from vet_agent.knowledge.retrieval import Retriever
from vet_agent.knowledge.vector_store import QdrantVectorStore, collection_name

app = typer.Typer(help="Vet-Agent CLI")


@app.callback()
def main() -> None:
    """Vet-Agent: agentic RAG over Plumb's Veterinary Drug Handbook."""
    # A no-op group callback keeps `ingest` registered as a named subcommand
    # (so `--help` lists it) instead of Typer collapsing a single-command app.


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="Path to the Plumb's handbook PDF"),  # noqa: B008
    toc_start: int = typer.Option(17, help="First page index (0-based) of the TOC"),  # noqa: B008
    toc_end: int = typer.Option(23, help="Last page index (0-based) of the TOC"),  # noqa: B008
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
        force=True,
    )
    if toc_start > toc_end:
        typer.echo(f"Error: --toc-start ({toc_start}) must not exceed --toc-end ({toc_end})")
        raise typer.Exit(code=1)
    if not pdf.is_file():
        typer.echo(f"Error: PDF not found at {pdf}")
        raise typer.Exit(code=1)

    typer.echo(f"Reading {pdf} ...")
    try:
        pages = extract_pages(pdf)
    except Exception as exc:  # pypdf raises several read errors for non-PDF/corrupt input
        typer.echo(f"Error: could not read PDF {pdf}: {exc}")
        raise typer.Exit(code=1) from exc
    monographs, chunks, report = run_ingestion(pages, toc_page_range=(toc_start, toc_end))

    # Always write artifacts first, so parse_report.json (with missing_headings) is
    # available for inspection even when the coverage gate below fails the run.
    write_artifacts(monographs, chunks, report, out_dir=out_dir)
    typer.echo(
        f"Parsed {report.drugs_parsed}/{report.toc_entries} TOC drugs, {len(chunks)} chunks, "
        f"{len(report.missing_headings)} missing headings, {len(report.anomalies)} anomalies. "
        f"Artifacts -> {out_dir}"
    )

    # Empty TOC means the page range is wrong; fail rather than silently writing
    # empty artifacts with a passing (0 > max_missing) gate.
    if report.toc_entries == 0:
        typer.echo(
            "Error: no TOC entries were parsed — check the --toc-start/--toc-end page range."
        )
        raise typer.Exit(code=1)

    # Coverage gate: fail loudly if too many TOC drugs could not be located.
    if len(report.missing_headings) > max_missing:
        typer.echo(
            f"Error: {len(report.missing_headings)} TOC drugs had no located heading "
            f"(allowed: {max_missing}). See 'missing_headings' in "
            f"{out_dir / 'parse_report.json'}. Fix the parser, or pass --max-missing "
            f"to proceed during local iteration."
        )
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
    eval_set: Path = typer.Option(Path("data/eval/retrieval_eval.yaml")),  # noqa: B008
    models: str = typer.Option("medembed-base,bge-base,qwen3-0.6b"),  # noqa: B008
    ks: str = typer.Option("1,3,5,10", help="Comma-separated k cutoffs"),  # noqa: B008
    cache_dir: Path = typer.Option(Path("data/embeddings")),  # noqa: B008
    out_dir: Path = typer.Option(Path("data/eval")),  # noqa: B008
) -> None:
    """Benchmark candidate embedders in-memory and write a scorecard."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    if not eval_set.is_file():
        typer.echo(f"Error: eval set not found at {eval_set}")
        raise typer.Exit(code=1)
    k_values = [int(x) for x in ks.split(",")]
    parsed = read_chunks(chunks)
    cases = load_eval_set(eval_set)
    scores: list[ModelScore] = []
    for key in [m.strip() for m in models.split(",")]:
        if key not in MODEL_REGISTRY:
            typer.echo(f"Error: unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}")
            raise typer.Exit(code=1)
        typer.echo(f"Benchmarking {key} ...")
        scores.append(benchmark_model(key, get_embedder(key), parsed, cases, k_values, cache_dir))
    write_scorecard(scores, k_values, out_dir)
    typer.echo(f"Scorecard -> {out_dir / 'benchmark_scorecard.md'}")


@app.command()
def load(
    chunks: Path = typer.Argument(Path("data/ingest/chunks.json")),  # noqa: B008
    model: str = typer.Option("", help="Override the configured embedding model"),  # noqa: B008
    no_prune: bool = typer.Option(False, "--no-prune"),  # noqa: B008
) -> None:
    """Idempotently embed + load chunks into Qdrant."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    settings = Settings()
    model_key = model or settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, collection_name(settings.qdrant_collection_prefix, model_key))
    report = load_chunks(
        read_chunks(chunks),
        embedder,
        store,
        prune=not no_prune,
        batch_size=settings.embedding_batch_size,
    )
    typer.echo(
        f"Loaded into '{collection_name(settings.qdrant_collection_prefix, model_key)}': "
        f"upserted={report.upserted} skipped={report.skipped} pruned={report.pruned}"
    )


@app.command()
def retrieve(
    query: str = typer.Argument(...),  # noqa: B008
    drug: str = typer.Option("", help="Filter by drug name"),  # noqa: B008
    section: str = typer.Option("", help="Filter by section_type, e.g. doses"),  # noqa: B008
    species: str = typer.Option("", help="Filter by species, e.g. dog"),  # noqa: B008
    top_k: int = typer.Option(5),  # noqa: B008
    rerank: bool = typer.Option(False, "--rerank"),  # noqa: B008
) -> None:
    """Filtered semantic search against the loaded Qdrant collection."""
    if not query.strip():
        typer.echo("Error: query must not be blank.")
        raise typer.Exit(code=1)
    settings = Settings()
    model_key = settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, collection_name(settings.qdrant_collection_prefix, model_key))
    reranker: Reranker | None = None
    if rerank:
        from vet_agent.knowledge.rerankers import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = Retriever(embedder, store, reranker=reranker)
    section_type = SectionType(section) if section else None
    hits = retriever.retrieve(
        query,
        drug=drug or None,
        section=section_type,
        species=species or None,
        top_k=top_k,
        rerank=rerank,
    )
    if not hits:
        typer.echo("No results.")
        return
    for h in hits:
        score = f"{h.score:.3f}" if h.score is not None else "n/a"
        typer.echo(f"[{score}] {h.drug_name} / {h.section_type.value} / p.{h.book_page}")
        typer.echo(f"    {h.text[:200]}")


if __name__ == "__main__":
    app()
