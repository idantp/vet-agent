import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from vet_agent.config import Settings
from vet_agent.eval.benchmark import ModelScore, benchmark_model, embed_corpus, write_scorecard
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
from vet_agent.tools.dose_extraction import (
    AnthropicRegimenExtractor,
    ExtractDoseRule,
    ExtractDoseRuleInput,
)
from vet_agent.tools.dose_math import CalculateDoseInput, calculate_dose
from vet_agent.tools.drug_index import DrugIndex
from vet_agent.tools.models import DoseRule, DoseRuleSet, NeedsClarification
from vet_agent.tools.retrieve import RetrieveMonograph, RetrieveMonographInput

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
def embed(
    models: str = typer.Option(
        "medembed-base",
        help=f"Comma-separated model keys ({', '.join(sorted(MODEL_REGISTRY))})",
    ),  # noqa: B008
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
    cache_dir: Path = typer.Option(Path("data/embeddings")),  # noqa: B008
) -> None:
    """Embed the corpus for one or more models and cache the vectors (the slow step).

    Run this per model to spread the load (e.g. one at a time, letting the machine cool
    between); `benchmark` then reuses the cache and runs fast.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    parsed = read_chunks(chunks)
    for key in [m.strip() for m in models.split(",")]:
        if key not in MODEL_REGISTRY:
            typer.echo(f"Error: unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}")
            raise typer.Exit(code=1)
        typer.echo(f"Embedding corpus with {key} ...")
        embed_corpus(get_embedder(key), parsed, cache_dir)
    typer.echo("Done embedding. Run 'vet-agent benchmark' to score (reuses the cache).")


@app.command()
def benchmark(
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
    eval_set: Path = typer.Option(Path("data/eval/retrieval_eval.yaml")),  # noqa: B008
    models: str = typer.Option("medembed-base,bge-base"),  # noqa: B008
    ks: str = typer.Option("1,3,5,10", help="Comma-separated k cutoffs"),  # noqa: B008
    cache_dir: Path = typer.Option(Path("data/embeddings")),  # noqa: B008
    out_dir: Path = typer.Option(Path("data/eval")),  # noqa: B008
) -> None:
    """Benchmark candidate embedders in-memory and write a scorecard.

    Reuses vectors cached by a prior `embed`/`benchmark`; embeds (slow) only on a miss.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
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


@app.command()
def dose(
    question: str = typer.Argument(..., help="Natural-language dose question"),  # noqa: B008
    drug: str = typer.Option(..., help="Drug name (resolved against the monographs)"),  # noqa: B008
    species: str = typer.Option(..., help="Patient species, e.g. dog"),  # noqa: B008
    weight: str = typer.Option(..., help="Patient weight (number)"),  # noqa: B008
    weight_unit: str = typer.Option("kg", help="kg or lb"),  # noqa: B008
    indication: str = typer.Option("", help="Indication hint, e.g. giardia"),  # noqa: B008
    all_regimens: bool = typer.Option(  # noqa: B008
        False, "--all-regimens", help="List every grounded regimen instead of picking one"
    ),
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),  # noqa: B008
) -> None:
    """Phase 3 demo: retrieve -> extract_dose_rule -> calculate_dose, with citations."""
    if not chunks.is_file():
        typer.echo(f"Error: chunks file not found at {chunks}")
        raise typer.Exit(code=1)
    try:
        weight_value = Decimal(weight)
    except InvalidOperation as exc:
        typer.echo(f"Error: weight must be a number, got '{weight}'")
        raise typer.Exit(code=1) from exc
    if weight_unit not in ("kg", "lb"):
        typer.echo(f"Error: weight-unit must be kg or lb, got '{weight_unit}'")
        raise typer.Exit(code=1)
    settings = Settings()
    if settings.anthropic_api_key is None:
        typer.echo("Error: VET_ANTHROPIC_API_KEY is required for dose extraction.")
        raise typer.Exit(code=1)

    model_key = settings.embedding_model
    embedder = get_embedder(model_key)
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantVectorStore(client, collection_name(settings.qdrant_collection_prefix, model_key))
    drug_index = DrugIndex.from_chunks(chunks)

    retrieve_tool = RetrieveMonograph(Retriever(embedder, store), drug_index)
    retrieved = retrieve_tool(
        RetrieveMonographInput(
            query=question, drug=drug, section=SectionType.DOSES, species=species
        )
    )
    if retrieved.kind == "drug_not_found":
        hint = (
            f" Did you mean: {', '.join(retrieved.suggestions)}?" if retrieved.suggestions else ""
        )
        typer.echo(f"Drug not found: '{retrieved.query}'.{hint}")
        raise typer.Exit(code=1)
    if retrieved.kind == "no_passages_found":
        typer.echo(f"No dose passages found for filters {retrieved.filters}.")
        raise typer.Exit(code=1)

    passage = retrieved.passages[0]
    typer.echo(f"Passage: {passage.drug_name} / doses / p.{passage.book_page}")

    extractor = AnthropicRegimenExtractor(
        settings.reasoning_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    extracted = ExtractDoseRule(extractor)(
        ExtractDoseRuleInput(
            passage=passage, indication=indication or None, all_regimens=all_regimens
        )
    )

    if isinstance(extracted, NeedsClarification):
        typer.echo(f"Needs clarification: {extracted.reason}")
        for c in extracted.candidates:
            typer.echo(f"  - {_describe_rule(c)}")
        return
    if isinstance(extracted, DoseRuleSet):
        typer.echo(f"{len(extracted.rules)} grounded regimen(s):")
        for rule in extracted.rules:
            typer.echo(f"  - {_describe_rule(rule)}")
        return

    result = calculate_dose(
        CalculateDoseInput(weight=weight_value, weight_unit=weight_unit, rule=extracted)  # type: ignore[arg-type]
    )
    dose_range = f"{result.dose_mg_low} mg"
    if result.dose_mg_high is not None:
        dose_range += f" - {result.dose_mg_high} mg"
    typer.echo(
        f"Dose: {dose_range} {result.route} {result.frequency} "
        f"({result.rule.mg_per_kg_low} mg/kg x {result.weight_kg} kg)"
    )
    typer.echo(f"Indication: {result.indication}")
    if result.notes:
        typer.echo(f"Notes: {result.notes}")
    typer.echo(
        f"Source: {result.drug_name}, Doses, p.{result.rule.book_page} "
        f"[{result.rule.source_logical_key}]"
    )
    typer.echo("Decision support only - consult a licensed veterinarian.")


def _describe_rule(rule: DoseRule) -> str:
    dose_str = f"{rule.mg_per_kg_low}"
    if rule.mg_per_kg_high is not None:
        dose_str += f"-{rule.mg_per_kg_high}"
    text = f"{rule.indication}: {dose_str} mg/kg {rule.route} {rule.frequency}"
    if rule.notes:
        text += f" ({rule.notes})"
    return text


if __name__ == "__main__":
    app()
