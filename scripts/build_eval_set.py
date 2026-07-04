"""Offline generation of the frozen retrieval eval set, with a human review gate.

Run by hand (NOT in CI):
    # 1) draft (needs VET_ANTHROPIC_API_KEY)
    uv run python scripts/build_eval_set.py generate --chunks data/ingest/chunks.json
    # 2) review/edit the query phrasings in data/eval/retrieval_eval.draft.yaml
    # 3) freeze the reviewed draft
    uv run python scripts/build_eval_set.py promote
"""

from pathlib import Path

import typer

from vet_agent.config import Settings
from vet_agent.eval.eval_set_builder import AnthropicQueryPhraser, build_eval_set, promote_eval_set, write_eval_set
from vet_agent.knowledge.loader import read_chunks

app = typer.Typer(help="Generate + promote the frozen retrieval eval set.")

DRAFT = Path("data/eval/retrieval_eval.draft.yaml")
FINAL = Path("data/eval/retrieval_eval.yaml")


@app.command()
def generate(
    chunks: Path = typer.Option(Path("data/ingest/chunks.json")),
    draft: Path = typer.Option(DRAFT),
    per_flow: int = typer.Option(25, help="Targets sampled per flow"),
    seed: int = typer.Option(0),
    other_fraction: float = typer.Option(
        0.1, help="Max share of exotic/food-animal species per flow (rest are dog/cat)"
    ),
) -> None:
    """Phrase queries with Claude and write a REVIEWABLE DRAFT (not the frozen set)."""
    settings = Settings()
    if settings.anthropic_api_key is None:
        typer.echo("Error: VET_ANTHROPIC_API_KEY is required to phrase queries.")
        raise typer.Exit(code=1)
    phraser = AnthropicQueryPhraser(
        settings.anthropic_api_key.get_secret_value(), settings.reasoning_model
    )
    cases = build_eval_set(
        read_chunks(chunks), phraser, per_flow=per_flow, seed=seed, other_fraction=other_fraction
    )
    write_eval_set(cases, draft)
    typer.echo(
        f"Wrote {len(cases)} DRAFT cases -> {draft}\n"
        "Review/edit the 'query' phrasings (labels are derived — leave them), "
        "then run: scripts/build_eval_set.py promote"
    )


@app.command()
def promote(
    draft: Path = typer.Option(DRAFT),
    out: Path = typer.Option(FINAL),
) -> None:
    """Validate the reviewed draft and freeze it to the committed eval set."""
    if not draft.is_file():
        typer.echo(f"Error: draft not found at {draft} (run 'generate' first).")
        raise typer.Exit(code=1)
    count = promote_eval_set(draft, out)
    typer.echo(f"Promoted {count} reviewed cases -> {out}")


if __name__ == "__main__":
    app()
