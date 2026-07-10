from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter

from vet_agent.ingestion.chunker import logical_key
from vet_agent.ingestion.models import Chunk, SectionType


class EvalCase(BaseModel):
    """One frozen retrieval-eval query with its metadata-derived ground truth."""

    query: str
    flow: str  # dose | contraindication | indication | other
    drug: str
    section: SectionType
    species: list[str]
    relevant_logical_keys: list[str]


_ADAPTER = TypeAdapter(list[EvalCase])


def load_eval_set(path: Path) -> list[EvalCase]:
    """Parse + validate a retrieval_eval.yaml file into EvalCase objects."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _ADAPTER.validate_python(data)


def derive_relevant_keys(
    chunks: list[Chunk], *, drug: str, section: SectionType, species: list[str]
) -> list[str]:
    """All logical_keys whose chunk matches (drug, section) and the species constraint.

    Species rule: if `species` is empty, any species matches; otherwise the chunk must
    share at least one species OR be a prose chunk tagged ["all"].
    """
    wanted = set(species)
    keys: list[str] = []
    for chunk in chunks:
        if chunk.drug_name.lower() != drug.lower():
            continue
        if chunk.section_type != section:
            continue
        if wanted and not (set(chunk.species) & wanted) and "all" not in chunk.species:
            continue
        keys.append(logical_key(chunk))
    return keys
