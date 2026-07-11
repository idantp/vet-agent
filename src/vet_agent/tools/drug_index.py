import difflib
from pathlib import Path

from pydantic import BaseModel

from vet_agent.tools.models import DrugNotFound

# High cutoff: a fuzzy hit is used to *filter retrieval*, so it must be near-certain.
# Suggestions use a lower cutoff — they are shown to the user, never acted on.
_MATCH_CUTOFF = 0.85
_SUGGESTION_CUTOFF = 0.6


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


class ResolvedDrug(BaseModel):
    """Internal resolution result — consumed by tools, not part of the result unions."""

    canonical: str
    exact: bool


class DrugIndex:
    """Resolves free-form drug names against the canonical monograph names."""

    def __init__(self, names: list[str]) -> None:
        self._by_norm = {_norm(n): n for n in names}

    @classmethod
    def from_chunks(cls, path: Path) -> "DrugIndex":
        from vet_agent.knowledge.loader import read_chunks  # lazy: avoids qdrant import cost

        return cls(sorted({c.drug_name for c in read_chunks(path)}))

    def resolve(self, query: str) -> ResolvedDrug | DrugNotFound:
        key = _norm(query)
        if key in self._by_norm:
            return ResolvedDrug(canonical=self._by_norm[key], exact=True)
        close = difflib.get_close_matches(key, list(self._by_norm), n=1, cutoff=_MATCH_CUTOFF)
        if close:
            return ResolvedDrug(canonical=self._by_norm[close[0]], exact=False)
        near = difflib.get_close_matches(key, list(self._by_norm), n=3, cutoff=_SUGGESTION_CUTOFF)
        return DrugNotFound(query=query, suggestions=[self._by_norm[n] for n in near])
