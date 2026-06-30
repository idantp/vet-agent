import random
from pathlib import Path
from typing import Protocol

import yaml

from vet_agent.eval.eval_set import EvalCase, derive_relevant_keys
from vet_agent.ingestion.models import Chunk, SectionType

FLOW_SECTIONS: dict[str, SectionType] = {
    "dose": SectionType.DOSES,
    "contraindication": SectionType.CONTRAINDICATIONS,
    "indication": SectionType.INDICATIONS,
}


class QueryPhraser(Protocol):
    def phrase(
        self, drug: str, section: SectionType, species: list[str], sample_text: str
    ) -> str: ...


class AnthropicQueryPhraser:
    """Phrases a natural-language vet question for a (drug, section, species) target."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic  # lazy: anthropic is a dev-only dependency

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def phrase(self, drug: str, section: SectionType, species: list[str], sample_text: str) -> str:
        from anthropic.types import MessageParam

        who = " and ".join(species) if species else "an animal"
        prompt = (
            f"Write ONE natural question a veterinarian would ask about the drug {drug}, "
            f"specifically its {section.value.replace('_', ' ')} for {who}. "
            f"Base it on this source text:\n\n{sample_text[:600]}\n\n"
            "Return only the question, no preamble."
        )
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[MessageParam(role="user", content=prompt)],
        )
        parts = [getattr(block, "text", "") for block in msg.content if block.type == "text"]
        return "".join(parts).strip()


def build_eval_set(
    chunks: list[Chunk], phraser: QueryPhraser, *, per_flow: int, seed: int
) -> list[EvalCase]:
    """Sample targets per flow, derive labels, and phrase a query for each (deterministic)."""
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    for flow, section in FLOW_SECTIONS.items():
        targets = sorted(
            {(c.drug_name, tuple(c.species)) for c in chunks if c.section_type == section}
        )
        rng.shuffle(targets)
        for drug, species_tuple in targets[:per_flow]:
            species = list(species_tuple)
            sample = next(
                (c.text for c in chunks if c.drug_name == drug and c.section_type == section),
                "",
            )
            keys = derive_relevant_keys(chunks, drug=drug, section=section, species=species)
            if not keys:
                continue
            cases.append(
                EvalCase(
                    query=phraser.phrase(drug, section, species, sample),
                    flow=flow,
                    drug=drug,
                    section=section,
                    species=species,
                    relevant_logical_keys=keys,
                )
            )
    return cases


def write_eval_set(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump(mode="json") for c in cases]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def promote_eval_set(draft_path: Path, final_path: Path) -> int:
    """Validate a human-reviewed draft and freeze it to the committed eval-set path.

    Re-parses the draft through load_eval_set (a malformed hand-edit fails loudly here,
    not later in the benchmark) and re-serializes canonically. Returns the case count.
    """
    from vet_agent.eval.eval_set import load_eval_set

    cases = load_eval_set(draft_path)
    write_eval_set(cases, final_path)
    return len(cases)
