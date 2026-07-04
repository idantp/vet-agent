import random
from pathlib import Path
from typing import Protocol

import yaml

from vet_agent.eval.eval_set import EvalCase, derive_relevant_keys
from vet_agent.ingestion.models import Chunk, SectionType

# The question "flows" the eval set covers, each mapped to the monograph section that
# answers it. Chosen to mirror the real questions veterinarians ask at the point of care:
# dosing, indication, contraindications, pregnancy/nursing safety, drug interactions,
# side effects, monitoring, and administration. One question per target, single section.
FLOW_SECTIONS: dict[str, SectionType] = {
    "dose": SectionType.DOSES,
    "indication": SectionType.INDICATIONS,
    "contraindication": SectionType.CONTRAINDICATIONS,
    "reproductive_safety": SectionType.REPRODUCTIVE_SAFETY,
    "interaction": SectionType.DRUG_INTERACTIONS,
    "adverse_effects": SectionType.ADVERSE_EFFECTS,
    "monitoring": SectionType.MONITORING,
    "administration": SectionType.CLIENT_INFORMATION,
}

# Flow-matched few-shot examples that anchor the STYLE/register of a real clinician's
# question (companion + exotic + food-animal species, terse and full-sentence). They are
# for OTHER drugs and must not be reused as content — only as phrasing exemplars.
_FEWSHOT_BY_SECTION: dict[SectionType, list[str]] = {
    SectionType.DOSES: [
        "What is the oral cephalexin dose for dogs?",
        "What's the recommended dose of gabapentin for a 25 kg dog with osteoarthritis?",
        "Meloxicam dose for a rabbit?",
    ],
    SectionType.INDICATIONS: [
        "What's pimobendan used for in dogs?",
        "Is doxycycline recommended for respiratory infections in birds?",
    ],
    SectionType.CONTRAINDICATIONS: [
        "Can I still use carprofen in a dog with kidney disease?",
        "Should I avoid this drug in liver disease?",
    ],
    SectionType.REPRODUCTIVE_SAFETY: [
        "Is this safe to use in a pregnant dog?",
        "Can I give it to nursing puppies?",
    ],
    SectionType.DRUG_INTERACTIONS: [
        "Can prednisone be given together with an NSAID?",
        "Does amoxicillin interact with phenobarbital?",
    ],
    SectionType.ADVERSE_EFFECTS: [
        "What adverse effects should I warn the owner about?",
        "Does this commonly cause vomiting or diarrhea in cats?",
    ],
    SectionType.MONITORING: [
        "What bloodwork should I monitor on long-term therapy, and how often?",
        "Are liver enzymes expected to rise on this drug?",
    ],
    SectionType.CLIENT_INFORMATION: [
        "Should this be given with food?",
        "Can the capsules be opened or compounded into a flavored suspension?",
    ],
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
        examples = _FEWSHOT_BY_SECTION.get(section, [])
        example_block = ""
        if examples:
            joined = "\n".join(f"- {e}" for e in examples)
            example_block = (
                "Examples of the style and register we want (these are for OTHER "
                f"drugs — do not reuse their drug names or content):\n{joined}\n\n"
            )
        prompt = (
            "You are a practicing veterinarian with the patient in front of you. You "
            "already know the species, breed, weight, age, diagnosis, current "
            "medications, and organ/pregnancy status — you are looking up "
            f"{drug} in a drug handbook at the point of care.\n\n"
            f"{example_block}"
            f"Write ONE realistic question you would actually ask about {drug} — "
            f"specifically its {section.value.replace('_', ' ')} for {who}. Phrase it the "
            "way a clinician would type it (you may include brief clinical context such as "
            "an indication or a patient weight if it fits). Ask about only this one topic. "
            f"It must be answerable from this source text:\n\n{sample_text[:600]}\n\n"
            "Return only the question, no preamble."
        )
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[MessageParam(role="user", content=prompt)],
        )
        parts = [getattr(block, "text", "") for block in msg.content if block.type == "text"]
        return "".join(parts).strip()


def _stratified_sample(
    targets: list[tuple[str, tuple[str, ...]]], per_flow: int, rng: random.Random
) -> list[tuple[str, tuple[str, ...]]]:
    """Pick up to per_flow targets, spread across species so rarer species surface.

    Targets are grouped by their species signature; we shuffle within each group
    (seeded) and round-robin across groups in sorted key order. When a section carries
    only one species signature (e.g. prose tagged ['all']), this degenerates to a plain
    shuffled pick over drugs. Deterministic for a fixed seed.
    """
    groups: dict[tuple[str, ...], list[tuple[str, tuple[str, ...]]]] = {}
    for target in targets:
        groups.setdefault(target[1], []).append(target)
    ordered_keys = sorted(groups)
    for key in ordered_keys:
        rng.shuffle(groups[key])
    picked: list[tuple[str, tuple[str, ...]]] = []
    cursors = {key: 0 for key in ordered_keys}
    while len(picked) < per_flow:
        progressed = False
        for key in ordered_keys:
            if cursors[key] < len(groups[key]):
                picked.append(groups[key][cursors[key]])
                cursors[key] += 1
                progressed = True
                if len(picked) >= per_flow:
                    break
        if not progressed:
            break
    return picked


def build_eval_set(
    chunks: list[Chunk], phraser: QueryPhraser, *, per_flow: int, seed: int
) -> list[EvalCase]:
    """Sample targets per flow (species-stratified), derive labels, phrase each query."""
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    for flow, section in FLOW_SECTIONS.items():
        targets = sorted(
            {(c.drug_name, tuple(c.species)) for c in chunks if c.section_type == section}
        )
        for drug, species_tuple in _stratified_sample(targets, per_flow, rng):
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
    """Validate a human-reviewed draft and freeze it to the committed eval-set path."""
    from vet_agent.eval.eval_set import load_eval_set

    cases = load_eval_set(draft_path)
    write_eval_set(cases, final_path)
    return len(cases)
