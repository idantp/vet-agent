"""Deterministic test double for the RegimenExtractor LLM seam."""

from vet_agent.tools.dose_extraction import ExtractedRegimen


class FakeRegimenExtractor:
    """Returns a canned regimen list, ignoring the passage text."""

    def __init__(self, regimens: list[ExtractedRegimen]) -> None:
        self._regimens = regimens

    def extract_regimens(self, passage_text: str) -> list[ExtractedRegimen]:
        return list(self._regimens)
