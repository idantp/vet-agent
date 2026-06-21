import re

# Maps any surface token (singular/plural) to a canonical species name.
_SPECIES_SYNONYMS: dict[str, str] = {
    "dog": "dog",
    "dogs": "dog",
    "cat": "cat",
    "cats": "cat",
    "horse": "horse",
    "horses": "horse",
    "ferret": "ferret",
    "ferrets": "ferret",
    "rabbit": "rabbit",
    "rabbits": "rabbit",
    "bird": "bird",
    "birds": "bird",
    "cattle": "cattle",
    "cow": "cattle",
    "cows": "cattle",
    "swine": "swine",
    "pig": "swine",
    "pigs": "swine",
    "sheep": "sheep",
    "goat": "goat",
    "goats": "goat",
}

# A dose sub-header is short, mostly uppercase, and ends with a colon.
_HEADER_RE = re.compile(r"^[A-Z][A-Z &/]{0,40}:$")


def _canonical_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    found = {_SPECIES_SYNONYMS[t] for t in tokens if t in _SPECIES_SYNONYMS}
    return sorted(found)


def parse_species_header(line: str) -> list[str]:
    """Return canonical species for a Doses sub-header line, else []."""
    stripped = line.strip()
    if not _HEADER_RE.match(stripped):
        return []
    return _canonical_tokens(stripped)


def is_species_header(line: str) -> bool:
    """True if the line is shaped like a Doses species sub-header, even when the
    species token is not in our vocabulary (e.g. ``AVIANS:``, ``RUMINANTS:``)."""
    return bool(_HEADER_RE.match(line.strip()))


def detect_species_mentions(text: str) -> list[str]:
    """Best-effort canonical species mentioned anywhere in prose text."""
    return _canonical_tokens(text)
