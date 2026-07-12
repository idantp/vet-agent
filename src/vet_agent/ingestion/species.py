import re

# Maps any surface token (singular/plural) to a canonical species name. Covers the
# species that appear as Doses sub-headers in Plumb's, including exotic/farm species.
_SPECIES_SYNONYMS: dict[str, str] = {
    "dog": "dog",
    "dogs": "dog",
    "cat": "cat",
    "cats": "cat",
    "horse": "horse",
    "horses": "horse",
    "equine": "horse",
    "ferret": "ferret",
    "ferrets": "ferret",
    "rabbit": "rabbit",
    "rabbits": "rabbit",
    "bird": "bird",
    "birds": "bird",
    "avian": "bird",
    "avians": "bird",
    "poultry": "poultry",
    "cattle": "cattle",
    "cow": "cattle",
    "cows": "cattle",
    "swine": "swine",
    "pig": "swine",
    "pigs": "swine",
    "sheep": "sheep",
    "goat": "goat",
    "goats": "goat",
    "ruminant": "ruminant",
    "ruminants": "ruminant",
    "camelid": "camelid",
    "camelids": "camelid",
    "llama": "llama",
    "llamas": "llama",
    "alpaca": "alpaca",
    "alpacas": "alpaca",
    "deer": "deer",
    "reptile": "reptile",
    "reptiles": "reptile",
    "turtle": "reptile",
    "turtles": "reptile",
    "tortoise": "reptile",
    "tortoises": "reptile",
    "amphibian": "amphibian",
    "amphibians": "amphibian",
    "fish": "fish",
    "hedgehog": "hedgehog",
    "hedgehogs": "hedgehog",
    "mammal": "small mammal",
    "mammals": "small mammal",
    "bee": "bee",
    "bees": "bee",
}

# Dose sub-headers that are NOT species (notes/warnings and parenteral routes); these
# must not start a new species group, or the doses below them get detached from their
# real species and mislabeled "unspecified".
_NON_SPECIES_HEADER_WORDS: frozenset[str] = frozenset(
    {
        "note",
        "notes",
        "warning",
        "warnings",
        "caution",
        "cautions",
        "im",
        "iv",
        "po",
        "sc",
        "sq",
        "it",
        "ip",
        "sl",
    }
)

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
    """True if the line is shaped like a Doses *species* sub-header.

    Allows species not in our vocabulary (so their doses become an 'unspecified' group
    rather than being misattributed to the preceding species), but excludes header-shaped
    lines that are notes/warnings or parenteral routes (``NOTES:``, ``IM:``) — those are
    content within the current species, not a new species boundary.
    """
    stripped = line.strip()
    if not _HEADER_RE.match(stripped):
        return False
    words = set(re.findall(r"[a-z]+", stripped.lower()))
    return bool(words) and not (words & _NON_SPECIES_HEADER_WORDS)


def detect_species_mentions(text: str) -> list[str]:
    """Best-effort canonical species mentioned anywhere in prose text."""
    return _canonical_tokens(text)


def canonical_species(text: str) -> str | None:
    """Canonicalize a single species string ("Dogs" -> "dog"); None if not exactly one."""
    tokens = _canonical_tokens(text)
    return tokens[0] if len(tokens) == 1 else None
