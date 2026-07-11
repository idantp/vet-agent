from decimal import Decimal
from types import SimpleNamespace

from vet_agent.tools.dose_extraction import AnthropicRegimenExtractor

RAW_REGIMENS = [
    {
        "indication": "giardiasis",
        "mg_per_kg_low": "25",
        "mg_per_kg_high": None,
        "route": "PO",
        "frequency": "q12h",
        "notes": "with fenbendazole",
    },
    {  # malformed: non-numeric dose -> must be dropped, not crash
        "indication": "bad",
        "mg_per_kg_low": "twenty-five",
        "mg_per_kg_high": None,
        "route": "PO",
        "frequency": "q12h",
        "notes": None,
    },
]


class _FakeMessages:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=self._content)


def _fake_client(content):
    return SimpleNamespace(messages=_FakeMessages(content))


def _tool_use_block(regimens):
    return SimpleNamespace(type="tool_use", input={"regimens": regimens})


def test_parses_forced_tool_use_and_drops_malformed_regimens():
    client = _fake_client([_tool_use_block(RAW_REGIMENS)])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    regimens = extractor.extract_regimens("passage text")
    assert len(regimens) == 1
    assert regimens[0].mg_per_kg_low == Decimal("25")
    assert regimens[0].notes == "with fenbendazole"


def test_request_forces_the_extraction_tool():
    client = _fake_client([_tool_use_block([])])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    extractor.extract_regimens("passage text")
    kwargs = client.messages.last_kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_regimens"}
    assert kwargs["tools"][0]["name"] == "record_regimens"
    assert kwargs["tools"][0]["strict"] is True
    assert "passage text" in kwargs["messages"][0]["content"]


def test_no_tool_use_block_returns_empty():
    client = _fake_client([SimpleNamespace(type="text", text="cannot comply")])
    extractor = AnthropicRegimenExtractor("claude-sonnet-5", client=client)
    assert extractor.extract_regimens("passage text") == []
