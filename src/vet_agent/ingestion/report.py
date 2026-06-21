import json
from pathlib import Path

from vet_agent.ingestion.models import Monograph, ParseReport, TocEntry


def build_parse_report(
    monographs: list[Monograph], toc: list[TocEntry], missing: list[TocEntry]
) -> ParseReport:
    """Summarize an ingestion run: TOC coverage, unlocated headings, empty monographs."""
    anomalies: list[dict[str, str]] = [
        {"drug": m.drug_name, "issue": "no sections parsed"} for m in monographs if not m.sections
    ]
    return ParseReport(
        toc_entries=len(toc),
        drugs_parsed=len(monographs),
        missing_headings=[e.drug_name for e in missing],
        anomalies=anomalies,
    )


def write_artifacts(monographs: list[Monograph], report: ParseReport, out_dir: Path) -> None:
    """Write monographs.json and parse_report.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: monograph text contains non-ASCII (e.g. μ, °, ≈) and the JSON
    # may carry it raw, so don't depend on the platform locale encoding (e.g. in Docker).
    (out_dir / "monographs.json").write_text(
        json.dumps([m.model_dump() for m in monographs], indent=2), encoding="utf-8"
    )
    (out_dir / "parse_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
