import json

from vet_agent.ingestion.models import Chunk, Monograph, Section, SectionType, TocEntry
from vet_agent.ingestion.report import build_parse_report, write_artifacts


def _monos() -> list[Monograph]:
    return [
        Monograph(
            drug_name="Metronidazole",
            book_page=873,
            sections=[Section(section_type=SectionType.DOSES, text="DOGS: 25 mg/kg")],
        ),
        Monograph(drug_name="Empty Drug", book_page=900, sections=[]),
    ]


def _toc() -> list[TocEntry]:
    return [
        TocEntry(drug_name="Metronidazole", book_page=873),
        TocEntry(drug_name="Empty Drug", book_page=900),
        TocEntry(drug_name="Lost Drug", book_page=950),
    ]


def test_build_parse_report_records_coverage_empty_and_missing():
    report = build_parse_report(_monos(), toc=_toc(), missing=[_toc()[2]])
    assert report.toc_entries == 3
    assert report.drugs_parsed == 2
    assert report.missing_headings == ["Lost Drug"]
    assert any(a["drug"] == "Empty Drug" for a in report.anomalies)


def test_write_artifacts_writes_files(tmp_path):
    monos = _monos()
    chunks = [
        Chunk(
            drug_name="Metronidazole",
            section_type=SectionType.DOSES,
            species=["dog"],
            book_page=875,
            text="DOGS: 25 mg/kg",
            ordinal=0,
        )
    ]
    report = build_parse_report(monos, toc=_toc(), missing=[])
    write_artifacts(monos, chunks, report, out_dir=tmp_path)

    report_data = json.loads((tmp_path / "parse_report.json").read_text())
    assert report_data["drugs_parsed"] == 2

    mono_data = json.loads((tmp_path / "monographs.json").read_text())
    assert mono_data[0]["drug_name"] == "Metronidazole"

    chunk_data = json.loads((tmp_path / "chunks.json").read_text())
    assert chunk_data[0]["drug_name"] == "Metronidazole"
    assert chunk_data[0]["species"] == ["dog"]
