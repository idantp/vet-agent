from typer.testing import CliRunner

from vet_agent.cli.main import app

runner = CliRunner()


def test_ingest_help_lists_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout


def test_ingest_requires_existing_pdf(tmp_path):
    missing = tmp_path / "nope.pdf"
    result = runner.invoke(app, ["ingest", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_ingest_rejects_inverted_toc_range():
    result = runner.invoke(app, ["ingest", "x.pdf", "--toc-start", "5", "--toc-end", "2"])
    assert result.exit_code != 0
    assert "toc-start" in result.stdout.lower()


def test_ingest_reports_clean_error_for_non_pdf_file(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not a PDF")
    result = runner.invoke(app, ["ingest", str(bad)])
    assert result.exit_code != 0
    assert "could not read pdf" in result.stdout.lower()
