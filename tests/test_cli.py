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
