.PHONY: install lint typecheck test check

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
