.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	uv run --no-sync python -c "from pathlib import Path; import shutil; roots=[Path('src'), Path('tests')]; caches=[Path('__pycache__'), Path('.mypy_cache'), Path('.pytest_cache')]; caches.extend(p for root in roots for p in root.rglob('__pycache__')); [shutil.rmtree(p) for p in caches if p.is_dir() and p.resolve().is_relative_to(Path.cwd().resolve())]"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
