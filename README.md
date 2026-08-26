# FluxFold

FluxFold is an agent memory system. It spends LLM cost on the write path and keeps the read path as pure vector search. Memories are organized into `subject`s: bounded units that are reviewed and split as they grow.

This repository is in early development. The current milestone is the experimental version: a Python library, dataset adapters, and a benchmark runner for `LongMemEval-S` and `LoCoMo_refined`. Public `add` / `search` usage, a CLI/TUI, and host connectors are not available yet.

## Status

Early development. Public APIs, the persistence format, and behavior may change without compatibility guarantees. Do not treat this package as production-ready.

## Capabilities

The experimental version is designed to provide:

- Memory extraction, batch subject linking, subject review, and subject split
- Public vector `search` over subjects and memories
- SQLite persistence with NumPy exact embedding scan
- Dataset adapters and a benchmark runner for `LongMemEval-S` and `LoCoMo_refined`

These are the implementation target. They are not implemented yet.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for Python versions, virtual environments, and lockfiles

## Development environment

From the repository root:

```bash
./scripts/setup-dev.sh
```

The script installs the Python version pinned in `.python-version`, syncs the project environment from `uv.lock`, and runs the quality checks below.

If uv and Python 3.12 are already available and you only need to refresh the environment:

```bash
uv sync
```

## Quality checks and build

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/fluxfold
uv run pytest
uv build
```

Always run these tools through `uv run`.

## Design

Project-level delivery form, toolchain, and evolution conditions: [`design.md`](design.md).

Memory Engine behavior, data model, and pipelines: [`design_detailed.md`](design_detailed.md).
