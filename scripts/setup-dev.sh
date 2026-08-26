#!/usr/bin/env bash
# One-time bootstrap for the FluxFold development environment.
# Day-to-day work still uses `uv run`; this script is not a task runner.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    cat <<'EOF'
uv is required but was not found on PATH.

Install uv from https://docs.astral.sh/uv/getting-started/installation/
Example:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
    exit 1
fi

echo "==> Using $(uv --version)"

PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
echo "==> Installing Python ${PYTHON_VERSION} if needed"
uv python install "${PYTHON_VERSION}"

if [[ -f uv.lock ]]; then
    echo "==> Syncing the project environment from uv.lock"
    uv sync --frozen
else
    echo "==> uv.lock is missing; resolving dependencies and syncing"
    uv sync
fi

echo "==> Running quality checks"
uv run ruff format --check .
uv run ruff check .
uv run mypy src/fluxfold
uv run pytest

echo "==> Building wheel and sdist"
uv build

echo
echo "Development environment is ready."
echo "Use uv run <command> for subsequent work."
