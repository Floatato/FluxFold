# [AGENTS.md](http://AGENTS.md)

> **Scope: experimental version only.** This file governs the implementation of the FluxFold
> *experimental version* (benchmark-oriented Memory Engine). It will be rewritten before work on
> the first official version starts. Do not extend it to cover CLI, TUI, connectors, or any
> official-version concern.

## Project Overview

FluxFold is an agent memory system built from scratch. Its core bet: **push all LLM cost to the
write path, keep the read path pure vector search.** Memories are organized into `subject`s —
bounded, self-specializing organization units that are reviewed and split as they grow.

- Delivery form: a Python **library**. No HTTP/gRPC service, no daemon, no network listener.
- Distribution / import package / CLI name: all `fluxfold`. Source under `src/fluxfold/`.
- Python `>=3.12`, managed with `uv`, single distribution, `src` layout, Apache-2.0.



## Authoritative Documents


| Document                      | Role                                                                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/design_detailed.md` §1.1–1.4 | **The implementation source of truth** for the experimental version: data model, storage, config values, pipelines, structured-output shapes. |
| `docs/design.md`                   | Project-level delivery form, toolchain, quality baseline, evolution conditions.                                                               |
| `docs/background.md`               | Why each mechanism exists, and which assumptions are still unverified.                                                                        |


`docs/design_detailed.md` §2.x is **official-version** design — out of scope, do not implement.
When a document and the code disagree, the document wins; if the design is genuinely wrong,
change the document in the same commit as the code, never silently diverge.

## Experimental Scope

**In scope:** memory extraction, candidate recall, batched subject linking, subject review,
subject split, public `search`, SQLite persistence, embedding management, dataset adapters for
`LongMemEval-S` and `LoCoMo_refined`, benchmark runner, build logs, tests.

**Out of scope (do not build):** CLI, TUI, connectors, MCP server, durable inbox, buffers,
embedding-based situation boundary detection, `flush`, users/streams tables, ablation studies,
graph DB, vector DB, ANN index, read-path BM25, rerankers, query rewriting.
Write-side name disambiguation may use BM25 alongside vectors.

## Target Layout

```text
src/fluxfold/           # library core; must not import benchmark or adapter code
tests/                  # pytest, mirrors src/fluxfold/ structure
benchmarks/             # dataset adapters + runner; depends on the library, never the reverse
data/                   # cloned LongMemEval and LoCoMo_refined; gitignored
runs/                   # local benchmark artifacts; gitignored
scripts/setup-dev.sh    # one-time local environment bootstrap
docs/                   # design source of truth and design rationale
pyproject.toml uv.lock .python-version LICENSE README.md .env.example
AGENTS.md AGENTS_CH.md
```

Dependency direction is one-way: `benchmarks/` → `src/fluxfold/`. Dataset adapters only
normalize source records into episodes; they must not contain extraction, linking, review,
split, write, or search logic.

## Commands

```bash
./scripts/setup-dev.sh           # first-time environment bootstrap
uv sync                          # refresh the environment
uv run ruff format --check .     # formatting
uv run ruff check .              # lint
uv run mypy src/fluxfold         # types
uv run pytest                    # tests
uv build                         # wheel + sdist
```

Always run tools through `uv run`. All four checks must pass before a change is considered done.

## Core Pipeline

```text
dataset session
→ normalize + persist immutable episode (content_hash, source_sequence)
→ memory extraction + anchor resolution (0..N memories with named anchors, or no_valuable_memory)
→ prepare same-name anchor subjects and actual memory IDs
→ per-memory, per-anchor subject-name recall (active/retired top 3 plus same-name subject; batch deduplication)
→ batched Subject linking           (one agent loop per episode; up to 5 association_search calls)
→ atomic commit                     (anchors, memberships, memories, versions, provenance, embeddings, subjects, reactivation, links, completion)
→ Subject split / Subject review    (split first when both are due)
→ Subject summary refresh           (durable episode targets, up to 5 concurrent rewrites)
```



## Coding Rules

- Fail fast. Catch only what you can genuinely recover from; let unexpected exceptions crash.
- No defensive branches or fallbacks for conditions that cannot happen (including those the data
model already prevents). Validate at boundaries (dataset input, model provider output) only.
- Experimental-version changes are not compatibility work: do not keep unused APIs, schema fields,
config knobs, migration paths, shims, or feature flags just because an earlier draft used them.
- When a change retires a path, delete all code that will no longer be used. Keep the tree small;
no commented-out leftovers, unused imports, or dead helpers.
- Comments, docstrings, LLM prompts, and other in-code prose are English.
- No speculative abstractions. Three similar lines beat a premature helper.
- Public library API in `src/fluxfold/` must be fully type-annotated and pass `mypy`.
- Ruff is both formatter and linter, targeting `py312`.



## Documentation Duties

- Behavior changes go into `docs/design_detailed.md` (Memory Engine) or `docs/design.md` (project-level)
in the same change. A code change without the matching doc update is incomplete.
- Describe only the resulting current design. Delete superseded descriptions instead of
recording that something is no longer used; git holds the history.
- `AGENTS.md` and `AGENTS_CH.md` must always be updated together and stay equivalent.
