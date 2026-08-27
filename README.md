# FluxFold

FluxFold is an experimental agent-memory library that spends LLM cost on the write path and keeps the read path as exact vector search. Memories are grouped into bounded `subject`s which can be reviewed and split as they grow.

The experimental version implements memory extraction, batch subject linking, optional association search, subject review and split, SQLite persistence, embedding-model rebuilds, and structured `search`. It also includes benchmark adapters and staged runners for LongMemEval-S and LoCoMo_refined. It is not production-ready and does not include a product CLI, TUI, connector, daemon, or network service.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

```bash
./scripts/setup-dev.sh
```

For an existing environment, run `uv sync`. The standard checks are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/fluxfold
uv run pytest
uv build
```

## Library usage

`FluxFold` is asynchronous. Applications provide one generation provider and one embedding provider; OpenAI-compatible adapters are included.

```python
import asyncio

from fluxfold import (
    EmbeddingModelInfo,
    EpisodeBlock,
    FluxFold,
    NormalizedEpisode,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
    Role,
)


async def main() -> None:
    generation = OpenAICompatibleGenerationProvider(
        model="gpt-4.1-mini",
        api_key="...",
    )
    embedding = OpenAICompatibleEmbeddingProvider(
        model_info=EmbeddingModelInfo(
            provider="openai-compatible",
            model="text-embedding-3-small",
            revision="unspecified",
            dimension=1536,
        ),
        api_key="...",
    )
    async with await FluxFold.open(
        db_path="fluxfold.sqlite3",
        generation_provider=generation,
        embedding_provider=embedding,
    ) as engine:
        space = await engine.create_or_open_space("example")
        await engine.add(
            space.memory_space_id,
            NormalizedEpisode(
                source_type="example",
                source_key="session-1",
                source_sequence=0,
                blocks=(
                    EpisodeBlock("message-1", 0, Role.USER, "I prefer window seats."),
                ),
            ),
        )
        result = await engine.search(space.memory_space_id, "What seat do I prefer?")
        print(result.render())


asyncio.run(main())
```

## Benchmarks

Copy the variable names from `.env.example` into your shell environment. The generation variables configure all write stages, QA answering, and judging; embedding variables are separate. Dataset files are supplied as local paths and are never downloaded automatically.

Each benchmark is split into build, answer, and score stages. A LongMemEval-S stratified sample selects one full instance from each of its seven categories by default:

```bash
uv run python -m benchmarks.scripts.build_sample \
  --dataset longmemeval \
  --data-path /data/longmemeval_s_cleaned.json \
  --run-dir runs/longmemeval-sample
uv run python -m benchmarks.scripts.answer_sample --run-dir runs/longmemeval-sample
uv run python -m benchmarks.scripts.score_sample --run-dir runs/longmemeval-sample
```

LoCoMo_refined sample mode requires one conversation ID or zero-based conversation index and processes all its sessions and questions:

```bash
uv run python -m benchmarks.scripts.build_sample \
  --dataset locomo_refined \
  --conversations-path /data/conversations.jsonl \
  --questions-path /data/questions.jsonl \
  --select conv-26 \
  --run-dir runs/locomo-sample
uv run python -m benchmarks.scripts.answer_sample --run-dir runs/locomo-sample
uv run python -m benchmarks.scripts.score_sample --run-dir runs/locomo-sample
```

Replace `_sample` with `_full` for complete datasets; full build scripts do not accept `--select`. Optional policy overrides are read from a TOML file passed as `--config`, with values under `[fluxfold]`.

Each run directory contains the SQLite database, immutable manifest, checkpoint, JSONL event and prediction files, complete Markdown build audit, search evidence, per-question scores, and JSON/Markdown summaries. Re-running a stage uses or validates the existing manifest; use a new run directory for a distinct run or configuration.

## Design

- [Project design](design.md)
- [Experimental Memory Engine design](design_detailed.md)
- [Design background and unverified assumptions](background.md)
