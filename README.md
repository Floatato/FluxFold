# FluxFold

FluxFold is an experimental agent-memory library that spends LLM cost on the write path and keeps the read path as exact vector search. Memories are grouped into bounded `subject`s which can be reviewed and split as they grow.

The experimental version is a Python library with adapters and runners for LongMemEval-S and LoCoMo_refined. It is not production-ready and has no product CLI, TUI, connector, daemon, or network service.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- `git` and `curl`

## Setup

```bash
./scripts/setup-dev.sh
```

Already have the environment:

```bash
uv sync
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

```bash
mkdir -p data
git clone --depth 1 https://github.com/mem-eval-suite/LoCoMo_refined.git data/LoCoMo_refined
git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git data/LongMemEval
mkdir -p data/LongMemEval/data
curl -L --fail -o data/LongMemEval/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

cp .env.example .env
```

Fill the empty values in `.env` for the three generation groups (`FLUXFOLD_BUILD_*`,
`FLUXFOLD_ANSWER_*`, `FLUXFOLD_SCORE_*`) and the embedding provider, then:

```bash
uv run python -m benchmarks.scripts.build_sample \
  --dataset longmemeval \
  --run-dir runs/longmemeval-sample
uv run python -m benchmarks.scripts.answer_sample --run-dir runs/longmemeval-sample
uv run python -m benchmarks.scripts.score_sample --run-dir runs/longmemeval-sample

uv run python -m benchmarks.scripts.build_sample \
  --dataset locomo_refined \
  --select conv-26 \
  --run-dir runs/locomo-sample
uv run python -m benchmarks.scripts.answer_sample --run-dir runs/locomo-sample
uv run python -m benchmarks.scripts.score_sample --run-dir runs/locomo-sample
```

Replace `_sample` with `_full` for complete datasets. Optional `--config` TOML overrides go under `[fluxfold]`.

## Design

- [Project design](design.md)
- [Experimental Memory Engine design](design_detailed.md)
- [Design background and unverified assumptions](background.md)
