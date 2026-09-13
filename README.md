# FluxFold

FluxFold is an experimental agent-memory library that spends LLM cost on the write path and keeps the read path as exact vector search. Memories carry named base `anchors` and are grouped into bounded `subject`s which can be reviewed and split as they grow. Linking recalls up to three subjects per memory and anchor, plus its same-name container, and assigns each anchor to the finest fitting subject before adding contextual links. Subject names are unique across active and retired objects; selected retired subjects are reactivated. Ambiguous anchor names receive a conditional LLM resolution pass. Public search excludes retired and empty subjects before selecting its top results.

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

Current experimental databases use schema version 6. Builds from earlier schemas need a new database; the library rejects incompatible schemas and does not migrate existing benchmark runs.

## Library usage

`FluxFold` is asynchronous. Applications provide a generation provider. Retrieval uses the local `sentence-transformers/all-MiniLM-L6-v2` model by default; the model is downloaded from Hugging Face on first use and then reused from the local cache. An OpenAI-compatible embedding adapter remains available for explicit configuration.

```python
import asyncio

from fluxfold import (
    EpisodeBlock,
    FluxFold,
    NormalizedEpisode,
    OpenAICompatibleGenerationProvider,
    Role,
)


async def main() -> None:
    generation = OpenAICompatibleGenerationProvider(
        model="gpt-4.1-mini",
        api_key="...",
    )
    async with await FluxFold.open(
        db_path="fluxfold.sqlite3",
        generation_provider=generation,
    ) as engine:
        space = await engine.create_or_open_space("example")
        await engine.add_episode(
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
`FLUXFOLD_ANSWER_*`, `FLUXFOLD_SCORE_*`). The benchmark uses the bundled local embedding model unless `.env` explicitly selects `openai-compatible`, then:

```bash
uv run python -m benchmarks.scripts.build_sample --dataset longmemeval
uv run python -m benchmarks.scripts.answer_sample --dataset longmemeval
uv run python -m benchmarks.scripts.score_sample --dataset longmemeval

uv run python -m benchmarks.scripts.build_sample \
  --dataset locomo_refined \
  --select conv-26
uv run python -m benchmarks.scripts.answer_sample --dataset locomo_refined
uv run python -m benchmarks.scripts.score_sample --dataset locomo_refined
```

`build` without `--run-dir` creates `runs/{dataset}_{month}.{day}_{HH:MM}_{seq}` using local time, for example `runs/longmemeval_8.27_21:02_1`. A second build in the same minute becomes `_2`. `answer` and `score` without `--run-dir` use the latest run of that dataset and mode. Pass `--run-dir` to override. Replace `_sample` with `_full` for complete datasets. Optional `--config` TOML overrides go under `[fluxfold]`. Build accepts `--memory-space-build-concurrency N` to override how many memory spaces are built at once (default 10).
