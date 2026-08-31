from __future__ import annotations

import asyncio

import numpy as np
import pytest

import fluxfold.providers as provider_module
from fluxfold.engine import FluxFold
from fluxfold.providers import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_REVISION,
    LocalMiniLMEmbeddingProvider,
)
from tests.fakes import FakeGenerationProvider


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert convert_to_numpy
        assert normalize_embeddings
        assert not show_progress_bar
        self.calls.append(sentences)
        vectors = np.zeros((len(sentences), 384), dtype=np.float64)
        for index in range(len(sentences)):
            vectors[index, index] = 2.0
        return vectors


def test_local_minilm_model_info() -> None:
    info = LocalMiniLMEmbeddingProvider().model_info
    assert info.provider == "sentence-transformers"
    assert info.model == DEFAULT_LOCAL_EMBEDDING_MODEL
    assert info.revision == DEFAULT_LOCAL_EMBEDDING_REVISION
    assert info.dimension == 384
    assert info.normalization == "l2"
    assert info.query_mode == "plain"
    assert info.document_mode == "plain"


def test_fluxfold_open_uses_local_embedding_by_default(tmp_path) -> None:
    async def scenario() -> None:
        engine = await FluxFold.open(
            db_path=str(tmp_path / "engine.sqlite3"),
            generation_provider=FakeGenerationProvider(),
        )
        space = await engine.create_or_open_space("default-local")
        assert space.space_key == "default-local"
        await engine.close()

    asyncio.run(scenario())


def test_local_minilm_embeds_normalized_float32(monkeypatch) -> None:
    async def scenario() -> None:
        fake = _FakeSentenceTransformer()
        loads = 0

        def load() -> _FakeSentenceTransformer:
            nonlocal loads
            loads += 1
            return fake

        monkeypatch.setattr(provider_module, "_load_local_minilm_model", load)
        provider = LocalMiniLMEmbeddingProvider()
        first = await provider.embed(
            ("first", "second"), input_type="document", timeout_seconds=60
        )
        second = await provider.embed(
            ("query",), input_type="query", timeout_seconds=60
        )

        assert loads == 1
        assert fake.calls == [["first", "second"], ["query"]]
        assert len(first.vectors) == 2
        assert len(second.vectors) == 1
        for vector in (*first.vectors, *second.vectors):
            assert vector.shape == (384,)
            assert vector.dtype == np.dtype("<f4")
            assert vector.flags.c_contiguous
            assert np.linalg.norm(vector) == pytest.approx(1.0)

    asyncio.run(scenario())


def test_local_minilm_empty_input_does_not_load(monkeypatch) -> None:
    async def scenario() -> None:
        def unexpected_load() -> None:
            raise AssertionError("empty input must not load the model")

        monkeypatch.setattr(
            provider_module, "_load_local_minilm_model", unexpected_load
        )
        response = await LocalMiniLMEmbeddingProvider().embed(
            (), input_type="query", timeout_seconds=60
        )
        assert response.vectors == ()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_type", ["", "passage", "QUERY"])
def test_local_minilm_rejects_unknown_input_type(input_type: str) -> None:
    async def scenario() -> None:
        provider = LocalMiniLMEmbeddingProvider()
        with pytest.raises(ValueError, match="unknown embedding input_type"):
            await provider.embed(("text",), input_type=input_type, timeout_seconds=60)

    asyncio.run(scenario())
