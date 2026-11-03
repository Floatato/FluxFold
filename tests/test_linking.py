"""Behavioral coverage for entity-local typed linking and domain lineage."""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from benchmarks.artifacts import ArtifactWriter, RunPaths
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from fluxfold import FluxFold, FluxFoldConfig
from fluxfold.decisions import (
    DecisionQuestion,
    DecisionResponse,
    TypeSafeDecisionProvider,
)
from fluxfold.errors import ErrorClass, ProviderError, StageFailure, ValidationError
from fluxfold.linking import Linker, validate_answers
from fluxfold.models import FullSplitOutput
from tests.fakes import (
    FakeDecisionProvider,
    FakeEmbeddingProvider,
    FakeGenerationProvider,
)
from tests.test_engine import _episode


class ChosenDecisions(FakeDecisionProvider):
    def __init__(self, choices=None, probabilities=None):
        super().__init__()
        self.choices = choices or {}
        self.probabilities = probabilities or {}

    async def decide(self, questions):
        self.requests.append(questions)
        answers = {}
        for key, question in questions.items():
            if question.kind == "choice":
                selected = self.choices[question.state["entity"]]
                answers[key] = {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {
                        name: float(name == selected) for name in question.options
                    },
                    "confidence": 1.0,
                }
            else:
                answers[key] = {
                    "type": "noul",
                    "noul": self.probabilities.get(question.state["domain"], 0.9),
                }
        return DecisionResponse(answers, self.model_id, 10, 0)


def memory(candidates, memory_id="fact"):
    return {
        "memory_id": memory_id,
        "content": "A durable fact",
        "candidates": {
            entity: [{"name": name, "similarity": 0.9} for name in names]
            for entity, names in candidates.items()
        },
    }


def test_two_rounds_ancestor_filter_and_entity_contextual_cap():
    async def scenario():
        provider = ChosenDecisions(
            {"Alice": "dental", "Bob": "driving"}, {"shared": 0.8, "low": 0.5}
        )
        events = []
        linker = Linker(
            provider,
            FluxFoldConfig(),
            lambda kind, **data: events.append({"event_type": kind, **data}),
            {},
        )
        result = await linker.link(
            [
                memory(
                    {
                        "Alice": ["Alice", "dental", "diet", "shared", "low"],
                        "Bob": ["driving", "trip", "commute", "shared", "hobby"],
                    }
                )
            ],
            {"dental": {"Alice", "health"}},
        )
        assert len(provider.requests) == 2
        assert {q.kind for q in provider.requests[0].values()} == {"choice"}
        nouls = list(provider.requests[1].values())
        assert {q.kind for q in nouls} == {"noul"}
        assert {q.state["domain"] for q in nouls} == {
            "diet",
            "shared",
            "low",
            "trip",
            "commute",
            "hobby",
        }
        assert sum(q.state["domain"] == "shared" for q in nouls) == 1
        assert all(set(q.state) == {"fact", "domain"} for q in nouls)
        output = result.memories[0]
        assert {a.domain for a in output.direct_assignments} == {"dental", "driving"}
        assert output.contextual_domains == ["commute", "diet", "hobby", "shared"]
        assert len(output.contextual_domains) + len(output.direct_assignments) == 6
        audit = next(
            event for event in events if event["event_type"] == "link_decision"
        )
        reasons = {
            item["domain"]: item["reason"] for item in audit["groups"][0]["contextual"]
        }
        assert reasons["Alice"] == "ancestor_of_direct"
        assert reasons["low"] == "below_threshold"

    asyncio.run(scenario())


def test_singleton_skips_model_and_shared_direct_deduplicates():
    async def scenario():
        provider = FakeDecisionProvider()
        linker = Linker(provider, FluxFoldConfig(), lambda *a, **k: None, {})
        output = await linker.link(
            [memory({"Alice": ["friendship"], "Bob": ["friendship"]})], {}
        )
        assert provider.requests == []
        assert len(output.memories[0].direct_assignments) == 2
        assert output.memories[0].contextual_domains == []

    asyncio.run(scenario())


def test_one_contextual_candidate_still_needs_judgment():
    async def scenario():
        provider = ChosenDecisions({"Alice": "dental"}, {"diet": 0.1})
        linker = Linker(provider, FluxFoldConfig(), lambda *a, **k: None, {})
        result = await linker.link([memory({"Alice": ["dental", "diet"]})], {})
        assert len(provider.requests[1]) == 1
        assert result.memories[0].contextual_domains == []

    asyncio.run(scenario())


def test_decision_capacity_split_reduces_state_and_logs_all_attempts():
    class Limited(FakeDecisionProvider):
        async def decide(self, questions):
            if len(questions) > 1:
                raise ProviderError(ErrorClass.CONTEXT_OVERFLOW, "too large")
            return await super().decide(questions)

    async def scenario():
        events = []
        linker = Linker(
            Limited(),
            FluxFoldConfig(),
            lambda kind, **data: events.append({"event_type": kind, **data}),
            {},
        )
        questions = {
            f"q{i}": DecisionQuestion("noul", {"fact": str(i)}, "Related?")
            for i in range(3)
        }
        result = await linker.ask("domain_linking_contextual", questions)
        assert result.keys() == questions.keys()
        requests = [e for e in events if e["event_type"] == "link_request"]
        assert [len(e["questions"]) for e in requests] == [3, 1, 2, 1, 1]
        assert all(
            e["state"]["items"].keys() == e["questions"].keys() for e in requests
        )

    asyncio.run(scenario())


def test_retry_logs_invalid_response_and_transport_failure(monkeypatch):
    class Flaky(FakeDecisionProvider):
        count = 0

        async def decide(self, questions):
            self.count += 1
            if self.count == 1:
                raise ProviderError(
                    ErrorClass.RATE_LIMITED, "busy", retry_after_seconds=0
                )
            if self.count == 2:
                return DecisionResponse({}, self.model_id, 3, 0)
            return await super().decide(questions)

    async def no_sleep(_):
        pass

    monkeypatch.setattr("fluxfold.linking.asyncio.sleep", no_sleep)

    async def scenario():
        events = []
        linker = Linker(
            Flaky(),
            FluxFoldConfig(),
            lambda kind, **data: events.append({"event_type": kind, **data}),
            {},
        )
        await linker.ask(
            "domain_linking_contextual", {"q": DecisionQuestion("noul", {}, "Related?")}
        )
        calls = [e for e in events if e["event_type"] == "llm_call"]
        assert [e["result"] for e in calls] == ["failed", "failed", "success"]
        assert calls[1]["input_tokens"] == 3
        assert len([e for e in events if e["event_type"] == "link_response"]) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": float("nan")},
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": 1.1},
        {"type": "choice", "choice": "x"},
    ],
)
def test_rejects_invalid_decision_values(answer):
    with pytest.raises(ValidationError):
        validate_answers({"q": DecisionQuestion("noul", {}, "Related?")}, {"q": answer})


def test_typesafe_adapter_uses_actual_sdk_wire_protocol():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(
            200,
            headers={"x-typesafe-request-id": "request-1"},
            json={
                "model": "jev-1.14.0",
                "usage": {"input_tokens": 17, "output_tokens": 0},
                "answers": {
                    "c": {
                        "type": "choice",
                        "choice": "A",
                        "probabilities": {"A": 0.8, "B": 0.2},
                        "confidence": 0.6,
                    },
                    "n": {"type": "noul", "noul": 0.9},
                },
            },
        )

    async def scenario():
        provider = TypeSafeDecisionProvider(api_key="test")
        await provider._client.aclose()
        provider._client = AsyncTypeSafeClient(
            api_key="test",
            transport=httpx2.MockTransport(handle),
            retry=RetryPolicy(max_retries=0),
            compression=False,
        )
        result = await provider.decide(
            {
                "c": DecisionQuestion("choice", {"fact": "fact"}, "Choose", ("A", "B")),
                "n": DecisionQuestion("noul", {"domain": "A"}, "Related?"),
            }
        )
        assert result.request_id == "request-1"
        assert result.input_tokens == 17
        assert result.answers["n"]["noul"] == 0.9
        assert requests[0]["questions"]["c"]["criteria"] == {"A": "A", "B": "B"}
        assert requests[0]["state"]["items"]["n"] == {"domain": "A"}
        await provider.close()

    asyncio.run(scenario())


def test_lineage_multi_parent_transitive_cycle_and_atomic_rollback(tmp_path):
    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "lineage.db"),
            generation_provider=FakeGenerationProvider(always_new_domain=True),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("lineage")
        operations = []
        for i, name in enumerate(["Alice", "Bob", "Charlie", "Diana"]):
            operations.append(
                (
                    await engine.add_episode(
                        space.memory_space_id, _episode(str(i), name, i)
                    )
                ).operation_id
            )
        store = engine._store
        ids = {e.name: e.object_id for e in store.domain_names(space.memory_space_id)}
        with store._transaction() as db:
            for parent, child in [
                ("Alice", "Bob"),
                ("Bob", "Charlie"),
                ("Diana", "Charlie"),
                ("Alice", "Bob"),
            ]:
                store._insert_lineage_edge(
                    db, space.memory_space_id, ids[parent], ids[child], operations[0]
                )
        assert store.domain_ancestors(space.memory_space_id)[ids["Charlie"]] == {
            ids[n] for n in ["Alice", "Bob", "Diana"]
        }
        with pytest.raises(ValidationError, match="cycle"):
            with store._transaction() as db:
                store._insert_lineage_edge(
                    db, space.memory_space_id, ids["Alice"], ids["Diana"], operations[0]
                )
                store._insert_lineage_edge(
                    db,
                    space.memory_space_id,
                    ids["Charlie"],
                    ids["Alice"],
                    operations[0],
                )
        assert ids["Diana"] not in store.domain_ancestors(space.memory_space_id)
        with pytest.raises(ValidationError, match="cycle"):
            with store._transaction() as db:
                store._insert_lineage_edge(
                    db, space.memory_space_id, ids["Alice"], ids["Alice"], operations[0]
                )
        await engine.delete_space(space.memory_space_id)
        assert store.domain_ancestors(space.memory_space_id) == {}
        await engine.close()

    asyncio.run(scenario())


def test_split_can_assign_each_memory_to_three_targets(tmp_path):
    async def scenario():
        generation = FakeGenerationProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "split.db"),
            generation_provider=generation,
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("split")
        result = None
        for i in range(3):
            result = await engine.add_episode(
                space.memory_space_id, _episode(str(i), f"Alice fact {i}", i)
            )
        original = engine._store.domain_snapshot(
            space.memory_space_id,
            engine._store.active_domain_ids(space.memory_space_id)[0],
        )
        payload = {
            "result": "full_split",
            "domains": [
                {
                    "name": name,
                    "links": [
                        {"memory_id": m.memory_id, "basis": "direct"}
                        for m in original.memories
                    ],
                }
                for name in ["Alice project", "Alice work", "Alice plans"]
            ],
        }
        output = FullSplitOutput.model_validate(payload)
        engine._validate_split(space.memory_space_id, output, original)
        # The normal split commit creates lineage edges in the same transaction.
        original_generate = generation.generate

        async def generate(request):
            if request.stage == "domain_split":
                from fluxfold.providers import GenerationResponse

                return GenerationResponse(json.dumps(payload), 1, 1, 2)
            return await original_generate(request)

        generation.generate = generate
        await engine._split_domain(
            space.memory_space_id,
            result.operation_id,
            original,
            engine._store.ensure_retrieval_signature(
                space.memory_space_id, engine._embedding.model_info
            ),
            "test",
        )
        ancestors = engine._store.domain_ancestors(space.memory_space_id)
        assert len(ancestors) == 3
        assert all(parents == {original.domain_id} for parents in ancestors.values())
        # A reused ancestor is rejected before calling the transaction.
        child = engine._store.domain_snapshot(
            space.memory_space_id, next(iter(ancestors))
        )
        output.domains[0].name = "Alice"
        output.domains[1].name = "Alice other"
        output.domains[2].name = "Alice third"
        with pytest.raises(ValidationError, match="cycle"):
            engine._validate_split(space.memory_space_id, output, child)
        await engine.close()

    asyncio.run(scenario())


def test_full_link_logs_are_separate_from_events_and_mark_commits(tmp_path):
    async def scenario():
        writer = ArtifactWriter(RunPaths(tmp_path / "logs"))
        engine = await FluxFold.open(
            db_path=str(tmp_path / "logs.db"),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
            event_sink=writer.event,
        )
        space = await engine.create_or_open_space("logs")
        await engine.add_episode(
            space.memory_space_id, _episode("one", "Alice likes hiking")
        )
        records = [
            json.loads(line)
            for line in writer.paths.link_decisions.read_text().splitlines()
        ]
        decision = next(r for r in records if r["event_type"] == "link_decision")
        commit = next(r for r in records if r["event_type"] == "episode_committed")
        assert decision["link_attempt_id"] == commit["link_attempt_id"]
        assert decision["groups"][0]["probabilities"] is None
        assert decision["status"] == "prepared"
        assert "single_candidate" in writer.paths.link_report.read_text()
        assert '"content": "Alice likes hiking"' not in writer.paths.events.read_text()
        assert writer.build_metrics()["link_single_candidate_count"] == 1
        restored = ArtifactWriter(writer.paths)
        assert restored.build_metrics() == writer.build_metrics()
        await engine.close()

    asyncio.run(scenario())


def test_multi_entity_contextual_links_commit_beyond_old_global_limit(tmp_path):
    from fluxfold.providers import GenerationResponse

    class Extraction(FakeGenerationProvider):
        joint = False

        async def generate(self, request):
            if (
                self.joint
                and request.stage == "memory_extraction"
                and "name_resolution" not in request.user_prompt
            ):
                return GenerationResponse(
                    json.dumps(
                        {
                            "result": "memories",
                            "memories": [
                                {
                                    "content": "A shared fact about three people",
                                    "entities": ["Alice", "Bob", "Charlie"],
                                }
                            ],
                        }
                    ),
                    1,
                    1,
                    2,
                )
            return await super().generate(request)

    async def scenario():
        generation = Extraction(always_new_domain=True)
        choices = {name: name for name in ["Alice", "Bob", "Charlie"]}
        engine = await FluxFold.open(
            db_path=str(tmp_path / "many-links.db"),
            generation_provider=generation,
            decision_provider=ChosenDecisions(choices),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("many-links")
        for index, name in enumerate(
            [
                "Alice",
                "Alice diet",
                "Alice travel",
                "Bob",
                "Bob work",
                "Bob driving",
                "Charlie",
                "Charlie family",
                "Charlie plans",
            ]
        ):
            await engine.add_episode(
                space.memory_space_id, _episode(str(index), name, index)
            )
        with engine._store._transaction() as db:
            for entity in choices:
                entity_id = db.execute(
                    "SELECT entity_id FROM entities WHERE name=?", (entity,)
                ).fetchone()[0]
                db.execute(
                    "INSERT OR IGNORE INTO entity_domains SELECT ?, domain_id FROM domains WHERE name LIKE ?",
                    (entity_id, entity + "%"),
                )
        generation.joint = True
        result = await engine.add_episode(
            space.memory_space_id, _episode("joint", "joint fact", 9)
        )
        assert result.links_created == 9
        with engine._store._connect() as db:
            bases = db.execute(
                "SELECT link_basis, count(*) FROM domain_memory_links WHERE opened_by_operation_id=? GROUP BY link_basis",
                (result.operation_id,),
            ).fetchall()
            assert dict(bases) == {"direct": 3, "contextual": 6}
        await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (400, {"error": {"code": "max_tokens_exceeded"}}, ErrorClass.CONTEXT_OVERFLOW),
        (413, {}, ErrorClass.CONTEXT_OVERFLOW),
        (401, {}, ErrorClass.AUTHENTICATION_OR_CONFIGURATION),
        (429, {"error": "busy"}, ErrorClass.RATE_LIMITED),
        (429, {"error": "quota exhausted"}, ErrorClass.QUOTA_EXHAUSTED),
        (529, {}, ErrorClass.SERVICE_UNAVAILABLE),
        (422, {}, ErrorClass.INVALID_REQUEST),
    ],
)
def test_typesafe_normalizes_errors_without_hidden_retries(status, body, expected):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(
            status,
            json=body,
            headers={"retry-after": "2", "x-typesafe-request-id": "failure"},
        )

    async def scenario():
        provider = TypeSafeDecisionProvider(api_key="test")
        await provider._client.aclose()
        provider._client = AsyncTypeSafeClient(
            api_key="test",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(handle),
            compression=False,
        )
        with pytest.raises(ProviderError) as caught:
            await provider.decide({"n": DecisionQuestion("noul", {}, "Related?")})
        assert caught.value.error_class == expected
        assert caught.value.request_id == "failure"
        assert len(requests) == 1
        await provider.close()

    asyncio.run(scenario())


def test_single_question_overflow_is_terminal():
    class Overflow(FakeDecisionProvider):
        async def decide(self, questions):
            raise ProviderError(ErrorClass.CONTEXT_OVERFLOW, "too large")

    async def scenario():
        linker = Linker(Overflow(), FluxFoldConfig(), lambda *a, **k: None, {})
        with pytest.raises(StageFailure) as caught:
            await linker.ask(
                "domain_linking_contextual",
                {"q": DecisionQuestion("noul", {}, "Related?")},
            )
        assert caught.value.error_class == ErrorClass.CONTEXT_OVERFLOW

    asyncio.run(scenario())
