from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from fluxfold import FluxFold, FluxFoldConfig
from fluxfold.errors import ErrorClass, StageFailure, ValidationError
from fluxfold.models import FullSplitOutput
from fluxfold.names import bm25_scores, normalized_name
from fluxfold.prompts import EXTRACTION_SYSTEM
from fluxfold.providers import GenerationResponse
from fluxfold.storage import PreparedLink
from tests.fakes import (
    FakeDecisionProvider,
    FakeEmbeddingProvider,
    FakeGenerationProvider,
)
from tests.test_engine import _episode


def response(value):
    return GenerationResponse(
        text=json.dumps(value), input_tokens=1, output_tokens=1, total_tokens=2
    )


class AliasProvider(FakeGenerationProvider):
    async def generate(self, request):
        payload = json.loads(request.user_prompt)
        if "name_resolution" in payload:
            self.requests.append(request)
            return response(
                {
                    "resolutions": [
                        {
                            "proposed_name": item["proposed_name"],
                            "canonical_name": item["candidates"][0]["name"],
                        }
                        for item in payload["name_resolution"]["items"]
                    ]
                }
            )
        if request.stage == "memory_extraction":
            self.requests.append(request)
            content = payload["episode"]["messages"][-1]["content"]
            return response(
                {
                    "result": "memories",
                    "memories": [
                        {
                            "content": content,
                            "entities": ["Alice" if "first" in content else "Alicia"],
                        }
                    ],
                }
            )
        return await super().generate(request)


def test_entity_alias_resolution_is_atomic_persistent_and_warns(tmp_path):
    async def scenario():
        events = []
        generation = AliasProvider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "aliases.db"),
            generation_provider=generation,
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("a")
        await engine.add_episode(
            space.memory_space_id, _episode("1", "Alice first fact", 0)
        )
        await engine.add_episode(
            space.memory_space_id, _episode("2", "Alicia second fact", 1)
        )
        bank = engine.memory_bank()[0]
        assert [domain.name for domain in bank.domains] == ["Alice"]
        assert len(bank.domains[0].memories) == 2
        with engine._store._connect() as db:
            assert db.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM memory_entities").fetchone()[0] == 2
        assert any(
            event["event_type"] == "name_normalized"
            and event["object_kind"] == "entity"
            for event in events
        )
        feedback = [
            json.loads(request.user_prompt)
            for request in generation.requests
            if "name_resolution" in request.user_prompt
        ][0]
        assert feedback["name_resolution"]["items"][0]["candidates"] == [
            {"name": "Alice"}
        ]
        await engine.close()

    asyncio.run(scenario())


def test_exact_name_reuse_works_when_domain_is_not_recalled(tmp_path, monkeypatch):
    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "exact.db"),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        first = await engine.create_or_open_space("first")
        await engine.add_episode(
            first.memory_space_id, _episode("1", "Alice likes hiking", 0)
        )
        result = await engine.add_episode(
            first.memory_space_id, _episode("2", "Alice bought land", 1)
        )
        assert result.domains_created == 0
        assert len(engine.memory_bank()[0].domains) == 1
        other = await engine.create_or_open_space("other")
        await engine.add_episode(
            other.memory_space_id, _episode("other-1", "Alice plays piano", 0)
        )
        assert len(engine._store.entity_names(other.memory_space_id)) == 1
        with engine._store._connect() as db:
            assert db.execute("SELECT count(*) FROM entities").fetchone()[0] == 2
            domain = engine._store.active_domain_ids(first.memory_space_id)[0]
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO domains SELECT 'duplicate', memory_space_id, ' ALICE ', normalized_name, summary, lifecycle_status, new_memory_count, summary_revision, created_at, updated_at, retired_at, retired_by_operation_id FROM domains WHERE domain_id=?",
                    (domain,),
                )
        await engine.close()

    asyncio.run(scenario())


def test_failed_link_does_not_persist_entities(tmp_path, monkeypatch):
    from fluxfold.linking import Linker

    async def fail(self, *args):
        raise StageFailure(
            "domain_linking_direct", ErrorClass.INVALID_STRUCTURED_OUTPUT, "invalid"
        )

    monkeypatch.setattr(Linker, "link", fail)

    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "failed.db"),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("a")
        with pytest.raises(StageFailure):
            await engine.add_episode(
                space.memory_space_id, _episode("1", "Alice likes hiking", 0)
            )
        assert engine._store.entity_names(space.memory_space_id) == ()
        assert engine._store.active_domain_ids(space.memory_space_id) == ()
        await engine.close()

    asyncio.run(scenario())


def test_batch_entity_proposals_resolve_against_earlier_proposals(tmp_path):
    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "batch.db"),
            generation_provider=AliasProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("a")
        mapping, _ = await engine._resolve_names(
            space.memory_space_id,
            ["Alice", "Alicia", " ALICE "],
            [],
            "memory_extraction",
            EXTRACTION_SYSTEM,
            "{}",
            {},
        )
        assert set(mapping.values()) == {"Alice"}
        await engine.close()

    asyncio.run(scenario())


def test_split_reuses_existing_target_and_unions_entity_membership(tmp_path):
    class Provider(FakeGenerationProvider):
        def __init__(self):
            super().__init__()
            self.split_output = None

        async def generate(self, request):
            if request.stage == "memory_extraction":
                payload = json.loads(request.user_prompt)
                entity = payload["episode"]["messages"][0]["content"]
                return response(
                    {
                        "result": "memories",
                        "memories": [
                            {"content": f"{entity} fact {index}", "entities": [entity]}
                            for index in range(6)
                        ],
                    }
                )
            if request.stage == "domain_split":
                self.requests.append(request)
                return response(self.split_output)
            return await super().generate(request)

    async def scenario():
        generation = Provider()
        engine = await FluxFold.open(
            db_path=str(tmp_path / "split.db"),
            generation_provider=generation,
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("a")
        first = await engine.add_episode(
            space.memory_space_id, _episode("1", "Alice", 0)
        )
        await engine.add_episode(space.memory_space_id, _episode("2", "Bob", 1))
        catalog = {
            entry.name: entry.object_id
            for entry in engine._store.domain_names(space.memory_space_id)
        }
        snapshot = engine._store.domain_snapshot(
            space.memory_space_id, catalog["Alice"]
        )
        ids = [memory.memory_id for memory in snapshot.memories]
        generation.split_output = {
            "result": "full_split",
            "domains": [
                {
                    "name": " bob ",
                    "links": [{"memory_id": mid, "basis": "direct"} for mid in ids[:3]],
                },
                {
                    "name": "Alice's travels",
                    "links": [{"memory_id": mid, "basis": "direct"} for mid in ids[3:]],
                },
            ],
        }
        signature = engine._store.ensure_retrieval_signature(
            space.memory_space_id, engine._embedding.model_info
        )
        # One transferred memory already has a contextual link to the reused target.
        with engine._store._transaction() as db:
            engine._store._insert_link(
                db,
                PreparedLink(ids[0], catalog["Bob"], "contextual"),
                first.operation_id,
                1,
            )
        output = FullSplitOutput.model_validate(generation.split_output)
        engine._validate_split(space.memory_space_id, output, snapshot)
        await engine._split_domain(
            space.memory_space_id, first.operation_id, snapshot, signature, "test"
        )
        memberships = engine._store.domain_entity_names(space.memory_space_id)
        assert memberships[catalog["Bob"]] == {"Alice", "Bob"}
        assert engine._store.domain_ancestors(space.memory_space_id)[
            catalog["Bob"]
        ] == {catalog["Alice"]}
        target = engine._store.domain_snapshot(space.memory_space_id, catalog["Bob"])
        assert len(target.memories) == 9
        assert target.new_memory_count == 2
        assert (
            engine._store.active_link_bases(space.memory_space_id)[
                (ids[0], catalog["Bob"])
            ]
            == "direct"
        )
        assert catalog["Bob"] in engine._store.pending_summary_refresh_domain_ids(
            space.memory_space_id, first.operation_id
        )
        assert catalog["Alice"] not in engine._store.active_domain_ids(
            space.memory_space_id
        )
        assert [
            request
            for request in generation.requests
            if request.stage == "domain_split"
        ][0].user_prompt.find("entities") == -1
        output.domains[1].name = "BOB"
        with pytest.raises(ValidationError, match="unique"):
            engine._validate_split(space.memory_space_id, output, snapshot)
        await engine.close()

    asyncio.run(scenario())


def test_name_scoring_and_zero_recall_defaults():
    assert normalized_name("  MiKe\tSmith ") == "mike smith"
    assert normalized_name("Project Atlas II") != normalized_name("Project Atlas")
    scores = bm25_scores("Texas land", ["Texas land", "piano", "Texas land management"])
    assert scores[0] > scores[2] > scores[1] == 0
    config = FluxFoldConfig()
    assert (
        config.search_domain_min_similarity == config.search_memory_min_similarity == 0
    )


def test_size_warnings_do_not_truncate_extraction_or_candidates(tmp_path, monkeypatch):
    class ManyMemories(FakeGenerationProvider):
        async def generate(self, request):
            if request.stage == "memory_extraction":
                return response(
                    {
                        "result": "memories",
                        "memories": [
                            {"content": f"Alice fact {index}", "entities": ["Alice"]}
                            for index in range(21)
                        ],
                    }
                )
            return await super().generate(request)

    async def scenario():
        events = []
        engine = await FluxFold.open(
            db_path=str(tmp_path / "warnings.db"),
            generation_provider=ManyMemories(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
            event_sink=events.append,
        )
        space = await engine.create_or_open_space("a")
        result = await engine.add_episode(
            space.memory_space_id, _episode("1", "many facts", 0)
        )
        assert result.memories_created == 21
        assert any(
            event["event_type"] == "extraction_memory_count_warning"
            and event["memory_count"] == 21
            for event in events
        )
        await engine.close()

    asyncio.run(scenario())


def test_model_switch_rebuilds_entity_vectors_and_space_delete_cleans_relations(
    tmp_path,
):
    async def scenario():
        path = tmp_path / "switch.db"
        engine = await FluxFold.open(
            db_path=str(path),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("a")
        await engine.add_episode(
            space.memory_space_id, _episode("1", "Alice likes hiking", 0)
        )
        old_id = engine._store.entity_names(space.memory_space_id)[0].object_id
        await engine.close()
        engine = await FluxFold.open(
            db_path=str(path),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(revision="2"),
        )
        rebuilt = await engine.rebuild_retrieval_embeddings(space.memory_space_id)
        with engine._store._connect() as db:
            row = db.execute(
                "SELECT entity_id, model_signature_id FROM entities"
            ).fetchone()
            assert tuple(row) == (old_id, rebuilt.model_signature_id)
        await engine.add_episode(
            space.memory_space_id, _episode("2", "Alice bought land", 1)
        )
        await engine.delete_space(space.memory_space_id)
        with engine._store._connect() as db:
            for table in ("entities", "entity_domains", "memory_entities"):
                assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", ["4", "5", "6", "7"])
def test_old_schema_rejected_before_applying_new_ddl(tmp_path, version):
    from fluxfold.storage import Store

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO schema_metadata VALUES ('schema_version', ?)", (version,)
        )
    with pytest.raises(ValidationError, match="incompatible"):
        Store(path, FluxFoldConfig())
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == [("schema_metadata",)]


def test_direct_assignments_cover_entities_without_counting_inherited_memberships(
    tmp_path,
):
    class Provider(FakeGenerationProvider):
        async def generate(self, request):
            if (
                request.stage == "memory_extraction"
                and "name_resolution" not in request.user_prompt
            ):
                return response(
                    {
                        "result": "memories",
                        "memories": [
                            {
                                "content": "A joint fact.",
                                "entities": [f"Person {i}" for i in range(6)],
                            }
                        ],
                    }
                )
            return await super().generate(request)

    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "coverage.db"),
            generation_provider=Provider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("coverage")
        first = await engine.add_episode(
            space.memory_space_id, _episode("1", "six people", 0)
        )
        assert first.domains_created == first.links_created == 6
        with engine._store._transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO entity_domains SELECT entity_id, domain_id FROM entities CROSS JOIN domains"
            )
        second = await engine.add_episode(
            space.memory_space_id, _episode("2", "six people again", 1)
        )
        assert second.links_created == 1
        await engine.close()

    asyncio.run(scenario())


def test_retired_domains_are_recalled_rebuilt_and_reactivated_without_old_links(
    tmp_path,
):
    class Provider(FakeGenerationProvider):
        split_output = None

        async def generate(self, request):
            if request.stage == "memory_extraction":
                return response(
                    {
                        "result": "memories",
                        "memories": [
                            {"content": f"Alice fact {index}", "entities": ["Alice"]}
                            for index in range(12)
                        ],
                    }
                )
            if request.stage == "domain_split":
                return response(self.split_output)
            return await super().generate(request)

    async def scenario():
        generation = Provider(always_new_domain=True)
        path = str(tmp_path / "lifecycle.db")
        engine = await FluxFold.open(
            db_path=path,
            generation_provider=generation,
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        space = await engine.create_or_open_space("lifecycle")
        sid = space.memory_space_id
        first = await engine.add_episode(sid, _episode("1", "Alice facts", 0))
        root = engine._store.domain_names(sid)[0].object_id

        async def split(domain_id, names, operation_id):
            snapshot = engine._store.domain_snapshot(sid, domain_id)
            ids = [memory.memory_id for memory in snapshot.memories]
            half = len(ids) // 2
            generation.split_output = {
                "result": "full_split",
                "domains": [
                    {
                        "name": name,
                        "links": [
                            {"memory_id": mid, "basis": "direct"} for mid in part
                        ],
                    }
                    for name, part in zip(names, [ids[:half], ids[half:]], strict=True)
                ],
            }
            await engine._split_domain(
                sid,
                operation_id,
                snapshot,
                engine._store.ensure_retrieval_signature(
                    sid, engine._embedding.model_info
                ),
                "test",
            )
            return set(ids)

        old_root_memories = await split(
            root, ["Alice work", "Alice travel"], first.operation_id
        )
        with engine._store._connect() as db:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO domains SELECT 'duplicate-root', memory_space_id, name, normalized_name, summary, 'active', new_memory_count, summary_revision, created_at, updated_at, NULL, NULL FROM domains WHERE domain_id=?",
                    (root,),
                )
        catalog = {
            entry.name: entry.object_id for entry in engine._store.domain_names(sid)
        }
        work = catalog["Alice work"]
        old_work_memories = await split(
            work, ["Alice work projects", "Alice work team"], first.operation_id
        )
        assert root not in engine._store.active_domain_ids(sid)
        assert work not in engine._store.active_domain_ids(sid)
        vector = (await engine._embed_queries(["Alice"]))[0]
        events = []
        engine._event_sink = events.append
        await engine._link_memories(
            sid,
            ["probe"],
            ["Alice"],
            [vector],
            [["Alice"]],
            [],
            episode_id="probe",
            link_attempt_id="probe",
        )
        candidates = next(e for e in events if e["event_type"] == "link_decision")[
            "groups"
        ][0]["candidates"]
        assert len(candidates) == 5
        assert "Alice" in {item["name"] for item in candidates}
        result = await engine.search(sid, "Alice")
        assert not {root, work} & {domain.domain_id for domain in result.domains}
        # Retrieval alone does not restore either retired domain.
        assert root not in engine._store.active_domain_ids(sid)
        await engine.close()
        engine = await FluxFold.open(
            db_path=path,
            generation_provider=generation,
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(revision="2"),
        )
        rebuilt = await engine.rebuild_retrieval_embeddings(sid)
        assert rebuilt.domains_embedded == 5
        assert len(engine._store.domain_names(sid)) == 5

        class RootDecision(FakeDecisionProvider):
            async def decide(self, questions):
                result = await super().decide(questions)
                for key, question in questions.items():
                    if question.kind == "choice":
                        result.answers[key] = {
                            "type": "choice",
                            "choice": "Alice",
                            "probabilities": {
                                name: float(name == "Alice")
                                for name in question.options
                            },
                        }
                return result

        engine._decision = RootDecision()
        second = await engine.add_episode(sid, _episode("2", "new Alice facts", 1))
        assert second.domains_created == 0
        restored = engine._store.domain_snapshot(sid, root)
        assert len(restored.memories) == 12
        assert not old_root_memories & {
            memory.memory_id for memory in restored.memories
        }
        assert work not in engine._store.active_domain_ids(sid)
        # Split also reuses a retired target with the same normalized name.
        await split(root, [" ALICE WORK ", "Alice travel"], second.operation_id)
        assert work in engine._store.active_domain_ids(sid)
        reused = engine._store.domain_snapshot(sid, work)
        assert len(reused.memories) == 6
        assert not old_work_memories & {memory.memory_id for memory in reused.memories}
        with engine._store._connect() as db:
            row = db.execute(
                "SELECT retired_at, retired_by_operation_id FROM domains WHERE domain_id=?",
                (work,),
            ).fetchone()
            assert tuple(row) == (None, None)
            assert (
                db.execute(
                    "SELECT count(*) FROM domains WHERE normalized_name='alice work'"
                ).fetchone()[0]
                == 1
            )
            assert (
                db.execute(
                    "SELECT count(*) FROM memory_operation_effects WHERE object_id=? AND effect_type='reactivated'",
                    (work,),
                ).fetchone()[0]
                == 1
            )
        await engine.close()

    asyncio.run(scenario())


def test_public_search_filters_empty_domains_before_top_k(tmp_path):
    from fluxfold.storage import PreparedDomain

    async def scenario():
        engine = await FluxFold.open(
            db_path=str(tmp_path / "empty.db"),
            generation_provider=FakeGenerationProvider(),
            decision_provider=FakeDecisionProvider(),
            embedding_provider=FakeEmbeddingProvider(),
            config=FluxFoldConfig(
                search_domain_top_k=1, search_domain_min_similarity=-1.0
            ),
        )
        space = await engine.create_or_open_space("empty")
        added = await engine.add_episode(
            space.memory_space_id, _episode("1", "Alice likes hiking", 0)
        )
        vector = (await engine._embed_queries(["empty exact match"]))[0]
        signature = engine._store.ensure_retrieval_signature(
            space.memory_space_id, engine._embedding.model_info
        )
        with engine._store._transaction() as db:
            engine._store._insert_domain(
                db,
                space.memory_space_id,
                PreparedDomain("empty", "empty exact match", None, vector, None),
                added.operation_id,
                signature,
                1,
            )
        public = engine._store._scan_domains(space.memory_space_id, vector, 1, -1)
        assert len(public) == 1 and public[0][0] != "empty"
        result = await engine.search(space.memory_space_id, "empty exact match")
        assert all(domain.domain_id != "empty" for domain in result.domains)
        await engine.close()

    asyncio.run(scenario())
