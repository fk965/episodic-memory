"""Tests for episodic_memory core functionality."""

from __future__ import annotations

import pytest

from episodic_memory import EpisodicMemory, SearchResult, Triple


class TestStore:
    def test_store_returns_id(self, memory):
        mem_id = memory.store("test trigger", "test judgment", "test reasoning")
        assert mem_id.startswith("mem_")
        assert len(mem_id) > 4

    def test_store_with_domain(self, memory):
        mem_id = memory.store(
            "trigger", "judgment", "reasoning", domain="ops"
        )
        record = memory.get(mem_id)
        assert record is not None
        assert record.domain == "ops"

    def test_store_with_metadata(self, memory):
        mem_id = memory.store(
            "trigger", "judgment", "reasoning",
            metadata={"source": "test", "version": 1},
        )
        record = memory.get(mem_id)
        assert record is not None
        assert record.metadata["source"] == "test"
        assert record.metadata["version"] == 1

    def test_store_empty_store_count(self, memory):
        assert memory.count() == 0

    def test_store_increments_count(self, memory):
        memory.store("t", "j", "r")
        assert memory.count() == 1
        memory.store("t", "j", "r")
        assert memory.count() == 2


class TestSearch:
    def test_search_empty(self, memory):
        results = memory.search("anything")
        assert results == []

    def test_search_finds_relevant(self, memory):
        memory.store(
            trigger="user wants to deploy to production",
            judgment="deploy on Friday afternoon is risky",
            reasoning="fewer engineers available for rollback",
            domain="ops",
        )
        memory.store(
            trigger="user asks about lunch options",
            judgment="recommend nearby restaurants",
            reasoning="user is hungry",
            domain="general",
        )
        results = memory.search("deploy to production", top_k=5)
        assert len(results) >= 1
        assert any("risky" in r.judgment for r in results)

    def test_search_top_k(self, memory):
        for i in range(10):
            memory.store(
                f"trigger {i}",
                f"judgment {i}",
                f"reasoning {i}",
                domain="test",
            )
        results = memory.search("test", top_k=3)
        assert len(results) == 3

    def test_search_domain_filter(self, memory):
        memory.store("t1", "j1", "r1", domain="ops")
        memory.store("t2", "j2", "r2", domain="coding")
        ops_results = memory.search("j1", domain="ops")
        assert all(r.domain == "ops" for r in ops_results)
        coding_results = memory.search("j2", domain="coding")
        assert all(r.domain == "coding" for r in coding_results)

    def test_search_min_score(self, memory):
        memory.store(
            "production deployment",
            "deploy on Friday is risky",
            "fewer engineers",
            domain="ops",
        )
        all_results = memory.search("deployment", min_score=0.0)
        strict_results = memory.search("deployment", min_score=0.99)
        assert len(all_results) >= len(strict_results)

    def test_search_returns_search_result_type(self, memory):
        memory.store("trigger", "my judgment", "reasoning", domain="test", metadata={"k": "v"})
        results = memory.search("my judgment")
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.judgment == "my judgment"
        assert r.reasoning == "reasoning"
        assert r.domain == "test"
        assert r.metadata == {"k": "v"}
        assert 0.0 <= r.distance <= 2.0


class TestUtilityWeightedSearch:
    """The flywheel: judgments that proved useful should rank higher."""

    def test_proven_judgment_ranks_above_equal_relevance(self, memory):
        # Two memories with identical text (identical similarity to query).
        proven = memory.store("deploy production", "freeze on Friday", "risky")
        unproven = memory.store("deploy production", "freeze on Friday", "risky")
        # One gets repeatedly adopted -> high utility.
        for _ in range(5):
            memory.verify(proven, adopted=True)

        results = memory.search("deploy production", use_utility=True)
        ids = [r.id for r in results]
        assert ids.index(proven) < ids.index(unproven)

    def test_utility_does_not_override_relevance(self, memory):
        # A high-utility but irrelevant memory must not outrank a relevant one.
        irrelevant = memory.store("lunch options", "eat tacos", "yummy")
        for _ in range(5):
            memory.verify(irrelevant, adopted=True)
        relevant = memory.store("deploy production", "freeze on Friday", "risky")

        results = memory.search("deploy production", use_utility=True, top_k=2)
        assert results[0].id == relevant

    def test_use_utility_off_is_pure_cosine(self, memory):
        proven = memory.store("deploy production", "freeze on Friday", "risky")
        memory.store("deploy production", "freeze on Friday", "risky")  # unproven twin
        for _ in range(5):
            memory.verify(proven, adopted=True)
        # With weighting off, tie broken by cosine only (order is by insertion
        # among equal scores) — both returned, neither forced ahead by utility.
        results = memory.search("deploy production", use_utility=False)
        assert len(results) == 2

    def test_corrected_judgment_ranks_below(self, memory):
        good = memory.store("deploy production", "freeze on Friday", "risky")
        bad = memory.store("deploy production", "freeze on Friday", "risky")
        memory.verify(good, adopted=True)
        memory.verify(bad, adopted=False, user_correction="wrong")

        results = memory.search("deploy production", use_utility=True)
        ids = [r.id for r in results]
        assert ids.index(good) < ids.index(bad)

    def test_more_evidence_outranks_less_at_equal_ratio(self, memory):
        # The v0.2.0 promise: a judgment adopted many times outranks one
        # adopted only once, even though both have a 100% adoption ratio.
        # The old raw-ratio formula scored both 1.0 and could not do this.
        well_tested = memory.store("deploy production", "freeze on Friday", "risky")
        barely_tested = memory.store("deploy production", "freeze on Friday", "risky")
        for _ in range(50):
            memory.verify(well_tested, adopted=True)
        memory.verify(barely_tested, adopted=True)

        assert memory.get(well_tested).utility_score > memory.get(barely_tested).utility_score  # type: ignore
        results = memory.search("deploy production", use_utility=True)
        ids = [r.id for r in results]
        assert ids.index(well_tested) < ids.index(barely_tested)


class TestVerify:
    def test_verify_adopted_increases_score(self, memory):
        mem_id = memory.store("trigger", "judgment", "reasoning")
        r1 = memory.get(mem_id)
        assert r1 is not None
        initial_score = r1.utility_score

        memory.verify(mem_id, adopted=True)
        r2 = memory.get(mem_id)
        assert r2 is not None
        assert r2.utility_score >= initial_score
        assert r2.adoption_count == 1

    def test_verify_corrected_decreases_score(self, memory):
        mem_id = memory.store("trigger", "bad judgment", "wrong reasoning")
        memory.verify(mem_id, adopted=True)
        r1 = memory.get(mem_id)
        assert r1 is not None
        after_adoption = r1.utility_score

        memory.verify(mem_id, adopted=False, user_correction="Actually, do it differently")
        r2 = memory.get(mem_id)
        assert r2 is not None
        assert r2.utility_score < after_adoption
        assert r2.correction_count == 1

    def test_verify_multiple_events(self, memory):
        mem_id = memory.store("trigger", "judgment", "reasoning")
        for _ in range(5):
            memory.verify(mem_id, adopted=True)
        for _ in range(3):
            memory.verify(mem_id, adopted=False)
        record = memory.get(mem_id)
        assert record is not None
        assert record.adoption_count == 5
        assert record.correction_count == 3
        assert 0.0 < record.utility_score < 1.0

    def test_verify_nonexistent(self, memory):
        with pytest.raises(Exception):
            memory.verify("nonexistent_id", adopted=True)


class TestExport:
    def test_export_empty(self, memory):
        triples = memory.export_triples()
        assert triples == []

    def test_export_only_verified(self, memory):
        memory.store("t", "j", "r")
        triples = memory.export_triples()
        # Without verify, utility is 0.0
        assert len(triples) == 0

    def test_export_with_verify(self, memory):
        mem_id = memory.store("t", "j", "r")
        memory.verify(mem_id, adopted=True)
        triples = memory.export_triples(min_utility=0.0)
        assert len(triples) >= 1
        assert isinstance(triples[0], Triple)
        assert "utility_score" in triples[0].outcome

    def test_export_min_utility_filter(self, memory):
        mem_id = memory.store("t", "j", "r")
        memory.verify(mem_id, adopted=True)
        high_utility = memory.export_triples(min_utility=0.9)
        low_utility = memory.export_triples(min_utility=0.0)
        assert len(high_utility) <= len(low_utility)


class TestPersistence:
    """Real cross-session persistence: close the store, reopen a *fresh*
    one at the same path, and confirm the data survived. The previous tests
    reused a single open connection and never actually exercised reload.
    """

    def test_persists_across_sessions(self, tmp_db_path, fake_embedder_cls):
        store = EpisodicMemory(embedder=fake_embedder_cls(), db_path=tmp_db_path)
        store.store("trigger", "judgment", "reasoning", domain="ops")
        store.close()

        reopened = EpisodicMemory(embedder=fake_embedder_cls(), db_path=tmp_db_path)
        try:
            assert reopened.count() == 1
        finally:
            reopened.close()

    def test_reload_persisted_data(self, tmp_db_path, fake_embedder_cls):
        store = EpisodicMemory(embedder=fake_embedder_cls(), db_path=tmp_db_path)
        mem_id = store.store("t", "j", "r")
        store.verify(mem_id, adopted=True)
        store.close()

        reopened = EpisodicMemory(embedder=fake_embedder_cls(), db_path=tmp_db_path)
        try:
            record = reopened.get(mem_id)
            assert record is not None
            assert record.adoption_count == 1
            assert record.utility_score > 0
        finally:
            reopened.close()


class TestGet:
    def test_get_nonexistent(self, memory):
        assert memory.get("nonexistent") is None

    def test_get_returns_record(self, memory):
        mem_id = memory.store("trigger", "judgment", "reasoning", domain="test")
        record = memory.get(mem_id)
        assert record is not None
        assert record.trigger == "trigger"
        assert record.judgment == "judgment"
        assert record.reasoning == "reasoning"
        assert record.domain == "test"


class TestEdgeCases:
    def test_very_long_text(self, memory):
        long_trigger = "x" * 10000
        mem_id = memory.store(long_trigger, "judgment", "reasoning")
        assert mem_id is not None

    def test_special_characters(self, memory):
        mem_id = memory.store(
            "trigger with 中文 and emoji 🎉",
            "judgment with spécial chars & <script>",
            "reasoning with 日本語",
        )
        record = memory.get(mem_id)
        assert record is not None
        assert "中文" in record.trigger

    def test_empty_strings_raises_error(self, memory):
        with pytest.raises(Exception):
            memory.store("", "", "")

    def test_multiple_domains_isolation(self, memory):
        memory.store("t1", "j1", "r1", domain="a")
        memory.store("t2", "j2", "r2", domain="b")
        assert memory.count() == 2
        a_results = memory.search("j1", domain="a")
        assert len(a_results) >= 1
        assert all(r.domain == "a" for r in a_results)
        assert all(r.domain == "b" for r in memory.search("j1", domain="b"))
