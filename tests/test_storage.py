"""Additional tests for storage and embed modules."""

from __future__ import annotations

import numpy as np
import pytest

from episodic_memory.storage import Storage, _serialize_vector, _deserialize_vector


class TestVectorSerialization:
    def test_roundtrip(self):
        vec = np.array([0.1, 0.2, 0.3, 0.0, -0.5], dtype=np.float32)
        blob = _serialize_vector(vec)
        restored = _deserialize_vector(blob)
        assert np.allclose(vec, restored)

    def test_zero_vector(self):
        vec = np.zeros(128, dtype=np.float32)
        blob = _serialize_vector(vec)
        restored = _deserialize_vector(blob)
        assert np.allclose(vec, restored)

    def test_high_dimension(self):
        vec = np.random.randn(768).astype(np.float32)
        blob = _serialize_vector(vec)
        restored = _deserialize_vector(blob)
        assert np.allclose(vec, restored)


class TestStorage:
    @pytest.fixture
    def storage(self):
        s = Storage()
        yield s
        s.close()

    def test_insert_and_count(self, storage):
        vec = np.zeros(384, dtype=np.float32)
        storage.insert("mem_1", "t", "j", "r", vec)
        assert storage.count() == 1

    def test_knn_empty(self, storage):
        vec = np.zeros(384, dtype=np.float32)
        assert storage.knn_search(vec) == []

    def test_knn_finds_nearest(self, storage):
        vec1 = np.zeros(384, dtype=np.float32)
        vec1[0] = 1.0
        storage.insert("mem_1", "t1", "j1", "r1", vec1)

        vec2 = np.zeros(384, dtype=np.float32)
        vec2[0] = 0.5
        storage.insert("mem_2", "t2", "j2", "r2", vec2)

        # Search with vector near mem_1
        query = np.zeros(384, dtype=np.float32)
        query[0] = 0.9
        results = storage.knn_search(query, top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "mem_1"

    def test_knn_domain_filter(self, storage):
        vec = np.zeros(384, dtype=np.float32)
        storage.insert("mem_1", "t", "j", "r", vec, domain="ops")
        storage.insert("mem_2", "t", "j", "r", vec, domain="coding")

        results = storage.knn_search(vec, domain="ops")
        assert all(r["domain"] == "ops" for r in results)

    def test_get_nonexistent(self, storage):
        assert storage.get("nope") is None

    def test_record_verification(self, storage):
        vec = np.zeros(384, dtype=np.float32)
        storage.insert("mem_1", "t", "j", "r", vec)
        storage.record_verification("mem_1", adopted=True)
        rec = storage.get("mem_1")
        assert rec is not None
        assert rec["adoption_count"] == 1
        assert rec["utility_score"] > 0


class TestEmbedEdgeCases:
    def test_serialize_roundtrip_many(self):
        for dim in [128, 384, 768, 1536]:
            vec = np.random.randn(dim).astype(np.float32)
            blob = _serialize_vector(vec)
            restored = _deserialize_vector(blob)
            assert np.allclose(vec, restored)
            assert len(restored) == dim