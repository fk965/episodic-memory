"""Tests for the embedding layer.

These never download the real Sentence-BERT weights: the model class is
stubbed so the Embedder's branching logic can be exercised offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from episodic_memory import embed as embed_module
from episodic_memory.embed import Embedder, cosine_similarity, normalize


class TestPureHelpers:
    def test_cosine_identical(self):
        v = normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_normalize_unit_length(self):
        v = normalize(np.array([3.0, 4.0], dtype=np.float32))
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_normalize_zero_vector(self):
        v = normalize(np.zeros(3, dtype=np.float32))
        assert np.linalg.norm(v) == 0.0


class _StubModel:
    """Stand-in for SentenceTransformer that needs no network."""

    def __init__(self, name: str):
        self.name = name

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, list):
            return np.array([[float(len(t))] * 3 for t in text], dtype=np.float32)
        return np.array([float(len(text))] * 3, dtype=np.float32)


class TestEmbedder:
    def test_dim(self):
        assert Embedder().dim == 384

    def test_missing_dependency_raises(self, monkeypatch):
        monkeypatch.setattr(embed_module, "_HAS_SENTENCE_TRANSFORMERS", False)
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            Embedder().embed("hello")

    def test_embed_single(self, monkeypatch):
        monkeypatch.setattr(embed_module, "_HAS_SENTENCE_TRANSFORMERS", True)
        monkeypatch.setattr(embed_module, "SentenceTransformer", _StubModel)
        vec = Embedder().embed("hello")
        assert vec.dtype == np.float32
        assert vec.shape == (3,)

    def test_embed_many(self, monkeypatch):
        monkeypatch.setattr(embed_module, "_HAS_SENTENCE_TRANSFORMERS", True)
        monkeypatch.setattr(embed_module, "SentenceTransformer", _StubModel)
        vecs = Embedder().embed_many(["a", "bb"])
        assert vecs.shape == (2, 3)

    def test_model_loaded_once(self, monkeypatch):
        monkeypatch.setattr(embed_module, "_HAS_SENTENCE_TRANSFORMERS", True)
        calls = []

        class Counting(_StubModel):
            def __init__(self, name):
                calls.append(name)
                super().__init__(name)

        monkeypatch.setattr(embed_module, "SentenceTransformer", Counting)
        e = Embedder()
        e.embed("one")
        e.embed("two")
        assert len(calls) == 1
