"""Shared test fixtures.

Tests must run fully offline and deterministically, so they never load the
real Sentence-BERT model (which downloads weights from HuggingFace on first
use). Instead they inject a hashing bag-of-words embedder that produces stable
vectors with the same cosine semantics the storage layer relies on.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class FakeEmbedder:
    """Deterministic, offline embedder for tests.

    Maps text to a fixed-dimension vector by hashing word tokens into buckets,
    then L2-normalizes. Texts sharing tokens land closer under cosine (np.dot
    on normalized vectors), which is what storage.knn_search measures.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            vec[_stable_bucket(token, self._dim)] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts):
        return np.stack([self.embed(t) for t in texts])


import tempfile
from pathlib import Path

import pytest

from episodic_memory import EpisodicMemory


@pytest.fixture
def memory():
    """In-memory store with an offline deterministic embedder."""
    store = EpisodicMemory(embedder=FakeEmbedder())
    yield store
    store.close()


@pytest.fixture
def persisted_memory():
    """File-backed store for persistence testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = EpisodicMemory(embedder=FakeEmbedder(), db_path=db_path)
    yield store
    store.close()
    Path(db_path).unlink(missing_ok=True)

