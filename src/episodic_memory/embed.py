from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


class Embedder:
    """Wrapper around a text embedding model.

    Defaults to Sentence-BERT (all-MiniLM-L6-v2), fully offline.
    Can be swapped with any callable that maps text → float vector.
    """

    def __init__(self, model: str | None = None):
        self._model_name = model or "all-MiniLM-L6-v2"
        self._model = None

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return 384  # all-MiniLM-L6-v2

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a float vector."""
        self._ensure_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple texts at once (batched for efficiency)."""
        self._ensure_model()
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector in-place."""
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v