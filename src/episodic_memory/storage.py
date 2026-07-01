from __future__ import annotations

import json
import sqlite3
import struct
import time
import uuid
from typing import Any

import numpy as np

from .scoring import utility_score


def _serialize_vector(vec: np.ndarray) -> bytes:
    """Pack float32 vector into bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(data: bytes) -> np.ndarray:
    """Unpack bytes into float32 vector."""
    return np.frombuffer(data, dtype=np.float32)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    trigger     TEXT NOT NULL,
    judgment    TEXT NOT NULL,
    reasoning   TEXT NOT NULL,
    domain      TEXT,
    metadata    TEXT DEFAULT '{}',
    embedding   BLOB NOT NULL,
    created_at  REAL NOT NULL,
    utility_score REAL DEFAULT 0.0,
    adoption_count  INTEGER DEFAULT 0,
    correction_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);

CREATE TABLE IF NOT EXISTS verification_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       TEXT NOT NULL REFERENCES memories(id),
    adopted         INTEGER NOT NULL,
    user_correction TEXT,
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_memory ON verification_events(memory_id);
"""


class Storage:
    """SQLite-backed storage for episodic memories with vector search."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def insert(
        self,
        memory_id: str,
        trigger: str,
        judgment: str,
        reasoning: str,
        embedding: np.ndarray,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO memories (id, trigger, judgment, reasoning, domain,
               metadata, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                trigger,
                judgment,
                reasoning,
                domain,
                json.dumps(metadata or {}),
                _serialize_vector(embedding),
                time.time(),
            ),
        )
        self._conn.commit()

    def knn_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        domain: str | None = None,
        min_score: float = 0.0,
        use_utility: bool = False,
        utility_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """KNN search by computing cosine similarity in Python.

        When ``use_utility`` is set, results are ranked by
        ``sim * (1 + utility_weight * utility_score)`` so judgments that
        proved useful surface above equally-relevant ones. The ``min_score``
        gate still applies to the raw cosine similarity, so utility can never
        pull a semantically irrelevant memory past the threshold.

        For small to medium stores (<10K records) this is fast enough.
        At scale, swap to sqlite-vec or pgvector.
        """
        if domain:
            rows = self._conn.execute(
                "SELECT id, trigger, judgment, reasoning, domain, metadata, "
                "embedding, utility_score FROM memories WHERE domain = ?",
                (domain,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, trigger, judgment, reasoning, domain, metadata, "
                "embedding, utility_score FROM memories"
            ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            stored_vec = _deserialize_vector(row[6])
            sim = float(np.dot(query_vec, stored_vec))
            if sim < min_score:
                continue
            utility = row[7]
            rank_score = sim * (1.0 + utility_weight * utility) if use_utility else sim
            scored.append((
                rank_score,
                {
                    "id": row[0],
                    "trigger": row[1],
                    "judgment": row[2],
                    "reasoning": row[3],
                    "domain": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "distance": 1.0 - sim,
                    "utility_score": utility,
                },
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, trigger, judgment, reasoning, domain, metadata, "
            "utility_score, adoption_count, correction_count, created_at "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "trigger": row[1],
            "judgment": row[2],
            "reasoning": row[3],
            "domain": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "utility_score": row[6],
            "adoption_count": row[7],
            "correction_count": row[8],
            "created_at": row[9],
        }

    def record_verification(
        self,
        memory_id: str,
        adopted: bool,
        user_correction: str | None = None,
    ) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO verification_events (memory_id, adopted, user_correction, created_at) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, int(adopted), user_correction, now),
        )

        # Bump the relevant counter, then recompute the confidence-weighted
        # utility from the new totals. We compute in Python (Wilson lower
        # bound) rather than inline SQL so the scoring logic lives in one
        # place and is unit-testable on its own.
        column = "adoption_count" if adopted else "correction_count"
        self._conn.execute(
            f"UPDATE memories SET {column} = {column} + 1 WHERE id = ?",
            (memory_id,),
        )
        row = self._conn.execute(
            "SELECT adoption_count, correction_count FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is not None:
            new_utility = utility_score(row[0], row[1])
            self._conn.execute(
                "UPDATE memories SET utility_score = ? WHERE id = ?",
                (new_utility, memory_id),
            )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def get_all(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, trigger, judgment, reasoning, domain, "
            "utility_score, adoption_count, correction_count FROM memories"
        ).fetchall()
        return [
            {
                "id": r[0],
                "trigger": r[1],
                "judgment": r[2],
                "reasoning": r[3],
                "domain": r[4],
                "utility_score": r[5],
                "adoption_count": r[6],
                "correction_count": r[7],
            }
            for r in rows
        ]

    def generate_id(self) -> str:
        return "mem_" + uuid.uuid4().hex[:12]
