"""Episodic Memory — a memory layer that remembers how your agent decided."""

from __future__ import annotations

from typing import Any, Sequence

from . import embed as embed_module
from .storage import Storage
from .types import MemoryRecord, SearchResult, Triple


class EpisodicMemory:
    """In-process episodic memory store for agent judgments.

    Usage::

        memory = EpisodicMemory()
        mem_id = memory.store(
            trigger="User asks to modify config.json",
            judgment="Production config changes must be confirmed first",
            reasoning="Direct config writes have caused outages before.",
            domain="ops",
        )
        results = memory.search("Can I edit production config?")
        memory.verify(mem_id, adopted=True)
    """

    def __init__(
        self,
        embedder: object | None = None,
        db_path: str = ":memory:",
    ):
        self._storage = Storage(db_path=db_path)
        self._embedder = embedder or embed_module.Embedder()

    # ── Public API ──────────────────────────────────────────────

    def store(
        self,
        trigger: str,
        judgment: str,
        reasoning: str,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new episodic memory.

        Args:
            trigger: What situation prompted this judgment.
            judgment: The judgment or decision itself.
            reasoning: Why this judgment was made.
            domain: Optional domain tag (e.g. "ops", "coding", "security").
            metadata: Optional dict of arbitrary metadata.

        Returns:
            Memory ID string (e.g. "mem_abc123").
        """
        if not trigger or not judgment or not reasoning:
            raise ValueError("trigger, judgment, and reasoning must be non-empty")
        memory_id = self._storage.generate_id()
        text_to_embed = f"{trigger}\n{judgment}\n{reasoning}"
        vector = self._embedder.embed(text_to_embed)
        self._storage.insert(
            memory_id=memory_id,
            trigger=trigger,
            judgment=judgment,
            reasoning=reasoning,
            embedding=vector,
            domain=domain,
            metadata=metadata,
        )
        return memory_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        min_score: float = 0.0,
        use_utility: bool = False,
        utility_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Search past judgments by semantic similarity.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.
            domain: Optional domain filter.
            min_score: Minimum similarity threshold (0.0 = no filter).
            use_utility: Rank by relevance *and* proven usefulness, so
                judgments that were repeatedly adopted surface higher.
            utility_weight: How strongly utility boosts ranking (only used
                when use_utility is True).

        Returns:
            List of SearchResult, ordered by relevance descending.
        """
        query_vec = self._embedder.embed(query)
        results = self._storage.knn_search(
            query_vec,
            top_k=top_k,
            domain=domain,
            min_score=min_score,
            use_utility=use_utility,
            utility_weight=utility_weight,
        )
        return [
            SearchResult(
                id=r["id"],
                judgment=r["judgment"],
                reasoning=r["reasoning"],
                trigger=r["trigger"],
                domain=r["domain"],
                distance=r["distance"],
                metadata=r["metadata"],
                utility_score=r["utility_score"],
            )
            for r in results
        ]

    def verify(
        self,
        memory_id: str,
        adopted: bool,
        user_correction: str | None = None,
    ) -> None:
        """Record whether a retrieved judgment was useful.

        Args:
            memory_id: ID returned by store().
            adopted: True if the agent followed this judgment.
            user_correction: User's correction if the judgment was wrong.
        """
        if self._storage.get(memory_id) is None:
            raise KeyError(f"No memory with id {memory_id!r}")
        self._storage.record_verification(memory_id, adopted, user_correction)

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a single memory record by ID."""
        row = self._storage.get(memory_id)
        if row is None:
            return None
        return MemoryRecord(**row)

    def count(self) -> int:
        """Number of stored memories."""
        return self._storage.count()

    def close(self) -> None:
        """Close the underlying storage connection."""
        self._storage.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def export_triples(
        self,
        min_utility: float = 0.0,
    ) -> list[Triple]:
        """Export (context, judgment, outcome) triples for fine-tuning.

        Only exports memories whose utility strictly exceeds the threshold,
        so unverified memories (utility 0.0) are excluded by default.
        """
        triples: list[Triple] = []
        for mem in self._storage.get_all():
            if mem["utility_score"] <= min_utility:
                continue
            triples.append(Triple(
                context=mem["trigger"],
                judgment=mem["judgment"],
                outcome={
                    "utility_score": mem["utility_score"],
                    "adoption_count": mem["adoption_count"],
                    "correction_count": mem["correction_count"],
                },
            ))
        return triples