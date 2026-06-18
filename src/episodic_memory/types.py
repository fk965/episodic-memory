from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    """A single episodic memory entry."""

    id: str
    trigger: str
    judgment: str
    reasoning: str
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    utility_score: float = 0.0
    adoption_count: int = 0
    correction_count: int = 0


@dataclass
class SearchResult:
    """Result from a semantic search query."""

    id: str
    judgment: str
    reasoning: str
    trigger: str
    domain: str | None
    distance: float
    metadata: dict[str, Any]
    utility_score: float


@dataclass
class Triple:
    """Exportable (context, judgment, outcome) triple for fine-tuning."""

    context: str
    judgment: str
    outcome: dict[str, Any]