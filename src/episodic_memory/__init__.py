"""Episodic Memory — public API surface.

Usage::

    from episodic_memory import EpisodicMemory
"""

from .core import EpisodicMemory
from .types import MemoryRecord, SearchResult, Triple

__all__ = [
    "EpisodicMemory",
    "MemoryRecord",
    "SearchResult",
    "Triple",
]

__version__ = "0.1.0"