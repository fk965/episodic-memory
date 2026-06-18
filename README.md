<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fk965/episodic-memory/main/docs/logo-dark.svg">
    <img alt="Episodic Memory" src="https://raw.githubusercontent.com/fk965/episodic-memory/main/docs/logo-light.svg" width="60%">
  </picture>
</p>

<h3 align="center">
A memory layer that remembers <em>how</em> your agent decided,<br>
not just <em>what</em> it knows.
</h3>

<p align="center">
  <a href="https://pypi.org/project/episodic-memory/"><img src="https://img.shields.io/pypi/v/episodic-memory" alt="PyPI"></a>
  <a href="https://github.com/fk965/episodic-memory/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fk965/episodic-memory" alt="License"></a>
  <a href="https://github.com/fk965/episodic-memory/actions"><img src="https://img.shields.io/github/actions/workflow/status/fk965/episodic-memory/test.yml" alt="CI"></a>
  <a href="https://codecov.io/gh/fk965/episodic-memory"><img src="https://img.shields.io/codecov/c/github/fk965/episodic-memory" alt="Coverage"></a>
  <a href="https://github.com/fk965/episodic-memory"><img src="https://img.shields.io/github/stars/fk965/episodic-memory" alt="Stars"></a>
</p>

---

## Why?

Most memory systems for AI agents store **facts**: "user prefers Python", "the API key is at X", "the last conversation was about Y".

They are **semantic memories** — they know *what* is true.

But agents don't just need to know things. They need to **judge** things:

- "Should I modify this config file directly, or ask the user first?"
- "The last time I changed database credentials, it broke the connection — try a different approach."
- "I've been burned by skipping tests in CI before. Never again."

These are **episodic memories**: decisions made, reasoning used, outcomes observed. They capture *how* the agent should behave, not just *what* is true.

**Episodic Memory** is a lightweight Python library that adds this missing dimension to your agent stack.

---

## How is this different?

| | Semantic Memory (Mem0, RAG, vector stores) | Episodic Memory (this) |
|---|---|---|
| **What it stores** | Facts, preferences, conversation history | Decisions, judgments, reasoning chains |
| **Query pattern** | "What does the user prefer?" | "How should I handle this situation?" |
| **Feedback loop** | None — stored facts are trusted | Utility-weighted: was the judgment adopted or corrected? |
| **Ranking** | Cosine similarity only | Dynamic: similarity × (1 + α · utility_score) |
| **Learning over time** | One-shot recall | Accumulates verified judgments — what works, what doesn't |
| **Training data output** | Not designed for this | Export (context, judgment, outcome) triples for fine-tuning |

**The key insight:** semantic memory answers "what is relevant?" Episodic memory answers "what has been proven correct?"

---

## Does it actually work? (Benchmark)

We built a synthetic judgment-recall task: 10 scenarios, each with two competing judgments that look alike to an embedder — one correct (validated), one wrong (corrected). Then we measure how often the **correct** judgment ranks first under each strategy.

```
metric                            cosine    +utility flywheel
--------------------------------------------------------
precision@1 (higher better)         0.40        0.90
mean rank of correct (lower)        1.90        1.30
```

Pure cosine retrieval finds the right judgment only **40%** of the time — because the embedder can't distinguish "looks relevant" from "is actually correct." Adding the utility flywheel brings it to **90%**.

The benchmark is fully reproducible — run it yourself:

```bash
pip install episodic-memory sentence-transformers
python benchmarks/judgment_recall.py
```

> **Note:** This is a synthetic proof-of-concept, not production validation. Real-world results depend on your domain, embedding quality, and verification coverage. The script is meant to be a starting point — we encourage you to adapt it to your own use case.

---

## Quickstart

```bash
pip install episodic-memory
```

```python
from episodic_memory import EpisodicMemory

# Create an in-process memory store
memory = EpisodicMemory()

# Save a judgment
memory.store(
    trigger="User asks agent to modify config.json in production",
    judgment="Production config changes must be confirmed with the user first",
    reasoning="Direct config writes have caused outages before. The agent should always propose, not execute.",
    domain="ops",
)
# → "mem_abc123"

# Later, when a similar situation arises, query past judgments
results = memory.search(
    query="Can I edit the production config file?",
    top_k=3,
)

for r in results:
    print(f"  [{r.distance:.2f}] {r.judgment}")
# → [0.21] Production config changes must be confirmed with the user first

# Did this judgment actually help? Let the feedback loop know.
memory.verify("mem_abc123", adopted=True)

# Next time, the utility-weighted search will rank this judgment higher.
results = memory.search("Can I edit the production config?", use_utility=True)
```

---

## The Utility Flywheel

The core idea: judgments that have been repeatedly validated should rank higher than untested or disproven ones, even when the semantic similarity is the same.

```
                      ┌──────────────────┐
                      │  store() saves   │
                      │  a judgment      │
                      └────────┬─────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────┐
  │  search(query, use_utility=True)        │
  │  ranks by sim × (1 + α · utility)      │
  └─────────┬───────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────┐
  │  Agent follows the judgment — or not?   │
  └─────────┬───────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────┐
  │  verify(id, adopted=True/False)         │
  │  updates utility_score                  │
  └─────────┬───────────────────────────────┘
            │
            ▼
  search() with use_utility=True (loop back)
```

The formula is simple: `rank_score = cosine_similarity × (1 + α · utility_score)`, where utility_score = adoption_count / (adoption_count + correction_count).

- ⍺ = 0.5 by default, adjustable via `utility_weight`
- Raw cosine similarity is always the base — utility can't override relevance, only disambiguate it
- `use_utility=False` (the default) preserves pure vanilla behavior

---

## When to use it

| Scenario | Without Episodic Memory | With Episodic Memory |
|---|---|---|
| Agent repeatedly makes the same mistake | Each run starts from scratch | Past judgment is retrieved — "last time this broke" |
| Agent needs to know your operation style | Hardcoded in system prompt, never evolves | Utility feedback loop reinforces what works |
| Onboarding new agents | Every agent needs its own instructions | Shared memory of accumulated operational wisdom |
| Debugging agent behavior | "Why did it do that?" is guesswork | Every judgment carries its reasoning chain |

---

## API Reference

### `EpisodicMemory(embedder=None, db_path=":memory:")`

Create a memory store. Defaults to in-memory SQLite; pass `db_path` for persistence.

| Parameter | Default | Description |
|---|---|---|
| `embedder` | `None` (auto: SentenceTransformer) | Custom embedding function: `callable(str) → list[float]` |
| `db_path` | `":memory:"` | Path to SQLite database file |

### `store(trigger, judgment, reasoning, domain=None, metadata=None) → str`

Save a new episodic memory. Raises `ValueError` if trigger, judgment, or reasoning is empty.

### `search(query, top_k=5, domain=None, min_score=0.0, use_utility=False, utility_weight=0.5) → list[SearchResult]`

Search past judgments by semantic similarity, optionally weighted by proven utility.

| Parameter | Default | Description |
|---|---|---|
| `query` | required | Natural language query |
| `top_k` | `5` | Max results |
| `domain` | `None` | Filter by domain |
| `min_score` | `0.0` | Minimum similarity threshold |
| `use_utility` | `False` | Rank by relevance × utility — the flywheel |
| `utility_weight` | `0.5` | How strongly utility boosts the ranking |

Returns list of `SearchResult(id, judgment, reasoning, trigger, domain, distance, metadata, utility_score)`.

### `verify(memory_id, adopted, user_correction=None) → None`

Record whether a retrieved judgment was useful. Raises `KeyError` if the memory_id doesn't exist.

| Parameter | Required | Description |
|---|---|---|
| `memory_id` | ✅ | ID returned by `store()` |
| `adopted` | ✅ | `True` if agent followed this judgment |
| `user_correction` | | User's correction if the judgment was wrong |

### `export_triples(min_utility=0.0) → list[Triple]`

Export (context, judgment, outcome) triples for fine-tuning. Only exports memories with `utility_score > min_utility` (strictly greater, so unverified memories are excluded by default).

### `close()`

Close the underlying storage connection. Also works as a context manager: `with EpisodicMemory() as m: ...`

---

## When NOT to use it

- **If your agent only needs facts and preferences** — use Mem0 or a vector store. This library is not designed for semantic memory.
- **If you don't have a way to verify judgments** — the flywheel needs feedback. Without `verify()`, the utility scores stay at 0 and the library behaves like a plain vector store.
- **For more than ~10K records** — the current `knn_search` scans all rows in Python. At scale, swap in `sqlite-vec` or `pgvector` (the interface is just `Storage`).

---

## Project Status

This is an early release (v0.1.0). The API is stable for the core loop (store → search → verify → weighted recall), but expect additions before 1.0:

- Pluggable ANN backends (sqlite-vec, pgvector)
- Time-decayed utility weighting
- Memory consolidation (merge duplicate judgments)
- Streaming export for online fine-tuning

---

## Related Work

- **Mem0** (⭐58k) — Universal semantic memory layer for AI agents. Complementary: use Mem0 for facts, episodic-memory for judgments.
- **LangGraph** (⭐35k) — Stateful agent orchestration. Integrates via its `MemorySaver` interface.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

Questions, ideas, or bugs? Open an [issue](https://github.com/fk965/episodic-memory/issues).

---

<p align="center">
  ⭐ If this project resonates, star it on GitHub — it helps others find it.
</p>