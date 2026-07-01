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
  <a href="https://pypi.org/project/episodic-judgment/"><img src="https://img.shields.io/pypi/v/episodic-judgment" alt="PyPI"></a>
  <a href="https://github.com/fk965/episodic-memory/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fk965/episodic-memory" alt="License"></a>
  <a href="https://github.com/fk965/episodic-memory/actions"><img src="https://img.shields.io/github/actions/workflow/status/fk965/episodic-memory/test.yml" alt="CI"></a>
  <a href="https://codecov.io/gh/fk965/episodic-memory"><img src="https://img.shields.io/codecov/c/github/fk965/episodic-memory" alt="Coverage"></a>
  <a href="https://github.com/fk965/episodic-memory"><img src="https://img.shields.io/github/stars/fk965/episodic-memory" alt="Stars"></a>
	  <a href="https://dev.to/fk965/ai-agents-remember-facts-but-cant-learn-from-mistakes-heres-a-fix-tags-ai-agents-2ml9"><img src="https://img.shields.io/badge/blog-dev.to-8B5CF6" alt="Blog"></a>
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
| **Feedback loop** | Optimized for storing/updating facts | Utility-weighted: was the judgment adopted or corrected? |
| **Ranking** | Primarily semantic similarity | Similarity × (1 + α · utility), where utility is confidence-weighted |
| **Optimized for** | Recalling what's relevant now | Surfacing what has held up over repeated use |
| **Training data output** | Not designed for this | Export (context, judgment, outcome) triples for fine-tuning |

**The key insight:** semantic memory answers "what is relevant?" Episodic memory adds "what has held up when we acted on it before?"

---

## Does the flywheel help? (Benchmark)

We built a synthetic judgment-recall task: 10 scenarios, each with two competing judgments that look alike to an embedder — one correct, one plausible-but-wrong. A held-out query restates the scenario, and we measure how often the **correct** judgment ranks first.

The honest starting point is the **cosine baseline**: a plain embedder picks the correct judgment only a minority of the time, because it can't tell "looks relevant" from "is actually correct." That gap is the whole reason this library exists.

The real question is not "can utility feedback close that gap under perfect labels" (trivially yes — if you already know which judgment is correct, ranking by that label wins by construction). It's whether the flywheel still helps when the feedback signal is **noisy**, since real adoption/correction signals always are. So the benchmark runs three regimes:

- **Perfect signal** — a ceiling, not a real-world claim.
- **Noisy signal (20% flipped)** — a realistic feedback source.
- **Noisy signal (30% flipped)** — a pessimistic one.

Run it yourself (downloads the Sentence-BERT weights on first run):

```bash
pip install episodic-judgment sentence-transformers
python benchmarks/judgment_recall.py
```

> **This is a synthetic proof-of-concept, not production validation.** The scenarios are hand-written and the "correct" labels are ours. Real-world results depend entirely on your domain, embedding quality, and — most of all — how reliable your verification signal is. Treat the script as a starting point to adapt to your own data, not as evidence the technique will work for you.

---

## Quickstart

```bash
pip install episodic-judgment
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

The ranking formula is `rank_score = cosine_similarity × (1 + α · utility_score)`. The `utility_score` is **not** a raw adoption ratio — that would score a judgment adopted once the same as one adopted a hundred times, so "repeatedly validated" would mean nothing. Instead it's the **Wilson score lower bound** on the adoption proportion, which rises with both the adoption rate *and* the amount of evidence. A judgment adopted 100× outranks one adopted 1× even though both have a 100% adoption rate.

- ⍺ = 0.5 by default, adjustable via `utility_weight`
- Raw cosine similarity is always the base — utility can't override relevance, only disambiguate it
- `use_utility=False` (the default) preserves pure vanilla behavior

---

## When to use it

| Scenario | Without Episodic Memory | With Episodic Memory |
|---|---|---|
| Agent repeatedly makes the same mistake | Each run starts from scratch | Past judgment is retrieved — "last time this broke" |
| Agent needs to know your operation style | Hardcoded in system prompt, never evolves | Utility feedback loop reinforces what works |
| Onboarding new agents | Every agent needs its own instructions | A shared judgment store they can read from (see the concurrency note below) |
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

## The hard part is the verification signal

Be clear-eyed about where the difficulty lives. This library packages the *easy* 20%: storing judgments and ranking them by a confidence-weighted score. The *hard* 80% — deciding, reliably, whether a past judgment was actually adopted or should have been corrected — is left to you, via `verify()`.

The flywheel is only as good as that signal. If your `adopted=True/False` calls are guesses, the utility scores are noise and you have a plain vector store with extra steps. Before adopting this library, make sure you have a trustworthy source of that signal: an explicit human thumbs-up/down, a downstream success metric, a test that passes or fails, a user who edits the agent's action. That source — not this code — is what determines whether episodic memory works for you.

---

## When NOT to use it

- **If your agent only needs facts and preferences** — use Mem0 or a vector store. This library is not designed for semantic memory.
- **If you have no reliable way to verify judgments** — see the section above. Without a real feedback signal the utility scores stay flat and this degrades to a plain vector store.
- **For concurrent multi-process access** — the store wraps a single SQLite connection and is not built for many agents writing at once. It's safe for one process (or read-mostly sharing); heavy concurrent writes need a real database behind the `Storage` interface.
- **For more than ~10K records** — the current `knn_search` scans all rows in Python. At scale, swap in an ANN backend (e.g. `sqlite-vec` or `pgvector`) behind the `Storage` interface.

---

## Project Status

Early release (v0.2.0). The core loop (store → search → verify → weighted recall) is stable and tested, but expect additions before 1.0:

- Pluggable ANN backends (sqlite-vec, pgvector) for >10K records
- Concurrent / multi-process access
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