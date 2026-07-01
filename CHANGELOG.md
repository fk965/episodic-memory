# Changelog

## 0.2.0 (2026-07-01)

Theme: make the docs and code tell the same story.

### Changed
- **Confidence-weighted utility.** `utility_score` is now the Wilson score lower bound on the adoption proportion, not the raw `adoption / (adoption + correction)` ratio. A judgment adopted 100× now outranks one adopted 1× at the same ratio — delivering the "repeatedly validated ranks higher" promise the old formula could not. Scoring lives in a new `episodic_memory.scoring` module and is unit-tested independently. This changes ranking output, hence the minor version bump.
- **Honest benchmark.** `benchmarks/judgment_recall.py` no longer only reports a perfect-signal run (which is circular — the labels decide the ranking). It now runs a perfect-signal *ceiling* plus noisy-signal regimes (20% / 30% flipped) to test whether the flywheel survives imperfect feedback. README reframed accordingly; the old "40% → 90%" headline is gone.
- **Docs de-overclaimed.** Corrected the storage description (plain SQLite + in-Python KNN, not sqlite-vec), scoped the multi-agent "shared memory" claim to single-process, and promoted "you must supply a reliable verification signal" from a footnote to a first-class section.

### Fixed
- Offline embed tests no longer error when `sentence-transformers` is absent (`SentenceTransformer` is now defined as `None` in the import fallback).
- Persistence tests now genuinely close and reopen the store at the same path, instead of reusing one open connection.
- Removed unused `Sequence` imports.

## 0.1.0 (2026-06-16)

- Initial release
- `store()` — save a judgment with trigger, reasoning, decision
- `search()` — semantic retrieval of past judgments
- `verify()` — utility feedback (was the judgment adopted or corrected?)
- Local SQLite storage with in-Python vector search, zero external API dependencies
- Default Sentence-BERT embeddings (fully offline capable)