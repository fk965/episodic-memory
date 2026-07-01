"""Benchmark: does utility-weighted retrieval beat pure cosine?

Premise (the episodic-memory thesis): an agent that *remembers which of its
past judgments proved correct* should retrieve better advice than one that
ranks memories by semantic similarity alone.

Setup: each scenario has two competing judgments that look alike to an
embedder — a CORRECT one (repeatedly adopted -> high utility) and a WRONG one
(corrected -> low utility). A held-out query restates the scenario. We measure
how often, and how highly, the CORRECT judgment is retrieved under each
ranking strategy.

This uses the REAL embedding model, not the test fake — a fake embedder would
make the semantic comparison meaningless.
"""

from __future__ import annotations

import random
import statistics

from episodic_memory import EpisodicMemory

# (scenario query, correct judgment, wrong-but-similar judgment)
SCENARIOS = [
    (
        "A teammate wants to push a hotfix to production on Friday evening",
        "Hold non-critical deploys until Monday so the full team can respond to incidents",
        "Friday evening deploys are fine, just merge it and go home",
    ),
    (
        "Should we mock the database in our integration tests?",
        "Integration tests must hit a real database; mocks hid a broken migration before",
        "Mock the database everywhere, it makes tests faster",
    ),
    (
        "User asks to delete all rows in the orders table to fix a bug",
        "Never run an unscoped delete on production data; take a backup and scope the fix",
        "Just run DELETE FROM orders, it will clear the bad state quickly",
    ),
    (
        "We need to store the user's session token in the browser",
        "Use httpOnly secure cookies so JavaScript cannot read the session token",
        "Put the session token in localStorage so the SPA can read it easily",
    ),
    (
        "A dependency has a new major version available, should we upgrade now?",
        "Pin the current version and upgrade behind tests; a blind major bump broke us before",
        "Always upgrade to the latest major immediately to stay current",
    ),
    (
        "How should we handle an API key that was committed to the repo?",
        "Rotate the key immediately and purge it from history; commits are forever public",
        "Just delete the line in a new commit, the key is gone now",
    ),
    (
        "The build is failing on a flaky test, can we skip it to ship?",
        "Quarantine the flaky test and file a fix; never delete tests to turn CI green",
        "Delete the failing test so the build passes and we can ship",
    ),
    (
        "Customer wants their personal data exported, what format is safe?",
        "Export only fields the user owns, over an authenticated channel with an audit log",
        "Dump the whole user table to a CSV and email it over for convenience",
    ),
    (
        "Should this new internal service be exposed without authentication?",
        "Require auth even for internal services; unauthenticated endpoints get abused",
        "Skip auth for internal services, the network is trusted enough",
    ),
    (
        "We hit a merge conflict during a rebase, how do we resolve it?",
        "Resolve conflicts by understanding both sides; never discard the other branch's work",
        "Just run git checkout --theirs on everything to make the conflict disappear",
    ),
]

# Verification events fed to each judgment.
ROUNDS = 5


def build_memory(noise: float = 0.0, seed: int = 0) -> tuple[EpisodicMemory, dict]:
    """Store each scenario's competing judgments and feed the flywheel.

    ``noise`` is the probability that any single verification event is
    *flipped* — a correct judgment recorded as corrected, or a wrong one
    recorded as adopted. noise=0.0 is the perfect-oracle ceiling; a realistic
    feedback source is noisy, which is the case that actually matters.
    """
    rng = random.Random(seed)
    mem = EpisodicMemory()  # real embedder by default
    correct_ids = {}
    for i, (_query, correct, wrong) in enumerate(SCENARIOS):
        cid = mem.store(f"scenario {i}", correct, "validated in practice", domain="eng")
        wid = mem.store(f"scenario {i}", wrong, "looked plausible", domain="eng")
        # Correct judgment: mostly adopted, but each signal may be flipped.
        for _ in range(ROUNDS):
            mem.verify(cid, adopted=rng.random() >= noise)
        # Wrong judgment: mostly corrected, same flip probability.
        for _ in range(ROUNDS):
            corrected = rng.random() >= noise
            mem.verify(wid, adopted=not corrected)
        correct_ids[i] = cid
    return mem, correct_ids


def evaluate(mem: EpisodicMemory, correct_ids: dict, use_utility: bool) -> dict:
    hits_at_1 = 0
    correct_ranks = []
    for i, (query, _c, _w) in enumerate(SCENARIOS):
        results = mem.search(query, top_k=10, use_utility=use_utility)
        ids = [r.id for r in results]
        target = correct_ids[i]
        if ids and ids[0] == target:
            hits_at_1 += 1
        if target in ids:
            correct_ranks.append(ids.index(target) + 1)
        else:
            correct_ranks.append(len(ids) + 1)
    return {
        "precision_at_1": hits_at_1 / len(SCENARIOS),
        "mean_rank_of_correct": statistics.mean(correct_ranks),
    }


def _report(title: str, noise: float, seed: int) -> None:
    mem, correct_ids = build_memory(noise=noise, seed=seed)
    baseline = evaluate(mem, correct_ids, use_utility=False)
    flywheel = evaluate(mem, correct_ids, use_utility=True)
    print(f"\n{title}  (signal noise = {noise:.0%})")
    print("=" * 56)
    print(f"{'metric':<28}{'cosine':>12}{'+utility':>12}")
    print("-" * 56)
    print(f"{'precision@1 (higher better)':<28}"
          f"{baseline['precision_at_1']:>12.2f}{flywheel['precision_at_1']:>12.2f}")
    print(f"{'mean rank of correct (lower)':<28}"
          f"{baseline['mean_rank_of_correct']:>12.2f}{flywheel['mean_rank_of_correct']:>12.2f}")
    print("=" * 56)
    mem.close()


def main() -> None:
    print("Building memory stores with the real embedding model...")
    print(f"{len(SCENARIOS)} scenarios, two competing judgments each.")
    print(
        "\nThe 'cosine' column is the honest finding: a plain embedder can't\n"
        "tell a correct judgment from a plausible-but-wrong one. The question\n"
        "is whether utility feedback helps — and whether it survives NOISE in\n"
        "that feedback, since real adoption signals are never perfect."
    )
    # Perfect-oracle ceiling (what the flywheel can do with clean signal)...
    _report("PERFECT SIGNAL (ceiling, not a real-world claim)", noise=0.0, seed=1)
    # ...and the case that actually matters: imperfect feedback.
    _report("NOISY SIGNAL (realistic)", noise=0.2, seed=1)
    _report("NOISY SIGNAL (realistic)", noise=0.3, seed=7)


if __name__ == "__main__":
    main()
