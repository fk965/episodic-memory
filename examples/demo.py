#!/usr/bin/env python3
"""
Episodic Memory Demo

Demonstrates the full workflow: store → search → verify → export.

Run:
    pip install episodic-memory
    python examples/demo.py
"""

from episodic_memory import EpisodicMemory


def main():
    print("=" * 60)
    print("  Episodic Memory — Demo")
    print("  Remembering how agents decide, not just what they know")
    print("=" * 60)

    # ── Initialize ──────────────────────────────────────────────
    memory = EpisodicMemory()
    print("\n[1] Initialized in-memory store.\n")

    # ── Store judgments ─────────────────────────────────────────
    judgments = [
        {
            "trigger": "User asks the agent to modify a production config file",
            "judgment": "Production config changes must be confirmed with the user first",
            "reasoning": "Direct config writes have caused outages in the past. "
                         "The agent should always propose changes, not execute them.",
            "domain": "ops",
        },
        {
            "trigger": "User asks the agent to deploy code on Friday afternoon",
            "judgment": "Avoid deploying on Friday unless it's an emergency",
            "reasoning": "If something breaks, fewer engineers are available to roll back over the weekend.",
            "domain": "ops",
        },
        {
            "trigger": "Agent needs to write a Python function",
            "judgment": "Always add type hints and a docstring to new functions",
            "reasoning": "Type hints catch bugs early, and docstrings help other agents "
                         "(and humans) understand the code without reading the implementation.",
            "domain": "coding",
        },
        {
            "trigger": "User asks for the agent's opinion on an approach",
            "judgment": "Give the honest assessment even if it contradicts the user's idea",
            "reasoning": "The user explicitly wants the agent's expertise. "
                         "Sugar-coating reduces trust over time.",
            "domain": "communication",
        },
    ]

    ids = []
    for j in judgments:
        mem_id = memory.store(**j)
        ids.append(mem_id)
        print(f"  Stored: [{j['domain']}] {j['judgment'][:55]}...")
    print(f"\n  → {len(ids)} total memories stored.\n")

    # ── Search ──────────────────────────────────────────────────
    print("[2] Searching: 'Should I deploy on a Friday?'\n")
    results = memory.search("Should I deploy on a Friday?")
    for r in results:
        print(f"  distance={r.distance:.3f}  [{r.domain}]")
        print(f"  → {r.judgment}")
        print(f"    (trigger: {r.trigger[:50]}...)")
        print()

    # ── Verify ──────────────────────────────────────────────────
    print("[3] Recording utility feedback (verification)...\n")
    # The deployment judgment was adopted — user agreed
    memory.verify(ids[1], adopted=True)
    print(f"  ✓ Verified 'deploy on Friday' → ADOPTED")

    # The config judgment was adopted too
    memory.verify(ids[0], adopted=True)
    print(f"  ✓ Verified 'production config' → ADOPTED")

    # The type hints judgment was corrected — user prefers simpler code
    memory.verify(ids[2], adopted=False, user_correction="Just add type hints, docstrings are unnecessary for internal code")
    print(f"  ✗ Verified 'type hints + docstring' → CORRECTED")

    # Show accumulated scores
    print("\n  After verification:")
    for mem_id in ids:
        record = memory.get(mem_id)
        if record:
            print(f"    {record.judgment[:50]:<50}  "
                  f"score={record.utility_score:.2f}  "
                  f"(adopted={record.adoption_count}, corrected={record.correction_count})")

    # ── Export ──────────────────────────────────────────────────
    print("\n[4] Exporting training triples...\n")
    triples = memory.export_triples(min_utility=0.0)
    print(f"  {len(triples)} triples exported:")
    for t in triples:
        print(f"  context:  {t.context[:50]}...")
        print(f"  judgment: {t.judgment[:50]}...")
        score = t.outcome.get("utility_score", 0)
        adopted = t.outcome.get("adoption_count", 0)
        corrected = t.outcome.get("correction_count", 0)
        print(f"  outcome:  score={score:.2f} (adopted={adopted}, corrected={corrected})")
        print()

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Total memories: {memory.count()}")
    print(f"  Total triples:  {len(triples)}")
    print("=" * 60)
    print("\n✨ Done. Install with:  pip install episodic-memory")
    print("   Docs:               github.com/fk965/episodic-memory")


if __name__ == "__main__":
    main()