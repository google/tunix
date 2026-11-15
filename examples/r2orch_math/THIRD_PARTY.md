# R²Orch reference provenance

Reference: **R²Orch: Recursive Self-Improvement for Agent Orchestration via
Self-Play within a Skill-Internalization Loop**, arXiv:2610.22740.

Repository: https://github.com/beita6969/R2Orch

Pinned revision: `12ceb7563aa50439687f21ba5da60ad74a75eb89`.

The graph runtime and training integration here are independently implemented
for Tunix. They are a Director-only adaptation, with explicit differences listed
in the README. The data importer reads original committed JSONL records from a
user-provided checkout; no original benchmark data or training source is vendored
in this directory. Source notices and upstream dataset terms remain applicable
to those external assets. The CPU fixture consists of synthetic local tasks.

The production AIME grader uses the exact pinned upstream mathematical scoring
definitions through the explicit reference-source adapter. This dependency is
recorded in the run manifest. Test oracles use the same pinned definitions to
detect numerical differences rather than comparing Tunix functions to themselves.
