# State

- Status: active
- Objective: Deliver one hardened DeepSWE clean-data evaluator and three reproducible Qwen3 workload families, each renderable for 64 or 256 TPU chips without changing workload semantics.
- Definition of done: Evaluation, renderer, profile-parity, timeout, dataset-fingerprint, trajectory-artifact, and adjacent DeepSWE CPU gates pass; six non-launchable manifests render from the three workload families; no target claim is made without a real cluster run.
- Task directory: `canon-zero-tim/tasks/p46-deepswe-eval-training-profiles/`
- Directory state: P46.1-P46.5 are published through `a4d165e854cc4c2320d8120e89aed185eaf61465`; the invalid-attempt retry and exact campaign finalizer are published by `a642ab267425a5b08b0cebb6e12c607f50f71831`
- Current phase: [P46.5 — true reward-only evaluation](phases/p46-5-reward-only-evaluation.md)
- Last verified fact: Returned 256-chip run `p46e25608` at source `bdc9681824743911d0691659604dec090dd42bc4` initialized Qwen3-4B DP32 x TP8 and attempted l0/p0, but produced only 62 valid trajectories plus two `MODEL_TIMEOUT` attempts. The old evaluator incorrectly emitted `P46_EVAL_SUBSHARD_PASS` because invalid attempts completed resume identities. The local repair persists consecutive attempts, retries invalid identities, rejects attempts after a valid result, and emits `P46_EVAL_PHYSICAL_INCOMPLETE`/nonzero until the exact valid physical count is present. A fail-closed global finalizer now requires all 58 exact-N16 summaries before emitting the merged candidate manifests. P46 33, P34, P39 and P44 gates pass; Python compile and diff checks pass.
- Next action: Read back the exact operator SHA containing `a642ab26`, start a new evaluation run id, rerun l0/p0 from all 64 identities, return the full trajectory JSONL archive/digests, then continue all 463 physical JobSets and require `P46_EVAL_CAMPAIGN_PASS` for 29,616 valid trajectories and 58 logical reports.
- Blockers: Full `p46e25608` trajectory JSONL is not present in git; the fixed 256-chip rerun, full 1851 x N16 campaign, L3 paired statistics, Kubernetes cleanup/throughput, Qwen3-4B three updates, and Qwen3-32B training remain target-unverified.
- Key artifacts: `HANDOFF.md`, `plan.md`, `log.md`, `../../cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`, `phases/p46-5-reward-only-evaluation.md`, dependency ledgers `../p39-deepswe-production/` and `../p44-deepswe-qwen4b-parity/`
- Updated: 2026-08-13T09:20:01Z
