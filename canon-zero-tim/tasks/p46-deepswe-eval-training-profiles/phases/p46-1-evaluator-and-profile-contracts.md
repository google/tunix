# P46.1 — hardened evaluator and frozen workload contracts

- Status: completed locally; target unverified

## Finding

- Confirmed: the current P39.5/P44.12 diff already supplies request abort, one-trajectory clocks, bounded reward/cleanup, shared rollout-batch deadlines, R2E delete confirmation, durable training trajectories, and device-resident optimizer defaults.
- Confirmed: `yuxzhang/swe-evaluation-dev@5113c0fb788a2c1f31344f6c3b1265d069bf11ea` streams result summaries and resumes n-sample evaluation, but enables prefix caching, keys resume too loosely, does not persist the complete trajectory, and can classify an incomplete n-sample task.
- Decision: preserve the immutable 1851-row clean source and produce versioned evaluation reports. Never overwrite or silently substitute the production whitelist.

## Execution

1. Separate pure evaluation manifest, resume, aggregation, and artifact functions from TPU/Kubernetes initialization so they are CPU-testable.
2. Persist one full redacted trajectory record per `(task_key, sample_index)` with model/data/source fingerprints and lifecycle timings.
3. Require exact configuration fingerprints for resume; reject duplicates, incompatible records, excess samples, and incomplete final classification.
4. Emit complete, Q4-learnable, Q32-candidate, all-pass, all-fail, broken, and incomplete reports with exact counts and SHA-256 digests.
5. Use Qwen3-4B-Instruct-2507, 16K response, temperature 1.0, prefix cache disabled, 4-task x 16-sample physical shards, concurrency 64, and a 3600-second shard boundary.
6. Freeze the Q4 three-update and Q32 full-training command contracts at 16K response and their separately signed deadlines.

## Exit gate

- Command: `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh`
- Pass: evaluator tests prove exact-N/fingerprint/resume/report behavior; six manifests render and pass normalized 64/256 parity; adjacent DeepSWE gates remain green.
- Fail: keep P46 active, preserve the failing evidence in `log.md`, and do not render or publish a launchable manifest.

## Result

Passed locally on 2026-08-13. The dedicated P46 gate returned
`P46_DEEPSWE_PROFILES_CPU_PASS cases=17`; six temporary JobSets rendered for
the three workload families on 64 and 256 chips. P39, P43, P44 and P34 static
adjacent gates passed. This is package evidence only and produced no target
claim.
