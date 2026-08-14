# P46.6 — persistent full clean-data washing

- Status: active

## Trigger

Returned 128-chip run `p46e12804` proved that Q4 rollout, reward-only capture,
clean-data selection, and seven solved samples execute, but it also exposed two
campaign blockers: the compatibility regex could manufacture malformed XML,
and one model initialization/JIT cycle was paid for only 64 trajectories.

## Decisions

- Q4 action compatibility is explicit and exact; Q32 remains strict/off.
- A deterministic repair is execution compatibility with provenance. An
  ambiguous or invalid model tool call is a completed model outcome, not an
  infrastructure retry. Adapter-created corruption is a hard failure.
- Full washing is one resident-runtime JobSet containing 463 sequential
  one-hour waves. It is not another l0/p0 smoke and does not fan out sandboxes.
- One explicit resume tag owns the PVC campaign across launch attempts. A
  launch run id may change, but source SHA, topology and the evaluation
  contract may not.
- The original 1851-row clean input remains immutable. The final promoted Q4
  learnable list contains exactly tasks with reward count 1 through 15 of N16.
- The operator-reported running `p46e12805` legacy-v5 job is not interrupted.
  Only after natural termination and producer-pod absence may its output be
  copied into a digest-sealed, read-only import snapshot under a fresh resume
  tag. The live source remains untouched.

## Implementation gate

1. Repair inline tags with existing tail closings, nested `parameter=path`, and
   signed top-level editor shorthands without guessing contradictory commands.
2. Add config-v4/trajectory-v6 action compatibility, resume-tag and diagnostic
   provenance.
3. Keep default `SWEAgent`/Q32 on `strict_xml`; enable v2 only in Q4 eval.
4. Add `--full-campaign`: initialize Q4 once, run all logical/physical waves,
   fsync each trajectory, write every logical digest, and finalize globally.
5. Preserve 16,384 response tokens, 50 steps, N16, concurrency 64, reward-only
   sampling, prefix-cache off, and a 3600-second deadline per physical wave.
6. Pin one immutable resume contract, hold a single-writer lease, isolate setup
   and logs by launch attempt, check out the original harness SHA after branch
   movement, and reconcile only same-tag orphan R2E sandboxes before resume.
7. Admit only an exact v5-to-v6 snapshot import: verify every digest, derived
   per-logical v3 fingerprint, clean task/sample identity, attempt sequence and
   provenance; preserve the legacy sampler SHA, pin the new harness SHA, and
   emit immutable v6 rows plus a receipt before TPU initialization.

## Exit gate

- Local: P46 CPU suite passes; the campaign orchestration test observes one
  runtime, 463 waves, 29,616 identities, and a final 48-identity wave.
- Recovery: CPU interruption creates 17 durable identities, a second launch
  runs exactly the missing 47; torn final JSON is ignored, contract drift and
  a concurrent writer fail closed, and full-campaign postflight accepts only
  58 logical markers plus the exact 29,616-identity campaign marker.
- Publication: an operator-approved commit is pushed to
  `yuxzhang/canon-zero-tim` and read back exactly. Implementation
  `c3a960acdc94173440144559bb95f1de36d31537` satisfies this gate at verified
  publication checkpoint `dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf`;
  executors repeat the ancestry check against the current branch HEAD.
- Transition: the old JobSet is left running. After its natural terminal state,
  one frozen `p46e12805` snapshot imports with `LEGACY_IMPORT_PASS`; any live,
  unsealed, drifted, duplicate or cross-contract input fails closed.
- Target: one full campaign emits
  `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616
  logical_shards=58`, with postflight cleanup and immutable campaign digests.

No target campaign has run. CPU PASS is not TPU, Kubernetes, throughput, or
data-washing completion evidence.
