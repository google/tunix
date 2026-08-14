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
- The original 1851-row clean input remains immutable. The final promoted Q4
  learnable list contains exactly tasks with reward count 1 through 15 of N16.

## Implementation gate

1. Repair inline tags with existing tail closings, nested `parameter=path`, and
   signed top-level editor shorthands without guessing contradictory commands.
2. Add config-v3/trajectory-v5 action compatibility and diagnostic provenance.
3. Keep default `SWEAgent`/Q32 on `strict_xml`; enable v2 only in Q4 eval.
4. Add `--full-campaign`: initialize Q4 once, run all logical/physical waves,
   fsync each trajectory, write every logical digest, and finalize globally.
5. Preserve 16,384 response tokens, 50 steps, N16, concurrency 64, reward-only
   sampling, prefix-cache off, and a 3600-second deadline per physical wave.

## Exit gate

- Local: P46 CPU suite passes; the campaign orchestration test observes one
  runtime, 463 waves, 29,616 identities, and a final 48-identity wave.
- Publication: an operator-approved commit is pushed to
  `yuxzhang/canon-zero-tim` and read back exactly.
- Target: one full campaign emits
  `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616
  logical_shards=58`, with postflight cleanup and immutable campaign digests.

No target campaign has run. CPU PASS is not TPU, Kubernetes, throughput, or
data-washing completion evidence.
