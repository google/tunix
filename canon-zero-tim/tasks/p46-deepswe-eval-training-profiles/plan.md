# Plan

## Outcome

Provide one operator-facing DeepSWE package with three workload families:

1. Qwen3-4B-Instruct-2507, 16K response, B4/G4, exactly three updates, and a 3600-second rollout-batch deadline.
2. Qwen3-4B-Instruct-2507 clean-data evaluation at 16K, logically 32 tasks x 16 samples but executed as resumable 4-task x 16-sample shards with a 3600-second shard deadline.
3. Qwen3-32B, 16K response, B8/G8, exactly 1000 updates, and a 5400-second rollout-batch deadline.

Training keeps TPU-resident optimizer state. The reviewed 1851-row source whitelist remains immutable. Evaluation produces versioned reports and candidate whitelists; it never silently replaces the production input. The 64/256 variants may differ only in registered topology, worker count, DP-local partitioning, and DP-derived carrier geometry. Main, credentials, cloud resources, precision, loss, reward, sampler semantics, and optimizer math remain out of scope.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P46.1 | Hardened evaluator and frozen workload contracts | Evaluator unit tests prove config fingerprinting, exact-N resume, full trajectory serialization, invalid-sample isolation, and curriculum classification | completed |
| P46.2 | Three dual-topology JobSet families | Six dummy manifests render; normalized 64/256 pairs differ only by the topology allowlist; Q4 uses 3600 s and Q32 uses 5400 s | completed |
| P46.3 | Operator documentation and release regressions | New P46 CPU gate plus P44/P39/P34 adjacent gates and `git diff --check` pass | completed |
| P46.4 | Remote execution campaign | Q4 evaluation smoke, Q4 three-update training, clean-data promotion, and Q32 training each return durable target evidence | active |

## Decisions

- Confirmed: the reference branch is `yuxzhang/swe-evaluation-dev` at `5113c0fb788a2c1f31344f6c3b1265d069bf11ea`; it provides n-sample reporting, streaming summaries, resume, and per-task failure isolation, but not the complete artifact and fingerprint contract required here.
- Decision: selectively port behavior; do not merge the reference branch wholesale.
- Decision: `32 x 16` is an evaluation report unit, not a 512-trajectory training update. Execute one concurrency wave of 4 tasks x 16 samples at a time and aggregate eight shards.
- Decision: Qwen3-32B remains B8/G8 and max concurrency 64. B16/G8 is not admitted until a real target proves 128 concurrent sandboxes can finish within the boundary.
- Decision: one-hour and ninety-minute deadlines bound rollout collection, not model initialization or first XLA compilation.
- Decision: a timed-out or incomplete batch never commits an optimizer update. A timed-out evaluation shard persists completed samples, confirms cleanup, and resumes by exact task/sample identity.
- Decision: Qwen3-4B evaluation is curriculum evidence, not Qwen3-32B ground truth. Q4 all-fail tasks remain a separate Q32 hard tier rather than being deleted.
- Confirmed: 1851 tasks at N16 and 64 trajectories per physical wave require 58 logical reports and 463 physical JobSets. The final logical report has 27 tasks and only physical indices 0-6.
- Decision: P46 maintains parameterized JobSet renderers rather than checked-in launch YAML with stale source/image/node-pool pins. A remote agent renders concrete YAML only after reading back the publication SHA.
- Publication status: implementation commit
  `e1b4009394c49ea015919bda0cfdb97c12c221b5` is published to the operator
  branch. Remote execution resolves the current branch HEAD dynamically and
  requires this implementation commit in its ancestry.
