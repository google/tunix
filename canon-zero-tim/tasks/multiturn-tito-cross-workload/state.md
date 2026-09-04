# State

- Status: release candidate committed; publication/target readback pending
- Objective: expose exact token-in/token-out as one explicit FrozenLake treatment selector for both P45 and M15, and make the explicit 300-update full-training record mode preserve replay-complete all-update A/B/C sidecars, crash-durable journals, and bounded actor-only red-policy snapshots without changing the legacy default or the training data path.
- Definition of done: the latest-tip P67 renderer emits closed legacy/exact and P45/M15 full-record identities; P45 and M15 pass host admission, exact reconstruction, request/trajectory/row joins, truthful coverage/counter classification, immutable journal reconstruction, all-update sidecar integrity, bounded pre-update actor snapshot classification, incremental GCS extraction, one-host observer neutrality, and pinned-image gates. In `record-full`, token red is recorded while the same trajectory trains unchanged, but the run is classified non-Zero-TIM. Identity corruption and all non-whitelisted numerical/backward faults remain fatal. DP8xTP8 execution remains a separate target gate.
- Task directory: `canon-zero-tim/tasks/multiturn-tito-cross-workload`
- Directory state: five-CL straight-line release on
  `local/p57-tito-pair-0902`; the runtime CLs are
  `c5d5ddd9`, `067cf3bf`, `dcde8a91`, and `ba533dd7`, followed by the
  documentation/evidence CL containing this file
- Current phase: T9d-3 — Perf-v2 observer-neutrality and durable target admission repair
- Last verified fact: T9d-3 host construction passes at P57 232/232, V1
  102/102, APC 31/31, flags 422/422, and focused one-host judge 5/5. The closed
  Perf-v2 DP1xTP4 exact-TiTO off/on carrier, O_EXCL single-writer receipt,
  production Orbax save/restore startup probe, before-backward update-0 token
  gate, and four bounded actor-snapshot categories are implemented. The first
  full fixed-image gate exits zero on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS`. No matched one-host
  observer-neutrality, real GCS/Orbax save, or DP8xTP8 target has run.
- Next action: verify the pushed fifth CL by full SHA, then request separate
  approval for a clean committed-tree v5p execution of
  `scripts/run_tito_onehost_neutrality_pair.sh`. Only after that pair passes may
  a separately approved P45/M15 DP8xTP8 record-full render/launch proceed.
- Blockers: DP8xTP8 target execution and all TPU/Kubernetes work are
  unverified and unauthorized. Because the shared trajectory engine changed,
  the DeepSWE DP1xTP4 controlled carrier remains a separate pending adjacency
  gate; scoped host/pinned regressions are green but do not replace it.
- Key artifacts: `HANDOFF.md`;
  `phases/t9c-full-record-and-durable-extraction.md`;
  `phases/t9b-engine-witness-and-multidiff-collection.md`;
  `../p58-deepswe-native-zero-comparison/state.md`;
  `../v1-phase4-three-full-recipes/phases/v1-p4-16-m15-nontito-curve-first.md`
- Updated: 2026-09-04T08:21:00Z
