# State

- Status: T9g source frozen / host and full pinned CPU image PASS / commit and push approved; target NOT RUN
- Objective: expose exact token-in/token-out as one explicit FrozenLake treatment selector for both P45 and M15, and make the explicit 300-update full-training record mode preserve replay-complete all-update A/B/C sidecars, crash-durable journals, and bounded actor-only red-policy snapshots without changing the legacy default or the training data path.
- Definition of done: the latest-tip P67 renderer emits closed legacy/exact and P45/M15 full-record identities; P45 and M15 pass host admission, exact reconstruction, request/trajectory/row joins, truthful coverage/counter classification, immutable journal reconstruction, all-update sidecar integrity, bounded pre-update actor snapshot classification, incremental GCS extraction, one-host observer neutrality, and pinned-image gates. In `record-full`, token red is recorded while the same trajectory trains unchanged, but the run is classified non-Zero-TIM. Identity corruption and all non-whitelisted numerical/backward faults remain fatal. DP8xTP8 execution remains a separate target gate.
- Task directory: `canon-zero-tim/tasks/multiturn-tito-cross-workload`
- Directory state: tracked task on published baseline
  `c07ea8fa10cbfc55957503b9831ed863f8c36814`; T9g source CL is
  `ae2e4884d7df20cdf7cda6dd4a910ea61280e443`, followed by its
  evidence/handoff CL. User approved T9g commit/push on 2026-09-07 to
  `yuxzhang/canon-zero-tim`, not target actions. Historical T9f source CL
  `89a58e24d02ed42b2bc39126eb592ca0a1426bd3` on baseline
  `2833977c1daae9971330e9be9bf16e546b9f0f4f` in
  `local/p57-tito-pair-0902`, followed by its evidence/handoff CL.
  The earlier T9f release had its separate approval on 2026-09-07 to
  `yuxzhang/canon-zero-tim`; target actions remain unauthorized.
  Historical published five-CL baseline:
  `local/p57-tito-pair-0902`; the runtime CLs are
  `c5d5ddd9`, `067cf3bf`, `dcde8a91`, and `ba533dd7`, followed by the
  documentation/evidence CL `a10c061a`. T9e is one additive follow-up CL
  rebased on integration tip `90fd0e55` and published as `290a67d8`.
- Current phase: T9g — explicit 32-chip / 128-trajectory full option; see `phases/t9g-dp4-tp8-b128-full.md`
- Last verified fact: T9g P57 248/248, V1 104/104, APC 12/12 and flags
  423/423 pass. The pinned-image real-runtime probe passes eight P45/M15
  old/new legacy/record-full cases and one DP4xTP8 forced-CPU fixed-reducer
  check. New global/local M/chunks/profile and TiTO 128-row/DP metadata
  negatives pass. Complete pinned-image rerun exits 0 with
  `V1_HP_EXACT_IMAGE_PASS`; the full raw log and 41 final code/gate hashes are
  retained in `evidence/t9g-host-20260907/receipt.json`
  (SHA256 `5df76e984f354109e50cc9ff5be9b1e584ab09017c4ce39baa0716d676fbf8fb`).
  No TPU, real storage or full-horizon performance is certified.
- Historical T9f fact: r09 raw evidence (three SHA checks PASS) shows policy 40
  synchronized, then a first-turn MODEL_TIMEOUT with zero responses failed
  the record-full row-map assertion. This is not a backward exception;
  updates 38/39 report finite gradients. The same empty-list assumption also
  existed in the sidecar and final coverage classifier. All are repaired
  with an explicit zero-response/zero-data receipt; IDs for actual completed
  responses remain mandatory. P57 238/238, V1 102/102 and flags 422/422 pass.
  Pinned-image focused collector, real learner off/on batch arrays and actual
  sidecar tests pass; complete image gate exits 0 with
  `V1_HP_EXACT_IMAGE_PASS`. The retained image console is partial due to tool
  truncation; this is an admission receipt, not a complete raw-log artifact.
  No patched TPU
  execution has occurred. The old source r09 tail is partial target evidence,
  not a complete full-run certification.
- Historical T9e fact: T9e removes the record-full per-trajectory latch and
  64-event cap, persists every valid token-difference event with a contiguous
  ordinal and replay-complete token/ledger identity, and lets update-0 and
  later token-red rows continue unchanged through full training. Missing or
  corrupt evidence and structural/numerical/backward faults remain fatal.
  Host construction passes at P57 234/234, V1 102/102, APC 12/12, and flags
  422/422. The complete fixed-image gate exits zero on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS` and explicit record-full,
  capsule-integrity, engine-witness, and GCS-durability receipts. Those gates
  did not exercise real GCS/Orbax or the matched one-host/DP8xTP8 target.
- Next action: verify delivered full HEAD and remote readback, then request
  the target render/launch approval before using HANDOFF Section 0.
  Select `dp4-tp8-b128` plus `both-exact` plus `record-full` for both
  TiTO runs. Publication does not execute them. Release audit and repeated
  host logs: `evidence/t9g-release-20260907/receipt.json`.
- Blockers: patched target execution and all TPU/Kubernetes work remain
  unauthorized, including DP4xTP8. Previous sorting/observer-neutrality and Orbax-exception
  review concerns are separate, not silently repaired by T9f. Because the shared trajectory engine changed,
  the DeepSWE DP1xTP4 controlled carrier remains a separate pending adjacency
  gate; scoped host/pinned regressions are green but do not replace it.
- Key artifacts: `HANDOFF.md`;
  `phases/t9c-full-record-and-durable-extraction.md`;
  `phases/t9b-engine-witness-and-multidiff-collection.md`;
  `../p58-deepswe-native-zero-comparison/state.md`;
  `../v1-phase4-three-full-recipes/phases/v1-p4-16-m15-nontito-curve-first.md`
- Updated: 2026-09-07
