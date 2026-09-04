# State

- Status: T9e all-event token-difference refinement implemented; host and pinned-image construction PASS; execution gates pending
- Objective: expose exact token-in/token-out as one explicit FrozenLake treatment selector for both P45 and M15, and make the explicit 300-update full-training record mode preserve replay-complete all-update A/B/C sidecars, crash-durable journals, and bounded actor-only red-policy snapshots without changing the legacy default or the training data path.
- Definition of done: the latest-tip P67 renderer emits closed legacy/exact and P45/M15 full-record identities; P45 and M15 pass host admission, exact reconstruction, request/trajectory/row joins, truthful coverage/counter classification, immutable journal reconstruction, all-update sidecar integrity, bounded pre-update actor snapshot classification, incremental GCS extraction, one-host observer neutrality, and pinned-image gates. In `record-full`, token red is recorded while the same trajectory trains unchanged, but the run is classified non-Zero-TIM. Identity corruption and all non-whitelisted numerical/backward faults remain fatal. DP8xTP8 execution remains a separate target gate.
- Task directory: `canon-zero-tim/tasks/multiturn-tito-cross-workload`
- Directory state: published five-CL straight-line baseline on
  `local/p57-tito-pair-0902`; the runtime CLs are
  `c5d5ddd9`, `067cf3bf`, `dcde8a91`, and `ba533dd7`, followed by the
  documentation/evidence CL `a10c061a`. T9e is one additive follow-up CL
  rebased on integration tip `90fd0e55`; resolve its exact SHA with
  `git rev-parse HEAD`. It is approved for push; no target execution is
  authorized.
- Current phase: T9e — all-event token-difference stream
- Last verified fact: T9e removes the record-full per-trajectory latch and
  64-event cap, persists every valid token-difference event with a contiguous
  ordinal and replay-complete token/ledger identity, and lets update-0 and
  later token-red rows continue unchanged through full training. Missing or
  corrupt evidence and structural/numerical/backward faults remain fatal.
  Host construction passes at P57 234/234, V1 102/102, APC 12/12, and flags
  422/422. The complete fixed-image gate exits zero on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal `V1_HP_EXACT_IMAGE_PASS` and explicit record-full,
  capsule-integrity, engine-witness, and GCS-durability receipts. No matched
  one-host observer-neutrality, real GCS/Orbax save, or DP8xTP8 target has run.
- Next action: verify the approved T9e follow-up by remote SHA readback.
  After a published clean SHA, separately seek approval to run the documented matched
  one-host observer-neutrality pair before seeking separate render/DP8xTP8
  launch approval. Do not commit, push, or launch TPU/Kubernetes work without
  that action's explicit approval.
- Blockers: DP8xTP8 target execution and all TPU/Kubernetes work are
  unverified and unauthorized. Because the shared trajectory engine changed,
  the DeepSWE DP1xTP4 controlled carrier remains a separate pending adjacency
  gate; scoped host/pinned regressions are green but do not replace it.
- Key artifacts: `HANDOFF.md`;
  `phases/t9c-full-record-and-durable-extraction.md`;
  `phases/t9b-engine-witness-and-multidiff-collection.md`;
  `../p58-deepswe-native-zero-comparison/state.md`;
  `../v1-phase4-three-full-recipes/phases/v1-p4-16-m15-nontito-curve-first.md`
- Updated: 2026-09-04T10:14:32Z
