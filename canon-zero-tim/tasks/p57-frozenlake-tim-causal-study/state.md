# State

## 2026-09-08 active extension — Standard Native64

- Implemented on `local/fl-standard64-0908`, originally tested on published base `06a0fdb9`; this three-CL publication is approved onto newer evidence-only base `069010dd`. Target remains unapproved.
- New opt-in Standard keeps rollout evidence but uses frozen trainer-old with no TIS. Legacy Native/IS/Zero and worker scheduling unchanged; inherited P78-off admission gap repaired.
- Local gates PASS: P57 263/263, V1 108/108, flags 439/439, full CPU pinned image and 11/11 final focused image checks. Real learner method uses mocked model logps; no real TPU backward/convergence claim.
- Operator procedure: [STANDARD64.md](STANDARD64.md); evidence: [manifest](evidence/standard64_local_0908/manifest.json), 22 program/gate hashes and 10 raw logs including failed intermediate gates.
- Committed code: admission `b7828eeff2c038ba02b29dcf6fc83f6291588f32`, Standard `48b7dc69ddd3eef93a3f304086da9523b79a5e32`. [Publication audit](evidence/standard64_local_0908/publication.json) proves all 22 hashes still match; post-rebase P57 263/263, V1 108/108 and flags439/439 PASS.
- Next: publication readback → clean published SHA/image/identity freeze → separately approved P45/M15 full. Target NOT RUN. Do not use dirty test manifests. Future commits/pushes also require new approval.

## Previous step-boundary campaign (historical; does not certify Standard)

- Status: G4 passed; source CL `ec9884e9` prepared for publication
- Objective: measure native/no-IS, native/token-IS, and complete zero-TIM/no-IS on both the original P45 FrozenLake workload and frozen M15 using signed 300-update training and seven-point rollout-only held-out curves.
- Definition of done: all six cells pass local/target arm receipts, complete 300 updates and exact evaluations at `0,50,100,150,200,250,300`, and produce within-workload `is-mismatch`, `zero-mismatch`, and `zero-is` contrasts under the registered claim ceiling.
- Task directory: `canon-zero-tim/tasks/p57-frozenlake-tim-causal-study`
- Directory state: tracked
- Current phase: [P57.1c — Perf v2 step-boundary isolation](phases/p57-1c-perf-v2-step-boundary.md)
- Last verified fact: approved one-host v5p G4 `p57c_g4_cb38cf67_r7` completed 3/3 real AdamW transactions, 12/12 strict alignment PASS with zero differing bytes, finite nonzero gradients, and a readable beta-zero semantic Perfetto. It crossed the former Step-1 rollout underflow boundary without a tracer red. Steady Steps 1/2 were 36.93s and 35.98s. P57 CPU is 172/172; the final pinned-image P45 gate emits `P57_PERF_V2_STEP_BOUNDARY_PASS` and `P45_EXACT_IMAGE_CPU_PASS`; V1 Phase4 is 90/90; flag audit is 395/395. No numerical flag or training math changed.
- Next action: verify the published two-CL stack, then obtain separate target-launch approval before fresh P45 and M15 full identities exercise G5.
- Blockers: production render and full P45/M15 launch remain outside the current authorization.
- Key artifacts: [plan.md](plan.md); [active phase](phases/p57-1c-perf-v2-step-boundary.md); [RUNBOOK.md](RUNBOOK.md); [HANDOFF.md](HANDOFF.md); [Wave 15 Incident](evidence/f45w15_timeline_tracer_incident/INCIDENT_REPORT.md)
- Updated: 2026-08-28T12:45:00Z
