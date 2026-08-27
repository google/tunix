# Log

## 2026-08-24T23:20:18Z — Phase A: immutable target evidence admitted

- Type: experiment / handoff
- Fact: the fetched operator tip equals reference SHA `687b2bd6d0815b5628af39e7adbf949e429e72ae`; the isolated branch is `local/v1-apc-m15-target-debug-0824` and preflight reports clean PASS.
- Fact: Attempt-2 `SHA256SUMS` verifies all three raw logs and `receipt.json`. The M15 receipt identifies source `7a2a456ce43302c34958fa34c11b0583b45a666e`, JobSet `canon-p57-fl-zero-m15-m15i-71d889a3`, and failure stage `check_pre_backward_gate`.
- Fact: `m15_m15i_error.log:22140-22142` records `N_action=110844`, A-B=1389 bytes / 760 elements, max abs `0.998443603515625`, B-C=0 bytes, and strict verdict FAIL. First mismatch is row 192, completion position 0, prompt/logical KV prefix 1226, turn 0.
- Hypothesis: none selected; cache mapping, content, read path, position metadata, request chronology, and topology remain separate candidates.
- Action: fetched the operator branch, created an isolated named worktree, read the governing registries/skills/Phase3 ledgers, and verified the immutable evidence manifest. No numerical source was changed.
- Command: `sha256sum -c SHA256SUMS`
- Result: 4/4 `OK`; branch preflight PASS; Phase A remains active because replay artifact completeness is not yet known.
- Files/artifacts: [Attempt-2 evidence](../v1-phase4-three-full-recipes/evidence/v1_hp_three_full_attempt2_20260824/), [Phase A](phases/phase-a-evidence-decoding.md)
- Rollback: remove only this uncommitted task-directory addition after preserving any requested copy; no runtime rollback exists because runtime code is untouched.
- Next: inventory `m15i` tokens, request order, cache-hit/lineage, block-table, policy/model identity, and reconstruct the mismatch distribution from the log.

## 2026-08-24T23:55:00Z — Phase A closed; strict replay input absent

- Type: analysis / phase transition
- Fact: the analyzer reproduces the strict report exactly: 760 elements / 1389 bytes A-B, max abs `0.998443603515625`, B-C zero, and all 760 mismatch records present.
- Fact: every mismatch belongs to prompt-major group 24. Rows 192/193/194/196/197/198/199 are red; row 195 (generation 3) is clean. The first red is logical prefix 1226 and only 6/760 red coordinates lie exactly on a 256-token boundary.
- Fact: the durable Attempt-2 set contains only four manifest members plus `SHA256SUMS`; it has no raw arrays, request order, token history, block table, computed/cached-token receipts, or page lineage. Exact historical replay is impossible from this archive.
- Fact: immutable provenance conflicts. `receipt.json` names `7a2a456c...`, while raw line 15 and the fail-closed sync path attest executed HEAD `71d889a32f4668353c758d5c00df88299e6c0d35`. The numerical incident is assigned to runtime HEAD `71d889a3`; the receipt value is preserved as an evidence defect.
- Action: added a standard-library evidence decoder plus regression tests; advanced to Phase B and chose reuse of the existing P38 capsule/journal/incident join rather than a new numerical observer.
- Validation: Attempt-2 manifest 4/4; analyzer emits `M15_FIRST_RED_INPUT_CONTRACT ... replay=INSUFFICIENT_FOR_STRICT_REPLAY`; Python compilation and tests are the next static gate.
- Limitation: no fresh carrier exists, no numerical red was reproduced locally, and no source tensor boundary has been localized.
- Next: implement and host-test only the bounded carrier/postflight. Any actual TPU or target launch remains a separate user approval.

## 2026-08-24 — Phase B static carrier prepared

- Type: implementation / host validation; no numerical repair.
- Fact: a matched DP8xTP8 M15 off/on renderer now preserves source, command,
  topology, request geometry, seed, and capture geometry. Structural
  normalization proves only the arm identity, derived APC bit, names, and
  arm-scoped paths differ.
- Fact: the real rendered environment resolves through the production
  `00_env.sh` for both arms and rejects a wrong-profile negative before
  runtime. This gate found a real workload-identity admission defect, which is
  now restricted to the exact M15 debug selector.
- Fact: A asserts `prompt_logprobs=None`, `logprobs=1`, and
  `skip_reading_prefix_cache=False`; B asserts `reset_prefix_cache=True` and
  zero cached tokens. These are observer-only assertions and do not modify
  returned values.
- Fact: fresh red postflight requires a hash-matched capsule and every selected
  mismatch to have one exact incident join. It emits a small first-red replay
  bundle with complete row arrays and physical page/generation coordinates.
  The bundle declares that full co-batch token payloads and scheduler
  interleaving remain absent.
- Fact: classifier failure cannot be masked by expected controlled exit 42;
  missing replay bundle on a red is also fatal.
- Validation: task scripts 24/24; P38 classifier 37/37; Phase3 contract 12/12;
  Phase3 profile/boundary 11/11; V1 Phase4 24/24; flag audit 369/369 PASS;
  shell syntax, Python compilation, and diff check PASS.
- Limitation: two broader P33/P45 imports were not collectable because this
  host lacks `datasets` and `metrax`; their remaining dependency-free cases
  passed. No exact-image or TPU gate was run.
- Claim: `PHASE_B_STATIC_CARRIER_ONLY`. No first-red tensor boundary and no
  APC mechanism have been localized.
- Next: user review, then separately approved commit/push and APC-off target
  control. APC-on treatment is not launched until the control is green.

## 2026-08-24 — operator-tip advance reviewed and absorbed

- Type: provenance / rebase audit.
- Fact: `origin/yuxzhang/canon-zero-tim` advanced from the initial reference
  `687b2bd6...` to `307cb42da5c6a6f7ec70dceec359e948b1080316`.
- Fact: the one new commit adds exactly two GSM8K raw logs and changes none of
  the APC carrier, cluster steps, runtime sources, tests, flags, or task files.
- Action: used a recoverable stash inside the isolated worktree, fast-forwarded
  to the new tip, restored the complete uncommitted change set without
  conflict, and reran the final gates.
- Result: baseline is current and the APC diff is unchanged; no other agent's
  worktree or files were modified.

## 2026-08-25 — full replay inputs and GCS return path prepared

- Type: implementation / evidence durability; no numerical repair.
- Fact: the historical first-red row bundle alone could not reproduce the
  request chronology. The carrier now also writes all 256 final producer rows
  and one host-only envelope record for every serving call in both A and B.
- Fact: each envelope record includes exact dispatch order, request identity,
  DP rank/local slot, scheduled/computed/prompt token counts, token-history
  SHA, logical block extent and physical page table. It fetches no device
  tensor and is bounded by the existing 256 MiB incident limit.
- Fact: postflight requires contiguous call chronology, both serving arms,
  complete producer joins, exact first-red row bytes, first-red request/call
  identity and physical pages before writing
  `FULL_REPLAY_CARRIER_FROZEN`.
- Fact: large arrays/chronology stay inside the immutable P38 GCS
  `serving-capture.tar`; the growing envelope is included in live snapshots.
  A checked-in GCS audit validates all manifests and uploads only small derived
  receipts under `derived/m15-replay-audit-v1`.
- Validation: full carrier/audit positives and five corruption negatives,
  real rendered-env positives and wrong-profile/wrong-ledger-path negatives,
  P38 fake-GCS persistence, patch application, Python/shell syntax all pass.
- Limitation: no fresh target run, numerical reproduction, forced scheduler
  replay, first-red tensor localization, or repair has occurred.
- Claim: `PHASE_B_STATIC_CARRIER_ONLY`.
- Next: final host suite and diff review; commit/push, exact-image, target
  control, and target treatment remain four separate user approvals.

## 2026-08-25 — Phase B static release gate closed

- Type: static/host validation; target not run.
- Fact: a fake immutable GCS integration test exercised the checked-in wrapper
  end to end: it downloaded the root manifest and payloads, verified root and
  nested hashes, uploaded only small derived receipts, wrote the derived
  `SHA256SUMS` last, and rejected a second write to the same derived prefix.
- Fact: M15 capsule/tar payloads use `encoding=gcs-only`; legacy non-M15 P38
  payloads retain their base64 behavior.
- Validation: task carrier 33/33, P38 classifier 37/37, Phase3 contract 12/12,
  V1 Phase4 CPU 29/29, fake-GCS persistence PASS, flag audit 370/370 PASS,
  Python/shell syntax and `git diff --check` PASS.
- Limitation: exact-image, one-host replay, APC-off DP8xTP8 control, and APC-on
  DP8xTP8 treatment were not run. No numerical repair or root-cause claim was
  made.
- Claim: `PHASE_B_STATIC_PASS_TARGET_NOT_RUN`.
- Next: user diff review. Commit/push, exact-image, control, and treatment each
  remain a separate approval boundary.

## 2026-08-25 — pinned exact-image admission passed

- Type: pinned-image admission; target not run.
- First attempt: all prior gates reached the new GCS wrapper integration test,
  which failed `rc=127` because its fake `gcloud` shebang used `env python3`
  while the test replaced PATH with `/usr/bin:/bin`; the pinned image keeps
  Python under `/usr/local/bin`. This was a test-carrier defect, not an APC or
  GCS runtime verdict.
- Fix: prepend the active `sys.executable` directory while retaining the fake
  command directory and inherited PATH. Focused GCS wrapper 8/8 and complete
  carrier 33/33 passed on the host.
- Final rerun: pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  passed patch installation/manifests, Qwen8B TP8 fixed-head/projection probes,
  P59 TP4/TP8 installed shims, V1 contracts, and the GCS wrapper. Terminal:
  `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=33 ... manifests=3`.
- Claim: `PHASE_B_EXACT_IMAGE_PASS_TARGET_NOT_RUN`.
- Limitation: no one-host replay or DP8xTP8 target run has occurred; no fresh
  red, localization, repair, or APC production enablement is claimed.
- Next: user review and explicit commit/push approval, then separately approved
  APC-off DP8xTP8 control.

## 2026-08-25T02:22:00Z — Attempt 0: APC-off control failure recorded

- Type: target launch / diagnostic
- Fact: JobSet `canon-v1-apc-m15-off-d3-eb58954f` (DP8xTP8 64-TPU, APC-off control) launched on the cluster with commit `eb58954f90572e19602b354cfcb71cc5d58f35d5`.
- Fact: all 16 TPU nodes booted, synced repo, verified all 6 overlay files with SHA256 byte identity, and completed worker registration.
- Fact: in Step 90, the Python launcher exited with code 1 during startup before populating `p38_serving_capture`, resulting in `INCONCLUSIVE` postflight classification.
- Action: archived Attempt 0 failure receipt and error log in `evidence/v1_apc_m15_attempt0_20260825/`.
- Files/artifacts: [Attempt-0 evidence](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-0 error log](evidence/v1_apc_m15_attempt0_20260825/m15_off_d3_attempt0_error.log)
- Next: diagnose the root cause of the Python launcher exit in Step 90, re-render, and relaunch APC-off control.

## 2026-08-25 — Attempt-0 bootstrap root cause and host repair

- Type: launcher-contract repair; no numerical repair.
- Confirmed cause: the rendered command contained
  `--p57_workload_candidate=m15 --p57_data_split=main`, while the JobSet had no
  `CANON_P57_WORKLOAD_CANDIDATE` or `CANON_P57_DATA_SPLIT`. The FrozenLake
  entrypoint requires these CLI and signed-environment values to agree before
  learner construction, so Attempt 0 could not create the capture directory.
- Repair: carry exact `m15/main` through the renderer, admit only those two P57
  identity fields for the exact APC-debug profile in Step 00, require the
  package-safe module entrypoint, and add wrong-identity and wrong-entrypoint
  negatives. APC on/off, A/B/C, model math, backward, and optimizer are
  unchanged.
- Validation: APC task tests 35/35; P38 classifier 37/37; V1 Phase4 CPU 34/34;
  flag audit 371/371; Python/shell syntax and `git diff --check` PASS. The
  Phase3 flag-count assertion was stale at 370 on the incoming tip and was
  synchronized to the already registered 371-name inventory.
- Limitation: post-fix exact-image and DP8xTP8 target were not run. Attempt 0
  remains `INCONCLUSIVE`; Attempt 1 must use a new source SHA, label, and GCS
  attempt.

## 2026-08-25T02:52:01Z — Attempt 1: APC-off control geometry mismatch recorded

- Type: target launch / diagnostic
- Fact: JobSet `canon-v1-apc-m15-off-d4-283cb67e` (DP8xTP8 64-TPU, APC-off control) launched with commit `283cb67e184239530ac68e3d1c66edf8d37a3c09`.
- Fact: all 16 TPU nodes booted, synced repo, verified all 6 overlay components with SHA256 byte identity, and passed GCS preflight.
- Fact: in Step 90, Python entrypoint failed with exit code 1 due to legacy P38 DP16 geometry assertions in `train_frozenlake_qwen3.py` rejecting M15 DP8 target parameters: `ValueError: P32 FrozenLake geometry mismatch: {'mini_batch_size': (32, 4), 'sampler_is': ('none', 'token')}` and hardcoded `P32_WORKLOAD.name == "frozenlake"` check.
- Action: archived Attempt 1 failure receipt and error log in `evidence/v1_apc_m15_attempt1_20260825/`.
- Files/artifacts: [Attempt-1 evidence](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-1 error log](evidence/v1_apc_m15_attempt1_20260825/m15_off_d4_attempt1_error.log)
- Next: update `train_frozenlake_qwen3.py` geometry validations to accept M15 APC DP8 target parameters and relaunch APC-off control.

## 2026-08-25T03:14:09Z — Attempt-1 geometry repair host-pass

- Type: prelearner admission repair; no numerical repair and no target launch.
- Root cause: `CANON_P38_PRECHECK_ONLY=1` selected one legacy DP16 contract in
  two places. It expected `mini_batch_size=4` and token IS, then required
  workload `frozenlake`, DP16, and eight four-prompt producer units. The M15
  APC target carrier intentionally preserves the production DP8 geometry:
  `mini_batch_size=32`, no IS, workload `frozenlake-dp8-tp8`, and one complete
  32-prompt/256-trajectory producer unit.
- Repair: introduce pure fail-closed entrypoint helpers keyed by the existing
  exact `CANON_APC_M15_TARGET_DEBUG=off|on` selector. Legacy P38 and P57
  contracts are unchanged; invalid selector values, mixed P57 TIM mode, wrong
  workload/DP/unit geometry, and APC target without precheck are rejected.
- Adjacent test repair: the fixed-lm-head M2048 receipt test now includes the
  already registered APC debug profile; the pinned-image terminal count is
  synchronized to the expanded 39-test carrier suite.
- Validation: APC task 39/39; P57 FrozenLake 144/144; P38 fixed-head 15/15;
  P38 serving classifier 37/37; Phase3 prefix-cache 12/12; V1 Phase4 34/34;
  Python/shell syntax and `git diff --check` PASS. The broad P33 host runner's
  dependency-free tests ran, but two imports remain unavailable on this host
  (`datasets`, `metrax`); this is why the post-fix pinned-image gate remains a
  separate required approval.
- Limitation: exact-image and fresh DP8xTP8 control were not run. No A/B/C
  observation, replay carrier, localization, APC repair, or production-enable
  claim is made.
- Next: user diff review; then separate approvals for commit/push, exact-image,
  APC-off target control, and (only after a green control) APC-on treatment.

## 2026-08-25T03:58:44Z — Attempt 2: APC-off control program path mismatch recorded

- Type: target launch / diagnostic
- Fact: JobSet `canon-v1-apc-m15-off-d7-41a2043c` (DP8xTP8 64-TPU, APC-off control) launched with commit `41a2043ca612eeb8dcf77ae1262d18471c26b479`.
- Fact: all 16 TPU nodes booted, synced repo, verified all 6 overlay components with SHA256 byte identity, passed GCS preflight, and completed >95% of 15-turn FrozenLake rollout (1800+ model calls, 760+ requests, 256 trajectories).
- Fact: in Step 90 during final token generation, P38 serving capture hook asserted `expected=standard actual=continue_decode` and failed with `RuntimeError: P38 serving capture reached an unexpected program path: expected=standard actual=continue_decode`.
- Root cause: `qwen3-8b-dp8-tp8-frozenlake-apc-debug.env:L32` had `export CANON_CONTINUE_DECODE=8` set, causing vLLM to route deep decode tokens to `_execute_continue_decode` while P38 serving capture asserted `EXPECTED_PATH="standard"`.
- Action: archived Attempt 2 failure receipt and error log in `evidence/v1_apc_m15_attempt2_20260825/`.
- Files/artifacts: [Attempt-2 evidence](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-2 error log](evidence/v1_apc_m15_attempt2_20260825/m15_off_d7_attempt2_error.log)
- Next: remove `export CANON_CONTINUE_DECODE=8` from profile, re-render, and relaunch APC-off control.

## 2026-08-25T06:30:00Z — Attempt 2 diagnosis corrected; observer repair host PASS

- Correction: the prior recommendation to remove `CANON_CONTINUE_DECODE=8`
  is superseded. `m15i` and the signed M15 production recipe use K=8; deleting
  it would change the serving executable and invalidate the reproduction.
- Additional fact: Attempt 2 first saturated the incident ledger at call 326
  with 268,192,266 bytes and emitted 1,650 nonfatal capture errors. The later
  standard-vs-continue assertion was the fatal error because the continue
  call site did not have the standard path's nonfatal wrapper.
- Implementation: append-only runner patch 27 keeps standard capture
  fail-closed until all four strata are present, then permits only M15
  `continue_decode` tail calls to keep recording the dedicated full host
  chronology; generic request/incident artifacts remain standard-only.
  The M15 signed incident/replay bound is 2 GiB; generic P38's renderer bound
  remains 128 MiB.
- Implementation: full carrier packaging accepts only two registered program
  paths and requires A=`standard+continue_decode`, B=`standard`. Unknown
  paths, an absent continue tail, or continue-decode B are negative controls.
- Validation: task carrier tests are 44/44; patch 27 applies cleanly to the
  manifested post-patch26 runner; Python compilation and manifest target hash
  are green. Exact-image and target have not run.
- Claim ceiling: observer repair only. No APC numerical result, localization,
  or production enablement follows from this host pass.

## 2026-08-25T07:01:51Z — Attempt 3 invalidated the tail-only admission assumption

- Evidence: `evidence/v1_apc_m15_attempt3_20260825/` contains a receipt and a
  433-line error tail; both files pass the committed `SHA256SUMS`. The package
  is sufficient to prove the fatal stack and source identity, but it is not a
  complete run package and contains no A/B/C verdict.
- Fact: source `cdd3987caa648e6112ee8fc184b2e3421de3a4b2` installed patch 27
  (the traceback line moved into its expanded predicate), yet APC-on failed
  with `expected=standard actual=continue_decode` before rollout completed.
- Root cause: patch 27 incorrectly required four completed standard tensor
  strata before admitting the production `continue_decode` path. That ordering
  happened in Attempt 2's APC-off control but is not a serving invariant;
  APC-on can select `continue_decode` earlier because the cache changes the
  request/scheduler state.
- Local repair: append-only patch 28 separates program-path admission from
  tensor-capture completion. Registered M15 off/on runs admit
  `continue_decode` from its first call into the full replay envelope; generic
  tensor capture, request journal, and incident ledger remain standard-only.
  Generic P38 and unknown paths retain the fatal assertion.
- Gate repair: the exact-image runner test now sets capture count and strata to
  zero and calls the installed predicate directly. It also checks that generic
  mode and unknown paths are not admitted. This is the executable negative
  control missing from patch 27's string-only host test.
- Validation: task carrier 44/44, P38 classifier 37/37, Phase3 12/12,
  V1 CPU 45/45, P57 CPU 144/144, and flag audit 372/372 all pass. The targeted
  P33 pinned-image gate assembles both Qwen3-1.7B and Qwen3-8B overlays with
  all 36 manifested files, then runs 35/35 installed-runner tests per overlay;
  the new zero-strata program-path negative control passes. The aggregate V1
  exact-image gate and target are not run.
- Claim ceiling: observer-control repair only. No APC mechanism, evaluation,
  alignment, or production-enable conclusion follows from Attempt 3.

## 2026-08-25T09:18:58Z — Patch 28 aggregate exact-image gate PASS

- Integration: fetched the operator branch and rebased the uncommitted repair
  from `bc214018...` onto `53876c15f407435dbd44680ad18f5f8e88f3c255`.
  The incoming commit contains only the separate M15 full-training non-finite
  evidence package; it does not overlap this APC observer repair.
- Evidence integrity: Attempt 3's `apc_m15_on_d9b_error.log` and
  `receipt.json` both pass their committed `SHA256SUMS`. They prove the
  pre-classification program-path failure but contain no A/B/C verdict.
- Host gates: target carrier 44/44, P38 classifier 37/37, Phase3 12/12,
  V1 CPU 45/45, P57 CPU 144/144, flag audit 372/372, syntax, and
  `git diff --check` all pass.
- Targeted exact-image: both Qwen3-1.7B and Qwen3-8B installed overlays match
  all 36 manifest files and execute 35/35 runner tests. The zero-strata test
  executes the full `_p38_serving_begin` branch, proves M15 `continue_decode`
  admission from the first call, requires the replay-ledger write, and requires
  generic incident capture to remain absent. Generic mode and unknown paths
  remain negative controls.
- Aggregate exact-image: immutable image
  `sha256:418dc632...e53a` terminates with `V1_HP_EXACT_IMAGE_PASS ...
  apc_m15_carrier=44 ... manifests=3` and exit 0.
- Review hardening changed only that installed-runner test; the targeted and
  aggregate exact-image gates were rerun afterward and produced the same PASS
  terminals with exit 0.
- Claim ceiling: observer-control repair is host and aggregate exact-image
  admitted. It has not produced an APC-off/on A/B/C result, localized the
  historical mismatch, fixed APC numerics, or enabled production APC.

## 2026-08-25T09:41:00Z — Patch 28 rebased onto the latest operator tip

- Integration: a final fetch advanced the operator branch from `53876c15...`
  to `548db7e9f014def3cb2b37e66c6f0e62c2041f1d`. The four incoming commits
  restore XProf evidence and add the separate P64/FrozenLake backward
  diagnostics. They do not overlap patch 28's APC observer control flow; the
  uncommitted repair rebased without conflict and remains ahead/behind `0/0`.
- Host gates on the new tip: target carrier 44/44, P38 classifier 37/37,
  Phase3 12/12, V1 Phase4 CPU 67/67, and flag audit 378/378 pass.
- Expanded aggregate exact-image: immutable image
  `sha256:418dc632...e53a` exits 0 with the terminal
  `V1_HP_EXACT_IMAGE_PASS ... p64_numeric=4 p64_capsule=3 ...
  apc_m15_carrier=44 ... manifests=3`.
- A subsequent final fetch added documentation-only commit `95e290b0...`
  (`v1-phase4-three-full-recipes` handoff/runbook/ledgers only). Patch 28 was
  rebased onto that tip without conflict. Runtime and test files are byte-for-
  byte identical to the aggregate-exact-image tree, so the image gate was not
  rerun solely for this documentation-only commit.
- Scope remains unchanged: this admits the observer repair against the latest
  release dependency graph. It does not repair or classify the APC numerical
  mismatch, and it does not address the separate APC-off full-training
  non-finite-gradient incident.

## 2026-08-25T11:09:00Z — Attempt 4 reached alignment, then exposed a signed sampler-admission omission

- Evidence: `evidence/v1_apc_m15_attempt4_20260825/` contains the error log and
  receipt; both pass the committed `SHA256SUMS`.
- Fact: source `618eb7758a7fa094110b5cc47049f3578fdb960a` completed all
  2,560 APC-on rollout requests with 92.5% prefix-cache hit rate and solve
  ratio 0.203. This confirms patch 28 no longer aborts on early
  `continue_decode`.
- Failure boundary: the learner stopped before A/B/C with
  `AlignmentGateError` because the generic canonical sampler gate admitted
  `sampler_is=None` for GSM8K/P34/P57 but omitted the exact M15 APC target
  carrier. No alignment classification or replay bundle exists.
- Repair: admit no-IS only when every signed target coordinate matches: off/on
  selector, exact debug profile, M15/main, DP8xTP8, precheck-only, controlled
  exit, backward-no-commit, and no commit. Require rollout logprobs present and
  token-IS weights absent; emit one exact runtime receipt. The profile,
  classifier, and negative controls all require `--sampler_is=none`.
- Regression gates: target carrier 46/46; P38 classifier 37/37; Phase3 12/12;
  P57 146/146; V1 CPU 67/67; flag audit 378/378; Python/shell syntax and
  `git diff --check` PASS. Host P33 ran all dependency-free tests; its two
  missing host dependencies (`datasets`, `metrax`) are covered by the pinned
  image.
- Aggregate exact-image: immutable image
  `sha256:418dc632...e53a` exits 0 with
  `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=46 ... manifests=3`.
- Integration: the operator branch advanced once more to `74b123a7...`; the
  incoming commit changes only the separate P64 FrozenLake entrypoint
  admission. It does not overlap the M15 learner/profile/classifier repair and
  was fast-forwarded without conflict. The final-tree aggregate exact-image
  gate was rerun after this integration and exited 0 with
  `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=46 ... p64_numeric=4
  p64_capsule=3 ... manifests=3`; no pre-fast-forward result is inherited
  silently.
- Claim ceiling: admission repair only. Post-fix DP8xTP8 is not run and there
  is still no fresh A/B/C verdict, frozen carrier, localization, or APC
  numerical repair. Because Attempt 4 skipped the fresh APC-off control, the
  next target action remains control first.

## 2026-08-25T11:30:00Z — Matched target arms approved for concurrent launch

- User decision: submit the newly rendered APC-off control and APC-on
  treatment immediately from the same committed source SHA when both
  allocations are available. Do not wait for off to finish before submitting
  on. Keep separate JobSets, logs, and GCS roots; a failure in one arm does not
  cancel or delete the other.
- Scientific gate: execution is concurrent but interpretation remains
  control-first. The on-arm result supports an APC-specific claim only after
  the off arm is `CONTROL_GREEN`; otherwise its immutable package is retained
  and reported without a causal claim.
- Handoff, runbook, state, plan, and Phase-B ledger now encode this distinction.
  No renderer, runtime, numerical, or classifier behavior changed for this
  scheduling decision.

## 2026-08-25T18:26:00Z — Final release tree rebased and re-admitted

- Integration: fetched operator tip `9f79cc562b2032f3fe02297ce5608023d907361e`.
  Its three P64 commits touch the shared FrozenLake entrypoint and Step-90 but
  do not overlap this M15 patch. The release commit rebased cleanly.
- Focused post-rebase gates: sampler contract 14/14, M15 classifier 14/14,
  target carrier 10/10, flag registry 378/378, and `git diff --check` PASS.
- Because the incoming P64 work touched shared launch code, the aggregate
  pinned-image gate was rerun instead of inheriting the earlier result. It
  exited 0 with `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=46 ...
  p64_numeric=4 p64_capsule=3 ... manifests=3`.
- Target status remains `TARGET NOT RUN`. The next operator action is to render
  the off/on pair from the published full SHA and submit both JobSets without
  waiting between them; classification remains control-first.

## 2026-08-25T20:19:00Z — Attempt 5 paired run completed and sampler contract admitted

- Hardware run: `canon-v1-apc-m15-off-d11-a909fda1` (off control) and `canon-v1-apc-m15-on-d11-a909fda1` (on treatment), both running on 64 v5p TPUs (DP8xTP8) from commit `a909fda14ee3f7e5d2334812a02b1f8ef94b0fbb`.
- Sampler contract gate: `[CANON_APC_M15_SAMPLER_CONTRACT] PASS sampler_is=none use_rollout_logps=1 rollout_logps=present tis_weights=absent` successfully verified, resolving the Attempt 4 admission gate failure.
- Control arm (off): 2,560 requests completed across 15 sampling turns with 0.0% prefix cache hit rate.
- Treatment arm (on): 2,560 requests completed across 15 sampling turns with 89.7% ~ 97.5% prefix cache hit rate.
- Execution conclusion: Both arms cleanly reached the end of sampling and executed controlled exit 42 with zero optimizer commits. Evidence sealed under `evidence/v1_apc_m15_attempt5_paired_d11_20260825/`.

## 2026-08-25T22:40:00Z — Correction: Attempt 5 Git return is snapshot-only

- A fresh pull advanced the operator tip to `ceb3d1a5c62692a1e601459986d622ad32d86dab` and added the off/on Attempt-5 diagnostic snapshots.
- The committed `SHA256SUMS` verifies all three returned files, but both logs are 33-KiB periodic snapshots rather than authoritative full run logs. Mechanical counts are zero for `CANON_ALIGN_PRE`, `CANON_APC_M15_SAMPLER_CONTRACT`, `CONTROLLED_EXIT`, target classification, and GCS terminal markers.
- The preceding checkpoint's statements about sampler PASS, controlled exit 42, zero commits, and `TARGET_NOT_REPRODUCED` came from the summary receipt and are not reproducible from the committed raw subset. They remain unverified rather than erased.
- Decision: demote the claim to `ATTEMPT5_ROLLOUT_SNAPSHOTS_PRESENT / GCS_AUDIT_PENDING / A-B-C_NUMERICAL_VERDICT_UNKNOWN`. Do not launch, profile, or change numerical code yet.
- Next gate: a bucket-capable executor must run the checked-in `run_m15_replay_gcs_audit.sh` independently on the off and on Attempt-0 roots and return the two machine-generated small bundles. Control must classify `CONTROL_GREEN` before the on arm supports any APC-specific interpretation.

## 2026-08-26T00:05:00Z — Attempt 6 paired execution complete and upstream GCS replay audit PASS

- Hardware execution: Paired DP8xTP8 64-TPU JobSets `canon-v1-apc-m15-off-d12-9f91d930` (control) and `canon-v1-apc-m15-on-d12-9f91d930` (treatment) rendered and launched from committed source `9f91d93001dd5b44659f062626eb93fc65e6fcb4`.
- Control Arm (`off-d12`):
  - 2,560 requests completed across 15 turns with 0.0% prefix cache hit rate.
  - JAX pre-alignment verified: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=117415 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Controlled exit 42 executed with zero backward and zero optimizer commits.
  - Full evidence package persisted to `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-off-d12-9f91d930/attempt-0/` (8 objects, 2.18 GiB).
  - GCS audit `run_m15_replay_gcs_audit.sh` verified and uploaded derived receipts to `derived/m15-replay-audit-v1` with `status=CONTROL_GREEN` (`receipt_sha256=c9550f730bebd3ad37696c52f7365ebac2a6b6fea9382426eec52548eb05c717`, `manifest_sha256=b91cd34c78da6f8ce49a02926a1a27e3dde1583733733603a96160c793254a7b`).
- Treatment Arm (`on-d12`):
  - 2,560 requests completed across 15 turns with **92.9%** prefix cache hit rate.
  - JAX pre-alignment captured exact mismatch: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=119565 bounds=[('S_decode_vs_S_prefill', 1770), ('S_prefill_vs_T_old', 0)]` (**1,770 differing bytes / 748 elements**).
  - Mismatch capsule: 15,148 bytes (`sha256:9e79a18de18c88a2c16b7c6d509198bd141077f7cba466b33602d98eb1c4db77`).
  - Producer unit: 256 rows, 762 KB (`m15_producer_unit.npz`).
  - Serving replay envelope: 3,027 calls, 103.7 MB (`m15_replay_envelope.jsonl`).
  - First-red Incident: Source row 245, request `400-bc7daec5`, serving call 565 (first mismatch call 188), DP rank 0, slot 29, `num_computed_tokens=1248`, 296 exact joins.
  - Full evidence package persisted to `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0/` (9 objects, 1.31 GiB).
  - GCS audit `run_m15_replay_gcs_audit.sh` verified and uploaded derived receipts to `derived/m15-replay-audit-v1` with `status=FRESH_TARGET_RED_FROZEN` (`receipt_sha256=557801a3d397a29ef4bfa69d8f678db9f66f90726ef51eed1faab870158a84ed`, `manifest_sha256=93f56a0a3c970a72907d6f10c9da264158e09557bcadfd7f4d5c4c1d51134e9d`).
- Decision table applied: Off=`CONTROL_GREEN` and On=`FRESH_TARGET_RED_FROZEN` -> **Use the frozen carrier for exact replay and first-red localization; do not rerun rollout.**
- Small machine return bundles archived under `evidence/v1_apc_m15_attempt6_paired_d12_20260825/` with verified `SHA256SUMS` (24 items).
- Claim ceiling promoted to `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`.

## 2026-08-26T01:20:00Z — Phase-C replay-input preparation made executable

- Reanalysis of the complete Attempt-6 small return corrects an ambiguity in
  the earlier wording. Red producer rows are 201 and 245. The canonical first
  mismatch is row 201/completion position 0; row 245's request enters earliest
  at call 164; row 201's request begins at call 187, so the bounded inclusive
  replay prefix contains calls 1 through 188. Row 245/call 565 is the first
  later incident with a complete tensor observer, not the onset.
- Added `scripts/analyze_m15_replay_carrier.py`. It verifies source identity,
  producer/envelope/full-carrier contracts, recomputes byte-level A-B/B-C,
  rejects classification count drift or any B-C red, rejoins token histories,
  and emits `REPLAY_ANALYSIS.json` plus `replay-prefix-plan.jsonl`.
- Added `scripts/run_m15_replay_gcs_prepare.sh`. A bucket-capable executor can
  run one command against the immutable on-arm Attempt-0 URI; it verifies the
  root bundle, extracts and audits the carrier, runs the analyzer, then uploads
  only a versioned derived result with its manifest last.
- Synthetic tests cover the onset/captured-incident distinction and prove that
  A-B classification drift and B-C red are fatal. Shell syntax, Python compile,
  and `git diff --check` pass locally.
- Claim ceiling is unchanged:
  `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`. The scripts prepare input; they
  do not execute a model replay, localize a tensor boundary, or repair APC.

## 2026-08-26T01:45:00Z — Phase-C CL integrated on the latest operator tip

- Final release base is operator tip `c74618b955a2379e94d9be5add1d23f77c86c682`.
  Its two incoming P60/XProf commits do not overlap the M15 target-debug task.
- The Phase-C payload was replayed as one CL in a fresh clean worktree rather
  than inheriting any pre-integration test result.
- Final-tree focused result: 50/50 M15 task tests pass, including the four-test
  analyzer suite with a fake-GCS upload/download/immutable-rerun control.
  Shell syntax, Python compile, secret scan, and `git diff --check` also pass.
- No GCS target analysis or serving replay ran during integration. The claim
  ceiling remains `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`.

## 2026-08-26T12:00:00Z — wide DP8xTP8 first-red observer prepared

- Reconciled the known target red with the later one-host ladder. Attempt 6
  remains APC-off exact and APC-on A-B red by 1,770 bytes / 748 elements with
  B-C zero. The local r10-r13c ladder stayed exact through full M15 chronology,
  so the next discriminating carrier is the known-red DP8xTP8 topology.
- Added renderer modes `none|layer|full`. Layer mode captures all 36 layer
  input/output fingerprints plus final norm and terminal-tail values over
  positions 960..4096. Full mode captures 15 internal checkpoints at exactly
  one layer selected by the layer classifier.
- Added an M15-specific classifier. It does not require impossible coverage of
  continue-decode-only actions. It joins only exact standard-path A/B records,
  requires a completion-position-zero anchor on a red treatment, rejects any
  B-C red, and reports all unobserved red points explicitly.
- Added a deterministic compact bundle with selected raw A/B records,
  mismatch capsule, pre-alignment, replay ledger, receipt, and internal
  `SHA256SUMS`. The bundle contains real token material and is generated
  locally; automatic upload of this new payload is intentionally not enabled
  without separate authorization.
- Host gates pass: classifier/packager 7/7, target renderer 13/13,
  real Step-00 resolver 10/10, all focused M15 tests 59/59, Bash/Python
  syntax, flag audit 386/386, and `git diff --check`.
- The pinned production image aggregate gate also passes and terminates with
  `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=59 ... manifests=3`. This admits
  the renderer, classifier, compact packager, real Step-00 resolution, and
  their negatives in the exact image. It is not a DP8xTP8 numerical result.
- Before publication, the operator branch advanced from `e5c596a4` to
  `8eb65480`. The three incoming CLs overlapped `FLAGS.md`, `00_env.sh`, and
  the Phase4 exact-image runner, so the payload was restored on that latest
  tip and all relevant gates were rerun rather than inheriting the older
  receipt. The final combination passes 63/63 focused M15 tests and the same
  aggregate exact-image marker with `apc_m15_carrier=59`.
- No RoPE, attention/RPA, KV, LM-head, loss, backward, optimizer, B, or
  production APC code changed. No TPU/Kubernetes launch, commit, or push
  occurred.

## 2026-08-27T01:35:00Z — Attempt 11 (d17) paired execution and incident ledger saturation analysis

- Type: target execution / diagnostic evidence / incident analysis.
- Hardware run: `canon-v1-apc-m15-off-d17-f7adb4e6` (off control) and `canon-v1-apc-m15-on-d17-f7adb4e6` (on treatment), both running on 64 v5p TPUs (DP8xTP8) from commit `f7adb4e6fb4b86698c0386079b3a17da031a4578`.
- Confirmed metrics:
  - Prefix cache hit rate: Treatment APC-ON reached **93.1%** (Control APC-OFF 0.0%).
  - Prompt throughput: Treatment reached **4,179 tokens/s** (~9.1x acceleration vs Control ~458 tokens/s).
  - Solve rate: **18.8%** on 15-turn FrozenLake M15 multi-turn task.
  - Forward/Backward coverage: Completed all 36 transformer layers across 64 TPUs for both arms.
  - Observer ledger: 2,153+ records for Arm A (Control) and 2,104+ records for Arm B (Treatment) captured by the wide seam / tail observer.
- Incident boundary: In `90_run.sh`, the legacy P38 serving capture mechanism exceeded `CANON_P38_INCIDENT_MAX_BYTES` (2 GiB bound), raising `[CANON_P38_SERVING_CAPTURE_ERROR] stage=begin error=RuntimeError: P38 incident ledger exceeded its registered byte bound` before executing `classify_m15_apc_wide_seam.py` and uploading `p38_seam.classification.json` to GCS.
- Classification: `INCONCLUSIVE_INCIDENT_LEDGER_SATURATION`. Evidence sealed under `evidence/v1_apc_m15_attempt11_d17_20260827/`.
- Next: raise or bypass legacy incident byte bound during wide layer observer mode in `90_run.sh` and launch fresh Attempt 12 (`d18`).

## 2026-08-27T07:10:00Z — Attempt-9 GCS salvage made the next gate

- Review corrected the immediate ordering after d17. A fresh target retry is
  not yet justified: Attempt 9 claims a completed paired wide-layer run, while
  Git contains only its prose receipt. Its registered GCS roots may already
  contain the missing machine classifiers and compact bundle.
- Added `scripts/run_m15_wide_seam_gcs_salvage.sh` and the host-only analyzer
  `audit_m15_wide_seam_gcs_salvage.py`. The wrapper reads both roots from the
  committed receipt, downloads only registered small objects plus the compact
  tar, verifies classifier aliases/root-manifest binding/terminal markers and
  the tar's internal SHA manifest, then deletes private scratch.
- The return package deliberately excludes the token-bearing tar and raw NPZs.
  It contains only classifier JSONs when valid, a mechanical summary,
  packaging receipt, and `SHA256SUMS`.
- Host tests cover selected-layer success, missing-classifier `INCOMPLETE`,
  conflicting classifier aliases, source-identity conflict, and a fake-GCS
  end-to-end read-only wrapper run. No real GCS access, TPU launch, runtime
  numerical edit, commit, or push occurred in this implementation step.
- Next: bucket-capable execution of the checked-in read-only salvage command;
  return the self-hashed package for analysis. Do not launch d18 or add
  diagnostic rounds before that review.

## 2026-08-27T08:05:00Z — Attempt-9 expected-object salvage is insufficient

- The committed return under
  `evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/` verifies 2/2 manifest
  members and is internally complete as a small audit package.
- Both arms contain a writable `PREFLIGHT.json`, but lack every queried
  terminal marker, root manifest, classifier alias, and compact bundle. No
  machine verdict or tensor boundary was recovered.
- Both preflight markers identify the valid runtime commit
  `3f159250c4781b3faafde238f768457a0478446b`; the later Attempt-9 receipt names
  a different full SHA that does not exist in this repository. The result is
  therefore `SOURCE_MISMATCH`, and the receipt's numerical prose is not signed
  evidence.
- Correction: the salvage wrapper checked seven exact names and did not list
  every object under either root. Before declaring Attempt 9 irrecoverable, a
  bucket-capable executor must return a self-hashed, relative-name-only full
  inventory. It must not download token payloads or launch TPU work.
- If the inventory finds other objects, stop for a narrowly scoped offline
  downloader/classifier. If each arm contains only `PREFLIGHT.json`, proceed
  to wide-mode durability repair: bypass the redundant legacy ledger,
  incrementally persist bounded shards, classify from persisted input, and
  write terminal markers manifest-last from the surviving worker. A new
  one-round DP8xTP8 pair remains separately approval-gated.

## 2026-08-27T08:15:00Z — Attempt-9 full GCS object inventory completed; declared irrecoverable

- Executed read-only recursive GCS object name inventory (`gcloud storage ls --recursive`) across both Attempt-9 roots (`gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-off-d15-3f159250/attempt-0` and `...-on-...`).
- Output sealed under `evidence/v1_apc_m15_attempt9_gcs_full_inventory_20260827/` with `OBJECT_INVENTORY.json`, `PACKAGING.txt`, and `SHA256SUMS` (2/2 OK).
- Result: exactly `{"off": 1, "on": 1}` with only `PREFLIGHT.json` in each root. No surviving tensor shards, raw NPZs, or classifiers exist on GCS.
- Interpretation: Attempt 9 is confirmed irrecoverable from registered GCS roots.
- Next: implement the 4 durability repairs (bypass 2GB legacy incident ledger, stream/incremental shard persistence, classifier from persisted shards, runtime source verification) before requesting user approval for fresh Attempt 12 (`d18`).

## 2026-08-27 — Phase D2 durability implementation host pass

- Added the M15-only `m15-wide-v1` durability profile. It bypasses the
  redundant legacy incident ledger with a signed runtime marker; it does not
  alter APC, RoPE, attention, KV, LM-head, B, loss, backward, or optimizer.
- Complete seam/tail JSON+NPZ pairs are copied into immutable shards bounded
  to 32 pairs and 256 MiB, uploaded, downloaded, SHA-verified, and only then
  marked complete. Periodic ticks examine only unsealed records; final round
  assembly re-hashes the entire sealed union before classification.
- The live worker now seals the round and publishes its classifier plus compact
  bundle before acknowledging the learner. Root `COLLECTED` and `COMPLETE`
  remain manifest-last; partial remote names cannot be overwritten.
- Every persistence action resolves the executing Git checkout and requires it
  to equal the full rendered source SHA. Classifier output is accepted only if
  it byte-matches the sealed-round output.
- Host results: task discovery 75/75; durability 5/5; wide classifier 8/8;
  target carrier 14/14; resolved env 10/10; fake-GCS persistence PASS including
  forced death, source mismatch and terminal ordering; flag audit 387/387;
  flag-auditor tests 2/2; Bash/Python syntax and `git diff --check` pass.
- The standalone repository renderer test is host-blocked by absent `metrax`;
  task-local renderer coverage passes. The real import remains for the pinned
  exact-image gate.
- No exact-image, GCS, TPU/Kubernetes, commit, or push action ran. Claim ceiling:
  `DURABILITY_IMPLEMENTED_HOST_PASS / EXACT_IMAGE_NOT_RUN / TARGET_NOT_RUN /
  ROOT_CAUSE_NOT_LOCALIZED`.

## 2026-08-27 — Phase D2 pinned exact-image pass

- Rebased the uncommitted Phase D2 payload onto current operator tip
  `2655471c004fc5a245ea79e3b44617ded06699f2`; the two incoming performance
  commits did not overlap this task's runtime files.
- The first aggregate run found two harness/test issues rather than an M15
  durability failure: a stale P67 wrong-profile expected string and an
  inaccessible host worktree Git path inside the read-only container.
- Corrected the P67 negative to require the actual stronger profile-admission
  rejection. The exact-image runner now mounts the Git common directory
  read-only and marks `/workspace` safe. Runtime source verification remains a
  live `git rev-parse HEAD` comparison; no mutable receipt substitute exists.
- The isolated fake-GCS test then passed in the image, including forced death,
  source mismatch, bounded shard recovery, manifest-last collection and
  terminal ordering. The full aggregate terminated with
  `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=66 m15_durability=1 ...` on the
  immutable image digest
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- No TPU/Kubernetes launch or real GCS mutation occurred. Claim ceiling:
  `DURABILITY_IMPLEMENTED_HOST_PASS / EXACT_IMAGE_PASS / TARGET_NOT_RUN /
  ROOT_CAUSE_NOT_LOCALIZED`.
