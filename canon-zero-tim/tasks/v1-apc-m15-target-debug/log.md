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
