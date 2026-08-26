# Log

## 2026-08-24 — implementation start

- Rejected the historical P58 DeepSWE Qwen3-4B DP1×TP4 no-commit carrier for
  this purpose: it launches `train_deepswe_nb.py` and does not exercise the
  current GSM8K/P59 update path.
- Reused the proven P59 DP4×TP1 geometry and the existing GSM8K vanilla stock
  admission, but moved both arms onto the same whole-update XProf window.
- Added a default-off matched-work selector and exact pre-update token/
  advantage receipts. Runtime and TPU validation remain pending.

## 2026-08-24 — one-host hardening

- A first Native development run completed 3/3 updates and produced a healthy
  159 MB XPlane plus 35 MB trace, but correctly exposed two postflight bugs:
  root-owned artifacts and reuse of the P55 segmented-backward/19-track census
  for a stock DP4 trainer. Added container-side permission normalization and
  arm-aware XPlane/Perfetto censuses. Reclassification proves 8/8 planes each
  contain 16 stock `jit__train_step` modules and zero decode modules.
- The first Zero-HP attempt failed before training because the runner omitted
  the registered P60 deterministic 1024/256 carrier. Both arms now require
  `CANON_P60_DETERMINISTIC_AB=1`; a narrow signed Native admission preserves
  stock numerical execution while fixing seed/schedule/shape.
- The second Zero-HP attempt reached a strict all-zero pre-gate and exposed
  that the receipt observed `ObservedTrainExample` rather than its underlying
  train example. The receipt now unwraps the host-only sidecar before hashing.
- Final Zero-HP run `zero_dev3_20260824` is GREEN: 3/3 updates, 51/51 strict
  alignment PASS, 8/8 planes with all five P59 backward families and no decode,
  813,929,492-byte XPlane, 33,974,285-byte trace JSON and 12,436-byte semantic
  Perfetto.

## 2026-08-24 — one-host final result

- Final Native run `native_dev2_20260824` is GREEN: 3/3 updates; every one of
  eight TensorCore planes contains exactly 16 stock `jit__train_step` modules
  and no decode module. Its XPlane is 159,449,612 bytes
  (`a2367fc94d4fa3643b5895e9cef068383c8d464bfc8b945bc95e0ab14186e4a6`),
  trace JSON is 34,779,292 bytes
  (`48dab817baf38549db84429f358f7f1f0ccb7d363ca59bceb175088f1886214f`),
  and semantic Perfetto is 15,302 bytes
  (`f63521cdd7263d0d3e0c1b4b92305aded078b07756d4bd2c52b0c0b6bb9e16c7`).
- Zero-HP `zero_dev3_20260824` remains GREEN on all eight planes. The five P59
  backward families are present, decode is absent, and the profiled transaction
  includes the fixed reducer and optimizer commit.
- The pair classifier returns `INCONCLUSIVE_INPUT_MISMATCH`, not FAIL. Both
  arms used the same source diff, image, model snapshot, DP4xTP1 topology,
  update 1 window, prompt ids, prompt mask, policy version and static shape.
  Completion ids/masks and advantages differ. Same seed therefore did not
  freeze stochastic work across numerically different inference programs.
- The `xprof-trace-analysis` comparison confirmed the expected program-shape
  change (Native monolithic `jit__train_step`; Zero-HP decomposed canonical
  forward/P59 backward), but is not a timing verdict because work differs.
  It also showed why completeness is gated on XPlane: Native trace JSON exposes
  11 modules on the selected plane while the full XPlane proves 16/16 on all
  planes.
- The repository `read-xprof` workflow was also applied to the full XPlanes.
  Native has 128 total `jit__train_step` programs across eight planes and
  18.47 device-seconds of monolithic update work. Zero-HP has the expected
  decomposed modules: canonical forward, P59 parallel backward, fixed reducer,
  replica comparison, and optimizer transaction. Its largest all-plane module
  families are forward layer (49.82s), fixed reduction (24.92s), parallel layer
  backward (13.17s), and replica comparison (9.54s). These numbers are
  attribution only and cannot be converted into an A/B speed ratio because the
  profiled work hashes differ.
- `scripts/analyze_gsm8k_xprof_pair.sh` was exercised on the real artifacts. It
  returned the expected exit 3, preserved both arm PASS verdicts, named the
  four differing arrays, produced the compact XProf comparison, and wrote a
  SHA ledger. This removes wildcard/manual-path ambiguity from the handoff.

## 2026-08-24 — operator handoff expansion

- Rewrote `HANDOFF.md` as a cold-start execution recipe: exact Native and
  Zero-HP commands, clean-vs-dirty evidence grade, expected arm-specific
  backward markers, output-root derivation, pair exit-code handling, artifact
  authority, and the complete return manifest. No runtime behavior changed.

## 2026-08-25 — P60-2 readability diagnosis and plan

- Re-read both immutable full XPlanes using the read-xprof method. Native
  TPU:0 contains 672 XLA-module events and a host `train` annotation; Zero-HP
  TPU:0 contains 59,028 module events and no host `train` annotation.
- Preserved the existing completeness verdict: Zero-HP backward is present on
  8/8 TensorCore planes, decode is absent, and the run passed 51/51 strict
  alignment. The defect is navigation/hierarchy, not capture loss.
- Confirmed that both historical captures already have non-empty device
  `Steps` rows on all eight TPU planes. Native additionally supplies the exact
  host `StepTraceAnnotation("train")`; revised Zero-HP must supply both.
- Traced the source difference to stock `PeftTrainer.train` owning a
  `StepTraceAnnotation("train")`, while G6 directly calls
  `_run_p28_g6_update` and only writes the separate official semantic
  `peft_train` span. Existing `CANON_XPROF_LABELS=1` names individual JITs but
  supplies no update/group parent intervals.
- Added the P60-2A..D phase plan, a cold-start implementation handoff, and a
  copyable executor prompt. The proposed change is profile-only and reuses the
  existing flag; it forbids new synchronization, numerical changes, per-layer
  span explosion, semantic-Perfetto changes, or raw-event filtering.
- No TPU job, Kubernetes object, commit, or push was created.

## 2026-08-25 — P60-2B implementation and local gates

- Created the executor-owned worktree
  `/home/yuxuan/code_rl_repro/worktrees/p60_2b_xprof_hierarchy_0825` at
  fetched tip `16db308b35c6e625d6a47c40b039ecfea317d9b3`.
- Added the shared labels helper and bounded host hierarchy without changing
  JIT/shard-map/reducer/optimizer expressions. The G6 entry now uses Native's
  exact `StepTraceAnnotation("train", step_num=...)` API; the fixed existing
  commit readiness boundary is merely enclosed by `optimizer_commit`.
- Added a pure interval/count validator and a real full-XPlane census requiring
  the one host step parent, 16 group transactions, and non-empty `Steps` lines
  on exactly eight TPU device planes. Revised Zero classification requires this
  census; Native's historical classifier behavior remains unchanged.
- Historical XPlane reads produced expected RED hierarchy verdicts while
  correctly recovering Native device `Steps` counts
  `672,672,576,576,576,576,576,576` and old Zero counts
  `59028,59028,37260,37260,37260,37260,37260,37260`.
- Local task tests passed 7/7; P59 host contracts passed 37/37; V1 Phase4 host
  contracts passed 34/34. The complete pinned CPU image gate passed existing
  V1/P59/P62 controls, labels off/on, the one-ULP fail-closed control, and the
  host annotation API probe.
- Final static gates: flag registry 371/371 PASS, branch preflight PASS,
  `git diff --check` clean, P60-2 doc set 12/12 PASS, changed-patch secret scan
  zero matches, and semantic Perfetto vocabulary unchanged.
- No TPU/Kubernetes action, commit, push, or image publication occurred.

## 2026-08-25 — P60-2C one-host canary preflight

- The user approved exactly one direct four-chip v5p Zero-HP development
  canary with the dirty-tree override, one update-level
  `train(step_num=1)` parent, and no Native rerun.
- Bound launch identity: branch
  `local/p60-2b-xprof-hierarchy-0825`, HEAD
  `16db308b35c6e625d6a47c40b039ecfea317d9b3`, label
  `p60_readable_zero_dev1_20260825`, and fresh root
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_dev1_20260825`.
- Preflight gates passed immediately before launch: branch runtime preflight
  PASS with the expected dirty development tree; flag registry 371/371 PASS;
  `git diff --check` exit 0; task suite 7/7 plus P60-2 document set and CPU
  contract PASS. Hostname, pinned local image ID, model/data assets, and fresh
  artifact root checks passed.
- A concurrent randomly named container was inspected before launch. It uses
  the same pinned image but is non-privileged and has no device mappings, so it
  does not own the TPU lane. No P51, P59, or V1 GSM8K XProf carrier is active.
- This run can produce analysis-grade target evidence only. Signed clean-SHA
  acceptance, Native rerun, Kubernetes, commit, push, and image publication
  remain outside the authorization.

## 2026-08-25 — P60-2C first canary RED and localized fix

- Ran the one authorized Zero-HP label
  `p60_readable_zero_dev1_20260825`. The container exited 0 after 1,501
  seconds, completed 3/3 optimizer updates, and produced 3 pre-alignment plus
  48 update-alignment PASS records with zero FAIL.
- The update-only window markers were exact: armed step 1, started at update
  entry step 1, and stopped at step-completed step 2. The XPlane is
  778,688,563 bytes with SHA-256
  `06b0c43c34361eab3a976d5870bd5b3b49a898500f741ab3676d06adc1da12a2`;
  the trace JSON is 33,971,974 bytes with SHA-256
  `4558f01b505cf496db2340e1d916fa29986cb3048bae272c606c18eb336c24db`.
- Old target gates passed: P59 backward families were present and decode was
  absent on all 8/8 TPU planes; semantic Perfetto was GREEN. The hierarchy
  census recovered exact child counts (16 forward groups, 16 reverse groups,
  16 of each transaction stage, one optimizer) and non-empty device Steps on
  8/8 planes.
- The first failing boundary was hierarchy parent capture:
  `train:count=0 expected=1` and
  `zero_tim_update:count=0 expected=1`. The classifier correctly returned
  FAIL with reason `hierarchy_census_rc=1`; the complete RED root was
  preserved.
- Root cause is confirmed from the real XPlane and pinned JAX TraceMe
  lifecycle: both parent annotation objects were constructed before
  `_canon_xprof_update_entry()` called `start_trace()`. Their intervals began
  outside the window; children constructed after start were captured.
- Applied the smallest local fix: move only the two parent constructions after
  the existing trace-start call and add a source-order regression test. No
  retry was launched; a fresh direct-TPU run requires separate approval.
- Post-fix gates passed: task suite 8/8 plus P60-2 document set; branch
  preflight PASS; flag registry 371/371 PASS; syntax and `git diff --check`
  exit 0. The complete pinned CPU-image ladder ended in
  `V1_HP_EXACT_IMAGE_PASS`,
  `P60_XPROF_ANNOTATION_API_PASS step=train step_num=1`, and
  `P60_2B_EXACT_IMAGE_PASS hierarchy_api=1 labels_off_on=1 one_ulp=1
  p59_v1=1 tpu_devices=0`.
- The no-new-sync diff audit shows the same three readiness calls removed and
  re-added only by indentation. Source counts remain `block_until_ready=15`,
  `device_get=21`, `optimization_barrier=0`, `jax.jit=21`, and
  `shard_map=57` across the audited runtime files.

## 2026-08-25 — P60-2C second canary authorized

- The user explicitly authorized one fresh direct-v5p Zero-HP retry after the
  parent-construction ordering fix passed local and pinned-image gates.
- Bound label and fresh root:
  `p60_readable_zero_dev2_20260825` and
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_dev2_20260825`.
  Native is not rerun and the dev1 RED root remains unchanged.
- Immediate preflight passed: fresh root, task suite 8/8, P60-2 document set,
  `git diff --check`, flag registry 371/371, branch runtime preflight, and an
  idle TPU lane. This remains a dirty-tree analysis-grade canary.

## 2026-08-25 — P60-2C second canary PASS

- Ran the single authorized retry `p60_readable_zero_dev2_20260825`. It
  completed 3/3 optimizer updates, 3/3 pre-alignment PASS records, and 48/48
  update-alignment PASS records with zero FAIL. The container ended after
  `TRAINING_DONE max_steps=3` with `docker_exit=0`; the wrapper returned GREEN.
- Capture markers were exact: armed step 1, started at update entry step 1,
  and stopped at step-completed step 2. The complete XPlane is 778,720,935
  bytes (`4ee534ed81ff4e721a5482ac42057048382161d569f6c608c9c56f29e7aa38fd`)
  and the trace JSON is 33,809,800 bytes
  (`dc68ba730cb54108d6be27147cdf315871819ec24193880473ed61bbcfb240a2`).
- The all-plane census passed: all five P59 backward families are present and
  decode is absent on 8/8 TPU planes. The semantic census passed with the
  unchanged official flat vocabulary and one profiled update.
- The repaired hierarchy census passed with one
  `train(step_num=1, _r=1)`, one `zero_tim_update`, 16 forward groups, 16
  reverse groups, 16 complete report/reduce/accumulate transactions, one
  optimizer, and non-empty `Steps` rows on 8/8 device planes.
- A direct read of the full XPlane confirmed parent containment and ordering.
  Reverse group 0 overlaps head, norm, layer, embed, and adjoint backward
  modules on each of the eight TPU planes, satisfying the navigation check.
- The post-training weakref finalizer traceback is the known cleanup-only
  message after `TRAINING_DONE`; it did not affect exit 0 or any artifact.
  This is analysis-grade dirty-tree evidence, not a signed clean-SHA receipt.
  No further TPU run, Native rerun, Kubernetes action, commit, push, or image
  publication is authorized.
- Post-run local closure passed: task suite 8/8, P60-2 document set 12/12,
  flag registry 371/371, branch runtime preflight, and `git diff --check`.
  The registry count is 371 on the fetched P62-inclusive source tip; 370/370
  was the correct count only on the earlier registry revision.

## 2026-08-25 — P60-2E truthful microstep metadata local PASS

- Type: code change and decision.
- Fact: Native and Zero-HP device `Steps` rows both contain numeric derived
  event names. Native host `train` events are microstep annotations; Zero-HP's
  decomposed all-forward-then-reverse schedule cannot truthfully reuse their
  contiguous monolithic shape.
- Action: kept one whole-update `train(step_num=1)`; added
  `micro_step=0..15` and one `is_last_accumulate=1` to the existing
  `gradient_accumulate` spans, plus `update_step=1` to optimizer commit.
  Strengthened the hierarchy census and added source/API/negative controls.
- Result: task 9/9 and document set 13/13 PASS. Pinned exact-image ended in
  `V1_HP_EXACT_IMAGE_PASS`,
  `P60_XPROF_ANNOTATION_API_PASS step=train step_num=1 micro_steps=0..15
  last_accumulate=15 optimizer_update=1 metadata=integer xplane=1 trace=1`,
  and `P60_2B_EXACT_IMAGE_PASS hierarchy_api=1 labels_off_on=1 one_ulp=1
  p59_v1=1 tpu_devices=0`.
- Negative result: strengthened census on immutable dev2 preserved every old
  count and 8/8 device Steps rows, then returned exactly 33 RED reasons for
  the intentionally absent new metadata. Dev2 remains valid P60-2C evidence
  and does not certify P60-2E.
- Static result: flag audit 371/371, branch preflight, diff, syntax, semantic
  vocabulary, no-new-sync, and secret gates pass. P60-2E is target not run;
  no TPU, Kubernetes, commit, push, or image publication occurred.

## 2026-08-25 — P60-2E same-track gate migrated to P63 tip

- Type: migration, gate hardening, and local/exact-image checkpoint.
- Source: fetched `origin/yuxzhang/canon-zero-tim` tip
  `cdd3987caa648e6112ee8fc184b2e3421de3a4b2`; created
  `local/p60-2e-microstep-latest-0825`. The root filesystem had no room to
  expand another checkout, so the independent worktree is data-disk-backed
  with an entry under the conventional `worktrees/` path. No existing
  worktree or evidence was deleted.
- Action: migrated the P60-2B/P60-2E diff, required every hierarchy span on
  the same `/host:CPU` `python3` line, added a wrong-track synthetic negative,
  and made the pinned annotation probe fail closed on process/thread drift.
  Documentation now says Native API-compatible only; Zero-HP retains one
  whole-update parent and does not match Native cadence/cardinality/program
  shape.
- Full-XPlane negative result: immutable dev2 retained every hierarchy count,
  the single host track, and device `Steps` counts
  `49124,49124,27356,27356,27356,27356,27356,27356`; it returned exactly 33
  missing-metadata reasons and no track reason.
- Local result: task suite 9/9 and P60-2 document set 13/13 PASS. Exact-image
  ended in `V1_HP_EXACT_IMAGE_PASS`,
  `P60_XPROF_ANNOTATION_API_PASS ... host_plane=/host:CPU
  host_line=python3 xplane=1 trace=1`, and
  `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Static result: flag audit 372/372, branch preflight PASS with dirty=21,
  `git diff --check`, Python/shell syntax, no-new-sync counts, unchanged
  official Perfetto vocabulary, and changed-file secret scan all pass.
- Boundary: P60-2E remains `TARGET NOT RUN`. No TPU, Kubernetes, commit, push,
  or image publication occurred.

## 2026-08-25 — P60-2E exact receipt and optimizer-tail gates closed

- Type: local gate hardening and evidence replay.
- Exact-image action: changed the annotation probe from one synthetic final
  accumulator to 16 actual `gradient_accumulate` spans. It now requires
  `group_index == micro_step == 0..15`, only 15 marked last, and every selected
  hierarchy event on one `/host:CPU` `python3` track before printing the
  `micro_steps=0..15` marker.
- Device action: made the module census require exactly TPU planes 0..7 and,
  on every plane, the five P59 backward families,
  `jit__precomputed_gradient_scaled_step=16`,
  `jit__precomputed_gradient_commit=1`, and decode absent. Added fail-closed
  CPU negatives for a missing commit, only 15 scaled steps, and a missing
  device plane.
- Local result: 10/10 task tests and the 13-file P60-2 document gate pass.
  The immutable dev2 XPlane replays GREEN on all 8/8 device planes with
  `scaled_step=16/16,commit=1/1` and decode absent.
- Exact-image result: the full P63-inclusive ladder ends in
  `V1_HP_EXACT_IMAGE_PASS`, then
  `P60_XPROF_ANNOTATION_API_PASS step=train step_num=1 micro_steps=0..15
  last_accumulate=15 optimizer_update=1 metadata=integer
  host_plane=/host:CPU host_line=python3 xplane=1 trace=1`, and finally
  `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Static result: flag audit 372/372, branch preflight PASS at
  `cdd3987caa648e6112ee8fc184b2e3421de3a4b2`, `git diff --check`, and shell
  syntax pass. Production synchronization-token counts remain identical to
  HEAD.
- Boundary: this closes the two local evidence ambiguities only. P60-2E is
  still `TARGET NOT RUN`; no TPU, Kubernetes, commit, push, or image
  publication occurred.

## 2026-08-25 — P60-2E clean-SHA core PASS, evidence packaging RED

- Source: locally committed implementation
  `da535c1d5cee7573671fa40809547a6972bec072`, clean source diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  The authorized fresh Zero-HP run root is
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_p60_2e_clean_20260825_r1`.
- Core result: 3/3 updates, 51/51 alignment, exact hierarchy metadata on the
  single host track, non-empty Steps on 8/8 planes, all required backward
  families and per-plane scaled-step×16 plus commit×1, decode absent, semantic
  census GREEN, and arm classification PASS with no reasons.
- Artifact result: full XPlane 768,217,129 bytes with SHA-256
  `fb9d41d1ccb50948aa7704b9d325ffcfae81659a4c3dc8d29e3f669f039d56bb`;
  trace JSON 33,812,925 bytes with SHA-256
  `c3251f92f57d7ac7900ec6ae0c68da93e1e226ed1ee5c0610fdf7b841a92178d`;
  semantic Perfetto 12,436 bytes with SHA-256
  `784a31c8924ff8a73b1893f96000eeee2049cb70954257efc16c92a69186c724`.
- Packaging result: `sha256sum -c SHA256SUMS` fails only on `driver.log`.
  The manifest records
  `f025bf90b76d668b37311ab420571e90ffaca3b2f10248bf082aaa2b259cd8d7`,
  the final file hashes to
  `d946011e6fa704f2db20fa75e170b20373296420335c33c91d043c1470769a8b`,
  and removing the last GREEN line reproduces the recorded value. Root cause:
  the old runner wrote the manifest before appending its terminal marker.
- Verdict: `CORE TARGET GATES PASS / EVIDENCE PACKAGING RED`. The immutable
  root was not modified. The weakref finalizer traceback occurs after
  `TRAINING_DONE` and is recorded as tail noise, not the packaging cause.

## 2026-08-25 — P60-2F additive ledger fix started

- Added a sourced finalization helper that selects GREEN/RED first, freezes a
  unique terminal marker in `driver.log`, atomically installs `SHA256SUMS`,
  verifies it immediately, and emits `SHA_LEDGER_PASS` only after success.
  Write or verification failure returns 98 with `SHA_LEDGER_RED`; no hashed
  file is written after verification.
- Focused CPU controls pass for execution GREEN/exit 0, execution RED/exit 1,
  and post-manifest tamper/exit 98. The tamper branch retains the already
  hashed execution GREEN marker but correctly withholds `SHA_LEDGER_PASS`.
- Final local result: task suite 11/11 and document set 14/14 pass. The full
  pinned-image P63/V1/P59 ladder passed on immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  and ended with `V1_HP_EXACT_IMAGE_PASS`, the complete P60 annotation API
  receipt, and `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Static result: flag audit 372/372, expected-branch preflight, syntax,
  `git diff --check`, changed-file secret scan, and no-production-training-
  source-change gates pass. No commit, amend, push, image mutation, Kubernetes
  action, or TPU rerun occurred.

## 2026-08-25 — P60-2F migrated to latest operator tip

- Fetched `origin/yuxzhang/canon-zero-tim` at
  `53876c15f407435dbd44680ad18f5f8e88f3c255`. The two incoming commits add
  FrozenLake/M15 evidence only; the P60 runtime/test tree is unchanged.
- Preserved the uncommitted P60-2F tree, rebased P60-2E without conflict as
  local commit `d0c6c67474d836664bab69eed665d96d6ff53a25`, and restored P60-2F
  without conflict. The prior clean target remains attributed to its actual
  immutable source `da535c1d5cee7573671fa40809547a6972bec072`.
- Latest-base gates pass: task suite 11/11, document set 14/14, flag audit
  372/372, branch preflight, `git diff --check`, and the complete pinned-image
  ladder ending in `V1_HP_EXACT_IMAGE_PASS`,
  `P60_XPROF_ANNOTATION_API_PASS ... micro_steps=0..15 ...`, and
  `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- No functional commit, push, TPU/Kubernetes launch, or mutation of the
  packaging-RED run root occurred. P60-2F remains TARGET NOT RERUN.

## 2026-08-25 — P60-2F historical clean-SHA Zero-HP TARGET PASS and latest-tip admission

- Historical source: authorized additive local commit
  `5549b5b6046f91406d1897b47618fca83c5fad7d`, tree
  `becf1f03e7659d80618c191ef6e05fce7ec3ba6c`, empty source diff, pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  Fresh root:
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_p60_2f_ledger_clean_20260825_r1`.
- Runtime: wrapper exit 0, 3/3 update receipts PASS, 3/3 pre-alignment plus
  48/48 micro-alignment PASS, no training/rank-0 traceback. One ignored
  weakref-finalizer traceback after `TRAINING_DONE` is preserved as tail noise;
  Docker returned 0. Global-step durations were 646.87, 224.61, and 192.93
  seconds versus 642.74, 225.70, and 192.33 seconds in the prior clean-SHA
  run; this is no observed smoke-test regression, not a matched unprofiled
  performance claim.
- Hierarchy: one same-track `train_step=1`; 16 forward groups, 16 reverse
  groups, 16 complete transactions; `micro_step=0..15`, unique last=15,
  optimizer update=1; non-empty Steps rows on all eight TPU planes.
- Device/semantic: all five backward families present on 8/8 planes, each
  plane has scaled-step×16 and commit×1, decode absent, and the single-update
  semantic census is GREEN. Classification is PASS with `reasons=[]`.
- Ledger: runner printed `SHA_LEDGER_PASS entries=9` before terminal Zero-HP
  GREEN; `driver.log` has exactly one terminal marker, no temporary manifest
  remains, and independent `sha256sum -c SHA256SUMS` passed 9/9. Manifest
  SHA-256 is
  `7faaa0d35ca112027f4638e95baa8d2738510eafdf96b5f11b790a8bb1bab0e0`.
- Artifacts: XPlane 768,320,714 bytes /
  `9f317e50e61cc92c5b1ce1c742904a3bb0c5c7ef2d8debb050e244cd9a051de8`;
  trace JSON 33,680,377 bytes /
  `263c79a0bbabf4269322a8adb552b29b438ffbc82402b89c7a21735e5be65d6a`;
  semantic Perfetto 12,436 bytes /
  `d4d465639658a9a49741d540a256ca13f5bba27d8610b876ca258a6d2c25e529`.
- Latest-tip integration: rebased onto
  `a909fda18ce97c885f9e5dcbd687e0b62c808c91`, replayed the implementation as
  `9493928fda4fac3186d4a7eaa49ad33ba59c8162` and the ledger change as
  `c87838d8a77ddca33800df024b3fef9edc503327`. P60 11/11, P59 37/37,
  V1/P64 67/67, flags 378/378, P59 DP4 exact-image, and the full
  aggregate-plus-P60 pinned exact-image ladder pass. The final P60 marker is
  `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Exact-image provenance: base `a909fda1` includes M15/P64 runtime and evidence
  changes, so the previous byte-identical closure is not reused. The complete
  aggregate-plus-P60 pinned exact-image ladder and the P60/P59/V1-P64 host
  gates were rerun on the final rebased three-commit tree before publication
  and exited 0.
- Verdict: historical `5549b5b6` is CLEAN-SHA TARGET PASS; integrated
  `c87838d8` is LOCAL/EXACT-IMAGE PASS / TARGET NOT RERUN. The prior
  packaging-RED root remains immutable. This evidence-only concern is
  published as the third CL in the approved stack after all publication gates;
  no image publication, Native rerun, or extra TPU launch occurred.

## 2026-08-25 — P60-2G Native-like train microsteps started

- Created independent data-disk worktree
  `local/p60-2g-native-steps-0825` at fetched operator tip `9f91d930`. The
  phase is local and uncommitted; no TPU, commit, push, or remote mutation is
  authorized.
- Baseline diagnosis: Native full XPlane has 16 real train accumulation steps
  16..31 plus terminal iterator probe 32. Historical P60-2F Zero-HP has one
  62.66-second `train(1)`. Its first reverse/reduce transaction contains
  first-use compilation; update 2 is the first warm post-sharding candidate.
- Implemented a profiling-only lifecycle schedule for the signed Zero-HP arm:
  the 16 real reverse/reduce/accumulate transactions become train 32..47 and
  the last real train closes after optimizer commit. Other workloads retain
  their existing envelope and labels-off remains a no-op.
- Added full-XPlane warm compiler rejection and a streaming trace-JSON UI gate.
  The immutable P60-2F JSON replay returned the expected RED:
  `trace_event_count=1000448`, `train=1`, `forward_group=16`, and zero loss,
  reverse, accumulator, or optimizer spans; 19 reasons before separating the
  full-XPlane-only `_r` marker.
- First focused CPU checkpoint: 12/12 tests PASS. Remaining work is the updated
  classifier/runner fixture replay, P59/V1/P64/flag/static gates, pinned
  exact-image probe, and final documentation closure.

## 2026-08-25 — P60-2G local and exact-image gates closed

- Host gates: P60 12/12 and document set 15/15, P59 37/37, V1/P64 67/67,
  flag audit 378/378, branch preflight, syntax, `git diff --check`, changed-
  diff secret scan, and no-new-sync token audit all PASS.
- Full pinned image ladder passed on immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  The container repeatedly reported zero TPU chips. Final P60 API receipt:
  `train_steps=32..47`, microsteps 0..15, last 15, optimizer update 2 owned by
  the last train, compiler events 0, one host track, XPlane and trace present.
- Old immutable P60-2F JSON fails the new UI gate with 1,000,448 events and no
  loss/reverse/optimizer tail. Its full XPlane preserves 8/8 Steps rows and all
  old transactions but fails the new step/metadata contract and reports three
  contained events in each of the three compiler families.
- Verdict: `LOCAL/EXACT-IMAGE PASS / TARGET NOT RUN`. No commit, push, Native
  rerun, TPU launch, image publication, or remote mutation occurred.

## 2026-08-26 — P60-2G bounded XProf delivery gate closed locally

- Type: additive artifact-budget hardening and remote handoff.
- Decision: retain the complete update-2 XPlane and UI trace while fixing the
  signed runner budget at soft 1,200,000,000 / hard 1,500,000,000 logical
  bytes. No new environment flag or operator override was added.
- Implementation: added `census_gsm8k_xprof_size.py`, a deterministic regular-
  file inventory and JSON receipt. The arm classifier recomputes the current
  file set, sizes, counts, limits, and status rather than trusting the census
  return code. Above the hard maximum the arm is RED but the root is preserved.
  The final manifest now directly covers every raw XProf file and semantic
  Perfetto in addition to censuses, classification, reports, and logs.
- Negative controls: PASS, soft-WARN, hard-RED, stale receipt, ordinary
  post-manifest tamper, and raw-XPlane post-manifest tamper all fire as
  intended. Focused P60 is 13/13 plus document set 15/15; P59 is 37/37; V1/P64
  is 67/67.
- Existing-artifact replay: immutable P60-2F resolves to exactly two XProf
  files totaling 802,001,091 bytes: XPlane 768,320,714 and trace JSON
  33,680,377. Marker:
  `V1_GSM8K_XPROF_SIZE_CENSUS_GREEN status=PASS xprof_bytes=802001091
  soft_warning_bytes=1200000000 hard_max_bytes=1500000000 files=2 xplanes=1
  traces=1 reasons=[]`.
- Exact-image: the complete aggregate ladder passed on pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
  the container repeatedly reported zero TPU chips and ended in
  `V1_HP_EXACT_IMAGE_PASS`, the 13-test P60 CPU PASS, the full annotation API
  receipt for train 32..47, and `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Final static gates: complete flag audit is 378/378/378 and
  `FLAG_AUDIT_PASS`; branch preflight PASS reports the expected local branch,
  base `9f91d930`, dirty=22, and zero remote credentials; `git diff --check`,
  Python AST for ten files, shell syntax, changed-diff secret scan, and the
  no-new-sync token audit all PASS.
- Handoff: `HANDOFF_P60_2.md` now contains the clean-tree Zero-HP-only remote
  command, fixed tracer tuple, size markers, inspection commands, and failure
  preservation rules. It intentionally withholds a source SHA until a local
  implementation commit is separately approved and created.
- Boundary: `LOCAL/EXACT-IMAGE PASS / TARGET NOT RUN`. No commit, push, TPU or
  Kubernetes launch, Native rerun, image publication, or remote mutation
  occurred.
- Integration check: remote-tracking tip advanced by one to
  `23e0bddfc4f742eecb66092db5bdc80def9571ca`. The incoming evidence commit
  changes only `tasks/v1-apc-m15-target-debug/`; its path intersection with the
  P60-2G concern is empty. The dirty worktree was not rebased. After explicit
  commit approval, rebase the clean implementation commit onto the then-current
  tip and rerun the focused/static gates before recording a launchable SHA.
