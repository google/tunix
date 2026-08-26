# State

- Status: P60-2G LOCAL/EXACT-IMAGE PASS / TARGET NOT RUN at verified runtime
  commit `fe94345c84c2181ed99997c6768f78d913b2da94` on operator base
  `23e0bddfc4f742eecb66092db5bdc80def9571ca`. P60-2F remains a historical
  CLEAN-SHA ZERO-HP TARGET PASS at
  `5549b5b6`, but under the new navigation criterion that artifact is
  `NUMERICAL/FULL-XPLANE PASS / NATIVE-LIKE UI FAIL / PERFORMANCE
  INCONCLUSIVE`.
- Objective: make the 16 real Zero-HP reverse/accumulate transactions visible
  as Native API train steps 32..47 in a warm update-2 UI capture, with the last
  real train owning optimizer commit and no numerical or synchronization
  change.
- Definition of done for P60-2B: host/static, numerical-neutrality,
  exact-image, flag, diff, and P60-2 document gates pass locally. The separately
  authorized P60-2C one-host certification has now passed its machine and
  full-XPlane readability gates.
- Worktree entry: `/home/yuxuan/code_rl_repro/worktrees/p60_2g_native_steps_0825`
  (data-disk-backed because the root filesystem was full).
- Branch: `local/p60-2g-native-steps-0825`
- Base: `23e0bddfc4f742eecb66092db5bdc80def9571ca`. The exact 22-path P60-2G
  implementation is locally committed at
  `fe94345c84c2181ed99997c6768f78d913b2da94`; its rebase was conflict-free
  because the incoming base commit is confined to
  `tasks/v1-apc-m15-target-debug/`. Historical target facts below retain their
  actual source SHAs.
- Current phase: [P60-2G — Native-like train microsteps and warm UI capture](phases/p60-2g-native-train-steps.md).
  The implementation is committed and clean; after rebase, 13/13 P60, 37/37
  P59, 67/67 V1/P64, 378/378 flags, static gates, and the full pinned
  exact-image ladder pass. The container reported zero TPU chips. Target is
  not run. Historical P60-2F evidence is preserved without relabeling its
  original acceptance.
- Current contract: labels-off remains an exact no-op. On the signed Zero-HP
  arm only, each real reverse/reduce/accumulate transaction owns one
  `StepTraceAnnotation("train")` with step 32..47 for captured update 2; the
  final train closes only after real optimizer commit. Forward/loss remain
  truthful update siblings, and the crossing aggregate `reverse_groups` span
  is omitted. Full XPlane requires same-track hierarchy, 8/8 Steps rows, exact
  backward/optimizer tail and zero captured compiler events. A separate
  streaming trace-JSON gate requires every train/reverse span and the optimizer
  tail to be UI-visible.
- Current artifact budget: the signed runner has no budget override. It records
  every regular `train/xprof` file by logical byte size, warns above
  1,200,000,000 bytes, and makes the arm RED above 1,500,000,000 bytes without
  truncating or deleting the artifact. The classifier recomputes the receipt,
  and the root SHA ledger directly covers raw XProf and semantic Perfetto.
- Preserved fact: the historical Native and Zero-HP captures are complete; the
  historical pair remains `INCONCLUSIVE_INPUT_MISMATCH` and cannot support a
  causal timing ratio.
- Next action: review the complete local diff and, only after explicit approval,
  create the implementation commit, rebase it onto the then-current operator
  tip, and rerun focused/static gates. Update `HANDOFF_P60_2.md` with that final
  full commit SHA before a remote operator runs the clean-tree Zero-HP-only
  command. Push and TPU launch each remain separate approval boundaries. Do
  not rerun Native.
- Current integrated result on `c87838d8...`: P60 task suite 11/11, document
  set 14/14, P59 37/37, V1/P64 67/67, complete 378/378 flag audit, branch
  preflight, syntax, diff, secret, P59 DP4 exact-image, and full
  aggregate-plus-P60 pinned exact-image gates pass. Base `a909fda1` contains
  M15/P64 runtime and evidence changes, so the complete pinned exact-image
  ladder and all focused host gates were rerun on the final rebased
  three-commit tree before publication.
- First target result: RED only at the new hierarchy gate. Training completed
  3/3 updates with 51/51 alignment PASS; old all-plane backward and semantic
  censuses passed; all bounded child spans and 8/8 device Steps rows were
  present. `train` and `zero_tim_update` were missing because their TraceMe
  objects were constructed before `start_trace()`.
- Fix result: task suite 8/8, P60-2 document set, branch preflight, flag audit
  371/371, syntax, diff check, and the full pinned exact-image ladder pass.
  No new synchronization token was added; the three readiness calls shown by
  the diff are exact reindentings of existing calls.
- Second target result: PASS. `p60_readable_zero_dev2_20260825` completed 3/3
  updates with 51/51 alignment PASS and wrapper exit 0. The 778,720,935-byte
  full XPlane has SHA-256
  `4ee534ed81ff4e721a5482ac42057048382161d569f6c608c9c56f29e7aa38fd`.
  The exact host hierarchy includes one Native API-compatible
  `train(step_num=1)`, one `zero_tim_update`, 16 forward groups, 16 reverse
  groups, 16 complete reduction transactions, and one optimizer commit.
  All eight device planes have non-empty `Steps` rows and all five P59
  backward families with decode absent.
- P60-2E clean-SHA target result: immutable run
  `v1_zero-hp_p60_readable_zero_p60_2e_clean_20260825_r1` used source
  `da535c1d5cee7573671fa40809547a6972bec072` with an empty source diff. It
  completed 3/3 updates and 51/51 alignment, passed the complete hierarchy
  metadata census, passed all five backward families plus scaled-step×16 and
  commit×1 on 8/8 planes, and had no decode. Its arm classifier verdict is
  PASS with no reasons.
- Packaging result: RED. The old runner hashed `driver.log` before appending
  its final GREEN line. `sha256sum -c SHA256SUMS` fails only for that file;
  removing the last line reproduces the recorded hash exactly. P60-2F now
  freezes the unique terminal marker before manifest creation, verifies the
  manifest immediately, emits `SHA_LEDGER_PASS` only afterward, and returns
  98 on a ledger failure. The immutable failed-package root is not modified.
- P60-2F historical clean-SHA target result: TARGET PASS. Fresh root
  `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_p60_2f_ledger_clean_20260825_r1`
  used source `5549b5b6046f91406d1897b47618fca83c5fad7d`, tree
  `becf1f03e7659d80618c191ef6e05fce7ec3ba6c`, empty source diff, pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`,
  and wrapper exit 0. It completed 3/3 updates and 51/51 alignment PASS, has
  the exact same-track host hierarchy with `train_step=1`, microsteps 0..15,
  last accumulator 15, optimizer update 1, and has all backward families plus
  scaled-step×16 and commit×1 with decode absent on 8/8 TPU planes. The arm
  classifier is PASS with no reasons and no training/rank-0 traceback was
  recorded. One ignored weakref-finalizer traceback after `TRAINING_DONE` is
  preserved as tail noise; Docker returned 0.
- Historical evidence-ledger result: PASS. `driver.log` contains exactly one
  terminal GREEN marker, the runner emitted `SHA_LEDGER_PASS entries=9`, no
  temporary manifest remains, and an independent `sha256sum -c SHA256SUMS`
  passed 9/9. Manifest SHA-256 is
  `7faaa0d35ca112027f4638e95baa8d2738510eafdf96b5f11b790a8bb1bab0e0`.
  XPlane is 768,320,714 bytes / SHA-256
  `9f317e50e61cc92c5b1ce1c742904a3bb0c5c7ef2d8debb050e244cd9a051de8`;
  trace JSON is 33,680,377 bytes /
  `263c79a0bbabf4269322a8adb552b29b438ffbc82402b89c7a21735e5be65d6a`;
  semantic Perfetto is 12,436 bytes /
  `d4d465639658a9a49741d540a256ca13f5bba27d8610b876ca258a6d2c25e529`.
- Runtime observation: historical target step durations were 646.87, 224.61,
  and 192.93 seconds versus 642.74, 225.70, and 192.33 seconds in the
  immediately preceding clean-SHA run. This is no observed regression in a
  single-run smoke comparison, not a matched unprofiled performance claim.
- Latest-tip integration result: commits `9493928f...` and `c87838d8...` sit
  on operator base `a909fda1...`. The append-only registry merge retains both
  the latest M15 row and the P60 row, and the subsequent publication-time
  rebase completed without conflicts. P64 group-0 diagnostic replay and its
  `sampler_is=none` admission remain nested inside the P60 hierarchy.
  P60/P59/V1/P64 host gates and both pinned exact-image ladders
  pass. No TPU was run for this integrated SHA, so its ceiling is
  LOCAL/EXACT-IMAGE PASS / TARGET NOT RERUN.
- Full-XPlane navigation result: group 0's reverse interval overlaps the
  head, norm, layer, embed, and adjoint backward module families on 8/8 TPU
  planes. Host parent/child containment and stage order pass mechanically.
- Blocker: none for historical P60-2F target acceptance. The integrated SHA is
  not target-certified, P60-2D remains pending, and this evidence-only concern
  is published as the third commit in the approved stack.
- Key artifacts: [P60-2 handoff](HANDOFF_P60_2.md), [plan](plan.md),
  [P60-2B phase](phases/p60-2b-hierarchy-instrumentation.md), and
  [operator runbook](RUNBOOK.md).
- All prior roots and the clean-SHA packaging-RED root remain preserved. The
  authorized P60-2G implementation commit, rebase, local validation, and
  handoff preparation have been consumed. Publication requires a final
  remote-tip equality check and ordinary fast-forward push; the push receipt
  is external to this static ledger. No image publication, Native rerun, or
  TPU launch is authorized.
- Updated: 2026-08-26.
