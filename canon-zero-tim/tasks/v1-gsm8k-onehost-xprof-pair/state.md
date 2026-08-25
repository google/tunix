# State

- Status: P60-2E CLEAN-SHA CORE TARGET GATES PASS / EVIDENCE PACKAGING RED;
  P60-2F LOCAL/EXACT-IMAGE PASS / TARGET NOT RERUN.
- Objective: make the Zero-HP P59 update capture human-readable as one training
  step with stable update, group, backward, reducer, and optimizer hierarchy,
  without changing its numerical program or adding synchronization.
- Definition of done for P60-2B: host/static, numerical-neutrality,
  exact-image, flag, diff, and P60-2 document gates pass locally. The separately
  authorized P60-2C one-host certification has now passed its machine and
  full-XPlane readability gates.
- Worktree entry: `/home/yuxuan/code_rl_repro/worktrees/p60_2e_microstep_latest_0825`
  (data-disk-backed because the root filesystem was full).
- Branch: `local/p60-2e-microstep-latest-0825`
- Base: `53876c15f407435dbd44680ad18f5f8e88f3c255` from the fetched
  `origin/yuxzhang/canon-zero-tim` tip. P60-2E was rebased without conflict as
  local commit `d0c6c67474d836664bab69eed665d96d6ff53a25`; the immutable clean-SHA
  target below correctly retains its historical source `da535c1d...`.
- Current phase: [P60-2F — evidence-ledger finalization](phases/p60-2f-evidence-ledger-finalization.md).
  The P60-2E implementation was committed locally as
  `da535c1d5cee7573671fa40809547a6972bec072` and received one clean-SHA
  Zero-HP target run. P60-2D remains pending.
- Implemented contract: labels-off is an exact no-op; labels-on adds the
  Native API-compatible `StepTraceAnnotation("train", step_num=1)` envelope
  and bounded update/group/transaction annotations. This is API compatibility
  only: Zero-HP keeps one whole-update parent and does not reproduce Native's
  microstep cadence, cardinality, or monolithic graph. Accumulator spans expose
  `micro_step=0..15` and one `is_last_accumulate=1`; optimizer commit exposes
  `update_step=1`. The full-XPlane census requires all hierarchy spans on the
  same `/host:CPU` `python3` track, all metadata, and non-empty `Steps` rows on
  all 8 TPU device planes. The device-module census separately requires the
  exact eight TPU planes and, on each, scaled-step×16 plus commit×1.
- Preserved fact: the historical Native and Zero-HP captures are complete; the
  historical pair remains `INCONCLUSIVE_INPUT_MISMATCH` and cannot support a
  causal timing ratio.
- Next action: review the P60-2F diff and stop for a separate additive-commit
  approval. A fresh target rerun would require a later, separate launch
  approval. Do not rerun Native.
- Current P60-2F result on the fetched `53876c15...` base: task suite 11/11,
  document set 14/14, full pinned exact-image ladder,
  GREEN/RED/post-manifest-tamper/duplicate-marker controls, 372/372 flag audit,
  branch preflight, syntax, diff, secret, and no-production-source-change gates
  pass.
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
- Full-XPlane navigation result: group 0's reverse interval overlaps the
  head, norm, layer, embed, and adjoint backward module families on 8/8 TPU
  planes. Host parent/child containment and stage order pass mechanically.
- Blocker: the clean-SHA target's core evidence is good, but its SHA ledger is
  invalid, so `TARGET PASS` is prohibited. P60-2F is uncommitted and has not
  been target-rerun.
- Key artifacts: [P60-2 handoff](HANDOFF_P60_2.md), [plan](plan.md),
  [P60-2B phase](phases/p60-2b-hierarchy-instrumentation.md), and
  [operator runbook](RUNBOOK.md).
- All prior roots and the clean-SHA packaging-RED root remain preserved. The
  authorized clean-SHA Zero-HP run has been consumed. No additive P60-2F
  commit, push, image publication, Native rerun, or further TPU retry is
  authorized.
- Updated: 2026-08-25.
