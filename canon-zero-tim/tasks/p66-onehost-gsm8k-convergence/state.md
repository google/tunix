# P66 state: one-host GSM8K convergence debug

- Status: ACTIVE; P66.3 G1 and G1.5 complete, G2 target replay not launched.
- Active phase: P66.3 — source-freeze review before the separately approved G2 target replay.
- Branch: `local/p66-gsm8k-onehost-convergence`
- Frozen source base: `1e5f7e835f4babe43a50496a5b998ea32cffcf71`
- Objective: identify and repair the first engine-VJP dependency responsible
  for the P59 TP gradient explosion before spending TPU time on optimizer or
  200-step convergence arms.
- Success condition: G1 must separate historical P59 from serial and repaired
  P59 on the same frozen full-Qwen TP4 input with strict Zero-TIM and zero
  optimizer commits; G2 must then reproduce the result on the signed P64
  target capsule before target or production admission.
- Hard stop: any `CANON_ALIGN verdict=FAIL`, input/pre-state mismatch, missing capture, non-finite metric, or failed manifest makes the arm non-admissible; it is not explained away as performance noise.
- Last verified fact: the full-Qwen DP1xTP4 discriminator and same-point oracle separate the
  regression. Serial S has 17/17 strict PASS, engine-VJP stable norm
  `6.0506024`, and zero padding-row cotangents in all 56 observations. The
  historical P59 U arm has the same strict input but grows real-row `dhidden`
  to `4.2658e19` at layer 0 and emits full engine-gradient stable norm
  `1.5402378e21`; padding-row cotangents remain zero. This is a backward
  regression in the old P59 TP composition, not a padding/RMSNorm-scale
  explanation. Checked-VMA P attempts `p66p8` through `p66p11` all preserve
  strict pre-alignment and zero optimizer commits but stop on progressively
  later structural ownership errors. P11 reaches the real layer-27 VJP and
  proves the layer primal returns hidden as `{V:(data,model)}` while its
  upstream cotangent is correctly `{V:data}`. The fixed all-gather/ring sum is
  numerically replicated but remains typed model-varying; this is the exact
  semantic hole hidden by historical `check_vma=False`. P13 is now the first
  complete checked-VMA PASS: 17/17 strict, zero commits, engine norm
  `6.05732584`, mapped norm `0.37858307`, all 310 leaves finite/nonzero, and
  all 56 padding row-layer cotangents zero. This is within `0.1112%` of the
  serial S norms and replaces the historical P59 `1.54e21` explosion. R also
  passes and is numerically identical to P across input hashes, model-before,
  engine/gradient summaries and sampled gradient hashes. Fixed gather is not
  the causal regression after VMA ownership is repaired. Final-source G1.5
  run `p66o2_20260826t0010z` passes all six same-point endpoints: rel-L2 is
  `5.71e-7` at head, exact at norm/embed, `9.49e-4` at layer 27,
  `3.33e-3` at layer 14, and `5.26e-3` at layer 0, all below the frozen
  `4e-2` cap. It is 17/17 strict, zero FAIL/commit, and its candidate evidence
  is exact to P13 under the independent observer-neutrality classifier.
- Release gap: FIXED LOCALLY. The exact historical comparator/tests were restored, and the wrapper now durably manifests KEEP, classified non-KEEP, and invalid outcomes without printing a false GREEN.
- Next action: audit the source-freeze diff and retain the default-off claim
  ceiling. G2 signed P64 DP8xTP8 replay still needs separate target-launch
  approval; do not infer it from the one-host PASS.
- P66.3 implementation status: G0 construction and the four-arm G1 campaign
  are green. Final S is ordinary, final U is the expected numerical red, and
  checked-VMA P plus gather-off R are ordinary and mutually exact. Four
  checked-VMA P attempts exposed
  the nested-map, duplicate-pcast, RPA output, and completed-fixed-sum VMA
  boundaries in order. The current default-off candidate reuses the outer TP
  map, preserves custom-call output VMA, and applies a TP pmean only to the
  already-identical completed fixed sum so its hidden output is honestly
  invariant. Host P66 4/4, P59 37/37, and VMA-on pinned-image TP4/TP8
  `manifests=2x37/37` pass. P12 then traversed all 28 checked-VMA layer
  pullbacks and stopped only at the input-embedding VJP, whose completed fixed
  vocab sum had the same model-varying typing defect. The same default-off
  invariant boundary is now applied there and P13 passed numerically. The
  single-variable gather-off R arm also passed with exact P gradient evidence.
  The pre-registered classifier returns `H1_VMA_SUPPORTED` with no contract
  reasons, same four-arm pre-alignment input, same S/P/R group hashes and
  model-before sample, and zero optimizer commits. This is a one-host causal
  result. G1.5 also passes on the final runtime tree with six endpoint receipts,
  a live negative control, and exact P13 observer-neutrality. Neither result is
  target certification.
- TPU status: P66.2 COMPLETE. P66.3 ordinary attempt
  `p66o1_20260825t1903z` strict pre-alignment PASS then carrier OOM, no
  gradient and zero commits. Early G1 harness failures are retained, repaired
  at their original gate, and superseded by final S/U/P/R evidence. G1 is
  COMPLETE with `H1_VMA_SUPPORTED`; G1.5 is COMPLETE with same-point oracle
  and observer-neutrality PASS. G2 P64 target replay is NOT RUN and needs
  separate launch approval.
- Commit/push: forbidden without a new explicit user approval.
- Upstream reconciliation: current fetched remote tip is `9f91d930` on
  2026-08-25; the newest P64 target evidence is commit `1406cc2d`. The three
  later commits are M15 Attempt-5 evidence/handoff updates, not a newer P64
  receipt. Direct pull is intentionally deferred because upstream and the
  current dirty diagnostic tree both modify adapter/learner/FLAGS runtime
  files. P64 was inspected read-only from the fetched commit; no evidence is
  missing from this evaluation.

## Decision after P66.2

- P59 update-level KEEP: continue to the next dependency closure and only then admit Native-200 vs Zero-full-200.
- P59 gradient KEEP but update-level REJECT: exclude P59 from the causal `Z-min` diagnostic arm, retain its explicit production claim ceiling, and test the next viable backward carrier rather than serial-training 200 steps.
- Carrier/input/alignment failure: repair at the same gate and rerun with fresh labels.
