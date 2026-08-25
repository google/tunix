# P60-2E — Expose truthful accumulator microsteps and optimizer update

- Status: local/exact-image pass; clean-SHA core target gates pass; evidence
  packaging red

## Finding

- Confirmed: both Native and Zero-HP device `Steps` rows use numeric derived
  event names; semantic `train` annotations live on `/host:CPU` `python3`.
- Confirmed: the full Native XPlane has 17 `train` events with step numbers
  16..32. Steps 16..31 cover 16 real gradient-accumulation iterations; 32 is
  the terminal iterator probe. Native's last real monolithic train step owns
  its optimizer update.
- Confirmed: Zero-HP executes all forward groups before its reverse/reduce/
  accumulate loop and commits once afterward. It would be misleading to draw
  16 Native-shaped contiguous `train` parents around this different schedule.
- Boundary: `train(step_num=1)` matches Native's annotation API only. It does
  not assert matching cadence, cardinality, or monolithic program shape;
  numeric device `Steps` rows are not semantic train-step labels.
- Decision: retain one whole-update `train(step_num=1)`, expose each real
  accumulator sink as `micro_step=0..15`, mark only 15 with
  `is_last_accumulate=1`, and mark the separate optimizer with
  `update_step=1`.

## Execution

1. Add integer metadata only to existing profile-gated annotations. Do not
   add a JIT, computation, synchronization, dynamic event name, or semantic
   Perfetto event.
2. Extend the pure/full-XPlane hierarchy census to require contiguous
   microsteps, the unique last accumulator, and the optimizer update number.
   Extend the device-module census to require exactly eight TPU planes and,
   on each plane, `jit__precomputed_gradient_scaled_step` exactly 16 times
   plus `jit__precomputed_gradient_commit` exactly once.
3. Add source-wiring and API positives; wrong-microstep, wrong-last,
   wrong-update, wrong-track, missing-commit, short-scaled-step, and missing-
   plane negative controls.
4. Prove the previous dev2 XPlane fails only the new metadata boundary, then
   run CPU, pinned exact-image, flag, branch, diff, no-sync, vocabulary, and
   secret gates.

## Exit gate

- Command: `bash tests/v1_gsm8k_xprof_pair/run_cpu.sh` and
  `bash tests/v1_gsm8k_xprof_pair/run_exact_image.sh`, followed by the flag
  audit, branch preflight, `git diff --check`, and source-token audits.
- Local pass: the exact-image receipt itself contains all 16 contiguous
  accumulator spans on one host track, one last accumulator, and one matching
  optimizer update. The module census requires the complete optimizer tail on
  8/8 planes; labels-off/on numerical gates remain exact.
- Target pass: a separately authorized fresh Zero-HP XPlane satisfies
  `micro_steps=0..15 last_accumulate=15 optimizer_update=1`, and every TPU
  plane has `scaled_step=16` plus `commit=1` with decode absent.
- Fail: do not weaken the metadata gate or reshape the numerical schedule to
  imitate Native. Preserve the old artifact and report target debt.

## Result

Local and pinned exact-image gates pass. The task suite is 10/10 and the P60-2
document set is 13 files. The exact-image API marker is:

```text
P60_XPROF_ANNOTATION_API_PASS step=train step_num=1 micro_steps=0..15 last_accumulate=15 optimizer_update=1 metadata=integer host_plane=/host:CPU host_line=python3 xplane=1 trace=1
```

Labels-off/on P59 DP2xTP2 and TP4/TP8 regressions, the one-ULP negative, and
the P63-inclusive V1 ladder pass on migrated base
`cdd3987caa648e6112ee8fc184b2e3421de3a4b2`. Flag audit is 372/372; branch
preflight, diff, syntax, and changed-patch secret scan pass. Source tokens are
unchanged versus HEAD: `block_until_ready=15`, `device_get=21`,
`optimization_barrier=0`, `jax.jit=21`, and `shard_map=57`. Semantic Perfetto
vocabulary is unchanged. The pure validator includes a wrong-host-track
negative, and the exact-image probe requires one `/host:CPU` `python3` track.

Running the strengthened census on immutable dev2 preserves all previous
counts, one common `/host:CPU` `python3` hierarchy track, and 8/8 device Steps
rows, then returns exactly 33 missing-metadata reasons: two fields on each of
16 accumulator spans plus optimizer `update_step`. This is the intended
fail-closed proof. Independently, the strengthened device-module census is
GREEN on that immutable XPlane: all 8/8 planes contain the five backward
families, `jit__precomputed_gradient_scaled_step` exactly 16 times, and
`jit__precomputed_gradient_commit` exactly once, with decode absent. The two
new tail negative controls fail on a missing commit and on only 15 scaled
steps. No TPU, Kubernetes, commit, push, or image publication occurred during
that local checkpoint.

The later clean implementation commit
`da535c1d5cee7573671fa40809547a6972bec072` received one authorized fresh
Zero-HP run. Its target XPlane satisfies the exact metadata, host-track,
8/8-device, backward, scaled-step×16, commit×1, decode-absent, 3/3-update, and
51/51-alignment gates; classification is PASS with no reasons. However, the
old runner generated `SHA256SUMS` before appending the terminal GREEN marker
to `driver.log`, so independent manifest verification fails. P60-2E therefore
has `CORE TARGET GATES PASS / EVIDENCE PACKAGING RED`, not TARGET PASS.
P60-2F owns the additive ledger-order repair and fresh-receipt requirement.
