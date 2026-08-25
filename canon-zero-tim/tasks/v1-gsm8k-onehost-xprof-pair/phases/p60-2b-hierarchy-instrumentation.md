# P60-2B — Add a profile-only Zero-HP hierarchy

- Status: passed (local); one-host target not run

## Finding

- Confirmed: the Zero-HP path starts XProf immediately before the G6 update and
  already blocks through the committed update before the window closes.
- Confirmed: the existing annotation flag is observational and default-off.
  Extending `CANON_XPROF_LABELS=1` is preferable to creating another flag.
- Hypothesis: one top-level step annotation and a bounded number of host trace
  scopes will make the existing device modules navigable without changing the
  JAX program.

## Required hierarchy

```text
train(step_num=<global step>)
└── zero_tim_update
    ├── forward_groups
    │   └── forward_group            ×16
    ├── loss_pullback                ×1
    ├── reverse_groups
    │   └── reverse_group            ×16
    │       ├── replay_forward       ×1/group
    │       ├── model_backward       ×1/group
    │       ├── report_adjoint       ×1/group
    │       ├── fixed_dp_reduce      ×1/group
    │       └── gradient_accumulate  ×1/group
    └── optimizer_commit             ×1
```

Existing `zt_tr_dp_parallel_bwd_layer_00..27`, head, norm, embed, and adjoint
module names remain the layer-level index. Do not add one host span per layer.

## Execution

1. Add one small shared helper that returns an exact no-op context when
   `CANON_XPROF_LABELS` is absent, empty, or `0`, and a validated
   `jax.profiler.TraceAnnotation` context when it is `1`. Reject any other
   value exactly as the current module-label helper does. Keep this helper in
   the profiling/observation layer so both the learner and canonical adapter
   use one parser and one naming contract.
2. At the G6 update entry, after the trace is armed and before numerical work,
   add `StepTraceAnnotation("train", step_num=<current global step>)` and one
   `zero_tim_update` parent. Keep the existing official semantic-Perfetto
   `peft_train` span unchanged.
3. In `segmented_dp_grpo_value_and_grad`, annotate forward-all, each forward
   group, loss pullback, reverse-all, each reverse group, replay-forward,
   model-backward, report adjoint, fixed reducer, and accumulator sink. In
   `_run_p28_g6_update`, annotate the optimizer transaction. Use a stable span
   name plus integer metadata such as `group_index`; do not create
   high-cardinality dynamic names.
4. Reuse `CANON_XPROF_LABELS`; do not add a new flag. Update its `FLAGS.md`
   description only if the implementation expands its documented semantics.
5. Add a hierarchy census that reads the full XPlane host plane and fails on
   missing, duplicated, or out-of-parent annotations. The census must remain
   separate from the existing all-plane backward census. It must also require
   the complete hierarchy on the same `/host:CPU` `python3` track, exactly
   eight `/device:TPU:N` planes, and a non-empty `Steps` line on each. This
   matches Native's annotation API and device-row presence, not its microstep
   cadence/cardinality or monolithic program shape. Factor interval validation
   into a pure function so synthetic positive/negative controls do not require
   constructing a protobuf fixture.
6. Extend the arm classifier so Zero-HP requires the hierarchy census only when
   this revised carrier is requested. Native remains governed by its stock
   `train` annotation and monolithic-module contract.

## Prohibited changes

- No new `jax.jit`, `shard_map`, scan, collective, reducer, gradient, optimizer,
  precision, loss, or sampling change.
- No new `block_until_ready`, host `device_get`, barrier, or synchronization.
- Do not rename existing XLA modules just to improve the screenshot.
- Do not add custom events to semantic Perfetto; P55 already demonstrated that
  custom nested semantic spans make the official timeline harder to read.
- Do not suppress or filter the 59,028 events in the raw artifact. The raw
  fragmentation is evidence; the hierarchy is an index over it.

## Verification ladder

1. `git diff --check`, syntax, and flag-registry audit.
2. Unit tests for absent/empty/`0` exact no-op, `1` positive path, and invalid
   value rejection.
3. Synthetic hierarchy-census positive control plus missing parent, duplicate
   group, orphan optimizer, wrong-count, wrong-host-track, and
   missing-device-`Steps` negative controls.
4. Existing P59 DP2xTP2 and TP4/TP8 serial-vs-parallel exact-gradient tests with
   labels off and on. A one-ULP injected gradient must still be detected.
5. Exact-image test proving the pinned JAX version accepts the chosen
   `TraceAnnotation` metadata and that existing P59/V1 gates remain green.
6. Diff review proving the instrumentation added no synchronization and did
   not touch semantic Perfetto event vocabulary.
7. A read-xprof static audit proving the carrier still uses `phase=update`,
   host tracer 1, Python tracer 0, one captured update, and an existing
   end-of-update readiness boundary. Do not add another readiness boundary.

## Exit gate

- Command: task-specific host suite, P59/V1 regression suites, flag audit, and
  the pinned exact-image gate recorded in the executor's checkpoint.
- Pass: every gate above is green; the code diff contains only observational
  annotation/census/classifier/documentation changes; no TPU launch has yet
  occurred.
- Fail: stop at the first red gate. Do not weaken counts, enable Python tracer,
  or add synchronization to make the trace look cleaner.

## Result

Implemented from fetched tip
`16db308b35c6e625d6a47c40b039ecfea317d9b3` in
`local/p60-2b-xprof-hierarchy-0825`.

- Shared exact parser/no-op helper, Native API-compatible
  `StepTraceAnnotation("train", step_num=1)`, bounded update/group/stage
  annotations, full-XPlane hierarchy/8-plane-`Steps` census, Zero-only
  classifier requirement, and local exact-image gate are present.
- Task suite: 7/7 PASS, including strong hierarchy negative controls;
  `P60_2_DOCSET_PASS files=12 phase=p60-2b`.
- Existing host suites: P59 37/37 PASS and V1 Phase4 34/34 PASS.
- Pinned exact image:
  `V1_HP_EXACT_IMAGE_PASS ...` and
  `P60_2B_EXACT_IMAGE_PASS hierarchy_api=1 labels_off_on=1 one_ulp=1 p59_v1=1 tpu_devices=0`.
- Source-token audit is unchanged for `block_until_ready` (15), `device_get`
  (21), `optimization_barrier` (0), `jax.jit` (21), and `shard_map`
  (57). Official semantic Perfetto vocabulary diff is empty.
- Latest flag registry is 371/371 PASS because the fetched tip includes P62;
  no flag was added by P60-2B.
- Historical full-XPlane negative probes confirm all eight Native and Zero
  device planes already have non-empty `Steps` rows. Native has 17 stock
  `train` microstep events (step numbers 16..32), whereas old Zero has none.
  The revised Zero carrier intentionally adds one whole-update
  `train(step_num=1)` parent; its device attribution and UI layout remain a
  P60-2C target fact, not a local claim.
- P60-2C remains `TARGET NOT RUN`. No TPU/Kubernetes action, commit, push, or
  image publication occurred.

The concern was later migrated unchanged to fetched P63-inclusive base
`cdd3987caa648e6112ee8fc184b2e3421de3a4b2` on
`local/p60-2e-microstep-latest-0825`. The hierarchy gate now also requires one
`/host:CPU` `python3` track and includes a wrong-track negative control.
