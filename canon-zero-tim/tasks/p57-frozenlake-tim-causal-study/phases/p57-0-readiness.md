# P57.0 — Readiness and treatment-contract admission

## Purpose

Make the experiment machinery technically valid before stock-only workload
discovery. This phase does not measure convergence and must not launch the
main campaign.

## Required changes

1. Register a P57.1 `stock-fast` calibration profile derived from P45 only for
   topology, model overlay, vLLM capacity, and resident placement.
2. Mechanically disable the complete inherited zero-TIM numerical bundle:
   12 presence-sensitive switches absent, 25 boolean/admission gates zero, and
   no canonical excess-precision XLA pin.
3. Add independent gates at render, resolved-container, training-entrypoint,
   receipt, and offline-classifier layers. Fixed lm-head off alone is rejected.
4. Preserve the registered Qwen3-8B K4096/TP8 fixed-lm-head geometry as a later
   zero-arm prerequisite, but do not use or inspect it during discovery.
5. Add a separate checkpoint evaluator. It must not share the training engine's
   prefix-cache state and must not re-enable P45 in-training evaluation.
6. Keep checkpoint cadence at 10 updates with LatestN(1) for bounded readiness
   runs, unless a later phase explicitly changes the retention contract.
7. Defer the complete paired training treatment contract to P57.2. Calibration
   readiness must not be misread as paired-campaign readiness.

## Local gates

- Registry unit tests cover K4096/TP8 positive dispatch, exact logical-vocab
  slicing, forward/VJP behavior, and wrong-geometry negatives.
- Renderer/profile tests prove stock-fast intent, resolve the inherited profile,
  and reject any canonical switch or admission leak.
- Calibration classifier rejects a missing or altered 37-switch zero-TIM-off
  attestation, malformed coverage, state mutation, and context-cap violations.
- Checkpoint evaluator can load a bounded P45-format checkpoint and evaluate an
  immutable map set without touching training state. Step 0 uses immutable
  base weights in checkpoint `new` mode; positive boundaries use exact GCS
  `resume`. Both explicitly sync actor weights into the rollout engine.
- `git diff --check`, syntax/compile checks, and exact-image gates pass.

## Target gate timing

P57.1 does not need a zero-arm hardware launch. The full stock-training and
zero-TIM treatment contracts, including Qwen3-8B/TP8 zero-head target checks,
are deferred to P57.2 after stock discovery freezes the workload.

## Exit gate

- P57.1 stock-fast intent is fail-closed at manifest and resolved-env layers.
- The JSON receipt/classifier reproduce the complete zero-TIM-off attestation.
- The isolated evaluator and materialized-map provenance path pass local and
  pinned-image lifecycle gates.
- No TPU target is implied by this local readiness exit.

## Claim boundary

Passing P57.0 certifies stock-fast calibration and measurement machinery only.
It says nothing about paired-training readiness, learning quality, or TIM's
functional importance.
