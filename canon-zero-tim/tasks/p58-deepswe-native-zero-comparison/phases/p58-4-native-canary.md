# P58.4N — Native 128-chip three-update canary

Status: active.

## Purpose

Prove that the untreated native DeepSWE-derived Qwen3-4B-Instruct path can
perform real rollout and training on the frozen P58 recipe before any zero arm
is launched. This is a native integration/training gate, not a numerical
treatment comparison and not zero-TIM evidence.

## Immutable contract

- source: exact post-push readback of `yuxzhang/canon-zero-tim`;
- image: registry digest, never a mutable tag;
- model/data: Qwen3-4B-Instruct-2507 and the frozen 1,012-task clean list;
- topology: one 128-chip `4x4x8` slice split into rollout DP8 x TP8 and trainer
  DP8 x TP8;
- recipe: B8 x G16, response 16,384, 50 turns, RLOO, fixed-context
  `sequence-mean-token-scale`, TPU-resident optimizer, optional interventions
  off, prefix cache off;
- stage: `three-update`, with exactly three optimizer commits;
- arm: `native` only. Rendering or applying `zero` is outside this phase.

## Exit gate

The native classifier must report `PASS` from complete, digest-verified
artifacts. It must prove exactly three commits, finite nonzero A-B treatment
dose, exact B-C, finite training values, device-resident optimizer state,
complete 128-row trajectory batches, journal continuity, sandbox cleanup, and
checkpoint/transaction integrity.

An all-filtered batch may add a trajectory batch without adding an optimizer
commit. It must have a zero-commit receipt and unchanged state. Interrupted
infrastructure, missing evidence, exact native A-B (`NO_TREATMENT`), or any
B-C drift is `INCONCLUSIVE`/failure under the classifier, never PASS.

## After the gate

Preserve and package the immutable run before proposing another launch. A
native canary PASS does not authorize a 1,000-update run and does not reactivate
zero; either decision requires a new user instruction.
