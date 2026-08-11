# P38.2e: schedule-aware GSM8K optimizer transaction

## Purpose

Repair the P38d5 G6 false rejection without weakening the optimizer
transaction. The production warmup schedule applies an effective learning
rate of exactly zero at update 0. Adam moment state must advance, but model
parameters must remain unchanged.

## Target fact

`debug_logs/p38_p38d5_gsm8k_full.raw.log` reached all 16 gradient
microbatches, the fixed-order DP reducer, and exactly one optimizer commit.
Every A/B/C boundary was exact. The gate then rejected the transaction because
the sampled optimizer state changed while the sampled model state did not.
The recipe constructs `warmup_cosine_decay_schedule(init_value=0.0, ...)`, so
model immutability at update 0 is required behavior, not evidence of a failed
commit.

## Implementation contract

- Construct one learning-rate schedule object and pass that exact object to
  both AdamW and the transaction observer.
- Preserve the public `commit_precomputed_gradients()` return value.
- Emit device-side scalar reductions for finite gradients and post-rounding
  parameter changes; do not transfer parameter arrays to the host.
- At effective LR 0, require zero changed parameter elements and a changed
  optimizer state when a learning signal exists.
- At positive LR, report a post-rounding zero update as
  `positive_lr_quantized_zero`; do not invent a model mutation.
- Keep accumulator reset, reference immutability, DP replica equality, and
  exactly-one-commit gates hard.

## Local gates

- Zero-LR real commit: nonzero gradient, optimizer moments change, model
  elements unchanged, transaction PASS.
- Constant positive LR real commit: changed parameter count and max absolute
  delta are nonzero.
- Classifier negative: any parameter change at effective LR 0 is rejected.
- Complete P33 CPU and exact-image gates.

## Target exit gate

A fresh source-pinned GSM8K full Attempt 0 must pass update 0 with
`effective_learning_rate=0.0`, `parameter_changed_elements=0`, changed Adam
state, all 16 active microbatches, and every hard numerical boundary green.
This target gate is not run locally.

## Local hardware limitation

The existing GSM8K L3 recipe requires one prompt with two generations, while
the legacy non-production G6 update path requires eight trajectories. The
local implementation does not relax either contract. Consequently, real
Qwen3-1.7B compile time and peak HBM for the scalar evidence remain NOT RUN;
the current local evidence consists of real Optax transaction tests plus the
complete frozen-image CPU and exact-image gates.

## Rollback

Revert the schedule registration, commit evidence sidecar, runtime
classification, and associated tests together. The optimizer schedule and
optimizer math are unchanged by this phase.
