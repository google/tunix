# P58.2 — Paired runtime and artifact contract

Status: completed locally on 2026-08-21.

## Delivered

- DP8 x TP8 per-role P58 profile and `4x4x8` renderer for both arms and both
  stages;
- complete native-bundle absence checks and zero-bundle presence checks;
- arm-aware alignment policy and postflight classifier;
- exact 1,012-task clean-data digest and B8 x G16 command contract;
- full gzip trajectory journal, batch solve/signal/compact-filter metrics,
  W&B forwarding, and atomic manifest/digest receipts;
- independent batch and optimizer counters with fail-closed resume; and
- native stock and zero canonical optimizer transaction receipts, including
  all-filtered no-commit in both paths.

## Exit evidence

The renderer tests prove the shared recipe signatures are equal and the
treatment signatures differ only in registered numerical fields. Environment
tests source the actual shell profiles after renderer variables for
native/zero x canary/full and pass `deepswe_contract.validate_environment`.
Classifier negatives reject native with no dose and zero with any drift.

Pinned-image terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```

This closes implementation wiring only. It does not activate P58.4 or prove a
real rollout, Pathways topology, HBM capacity, native mismatch dose, or zero
exactness on TPU.
