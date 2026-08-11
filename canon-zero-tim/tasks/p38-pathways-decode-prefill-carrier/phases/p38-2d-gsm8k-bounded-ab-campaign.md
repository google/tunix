# P38.2d: bounded GSM8K A/B campaign

## Purpose

Run the user-approved 200-update GSM8K campaign without hiding the sparse
decode-versus-prefill observation. This is an operational exception, not a
zero-TIM repair. FrozenLake remains strict and is admitted only for the
backward-no-commit diagnostic.

## Frozen contract

Only `gsm8k + full + train + commit` may set
`CANON_GSM8K_AB_REPORT_ONLY=1`. The renderer sets the switch to `0` for every
other queue entry, and both the environment preflight and runtime gate reject
an out-of-scope value.

`S_decode_vs_S_prefill` may be reported without stopping only when all of the
following hold:

- shapes and dtypes are valid;
- all masked values are finite;
- `max_abs <= 1e-4`;
- `differing_bytes / total_bytes <= 4e-3`;
- `S_prefill_vs_T_old` remains exact;
- after backward, `T_old_vs_T_current` and `r` remain exact;
- clip and TIS hit counts remain zero;
- gradients, DP reduction, optimizer commit, and replica gates remain green.

The loss is unchanged. It still consumes rollout log probabilities, and no old
log probability is recomputed. Therefore `w` and `w*r` may be non-unit only on
the reported A/B entries. Every record preserves the exact mismatch evidence
and is labeled `PASS_WITH_REPORTED_DRIFT`; the final classifier is labeled
`PASS_WITH_AB_REPORT_POLICY` and `claim_level=alignment-degraded`.

## Jobs

Render all queue manifests from one source commit, but apply only:

1. `jobset-p33-frozenlake-backward-no-commit.yaml`;
2. `jobset-p33-gsm8k-full.yaml`.

The FrozenLake job performs no optimizer commit and keeps all numerical
boundaries hard. The GSM8K job performs 200 updates under the bounded policy.
Neither result may be described as a full zero-TIM closure if any A/B drift is
reported.

The GSM8K full JobSet may restart up to three times. Checkpointing remains
disabled, so every retry starts again at update 0. This is an operational
availability tradeoff, not checkpoint/resume support. Because the simple
JobSet policy retries every nonzero exit, it can also repeat a numerical gate
failure; every attempt number must remain visible and only a complete attempt
may be classified. FrozenLake and all diagnostic entries retain
`maxRestarts: 0`.

## Verification

Before push:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace:ro" -w /workspace -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh

bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh
```

The negative controls must reject the policy on FrozenLake, no-commit, short,
and non-training configurations, as well as invalid, nonfinite, or
out-of-budget A/B observations.

## Rollback

Set `CANON_GSM8K_AB_REPORT_ONLY=0` or revert this phase commit. The original
strict gate is then restored without changing precision, loss, optimizer, W&B,
or credential configuration. Preserve all reports from a degraded campaign.
