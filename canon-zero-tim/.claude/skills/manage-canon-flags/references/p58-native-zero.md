# P58 DeepSWE Native/Zero flag contract

Read this reference only for
`cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env`. The profile and
`tunix/rl/deepswe_contract.py` remain the executable sources of truth.

## Treatment split

| Contract | Native | Zero |
|---|---:|---:|
| `CANON_P58_TIM_ARM` | `native` | `zero` |
| `CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER` | `1` | `0` |
| `CANON_PROMPT_PROCESSED_LOGPROBS` | `0` | `1` through canonical profile |
| `CANON_ENGINE_MODULE_C` | `0` | `1` through canonical profile |
| `CANON_RPA_VJP2` / `CANON_VJP2_MAX_SEQS` | `0 / 0` | canonical admitted values |
| `CANON_FIXED_AR`, `CANON_FIXED_AR_EMBED`, `CANON_LOGPROB_M` | absent | canonical admitted values |
| Pallas/segmented numerical bundle | disabled or absent as declared by profile | complete canonical bundle |
| `CANON_DEEPSWE_ALIGNMENT_WARN_ONLY` | `1` for finite A-B/B-C serving-path observation; trainer repeat strict | `0`, strict exactness |
| `CANON_P32_DP_REDUCTION_ADMITTED` | `0` | `1` |

Do not delete Native's explicit zeros or absences. Do not turn on canonical
processed logprobs to satisfy the Native `S_prefill` API.

Native alignment warning admission is independently signed. Require
`CANON_P58_TIM_ADMITTED=1`, no competing P39/P43/P44 workload mode, and an
exact `CANON_P34_RUN_STAGE/CANON_P58_EXPECTED_UPDATES` pair of
`three-update/3` or `full/1000`. Do not place P58 full training in a generic
short-debug-stage branch. Zero keeps warning-only at zero and remains strict.

For signed P58 Native, `warning_boundaries` must contain exactly
`S_decode_vs_S_prefill` and `S_prefill_vs_T_old`. These compare untreated
serving decode/prefill and serving/trainer programs, so any shape-valid finite
difference is treatment evidence rather than a training veto. The trainer
repeat boundary `T_old_vs_T_current`, derived ratio `r`, nonfinite values,
invalid shapes, weights, replicas, transactions, and optimizer state remain
fail-closed. Zero admits no warning boundary and requires all comparisons
exact. Do not copy P38's diagnostic-only B-C requirement into P58 Native.

## Why the Native observer exists

P58f04 completed 128 trajectories and exact live-weight attestation, then
failed before trainer forward because processed B accepted only the canonical
engine flag. Returning stock raw prompt logprobs would be semantically wrong:
the pinned helper rolls target IDs across a DP-packed buffer and can cross
request or padding boundaries.

The Native observer overlay is limited to the post-rollout B request. It uses
decode-equivalent temperature/top-k/top-p transforms and absolute
request-history targets. It does not participate in rollout token selection,
trainer forward, loss, backward, optimizer, or commit counting. Its flag is
therefore observational and separate from the canonical numerical treatment.

## Required Native evidence

Require all of the following:

```text
[P58.NATIVE] STOCK_PREFLIGHT_PASS ... overlay=absent
[P58.STOCK_OBSERVER] OVERLAY_PASS ... canonical_bundle=off treatment=observer-only
[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS ... targets=absolute-request-history treatment=observer-only
[P34.WEIGHTS] EXACT ...
[P58.NATIVE] RUNTIME_PATH_PASS canonical_markers=0 canonical_overlay=skipped stock_observer=observer-only
```

The processed observer marker must occur exactly once. Any canonical engine
marker in Native, a missing observer marker, or more than one observer marker is
a hard treatment failure.

Zero must not install or emit the P58 stock-observer markers. It retains the
canonical overlay and strict A=B=C postflight.

## Regression set

- `tests/p58_deepswe_native_zero/test_profile.py`
- `tests/p58_deepswe_native_zero/test_environment_contract.py`
- `tests/p58_deepswe_native_zero/test_stock_prompt_observer.py`
- `tests/p58_deepswe_native_zero/probe_stock_prompt_observer.py`
- `tests/rl/rollout/vllm_rollout_canonical_test.py`
- `tests/p58_deepswe_native_zero/run_exact_image.sh`

P57 stock-fast is the neighboring negative control: the P58 observer flag must
not contaminate it.
