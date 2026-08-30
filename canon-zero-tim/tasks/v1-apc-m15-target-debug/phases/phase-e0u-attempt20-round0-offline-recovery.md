# Phase E0u — Attempt-20 treatment round-0 offline recovery

## Purpose

Attempt 20 executed on DP8×TP8 and preserved three exact APC-off rounds, but
the APC-on arm released resources after its round-0 classifier-input checkpoint
and before classification/`ROUND_COMPLETE`. This phase asks one bounded
question without another target launch:

```text
Can the already uploaded treatment round-0 classifier input be retrieved,
self-verified, and classified by its archived source-bound classifier?
```

It does not ask whether the treatment is stable for three rounds and cannot
return `TARGET PASS` or authorize a numerical repair.

## Frozen facts

- Target source:
  `97e813de84f6c8b3e2ba911fc96ff8397b199603`.
- Committed compact return:
  `evidence/v1_apc_m15_attempt20_e0_kv3_salvage_return_20260830/`.
- Its `SHA256SUMS` file has SHA256
  `986491ae7dd08a5643c832b4e7c1218000eaca652d257e3055e76be2129a32fc`.
- APC-off rounds 0/1/2 are independently exact at A−B and B−C, but root
  terminal state is absent.
- APC-on has zero completed rounds. Round 0 has a returned
  `classifier_input_receipt` presence signal but no returned `ROUND_INPUT`,
  classification, or completion. Rounds 1/2 have no checkpoint receipt.
- The recoverable classifier archive has no root runtime marker proving B
  full reset or all cached-token counts zero. B-C counters may be returned,
  but these runtime receipts must remain explicitly unavailable.
- The last admitted numerical tensor interval remains D3e Layer 0
  `k_post_rope -> rpa_output`, shape `[2048,1,15,8]`. Attempt 20 has not
  narrowed it.

## Single executor command

Run only on the bucket-capable machine, from a clean `local/*` worktree at the
exact published SHA containing this phase and script. Reuse the immutable
Attempt-20 render directory only as a signed locator; do not apply its YAMLs.

```bash
OUT=<preserved-original-Attempt20-render-dir>
RETURN=/mnt/disks/tunix-data/m15-attempt20-on-r0-recovery-k02
test -d "$OUT"
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt20_on_round0_offline_recovery.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data
```

`$OUT` must be the original verified Attempt-20 render directory. The wrapper
derives the registered treatment evidence root from that render and never
prints it. It performs read-only GCS downloads and local CPU classification;
it performs no GCS write, Kubernetes operation, or TPU work. Real GCS access
requires a separate explicit user approval.

## Fail-closed results

| Terminal status | What exists | Required answer |
|---|---|---|
| `ROUND0_LIVE_KV_FINGERPRINT_DIFFERS` | one verified red treatment round classified as live-KV fingerprint different | report the one-round branch and `NO 3/3`; do not patch |
| `ROUND0_LIVE_KV_FINGERPRINT_EQUAL` | one verified red treatment round classified as live-KV fingerprint equal | report the one-round branch and `NO 3/3`; do not patch |
| `ROUND0_TARGET_NON_REPRODUCTION` | one verified exact treatment round | report round-0 non-reproduction, not bug fixed |
| `CLASSIFIER_INPUT_UNAVAILABLE` | one or more required remote objects were not retrieved | report `classification=NONE`; do not infer absence, equality, or difference |
| `INVALID_OR_CLASSIFIER_FAILED` | receipt/archive/source/classifier/binding failed validation | report `classification=NONE`; preserve scratch and raw log |
| `ORIGINAL_RENDER_UNAVAILABLE` | the preserved signed Attempt-20 locator is absent | report `classification=NONE`; do not re-render or guess a root |

Every result fixes these fields:

```text
three_round_verdict=false
terminal_pair_complete=false
target_rerun=false
numerical_repair_authorized=false
remote_mutation=false
B_full_reset_runtime_receipt_available=false
all_num_cached_tokens_zero_runtime_receipt_available=false
```

No stdout, chat response, or Git return may contain the registered GCS root,
archive contents, token/capsule/replay rows, raw NPZ payloads, credentials, or
remote configuration.

## Returned files

On successful classification, return only:

```text
ATTEMPT20_ON_R0_RECOVERY.json
on.round-000000.kv-observer-classification.json
on.round-000000.classifier-input-receipt.json
on.round-000000.classifier-input-sha256sums
SHA256SUMS
```

On an unavailable/invalid input, return only the fail-closed recovery JSON and
its `SHA256SUMS`. In either case return the sanitized terminal marker, local
raw-log path/SHA, output manifest SHA, and no large payload.

## Exit gate and next decision

Phase E0u exits when the command returns one self-hashed compact result or one
self-hashed `classification=NONE` result. A recovered round-0 mechanism branch
may guide the next discriminator, but it does not satisfy the E0s three-round
stability contract. Full certification still requires a separately approved,
fresh matched APC-off/APC-on DP8×TP8 pair with three complete rounds per arm.

## Implementation validation

- Focused recovery tests: 6/6 PASS, including refusal to reconstruct a
  missing original Attempt-20 render.
- Canonical host aggregate:
  `M15_E0U_HOST_PASS task_discovery=199 return=1 round0_recovery=6 v1_cpu=91
  p3_prefix_cache=31 persistence=1 flags=409 manifest=dae6dfa8 syntax=1
  diff_check=1 exact_image=0 target_rerun=0 gcs=0 kubernetes=0 tpu=0`.
- Raw host log: `/tmp/m15-e0u-host-gate-20260830-r4.log`, SHA256
  `738d8df5a7ca9adef35735375e319c242f225b1404e14ef818fb72ef5ba4c4bf`.
- Official pinned exact-image: NOT RUN.
- Real GCS recovery: NOT RUN.
- Target rerun: NOT RUN.
