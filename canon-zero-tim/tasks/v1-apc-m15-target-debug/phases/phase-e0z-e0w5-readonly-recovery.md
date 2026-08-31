# Phase E0z — e0w5 read-only evidence recovery

## Purpose

Recover and mechanically audit the already executed e0w5 DP8xTP8 pair before
considering another target launch or a numerical repair. This phase changes
evidence handling only. It does not change A/B/C, APC reads, attention, RoPE,
RPA, KV values, model arithmetic, training, or production defaults.

## Immutable binding

```text
target source: 2f61f8fc7cf073964a9adbd30e78de872426a4d2
run label: e0w5
topology: DP8xTP8, 64 TPU per arm
arms: APC off / APC on; APC is the only treatment difference
program: exact TiTO, layer observer, three diagnostic rounds
B: reset_prefix_cache=True and all cached-token counts zero
backward: 0
optimizer commits: 0
```

The Git incident subset reports off rounds 0/1 exact and on round 0 red at
A-B 615 bytes / 262 elements with B-C zero. Those statements are reported
facts, not an admitted three-round verdict.

## Entry gate

- clean `local/*` worktree at the full published analysis SHA;
- target source is an ancestor of the analysis source;
- original e0w5 render directory is present and its complete manifest verifies;
- render source, label, exact-TiTO/layer/three-round contract, JobSet identities,
  zero-update contract, and paired APC-only difference all verify;
- committed e0w5 incident manifest verifies;
- GCS credentials permit reads; no write or Kubernetes permission is needed.

If any entry condition fails, return a bounded refusal. Do not reconstruct the
render or silently switch run labels.

## Operation

Run `scripts/run_m15_e0w5_gcs_return.sh` once, without a pipe. It calls the
official small multiround return, inventories all three rounds, validates
official classifier schemas and names, classifier-input checkpoints, ordered
stage receipts, remote bundle presence, round completion and root markers.
Token-bearing tars remain remote. Failure scratch and a local raw log are
preserved.

This operation is read-only: `gcs_read=1`, `gcs_write=0`, `kubernetes=0`,
`tpu=0`. It is not a target rerun.

## Exit classifications

- `COMPLETE`: all six rounds seal and both roots are complete. Still requires
  review of runtime B/TiTO receipts and localization fields; no automatic
  target PASS or repair authority.
- `ROUNDS_RECOVERED_ROOT_INCOMPLETE`: six numerical rounds exist but terminal
  root evidence is incomplete; analysis-grade only.
- `PARTIAL_ROUNDS_RECOVERED`: at least one independently sealed round exists,
  but no paired three-round verdict.
- `ROUND_STAGE_FAILURE_IDENTIFIED` or `ROUND_STAGE_PROGRESS_ONLY`: the durable
  boundary is an evidence pipeline stage, not a numerical tensor boundary.
- `NO_DURABLE_ROUND`: inspect preserved scratch/raw log; do not infer equality.
- official audit failure: preserve the downloaded small evidence and report
  the rejection exactly.

Exit code 3 means `INCONCLUSIVE` evidence, including a valid partial return.

## Claim ceiling

`FIRST_RED_LOCALIZED` remains false unless an admitted classifier supplies and
review confirms the last exact tensor, first red tensor, shape ledger,
request/call/token/cache/page coordinate, and source `file:line`. Candidate
sets remain candidate sets. Missing B full-reset, cached-token-zero, or exact
TiTO runtime receipts must be returned as `NONE`.

No e0w6 launch and no numerical edit is part of this phase. The next operation
must be chosen only after reviewing the bounded return with the user.
