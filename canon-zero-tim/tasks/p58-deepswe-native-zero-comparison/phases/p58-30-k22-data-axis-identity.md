# P58.30 — K22 grouped-trainer data-axis identity

Status: `COMPLETED FOR REDUCER-AXIS SCOPE / K23 CROSSED`

## K22 evidence boundary

K22 is not another rollout, TiTO, fixed-head, alignment, or scan-placement
failure. The committed raw tail shows the P59 reverse path reaching layer 0
and then stopping at the reducer safety boundary:

```text
FunctionalMappingError: P59 report and grouped trainer data axes differ
```

The incident report records a 128-device disaggregated DP8xTP8 rollout role
plus DP8xTP8 trainer role, 128 trajectories, 4 solved tasks, exact A=B=C over
393,135 action tokens, and a complete layer-35-to-layer-0 pullback. Only the
final 100-line error tail is committed, so the pre-tail receipts must remain
analysis-grade rather than being treated as independently reproducible.
Preserve the immutable package at
`canon-zero-tim/evidence/p58_k22_data_axis_mismatch_incident/`.

## Root cause

The grouped reverse previously treated `self._dp_axis` as the trainer data
axis when `CANON_P34_DEEPSWE=1`. That member is an engine-facing alias and was
`data`. The rank-parallel report adjoint independently and correctly inspected
the trainer state's `NamedSharding`, whose mesh axes were `("dp", "tp")`, and
therefore returned `dp`. The reducer's fail-closed consistency check compared
`data` with `dp` and raised after the pullback.

This was a role-identity bug. It did not indicate nonfinite gradients, a
different reduction, or a DP8 topology mismatch.

## Repair and regression contract

Grouped reverse now resolves its data axis from the trainer state for every
workload. The local hardening puts this behind
`_p32_grouped_trainer_dp_axis` and intentionally never reads the adapter's
serving alias.

The forced-four-device regression matrix is:

| Trainer state mesh | Stale adapter alias | Required result |
|---|---|---|
| `("dp", "tp")` | `data` | `dp` |
| `("data", "model")` | `data` | `data` |
| `("fsdp", "tp")` | `data` | hard error |

These tests guard both DeepSWE and neighboring ordinary grouped workloads.
They do not infer identity from mesh shape, and they do not admit FSDP or a
third naming convention.

No flag, recipe value, model, clean-data selector, sampling distribution,
loss aggregation, precision, optimizer placement, DP/TP split, timeout,
TiTO, or Zero-HP path changes.

## Construction gates

Before publication require:

```text
forced-four-device grouped trainer-axis tests: 3/3 PASS
P34_STATIC_PASS suites=10
flag audit: declared=409 actual=409 unique=409 changed_names=0
P58_EXACT_IMAGE_CPU_PASS ... grouped_trainer_axis=3
git diff --check: PASS for authored source/docs
```

On operator parent `110146c6f48e997fd426226333d2f39cb3486840` plus the
local hardening diff, the focused pinned-image tests pass 3/3, P34 static
passes ten suites, the flag audit passes 409/409 with `changed_names=0`, and
the complete digest-pinned P58 image gate exits zero with
`grouped_trainer_axis=3` and `P58_EXACT_IMAGE_CPU_PASS`.

## K23 closure and claim ceiling

K23 used the clean published successor and crossed this phase's reducer-axis
boundary. It emitted `gradient_reducer_ready dp_axis=dp dp_size=8` and
completed group 1/16 with exact replicas plus a finite nonzero gradient. It
then failed before accumulator mutation at
`segmented update accumulation changed: 8 != 16`; that new issue belongs to
P58.31, not this phase.

K23 preserved the DeepSWE/TiTO, clean-data, Rescore-B, and strict A=B=C
contracts. It did not:

1. complete every one of the 16 grouped forward/reverse/reduction
   transactions with finite nonzero gradients;
2. preserve exact post-reduction replicas and all optimizer safety checks;
3. emit exactly the intended first optimizer commit and durable checkpoint;
4. continue while later updates remain finite and the signed 1,000-step
   campaign contracts hold.

The reducer-axis repair is therefore complete for its scope, while the full
training result remains `INCONCLUSIVE`. See P58.31 and the immutable K23
incident package for the active accumulator-geometry repair.
