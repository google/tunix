# P58.28 — K11 prompt-only row admission in grouped reverse

Status: `LOCAL REPAIR / COMPLETE PINNED-IMAGE PASS / TARGET NOT RERUN`

## Incident

K11 ran source `2f61f8fc7cf073964a9adbd30e78de872426a4d2` on the
real 128-device DP8xTP8 target. It completed all 128 multi-turn trajectories,
produced 427,594 action tokens, finished Rescore-B in 109.5 seconds, and
passed strict Step-0 pre-alignment with exact A=B=C. The first segmented
reverse then stopped in `_p32_group_spec` because three DP ranks in one group
had no completion-valid tokens:

```text
n=[4874,1737,4415,1819,3436,3538,1811,5103]
prompt=[1808,1737,1876,1819,1863,1800,1811,1740]
completion=[3066,0,2539,0,1573,1738,0,3363]
```

The immutable incident is
`canon-zero-tim/evidence/p58_k11_deepswe_empty_completion_incident/`; raw
error SHA-256 is
`21a5bbda5c11e6372e393433835047969c372d9d828b27da475122b6d4d15b0c`.

## Root cause

The shared P32 group builder inherited a single-turn assumption that every DP
rank must contain at least one completion-valid token. DeepSWE intentionally
preserves some turn-zero environment failures/timeouts as prompt-only rows.
For such a row, `completion_valid_mask` is empty and the earlier subset gate
proves that `completion_mask` (the policy action mask) is also empty.

The row is mathematically neutral:

- `sequence-mean-token-scale` multiplies by the action mask and excludes the
  row from both numerator and effective-row denominator;
- grouped forward returns zero logprob and entropy values wherever
  `completion_valid_mask` is false;
- grouped reverse masks both logprob and entropy cotangents by that same
  validity mask before scattering them into the packed sequence;
- the fixed DP reducer already admits equal or zero rank gradients.

The error was therefore an overly strict construction assertion, not a
rollout, reward, alignment, or optimizer failure.

## Repair

`_p32_group_spec` now has a keyword-only
`allow_empty_completion=False`. The default preserves the fail-closed
single-turn/GSM8K contract. Only
`segmented_dp_grpo_value_and_grad` with the already validated
`CANON_P34_DEEPSWE=1` workload identity passes `True`.

Prompt validity and at least two total real tokens remain mandatory on every
rank. The repair does not insert a fake completion token, drop/resample a
trajectory, change a reward or advantage, change loss normalization, change
DP reduction, or skip the fixed optimizer transaction. When prompt-only rows
are present, the learner emits:

```text
[P34.EMPTY_COMPLETION] admitted_rows=<n> coordinates=<group/rank pairs> semantics=zero-loss-zero-gradient
```

## K11 shape ledger

| Quantity | Value |
|---|---:|
| Global trajectories | 128 |
| Trainer mesh | DP8xTP8 |
| Rank-major reverse groups | 16 |
| Rows in the failing group | 8 |
| Prompt-only rows in that group | 3 |
| Prompt / completion compiled widths | 4096 / 16384 |
| Local canonical M | 256 |
| Largest real sequence in the failing group | 5103 |
| Fixed chunks for the failing group | 20 |
| Global engine M per call | 2048 |

No compiled width, scheduler capacity, or topology changes.

## Gates

- P58 loss contract proves a fully masked row is finite zero and excluded
  from the effective-row denominator.
- An AST gate proves the new argument defaults to false and only the P34
  branch supplies the DeepSWE opt-in.
- A forced-16-CPU-device grouped forward/reverse test proves prompt-only rows
  return zero outputs and that arbitrary cotangents on those invalid cells do
  not alter any engine or cache gradient.
- A second regression replays the exact K11 DP8 length vectors and requires
  20 M256 chunks.
- Focused tests pass in pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- P34 static passes ten suites and the deterministic flag audit passes
  409/409 with `changed_names=0`.
- The complete pinned P58 image gate exits zero with
  `P58_EXACT_IMAGE_CPU_PASS ... p34_empty_completion=2 regressions=1`.

## Claim ceiling and next target

This repair admits the exact K11 prompt-only shape into segmented reverse. It
does not prove the real DP8xTP8 backward, first optimizer commit, checkpoint,
or 1,000-update campaign. No target launch is authorized by this phase.

After complete local gates, explicit source publication approval, matching
image publication, and separate launch approval, a fresh Attempt-0 must start
from the final clean remote readback SHA. It must preserve the K11 TiTO,
trajectory, A=B=C and empty-row receipt, cross segmented forward/reverse, and
produce exactly one valid first optimizer transaction before this boundary is
considered target-proven.
