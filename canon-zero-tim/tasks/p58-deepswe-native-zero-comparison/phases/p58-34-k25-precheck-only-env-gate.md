# P58.34 — K25 precheck-only environment admission

Status: `LOCAL CONSTRUCTION PASS / TARGET NOT RUN`

## Incident

K25 `canon-p58-ds4b-zero-hp-full-k25` ran on 128 TPU v5p devices as
rollout DP8xTP8 plus trainer DP8xTP8. It crossed P58.33 by emitting:

```text
[P58.CHECKPOINT] PASS mode=disabled cli=none resume=unsupported
```

It returned all 128 trajectories, including 115 complete rows, five solved
rows, thirteen compact-filtered rows (twelve `MODEL_TIMEOUT`, one
`MAX_CONTEXT_LIMIT`), three effective prompt groups, and 46 final nonzero
advantages. Rescore B completed for all 128 rows. The run then stopped before
the first alignment comparison, backward, or optimizer transaction because
the exact Zero-HP warning policy rejected an absent
`CANON_P38_PRECHECK_ONLY` value.

This is a policy-admission representation defect, not a numerical RED. The
shell/profile contract interprets an absent or empty value as disabled, while
the Python warning-policy identity had required the literal string `0`.

Immutable analysis-grade evidence:
`canon-zero-tim/evidence/p58_k25_precheck_only_env_gate_incident/`.

## Repair

The one parser remains authoritative and fail closed:

| Raw value | Meaning | Zero-HP/full warning admission |
|---|---|---|
| absent | production, precheck disabled | admitted |
| empty | shell-compatible disabled representation | admitted |
| `0` | explicitly disabled | admitted |
| `1` | P38 precheck-only diagnostic | rejected; diagnostic remains strict |
| any other value | malformed | rejected by the parser |

The warning identity remains otherwise exact: admitted P58 Zero arm,
`qwen3-4b-dp8-tp8-deepswe-v1-hp`, full/1,000 updates, commit enabled,
DP8xTP8, 128 trajectories, and no checked-VMA or seam diagnostic selector.
Only finite A-B and directly derived weight observations may warn. B-C,
trainer repeat, nonfinite values, shapes, gradients, replicas, transactions,
optimizer state, OOM, and evidence failures remain hard stops.

## Next target

K25 made zero optimizer commits and no trainer checkpoint exists, by design.
It cannot be resumed. The next separately approved target is a fresh K26
from the final clean remote-read source SHA and matching digest-pinned image.
Require the checkpoint-disabled marker, policy ID
`deepswe-zero-hp-ab-warning-v1`, the actual pre-alignment verdict, all sixteen
reverse groups, and the first valid TPU-resident optimizer transaction.

K25 proves only rollout/rescore reachability and the P58.33 checkpoint
boundary. It does not prove A=B=C, warning behavior on a real finite A-B
difference, backward, an optimizer update, convergence, or Zero-TIM.

## Construction evidence

- alignment-policy truth table: 15/15 PASS;
- P34 adjacency: `P34_STATIC_PASS suites=10`;
- flag registry: `declared=409 actual=409 unique=409 changed_names=1`,
  `FLAG_AUDIT_PASS`;
- Python compilation and diff hygiene: PASS;
- local K26 review render: `P58_DEEPSWE_TIM_RENDER_PASS`; warning-only is
  exactly `1`, precheck-only is absent, checkpoint-none occurs once, and save
  cadence is absent;
- complete digest-pinned image gate:
  `P58_EXACT_IMAGE_CPU_PASS ... alignment_policy=1 zero_hp_full=1
  checked_vma_diagnostic=1 ...`.

No image publication, Kubernetes mutation, TPU launch, commit, or push is
part of this local repair. A K26 target is still required.
