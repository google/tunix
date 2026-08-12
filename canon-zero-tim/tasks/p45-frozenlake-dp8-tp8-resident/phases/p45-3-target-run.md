# P45.3 — 64-chip target run

- Status: pending

## Objective

Run exactly one P45 full or full-eval manifest on a 64-chip v5p slice and prove
that Qwen3-8B can retain Adam state on TPU across real updates without OOM or
placement drift.

## Entry point

Follow `../HANDOFF.md`. Dry-run both generated manifests, apply exactly one,
and copy the first update JSON while the pod is still live.

## Stop conditions

- TPU OOM or missing/invalid HBM evidence;
- optimizer placement other than `device-resident` before reverse or after
  commit;
- nonzero optimizer H2D/D2H time;
- non-finite gradient/parameter delta;
- invalid optimizer transaction, replica mismatch, topology/bucket drift, or
  accumulator/reference mutation.

Finite alignment drift is warning-only by the requested convergence contract.
It cannot promote the run to a strict zero-TIM claim.
