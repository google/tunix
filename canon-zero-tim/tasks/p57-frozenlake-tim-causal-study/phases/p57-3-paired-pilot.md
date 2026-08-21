# P57.3 — Paired one-seed pilot

## Purpose

Verify that the frozen pair can operate for 50 updates and produce analyzable
checkpoints. This is an operational gate, not the scientific comparison.

## Design

- Arms: zero TIM and finite TIM.
- Paired seed: 42.
- Horizon: 50 updates.
- Checkpoints: every 10 updates, LatestN(1) during training; export the
  preregistered evaluation checkpoints before retention removes them.
- Evaluation: isolated held-out evaluation at updates 0, 20, and 50.
- No in-training evaluation and no arm-specific performance setting.

## Gates

- Intent diff shows only the registered numerical zero-TIM bundle and its
  corresponding arm label/policy; all nonnumerical settings are identical.
- Zero arm stays A=B=C bytewise exact.
- Mismatch arm retains finite A-B dose and B=C exactness.
- Both arms complete 50 valid transactions, checkpoints, and evaluations.
- Neither arm is at task floor or ceiling; both show a usable learning signal.
- No OOM, nonfinite, replica, transaction, truncation, or checkpoint fault.
- Context, group effectiveness, action validity, and resource use stay inside
  the frozen calibration envelope.

## Decision branches

- Pass: request approval for P57.4.
- Operational failure unrelated to treatment: repair the shared machinery and
  rerun both arms from the same initial state.
- Zero exactness loss or missing mismatch dose: return to P57.0/P57.2.
- Benchmark floor/ceiling: mark the frozen benchmark invalid. Do not silently
  select a new task after comparing arms.

## Claim boundary

The pilot cannot be cited as evidence that one arm learns better. Its seed and
horizon exist only to validate the campaign machinery.
