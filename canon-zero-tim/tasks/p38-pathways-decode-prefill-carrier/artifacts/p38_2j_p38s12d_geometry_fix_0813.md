# P38.2j P38s12d geometry failure and local fix

Date: 2026-08-13 UTC

## Target evidence

P38s12d used source `bdc96818` and correctly rendered the intended stock
concurrency-32 command. It did not reach rollout, capture, alignment,
backward, or an optimizer operation. The FrozenLake recipe rejected the
command during startup with:

```text
ValueError: P32 FrozenLake geometry mismatch: {'max_concurrency': 32}
```

The later missing-capture and stale-`run.log` messages are consequences of
that startup failure. P38s12d contains no numerical carrier evidence and is
`INCONCLUSIVE`.

## Root cause

The P38 renderer and intent-diff gate admitted `--max-concurrency 32`, but the
recipe still hard-coded max concurrency 256 for every canonical FrozenLake
invocation. The renderer contract therefore could not reach the workload it
was designed to test.

## Local repair

The recipe now delegates max-concurrency validation to a shared fail-closed
contract. Concurrency 256 remains valid everywhere. Concurrency 32 is valid
only when all of the following identify the bounded stock P38 arm:

- workload is exactly `frozenlake`;
- stage is `backward-no-commit`, no-commit is enabled, and launch is admitted;
- P38 precheck, controlled exit, capture directory, and request journal are
  configured;
- capture path is `standard`; and
- KV-unified is disabled.

Missing any guard, selecting DP8xTP8, selecting KV-unified, or requesting any
other concurrency remains a hard error. Full training and evaluation retain
concurrency 256.

## Verification

- Python compilation: PASS.
- Pinned-image focused workload + renderer tests: 59/59 PASS.
- Pinned-image complete P33 CPU/adjacent gate: PASS; workload suite 85/85,
  alignment 34/34, adjacent learner 15/15, all P38 renderer/classifier/
  postflight and negative controls green; terminal marker
  `[P33.WORKLOAD] CPU_GATE PASS`.
- `git diff --check`: PASS.

No cluster run, backward, optimizer operation, commit, or push occurred.
The next target action remains one newly rendered, source-pinned, stock-only
P38s12b concurrency-32 attempt. Do not reuse the P38s12d YAML or source SHA.
