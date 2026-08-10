# P38.1 — Durable and localizable pre-backward evidence

- Status: complete

## Finding

- Confirmed: `_masked_bitwise_difference` already computes element counts and a first value but
  loses the original two-dimensional coordinate and exact bit pattern.
- Confirmed: `check_pre_backward` writes the complete record under `/tmp/canon-state`, while its
  stdout line contains only differing-byte counts. `90_run.sh` classifies P33 evidence only after
  a zero workload exit, so a hard pre-backward failure strands the JSON inside the failed pod.
- Confirmed: r35 README labels differing bytes as tokens. FrozenLake's 70 differing bytes imply
  between 18 and 70 different float32 elements, not 70 established token mismatches.

## Execution

1. Preserve up to 1024 mismatch records with row, completion position, exact float bits, XOR,
   differing byte offsets, ULP distance, absolute delta, and token id.
2. Flush and fsync the JSONL before raising, print a compact JSON copy and report SHA to stdout,
   and have the runner repeat failed pre-alignment artifacts before postflight exits.
3. Keep both pre-backward boundaries fail-closed. Do not add a tolerance or diagnostic training
   override in this phase.
4. Correct the r35 evidence units without changing the original raw logs.
5. Add positive and negative controls for coordinates, signed zero, one ULP, high-amplitude sparse
   drift, record truncation, invalid shapes, and nonzero workload exits.

## Exit gate

- Command: `JAX_PLATFORMS=cpu python3 -m unittest discover -s tests/rl -p alignment_test.py`
- Command: `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh`
- Pass: every registered test completes; a deliberately failed workload exposes the full
  pre-alignment JSON and SHA in stdout; one-bit and invalid-shape controls remain rejected.
- Fail: missing measurement, unbounded output, changed gate semantics, or a failed run without a
  durable stdout record keeps P38.1 active.

## Result

PASS locally in the pinned FrozenLake image. The focused alignment suite completed 18/18 tests.
The complete P33 CPU gate also passed and exercised the injected nonzero workload exit, which
preserved the report SHA and complete JSON in stdout. Static shell, Python, and diff checks passed.
No TPU result was produced and no release numerical gate was promoted by this phase.
