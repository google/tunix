# P38.2g6 local gate — 2026-08-12 UTC

## Scope

This checkpoint verifies the default-off standard/mixed runner capture
construction against the pinned `tunix_frozenlake_image:vllm-tpu0.25.0`
image. It does not claim a target Pathways capture or a numerical repair.

## Results

- Serving classifier: 26/26 PASS.
- P38 JobSet renderer: 5/5 PASS.
- Serving shell postflight and its fail-closed controls: PASS.
- Exact-image overlay gate:
  - Qwen3-1.7B: 20/20 PASS;
  - Qwen3-8B: 20/20 PASS;
  - manifest: 29/29 entries match for both overlays.
- Full pinned-image CPU regression gate: PASS. Its terminal marker was
  `P33.WORKLOAD CPU_GATE PASS`; it included 81 workload tests, 31 alignment
  tests, the 26-case serving classifier suite, and adjacent contract/negative
  controls.
- `git diff --check`: PASS.

Installed `tpu_runner_p21_l30.py` SHA-256:

`a7bdc527182ad115385e60005cff8c4e135efd2714eb97a2e929dc3dbc45e890`

## Claim ceiling

The standard path is locally reachable under a fake mixed scheduler state,
its packed-token mapping is tested, the completion hook serializes numeric
step-major samples, and wrong-path/async controls reject. Real Qwen3-8B
Pathways execution, four production records, exact capsule join, block-table
evidence, first-divergence localization, backward, and optimizer behavior are
NOT RUN.

## Next

Publish this source, then follow the P38s7 stock-only Attempt-0 protocol at
the top of `HANDOFF.md`. Do not force-enable continue decode and do not rerun
the KV-unified arm.
