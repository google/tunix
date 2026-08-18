# P38s22 analysis receipt — 2026-08-18

## Run-level status

`CLASSIFICATION_COMPLETE` / `TARGET_DISCRIMINATOR_EVALUATED`.

The committed evidence directory verifies its own 8-file SHA manifest. The target run completed all 3 frozen diagnostic rounds on 64 TPU (`DP16xTP4`, concurrency 256) with zero backward, zero optimizer commits, and controlled exit 42.

## Admitted numerical evidence

- Round 0: N_action=45,276; B-C exact (0 differing bytes).
- Round 1: N_action=44,695; B-C exact (0 differing bytes).
- Round 2: N_action=53,617; A-B 8 elements / 15 bytes; max_abs=0.289223; B-C=0 elements / 0 bytes.
- **S_prefill vs T_old (B-C)**: 100% bitwise exact (0 differing bytes, max_abs=0.0) across all 3 completed rounds.
- **S_decode vs S_prefill (A-B)**: Remains sparse red (15 differing bytes / 8 differing tokens in Round 2).

## Verdict and Decision

According to the P38.2w decision table:
`A-B remains red; B-C exact` -> `CANON_MM_ALGO` (`DotAlgorithmPreset.BF16_BF16_F32`) is rejected as a causal repair for the decode vs prefill carrier.

Generic dot algorithm / precision flags do not eliminate the carrier in `lm_head`. Move directly to a dedicated fixed-tile Pallas `lm_head` kernel design.
