# P38s22 analysis receipt — 2026-08-18

## Run-level status

`ANALYSIS_GRADE_TARGET_DISCRIMINATOR_EVALUATED`.

The target run completed all 3 frozen diagnostic rounds on 64 TPU
(`DP16xTP4`, concurrency 256) with zero backward, zero optimizer commits, and
controlled exit 42. The returned Git directory verifies its own 18-file
manifest, but the round-archive receipts require the independent offsite audit
registered in P38.2w1 before run-level durability can be called complete.

## Admitted numerical evidence

- Round 0: N_action=45,865; A-B 48 elements / 82 bytes; max_abs=0.263157; B-C exact.
- Round 1: N_action=43,982; A-B 10 elements / 14 bytes; max_abs=0.0160103; B-C exact.
- Round 2: N_action=53,617; A-B 8 elements / 15 bytes; max_abs=0.289223; B-C=0 elements / 0 bytes.
- **S_prefill vs T_old (B-C)**: 100% bitwise exact (0 differing bytes, max_abs=0.0) across all 3 completed rounds.
- **S_decode vs S_prefill (A-B)**: Remains sparse red in every round: 66 elements / 111 bytes across 143,464 actions in total.

The returned `p38_terminal.classification.json` is not admitted by this
receipt. P38s22 disabled the terminal observer and returned no raw terminal
JSON/NPZ inputs or classifier provenance. The lm-head interval localization
continues to come from the admitted P38s21 selected-point evidence.

## Verdict and Decision

According to the P38.2w decision table:
`A-B remains red; B-C exact` -> `CANON_MM_ALGO` (`DotAlgorithmPreset.BF16_BF16_F32`) is rejected as a causal repair for the decode vs prefill carrier.

Generic dot algorithm / precision flags do not eliminate the carrier in `lm_head`. Move directly to a dedicated fixed-tile Pallas `lm_head` kernel design.

This target decision does not depend on the unadmitted terminal classification.
Formal durability remains pending the read-only `P38S22_OFFSITE_AUDIT_RUNBOOK.md`
gate and does not require another TPU run.
