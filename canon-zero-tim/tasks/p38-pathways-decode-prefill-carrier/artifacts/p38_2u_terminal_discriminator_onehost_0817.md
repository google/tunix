# P38.2u terminal discriminator one-host receipt

Date: 2026-08-17 UTC

This receipt is a construction, negative-control, and endpoint-neutrality gate.
It does not reproduce or close the 64-TPU A-B carrier.

## Frozen source

Both final arms used base source
`a5145cd602283146527830e7038f077f93cbe203` and worktree diff SHA-256
`f049d26fa64ed15d362c7825d12af72609a9e50cf0e43310e7f7f294924ca6b9`.
No executable or classifier source changed between the two arms.

Critical payload hashes:

- patch 20: `3c16e901e5a8ed16c56dbfca54fdc4236b76adef692ebdec7c9fbec370e3666c`;
- terminal shim: `c3b024a7f0514f044a8264db250c9458302d317790a316010f3c27d6cfbca0ac`;
- terminal classifier: `97c67287ba8f13d9ef44d03279369c50f46b202f008ec59b3e7dcc00e9362cf7`;
- one-host driver: `2d3ebcdabb7a9d05d3fd0dd1ee3a034c033ce534a4671f35baac3a2353f7719d`.

The pinned image gate installed both Qwen overlays with 33/33 manifest files
and ran 34/34 runner tests per overlay:

```text
P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5
runner_tests_per_overlay=34 overlays=2
```

The complete pinned-image CPU suite ended with
`[P33.WORKLOAD] CPU_GATE PASS`. The real TPU positive/one-bit-negative unit
gate ran 3/3 tests successfully.

## Real v5p arms

Both arms used Qwen3-8B, `DP1xTP4`, prefix cache disabled, three frozen
rounds, zero backward, zero optimizer commits, and controlled exit 42.

The terminal-discriminator arm completed in 471 seconds:

```text
round 1: N_action=409, A-B=0 bytes
round 2: N_action=565, A-B=0 bytes
round 3: N_action=897, A-B=0 bytes
PASS mode=terminal-discriminator rounds=3 backward=0 optimizer_commits=0
```

It wrote 130 terminal JSON/NPZ pairs (42 MiB). The classifier joined 155 A
rows to B and returned:

```text
classification=terminal_rows_exact
stage_counts={exact: 155}
missing_b_rows=0
red_rows=[]
```

For every joined row the exact selected final-hidden bytes and the shared
fixed-four-row raw/processed logit evidence were equal. Classification SHA:
`4f3810f64f270ee25bfff3bc2d272b7f33135817f8467cbb2432119ea9ff2582`.

The observer-off arm completed in 451 seconds with the same three action
counts and exact A-B/B-C endpoints. `classify_p38_seam_neutrality.py` compared
the complete three-round alignment records except timestamps and returned:

```text
status=PASS
classification=observer_endpoint_bitwise_neutral
```

Alignment report SHA-256 values:

- terminal-discriminator:
  `c9e9ac93c216ffa658176533117c97a46f05562d277e6aed56e25edf8aa13138`;
- observer-off:
  `c032bf0e03c9fab0668cea231ed77a93c9b623758bb2b8585771806b84a5c3c3`.

## Two observer bugs caught before target launch

The first prototype fused source-shape-dependent gather and vocabulary
reduction. It falsely classified 148 locally exact rows as reduction drift.
The admitted design gathers separately, pads to exactly four rows, and runs
raw plus processed logits through one shared executable identified by
`reduction_program=shared-fixed-four-row-v1`.

The corrected raw observer then exposed two rows where the legacy
shape-dependent tail observer's intermediates differed even though the
production endpoint and the shared observer were exact. The final classifier
therefore records those rows under
`legacy_shape_dependent_tail_drift_rows` but does not let them outrank the
shared observer or production endpoint. Older terminal records without the
fixed-four program identity are rejected.

## Verdict

The P38.2u implementation is locally admitted for user review and publication.
It is default-off and endpoint-neutral on real v5p. After an explicit user-
approved commit/push, exactly one source-pinned P38s19 64-TPU stock diagnostic
may run. A target result remains diagnostic multiword logit evidence, not full
logit byte equality and not a zero-TIM closure.
