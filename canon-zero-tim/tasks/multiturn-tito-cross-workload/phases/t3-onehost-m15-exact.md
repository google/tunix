# T3 — One-host M15 exact TiTO certification

- Status: complete at one-host scope; DP8xTP8 unverified

## Motivation

The legacy r7 arm proved that this bounded M15 carrier happened to be
retokenization-stable. It did not execute the exact token-input path. The user
requires the M15 TiTO implementation itself to be proven healthy, so T3 is
reopened even though TiTO is not needed to repair the observed legacy trace.

## Registered carrier

- same Qwen3-8B M15/main DP1xTP4, seed, data, three rounds, generation shape,
  concurrency one, and pinned image as legacy r7;
- APC off, `CANON_M15_TOKEN_CONTINUITY=exact`;
- first prompt follows the ordinary path; every later prompt is the exact
  integer concatenation of the initial prompt, sampled assistant IDs, and
  nonterminal environment IDs;
- serving-consumed unpadded prompt IDs are compared to that exact ledger at
  every exercised later turn;
- strict A/B/C, B full reset/all cached-token counts zero, controlled exit,
  zero backward, zero optimizer commit, no eval/checkpoint/W&B.

## Gates

1. Host selector admits exact only for the isolated one-host identity and
   continues to reject APC-on, topology/profile leakage, eval, checkpoint,
   malformed token arrays, and caller-owned prompt overrides.
2. Each exact receipt must be `TOKEN_STREAM_EQUAL`; one different or missing
   receipt is fatal, not a scientific-success branch.
3. All three A-B and B-C comparisons must be zero bytes and finite; any
   alignment red is fatal.
4. Exactly three explicit B-full-reset/all-cached-zero receipts and the signed
   controlled exit are required; backward and optimizer commit remain zero.
5. Cross-arm comparison against legacy r7 requires identical ordered prompt
   token lengths/SHA values and identical per-round action-token and token-mask
   hashes. A mismatch is `INCONCLUSIVE_CROSS_ARM` and blocks any TiTO claim.
6. Production remains selector-absent. DP1xTP4 success is not DP8xTP8
   certification and does not independently authorize a production default.

## Result

Verified by exact r8 on pinned image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with source `0b90ff75ef7581c4230c0253df67779d06066792` plus recorded diff
`e788a1c8571ef335b8851c45d310b96a855d76249d42bf7fc2ddb38450a75a64`:

- 17/17 `exact` token receipts are equal and zero are different;
- three B-full-reset/all-cached-zero receipts are present;
- rounds contain 515, 766, and 748 action tokens, with A-B and B-C both
  exactly zero bytes in every round;
- backward and optimizer commits are both zero; controlled exit is 42;
- all seven external manifest entries verify; raw log SHA is
  `c3f026734255dc06a4cfca2d82f0769e62ffdafeca2926b68331c45c93d4dd64`;
- r7/r8 cross-arm comparison is `MATCH` for all 17 ordered prompt receipts
  and all three round token/action-mask hashes.

Durable summaries are `evidence/m15-onehost-r8-exact-summary.json` and
`evidence/m15-onehost-r7-r8-cross-arm.json`. The observed total wall time is
499 seconds versus 489 seconds for legacy r7; this single bounded pair is not
a component-level performance claim.

Not verified because no DP8xTP8 production target was run. Production M15
therefore remains selector-absent.
