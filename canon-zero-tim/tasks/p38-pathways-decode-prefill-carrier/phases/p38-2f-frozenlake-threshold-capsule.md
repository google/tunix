# P38.2f: FrozenLake KV-threshold mismatch capsule

## Purpose

Turn the sparse P38d5 FrozenLake red boundary into a replayable input rather
than paying for another opaque 64-chip run.

## Target fact

`debug_logs/p38_p38d5_frozenlake_bwd.raw.log` recorded 25 differing action
elements out of 48,946, with maximum absolute difference `0.153839111328125`.
All localized mismatches have logical KV prefix at least 1791. The earliest is
row 238 at sequence-chunk offset 255. Row 255 first differs at KV prefix 1800
in turn 2, and none of the 25 records is adjacent to an environment token.
This supports a depth/chunk boundary hypothesis; it does not prove an
attention tile or page-size cause.

## Capsule contract

Only the FrozenLake `backward-no-commit` JobSet receives a capsule path. On a
blocking pre-backward red, the learner persists at most two localized rows:

- prompt ids and mask;
- completion ids, sequence-valid mask, and action mask;
- `S_decode`, `S_prefill`, and `T_old`;
- policy version and sampling values;
- per-array shapes, dtypes, SHA-256 values, source record hash, and selected
  global row ids.

The NPZ is written before the numerical exception. The runner prints its
byte count, SHA-256, and base64 payload to stdout. The extraction tool verifies
transport SHA, schema, and every embedded array SHA before writing a local
file. Existing paths, invalid bounds, missing prompt ids, or missing localized
rows fail closed.

## Next replay after target capture

Recover the capsule from the immutable pod log, then use the existing
direct-attached DP1xTP4 host to run a no-backward single-row sweep. Keep prefix
cache disabled and record raw target logit, vocabulary normalizer, processed
logprob, q/kv lengths, page-table digest, and cache digest for every arm.

Pre-registered prefix points include the captured onset and
`1790/1791/1792`, `1800`, `1931`, `2047/2048/2049`, and `3839/3840` when the
captured row is long enough. Compare teacher-forced q_len=1 decode with the
unchanged M256 rescore, then sweep chunk sizes only as diagnostic arms. Add a
single-turn sequence at the same depth as a negative control.

- Raw target logit first differs: localize upstream through attention/KV.
- Raw target is exact but normalizer differs: localize the vocabulary
  reduction tail.
- Only processed logprob differs: localize transform/gather/subtraction.
- No one-host reproduction: retain a Pathways-only classification; do not
  call the carrier repaired.

## Rollback

Leave `CANON_P38_MISMATCH_CAPSULE` empty or revert this phase. Normal training,
precision, prefix-cache policy, attention kernels, and loss are unchanged.
