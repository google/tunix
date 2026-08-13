# P38.2j P38s12e evidence audit

Date: 2026-08-13 UTC

## Verdict

`P38s12e` is `INCONCLUSIVE_WRONG_RUN_DUPLICATED`. It is not a new
concurrency-32 measurement.

## Direct evidence

- All 365 `[sync] HEAD` records equal
  `bdc9681824743911d0691659604dec090dd42bc4`.
- Every JobSet/state path is `p38s12d-bdc96818`; the repaired commit
  `6c3938a6f2fe` never appears.
- The 41,675-line file decomposes exactly into five copies of the first pod's
  199-line output plus 360 copies of the second pod's 113-line output:
  `5 * 199 + 360 * 113 = 41,675`.
- The first repeated output fails before rollout with
  `P32 FrozenLake geometry mismatch: {'max_concurrency': 32}`. The second
  repeatedly refuses an existing `CANON_RUN_LOG`.
- There are zero capture-init/observe/error records, zero alignment records,
  zero depth-sufficiency records, and no controlled terminal result.
- `pre-alignment.jsonl` is empty. `serving-classification.json` contains five
  concatenated `INCONCLUSIVE` JSON objects and is not one JSON document.
- `SHA256SUMS` passes. That proves only byte transport of the wrong artifacts.

## Operator correction

Use a fresh `p38s12f` run id and source containing `6c3938a6f2fe`. Fetch
`head.follow.log` while streaming, then fetch `head.full.log` exactly once with
`>` after the pod is terminal. Never append full logs in a polling loop.
Before sealing, require one source marker, one command, no old run identity,
one nonempty pre-alignment JSON record, and one parseable PASS classification
whose source matches the selected commit.

No source code, numerical kernel, target experiment, backward, or optimizer
operation was changed or executed by this audit.
