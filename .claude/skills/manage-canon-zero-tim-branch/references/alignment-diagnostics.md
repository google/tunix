# Alignment diagnostics

Load this reference before changing or operating an A/B/C alignment gate.

## Contents

- Keep the values and boundaries explicit
- Evidence contract
- Diagnostic versus training manifests
- Verdicts
- Diagnose a serving cache or page-state carrier

## Keep the values and boundaries explicit

Use these names consistently:

```text
A         S_decode    logprob emitted by rollout sampling
B         S_prefill   native engine prefill rescore
C-old     T_old       trainer-side pre-backward value
C-current T_current   value returned by the actual differentiable training program
```

Measure A-B and B-C before backward. Measure C-old/C-current only in a registered no-commit
backward or an admitted training step. Do not recompute or substitute old logprobs to hide a red
boundary.

## Evidence contract

For each masked boundary, require:

- `differing_bytes / total_bytes` and `differing_elements / total_elements`;
- byte and element fractions with the correct denominator;
- masked hashes for both arms;
- original sequence row, completion position, and token id;
- exact A/B scalar bits, XOR bits, differing byte offsets, ULP distance, and absolute delta;
- the maximum-absolute mismatch, not only the first mismatch;
- an explicit expected/actual count and truncation marker for bounded records.

Byte density only answers how frequently bytes differ. It does not bound logprob amplitude or the
importance weight. Never call a sparse result one ULP without the exact bit records.

Write JSONL, flush, and fsync before raising the hard-gate exception. Print a bounded strict-JSON
copy and the report SHA-256 to stdout. On nonzero workload exit, make the runner repeat the saved
record into stdout because pod-local paths may disappear.

Encode nonfinite values explicitly so a discovered NaN or infinity cannot crash the evidence
serializer. Reject an invalid shape loudly; equal invalid sentinels are not a hash match.

## Diagnostic versus training manifests

A diagnostic manifest must enforce all of these:

```text
one alignment transaction
optimizer commit disabled
Attempt 0
zero retries
source and runtime pinned
hard B-C gate retained
full evidence emitted before exit
```

Do not rely on an expected A-B red to stop a full manifest. If that sparse red is not sampled, the
job may enter training. Read the active P38 handoff for the current order. The current split keeps
FrozenLake on a strict no-commit root-cause track while an explicitly admitted GSM8K full campaign
may run with warning-only alignment and a convergence-only claim ceiling.

## Verdicts

- A-B red and B-C exact: the serving decode/prefill carrier reproduced. Localize before repair.
- A-B exact and B-C exact on sparse GSM8K: `NON_REPRODUCTION`, not repaired; run the fallback.
- B-C red, invalid shape, missing target line, retry, source drift, or infrastructure disconnect:
  numerical verdict not admitted.
- All forward boundaries exact: proceed only to a registered no-commit actual-model
  C-old/C-current gate. Generic probes do not replace the production model gate.

Do not introduce a tolerance, precision change, old-logprob substitution, or optimizer commit as
a diagnostic convenience. A user-approved warning-only committing campaign is a separate policy
track: encode it in a workload-specific default-off flag, retain every mismatch, keep all
non-alignment safety gates hard, and label the result `convergence-only`.

## Diagnose a serving cache or page-state carrier

Use this order when production decode differs from canonical rescore but a clean local replay does
not reproduce the captured decode values:

1. Capture the actual `continue_decode` serving call, not only prompt-logprob or adapter metadata.
2. Record only scheduled requests without compacting their physical slots. Join the selected row
   to the mismatch by request ID and exact token-history hash; row ordinal is not a durable key.
3. Record the call ordinal, request/DP/slot/index mappings, q/kv lengths, query starts,
   distribution, physical block table, cache identity, allocation/reuse generation when available,
   read range, write range, and per-layer K/V page hashes before and after every mutating event.
4. Replay stock first. Require the complete A and B action vectors, counts, bits, and masked hashes
   to reproduce exactly. If stock is not exact, stop every repair and topology arm.
5. Change one variable only after stock reproduces. For relocation or contiguity arms, prove
   logical page-content equivalence at every write event. For padding sanitization, add a finite
   poison control after proving the sentinel page is absent from all valid rows and does not alias
   live data.

Interpret narrow kernel probes conservatively:

- Same KV bytes producing exact decode/prefill output excludes only the tested arithmetic path and
  configuration; it does not prove production page ownership or scheduler correctness.
- Injecting stale or wrong page bytes and reproducing the observed magnitude proves sufficiency,
  not that production supplied those bytes.
- A full-page rewrite succeeding in isolation does not exclude partial writes, offsets, async
  visibility, stale block tables, or lifecycle bugs in the real multi-host serving envelope.

Claim a scheduler ownership bug only after production evidence shows overlapping live ownership or
an ownership-generation mismatch. Claim a stale block-table bug only after a request retains a page
whose ownership changed. If ownership and mapping are valid but a post-write page hash differs from
the canonical K/V hash, localize the writer, offset, or partial-write path instead.

The current RPA v3 public contract cannot isolate writer-only behavior because
`update_kv_cache=False` skips the write and forces all-cache reads. Treat a two-pass
write-then-read experiment as a combined `U` arm. If it becomes exact, the combined mechanism is
causal but writer and read-source remain inseparable.
