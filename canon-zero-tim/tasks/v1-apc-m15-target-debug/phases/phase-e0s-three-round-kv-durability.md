# Phase E0s — three-round targeted-KV durability

## Question

Can the D3e-bound Layer-0 live-KV discriminator produce the same mechanism
classification in three independent frozen-weight DP8×TP8 M15 rounds, while
making every completed round recoverable before a later round or root
collection can fail?

Attempt 18's one round was intentional under its old contract. It is not a
three-round stability result, and its returned package is provenance-rejected.
This phase is additive: legacy `observer=kv` remains one round; the new target
identity is `observer=kv3`, `CANON_P38_DURABILITY_PROFILE=m15-e0-kv-v1`, and
`CANON_P38_DIAGNOSTIC_ROUNDS=3`.

## Frozen numerical contract

```text
A = rollout decode (APC off for control, APC on for treatment)
B = serving prefill rescore of A action IDs, reset_prefix_cache=True
C = trainer old-policy forward

required each round: B-C = 0
required control:      A-B = 0
```

B full reset and zero cached-token receipts are immutable. This phase changes
no RoPE, RPA, attention, KV values, LM head, loss, backward, optimizer, model
weights, sampling recipe, production flag, or production APC default.

## Per-round transaction

For diagnostic rounds 0, 1, and 2, in order:

1. capture exactly eight targeted A aliases and their eight matched B records;
2. isolate the current round's alignment, replay-envelope rows, optional
   immutable mismatch capsule, and source-bound classifier;
3. self-hash a deterministic classifier-input archive, upload it, download it,
   and verify its hash before classification;
4. run the official classifier on the isolated round;
5. self-hash the final round archive, upload the archive and compact receipts,
   download them, and verify them;
6. publish `ROUND_COMPLETE.json` and verify its remote readback;
7. only then publish the learner ACK and advance to the next round.

Record indices remain globally monotonic and are never reused. Candidate
membership and the 128 MiB KV byte budget reset per round. A/B pairs may not
cross a round boundary. Periodic snapshots are disabled for this profile so
they cannot delay the ordered seal transaction.

The redundant incident ledger is bypassed only in this profile; the
round-filtered replay envelope plus sealed KV records are the replacement.
Other profiles retain their existing incident-ledger behavior.

## Stability verdict

- APC-off must be `CONTROL_EXACT_3_OF_3`.
- APC-on exact in all rounds is `TARGET_NON_REPRODUCTION_3_OF_3`.
- APC-on red with equal live-KV fingerprints in all rounds is
  `LIVE_KV_FINGERPRINT_EQUAL_3_OF_3`.
- APC-on red with different live-KV fingerprints in all rounds is
  `LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3`.
- Any mixed three-round outcome, red control, B-C red, missing alias, missing
  unique future-prefix binding, missing checkpoint, or missing readback fails
  closed.

The fingerprint claim remains
`bit-level-diagnostic-fingerprint-not-full-kv-bytes`; even a stable 3/3 result
does not by itself authorize a numerical repair.

## Recovery contract

The return script reads the four small per-round receipts first:

```text
ROUND_INPUT.json
kv-observer-classification.json
classifier-input/CLASSIFIER_INPUT_RECEIPT.json
ROUND_COMPLETE.json
```

It does not require root `COLLECTED.json` or `COMPLETE.json` to recover an
already completed round. Three recovered rounds with missing root terminal
state produce `ROUNDS_RECOVERED_ROOT_INCOMPLETE`; partial rounds produce
`ROUND_EVIDENCE_PARTIAL`. Both are INCONCLUSIVE at the target-certification
level and preserve local scratch/output.

## Gates

1. host implementation, fake-GCS upload/readback, round-2 failure injection,
   renderer/resolved-env, syntax, manifest reconstruction, flags audit;
2. separately approved official pinned exact-image aggregate;
3. separately approved fresh matched DP8×TP8 APC-off/on launch from one exact
   published SHA and fresh labels;
4. separately approved read-only GCS salvage/terminal return.

No later gate inherits approval from an earlier gate.

## Current status

Local host focused gates and fake-GCS durability/failure injection pass.
The aggregate host gate also passes task discovery 193/193, V1 CPU 91/91, P3
31/31, flag registry 398/398, syntax, static manifest binding, return recovery,
and diff checks. Official pinned exact-image and DP8×TP8 target have not run.
Phase E remains closed and no numerical repair is implemented or authorized.
