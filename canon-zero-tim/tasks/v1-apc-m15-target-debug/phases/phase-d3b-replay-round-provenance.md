# Phase D3b replay-round provenance

## Objective

Repair the Attempt-15 round-seal instrumentation failure without changing any
APC or model arithmetic.  Every `m15-apc-serving-envelope-v1` row must carry
the live diagnostic round so the cumulative replay ledger can be partitioned
into sealed round 0, 1, and 2 inputs.

Attempt 15 is not a prefix-cache fix result.  Both arms passed the Round-0
pre-alignment gate, then the control plane stopped before backward or optimizer
commit because replay records omitted their round identity.

## Immutable Attempt-15 facts

- Runtime source was independently bound as
  `57d9ab8e25de3b2404e983e9a139d78b151a58f8` by both live-worker logs.
- APC-off: `N_action=120889`, A-B=0 bytes, B-C=0 bytes.
- APC-on: `N_action=130468`, A-B=0 bytes, B-C=0 bytes.
- Both records say `backward=0 optimizer_commits=0`.
- The learner requested the Round-0 seal; assembly exited 2 on
  `replay round is invalid at line 1`.
- The returned replay head contains 20/20 valid envelope rows and 20/20 omit
  `diagnostic_round`.
- The existing failure receipt and learner fail-fast path worked as designed.

The incident report's statement that backward Pallas hot paths ran is
withdrawn.  Its full source SHA is also superseded by the independently
verified live-worker runtime-source receipt above.

## Implementation

The published patch chain remains immutable.  Patch 26 is not edited.
Additive patch 33 inserts exactly one field into the replay producer:

```python
"diagnostic_round": int(_p38_seam_round()),
```

The value is read when each host-only envelope is serialized.  It does not
fetch a device value, change request order, alter cache state, add a JIT, or
touch A, B, C, RoPE, RPA, KV contents, LM head, loss, backward, or optimizer.
The assembler remains fail-closed for missing, invalid, or foreign rounds.

An AST probe validates the final installed runner rather than merely finding a
string in a patch.  It requires the field to be present in the one record with
schema `m15-apc-serving-envelope-v1` and to come from exactly
`int(_p38_seam_round())`.  Missing fields and a hard-coded `0` are negative
controls.

## Local gates

- M15 target-debug discovery: 139/139 PASS.
- Focused target carrier: 17/17 PASS.
- Wide durability: 8/8 PASS, including round-1 replay selection.
- Wide classifier: 10/10 PASS.
- P38 persistence: PASS, including three worker rounds and fail-fast failure.
- Flag registry: 395/395, `FLAG_AUDIT_PASS`.
- Patch 33 applies to the manifest-matching prepublish runner; the installed
  source compiles and the AST probe passes.
- Bash/Python syntax and `git diff --check`: PASS.

The new installed runner SHA is registered as
`c527d31a6343c673a3c93988b15db37d85000956098a737136bac9af8387bc81`.
This is construction evidence, not exact-image or target evidence.

## Remaining gates

1. A separately approved pinned exact-image run must build the complete
   overlay, verify the manifest, and emit
   `M15_REPLAY_ROUND_PROVENANCE_PASS` plus the aggregate marker with
   `apc_m15_carrier=68 m15_round_provenance=1`.
2. Commit and push remain separate user approvals.
3. A fresh matched DP8xTP8 pair remains a separate launch approval.  It must
   seal and ACK all three rounds and return the official per-round classifiers.

## Claim ceiling

`REPLAY_ROUND_PROVENANCE_LOCAL_PASS / NUMERICAL_PATH_UNCHANGED /
EXACT_IMAGE_NOT_RUN / TARGET_NOT_RUN / ROUND0_STOCHASTIC_EXACT_ONLY /
FIRST_RED_NOT_LOCALIZED / NUMERICAL_FIX_NOT_AUTHORIZED / PHASE_E_CLOSED`.
