# P38.2x — dedicated fixed-tile Pallas lm-head

Status: active. Implementation, CPU/static, exact-image, and real-weight
one-host construction gates pass. P38s23 is prepared but not launched.

## Entering evidence

P38s21 localizes the first measured red interval to `lm_head_logits` while the
selected final-hidden rows remain exact. P38s22 then rejects the generic
`BF16_BF16_F32` dot-algorithm preset in three independently sealed rounds:
66 A-B elements / 111 bytes across 143,464 actions, with exact B-C.

Code archaeology shows why this freedom remains. The seven transformer
projections are registered Pallas sites, but `JaxLmHead` intentionally does
not inherit `JaxEinsum` and still calls a separate `TD,DV->TV` einsum. The
current local serving shapes are decode M16 and prefill M256.

## Shape ledger

These row counts are deliberately distinct:

- caller-local semantic rows: decode M16; prefill M256;
- fixed lm-head kernel rows: M256 for both paths;
- hidden reduction width: K4096;
- global vocabulary width: V151936;
- TP4-local vocabulary width: N37984;
- fixed local vocabulary extent: N38144 (N37984 plus 160 zero columns);
- Pallas tiles: BM128, BN256, BK256.

The output is sliced back to the caller's semantic M and real local N before
`shard_map` reassembles the global vocabulary. `CANON_LOGPROB_M=256` remains a
separate downstream contract and is not renamed or reused here.

## Deliverable

1. Register default-off numerical flag `CANON_P38_FIXED_LM_HEAD` with an
   explicit sunset.
2. Intercept only `JaxLmHead.__call__` when the flag is exactly `1`; the flag-off
   method remains the inherited original without a wrapper.
3. Fail closed on any model, TP width, dtype, equation, M, K, N, missing mesh,
   missing canonical dependency, or conflicting diagnostic outside the ledger.
4. Reuse the promoted P22.XK primal/custom-VJP stack. Both M16 and M256 enter
   the same Pallas shape `[256,4096] @ [4096,38144]`.
5. Emit a compile-time PATHTRACE containing semantic and fixed shapes.

## Gate ladder

1. CPU/static: flag parser, shape ledger, wrong M/K/N/dtype/TP negatives,
   source wiring, manifest, Python/Bash syntax, and one-bit comparator negative.
2. Exact image: install the Qwen3-8B TP4 overlay and attest the new module and
   flag-on lm-head hook without changing the default profile.
3. Real v5p: load real Qwen3-8B BF16 lm-head weight, run four deterministic
   seeds at M16 and M256 through the fixed construction, require shared rows to
   be bitwise exact, require production tile/path receipts, and require the
   injected one-bit negative to report exactly one.
4. Only after 1-3 pass and a separate user launch approval: P38s23, one slim
   three-round 64-TPU stock arm with this flag as the single numerical change.

Current gate result: 4/4 seeds are fixed-M cross-shape exact, the one-bit
negative reports exactly one, and fixed-versus-stock differs at 211--268
selected elements per seed. Receipt:
`../artifacts/p38_2x_fixed_lm_head_onehost_0818.md`.

## Target decision table

- A-B becomes zero in all rounds and B-C stays exact: candidate causal repair;
  proceed to P38.2h backward-no-commit before any production default.
- A-B remains red and B-C stays exact: fixed lm-head program freedom is
  rejected; revert the flag-on arm and reopen the remaining tail interval.
- B-C becomes red, any dependency/path receipt is absent, or any target
  contract fails: instrumentation/configuration failure; no numerical claim.

## Claim ceiling and rollback

One-host exactness is construction evidence only and cannot prove Pathways
repair. P38s23 is forward-only and cannot admit backward, optimizer, training,
or production performance. The known decode cost is up to 16x lm-head row work
locally; performance is measured but never traded against bitwise admission.

Rollback is `CANON_P38_FIXED_LM_HEAD=0` or unset. No existing canonical default,
prefix-cache setting, runner geometry, checkpoint, or source evidence object is
changed by this phase.
