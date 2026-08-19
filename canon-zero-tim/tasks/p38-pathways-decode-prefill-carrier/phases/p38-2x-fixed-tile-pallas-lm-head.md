# P38.2x — dedicated fixed-tile Pallas lm-head

Status: active. P38.2x2 passed the one-host construction gate and P38s23r2
measured one exact 64-TPU forward round (`A-B=0`, `B-C=0`, 49,177 actions).
That run is analysis-grade only because the shared full-forensics snapshot
worker starved the critical round seal beyond 900 seconds. P38s23r3 is the
registered durability-only retry: identical numerical arm, three frozen
rounds, and a round-priority minimal alignment evidence profile.

## Entering evidence

P38s21 localizes the first measured red interval to `lm_head_logits` while the
selected final-hidden rows remain exact. P38s22 then rejects the generic
`BF16_BF16_F32` dot-algorithm preset in three independently sealed rounds:
66 A-B elements / 111 bytes across 143,464 actions, with exact B-C.

Code archaeology shows why this freedom remains. The seven transformer
projections are registered Pallas sites, but `JaxLmHead` intentionally does
not inherit `JaxEinsum` and still calls a separate `TD,DV->TV` einsum. The
the live request-count compiler uses a power-of-two bucket ladder while the
canonical prefill/rescore endpoint uses M256.

## Shape ledger

These row counts are deliberately distinct:

- admitted caller rows: exact request buckets M8/16/32/64/128/256 and exact
  learner-rescore M4096;
- fixed lm-head kernel rows: M256 for every request bucket; M4096 is mapped to
  exactly 16 independent M256 calls;
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
4. Reuse the promoted P22.XK primal/custom-VJP stack. Every registered M enters
   the same Pallas shape `[256,4096] @ [4096,38144]`.
5. Emit a compile-time PATHTRACE containing semantic and fixed shapes.

## Gate ladder

1. CPU/static: flag parser, shape ledger, wrong M/K/N/dtype/TP negatives,
   source wiring, manifest, Python/Bash syntax, and one-bit comparator negative.
2. Exact image: install the Qwen3-8B TP4 overlay and attest the new module and
   flag-on lm-head hook without changing the default profile.
3. Real v5p: load real Qwen3-8B BF16 lm-head weight, run four deterministic
   seeds at M8/16/32/64/128/256 through the fixed construction, require every
   bucket to equal the corresponding M256 reference rows bitwise, require all
   production tile/path receipts, then run M4096 and require all 16 chunks to
   equal direct M256 reference calls. Require the one-bit negative to report 1.
4. Only after 1-3 pass and a separate user launch approval: one slim
   three-round 64-TPU stock arm with this flag as the single numerical change.

The first P38s23 never passed item 4: `capture_model()` invoked compute_logits
at M32, and the deliberately narrow M16/M256 validator raised before rollout.
P38.2x1 corrects the test envelope without admitting arbitrary row counts:
only `(8,16,32,64,128,256)` is legal, while M1/M7/M24/M257 remain red.

Current repaired gate result: 24/24 bucket comparisons are exact,
`max_abs=0`, the one-bit negative reports exactly one, all six lowerings carry
custom calls, and fixed-versus-stock differs at 211--268 selected elements per
seed. Receipt:
`../artifacts/p38_2x1_fixed_lm_head_bucket_onehost_0818.md`.

P38s23r1 subsequently reached the learner and exposed exact M4096. Falling
back to stock there is forbidden because it would give B/M256 and C/M4096
different lm-head programs and confound B-C. P38.2x2 instead uses `lax.map`
over 16 M256 chunks. CPU/static and pinned-image gates pass. Its real-v5p gate
then passed its real-v5p gate. P38s23r2 reached the numerical discriminator.

## P38s23r2 result and P38s23r3 durability amendment

P38s23r2/source `6814774eef70aa0c67610eab9f355d964d420378`
emitted all seven fixed-lm-head receipts and measured one exact round:
`N_action=49,177`, A-B differing bytes/elements `0/0`, B-C `0/0`, and
`max_abs=0.0`. It then published `round-000000.request` and timed out after
900 seconds without an ACK. No rounds 1/2, controlled exit, backward, or
optimizer result is admitted.

The timeout was not numerical. The only worker was already inside a periodic
full-forensics snapshot and shell priority cannot preempt an in-flight GCS
transfer. A second success-path defect was also present: exact rounds do not
create mismatch capsules, while the old round staging and root collection
required one. Finally, stock postflight required a mismatch join even though
the fixed-lm-head success case is exact by definition.

P38s23r3 fixes only this evidence/control plane:

- `CANON_P38_DURABILITY_PROFILE=round-alignment-v1` is exclusive to the
  fixed-lm-head arm;
- periodic live snapshots and KV/seam/tail/terminal observers are absent;
- the worker services only ordered round seals and terminal requests;
- each round archive contains `run.log`, one scoped `pre-alignment.jsonl`, an
  inventory, and a mismatch capsule only when the round is red;
- exact fixed-lm-head postflight does not require a mismatch join or capsule;
- three immutable round archives and root terminal markers remain mandatory.

Local fake-GCS gates cover round priority, exact-round capsule absence,
three-object archives, root collect/complete, manifest verification, both
scientific outcomes, and a truncated-head negative. Pinned-image renderer
(18), fixed-lm-head (8), serving-classifier (36), and complete P33 adjacent
CPU gates pass. The target launch remains separate and requires explicit user
approval after publication.

## Target decision table

- A-B becomes zero in all rounds and B-C stays exact: candidate causal repair;
  proceed to P38.2h backward-no-commit before any production default.
- A-B remains red and B-C stays exact: fixed lm-head program freedom is
  rejected; revert the flag-on arm and reopen the remaining tail interval.
- B-C becomes red, any dependency/path receipt is absent, or any target
  contract fails: instrumentation/configuration failure; no numerical claim.

## Claim ceiling and rollback

One-host exactness is construction evidence only and cannot prove Pathways
repair. P38s23r3 is forward-only and cannot admit backward, optimizer,
training, or production performance. The registered M8 bucket makes the
worst-case lm-head row-work multiplier 32x; performance is measured but never
traded against bitwise admission.

Rollback is `CANON_P38_FIXED_LM_HEAD=0` or unset. No existing canonical default,
prefix-cache setting, runner geometry, checkpoint, or source evidence object is
changed by this phase.
