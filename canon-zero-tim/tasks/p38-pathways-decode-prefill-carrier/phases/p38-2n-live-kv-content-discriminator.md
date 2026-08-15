# P38.2n — Live-KV content discriminator

Status: complete at analysis level. N3 completed and P38s17 ran, but the
originally archived N4 classification was not reproducible. Correct
reclassification selects the equal-fingerprint/program-envelope branch.

## Goal

Resolve the remaining production boundary without changing model weights,
decode geometry, concurrency, canonical M, attention implementation, or
training behavior:

1. live serving KV content differs from deterministic clean recomputation; or
2. live KV content is exact and the first difference is in the decode program
   envelope (`q_norm -> post-RoPE -> RPA -> residual -> MLP -> logits ->
   normalizer`).

This phase does not select a repair until one branch is measured.

## Entering evidence

- P38s16 ran three frozen-weight rounds at production shape (`DP16xTP4`,
  global padded rows 4096, local canonical logprob rows 256), with no backward
  and no optimizer commit.
- B-C was bitwise exact in all rounds. A-B was red at 32 / 17 / 11 elements.
- `audit_p38_single_active.py` validates 3,686 ledger records / 44,676 request
  entries and joins all 60 mismatch elements exactly. There is one naturally
  single-active mismatch: round 2, row 255, call 4223, request
  `2529-a6d304ba`, prefix 2209, `abs(A-B)=0.000301361083984375`.
- The call retained the production fixed-M avals and exact token history. It
  did not retain live KV bytes. Observation generations are not allocator
  generations and do not prove content equality.
- Concurrency 32, instantaneous co-batch >1, shape-one compilation, same-DP
  simultaneous page aliasing, and KV-unified reads are closed as necessary
  causes. Historical live cache state remains open.

## Deliverables and gates

### N0 — Reproducible host-only audit

Run:

```bash
python3 \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/audit_p38_single_active.py \
  --evidence-dir \
    canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/evidence/p38s16 \
  --output \
    canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/artifacts/p38s16_single_active_audit_0814.json \
  --expected-target-call 4223
```

Gate: 60 joined mismatch elements, zero missing/ambiguous joins, exactly one
call-4223 target, one fixed-M geometry signature. The output claim level is
`exact-host-join-not-kv-content`.

### N1 — Completion-last evidence transport

The live snapshot worker owns final persistence:

1. head produces classification and archive;
2. head atomically requests `collect`;
3. worker validates/uploads artifacts and writes a collect ACK;
4. head runs every numerical/shape/depth/transport postflight;
5. only after all pass, head atomically requests `complete`;
6. worker writes `COMPLETE.json`, ACKs, and is stopped.

Any failure before step 5 may leave durable `COLLECTED` evidence but must not
leave `COMPLETE`. Tests must cover worker-owned collect/complete, repeated or
malformed control files, missing artifacts, and a normal controlled exit.

### N2 — One-host E0-lite preflight

Call 4223 may be replayed on DP1xTP4 only as an input/oracle construction
preflight. It is not strict E0 because DP/mesh/sharding differ from production.

- If REF does not reproduce B/T-old, repair the clean recomputation contract.
- If local R0 reproduces A, record it as a candidate program signal but do not
  claim production executable identity.
- If local R0 is exact with B or produces a third value, do not select a repair;
  proceed with the live content observer.

Completed result: `E0_LITE_ENVELOPE_NOT_REPRODUCED`. REF reproduced all 646
production B/T-old values exactly. R0/R1 were repeat-exact but differed from
production A/B at 428 values (`max_abs=29.4570369720459`). The negative
control and 399-leaf weight attestation passed; no backward or optimizer
commit ran. This closes local operator counterfactuals and leaves N3 as the
next executable work.

### N3 — Default-off neutral live-KV observer

Before target admission, implement and rehearse one shared fingerprint
callable for live and clean KV. Requirements:

- disabled by default and admitted only by the strict P38 no-backward profile;
- executes after the decode result for the observed call is already produced;
- bounded to natural single-active requests in the registered deep interval;
- records layer, logical page, physical page, valid-token extent, and a
  deterministic bit-level fingerprint plus small fixed samples;
- the clean oracle uses the exact token history and the same weights;
- one-bit injection must change the fingerprint;
- observer-on/off one-host outputs must be bitwise equal;
- unknown dtype, page mapping, sharding, partial record, or size overflow fails
  closed;
- local overhead and bytes must be recorded before target admission.

Do not call a compact fingerprint a cryptographic hash. Collision risk remains
part of the claim ceiling unless full bytes are compared.

Primitive status (2026-08-14): the shared all-prefix callable,
DP-local-to-global page mapping, byte bound, and fixed samples pass CPU tests
and a real four-chip v5p TP4 rehearsal. One end-of-request table covers every
valid extent 1..256 for each page, so a later mismatch can index the exact
historical prefix without observing every decode token. For 36 layers and nine
selected pages it read 339,738,624 bytes and returned 5,308,416 bytes; first
compile took 34.276 s, a warm observation took 0.9514 s, and host transfer took
0.0078 s.
Observer-off/on endpoints were bitwise equal, repeats were exact, and a
one-bit mutation of a normal non-zero BF16 value changed the fingerprint.
Flipping the low bit of BF16 +0 was rejected as a negative control because the
TPU path may flush that subnormal back to zero.

N3 completed on 2026-08-15. Patch 16 wires the same fixed-shape callable to:

- A: the completed live decode request after its sampled result exists; and
- B: the exact token-prefix clean rescore immediately after `model_fn` has
  materialized the final prompt chunk, outside `maybe_forbid_compile`.

The real Qwen3-8B DP1xTP4 `p38_2n_kvobs_r6` rehearsal produced exactly three
A and three B records. Token histories, valid extents, provenance, and record
SHA identities passed; all three local pairs were fingerprint-exact and the
classifier returned `observer_pairs_valid_red_join_pending`. A-B itself was
exact locally, so this proves wiring/neutrality only. See
`artifacts/p38_2n_kv_observer_onehost_0815.md`.

The rejected r1-r5 rehearsals found and closed three runtime bugs: consumed
prompt-logprob identity, prompt-only requests absent from sampled output rows,
and observer compilation under the engine's compile-forbidden context. An AST
regression test now enforces the final hook placement.

Packaging gates pass in the pinned image: both Qwen3-1.7B and Qwen3-8B
overlays verify all 30 manifest entries and pass 29 runner tests. The focused
P38 observer/postflight/persistence gates pass. The broader host CPU suite
reaches an unrelated pre-existing environment mismatch: its host
`tpu_inference` lacks `compute_and_gather_logprobs`; the pinned-image gate is
the authoritative engine identity for this phase.

### N4 — One production-shape discriminator

Freeze the source and run one stock P38 diagnostic at the existing
`DP16xTP4`, concurrency 256, prefix-cache-off geometry. No backward, optimizer
commit, KV-unified arm, concurrency arm, or repair arm is permitted.

Admission is conditional on publishing the reviewed worktree and rendering
from that exact immutable source SHA. Use a new run id (`p38s17` is reserved),
`--stock-only`, and `--max-concurrency 256`. Do not manually edit the YAML.

Post-hoc join every new red mismatch to its exact call and compare live versus
clean fingerprints:

| Readout | Verdict | Next action |
|---|---|---|
| live KV differs from clean oracle at a layer/page before the red token | cache write/lifecycle/content carrier | localize the first dirty layer/page and repair that writer/lifecycle path |
| all compared live KV equals clean oracle | KV-content hypothesis rejected for the observed incident | run the ordered in-situ seam walk, beginning at q_norm/post-RoPE |
| no joined red call or insufficient content coverage | inconclusive instrumentation | fix coverage only; no mechanism claim |
| observer changes normal outputs or negative control stays green | invalid observer | reject the run and repair the observer |

## Claim ceiling

- P38s16 closes instantaneous environment variables, not historical state.
- A one-host replay is E0-lite and cannot prove production program identity.
- Host page tables and observation generations do not prove device content.
- A live observer may perturb timing even when it does not alter the current
  output. Report that limitation and require repeated carrier reproduction.
- N3 passing authorizes only the single stock N4 discriminator after immutable
  source publication. Repair, backward, optimizer commit, and strict
  full-training admission remain forbidden until N4 selects and validates a
  mechanism branch.

## Corrected N4 result

P38s17 produced three A-B-red/B-C-exact rounds. Reclassification from the six
observer records and exactly the three immutable round capsules reports
`live_kv_fingerprint_equal_on_red_row`: zero valid-region aggregate and sample
differences, with a red row joined in every round. The committed directory is
only a live snapshot and has no terminal `COLLECTED`/`COMPLETE` markers, so the
result is analysis-level. P38.2o owns evidence-chain repair and the ordered
decode seam walk. No cache-writer repair is selected by this phase.
