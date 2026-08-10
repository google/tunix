# P35.3b bounded exact-input replay

Status: locally complete; target r30 pending

## Failure being repaired

Target attempt r29 completed rollout, A/B/C production measurements and the P35.2
attestations, then lost the Pathways IFRT connection during the first captured/live replay.
The run emitted neither the base P35 report nor the exact-replay report. The archived log
proves an `UNAVAILABLE: Socket closed` error and two captured B records. It does not prove OOM,
host transfer, worker eviction or a numerical mismatch.

Each captured record logically forms float32 logits with shape `(4096, 151936)`, about 2.49 GB.
The old replay tail materialized sampling output, canonical target logprobs and two target
gathers as separate eager dispatches, then started later replay arms without an explicit
record boundary. The old learner also wrote the already-complete P35.2 report only after the
optional replay succeeded.

## Bounded repair

1. Write an immutable preliminary A/B/C report before entering optional exact replay. Use a
   separate `.pre_replay.json` path and marker so it cannot be mistaken for final P35.3
   completion.
2. Preserve the original diagnostic sampling/logprob program boundaries. A fused target-only
   JIT candidate was tested first and rejected because it changed 178/256 CPU target logprobs
   by about one ULP. Zero-TIM does not admit that tradeoff.
3. Print replay-arm and per-record begin/complete markers, including logical logits shape and
   byte count.
4. Block every captured record before submitting the next record or replay arm. This bounds
   asynchronous work; it does not claim to reduce the model's intrinsic peak logits memory.
5. Keep all behavior behind the existing default-off P35 switches. Do not change production
   precision, loss, sampling, backward, optimizer or checkpoint behavior.

## Gates

- CPU: replay repeats remain exact with the original diagnostic value path, and the fused-tail
  negative experiment remains recorded as rejected rather than being reclassified as a pass.
- Exact image: focused replay and signed-zero/one-bit negative controls pass in the pinned
  FrozenLake image.
- One-host v5p TP4: focused real-device replay completes with four TPU devices and preserves
  the same bitwise test results. This is a code-mechanics gate only because the direct-attached
  host did not reproduce the 64-chip carrier.
- Target: only a new Attempt-0 Pathways run can decide whether the execution repair removes the
  r29 infrastructure interruption and can classify the exact-input boundary.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. No production path or default is
changed. Preserve r28 and r29 evidence unchanged.

## Local result

PASS for the admitted scope. The complete CPU contract, both exact-image overlays and the
four-device TP4 smoke are green. The one-host raw log contains four replay-arm begin markers over
two records and eight matching record-complete markers. Its first record deliberately contains no
action predictor. The cluster postflight negative control also proves that a replay failure
retains and hashes the preliminary A/B/C report. See
`../artifacts/p35_3b_local_gate.md`.

The 64-chip target remains untested with this repair. Do not claim that r29's IFRT interruption
is fixed until r30 completes on Pathways.
