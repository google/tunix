# P58.5N — Native 128-chip full campaign

Status: active.

## Purpose

Run the untreated native DeepSWE-derived Qwen3-4B-Instruct path for exactly
1,000 committed optimizer updates on the frozen P58 recipe. This is the user's
direct-full decision: updates 1–3 are live monitoring milestones within one
continuous job, not a separate canary and not an early-stop condition.

This phase proves native integration and training only. It does not run or
validate the deferred zero arm and cannot establish a paired treatment effect.

## Immutable contract

- source: exact post-push readback of `yuxzhang/canon-zero-tim`;
- image: registry digest, never a mutable tag;
- model/data: Qwen3-4B-Instruct-2507 and the frozen 1,012-task clean list;
- topology: one 128-chip `4x4x8` slice split into rollout DP8 x TP8 and trainer
  DP8 x TP8;
- Kueue affinity: worker sentinel `tpu-v5p-slice` delegates the concrete pool
  to ResourceFlavor and must not appear as a literal node-pool selector;
- recipe: B8 x G16, response 16,384, 50 turns, RLOO, fixed-context
  `sequence-mean-token-scale`, TPU-resident optimizer, optional interventions
  off, prefix cache off;
- stage/run: `full`, fresh run-id `p58f05`, exactly 1,000 optimizer commits;
- arm: `native` only. Rendering or applying `zero` is outside this phase.

## Admission gate

The rendered worker must retain `google.com/tpu: 128`, TPU accelerator
`tpu-v5p-slice`, and exact topology `4x4x8`, while omitting literal
`cloud.google.com/gke-nodepool: tpu-v5p-slice`. Server-side dry-run must pass,
then Kueue must report `QuotaReserved=True` before runtime diagnosis begins.
Failure here is admission `INCONCLUSIVE`, not training evidence.

## Online monitoring

- admission: quota reservation, concrete flavor, 32 four-chip worker pods, Pathways
  device count 128;
- first completed batch: 128 journal rows, sandbox-start/environment/model
  timeout split, cleanup receipts, solve/signal group metrics;
- commits 1–3: finite forward/backward, finite nonzero native A-B, exact B-C,
  device-resident optimizer, monotonic transaction and journal state;
- later milestones: checkpoint 8, then updates 32, 100, and every 100 updates;
- evaluation remains at the signed recipe cadence; checkpoint cadence remains
  every 8 committed updates.

A healthy job continues after update 3. Stop only for a signed hard failure,
not because the former canary horizon has been crossed.

## Exit gate

The native full classifier must report `PASS` from complete, digest-verified
artifacts and exactly 1,000 commits. It must prove finite nonzero A-B treatment
dose, exact B-C, finite training values, TPU-resident optimizer state,
complete 128-row trajectory batches, journal continuity, sandbox cleanup,
evaluation/checkpoint cadence, and transaction integrity.

An all-filtered batch may advance `batch_index` without an optimizer commit;
it must preserve unchanged optimizer state and may make the number of consumed
batches exceed 1,000. Partial/tampered evidence, exact native A-B
(`NO_TREATMENT`), or any B-C drift cannot be promoted.

## Attempt boundary

P58c05 and p58f01 through p58f04 are immutable `INCONCLUSIVE` evidence.
P58f01 exposed sandbox LocalQueue and reset-time provenance faults. P58f02
exposed a CPU-flavor/node-pool mismatch; moving the head and sandboxes to
`cpu-np` was the correct repair. P58f03 then completed 128 real trajectories
in 616.3 seconds and durably journaled them, proving that rollout and sandbox
throughput are no longer the first failure. It stopped before trainer forward,
backward, or update because native was routed to a canonical-adapter-only
weight gate.

The repaired weight gate uses an exact read-only live-weight observer for signed P58
native and keeps the canonical registered-adapter path for zero. Native still
has no numerical hook. Any mismatch, invalid mesh, missing signature, or
leaked adapter remains fatal. P58f03 has no optimizer checkpoint and is not a
resumable training root; preserve its trajectory journal as diagnostic
evidence only.

P58f04 completed 128 real trajectories in 557.2 seconds, durably journaled
them, and passed exact live-weight attestation for 398 leaves and
4,022,468,096 elements. It then failed before trainer forward/backward/update
because processed `S_prefill` was wired only to the canonical processed engine,
which native correctly disables. The repair adds an independent, observer-only
stock B overlay gated solely by the signed P58 native tuple. Native retains
`CANON_PROMPT_PROCESSED_LOGPROBS=0`, `CANON_ENGINE_MODULE_C=0`, and every other
zero-TIM disable/absence. Zero retains the canonical engine and sets the stock
observer to zero. The next attempt is fresh native `p58f05` after publication
and readback. Zero remains deferred.
