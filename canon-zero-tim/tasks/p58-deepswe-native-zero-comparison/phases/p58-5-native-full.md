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
- stage/run: `full`, fresh run-id `p58f01`, exactly 1,000 optimizer commits;
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

P58c05 is immutable Kueue-admission evidence and has no resumable journal or
checkpoint. Do not reuse its YAML or root. The next attempt is fresh native
`p58f01` from the published renderer repair. Zero remains deferred.
