# Phase P35.3c — First-record stage localization

Status: locally complete; not committed or published; no target run

## Objective

Localize the 64-chip r30 IFRT disconnection inside the first captured/live replay record without
changing the model, sampling or canonical logprob numerical callables. This is an infrastructure
stage probe. It cannot produce a B/C numerical verdict.

## Corrected evidence boundary

- r30 wrote the preliminary A/B/C report and entered `R0_live_first` record 1 of 2.
- The record emitted no completion marker before the IFRT client observed `Socket closed`.
- The canonical target-logprob `compute_and_gather` callable was already `jax.jit`; the Python
  traceback does not establish that an unjitted scorer caused the peer exit.
- The complete logical logits shape is `(4096, 151936)` float32, but the inspected replay contains
  no complete-logits host conversion. A 2 GiB host-transfer limit is not the r30 explanation.
- The client log lacks the proxy, resource-manager, worker and node evidence required to classify
  OOM, transport, service restart or a compiler/runtime failure.

## Admitted implementation

`CANON_P35_REPLAY_STAGE_PROBE=1` is valid only with the existing P35 envelope and exact-replay
gates. It executes only `R0_live_first`, record 1, and inserts readiness boundaries after:

1. model output and updated caches;
2. final full-vocabulary logits;
3. processed sampling logits;
4. the existing jitted canonical target-logprob callable;
5. raw and processed target gathers;
6. compact record outputs.

The first stage creates and fsyncs an empty report before waiting, so a failure in the first
readiness barrier remains classifiable. Each stage prints one `STAGE_BEGIN`, waits for device
completion, appends one fsynced JSONL `ready` event, then prints one `STAGE_READY`. Six ready
stages produce exactly one
`STAGE_PROBE_COMPLETE ... NO_NUMERICAL_VERDICT` marker and a deliberate diagnostic exit before
the second record, repeat arms, backward or optimizer.

The stage probe does not wrap `model_fn`, `compute_logits_fn`, `_sample` or the canonical scorer in
a new outer JIT. Readiness barriers alter asynchronous scheduling and are therefore diagnostic,
not evidence that the original unbarriered replay is fixed.

## Fail-closed gates

- The renderer must explicitly receive `--stage-probe`; ordinary P35 rendering writes
  `CANON_P35_REPLAY_STAGE_PROBE=0`.
- The cluster preflight requires unique stage report and classification paths.
- Postflight accepts only diagnostic exit 1, one preliminary base marker, zero final numerical
  report markers, six ordered begin/ready pairs and one non-numerical completion marker.
- The JSONL classifier records `last_ready_stage` and `first_missing_stage`. It requires exactly
  the six ordered stages for `R0_live_first`, record 1 to return `COMPLETE`; a missing, duplicate,
  reordered, record-count drift or second-record event remains persisted as `INCONCLUSIVE`.
- A successful stage probe preserves and classifies the preliminary A/B/C report, but its own
  classification contains `numerical_verdict: false`.

## Decision table

| Last ready stage | First missing ready stage | Next discriminator |
|---|---|---|
| none | model | model/cache depth or worker-side model execution |
| model | logits | final norm/lm-head/full-logits materialization |
| logits | sample | production sampling transform |
| sample | logprobs | existing canonical processed-logprob callable |
| logprobs | target_gathers | raw/processed target gathers |
| target_gathers | record_outputs | compact scatter/output assembly |
| record_outputs | none | async liveness or dispatch accumulation in the unbarriered replay |

If the canonical-logprob or target-gather stage is isolated, a shared compact observer may be
tested next. It must reuse the production canonical scorer, derive the implied normalizer as
`processed_target - logprob`, and pass a standalone-versus-observer bitwise gate. A new
`jax.nn.logsumexp` reduction is not admitted as equivalent evidence.

## Target evidence to archive

- coordinator run log and preliminary A/B/C JSON;
- stage JSONL and its classification JSON;
- Pathways proxy, resource-manager and every worker log;
- pod status, termination reason, node events and available memory/HBM telemetry;
- rendered JobSet and source SHA.

## Exit gate

Local completion requires focused adapter tests, stage-classifier negative controls, renderer and
cluster-postflight controls, the complete CPU contract, exact-image overlays and one real-device
TP4 mechanics gate. Target completion requires one source-pinned Attempt 0 whose last ready stage
is mechanically classified. It creates no numerical P35.3 verdict.

Local result: PASS on 2026-08-10 UTC. Focused classifier/renderer tests pass 12/12, the focused
adapter stage test passes, the complete P33/P35 CPU contract passes, both exact-image overlays
pass, and a real four-device v5p TP4 test materializes the production-shape local logits array
`(256, 151936)` and completes the six-stage mechanics. That TP4 test uses a synthetic forward; it
does not exercise Qwen, Pathways or the target failure. Evidence and reproducible commands are in
`../artifacts/p35_3c_local_gate.md`.

## Rollback

Leave `CANON_P35_REPLAY_STAGE_PROBE` unset or set it to `0`. The stage barriers, append-only
evidence and diagnostic stop are then unreachable. The existing P35.3 numerical replay and all
production serving, training, backward, optimizer, precision and sampling paths remain unchanged.
