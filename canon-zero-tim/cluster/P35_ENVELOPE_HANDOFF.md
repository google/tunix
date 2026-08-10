# P35 Pathways exact-envelope handoff

Updated: 2026-08-10 UTC

The bounded-r30 implementation ran from `78bde02f059d4984eb4fd2ac7079668b94fee980` and remains
infrastructure-inconclusive. P35.3c first-record stage localization is locally complete at reviewed
source pin `7484ab7844ca79fda6399f6f6dcd475ef8c6d632`; it has no target run yet.

## Current target result

P35.2 is complete on one source-pinned 64-chip DP16xTP4 Pathways attempt (r28):

- A, native serving with dynamic packing, versus B, grouped native serving: 0/3,244
  selected action elements and 0/12,976 bytes differ.
- B versus C, the canonical adapter: 1,529/3,244 selected action elements and
  3,106/12,976 bytes differ.
- Direct A versus C has the same red counts, so the three-arm probe reproduced the production
  boundary and its negative control fired.
- All 310 mapped/live leaves were bitwise equal, but the comparison normalized
  `pinned_host->device`. Equal bits do not remove memory placement as a compiled-program
  variable.

The valid conclusion is `adapter_envelope_carrier`. It excludes dynamic serving packing for the
selected group. It does not yet distinguish weight memory placement, physical metadata/cache
construction, or the adapter's outer `lax.map` program.

P35.3 r29 and r30 are infrastructure-inconclusive. r30 completed rollout, wrote the preliminary
A/B/C report, and entered the first captured/live replay record. The IFRT socket closed before
that record emitted a completion marker. The log proves two captured B records and a logical
float32 logits shape of `(4096, 151936)` per record. It does not prove OOM, host transfer, an
unjitted scorer, transport saturation or worker eviction.

P35.3b preserves the original numerical program boundaries and serializes every captured record.
It writes `p35_envelope.pre_replay.json` before optional replay, prints the record count and
logical logits size, blocks each record before submitting the next, and releases full-vocabulary
temporaries at that boundary. A fused-tail candidate was rejected because it changed 178/256 CPU
target logprobs by about one ULP.

P35.3c is the next diagnostic, not another full replay. It preserves all numerical callables and
waits after model, logits, sampling, the already-jitted canonical scorer, target gathers and
compact output assembly. It stops after `R0_live_first` record 1 with
`NO_NUMERICAL_VERDICT`. A successful stage probe classifies infrastructure progress only.

Local gates passed on CPU, both exact-image overlays and a real four-device v5p TP4
production-shape mechanics test. The TP4 test used a synthetic forward at local M256 and
vocabulary 151936; it did not reproduce Qwen or Pathways. Do not run the commands below unless the
target branch resolves to the reviewed source pin and the 64-chip attempt is separately approved.

## P35.3 arms

The next run keeps A/B/C and adds four in-process replay arms before backward:

1. R0: exact captured B IDs, positions, attention metadata and page tables; fresh cache; live
   serving leaves; direct canonical model entry.
2. R1: exactly the R0 tensors and direct entry; bitwise-equal trainer-mapped leaves.
3. R2: trainer-mapped leaves and direct entry; adapter-generated metadata and fresh cache.
4. R3: the unchanged production adapter envelope replayed on the complete original batch.
5. C: the original production adapter value already measured in the batch.

This gives a single-variable chain:

| Boundary | Variable isolated |
|---|---|
| B vs R0 | Whether the captured direct replay reproduces the serving result |
| R0 vs R1 | Live-serving versus trainer-mapped leaf placement, with equal bits |
| R1 vs R2 | Captured serving metadata/cache contract versus adapter construction |
| R2 vs R3 | Direct adapter group versus the production outer `lax.map` envelope |
| R3 vs C | Repeat/production anchor |

B-vs-R0 and R3-vs-C are hard anchors. Either red makes the measurement `INCONCLUSIVE`; it is
not reclassified as a carrier. B-vs-C must remain red. R0, R1 and R2 are repeated, and every
repeat must be bitwise exact.

## Full numerical-replay contract

The run must satisfy all of the following before its numerical classification is usable:

- Attempt 0, source-pinned SHA, DP16xTP4, canonical local M256 and response 256;
- all P35.2 weight, token, mask, device-order and metadata attestations;
- exactly one preliminary P35.2 marker, one final P35.2 marker and one P35.3 replay marker;
- effective injected-drift negative controls;
- immutable report and classification paths;
- diagnostic exit 1 before backward, converted to success only after both classifiers return
  `COMPLETE`;
- no training commit, optimizer update, W&B mutation, precision change or sampling change.

The postflight prints SHA-256 for all five JSON artifacts. If replay fails after the preliminary
marker, it still prints the preliminary report SHA before failing closed. The detailed files
still live under `CANON_STATE`, which is `/tmp` on the coordinator host. Copy them before
deleting the Pod; the log SHA is provenance, not a replacement for the files.

## r31 stage-probe operator commands

Run these only after the reviewed implementation commit is published to
`yuxzhang/canon-zero-tim`. The stage-probe renderer flag is mandatory:

```bash
git fetch origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse origin/yuxzhang/canon-zero-tim)"
RUN_ID=r31
OUT="/tmp/canon-p35-gsm8k-${RUN_ID}.yaml"

python3 canon-zero-tim/cluster/render_p35_jobset.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --stage-probe \
  --output "$OUT"
kubectl apply --dry-run=server -f "$OUT"
```

After the dry run passes, confirm the concrete SHA and apply that one rendered file. Do not edit
the YAML between dry run and apply.

```bash
kubectl apply -f "$OUT"
JOBSET="canon-p35-gsm8k-env-${RUN_ID}-${SOURCE_SHA:0:8}"
kubectl logs -f "jobs/${JOBSET}-pathways-head-0"
```

Before deleting the JobSet, resolve the coordinator Pod and copy client and service evidence. A
missing optional file is itself evidence; list the state directory before copying:

```bash
POD="$(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${JOBSET}" \
  -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head \
  -o jsonpath='{.items[0].metadata.name}')"
STATE="/tmp/canon-state/${JOBSET}"
DEST="canon-zero-tim/debug_logs/p35_${RUN_ID}"
mkdir -p "$DEST"
kubectl logs "$POD" -c jax-tpu > "$DEST/head_jax_tpu.raw.log"
kubectl logs "$POD" -c pathways-proxy > "$DEST/pathways_proxy.raw.log"
kubectl logs "$POD" -c pathways-rm > "$DEST/pathways_rm.raw.log"
kubectl exec "$POD" -c jax-tpu -- find "$STATE" -maxdepth 2 -type f -print \
  > "$DEST/state_inventory.txt"
kubectl cp "$POD:$STATE/p35_envelope.pre_replay.json" \
  "$DEST/p35_envelope.pre_replay.json"
kubectl cp "$POD:$STATE/p35_envelope.classification.json" \
  "$DEST/p35_envelope.classification.json"
kubectl cp "$POD:$STATE/p35_replay_stages.jsonl" \
  "$DEST/p35_replay_stages.jsonl"
kubectl cp "$POD:$STATE/p35_replay_stages.classification.json" \
  "$DEST/p35_replay_stages.classification.json"
kubectl cp "$POD:$STATE/p35_metadata" "$DEST/p35_metadata"
kubectl get pods -o wide > "$DEST/pods_wide.txt"
kubectl get events --sort-by=.lastTimestamp -o yaml > "$DEST/events.yaml"
while read -r worker; do
  worker_name="${worker#pod/}"
  kubectl logs "$worker" -c pathways-worker \
    > "$DEST/${worker_name}.raw.log" 2>&1
  kubectl get "$worker" -o yaml > "$DEST/${worker_name}.pod.yaml"
done < <(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${JOBSET}" \
  -l jobset.sigs.k8s.io/replicatedjob-name=pathways-worker \
  -o name)
find "$DEST" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$DEST/SHA256SUMS"
```

If the label resolves zero or multiple coordinator Pods, stop and resolve it explicitly; do not
guess. Preserve every red or inconclusive artifact.

## Reading the r31 stage result

Only `p35_replay_stages.classification.json` decides whether the probe completed mechanically.
`measurement_verdict=COMPLETE` must be accompanied by `numerical_verdict=false` and exactly six
ordered stages. If the probe fails, the final `STAGE_BEGIN` without its matching `STAGE_READY`
identifies the first incomplete stage. Use the decision table in
`tasks/p35-envelope-discriminator/phases/p35-3c-first-record-stage-probe.md`.

Do not infer a numerical carrier from r31 and do not accept a successful JobSet as B=C. The full
P35.3 replay remains deferred until the infrastructure stage is localized.

## Reading a later full replay

Only the P35.3 classification JSON decides the branch:

- `weight_memory_placement_carrier`: R0/R1 is the only red internal boundary.
- `metadata_cache_construction_carrier`: R1/R2 is the only red internal boundary.
- `adapter_outer_program_carrier`: R2/R3 is the only red internal boundary.
- `mixed_exact_replay_carriers`: more than one internal boundary is red.
- `INCONCLUSIVE`: an anchor, attestation, repeat, negative control or transitivity check failed.

P35.3 localizes the final logprob boundary and compact captured-input stages. It does not by
itself prove actual-model THIRDPROG or gradient correctness. Those remain later gates.

## Rollback

Leave `CANON_P35_ENVELOPE` and `CANON_P35_EXACT_REPLAY` unset. The diagnostic methods are then
unreachable. Do not change precision, canonical M, DP/TP geometry, sampling, loss, fixed
reductions, VJP, optimizer semantics, W&B or Hugging Face configuration.
