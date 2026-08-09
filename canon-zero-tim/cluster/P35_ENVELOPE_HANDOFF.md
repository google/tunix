# P35 Pathways exact-envelope handoff

Updated: 2026-08-09 UTC

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

## Fail-closed contract

The run must satisfy all of the following before its numerical classification is usable:

- Attempt 0, source-pinned SHA, DP16xTP4, canonical local M256 and response 256;
- all P35.2 weight, token, mask, device-order and metadata attestations;
- exactly one P35.2 report marker and one P35.3 replay marker;
- effective injected-drift negative controls;
- immutable report and classification paths;
- diagnostic exit 1 before backward, converted to success only after both classifiers return
  `COMPLETE`;
- no training commit, optimizer update, W&B mutation, precision change or sampling change.

The postflight prints SHA-256 for all four JSON artifacts. The detailed files still live under
`CANON_STATE`, which is `/tmp` on the coordinator host. Copy them before deleting the Pod; the
log SHA is provenance, not a replacement for the files.

## Operator commands

Run these only after the reviewed implementation commit is published to
`yuxzhang/canon-zero-tim`. Use the next unused run id (`r29` is reserved by this handoff):

```bash
git fetch origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse origin/yuxzhang/canon-zero-tim)"
RUN_ID=r29
OUT="/tmp/canon-p35-gsm8k-${RUN_ID}.yaml"

python3 canon-zero-tim/cluster/render_p35_jobset.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
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

Before deleting the JobSet, resolve the coordinator Pod and copy the evidence:

```bash
POD="$(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=${JOBSET}" \
  -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head \
  -o jsonpath='{.items[0].metadata.name}')"
STATE="/tmp/canon-state/${JOBSET}"
DEST="canon-zero-tim/debug_logs/p35_${RUN_ID}"
mkdir -p "$DEST"
kubectl logs "$POD" > "$DEST/head_jax_tpu.raw.log"
kubectl cp "$POD:$STATE/p35_envelope.json" "$DEST/p35_envelope.json"
kubectl cp "$POD:$STATE/p35_envelope.classification.json" \
  "$DEST/p35_envelope.classification.json"
kubectl cp "$POD:$STATE/p35_exact_replay.json" "$DEST/p35_exact_replay.json"
kubectl cp "$POD:$STATE/p35_exact_replay.classification.json" \
  "$DEST/p35_exact_replay.classification.json"
kubectl cp "$POD:$STATE/p35_metadata" "$DEST/p35_metadata"
find "$DEST" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$DEST/SHA256SUMS"
```

If the label resolves zero or multiple coordinator Pods, stop and resolve it explicitly; do not
guess. Preserve every red or inconclusive artifact.

## Reading the result

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
