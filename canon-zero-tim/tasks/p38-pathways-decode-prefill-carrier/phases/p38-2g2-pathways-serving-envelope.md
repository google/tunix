# P38.2g2: source-pinned Pathways serving envelope

Status: local admission hardening complete and gated; uncommitted; target run
not authorized or run.

## Objective

Measure the FrozenLake decode carrier in the executable that actually produced
it. The earlier DP1xTP4 replay reconstructed a plausible schedule but did not
reproduce captured decode, so no local KV counterfactual is admissible.

This phase adds two default-off mechanisms:

1. a bounded production decode capture around the real `continue_decode`
   program; and
2. one combined all-cache-read counterfactual (`U`) using the historical
   two-pass operation.

Neither mechanism changes stock behavior when its environment variable is
unset. No backward or optimizer commit is part of this phase.

## Pinned source audit

Image tag: `tunix_frozenlake_image:vllm-tpu0.25.0`

Local image ID:
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.

Archived sources and SHA-256 values:

- `artifacts/rpa_v3_kernel_pinned_0811.py`:
  `48287ee07d52e3792aaa570420f38c0abfd096013c4727eaacfb4c111207a7ba`;
- `artifacts/pinned_decode_loop.py`:
  `b5602ec75073e9740e3661f68578baaedf41bc1047fafea2b79f041856cc10f5`;
- `artifacts/pinned_ragged_kv_cache_update.py`:
  `e889841c5ee91c8589f1a47a723aa0daf555a3a1692a153a8f43fb6da9849b0c`;
- `artifacts/pinned_attention_metadata.py`:
  `39cf16b5197336ae365fdfebba521664d2f2a5d5655cde25a402bb2814e0754b`.

The audit establishes these facts:

- production decode can bypass the ordinary runner call and execute multiple
  steps inside the donated-cache `continue_decode` `lax.while_loop`;
- RPA v3 with `update_kv_cache=True` dynamically joins cached and fresh K/V
  and performs an asynchronous fused cache write;
- `update_kv_cache=False` simultaneously skips that write and reads all K/V
  from cache, assuming another operation has already populated it;
- the public RPA v3 API does not expose a write-only mode;
- the v2 `kv_cache_update` writer is available, but using it changes the write
  kernel as well as the read source.

Therefore a clean write-only `W` arm cannot be constructed from the current
public v3 API. It must not be reported as a single-variable experiment. The
first counterfactual is the combined historical operation `U`: run stock once
to populate cache, discard its attention output, then run the same attention
with `update_kv_cache=False` and use the second output. Phase 13 proved that
this operation executed but had no effect in the short-context four-chip
domain; it is a new-domain causal arm, not a previously proven repair.

## Capture contract

The capture must include the real continue-decode call, not only prompt
logprob calls. For every admitted record it writes before dispatch:

- attempt-unique call ordinal and request IDs by DP rank;
- scheduler token counts, live slot/index mapping, and co-batch membership;
- input IDs, input positions, active mask, block tables, sequence lengths,
  query starts, and request distribution;
- logical request lengths and exact physical page IDs;
- cache shape, dtype, sharding, page size, D/P/M block tuples, and source
  fingerprint;
- maximum requested and static decode-loop steps.

After dispatch it adds actual steps, generated token IDs, returned logprob
buffers, and final positions/sequence lengths. The pre-dispatch record must
survive a backend disconnect. Capture is bounded, collision-failing, and
disabled by default. Missing expected decode records make classification
inconclusive.

The request record also preserves each live request's token history and the
physical-to-logical token selector. This permits an exact token-history join
to a recovered mismatch capsule; row ordinal alone is not an admitted join
key.

Only requests with a positive scheduled-token count are captured. Their
original scheduler slot is preserved even when an idle live request is
filtered, so filtering cannot silently compact the row mapping. Every selected
request must be a one-token continue-decode request and must agree across the
request list, DP assignment, global token row, attention row, selector range,
sequence length, query-start range, and physical block-table row.

## Local implementation gate

- patches 08 and 09 install from the pinned image for Qwen3-1.7B and Qwen3-8B;
- each installed overlay matches its 29-file SHA manifest;
- both installed logprob-chunking suites pass 13/13, including scheduled-only
  selection, preserved physical-slot mapping, empty-selection rejection, and
  selector-drift rejection;
- the serving classifier passes 18 tests, including missing-post,
  missing-page, missing-sampling-metadata, corrupt-SHA, count, internal mapping,
  required-capsule, missing-join, and ambiguous-join controls;
- the dedicated renderer passes five tests, including stock/U separation,
  zero retries, poison-SHA quoting, missing-capsule rejection, and overwrite
  refusal;
- the archive transport extractor passes four tests; and
- the shell postflight accepts exactly one complete precheck stop, rejects a
  numerical red without that marker, rejects a stock U hit, rejects U without
  a hit, and accepts exact U only with a positive hit; and
- the complete pinned-image P33 CPU gate passes.

These are construction gates only. They do not show that production decode
was captured or that `U` changes the FrozenLake boundary.

## Render and target order

After the source is explicitly committed and pushed, render both diagnostic
manifests from that exact commit:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38cap1"
OUT="/tmp/p38-serving-$RUN_ID"
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
```

Server-side dry-run both, but apply only stock first. Do not apply the whole
directory and do not run both arms concurrently:

```bash
kubectl apply --dry-run=server -f "$OUT/jobset-p38-serving-stock.yaml"
kubectl apply --dry-run=server -f "$OUT/jobset-p38-serving-unified.yaml"
kubectl apply -f "$OUT/jobset-p38-serving-stock.yaml"
```

The stock run is expected to stop at the known pre-backward A/B gate. Archive
the complete head log, then recover the collision-protected serving tar:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py \
  --log /path/to/stock.raw.log \
  --output /path/to/stock-serving.tar
```

Only after the stock run contains one PASS serving-capture classification,
one verified archive, the expected hard A/B red, Attempt 0, and zero optimizer
commits may the operator apply the unified manifest. The unified result is a
combined mechanism test; it is not a writer-only verdict.

Both manifests set `CANON_P38_PRECHECK_ONLY=1`. If U makes the pre-backward
record exact, the learner emits exactly one
`[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD` marker and terminates
before backward. The wrapper accepts that expected exit only when the serving
capture also classified PASS. A stock hard red remains a nonzero diagnostic
outcome and is never converted to success.

## Target ladder

1. Run stock capture only. It must reproduce the known decode output for the
   selected request and preserve all required fields.
2. Run `U` as a separate default-off source-pinned diagnostic. Prefix cache,
   precision, fixed-M, block sizes, weights, prompts, and all other controls
   remain unchanged.
3. If `U` makes the forward boundary exact, rerun GSM8K A/B and the VJP2
   chain/oracle gates, then run FrozenLake backward-no-commit.
4. Only an exact forward plus healthy/correct gradient and zero optimizer
   commits may advance to FrozenLake full training.

The first target run is diagnostic and uses Attempt 0. Operational JobSet
restarts are not allowed for this numerical classification.

## Pre-registered verdict

- Stock capture absent or not joinable to the mismatch row: `INCONCLUSIVE`.
- Stock reproduces and `U` remains red: cache/fresh unification is falsified
  for this carrier; inspect page-gather ordering and the first divergent layer.
- Stock reproduces and `U` becomes exact: the combined cache-write/read
  mechanism is causal in the target domain. This does not distinguish the
  writer from the read-source change.
- Any source drift, retry, cache collision, missing count, B/C regression, or
  infrastructure disconnect voids downstream numerical interpretation.

## Rollback

Leave the new P38 capture and KV-unified variables unset. Remove only the new
default-off patch and its manifest/test entries; stock attention, precision,
training loss, and optimizer behavior are otherwise unchanged.
