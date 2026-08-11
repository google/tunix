# P38 target handoff

Scope: this handoff covers only the GSM8K/FrozenLake P38 workstream. For the
parallel Qwen3-32B DeepSWE workstream, read
`../p39-deepswe-production/HANDOFF.md`. P38 evidence cannot promote P39, and
P39 evidence cannot promote P38.

## Purpose

P38.2 separates two observed flag-on `S_decode_vs_S_prefill` signatures. GSM8K is a tail-aval
candidate; FrozenLake contains a `0.10390` maximum difference and requires upstream/multi-turn
localization. The original strict probes are pre-backward diagnostics. The
later P38.2d amendment separately admits GSM8K full training under a bounded
A/B reporting policy; FrozenLake remains strict and no-commit. A P38.2d GSM8K
run is `alignment-degraded`, not a zero-TIM completion claim.

## Proven locally

- The alignment test suite passes 26/26 in `tunix_frozenlake_image:vllm-tpu0.25.0`.
- The complete P33 CPU gate passes, including a deliberately failed workload whose pre-alignment
  JSON and SHA survive in stdout.
- The existing hard gate is unchanged: any nonzero pre-backward boundary still exits nonzero.
- A signed GSM8K DP1xTP4 direct-attached run observed 11,340 action tokens with
  `S_decode_vs_S_prefill=0/45360 bytes` and
  `S_prefill_vs_T_old=0/45360 bytes`; the classifier verdict is
  `LOCAL_NOT_REPRODUCED`.
- A production-shape canonical-tail control compared 38,895,616 f32 elements
  across two outer JIT programs with zero differences and detected an injected
  one-bit negative control.
- A model-free DP1xTP4 aval probe ran the live sampling transform at M16/M256
  and the live canonical scorer at M256/M256. Its transform HLO digests were
  different but all five numerical comparisons were exact. This is
  `MODEL_FREE_NOT_REPRODUCED`, not a target fix.
- A synthetic multi-turn mismatch now records turn index, action-run offset,
  completion and sequence chunk coordinates, logical KV prefix length, and
  distance to the next M256 boundary. The complete CPU gate passed.
- A real zero-LR Adam commit with 16 active gradient microbatches advances
  optimizer state, keeps all parameter elements unchanged, and passes the new
  schedule-aware transaction gate. A positive constant-LR control reports
  nonzero post-rounding parameter changes.
- A blocking pre-backward mismatch can persist at most two replay rows to a
  hash-attested NPZ. The failed-run wrapper emits base64 to stdout, and
  `scripts/extract_p38_capsule.py` rejects corrupt transport or array hashes.

## Not proven

- P38d5 ran on 64 target chips, but the new schedule-aware commit evidence and
  mismatch capsule were implemented afterward and remain untested there.
- The FrozenLake A-B carrier has not been localized to a specific operator,
  page layout, or attention tile. Its observed onset is a coordinate, not a
  causal repair.
- GSM8K P38d5 measured `T_old_vs_T_current=0` across all 16 microbatches.
  FrozenLake still stops pre-backward, so its actual-model third-program
  boundary remains unmeasured.
- The new schedule-aware commit evidence and mismatch capsule have not run on
  DP16xTP4 target hardware.

The one-host result is a construction gate, not evidence that r35 was repaired.
Its immutable local artifacts are:

- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2/pre_alignment.jsonl`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`

## Source and render the model-free target probe

Run only after the P38 patch has been reviewed, committed, and pushed with explicit approval.
Use a clean `yuxzhang/canon-zero-tim` worktree and replace `p38a0` if that run id already exists.

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38a0"
TARGET="/tmp/jobset-p38-aval-$RUN_ID.yaml"
python3 canon-zero-tim/cluster/render_p38_aval_jobset.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output "$TARGET"
kubectl apply --dry-run=server -f "$TARGET"
```

The dry run must pass before resource allocation. Do not apply the rendered directory and do not
queue `gsm8k-full` or `frozenlake-full` in this phase.

## Stage 1 target run: model-free aval discriminator

The external operator may apply exactly one manifest after confirming resource
approval. It uses no model, workload, backward, optimizer, checkpoint, or W&B:

```bash
kubectl apply -f "$TARGET"
```

Require Attempt 0, zero restarts, the source commit printed by the renderer, and the proxy
`XLA_FLAGS=--xla_allow_excess_precision=false` contract. A failed numerical gate is an expected
diagnostic outcome; do not restart it automatically.

Return the complete head-pod stdout plus the durable
`CANON_P38_AVAL_REPORT`. The report must contain five completed comparisons,
the registered DP16xTP4 shape table (transform M16/M4096, score M256/M4096),
sharding specs, HLO digests, and a one-element negative control. Missing fields
make the run inconclusive. A fully exact model-free result does not prove the
production boundary; it advances to Stage 2.

## Stage 2 target runs: both production boundaries

After Stage 1 is classified, render the existing no-commit production probes
from the same source commit. Do not substitute one workload for the other:

```bash
RUN_ID="p38prod0"
OUT="/tmp/p38-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

GSM="$OUT/jobset-p33-gsm8k-alignment-short.yaml"
FL="$OUT/jobset-p33-frozenlake-alignment-short.yaml"
kubectl apply --dry-run=server -f "$GSM"
kubectl apply --dry-run=server -f "$FL"
```

With separate resource approval, apply only `GSM` and `FL`. Both stop before
backward and optimizer commit. GSM8K tests the low-amplitude tail candidate.
FrozenLake independently tests the `0.10390` multi-turn signature and must
emit turn, action-run, M256 chunk, and logical-KV coordinates for every
reported mismatch.

## Evidence to return

Archive the complete head-pod stdout and report its SHA. The raw log must contain:

- `[CANON_ALIGN_PRE_JSON]` with both boundary records;
- `[CANON_ALIGN_PRE_EVIDENCE]` with the report SHA;
- on failure, `[CANON_PRE_ALIGN_ARTIFACT]` and every
  `[CANON_PRE_ALIGN_ARTIFACT_JSON]` row;
- the exact source commit, Attempt 0 marker, proxy XLA environment, mesh order, local canonical
  row count, `N_action`, and workload exit code.

For every mismatch, preserve coordinate, token id, exact A/B bits, XOR, byte offsets, ULP
distance, and absolute delta. The report is inconclusive if a target line is missing; absence is
not equality.

## Pre-registered verdict

- A-B nonzero and B-C zero: P38.2b reproduces the GSM8K serving carrier; classify the transform,
  score, and implied-normalizer fields before selecting a repair. FrozenLake is still required.
- A-B zero and B-C zero: GSM8K did not reproduce the sparse r35 carrier. This is not proof of a
  fix; P38.2c FrozenLake remains independently required.
- B-C nonzero, an invalid shape, missing evidence, source drift, a retry, or an infrastructure
  disconnect: the numerical result is not admitted.

FrozenLake evidence must additionally identify the turn index, assistant-run offset, canonical
chunk index, logical KV prefix length, and whether the mismatch is adjacent to a turn or M256
boundary. A tail-only repair is not admitted for a `0.10390` upstream signature.

No tolerance, report-only committing mode, old-logprob recomputation, precision change, or
optimizer commit is authorized by this handoff.

The paragraph above records the original strict handoff. It is superseded only
by the user-approved P38.2d amendment in
`phases/p38-2d-gsm8k-bounded-ab-campaign.md`: committed report-only behavior is
limited to bounded `S_decode_vs_S_prefill` drift in GSM8K full. It does not
apply to FrozenLake, B/C, old/current, gradients, DP reduction, or optimizer
integrity, and it disables a zero-TIM completion claim for that campaign. Old
logprob recomputation and precision changes remain forbidden.

## P38.2d operator handoff

After pulling the source commit, render the P33 queue with a fresh run id. The
renderer must show `CANON_GSM8K_AB_REPORT_ONLY=1` only in `gsm8k-full`; every
other YAML must show `0`. Apply only the FrozenLake backward-no-commit and
GSM8K full manifests. Do not apply FrozenLake full.

The GSM8K classifier may exit successfully as
`PASS_WITH_AB_REPORT_POLICY`. This means the run completed under a downgraded
admission policy, not that A=B=C was proven. Archive the raw log and all
alignment/update JSONL before deleting either JobSet.

The refreshed FrozenLake backward-no-commit manifest must contain
`CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=2` and a run-isolated
`CANON_P38_MISMATCH_CAPSULE` path. On the expected hard red, archive all
`[CANON_P38_CAPSULE_ARTIFACT]` and `[CANON_P38_CAPSULE_B64]` lines. Recover the
file without editing the raw log:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py \
  --log /path/to/frozenlake-head.raw.log \
  --output /path/to/p38-frozenlake-capsule.npz
```

Do not rerun FrozenLake before the capsule has passed transport and embedded
array SHA checks. Prefix cache remains disabled. The next action is the
single-row prefix sweep specified in
`phases/p38-2f-frozenlake-threshold-capsule.md`, not a full training launch.

## Current FrozenLake execution order

The published source is `e9cfe298`. The first refreshed
`frozenlake-backward-no-commit` run is expected to stop at the pre-backward A-B
gate. Treat it as P38.2f capsule capture; do not claim that backward ran merely
because of the manifest name. Do not add KV-unified or another numerical arm
to this first target run.

After capsule recovery, follow
`phases/p38-2g-frozenlake-causal-replay.md`: R0 stock multi-turn reproduction,
R1 same-depth single-turn control, R2 MIXED-only two-pass cache write/read, and
R3 all-distribution two-pass. Each arm requires an independent cache seed.
Only a selected candidate with exact forward boundaries may advance to a new
target backward-no-commit run. That later run must prove gradient and DP
reducer health while committing neither parameters nor optimizer state.

GSM8K full may run in parallel on independent resources. Its schedule-aware
transaction result is independent evidence and cannot promote FrozenLake.

## P38.2g implementation ready in the current worktree

The current uncommitted worktree on top of `e9cfe298` contains the locally
admitted R0/R1 replay implementation. It has not been published and has not
run on a real Qwen3-8B TPU model. Do not tell the target operator to pull it
until the user explicitly approves commit and push.

After P38.2f produces a capsule, recover it with
`scripts/extract_p38_capsule.py`, copy the verified NPZ to the authorized
DP1xTP4 host, and run:

```bash
scripts/run_p38_frozenlake_replay.sh \
  /absolute/path/to/recovered-p38-capsule.npz <unique-label>
```

The runner verifies the capsule before model initialization, loads Qwen3-8B,
executes R0/R1/reference twice with fresh caches, runs a one-bit negative
control, requires bitwise-equal actor/engine leaves, writes a bounded report,
and exits before the agentic learner/backward with zero optimizer commits.
R0 is `mask-derived-v1`; exact serving scheduler metadata was not captured.

Expected classification is one of:

- `MULTITURN_SCHEDULE_CARRIER_CANDIDATE`: R0 is red and R1 is exact against
  the fixed-chunk reference;
- `LOCAL_CARRIER_NOT_REPRODUCED`: R0 is exact locally;
- `LOCAL_CARRIER_NOT_ISOLATED`: R0 and R1 are both red.

All three are measurement outcomes, not production repair admission. R2/R3
must remain absent until a verified target capsule makes R0/R1 interpretable.
Local proof and exact commands are in `artifacts/p38_2g_local_gate.md`.

Real Qwen3-8B synthetic controls have now exercised the runner. At prompt
lengths 256 and 1788, R0 and R1 were bitwise exact with each other, while both
were red against REF at all eight scored actions. The shallow maximum was
larger than the deep maximum. This is `LOCAL_CARRIER_NOT_ISOLATED`, not a
production reproduction and not evidence for a KV-1791 threshold. See
`artifacts/p38_2g_onehost_synthetic_0811.md`.

Do not implement or interpret R2/R3 from the synthetic result. The target
operator must still produce a verified P38.2f capsule. If target R0/R1 show the
same broad split, add an exact serving-envelope control before changing KV
update behavior.
# 2026-08-11 target-row update: local serving envelope rejected

The verified P38e1 source row 191 has now run on real Qwen3-8B DP1xTP4. Do not
implement or interpret local R2/R3 from the current mask-derived schedule.

- R0 equals R1 bitwise at raw target, processed target, normalizer, and logprob.
- R0/R1 differ from REF at 395 of 517 action logprobs.
- REF logprob SHA exactly equals captured `S_prefill`/`T_old`.
- R0/R1 do not equal captured `S_decode`.
- Measurement integrity passed; classification is
  `LOCAL_CARRIER_NOT_ISOLATED`; no production repair is admitted.
- Evidence: `artifacts/p38_2g_onehost_target_row191_0811.md`.

The next implementation belongs in the actual serving envelope. The existing
P18/P35 capture in `patches/tpu_inference/06-tpu-runner.patch` and
`07-tpu-runner-p35-metadata.patch` records metadata only when
`input_batch.num_prompt_logprobs` is non-empty, which captures rescore B but not
ordinary decode A. Add a separate default-off P38 serving-metadata capture that
also executes for decode and records, per scheduler call:

1. monotonically increasing call ordinal and the live request/slot IDs needed
   to join a capsule row back to its serving request;
2. input IDs and positions;
3. attention input positions, block tables, sequence lengths, query starts,
   and request distribution;
4. exact logical-to-physical page IDs used by the selected requests;
5. cache shape, dtype, sharding, page size, and configured D/P/M block tuples;
6. whether the call is decode, prefill, or mixed and the effective
   `update_kv_cache` value.

The capture must be bounded, attempt-unique, fail on overwrite, print completed
record counts, and include a negative control proving that missing decode
records reject classification. The mismatch capsule must also preserve the
row-to-request-ID mapping; row order alone is not an admissible join key.

The pinned-source audit is recorded in
`phases/p38-2g2-pathways-serving-envelope.md`. Production A can execute inside
`runner/decode_loop.py::continue_decode`, so capture limited to ordinary
`model_fn` or prompt-logprob calls is invalid. The v3 public API also cannot
construct a clean write-only arm: `update_kv_cache=False` both skips the write
and forces all-cache reads. Do not label any v2-writer experiment as a
single-variable `W` arm.

After an exact serving record exists, run stock first. Only if that record
reproduces captured `S_decode` may a separate source-pinned diagnostic enable
the combined historical `U` arm: stock RPA writes the cache, its output is
discarded, and a second RPA call with `update_kv_cache=False` supplies the
attention output. This can establish causality for the combined mechanism but
cannot distinguish fused-write effects from read-source effects.

Keep prefix cache disabled, backward disabled, optimizer commits zero, and the
precision/fixed-M/fixed-reduction configuration unchanged. Rollback is leaving
the new capture and counterfactual environment variables unset.

## P38.2g2 local handoff: implementation gated, target not run

The current dirty worktree now contains the implementation described above:

- patch 09 captures the actual donated-cache `continue_decode` call, including
  request IDs, full current token histories, physical page IDs, scheduler and
  attention metadata, sampling leaves, the physical/logical selector, and
  bounded post-dispatch outputs;
- patch 08 adds only the combined two-pass `U` arm. It cannot distinguish the
  fused writer from the read-source change and must never be named `W`;
- `render_p38_serving_jobsets.py` renders separate stock and U Attempt-0
  manifests; and
- `90_run.sh` classifies and emits a SHA-verified tar as base64 so evidence
  survives pod deletion. `extract_p38_serving_archive.py` recovers it. Both
  manifests force `CANON_P38_PRECHECK_ONLY=1`, so an exact U arm stops before
  backward rather than falling through the misleadingly named workload stage.

Local gates are green: pinned-image install 29/29 for both model overlays,
logprob chunking 10/10 for each overlay, ten serving-classifier controls,
four renderer controls, four archive-transport controls, and the complete
P33 CPU suite. A shell postflight also proves exact precheck stop is accepted
while a red stop is rejected. No Pathways/TPU target result exists.

Do not tell an operator to pull this worktree until the user explicitly
approves commit and push. After publication, follow the exact commands in
`phases/p38-2g2-pathways-serving-envelope.md`: render both, dry-run both,
apply stock only, archive/classify it, and apply U only if stock reproduced the
known red with complete capture evidence. Never apply the output directory.
