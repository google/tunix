# M15 APC target-debug handoff

## START HERE — run the DP8xTP8 wide layer observer

The one-host ladder is exhausted: real scheduler publication, 32-request
composition, `continue_decode=8`, and full M15 chronology all stayed exact on
DP1xTP4. The root remains a scale/topology seam. Do not run another one-host
replay and do not guess a RoPE/page repair.

This source prepares a known-red target localization run. It changes no model
arithmetic and keeps production APC off. The first target run attaches one
identical observer to an APC-off control and APC-on treatment:

- all 36 layer input/output fingerprints;
- final norm and terminal tail;
- positions 960..4096, covering the historical 1226 and Attempt-6 prompt
  boundaries;
- exact request/call/token-prefix/page receipts;
- automatic M15-aware classification and compact selected-record bundle.

After the user explicitly approves commit/push and the exact source SHA is
available, render only with:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<40-character-published-sha>
RUN_ID=<fresh-unique-label>
OUT=/tmp/v1-apc-m15-wide-${RUN_ID}
test ! -e "$OUT"
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --observer layer \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Expected manifests:

```text
jobset-v1-apc-m15-off-layer.yaml
jobset-v1-apc-m15-on-layer.yaml
```

If both allocations are available, the user may submit the two standalone
commands without waiting between them. Do not append pipes, `tee`, `&&`, or a
monitor:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off-layer.yaml"
```

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on-layer.yaml"
```

Interpret control first. The off arm must remain A-B=0 and B-C=0. The on arm
must keep B-C=0; if red, it must emit one of:

```text
M15_LAYER_FIRST_RED_LOCALIZED
M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED
```

If the first result selects layer `L`, do not guess or edit YAML. A separately
approved follow-up is rendered by:

```bash
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "${RUN_ID}-full-l${L}" \
  --observer full \
  --seam-layer "$L" \
  --output-dir "/tmp/v1-apc-m15-${RUN_ID}-full-l${L}"
```

Return exactly:

1. full source SHA, both JobSet names, attempts, Kubernetes terminal states,
   and both GCS Attempt-0 URIs;
2. the complete `CANON_ALIGN_PRE` line for each arm;
3. both `p38_seam.classification.json` files and their SHA-256;
4. the `CANON_APC_M15_SEAM_BUNDLE` path/size/SHA marker for each arm;
5. on red: `classification`, `gate`, `selected_layer`, `last_exact_boundary`,
   `first_red_boundary`, `coverage`, and `source_interval` from the JSON;
6. any nonzero return code plus complete stderr/raw-log tail.

The compact bundle contains real token/capsule material and is generated in
the pod, but this change does not automatically upload that new payload. Do
not add it to GCS until the user explicitly authorizes that payload. Existing
P38 root artifacts and seam classification follow the existing durability
path unchanged.

Current claim ceiling is `WIDE OBSERVER READY / TARGET NOT RUN / ROOT CAUSE
NOT LOCALIZED`. See [Phase D](phases/phase-d-wide-target-observer.md).

## Historical Phase C replay input and Attempt-6 evidence

Attempt 6 paired execution (`d12-9f91d930`, source commit `9f91d93001dd5b44659f062626eb93fc65e6fcb4`) ran on 64 TPUs (DP8xTP8) for both control and treatment arms, persisted complete raw payloads to GCS Attempt-0 roots, and successfully passed the GCS replay audit `run_m15_replay_gcs_audit.sh`:

- **Control Arm (`canon-v1-apc-m15-off-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, 0.0% prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=117415 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - GCS Audit Verdict: `CONTROL_GREEN` (`receipt_sha256=c9550f73...`, `manifest_sha256=b91cd34c...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, **92.9%** prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=119565 bounds=[('S_decode_vs_S_prefill', 1770), ('S_prefill_vs_T_old', 0)]` (**Captured exact mismatch of 1,770 bytes / 748 elements**).
  - Canonical first mismatch: row 201, completion position 0, logical prefix 1066; its request starts at call 187 and the bounded interval ends at call 188.
  - Earliest request belonging to any red row: row 245 at call 164.
  - First fully captured tensor incident: row 245, request `400-bc7daec5`, serving call 565, DP rank 0, slot 29, `num_computed_tokens=1248`, 296 exact joins. This is not the onset.
  - Mismatch Capsule: 15,148 bytes (`sha256:9e79a18d...`).
  - Producer Unit: 762 KB, 256 rows (`m15_producer_unit.npz`).
  - Replay Envelope: 103.7 MB, 3,027 calls (`m15_replay_envelope.jsonl`).
  - GCS Audit Verdict: `FRESH_TARGET_RED_FROZEN` (`receipt_sha256=557801a3...`, `manifest_sha256=93f56a0a...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

### Phase C Execution Summary (Replay Input Plan Prepared)

`run_m15_replay_gcs_prepare.sh` was executed on `canon-v1-apc-m15-on-d12-9f91d930/attempt-0` and terminated with:
```text
[M15.APC.REPLAY.PREPARE] COMPLETE status=M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED analysis_sha256=a3c381f8d5e8143ac266a96fb082679e86d85a96eb749255696aaebe649ceff0 manifest_sha256=ed0c67413e51acd639e79dbb95df8698ed8e0386ea606dfcfcb0b1a4fb3e2355 prefix_sha256=b8c00fc704cdd698318a2088c70b9593737a996da8eda1e55d98986d5a8f30a7 prefix_bytes=3938394 red_rows=201,245 replay_prefix_end_call=188 destination=gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0/derived/m15-replay-input-plan-v1
```

Evidence sealed in `evidence/v1_apc_m15_attempt6_paired_d12_20260825/` and `evidence/v1_apc_m15_replay_input_plan_d12_20260826/`.

Current claim ceiling:
```text
M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED
```

Next phase: **Phase D (Deterministic replay and tensor-level tap)**.

## NEXT ACTION — Phase D: Deterministic Replay Harness Execution

The replay plan is prepared (`replay-prefix-plan.jsonl`, 188 calls, 3.94 MB).
The next action is to execute the deterministic replay harness using the saved 188 prefix calls to:
1. Prime the cache from Call 1 through Call 188.
2. Verify reproduction of the exact 1,770 byte A-B mismatch on Row 201/245.
3. Tap the attention / RoPE / block-table layers to pinpoint the exact numerical root cause.

## Scope and current ceiling

This task is the independent APC/prefix-cache numerical lane. It does not
change production APC defaults and it does not authorize a TPU launch,
commit, or push.

Current immutable facts:

- one-host Qwen3-8B DP1xTP4 Phase3 G-A through G-D is exact;
- M15 `m15i` on DP8xTP8 is A-B red by 1389 bytes / 760 elements and B-C exact;
- the first historical red is row 192, completion position 0, logical prefix
  1226;
- the historical archive has hashes but not reversible tokens/request order/
  cache lineage, so `m15i` itself is not an exact replay carrier;
- no numerical source has been repaired and all production full recipes remain
  APC-off.
- Attempt 0 (`canon-v1-apc-m15-off-d3-eb58954f`) is `INCONCLUSIVE`: it never
  reached alignment or created a serving capture. Its command selected
  `m15/main` on the CLI but omitted `CANON_P57_WORKLOAD_CANDIDATE` and
  `CANON_P57_DATA_SPLIT`, which the FrozenLake entrypoint requires to match.
- The bootstrap repair carries exact `m15/main` identity through the renderer,
  profile, and Step-00 resolver and preserves the package-safe module
  entrypoint. It changes no APC, model, alignment, backward, or optimizer math.
- Attempt 1 (`canon-v1-apc-m15-off-d4-283cb67e`) is also prelearner-only: it
  passed all overlays and GCS preflight, then legacy P38 DP16 assertions
  rejected the M15 carrier's `(mini_batch_size, sampler_is)=(32, none)` and
  would next have rejected its DP8 workload/unit identity. No A/B/C verdict or
  replay payload was produced.
- The bounded follow-up keeps legacy P38 exactly at `frozenlake`, DP16,
  8 x 4-prompt producer units and token IS, while only
  `CANON_APC_M15_TARGET_DEBUG=off|on` admits `frozenlake-dp8-tp8`, DP8,
  1 x 32-prompt unit and no IS. Cross-mode and partial geometry negatives are
  executable host tests. It changes no numerical code.
- Attempt 2 (`canon-v1-apc-m15-off-d7-41a2043c`) finally reached the real
  DP8xTP8 rollout and completed more than 1,800 serving calls plus all four
  standard capture strata. It did not reach A/B/C classification: the
  incident ledger saturated at call 326 (268,192,266 bytes) and the drain tail
  later entered the production `continue_decode` path, which the old
  single-path observer rejected.
- Removing `CANON_CONTINUE_DECODE=8` is explicitly rejected because `m15i`
  used it. Attempt 3 proved patch 27's remaining assumption was also wrong:
  APC-on can enter `continue_decode` before four standard tensor strata have
  been captured. Append-only patch 28 admits that registered M15 program path
  from its first call and writes only the dedicated host replay envelope;
  standard tensor capture and generic request/incident evidence stay
  standard-only. The M15-only incident/replay ceiling remains 2 GiB. A frozen
  red must attest A=`standard+continue_decode` and B=`standard`; unknown paths
  and any non-M15 use remain fatal, while a B-side continue path is rejected
  by packaging.
- Attempt 4 (`canon-v1-apc-m15-on-d10-618eb775`) proves patch 28 reached the
  end of the real rollout: 2,560 requests completed, prefix-cache hit rate was
  92.5%, and solve ratio was 0.203. It then failed before A/B/C because the
  generic alignment gate did not admit this carrier's signed
  `sampler_is=None` recipe. Its two committed files pass `SHA256SUMS`; they
  prove the fatal admission boundary but are not a complete replay package.
- Attempt 5 paired run (`d11-a909fda1`, source `a909fda1`) produced hash-valid
  Git snapshots for both arms.  The snapshots show 0.0% cache hits off and
  approximately 89.4%--97.5% on, but contain no alignment, sampler,
  controlled-exit, classification, or GCS-terminal markers.  The accompanying
  receipt is an unverified summary until the GCS-side audit returns.

Claim ceiling: `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`.

The exact remote procedure is in [RUNBOOK.md](RUNBOOK.md). The execution agent
must run those commands rather than constructing a new carrier by hand.

## Prepared bounded carrier

The new renderer creates a matched pair from one committed source tree:

- `off`: APC-off shared-serving control;
- `on`: production-congruent cache-read treatment.

Both use the exact M15 main geometry: DP8xTP8, 32 prompts, 8 generations,
256 trajectories, concurrency 256, `vllm_max_num_seqs=32`, batched tokens 256,
15 turns, prompt 4096, response 8192, temperature 0.7, seed 42. Both stop after
one strict pre-alignment round with zero backward and zero optimizer commit.
Both deliberately use `--sampler_is=none`: A supplies rollout logprobs as the
old-policy source and no token-IS correction weights may exist.

The only intended cross-arm values are
`CANON_APC_M15_TARGET_DEBUG=off|on` and derived
`CANON_VLLM_ENABLE_PREFIX_CACHING=0|1`; a structural test rejects any other
document difference after arm-path normalization.

A must attest:

```text
prompt_logprobs=None
logprobs=1
skip_reading_prefix_cache=False
```

B must attest `reset_prefix_cache=True` and zero cached tokens. The classifier
rejects any B-C byte difference, optimizer marker, wrong source, wrong command,
missing capsule/journal/incident join, or M15 classifier failure hidden behind
the expected controlled exit code 42.

For every arm, postflight/GCS collection preserves:

- `m15_producer_unit.npz`: all 256 final token/logprob rows;
- `m15_replay_envelope.jsonl`: every serving call's exact host-side dispatch,
  request, DP slot, prefix hash/position and physical page table.

For a fresh red, postflight additionally creates
`p38_serving_capture/m15_first_red_replay/` with:

- `first_red_capsule.npz`: the earliest exact incident row's complete prompt,
  completion, masks, A/B/C, policy version, and sampling values;
- `first_red_contract.json`: request/call/position, DP slot, physical pages,
  page generations, and co-batch request IDs;
- `SHA256SUMS`.

It then creates `m15_full_replay_carrier/`. Its request-row join proves that
every scheduled token history comes from the saved 256-row producer and that
the first-red request/call/pages match the incident ledger. The full carrier
records scheduler dispatch order but has not yet forced that order through a
replay harness, so it remains input evidence rather than a mechanism verdict.

## CPU validation already run

From the independent worktree:

```bash
cd /home/yuxuan/code_rl_repro/worktrees/m15_eval_fix_0825
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_analyze_m15i_evidence.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_classify_m15_apc_target_run.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_first_red_replay.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_full_replay_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py -v
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py -v
python3 canon-zero-tim/tests/p3_prefix_cache/test_contract.py -v
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base ff913a84
bash -n canon-zero-tim/cluster/steps/00_env.sh canon-zero-tim/cluster/steps/90_run.sh
git diff --check
```

All task-specific tests and the flag audit pass. After the Attempt-4 repair,
the task carrier is 46/46, P57 is 146/146, V1 CPU is 67/67, and flags are
378/378. The pinned image
`sha256:418dc632...e53a` was then run on the final runtime/test tree. The first
attempt exposed an image-only test PATH defect (`python3` lives under
`/usr/local/bin`); after the test inherited the active interpreter directory,
the full rerun terminated with `V1_HP_EXACT_IMAGE_PASS ...
apc_m15_carrier=46 ... manifests=3`. This is exact-image admission, not a
one-host numerical replay or DP8xTP8 target result.

The worktree was initially created at reference `687b2bd6...`. The operator
branch later advanced through `ff913a84` to `9f79cc56`; the intervening raw-log,
P58 seed, and P64 shared-entrypoint changes were reviewed, then the release
commit was rebased without conflict before the final aggregate gate.

## Next action before any launch

Do not relaunch source `eb58954f...`; its missing signed identity is
deterministically invalid. Patch 28 has passed the targeted and aggregate
exact-image gates on the current tree. First publish the observer repair, then
verify that the committed tree is identical to the admitted tree and render
only from that new full SHA. One paired-launch approval covers both target
arms. Use a unique label and a new output directory:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<full-committed-sha>
RUN_ID=<new-unique-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Do not edit the rendered YAML. Do not render from a dirty or abbreviated SHA.

If any runtime or test file changes before publication, rerun the dependency-
complete pinned-image gate against the immutable production image:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

The post-fix terminal must include `apc_m15_carrier=46`. The current
Attempt-4 repair tree produced that terminal with exit 0. This remains a CPU/image
admission gate, not a DP8xTP8 numerical result.

The installed-runner test is not a string-only predicate check: with zero
captured records and zero strata it executes `_p38_serving_begin`, requires the
M15 replay ledger call, and requires generic incident capture to remain absent.

Before applying the YAML, the rendered environment must contain all four
members of one workload identity:

```text
CANON_P57_WORKLOAD_CANDIDATE=m15
CANON_P57_DATA_SPLIT=main
--p57_workload_candidate=m15
--p57_data_split=main
--sampler_is=none
```

The checked-in renderer and real Step-00 resolver now enforce this and reject
wrong identity or file-path-entrypoint negatives before TPU work begins.
It must also contain `CANON_CONTINUE_DECODE=8`,
`CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard`, and
`CANON_P38_INCIDENT_MAX_BYTES=2147483648`. This combination is intentional:
the tensor records stay single-path while the replay envelope attests the
mixed production tail.

Do not relaunch Attempt-4 source `618eb775...`: it deterministically lacks the
new sampler admission. Attempt 4 has no matched fresh APC-off arm, so it cannot
substitute for either member of the new pair below.

## Paired launch — submit both without waiting

After one explicit paired-launch approval, issue both standalone commands
immediately. Do not append a pipe, `tee`, `&&`, or a monitor to either command:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off.yaml"
```
```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on.yaml"
```

Do not wait for off to finish before submitting on. The arms may execute and
fail concurrently; they still use distinct JobSets, logs, and JobSet-derived
GCS roots. A failure in one arm must not cancel or delete the other arm.

Interpretation remains control-first even though execution is concurrent.
First classify off and require `CONTROL_GREEN`, B-C zero, plus
`PREFLIGHT.json`, `COLLECTED.json`, and `COMPLETE.json`. If off is red or
inconclusive, preserve and report the on package, but do not use on to make an
APC-specific causal claim.

After a green control, the treatment has two admissible outcomes:

- `FRESH_TARGET_RED_FROZEN`: proceed to Phase C/D using the bundled first-red
  carrier; do not infer RoPE/page/cache mechanism yet;
- `TARGET_NOT_REPRODUCED`: this one bounded target observation was exact at
  representative depth/cache occupancy; it does not prove APC fixed.

Any `INCONCLUSIVE`, B-C red, missing join, missing GCS terminal marker, or
unexpected optimizer/backward evidence is a hard stop.

## What the remote execution agent must return

Large GCS evidence remains durable and must not be added wholesale to Git. On
the machine that can read the bucket, run the checked-in GCS audit:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_audit.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Return these small outputs plus their SHA values:

1. `PREFLIGHT.json`, `COLLECTED.json`, `COMPLETE.json`, and remote
   `SHA256SUMS`;
2. `serving-classification.json`;
3. the derived audit URI and `RETURN_RECEIPT.json`;
4. from the derived audit:
   - `m15-classification.json`;
   - on a red, `first-red-contract.json`, `replay-contract.json`, and
     `request-row-joins.jsonl`;
5. the raw log, or if it is too large, an immutable raw-log URI/SHA plus every
   line containing the alignment-pre marker, M15 APC marker prefix,
   `P3_APC_CONFIG`,
   `Prefix cache hit rate`, `CONTROLLED_EXIT`, and `FATAL`;
6. the exact source SHA, JobSet name, attempt number, GCS prefix, and Kubernetes
   terminal status.

The audit script verifies the remote root manifest and both nested replay
manifests before uploading its own `SHA256SUMS` last. Do not call a hash-valid
subset a complete upload; the three terminal markers and required
classifications remain separate completeness gates.

## Rollback

The new selector is default-off, so publication does not change production
behavior. Reverting this bounded carrier must remove the
renderer/profile/classifier/marker additions as one concern. Production
recipes remain APC-off throughout.


## Attempt 7 M15 Target Debug Runs (d13-663cb547)

Attempt 7 dual-arm execution (`d13-663cb547`, source commit `663cb5474490173cfaf33fce3a323d95e5fc2ee1`) was launched on dual 64-TPU allocations:
- **Control Arm (`canon-v1-apc-m15-off-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- **Treatment Arm (`canon-v1-apc-m15-on-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- Retained evidence: `evidence/v1_apc_m15_attempt7_d13_20260826/`.
