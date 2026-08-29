# M15 APC target carrier runbook

This runbook is for the remote execution agent. The agent runs checked-in
commands; it does not edit YAML, numerical code, or evidence. Large payloads
remain in GCS exactly like the earlier P38/lm-head investigation. Only small
machine-generated receipts are returned through Git or chat.

## Current operation: certify provenance hardening, then recover read-only

Do not render, apply, restart, or launch a JobSet. Commit
`971bb2281417ecb6e33cfa6bb68a422f7fd24f00` contains a four-file
Attempt-18 return whose local manifest verifies, but the return is rejected as
`OFFICIAL_RETURN_PROVENANCE_FAIL`: the classifier path/SHA cannot match the
pinned runtime source, unrelated inputs and both arm manifests collapse to
one digest, runtime fields are missing, paths are impossible, and the raw
terminal receipt is absent. `LIVE_KV_FINGERPRINT_EQUAL` is not admitted and
Phase E remains closed.

The hardened tree makes 971bb228 a locked negative regression and preserves
both rejected snapshots. It must first receive separate commit/push approval.
After publication, obtain separate approval and pass the official pinned-image
aggregate with the exact identity and `m15_e0=30` marker specified in
`HANDOFF.md`. Only after a third approval for this one GCS read may a
bucket-capable executor use a clean exact-SHA `local/*` worktree and the
original preserved `e01` render directory:

```bash
ANALYSIS_SOURCE=<full-published-provenance-hardening-SHA>
RENDER=<preserved-attempt18-e01-render-directory>
RETURN=/mnt/disks/tunix-data/m15-e0-return-recovery-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt18_e0_return_recovery.sh \
  "$ANALYSIS_SOURCE" "$RENDER" "$RETURN" /mnt/disks/tunix-data
```

Run directly without a pipe and never reuse an output path. The operation
reads existing GCS evidence only; it does not write GCS, query/apply
Kubernetes, or use TPU. It verifies root manifests and terminal markers,
retrieves both classifier JSONs, and requires the exact runtime classifier,
complete per-record/comparison/red-join fields, distinct provenance digests,
basename-only source paths, B full-reset/all-cached-zero, zero-commit receipts,
and a preserved raw log. Raw-log/token/capsule/archive payloads remain outside
the returned directory and chat.

Require the official return markers, `M15_E0_RETURN_INTAKE_PASS`, and both
`[M15.E0.RECOVERY]` terminal lines specified at the top of `HANDOFF.md`.
Return exactly the four files listed there plus sanitized markers and local
raw-log path/SHA. Import the result only into a fresh additive evidence
directory; never overwrite the rejected 971bb228 directory. Preserve all
failure scratch. A missing render or provenance failure is `INCONCLUSIVE` /
`OFFICIAL_RETURN_PROVENANCE_FAIL`, not permission to infer a bucket root or
launch again.

## Superseded operation: prepare the E0 Layer-0 live-KV pair; local CPU only

D3e is complete and committed: the canonical completion-position-zero action
is `FIRST_RED_LOCALIZED` at Layer 0 `k_post_rope -> rpa_output`, shape
`[2048,1,15,8]`, with A-B=207 bytes / 95 elements and B-C=0. The evidence is
still analysis-grade partial, not a complete target PASS. E0 observes whether
the uniquely future-bound A request already has a different stored Layer-0 KV
fingerprint before RPA. It is not a numerical repair.

The executor must receive the exact full published E0 SHA. From a new clean
`local/*` worktree at that SHA, run only the prepare command first:

```bash
SOURCE_COMMIT=<full-published-E0-follow-up-SHA>
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/m15-e0-kv-${RUN_ID}
test ! -e "$OUT"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt18_e0_kv_pair.sh \
  "$SOURCE_COMMIT" "$RUN_ID" "$OUT"
```

This verifies the committed D3e manifest, reruns focused host gates, and emits
two immutable YAMLs, `D3E_ADMISSION.json`, `KV_CLASSIFIER_RUNTIME.json`,
`RUN_CONTRACT.json`, and `SHA256SUMS`. If host Python lacks NumPy, the focused
classifier can use only the registered exact image ID already present locally,
with `--pull=never` and `--network=none`; the selected route is recorded in the
self-hashed receipt. This is not the official exact-image aggregate. The
wrapper does not access GCS or Kubernetes and does not launch TPU.
Require both terminal markers:

```text
[M15.E0.KV] RENDER_PASS ... rounds=1 layer=0 aliases=8 pages=96 ...
[M15.E0.KV] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0
```

Stop on any source, clean-tree, manifest, classifier-runtime receipt,
pair-normalization, or marker failure. The wrapper preserves and prints its
scratch path on failure. Do not edit either YAML. Next, request explicit
approval for the complete
official pinned-image aggregate on
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
Pinned-image PASS still does not authorize launch. A later DP8xTP8 launch and a
later compact GCS read each require separate approvals. Exact commands,
required markers, failure preservation, and the decision table are at the top
of `HANDOFF.md` and in `phases/phase-e0-layer0-live-kv-discriminator.md`.

## Historical operation: D3e canonical first-action reclassification; CPU/GCS read only

D3d has completed. Do not run its old entrypoint directly and do not launch a
JobSet. The verified return uniquely binds source row 217 / completion
position 0 to one A request, but the old gate mixed that first-action boundary
with later joinable red actions.

The D3e analysis tree has passed host and the separately approved official
pinned exact-image aggregate. It must now be committed/pushed only with
explicit user approval. After the user supplies that full published SHA and
separately approves GCS read access, a bucket-capable executor uses a clean
`local/*` worktree and one fresh local output directory:

```bash
RETURN=/mnt/disks/tunix-data/m15-d3e-canonical-action-return-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d3e_canonical_action.sh \
  "$RETURN" /mnt/disks/tunix-data
```

Run directly without a pipe. The wrapper delegates to the existing immutable
D36 recovery, then independently verifies the D3e decision scope, A-B/B-C
boundary, unique future-prefix binding, fingerprint geometry, source anchors,
and presence of cache-page coordinates. It performs GCS reads only and no GCS
write, Kubernetes query, or TPU launch.

Required final markers:

```text
M15_D3E_CANONICAL_ACTION_REVIEW_PASS status=<status> decision_scope=completion-position-zero ... numerical_repair_authorized=0
[M15.D3E.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D3E.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Return only the three JSON files, `SHA256SUMS`, its SHA256, and sanitized
terminal markers. Preserve any failed scratch directory. A localized return
is review input only; it does not authorize Phase E. A preserved candidate set
means direct producer/request provenance and original checkpoint-shape
metadata must be added before a separately approved fresh matched DP8xTP8
pair. Full details are at the top of `HANDOFF.md` and in
`phases/phase-d3e-canonical-first-action-scope.md`.

## Superseded operation: Attempt-17 D3d offline binding; retained for provenance

Attempt 17 (`d36`) already preserved one sealed APC-on treatment round with
A-B=207 differing bytes / 95 elements, B-C=0, and
`M15_INTERNAL_FIRST_RED_CANDIDATE_SET`. Do not launch another JobSet yet. The
sealed compact bundle contains the selected seam rows, mismatch capsule, and
full replay ledger needed to test a stable source-row/request join offline.

The analysis change must first be committed and pushed with explicit user
approval. The remote executor then creates a clean `local/*` worktree at that
exact analysis commit. GCS read access is a separate approval. No TPU or
Kubernetes approval is needed for this operation because the wrapper neither
queries nor mutates them.

Run exactly one command on the bucket-capable machine:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d36_offline_binding.sh \
  /mnt/disks/tunix-data/m15-d36-offline-binding-return
```

The output directory must not exist. The wrapper:

1. requires a clean `local/*` analysis worktree and canonical preflight PASS;
2. verifies the committed 84-member Attempt-17 evidence package;
3. reconstructs the runtime-source d36 full/Layer-0 render and matches both
   JobSet identities to the committed receipts;
4. performs the existing small multiround GCS return;
5. reads only treatment Round 0's `WIDE_SHA256SUMS` and compact bundle;
6. verifies the outer bundle digest, safe tar membership, internal manifest,
   and byte-identical committed classification;
7. reruns the official classifier with fail-closed future-prefix binding; and
8. writes a three-file, self-hashed local return.

It does not upload, write, delete, restart, apply, or launch remote state. The
large token-bearing bundle exists only in the registered evidence root and
ephemeral scratch. On failure, scratch is preserved on the executor for
diagnosis; do not return its payload.

Success has two terminal lines:

```text
[M15.D36.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D36.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Return only `D36_OFFLINE_REVIEW.json`, `D36_RECLASSIFICATION.json`,
`REMOTE_MULTIROUND_SUMMARY.json`, `SHA256SUMS`, the terminal lines, and the
manifest SHA. Do not return bucket roots, credentials, tar/capsule/ledger
payloads, or raw token prefixes.

`FIRST_RED_LOCALIZED` is review input, not an automatic repair authorization.
It must contain last exact, first red, shape, request/call/token/cache/page
coordinates, and source anchors. `FIRST_RED_CANDIDATE_SET_PRESERVED` means the
next implementation is one observational producer-row/request provenance
field followed by host and exact-image gates; only then may a new matched
DP8xTP8 pair be proposed. Phase E stays closed in both cases until explicitly
reviewed.

## Historical operation: recover the already-run d33 pair

Do not render, apply, restart, or delete a JobSet. The submitted d33 package is
only a five-file analysis subset, so the current operation is a read-only
recovery of the per-round objects that the runtime already attempted to seal.
The remote agent needs a clean current checkout, read-only access to the two
existing JobSets, and read access to their registered bucket roots.

```bash
RETURN=/tmp/v1-apc-m15-attempt14-d33-operator-return
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/recover_m15_attempt14_d33_operator_return.sh \
  "$RETURN" /mnt/disks/tunix-data default
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

This dedicated entrypoint removes the missing-original-`$OUT` ambiguity. It
verifies the committed Attempt-14 subset, derives the exact source and two
JobSet/object identities, emits a `LOCATOR_ONLY` recovery receipt, and calls
the existing official operator-return wrapper. The final package binds the
locator receipt, all recovered official classifiers, numerical summary,
sanitized JobSet status, and remote `run.log` SHA/size receipts under one
manifest. It never downloads or returns the large logs or token-bearing
archives.

Return the directory unchanged. A hand-written classification or receipt is
not acceptable. `COMPLETE` admits six-round scientific review;
`ROUNDS_RECOVERED_ROOT_INCOMPLETE` and `PARTIAL_ROUNDS_RECOVERED` remain
analysis-grade; `NO_DURABLE_ROUND` is the only status that makes a fresh target
run worth discussing. Phase E and all APC numerical edits remain blocked until
the machine return is reviewed.

Evidence publication, if separately approved, is additive: copy the entire
return into a new `v1_apc_m15_attempt14_d33_operator_return_20260828/`
directory, verify its manifest in place, and leave the prior five-file subset
untouched. Do not select or rename individual classifier files.

## Historical operation: launch the three-round pair

The already-completed d33 launch used the three-round durability path below.
Do not run it again for the current task. It is retained for provenance:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
```

## Historical operation: publish Phase D2; target launch remained approval-gated

Do not launch a JobSet. The checked-in bounded-shard host gates in the first
section of `HANDOFF.md` and the pinned exact-image aggregate gate pass. The forced-death
fake-GCS case is not optional: a completed shard must remain independently
extractable and SHA-valid while root `COLLECTED.json` and `COMPLETE.json` are
absent. Also require the runtime-source mismatch negative, published-output
drift negative, missing-pair negative and tampered-shard negative.

The exact-image terminal marker is `V1_HP_EXACT_IMAGE_PASS ...
apc_m15_carrier=66 m15_durability=1 ...`. Publish the reviewed source only.
Do not access real GCS or launch TPU/Kubernetes until the user separately
approves the DP8xTP8 pair.

## Historical operation: read-only Attempt-9 full object inventory (complete)

Do not launch TPU work and do not rerun the earlier exact-name salvage. That
audit is complete: its 2/2 returned files verify, both arms expose only
`PREFLIGHT.json` among the seven queried names, and the later receipt's full
source SHA conflicts with both runtime markers and does not exist in the Git
object database.

Follow the exact command at the top of `HANDOFF.md`. It recursively lists names
under both registered roots, strips the bucket roots from the return, downloads
no payload, mutates no remote state, and emits only `OBJECT_INVENTORY.json`,
`PACKAGING.txt`, and `SHA256SUMS`.

If any object other than `PREFLIGHT.json` exists, return the inventory and stop;
the analysis owner will prepare a downloader for only those names. If both
roots contain only `PREFLIGHT.json`, Attempt 9 is irrecoverable from its
registered GCS roots. Neither outcome permits a target relaunch or numerical
repair.

## Conditional operation after Phase D2 publication: DP8xTP8 first-red walk

Do not run the older Phase-C preparation again. The one-host r10-r13c ladder
was exact and did not reproduce the target. The next useful run is the known-
red 64-chip M15 topology with the layer observer already attached.

This one-round section is superseded by Phase D3.  It is retained only to
explain older run receipts.  Do not copy its renderer command for the next
launch; use `prepare_m15_multiround_pair.sh` above.

Preflight after a published source exists:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
SOURCE_SHA=<full-published-sha>
test "$(git rev-parse "$SOURCE_SHA")" = "$SOURCE_SHA"
RUN_ID=<fresh-label>
OUT=/tmp/v1-apc-m15-wide-${RUN_ID}
test ! -e "$OUT"
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --observer layer \
  --output-dir "$OUT"
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py
sha256sum "$OUT"/*.yaml
```

The two expected files are `jobset-v1-apc-m15-off-layer.yaml` and
`jobset-v1-apc-m15-on-layer.yaml`. Submit each with a standalone `kubectl
apply`; they may be submitted concurrently when the user approves the pair.
Never hand-edit a rendered environment.

Postflight is automatic. The live worker first seals bounded observer shards,
then runs `classify_m15_apc_wide_seam.py` and
`package_m15_apc_wide_seam.py` from their sealed union. `90_run.sh` only
verifies that sealed output and asks the live worker for terminal publication.
Acceptance requires:

- off: A-B=0, B-C=0, observer A/B records present;
- on: B-C=0 and either exact treatment or a joined first-red classification;
- a red on arm has at least one completion-position-zero standard-path join;
- all raw selected NPZ files match their JSON SHA;
- each shard, the wide-round receipt, root manifest, and compact tar internal
  `SHA256SUMS` verify;
- runtime checkout SHA equals the full rendered source SHA;
- the legacy incident ledger is absent and exactly one signed bypass marker is
  present;
- zero backward and zero optimizer commits.

Read `p38_seam.classification.json`, not a prose summary. If it reports
`M15_LAYER_FIRST_RED_LOCALIZED`, take `selected_layer=L` and render the
conditional full run:

```bash
FULL_OUT=/tmp/v1-apc-m15-${RUN_ID}-full-l${L}
test ! -e "$FULL_OUT"
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "${RUN_ID}-full-l${L}" \
  --observer full \
  --seam-layer "$L" \
  --output-dir "$FULL_OUT"
```

Do not launch full mode before layer mode chooses `L`. A full-mode PASS must
say `FIRST_RED_LOCALIZED` and return a last exact/first red checkpoint plus
source `file:line`. Only then may a repair phase begin.

The compact bundle contains true token/capsule data. `m15-wide-v1` stores it
under the registered Attempt-0 evidence root but never creates or uploads the
old whole-capture tar. Return only its path/size/SHA and classifier fields;
never place token data in Git or chat.

### Current decision table

| Observation | Decision |
|---|---|
| off red or B-C red | hard stop; carrier/shared contract invalid |
| on exact | one target non-reproduction; no fix claim |
| layer N red | full observer at mechanically selected layer only |
| layer exact, tail red | localize reported LM-head/normalizer tail interval |
| no first-action join | evidence incomplete; do not change numerics |
| full `FIRST_RED_LOCALIZED` | open minimal Phase-E repair |

Everything below is retained as historical carrier context.

## What this run is for

One fresh target observation records enough information to replay a red
without running the FrozenLake environment or sampling new actions again:

1. all 256 final prompt/completion token streams and A/B/C logprobs;
2. every serving call's request dispatch order, DP rank/local slot, token
   prefix identity, logical position, and physical page table;
3. the exact first-red request/call/page-generation join.

The run still executes a real M15 rollout once because historical `m15i` did
not save these inputs. Later replay may skip environment generation, but must
still execute the real serving decode and independent full-reset B arm.

This carrier does not modify or repair RoPE, attention, KV values, lm-head,
loss, backward, or optimizer. Both JobSets stop before backward and commit.

## Attempt 6 is complete — prepare Phase-C input without a new rollout

Attempt 6 supplied the required matched pair. Off is `CONTROL_GREEN`; on is
`FRESH_TARGET_RED_FROZEN` with A-B=1,770 bytes / 748 elements and B-C=0. The
large on-arm carrier remains in its immutable GCS Attempt-0 root. A
bucket-capable agent prepares the next small input plan with one checked-in
command:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_prepare.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0 \
  /mnt/disks/tunix-data
```

The wrapper performs, in order:

1. root `SHA256SUMS` validation and terminal-state checks;
2. safe extraction of `serving-capture.tar`;
3. the existing full replay-carrier audit;
4. independent byte-level A-B/B-C recomputation from `m15_producer_unit.npz`;
5. exact request-history joins and separation of onset from the later captured
   incident;
6. generation of `replay-prefix-plan.jsonl` for calls 1 through 188;
7. upload of only the derived self-hashed files under
   `derived/m15-replay-input-plan-v1/files/`, with `SHA256SUMS` uploaded last.

Success is exactly:

```text
[M15.APC.REPLAY.PREPARE] COMPLETE status=M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED ... red_rows=201,245 replay_prefix_end_call=188 ...
```

Fetch the small return into a fresh directory and independently verify it:

```bash
DERIVED=gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0/derived/m15-replay-input-plan-v1/files
RETURN=/tmp/m15-replay-input-plan-v1-return
test ! -e "$RETURN"
mkdir -p "$RETURN"
gcloud storage cp "$DERIVED/REPLAY_ANALYSIS.json" "$RETURN/"
gcloud storage cp "$DERIVED/UPSTREAM_AUDIT_RECEIPT.json" "$RETURN/"
gcloud storage cp "$DERIVED/replay-prefix-plan.jsonl" "$RETURN/"
gcloud storage cp "$DERIVED/SHA256SUMS" "$RETURN/"
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

Return the terminal marker, `REPLAY_ANALYSIS.json`,
`UPSTREAM_AUDIT_RECEIPT.json`, `SHA256SUMS`, and the URI/SHA/size of the JSONL.
The terminal marker prints the JSONL SHA and byte size. On failure, return
stderr and the exact return code. Do not commit the large JSONL or original
tar.

Interpretation is fail-closed: row 201/completion position 0 is the canonical
first mismatch; row 245/call 164 is the earliest request belonging to any red
row; row 201's request starts at call 187, so the plan covers its first output
interval through call 188; row 245/call 565 is the first later fully captured
tensor incident. The plan is input evidence, not a model replay or a root-cause
verdict.

## Attempt-0 through Attempt-4 incidents

Never relaunch source `eb58954f...`. That Attempt-0 command carried
`--p57_workload_candidate=m15 --p57_data_split=main`, but the rendered
environment omitted the matching signed `CANON_P57_*` fields. The workload
entrypoint therefore exited before learner construction, capture creation, or
any A/B/C numerical verdict. `INCONCLUSIVE` is the permanent classification.

The repaired source requires exact `m15/main` in both CLI and environment and
keeps the package-safe `python3 -u -m
examples.frozenlake.train_frozenlake_qwen3` entrypoint. This is a bootstrap
contract fix only; production APC remains off.

Never relaunch source `283cb67e...`. Attempt 1 proved that bootstrap repair
worked: all overlays and GCS preflight passed. It then stopped before learner
construction because `train_frozenlake_qwen3.py` reused the legacy P38 DP16
geometry for this different carrier. The old contract expected
`mini_batch_size=4`, token IS, workload `frozenlake`, and eight producer units;
the signed M15 target is `mini_batch_size=32`, no IS,
`frozenlake-dp8-tp8`, and one full producer unit. The current source keeps both
contracts separately fail-closed. This is still admission-only and is not an
APC numerical fix.

Never relaunch source `41a2043c...`. Attempt 2 was the first run to exercise
the real DP8xTP8 M15 serving envelope: it completed more than 1,800 serving
calls, all four standard tensor captures, and most of the 15-turn rollout.
It then exposed two observer defects, not a model mismatch:

- the incident ledger reached its 256 MiB ceiling at call 326 and emitted
  1,650 nonfatal capture errors;
- the drain tail entered the production `continue_decode` program, while the
  old observer asserted that every call in the process must be `standard`.

Do **not** remove `CANON_CONTINUE_DECODE=8`. The historical `m15i` production
red used that program; removing it changes the experiment. The current repair
keeps tensor records and generic request/incident ledgers standard-only,
admits M15 `continue_decode` into the dedicated full replay chronology from
its first production call, and never treats tensor-strata completion as a
program-path prerequisite. A red
carrier must prove A used both `standard` and `continue_decode`; B must remain
the independent full-reset `standard` path. The M15-only signed ledger bound
is 2 GiB, based on Attempt 2's 268,192,266 bytes at call 326 and roughly 1,894
observed calls. Ordinary P38 renderer limits are unchanged.

Never relaunch source `cdd3987c...`. Attempt 3 proved that APC-on can enter
`continue_decode` before any complete set of standard tensor strata exists.
Patch 28 removed that invalid ordering assumption without broadening tensor or
incident capture.

Never relaunch source `618eb775...`. Attempt 4 completed all 2,560 APC-on
rollout requests at 92.5% prefix-cache hit rate, proving the patch-28 program
path repair took effect. It then stopped before A/B/C because the generic
alignment gate rejected `sampler_is=None`. The current repair admits no-IS
only for the exact signed M15 target identity and requires this one runtime
receipt:

```text
[CANON_APC_M15_SAMPLER_CONTRACT] PASS sampler_is=none use_rollout_logps=1 rollout_logps=present tis_weights=absent
```

Missing/duplicate receipt, a token sampler, any TIS weights, or a neighboring
workload/profile/topology is fatal. This is an admission repair, not an APC
numerical repair.

## Approval boundaries

The following are three separate user decisions:

1. commit and push the prepared source;
2. run the exact-image gate;
3. launch the matched APC-off/APC-on pair. One explicit pair-launch approval
   covers both standalone submissions, which run concurrently.

Do not infer one approval from another. Do not launch from a dirty tree or an
abbreviated SHA.

## Render

After the source has been committed and pushed:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<40-character-committed-sha>
RUN_ID=<new-unique-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Expected files:

```text
jobset-v1-apc-m15-off.yaml
jobset-v1-apc-m15-on.yaml
```

Never edit either file. The renderer fixes M15/main, seed 42, DP8xTP8,
32 prompts x 8 generations, concurrency 256, 15 turns, 4096 prompt tokens,
8192 response tokens, temperature 0.7, one diagnostic round, zero backward,
and zero optimizer commit. It also deliberately preserves
`CANON_CONTINUE_DECODE=8`, standard-only four-stratum tensor capture, and the
2 GiB M15 incident/replay byte bound. It also fixes
`--sampler_is=none`; do not hand-edit that to `token`.

The renderer and Step-00 resolver must reject any CLI/environment identity
split. A valid rendered arm carries:

```text
CANON_P57_WORKLOAD_CANDIDATE=m15
CANON_P57_DATA_SPLIT=main
--p57_workload_candidate=m15
--p57_data_split=main
--sampler_is=none
```

## Exact-image admission

The current patch-28 tree passed this gate on 2026-08-25. It remains the
canonical rerun command if any runtime or test file changes before publication:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Expected post-fix terminal marker includes `apc_m15_carrier=46`. The nested
P33 gate must also report `runner_tests_per_overlay=35`; its new installed-
runner test sets captured records/strata to zero, executes the full
`_p38_serving_begin` branch, and proves M15 `continue_decode` writes the replay
ledger without entering generic incident/tensor capture. The same path remains
rejected outside M15 debug. This is not a target numerical result.

## Paired launch

Launch commands must be standalone. Do not append `tee`, a pipe, `&&`, or a
monitor.

After paired-launch approval, submit both manifests immediately as separate
commands:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off.yaml"
```
```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on.yaml"
```

Do not wait for the control before submitting treatment. The two JobSets may
run and fail concurrently; keep their logs and GCS evidence separate. A
failure in one arm does not stop or delete the other.

Classify off first after both return. Only `CONTROL_GREEN`, B-C zero, and all
three GCS terminal markers make the on-arm result interpretable as an APC
comparison. If off is red or inconclusive, retain and report on, but make no
APC-specific causal claim from it.

After a green control, accepted treatment outcomes are:

- `FRESH_TARGET_RED_FROZEN`: a complete replay carrier must also be frozen;
- `TARGET_NOT_REPRODUCED`: one target observation was exact, no fix claim.

Any other classification is a hard stop.

## GCS layout

Each JobSet writes to:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0/
```

Root completeness requires:

```text
PREFLIGHT.json
COLLECTED.json
COMPLETE.json
SHA256SUMS
run.log
pre-alignment.jsonl
serving-classification.json
serving-capture.tar
```

The large `serving-capture.tar` contains, for both clean and red runs:

```text
m15_producer_unit.npz
m15_replay_envelope.jsonl
m15_apc_target.classification.json
```

The complete `m15_replay_envelope.jsonl` must contain both `standard` and
`continue_decode` records for serving arm A, and only `standard` records for
serving arm B. `m15_full_replay_carrier/replay_contract.json` records the
mechanically checked `program_paths_by_arm` map. Missing the continue tail,
placing B on it, or observing any unknown path makes packaging inconclusive.

For a red treatment it must additionally contain:

```text
m15_first_red_replay/first_red_capsule.npz
m15_first_red_replay/first_red_contract.json
m15_first_red_replay/SHA256SUMS
m15_full_replay_carrier/replay_contract.json
m15_full_replay_carrier/request_row_joins.jsonl
m15_full_replay_carrier/SHA256SUMS
```

The periodic live snapshot also includes the growing replay envelope. A pod
loss therefore leaves a bounded chronology snapshot in GCS even before final
collection, but only the terminal `COMPLETE.json` admits a finished attempt.
The M15 carrier intentionally prints only SHA/size/`encoding=gcs-only` receipts
to the pod log; it does not base64-duplicate the large NPZ/tar into that log.

## Run the GCS-side audit

Run this on the machine that can read the bucket. It downloads and verifies
the immutable root, checks the nested producer/envelope/first-red joins, and
uploads a small derived audit beside the large payload:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_audit.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Expected terminal marker:

```text
[M15.APC.GCS] COMPLETE status=<status> ... destination=.../derived/m15-replay-audit-v1
```

The derived prefix contains only small receipts under `files/`; its
`SHA256SUMS` is uploaded last. Raw logs, the complete producer NPZ, and the
serving envelope remain in the original GCS attempt and are not committed.

## What to return

Return only:

1. the exact source SHA, JobSet, attempt number, Kubernetes terminal state,
   and source GCS URI;
2. the one-line `[M15.APC.GCS] COMPLETE ...` output;
3. the derived GCS URI;
4. the small derived `RETURN_RECEIPT.json`, `SHA256SUMS`,
   `m15-classification.json`, and, on red, `replay-contract.json`;
5. any nonzero command return code and its complete stderr.

Do not manually summarize the large NPZ/JSONL and do not add them to Git.

## Claim ceiling

Even a successful red capture means only:

```text
FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN
```

The next phase must execute the carrier through serving. It may then say
`ONEHOST_NOT_REPRODUCED` or advance to first-red localization. It may not say
that APC, RoPE, pages, or topology is the root cause merely because the
carrier was captured.
