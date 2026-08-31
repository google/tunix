# P58 DeepSWE native-first training handoff

## START HERE — Step 65 timeout probe remains retired

P58 production manifests must not set `CANON_EXPECTED_SLICE_DEVICES`,
`CANON_DEVICE_PROBE_TIMEOUT_SECS`, or `CANON_WORKER_INITIAL_SYNC_SECONDS`.
Presence of the expected-device variable launches a temporary JAX/Pathways
client whose disconnect can cancel healthy workers; a longer probe timeout
does not repair that lifecycle. Leaving all three variables absent skips Step
65 and retains the 180-second worker quiet-period default. Use the training
process's exact `split_4x4x8_role_devices` admission as the authoritative
128-device topology check. Do not reintroduce this probe in a rendered JobSet.

## START HERE — K15 lazy-scan mesh repair is local; K16 target not run

K15 completed all 128 multi-turn R2E trajectories across 32 TPU hosts on the 128 TPU v5p slice:
116 finished naturally, 12 max-turn truncated, 0 timeouts/environment failures.
Solved 3 SWE tasks in Step 0 (`Reward = 1.0`), producing 31 non-zero advantage samples (24.2%) across 407,262 action tokens.
Rescore-B prefill passed, and strict Step-0 pre-alignment passed with 100% exact match:
`[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=407262 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)] diff_bytes=0 diff_elements=0 hash=1ef8b0406cb2...`

The run crashed when entering segmented backward in `run_layers_fwd_tape_scan`:
`ValueError: Received incompatible devices for jitted computation. Got argument stacked_leaves[0] of zt_tr_fwd_scan with shape bfloat16[36,2560] and with device ids [2, 3, 18, 19, ...] on platform TPU and shard_map inside jit with device ids [0, 4, 8, 12, 1, 5, ...] on platform TPU at qwen3_p22xh.py:144:11 (P22XHRmsNorm.__call__)`

Root Cause:
In the actual 128 TPU disaggregated topology, rollout uses 64 devices at
DP8xTP8 and trainer uses a disjoint 64 devices at DP8xTP8. The immutable
incident prose says `DP32xTP4`, but raw lines 3–6 supersede that stale label.
Serving mesh is devices `[0, 4, 8, 12...]` while trainer mesh is
`[2, 3, 18, 19...]`.
During rollout, `linear._CANON_MESH` is initialized to the serving mesh.
While per-layer functions were bound with `_canonical_fixed_ar_execution_mesh` in `SegmentedEngine.__init__`, the P71/P50 scan methods (`run_layers_fwd_tape_scan`, `run_layers_scan`, `run_layers_tape_scan`, `run_layers_rev_scan`) invoked lazy JIT scan functions directly without `_canonical_fixed_ar_execution_mesh`.
When JAX jitted `zt_tr_fwd_scan`, `P22XHRmsNorm.__call__` read the rollout `_CANON_MESH` instead of trainer mesh, failing device compatibility check.

The local P58.29 repair, based on unpublished parent
`55553dfe0c3c895de81c66191e5082ed9ec41a32`, promotes the existing segmented
execution-mesh binder to an instance method and applies it to all four lazy
scan JITs: plain forward scan, tape scan, P71 forward-tape scan, and reverse
scan. Colocated mode returns the original callable by identity. The disjoint
positive and colocated negative pass 2/2; P34 static passes ten suites; the
flag audit passes 409/409; and the complete digest-pinned image gate passes
with `disaggregated_scan_mesh=2`.

Do not launch from this dirty worktree. No commit/push, image publication, or
cluster launch has occurred. After separate approval for each transition,
fetch the final clean remote readback SHA and matching image for K16. K16 must
preserve K15's TiTO/clean-data/rollout/Rescore-B/exact A=B=C receipts, cross
the former `zt_tr_fwd_scan` trace, complete segmented reverse with finite
nonzero gradients, and produce exactly the intended first optimizer commit
and checkpoint receipts. Until then this is source/image admission only, not
a repaired target or Zero-TIM PASS.

Immutable incident: `canon-zero-tim/evidence/p58_k15_disaggregated_mesh_scan_incident/`. See `phases/p58-29-k15-disaggregated-mesh-scan.md`.

## START HERE — K11 prompt-only grouped-reverse repair is local, not published

K11 is not another rollout, TiTO, dataset, topology, or alignment failure.
Source `2f61f8fc7cf073964a9adbd30e78de872426a4d2` completed all 128
multi-turn R2E trajectories across 32 TPU hosts, produced 427,594 action
tokens, finished Rescore-B in 109.5s, and passed strict Step-0 pre-alignment
with A-B=0 and B-C=0 (0 differing bytes, 0 differing elements).

The run stopped during segmented gradient computation in `_p32_group_spec`
because DP ranks containing 0 completion tokens (from turn-0 environment errors
or timeouts) triggered the single-turn `host_completion_length < 1` assertion:
`FunctionalMappingError: P32 grouped reverse requires nonempty prompt/completion on every rank`.
Because zero-completion rows have `action_mask=0`, they contribute zero loss
and zero gradient. See the immutable incident directory
`canon-zero-tim/evidence/p58_k11_deepswe_empty_completion_incident/`.

P58.28 repairs only this assertion. `_p32_group_spec` retains a default-false
empty-completion admission; the validated P34 DeepSWE branch is the only
caller that opts in. Prompt validity and at least two real tokens remain hard
requirements. No fake token is inserted and no row is dropped or resampled.
The exact K11 DP8 vector is covered by regression, and prompt-only rows are
proven to have zero forward outputs and zero reverse cotangent contribution.
An admitted batch prints `[P34.EMPTY_COMPLETION] ...
semantics=zero-loss-zero-gradient`.

The local work is based on operator parent
`9f6b9c7eb6c32792604a966a7c0b8d9efa4072aa`. P34 static, the 409/409 flag
audit, and the complete pinned-image gate pass; its terminal marker includes
`p34_empty_completion=2`. The repaired target has not run. Do not launch from
this dirty worktree. After separate commit/push and matching-image approval, fetch
the final clean remote readback SHA and rerun Attempt-0. Require the K11
strict A=B=C receipts plus segmented backward and the first optimizer commit.
See `phases/p58-28-k11-empty-completion-reverse.md`.

## START HERE — K10 passed strict rollout alignment, then hit a workload interface mismatch

K10 is not another rollout, TiTO, dataset, topology, or alignment failure.
Source `0e954153cdfd21ee79ebf57eaa6afb4bf273aff0` completed all 128
multi-turn R2E trajectories, produced 404,028 action tokens, finished
Rescore-B, and passed strict Step-0 pre-alignment with A-B=0 and B-C=0. It
also proves the P58.26/K09 startup repair on the real DP8xTP8 target.

The run stopped before segmented forward/backward because the generic DP
adapter reads `workload.name`, while `DeepSWEWorkload` historically stored the
same signed identity only as `contract_name`. The local P58.27 repair adds a
read-only `name` property returning `contract_name`; it does not add a second
serialized recipe field. Tests cover every registered DeepSWE contract and
the real P58 4096/16384 token-width call.

Local validation on operator parent
`98d102eb27fe05fcee327688d0aa6d236b32be4a` passes P34 static ten
suites, focused DeepSWE 6/6, flag audit 409/409 with `changed_names=0`, and
the complete pinned P58 gate with `deepswe_workload_identity=1` and
`P58_EXACT_IMAGE_CPU_PASS`. This is not a repaired target PASS. After separate
publication and launch approval, a fresh Attempt-0 must preserve the K10
strict alignment, cross segmented forward/backward, and produce the first
valid optimizer commit. See `phases/p58-27-k10-workload-identity.md` and the
immutable incident directory
`canon-zero-tim/evidence/p58_k10_deepswe_workload_attribute_incident/`.

## START HERE — K09 is a pre-rollout Python scope failure

K09 is not a model, TiTO, R2E dataset, or DP8xTP8 admission failure. Source
`0b62b6bbd3d9fa44268c7640047d4b60047cb4d5` reached all of these receipts:

```text
[DEEPSWE.TITO] ADMISSION_PASS ... retokenize_sampled_tokens=0
[P34.DATASET] CLEAN_DATA_PASS source_rows=4578 filtered_rows=1012 ...
[P34.DEVICE_INVENTORY] PASS devices=128 ...
[P34.TOPOLOGY] PASS rollout_devices=64 trainer_devices=64 ...
Rollout Mesh: dp=8,tp=8
Train Mesh: dp=8,tp=8
```

It then read `P58_Q4_TP4_TRAJECTORY_REPLAY` during shared `ClusterConfig`
construction even though that diagnostic name had only been assigned inside
the one-host branch. The local P58.26 repair binds it to `False` before that
branch and short-circuits replay geometry on `ONEHOST_SMOKE`. The regression
executes both the full-mode negative path and one-host positive path, and a
scope audit rejects future one-host uppercase names escaping unbound.

The repair is locally validated on exact operator parent
`0d224e4a0e8c278f1bf9f699af235fdea83ef327`, including both latest shared
Qwen explicit-mesh resharding changes. P34 static passes ten suites, focused
P58 passes 49/49, script contract passes 10/10, and the flag audit passes
409/409 with `changed_names=0`. The complete pinned-image gate exits zero with
`P58_EXACT_IMAGE_CPU_PASS ... regressions=1` on image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
This is local source/image admission; it is not a published source SHA or a
128-chip target result.

Do not claim the full target is healthy from this source repair. After source
publication, matching-image publication, capacity admission, and separate
launch approval, use a fresh Attempt-0 run. It must progress past the former
line 1804 and produce a real TiTO continuation plus a complete 128-row Step-0
journal before any numerical conclusion. K09 has no trajectory, backward,
optimizer commit, checkpoint, or resumable training state. See
`phases/p58-26-k09-full-startup-scope.md` and the immutable incident package
under `canon-zero-tim/evidence/p58_k09_deepswe_unbound_variable_incident/`.

## START HERE — default full YAML now fail-closes on TiTO

For every P58 Native, Native+IS, Zero, and Zero-HP render, require both the raw
container environment and JobSet provenance before any launch:

```text
CANON_P34_DEEPSWE=1
canon.zero-tim/token-transport=tito
```

Do not add a separate optional TiTO switch.  The existing DeepSWE workload
identity is the single selector, and the renderer rejects a missing/wrong
label or raw value.  `prepare_deepswe_zero_hp_full.sh` must finish with
`transport=token-in-token-out launch=not-executed`.  These fields are part of
the Native/Zero recipe signature, so both arms use identical token transport.

Local construction on source
`18f29c56daf471cc0ac011396d7c7a09f35d695b` plus its recorded dirty diff is
green: focused 50/50, P34 static 10 suites, flag audit 409/409, and the pinned
image gate ends in `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 ...
regressions=1`.  Direct-v5p evidence
`p58s25titoctl_20260830t0713z` emitted one admission and 23 continuation
receipts, then returned `EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS` with exact
A=B=C over 2,413 action tokens and controlled exit 42 before backward.  Bundle
SHA-256 is
`a68925aa95aaeddcdc9f3f0be625aa92418b221959e1ef11cdc8c7f0ebbbcb35`.
This proves only DP1xTP4 TiTO/alignment, not backward, TP8, Pathways, optimizer
commit, or DP8xTP8 production.

The operator branch advanced while the hardware evidence ran.  Publication
preparation preserved the edits in a named stash, cleanly rebased onto exact
operator parent `cd32949e9b63b927e99f3cfba724f4f5f6d03cda`, and restored
without conflict.  A final non-overlapping Qwen3 embedder-sharding commit then
advanced the publication parent to
`e89272d1d6c99b8f3c5014f0974b4fe57f2a4156`; the P58 delta was rebased again
and the focused, P34, flag-audit, render, and complete exact-image gates passed
there.  The shared M15 exact-token runtime remains mutually exclusive with
DeepSWE and does not change this selector.  Do not launch from the old
evidence SHA or a local dirty tree: fetch the final remote readback SHA
containing this entry and verify it cleanly.  Image publication and Kubernetes
launch remain separately gated.

## START HERE — P58.25 makes TiTO common to every DeepSWE arm

Do not preserve or reintroduce the historical multi-turn text/re-tokenize
path for Native. DeepSWE is token-in/token-out: Native, Native+IS, Zero,
Qwen3-32B, diagnostics, and one-host all carry the exact sampled assistant
token IDs into the next request. R2E environment observations originate as
text and are encoded once; those IDs are then reused. The continuation request
must be an integer token list with `apply_chat_template=False`.

The startup log must contain exactly one line matching:

```text
[DEEPSWE.TITO] ADMISSION_PASS contract=<contract> arm=<arm> mode=token-in-token-out retokenize_sampled_tokens=0
```

An ordinary P58 training rollout must also contain one or more:

```text
[DEEPSWE.TITO] CONTINUATION turn=<positive> prompt_tokens=<positive> sha256=<64-hex>
```

`90_run.sh` rejects a missing/duplicate admission receipt and rejects an
ordinary P58 run that never exercises a multi-turn continuation. Native and
Zero are now compared on one identical TiTO transport. Their differences are
the registered numerical/alignment/IS treatments only.

The source concern was rebased onto exact operator parent
`509d3866b39228ce7df29d4eb3e5394591c69de0`. Its collector overlap with the
upstream observer-only M15 token verifier was reconciled by sharing the strict
reconstruction helper while keeping M15 observer-only and DeepSWE exact input
separately admitted. Focused host, P34 static, 409/409 flag-audit, and
post-rebase digest-pinned complete image gates pass. The image gate observed a
real focused continuation receipt and ended in
`P58_EXACT_IMAGE_CPU_PASS ... regressions=1`. Use the final remote readback SHA
that contains P58.25; do not substitute the parent SHA. Real target evidence
remains pending. Source commit/push was explicitly authorized, but image
publication, render/apply, and TPU/Kubernetes launch remain separately
unapproved. See `phases/p58-25-deepswe-tito.md`.

## K04 START HERE — use JobSet-level exclusive topology

K03 is immutable infrastructure `INCONCLUSIVE`. Kueue admitted the JobSet and
the CPU head started, but `vpod.kb.io` rejected indexed TPU worker followers
because the worker Pod template had no
`cloud.google.com/gke-nodepool`. No rollout, trajectory, trainer program,
backward, optimizer commit, or checkpoint was produced.

K03 put JobSet's exclusive-topology annotation on the worker Pod template.
That is the wrong scope: the controller needs it at JobSet metadata to bind
the indexed Job and its followers to the Kueue-selected or NAP-created pool.
For K04 use either the existing Kueue sentinel `tpu-v5p-slice` or an actual
pool verified against the selected flavor, and render only through:

```bash
bash canon-zero-tim/tasks/v1-system-optimization-workload-rollout/prepare_deepswe_zero_hp_full.sh \
  <approved-40-character-sha> \
  <matching-registry-image@sha256:digest> \
  <fresh-output.yaml> \
  <fresh-run-id> \
  <worker-nodepool-or-kueue-sentinel> \
  <model-pvc>
```

The wrapper is render-only and must emit
`V1_DEEPSWE_ZERO_HP_RFULL_READY ... launch=not-executed`. Require exactly one
top-level `alpha.jobset.sigs.k8s.io/exclusive-topology` annotation whose value
is `cloud.google.com/gke-nodepool`, and no copy on the worker Pod template.
Sentinel renders must omit a literal nodepool selector while retaining
accelerator `tpu-v5p-slice` and topology `4x4x8`; explicit-pool renders must
retain that real value exactly. Require server-side dry-run before a
separately approved apply. Do not hand-edit K03 YAML. Dynamic inference-package
discovery remains for image-layout drift, but the client image must still be
immutable and source-matched.

## PUBLICATION CHECKPOINT — optimized Qwen3-4B Zero/full wiring

The implementation commit is
`fb178803d53ff562cefdfdc8e7b3fac3563d9d6e`. It is a descendant of the
fetched operator tip `4ce03fad6e10466acece308a3fe05b41af3825c2`; the final
publication commit is the remote tip that contains this implementation. A
remote executor must fetch `yuxzhang/canon-zero-tim`, record its exact
40-character readback SHA, and prove that it contains the implementation
commit. Do not substitute either hash by hand in an old YAML.

The final fixed-image construction gate exits zero with
`P58_EXACT_IMAGE_CPU_PASS ... trajectory_replay_b2g2=1
system_optimization=1 ... m15_token=1 regressions=1`. Rebase integration
preserves upstream M15 runner patch 36 and installs the P58 observer as patch
37; the combined runner manifest SHA-256 is
`dae6dfa8a45bfd0a34b41baa9ec7c258229e8824c427a2fb863b620add074f98`.
The focused P58 observer probe passes 8/8 and the upstream M15 target-carrier
and three-round contracts pass 21/21 and 3/3. Flag audit is 408/408 with 12
registered non-settable markers.

A clean render from the implementation commit produced manifest SHA-256
`61b837dbc9915373c931eebfbbee0fc67c75f9726d7db3893b108c67eac1331c`
and `launch=not-executed`. That manifest was a local verification artifact,
not a launch input. Render again from the final remote readback. The resolved
production identity is Qwen3-4B-Instruct-2507, Zero/full, DP8xTP8 per role,
B8xG16, 1,000 updates, TPU-resident optimizer, and the latest registered
system-optimization tuple described below. `CANON_DP_COLLECTIVE_REDUCE`
remains absent because DP8 target certification does not exist. No image was
published and no Kubernetes/Pathways/TPU job was launched by publication.

## START HERE — future Zero-HP full training must use the P74-enabled wrapper

This section defines the production wiring for the next DeepSWE Qwen3-4B
Zero/full/HP training attempt. It does not override the current P58.19 seam
localization queue: the selector-absent 1,000-update full run remains blocked
until the strict A-B prealignment case is resolved and a new launch is
separately approved.

Status is `IMPLEMENTATION COMMITTED / PINNED-IMAGE CPU PASS / PUBLICATION
READBACK REQUIRED / TARGET NOT RUN`. After fetching and reading back the final
published clean SHA, render with:

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  <exact-local-image-sha256-id>

bash canon-zero-tim/tasks/v1-system-optimization-workload-rollout/prepare_deepswe_zero_hp_full.sh \
  <approved-40-character-sha> \
  <matching-registry-image@sha256:digest> \
  <fresh-output.yaml> \
  <fresh-run-id> \
  <worker-nodepool> \
  <model-pvc>
```

The wrapper never launches and must emit
`V1_DEEPSWE_ZERO_HP_RFULL_READY ... launch=not-executed`. The resolved trainer
environment must retain the full P58 numerical protection and add the
registered system tuple:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1        # derived compatibility alias
CANON_P67_P66_VMA_P59_ONLY=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_P63_OVERFLOW_SAFE_CLIP=1
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
```

`CANON_DP_COLLECTIVE_REDUCE` must remain absent. P74 is source behavior, not a
new switch. Native raw, Native+IS, ordinary non-HP Zero, three-update,
checked-VMA diagnostic, and seam-localization arms must keep all four new
receipt/scan selectors absent.

The exact-image terminal is
`P58_EXACT_IMAGE_CPU_PASS ... system_optimization=1 ... regressions=1`; this
is construction evidence only. A future target must still pass strict A=B=C,
16-group first-update admission, coherent AdamW `0 -> 1`, all 1,000 commits,
performance/XProf receipts, and the P58 final classifier. Do not reuse a
previous P58 YAML, run ID, image tag, or evidence root.

## COMPLETED LOCAL DIAGNOSTIC — P58.23 B2xG2 optimized one-host backward

P58.23 is complete.  P58.22 already proved a real R2E
rollout and strict A=B=C for Qwen3-4B-Instruct-2507 on direct DP1xTP4.  Do not
spend another 1–2 hours compiling the serial reference.  The accepted carrier
replays one immutable strict-exact real prompt repeated as two physical
groups, with two generations per group, through the current
P28/P30/P71-forward optimized trainer path.

Hard geometry: global `B=2`, `G=2`, four trajectories, K=`2048+512=2560`,
`batch_size=2`, `mini_batch_size=2`.  Memory microbatches may remain 1; they
do not make this a batch-size-one run.  Both groups have rewards `[1,0]`.
P59 is intentionally off on DP1 and remains a DP8 target-only claim.

Replay source:

```text
/mnt/disks/tunix-data/deepswe-replay-sources/p58-q4-b2g2-k2560-v2
manifest 482d7934a95207d0d77bb4857fbb200d7b367cbf437dda6585937b20909afa8f
journal  091a9273c2067876fbee1996ee853e3c8e861352e307cd5fb94fea2563aec456
```

The successful fresh label was:

```text
p58s23optb2g2g_20260830t0132z
```

Its immutable artifact root is
`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s23optb2g2g_20260830t0132z`;
the return-bundle SHA-256 is
`7d33ee791146d2309c16866d8e30f15f0f012e05e88f6c795b587938f973f795`.
The classifier returned strict A=B=C over 1,254 action tokens, exact finite
nonzero repeated gradients, device-resident unchanged optimizer state, and
zero commits.  Its compiled profiled repeat took 12.418 seconds and peak HBM
was 52.5 GiB.  This is the accepted P58.23 receipt; do not rerun merely to
replace the label.

If a deliberate reproduction is separately approved, use a fresh label only:

```bash
P58_ONEHOST_ALLOW_DIRTY=1 \
  bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_trajectory_replay_docker.sh \
  <fresh-label>
```

The wrapper enforces a 1,800-second cold bound and compilation cache
`/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-systemopt-b2g2-k2560`.
Require four nonzero advantages, strict A=B=C, finite nonzero backward,
unchanged parameters/optimizer, zero commits, and a verified return bundle.
The duplicate prompt is intentional and proves only physical B=2/RLOO/train
shape; it is not prompt-diversity evidence.  Never substitute replay v1: its
Coverage group is historical alignment-red evidence.  Timeout is
`ZERO_TIM_BACKWARD_INCOMPLETE`, not PASS.  Replay validates the trainer path
but does not itself run R2E/decode, TP8, Pathways, P59, or an optimizer update.
See `phases/p58-23-qwen4b-systemopt-b2g2.md`. The next executor action is to
fetch and read back the final operator SHA, prove it contains implementation
commit `fb178803d53f`, and only then prepare the separately approved TP8
promotion. Do not launch or claim TP8 from this local receipt.

## 2026-08-28 UTC — DeepSWE P58.19e incident intake (`canon-p58-seamcoarse-full-p58s19e`, 128 TPU)

### Incident Summary
Target run `canon-p58-seamcoarse-full-p58s19e` executed Step 0 multi-turn rollout on 128 TPU v5p (33 Pods):
- **Patch 34 Single-Round Dynamic Budget Extension (Verified PASS)**: Scaled from 635 to **1,790+ Seam and Tail Observer Records** (`arm=A`) with 1,007+ request journals without continue-decode or tensor-strata exceptions.
- **Deep Multi-Turn Tool Calling (Verified PASS)**: Context lengths reaching 3,769+ tokens, tool calls deepened up to step 10 across 128 RepoEnv sandboxes.
- **Fatal Error**: At step 0 rollout, the accumulated `.npz` records reached 4.3 GiB, crossing the registered `_SEAM_MAX_BYTES` (4 GiB) limit in `p38_seam_capture.py` -> `RuntimeError: P38 seam evidence exceeded its registered output byte bound`.
- **Sealed Incident Package**: `evidence/p58s19e_byte_bound_incident/` (`INCIDENT_REPORT.md`, `RAW_ERROR.log`, `run.log`, `env.sh`, `SHA256SUMS`).

### Reconciled action for p58s19f
Raise `_SEAM_MAX_BYTES` to 16 GiB (or 32 GiB) per round to accommodate the full completion of all 128 multi-turn trajectories across 3 sequential rounds.

## 2026-08-28 UTC — P58.19e per-round observer budget repair (local only)

The sealed `p58s19d` failure is an instrumentation-capacity failure, not a
model, R2E, alignment, or training result.  It proves the continue-decode
bypass and `[1686,4096)` coverage with 635+ records, then the P58 observer
exhausted its cumulative 1 GiB limit during round 0.

The local repair on base `af006872b64c2d6327588b4d4cef757242ddc222`
derives 4 GiB per diagnostic round and adds append-only runner patch 34 after
the upstream M15 replay-provenance patch 33.  The P58 patch
patch extends the already shipped monotonic round-budget reset to exact
`p58-seam-v1`; M15 behavior remains admitted, foreign profiles remain no-op,
record indices never reset, and no records are deleted.  Postflight requires
all six seam/tail round-start receipts.  Model, data, sampling, loss,
alignment, checked-VMA, continue-decode, backward, and optimizer contracts are
unchanged.

Pinned-image assembly installs all 37 Qwen3-4B files.  The P58 and M15
dynamic probes both pass, and the complete pinned-image gate emits
`P58_CONTINUE_DECODE_OVERLAY_PASS cases=5 tensor_capture=standard-only
round_budget=p58+m15` followed by `P58_EXACT_IMAGE_CPU_PASS ...
continue_decode_observer=1 ... m15_token=1 regressions=1`.  This work is not
committed or pushed, and no image, Kubernetes object, Pathways run, or TPU
target was created.  A fresh 128-chip rerun remains separately approval-gated.

## 2026-08-28 UTC — DeepSWE P58.19d incident intake (`canon-p58-seamcoarse-full-p58s19d`, 128 TPU)

### Incident Summary
Target run `canon-p58-seamcoarse-full-p58s19d` executed Step 0 multi-turn rollout on 128 TPU v5p (33 Pods):
- **Continue-Decode Observer Bypass (Verified PASS)**: Commit `cf56b21a` containing `32-tpu-runner-p58-mixed-program-path.patch` successfully bypassed `continue_decode` without throwing `expected=standard actual=continue_decode`.
- **Target Seam Window Coverage (Verified PASS)**: Multi-turn tool execution (`search`, `file_editor` up to step 4) covered bands `[12, 15]` (`3072..4095`), emitting over **635 Seam Observer Records** (`arm=A`) and **Tail Observer Records** with valid SHA256 checksums.
- **Fatal Error**: At step 0 rollout, the accumulated `.npz` records crossed the registered `_SEAM_MAX_BYTES` (1 GiB) limit in `p38_seam_capture.py`:
  ```text
  RuntimeError: P38 seam evidence exceeded its registered output byte bound
  ```
- **Sealed Incident Package**: `evidence/p58s19d_byte_bound_incident/` (`INCIDENT_REPORT.md`, `RAW_ERROR.log`, `SHA256SUMS`).

### Reconciled action for p58s19e

Use the scoped per-round 4 GiB repair above; a cumulative 4 GiB value alone is
insufficient for three rounds.  Source publication, matching image
publication, render/server dry-run, and launching
`canon-p58-seamcoarse-full-p58s19e` each remain separately approval-gated.

## 2026-08-28 UTC — P58.19d continue-decode observer repair (published)

The repair was rebased onto operator source
`57d9ab8e25de3b2404e983e9a139d78b151a58f8` and published as
`ed8ce99a0fa4187e0619237e071990b90d453d72`.  The sealed `p58s19c`
incident proves the repaired `[1686,4096)` window emitted 113 seam records,
then the observer rejected `program_path=continue_decode` while configured
for standard-path tensor capture.  Its `RAW_ERROR.log` is a short incident
excerpt rather than a complete raw run; do not infer a three-round result.

Do **not** disable `CANON_CONTINUE_DECODE`.  The signed P58 high-performance
carrier requires value 8.  The local repair instead admits continue-decode
only for exact `p58-seam-v1`: scheduler chronology remains visible, incident
and tensor payloads are skipped, an exact
`CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS ... tensor_capture=0` receipt is
printed, and the hook returns before candidate construction.  Standard path
remains the sole tensor-strata source.  Other profiles and unknown program
paths keep the hard error.  Postflight requires the bypass receipt for P58.19
coarse and rejects it everywhere else.

The installed-overlay probe and complete pinned-image gate pass.  Terminal
receipts are `P58_CONTINUE_DECODE_OVERLAY_PASS cases=5
tensor_capture=standard-only` and `P58_EXACT_IMAGE_CPU_PASS ...
continue_decode_observer=1 ... regressions=1`; all 37 Qwen3-4B overlay files
match MANIFEST.  Focused P58 suites pass 52/52, P34 static passes 10 suites,
and the deterministic flag audit passes 394/394/394 with the new runtime name
registered as a marker rather than a settable flag.  The implementation was
read back from the operator branch after a fast-forward push.  No image,
Kubernetes object, Pathways run, or TPU target was created.

## 2026-08-28 UTC — DeepSWE P58.19c incident intake (`canon-p58-seamcoarse-full-p58s19c`, 128 TPU)

### Incident Summary
Target run `canon-p58-seamcoarse-full-p58s19c` executed Step 0 multi-turn rollout on 128 TPU v5p:
- **Observation Window Coverage (Verified PASS)**: Widened window `[1686, 4096)` successfully captured **113 seam records** (`p38_seam_records=113`), fixing the previous `p58s19b` zero-record failure.
- **Fatal Error**: TPU Runner entered continuous decode `_execute_continue_decode`, invoking `_p38_serving_begin(program_path="continue_decode")`. `_p38_serving_begin` raised:
  ```text
  RuntimeError: P38 serving capture reached an unexpected program path: expected=standard actual=continue_decode
  ```
- **Sealed Incident Package**: `evidence/p58s19c_continue_decode_incident/` (`INCIDENT_REPORT.md`, `RAW_ERROR.log`, `SHA256SUMS`). Full multi-pod logs mirrored at `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/canon-p58-seamcoarse-full-p58s19c/attempt-0`.

### Reconciled action before rerun

Preserve `CANON_CONTINUE_DECODE=8` and use the scoped P58.19d observer repair
above.  Disabling it changes the signed carrier and is not an admissible
repair.  A rerun remains separately approval-gated after source and matching
image publication.

## 2026-08-28 UTC — P58.19c local seam-window coverage repair

`p58s19b` proves only that the observer initialized but emitted zero records:
`init=1 records=0 classifier=1`. Its sealed `RAW_ERROR.log` is a 26-line
incident excerpt, not the complete run log, so the earlier statement that all
initial prompts were shorter than 3,072 is not promoted to a runtime fact.
The immutable incident files remain unchanged.

The local repair replaces the zero-hit `[3072,4608)` window with the
evidence-derived `[1686,4096)` window and serving-capture strata
`1686,2512,3072,3584,4096`. This one window contains every known first-red
prefix available to this lane: 2,513 and 3,715 from `p58z07`, plus 3,438,
3,880, and 4,032 from the P58.18 ON-A/OFF/ON-B evidence. It does not change
the model, data, sampling, numerical treatment, geometry, backward, or
optimizer contract. Source publication required a separate user-approved
commit/push. That approval is complete: implementation
`b231ef39d0d2f5c270561f9acd1a26a6b0503654` is the P58.19c source
publication for this delivery. No target retry is authorized by this handoff.

Construction validation is complete on source base
`b231ef39d0d2f5c270561f9acd1a26a6b0503654`: focused renderer/profile/
classifier tests pass 45/45, P34 static passes 10 suites, flag audit passes
394/394/394, diff hygiene passes, and the complete digest-pinned image gate
ends in `P58_EXACT_IMAGE_CPU_PASS ... coarse_seam=1 ... regressions=1`. The
bare host lacks optional `metrax`, so its environment-contract import is not
claimed as a PASS; the same contract passes in the pinned dependency image.
No Pathways/TP8 target has been run. A separately approved retry must return
the complete raw log and request/scheduler journal, not another short excerpt,
and must show observer records plus one classification for every round.

## 2026-08-28 UTC — DeepSWE P58.19b incident intake

### Incident Summary (`canon-p58-seamcoarse-full-p58s19b`, 128 TPU)
Target run `canon-p58-seamcoarse-full-p58s19b` failed at the Step 0 postflight gate:
- Error: `FATAL: P38 seam observer contract failed: init=1 records=0 classifier=1`
- Proven boundary: the configured `[3072,4608)` standard-path window emitted
  no seam record in this attempt. The incident's prompt-length explanation is
  a hypothesis because the returned artifact does not contain the full log or
  per-request scheduler journal.
- Sealed incident artifacts: `evidence/p58s19b_seam_observer_contract_incident/` (`INCIDENT_REPORT.md`, `RAW_ERROR.log`, `SHA256SUMS`).

### Action Required After Publication
Render a fresh JobSet with the registered selector and verify the resolved
`[1686,4096)` tuple. Image publication and Kubernetes apply remain separately
approval-gated.

---

## 2026-08-28 UTC — P58.19 coarse seam localization published

The implementation commit is
`f58a97748a8895835fba4944f5c5a34ba8bee352` on
`yuxzhang/canon-zero-tim`. Immediate post-push readback matched local HEAD and
the operator remote-tracking ref with ahead/behind `0/0`. The publication
ledger is a later documentation-only tip. Fetch the current operator tip into
a clean checkout, prove that it contains the implementation commit, and
record that checkout's actual 40-character HEAD; do not silently substitute
an older P58 tip.

P58.18 is closed by the sealed `p58aba01` result: ON-A, OFF, and ON-B all
returned finite A-B RED, exact B-C, controlled exit, and zero backward/commit.
The only supported conclusion is `CHECKED_VMA_NOT_SUFFICIENT`.  Do not ship a
P67-only repair and do not spend another 1,000-update training run on the same
pre-backward failure.

P58.19 prepares one default-off diagnostic selector,
`CANON_P58_SEAM_LOCALIZATION=coarse`.  It renders exactly one 128-chip JobSet,
not three JobSets: rollout DP8xTP8 and trainer DP8xTP8 remain disjoint, while
the same process executes three sequential frozen-weight Step-0 rounds.  Each
round keeps Qwen3-4B-Instruct-2507, the reviewed 1,012-task list, B8xG16 (128
trajectories), 16,384 response tokens, 50 turns, temperature/top-p/top-k
`1.0/1.0/0`, seed 42, concurrency 128, fixed lm-head, continue-decode 8,
prefix cache off, exact B-C, and the production Zero-HP numerical tuple.  VJP,
backward, optimizer commit, and checkpoint advancement are unreachable.

The selector derives the complete observer/durability tuple.  It records
bounded coarse fingerprints for every Transformer block input/output, final
norm, and the terminal logprob path over logical KV prefixes `[1686,4096)`.
Never hand-edit subordinate P38/M15 fields and never combine this selector
with Native, three-update, warning-only, checked-VMA diagnostic, or partial
observer settings.  Selector absence leaves production P58 Zero/full
unchanged.

After this published source is paired with a separately approved digest-pinned
image, an executor starts from a clean checkout of the exact published SHA and
runs only the render wrapper:

```bash
git merge-base --is-ancestor \
  f58a97748a8895835fba4944f5c5a34ba8bee352 HEAD
export P58_EXPECT_SOURCE_SHA="$(git rev-parse HEAD)"
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_coarse_seam_localization.sh \
  <fresh-run-id> \
  <matching-image@sha256:...> \
  <worker-nodepool-or-auto> \
  /tmp/p58-coarse-seam.yaml
```

The wrapper refuses a dirty checkout, SHA drift, a mutable image tag, or an
existing output file.  It contains no `kubectl` call.  Image publication,
server dry-run, apply, and target monitoring remain separate explicit
approvals. Do not launch from a development worktree.

A return is complete only when the persistent run root contains all three
round seals and the aggregate result:

```text
run.log
pre_alignment.jsonl
debug/*.trajectories.jsonl.gz
p58-seam.classification.json
p38_gcs_rounds/000000/ROUND_COMPLETE.json
p38_gcs_rounds/000000/p58-seam.round.classification.json
p38_gcs_rounds/000001/ROUND_COMPLETE.json
p38_gcs_rounds/000001/p58-seam.round.classification.json
p38_gcs_rounds/000002/ROUND_COMPLETE.json
p38_gcs_rounds/000002/p58-seam.round.classification.json
```

Return the full persistent root or its verified GCS run root, not a copied log
excerpt.  Every round is classified, archived, uploaded, read back, and ACKed
before the next begins.  Missing joins/rounds, B-C drift, nonfinite values,
observer endpoint drift, or any backward/commit evidence is INCONCLUSIVE or
FAIL, never PASS.  The aggregate classifier requires one common first-red
coarse signature across all three rounds; otherwise preserve the evidence and
refine only the common interval if one exists.

Local host tests cover selector truth tables, renderer/profile rejection,
classification, and durability wiring.  The existing DP1xTP4 one-host seam
carrier does **not** execute the new Qwen3 TP8 layer observer and therefore
cannot certify observer neutrality for the exact DP8xTP8 carrier.  A real
v5p/Pathways target has not been launched.  The complete local
dependency-image gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ...
checked_vma_aba=1 coarse_seam=1 ... regressions=1`; this is construction
evidence only.  P58.19 cannot certify backward, optimizer correctness, full
training, convergence, or general Zero-TIM readiness.

## 2026-08-28 UTC — P58.18 triplicate complete: Case 2 (CHECKED_VMA_NOT_SUFFICIENT) sealed

The three independent exact-geometry Step-0 diagnostic JobSets (`p58aba01`: `ON-A`, `OFF`, `ON-B`) on 128 TPU chips have all completed with Controlled Exit 42:

- **ON-A (`canon-p58-vmaon-full-p58aba01-ona`)**: 128 trajectories (3 solved / 120 complete). Alignment: $S_{decode} - S_{prefill} = 47,645$ B (21,717 elements), $S_{prefill} - T_{old} = 0$ B. Verdict: `A_B_RED_WITH_CHECKED_VMA_ON`.
- **OFF (`canon-p58-vmaoff-full-p58aba01-off`)**: 128 trajectories (6 solved / 118 complete). Alignment: $S_{decode} - S_{prefill} = 39,787$ B (18,068 elements), $S_{prefill} - T_{old} = 0$ B. Verdict: `A_B_RED_WITH_CHECKED_VMA_OFF`.
- **ON-B (`canon-p58-vmaon-full-p58aba01-onb`)**: 128 trajectories (3 solved / 120 complete). Alignment: $S_{decode} - S_{prefill} = 36,323$ B (16,653 elements), $S_{prefill} - T_{old} = 0$ B. Verdict: `A_B_RED_WITH_CHECKED_VMA_ON`.

### Wave-level classification result:
```text
P58_CHECKED_VMA_ABA_CLASSIFICATION verdict=PASS decision=CHECKED_VMA_NOT_SUFFICIENT backward=0 optimizer_commits=0
```

### Interpretation & Next Action:
1. **Case 2 triggered**: Checked-VMA is not sufficient to explain the decode-vs-prefill divergence in DeepSWE Qwen3-4B.
2. **Do NOT ship a P67-only repair**: Since `checked_vma=off` also produces A-B RED with exact B-C, checked-VMA is not sufficient. This does not prove the seam is independent of checked-VMA.
3. **Evidence sealed**: Complete self-hashed package is sealed in `evidence/p58aba01_checked_vma_aba_wave/`.
4. **Next phase**: Promote exact-geometry decode/prefill seam replay on DeepSWE to isolate whether rotary embeddings, ragged paged attention kernel, or tensor parallel communication is the source of the token logprob divergence.

---

## Historical — 2026-08-28 P58.18 triplicate preparation (complete)

This section supersedes the single checked-VMA-off launch instruction below.
The next numerical experiment is three independent exact-geometry Step-0
JobSets: logical `ON-A/OFF/ON-B`. They use identical published source,
digest-pinned image, Qwen3-4B recipe, clean data, geometry, seed, and artifact
contract. Only `CANON_P58_CHECKED_VMA_DIAGNOSTIC=on|off` differs. All three
exit before VJP/backward/optimizer commit; none is a training run.

The operator requested concurrent submission. This needs 384 TPU chips in
aggregate, three CPU head nodes (required anti-affinity), and up to 384 R2E
sandboxes. At the signed sandbox request that is 768 CPU and 1,536 GiB of
sandbox memory, excluding head containers. If Kueue or the CPU pool cannot
admit all three together, preserve Pending/admission evidence; do not silently
call a staggered run temporal ABA evidence.

After this approved source commit/push and a separately approved matching image
publication, the remote executor must fetch the final tip, verify a clean
checkout, and run the render-only wrapper:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_aba_wave.sh \
  <fresh-wave-id-at-most-12-chars> \
  <matching-image@sha256:...> \
  <worker-nodepool> \
  /tmp/<fresh-p58-aba-output-dir>
```

The wrapper creates and independently verifies:

```text
jobsets/01-on-a.yaml
jobsets/02-off.yaml
jobsets/03-on-b.yaml
wave-render-receipt.json
wave-verify.json
```

It contains no `kubectl` call. Inspect `wave-verify.json` for `PASS`, source
and image identity, three unique JobSet names/roots, `aggregate_tpu_chips=384`,
`backward=0`, and `optimizer_commits=0`. Server dry-run and applying the three
files remain separate, explicitly approved operations. Applying all three
objects makes them eligible concurrently; actual concurrency must be proved
from Kueue admission and start timestamps, not inferred from submission.

Return each complete persistent run root and its
`p58_checked_vma_{on|off}.classification.json`. Then run the local aggregate
classifier against the three returned classifications and the original
`wave-verify.json`:

```bash
python3 canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/classify_p58_checked_vma_aba_wave.py \
  --wave-verify <wave-output>/wave-verify.json \
  --on-a <on-a-root>/p58_checked_vma_on.classification.json \
  --off <off-root>/p58_checked_vma_off.classification.json \
  --on-b <on-b-root>/p58_checked_vma_on.classification.json \
  --output <fresh-output>/p58_checked_vma_aba.classification.json
```

The decisive pattern is ON RED / OFF exact / ON RED with B-C exact in every
arm. RED/RED/RED means checked VMA is not sufficient. Different ON outcomes
are inconclusive. Do not require cross-run token identity, average away a
nonreplicating ON arm, loosen B-C, or continue into backward.

Local construction validation is complete in the dependency-bearing pinned
CPU image: the full suite ends with `P58_EXACT_IMAGE_CPU_PASS ...
checked_vma_diagnostic=1 checked_vma_aba=1 ... regressions=1`. This does not
authorize or substitute for the three real 128-chip target arms.

### Resolution after the triplicate returns

Judge each arm with its committed classifier before choosing a repair. Never
infer the result from the JobSet exit code or from a maximum delta alone.

#### Case 1 — ON RED / OFF exact / ON RED: repair P67 identity scoping

This pattern, with exact B-C in all three arms, reproduces checked VMA as the
causal discriminator. It does **not** justify leaving checked VMA disabled:
OFF is a diagnostic control, while P59 backward still needs the checked-VMA
ownership repair. The production fix is to stop using abstract-mesh shape as
the identity of a P59 pullback.

Implement one internal, non-user-settable P59 pullback-scope API. Enter it in
`tunix/rl/canonical_qwen3_adapter.py` only while `_p59_parallel_map` traces the
registered local pullback, and reset it in `finally`. The serving/decode and
prefill paths must never enter this scope, even when their live abstract mesh
is exactly `(data,model)=(8,8)` with both axes Manual. Do not add another
independently combinable environment flag and do not key the scope to DP/TP
sizes, axis names, `CANON_P59_RANK_PARALLEL_BACKWARD`, or the presence of a
trainer-shaped mesh.

Replace the topology-shaped P67 decisions at every checked-VMA consumer, not
only the first helper found:

- `canon-zero-tim/src/engine_shims/p22_pallas_matmul.py`: operand pcasts and
  output `ManualAxisType`;
- `canon-zero-tim/src/engine_shims/linear_p22xf.py`: the replicated-TP pmean
  path (and audit every `_p59_local_tp_context` caller so topology is not used
  as treatment identity);
- `canon-zero-tim/patches/tpu_inference/02-embed.patch`: embedding pmean;
- `canon-zero-tim/patches/tpu_inference/29-rpa-p66-vma-output.patch`: RPA
  output-shape VMA annotations.

The internal scope must fail closed on nesting/ownership drift and emit
trace-time receipts naming the P59 module and whether each consumer observed
the scope. Receipts are observation only; they must not add device reads or
JAX program boundaries.

Required repair gates, in order:

1. Host truth-table tests over identity=`serving|trainer-pullback`,
   data=`1|2|8`, and model=`4|8`: every serving case is scope-off; every
   registered trainer pullback is scope-on. Shape alone must never flip it.
2. Forced-device positive tests prove checked VMA still owns the P59 local
   pullback and its transpose collectives. DP1xTP8 and DP8xTP8 serving
   negatives prove no pcast, pmean, or RPA output annotation is activated.
3. Re-run the complete pinned-image and installed-shim gates. Preserve exact
   Native/Zero treatment isolation and the production selector-absent tuple.
4. Publish a matching source/image only with separate approval, then run one
   fresh exact-geometry `checked-VMA=on` Step-0/no-commit arm. Require strict
   A-B=0, B-C=0, 128 durable rows, and zero VJP/backward/commit.
5. Only after that fresh ON arm is exact may a selector-absent Zero-HP full
   job start. Its first strict Step-0 gate, checked-VMA backward receipts,
   finite nonzero gradients, first-update transaction, and checkpoint remain
   hard gates. Never resume a pre-repair checkpoint.

#### Case 2 — ON RED / OFF RED / ON RED: do not ship a P67 repair

Checked VMA is not sufficient to explain the seam. Freeze the triplicate and
promote exact-geometry decode/prefill seam replay. Join the first A-B mismatch
to its durable trajectory, verify token/action-mask and `i±1` shift controls,
then instrument the smallest existing seam around that call. Reuse the
lm-head/prefix-cache three-run methodology: two identical baseline replicates
around one single-selector treatment, immutable per-run roots, and a
classifier that requires B-C exact. Do not change precision, sampling, loss,
prefix cache, optimizer, or alignment gates to make the seam disappear.

#### Case 3 — both ON arms exact and OFF exact: baseline RED not reproduced

Do not claim a repair. Preserve the result as a non-reproduction and compare
source/image provenance plus per-arm trajectory identities against p58z07 and
p58z08. A full training run remains blocked until one fresh selector-absent
Step-0 strict gate is exact on the final source/image.

#### Case 4 — ON arms disagree, B-C is RED, or an arm is incomplete

The aggregate result is inconclusive/invalid. Do not average the ON arms and
do not reinterpret an infrastructure failure as numerical evidence. First
repair only the missing durability/admission/infra prerequisite, then rerun
the missing or nonreplicating zero-commit arm from a fresh root. Any B-C RED,
nonfinite value, malformed artifact, VJP/backward evidence, or optimizer
activity remains a hard classifier failure.

## 2026-08-27 p58z08 intake — wrong arm for P58.17; rerun only the discriminator

The operator branch is still at
`5d4f2fceb6996bb0a5e2149a21c8fd846d89dcb5` after a fresh pull. There is no
newer target log. The newly archived `p58z08` run used source
`395c0e0de8626c96e85457b997efddd2dd2dec48` and the ordinary job identity
`canon-p58-ds4b-zero-hp-full-p58z08`. It is useful analysis-grade evidence,
but it is **not** the P58.17 checked-VMA-off discriminator and must not be
reported as a failed discriminator.

The log proves a healthy Step-0 rollout/data path: 128 durable trajectories,
120 `SUCCEEDED`, five `MODEL_TIMEOUT`, three `MAX_CONTEXT_LIMIT_REACHED`, four
solved trajectories, two effective prompt groups, and 30 admitted nonzero
advantages. The strict pre-backward gate then stopped before VJP, backward,
AdamW, update, or checkpoint commit. Over `N_action=389067`,
`S_prefill_vs_T_old` is exact while `S_decode_vs_S_prefill` differs in 17,507
logprob elements / 39,031 serialized bytes. The first finite delta is
`0.02544403076171875` at logical prefix 2,141, turn 1, immediately after an
environment token; the later maximum is `9.499740600585938`.

The incident report's phrase "39,031 token logprob differences" is a metric
label error: 39,031 is the byte count; 17,507 is the differing-element count.
Keep the immutable report and use this correction in all later claims.

Most importantly, the raw log contains zero occurrences of all of the
following:

```text
CANON_P58_CHECKED_VMA_DIAGNOSTIC
P58_CHECKED_VMA_DIAGNOSTIC_CLASSIFICATION
zero-hp-vmaoff-precheck
CANON_P38_PRECHECK_ONLY
CANON_P38_PRECHECK_CONTROLLED_EXIT
```

It instead says that P59 checked VMA, the first-update gate, and P63 clipping
were enabled. This is an arm-selection failure: an ordinary 1,000-update
Zero-HP YAML was launched after P58.17 had already supplied a dedicated
Step-0 diagnostic renderer.

### Exact next run

Do not rerun `zero-hp-full`, do not hand-edit its YAML, and do not switch the
strict Zero lane to warning-only. Fetch the final operator tip, build/read
back the matching digest-pinned image as required by the run contract, then
use only the render-only wrapper documented below:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-current-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_off_diagnostic.sh \
  <fresh-short-run-id> \
  <matching-image@sha256:...> \
  <worker-nodepool-or-auto> \
  /tmp/p58-checked-vma-off.yaml
```

Before any separately approved apply, inspect the rendered artifact. Its name
must contain `zero-hp-vmaoff-precheck`; its labels must state
`diagnostic=p58-checked-vma-off`, `backward=0`, and `optimizer-commits=0`.
The runtime must print the selector and its derived tuple with checked VMA,
P66 alias, P67 scoping, first-update gate, and P63 all zero. Absence of any of
those receipts is a wrong-arm result and must stop before spending TPU time.

Return the complete run root and classifier. The decision is deliberately
binary:

- `A_B_EXACT_WITH_CHECKED_VMA_OFF`, with B-C exact: checked-VMA/P67 scoping is
  causally implicated. Repair P67 by keying the four VMA consumers to an
  explicit P59 pullback identity, never to abstract-mesh shape. Add serving
  negatives for DP1xTP8 and DP8xTP8 plus trainer positives, then rerun one
  checked-VMA-on Step-0 strict gate before any full training.
- `A_B_RED_WITH_CHECKED_VMA_OFF`, with B-C exact: checked VMA is not a
  sufficient cause. Preserve the result and promote exact-geometry
  decode/prefill seam replay, beginning at the first environment-to-action
  boundary; do not weaken the gate or change optimizer/loss/sampling.

Either outcome is a diagnostic result only. A full 1,000-update Zero-HP run
becomes eligible only after the selected repair has passed a fresh strict
Step-0 A=B=C gate. Image publication, Kubernetes dry-run/apply, and TPU launch
remain separately approval-gated.

## 2026-08-27 P58.17 exact-geometry selector — source published, target not run

Supersede the last sentence of the one-host section below: the production
renderer now has a dedicated default-off exact-geometry discriminator. Do not
launch another full 1,000-update retry and do not hand-edit the Zero-HP YAML.

The target is Qwen3-4B-Instruct-2507 on 128 TPU chips with disjoint rollout
DP8xTP8 and trainer DP8xTP8 roles. It keeps the 1,012-task clean whitelist,
B8xG16, 16K response, 50 turns, seed 42, 128 rollout concurrency, prefix cache
off, fixed lm-head, continue-decode 8, and full trajectory/debug journals.
One selector, `CANON_P58_CHECKED_VMA_DIAGNOSTIC=off`, atomically disables
checked VMA, its P66 alias, P67 scoping, first-update gate, and P63 clip. After
one real Step-0 rollout and strict A/B/C pre-alignment it exits code 42 before
fixed-head VJP, P59/P66 backward, or any optimizer commit.

The implementation is published as
`b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9` on the operator branch and its
first remote readback matched HEAD/FETCH_HEAD/tracking with ahead/behind
`0/0`. Fetch the final operator tip after this documentation checkpoint and
record that actual 40-character SHA; never substitute the older base
`9177b00b62d07a7d26a292126ba37b42f174f6de`. No matching image or target run
exists yet. Build/publish a matching image only after separate approval.

On the remote operator checkout, after source/image publication and readback:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_off_diagnostic.sh \
  <short-fresh-run-id> \
  <matching-image@sha256:...> \
  <worker-nodepool-or-auto> \
  /tmp/p58-checked-vma-off.yaml
```

The wrapper only renders. It requires a clean tree and exact equality among
HEAD, `P58_EXPECT_SOURCE_SHA`, and `origin/yuxzhang/canon-zero-tim`; it never
calls Kubernetes. Server dry-run and apply remain separately user-approved
actions. A valid render contains `zero-hp-vmaoff-precheck`, profile
`qwen3-4b-dp8-tp8-deepswe-v1-hp.env`, and diagnostic/backward/optimizer labels
`p58-checked-vma-off/0/0`.

Return the complete persistent run root, not selected log snippets. Required
files are `env.sh`, `run.log`, `weight_attestation.jsonl`,
`pre_alignment.jsonl`, `debug/run_manifest.json`, the Step-0 compressed
trajectory journal, `debug/batch_metrics.jsonl`, and
`p58_checked_vma_off.classification.json`. `updates.jsonl` must be absent or
empty. The classifier requires exactly 128 durable rows, finite A/B/C, exact
B-C, one precheck round, and zero VJP/backward/commit evidence. It returns
`A_B_EXACT_WITH_CHECKED_VMA_OFF` or `A_B_RED_WITH_CHECKED_VMA_OFF`. The first
supports the topology-shaped checked-VMA leak hypothesis; the second says
checked VMA is not a sufficient cause and promotes seam replay. Neither is a
full-training or Zero-TIM certification.

## 2026-08-27 P58.17 decode-vs-prefill seam probe — locally executed, source published

This section supersedes the P58.16 instruction to launch another full training
retry. Immutable `p58z07` already proved the loader-metadata repair and
returned all 128 Step-0 slots. It stopped before backward because
`S_decode_vs_S_prefill` had 32,952 differing elements / 71,797 differing
bytes over 379,496 action tokens; `S_prefill_vs_T_old` was exact. The first
delta was `4.35257e-3`, not the later `11.87498` maximum. All 1,024 bounded
mismatch records join exactly to durable artifact rows 49 and 62, and a
shift discriminator refutes a simple one-token offset.

The local P58.17 source adds a single-task DP1xTP4 Zero-HP carrier and an
automatic artifact classifier/bundler. The carrier has now run locally, but
that run remains development evidence. The source is published in
`b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9`. Fetch the final operator tip
containing the publication checkpoint and record its actual 40-character SHA;
do not run from historical base
`019d7a7e1cb7763b2ad4ffdc35e84bf9c217afe4` or substitute another SHA.

On one direct-attached four-chip v5p host, use a fresh clean `local/*` branch.
Do not use Pathways or Kubernetes. Confirm the default local Qwen3-4B snapshot,
R2E-Gym checkout, dataset cache, and Pillow Docker image already exist. Then:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch -c local/p58-seam-p58s01 origin/yuxzhang/canon-zero-tim
git rev-parse HEAD
git status --short --branch
export P58_ONEHOST_EXPECT_HOSTNAME=THE_EXACT_OUTPUT_OF_HOSTNAME
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_seam_probe_docker.sh p58s01
```

Optional path overrides are `DEEPSWE_TRAIN_PYTHON`,
`DEEPSWE_QWEN4B_MODEL_PATH`, `DEEPSWE_DATASET_CACHE`,
`DEEPSWE_R2EGYM_ROOT`, and `P58_ONEHOST_EVIDENCE_ROOT`. Do not set
`P58_ONEHOST_ALLOW_DIRTY=1` for returned evidence. The runner requires a clean
tracked tree, exact hostname, four TPU devices, local Qwen3-4B snapshot,
R2E-Gym commit `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`, and the immutable local task
image. It launches one real task with G2, response 4,096, 16 turns, serial
scheduling, prefix cache off, strict pre-alignment, and zero optimizer commits.
Its outer timeout is two hours; per-trajectory and batch deadlines remain
3,000 and 3,600 seconds.

The host-side Docker wrapper is required on this machine: host Python cannot
resolve TPU metadata, while the pinned privileged image consumes the four
`/dev/vfio` devices. It uses host network/IPC/UTS and mounts the Docker socket
so R2E creates sibling sandbox containers. It does not use Kubernetes.

Return exactly these two files named by the final `RETURN_FILES` marker; do
not manually select individual JSON/log files:

```text
P58_SEAM_PROBE_RETURN.tar.gz
P58_SEAM_PROBE_RETURN.tar.gz.sha256
```

The completed local development run is:

```text
artifact=/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s17_20260827t1045z
outcome=FINITE_RED_REPRODUCED
trajectories=2 SUCCEEDED=2 N_action=4808 optimizer_commits=0
A-B differing_elements=2488 max_abs=1.3662147521972656
B-C differing_elements=988
bundle_sha256=6285b5d2e8958ee85bd4b4190beaa240c7239ad6d07165a0948d7ba7f2b32eee
```

Shift 0 is much closer than shifts -1/+1, so the carrier refutes a simple
token offset. It does not reproduce the exact `p58z07` signature: the remote
DP8xTP8 run had a small first A-B delta and exact B-C, while this TP4 run has
a larger first A-B delta and finite B-C RED. The one-host result therefore
proves the real pipeline and a finite seam failure, but it cannot adjudicate
the P67 topology-shaped checked-VMA hypothesis. The runner-only overlay is
intentional: the remaining generated Qwen3/linear/embed/attention/RPA shims
are signed for TP8 and are excluded rather than falsely exercised on TP4.

`FINITE_RED_REPRODUCED` is diagnostic PASS evidence only.
`EXACT_ON_THIS_CARRIER` would be useful TP4 non-reproduction evidence.
`MALFORMED_OR_INCOMPLETE_EVIDENCE` or `INCONCLUSIVE_NO_ACTION_TOKENS` is not a
pass. This is not forced-token decode replay: the public serving API cannot
force the historical sampled IDs through incremental decode. It cannot
certify TP8, disaggregated Pathways, backward, optimizer, or convergence.

The next remote experiment is the admitted, zero-commit Step-0 selector
documented in the newer section above. Compare it against the immutable
`p58z07` signature and return pre-alignment plus full trajectory evidence.
Do not edit environment variables after rendering and do not launch another
1,000-update full run for this diagnosis.

See `phases/p58-17-decode-prefill-seam-probe.md`. No image publication,
Kubernetes mutation, TPU launch, commit, or push is authorized by this text.

## 2026-08-27 P58.16 loader-metadata override — source published

This section supersedes P58.15's `p58z05` launch instruction. The latest
immutable target is `p58z06`. It admitted 128 devices, the clean 1,012-task
list, disjoint DP8xTP8 rollout/trainer roles, and completed vLLM warmup, then
failed during adapter initialization before any rollout:

```text
[CANON_ADAPTER] live engine contract ... state_leaves=398 ...
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
FunctionalMappingError: canonical trainer execution trainer-mesh reconstruction changed the NNX state tree
```

This was not a bad trajectory or training collapse. Pathways dummy loading
adds `_is_loaded=True` to every live parameter, and Flax includes that loader
provenance in the State treedef. The weight-free trainer clone correctly has
no such marker. P58.15 compared raw treedefs and falsely rejected the otherwise
matching logical 398-leaf state; segmented backward contained the same latent
check.

Published implementation
`dba5211ac4945fefb50337603c800d9f8e3d37b5` removes only exact
`_is_loaded=True` from copied Variables for contract comparison. A
false/non-boolean marker fails, while all other metadata, paths/types, leaf
count, shapes, and dtypes remain exact. It does not change data, B8xG16, seed,
16K/50-turn horizon, sampling, loss, strict A=B=C, optimizer, compact
filtering, or any Native/Zero selector.

Focused forced-device tests and the complete pinned-image gate pass with:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

The exact source commit above is published and read back from
`yuxzhang/canon-zero-tim`. No matching image was published and no remote target
was launched. After the user separately approves image publication, build/read
back that exact source image, rerun the full gate, and pass sandbox admission.
A Kubernetes launch needs another approval and must use fresh id `p58z07`;
never resume/overwrite `p58z01`-`p58z06`.

At startup require exactly one each:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer state contract PASS relation=disjoint leaves=398 normalized_loader_metadata=_is_loaded live_markers=398 reconstruction_markers=0
[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt relation=disjoint graph=abstract-clone mesh_bound_jits=2
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
```

Then require trainer old/current logps, strict A=B=C, finite nonzero 16-group
backward, and one coherent update-0 transaction before continuing the same
1,000-update job. Evidence is under
`evidence/p58z06_nnx_loader_metadata_error/`; see
`phases/p58-16-nnx-loader-metadata.md`.

## 2026-08-26 P58.15 nested-JIT trainer-mesh override — source published

This is the highest-priority P58 handoff. `p58z04` already ran from source
`3f159250c4781b3faafde238f768457a0478446b`; preserve it as immutable trigger
evidence and do not reuse its image or artifact root.

`p58z04` emitted both P58.14 placement receipts and completed all 128 Step-0
trajectories in 1,709 seconds. Eight `MODEL_TIMEOUT` and one
`MAX_CONTEXT_LIMIT_REACHED` rows were accepted compact statuses. The actual
crash happened afterward, in the first trainer old-policy-logprob call:

```text
ValueError: Received incompatible devices for jitted computation.
trainer state devices: one 64-device role
jit inside jit devices: the disjoint rollout 64-device role
```

P58.14 moved the adapter's visible shardings to trainer devices, but vLLM's
prebuilt `model_fn` and `compute_logits_fn` each contain an inner `jax.jit`
whose output shardings captured rollout devices at engine initialization. The
old CPU mock used plain functions and did not model that closure.

The local P58.15 repair rebuilds the exact live model graph weight-free on the
trainer mesh with `nnx.eval_shape`, rejects any state tree/shape/dtype drift,
and constructs trainer-bound model/logits JITs with the original static and
donation contract. The segmented forward/backward path uses the same graph.
The installed fixed-AR mesh global is changed only during a locked trace and
is restored immediately. Serving remains rollout-bound; Native and colocated
behavior and all algorithmic settings are unchanged.

Local dependency-image CPU validation covers both the nested model/logits JIT
and segmented layer backward on disjoint 2+2 devices, finite nonzero gradients,
and partial-overlap rejection. The terminal marker is:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

This is not 128-chip proof. Implementation commit
`f60cdd569c2737df6cb2968125c8e42680938981` is published on
`yuxzhang/canon-zero-tim`. Fetch/read back the final operator tip and prove it
contains that commit, build and pin the matching image, rerun the full gate,
pass sandbox admission, and obtain separate Kubernetes launch approval. Use
fresh id `p58z05`; never resume/overwrite `p58z01` through `p58z04`.

At startup require exactly one each:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt relation=disjoint graph=abstract-clone mesh_bound_jits=2
```

The full classifier now fails if any receipt is missing or duplicated. Then
require trainer old/current logps, strict A=B=C, finite nonzero 16-group
backward, and one coherent update-0 commit. Passing update 0 continues the
same 1,000-update job. See `phases/p58-15-nested-jit-trainer-mesh.md` and
`evidence/p58z04_disaggregated_mesh_error/`.

## 2026-08-26 P58.14 disaggregated trainer-mesh override — source published

This is the highest-priority P58 handoff. Implementation commit
`dce0e93777548b7623e4f41702144f8d00f242f5` is published on
`yuxzhang/canon-zero-tim`. Do not launch an older operator tip or reuse the
`p58z03` runtime image.

Immutable `p58z03` facts:

- source `8eb65480d3705d96ab282799ad5a6c1901596248`, Qwen3-4B-Instruct,
  128 chips, disjoint rollout DP8xTP8 and trainer DP8xTP8 roles;
- all 128 Step-0 trajectories returned and fixed-head global/local
  M=`2048/256` was admitted;
- the first canonical trainer old-policy-logprob JIT combined trainer-state
  devices with rollout-bound sharding constraints and failed with
  `Received incompatible devices for jitted computation`;
- no trainer logprob completed, no alignment completed, no forward/backward
  executed, and no optimizer commit or checkpoint exists. Pallas/VJP
  `PATHTRACE` lines before the error are tracing evidence only.

The repair passes trainer state into adapter construction, derives an
engine-axis execution mesh on the exact trainer devices, and binds the
differentiable input/cache/sample/output path there. Serving remains on
rollout devices. The canonical log-softmax factory/math is unchanged, but
serving and trainer receive separate mesh-bound instances because `shard_map`
captures physical devices. DP/TP drift and partial overlap fail closed. Native
and colocated paths remain unchanged.

Local verification includes a forced four-CPU-device disaggregated
`jax.jit(value_and_grad)` with finite nonzero gradient, its partial-overlap
negative, colocated regressions, and the complete dependency-image CPU gate:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=2 trainer_devices=2 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=3 ... regressions=1
```

The local image has no `/dev/vfio`; this is not Pathways/TPU evidence. The
execution sequence is: fetch/read back the final operator SHA and require it
to contain `dce0e93777548b7623e4f41702144f8d00f242f5`, build and pin its
matching image, rerun the full gate, pass sandbox capacity, obtain separate
launch approval, and render fresh `p58z04`. Require the same placement lines
with `64/64`, then completed
trainer old/current logps, strict A=B=C, finite nonzero 16-group backward, and
the coherent update-0 transaction. A passing first update continues the same
1,000-update job. Never resume or overwrite `p58z01` through `p58z03`.

See `phases/p58-14-device-sharding-mismatch.md`. Preserved evidence is under
`evidence/p58z03_device_sharding_error/` and its `SHA256SUMS` verifies.

## 2026-08-26 P58.13 Qwen3-4B M2048 + FrozenLake P59-only VMA override

This is a completed historical source checkpoint. Implementation commit
`bea1aabde39c43c13ca4eaefab989301c6e8b46c` is published and read back on
`yuxzhang/canon-zero-tim`; the full pinned-image construction gate passed, and
matching target `p58z03` subsequently exposed P58.14.

Immutable `p58z02` facts:

- Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16, rollout DP8xTP8 plus
  trainer DP8xTP8 on 128 chips;
- the P58.12 engine-global seed route passed;
- all 128 Step-0 collector rows returned in one 1,514.2-second wave;
- one `MODEL_TIMEOUT` and two `MAX_CONTEXT_LIMIT_REACHED` rows were retained
  under the compact-status policy, so the batch was not timeout-free;
- the hard failure came later in trainer canonical per-token-logprob forward,
  before alignment completion, backward, AdamW, or an optimizer commit:

```text
[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS data=8 static_width=20480 chunks=80 global_M=2048 local_M=256
ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256, 4096), got (2048, 2560)
```

The repair registers learner M `(2048,4096)` only for exact Qwen3-4B TP8
`(hidden=2560,tp=8)`, retains the existing Qwen3-8B TP8 registration, and
keeps every other geometry at `(4096,)`. Qwen3-32B TP8 remains a negative for
M=2,048; do not broaden this to all TP8 models.

The Zero-HP profile also imports the latest FrozenLake Wave-5 repair:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1       # internal alias derived by 00_env
CANON_P67_P66_VMA_P59_ONLY=1   # scope metadata to exact P59 backward
```

Wave 5 proved strict A-B=0/B-C=0 for both `p66-off` and `serving-scope`; the
scoped arm is preferred because it preserves checked-VMA backward while
restoring the historical ordinary-serving graph. P67 is admitted only for
the exact P58 Zero/full, Qwen3-4B DP8xTP8, 1,000-update HP tuple. Native raw,
Native+IS, non-HP Zero, Qwen3-32B, and unrelated profiles remain off. This is
a numerical graph repair, not a warning-only gate: fresh DeepSWE Zero still
requires A=B=C exactly.

Construction evidence:

- 50/50 focused host tests pass;
- installed Qwen3-4B overlay matches 37/37 and reports
  `learner_M=2048,4096`;
- independent Qwen3-32B exact-image gate reports `learner_M=4096`;
- complete gate ends with `P58_EXACT_IMAGE_CPU_PASS ...
  qwen4b_fixed_head=1 checked_vma=1 vma_p59_only=1 first_update=1 ...`.

The image had no `/dev/vfio`; target A=B=C, backward, optimizer, and
convergence are not proven. Preserve `p58z02` under
`evidence/p58z02_backward_fixed_lm_head_error/` (run-log SHA-256
`7349c7965f31e2c84dfd98f8cb7fe175f9b2d4281759d0bb5c07bb336ef8784d`).
It is not a resumable trainer checkpoint.

Historical execution produced `p58z03`; do not rerun that source unchanged or
resume its nonexistent trainer checkpoint. Follow the P58.14 section above
for the fresh `p58z04` sequence. See
`phases/p58-13-backward-fixed-lm-head-m2048.md` for the completed source gate.

## 2026-08-26 P58.12 JAX engine-seed/cleanup override — source published

This is the highest-priority P58 handoff. Implementation commit
`c10fbe0487d1f6635975b84806f1efdce6bc95c1` is published on
`yuxzhang/canon-zero-tim` and preserves immutable Zero-HP Attempt-0 evidence under
`evidence/p58z01_attempt0_seed_exception/`. `p58z01` admitted all 128 TPU
devices, loaded 1,012 clean tasks, launched 128 R2E sandboxes, and initialized
vLLM. The first Step-0 model call then failed before any trajectory:

```text
ValueError: JAX does not support per-request seed.
```

P58.10 had put seed 42 in `RolloutConfig.seed`, which Tunix forwarded to
`SamplingParams.seed`. The P58.12 published repair instead passes the same signed
42 through global vLLM `EngineArgs.seed` and rejects any JAX per-request seed
before generation. Require both startup receipts exactly once:

```text
[P58.SEED] PASS dataset_seed=42 rollout_seed=42 scope=engine-global async_completion_order=not-claimed
[VLLM.JAX_SEED] PASS engine_seed=42 request_seed=none scope=engine-global
```

W&B, durable manifests, one-host artifacts, classifiers, and postflight use
the same engine-global scope. Async sandbox completion order remains explicitly
unclaimed. This preserves the fixed-seed comparison across Native raw,
Native+IS, and Zero; it does not change sampling parameters or the Zero
numerical bundle.

Abort cleanup also hit kubernetes-client's exact empty-body
`AttributeError: 'NoneType' object has no attribute 'decode'`. The local patch
treats only that exact defect as an ambiguous response, reads until confirmed
404, and reissues the same exactly-scoped DELETE if the Pod is still present.
Every other AttributeError/API failure or an unconfirmed deletion remains
fatal; no namespace-wide cleanup is introduced.

Current status is `SOURCE PUBLISHED / CONSTRUCTION PASS / TARGET RETRY NOT
RUN`. Focused P58, P34, P57, flag-audit, and complete digest-pinned image gates
pass; the image exposes no `/dev/vfio`, so this is not target evidence. The
execution agent must fetch the final operator tip, prove it contains
`c10fbe0487d1f6635975b84806f1efdce6bc95c1`, build and pin the matching image,
then launch fresh `p58z02` only after separate image/launch approvals.
Do not resume/overwrite `p58z01`: it has no trajectory or trainer checkpoint.
P58.11's unchanged strict A=B=C, checked-VMA, first-update, stable-clip, and
1,000-commit gates apply after Step 0 begins. See
`phases/p58-12-jax-engine-seed-cleanup.md` and the top P58.12 runbook override.

## 2026-08-26 P58.11 strict Zero-HP override — source published

This is the highest-priority P58 source instruction. The user reactivated the
Qwen3-4B-Instruct strict Zero-HP full campaign. P58.11 adds the shared
checked-VMA backward repair, first-update admission, and overflow-safe clip to
the existing `--arm zero --high-performance` recipe without changing its
scientific workload:

```text
model/tasks:       Qwen/Qwen3-4B-Instruct-2507 / promoted 1,012 tasks
batch:             B8 x G16 = 128 trajectories
roles:             rollout DP8xTP8 + trainer DP8xTP8 (128 chips total)
context/turns:     response 16,384 / max turns 50
training:          1,000 commits, seed 42, TPU-resident AdamW
alignment:         strict A=B=C; sampler IS/TIS and group filter off
backward shape:    global M2048, local M256, 16 rank-major groups
```

The exact HP profile now derives this closed numerical bundle:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1        # internal derived compatibility alias
CANON_P67_P66_VMA_P59_ONLY=1    # P58.13 serving-scope repair
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_P63_OVERFLOW_SAFE_CLIP=1   # max norm remains 1.0
```

The eight outer prompt chunks are not the accumulator denominator. Update 0
must emit a precommit receipt with `microsteps=16` and
`accumulator_denominator=16.0`, then a coherent `train_steps 0 -> 1` commit
receipt before outer weight sync/checkpoint. Every update must carry checked-
VMA and P63 evidence. More precisely, a legal all-compact backward attempt
carries P59/checked-VMA receipts plus a zero-commit journal row, while P63 and
global-step receipts occur only for commits. Postflight reconciles the ordered
attempt stream and still requires exactly 1,000 commits. Native raw, Native+IS,
ordinary non-HP Zero, and neighbor DeepSWE recipes must keep all four
operator-facing flags absent.

The implementation is published to `yuxzhang/canon-zero-tim`. It was
constructed on `644beb38cee2388862941019269ad264a581064f` and fast-forwarded
without overlap over V1-only evidence tip
`4003f61cabb6f2d5e43d4c217cebb4dca2c3d217` before publication. Focused and
adjacent CPU tests,
the real P58 16-group/0-to-1 CPU commit regression, flag audit 383/383, and the
complete pinned-image gate pass; its terminal includes
`zero_hp_full=1 checked_vma=1 first_update=1 stable_clip=1`. The pinned image
has no `/dev/vfio`, so this is construction evidence only. The execution agent
must fetch `yuxzhang/canon-zero-tim`, read back the exact current 40-character
tip, build/pin the matching image, rerun the complete P58
exact-image gate, perform the existing sandbox-capacity admission, render a
fresh Attempt-0 `--stage full --arm zero --high-performance` JobSet, and obtain
separate launch approval. Source publication does not authorize image
publication, Kubernetes apply, or TPU execution. A first-update PASS continues
the same 1,000-update job; it is not a one- or three-update stop.

Construction evidence cannot certify DP8xTP8 target behavior. Until a real
run completes, report `TARGET NOT RUN`. See
`phases/p58-11-qwen4b-zero-checked-vma.md` and the top P58.11 override in
`cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## 2026-08-25 P58.10 fixed-seed override — published, launch separately gated

This is the newest source checkpoint. It adds one shared fixed-seed contract
to all three P58 recipes without changing the selected Native+IS treatment:

```text
CLI:               exactly one --seed=42
dataset shuffle:   seed 42
rollout sampler:   RolloutConfig.seed=42
W&B/manifest:      dataset_seed=42, rollout_seed=42,
                   seed_scope=config-level
runtime marker:    [P58.SEED] PASS dataset_seed=42 rollout_seed=42
```

Missing, duplicate, or non-42 CLI values fail closed. Native raw, Native+IS,
and Zero-HP use the same value, so seed is not a treatment difference. This
does not claim bitwise-identical end-to-end trajectories: vLLM scheduling,
R2E sandbox completion, and `asyncio.as_completed` ordering remain
asynchronous. The seed fixes the configured sampling stream and data shuffle,
not external completion order.

The implementation was built in
`/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
`local/p58-fixed-seed-0824`, then replayed over latest fetched operator tip
`ff646a4d76f58e9f328bc640f44d362637eb1432`. It passes 33/33 focused tests and
the complete pinned-image P58 gate. Implementation commit
`9597de3d99fbf65c87f4fea3d86e639cca0b7abe` was pushed only to
`yuxzhang/canon-zero-tim`; immediate local/FETCH_HEAD/remote-tracking readback
was exact with ahead/behind `0/0`. Fetch the final operator tip containing
that commit and pin the exact read-back 40-character SHA in the rendered YAML.
The Native-raw archival/stop decision below is unchanged. Fresh Native+IS is
source-ready, but launch remains separately user-gated.

See `phases/p58-10-fixed-seed.md` and
`cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## 2026-08-24 execution decision — stop Native raw, launch fresh Native+IS

This is the highest-priority execution instruction and supersedes every
native-raw launch/resume instruction later in this historical handoff. The
operator reports that the currently running Native/no-IS campaign's training
reward has dropped sharply and considers the run collapsed. The onset update
is not established; do not assign the event to any fixed optimizer step. The
exact run id, W&B series, raw log, and checkpoint receipts have not yet been
ingested into this worktree, so the reward collapse
is operator-reported evidence rather than a locally verified diagnosis. The
execution decision does not wait for a root-cause classification:

1. stop the exact currently running Native/no-IS JobSet;
2. preserve its full evidence, including the reward-drop onset, as an
   immutable failed/collapsed Native-raw attempt;
3. never resume that optimizer checkpoint and never relaunch Native raw;
4. launch a fresh Native+IS full run only from the original frozen base model,
   with a new run id, run root, W&B run, and checkpoint directory.

Before stopping, resolve the exact JobSet name rather than guessing. Require
all of the following from its rendered YAML/resolved environment:

```text
canon.zero-tim/arm: native
CANON_P58_TIM_ARM=native
CANON_P34_DISABLE_SAMPLER_IS=1
CANON_P34_DISABLE_TIS=1
no --sampler_is=token
no canon.zero-tim/sampler-recipe=token-is
```

Preserve the exact rendered YAML and digest, source SHA, image digest, JobSet
and Workload YAML, head/worker logs, run log, W&B URL/export, trajectory
journals and their digests, update receipts, optimizer/checkpoint inventory,
and metrics covering the last stable reward region, the reward-drop onset, and
all subsequent completed batches. At minimum retain solve ratio,
all-zero/all-one/mixed/effective group counts, nonzero-advantage ratio,
completion lengths, sampler-trainer logp/prob diffs, policy ratio/clip metrics,
gradient/update norms, and A/B/T-old/T-current observations. Do not truncate
the evidence export at an assumed optimizer step.

Only after the identity and evidence above are preserved, the remote executor
is authorized by this decision to delete that exact Native-raw JobSet and wait
for its deletion:

```bash
JOBSET='<exact-running-native-raw-jobset-name>'
kubectl -n default get jobset "$JOBSET" -o yaml
kubectl -n default get pods \
  -l "jobset.sigs.k8s.io/jobset-name=$JOBSET" -o wide
kubectl -n default delete jobset "$JOBSET" --wait=true --timeout=10m
kubectl -n default wait --for=delete "jobset/$JOBSET" --timeout=10m
```

Do not substitute a wildcard, namespace-wide delete, or a guessed name. After
the JobSet is gone, confirm the Pathways head/workers are gone. Enumerate any
remaining R2E sandboxes and delete only Pods proven by run provenance to
belong to this exact attempt; preserve cleanup receipts. Never delete unrelated
R2E workloads.

The replacement experiment is the registered Native+IS recipe:

```text
model/data/geometry: unchanged Qwen3-4B-Instruct-2507, 1,012 tasks,
                     B8 x G16, 16K, 50 turns, 128 chips
renderer:            --stage full --arm native --sampler-is
sampler tuple:        CANON_P34_DISABLE_SAMPLER_IS/TIS=0/0
runtime:              sampler_is=token, threshold=2.0
old policy logps:     trainer logps
correction:           token TIS weights present
group filter:         absent
optimizer:            TPU resident; no host offload
restart policy:       exact Attempt-0
seed:                 42 for dataset shuffle and rollout sampler
horizon:              1,000 committed updates
```

Use a fresh run id such as `p58is01`; do not reuse the Native-raw run root,
W&B run, or checkpoint. The renderer must emit
`P58_DEEPSWE_TIM_RENDER_PASS arm=native stage=full recipe=native-is`, the
JobSet name must contain `native-is`, and its label must contain
`canon.zero-tim/sampler-recipe=token-is`. On the first effective batch require
exactly one marker:

```text
[P58.TIM_RECIPE] PASS recipe=native-is sampler_is=token old_logps=trainer tis_weights=present threshold=2.0 group_filter=none
[P58.SEED] PASS dataset_seed=42 rollout_seed=42 scope=config-level async_completion_order=not-claimed
```

A `native-raw` marker, a `1:1` or partial sampler tuple, missing trainer logps,
missing TIS weights, group filtering, host optimizer offload, prefix cache, or
resume from the collapsed Native checkpoint is a hard stop.

Publication status: on 2026-08-24 the user explicitly authorized commit and
push of this Native+IS refinement. Implementation commit
`2aedd73c957abba29d21d05b866a996af2f66dfd` was replayed over operator tip
`7b85b42d0a019d70f32a7dc9712c538ad42f5cb5`, pushed only to
`yuxzhang/canon-zero-tim`, and its first post-push readback matched local HEAD,
`FETCH_HEAD`, and the remote-tracking ref with ahead/behind `0/0`. Fetch the
final operator tip containing this publication checkpoint and pin that exact
40-character SHA in the rendered YAML. Stopping and archiving the current
Native-raw job may proceed now. Do not silently launch an older branch and
call it Native+IS.

## 2026-08-24 P58.9 publication override — launch remains separately gated

This is the current checkpoint. It supersedes older execution wording below
without deleting historical evidence. Work only from
`/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824`, branch
`local/p58-is-zero-refine-0824`, originally based on operator tip
`614156c1ab067192ab65b2969543e23904f192be`. It was replayed over
`7b85b42d0a019d70f32a7dc9712c538ad42f5cb5` and published as implementation
commit `2aedd73c957abba29d21d05b866a996af2f66dfd`. Do not use the older dirty P58
worktree and do not touch `main`. The execution decision at the top of this
handoff authorizes the remote executor to preserve/stop the exact Native-raw
run and then apply only the fresh Native+IS YAML after final SHA readback and
all listed render/admission checks.

P58 now maintains three closed production recipes on the same Qwen3-4B,
1,012-task, B8 x G16, 16K, 50-turn, 128-chip DP8 x TP8-per-role setup:

| Recipe | Renderer selector | Sampler tuple | Required first-effective-batch evidence |
|---|---|---|---|
| Native raw | `--arm native` | disable sampler/TIS `1:1` | one `recipe=native-raw`, old logps=rollout, TIS absent |
| Native IS | `--arm native --sampler-is` | `0:0` | one `recipe=native-is`, token IS threshold 2.0, old logps=trainer, TIS present |
| Zero HP | `--arm zero --high-performance` | `1:1` | strict Zero/P59/fixed-head receipts; no Native recipe marker |

All mixed or partial tuples fail closed. Native-IS does not enable group clip
filtering, flat-group resampling, host optimizer offload, prefix cache, or a
Zero numerical switch. The original Native-vs-Zero estimand remains distinct;
Native-IS is a mitigation arm.

The renderer also restores exact Attempt-0:
`failurePolicy={maxRestarts: 0, restartStrategy: Recreate}`. The prior retry
setting reused a persistent run root without attempt isolation. Five
Pathways/IFRT/GRPC keepalive environment names were removed because pinned
image inspection found no code consumer; they were configuration-shaped text,
not a proven recovery mechanism.

Focused host gates pass after replay: renderer/profile/sampler-recipe/stock-
observer aggregate 40/40, Python/Bash syntax, and diff hygiene. Bare-host
environment-contract import is `INCONCLUSIVE` because this shell lacks
`metrax`. The complete P58 exact-image gate passes in
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with terminal marker `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
zero_hp_full=1 ... p59_real_shim=4 p59_rpa=2 ... m15_token=1
regressions=1`. No target or one-host TPU PASS exists for this delta.

See `phases/p58-9-native-is-attempt-zero-refine.md` and the top of
`cluster/P58_DEEPSWE_TIM_RUNBOOK.md`. Source publication is complete. This
agent did not publish an image, apply Kubernetes resources, stop a live job,
or execute TPU training.

## 2026-08-23 P58.6/P58.7/P58.8 override

This checkpoint supersedes the later native-only execution wording in this
historical handoff. The current local worktree is
`/home/yuxuan/code_rl_repro/worktrees/p58_zero_hp_release3_0823`, branch
`local/p58-zero-hp-release3-0823`. The release was originally rebuilt from
operator tip `ccbcf572dc903bb1cce12f897cbdb05aec94922a` by
migrating only prior dirty hunks and new files, preserving the upstream P57
evaluation-cycle, final-only checkpoint, and lazy NumPy host-render fixes. The
branch has since fast-forwarded through immutable V1 evidence to
`614156c1ab067192ab65b2969543e23904f192be`; the older dirty and release
worktrees were not rebased, reset, or modified.

The three user-requested TODOs are implemented:

1. P58.6 provides matched direct-four-chip Native and optimized Zero-HP
   no-commit update XProf/Perfetto carriers, immutable provenance/work hashes,
   state neutrality, arm classifiers, cross-arm classification, and sealed
   packages. See `phases/p58-6-onehost-native-zero-xprof.md`.
2. P58.7 provides a default-off optimized strict-Zero Qwen3-4B DP8 x TP8 full
   profile, exact renderer/admission tuple, P59 and fixed-head receipts,
   update XProf/Perfetto, and a 1,000-update postflight/performance ledger. APC
   remains off. See `phases/p58-7-qwen4b-zero-hp.md`.
3. P58.8 repairs the P59 TP4/TP8 nested-engine mesh boundary exposed by the
   first GSM8K full log and the signed P57 Zero/full W&B project admission
   exposed by the FrozenLake log. See `phases/p58-8-p59-tp-mesh.md`.

Before the current V1 Attempt-3 RPA repair, the complete pinned-image gate
passed on the reconstructed release tree with
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with terminal marker
`P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 zero_hp_full=1 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 regressions=1`.
The V1 exact-image gate independently passed with
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 perfetto_window=1 manifests=3`.
The current additive Attempt-3 gate changes both expected terminals to include
`p59_rpa=2` and `m15_token=1`; that gate has not run and remains separately
approval-bound. Current host adjacency is P59 34/34, P57 144/144, V1 21/21,
APC 31/31, and flags 366/366. The FP32 TP rank sums include operand barriers;
the historical complete exact-image runs execute TP4/TP8 fixed-head markers with
`all_gather_rank_order_f32_barrier`, installed projections remain
`serial_parallel=exact`, and both manifests are 36/36.

No direct TPU pair or DP8 x TP8 target was run. The approved release is four
functional commits plus one audit-only release-gate commit. The latter excludes
immutable logs and Markdown marker contracts from changed-settable-flag
discovery while preserving the independent 366-name registry inventory.
Publication is authorized, but the runnable source is only the exact
operator-branch SHA read back after push. No image publish, Kubernetes apply,
or TPU launch is authorized here. CPU/pinned-image admission must not be
promoted to target certification. The exact operator commands and artifact
rules are in `RUNBOOK_P58_6_7.md`. Any real Zero `CANON_ALIGN ... verdict=FAIL`
kills the candidate. P59 claims ordinary-JAX FP64 gradient correctness, not
serial-AdamW weight-trajectory identity.

The corrected P59 admission gate is
`canon-zero-tim/tests/p59_backward/run_tp4_tp8_installed_shim_exact_image.sh`.
It executes real installed Qwen1.7B/TP4 and Qwen8B/TP8 projection branches plus
the fixed-head/report-adjoint/fixed-reducer composition with zero commits. The
available four-chip v5p cannot form the minimum real DP2 x TP4 composition, so
this is not recorded as a one-host TPU PASS.

## Current checkpoint

P58 was developed in isolated worktree
`/home/yuxuan/code_rl_repro/worktrees/p58_deepswe_native_zero_0821` and approved
for publication to `yuxzhang/canon-zero-tim`. The implementation commit is
`c5bdc9d993dfaf1a6956335609fbf259f9ed95f7`; its first post-push readback
matched exactly with ahead/behind `0/0`. Always obtain the runnable
40-character source SHA by fetching and reading back that branch after push;
the latest branch may contain this later publication-evidence-only commit. Do
not infer the runnable SHA from the historical base or a mutable tag.

The latest P58 admission repair and direct-full phase implementation commit is
`abbc76008e0a7fcb63562c27d5cf4608fb4f4e90`. Its first post-push readback
matched exactly with ahead/behind `0/0`. This documentation checkpoint advances
the branch once more, so the executor must still fetch and use the final
operator-branch SHA rather than pinning the implementation commit directly.

Latest source intake fast-forwarded this isolated worktree through immutable
p58f09 evidence to operator tip
`3edf480072126145acc2df259419e12dd2737c69`. P58f07 proved the published finite
Native B-C warning repair, then exposed an over-strict Native trainer-program
observer gate. The correction was published as
`81622977bf15393798c671e578ee059d1268e78b`; its first readback matched local
HEAD, `FETCH_HEAD`, and `origin/yuxzhang/canon-zero-tim` with ahead/behind
`0/0`. This documentation checkpoint advances the branch once more, so the
executor must fetch and exactly read back the final operator tip. The p58f09
repair was published as `678bc5cfbcec386fd655e6685365c937e826d547`; its
first readback matched local HEAD, `FETCH_HEAD`, and the remote-tracking branch
with ahead/behind `0/0`. Source intake then fast-forwarded to exact operator tip
`28817bfb3a14c95f42b3950f03380d1c6c03d336`, which contains immutable p58f10
timeout evidence. P58f10 reached Step-0 rollout but the B8 x G16 batch was
throttled into two waves by concurrency 64; only 5/8 prompt groups completed
before the 3,600-second hard batch deadline. The local repair makes all 128
trajectories one wave, matching rollout DP8 x max-seqs16 capacity. It was
published as implementation commit
`44b6fb4527a8a05bf649b5140d12142e2abef83f`; its first remote readback matched
local HEAD, `FETCH_HEAD`, and the remote-tracking branch with ahead/behind
`0/0`. Source intake then fast-forwarded to exact operator tip
`e92b0120a7df371569cc8646eb7b8a9367ebbe86`, which adds immutable p58f11
evidence. P58f11 proved the one-wave concurrency repair by completing all 128
trajectories and 8/8 groups in 1,209.2 seconds, then stopped on a missing
`prompts` key in the single reset-timeout fallback row. The repair was
published as implementation commit
`43614af55ed98423b757945642fa5444ae484ecc`; its first remote readback matched
local HEAD, `FETCH_HEAD`, and the remote-tracking branch with ahead/behind
`0/0`. Latest source intake reached exact operator tip
`5f449cc8def801b4a61387ef664b2cb1f7ab05cf`, which contains immutable p58f12
evidence plus a later P57-only checkpoint change. After explicit user approval,
the p58f12 repair described below was committed as
`135867f04bfa0fc90ea1d4528ba59f365573a78b` after a conflict-free rebase over
non-overlapping P57 evidence commit
`e7958a27851931ab9bcff232088efd95bbc12021`; this publication-evidence
checkpoint follows it. Fetch the final `yuxzhang/canon-zero-tim` tip and prove
the remote readback matches before use. The historical next id was `p58f13`;
the 2026-08-24 execution decision above supersedes it with fresh Native+IS id
`p58is01`.

The user previously waived P58.3 and the separate three-update stop, then chose
the native 128-chip full 1,000-update stage. That historical phase remains
waived rather than promoted. For the p58f05 repair, the user later requested a
new bounded direct-attached one-host gate before publication. Its runner is
implemented, but this container exposes no `/dev/vfio` and returned
`P58_ONEHOST_ALIGNMENT_BLOCKED`; it is not a TPU PASS. Updates 1–3 remain live
monitoring milestones in the same full job, not an early-stop condition. Zero
is not optimized enough for launch and is explicitly deferred. No Kubernetes
apply or TPU launch is authorized by this handoff alone.

Native attempts `p58c01`, `p58c02`, and `p58c03` are bootstrap
`INCONCLUSIVE` results.
P58c01 failed in `00_env.sh`; its published fix preserves native
`CANON_P32_DP_REDUCTION_ADMITTED=0`, exports three unrelated FrozenLake zeros,
and passes the renderer-to-real-`00_env.sh` regression. The fix implementation commit
`acd3136267214b367a6755d0ba28d80e883d6753` was pushed and its first remote
readback matched exactly with ahead/behind `0/0`. Fetch again and use the
final operator-branch SHA because this publication note is a later docs commit.

P58c02 then initialized Pathways and stopped before importing the model: direct
file execution of `/app/examples/deepswe/canonical_entrypoint.py` did not put
`/app` on `sys.path`, so its package-qualified `examples.deepswe` target could
not be found. The local fix derives the repository root from `__file__`, adds
it before the package import, and changes native stock preflight to exercise
the identical direct-file entrypoint. The exact command now exits zero from
`/tmp` in the pinned image, and the complete exact-image gate passes. These
changes were published as `82d82f72a7220d945737d95f6266b5b7e2cfe706`;
the first post-push readback matched exactly with ahead/behind `0/0`. Fetch the
final operator tip because this publication checkpoint advances it once more.

P58c03 proved that the preceding admission, install, stock-engine, Pathways,
and direct-entrypoint fixes work, then stopped before model initialization.
`00_env.sh` correctly removed native-only presence-sensitive zero-TIM switches
inside its child shell, but its generated `env.sh` contained exports only.
When the parent entrypoint sourced it, the raw renderer value
`CANON_LOGPROB_M=256` remained present and the DeepSWE Python contract
correctly rejected the native environment. The W&B-run fatal printed after
that exit is derivative, not the first failure.

The fix turns the generated `env.sh` into an authoritative snapshot of
all managed non-secret namespaces: it clears the caller's managed values,
then exports the exact resolved set. Secret injection variables and token
values are neither cleared nor serialized. The exact regression seeds the
raw parent with `CANON_LOGPROB_M=256`, executes real `00_env.sh`, sources its
snapshot, verifies native absences, and passes the Python contract. Focused
P58/P34 tests, the P57 81-test adjacent suite, and the full pinned-image gate
pass. It was published as `c0ca41805bd65a4fdede4825ed2835cdce6e13ed`;
the first post-push remote readback matched exactly with ahead/behind `0/0`.
Fetch the final operator tip because this publication-evidence checkpoint
advances the branch once more.

P58c04 proved the complete bootstrap and initialization chain through real
128-chip Pathways discovery, Qwen3-4B/vLLM initialization, W&B initialization,
and entry into `run_producers_from_stream`. It then requested all 128 RepoEnv
sandboxes concurrently. No sandbox was logged Running before the 1,200-second
start deadline, and the interleaved log retains at least 121 explicit timeout
records. The pinned R2E `start_container` swallowed the start `TimeoutError`,
deleted the pod, and returned with `container=None`; later
setup attempted a websocket exec into that deleted pod. Kubernetes' real 404
was then obscured by the client library's `None.decode` AttributeError. The
websocket payload decoder is not the root cause and must not be patched or
made permissive.

The local repair bypasses the upstream exception-swallowing wrapper only for
the Kubernetes backend, propagates the original timeout after confirmed pod
deletion, and proves that a reset-time start failure becomes the existing
signed `ENV_TIMEOUT` trajectory status. Docker behavior remains delegated to
upstream. A bounded timeout marker preserves pod phase and scheduler
conditions without inspecting the pod spec/environment. At the p58c04 repair
checkpoint, the P58 renderer used reference sandbox concurrency 64, so the
unchanged B8 x G16 batch was created in two waves. That historical choice is
superseded by the p58f10 one-wave repair below. This changes neither
data, sampling, RLOO/loss, meshes, optimizer, nor update horizon. Two newly
shared stock-contract booleans are explicitly zeroed in the native profile;
that is compatibility hardening, not a new treatment. Focused tests and the
full pinned-image gate pass. The trajectory journal and W&B now retain bounded
timeout provenance: status; sandbox/model/environment/reward/deadline stage;
unschedulable; and insufficient CPU/memory counts and ratios. Raw scheduler
messages stay in the run log. These changes were published as
`174fcf3a42af3e9cd465307843a1c19a08098c99`; its first remote readback matched
with ahead/behind `0/0`. Fetch the final operator tip after the publication
checkpoint rather than pinning this implementation commit directly.

P58c05 never reached the runtime. Its Workload remained
`QuotaReserved=False`; Kueue reported that flavor `0xv5p-8` did not match the
worker node affinity. The rendered worker combined exact `4x4x8` topology with
literal node-pool selector `tpu-v5p-slice`. That value is a Kueue sentinel, not
a concrete node pool, so it contradicted ResourceFlavor admission. No JobSet
pod, Pathways process, model, sandbox, trajectory, optimizer action, or
checkpoint started. The evidence under `evidence/p58c05_admission/` is
immutable and there is no resumable state.

The local repair makes all registered Kueue sentinels delegate concrete pool
affinity to ResourceFlavor while retaining the TPU accelerator and exact
topology. Explicit real node-pool names remain exact. The next run is fresh
native full-stage `p58f01`, not a retry or resume of p58c05.

P58f01 proved that repair: it reached 128 Pathways devices, the exact 64/64
role split, Qwen3-4B/vLLM and online W&B initialization, and the rollout
producer. It did not produce a usable R2E trajectory. All 128 environment
resets timed out, and at least 127 bounded Pod markers say
`PodScheduled=False`, reason `SchedulingGated`. The runtime-created standalone
Pods lacked `kueue.x-k8s.io/queue-name`; on this cluster Kueue therefore added
an admission gate but had no LocalQueue through which to admit them. After the
all-timeout batch completed, a second local bug raised
`policy_version is missing from trajectory task`: environment reset had failed
before `_model_call` assigned that provenance, so the batch crashed before
the P58 journal boundary. P58f01 is `INCONCLUSIVE`, immutable, and not
resumable. Its raw log SHA-256 is
`16c513c773ac2bfb1542178b4e42b03098bb9114564106b03f83c0195a0d542f`.

The repair derives `R2E_K8S_QUEUE_NAME` from the parent JobSet queue label,
persists it through the authoritative environment snapshot, validates it
without normalization, and applies it to every sandbox Pod. It also assigns
the current `policy_version` when the environment is constructed, before
reset, while retaining the strict downstream missing-provenance check.
`SchedulingGated` is now a separate bounded trajectory/W&B dimension. The next
fresh native full attempt is `p58f02`; do not reuse the p58f01 root.

The p58f01 repair was published as
`c67e9d5bfa3f1b3b592a2440075eb165e073e6ac`; its first remote readback matched
exactly with ahead/behind `0/0`. This publication checkpoint advances the
branch once more, so the executor must fetch and use the final operator tip
rather than pinning the implementation commit directly.

P58f02 then reached Step 0 but the sandboxes stayed `SchedulingGated`: the
cluster's `cpu-user` flavor requires `nodeSelector: cpu-np`, while the job was
requesting `deepswe-cpu-pool`. The user confirmed that moving the CPU head and
sandboxes to `cpu-np` resolves this; a general in-process CPU fallback is not
part of the solution. That routing repair was published in source
`7208d7b330759ac7dc31493ece65d32a6c355308`.

P58f03 used that source and completed the first real rollout batch in 616.3
seconds. Its durable journal has 128 rows: 126 `SUCCEEDED`, two
`MAX_CONTEXT_LIMIT_REACHED`, three solved trajectories, two mixed/effective
groups, and 32 nonzero advantages. No sandbox-start timeouts occurred. The raw
log is `evidence/p58f03/run.log`, SHA-256
`fdb958d5e1db8bafa25b6df8c3223a3c6a642d00c6a1915bb34a8e17b5bcf600`.
The journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f03/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`26c92d2153865cc14296303fcb97afd98f857744e50574032b6eba8631f23a9e`.

P58f03 then stopped before trainer forward/backward/update. The generic P34
weight gate called `attest_canonical_engine_weights`, which intentionally
requires a registered canonical adapter, while native correctly runs with
`CANON_ENGINE_MODULE_C=0`. The first failure was therefore a routing/contract
bug, not rollout throughput or weight drift. The local repair exposes a shared
exact-live-weight interface: zero still delegates to its registered canonical
adapter; signed P58 native performs only the same pure leaf mapping and
bitwise live-weight comparison. It neither registers an adapter nor changes
serving math. Missing/mismatched weights, invalid DP8 x TP8 mesh, unsigned
native routing, and a leaked native adapter remain fatal. Focused routes, all
15 rollout canonical tests, and the full pinned P58 exact-image gate pass.
The implementation was published as
`234eaddb8e3543083927aa10effe101abef18a91`; its first remote readback matched
exactly with ahead/behind `0/0`. This publication-evidence checkpoint advances
the branch once more, so fetch and pin the final remote tip rather than the
implementation commit directly. That repair was exercised by fresh native
`p58f04` below rather than by resuming p58f03. Zero remains deferred.

P58f04 completed the next real rollout batch in 557.2 seconds and durably
journaled 128 rows: 125 `SUCCEEDED`, three `MAX_CONTEXT_LIMIT_REACHED`, six
solved trajectories, five all-failed groups, one mixed/effective group, two
incomplete groups, and 16 nonzero advantages. It proved the preceding repair
with `[P34.WEIGHTS] EXACT` over 398 leaves and 4,022,468,096 elements. The raw
log is `evidence/p58f04/run.log`, SHA-256
`a7b0cda5e7d359c7e320b29f8af197db0dd6c46dc34850aa55ffb350fb766fdd`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f04/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`e39caf5df63ba54406a36427a413dea562e5771f4c52b30c840229d3178c1f3b`.

P58f04 then failed before trainer forward/backward/update. The shared
processed-`S_prefill` interface required the canonical
`CANON_PROMPT_PROCESSED_LOGPROBS=1` engine path, while native correctly keeps
that flag and `CANON_ENGINE_MODULE_C` at zero. Reusing the stock raw helper
would be wrong because it rolls targets across a DP-packed buffer and can cross
request/padding boundaries. Enabling the canonical flag would contaminate the
native treatment.

The local repair adds a separately signed, observer-only P58 native stock-B
overlay. It is installed only after the six stock files verify; it changes one
runner call site plus one helper under an exact two-file manifest. It applies
decode-equivalent temperature/top-k/top-p transforms and derives targets from
absolute request history. It does not enter generation, trainer forward, loss,
backward, optimizer math, or commits. Native still has
`CANON_PROMPT_PROCESSED_LOGPROBS=0`, `CANON_ENGINE_MODULE_C=0`, and every other
zero-TIM numerical switch disabled/absent. Zero sets the new P58 observer flag
to zero and retains the complete canonical engine. Mixed tuples fail closed.
P58f05 proved the observer repair. It completed the next 128-row batch in
486.4 seconds: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, six solved,
two mixed/effective groups, and 32 nonzero advantages. All timeout dimensions
were zero. Exact live weights passed over 398 leaves and 4,022,468,096
elements, and one observer marker covered all 2,048 prompt rows. The raw log
is `evidence/p58f05/run.log`, SHA-256
`73def19531ca1a9ef083a30d11ceb89696afcbe4125bd128f7ff0e7152ec06a6`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f05/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`90c179d799bb97416f1a4e6cf944a15326cef56360da179c771fad79fa02bcac`.

P58f05 then attached the alignment sidecar and stopped before trainer
forward/backward/update. `gsm8k_ab_report_policy()` already recognized the P58
arm and enforced Native-warning/Zero-strict, but its workload admission placed
P58 in a branch that accepted only `one-update/three-update`. The signed
`CANON_P34_RUN_STAGE=full` plus `CANON_P58_EXPECTED_UPDATES=1000` tuple was
therefore incorrectly rejected. This is a stale stage enumeration, not an
alignment red or missing treatment dose.

The published p58f05 repair separates P58 from the P39/P43/P44 debug-update branch and
admits only its signed Native tuple: `CANON_P58_TIM_ADMITTED=1`, no competing
DeepSWE mode, and an exact `three-update/3` or `full/1000` stage/horizon. It
does not add a flag. P58f06 proves it: the 492.7-second rollout durably wrote
128 rows (126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, three solved),
with five all-failed groups, one mixed/effective group, two incomplete groups,
and 31 effective nonzero advantages. All timeout dimensions were zero. Exact
live weights passed over 398 leaves/4,022,468,096 elements and the Native
processed-B observer covered all 2,048 prompt rows. The raw log is
`evidence/p58f06/run.log`, SHA-256
`34c6830d5b4179cf8ccdd697a0b03d9764fc75ffefa9313d5a1910914e774fd9`.
The trajectory journal is
`/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f06/debug/batch-000000.trajectories.jsonl.gz`,
SHA-256
`ddaefb3c0efc8eb7f29724c80b5aa88ab38e8b49e7bd3cf7134c4916afe2e6f3`.

Alignment then executed over 405,827 action tokens. Both
`S_decode_vs_S_prefill` and `S_prefill_vs_T_old` were shape-valid and finite;
the former was already a warning, but the P58-specific tuple still treated the
latter as blocking. That contradicted the untreated Native treatment and the
user's earlier decision that finite B-C must not stop Native training. The
local correction makes both finite serving-path boundaries warnings and
updates the classifier to accept a finite nonzero dose on either. Nonfinite,
shape, weight, replica, transaction, and optimizer errors remain hard. Zero
remains strict at all boundaries. P58f06 has no optimizer checkpoint and is
not resumable training state.

P58f07 completed all 128 real SWE RepoEnv trajectories (`N_action=436,464`),
passed pre-backward with finite A-B/B-C warnings, completed Rescore B in 26.9
seconds, and entered real value-and-grad/backward. It then stopped at the
first post-backward gate on `T_old_vs_T_current` and derived
`r_all_exactly_1`. The durable launcher marker from this attempt family shows
`T_old` was computed by one standalone 128-trajectory trainer program, while
the frozen update structure computes `T_current` in eight ordered
16-trajectory value-and-grad programs. The arrays therefore came from
different batch programs; exactness is therefore not a valid admission
requirement for an untreated Native arm.

The corrected contract preserves the complete stock quality-fix program,
including the standalone 128-trajectory `T_old` rescore. With
`use_rollout_logps=true` and sampler-IS disabled, the loss uses rollout A as
`old_per_token_logps`; `T_old` is observer-only. Signed Native now records any
shape-valid finite `T_old_vs_T_current` and finite derived ratio drift as a
warning, while the classifier requires that boundary to be present and
finite. Zero remains exact. B8 x G16, all 128 training rows, rollout logps,
loss, eight-step gradient accumulation, optimizer placement and math, commit
cadence, and every Native/Zero numerical flag remain unchanged.

P58f07 has no durable optimizer receipt or checkpoint and is not resumable
training state. P58f08 then stopped before rollout: six concurrent Pathways
heads already occupied all six `cpu-np` nodes, so Kubernetes packed the next
host-network head onto an occupied node. Port 29001 connected its CL/956357083
worker to a foreign CL/42 ResourceManager. A follow-up placement on
`deepswe-cpu-pool` started the head but could not maintain the worker scheduler
pipe across the node-pool subnet boundary. The correct infrastructure repair
is therefore not a CPU-pool or Pod-network change: retain `cpu-np` and
`hostNetwork:true`, and require hostname anti-affinity between every JobSet
`pathways-head` Pod.

P58f09 proved correct Pathways attachment and completed all 128 Step-0 rollout
slots in 1,699.1 seconds. Reset-deadline rows that terminated before first
observation had `agent.trajectory.task=None`, even though `env.task` still
contained the original input. Learner `merge_micro_batches()` dereferenced
that value and crashed before the P58 journal, alignment, forward, backward,
optimizer receipt, or checkpoint. The local repair preserves the agent task
when present, otherwise falls back to `env.task`, and fails closed if neither
is a dictionary. Compact timeout/context rows retain the existing zero policy
mask and are neither dropped nor resampled. Renderer validation requires the
exact hostname anti-affinity plus retained head/worker host networking,
JobSet DNS, and RM/PATHWAYS_HEAD route. P58f08 and p58f09 are not resumable
training state.

P58f10 ran the source containing the prior placement/input repairs and entered
real Step-0 rollout. The batch deadline prevented post-rollout merge, so the
original-input fallback remains target-unproven despite exact-image coverage.
Its 128 trajectories were still admitted with
`max_concurrency=64`, creating two sequential waves. At 3,600 seconds only
5/8 prompt groups were complete, so the batch orchestrator correctly failed
closed before durable journal, trainer, optimizer receipt, or checkpoint. The
published repair sets concurrency to 128, exactly the raw batch and exactly rollout
DP8 x max-seqs16 capacity. Episode 3,000 s, cleanup 300 s, and batch 3,600 s
remain unchanged. Individual timeout/context outcomes still become compact
zero-mask rows; only a whole one-wave batch that cannot drain is fatal. P58f10
is not resumable. After separate publication/readback, use fresh Native
`p58f11`. Zero remains deferred.

P58f11 ran the one-wave B8 x G16 geometry successfully: all 128 trajectories
completed in 1,209.2 seconds. `group_id=7`, `pair_index=14` terminated during
`env.reset`, so it used the pre-observation fallback. `SWEEnv` had stored the
normalized dataset row in `self.entry` but called `BaseTaskEnv` without a
task; only `policy_version` existed in `env.task`. The fallback was therefore
a dictionary without `prompts`, and learner processing raised
`KeyError: 'prompts'` before the durable P58 journal, alignment, trainer,
optimizer receipt, or checkpoint.

The published repair seeds `SWEEnv.task` with the normalized prompt before any
sandbox work and uses the policy-seeded environment task as the authoritative
training input for every generation. Successful and reset-timeout rows now
have the same schema. A future policy-seeded task missing `prompts` fails
immediately at collection. Compact-filter masks and the no-drop/no-resample
recipe are unchanged. The exact-image gate passes the positive timeout path,
the normal-path authority check, and a missing-key negative control. P58f11
is immutable and not resumable; at that historical checkpoint the next run was
`p58f12`.

P58f12 target-proved that repair by writing a valid 128-row Step-0 journal.
However, all 128 R2E Pods remained Kueue `scheduling_gated` until sandbox-start
timeout. Every row was therefore signed compact-filtered `ENV_TIMEOUT`, with
zero completion/action tokens; no model call occurred and `generate()` never
created sampling-transform provenance. The processed-B observer still tried
to rescore and raised `processed S_prefill must follow generate()` before
alignment, backward, optimizer, or checkpoint. Effective sandbox throughput
was zero. This is a `cpu-np`/Kueue scheduling-capacity failure, not evidence
that vLLM max-seqs or model generation was too slow.

The local repair completes the preregistered ordinary all-filtered no-commit
path. When and only when signed P58 durable metrics prove every row is compact
filtered, zero completion targets skip the observer engine after structural
and signature validation and record `engine_called=false`; no fake zero
log-probability values are introduced. Alignment accepts the empty policy mask
only with that provenance. For model/context/runtime all-compact outcomes, the
trainer makes no optimizer commit, the outer learner suppresses weight sync
and all committed-step advances, `batch_index` advances, and the next clean
prompt batch is consumed without resampling.

An entire batch that timed out before sandbox start is not treated as training
data. After its 128-row journal and bounded metrics are durable, the new
circuit breaker emits `[P58.SANDBOX_CAPACITY] BLOCKED` with
`optimizer_commits=0 prompts_consumed_after_batch=0` and raises
`BLOCKED_SANDBOX_CAPACITY` before processed rescore, alignment, trainer, or a
later prompt batch. Any inconsistent infrastructure signature fails closed.
P58f12 is immutable and not resumable trainer state. The former `p58f13`
Native-raw instruction is superseded; fresh `p58is01` Native+IS is next only
after publication/readback and live CPU sandbox admission evidence.

`origin/main` was reviewed read-only at
`c7d8950f12a9c55a976bf2e1a0d8b447d71c20b3`. Its Agent
Sandbox/SandboxFleet commit `e789573964b6f695ded85fe519040bd06a2b9f37`
is not integrated or enabled: it does not create Kueue quota, currently treats
prewarm failures as warnings, and current-plus-lookahead sizing can request
256 sandboxes for B8 x G16. A later port requires its own default-off,
Kueue-aware, fail-closed phase. Never modify or push `main`.

Never modify or push `main`. The publication target is exclusively
`yuxzhang/canon-zero-tim`; the p58f09 repair is published there as
`678bc5cfbcec386fd655e6685365c937e826d547`, and the p58f10 one-wave repair as
`44b6fb4527a8a05bf649b5140d12142e2abef83f`. Always fetch the later final
documentation tip before rendering.

## What was implemented

- additive P58 DP8 x TP8 per-role workload/profile and a `4x4x8` paired
  renderer for `native|zero` and `three-update|full`;
- frozen Qwen3-4B-Instruct-2507 B8 x G16 recipe on the 1,012-task clean list;
- explicit 16,384 `sequence-mean-token-scale` norm and effective-row
  denominator matching the pinned DeepSWE quality-fix compact-filter path;
- denominator-weighted eight-way gradient accumulation for the stock trainer;
- matching global denominator behavior in the canonical segmented path;
- all-filtered no-commit for both paths, with no resampling;
- durable full-trajectory P58 journal, separate `batch_index` and
  `optimizer_step`, restart continuity/digest verification, per-batch solve and
  signal metrics, and W&B forwarding;
- native stock-engine verification and absence checks for the complete
  canonical numerical bundle;
- independent native-only processed-B observer with absolute request-history
  targets, exact two-file manifest, and mutually exclusive Native/Zero flags;
- native finite A-B/B-C/T_old-T_current warning boundaries with finite ratio
  diagnostics, and zero all-boundary strictness;
- native stock optimizer transaction receipts plus zero explicit fixed-tree
  transaction receipts;
- P58 fail-closed postflight classifier and automatic invocation from
  `90_run.sh`; and
- negative/regression controls for P34/P44 and the shared trainer/loss paths.
- authoritative resolved-environment reload semantics so child-shell unsets
  remain absent in all later entrypoint steps.
- required hostname anti-affinity for fixed-port Pathways heads while
  preserving host-network transport; and
- pre-observation reset-timeout original-input recovery from the environment,
  with a durable normalized prompt, one schema for normal/timeout rows, and a
  hard error when no mapping or required prompt exists; and
- exact one-wave rollout admission: B8 x G16 = concurrency 128 = rollout DP8 x
  max-seqs16, without extending the signed timeout hierarchy;
- a P58 infrastructure circuit breaker that stops after durable evidence when
  every trajectory timed out before sandbox start, without rescore, trainer,
  optimizer commit, or consumption of later prompts; and
- a production-shaped one-Pod Kueue admission probe plus a read-only verifier
  for the exact queue, `cpu-np` routing, Pod gate, and selected node.

The exact run instructions and artifact interpretation are in
`canon-zero-tim/cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## Validated locally

Pinned local image ID:

```text
sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1
```

The gate covers the P58 loss oracle, unequal-effective-row gradients, real
trainer accumulation, stock/canonical all-filter discard, journal resume,
native-dose/zero-exact classifier negatives, both renderer arms/stages,
environment resolution, the full alignment suite, and relevant P34/P44
regressions. It now also covers the three sandbox-capacity circuit-breaker
controls and the production-shaped probe/verifier's Running, Pending, and
unmanaged-Pod cases. The standalone probe suite passes 4/4 on host.

The current host environment contract passes 8/8. Previously published host
validation remains profile 2/2, renderer 15/15, alignment policy 9/9, P34
static 10 suites, and P57 adjacency 105/105. In the pinned image, classifier
5/5 and the shared alignment
regression 42/42 pass; the targeted agentic/trajectory batch passes 13/13,
including the new infrastructure signature controls plus
reset-timeout prompt preservation, policy-seeded normal-path authority, and
missing-input/missing-prompt fail-closed controls. Python compilation,
the 320/320 flag-registry audit, and `git diff --check` pass. The complete
pinned-image gate emits the terminal marker above.

P58f12 is the latest target execution. It proved the 128-row journal schema but
ran zero real R2E sandboxes and no model generation; p58f07 remains the latest
attempt to enter real value-and-grad/backward, also without an optimizer
receipt/checkpoint. The training venv loads JAX/libtpu. After the self-created,
unlocked zero-byte libtpu lock was removed, the runtime could not obtain
`CHIPS_PER_HOST_BOUNDS` from instance metadata; the bounded runner emitted
`P58_ONEHOST_ALIGNMENT_BLOCKED reason=device_inventory_timeout timeout_secs=30`
instead of PASS. No one-host or CPU test proves live Kueue admission, 128-chip
Pathways, real R2E rollout, or TPU training.

## Next executor sequence — native only

1. Read `state.md`, `plan.md`, this handoff, the superseded P58.4N phase file,
   the active `phases/p58-5-native-full.md`, and
   `cluster/P58_DEEPSWE_TIM_RUNBOOK.md` completely.
2. Fetch `yuxzhang/canon-zero-tim`, detach at its exact remote-tracking SHA,
   prove a second remote readback matches, and require a clean tree. Never use
   `main`.
3. Rerun syntax, `git diff --check`, the P58 renderer/profile/environment
   tests, and the pinned exact-image gate. On a real direct-attached four-chip
   v5p host, also run
   `tests/p58_deepswe_native_zero/run_onehost_alignment_v5p.sh`; require its
   renderer-profile-policy PASS marker without treating it as a Qwen/R2E or
   DP8 x TP8 training result.
4. Publish or select a client image by immutable registry digest and verify the
   mounted Qwen3-4B-Instruct-2507 weights and frozen clean-list digest without
   printing credentials.
5. Follow the runbook's `P58 sandbox capacity gate` exactly. Derive a real
   `docker_image` from the frozen clean list, render the production-shaped
   one-Pod probe, preserve its digest, and run server-side dry-run. Applying
   the probe is a separate user/operator-approved Kubernetes mutation. Once
   applied, require
   `P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only`, preserve Pod,
   matching Workload, LocalQueue, ClusterQueue, ResourceFlavor and `cpu-np`
   node evidence, then delete only that exact probe and confirm deletion.
   Separately confirm capacity for 128 x 2 requested CPU = 256 CPU and
   128 x 4 GiB = 512 GiB requested memory, plus head/cluster overhead. A
   one-Pod PASS is necessary but does not prove full-batch capacity. Never
   remove the queue label to bypass Kueue.
6. After the exact Native/no-IS JobSet is archived and deleted, and only after
   P58.9 is published with an exact remote readback, render
   `arm=native, stage=full, --sampler-is` with fresh run-id `p58is01` and
   worker sentinel `tpu-v5p-slice`. Start from the original frozen base; never
   resume the collapsed Native checkpoint. Require renderer recipe
   `native-is`, JobSet label `canon.zero-tim/sampler-recipe=token-is`, sampler
   disable tuple `0:0`, token threshold `2.0`, trainer old logps, present TIS
   weights, and no group filter. Require exact `4x4x8` topology and no literal
   `cloud.google.com/gke-nodepool: tpu-v5p-slice`; require B8 x G16 =
   concurrency 128 = rollout DP8 x max-seqs16; require head pool `cpu-np`,
   head and worker host networking, exact required hostname anti-affinity over
   the JobSet `pathways-head` label, both JobSet DNS-publication settings, and
   the exact generated head DNS in both worker RM fields. Preserve the
   YAML/digest and run server-side dry-run before the separately approved apply.
7. If an ordinary model/context/runtime all-compact batch occurs, require all
   of these markers before allowing the loop to consume the next prompt batch
   without a commit:
   `[CANON_RESCORE] empty_completion_batch ... engine_called=0`, signed
   alignment with `N_action=0 ... no_signal_admitted=true`,
   `[DEEPSWE.COMPACT_FILTER] optimizer_boundary_skipped effective_rows=0`, a
   Native optimizer transaction with `commits=0`, and
   `[P58.COMPACT_FILTER] ... optimizer_commits=0 ... weight_sync=0`. Require
   identical trainer/RL/optimizer/policy versions and an incremented
   `batch_index`; never retry the same prompt batch. If instead
   `all_sandbox_start_timeout_batch=1`, require the durable journal and
   `[P58.SANDBOX_CAPACITY] BLOCKED ... optimizer_commits=0
   prompts_consumed_after_batch=0`, followed by `BLOCKED_SANDBOX_CAPACITY`.
   That JobSet must stop before rescore/trainer or a later prompt batch; return
   to the capacity gate with a fresh run id after the infrastructure issue is
   resolved.
8. Require stock preflight, one P58 stock-observer processed-B marker, exactly
   one signed `[P58.TIM_RECIPE] ... recipe=native-is ...` marker, exact live
   weights, shape-valid finite Native boundaries/ratios, finite
   forward/backward, and the first optimizer commit.
   Then monitor commits 1–3 without stopping a healthy job. Continue through
   checkpoint 8, updates 32 and 100, then every 100 updates.
9. Require the full Native-arm classifier JSON to say `PASS` and the separate
   Native+IS recipe receipts to pass, including a finite nonzero serving-path
   dose on A-B or B-C, finite trainer-program observation, exactly 1,000
   commits, device optimizer, complete journal, cleanup, evaluation,
   checkpoint, and transaction receipts.
10. Do not render or apply Native raw or Zero.

Do not reuse any failed `p58c01` through `p58c05` or `p58f01` through `p58f12`
YAML/run root. P58f03 through p58f07 have diagnostic trajectory/alignment
evidence but no durable trainer update or optimizer checkpoint, so none is
resumable training state. The attempts remain immutable failure evidence.
P58f08 has no trajectory at all; p58f09 completed rollout processing but
crashed before the durable journal; p58f10 timed out at the batch orchestrator
before the journal; p58f11 completed the batch but failed learner preprocessing
before the journal. P58f12 has a valid diagnostic trajectory journal but no
trainer/optimizer checkpoint, so it is also not resumable training state.
if a CL mismatch recurs, collect all three head-container logs plus one worker
log and verify its resolved RM address before deleting the failed JobSet.
Earlier evidence remains under `evidence/p58c01/`, `evidence/p58c02/`,
`evidence/p58c03/`, and
`evidence/p58c04/`. The
p58c03 hashes are `15aa9968200c55a02ef47c72c5e209277397835e1752a4dbd9699fce3b2c42b4`
for `run.log` and
`d5e8b5b1941aa5632fa6267cfdac445727c175bf8d2bbcc79c1ece7cf7aba1e2`
for `head_container.log`.
P58c04 hashes are
`f5caf2efb70bfec083a4454e441ce7f4b5b0632abbd206439ba9497bca5a6a40`
for `run.log` and
`a311eb64ee30b1fa0a168b68d9f17661756ed9cb3b272dd19d9bdddbc7f34666`
for `env.sh`.
P58c05 admission hashes are
`d0845e3da4fc106afa3e0f8aa4af387cf44335f21ba696713fd382bbc32b4cf5`
for `workload.yaml` and
`cbcf60c467c758601f42221ce050f5dac329ab1f696ba735c60ac809b33fec05`
for `workload_describe.txt`.

## Important operational semantics

- `use_rollout_logps=true` remains enabled. For the active Native+IS recipe,
  token sampler-IS and TIS correction are enabled only through the registered
  `0:0` tuple at threshold `2.0`; trainer logps define the old policy and TIS
  weights must be present. Group clip/filter, degenerate-group masking, and
  flat-group resampling remain off. Native raw is retired and must not resume.
- All-zero/all-one reward groups remain. They naturally produce zero RLOO
  advantage and are logged.
- Compact-filter statuses are not malformed trajectories. They remain in the
  full journal but have zero policy mask. Structural missing/duplicate/parser
  failures remain fatal.
- A Kubernetes sandbox start exception must propagate after deletion is
  confirmed. `ENV_TIMEOUT` is an admitted compact-filter status; a
  half-created RepoEnv with `container=None` is forbidden. If an entire
  Native+IS batch has zero confirmed Running pods, classify infrastructure
  capacity/scheduling before another launch instead of patching websocket
  decode or inventing a successful trajectory.
- Read `deepswe/all_sandbox_start_timeout_batch` first. Value `1` means the
  effective R2E environment throughput was zero and the model was not the
  first bottleneck. A zero sandbox-start ratio plus a nonzero
  `deepswe/status/model_timeout_ratio` instead points to model-serving
  throughput. W&B dimensions are fixed and low-cardinality; detailed
  scheduler text is available only in the bounded raw marker.
- If an ordinary model/context/runtime batch is entirely compact-filtered,
  `batch_index` advances but trainer/RL steps, `optimizer_step`,
  `policy_version`, weight sync, and commit count do not; the next prompt batch
  is consumed without resampling. If all rows timed out before sandbox start,
  the durable journal is followed by `BLOCKED_SANDBOX_CAPACITY` and no later
  prompt is consumed. A partial/digest-mismatched journal always stops
  fail-closed. Do not describe p58f12 as resumable trainer state.
- The native arm is stock numerical training plus observation. It must not
  inherit `CANON_FIXED_AR`, `CANON_LOGPROB_M`, the canonical module, VJP2, or
  the excess-precision pin. The zero arm retains the complete bundle.
- Native processed B must come only from the P58 stock observer while
  `CANON_PROMPT_PROCESSED_LOGPROBS=0`; require exactly one
  `[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS` marker. Zero must keep
  the stock observer off and use its canonical processed engine. Never enable
  both to make a run pass.
- Exact live-weight attestation is shared evidence, not a native numerical
  treatment. Native may use only the observer interface and must keep the
  canonical adapter absent; zero uses the registered adapter. Require exact
  leaf equality and public DP8 x TP8 mesh provenance before A/B/C.
- `env.sh` is an authoritative managed-environment snapshot, not a layered
  override. If the parent retains a renderer variable that the profile made
  absent, the p58c03 regression must fail before publication.
- `CANON_P34_TRAJECTORY_CAPTURE=0` is intentional. P58 uses its own full
  trajectory journal and does not enable the older large P34 alignment-tensor
  capture mode.
- Optimizer state is TPU device-resident in both arms. Host offload is a hard
  configuration error.

## Claim ceiling

A native 128-chip PASS proves only that the untreated Qwen3-4B clean-data
training path completed the signed 1,000-update full campaign. It does not
estimate a native-versus-zero effect, prove zero-TIM, isolate one kernel,
reproduce DeepSWE-32B, prove packing, or establish 256-chip production
behavior. No finite Native serving-path mismatch on either A-B or B-C is
`NO_TREATMENT`; missing evidence or interrupted execution is inconclusive.


## P58.14 historical append correction

The earlier append incorrectly described JAX tracing markers as completed
36-layer VJP/backward execution. The authoritative P58.14 account is the
highest-priority section at the top of this handoff and
`phases/p58-14-device-sharding-mismatch.md`: rollout completed, but trainer
execution did not begin before the disjoint-device JIT error. Retained
evidence remains under `evidence/p58z03_device_sharding_error/`.
