# P58 Qwen3-4B DeepSWE native-first runbook

P58 retains a paired 128-chip study design. Each arm uses one `4x4x8` v5p
slice, synchronously split into a 64-device rollout role and a 64-device
trainer role. Both roles are DP8 x TP8. The two arms share data, seeds,
sampling, loss, optimizer, deadlines, artifacts, and update horizon.

- `native` preserves the stock serving/trainer numerical programs from the
  pinned DeepSWE quality-fix lineage. Finite A-B and B-C differences are the
  measured serving-path treatment dose; finite T_old-T_current and derived
  ratio differences are additional stock-program observations. Nonfinite
  values, invalid shapes, replica/transaction failures, and corrupt evidence
  remain fatal.
- `zero` enables the complete canonical numerical bundle. A, B, and C must be
  exact at every admitted boundary.

P58 does not modify `main`. Rendering and local validation do not authorize a
Kubernetes apply. An operator must separately approve image publication and
each launch.

## Default-full TiTO admission gate

All P58 arm/stage renders must contain both:

```text
env: CANON_P34_DEEPSWE=1
label: canon.zero-tim/token-transport=tito
```

The raw environment value is required before profile sourcing; the label is
durable JobSet provenance.  Both are in the paired recipe signature.  A
missing/wrong value is a renderer failure, not a warning.  TiTO has no
independent disable flag: it is selected by the common DeepSWE identity for
Native, Native+IS, Zero, and Zero-HP alike.

For the default Qwen3-4B-Instruct Zero-HP full render, require the wrapper
terminal marker to include:

```text
V1_DEEPSWE_ZERO_HP_RFULL_READY transport=token-in-token-out launch=not-executed
```

Before any 128-chip target, run the focused renderer tests, P34 static, flag
audit, and digest-pinned P58 exact-image gate.  The current local proof on
source `18f29c56daf471cc0ac011396d7c7a09f35d695b` plus its recorded dirty diff
also has a real direct-v5p controlled carrier:

```text
label: p58s25titoctl_20260830t0713z
classification: EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS
N_action: 2413
S_decode_vs_S_prefill differing_elements: 0
S_prefill_vs_T_old differing_elements: 0
process status: 42 (controlled pre-backward exit)
bundle sha256: a68925aa95aaeddcdc9f3f0be625aa92418b221959e1ef11cdc8c7f0ebbbcb35
```

That run validates Qwen3-4B DP1xTP4 real-R2E TiTO and strict alignment only.
It does not validate backward, TP8, Pathways, 128-chip behavior, optimizer
commit, or production readiness.  Publication preparation cleanly rebased the
implementation onto exact operator parent
`cd32949e9b63b927e99f3cfba724f4f5f6d03cda`, then rebased once more onto
`e89272d1d6c99b8f3c5014f0974b4fe57f2a4156` after a non-overlapping Qwen3
embedder-sharding update; render only from the final clean remote readback SHA
containing this entry, never from the older evidence SHA.

## K03 correction: exclusive topology belongs on the JobSet

K03 failed before rollout when `vpod.kb.io` rejected indexed worker followers.
The manifest placed
`alpha.jobset.sigs.k8s.io/exclusive-topology=cloud.google.com/gke-nodepool`
on the worker Pod template instead of JobSet metadata, so follower admission
lacked the JobSet-level placement context.

P58 now requires that annotation exactly once on JobSet metadata and forbids
the worker-Pod copy. Kueue-managed values `auto`, `none`, `any`, and
`tpu-v5p-slice` omit literal nodepool affinity and let the selected flavor or
NAP pool satisfy the JobSet placement contract. A concrete pool remains legal
and exact. Every render retains:

```text
cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
cloud.google.com/gke-tpu-topology: 4x4x8
```

Before apply, verify the selected ResourceFlavor or explicit pool read-only,
then require a server-side dry-run. Never hand-edit K03 YAML. K03 produced no
trajectory or checkpoint and is not resumable.

## Publication/readback gate for the optimized Zero/full profile

Implementation commit `fb178803d53ff562cefdfdc8e7b3fac3563d9d6e` contains
the accepted P58.23 one-host evidence and production wiring. Before rendering,
fetch the operator branch, record its exact remote SHA, and verify that commit
is an ancestor. Run the complete P58 fixed-image gate, then use only
`prepare_deepswe_zero_hp_full.sh`; it renders but never applies the JobSet.

The resolved full profile must be Qwen3-4B-Instruct-2507, B8xG16, 1,000
updates, rollout DP8xTP8 plus trainer DP8xTP8, strict alignment, and
device-resident optimizer. Require P59 rank-parallel backward/checked VMA,
P67 P59-only VMA scope, first-update and P63 finite/clip gates,
`fingerprint-hybrid`, `first-group-warmup`, `batched-commit`, and P71 forward
scan. `CANON_DP_COLLECTIVE_REDUCE` must be absent. The last local render was
verification only (manifest SHA-256
`61b837dbc9915373c931eebfbbee0fc67c75f9726d7db3893b108c67eac1331c`);
render a fresh YAML from remote readback and obtain separate launch approval.

## P58.23 direct-v5p optimized backward — completed

Do not use global batch size one.  The accepted one-host backward carrier is
Qwen3-4B-Instruct-2507 DP1xTP4 with global B2xG2, four immutable real
trajectory rows, prompt/response `2048/512`, and K=2560.  One strict-exact
real Scrapy prompt pair is repeated as two physical groups; both groups are
mixed `[1,0]`; the classifier requires four finite nonzero advantages and
zero per-group sums.  Global batch and mini-batch are 2.  Trainer/logprob
microbatches stay 1 only to bound HBM.

The treatment is the current optimized trainer route: P28 segmented
forward/train plus G6, P29 full train, P30 sparse/reuse/release/reshard, and
P71 forward scan.  P59 is off because DP1 cannot exercise rank-parallel
backward.  Do not launch a serial-reference arm.

The replay source directory and required manifest/journal hashes are:

```text
/mnt/disks/tunix-data/deepswe-replay-sources/p58-q4-b2g2-k2560-v2
482d7934a95207d0d77bb4857fbb200d7b367cbf437dda6585937b20909afa8f
091a9273c2067876fbee1996ee853e3c8e861352e307cd5fb94fea2563aec456
```

Run with a fresh label:

```bash
P58_ONEHOST_ALLOW_DIRTY=1 \
  bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_trajectory_replay_docker.sh \
  <fresh-label>
```

The wrapper enforces 1,800 seconds and compilation cache
`/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-systemopt-b2g2-k2560`.
A valid return has strict A=B=C, finite nonzero backward, unchanged model and
TPU optimizer state, zero commits, and a verified package.  Timeout or a
missing terminal marker is incomplete.  This replay does not execute a fresh
sandbox/rollout and cannot certify TP8, Pathways, P59, 128-chip behavior, an
optimizer update, prompt diversity, or production readiness.  Replay v1 is
preserved failure evidence only; its Coverage rows were alignment-red and are
forbidden in the acceptance carrier.

Accepted target `p58s23optb2g2g_20260830t0132z` returned all required markers:
strict A=B=C over 1,254 action tokens, four finite nonzero advantages, two
trajectory microsteps, repeat-exact gradient norm `8.544539451599121`, device
optimizer state unchanged, and zero commits.  Its profiled repeat took 12.418
seconds and peak HBM was 52.5 GiB.  Artifact root:
`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s23optb2g2g_20260830t0132z`;
return-bundle SHA-256:
`7d33ee791146d2309c16866d8e30f15f0f012e05e88f6c795b587938f973f795`.
Do not rerun this gate merely to obtain a new label.  Publication and TP8
promotion are separate approvals, and global batch size one remains forbidden.

## Production Zero-HP full system-optimization route

The production Qwen3-4B Zero/full/HP renderer now consumes the P74-era system
bundle. This is wiring readiness, not launch readiness: P58.23 has completed
only the local DP1xTP4 replay gate, and no selector-absent 1,000-update full
run may launch until the TP8 promotion is separately defined, constructed,
and explicitly approved by the user.

After the implementation is committed, published, paired with a digest-pinned
image, and checked out clean at the approved SHA, render only with:

```bash
bash canon-zero-tim/tasks/v1-system-optimization-workload-rollout/prepare_deepswe_zero_hp_full.sh \
  <approved-40-character-sha> \
  <matching-registry-image@sha256:digest> \
  <fresh-output.yaml> \
  <fresh-run-id> \
  <worker-nodepool-or-kueue-sentinel> \
  <model-pvc>
```

Require `V1_DEEPSWE_ZERO_HP_RFULL_READY ... launch=not-executed`, then inspect
the resolved environment for checked-VMA/P67/first-update/P63 plus:

```text
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
```

`CANON_DP_COLLECTIVE_REDUCE` must remain absent. Native raw, Native+IS,
ordinary Zero, three-update, checked-VMA diagnostics, and seam diagnostics do
not inherit these selectors. The wrapper has no apply step; server dry-run,
capacity admission, Kubernetes apply, and target monitoring each remain
separately approval-gated.

## P58.19 three-round coarse seam localization — render-only recipe

The current diagnostic is one 128-chip JobSet with three sequential
frozen-weight rounds.  It is not three jobs and it is not training.  Every
round uses rollout DP8xTP8 plus trainer DP8xTP8, B8xG16 (128 trajectories),
Qwen3-4B-Instruct-2507, the reviewed 1,012-task list, 16K response, 50 turns,
seed 42, concurrency 128, prefix cache off, fixed lm-head, continue-decode 8,
and strict B-C.  Backward and optimizer commit remain zero.

`continue-decode 8` is part of this signed carrier and must not be disabled to
work around an observer error.  The `p58-seam-v1` capture hook treats
`standard` as the only tensor-strata source.  If the scheduler enters
`continue_decode`, the hook preserves chronology, emits
`CANON_P58_CONTINUE_DECODE_OBSERVER_BYPASS ... tensor_capture=0`, and returns
before incident/tensor capture.  Postflight requires at least one exact bypass
marker for this selector and rejects the marker for every other workload.
Unknown program paths still fail closed.

The seam/tail observer budget is 4 GiB **per diagnostic round**, not one
cumulative run budget.  Record indices remain monotonic across rounds; only
the byte counter resets after a sealed `0→1` or `1→2` transition.  Before a
successful postflight, require one exact
`[P38_OBSERVER_ROUND_BUDGET] ... bytes=0` receipt for each
`label={seam,tail}` and `round={0,1,2}`.  Missing receipts, a round jump, or a
foreign profile receiving this reset is fatal.  Do not raise the value in a
rendered YAML or delete earlier records to recover space.

Prepare from a clean checkout only after the source and matching digest-pinned
image have each been published with explicit approval:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_coarse_seam_localization.sh \
  <fresh-run-id> \
  <matching-image@sha256:...> \
  <worker-nodepool-or-auto> \
  /tmp/p58-coarse-seam.yaml
```

Required render marker:

```text
P58_COARSE_SEAM_PREPARE_PASS ... rounds=3 backward=0 optimizer_commits=0
```

The wrapper only renders YAML.  Before any separately approved apply, inspect
the rendered identity: `diagnostic=p58-seam-localization`,
`diagnostic-rounds=3`, `backward=0`, `optimizer-commits=0`, exactly one
`CANON_P58_SEAM_LOCALIZATION=coarse`, and no P58.18 checked-VMA selector.  Do
not hand-set subordinate P38 fields; `00_env.sh` must attest the derived
`p58-seam-v1` durability profile, per-round 4 GiB byte budget, and bounded
`[1686,4096)` layer observer,
with capture strata `1686,2512,3072,3584,4096`. These values cover the five
known first-red prefixes from p58z07 and P58.18. A repeat with `records=0`
remains INCONCLUSIVE and must return the complete log plus request journal;
do not lower the window ad hoc in rendered YAML.

During execution, a round may advance only after its P58 round classifier is
PASS and the archive has passed upload/read-back verification.  A completed
return contains three distinct `ROUND_COMPLETE` receipts, their three
`p58-seam-round.classification.json` files, and the aggregate
`p58-seam.classification.json`.  The aggregate requires exactly three
precheck markers, one controlled exit, finite positive A-B in each round,
exact B-C, a common first-red coarse signature, and no VJP/backward/commit.

Decision routing:

| Result | Next action |
|---|---|
| same first-red layer/checkpoint in all rounds | prepare a separately reviewed 15-checkpoint fine scan of that layer |
| different checkpoints but a common interval | retain evidence and refine only that interval |
| backbone exact; terminal path first red | route to the LM-head/log-normalizer discriminator |
| layer-0 input already red | inspect embedding, position, and KV handoff |
| missing join/round, B-C red, endpoint drift, or training activity | INCONCLUSIVE/FAIL; repair the prerequisite only |

Return the entire persistent or verified GCS run root, including `run.log`,
`pre_alignment.jsonl`, full debug trajectories, per-round seals, and the
aggregate classification.  The legacy DP1xTP4 one-host carrier does not
exercise this TP8 layer observer, so its result cannot substitute for the
exact target.  No construction marker authorizes image publication,
Kubernetes apply, or a TPU launch.

Local construction gate:

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Required terminal substring is `continue_decode_observer=1 ...
checked_vma_aba=1 coarse_seam=1 ... regressions=1`.  This CPU dependency-image result checks contracts and
neighboring workloads; it does not run Qwen3 on TPU and is not target evidence.

## Historical: P58.18 checked-VMA matched triplicate — completed recipe

The next Zero diagnostic is three separate 128-chip Step-0 JobSets, logically
named `ON-A/OFF/ON-B`. Each uses rollout DP8xTP8 plus trainer DP8xTP8,
Qwen3-4B-Instruct-2507, the reviewed 1,012-task list, B8xG16, 16K response,
50 turns, seed 42, concurrency 128, fixed lm-head, continue-decode 8, prefix
cache off, and complete trajectory/pre-alignment artifacts. Every arm exits
code 42 before VJP/backward/optimizer.

`on` derives checked-VMA/P66/P67=`1/1/1`; `off` derives `0/0/0`. Both derive
first-update/P63=`0/0`, keeping the zero-commit controls matched. With the
selector absent, the production Zero-HP profile remains `1/1/1/1/1`. Never
set the five subordinate flags by hand.

Prepare all three from one clean published revision:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_aba_wave.sh \
  <fresh-wave-id-at-most-12-chars> \
  <matching-image@sha256:...> \
  <worker-nodepool> \
  /tmp/<fresh-p58-aba-output-dir>
```

This is render-only. It refuses a dirty checkout, source/origin mismatch,
unpinned image, or existing output root. It writes three YAMLs beneath
`jobsets/`, plus render and verify receipts. Required terminal markers are:

```text
P58_CHECKED_VMA_ABA_RENDER_PASS ... jobs=3 tpu_request=384 ...
P58_CHECKED_VMA_ABA_VERIFY_PASS jobs=3 selectors=on,off,on ...
P58_CHECKED_VMA_ABA_WAVE_READY ... jobs=3 tpu_request=384 ...
```

Concurrent submission requires aggregate capacity, not three independent
per-job assumptions: 384 TPU chips, three anti-affined CPU head nodes, and
384 sandbox slots (768 CPU and 1,536 GiB requested memory at the signed
per-sandbox contract). Server-side dry-run, aggregate capacity inspection,
and apply are separate user-approved steps. A directory apply makes all three
eligible concurrently, but only Kueue admission and Pod start timestamps prove
that all three actually overlapped.

After completion, keep each full run root. Every per-arm classifier must be
PASS, exact B-C, one valid A-B outcome, exactly 128 durable trajectories, and
zero VJP/backward/commit. Aggregate with:

```bash
python3 canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/classify_p58_checked_vma_aba_wave.py \
  --wave-verify <wave-output>/wave-verify.json \
  --on-a <on-a-root>/p58_checked_vma_on.classification.json \
  --off <off-root>/p58_checked_vma_off.classification.json \
  --on-b <on-b-root>/p58_checked_vma_on.classification.json \
  --output <fresh-output>/p58_checked_vma_aba.classification.json
```

Concurrent ON-A/OFF/ON-B is a matched OFF control with two ON replicates, not
a temporal ABA sandwich. ON RED / OFF exact / ON RED supports the selector as
the reproduced discriminator. RED/RED/RED rejects checked VMA as sufficient.
Nonreplicating ON controls are inconclusive. Cross-run token identity is not a
hard gate; per-arm durability, finiteness, exact B-C, and zero training are.
The local dependency-image construction gate is green only when its terminal
marker contains `checked_vma_diagnostic=1 checked_vma_aba=1`; it is not TPU
target evidence.

## 2026-08-27 P58.17 exact-geometry checked-VMA-off diagnostic — source published

The next target is not another 1,000-update retry. It is one real Step-0
Qwen3-4B-Instruct-2507 diagnostic on the exact 128-chip disaggregated geometry:
rollout DP8xTP8 plus trainer DP8xTP8. It keeps the reviewed 1,012-task clean
list, B8xG16 (128 trajectories), 16K response, 50 turns, seed 42, prefix cache
off, fixed lm-head, continue-decode 8, full trajectory journaling, and strict
A/B/C pre-alignment. It then exits before VJP/backward/optimizer commit.

`CANON_P58_CHECKED_VMA_DIAGNOSTIC=off` is the only selector. The profile
derives checked VMA, the P66 compatibility alias, P67 scoping, first-update
gate, and P63 clip to zero as one fail-closed tuple. The selector cannot be
combined with `--high-performance`, Native, three-update, warning-only, or
manually supplied subordinate values. When it is absent, the production
Zero-HP tuple stays unchanged at checked-VMA/P67/first-gate/P63 = 1/1/1/1.

The implementation is published as
`b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9`; its first remote readback
matched HEAD/FETCH_HEAD/tracking with ahead/behind `0/0`. The older base
`9177b00b62d07a7d26a292126ba37b42f174f6de` does not contain P58.17. Fetch the
final operator tip after the publication checkpoint and never improvise the
flags in YAML. After the user separately approves a matching digest-pinned
image, use:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-character-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_off_diagnostic.sh \
  <short-fresh-run-id> \
  <matching-image@sha256:...> \
  <worker-nodepool-or-auto> \
  /tmp/p58-checked-vma-off.yaml
```

This wrapper is render-only. It refuses a dirty tree, requires local HEAD and
`origin/yuxzhang/canon-zero-tim` to equal `P58_EXPECT_SOURCE_SHA`, and refuses
to overwrite an existing YAML. Kubernetes server dry-run and apply each need
separate operator approval. Never hand-edit its output.

The rendered JobSet name contains `zero-hp-vmaoff-precheck`; labels state
`diagnostic=p58-checked-vma-off`, `fixed-lm-head=1`, `backward=0`, and
`optimizer-commits=0`. A valid target must end with exactly one profile marker,
one precheck round, one code-42 controlled exit, and one
`P58_CHECKED_VMA_DIAGNOSTIC_CLASSIFICATION`. Its outcome is either
`A_B_EXACT_WITH_CHECKED_VMA_OFF` or `A_B_RED_WITH_CHECKED_VMA_OFF`; B-C must
remain exact. Any fixed-head VJP, P59/P66 backward, global step, nonempty
update report, missing trajectory, or nonfinite value fails closed.

Return the entire persistent root:

```text
/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/
  env.sh
  run.log
  weight_attestation.jsonl
  pre_alignment.jsonl
  debug/run_manifest.json
  debug/batch-000000.trajectories.jsonl.gz
  debug/batch_metrics.jsonl
  p58_checked_vma_off.classification.json
```

`updates.jsonl` must be absent or empty. This target decides whether checked
VMA is sufficient to explain `p58z07`; it does not test training, optimizer
correctness, convergence, or production readiness.

## 2026-08-27 P58.17 one-host decode/prefill seam probe — local result

Do not launch another full P58 training job merely to repeat the `p58z07`
pre-backward RED. The bounded first step is the tracked one-host carrier:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch -c local/p58-seam-p58s01 origin/yuxzhang/canon-zero-tim
git rev-parse HEAD
git status --short --branch
export P58_ONEHOST_EXPECT_HOSTNAME=THE_EXACT_OUTPUT_OF_HOSTNAME
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_seam_probe_docker.sh p58s01
```

Run only from the final published operator tip on a clean named `local/*`
branch and one direct-attached four-chip v5p host. Implementation
`b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9` contains P58.17; no executor may
use historical base `019d7a7e1cb7763b2ad4ffdc35e84bf9c217afe4` as though it did.
The script performs read-only prerequisite/identity checks, installs the
canonical overlay into the fresh artifact root, launches one real signed
Pillow task with Qwen3-4B DP1xTP4/G2/4K/16 turns, preserves full trajectories,
and runs strict decode-vs-prefill alignment with no optimizer commit.

Use the Docker wrapper, not host Python. This host exposes TPU through
`/dev/vfio`; the pinned privileged image is the proven direct-v5p carrier and
mounts the host Docker socket for sibling R2E sandboxes.

The final success marker names a ready-to-return tarball and checksum. Return
exactly those two files; the tarball contains the raw log, complete trajectory
journal, pre-alignment, classifier, manifest, checksums, and return note. The
classifier joins every bounded mismatch to the durable token/action-mask/
decode-logprob arrays. A finite RED and an exact TP4 result are both bounded
diagnostic outcomes. Missing action tokens, non-finite values, malformed
schema, count drift, or an unjoinable mismatch fail closed. Exact TP4 does not
clear DP8xTP8; it requires a later separately approved exact-geometry carrier.
The script does not replay fixed historical token IDs and makes no TP8,
Pathways, backward, optimizer, or production claim.

The local direct-v5p execution `p58s17` returned:

```text
PASS / FINITE_RED_REPRODUCED
2/2 SUCCEEDED trajectories; N_action=4808; optimizer_commits=0
S_decode_vs_S_prefill: 2488 differing elements, max_abs=1.3662147521972656
S_prefill_vs_T_old: 988 differing elements
bundle SHA-256: 6285b5d2e8958ee85bd4b4190beaa240c7239ad6d07165a0948d7ba7f2b32eee
```

This proves the runnable real-R2E carrier and rejects a simple one-token
shift. It is not the same fingerprint as `p58z07`, where B-C was exact.
One-host uses only the generated canonical runner overlay; the TP8-only
Qwen3/linear/embed/attention/RPA overlays are deliberately excluded. The next
causal experiment is therefore a separately admitted Step-0/no-commit
checked-VMA-off selector on the exact 128-chip DP8xTP8+DP8xTP8 topology. Do
not hand-edit the full profile or launch a new 1,000-update job to obtain it.

Full rationale and return protocol are in
`tasks/p58-deepswe-native-zero-comparison/HANDOFF.md` and
`phases/p58-17-decode-prefill-seam-probe.md`.

## 2026-08-27 P58.16 NNX loader-metadata retry override — source published

Do not follow the older P58.15 instruction to launch `p58z05`. Immutable
`p58z06` completed model load/warmup on the exact 128-device Qwen3-4B Zero-HP
geometry, then failed before rollout because the live Pathways dummy loader
adds `_is_loaded=True` to all 398 NNX parameters while the weight-free trainer
clone does not. Flax includes this provenance in raw State treedef equality.
No trajectory, backward, optimizer commit, or checkpoint exists; the run is
not resumable. The evidence log does not embed its source SHA, so do not infer
one.

P58.16 normalizes only exact `_is_loaded=True` on copied Variables. Any other
marker value and every other metadata/path/type/leaf/shape/dtype drift remain
fatal. The same contract protects segmented backward. The local complete
pinned-image gate passes:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

Implementation `dba5211ac4945fefb50337603c800d9f8e3d37b5` is published and
read back from `yuxzhang/canon-zero-tim`. No matching image was published.
Stop here until the user separately approves image publication. Require exact
source/image readback, rerun the complete gate, and pass sandbox capacity. A
later separately approved launch must use fresh `p58z07` and require exactly
one of all four receipts:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer state contract PASS relation=disjoint leaves=398 normalized_loader_metadata=_is_loaded live_markers=398 reconstruction_markers=0
[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt relation=disjoint graph=abstract-clone mesh_bound_jits=2
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
```

Only after those receipts may the operator wait for trainer old/current logps,
strict A=B=C, finite nonzero 16-group backward, and the coherent update-0
transaction. `p58z01`-`p58z06` remain immutable and must not be resumed.

## 2026-08-26 P58.15 nested-JIT retry override — source published

This section supersedes the P58.14 instruction to launch `p58z04`. That target
already ran from `3f159250c4781b3faafde238f768457a0478446b`, emitted the two
P58.14 placement receipts, and completed all 128 Step-0 trajectories in 1,709
seconds. Its eight `MODEL_TIMEOUT` and one `MAX_CONTEXT_LIMIT_REACHED` rows
were compact-filter statuses, not the fatal event.

The hard error occurred in the first trainer old-policy-logprob call:

```text
ValueError: Received incompatible devices for jitted computation.
```

Trainer state was correctly resident on the trainer 64-device role, but the
`jit inside jit` device list was still the disjoint rollout role. P58.14
rebound the adapter's explicit shardings but reused vLLM's `model_fn` and
`compute_logits_fn`; those inner JITs captured rollout output shardings when
the engine was initialized. Preserve `p58z04`; it has no optimizer checkpoint
and is not resumable.

P58.15 reconstructs the identical live NNX graph weight-free on the trainer
mesh, validates exact tree/shape/dtype equality, rebuilds the two nested JITs
on trainer devices, and uses that graph for segmented trainer backward.
Serving remains rollout-bound. Native, colocated, strict A=B=C, loss,
sampling, optimizer, and B8xG16 are unchanged.

Before any target, the complete matching-image gate must end with:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

This CPU gate is not Pathways/TPU proof. Implementation commit
`f60cdd569c2737df6cb2968125c8e42680938981` is published only on
`yuxzhang/canon-zero-tim`; `main` remains untouched. Fetch/read back the final
operator tip and prove it contains that implementation. Image publication and
Kubernetes launch remain separately approval-gated. After matching-image
readback and sandbox admission, use fresh Attempt-0 id `p58z05` and require
exactly one of each:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt relation=disjoint graph=abstract-clone mesh_bound_jits=2
```

The full classifier rejects missing/duplicate receipts. Next require completed
trainer old/current logps, strict A=B=C, finite nonzero 16-group backward, and
one coherent update-0 transaction. If it passes, continue the same full job to
1,000 commits. Never resume or overwrite `p58z01` through `p58z04`.

## 2026-08-26 P58.14 trainer-mesh retry override — source published

This section supersedes the P58.13 instruction to launch `p58z03`. That target
already ran from `8eb65480d3705d96ab282799ad5a6c1901596248`, completed all
128 Step-0 trajectories, and proved fixed-head global/local M=`2048/256`.
It then failed before trainer execution because the canonical differentiable
adapter applied rollout-role sharding constraints to trainer-role state:

```text
ValueError: Received incompatible devices for jitted computation.
```

The two device-id lists are fully disjoint 64-device roles, not merely a
different ordering. Pallas/VJP `PATHTRACE` lines before the exception were
emitted during tracing and do not prove forward/backward execution. Preserve
`p58z03`; it has no optimizer checkpoint and is not resumable.

The local P58.14 repair passes trainer state into adapter construction and
rebinds only the differentiable canonical forward to an engine-axis mesh over
the trainer devices. Serving remains rollout-bound. The mesh-bound canonical
log-softmax callable is instantiated once per role from the same factory and
math. Exact DP/TP equality is required; partial device overlap fails closed.
Native and colocated behavior are unchanged.

The full local dependency-image CPU gate passes with:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=3 ... regressions=1
```

This does not prove 128-chip Pathways execution. Implementation commit
`dce0e93777548b7623e4f41702144f8d00f242f5` is published. Do not render or
launch until the final operator SHA is read back and shown to contain that
commit, and a matching digest-pinned image passes the complete gate plus
sandbox admission. Then obtain separate launch approval and use fresh
Attempt-0 id `p58z04`.

At startup require exactly one of each:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
```

Then require completed trainer old/current logps, strict A=B=C, finite
nonzero 16-group backward, and one coherent update-0 transaction. If update 0
passes, continue the same full job toward 1,000 commits. Never resume or
overwrite `p58z01`, `p58z02`, or `p58z03`.

## 2026-08-26 P58.13 M2048/P59-only VMA retry override — source published

This section supersedes the P58.12 instruction to launch `p58z02`. That target
already ran: it proved the engine-global seed route and returned all 128
Step-0 rows in 1,514.2 seconds, then stopped in the first trainer canonical
per-token-logprob forward before backward or AdamW:

```text
[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS data=8 static_width=20480 chunks=80 global_M=2048 local_M=256
ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256, 4096), got (2048, 2560)
```

The batch included one `MODEL_TIMEOUT` and two
`MAX_CONTEXT_LIMIT_REACHED` rows. Those statuses were retained under the
signed compact filter and were not the fatal error. Preserve `p58z02`; it has
no optimizer commit or resumable trainer checkpoint.

P58.13 admits M=2,048 only for Qwen3-4B TP8 `(hidden=2560,tp=8)`. Qwen3-8B
keeps its existing admission; Qwen3-32B and every other geometry remain at
learner M=4,096. It also imports the target-proven FrozenLake Wave-5 serving
repair into only the strict P58 Zero-HP full profile:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1       # internal derived alias
CANON_P67_P66_VMA_P59_ONLY=1
```

P67 prevents checked-VMA metadata from leaking into ordinary serving and
keeps it inside the exact P59 manual data/model backward. It must remain
absent from Native raw, Native+IS, three-update/non-HP Zero, Qwen3-32B, and
unrelated profiles. It does not weaken alignment: Zero-HP still hard-stops on
any A-B or B-C differing byte.

Before any target, require the complete pinned-image marker:

```text
P58_EXACT_IMAGE_CPU_PASS ... qwen4b_fixed_head=1 checked_vma=1 vma_p59_only=1 first_update=1 ... regressions=1
```

Implementation commit `bea1aabde39c43c13ca4eaefab989301c6e8b46c` is
published and exact remote readback matched. Require the fetched operator tip
to contain that commit, then publish/read back the matching image and pass
sandbox-capacity admission. Obtain separate launch approval before using the
same full renderer command below with a fresh run id `p58z03`. Do not reuse
`p58z01`/`p58z02` roots or checkpoints. The fresh target must prove fixed-head
global/local M=`2048/256`, strict A-B/B-C `0/0`, finite trainer forward and
16-group backward, then exactly one coherent update-0 optimizer transaction.
If those pass, continue the same job toward 1,000 commits.

## 2026-08-26 P58.12 JAX engine-seed retry override — source published

This is the highest-priority execution instruction. `p58z01` proved 128-device
admission, clean-data loading, 128 sandbox launches, and vLLM initialization,
then failed on its first Step-0 generation because JAX rejects a per-request
`SamplingParams.seed`. Its abort cleanup separately encountered the
kubernetes-client empty-response `None.decode` defect. The run produced no
trajectory, backward, optimizer transaction, or resumable trainer checkpoint.
Preserve it and do not relaunch the published tip unchanged.

The published P58.12 repair keeps seed 42 but routes it only through global vLLM
`EngineArgs.seed`. It fails early if a JAX caller supplies a per-request seed.
Every P58 target must emit exactly one of each:

```text
[P58.SEED] PASS dataset_seed=42 rollout_seed=42 scope=engine-global async_completion_order=not-claimed
[VLLM.JAX_SEED] PASS engine_seed=42 request_seed=none scope=engine-global
```

W&B and `run_manifest.json` must use
`seed_scope=engine-global; async completion order not claimed`. This fixes the
shared RNG configuration, not asynchronous R2E completion order. Cleanup may
tolerate only the exact client-side `None.decode` exception and must still
confirm the exact run-owned Pod reaches 404 within its bounded deadline;
unrelated errors and unconfirmed deletion remain fatal.

This repair passes focused, adjacent, flag-audit, and complete pinned-image
construction gates. Implementation commit
`c10fbe0487d1f6635975b84806f1efdce6bc95c1` is published and read back. Before
target use, fetch the final operator tip and prove it contains that commit,
build and pin its matching image, rerun the full P58 exact-image gate, and pass
the existing sandbox-capacity gate. Then obtain separate launch approval and
render a fresh run id such as `p58z02` using the unchanged command below
(`--stage full --arm zero --high-performance`). Do not resume or overwrite
`p58z01`. After the two seed receipts pass, retain every P58.11 strict
alignment, checked-VMA, first-update, stable-clip, and 1,000-commit
requirement.

## 2026-08-26 P58.11 strict Zero-HP execution override

This section supersedes older statements below that defer Zero-HP. The user
has reactivated the P58 Qwen3-4B-Instruct strict Zero-HP full campaign. The
published P58.11 implementation passes focused/adjacent CPU gates, a real
16-group P58 CPU optimizer transaction, flag audit 383/383, and the complete
pinned-image construction gate. Fetch and record the exact current operator
tip; do not render or launch from an older SHA.

P58.11 keeps the signed recipe unchanged: promoted 1,012 tasks, B8 x G16,
16,384 response tokens, 50 turns, seed 42, rollout DP8 x TP8 plus trainer
DP8 x TP8, TPU-resident AdamW, strict A=B=C, and 1,000 commits. With the later
P58.13 serving-scope repair, the HP profile derives four operator-facing
production flags:

```text
CANON_P59_CHECKED_VMA=1
CANON_P67_P66_VMA_P59_ONLY=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_P63_OVERFLOW_SAFE_CLIP=1
```

`00_env.sh` derives `CANON_P66_P59_CHECK_VMA=1` internally. Partial bundles,
Native arms, non-HP Zero, warning-only alignment, wrong stage/horizon, or an
inherited `CANON_P32_WORKLOAD` fail closed. The P63 max norm is the existing
DeepSWE `1.0`: stock-finite norm behavior is unchanged; only independently
all-finite FP32 norm overflow uses max-scaled L2; NaN/Inf remains fatal.

Shape admission is exact:

| Quantity | Value |
|---|---:|
| prompts x generations | 8 x 16 |
| global / DP-local trajectories | 128 / 16 |
| outer prompt chunks | 8 |
| rank-major backward groups | 16 |
| first-update denominator | 16.0 |
| global / local canonical M | 2,048 / 256 |

Before publication, run the focused host gates and the full dependency-bearing
gate:

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Fetch the final operator tip and prove the exact
local/FETCH_HEAD/remote-tracking 40-character SHA. Build and pin the matching
runtime image; do not reuse the construction image above as source provenance.
After separate image-publication approval, the existing one-sandbox capacity
gate, and separate launch approval, render only:

```bash
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p58-zero-hp-full.yaml \
  --source-commit <exact-published-40-char-sha> \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image <matching-digest-pinned-image> \
  --run-id <fresh-attempt-0-id> \
  --stage full --arm zero --high-performance \
  --worker-nodepool <pool-or-auto>
```

At update 0 require exactly two `[V1.FIRST_UPDATE]` JSON receipts: precommit
must show workload `p58-qwen4b-tim-128`, DP8/TP8, 16 microsteps, denominator
16, all-finite/nonzero stable norm in `(0,1e6]`; commit must show one valid
optimizer transaction and `train_steps 0 -> 1` before outer sync/checkpoint.
Also require `[P59.CHECKED_VMA]` and `[P63.STABLE_CLIP]` receipts. If these pass,
continue the same job to 1,000 commits. Do not stop at one or three updates.
The P59/checked-VMA marker count follows ordered backward attempts, whereas
P63/global-step/optimizer markers follow committed updates. A legal
all-compact attempt therefore adds one zero-commit update-journal row and one
P59/checked-VMA pair but no P63/global-step row. The P58.11 postflight
reconciles these streams by journal order and excludes skipped-attempt PERF
rows from committed-step timing; it still requires exactly 1,000 commits.
Any strict alignment difference, nonfinite gradient, invalid transaction, or
missing receipt is fatal. Construction PASS remains `TARGET NOT RUN` until the
real DP8xTP8 campaign completes.

## 2026-08-24 P58.10 fixed-seed override

P58 now signs one fixed seed across Native raw, Native+IS, and Zero-HP:

```text
--seed=42
dataset shuffle seed = 42
rollout sampler seed = 42
```

The renderer requires exactly one `--seed=42`; missing, duplicate, or drifted
values are invalid. The first startup must emit exactly one
`[P58.SEED] PASS dataset_seed=42 rollout_seed=42 scope=config-level
async_completion_order=not-claimed` marker. W&B and `run_manifest.json` must
also report dataset and rollout seed 42. The scope text matters: R2E sandbox
completion and async collection order remain nondeterministic, so this is not
a bitwise end-to-end replay guarantee.

P58.10 is published as implementation commit
`9597de3d99fbf65c87f4fea3d86e639cca0b7abe`, replayed over fetched operator
tip `ff646a4d76f58e9f328bc640f44d362637eb1432`. Immediate local/FETCH_HEAD/
remote-tracking readback matched with ahead/behind `0/0`. Before rendering,
fetch the final operator tip containing that implementation commit and pin its
exact 40-character SHA. Source publication does not authorize a target launch.

## 2026-08-24 local P58.9 execution override

The published P58.9 refinement was built at
`/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824` on local branch
`local/p58-is-zero-refine-0824`, originally based on operator tip
`614156c1ab067192ab65b2969543e23904f192be`. The implementation was replayed
over `7b85b42d0a019d70f32a7dc9712c538ad42f5cb5`, published as
`2aedd73c957abba29d21d05b866a996af2f66dfd`, and passed exact first remote
readback. Before use, fetch the final operator tip containing that commit and
replace `<published-40-char-sha>` below with the exact fetched/read-back SHA;
never use the mutable branch name as provenance.

The renderer admits exactly three recipe shapes:

```bash
# Untreated Native numerical program; rollout logps are the old policy.
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p58-native-raw.yaml \
  --source-commit <published-40-char-sha> \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image <digest-pinned-image> --run-id <fresh-id> \
  --stage full --arm native --worker-nodepool <pool-or-auto>

# Identical Native program plus registered token TIS at threshold 2.0.
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p58-native-is.yaml \
  --source-commit <published-40-char-sha> \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image <digest-pinned-image> --run-id <fresh-id> \
  --stage full --arm native --sampler-is \
  --worker-nodepool <pool-or-auto>

# Optimized strict Zero. Sampler IS remains forbidden here.
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p58-zero-hp.yaml \
  --source-commit <published-40-char-sha> \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image <digest-pinned-image> --run-id <fresh-id> \
  --stage full --arm zero --high-performance \
  --worker-nodepool <pool-or-auto>
```

Native raw resolves `CANON_P34_DISABLE_SAMPLER_IS/TIS=1/1`. Native IS
resolves exactly `0/0`, passes `--sampler_is=token` and
`--sampler_is_threshold=2.0`, and must emit exactly one
`[P58.TIM_RECIPE] ... recipe=native-is ... old_logps=trainer
tis_weights=present` marker. Native raw emits exactly one corresponding
`native-raw ... old_logps=rollout tis_weights=absent` marker. Zero and Zero-HP
remain `1/1` and reject `--sampler-is`. No recipe enables group filtering or
host optimizer offload.

Every rendered P58 JobSet must contain the exact Attempt-0 policy
`maxRestarts: 0, restartStrategy: Recreate`. Do not increase retries until the
run root, report paths, W&B identity, and postflight are all attempt-scoped and
tested. The renderer intentionally omits unconsumed
`PATHWAYS_HEARTBEAT_TIMEOUT_SEC`, `IFRT_PROXY_TIMEOUT_SECONDS`, and GRPC
keepalive environment names.

Before any publication or launch run:

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

This gate is construction evidence only. Native-IS and Zero-HP target jobs are
separate experiments. Native-IS is now the selected replacement experiment.
P58.9 and P58.10 are published; target launch remains separately gated and
Zero-HP is still deferred.

Current execution decision: stop and archive the exact running Native/no-IS
campaign after the operator observed a sharp training-reward drop and judged
the run collapsed. The onset update is not established; do not assign the
event to a fixed optimizer step. Do not resume its optimizer checkpoint
and do not relaunch Native raw. Preserve the rendered YAML, source/image
provenance, logs, full W&B export, trajectories, update receipts, checkpoint
inventory, and metrics spanning the last stable reward region, reward-drop
onset, and subsequent completed batches before deleting only that exact JobSet
and its proven run-owned sandboxes. Then launch a fresh Native+IS 1,000-update
campaign from the original frozen base, with a new run id, run root, W&B run,
and checkpoint directory. Source publication and first readback are complete.
The replacement launch still requires final tip readback, digest-pinned image,
the render/admission checks below, and the exact Native-raw archival/cleanup
boundary in the handoff.

P58.3 was explicitly waived, not passed, and the user superseded the separate
three-update stop. The later p58f05 repair has a separate bounded one-host
admission gate; it does not retroactively promote P58.3. The first local
attempt at that gate was blocked because the container exposed no `/dev/vfio`,
so it is not a TPU PASS. Updates 1–3 are monitored inside the same full job and
do not terminate a healthy run. Zero is deferred while its optimization work
continues and must not be rendered or applied. A native result cannot be
reported as a paired comparison.

Attempt history: native `p58c01` failed in `00_env.sh` before any TPU program;
that admission fix was published as
`acd3136267214b367a6755d0ba28d80e883d6753`. Native `p58c02` initialized
Pathways but failed before model import because direct execution of the wrapper
did not make the repository root importable. Native `p58c03` passed those
boundaries, then stopped before model initialization because the parent
entrypoint retained the renderer's stale `CANON_LOGPROB_M=256` after the
native profile had unset it in child-shell `00_env.sh`. All three roots are
immutable and have no resumable trajectory state. Use a final operator-branch
readback SHA containing the authoritative environment-snapshot fix. Native
`p58c04` passed all bootstrap gates and initialized Pathways, Qwen3-4B, vLLM,
W&B, and the rollout loop, but all 128 concurrently requested RepoEnv pods
remained unconfirmed Running until their 1,200-second start deadline. Pinned
R2E swallowed those timeouts, then attempted setup against deleted pods; the
real Kubernetes 404 was obscured by the client's `None.decode` error. P58c04
is also immutable and has no resumable trajectory state. Use a final
operator-branch readback SHA containing the fail-closed start repair. Native
`p58c05` then failed even earlier: its Workload remained
`QuotaReserved=False` because the rendered worker treated the Kueue sentinel
`tpu-v5p-slice` as literal node-pool affinity, so flavor `0xv5p-8` could not
match. No workload pod or training process started. The renderer repair
delegates concrete node-pool selection to Kueue for registered sentinels while
retaining exact `4x4x8` topology. Use fresh full-stage run-id `p58f01`. Never
reuse a p58c01 through p58c05 YAML/root.

Native `p58f01` then passed JobSet admission and the complete 128-device
Pathways/Qwen3-4B/vLLM initialization chain, but every runtime-created R2E Pod
remained `SchedulingGated`. Those standalone Pods lacked the parent JobSet's
Kueue queue label, so all 128 resets timed out. The resulting all-timeout
batch exposed a second bug: `policy_version` was assigned only on the first
model call, which reset-time failures never reached, and strict processing
crashed before journaling. The repaired path derives the sandbox queue from
the parent JobSet, writes it to every Pod, seeds policy provenance before
reset, and records `scheduling_gated` separately. P58f01 is immutable,
`INCONCLUSIVE`, and has no resumable state.

Native `p58f02` passed initialization and started Step 0, but sandboxes remained
`SchedulingGated` because `multislice-queue` CPU flavor `cpu-user` requires
`nodeSelector: cpu-np`, whereas sandboxes defaulted to `deepswe-cpu-pool`.
The fix routes sandboxes and head pod to `cpu-np` (`NODE_SELECTOR_VAL=cpu-np`).
P58f02 is immutable, `INCONCLUSIVE`, and has no resumable state. Use fresh
full-stage run-id `p58f03`; never reuse its YAML/root.

Native `p58f03` proved that CPU routing repair. It completed the first real
rollout batch in 616.3 seconds and wrote 128 durable rows: 126 `SUCCEEDED`, two
`MAX_CONTEXT_LIMIT_REACHED`, three solved trajectories, two mixed/effective
prompt groups, and 32 nonzero advantages. Sandbox-start timeout count was
zero. The run then stopped before trainer forward, backward, or update because
the shared P34 weight gate called a canonical-adapter-only method even though
native correctly had `CANON_ENGINE_MODULE_C=0`. P58f03 is immutable and
`INCONCLUSIVE`; it has a useful trajectory journal but no optimizer checkpoint
and is not a resumable training root.

The repaired gate is arm-aware and remains fail-closed. Zero delegates exact
live-weight comparison to its registered canonical adapter. Signed P58 native
uses the same pure trainer-to-engine leaf mapping and bitwise comparison as a
read-only observer; it does not construct/register that adapter or replace a
serving function. Missing or unequal leaves, DP8 x TP8 mesh drift, an unsigned
route, or a leaked canonical adapter in native is fatal. The receipt exposes
contract axis names `dp/tp` even though vLLM internally names them `data/model`.

Native `p58f04` proved that weight repair: after a 557.2-second rollout it
wrote 128 rows (125 `SUCCEEDED`, three `MAX_CONTEXT_LIMIT_REACHED`, six solved,
one mixed/effective group, 16 nonzero advantages) and emitted
`[P34.WEIGHTS] EXACT` for 398 leaves and 4,022,468,096 elements. It then failed
before trainer forward/backward/update because the shared processed
`S_prefill` call accepted only `CANON_PROMPT_PROCESSED_LOGPROBS=1`. Native
correctly keeps that canonical flag at zero, so the fail-closed rejection was
correct but the arm routing was incomplete. P58f04 is immutable,
`INCONCLUSIVE`, and has no resumable optimizer state.

The repair uses two mutually exclusive processed-B implementations. Native
keeps the complete zero-TIM bundle disabled/absent and alone sets
`CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1`. After all six stock files are
verified, a two-file, digest-pinned observer overlay computes only the
post-rollout B values with decode-equivalent temperature/top-k/top-p transforms
and absolute request-history targets. It never participates in generation,
trainer forward, loss, backward, optimizer math, or commit accounting. Zero
sets the P58 observer to zero and retains
`CANON_PROMPT_PROCESSED_LOGPROBS=1`, `CANON_ENGINE_MODULE_C=1`, and the complete
canonical engine. Environment and runtime contracts reject any mixed tuple.
Native `p58f05` proved that observer repair. Its 486.4-second rollout wrote 128
durable rows: 126 `SUCCEEDED`, two `MAX_CONTEXT_LIMIT_REACHED`, six solved,
two mixed/effective groups, and 32 nonzero advantages. All timeout dimensions
were zero. Exact live weights passed over 398 leaves and the processed-B
observer covered all 2,048 prompt rows. The alignment sidecar then attached,
but the run stopped before trainer forward/backward/update because the warning
policy admitted P58 only in an obsolete short-update branch and rejected the
signed `stage=full, expected_updates=1000` tuple.

The p58f05 repair admits P58 Native only with `CANON_P58_TIM_ADMITTED=1`, no competing
P39/P43/P44 mode, and an exact `three-update/3` or `full/1000` pair. It does
not enable a zero-TIM numerical flag. P58f06 proved that admission and reached
alignment after 128 durable trajectories, exact weights, and a complete Native
processed-B observation. Both A-B and B-C were shape-valid and finite across
405,827 action tokens, but the P58-specific warning tuple still blocked B-C
before trainer forward. The correction makes both finite serving-path
boundaries warnings while keeping nonfinite/shape, weights, replica,
transaction, and optimizer errors hard. Zero remains strict.
P58f07 proved the warning repair: all 128 real RepoEnv trajectories completed,
pre-backward passed with finite A-B/B-C warnings, and the trainer entered real
value-and-grad/backward. It then stopped on `T_old_vs_T_current` because the
Native gate incorrectly required an observer-only stock rescore to match the
value-and-grad primal exactly. The correction preserves the stock
quality-fix 128-trajectory observer and makes every shape-valid finite Native
program boundary and derived ratio observational. With rollout logprobs on and
sampler-IS off, the loss uses rollout A rather than observer `T_old`. Zero
remains exact, and no training data, accumulation, loss, optimizer, or
numerical treatment flag changes.
P58f08 never reached rollout. Its CPU head inherited `hostNetwork:true` and
shared node port 29001 with another Pathways ResourceManager. Six concurrent
heads already occupied the six available `cpu-np` nodes, so the scheduler
packed this seventh head onto an occupied host. The P58 CL/956357083 worker
resolved that foreign CL/42 server and failed strict compatibility. Moving the
head to `deepswe-cpu-pool` was tested and rejected because the worker could not
maintain its scheduler pipe across the node-pool subnet boundary. The correct
repair keeps `cpu-np` and the proven host-network transport, but adds required
hostname anti-affinity between every JobSet `pathways-head` Pod so Kubernetes
or the autoscaler must supply an unused CPU node.

P58f09 proved correct Pathways attachment and completed all 128 Step-0 rollout
slots in 1,699.1 seconds. Some environment resets ended at the admitted
3,000-second trajectory deadline before first observation. Those rows had
`agent.trajectory.task=None`, although the original input remained in
`env.task`; learner `merge_micro_batches()` then crashed on the `None` before
the P58 journal, alignment, forward, backward, update, or checkpoint. The
collector repair preserves the agent task when present, otherwise uses the
environment task, and fails closed if neither is a dictionary. Timeout/context
rows retain their compact status and zero policy mask; they are not removed or
resampled. No training or numerical parameter changed.

Native `p58f10` ran the source containing that repair and entered Step-0
rollout, then exposed a separate scheduling-geometry error. The hard batch
timeout prevented post-rollout merge, so the original-input fallback remains
target-unproven despite its exact-image coverage. B8 x G16 produces 128 trajectories, but
`max_concurrency=64` split them into two sequential waves. The unchanged
3,600-second rollout-batch deadline expired with only 5/8 prompt groups
complete. No journal, trainer program, optimizer receipt, or checkpoint was
created. The repair sets concurrency to 128, matching both the raw batch and
the provisioned rollout capacity DP8 x max-seqs16. Episode 3,000 s, cleanup
300 s, and batch 3,600 s remain unchanged. Use fresh full-stage run-id
`p58f11` after fetching and exactly reading back the final operator tip; never reuse p58f08,
p58f09, or p58f10 YAML/root. None has optimizer state to resume.

Native `p58f11` proves that repair: all 128 trajectories and all 8 prompt
groups completed in one wave in 1,209.2 seconds. One generation terminated in
`env.reset` and exercised the compact fallback. It then exposed a task-schema
bug before journaling: `SWEEnv.entry` had the normalized prompt, but inherited
`SWEEnv.task` had only `policy_version`, so learner processing raised
`KeyError: 'prompts'`. The repaired source seeds `SWEEnv.task` with a
singleton-batched normalized prompt before reset and uses that policy-seeded
task for both normal and pre-observation termination rows. Missing `prompts`
now fails at collection. At that historical checkpoint the next run was
`p58f12`; never reuse p58f11 YAML/root.

Native `p58f12` proved the normalized-prompt repair and wrote a valid durable
128-row Step-0 journal. All 128 R2E Pods nevertheless remained Kueue
`scheduling_gated` until sandbox-start timeout, so every trajectory was signed
compact-filtered `ENV_TIMEOUT`, completion/action token counts were zero, and
`generate()` was never called. Processed-B rescore then failed because no
rollout sampling-transform provenance existed. No alignment, backward,
optimizer commit, or checkpoint followed. The journal is diagnostic evidence,
not resumable trainer state. This batch diagnoses zero CPU sandbox admission
throughput; do not respond by increasing vLLM max-seqs or model concurrency.

The local repair first completes the signed empty-completion observer path
without changing the training algorithm. Zero completion targets validate
structure and the observer signature, skip the engine, and record
`engine_called=false`; any nonempty target still requires real
post-`generate()` sampling provenance. Alignment admits zero policy actions
only with durable P58 all-compact provenance. For ordinary
model/context/runtime all-compact outcomes, the existing zero-gradient
transaction makes no optimizer commit, the outer learner suppresses weight
sync and every committed-step advance, only `batch_index` advances, and the
next prompt batch is consumed without resampling.

A durable full sandbox-start outage uses a separate circuit breaker: emit
`[P58.SANDBOX_CAPACITY] BLOCKED` and raise `BLOCKED_SANDBOX_CAPACITY` before
rescore/alignment/trainer or any later prompt consumption. For the selected
replacement, use a fresh Native+IS run id such as `p58is01` only after exact
final remote readback and the `cpu-np`/Kueue capacity gate below.

The direct-entrypoint implementation commit is
`82d82f72a7220d945737d95f6266b5b7e2cfe706`. Resolve the final runnable SHA by
fetching the operator branch after the later publication checkpoint; do not
launch from the historical p58c02 source.

The authoritative resolved-environment snapshot implementation commit is
`c0ca41805bd65a4fdede4825ed2835cdce6e13ed`. Its first post-push readback
matched exactly with ahead/behind `0/0`; still fetch the final branch tip after
the publication-evidence checkpoint rather than pinning this historical
implementation commit directly.

## 1. Frozen recipe

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Clean data | 1,012 promoted P46 tasks |
| Clean SHA-256 | `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` |
| Prompt batch / generations | B8 x G16 = 128 raw trajectories |
| Sandbox concurrency | 128; exactly one wave matching both the raw batch and rollout DP8 x max-seqs16 capacity |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0, top-p 1.0, top-k 0 |
| Roles | rollout DP8 x TP8 + trainer DP8 x TP8 |
| Objective | RLOO; `sequence-mean-token-scale`; fixed norm 16,384 |
| Trainer observer | Stock quality-fix `T_old`: one prompt-counted 128-trajectory rescore for B8 x G16; observer-only under `use_rollout_logps=true` |
| PPO | epsilon 0.20, epsilon-high 0.28, beta 0 |
| Optimizer | Adam 1e-6, betas 0.9/0.99, weight decay 0.01, grad clip 1.0 |
| Optimizer placement | TPU device-resident; host offload forbidden |
| Update geometry | prompt mini-batch 8; 128 trajectory mini-batch; trajectory micro-batch 16; accumulation depth 8 |
| Optional interventions | sampler-IS off; group clip/filter off; degenerate masking off; flat-group resampling off |
| Prefix cache | off |
| Active horizon | full campaign, exactly 1,000 commits; commits 1–3 are monitoring milestones |

Compact filtering is part of the shared recipe. These terminal statuses are
journaled but get an all-zero policy mask:

```text
MAX_STEPS_REACHED
MAX_CONTEXT_LIMIT_REACHED
TIMEOUT
ENV_TIMEOUT
MODEL_TIMEOUT
REWARD_TIMEOUT
```

Partial filtering uses
`sum(mask * token_loss) / (B_eff * 16384)`. If all 128 rows are filtered by
ordinary model/context/runtime outcomes, the transaction is discarded without
an optimizer commit or weight sync and the next data batch is consumed. It is
not resampled. Trainer/RL global steps and `policy_version` remain unchanged;
`batch_index` advances while `optimizer_step` remains the actual committed
trainer step. A full sandbox-start outage instead journals and stops before a
later prompt. This separation prevents artifact collisions without allowing
an infrastructure outage to scan the clean list.

Timeout nesting is fixed: turn 300 s, step/reward 600 s, trajectory 3,000 s,
sandbox 3,300 s, cleanup 300 s, and the shared rollout-batch deadline 3,600 s.
The renderer requires B x G = `max_concurrency` = rollout DP x
`rollout_vllm_max_num_seqs` = 128. A per-trajectory timeout returns through the
normal compact zero-mask path and does not abort sibling trajectories. The
rollout-batch deadline remains a hard orchestration bound: if the complete
one-wave batch cannot drain by 3,600 seconds, fail closed rather than fabricating
missing rows or extending the signed one-hour batch.

Kubernetes sandbox start is fail-closed. A pod that does not become Running
within 1,200 seconds is deleted and confirmed absent, and its original
`TimeoutError` must reach the trajectory collector as signed `ENV_TIMEOUT`.
R2E must never return a RepoEnv with `container=None` or continue with a
websocket exec into a deleted pod. If a target run again reports zero Running
pods across a batch, preserve pod scheduling/events evidence and treat it as
CPU-pool capacity/admission work; do not patch the websocket decoder.
The bounded marker `[P34.R2E] KUBERNETES_START_TIMEOUT` records only pod name,
phase, and scheduler condition/reason/message; it never serializes the pod
spec or environment.

## 2. Fetch, pin, and validate the published source

Only launch a clean, freshly read-back publication on
`yuxzhang/canon-zero-tim`. The publication SHA is deliberately resolved at
execution time so this versioned document never contains a stale or
self-referential commit ID.

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test "$(printf '%s' "$SOURCE_SHA" | wc -c)" -eq 40
test -z "$(git status --porcelain)"
```

Run the pinned-image gate with the exact launch image. A registry digest is
required by the renderer; a local Docker image ID is only suitable for the
local gate.

```bash
bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Required terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1
```

That marker proves CPU/image wiring, not TPU execution, HBM, real R2E rollout,
native mismatch dose, or zero exactness.

For the p58f05 admission repair, run the bounded direct-attached v5p gate when
an actual four-device host is available:

```bash
DEEPSWE_TRAIN_PYTHON=/mnt/disks/tunix-data/venvs/train/bin/python \
  bash canon-zero-tim/tests/p58_deepswe_native_zero/run_onehost_alignment_v5p.sh
```

Required terminal marker:

```text
P58_ONEHOST_ALIGNMENT_ADMISSION_PASS ... devices=4 scope=renderer-profile-policy
```

The gate requires four real v5p devices, executes a TPU matmul, then checks the
P58 alignment policy and the renderer-derived full-stage environment. It does
not run Qwen/R2E rollout, trainer forward/backward, DP8 x TP8, Pathways, or an
optimizer update. A missing TPU metadata path or device inventory is
`BLOCKED_DIRECT_TPU_METADATA`, not PASS; do not inject fake topology metadata.
This repair-only check does not retroactively promote the waived P58.3 phase.

Before rendering, verify read-only that the mounted PVC contains
`Qwen3-4B-Instruct-2507`, the R2E dependency imports, the clean JSONL has 1,012
lines, and its digest matches the frozen value. Never print secret values.

## 2A. P58 sandbox capacity gate

P58f12 had 128/128 sandboxes `scheduling_gated`. Before reserving TPUs, prove
one production-shaped sandbox can pass the exact Kueue/CPU route. This probe
checks admission only; it does not execute R2E, model rollout, or training.

First derive a real task image from the frozen clean list and render a unique
Pod. Do not substitute a generic image, hand-edit the YAML, or overwrite a
previous probe artifact.

```bash
CLEAN_JSONL='canon-zero-tim/clean_data/p46_q4_learnable/p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl'
TASK_IMAGE="$(head -n 1 "$CLEAN_JSONL" | jq -er '.docker_image')"
PROBE_RUN_ID='p58is01'
PROBE_POD="canon-p58-sandbox-probe-${PROBE_RUN_ID}"
PROBE_YAML="/tmp/${PROBE_POD}.yaml"

python3 canon-zero-tim/cluster/render_p58_sandbox_probe.py \
  --run-id "$PROBE_RUN_ID" \
  --task-image "$TASK_IMAGE" \
  --output "$PROBE_YAML"
sha256sum "$PROBE_YAML"
kubectl apply --server-side --dry-run=server -f "$PROBE_YAML"
```

Required render marker:

```text
P58_SANDBOX_PROBE_RENDER_PASS pod=canon-p58-sandbox-probe-p58is01 ...
```

Applying the probe is a separate user/operator-approved Kubernetes mutation.
Only after that approval:

```bash
kubectl apply -f "$PROBE_YAML"
kubectl -n default wait --for=condition=Ready "pod/$PROBE_POD" \
  --timeout=10m || true
P58_SANDBOX_PROBE_POD="$PROBE_POD" \
  bash canon-zero-tim/cluster/steps/p58_verify_sandbox_capacity.sh
```

The only pass marker is:

```text
P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only ...
```

The verifier requires an Active `multislice-queue` LocalQueue, an Active
backing ClusterQueue, at least one Ready schedulable `cpu-np` node, a Running
probe labeled `kueue.x-k8s.io/managed=true` with no scheduling gate, the exact
queue/node selector, and a selected node that actually belongs to `cpu-np`.
Preserve read-only evidence before cleanup:

```bash
kubectl -n default get pod "$PROBE_POD" -o yaml
kubectl -n default describe pod "$PROBE_POD"
kubectl -n default get workloads.kueue.x-k8s.io -o wide
kubectl -n default get localqueue.kueue.x-k8s.io multislice-queue -o yaml
CLUSTER_QUEUE="$(kubectl -n default get localqueue.kueue.x-k8s.io \
  multislice-queue -o jsonpath='{.spec.clusterQueue}')"
kubectl get clusterqueue.kueue.x-k8s.io "$CLUSTER_QUEUE" -o yaml
kubectl get resourceflavors.kueue.x-k8s.io -o yaml
kubectl get nodes -l cloud.google.com/gke-nodepool=cpu-np \
  -o custom-columns=NAME:.metadata.name,UNSCHEDULABLE:.spec.unschedulable,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory
```

Identify and preserve the Workload associated with the probe from the Pod UID,
owner/labels, and timestamps; do not guess a Workload name. A one-Pod PASS is
necessary but is not proof that the full 128-Pod wave can be admitted. The
production request floor is 128 x 2 CPU = 256 requested CPU and 128 x 4 GiB =
512 GiB requested memory, plus the Pathways head and cluster overhead. Confirm
the ClusterQueue quota/usage, selected ResourceFlavor, ready-node capacity,
and autoscaler behavior can supply that request, or obtain explicit operator
signoff before applying the full JobSet. Container limits are 4 CPU/8 GiB each
and must also remain feasible on selected nodes.

If the probe remains Pending or gated, the verifier emits
`P58_SANDBOX_CAPACITY_BLOCKED`. Preserve Pod conditions/events, the matching
Workload's `QuotaReserved`/admission checks, LocalQueue/ClusterQueue status,
selected ResourceFlavor, and `cpu-np` readiness. Do not launch TPUs, tune vLLM,
increase `max_num_seqs`, or remove the queue label. The returned p58f12 log is
insufficient to distinguish quota exhaustion from a flavor/admission issue.

When evidence is preserved, delete only the exact probe and confirm recovery:

```bash
kubectl -n default delete pod "$PROBE_POD"
kubectl -n default wait --for=delete "pod/$PROBE_POD" --timeout=5m
```

This deletion is also a Kubernetes mutation and requires operator authority.
Report the exact deleted Pod name and whether deletion was confirmed.

## 3N. Render and launch the fresh Native+IS full campaign

Use the exact source SHA, image digest, CPU pool, Kueue worker sentinel, PVC,
and a unique run id. Never hand-edit rendered YAML. This phase permits only
the registered Native+IS recipe. It is invalid until the old Native/no-IS
JobSet has been stopped and archived and the exact published operator SHA
containing P58.9 has been fetched and read back. Load the original frozen base
model; never resume the collapsed Native/no-IS checkpoint.

```bash
CLIENT_IMAGE_DIGEST='registry.example/tunix@sha256:<64-hex-digest>'
CPU_NODEPOOL='cpu-np'
TPU_NODEPOOL='tpu-v5p-slice'
MODEL_PVC='haoyugao-cpu-np-pvc'
RUN_STEM='p58is01'
STAGE='full'

ARM='native'
RECIPE='native-is'
OUTPUT="/tmp/p58-${RECIPE}-${STAGE}-${RUN_STEM}.yaml"
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_STEM" \
  --stage "$STAGE" \
  --arm "$ARM" \
  --sampler-is \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
sha256sum "$OUTPUT"
kubectl apply --server-side --dry-run=server -f "$OUTPUT"
```

The renderer must emit
`P58_DEEPSWE_TIM_RENDER_PASS arm=native stage=full recipe=native-is`.
The JobSet name must contain `native-is`, its label must contain
`canon.zero-tim/sampler-recipe=token-is`, and the resolved environment must
contain the exact sampler/TIS disable tuple `0:0`. A `native-raw` marker,
tuple `1:1`, or a partial tuple is a hard stop.

`tpu-v5p-slice` is a Kueue-managed sentinel, not a concrete node-pool name.
Before server-side dry-run, inspect the rendered worker and require all of the
following:

```text
google.com/tpu: 128
cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
cloud.google.com/gke-tpu-topology: 4x4x8
no cloud.google.com/gke-nodepool: tpu-v5p-slice
JobSet label kueue.x-k8s.io/queue-name: multislice-queue
jax-tpu env R2E_K8S_QUEUE_NAME: multislice-queue
head nodeSelector cloud.google.com/gke-nodepool: cpu-np
head hostNetwork: true
head dnsPolicy: ClusterFirstWithHostNet
head required podAntiAffinity selector: jobset.sigs.k8s.io/replicatedjob-name In [pathways-head]
head required podAntiAffinity topologyKey: kubernetes.io/hostname
JobSet spec.network.enableDNSHostnames: true
JobSet spec.network.publishNotReadyAddresses: true
worker hostNetwork: true
worker dnsPolicy: ClusterFirstWithHostNet
worker --resource_manager_address: <jobset>-pathways-head-0-0.<jobset>:29001
worker PATHWAYS_HEAD: <jobset>-pathways-head-0-0.<jobset>
```

Kueue's selected ResourceFlavor supplies the concrete pool affinity. If the
literal sentinel appears as node-pool affinity, stop before apply; that is the
p58c05 admission bug.

If the head loses host networking, uses `deepswe-cpu-pool`, or lacks the exact
required hostname anti-affinity, stop before apply. The anti-affinity is the
fixed-port isolation; changing the proven transport is not. After apply,
confirm each Pathways head is on a distinct CPU hostname. If a strict-CL
mismatch recurs despite these invariants, preserve logs from
`pathways-proxy`, `pathways-rm`, `jax-tpu`, and one `pathways-worker` plus the
worker's resolved RM address before deletion. Do not wait for rollout: this is
an immediate bootstrap failure.

After apply, inspect the first sandbox before waiting for a whole batch. Its
metadata must contain `kueue.x-k8s.io/queue-name=multislice-queue`; its Kueue
Workload must become admitted, and the `kueue.x-k8s.io/admission` scheduling
gate must disappear before the Pod can be called healthy. If it remains gated,
preserve the Pod conditions and matching Workload/LocalQueue status and stop;
do not tune model concurrency or wait another 1,200 seconds first.

Before apply, preserve the resolved-environment regression result. It must
prove that a parent process seeded with the renderer's
`CANON_LOGPROB_M=256` loses that variable after sourcing the native
`env.sh`, while the zero arm still resolves it to `256`. Do not work around a
failure by relaxing `deepswe_contract.validate_environment`; absence is part
of the native treatment definition.

The explicit launch boundary, only after operator approval, is:

```bash
kubectl apply -f /tmp/p58-native-is-full-${RUN_STEM}.yaml
```

Do not produce or apply a Native raw or Zero YAML in this phase. Preserve the
exact Native+IS YAML and digest with the returned run. Before admitting the
first optimizer update, require exactly one:

```text
[P58.TIM_RECIPE] PASS recipe=native-is sampler_is=token old_logps=trainer tis_weights=present threshold=2.0 group_filter=none
```

## 4. Evidence and full-campaign interpretation

Each run root is:

```text
/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>/
```

Important artifacts:

```text
run.log
weight_attestation.jsonl
pre_alignment.jsonl
alignment.jsonl
updates.jsonl
p58_deepswe_<arm>_<stage>.classification.json
debug/run_manifest.json
debug/batch_metrics.jsonl
debug/batch-<batch_index>.trajectories.jsonl.gz
```

Before interpreting A/B/C, require one exact weight receipt for the active
arm. Zero obtains it through the registered canonical adapter. Native obtains
the same bitwise trainer-to-live-engine proof through an observer-only route;
the adapter must remain absent. The log must contain `[P34.WEIGHTS] EXACT`.
Missing/mismatched leaves, invalid mesh, or native adapter leakage is a hard
failure and must not be converted to warning-only.

For native, also require exactly one engine marker before B is accepted:

```text
[P58.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS ... targets=absolute-request-history treatment=observer-only
```

The preceding install log must say `canonical_bundle=off`, while the native
environment retains `CANON_PROMPT_PROCESSED_LOGPROBS=0` and
`CANON_ENGINE_MODULE_C=0`. Seeing a canonical engine marker or either flag at
one is treatment contamination and a hard stop. Conversely, the zero arm must
set `CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=0`; it must never install or emit
the stock-observer marker.

The gzip files contain the complete redacted conversation/tool trajectory,
raw final reward, training reward, advantage, status, task identity,
`batch_index`, and `optimizer_step`. Inspect without editing:

W&B receives both counts and ratios for solved trajectories, all-solved,
all-failed, mixed, incomplete, and effective prompt groups, plus compact-filter
counts/ratios. `effective` and `nonzero_advantage` mean usable policy signal:
compact-filtered rows retain their raw advantage in the journal but do not
inflate these metrics. A separate `raw_nonzero_advantage_ratio` is retained for
audit.

Timeout telemetry is deliberately low-cardinality. W&B receives counts and
ratios for all timeout statuses, `ENV_TIMEOUT`, sandbox-start timeouts,
`scheduling_gated` and unschedulable sandboxes, and insufficient CPU/memory,
plus the batch booleans
`deepswe/all_env_timeout_batch` and
`deepswe/all_sandbox_start_timeout_batch`. Full scheduler messages remain only
in the raw `[P34.R2E] KUBERNETES_START_TIMEOUT` log marker; they are never used
as W&B keys or values. Interpret the first completed batch as follows:

| Observation | First boundary | Action |
|---|---|---|
| `all_sandbox_start_timeout_batch=1` | no R2E pod became Running; model rollout was not the bottleneck | preserve pod events and inspect CPU-nodepool scheduling/capacity |
| sandbox-start ratio is nonzero but below one | partial sandbox admission/throughput | inspect scheduler reasons before tuning model concurrency |
| sandbox-start ratio is zero and `status/model_timeout_ratio` is nonzero | sandbox ran; model generation exceeded its deadline | investigate serving throughput/model limits |
| `timeout_stage_histogram.environment_step` is nonzero | sandbox started; repository command execution timed out | inspect R2E task/runtime behavior |

The W&B batch metrics are emitted only after a 128-row trajectory batch has
been journaled. If the process dies before that boundary, use the bounded raw
timeout markers and Kubernetes events; absence of W&B metrics is not evidence
that the sandboxes ran.

```bash
RUN_ROOT='/mnt/disks/linchai_data/deepswe_zero_tim/<jobset-name>'
jq . "$RUN_ROOT/debug/run_manifest.json"
jq -c '{step,optimizer_step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,compact_filtered_trajectories,status_histogram,timeout_stage_histogram,timeout_scheduler_reason_histogram,timeout_resource_histogram,all_env_timeout_batch,all_sandbox_start_timeout_batch}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" \
  | head -n 1 | jq .
jq . "$RUN_ROOT/p58_deepswe_<arm>_<stage>.classification.json"
```

Full-stage PASS requires exactly 1,000 committed update records. There may be
more than 1,000 trajectory batches if an ordinary model/context/runtime batch
was entirely compact-filtered; every such extra batch must have a zero-commit
receipt, no weight sync, unchanged trainer/RL/policy state, an incremented
batch index, and the same optimizer step as its successor. Require these
bounded markers:

```text
[CANON_RESCORE] empty_completion_batch targets=0 ... engine_called=0
CANON_ALIGN... N_action=0 ... all_compact_filtered=true ... no_signal_admitted=true
[DEEPSWE.COMPACT_FILTER] optimizer_boundary_skipped effective_rows=0 train_steps=<N>
[P58.NATIVE] optimizer_transaction ... commits=0 mode=compact...
[P58.COMPACT_FILTER] all_filtered=1 optimizer_commits=0 train_steps=<N> global_steps=<N> ... weight_sync=0
```

The subsequent journal may use a larger `batch_index` but must still report
`optimizer_step=<N>` until a real commit occurs. Any partial journal, missing
digest, duplicate/missing trajectory, wrong task identity, unsigned filtered
status, or state advance on the no-commit path is fatal.

A full sandbox-start outage uses a different terminal contract. After the
durable journal and bounded timeout metrics, require:

```text
[P58.SANDBOX_CAPACITY] BLOCKED ... optimizer_commits=0 prompts_consumed_after_batch=0 trajectory_path=<...> trajectory_sha256=<...>
BLOCKED_SANDBOX_CAPACITY: all P58 trajectories timed out before sandbox start
```

No processed rescore, alignment, trainer call, weight sync, optimizer commit,
or later prompt batch may follow. Preserve the run as `INCONCLUSIVE`, repair
capacity, and use a fresh run id. Do not let a persistent cluster outage scan
past the frozen clean-task ordering.

Monitor without stopping the healthy job:

| Milestone | Required evidence |
|---|---|
| Kueue admission | `QuotaReserved=True`, selected TPU flavor, 32 four-chip worker pods, 128 Pathways devices |
| first completed batch | 128 journal rows; timeout split; cleanup; solve, all-zero/all-one/mixed/effective-group metrics |
| commits 1–3 | finite forward/backward; finite nonzero A-B or B-C dose; all Native program boundaries and ratios finite; TPU optimizer; monotonic transaction/journal state |
| commit 8 | first expected checkpoint artifact and digest |
| commits 32, 100, then each 100 | continued finite training, checkpoint/evaluation cadence, no journal or cleanup drift |

Crossing commit 3 is not a stop condition. The classifier cannot say full
`PASS` until commit 1,000 and complete postflight evidence exist.

The native classifier requires at least one finite, nonzero mismatch across
`S_decode_vs_S_prefill` or `S_prefill_vs_T_old`. Exactness on both Native
serving boundaries is `NO_TREATMENT`, not a successful comparison. Native
`T_old_vs_T_current` may differ only when the boundary and derived ratios are
shape-valid and finite. The zero classifier requires all boundaries exact.
Both require device-resident optimizer evidence and no blocking reds.

Useful scan:

```bash
grep -aE '\[P58\.|CANON_ALIGN_PRE_JSON|CANON_ALIGN\]|COMPACT_FILTER|update_step_committed|optimizer_transaction|ONLINE_RUN_PASS|Traceback|OOM|RESOURCE_EXHAUSTED|CANCELLED|IFRT' \
  "$RUN_ROOT/run.log"
```

The run may not be promoted merely because Python exits zero. Require the
classification JSON verdict `PASS` and preserve its digest with the rendered
manifest and raw log.

## 5. Completion and follow-up

At update 1,000, preserve the full native classifier, raw log, run manifest,
all journal/checkpoint/evaluation digests, rendered YAML, source SHA, and image
digest before declaring the campaign complete. A later zero canary or paired
campaign still requires a separate user decision and must restore the paired
invariants. P58 does not claim Qwen3-32B or 256-chip production readiness.

## 6. Stop and escalation rules

Stop rather than retrying the same manifest if any of these occurs:

- source/image/data digest drift;
- native/zero processed-B treatment mixing, a missing or duplicate native
  stock-observer marker, or a canonical engine marker in native;
- native has no observed finite serving-path mismatch dose;
- any Native boundary or derived ratio is nonfinite/invalid, or any Zero
  boundary differs;
- NaN/Inf, invalid shape, replica drift, optimizer/weight attestation failure;
- host optimizer offload, prefix cache, unregistered sampler-IS, a mixed
  sampler/TIS tuple, group filtering, or flat resampling appears;
- fewer/more than 128 raw trajectory records in any batch;
- journal continuity or digest failure;
- `BLOCKED_SANDBOX_CAPACITY` after a durable full sandbox-start-timeout batch;
- sandbox cleanup failure, OOM, IFRT/CANCELLED, or deadline nesting drift; or
- classifier verdict is not `PASS`.

Archive the exact YAML, its SHA-256, source SHA, image digest, raw log,
artifacts, and classification. A failed prerequisite or interrupted target run
is `INCONCLUSIVE`, never PASS.
