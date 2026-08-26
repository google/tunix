# P58 DeepSWE native-first training handoff

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
