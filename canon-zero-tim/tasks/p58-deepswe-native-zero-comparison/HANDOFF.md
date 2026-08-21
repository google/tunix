# P58 DeepSWE native-first training handoff

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

The user changed the execution order again: waive the optional one-host sanity
and the separate three-update stop, then run only the native 128-chip full
1,000-update stage. Updates 1–3 are live monitoring milestones in the same job,
not an early-stop condition. Zero is not optimized enough for launch and is
explicitly deferred. No Kubernetes apply or TPU launch is authorized by this
handoff alone.

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
conditions without inspecting the pod spec/environment. The P58 renderer now
uses the reference sandbox concurrency 64, so
the unchanged B8 x G16 batch is created in two waves. This changes neither
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

Never modify or push `main`. The publication target is exclusively
`yuxzhang/canon-zero-tim`; the implementation is present there.

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
- native finite A-B warning boundary with B-C strict, and zero all-boundary
  strictness;
- native stock optimizer transaction receipts plus zero explicit fixed-tree
  transaction receipts;
- P58 fail-closed postflight classifier and automatic invocation from
  `90_run.sh`; and
- negative/regression controls for P34/P44 and the shared trainer/loss paths.
- authoritative resolved-environment reload semantics so child-shell unsets
  remain absent in all later entrypoint steps.

The exact run instructions and artifact interpretation are in
`canon-zero-tim/cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## Validated locally

Pinned local image ID:

```text
sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```

The gate covers the P58 loss oracle, unequal-effective-row gradients, real
trainer accumulation, stock/canonical all-filter discard, journal resume,
native-dose/zero-exact classifier negatives, both renderer arms/stages,
environment resolution, the full alignment suite, and relevant P34/P44
regressions.

One earlier attempt to run the complete legacy P34 static wrapper reached its
final device-probe test and timed out because this host has no TPU. The nine
preceding P34 suites passed. That device-probe result is `INCONCLUSIVE` and is
not represented as TPU evidence; the final P58 image gate runs the directly
relevant P34 contract/environment/renderer regressions instead.

No one-host real Qwen/R2E rollout and no 128-chip target training update has run. A
fresh inventory found Qwen3-4B weights but no `libtpu.so`, so this host cannot
run the requested direct-attached v5p validation. The current claim is
implementation plus CPU/exact-image validation only.

## Next executor sequence — native only

1. Read `state.md`, `plan.md`, this handoff, the superseded P58.4N phase file,
   the active `phases/p58-5-native-full.md`, and
   `cluster/P58_DEEPSWE_TIM_RUNBOOK.md` completely.
2. Fetch `yuxzhang/canon-zero-tim`, detach at its exact remote-tracking SHA,
   prove a second remote readback matches, and require a clean tree. Never use
   `main`.
3. Rerun syntax, `git diff --check`, the P58 renderer/profile/environment
   tests, and the pinned exact-image gate.
4. Publish or select a client image by immutable registry digest and verify the
   mounted Qwen3-4B-Instruct-2507 weights and frozen clean-list digest without
   printing credentials.
5. Render only `arm=native, stage=full` with fresh run-id `p58f01` and worker
   sentinel `tpu-v5p-slice`. Require exact `4x4x8` topology and no literal
   `cloud.google.com/gke-nodepool: tpu-v5p-slice`; preserve the YAML/digest and
   run server-side dry-run before the separately approved apply.
6. Monitor Kueue admission, the first completed trajectory batch, and commits
   1–3 without stopping a healthy job. Continue through checkpoint 8, updates
   32 and 100, then every 100 updates.
7. Require the full native classifier JSON to say `PASS`, including finite
   nonzero A-B, exact B-C, exactly 1,000 commits, device optimizer, complete
   journal, cleanup, evaluation, checkpoint, and transaction receipts.
8. Do not render or apply zero.

Do not reuse any failed `p58c01` through `p58c05` YAML/run root. None
contains resumable trajectory state. They remain immutable failure evidence
under `evidence/p58c01/`, `evidence/p58c02/`, `evidence/p58c03/`, and
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

- `use_rollout_logps=true` is shared. Sampler-IS, TIS correction, group
  clip/filter, degenerate-group masking, and flat-group resampling are off.
- All-zero/all-one reward groups remain. They naturally produce zero RLOO
  advantage and are logged.
- Compact-filter statuses are not malformed trajectories. They remain in the
  full journal but have zero policy mask. Structural missing/duplicate/parser
  failures remain fatal.
- A Kubernetes sandbox start exception must propagate after deletion is
  confirmed. `ENV_TIMEOUT` is an admitted compact-filter status; a
  half-created RepoEnv with `container=None` is forbidden. If an entire
  p58f01 batch has zero confirmed Running pods, classify infrastructure
  capacity/scheduling before another launch instead of patching websocket
  decode or inventing a successful trajectory.
- Read `deepswe/all_sandbox_start_timeout_batch` first. Value `1` means the
  effective R2E environment throughput was zero and the model was not the
  first bottleneck. A zero sandbox-start ratio plus a nonzero
  `deepswe/status/model_timeout_ratio` instead points to model-serving
  throughput. W&B dimensions are fixed and low-cardinality; detailed
  scheduler text is available only in the bounded raw marker.
- If an entire batch is compact-filtered, `batch_index` advances but
  `optimizer_step` and commit count do not. Relaunching into a complete
  journal continues at the next batch index; a partial/digest-mismatched
  journal stops fail-closed.
- The native arm is stock numerical training plus observation. It must not
  inherit `CANON_FIXED_AR`, `CANON_LOGPROB_M`, the canonical module, VJP2, or
  the excess-precision pin. The zero arm retains the complete bundle.
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
behavior. Native exact A-B is `NO_TREATMENT`; missing evidence or interrupted
execution is inconclusive.
