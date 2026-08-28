# State

## Current P58.18 checked-VMA matched-triplicate checkpoint (2026-08-28)

- Status: ON-A/OFF/ON-B exact-geometry implementation and construction gates
  are complete; source publication is approved for this delivery. The remote
  executor must fetch and record the final operator-branch tip. Target not run;
  no image, Kubernetes object, or TPU work was created.
- Each independent JobSet keeps the signed Qwen3-4B-Instruct-2507 carrier:
  128 chips split into rollout DP8xTP8 and trainer DP8xTP8, clean 1,012 tasks,
  B8xG16, 16K, 50 turns, seed 42, concurrency 128, fixed lm-head,
  continue-decode 8, prefix cache off, strict Step-0 A/B/C, full trajectory
  journal, then controlled exit before VJP/backward/optimizer.
- Selector contract: `on` derives checked-VMA/P66/P67=`1/1/1`; `off` derives
  `0/0/0`. Both diagnostic arms force first-update/P63=`0/0` because backward
  is forbidden. Selector absent leaves production Zero-HP `1/1/1/1/1`
  unchanged.
- Parallel plan: three JobSets request 384 TPU chips, three anti-affined CPU
  heads, and aggregate sandbox concurrency 384 (768 requested CPU and 1,536
  GiB at the signed per-sandbox requests). Render PASS is not aggregate
  capacity or Kueue admission evidence.
- Interpretation: concurrent ON-A/OFF/ON-B is one matched OFF control plus two
  ON replicates, not a temporal ABA sandwich. Cross-run token identity is not
  required. Each arm must independently return exact B-C and valid finite or
  exact A-B evidence with zero training activity.
- Validation: Python/shell syntax; focused renderer 27/27, profile 9/9,
  per-arm classifier 7/7, and wave-contract 4/4; deterministic flag audit
  `393/393/393`; and the complete pinned dependency-image gate all pass. The
  terminal image marker includes `checked_vma_diagnostic=1
  checked_vma_aba=1 ... regressions=1`. This is construction evidence, not a
  Pathways/TP8 target result.
- Phase: `phases/p58-18-checked-vma-aba-wave.md`.

## Current p58z08 intake correction (2026-08-27)

- Status: analysis complete; P58.17 exact-geometry checked-VMA-off target is
  still NOT RUN. A fresh pull left the operator worktree at
  `5d4f2fceb6996bb0a5e2149a21c8fd846d89dcb5` with no newer target artifact.
- `p58z08` used source `395c0e0de8626c96e85457b997efddd2dd2dec48`
  and the ordinary `zero-hp-full` job identity. It did not contain the P58.17
  selector, diagnostic job name, precheck marker, controlled exit, or
  diagnostic classification. Checked VMA, the first-update gate, and P63 were
  enabled. Therefore it is not a failed checked-VMA-off arm.
- Rollout/data fact: 128 rows; 120 succeeded, five model-timeout, three
  context-limit; four solved trajectories; two effective prompt groups; 30
  admitted nonzero advantages. This is not an all-zero or sandbox-capacity
  failure.
- Numerical fact: `N_action=389067`; B-C (`S_prefill_vs_T_old`) is exact; A-B
  has 17,507 differing elements and 39,031 differing bytes. First delta is
  `0.02544403076171875` at an environment-to-action boundary; later maximum is
  `9.499740600585938`. The incident report incorrectly labels the byte count
  as token differences; preserve the report and use this correction.
- First failing boundary: strict pre-backward A-B gate. No VJP, backward,
  optimizer update, or checkpoint commit executed. The archived raw
  log/report lack a complete packaged run and classifier, so the evidence is
  analysis-grade.
- Next action: render and, only after separate approval, run the exact P58.17
  128-chip `zero-hp-vmaoff-precheck` discriminator. Exact A-B implicates the
  topology-shaped P67/VMA scope and triggers an explicit pullback-identity
  repair; finite-red A-B with exact B-C promotes seam replay. Do not launch
  another ordinary 1,000-update Zero-HP job before this gate.

## Current P58.17 decode-vs-prefill seam checkpoint (2026-08-27)

- Status: one-host diagnostic complete; exact-geometry checked-VMA-off
  discriminator implemented and host-tested, target not run. Implementation
  `b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9` is published on the operator
  branch and its first remote readback matched HEAD/FETCH_HEAD/tracking with
  ahead/behind `0/0`. No image publication or Kubernetes mutation occurred.
  One real direct-attached four-chip v5p carrier ran locally.
- Immutable target fact: `p58z07` ran source
  `ef46b0b3a5d8754160f0cce323ec3861b04dccdc` on disjoint rollout/trainer
  DP8xTP8 roles. It returned 121 `SUCCEEDED` and seven `MODEL_TIMEOUT` rows,
  six solved trajectories, and 31 admitted nonzero advantages. The
  P58.16 four placement/state/JIT/scorer contracts passed. The strict
  pre-backward gate then stopped before VJP/AdamW/checkpoint.
- Numerical fact: `N_action=379496`; `S_prefill_vs_T_old` is exact;
  `S_decode_vs_S_prefill` has 32,952 differing elements and 71,797 differing
  serialized bytes. The earlier log entry calling 71,797 a token count is
  corrected here. First mismatch absolute delta is `0.00435257`; maximum is
  `11.87498` later in the trajectory.
- Artifact join: all 1,024 reported mismatches exactly match durable token ID,
  action mask, and decode logprob. They map to trajectory rows 49 and 62, the
  same signed Pillow task. Shift-0 median absolute delta is `0.0040245`; -1/+1
  medians are about `0.4952/0.4922`, refuting a simple token offset.
- Implementation: `classify_decode_prefill_probe.py` fails closed on missing,
  non-finite, count-drifted, or unjoinable evidence. The default-off seam
  selector extends only the Zero-HP DP1xTP4 backward-no-commit carrier to the
  frozen task, G2, 4K response, 16 turns, serial scheduling, strict alignment,
  durable trajectory output, and automatic return bundle.
- Local result: `p58s17` returned two successful real R2E trajectories,
  `N_action=4808`, and diagnostic `PASS / FINITE_RED_REPRODUCED` with zero
  optimizer commits. A-B differs at 2,488 elements (`max_abs=1.3662147522`),
  and B-C differs at 988 elements. Shift 0 is decisively closer than shifts
  -1/+1, so a simple token displacement is rejected.
- Evidence: bundle
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s17_20260827t1045z/P58_SEAM_PROBE_RETURN.tar.gz`
  has SHA-256
  `6285b5d2e8958ee85bd4b4190beaa240c7239ad6d07165a0948d7ba7f2b32eee`.
- Repair intake: the one-host trainer now uses topology-aware device order;
  the Zero path installs the generated runner as a real package overlay;
  alignment normalizes inactive `top_k/top_p=None`; and the final runner pins
  the frozen whitelist SHA in its manifest. TP8-only model/kernel overlays are
  excluded from TP4 by construction.
- Validation: focused probe/one-host tests pass 11/11; the pinned-image
  alignment suite passes 43/43; and the complete P58 exact-image gate ends in
  `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 ...
  disaggregated_trainer_mesh=4 ... regressions=1`.
- Claim boundary: this carrier does not match `p58z07` exactly because local
  B-C is also RED. It is not forced-token replay and cannot certify
  TP8, DP8, disaggregated Pathways, backward, optimizer correctness, or
  convergence. The next decisive experiment is an admitted, Step-0/no-commit
  checked-VMA-off selector on the exact DP8xTP8+DP8xTP8 carrier; do not
  improvise the selector against the production full profile.
- Exact-geometry implementation: renderer flag
  `--checked-vma-off-diagnostic` creates a 128-chip Zero/full HP carrier and
  sets the single selector `CANON_P58_CHECKED_VMA_DIAGNOSTIC=off`. The profile,
  `00_env.sh`, authoritative `env.sh` reload, Python contract, runtime marker,
  controlled-exit postflight, durable classifier, flag registry, and
  render-only preparation wrapper all fail closed. The selector derives
  checked VMA/P66 alias/P67/first-update gate/P63 to zero and preserves every
  other signed P58 Zero-HP serving/data/geometry field. Production default is
  unchanged when absent.
- Target evidence contract: 128 durable rows; finite positive action count;
  exact B-C; either exact or finite-red A-B classification; exactly one Step-0
  precheck and code-42 exit; no fixed-head VJP, P59/P66 backward, global step,
  or optimizer update. The full run root is returned. Target remains NOT RUN.
- Phase: `phases/p58-17-decode-prefill-seam-probe.md`.

## Historical P58.16 pre-target NNX loader-metadata checkpoint (2026-08-27)

- Status: implementation commit
  `dba5211ac4945fefb50337603c800d9f8e3d37b5` is published and read back on
  `yuxzhang/canon-zero-tim`; pinned dependency-image CPU gates PASS. No
  matching image was published and no 128-TPU retry has run. `main` is
  untouched.
- Source intake: the clean isolated worktree fast-forwarded from
  `a04b65febcb5e163bf1f30bf33065decbe29651f` through five remote commits to
  `959a3258fe70230c483cec9a25b191b7b3d4ab4b`. The incoming P58 artifact is
  the 7,958-line `p58z06` raw log; the other incoming runtime changes belong
  to P68/DP-collective and M15 concerns and do not repair this failure.
- Immutable target fact: `p58z06` loaded the exact 1,012-task clean list,
  admitted 128 TPU devices with disjoint DP8xTP8 rollout/trainer roles, and
  completed vLLM model warmup. It then failed during adapter initialization
  after the first placement receipt, before rollout. No trajectory, trainer
  logprob, alignment, backward, AdamW, commit, or checkpoint exists. The raw
  log lacks a source SHA; none is inferred. Evidence checksums pass under
  `evidence/p58z06_nnx_loader_metadata_error/`.
- Root cause: Pathways dummy loading adds `_is_loaded=True` to every one of
  the 398 live NNX parameters. Flax includes variable metadata in State
  treedefs, while the weight-free trainer `nnx.eval_shape` reconstruction has
  no loader marker. P58.15 compared raw treedefs and rejected this provenance
  difference; segmented backward contained the same latent comparison.
- Repair: compare logical NNX treedefs after removing only exact
  `_is_loaded=True` from copied Variables. False/non-boolean values fail;
  every other metadata field, path/type, leaf count, shape, and dtype remains
  exact. The same check guards segmented backward. The full classifier now
  requires exactly one 398-leaf state-contract receipt.
- Validation: Python compilation and diff hygiene PASS; forced four-CPU-device
  nested-JIT forward/backward, segmented pullback, partial-overlap negative,
  and false-marker negative PASS; Zero-HP classifier 7/7 PASS. The complete
  pinned-image gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ...
  disaggregated_trainer_mesh=4 ... regressions=1`. No TPU is exposed locally,
  so this is construction evidence only.
- Next action: source publication is complete. Matching-image publication,
  Kubernetes apply, and target launch remain separately approval-gated. Once
  the matching image is published/read back and sandbox admission passes, use
  fresh `p58z07`; never resume/overwrite `p58z01`-`p58z06`. Require four
  placement/state/JIT/scorer receipts, trainer old/current logps, strict
  A=B=C, finite nonzero 16-group backward, and one coherent update-0 commit.
- Phase: `phases/p58-16-nnx-loader-metadata.md`.

## Current P58.15 nested-JIT trainer-mesh checkpoint (2026-08-26)

- Status: implementation commit
  `f60cdd569c2737df6cb2968125c8e42680938981` published and
  dependency-image CPU gates PASS; 128-TPU retry not run.
- Source intake: `git pull --ff-only` first advanced the isolated P58 worktree
  to `a36cbd1b156e013a75af4071e91a238be49bc95b`, then reconciled the local
  repair over `98a2dfd9e8ece301374fdfb55518b3bc9ebef4d4`. Immediately before
  publication the remote advanced again to exact base
  `be758e68faa9db5b06be153a0656c4c861e3119f`; that incoming commit contains
  M15 evidence only. Neither intervening commit overlaps P58. The repair was
  rebased, revalidated, committed as the exact implementation SHA above, and
  published only to the operator branch; `main` is untouched.
- Immutable target fact: `p58z04`, built from
  `3f159250c4781b3faafde238f768457a0478446b`, emitted both P58.14 placement
  receipts and completed all eight prompt groups / 128 Step-0 trajectories in
  1,709 seconds. Eight `MODEL_TIMEOUT` and one
  `MAX_CONTEXT_LIMIT_REACHED` rows were compact statuses. The first hard
  failure was the first trainer old-policy-logprob call: trainer state occupied
  one 64-device role while a `jit inside jit` still named the disjoint rollout
  role. No trainer logprob, alignment, backward, AdamW, commit, or checkpoint
  completed. Evidence checksum passes.
- Root cause: P58.14 rebound explicit adapter shardings but reused vLLM
  `model_fn` and `compute_logits_fn`. Both nested JITs were created at engine
  initialization with rollout-mesh output shardings. Its prior CPU mock used
  plain Python functions and therefore missed this captured-device closure.
- Repair: disaggregated trainer execution reconstructs the same live NNX graph
  weight-free with `nnx.eval_shape`, validates exact state tree/shape/dtype,
  and rebuilds both nested JITs on the trainer execution mesh. The segmented
  trainer forward/backward uses the same reconstructed graph. The fixed-AR
  mesh global is rebound only inside a locked trace context and restored.
  Native, colocated, serving, sampling, loss, strict A=B=C, optimizer, and the
  signed B8xG16 recipe are unchanged.
- Validation: forced 2-rollout + 2-trainer CPU tests execute the real
  nested-JIT `value_and_grad` and segmented layer pullback with finite nonzero
  gradients; partial overlap still fails closed. The complete pinned-image
  gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ...
  disaggregated_trainer_mesh=4 ... regressions=1`. No `/dev/vfio` is present,
  so this is construction evidence, not target proof.
- Next action: fetch/read back the final operator tip and prove it contains
  implementation `f60cdd569c2737df6cb2968125c8e42680938981`, build/read back
  a matching image, rerun the complete
  gate, pass sandbox admission, obtain separate launch approval, and render
  fresh `p58z05`. Require exactly the original two placement receipts plus
  `trainer model callables rebuilt ... mesh_bound_jits=2`, then trainer
  old/current logps, strict A=B=C, finite nonzero 16-group backward, and one
  coherent update-0 transaction. Never resume/overwrite `p58z01`-`p58z04`.
- Evidence/phase: `evidence/p58z04_disaggregated_mesh_error/` and
  `phases/p58-15-nested-jit-trainer-mesh.md`.

## Current P58.14 disaggregated trainer-mesh checkpoint (2026-08-26)

- Status: source implementation
  `dce0e93777548b7623e4f41702144f8d00f242f5` published and dependency-image
  CPU gate PASS; no 128-TPU retry has run.
- Source intake: clean worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, initially pulled at
  `3820b168457830112e6ce4b505fcedc9691bd705` and finally reconciled to exact
  operator tip `bde8f4c6e055ff077b24af716857786ce967f422`, then publication-time
  tip `9ae21d22c2c096d4c2b39724b40e87768ece8934`. The intervening
  FrozenLake source and M15 evidence commits did not overlap the P58 repair.
  `main` is untouched.
- Immutable target fact: `p58z03`, built from
  `8eb65480d3705d96ab282799ad5a6c1901596248`, returned all 128 Step-0
  trajectories and proved the P58.13 fixed-head M=`2048/256` admission. It
  then failed while compiling the first canonical trainer old-policy-logprob
  forward because trainer-state arrays and adapter sharding constraints named
  disjoint 64-device role sets. It did not complete trainer logprobs,
  alignment, forward execution, backward, AdamW, a commit, or a checkpoint.
  Earlier Pallas/VJP markers were tracing receipts, not execution proof.
- Root cause: the canonical adapter was constructed from the rollout runner
  only, so its differentiable input/cache/output/sample constraints and
  mesh-bound log-softmax scorer retained rollout devices while consuming
  trainer-resident state.
- Repair: adapter registration now supplies trainer state; the adapter derives
  the exact trainer `dp,tp` mesh, preserves the engine axis topology on those
  devices, and binds only the differentiable trainer forward there. Serving
  remains rollout-bound. Disaggregated serving/trainer scorers are separate
  mesh-bound instances from the same factory/math. DP/TP mismatch and partial
  device overlap fail closed; colocated and Native paths retain their prior
  behavior.
- Validation: Python compilation and diff hygiene pass. A forced four-CPU
  disaggregated `jax.jit(value_and_grad)` executes with finite primal,
  finite/nonzero gradient, and a partial-overlap negative. Existing colocated
  adapter regressions pass. The complete local dependency-image gate exits
  zero with `P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=3 ...
  regressions=1`. The image has no `/dev/vfio`, so no TPU/Pathways claim is
  made. An unrelated pulled stale flag-count assertion was corrected from 385
  to the authoritative 386; its 31-test suite passes.
- Next action: fetch the final operator tip and prove it contains implementation
  commit `dce0e93777548b7623e4f41702144f8d00f242f5`. Then build/pin a matching
  image, rerun the complete gate, pass sandbox admission, obtain separate
  launch approval, and render fresh `p58z04`.
  Require the two `[CANON_ADAPTER.PLACEMENT]` receipts, completed trainer
  logprobs, strict A=B=C, finite nonzero backward, and the coherent update-0
  transaction before continuing the same 1,000-update job. Never resume or
  overwrite `p58z01` through `p58z03`.
- Evidence/phase: `evidence/p58z03_device_sharding_error/` and
  `phases/p58-14-device-sharding-mismatch.md`.

## Current P58.13 Qwen3-4B M2048/P59-only VMA checkpoint (2026-08-26)

- Status: completed source repair; implementation published and pinned-image
  construction PASS. Target `p58z03` proved M=2,048 admission and exposed the
  P58.14 trainer-mesh bug before trainer execution.
- Source: worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`. Implementation commit
  `bea1aabde39c43c13ca4eaefab989301c6e8b46c` is published on
  `yuxzhang/canon-zero-tim`, rebased over the latest FrozenLake P67 full-run
  promotion `c73c9a6c3676c9a1ba27e9b871b0f2e14ff6adb4`, and exact
  local/FETCH_HEAD/remote-tracking readback matched at `0/0`. `main` is
  untouched.
- Immutable target fact: `p58z02` proved the P58.12 global JAX seed route and
  returned all 128 Step-0 rows in one 1,514.2-second wave. It contained one
  `MODEL_TIMEOUT` and two `MAX_CONTEXT_LIMIT_REACHED` rows; those were compact
  statuses, not the crash. The first hard failure was later in trainer
  canonical per-token-logprob forward, before alignment completion, backward,
  AdamW, or any optimizer commit.
- Root error: Qwen3-4B TP8 produced caller-global fixed-head semantic M=2,048
  (`data=8`, local M=256), but the fixed-head registry admitted M=2,048 only
  for Qwen3-8B TP8. Qwen3-4B `(hidden=2560,tp=8)` fell back to learner
  M=`(4096,)` and rejected shape `(2048,2560)`.
- Fixed-head repair: register learner M `(2048,4096)` only for exact Qwen3-4B
  TP8 and retain the existing exact Qwen3-8B TP8 registration. Every other
  geometry retains `(4096,)`; Qwen3-32B TP8 remains a negative for M=2,048.
- Shared FrozenLake repair: Wave 5 proved strict A-B/B-C `0/0` with
  checked-VMA scoped to P59 backward. Exact P58 Zero-HP now derives
  `CANON_P67_P66_VMA_P59_ONLY=1` together with checked-VMA. Native raw,
  Native+IS, non-HP Zero, Qwen3-32B, and unrelated profiles remain off. This
  preserves the serving graph; it does not relax the strict A=B=C gate.
- Validation: 50/50 focused host tests, P34 static 10 suites, P57 146/146,
  and the flag-registry regression pass. The Qwen3-4B installed overlay
  matches 37/37 files and reports `learner_M=2048,4096`; the independent
  Qwen3-32B image gate reports `learner_M=4096`. The complete pinned image
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... qwen4b_fixed_head=1
  checked_vma=1 vma_p59_only=1 first_update=1 ... regressions=1`. No
  `/dev/vfio` is visible, so no TPU target is claimed.
- Historical transition: the matching target was rendered as fresh `p58z03`.
  Preserve it as immutable P58.14 trigger evidence; never resume or overwrite
  `p58z01`, `p58z02`, or `p58z03`.
- Evidence/phase: `evidence/p58z02_backward_fixed_lm_head_error/` and
  `phases/p58-13-backward-fixed-lm-head-m2048.md`.

## Current P58.12 JAX engine-seed/cleanup checkpoint (2026-08-26)

- Status: completed source repair; `p58z02` target proved the engine-global
  seed route and exposed the later P58.13 trainer-logprob fixed-head failure.
- Source: worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`. The repair was built on exact pulled base
  `7f6fc071082f291bf926b1c5bc79021733628c2e`; implementation commit
  `c10fbe0487d1f6635975b84806f1efdce6bc95c1` was pushed only to
  `yuxzhang/canon-zero-tim` and immediate local/FETCH_HEAD/remote-tracking
  readback matched with ahead/behind `0/0`. The published history preserves
  immutable `p58z01` Attempt-0 evidence. `main` is untouched.
- Last verified target fact: `p58z01` admitted 128 TPU devices, the exact
  DP8xTP8 roles, 1,012 clean tasks, 128 R2E sandboxes, and a compiled vLLM
  engine. The first Step-0 generation failed because P58.10 routed seed 42 to
  JAX `SamplingParams.seed`, which is unsupported. Abort cleanup then hit the
  kubernetes-client empty-body `None.decode` defect. No trajectory, backward,
  optimizer commit, or resumable checkpoint exists.
- Repair: seed 42 remains common to dataset and rollout, but JAX receives it
  through global `EngineArgs.seed`; per-request JAX seed now fails early rather
  than being silently dropped. Runtime/postflight require the exact
  `[P58.SEED] ... scope=engine-global` and `[VLLM.JAX_SEED] ...` receipts.
  W&B/manifests/classifiers use the same bounded scope. R2E cleanup handles
  only the exact empty-body decode defect, confirms 404 within the existing
  deadline, and retries the exact Pod DELETE when its first outcome is
  ambiguous; unrelated errors remain fatal.
- Frozen workload: Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, rollout DP8 x TP8 plus trainer DP8 x TP8, 128 chips,
  TPU-resident optimizer, strict A=B=C, and 1,000 commits. No numerical or
  algorithmic flag changed.
- Validation: syntax/compile/diff hygiene pass; focused sampler 7/7, one-host
  artifact 5/5, and bounded cleanup regression pass; P34 static emits
  `suites=10`; P57 passes 146/146; flag audit passes 385/385. The complete
  pinned image exits zero with `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1
  checked_vma=1 first_update=1 stable_clip=1 ... regressions=1`. It reports no
  `/dev/vfio`, so no TPU target is claimed.
- Latest-tip reconciliation: the operator branch advanced during validation
  by three P4.10/P66 commits. The local branch fast-forwarded to
  `7f6fc071082f291bf926b1c5bc79021733628c2e`; shared `90_run.sh` retained both
  the new FrozenLake diagnostics and the P58 seed receipts without conflict.
  Final gates are rerun on that exact tip.
- Next action: the execution agent fetches the final operator tip, proves it
  contains implementation commit `c10fbe0487d1f6635975b84806f1efdce6bc95c1`,
  and builds/pins the matching image. Image publication, Kubernetes
  application, and a fresh `p58z02` target each require their separate
  approval. Never resume or overwrite `p58z01`.
- Phase: `phases/p58-12-jax-engine-seed-cleanup.md`.

## Current P58.11 checked-VMA Zero-HP checkpoint (2026-08-26)

- Status: implementation published and construction gates passed; target
  Attempt 0 (`p58z01`) reached vLLM initialization but failed before first
  generation on the P58.10 JAX per-request seed route. Superseded as the
  active repair by P58.12; its numerical gates remain the retry contract.
- Source intake: clean worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, constructed at fetched operator tip
  `644beb38cee2388862941019269ad264a581064f`, then fast-forwarded without
  overlap over V1-only evidence tip
  `4003f61cabb6f2d5e43d4c217cebb4dca2c3d217` before publication.
- Objective: admit the shared P66 checked-VMA P59 backward repair,
  `CANON_V1_HP_FIRST_UPDATE_GATE`, and P63 overflow-safe global-norm clipping
  into exactly the strict P58 Qwen3-4B Zero-HP full profile. Native raw,
  Native+IS, ordinary Zero, P44/P46, and diagnostics remain unchanged.
- Shape correction: P58 has eight outer 16-trajectory chunks but sixteen
  rank-major DP8 gradient groups. The real accumulator denominator and
  first-update microstep count are 16. Global canonical M is 2,048 and the
  shard-local/kernel M is 256.
- Frozen recipe: Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, rollout DP8 x TP8 plus trainer DP8 x TP8, 128 chips,
  optimizer resident, prefix cache/APC/sampler IS/group filtering off, strict
  A=B=C, and exactly 1,000 committed updates.
- Implementation: the exact Zero-HP profile derives checked-VMA, the internal
  P66 compatibility alias, the first-update gate, and P63 max-norm 1.0. The
  runtime uses `contract_name` as DeepSWE workload identity, executes the
  16-group precommit/commit gate, persists P63 evidence, and exports stable
  norm, naive-norm-finite, fallback, and clip-factor W&B metrics. Postflight
  requires exactly 1,000 P63 commit receipts, P59/checked-VMA receipts for
  every ordered attempt, and exactly two update-0 first-update receipts. A
  legal all-compact attempt is reconciled as zero-commit and removed from
  committed-step timing. Native, Native+IS, and non-HP Zero remain isolated.
- Additional commit-boundary repair: P58 intentionally carries the shared P33
  launch-admission bit, but `DeepSWEWorkload` has `contract_name`, not `name`.
  The old unconditional schedule check would therefore fail at the first real
  P58 optimizer commit. Schedule identity now uses the normalized workload
  identity; a real CPU P58 16-group/0-to-1 commit regression covers it.
- Validation: syntax, Python compilation, and diff hygiene pass. Focused
  profile 7/7, classifier 5/5, first-update 6/6, stable-clip source 3/3,
  exact-image environment 12/12, P63 validator/commit 10/10, and P58 real CPU
  first-commit integration 1/1 pass. Adjacent P34 emits
  `P34_STATIC_PASS suites=10`; P59 host 37/37, V1 Phase4 76/76, and P57
  146/146 pass. Flag audit is 383/383 with `FLAG_AUDIT_PASS`. Complete pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 checked_vma=1
  first_update=1 stable_clip=1 ... regressions=1`.
- Claim ceiling: the pinned container reports no `/dev/vfio`; no TPU,
  Pathways, R2E sandbox, Kubernetes, or 128-chip target was run. This is local
  construction and one CPU optimizer transaction only, not target evidence.
- Next action: P58.12 source is published. After matching-image readback,
  sandbox-capacity admission, and separate launch approval, use the corrected
  engine-global seed route for a fresh `p58z02` JobSet. The P58.11
  first-update and 1,000-commit gates remain unchanged.
- External effects: source commit/push only. No image publication, Kubernetes
  mutation, TPU run, model download, credential change, or artifact deletion.
- Phase: `phases/p58-11-qwen4b-zero-checked-vma.md`.

## Current P58.10 fixed-seed checkpoint (2026-08-24)

- Status: implementation published and exact-read back; pinned-image
  construction PASS. Worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, was built on operator tip
  `687b2bd6d0815b5628af39e7adbf949e429e72ae`, then replayed without conflict
  over latest fetched tip `ff646a4d76f58e9f328bc640f44d362637eb1432`.
  Implementation commit `9597de3d99fbf65c87f4fea3d86e639cca0b7abe`
  was pushed only to `yuxzhang/canon-zero-tim`; immediate local/FETCH_HEAD/
  remote-tracking readback matched with ahead/behind `0/0`. The older P58
  worktree was already dirty with unrelated P59/V1 work and was not modified.
- Contract: every P58 Native-raw, Native+IS, and Zero-HP render contains
  exactly one `--seed=42`. Any missing, duplicate, or different seed is
  rejected. The training entry point requires 42 for P58, uses it for both
  dataset shuffle and `RolloutConfig.seed`, and emits `[P58.SEED] PASS`.
- Provenance: W&B records `seed`, `rollout_seed`, and `seed_scope`; the durable
  run manifest records `dataset_seed=42`, `rollout_seed=42`, and the same
  scope. P58 classifiers require those fields.
- Claim boundary: this is configuration-level reproducibility. vLLM/R2E
  generation and sandbox collection remain asynchronous, so bitwise-equal
  trajectory order or identical end-to-end artifacts across independent jobs
  is not claimed.
- Validation: Python compilation and `git diff --check` pass; focused
  renderer/sampler/one-host tests pass 33/33. Bare-host artifact/classifier
  imports are unavailable because `metrax` is absent. The complete pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
  onehost_xprof=1 zero_hp_full=1 ... regressions=1`.
- Next action: fetch the final operator tip containing implementation commit
  `9597de3d`, render the fresh Native+IS full JobSet, and verify both its
  unique `--seed=42` argument and first `[P58.SEED] PASS dataset_seed=42
  rollout_seed=42 ...` marker. Launch remains separately user-gated.
- Source was committed and pushed only under explicit approval. No image
  publication, Kubernetes mutation, live-job stop, or TPU execution occurred.
- Phase: `phases/p58-10-fixed-seed.md`.

## Current P58.9 publication checkpoint (2026-08-24)

- Status: Native-IS target selected and source publication complete.
  Implementation commit `2aedd73c957abba29d21d05b866a996af2f66dfd`
  was pushed only to `yuxzhang/canon-zero-tim`; first remote readback matched
  local/FETCH_HEAD/remote-tracking refs with ahead/behind `0/0`. Worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824`, branch
  `local/p58-is-zero-refine-0824`, was originally based on
  `614156c1ab067192ab65b2969543e23904f192be` and replayed over exact operator
  tip `7b85b42d0a019d70f32a7dc9712c538ad42f5cb5` before publication.
- Scope: add a third executable recipe, Native+token-IS, without changing the
  existing Native-raw or Zero-HP numerical programs. Native+IS is selected
  only by renderer `--arm native --sampler-is`, resolves the existing disable
  tuple to `0:0`, uses threshold `2.0`, and requires trainer-old logps plus
  materialized TIS weights. Native-raw and every Zero recipe remain `1:1`.
  Partial tuples and IS on Zero fail closed. Group filtering remains absent.
- Retry correction: P58 is restored to exact Attempt-0
  (`maxRestarts: 0`, `restartStrategy: Recreate`). The prior JobSet retry used
  the same persistent run root without attempt isolation. Five renderer-only
  keepalive environment names had no exact-image code consumer and were
  removed rather than represented as recovery controls.
- Shared recipe remains Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, 128-chip synchronous disaggregated DP8 x TP8 per role,
  one-hour batch deadline, TPU-resident optimizer, prefix cache off, and full
  1,000 committed updates.
- Latest target fact: the operator reports a sharp training-reward drop in the
  live Native/no-IS run and judges the run collapsed. The onset update is not
  established and must not be recorded as a fixed optimizer step. Its
  exact run id, log, W&B series, and checkpoint receipts are not yet present
  locally, so the root cause is not independently classified here. The
  execution decision is final: preserve that run as immutable Native-raw
  failure evidence, stop its exact JobSet, do not resume its optimizer
  checkpoint, and do not launch Native raw again.
- Evidence after replay: focused renderer/profile/sampler/observer tests 40/40,
  Python/Bash syntax, and diff hygiene pass. Bare-host environment import
  is `INCONCLUSIVE` because `metrax` is absent. The complete pinned-image gate
  passes with `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
  zero_hp_full=1 ... p59_real_shim=4 p59_rpa=2 ... m15_token=1
  regressions=1`.
- Claim ceiling: implementation plus pinned-image construction only. No direct
  TPU run, Pathways target, optimizer commit, full training, or performance
  result exists.
- Next action: the remote executor identifies and archives the exact live
  Native-raw JobSet, then deletes only that JobSet and verifies cleanup. After
  fetching the final operator tip and proving it contains implementation
  commit `2aedd73c`, render and launch a fresh `--arm native
  --sampler-is` full run from the original frozen base checkpoint using a new
  run id/root/W&B/checkpoint. Never resume the collapsed Native-raw state.
- Blocker: source publication is not blocked. Remote execution must still
  preserve/stop the exact Native-raw run, pin the final read-back source SHA
  and image digest, pass render/admission checks, and never resume the
  collapsed optimizer state.
- Phase: `phases/p58-9-native-is-attempt-zero-refine.md`.

## Current P58.6/P58.7/P58.8 checkpoint (supersedes the legacy P58.5N snapshot below)

- Status: active; P58.6/P58.7 are implemented and the approved four functional CLs plus one audit-only release CL on latest tip `ccbcf572` pass the post-barrier host and pinned-image admission ladder. Publication is authorized; exact remote readback and hardware targets remain open.
- Objective: produce a matched one-host Qwen3-4B Native versus optimized Zero-HP XProf/Perfetto pair, then run the optimized strict-Zero Qwen3-4B DeepSWE-derived DP8 x TP8 full 1,000-commit campaign after separate publication and launch approval.
- Worktree: `/home/yuxuan/code_rl_repro/worktrees/p58_zero_hp_release3_0823`, branch `local/p58-zero-hp-release3-0823`, based exactly on latest fetched operator tip `ccbcf572dc903bb1cce12f897cbdb05aec94922a`. It retains all earlier immutable failure evidence and P58 restart/keepalives plus the new P57 evaluation-cycle, final-only checkpoint, and lazy NumPy host-render fixes. Older dirty and release worktrees remain untouched.
- Current phase: P58.8 latest-tip release publication. The CL-A/B/C/D/E scope and reverse rollback order are in `RELEASE_CL_PLAN.md`; P58.6 direct-host pair, P58.7 target, and P59 hardware target remain `NOT RUN`.
- Implementation: P58.6 has two thin wrappers, one fail-closed common driver, signed provenance/work/state hashes, a warmup plus identical no-commit update capture, XPlane/trace/semantic Perfetto requirements, arm classifier, cross-arm classifier, and immutable package sealing. P58.7 has an additive default-off Zero-HP profile, renderer selector, DeepSWE exact admission, Qwen3-4B TP8 fixed-head receipts, P59 receipts, update XProf/Perfetto wiring, and a 1,000-update strict postflight/performance ledger. P58.8 repairs the fetched Phase4 P59 TP4 nested-mesh first red with an exact-device two-axis engine carrier covering TP4/TP8, and scopes the P57 Zero/full W&B project to its signed profile.
- Numerical policy: any real Zero alignment FAIL kills the candidate. P59 is accepted under ordinary-JAX FP64 gradient correctness; serial-AdamW weight trajectory identity is explicitly not claimed. APC stays off for P58.7.
- Latest-tree host validation: P59 30/30, current P57 136/136 including the wrong-profile W&B negative control, V1 12/12, APC 31/31, flag registry 366/366, Python/Bash syntax, and `git diff --check` PASS. Bare-host P58 discovery executes 51 tests but cannot collect four modules because this shell lacks `metrax`; this is `INCONCLUSIVE` dependency coverage, superseded for those modules by the complete pinned-image gate.
- Pinned-image validation: image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passes `P59_TP_SHIM_EXACT_IMAGE_PASS fixed_head=2 installed_projection=2 report_adjoint=2 fixed_reducer=2 topologies=DP2xTP4,DP2xTP8 optimizer_commits=0 manifests=2x36/36`. Both P58 and V1 complete exact-image suites terminate with `p59_real_shim=4`. Positive paths execute the modified P59-local fixed-head/projection VJPs and emit `all_gather_rank_order_f32_barrier`; negative paths keep ordinary global outputs and exact device-index maps. TP8 initially exposed BF16 per-rank-add error (`0.5` max abs versus FP64 while the serial probe was exact); fixed-rank TP partials now accumulate in barrier-pinned FP32 and cast once, after which TP4 and TP8 serial/parallel probes are byte-exact.
- Release-tree correction: the unrelated APC-on availability decision from the older dirty tree is excluded. A first complete P58 exact-image rerun then failed only because its vLLM test double omitted the stock `num_cached_tokens` field; the mock now supplies zero. P58 and V1 exact-image reruns both terminated PASS on that first release tree before the operator tip advanced. No real run, numerical gate, or optimizer commit was involved in the construction red.
- Latest-tip reconciliation: the earlier `24b1bbcf` tree preserved P58 `maxRestarts=3` and Pathways/IFRT/GRPC keepalives. The new `ccbcf572` reconstruction additionally preserves all three later P57 fixes while migrating only P58/P59 dirty hunks. FP32 TP accumulation now uses fixed-reducer-style operand barriers, and both complete exact-image gates passed after that change.
- Target not run: no direct TPU XPlane/Perfetto pair, DP8 x TP8 Pathways run, 4B/TP8 fixed-head target, P59 target, full evaluation/checkpoint horizon, or 1,000-commit result exists. The fifth CL only prevents immutable log markers from masquerading as settable flags during changed-base auditing; it changes no runtime. No image publication, YAML apply, or TPU launch occurred.
- Hardware limitation: the available direct one-host v5p exposes four chips, while every registered production fixed-head geometry is TP4 or TP8. A real P59 DP>1 composition therefore needs at least DP2 x TP4 (eight chips); the requested DP2 x TP2 fixed-head program would be an unregistered artificial geometry. No one-host TPU composition PASS is claimed.
- Next action: push the approved five-CL stack and require exact remote readback. After publication, keep P58.6 and P58.7 deferred while the separately approved V1 GSM8K DP16 x TP4 full run performs the first repaired target certification; P45 and M15 follow only after their preceding full postflight.
- Latest evidence: `evidence/p58rel3-p58-exact-image-20260823/`, `evidence/p58rel3-v1-exact-image-20260823/`, and `evidence/p58rel3-release-tree-20260823/`; committed runtime/test delta manifest SHA `babc1c708f7cee01c14e465058991013fd5483e6a0a75b7c367a22cd44e329da`.
- Runbook: `RUNBOOK_P58_6_7.md`. Phase files: `phases/p58-6-onehost-native-zero-xprof.md`, `phases/p58-7-qwen4b-zero-hp.md`, and `phases/p58-8-p59-tp-mesh.md`.
- Updated: 2026-08-23 UTC, publication closeout.

## Legacy P58.5N snapshot (historical; no longer current)

- Status: active
- Objective: run the untreated native DeepSWE-derived Qwen3-4B-Instruct path as a full 1,000-update 128-chip campaign on the frozen clean-data recipe. Preserve the zero arm for the eventual causal comparison, but do not launch it until its optimization work is complete and the user explicitly reactivates it.
- Definition of done for the active phase: the published native source and image pass preflight; one 128-chip native JobSet runs rollout DP8 x TP8 plus trainer DP8 x TP8; exactly 1,000 optimizer commits complete; full trajectory, signal, mismatch, optimizer-placement, cleanup, evaluation, and checkpoint evidence is durable; at least one finite nonzero Native serving-path mismatch is observed across A-B or B-C; every Native numerical boundary is shape-valid and finite; and the native full-stage classifier says `PASS`. The first three commits are online monitoring milestones, not an early-stop canary. This does not complete the deferred two-arm comparison.
- Task directory: `canon-zero-tim/tasks/p58-deepswe-native-zero-comparison`
- Directory state: source intake was fast-forwarded from exact operator tip `5f449cc8def801b4a61387ef664b2cb1f7ab05cf`, including immutable p58f12 evidence and a later P57-only checkpoint change. During publication the remote advanced by P57-only evidence commit `e7958a27851931ab9bcff232088efd95bbc12021`; the first normal push was safely rejected, that commit was reviewed as non-overlapping, and the P58 commits were rebased without conflict. The ordinary all-compact no-commit path, sandbox-capacity circuit breaker/probe, tests, and reconciled handoff are implementation commit `135867f04bfa0fc90ea1d4528ba59f365573a78b`. This publication-evidence checkpoint follows it; executors must fetch and exactly read back the final operator tip rather than pinning the implementation commit. No zero-TIM flag was enabled, deleted, or repurposed; `main` remains untouched.
- Current phase: P58.5N — native 128-chip full 1,000-update campaign.
- Completed/closed phases: P58.1 loss/compact-filter/accumulation contract; P58.2 paired profile, renderer, full-trajectory journal, classifier, negative controls, and exact-image regression gate. P58.3 one-host sanity was explicitly `WAIVED` by the user on 2026-08-21; it is not a PASS. P58.4N three-update was superseded by the user's direct-full decision after p58c05 failed before execution; it is not a PASS.
- Last verified fact: `p58f12` proved the p58f11 normalized-prompt repair and wrote a valid durable 128-row Step-0 journal, but all 128 RepoEnv pods remained Kueue `scheduling_gated` until sandbox-start timeout. Every row became signed compact-filtered `ENV_TIMEOUT`; no rollout generation occurred, so no sampling-transform provenance existed. Learner preprocessing nevertheless requested processed `S_prefill` and raised `RuntimeError: processed S_prefill must follow generate()` before alignment/backward/update/checkpoint. Effective R2E throughput was zero; this is CPU sandbox scheduling/capacity evidence, not a model/vLLM-throughput diagnosis.
- Local repair: a signed P58 all-compact batch now takes an explicit empty-completion observer path: it validates prompt/completion structure and the observer signature, skips the engine because there are zero scored targets, and records `engine_called=false` rather than fabricating log probabilities. Alignment admits zero action tokens only with durable all-compact P58 provenance and still rejects unsigned/partial zero-action input, nonfinite values, or nonzero gradients. Ordinary all-compact batches caused by model/context/runtime outcomes complete the existing zero-gradient/no-commit transaction, suppress weight sync plus `policy_version`/RL/trainer/optimizer-step advance, increment only durable `batch_index`, and consume the next prompt batch without resampling. A full `all_sandbox_start_timeout_batch` is different: after its journal and bounded metrics are durable, the learner emits `[P58.SANDBOX_CAPACITY] BLOCKED ... optimizer_commits=0 prompts_consumed_after_batch=0` and raises `BLOCKED_SANDBOX_CAPACITY` before rescore/alignment/trainer or the next prompt batch. Inconsistent infrastructure evidence fails closed.
- Validation: shell syntax, Python compilation, `git diff --check`, and the production-shaped probe/verifier regression pass 4/4 on host. The complete pinned-image gate at `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` exits zero after the new three circuit-breaker tests, probe tests, empty-rescore tests, full P58 suites, 60 shared-common tests, 42 alignment tests, trainer accumulation/compact tests, P34/P44 adjacency, and stock observer; terminal marker is `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`. Host-only full import remains unavailable because this shell lacks `metrax`; the pinned image supplies it. No target TPU/Pathways/Kueue execution was run. Prior direct-one-host metadata/device exposure remains blocked and is not a TPU PASS.
- Next action: fetch and exactly read back the final `yuxzhang/canon-zero-tim` tip. Before fresh native full `p58f13`, a remote operator must render one real-task-image sandbox probe, separately approve/apply it, and obtain `P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only`; also confirm Kueue/node capacity for the 128-Pod request floor (256 requested CPU and 512 GiB requested memory, plus head/cluster overhead). Retain concurrency 128, B8 x G16, all deadlines, topology, TPU-resident optimizer, and every Native/Zero flag. Require a later effective batch to reach finite Native boundaries/backward and the first TPU optimizer commit. Do not bypass Kueue, enable SandboxFleet, or render/apply zero.
- Blockers: p58f08 through p58f12 have no optimizer checkpoint and are immutable `INCONCLUSIVE`; p58f12 has a diagnostic trajectory journal but no resumable trainer state. Exact live ClusterQueue/ResourceFlavor/node capacity is not present in the returned log and must be confirmed remotely. Main `c7d8950f12a9c55a976bf2e1a0d8b447d71c20b3` contains Agent Sandbox/SandboxFleet commit `e789573964b6f695ded85fe519040bd06a2b9f37`, but it is deliberately not integrated: it cannot create quota, is not yet P58 Kueue/fail-closed, and its lookahead can request 256 sandboxes for B8 x G16. Zero launch remains deferred.
- Key artifacts: `plan.md`; `HANDOFF.md`; `log.md`; `phases/p58-5-native-full.md`; `../../debug_logs/p58_p58f12_deepswe_empty_batch_rescore.raw.log` (SHA-256 `10f718fb6221e3bfb3ae509ff394fbf6ea44caab1a9388c3ae1033f6410e109a`); its classification JSON (SHA-256 `e0831acd814b5398726e3a24f5063a73090e51bcbf9bcbf3d9b39614d9b626e3`); target journal `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f12/debug/batch-000000.trajectories.jsonl.gz` (SHA-256 `d4453eb0873a89933ebd1ccd281fd97c22f86f2870f3ac76dff7fefecae8986c`); prior immutable evidence listed in `log.md`; `../../cluster/P58_DEEPSWE_TIM_RUNBOOK.md`; `../../tests/p58_deepswe_native_zero/`.
- Updated: 2026-08-23 UTC
