# 2x2 one-host XProf matrix (backward / rollout x native / zero)

Four thin wrappers over `run_onehost_gsm8k_xprof_common.sh` for separated
profiling and debugging. Each takes one argument: a unique label
(never reuse labels; failed run dirs are never deleted).

| script | phase | what the device trace holds |
|---|---|---|
| run_onehost_xprof_backward_native.sh | update | native monolithic train steps |
| run_onehost_xprof_backward_zero.sh | update | the whole zero-TIM backward (blocks, reduce, receipts) |
| run_onehost_xprof_rollout_native.sh | step | first ~25s of native decode |
| run_onehost_xprof_rollout_zero.sh | step | first ~25s of zero-TIM decode |

Contract note: phase and TPU trace mode are a signed pair — update takes
CANON_XPROF_TPU_TRACE_MODE=TRACE_ONLY_XLA, step must leave it empty (the
learner admits the TPU trace mode only for the update window). The
wrappers set both; the arm contract rejects every other combination.

Observed on a zero-arm rollout smoke (3 updates, ~26 min, docker exit 0,
commit norms bitwise on anchor): xplane 1.6 GB written plus a perfetto
trace, and the classifier reports six reds, all update-phase
expectations — `xprof_bytes=1706271783 exceeds_hard_max=1500000000`
(the step window traces far more than the update window; reduce
CANON_XPROF_HOST_TRACER or the capture step count if a smaller artifact
is wanted), `xprof_start_step`/`xprof_stop_step`, `xprof_census_rc`,
`size_census_rc`, and the size receipt. The vLLM teardown
`AttributeError: 'Qwen3ForCausalLM' object has no attribute 'modules'`
is known benign noise.

Rollout mode is DIAGNOSTIC: the census/classifier expectations are written
for the update phase, so rollout runs are EXPECTED to exit 1 with census
reds; the xprof/trace/perf artifacts under
/mnt/disks/tunix-data/gsm8k-onehost-xprof/<run>/train/ are the deliverable.
Host tracer and the engine [PERF] spans cover the full phase in all modes.

## Geometry: two registered one-host geometries

`V1_GSM8K_XPROF_GEOMETRY` selects the carrier mesh for BOTH arms; labels
are auto-prefixed `dp2tp2-` so runs can never be confused.

| geometry | mesh | why | warm update | commit-norm anchor (bitwise) |
|---|---|---|---|---|
| `dp4-tp1` (default) | data4 | fastest backward; the only geometry where `CANON_P71_SCAN=bwd` runs | ~15.7-16.0 s | 1.4907878637313843 / 2.2041752338409424 / 2.6263937950134277 |
| `dp2-tp2` | data2 x model2 | representative (TP collectives present, like the DP16xTP4 target); 32 groups of one row per rank | ~26.7-30.2 s | 1.6838101148605347 / 3.3025829792022705 / 1.8203867673873901 |

The anchors are geometry-scoped: never compare one geometry's norms (or
walls) to the other's. At dp2-tp2 the profile force-enables
`CANON_P66_P59_CHECK_VMA=1` (TP>1 transpose correctness; the drift guard
refuses to launch without it) and the launcher mounts the seventh
engine shim (the annotated RPA kernel) — both are automatic.
`CANON_P71_SCAN=bwd` at dp2-tp2 refuses by design ("P71 bwd block
supports TP1 only"); use `fwd` there.

## Running the LATEST optimized zero-TIM backward (this is the current recipe)

Prereqs on a fresh host: pinned image
tunix_frozenlake_image:vllm-tpu0.25.0 (id 418dc632...), the
/mnt/disks/tunix-data data layout (gsm8k_zero_tim, hf cache,
claude_work/canon_env.sh), ~/.netrc with api.wandb.ai, 4 TPU devices, and:

    export V1_GSM8K_XPROF_EXPECT_HOSTNAME=$(hostname)   # non-canonical hosts

Launch (always from the PHYSICAL worktree path, never through a symlink;
never append a pipe to the launch command):

    V1_GSM8K_XPROF_ALLOW_DIRTY=1 \
    CANON_DP_COMPARE_MODE=fingerprint-hybrid \
    CANON_DP_DISTINCT_SCHEDULE=first-group-warmup \
    CANON_DP_FINITE_FETCH=batched-commit \
    CANON_P71_SCAN=bwd \
    bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_xprof_backward_zero.sh "<host>_<date>"

Representative-geometry variant (TP present; swap the scan rung to fwd):

    V1_GSM8K_XPROF_GEOMETRY=dp2-tp2 V1_GSM8K_XPROF_ALLOW_DIRTY=1 \
    CANON_DP_COMPARE_MODE=fingerprint-hybrid \
    CANON_DP_DISTINCT_SCHEDULE=first-group-warmup \
    CANON_DP_FINITE_FETCH=batched-commit \
    CANON_P71_SCAN=fwd \
    bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_xprof_backward_zero.sh "<host>_<date>"

For the P74 checked-VMA treatment, use the signed wrapper below. It pins the
same DP2xTP2 carrier settings and leaves `CANON_P66_P59_CHECK_VMA=1` under the
profile's mandatory drift guard; it does not add a chunk-count flag:

    V1_GSM8K_XPROF_ALLOW_DIRTY=1 \
    bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_xprof_backward_p74_dp2tp2.sh "<host>_<date>_p74"

Omit `V1_GSM8K_XPROF_ALLOW_DIRTY=1` for a clean committed carrier. Launch only
from the physical worktree path, with a fresh label, and never append a pipe to
the command. The wrapper automatically emits:

- `train/p74_gap_census.txt`: one-line GREEN/RED verdict;
- `train/p74_gap_receipt.json`: all 64 seed-to-head windows, mean/max gap,
  exact P74 partition-module coverage, seven old D2H/H2D victim counts, and the
  captured `[PERF] p32_vag_reverse` wall;
- the receipt and census in the root `SHA256SUMS` ledger.

The fail-closed P74 gate is mean gap <=70 ms, exactly 64/64 windows passing
through `jit__p74_identity_head_cotangent_partition`, and zero overlap from
the old transfer family. Historical matched captures replay through the same
census as follows: r3 before the accepted device repartition was 150.746
ms/chunk and 726 ms/group; r4 was 0.063 ms/chunk and 459 ms/group. There are
two chunks per group in this frozen carrier, so the exact two boundary bubbles
shrank by 301.366 ms/group, while the independently measured end-to-end group
wall shrank by 267 ms (36.8%). These are different clocks and are deliberately
reported separately.

`num_chunks` remains the data-derived P32 specialization
`ceil(max_real_tokens / local_M)` with `local_M=256`; P74 neither hard-codes 2
nor fuses chunks. It removes the checked-VMA host materialization at every
existing chunk boundary, so longer contexts with more chunks receive the same
per-boundary fix.

A six-update horizon (dark-time and warmup-knife studies) is available
on either geometry with CANON_P33_RUN_STAGE=six-update.

Flag notes: the P68 batched receipts, the P70 jitted strips, the prep
hoist, and the reducer program cache are flagless and always on; the
v1-hp profile pins CANON_BATCHED_EVIDENCE=1 in-container (an off arm via
docker -e is impossible by design). CANON_P71_SCAN=fwd if the bwd block
stage is not desired; CANON_DP_COLLECTIVE_REDUCE defaults to psum on the
DP4 one-host profile only.

Acceptance anchors (cross-machine, same topology/image/data —
**DP4xTP1 only**; a DP2xTP2 run must never be compared against these
norms, its own anchors are established by its first green capture):
commit_gradient_norm per update must be bitwise
1.4907878637313843 / 2.2041752338409424 / 2.6263937950134277;
warm update wall ~15.7-16.0 s; the only expected classifier red in
backward mode is trace_census (trace.json export truncation, benign).
The module census takes the launched CANON_P71_SCAN value and asserts
that rung's backward inventory: off/fwd must show the per-layer
pullbacks (28 x 32 executions) and no block program, bwd must show
exactly ceil(28 / 7) = 4 `bwd_block_NN` programs at 32 executions each
and no per-layer program.  Either family appearing in the other mode is
a red, so a silent fallback cannot pass.

## Mesh geometry selector (P72)

`V1_GSM8K_XPROF_GEOMETRY` selects the carrier mesh on the same four chips
for BOTH arms; it is orthogonal to the backward/rollout x native/zero
matrix above ("2x2" in this file's title is that matrix, not the mesh):

| value | mesh | zero workload | groups/update | zero engine overlay |
|---|---|---|---|---|
| unset / `dp4-tp1` | data=4 x model=1 | gsm8k-p59-dp4-tp1 | 16 | qwen1p7b_tp1 |
| `dp2-tp2` | data=2 x model=2 | gsm8k-p59-dp2-tp2 | 32 | qwen1p7b_tp2 |

The default is byte-identical to the pre-P72 launcher.  A dp2-tp2 label is
automatically prefixed `dp2tp2-` so its runs can never be confused with
dp4 runs; every other value refuses in the launcher, the container
entrypoint, and the arm contract.  Global work is identical (prompts 8,
generations 8, trajectories 64, max_steps per stage); only the cut
changes: each dp2 rank carries twice the rows and the model shards across
model=2, so TP collectives are present in rollout and training.  The
DP4 norm anchors above and the DP4 warm-wall expectation do not transfer.
