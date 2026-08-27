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

Flag notes: the P68 batched receipts, the P70 jitted strips, the prep
hoist, and the reducer program cache are flagless and always on; the
v1-hp profile pins CANON_BATCHED_EVIDENCE=1 in-container (an off arm via
docker -e is impossible by design). CANON_P71_SCAN=fwd if the bwd block
stage is not desired; CANON_DP_COLLECTIVE_REDUCE defaults to psum on the
DP4 one-host profile only.

Acceptance anchors (cross-machine, same topology/image/data):
commit_gradient_norm per update must be bitwise
1.4907878637313843 / 2.2041752338409424 / 2.6263937950134277;
warm update wall ~15.7-16.0 s; the only expected classifier red in
backward mode is trace_census (trace.json export truncation, benign) plus
xprof_census when bwd blocks are on (census still expects per-layer
module names; instrument follow-up).
