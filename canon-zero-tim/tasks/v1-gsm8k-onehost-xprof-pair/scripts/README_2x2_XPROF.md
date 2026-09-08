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
| `dp2-tp2` + `CANON_DP_REDUCE_ONCE=1` (`CANON_P32_KEEP_TAPE=stream`) | data2 x model2 | K2: one fixed-order DP reduction per update; the sum over groups precedes the sum over ranks, so this is a deliberately re-pinned anchor | ~16.4 s (run k2_reduce_once_20260902_r4) | 1.6838101148605347 / 3.3025834560394287 / 1.8203867673873901 |
| `dp4-tp1` + `CANON_DP_REDUCE_ONCE=1` (`CANON_P32_KEEP_TAPE=stream`) | data4 | K2 on dp4-tp1: the reassociated sum came out bitwise equal to the per-group anchor (run `v1_zero-hp_k3host_dp4_20260902_r1`, 2026-09-02; warm 7.68 s vs 11.55 s stream-only, peak HBM 65.6 GiB vs 59.5) | ~7.7 s | 1.4907878637313843 / 2.2041752338409424 / 2.6263937950134277 |

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

## v2_dispatch (2026-09-06): fewer program launches per chunk pass

Measured with `census_dispatch_count.py` (TPU:0 "XLA Modules" events of the
captured update, normalised by the chunk passes read from `raw.log`):

| dp2-tp2 (32 groups, 64 chunk passes) | launches / update | warm update | anchor / A=B=C |
|---|---|---|---|
| base control 640836c6 + spelling fix (`v1_zero-hp_dp2tp2-v2disp_basefix_dp2_20260906_r3`) | 23,146 | 13.08 s | registered |
| Phase 1 knives (`…v2disp_p1_dp2_20260906_r4`, r5 byte-identical) | 9,774 | 10.73 s | bitwise / 96 of 96 |
| + one-program stream step (`…v2disp_p13c_dp2_20260906_r1`) | 9,006 | 10.57 s | bitwise / 96 of 96 |
| + `CANON_P71_SCAN=fwd_block` (`…v2disp_p3_dp2_20260906_r1`) | 7,278 | 9.95 s | bitwise / 96 of 96 |
| `CANON_P71_SCAN=bwd` at TP2 (`…v2disp_p4_dp2_20260906_r1`) | 7,598 | — | NOT bitwise: sealed |
| `CANON_P32_CHUNK_BATCH=2`, per-layer forward (`…v2disp_p5_dp2_20260906_r1`) | 6,734 | 9.38 s | bitwise / 96 of 96 |
| `CANON_P32_CHUNK_BATCH=2` + `fwd_block` (`…v2disp_p5b_dp2_20260906_r2`) | 6,734 | 9.60 s | bitwise / 96 of 96 |
| + one program per reverse chunk, nesting the built mapped pullbacks with optimization barriers (`…v2disp_p7_dp2_20260906_r1`, branch `local/v2-dispatch-p7`, not in this lane) | 4,366 | 6.14 s | NOT bitwise (norms +3.4e-5 / -1.5e-4 / +2.1e-3): sealed |
| Phase 8 host-glue folds on the certified stack: sidecar rows sliced on the host, spec arrays committed, one split program, one cast program, one relabel per update, one loss program (`…v2disp_p8_dp2_20260906_r3`, lane 569c978d) | 3,680 | 7.55 s | bitwise / 96 of 96 (HBM +4.2%) |
| Phase 9: mapped pullback programs pinned to their operands' spellings (no per-call relabel), forward tail as one program, cached chunk-start scalars, one-launch replay check (`…v2disp_p9_dp2_20260907_r1`, lane 2b298a91) | 3,200 | 5.48 s | bitwise / 96 of 96 (HBM −1.0%) |
| + one program per layer over the two reverse chunks of a group, calling the built mapped pullback twice with the cache cotangent threaded in-graph and emitted as an output (`…v2disp_p10_dp2_20260907_r1`, branch `local/v2-dispatch-p10`, not in this lane) | 2,304 | 5.40 s | NOT bitwise (norms +7.4e-4 / −2.0e-4 / +1.6e-3; the bootstrap group's per-chunk norm is bitwise, every batched group's differs): sealed (HBM +0.4%) |
| Phase 11: one zeros program per group plus the zero cache cotangents from the first chunk's rebuild, host row lengths for every group's spec (no lengths program, no per-group sync), the train example committed to the engine mesh once per update (`…v2disp_p11_dp2_20260907_r1`, lane 7ac9338b) | 3,040 | 5.44 s | bitwise / 96 of 96 (HBM +0.1%) |

| dp2-tp2-long (8 groups, 69 chunk passes) | launches / update | warm update | anchor / A=B=C |
|---|---|---|---|
| Phase 1 knives (`…dp2tp2long-v2disp_p1_long_20260906_r1`) | 7,371 | 9.28 s | bitwise / all |
| lane tip 40ae9f83, selectors off (`…dp2tp2long-v2disp_p7r4_long_ctl_20260906_r1`) | 7,179 | 9.1 s | bitwise / all |
| + `CANON_P71_SCAN=fwd_block` + `CANON_P32_CHUNK_BATCH=2` (`…dp2tp2long-v2disp_p7r4_long_sel_20260906_r1`) | 4,789 | 8.86 s | bitwise / all |
| lane e197f0e3 (Phase 8 and 9 host-glue folds) with the same selectors (`…dp2tp2long-v2disp_p10_long_ctl_20260907_r1`) | 3,071 | 7.23 s | bitwise / all (peak HBM 39.39 → 43.18 GiB, +9.6%, reached inside the first update's train phase; unattributed) |

A batched program must take the engine state as operands on every call;
capturing the leaves at build time ran the third update on stale
parameters (`…v2disp_p5b_dp2_20260906_r1`, alignment gate RED) until
1f77ddf4 made them operands.

Where the remaining launches come from (attribution capture
`v1_zero-hp_dp2tp2-v2disp_p7r3_pytrace_dp2_20260906_r2`, same code and
anchors as the certified stack, `CANON_XPROF_PYTHON_TRACER=1`, analysed
with `attribute_pytrace_launches.py`; per captured update of 6,734):

| innermost tunix frame | launches | what it is |
|---|---|---|
| `_p59_parallel_map.invoke` | 1,984 | the mapped pullback programs (necessary) |
| `agentic_rl_learner.py:2023 <lambda>` (report sidecar `value[rows]` per microbatch) | 1,536 | advanced indexing, 8 launches per leaf per group |
| `_p32_forward_group` | 608 | 288 implicit reshards of call operands (`shard_args -> shard_device_array`, seen as `jit__multi_slice`), the batched forward, the tail gather/where |
| `_p32_group_chunk_inputs` (reverse, per chunk) | 320 | 192 implicit reshards of the three spec arrays + the fused builder + a scalar |
| `_transform_value` (weight sync casts) | 310 | one `astype` launch per leaf per update |
| `grouped_inputs[k][index]` generator | 256 | one slice + squeeze per array per group |
| eager loss value / scale glue | 483 | per-group reductions, masks, clips |
| everything else | ~1,237 | the reverse's own programs, group spec, checks, accumulate |

After Phase 8 the same census reads 3,680 per update: the 1,792 mapped
pullbacks, 318 converts, 131 residual reshards, and one launch per chunk
pass for each remaining named program.  After Phase 9 it reads 3,200, and
the host plane's `shard_args` copies fall from 62,285 to about 1,300 per
update (`…v2disp_p9_pytrace_dp2_20260907_r1`): the mapped pullbacks no
longer relabel anything per call.

Both ways of nesting the built mapped pullbacks into a larger program are
sealed at TP2: across the layers of one chunk with optimization barriers
(`…v2disp_p7_dp2_20260906_r1`, +3.4e-5 on the first update's norm) and
across the two chunks of one layer with every boundary cotangent an output
(`…v2disp_p10_dp2_20260907_r1`, +7.4e-4).  In the second run the first
group of the process, which still ran per chunk to build the maps, has a
bitwise micro gradient norm while all 31 batched groups differ, so the
nested program's arithmetic is what moved -- the bf16 cache cotangent
handed from one chunk to the next inside one program is not the value the
standalone program emitted.  Emitting a boundary value as an output does
not pin its arithmetic; the forward chunk batch stays bitwise because its
cross-chunk boundary is a Pallas cache write, not a fusible XLA value.  On
this lane the reverse's launch floor is therefore one mapped pullback per
layer per chunk pass (28 of the 36 launches a pass would keep).

An fp64 arbitration of those sealed classes (Phase 13, evidence in
`../evidence/p13_fp64_arbitration_20260907/`): the certified gradient and
the sealed p10 variant's gradient, both captured at the first update, sit
0.2122 and 0.2124 (rel_l2) from an fp64 re-derivation of the same update
on the CPU and 1.30e-2 from each other (one_minus_cos 8.4e-5); the same
stock model in bf16 on the CPU sits 0.127 from fp64.  The certified
backward's gradient norms are 6.5% below fp64 and shrink with depth
(layer 27 ≈ 0.99, lower layers 0.86–0.94) while the CPU bf16 twin does
not.  A stock TPU backward of the same update (XLA autodiff of the stock
model at the same loss and point) sits 0.145 from fp64 with unbiased
norms, so the shortfall is canonical-specific; the canonical forward's
logps are unbiased against fp64 (signed mean +6e-4 nat, the stock
forward -6e-4), and nine fp64 single-placement emulations of the
engine backward (softmax and attention-core bf16, chunking, the bf16
cache carry, per-layer bf16 cotangents, per-chunk bf16 accumulation)
plus a real f32-accumulator trial capture (rel_l2 2e-4 from the
certified gradient) all reproduce nothing of it.  The remaining
difference is the engine forward's own numerical point, which zero-TIM
pins; the gradient is that function's exact VJP.  Evidence and the
probe script live next to the arbitration.  The user-approved re-pin
rule for backward
knives (fp64-equidistance plus guard rails plus a byte-identical repeat)
is in `tasks/v2_dispatch/GOAL.md`.

Phase 15 (2026-09-07): each reverse chunk of the grouped backward runs as
one program (zt_tr_bwd_chunk, c61a7497) that calls the already-built
mapped pullbacks in-graph, the Phase 7 class redone on the certified
lane.  dp2-tp2: 3,040 -> 736 launches per update (11.5 per chunk pass),
warm reverse 5.33 -> 4.94 s, peak HBM 38.27 -> 39.21 GiB, A=B=C and every
census green; long: 3,007 -> 526 launches, 7.20 -> 6.81 s, 39.96 -> 40.97
GiB.  The anchors moved by 3e-5..2e-4 relative and were re-pinned under
the user-approved fp64 rule: the new gradient sits 0.2128 from fp64
against the certified 0.2122 (one_minus_cos 2.195e-2 vs 2.181e-2),
1.255e-2 from the certified gradient, every per-kind and per-group ratio
within the rule, two byte-identical dp2-tp2 runs
(`../evidence/p15_reverse_chunk_20260907/`); the judge carries the new
anchors under mode reduce-once+chunk.  Since the first host-glue knife
the dp2-tp2 census reads 23,146 -> 736.

Defaults after Phase 16 (2026-09-07): the certified GSM8K recipe's two
forward selectors are the defaults -- an unset or empty
`CANON_P71_SCAN` selects `fwd_block` and an unset or empty
`CANON_P32_CHUNK_BATCH` selects 2 -- so the dp2-tp2 and long launches
above no longer need them in EXTRA_ENV; `CANON_P71_SCAN=off` and
`CANON_P32_CHUNK_BATCH=1` select the per-layer forward and the
per-chunk loop explicitly, and a recipe that selects the layer-scan
rung (`CANON_P28_LAYER_SCAN`) keeps both legacy paths by default.  The
module census reads the same defaults for an empty value.  Fresh
certifications with only CANON_DP_REDUCE_ONCE=1 in the environment
(`../evidence/p16_defaults_fresh_20260907/`): dp2-tp2 twice (736 launches
per update, anchors bitwise and byte-identical across the pair, 39.21 GiB,
warm reverse 4.96 / 4.98 s) and long twice (526 launches, anchors
bitwise across the pair, 40.22 GiB, 6.79 / 6.88 s), every census green.
Not run on this lane: dp4-tp1 (Not verified because the lane's six phases
were certified on dp2-tp2 and dp2-tp2-long only) and the 8B carrier
(CAPACITY_REJECT on one host).

Phase 14 (2026-09-07): the +3.7 GiB that dp2-tp2-long's peak HBM had
gained since the Phase 8/9 host-glue knives (43.08 vs 39.39 GiB) is a
pure dispatch-lead transient -- the resident profile per update is
byte-for-byte the pre-knife one; the faster host queues whole chunks of
layer pullbacks whose outputs PJRT allocates at dispatch.  The GSM8K
rank-parallel reverse now bounds that lead with the two P77 readiness
waits at every chunk boundary, waiting on one leaf of the last
dispatched program (c12363c5, c6a5f476; no flag, no value read, the
FrozenLake carriers unchanged): long 39.96 GiB and dp2-tp2 38.27 GiB with
anchors bitwise and the census unchanged (3,007 / 3,040), at the cost of
about 0.7 s of warm reverse per update on both geometries (the drains
expose host work the queue used to hide; evidence in
`../evidence/p14_long_hbm_20260907/`).

After Phase 11 the census reads 3,040 per update, 47.5 per chunk pass: the
28 mapped pullbacks, one launch per chunk pass for each named program of
the prologue (fused metadata, entry rebuild, norm and head recompute, rows
pullback, P74 partition, head and norm pullbacks) and epilogue (embed
pullback, accumulate), and per group the zeros program, the pack program,
the report adjoint, the assembly and the checks; `jit__multi_slice` is down
to 2 per update.

Every `jit__multi_slice` in the census is an implicit reshard at a program
call boundary, not a slice: pin the producers' output shardings instead of
touching the indexing.  Host side, 60,096 of the 62,285 `shard_args`
events per update are the per-invoke mesh relabels of the mapped
pullbacks' leaves (`_p59_align_to_mesh`), which change only once per
update.

Rules that came out of it:

- A treatment needs a control on the *same* base commit; the 2026-09-04
  controls are 40 commits behind and the base itself failed its second
  update (fused report accumulate signature) until the spelling
  canonicalization landed.  Run the no-knife base first when the red is in
  code you did not touch.
- New shardings are derived from `_input_sharding`, never from the trainer's
  axis name: the engine mesh is named `('dp', 'tp')` here.
- The P74 gap window opens at `jit_zt_tr_bwd_logprob` (the folded rows
  pullback) and the module census counts `jit__precomputed_gradient_adopt_scaled_step`
  as the scaled step; both are equal-power replacements, not removals.
- Nesting already-compiled programs into one program is bitwise on the
  TPU only when every boundary value is a program output (the forward
  batch keeps each layer's hidden and cache as outputs); the reverse chunk
  program left its intermediate cotangents in-graph and moved the anchors
  even with an optimization barrier on every former boundary, and XLA:CPU
  shows the same context dependence through FMA contraction.  Verify a
  boundary's arithmetic with a probe before spending a certification run.
- dp4-tp1 is not verified: the base's fused report accumulate rejects the
  TP1 embedding cotangent VMA (`v1_zero-hp_v2disp_basefix_dp4_20260906_r1`).

## tasks/v2_integrate (2026-09-08): the merged branch, its fresh certifications and the bounded-lead knife

`local/v2-integrate` = 6842edae + the v2 one-host line squashed into eight
thematic commits + the dispatch lane squashed into four + a merge of the
TiTO release line (cd955f99) + four follow-ups (demo sys.path, P61 capture
on the FrozenLake no-commit carrier, the pack-budget backpressure knife, the
FrozenLake launcher's second physical worktree).

Fresh certifications on the merged tree (defaults: fwd_block + two-chunk
batch + reverse chunk program; EXTRA_ENV `CANON_DP_REDUCE_ONCE=1`, tape
`stream`):

| run | dispatch/update | anchors (reduce-once+chunk) | peak HBM | warm reverse |
|---|---|---|---|---|
| `…dp2tp2-v2int_a_dp2_20260908_r2` (d2c40608) | 736 | bitwise | 39.21 GiB | 4.986 s |
| `…dp2tp2long-v2int_a_long_20260908_r2` | 526 | bitwise | 40.22 GiB | 6.823 s |
| `…dp2tp2long8k-v2int_a_long8k_20260908_r2/r3` | truncated capture (structural) | pinned from two byte-identical runs: 5.4485979080200195 / 4.931449890136719 / 3.995054006576538 | 46.83 GiB | 12.42 s |
| `…dp2tp2-v2int_c_dp2_20260908_r1` (8c9d49f1, knife d=2) | 736 | bitwise | 39.21 GiB | 4.773 s (-4.3%) |
| `…dp2tp2long-v2int_c_long_20260908_r1` (8c9d49f1, knife d=2) | 526 | bitwise | 42.40 GiB (+5.4%) | 6.438 s (-5.6%) |
| `…dp2tp2-v2int_c_dp2_20260908_r2` (b7cd9502, lead budget 0 = the shipped tree) | 736 | bitwise | 39.21 GiB | 5.003 s |

The two-chunk lead (`_P77_LEAD_PACK_BUDGET_GIB`) costs one in-flight gradient
pack: over the +5% HBM gate on dp2-tp2-long, so the shipped tree keeps the
Phase 14 two waits at every chunk boundary on every carrier (FrozenLake
included) and the lead stays a certified-but-disabled path until the 64-chip
admission run shows its headroom.

The long8k xprof/P74 censuses stay RED on every run of that geometry (the
profile capture ends before the update does: span 11.6 s vs a 12.4 s
reverse; the legacy 2026-09-03 run had the same shape), so long8k is
certified by anchors, HBM, timing and the hierarchy/semantic censuses only.
The first launch of the merged tree (`v2int_a_dp2_20260908_r1`) died on an
import: the demo had put the tunix package directory on sys.path, which let
tunix/examples shadow the repo's examples namespace once the TiTO line
imported examples.frozenlake at module level (fixed in d2c40608).
