# DeepSWE-4B — 128 v5p chips, native vs Zero-TIM

Run the DeepSWE agentic coding RL recipe — Qwen3-4B-Instruct-2507 solving R2E-Gym repository tasks,
one sandbox container per trajectory — on 128 v5p chips, in either of the two arms this branch
renders: native, the untreated numerical baseline, and Zero-TIM, where the canonical engine overlay
makes the rollout logprobs and the trainer's forward bit-identical.

## The recipe

Every value here was read back from a rendered manifest. Shared by both arms: Qwen3-4B-Instruct-2507,
read from the model PVC `haoyugao-cpu-np-pvc` (volume `p34-data`) at
`/mnt/disks/linchai_data/models/Qwen3-4B-Instruct-2507`; 32 worker pods of 4 chips — instance type
`4x4x8`, 128 v5p chips — split into two disjoint 64-device roles, rollout DP8×TP8 and trainer DP8×TP8;
`--stage full`, which is 1000 optimizer updates (`--max_steps=1000`, and the pod asserts
`CANON_P58_EXPECTED_UPDATES=1000`); 8 prompts × 16 generations = 128 trajectories per update; prompt
limit 4096, response limit 16384, at most 50 turns per episode; RLOO advantages with
`sequence-mean-token-scale` aggregation, learning rate 1e-6, seed 42, and `--ckpt_dir=none`, so there
is no checkpoint and no resume. The dataset is the R2E-Gym subset, split `train`, 4578 source rows,
filtered to the reviewed 1012-task whitelist
`canon-zero-tim/clean_data/p46_q4_learnable/p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl`,
whose sha256 both the renderer and `canon-zero-tim/cluster/steps/00_env.sh` assert before anything
runs. Sandbox runtime `direct` puts one R2E pod per trajectory in the JobSet's own namespace
(`trellis`) on CPU node pool `sandbox-cpu-pool`, driven through `pods/exec` by the Pathways head,
which sits on `cpu-np` under service account `xpk-sa`. Admission is Kueue queue `multislice-queue`.

## Where the arms differ

| | native | Zero-TIM |
|---|---|---|
| Old-policy logprobs | the rollout logprobs, uncorrected — nothing makes them match what the trainer computes | the rollout logprobs, which the canonical overlay makes bit-identical to the trainer's |
| Sampler correction | none | none |
| Render script | `canon-zero-tim/recipes/deepswe-4b/render_native.sh` | `canon-zero-tim/recipes/deepswe-4b/render_zero_tim.sh` |
| Asserted at update 0 | no canonical marker leaked into a stock run: `[P58.TIM_RECIPE] PASS recipe=native-raw sampler_is=none old_logps=rollout tis_weights=absent threshold=inactive group_filter=none`, then `[P58.NATIVE] RUNTIME_PATH_PASS canonical_markers=0` | decode, prefill re-score and the trainer agree to the byte: `[CANON_ALIGN_PRE] step=0 verdict=PASS` with every entry of `bounds` zero |

Both arms send the same trainer command line; the profile and the engine are what differ. Zero-TIM
renders the strict system-optimization lane (`--system-optimization-arm treatment`), which sets
`CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0`, so a drift fails the run instead of warning it; the renderer's
`--high-performance` sets that flag to `1` instead, and these wrappers do not use it.

The Standard and token-level TIS arms of the Figure 4 page do not exist here. The renderer admits two
arms only — `_ARMS = ("native", "zero")` in `canon-zero-tim/cluster/render_deepswe_comparison.py` —
plus a `--sampler-is` modifier accepted for `native` alone, which makes a separate `native-is` recipe,
not a TIS arm comparable to Zero-TIM. Standard means "the trainer re-scores the sampled tokens and
that is the old policy", which on FrozenLake is `--old_logps_source=trainer` in
`examples/frozenlake/train_frozenlake_qwen3.py`; `examples/deepswe/train_deepswe_nb.py` has no such
flag, and its nearest relative `--use_rollout_logps` defaults to off but the renderer emits it
unconditionally. A DeepSWE Standard arm is a code change, not a configuration.

Before launching, have these six in hand: the branch tip SHA you cloned; the runtime image pinned by
digest in your registry; namespace `trellis` with room on `sandbox-cpu-pool` for 128 concurrent
sandboxes; the model PVC name; a GCS scratch prefix; a W&B API key.

## Steps 1 and 2 — environment and image

As in steps 1–2 of `canon-zero-tim/blog_reprod/README.md`: clone the branch and render at the tip you
intend to run, because the pods re-fetch that tip and refuse any other commit; then pull or rebuild
the runtime image and push it where the cluster can read it. DeepSWE has no image of its own — the
base manifest `canon-zero-tim/cluster/jobset-64chip.yaml` pins the training container by tag
(`tunix_frozenlake_image:vllm-tpu0.25.0`) and the renderer overwrites that one container with the
`<client-image@sha256>` you pass, which the wrappers check because the renderer refuses an image that
is not digest-pinned. The Pathways proxy and server containers keep the base manifest's own digests.

## Step 3 — render the two arms

```bash
SHA="$(git rev-parse HEAD)"
OUT="<an absolute, empty directory outside the checkout>"
IMAGE="<your-registry>/tunix_frozenlake_image@sha256:<64 hex>"
POOL="<the TPU node pool holding the 32 v5p hosts>"
bash canon-zero-tim/recipes/deepswe-4b/render_zero_tim.sh "$SHA" "$OUT/zero" dszero "$IMAGE" "$POOL"
bash canon-zero-tim/recipes/deepswe-4b/render_native.sh "$SHA" "$OUT/native" dsnat "$IMAGE" "$POOL"
```

A run id is 1–9 lowercase letters, digits or hyphens: the JobSet name is a 27-character prefix such
as `canon-p58-128s-zsoptt-full-` plus the run id, and the renderer caps a JobSet name at 36.

expected: `P58_DEEPSWE_TIM_RENDER_PASS arm=zero stage=full recipe=zero-systemopt-treatment
topology=128 transport=token-in-token-out output=<OUT>/zero/jobset-deepswe-4b-zero-1000.yaml` from
the first script, then the same line with `arm=native recipe=native-raw` and
`jobset-deepswe-4b-native-1000.yaml` from the second. Each script ends with `MANIFEST=<path>`.
Seconds per arm, nothing is launched, and rendering is offline: a placeholder digest and node pool
render fine, because the renderer checks shapes, never the registry or the cluster. The wrappers
create the output directory, refuse to overwrite a manifest already there, and default the three
placement flags to what the last 128-chip run used — namespace `trellis`, sandbox pool
`sandbox-cpu-pool`, head pool `cpu-np` — overridable with `DEEPSWE_JOBSET_NAMESPACE`,
`DEEPSWE_SANDBOX_NODEPOOL` and `DEEPSWE_CPU_NODEPOOL`.

## Step 4 — launch one arm

Start a persistent log collector first: restarted pods lose their logs, and on this workload the head
pod's stdout has been seen to rotate past the first updates.

```bash
python3 canon-zero-tim/workloads/frozenlake-three-arm/scripts/collect_jobset_logs_to_gcs.py \
  --jobset "<metadata.name from the manifest>" --source-sha "$SHA" \
  --gcs-prefix "<gs://your-bucket/your-prefix>" --output-dir "<local-evidence-dir>" \
  --namespace trellis --expected-workers 32
kubectl apply -n trellis -f "$MANIFEST"
```

| Marker that must appear | Red flag |
|---|---|
| Zero-TIM, per update: `[CANON_ALIGN_PRE] step=<n> verdict=PASS`. Alignment is audited on update 0 and every tenth after it (`CANON_ALIGNMENT_AUDIT_EVERY=10`, from the profile); on an audited update `bounds` reads `[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]`, in between it is empty | any non-zero entry in `bounds` |
| Zero-TIM, at startup: `[PATHTRACE] CANON_FIXED_AR=1 fixed-order tree at …` and `[PATHTRACE] CANON_FIXED_AR_EMBED=1 fixed-order embed gather (tp=8)` | no `[PATHTRACE]` line at all: `canon-zero-tim/cluster/steps/90_run.sh` then prints `FATAL: no PATHTRACE for the fixed-order reductions` and voids the run whatever its exit code |
| Both arms, once: `[DEEPSWE.TITO] ADMISSION_PASS contract=<c> arm=<arm> mode=token-in-token-out retokenize_sampled_tokens=0`, then many `[DEEPSWE.TITO] CONTINUATION turn=… prompt_tokens=… sha256=…` | either count at zero; `90_run.sh` greps both and exits non-zero |
| Both arms, per update: `[P34.WEIGHTS] EXACT step=<n> … devices=64` and a line ending `update_step_committed train_steps=<n>` (its prefix is built from the trainer DP size, so `[CANON_P34_DP8]` here) | a finished rollout batch with no commit line behind it |
| native only, once each: `[P58.TIM_RECIPE] PASS recipe=native-raw …` and `[P58.NATIVE] RUNTIME_PATH_PASS …` | any canonical marker in a native run |

Two failures are worth recognising early, both seen on this recipe. A crash in the first update's
old-policy logprob forward with a Pathways `CompilationResponseProto` above 2147418111 bytes is the
compile-response limit, not a numerics bug, and it costs about an hour of compile before it reports.
A batch that loses most of its trajectories to `ENV_TIMEOUT` is the sandbox pool degrading, which
starves data-parallel ranks and trips a fail-closed contract in the backward. The profiler is off for
`--stage full`, so expect no XProf artifacts.

## Step 5 — export and register the run

Follow `canon-zero-tim/results/README.md`: export the run, add one `RUNS.tsv` row, re-check.

```bash
python3 canon-zero-tim/results/check_results.py
```

expected: `RESULTS_CHECK PASS runs=<n>`. For the 128-chip runs this branch knows about the W&B history
is empty, so `canon-zero-tim/blog_reprod/export_wandb.py` has nothing to fetch and the realistic path
is the `receipts-only` grade: rebuild `history.csv`, `config.yaml` and `summary.json` from the run's
console receipts and the per-step metrics it wrote under `CANON_STATE`, since `check_results.py`
accepts exactly those three names and a `_step` column contiguous from 0.

## Step 6 — what counts as reproduced

* Zero-TIM: every audited update reports `verdict=PASS` with every `bounds` entry exactly 0 — not
  small. That is decode, prefill re-score and the training forward agreeing to the byte over every
  action token of the update.
* The first update commits: one `update_step_committed train_steps=1` behind a `[P34.WEIGHTS] EXACT`
  line, with a finite gradient and a valid optimizer transaction.
* Wall clock is in the right ballpark: about 1843 s per update in the steady state on the Zero-TIM
  treatment arm, with update 0 near 2900 s because it carries the first trace compilation.

## Current data

This recipe has been launched on 128 chips several times and has never finished: bd07 on 2026-09-15,
bd10 on 2026-09-16 — five optimizer updates at about 1843 s each before the sandbox pool degraded —
and k34 on 2026-09-01 with 22 updates. No W&B history exists for any of them and nothing has been
exported to `canon-zero-tim/results/` yet. The console evidence for bd10 sits at archive commit
`678170d29`: read it with
`git show 678170d29:canon-zero-tim/debug_logs/canon_p58_deepswe_bd10_perf_20260916/ANALYSIS.md`,
beside `run.log.gz`, `pre_alignment.jsonl`, `updates.jsonl.gz` and `env.sh` in that directory.
