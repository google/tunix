# GSM8K DP16×TP4 — the Zero-TIM arm and its native control

Reproduce the GSM8K Zero-TIM run: Qwen3-1.7B on 64 v5p chips as DP16×TP4, 200 updates, with the
canonical engine overlay making rollout decode, prefill re-score and the training forward
bit-identical, so no sampler correction is applied. Steps 1–4 need the chips; there is nothing to
redraw yet, because no run of this recipe has ever finished (see "Current data").

## The recipe

Both arms train `Qwen/Qwen3-1.7B` in bfloat16, seed 42, and run the same command, taken from the
rendered manifest's `CANON_RUN_CMD`: `examples/math_gsm8k/qwen3_grpo_demo.py` with `--mesh_dp=16
--mesh_tp=4 --max_steps=200 --batch_size=32 --num_generations=8 --mini_batch_size=32
--train_micro_batch_size=32 --train_trajectory_micro_batch_size=16
--compute_logps_micro_batch_size=32 --max_prompt_length=1024 --max_response_length=1024
--max_concurrency=256 --rollout_vllm_hbm_utilization=0.20 --rollout_vllm_max_num_seqs=16
--rollout_vllm_max_num_batched_tokens=256 --wandb_project=zero-tim-gsm8k-dp16-tp4`. That is 32
prompts × 8 generations = 256 trajectories per update, 200 updates, on one JobSet of 16 workers ×
4 chips (`tpu-v5p-slice`, topology `4x4x4`, namespace `default`, `maxRestarts: 0` on the Zero arm
and 3 on the native one).

Neither the dataset nor the weights are in this tree. Every fresh container downloads the OSS TFDS
`gsm8k` builder into `artifacts/qwen3_grpo_gsm8k_vtc/data` and pulls `Qwen/Qwen3-1.7B` from the
Hugging Face Hub (public, no token), so the pods need outbound network to both. The only thing
pre-staged from GCS is the JAX compilation cache, restored by
`canon-zero-tim/cluster/steps/28_sync_cache.sh`; a cold cache costs only compile time.

| | Zero-TIM | Native control |
|---|---|---|
| Profile | `canon-zero-tim/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env` | `canon-zero-tim/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env` |
| Engine | canonical overlay: 37 installed files, 6 stock engine files replaced | the image's stock vllm-tpu engine, untouched |
| Alignment | strict — `CANON_GSM8K_ALIGNMENT_WARN_ONLY=0`, a warning is a failure | off |
| Render script | `canon-zero-tim/recipes/gsm8k/render_zero_tim.sh` | `canon-zero-tim/recipes/gsm8k/render_native.sh` |
| JobSet / W&B run name | `canon-v1hp-gsm8k-<run-id>-<sha8>` | `canon-v1ctl-gsm-nat-<run-id>-<sha8>` |

Both arms report to the same W&B project and group, `zero-tim-gsm8k-dp16-tp4` /
`qwen3-1p7b-dp16-tp4`, so only the run name separates them, and both force `WANDB_MODE=online`.
Before launching, have these six in hand: the branch tip SHA you cloned; the runtime image pinned
by digest in your registry; the JobSet namespace (`default`); a GCS scratch prefix; a Kubernetes
secret `yuxzhang-secrets` holding `WANDB_API_KEY` (and `HF_TOKEN`); 64 free v5p chips.

## Steps 1 and 2 — environment and image

Identical to steps 1 and 2 of `canon-zero-tim/blog_reprod/README.md`: clone the branch and note
`git rev-parse HEAD` — the pods re-fetch the branch tip and refuse any other commit, so render at
the tip you intend to run — then pull or rebuild the same runtime image,
`tunix_frozenlake_image:vllm-tpu0.25.0`, and push it where the cluster can read it.

## Step 3 — render

```bash
SHA="$(git rev-parse HEAD)"
OUT="<an absolute, empty directory outside the checkout>"
bash canon-zero-tim/recipes/gsm8k/render_zero_tim.sh "$SHA" "$OUT/zero" g64zero
bash canon-zero-tim/recipes/gsm8k/render_native.sh "$SHA" "$OUT/native" g64nat
```

A run id is 1–16 lowercase letters, digits or hyphens, starting and ending with a letter or digit.
Both scripts refuse a dirty worktree, a `HEAD` other than the SHA you pass, and an output directory
that already exists.

expected: `V1_HP_GSM8K_P74_MANIFEST_PASS` then `V1_HP_GSM8K_P74_WAVE_READY manifests=1
source=<SHA> output=<OUT>/zero launch=not-executed` from the first script, and
`V1_GSM8K_NATIVE_FULL_MANIFEST_PASS` then `V1_GSM8K_NATIVE_FULL_READY manifests=1 source=<SHA>
output=<OUT>/native treatment=native-mismatch launch=not-executed` from the second. Seconds each,
nothing is launched, and rendering is deterministic. Each script ends with `MANIFEST=<path>`:

* `<OUT>/zero/jobset-v1-hp-gsm8k-dp16tp4-p74.yaml`
* `<OUT>/native/jobset-v1-gsm8k-native-mismatch-full.yaml`

## Step 4 — launch one arm

Start a persistent log collector first — restarted pods lose their logs, and the Zero arm's
`maxRestarts` is 0, so its only evidence is what you collected.

```bash
python3 canon-zero-tim/workloads/frozenlake-three-arm/scripts/collect_jobset_logs_to_gcs.py \
  --jobset "<metadata.name from the manifest>" --source-sha "$SHA" \
  --gcs-prefix "<gs://your-bucket/your-prefix>" --output-dir "<local-evidence-dir>" \
  --namespace default --expected-workers 16
kubectl apply -f "$MANIFEST"
```

That collector is workload-agnostic despite its directory; the GSM8K JobSet also has 16 workers.

| When | Marker that must appear | Red flag |
|---|---|---|
| step 0, Zero arm | `[env] V1 high-performance GSM8K requires strict alignment` is the refusal, so its absence is the pass; then `all 37 files match (qwen1p7b)` from the install, `[PATHTRACE]` lines, and `[CANON_ALIGN_PRE] step=0 verdict=PASS` | a clean start with no `[PATHTRACE]` line: the overlay never installed |
| step 0, native arm | `[GSM8K.NATIVE] ZERO_TIM_OFF_PASS p32=absent canonical_engine=off alignment=off p59=off v1=off`, then `[GSM8K.NATIVE] STOCK_PREFLIGHT_PASS files=6 driver_import=pass canonical_overlay=absent alignment=off` and `GSM8K_NATIVE_STOCK_PATH … canonical_overlay=skipped alignment=off` | any canonical overlay line at all |
| per update, Zero arm | one `[CANON_ALIGN_PRE] step=<n> verdict=PASS` and 16 `[CANON_ALIGN] step=<n> verdict=PASS …` lines, every boundary reporting `differing_bytes` 0; in W&B, `sampler_trainer/logp_diff_mean` == 0 | any `[CANON_ALIGN_WARNING]`; any `[RESCORE.RETRY] rows=…` line means the engine dropped prompt logprobs and the retry from `tunix/rl/rollout/vllm_rollout.py` had to cover for it — the run survives, but say so |
| at the end | `V1_HP_FULL_CLASSIFICATION verdict=PASS recipe=gsm8k zero=3400/3400 fail=0`, written by `canon-zero-tim/workloads/full-recipes/scripts/classify_full_recipe.py`, plus its `_JSON` twin | `verdict=PASS_WITH_ALIGNMENT_WARNINGS`, or any `zero=` numerator below 3400 |

3400 is `200 × (1 + 256/16)` — one pre-alignment and 16 alignment receipts per update, for 200
updates; the classifier also requires at least one P63 fallback receipt.

## Step 5 — export and register the run

```bash
export WANDB_API_KEY=<your key>
python3 canon-zero-tim/blog_reprod/export_wandb.py --entity yuxzhang-google \
  --project zero-tim-gsm8k-dp16-tp4 --run <wandb-run-id> \
  --out canon-zero-tim/results/gsm8k/<arm>/<run-id>
python3 canon-zero-tim/results/check_results.py
python3 canon-zero-tim/results/plot_curves.py --workload gsm8k
```

`<arm>` is `zero_tim` or `native`. Append one `canon-zero-tim/results/RUNS.tsv` row per run, graded
`complete` or `incomplete`; `canon-zero-tim/results/README.md` owns the layout.
expected: `RESULTS_CHECK PASS runs=<n>` and `PLOT_CURVES PASS workload=gsm8k runs=<n> …`.

## Step 6 — what counts as reproduced

* The Zero arm's logprob difference is exactly 0 at every step — not small — and every alignment
  boundary reports 0 differing bytes, all 3400 receipts.
* The run reaches update 200 and prints `V1_HP_FULL_CLASSIFICATION verdict=PASS recipe=gsm8k`.
* The solve rate is worth plotting against the native control — but not step by step, because the
  same seed is not the same trajectory.

A numeric yardstick is not available on this host: the archived run's W&B project could not be read
here (no `WANDB_API_KEY`), and no GSM8K export exists under `canon-zero-tim/results/` yet.

## Current data

None on this branch. The one 64-chip run of this recipe,
`yuxzhang-google/zero-tim-gsm8k-dp16-tp4/fmrug2iu` (JobSet `canon-v1hp-gsm8k-gfull1-799a0bd1`,
2026-08-28), reached update 64 and then died: on the 65th rollout the engine returned a single
prompt logprob for a 1130-token sequence and the fail-closed length check in the prefill re-score
raised. The fix `976c54f43` retries such a row once and prints `[RESCORE.RETRY]` instead of
crashing; it is on this branch but has never been re-run on 64 chips, so the crash is repaired in
code and unproven in practice. The native control has never completed either — five attempts on
2026-08-30 died during model construction and sharding, and the repair was never re-launched.
Renders and the CPU pod-install chain are verified for both arms; nothing here has been applied to
a cluster from this tip.
