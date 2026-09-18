# Figure 4 — FrozenLake short-horizon, three arms

Reproduce the three curves of Figure 4 — Standard, token-level TIS and Zero-TIM — on the FrozenLake
short-horizon recipe, Qwen3-8B on 64 v5p chips.
Steps 1–4 need the chips; steps 5–6 redraw the archived runs on any machine.
Nothing in this tree has been launched on a cluster yet; rendering, the pod install chain, the image
rebuild and one-host training were verified (details in `canon-zero-tim/README.md`).

## The shared recipe

Shared by all three arms: Qwen3-8B on 64 v5p chips as DP8×TP8; FrozenLake with grid sides 2–9, at
most 5 turns per episode, prompt limit 4096, generation limit 2048; 32 prompts × 8 generations = 256
trajectories per update; GSPO-token loss with RLOO advantages, one optimizer iteration per fresh
batch, learning rate 1e-6, clip 0.003 / 0.005; temperature 0.7, top-p 1, top-k 0; seed 42; 300
updates requested, the first 200 plotted.

## Where the arms differ

| | Standard | Token-level TIS | Zero-TIM |
|---|---|---|---|
| Old-policy logprobs | trainer re-score (`old_logps_source: trainer`) | trainer re-score | the rollout logprobs themselves, bit-identical to the trainer's |
| Sampler correction | none (`sampler_is: none`) | token-level TIS, threshold 2.0 | none |
| Render script | `canon-zero-tim/recipes/frozenlake-short-horizon/render_standard.sh` | `canon-zero-tim/recipes/frozenlake-short-horizon/render_tis.sh` | `canon-zero-tim/recipes/frozenlake-short-horizon/render_zero_tim.sh` |
| Reference endpoint (trailing-10 solve rate at display step 200) | 0.6265625 → 62.7% | 0.66875 → 66.9% | 0.882421875 → 88.2% |

Zero-TIM additionally runs the canonical engine overlay with exact token-in-token-out and evaluation
off; Standard and TIS run the stock vllm-tpu engine with evaluation every 50 updates. All three are
configured for 300 updates; Figure 4 uses the first 200 observations (W&B `_step` 0–199).

Before launching, have these five in hand: the branch tip SHA you cloned; the runtime image pinned
by digest in your registry; the JobSet namespace (`default`); a GCS scratch prefix; a W&B API key.

## Step 1 — environment

```bash
git clone --depth 1 --branch yuxzhang/canon-zero-tim https://github.com/google/tunix.git
cd tunix
git rev-parse HEAD
export R2E_K8S_NAMESPACE=default
python3 -c "import yaml; print(yaml.__version__)"
```

expected: `git rev-parse HEAD` prints a 40-character SHA and PyYAML prints its version. ≈1 min. The
pods re-fetch the branch tip and refuse any other commit, so clone and render at the tip you intend
to run (`canon-zero-tim/README.md` §0.3).

## Step 2 — the runtime image

```bash
docker pull europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image:vllm-tpu0.25.0
bash canon-zero-tim/image/build_frozenlake_image.sh
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0 --devices 4
```

expected: `stack OK (vllm-tpu 0.25.0)`. Pull the image or rebuild it with the second command, then
push yours where the cluster can read it. ≈10 min to pull, 343 s to rebuild, seconds to verify. The
last line needs a host with chips; it is the only check that catches a libtpu ABI mismatch.

## Step 3 — render the three arms

```bash
SHA="$(git rev-parse HEAD)"
OUT="<an absolute, empty directory outside the checkout>"
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_standard.sh "$SHA" "$OUT/standard" f4std
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_tis.sh "$SHA" "$OUT/tis" f4is
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_zero_tim.sh "$SHA" "$OUT/zero" f4zero
```

A run id is 1–12 lowercase letters, digits or hyphens, starting and ending with a letter or digit.

expected: `P57_THREE_ARM_RENDER_PASS wave=standard output=<OUT>/standard` from the first script,
the same line with `wave=is` from the second, and from the third `V1_P67_FROZENLAKE_WAVE_READY
manifests=2 source=<SHA> output=<OUT>/zero token_continuity=both-exact
token_continuity_debug=record-full p45_length_sort=off launch=not-executed`.

Each script ends with `MANIFEST=<path>`. Seconds per arm, nothing is launched, and rendering is
deterministic. Each wrapper also writes a long-horizon manifest beside the one you want — apply only
these three files:

* `<OUT>/standard/short-horizon/jobset-frozenlake-short-horizon-standard-300.yaml`
* `<OUT>/tis/short-horizon/jobset-frozenlake-short-horizon-is-300.yaml`
* `<OUT>/zero/frozenlake-short-horizon/jobset-frozenlake-short-horizon-zero-300.yaml`

## Step 4 — launch one arm

Start a persistent log collector first — restarted pods lose their logs.

```bash
python3 canon-zero-tim/workloads/frozenlake-three-arm/scripts/collect_jobset_logs_to_gcs.py \
  --jobset "<metadata.name from the manifest>" --source-sha "$SHA" \
  --gcs-prefix "<gs://your-bucket/your-prefix>" --output-dir "<local-evidence-dir>" \
  --namespace default --expected-workers 16
kubectl apply -f "$MANIFEST"
```

| Arm | Marker that must appear | Red flag |
|---|---|---|
| Standard | one per update: `[P57.TIM_STANDARD] PASS step=<n> rows=<n> groups=<…> old_logps=trainer tis_weights=absent rollout_logps=present trainer_rescore=training-input policy_version=matched`, re-checked over the whole log by `canon-zero-tim/workloads/frozenlake-three-arm/scripts/classify_standard_receipts.py` | any `[P57.TIM_PURITY]` line |
| Token-level TIS | exactly one `[P57.TIM_PURITY] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input` | any other count — `canon-zero-tim/cluster/steps/90_run.sh` greps it and exits non-zero |
| Zero-TIM | exactly one `[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only`, plus `[PATHTRACE]` lines in the log and, in W&B, `sampler_trainer/train/logp_diff_mean` == 0 and `canonical/train/alignment_max_differing_bytes` == 0 at every step | a clean exit with no `[PATHTRACE]` line: the overlay never installed |

Three runs need 192 chips; on 64 chips run them one after another.

Stop condition: once W&B shows the row `_step = 199` (the 200th observation) the Figure 4 data for
that arm is complete — let the job finish its 300 updates or delete the JobSet; step 5 takes only
`_step` 0–199 and refuses a run that has fewer.

## Step 5 — export the runs from W&B

```bash
export WANDB_API_KEY=<your key>
RUN=canon-zero-tim/results/frozenlake-short-horizon/<arm-dir>/<run-id>   # standard, tis, zero_tim
python3 canon-zero-tim/blog_reprod/export_wandb.py --project zero-tim-p57-frozenlake-tim \
  --run <run-id> --out $RUN                                       # the first three: once per arm
python3 canon-zero-tim/blog_reprod/cut_plotted_columns.py \
  $RUN/history.csv canon-zero-tim/blog_reprod/data/<arm>.csv
python3 canon-zero-tim/blog_reprod/compare_history.py \
  $RUN/history.csv <an archived export>/history.csv
python3 canon-zero-tim/blog_reprod/make_manifest.py --source-commit "$SHA" \
  --standard <run-id> --tis <run-id> --zero <run-id>               # this one: once, at the end
```

expected: `WANDB_EXPORT_PASS run=<entity>/zero-tim-p57-frozenlake-tim/<run-id> history_rows=…
config_keys=… summary_keys=… out=…`, then no output from the cut, then `COMPARE_HISTORY PASS
steps=…/… columns=…/… cells_compared=… differences=0` (that one re-checks an export you already
have; skip it for a fresh run). `<arm>` is `standard.csv`, `importance_sampling.csv` or
`zero_tim.csv`; add `--entity` (our runs live under `yuxzhang-google`). ≈1 min per run. The
exports live under `canon-zero-tim/results/`, one home for every workload's raw W&B exports —
`canon-zero-tim/results/README.md` says how to register a run there.

expected: `PASS: wrote data/manifest.json (source_commit <SHA>; 3 arms; 200 rows each)`. Each arm
needs at least 200 training observations — steps 0 to 199 — and a `data/<arm>.csv` cut from the
history it is listed with, or the manifest is refused. Step 6 pins the 13 shared recipe values
above and rejects a run configured differently, so a rerun has to use them.

## Step 6 — figure, spreadsheet, comparison

```bash
python3 canon-zero-tim/blog_reprod/check_blog_reprod_data.py
python3 canon-zero-tim/blog_reprod/build_end_to_end_results_measured.py \
  --data-dir canon-zero-tim/blog_reprod/data --output-dir canon-zero-tim/blog_reprod/figures
python3 canon-zero-tim/blog_reprod/make_spreadsheet.py
```

expected: `PASS: 9 sha256 hashes match data/manifest.json` — the manifest step 5 wrote — then
`PASS: 200 verified local observations per arm; source data unchanged; …/figure4.svg`, then
`PASS: wrote …/figure4.xlsx (… bytes) …`. Under a second each. On the archived data the endpoints are 0.6265625 → 62.7%,
0.66875 → 66.9% and 0.882421875 → 88.2%.

What counts as reproduced:

* Zero-TIM's logprob difference is exactly 0 at every step — not small.
* The ordering is Zero-TIM > TIS > Standard with a clear gap, and the Standard and TIS differences
  stay in the 1e-2 band (Standard 0.0144–0.0233, TIS 0.0049–0.0254).
* Do not compare step by step: the same seed is not the same trajectory.

Run ids, executed commits and file hashes: `canon-zero-tim/blog_reprod/data/manifest.json`.
