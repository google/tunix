# FrozenLake long-horizon — Standard, token-level TIS, Zero-TIM

Run the three arms of Figure 4 on the long-horizon FrozenLake recipe — fifteen turns instead of
five, generation limit 8192 instead of 2048 — Qwen3-8B on 64 v5p chips. Steps 1–4 need the chips;
steps 5–6 redraw the archived runs on any machine. Four runs are archived under
`canon-zero-tim/results/frozenlake-long-horizon/`; none reached the configured 300 updates, and
nothing in this tree has been launched from it.

## The shared recipe

Shared by all three arms: Qwen3-8B on 64 v5p chips as DP8×TP8 (16 workers × 4 chips); FrozenLake
with grid sides 5–12, at most 15 turns per episode, prompt limit 4096, generation limit 8192; 32
prompts × 8 generations = 256 trajectories per update; GSPO-token loss with RLOO advantages, one
optimizer iteration per fresh batch, learning rate 1e-6, clip 0.003 / 0.005; temperature 0.7, top-p
1, top-k 0; seed 42; 300 updates requested, the first 200 compared.

Those values come from the rendered manifests' own training command (`--env_max_steps=15
--max_response_length=8192`, against the short-horizon `5 / 2048`), which selects one registered
envelope: `Recipe("m15", 5, 12, 15, …)` in `examples/frozenlake/p57_workloads.py`.

## Where the arms differ

| | Standard | Token-level TIS | Zero-TIM |
|---|---|---|---|
| Old-policy logprobs | trainer re-score (`old_logps_source: trainer`) | trainer re-score | the rollout logprobs themselves, bit-identical to the trainer's |
| Sampler correction | none (`sampler_is: none`) | token-level TIS, threshold 2.0 | none |
| Render script | `canon-zero-tim/recipes/frozenlake-long-horizon/render_standard.sh` | `canon-zero-tim/recipes/frozenlake-long-horizon/render_tis.sh` | `canon-zero-tim/recipes/frozenlake-long-horizon/render_zero_tim.sh` |
| Reference trailing-10 solve rate, at the archived run's last logged step | 0.007421875 → 0.7% at step 182 | 0.112109375 → 11.2% at step 149 | 0.903125 → 90.3% at step 234 |

The reference row is the `trailing10_last` column of
`canon-zero-tim/results/frozenlake-long-horizon/summary.csv` — the smoothed value at each run's
**last** step, not a peak and not the same step for all three. The peaks were 51.95% (Standard,
`_step` 53), 48.44% (TIS, `_step` 117) and 99.22% (Zero-TIM, `_step` 217). Zero-TIM additionally
runs the canonical engine overlay with exact token-in-token-out and evaluation off; Standard and
TIS run the stock vllm-tpu engine with evaluation every 50 updates. A fourth arm, `mismatch`, is
archived beside the three: rollout logprobs as the old policy with no correction and no bitwise
guarantee — the control for what Zero-TIM's premise costs when it is not true. The plotter draws
it because it is in the directory; this page does not compare against it.

Before launching, have these five in hand: the branch tip SHA you cloned; the runtime image pinned
by digest in your registry; the JobSet namespace (`default`); a GCS scratch prefix; a W&B API key.

## Steps 1–2 — environment and image

**Step 1** — clone the branch shallow at its tip and check PyYAML, exactly as
`canon-zero-tim/blog_reprod/README.md` step 1 does: the pods re-fetch the branch tip and refuse any
other commit, so render at the tip you intend to run. **Step 2** — pull or rebuild the one runtime
image all arms share and verify its TPU stack, exactly as that page's step 2 does; the horizon
changes nothing about the image.

## Step 3 — render the three arms

```bash
SHA="$(git rev-parse HEAD)"
OUT="<an absolute, empty directory outside the checkout>"
bash canon-zero-tim/recipes/frozenlake-long-horizon/render_standard.sh "$SHA" "$OUT/standard" lh15std
bash canon-zero-tim/recipes/frozenlake-long-horizon/render_tis.sh "$SHA" "$OUT/tis" lh15is
bash canon-zero-tim/recipes/frozenlake-long-horizon/render_zero_tim.sh "$SHA" "$OUT/zero" lh15zero
```

A run id is 1–10 lowercase letters, digits or hyphens, starting and ending with a letter or digit:
the renderers accept 16, and each wrapper spends six on the derived `<run-id>-short`.

expected: `P57_THREE_ARM_RENDER_PASS wave=standard output=<OUT>/standard` from the first script, the
same line with `wave=is` from the second, and from the third `V1_P67_FROZENLAKE_WAVE_READY
manifests=2 source=<SHA> output=<OUT>/zero token_continuity=both-exact
token_continuity_debug=record-full p45_length_sort=off launch=not-executed`.

Each script ends with `MANIFEST=<path>`. Seconds per arm, nothing is launched, and rendering is
deterministic. Every wrapper renders the short-horizon manifest too, under its own directory — do
not apply it. Apply only these three:

* `<OUT>/standard/long-horizon/jobset-frozenlake-long-horizon-standard-300.yaml`
* `<OUT>/tis/long-horizon/jobset-frozenlake-long-horizon-is-300.yaml`
* `<OUT>/zero/frozenlake-long-horizon/jobset-frozenlake-long-horizon-zero-300.yaml`

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

The receipts do not depend on the horizon: `canon-zero-tim/cluster/steps/90_run.sh` keys the
contract on `CANON_P57_RUN_KIND` and `CANON_P57_TIM_ARM` only. Three runs need 192 chips; on 64
chips run them one after another.

Stop condition: once W&B shows the row `_step = 199` (the 200th observation) the comparison window
for that arm is complete — the same rule Figure 4 uses. Budget days, not hours: the median
`_timestamp` gap per update in the archived histories is 2833 s for Zero-TIM, 428 s for Standard
and 397 s for TIS, so Zero-TIM alone needs ≈6.3 days to reach `_step = 199` and ≈9.5 days for 300.

## Step 5 — export the runs from W&B

```bash
export WANDB_API_KEY=<your key>
RUN=canon-zero-tim/results/frozenlake-long-horizon/<arm-dir>/<run-id>   # standard, tis, zero_tim
python3 canon-zero-tim/blog_reprod/export_wandb.py --entity yuxzhang-google \
  --project zero-tim-p57-frozenlake-tim --run <run-id> --out $RUN
python3 canon-zero-tim/results/check_results.py
```

expected: `WANDB_EXPORT_PASS run=<entity>/zero-tim-p57-frozenlake-tim/<run-id> history_rows=…
config_keys=… summary_keys=… out=…` once per arm. Then append one `canon-zero-tim/results/RUNS.tsv`
row per run — columns in `canon-zero-tim/results/README.md` — and run the checker.

expected: `RESULTS_CHECK PASS runs=<n>`, `<n>` counting every run directory under
`canon-zero-tim/results/`; a directory with no row and a row with no directory are both failures.

## Step 6 — plot and judge

```bash
python3 canon-zero-tim/results/plot_curves.py --workload frozenlake-long-horizon
```

expected: `PLOT_CURVES PASS workload=frozenlake-long-horizon runs=4 updates=234
out=<repo>/canon-zero-tim/results/frozenlake-long-horizon` on the archived data, with `runs` and
`updates` growing as you add yours. It rewrites that directory's `curves.svg` and `summary.csv`.

What counts as reproduced:

* Zero-TIM's logprob difference is exactly 0 at every step — not small. On the archived run
  `summary.csv` reports `logp_diff_mean_max` = 0 over all 234 steps, against 0.0278 for Standard
  and 0.0325 for TIS.
* Over the window Zero-TIM stays far above both stock arms, corrected or not, with the archived
  numbers as the yardstick: 90.3% trailing-ten and a 99.2% peak against 0.7% / 51.9% (Standard)
  and 11.2% / 48.4% (TIS). Arms that do not separate this widely have not reproduced the result.
* Do not compare step by step: the same seed is not the same trajectory.

## Current data

Four archived runs, none complete against the 300 configured updates: Standard `4m9p2ylk` stopped
at 182 observations, TIS `oxkvtl0d` at 149, Zero-TIM `3osny0pb` at 234, mismatch `2i5f31c2` at 149.
Only Zero-TIM covers the whole first-200 window — Standard ends at `_step = 181`, TIS at
`_step = 148` — so the archived comparison is truncated on two of the three arms, and a fresh wave
is what would close it. Run ids, W&B projects, executed commits, step counts and grades are the
`frozenlake-long-horizon` rows of `canon-zero-tim/results/RUNS.tsv`.
