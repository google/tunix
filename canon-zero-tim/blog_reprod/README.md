# Figure 4 — FrozenLake short-horizon, three arms

Reproduce the three curves in Figure 4 of the blog post — Standard, token-level TIS and Zero-TIM, one
FrozenLake short-horizon recipe, Qwen3-8B on 64 v5p chips — from a clone of this branch. Steps 1–5
need the chips; steps 6–7 redraw the archived runs on any laptop, and are where to start if you only
want the figure back.

## The recipe

Three complete configurations of one recipe; everything not in this table is shared.

| | Standard | Token-level TIS | Zero-TIM |
|---|---|---|---|
| Model and chips | Qwen3-8B, 64 v5p as DP8×TP8 | same | same |
| Data and workload | FrozenLake, grid sides 2–9, ≤5 turns, prompt limit 4096, generation limit 2048; 32 prompts × 8 generations = 256 trajectories per update; 300 updates requested | same | same |
| Optimization | GSPO-token loss, RLOO advantages, one optimizer iteration per fresh batch, lr 1e-6, clip 0.003 / 0.005, seed 42, temperature 0.7, top-p 1, top-k 0 | same | same |
| Old-policy logprobs | trainer re-score (`old_logps_source: trainer`) | trainer re-score | the rollout logprobs themselves — identical to the trainer's by construction |
| Sampler correction | none (`sampler_is: none`) | token-level TIS, threshold 2.0, set in `examples/frozenlake/train_frozenlake_qwen3.py`, not by a flag | none |
| Token handling and evaluation | evaluation every 50 updates | evaluation every 50 updates | exact token-in-token-out both directions (`--token-continuity both-exact`, debug `record-full`); evaluation off (`eval_every_n_steps: 0`) |
| Engine path | stock vllm-tpu plus observer (`CANON_P57_INFERENCE_REGIME=stock-fast`) | same | canonical overlay: 40 patches plus the shim chain, verified against `canon-zero-tim/MANIFEST.sha256` |
| Render command | `canon-zero-tim/recipes/frozenlake-short-horizon/render_standard.sh` | `canon-zero-tim/recipes/frozenlake-short-horizon/render_tis.sh` | `canon-zero-tim/recipes/frozenlake-short-horizon/render_zero_tim.sh` |
| Reference endpoint (trailing-10 solve rate at display step 200) | 0.6265625 → 62.7% | 0.66875 → 66.9% | 0.882421875 → 88.2% |

## Step 1 — clone at the tip you will run

```bash
git clone --depth 1 --branch yuxzhang/canon-zero-tim https://github.com/google/tunix.git
cd tunix
git rev-parse HEAD
```

expected: `git rev-parse HEAD` prints the 40-character tip; pass that same SHA to every render below,
because the pods fetch the tip and refuse anything else (`canon-zero-tim/README.md` §0.3). ≈1 min.

## Step 2 — the runtime image

Pull our image or rebuild it; push yours where the cluster can read it and pin it by digest in
`canon-zero-tim/cluster/jobset-64chip.yaml`.

```bash
docker pull europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image:vllm-tpu0.25.0
bash canon-zero-tim/image/build_frozenlake_image.sh   # ... or rebuild from canon-zero-tim/image/
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0
```

expected: `stack OK (vllm-tpu 0.25.0)`, and 275/275 lines shared between
`canon-zero-tim/image/requirements.frozenlake.lock.txt` and `pip freeze --exclude-editable` in the
image. ≈10 min to pull, 343 s to rebuild (measured 2026-09-17), seconds to verify; on a host with
chips also pass `--devices 4`, the only check that catches a libtpu ABI mismatch.

## Step 3 — the rendering host

```bash
export R2E_K8S_NAMESPACE=default
python3 -c "import yaml; print(yaml.__version__)"
git status --porcelain --untracked-files=all
```

expected: `R2E_K8S_NAMESPACE` set (`canon-zero-tim/cluster/steps/00_env.sh` requires it and no
renderer writes it), PyYAML importable, **no output** from `git status` — the wrappers refuse a
dirty tree or a HEAD other than the SHA you pass. Seconds.

## Step 4 — render the three arms

One script per arm: tip SHA, a fresh output directory outside the checkout, and a run id that lands
in the JobSet name and the W&B run name — 1–12 lowercase letters, digits or hyphens, starting and
ending with a letter or digit (no underscores; the wrapper derives `<run-id>-m15` and
`<run-id>-campaign` from it and rejects anything else before rendering).

```bash
SHA="$(git rev-parse HEAD)"                              # the tip from step 1
OUT="<absolute-fresh-directory-outside-the-checkout>"
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_standard.sh "$SHA" "$OUT/standard" f4std
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_tis.sh "$SHA" "$OUT/tis" f4is
bash canon-zero-tim/recipes/frozenlake-short-horizon/render_zero_tim.sh "$SHA" "$OUT/zero" f4zero
```

expected: `P57_THREE_ARM_RENDER_PASS wave=standard output=<OUT>/standard` for the first two
(`wave=is` for the second), and for the third `V1_P67_FROZENLAKE_WAVE_READY manifests=2
source=<SHA> output=<OUT>/zero token_continuity=both-exact token_continuity_debug=record-full
p45_length_sort=off launch=not-executed`; then, as each script's last line,
`MANIFEST=<OUT>/standard/p45/jobset-p57-frozenlake-standard-300.yaml` — the one file to apply.
Seconds per arm, nothing is launched. Each wrapper also writes a long-horizon (M15) manifest beside
it: a different workload, not part of Figure 4. Rendering is deterministic — same SHA and run ids,
same bytes.

## Step 5 — launch one arm

Three runs need 192 chips; on 64 they go one after another. Start a persistent log collector first —
restarted pods lose their logs.

```bash
python3 canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/collect_jobset_logs_to_gcs.py \
  --jobset "<metadata.name from the manifest>" --source-sha "$SHA" \
  --gcs-prefix "<gs://your-bucket/your-prefix>" --output-dir "<local-evidence-dir>" \
  --namespace default --expected-workers 16
kubectl apply -f "$MANIFEST"
```

expected: `P57.TIM_STANDARD` — one receipt per update in the Standard arm, `[P57.TIM_STANDARD] PASS
step=<n> rows=<n> groups=<…> old_logps=trainer tis_weights=absent rollout_logps=present
trainer_rescore=training-input policy_version=matched`, re-checked over the whole log by
`canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_standard_receipts.py`; a
legacy `[P57.TIM_PURITY]` line in a Standard run is a failure.

expected: `P57.TIM_PURITY` — exactly one `… PASS sampler_is=token old_logps=trainer
tis_weights=present trainer_rescore=training-input` in TIS, exactly one `… PASS sampler_is=none
old_logps=rollout tis_weights=absent trainer_rescore=observer-only` in Zero-TIM;
`canon-zero-tim/cluster/steps/90_run.sh` greps both counts and exits non-zero on the wrong one. For
Zero-TIM also: `sampler_trainer/train/logp_diff_mean` and
`canonical/train/alignment_max_differing_bytes` exactly 0 at every step in W&B, and `[PATHTRACE]`
lines in the log — switches on, no `[PATHTRACE]`, clean exit means the overlay never installed.

Wall clock from the archive: Standard finished 301 steps in 12.65 h (136.9 s/step); Zero-TIM was
still at step 216 after 38.06 h (609.7 s/step); TIS's was not recorded. All three report to W&B
project `zero-tim-p57-frozenlake-tim`, set by `…/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env`.

## Step 6 — export the runs from W&B

```bash
export WANDB_API_KEY=<your key>
python3 canon-zero-tim/blog_reprod/export_wandb.py --project zero-tim-p57-frozenlake-tim \
  --run <run-id> --out canon-zero-tim/blog_reprod/runs/<run-id>
```

expected: `WANDB_EXPORT_PASS run=<entity>/zero-tim-p57-frozenlake-tim/<run-id> history_rows=…
config_keys=… summary_keys=… out=…`, and `history.csv`, `config.yaml`, `summary.json` side by side.
Add `--entity` (or `WANDB_ENTITY`) if the project is not under your default entity; the key is read
from the environment only. ≈1 min per run.

Then cut the five plotted columns out of each `history.csv` for steps 0–199 — the figure builder's
`--repo` mode cannot do it, because it re-exports from `wandb_exports/` at the archive commit and
needs a full clone of the pre-prune history, not `runs/`. From `canon-zero-tim/blog_reprod/`, once
per arm, into `data/standard.csv`, `data/importance_sampling.csv` and `data/zero_tim.csv`:

```bash
python3 - runs/<run-id>/history.csv data/<arm>.csv <<'PY'
import csv, sys
FIELDS = ["_step", "rewards/train/solve_ratio", "sampler_trainer/train/logp_diff_mean",
          "sampler_trainer/train/logp_diff_max", "canonical/train/alignment_max_differing_bytes"]
rows = [r for r in csv.DictReader(open(sys.argv[1], newline=""))
        if r["_step"] and 0 <= int(float(r["_step"])) < 200]
rows.sort(key=lambda r: int(float(r["_step"])))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
PY
```

expected: the three files this directory already ships, byte for byte, when the inputs are the
archived histories (verified 2026-09-17 by rebuilding all three and comparing SHA-256). A column an
arm never logged stays empty; regenerate `data/manifest.json` only if you re-derive its hashes.

## Step 7 — figure, spreadsheet, comparison

```bash
python3 canon-zero-tim/blog_reprod/check_blog_reprod_data.py
python3 canon-zero-tim/blog_reprod/build_end_to_end_results_measured.py \
  --data-dir canon-zero-tim/blog_reprod/data --output-dir canon-zero-tim/blog_reprod/figures
python3 canon-zero-tim/blog_reprod/make_spreadsheet.py
```

expected: `PASS: 9 sha256 hashes match data/manifest.json`, then `PASS: 200 verified local
observations per arm; source data unchanged; …/figure4.svg`, then `PASS: wrote …/figure4.xlsx
(531684 bytes) …`. Under a second each (stdlib only, then PyYAML, then openpyxl and PyYAML). On the
archived data the endpoints are 0.6265625 → 62.7%, 0.66875 → 66.9% and 0.882421875 → 88.2%, and the
rebuilt SVG and PNG are byte-identical to those in `canon-zero-tim/blog_reprod/figures/`.

A rerun will **not** land on those endpoints and does not have to. What counts as reproduced: the
ordering and size of the gap (Zero-TIM well clear of the other two, TIS between), a Zero-TIM logprob
difference that is exactly zero rather than small, and Standard's and TIS's differences staying in
the archive's 1e-2 band (Standard 0.0144–0.0233, TIS 0.0049–0.0254). To check a re-export against
the archive instead of eyeballing it:

```bash
python3 canon-zero-tim/blog_reprod/compare_history.py <new>/history.csv <archived>/history.csv
```

expected: `COMPARE_HISTORY PASS steps=…/… columns=…/… cells_compared=… differences=0`. It compares
parsed cells, not bytes — the vendored exports were written with CRLF by an older tool, this
exporter writes LF.

## What "reproduced" means

* **The arms differ in more than one thing.** Three source commits, evaluation schedules, TiTO
  settings and backward implementations: this compares three complete configurations, not one knob.
* **The same seed is not the same trajectory.** Seed 42 does not guarantee identical sampled
  trajectories, gradient bits or curves on a rerun; compare shapes, not step-by-step values.
* **The Zero-TIM telemetry is analysis-grade, not a certificate.** All 200 exported values of
  `sampler_trainer/train/logp_diff_mean`, `sampler_trainer/train/logp_diff_max` and
  `canonical/train/alignment_max_differing_bytes` are exactly `0` in
  `canon-zero-tim/blog_reprod/data/zero_tim.csv`, but those receipts were sampled under warning-only
  admission and do not cover every step — `canon-zero-tim/blog_reprod/data/manifest.json` records
  `signed_full_run_certification: false`, and a missing receipt is not a zero.
* **The endpoints are what those runs did**, not targets: 62.7% / 66.9% / 88.2%. The Zero-TIM run
  had not reached 300 updates when it was exported.
* **Forward alignment and healthy backward are separate checks.** A run can be bitwise-aligned in
  the forward pass and still be training badly; look at gradient norms and update finiteness too.

## Provenance

| Arm | W&B run | Executed source | History rows |
|---|---|---|---|
| Standard | `jff877lt_canon-p57-fl-stan-r01-567c96d5` | 567c96d5 | 301 |
| Token-level TIS | `8zjz4li7_canon-p57-fl-is-i45g-ccbcf572` | ccbcf572 | 301 |
| Zero-TIM | `tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9` | 06a0fdb9 | 216 |

The three executed commits are ancestors of this branch, so `git show 567c96d5:<path>` works in a
full clone — provenance, not something you can re-run (`canon-zero-tim/README.md` §0.3). Exports and
plotted CSVs come from archive commit a7255cfc4b15a29ed9cabcd67a54cac416cd8b74, pinned by nine
SHA-256 hashes in `canon-zero-tim/blog_reprod/data/manifest.json` — among them
`09eb10ee12467c93007da3a221be5c55c52e0f56eb6e34f5b49e513700d70cec` for `data/standard.csv`, and the
checked-in PNG `4465b15ab936c2d812568fa806909b810cc2e5a3ada1082e1598d306fbc2cccc`.

## Verified, and not

Verified 2026-09-17 on a host without TPUs: the three render wrappers exit 0 with the step-4 PASS
lines and produce manifests byte-identical to the campaign-named wrappers' for the same run ids; the
overlay installs inside the real image (`all 37 files match (qwen8b_tp8)`); the image rebuilds to the
same 275-package set; the nine data hashes match; figure and spreadsheet rebuild, the figure
byte-identically; the step-6 CSV recipe reproduces all three `data/*.csv` byte for byte; and
`canon-zero-tim/blog_reprod/export_wandb.py` passes its unit tests and exported a real W&B run.

Verified 2026-09-17 on one v5p-8 host (real image, real TPUs, this branch's tip), Zero-TIM only:
`canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/run_perf_v2_onehost.sh` with the
exact-token-continuity arm (`tito-on`, DP1×TP4, production dataset via `P57_PERF_V2_DATA_DIR`)
installed the overlay (`all 37 files match (qwen8b)`), ran three optimizer commits with finite
gradients (commit gradient norms 15.51 / 6.68 / 6.51) and held all 12 strict-alignment rows — 36
boundaries — at zero differing bytes, with a green semantic census; and
`canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/run_frozenlake_dp2tp2_onehost.sh p45 r3 … measure`
(DP2×TP2, the same trainer knobs as the 64-chip profile) reported `strict_exact: true` over 26
boundaries with finite gradient norms. Those carriers cap rollouts at two turns, so every trajectory
was single-turn and exact token continuity itself was **not exercised** there (`token_verdict:
UNEXERCISED`); the 64-chip recipe's five-turn rollouts are where it is exercised.

Not verified: **no cluster job was launched** — no 64-chip run of these manifests, no `--devices` ABI
check; the Standard and TIS arms have no one-host coverage (the host driver's stock-engine arms crash
in the backward pass under its `CANON_P66_P59_CHECK_VMA=1` profile — a pre-existing defect on record
since 2026-09-10, unrelated to the 64-chip path); and the three archived runs could not be
re-exported live, because the credentials on this host cannot see project
`zero-tim-p57-frozenlake-tim`, so comparing a live export against `canon-zero-tim/blog_reprod/runs/`
is **not verified** (the exporter was exercised against a different, reachable run).
