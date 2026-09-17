# canon-zero-tim

Make the rollout engine, its re-scorer, and the training forward produce **bit-identical**
logprobs — then keep them that way while the model trains. This is the only document on the
branch; Part 1 is an end-to-end recipe for the three runs in Figure 4 of the blog post —
Standard, token-level TIS, Zero-TIM — on FrozenLake short-horizon with Qwen3-8B on 64 v5p chips.

```
A = engine decode logprobs      the behaviour policy the tokens were actually sampled from
B = engine prefill re-score     the same engine, same tokens, scored in one pass
C = differentiable training forward
goal: A = B = C, bitwise, over the full distribution
```

## 0. What this branch is

`yuxzhang/canon-zero-tim` is the tunix repository plus exactly one extra top-level directory.
Everything above `canon-zero-tim/` is tunix `main` with the Zero-TIM product changes applied in
place (`tunix/`, `examples/`, `tests/`). `canon-zero-tim/` is the delivery package: engine overlay,
cluster launcher, image recipe, Figure 4 data. Clone shallow — the full history is about 1.6 GB of
blobs, most of it retired evidence and console logs:

```bash
git clone --depth 1 --branch yuxzhang/canon-zero-tim https://github.com/google/tunix.git
```

### 0.1 Layout

```
<repo root>                 tunix main: .gemini .github .gitignore .pre-commit-config.yaml
├── Dockerfile LICENSE README.md build_docker.sh pyproject.toml readthedocs.yml requirements/
├── docs/ examples/ scripts/ tests/ tunix/     the Zero-TIM implementation lives here
└── canon-zero-tim/                            the only extra top-level directory
    ├── README.md  install.sh   this file; assembling the canonical engine overlay
    ├── MANIFEST.sha256  STOCK_MANIFEST.sha256   installed-overlay and upstream file SHAs
    ├── P57_STOCK_OBSERVER_MANIFEST.sha256  P58_STOCK_OBSERVER_MANIFEST.sha256
    ├── FLAGS.md             runtime flag registry; three package tests read it
    ├── image/               Dockerfile.frozenlake, build_frozenlake_image.sh,
    │                        requirements.frozenlake.lock.txt, verify_tpu_stack.py
    ├── blog_reprod/         Figure 4, self-contained: data/ runs/ figures/ figure4.xlsx,
    │                        both builders and their tests
    ├── cluster/             renderers (render_*.py), entrypoint.sh, two JobSet templates,
    │                        profiles/ (47 .env: workload × geometry × numerics) and
    │                        steps/ (00_env → 10_sync_repo → 30_install_canon →
    │                        40_overlay_engine → 90_run)
    ├── src/                 engine_shims/ (shim chain, model modules) + stock observers
    ├── patches/             tpu_inference/ (40 ordered diffs) + observer and r2egym patches
    ├── tasks/               only <task>/scripts/: runtime helpers 90_run.sh calls in the
    │                        pod, plus the launch wrappers used in §1.3
    ├── tests/               package tests; the run_cpu.sh suites are host-runnable
    └── clean_data/          DeepSWE task whitelists; renderers assert their SHA-256
```

### 0.2 Everything removed is in the archive

The branch was pruned on 2026-09-17. Everything removed — phase documents, evidence directories,
debug logs, W&B console exports, per-task runbooks and handoffs — is pinned twice at commit
678170d29b3e6a6bbceb131b42247f4abc731da0: the branch `yuxzhang/canon-zero-tim-archive-20260917` and
a tag of the same name. To read a file that used to be here:

```bash
git fetch origin yuxzhang/canon-zero-tim-archive-20260917
git show 678170d29b3e6a6bbceb131b42247f4abc731da0:wandb_exports/ablation_summary.md
```

### 0.3 The pods run the branch tip, not the commit you render

The training container in `canon-zero-tim/cluster/jobset-64chip.yaml` re-syncs itself, and
`canon-zero-tim/cluster/steps/10_sync_repo.sh` then refuses any HEAD other than
`CANON_EXPECT_COMMIT` — the `--source-commit` the renderer baked into the manifest:

```
git fetch -q origin yuxzhang/canon-zero-tim && git reset -q --hard FETCH_HEAD   # jobset-64chip.yaml
[sync] REFUSING: expected commit $CANON_EXPECT_COMMIT, got $HEAD_SHA            # 10_sync_repo.sh
```

So render against the tip you intend to run: the rendered SHA is an assertion, not a checkout. And
you cannot re-run a historical SHA with these templates — the pod fetches the tip, sees the
mismatch and refuses. That would mean editing the template's sync block; nobody here has.

### 0.4 Codenames

The code paths keep the campaign codenames; this file uses plain names in prose.

| Code name | Plain name |
|---|---|
| P45 | FrozenLake short-horizon: grid sides 2–9, ≤5 turns, generation limit 2048 |
| M15 | FrozenLake long-horizon: 15 turns, generation limit 8192 — not in Figure 4 |
| p57 / p58 | the three-arm FrozenLake comparison (this figure) / the DeepSWE comparison |
| p38 / p22x / p33 | decode-prefill alignment carrier / engine shim series / JobSet base |
| p64, p67, p74 / v1-hp | v1 phase-4 full-recipe waves / the high-performance numerics profile |
| stan / is / zero / mism | arms: Standard / token-level TIS / Zero-TIM / mismatch |
| r01, i45g, r10a, p1std … | wave ids; they appear in run names and JobSet names |

## 1. Reproduce blog Figure 4

### 1.1 The three runs

Figure 4 has two panels — sampler-versus-trainer logprob difference and training solve rate — for
three treatments of one FrozenLake short-horizon recipe.

| Arm (blog label) | W&B run id | Executed source | Key flags | Engine |
|---|---|---|---|---|
| Standard | `jff877lt_canon-p57-fl-stan-r01-567c96d5` | 567c96d5 | `--old_logps_source=trainer --sampler_is=none`, eval every 50 | stock |
| Token-level TIS | `8zjz4li7_canon-p57-fl-is-i45g-ccbcf572` | ccbcf572 | `--sampler_is=token` (threshold 2.0), eval every 50 | stock |
| Zero-TIM | `tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9` | 06a0fdb9 | `--high-performance --disable-eval`, exact TiTO | overlay |

A run name encodes arm, wave id and the first 8 characters of the executed commit; rendered JobSets
use the same pattern (`canon-p57-fl-stan-p1std-678170d2`). All three commits are ancestors of this
branch, so `git show 567c96d5:<path>` works in a full clone — but they are provenance only (§0.3).
The TIS threshold is not a flag: `examples/frozenlake/train_frozenlake_qwen3.py` sets it to 2.0.

**Shared recipe.** Qwen3-8B; 64 v5p chips as DP8×TP8; 32 prompts × 8 generations = 256
trajectories per update; GSPO-token loss with RLOO advantages; one optimizer iteration per fresh
batch; seed 42; learning rate 1e-6; clipping 0.003 / 0.005; temperature 0.7, top-p 1, top-k 0;
loss aggregation `sequence-mean-token-mean`; grid sides 2–9; ≤5 turns; prompt limit 4096;
generation limit 2048; 300 updates requested, first 200 training observations plotted. The
`recipe` tab of `canon-zero-tim/blog_reprod/figure4.xlsx` compares all 24 recorded config keys
across the arms and marks exactly three as differing: `old_logps_source`, `sampler_is`,
`eval_every_n_steps`.

**Observed wall clock**, from the archived ablation summary (§0.2): Standard finished 301 steps in
12.65 h at 136.9 s/step; Zero-TIM was still running at step 216 after 38.06 h at 609.7 s/step. The
TIS run is absent from that table, so its runtime is **not verified**. These are what those runs
did, not promises about yours.

### 1.2 The runtime image

One image serves all three arms, built 2026-07-27, 12.2 GB:

```
europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image:vllm-tpu0.25.0
```

That is our registry path — replace it with yours and pin it by digest, as the JobSet template's
header says: a floating tag is incompatible with a bitwise contract. The whole TPU stack follows
from one pin, asserted by `canon-zero-tim/image/verify_tpu_stack.py`:
vllm-tpu 0.25.0 → tpu_inference 0.25.0 → jax/jaxlib 0.10.2 → libtpu 0.0.42.1. Alongside: flax
0.12.4, qwix 0.1.8, torch 2.10.0, transformers 5.12.1, numpy 2.3.5, on a `python:3.12-slim` base
(vllm-tpu ships a cp312-only wheel). `canon-zero-tim/image/requirements.frozenlake.lock.txt` holds
275 pins and is a freeze of that image: `pip freeze --exclude-editable` in the image matches it
275/275, nothing only in the lock, nothing only in the image (verified 2026-09-17).

**Rebuild** from the repository root; the script reads `IMAGE_REPO` (default
`tunix_frozenlake_image`), `TAG`, `VLLM_TPU_VERSION` (default 0.25.0), `NO_LOCK=1` (re-resolve
instead of using the lock), `PROBE=1` (resolve only) and `PUSH=1`:

```bash
bash canon-zero-tim/image/build_frozenlake_image.sh
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0 --devices 4
```

The build asserts the versions in its final layer (L1), so a wrong stack fails the build; the same
check with `--devices` (L2) is the only one that catches a libtpu ABI mismatch, because the symbols
resolve when libtpu loads, and it needs real chips. Also diff a rebuilt image's
`pip freeze --exclude-editable` against the lock: expect 275/275. **Not pinned**, so two things can
drift: the `python:3.12-slim` base digest and the apt packages
(`build-essential git curl ca-certificates`) resolved on build day. A rebuild on 2026-09-17 from
this exact recipe (locked mode, 343 s) resolved all 275 pins and produced the **same package set**:
`pip freeze --exclude-editable` of the rebuilt image, of the original image and the lock file are one
and the same 275-line list. The image is not bit-identical, though: the base tag had moved
(`python:3.12-slim` = `python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`
on that day, Debian 13, Python 3.12.14, versus Python 3.12.13 in the original; 173 dpkg packages in
both, versions not compared), and the rebuilt image is 11.5 GB versus 12.2 GB. If you need the
interpreter patch level fixed too, pin the base by digest in the Dockerfile before building.

**What the image decides.** Only the Python environment — its own `/app` checkout is a July commit
from an unrelated branch and does not matter, because the pod resets the tree to the branch tip
(§0.3). It is not quite complete either: `canon-zero-tim/cluster/steps/30_install_canon.sh`
pip-installs `gymnasium==1.3.0` (plus `sentencepiece` and `tiktoken`, both in the lock) at pod
start if they are not importable, so nodes need to reach PyPI.

**Two engine paths.** Standard and TIS run the image's stock vllm-tpu with an observer — the
renderer writes `CANON_P57_INFERENCE_REGIME=stock-fast` into their manifests. Zero-TIM gets the
canonical overlay, installed in the pod by `30_install_canon.sh`, which calls

```bash
bash canon-zero-tim/install.sh "$OUT" --from-path "$SP" --model qwen8b_tp8
```

with `$SP` the container's `tpu_inference` directory: apply the 40 diffs in
`canon-zero-tim/patches/tpu_inference/`, lay down the shim chain from
`canon-zero-tim/src/engine_shims/`, verify every produced file against
`canon-zero-tim/MANIFEST.sha256`. Against the real image on a CPU host it ends:

```
[4/4] verifying against MANIFEST.sha256
      all 37 files match (qwen8b_tp8)
```

A SHA mismatch is fatal on purpose: the chain is loaded by absolute-path import, so a stale member
does not raise — it silently reverts the run to stock while every switch still reads "on".

### 1.3 Launch on 64 v5p chips

**On the rendering host:** a clean checkout at exactly the SHA you render (both wrappers refuse a
dirty tree or a different HEAD); a `python3` with PyYAML; `R2E_K8S_NAMESPACE=default` exported,
which the preflight in `canon-zero-tim/cluster/steps/00_env.sh` requires and no renderer writes.

**On the cluster** — our values, replace them: a GKE cluster running Pathways JobSets; 64 v5p chips
as 16 worker pods × 4 chips; the image above, pinned by digest; a GCS scratch prefix,
`gs://yuxzhang-tunix-models/tmp/canon-zero-tim` in the template; W&B credentials. All three arms
report to project `zero-tim-p57-frozenlake-tim`, set by
`canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env`, which the Zero-TIM profile
`canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env` sources.

**Render** from the repository root, with `SOURCE` the 40-character tip you will run and `OUT` a
fresh directory outside the worktree. Use fresh run ids: they go into the JobSet name, the W&B run
name and the manifest.

```bash
SOURCE="<the-40-character-branch-tip-you-will-run>"
OUT="<absolute-fresh-output-directory-outside-the-worktree>"

bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  standard "$SOURCE" "$OUT/standard" "<standard-p45-id>" "<unused-m15-id>" "<campaign>"
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT/is" "<is-p45-id>" "<unused-m15-id>" "<campaign>"
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  "$SOURCE" "$OUT/zero" "<zero-campaign>" "<zero-p45-id>" "<unused-m15-id>" \
  --token-continuity both-exact --token-continuity-debug-mode record-full \
  --train-geometry dp8-tp8-b256
```

Expected last line of each, in order:

```
P57_THREE_ARM_RENDER_PASS wave=standard output=<OUT>/standard
P57_THREE_ARM_RENDER_PASS wave=is output=<OUT>/is
V1_P67_FROZENLAKE_WAVE_READY manifests=2 source=<SOURCE> output=<OUT>/zero token_continuity=both-exact token_continuity_debug=record-full p45_length_sort=off launch=not-executed
```

Rendering launches nothing. Both wrappers also write long-horizon (M15) manifests — a different
workload, not part of Figure 4 — so apply exactly these three files and nothing else:

```
<OUT>/standard/p45/jobset-p57-frozenlake-standard-300.yaml
<OUT>/is/p45/jobset-p57-frozenlake-is-300.yaml
<OUT>/zero/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml
```

On commit 678170d29b3e6a6bbceb131b42247f4abc731da0 with run ids `p1std` / `p1is` / `p1z` the three
commands exited 0, and two separate renders produced the same Standard manifest
(`49443d060c0b1b68366beac344df5f697be91f1746ad2d362f338ad042f4748f`): these manifests carry no
timestamps. A different source SHA or run id does change them — both are written into the YAML.

Three simultaneous runs need 192 chips; on 64 they run one after another. Before applying, start
one persistent log collector per JobSet — restarted pods lose their logs:

```bash
python3 canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/collect_jobset_logs_to_gcs.py \
  --jobset "<metadata.name from the manifest>" --source-sha "$SOURCE" \
  --gcs-prefix "<gs://your-bucket/your-prefix>" --output-dir "<local-evidence-dir>" \
  --namespace default --expected-workers 16
kubectl apply -f "$OUT/standard/p45/jobset-p57-frozenlake-standard-300.yaml"
```

**Check the result, not just the launch.** A zero exit code is not an admission, and a complete
300-update run and a 200-observation plot are two different checks. Keep, per run: source, image
and config identities; raw worker logs; admission and optimizer receipts; the W&B exports. Then:

* Zero-TIM: `sampler_trainer/train/logp_diff_mean` and
  `canonical/train/alignment_max_differing_bytes` (both logged from
  `tunix/rl/agentic/agentic_rl_learner.py`) must be 0 at every step. A run with the switches on and
  no `[PATHTRACE]` lines did not install the chain, and the exit code will not tell you.
* Standard: every step must log `old_logps=trainer tis_weights=absent`; the postflight is
  `python3 canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_standard_receipts.py --run-log <log> --expected-updates 300 --output <json>`,
  which also rejects a legacy TIS receipt appearing in a Standard run.
* TIS must keep the registered token-level correction; Zero must report exact TiTO and its
  numerical profile. Check gradient and update finiteness separately from forward alignment:
  forward-bitwise and healthy-backward are independent failure modes.

Nothing past rendering was executed against a cluster while this file was written — **not verified** here.

### 1.4 Data → spreadsheet → figure

`canon-zero-tim/blog_reprod/` redraws Figure 4 from vendored data — no network, no cluster, no W&B
account: the three plotted CSVs and their manifest, the three W&B exports (`config.yaml`,
`history.csv`, `summary.json`; the 60 MB console logs are excluded), the two builders with their
tests, `figure4.xlsx`, and `canon-zero-tim/blog_reprod/figures/figure4.svg` with its `.png`. Run
from the repository root; the first command needs only the standard library, the second PyYAML, the
third openpyxl and PyYAML (it imports the figure builder, which imports `yaml` at module level):

```bash
python3 canon-zero-tim/blog_reprod/check_blog_reprod_data.py
python3 canon-zero-tim/blog_reprod/build_end_to_end_results_measured.py \
  --data-dir canon-zero-tim/blog_reprod/data \
  --output-dir canon-zero-tim/blog_reprod/figures
python3 canon-zero-tim/blog_reprod/make_spreadsheet.py
```

The first checks nine hashes — three plotted CSVs, six W&B export files — against the manifest:

```
09eb10ee12467c93007da3a221be5c55c52e0f56eb6e34f5b49e513700d70cec  data/standard.csv
e61d8ce7ab297069a2c8fe3a6025262f3efd560d14c2d59c56658bcefeff96e9  data/importance_sampling.csv
ea1af6c35612753b422f19fdaeb24361fb6d7c55d4cad39719bbba2bb06ff670  data/zero_tim.csv
PASS: 9 sha256 hashes match data/manifest.json (source_commit a7255cfc4b15a29ed9cabcd67a54cac416cd8b74)
```

The second prints `PASS: 200 verified local observations per arm; source data unchanged; …`, writes
the SVG and PNG, and labels these final ten-observation means:

```
standard             0.6265625     -> 62.7%
importance_sampling  0.66875       -> 66.9%
zero_tim             0.882421875   -> 88.2%
```

The SVG carries the same numbers as `data-arm` / `data-step` / `data-value` attributes, and the
checked-in SVG and PNG are byte-identical to the blog package's (PNG SHA-256
`4465b15ab936c2d812568fa806909b810cc2e5a3ada1082e1598d306fbc2cccc`). The PNG comes from
`gdk-pixbuf-thumbnailer`; without that binary the command fails after writing the SVG.

The third writes `figure4.xlsx` (67,133 bytes as generated) with six sheets: `README`;
`curves_wide` (201×10: display step plus, per arm, `solve_ratio`, its trailing ten-step mean and
`logp_diff_mean`); `curves_long` (1801×4); `recipe` (24 config keys × three arms plus a `differs`
column); `provenance` (run ids, W&B project, executed SHAs, the three CSV hashes, line ranges); and
`endpoints`. Raw curve cells hold the CSV's own number strings, so the workbook shows the digits
that were plotted. Regenerating it changes its bytes but not its contents — openpyxl stamps the
wall clock into `docProps/core.xml` and the zip member timestamps, so compare the worksheet XML,
not the file hash. `python3 -m unittest` passes on
`canon-zero-tim/blog_reprod/make_spreadsheet_test.py` (5 tests) and on the two builder tests (19).

### 1.5 Evidence boundary, and what "reproduced" means

* **The arms differ in more than one thing.** Three different source commits, evaluation
  schedules, TiTO settings and backward implementations: this compares three complete
  configurations, not one knob.
* **The same seed is not the same trajectory.** Seed 42 does not guarantee identical sampled
  trajectories, gradient bits or curves on a rerun; compare shapes, not step-by-step values.
* **The Zero-TIM telemetry is analysis-grade, not a certificate.** All 200 exported values of
  `sampler_trainer/train/logp_diff_mean`, `logp_diff_max` and
  `canonical/train/alignment_max_differing_bytes` are exactly `0` in
  `canon-zero-tim/blog_reprod/data/zero_tim.csv`, while Standard's mean difference stays in
  0.0144–0.0233 and TIS's in 0.0049–0.0254. But the raw receipts behind those zeros were sampled
  under warning-only admission and do not cover every step continuously; the manifest records
  `signed_full_run_certification: false`. A missing receipt is not a zero: strict Zero-TIM needs
  both decode-versus-independent-prefill and prefill-versus-trainer to show zero differing bytes on
  the sampled action mask.
* **The endpoints are what those runs did**, not targets: 62.7% / 66.9% / 88.2%. The Zero-TIM run
  had not reached 300 updates when it was exported.
* **Verified while assembling this branch**, on the pre-prune commit
  678170d29b3e6a6bbceb131b42247f4abc731da0 and again on the pruned tree: the three render wrappers
  (exit 0 with the PASS lines above; the manifests differ only where the source SHA is written
  into them), the overlay install inside the real image (`all 37 files match (qwen8b_tp8)`, log
  byte-identical across the prune), the nine data hashes, and the figure and spreadsheet rebuilds.
  The 19 host-runnable package test suites were run before and after the prune as the prune's
  own gate: every exit code and count is unchanged except the two suites that lost the deleted
  evidence-reading tests (p57_frozenlake_tim 265→263, v1_phase4 109→101, both still OK); two
  suites (`p33_workloads`, `p43_deepswe_debug`) were already failing before the prune for
  unrelated reasons and fail identically after it.
* **Not verified**: any TPU execution. No job was launched and no `--devices` ABI check was run
  (a rebuilt image was compared against the running one at the package level only, §1.2).
  Reproducing the curves end to end is still a
  first-launch exercise — and from a new tip, that first 64-chip launch is itself this branch's
  end-to-end gate (§0.3).

## 2. Part 2 — other recipes

Part 2 — other recipes (GSM8K 64-chip, FrozenLake long-horizon, DeepSWE-4B 128-chip): see the next revision.
