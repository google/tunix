# canon-zero-tim

Make the rollout engine, its re-scorer, and the training forward produce **bit-identical**
logprobs — then keep them that way while the model trains.

```
A = engine decode logprobs      the behaviour policy the tokens were actually sampled from
B = engine prefill re-score     the same engine, same tokens, scored in one pass
C = differentiable training forward
goal: A = B = C, bitwise, over the full distribution
```

This file is the general guide: what is here, how the pods get their code, which image they run,
what the codenames mean, which recipes are documented. It holds no step-by-step operations — every
documented recipe has its own page, starting with `canon-zero-tim/blog_reprod/README.md` (Figure 4).

## 0. What this branch is

`yuxzhang/canon-zero-tim` is the tunix repository plus exactly one extra top-level directory.
Everything above `canon-zero-tim/` is tunix `main` with the Zero-TIM product changes applied in
place (`tunix/`, `examples/`, `tests/`); `canon-zero-tim/` is the delivery package: engine overlay,
cluster launcher, image recipe, render entry points, Figure 4 data. Clone shallow — the history is
about 1.6 GB of blobs, most of it retired evidence and console logs:

```bash
git clone --depth 1 --branch yuxzhang/canon-zero-tim https://github.com/google/tunix.git
```

### 0.1 Layout

```
<repo root>                 tunix main, unchanged: Dockerfile LICENSE README.md .github
├── docs/ examples/ scripts/ tests/ tunix/     the Zero-TIM implementation lives here
└── canon-zero-tim/                            the only extra top-level directory
    ├── README.md  install.sh   this file; assembling the canonical engine overlay
    ├── MANIFEST.sha256  STOCK_MANIFEST.sha256  P57/P58_STOCK_OBSERVER_MANIFEST.sha256
    ├── FLAGS.md             runtime flag registry; three package tests read it
    ├── recipes/             plain-named render entry points, one directory per recipe
    ├── image/               Dockerfile.frozenlake, build_frozenlake_image.sh,
    │                        requirements.frozenlake.lock.txt, verify_tpu_stack.py
    ├── blog_reprod/         Figure 4, self-contained: README.md, data/ runs/ figures/,
    │                        figure4.xlsx, both builders, the W&B exporter, their tests
    ├── cluster/             renderers (render_*.py), entrypoint.sh, two JobSet templates,
    │                        profiles/ (47 .env) and steps/ (00_env → 10_sync_repo →
    │                        30_install_canon → 40_overlay_engine → 90_run)
    ├── src/                 engine_shims/ (shim chain, model modules) + stock observers
    ├── patches/             tpu_inference/ (40 ordered diffs) + observer and r2egym patches
    ├── workloads/           only <workload>/scripts/: pod-side helpers 90_run.sh calls,
    │                        plus the render wrappers recipes/ calls
    ├── tests/               package tests; the run_cpu.sh suites are host-runnable
    └── clean_data/          DeepSWE task whitelists; renderers assert their SHA-256
```

### 0.2 Everything removed is in the archive

The branch was pruned on 2026-09-17. Everything removed — phase documents, evidence directories,
debug logs, W&B console exports, runbooks and handoffs — is pinned twice at commit
678170d29b3e6a6bbceb131b42247f4abc731da0, by the branch and tag
`yuxzhang/canon-zero-tim-archive-20260917`. To read a file that used to be here:

```bash
git fetch origin yuxzhang/canon-zero-tim-archive-20260917
git show 678170d29b3e6a6bbceb131b42247f4abc731da0:wandb_exports/ablation_summary.md
```

### 0.3 The pods run the branch tip, not the commit you render

The training container in `canon-zero-tim/cluster/jobset-64chip.yaml` re-syncs itself, and
`canon-zero-tim/cluster/steps/10_sync_repo.sh` refuses any HEAD other than `CANON_EXPECT_COMMIT`,
the `--source-commit` the renderer baked into the manifest:

```
git fetch -q origin yuxzhang/canon-zero-tim && git reset -q --hard FETCH_HEAD   # jobset-64chip.yaml
[sync] REFUSING: expected commit $CANON_EXPECT_COMMIT, got $HEAD_SHA            # 10_sync_repo.sh
```

So render against the tip you intend to run: the rendered SHA is an assertion, not a checkout, and
a historical SHA cannot be re-run with these templates — the pod fetches the tip and refuses.

### 0.4 The runtime image

One image serves all three FrozenLake arms, built 2026-07-27, 12.2 GB:

```
europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image:vllm-tpu0.25.0
```

That is our registry path — replace it with yours and pin it by digest, as the JobSet template's
header says: a floating tag is incompatible with a bitwise contract. The whole TPU stack follows
from one pin, asserted by `canon-zero-tim/image/verify_tpu_stack.py`: vllm-tpu 0.25.0 →
tpu_inference 0.25.0 → jax/jaxlib 0.10.2 → libtpu 0.0.42.1, alongside flax 0.12.4, qwix 0.1.8,
torch 2.10.0, transformers 5.12.1 and numpy 2.3.5 on a `python:3.12-slim` base (vllm-tpu ships a
cp312-only wheel). `canon-zero-tim/image/requirements.frozenlake.lock.txt` holds 275 pins and is a
freeze of that image — `pip freeze --exclude-editable` matches it 275/275 (verified 2026-09-17):

```bash
bash canon-zero-tim/image/build_frozenlake_image.sh
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0
python3 canon-zero-tim/image/verify_tpu_stack.py --vllm-tpu 0.25.0 --devices 4
```

The build asserts the versions in its final layer, so a wrong stack fails the build; the same check
with `--devices` is the only one that catches a libtpu ABI mismatch, and it needs real chips.
**Not pinned**: the base image digest and the apt packages resolved on build day. A rebuild on
2026-09-17 (locked mode, 343 s) resolved all 275 pins into the **same package set** — rebuilt
freeze, original freeze and lock file are one 275-line list — but not the same image: the base tag
had moved to `python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`
(Debian 13, Python 3.12.14 against 3.12.13; 173 dpkg packages both sides, versions not compared) and
the rebuild is 11.5 GB. Pin the base by digest if you need the interpreter fixed too.

Two engine paths run on top of it. Standard and TIS use the image's stock vllm-tpu with an observer
(`CANON_P57_INFERENCE_REGIME=stock-fast`). Zero-TIM gets the canonical overlay, installed in the pod
by `canon-zero-tim/cluster/steps/30_install_canon.sh` calling `canon-zero-tim/install.sh`: the 40
diffs in `canon-zero-tim/patches/tpu_inference/`, the shim chain from
`canon-zero-tim/src/engine_shims/`, then every produced file verified against
`canon-zero-tim/MANIFEST.sha256`. That SHA check is fatal on purpose — the chain is loaded by
absolute-path import, so a stale member silently reverts the run to stock while every switch reads
"on".

### 0.5 Codenames

Directories, renderers and test suites are plain-named. The codenames survive only where renaming
them would change what a run produces: environment variables (`CANON_P57_*`), profile file names,
run ids, JobSet and W&B names, the helper scripts under `<workload>/scripts/`, and the shim and
patch file names under `src/` and `patches/`. Read the table below when an old name turns up in a
log line, a manifest, an archived run or the archive branch.

| Historical code name | Plain name |
|---|---|
| P45 | FrozenLake short-horizon: grid sides 2–9, ≤5 turns, generation limit 2048 |
| M15 | FrozenLake long-horizon: 15 turns, generation limit 8192 |
| p57 / p58 | the three-arm FrozenLake comparison / the DeepSWE comparison |
| p38 / p22x / p33 | decode-prefill alignment carrier / engine shim series / JobSet base |
| p64, p67, p74 / v1-hp | v1 phase-4 full-recipe waves / the high-performance numerics profile |
| stan / is / zero / mism | arms: Standard / token-level TIS / Zero-TIM / mismatch |
| r01, i45g, r10a, p9std … | wave ids; they appear in run names and JobSet names |

## 1. Recipes

| Recipe | Documented in | Status |
|---|---|---|
| Figure 4 — FrozenLake short-horizon, three arms (Qwen3-8B, 64 v5p) | `canon-zero-tim/blog_reprod/README.md` | archived runs and vendored data; reproduction verified as far as rendering — nothing was launched |
| GSM8K 64-chip Zero-TIM | not yet documented | the archived run crashed at step 64 (prefill re-score context overrun); entry point `canon-zero-tim/workloads/full-recipes/scripts/prepare_gsm8k_full_dp16tp4_p74.sh` |
| FrozenLake long-horizon, Standard / TIS / Zero-TIM | not yet documented | archived runs incomplete: Standard crashed at 181/300, TIS stopped at 148/300, Zero-TIM reached 52–59 steps at ≈2550 s/step. The Figure 4 wrappers already render the long-horizon manifests next to the short-horizon ones |
| DeepSWE-4B 128-chip Zero-TIM | not yet documented | never launched, render pending; entry point `canon-zero-tim/cluster/render_deepswe_comparison.py` |
| DeepSWE-4B 128-chip Standard / TIS | not yet documented | the renderer has no such arm — `canon-zero-tim/cluster/render_deepswe_comparison.py` line 52 reads `_ARMS = ("native", "zero")`. Adding them is a new feature, not documentation |

A documented recipe keeps its entry points in `canon-zero-tim/recipes/<recipe>/`, one script per
arm, each taking `<tip-sha40> <out-dir> <run-id>` and printing the JobSet path to apply. They are
deliberately thin: they translate those three arguments into the six the wave wrappers under
`canon-zero-tim/workloads/` expect, and change nothing about what gets rendered.

## 2. What has been verified

Everything below was checked on this branch's current tip; a row that says "not verified" says why.

| What | Status |
|---|---|
| Rendering the three arms | Verified 2026-09-17: the three `canon-zero-tim/recipes/frozenlake-short-horizon/` entry points exit 0 with their PASS lines, and their P45 manifests are byte-identical to the ones the older wave wrappers render for the same run ids. |
| Pod install chain, `00_env` → `50_verify_overlay` | Verified 2026-09-18 on CPU: the real runtime image replayed `canon-zero-tim/cluster/entrypoint.sh` with the env taken from the three rendered manifests — `probe-only` and `install-only` exit 0 for all three arms, Zero-TIM reaching `all 37 files match (qwen8b_tp8)` and the stock arms `canonical_overlay=skipped observer_overlay=installed`. Steps `60_wait_workers`, `65_probe_devices`, `80_model_init` and `90_run.sh` only run on a cluster and are not covered. |
| Image rebuild | Verified 2026-09-17: a locked rebuild took 343 s and resolved all 275 pins into the same package set as `canon-zero-tim/image/requirements.frozenlake.lock.txt`. The base image digest and apt packages are not pinned, and the `--devices` ABI check needs chips and was not run. |
| Zero-TIM training and A = B = C on one host | Verified 2026-09-17 on one v5p-8 (real image, real chips): the overlay installed 37/37, three optimizer commits had finite gradient norms (15.51 / 6.68 / 6.51), 12 strict-alignment rows — 36 boundaries — held at zero differing bytes with a green semantic census; a second carrier at DP2×TP2 reported `strict_exact: true` over 26 boundaries with finite norms. |
| Multi-turn exact token continuity on one host | Not verified on one host: the carrier's frozen geometry contracts (`examples/frozenlake/train_frozenlake_qwen3.py`, P28 G6 and P27) admit only a 64-token whole-episode response budget, so every one-host trajectory is single-turn and token continuity is never compared (`token_verdict: UNEXERCISED` in both 2026-09-18 runs); the 64-chip recipe (prompt 4096, response 2048, five turns) is where it is exercised |
| Standard and TIS training | Not verified: the stock-engine arms crash in the backward pass under the one-host profile's `CANON_P66_P59_CHECK_VMA=1` — a defect on record since 2026-09-10, in the image's own attention kernel and unrelated to the 64-chip path — and the one-host classifier accepts neither arm, so no Standard or TIS update was reached. |
| 64-chip launch | Not verified: no cluster job has been launched from this tree. Whether the API server accepts these JobSets, and everything from `60_wait_workers` on, is untested. |
| Data → figure → spreadsheet | Verified 2026-09-17: the nine hashes in `canon-zero-tim/blog_reprod/data/manifest.json` match, the figure rebuilds byte-identically to the checked-in SVG and PNG, the spreadsheet rebuilds, and cutting the plotted columns out of the archived W&B exports reproduces all three `canon-zero-tim/blog_reprod/data/` CSVs byte for byte. Re-exporting those runs live is not verified — the credentials on the assembling host cannot see the W&B project. |

This branch carries analysis-grade telemetry and a reproducible data-to-figure path, not a signed certification. The Zero-TIM run's exported `sampler_trainer/train/logp_diff_mean`, `sampler_trainer/train/logp_diff_max` and `canonical/train/alignment_max_differing_bytes` are exactly zero at every plotted step, but the receipts behind those zeros were sampled under warning-only admission and do not cover every step continuously, and `canon-zero-tim/blog_reprod/data/manifest.json` records `signed_full_run_certification: false`: a missing receipt is not a zero.

The arms differ in more than one thing — source commit, evaluation schedule, TiTO setting, backward implementation — so they compare three configurations, not one knob, and seed 42 does not make a rerun trajectory-identical.
