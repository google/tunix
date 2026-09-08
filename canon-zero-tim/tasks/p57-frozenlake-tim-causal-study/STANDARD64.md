# Native64 Standard — trainer-old / no-TIS operator handoff

## Scope / release boundary

Implementation branch `local/fl-standard64-0908`; originally tested on published
source `06a0fdb99897d3d50a2f6443e990ce753936f1ee`. The user approved this
three-CL commit/push on 2026-09-08. Publication preserves the newer evidence-only
base `069010dd2a6ecf7fdb000a2cec186c8b1fa94d4f`; the release audit binds
committed code to the original test manifest. Target: **NOT RUN**.
No TPU/Kubernetes launch is authorized by this publication.
See EVIDENCE.md for host/image receipts; they do not certify target backward,
convergence, or performance. This is not a relabel of historical Native.

| Wrapper wave / arm | Frozen loss denominator | Extra TIS |
|---|---|---|
| `standard` / `standard` (new) | trainer recomputed before updates | absent (`None`) |
| `native` / `mismatch` (existing Bypass) | recorded rollout probabilities | absent |
| `is` / `is` (existing) | trainer recomputed before updates | existing token correction, threshold 2 |

Standard passes `--old_logps_source=trainer --sampler_is=none`. Rollout A
remains available for observation; trainer-old is captured once in TrainExample
and frozen across optimizer iterations. GSPO-token's ratio/clipping, RLOO,
loss aggregation, backward and optimizer kernels are unchanged. No TIS tensor
is generated. Missing A/C, wrong shape, policy-version drift or conflicting
options fail closed. Never fall back to rollout or stop-gradient(current).

## Recycled recipe

Each run: Qwen3-8B, Native `stock-fast`, **64 chips / DP8×TP8**, 16 workers ×
4 chips, v5p `4x4x4`. Worker configuration is equal to existing Native after
normalizing only the JobSet-derived head address. Autoscale, exclusive topology,
node selector, queue and tolerations remain unchanged. No hand-edited YAML.

Common: 32 prompts × 8 generations = 256 trajectories; 300 updates; seed 42;
GSPO-token + RLOO; LR 1e-6; AdamW (0.9, 0.95), weight decay 0, beta 0;
epsilon 0.003/0.005; temperature 0.7, top-p 1, top-k 0; prompt limit 4096.
P45: 5 turns, response 2048. M15-main: 15 turns, response 8192.

Inherit Native's **legacy tokenization, not TiTO**, eval at policy steps
0/50/100/150/200/250/300 (100 held-out prompts × 8), and final-only checkpoint
300/latest1. Do not borrow Zero's no-eval/no-checkpoint or numerical flags.
P78 segmented actor-logps is explicitly off: the inherited source required
zero in Python admission but omitted it from the profile/offline classifier;
this change closes that off-contract gap, not an optimization rollout.

W&B project: `zero-tim-p57-frozenlake-tim`; groups: `p57-standard` and
`p57-standard-m15-main`. Label analyses “trainer-old/no-TIS.” No successful
historical Native run ID was supplied. This recycles published infrastructure,
not a proven exact historical run identity. Before reusing +TIS/Zero curves,
match source, model/checkpoint/tokenizer/data hashes, seed, batch/mesh, TiTO,
loss/backward/optimizer, eval and horizon. Different settings are confounders.

## Render, collect and launch after separate approval

Checkout the approved **published full SHA**, read that revision's AGENTS and
branch/flag skills, verify clean source and remote readback. Freeze the production
image digest and model/data identities against the intended comparison; a local
Docker image ID is not a registry manifest digest. Use fresh run IDs and an
output directory outside the clean source worktree. The existing wrapper now
accepts `standard` and rejects a dirty source or reused output root:

```bash
SOURCE="<approved-published-40-character-sha>"
OUT_STANDARD="<new-absolute-output-directory>"
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  standard "$SOURCE" "$OUT_STANDARD" \
  "<fresh-p45-run-id>" "<fresh-m15-run-id>" "<fresh-campaign-root>"
```

Require two `P57_THREE_ARM_MANIFEST_PASS`,
`P57_THREE_ARM_WAVE_PASS wave=standard manifests=2`,
`P57_THREE_ARM_RENDER_PASS wave=standard`, and both YAML SHA256 values.
Verification includes real `00_env.sh`, resolved geometry and stock admission.
Do not launch temporary test manifests from an uncommitted source.

Start one persistent collector per exact JobSet as described in RUNBOOK.md
“External worker-log collection”: 16 expected workers, separate fresh local
directory and protected GCS evidence prefix per run. After **specific launch
approval**, apply without a trailing pipeline:

```bash
kubectl apply -f "$OUT_STANDARD/p45/jobset-p57-frozenlake-standard-300.yaml"
kubectl apply -f "$OUT_STANDARD/m15/jobset-p57-frozenlake-standard-m15-main-300.yaml"
```

Two concurrent runs require 128 chips. No capacity check or target launch was
performed here. First real update in each full is the first target boundary;
this implementation does not add a mandatory short run.

## Required receipts / what to return

First update: existing stock runtime, weight-sync and observer receipts;
actual Standard batch receipts; a real optimizer commit and finite gradient/
update diagnostics. Each Standard receipt names step, rows, group IDs,
`old_logps=trainer tis_weights=absent rollout_logps=present`,
`trainer_rescore=training-input policy_version=matched`.

The new classifier accepts per-group or merged receipts; requires each update's
32 distinct prompt IDs, 8 trajectories per group; rejects missing/duplicate
coverage, wrong source/weights and legacy purity labels. `90_run.sh` runs it
automatically, in addition to existing P33 full/P57 in-process eval postflight:

```bash
python3 canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_standard_receipts.py \
  --run-log "<complete-single-attempt-head-log>" --expected-updates 300 \
  --output "<new-standard-source-classification.json>"
```

Require `P57_STANDARD_RECEIPTS_PASS`, 9,600 groups / 76,800 trajectories, all
300 updates, and complete full/eval/checkpoint receipts. Source coverage alone
does not prove optimizer success. Native finite mismatch follows the existing
warning observer; it is not Zero-TIM evidence. Nonfinite/missing/optimizer
failures cannot be downgraded. Return source/YAML/image identities, head+worker
logs and terminal status, raw artifact SHA256, W&B run IDs, first-commit receipts,
all classifications and unresolved reds. Export final classifications from pod
state; the worker-log collector does not promise to archive arbitrary state.

## Checks / rollback / cost

```bash
env JAX_PLATFORMS=cpu bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base 06a0fdb99897d3d50a2f6443e990ce753936f1ee
git diff --check
```

Fixed-image CPU entry: `canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`
with the approved immutable local image ID. Includes actual learner method
(mocked model logps), default regression, real env reload/Python admission,
stock runtime and overlay probes. Not a real Qwen TPU backward test.

Rollback: stop selecting `standard`; old wave names retain meaning. A targeted
source revert requires approval; preserve the independent P78-off repair if
still needed. Preserve all evidence. Cost: trainer rescore is required (often
already paid by the observer) plus small host receipts; no speedup claim.

Publication is split into an independent P78-off admission repair, the Standard
arm, and a docs/evidence ledger CL. The ledger does not change runtime programs.
Use the full published release SHA containing all three; never render from an
intermediate code CL alone. Verify remote readback and clean source before use.

| CL | Committed source | Scope / downside / rollback |
|---|---|---|
| Admission | `b7828eeff2c038ba02b29dcf6fc83f6291588f32` | 4 files, +26/-3; P78-off contract and exact-SHA historical compatibility. Compatibility covers only two immutable archives; preserve this repair when disabling Standard. |
| Standard | `48b7dc69ddd3eef93a3f304086da9523b79a5e32` | 21 files, +691/-32; opt-in denominator, source receipts, wiring, registry and gates. Intentionally differs from Bypass; costs trainer rescore. Deselect Standard or, with approval, revert this code CL independently. |
| Ledger | The commit containing this handoff and publication audit | Docs/evidence only; local raw logs are host-local, not signed target artifacts. Preserve the ledger and append corrections when withdrawing a claim. |

[Publication audit](evidence/standard64_local_0908/publication.json) verifies
22/22 program/gate files and 10/10 original raw logs against the unchanged test
manifest. All tracked code-CL blobs outside the 14 newly archived upstream log
files match pre-rebase source. Post-rebase P57 263/263, V1 108/108 and flags
439/439 PASS. No additional image/TPU execution was needed for this evidence-only
rebase; no target certification is inferred.
