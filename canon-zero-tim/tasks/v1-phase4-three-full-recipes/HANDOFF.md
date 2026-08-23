# V1 Phase4 three-full handoff

## Mission and current boundary

Prepare exactly three strict optimized Zero-TIM full-training recipes from one
approved immutable source: GSM8K Qwen3-1.7B DP16xTP4 for 200 updates,
FrozenLake P45 Qwen3-8B DP8xTP8 for 300 updates, and FrozenLake M15-main
Qwen3-8B DP8xTP8 for 300 updates. M15 is a production/scientific recipe, not a
canary. The original three-recipe stack is published in the operator history.
The current repair release is staged in
`/home/yuxuan/code_rl_repro/worktrees/p58_zero_hp_release3_0823`, branch
`local/p58-zero-hp-release3-0823`, on exact base
`ccbcf572dc903bb1cce12f897cbdb05aec94922a`. Its four functional repair CLs
plus one audit-only release CL are authorized for publication. Render only
from the exact 40-character operator-branch SHA read back after that push, and
require a clean worktree before rendering.

Do not push, rerun the pinned image, publish an image, apply a JobSet, or occupy
TPU resources without the separate user approval for that boundary. Never
launch through a pipe. Run IDs, campaign roots, and evidence directories are
first-use only; preserve every failed run.

The first published target logs are immutable failures rather than campaign
passes. GSM8K DP16 x TP4 stopped at the P59 outer-trainer/inner-engine mesh
boundary; FrozenLake DP8 x TP8 stopped earlier at signed P57 W&B admission.
The local P58.8 follow-up repairs both and passes forced TP4/TP8 plus extended
P58/V1 pinned-image gates, but no real target rerun has certified it. Publish
and read back that repair before rendering fresh run IDs.

## Resolved bundle

- All recipes: automatic P47a, continue-decode K=8, fixed-AR gather,
  DP-aware gathered logprobs, logprob step fusion, fixed LM head, resident
  trainer placement, batched report, and P59 rank-parallel backward.
- GSM8K only: batched evidence on; APC off.
- FrozenLake only: APC on and batched evidence off. B rescore remains an
  independent full recomputation with `reset_prefix_cache=True`.
- Explicitly off: batched reverse, fused tree ops, norm-matmul, sample-split,
  engine-logprob-readback, anchor overlap, and vanilla/non-Zero paths.
- FrozenLake held-out rollout-only eval runs at pre-update policy steps
  `0,50,...,250` and after training at 300. Runtime receipts map the first six
  to enclosing global timing rows `1,51,...,251`; final 300 maps to `none`.
  Eval never enters trainer forward, backward, or optimizer.

## Claim provenance and ceilings

- The published operator history supplies the P56 serving, P59/APC foundation,
  three-recipe integration, and P57 300-update signed in-process-eval setup.
  The current release adds the P59 TP4/TP8 and signed P57 W&B repairs; target
  certification remains pending after publication.
- P56 knives have one-host KEEP evidence. Their complete current profiles and
  DP8/DP16 target geometries have not run at target scale.
- P59 is accepted under ordinary-JAX FP64 gradient correctness: the oracle is
  rel-L2 `3.91e-16`, the frozen real-Qwen gradient gate records `1.582%`, and
  DP4 reverse measured 3.605x. Serial and parallel AdamW first-step deltas
  differ by rel-L2 `9.976%`; do not claim trajectory identity.
- APC passed Phase3 one-host G-A through G-D, including the dirty-page negative
  control and matched performance/XProf. G-E and both DP8xTP8 full targets are
  unverified. Strict A(APC)-B(full-reset)=0 bytes remains mandatory.
- Qwen3-8B TP8 fixed-head code/overlay construction and pinned-image
  construction gate are green. The DP8xTP8 target is pending; TP4
  certification does not transfer to TP8.
- The current supported bundle passed a Qwen3-1.7B DP4xTP1 one-host v5p proxy:
  3/3 optimizer transactions, 51/51 strict alignment PASS, 0 FAIL. This proxy
  excludes APC and fixed LM head because those registered geometries are not
  represented by 1.7B/TP1. It does not certify any 64-chip topology or
  performance.

## Admission commands

From the worktree root, run the host gates exactly:

```bash
bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
python3 -m unittest discover -s canon-zero-tim/tests/p59_backward -p 'test_*.py'
python3 -m unittest discover -s canon-zero-tim/tests/p3_prefix_cache -p 'test_*.py'
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
git diff --check
```

The approved pinned-image gate was executed against image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:

```bash
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh
```

Require the exact terminal marker:
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 perfetto_window=1 manifests=3`.
This is an exact-image admission receipt, not a signed raw-log artifact: the
stdout/stderr log was not durably preserved, so no raw-log path or SHA exists.

The separately approved one-host v5p integration proxy is frozen at evidence
root
`/mnt/disks/tunix-data/logp_probe_1host/p59_dp4_v1_v1hp_20260823_0824utc`.
Its terminal marker is
`[P59.DP4] GREEN kind=v1 zero_tim=51/51 fail=0`; all six entries in its
`SHA256SUMS` verify.

## Launch and postflight

Render only from the approved pushed 40-character SHA using `RUNBOOK.md`.
Require three manifest PASS receipts and freeze every YAML hash. With separate
approval for each apply, launch GSM8K full first and classify its complete
strict alignment, P59, timing, XProf, and Perfetto evidence. Only then apply
P45, followed by M15, from the same SHA. A GSM8K green does not certify APC,
TP8 fixed head, DP8xTP8, FrozenLake evaluation, or M15 workload geometry.

Any real `CANON_ALIGN` or `CANON_ALIGN_PRE verdict=FAIL` kills that recipe.
Missing horizon, receipts, trace, checkpoint, or artifacts is INCONCLUSIVE,
not PASS. Performance judgment comes from `[PERF]`; XProf/Perfetto provides
operation attribution and never overrides the bitwise gate.
