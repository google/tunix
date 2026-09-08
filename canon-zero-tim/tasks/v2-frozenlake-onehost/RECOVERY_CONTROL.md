# P45 recovered-baseline control

`p45-dp2tp2-r2-recovery-control-v1` is an additional offline contract for the
conservative-key baseline after withdrawal of an unadmitted P59 key change.
It is not an alternative way to admit the one-program optimization.

## Fixed scope

P45 Qwen3-8B, DP2xTP2, r2, certify capsule replay, backward-no-commit.
The standard classifier still runs with required registered anchors and
retains its complete verdict and reasons in `standard_classification`.

| Receipt field | Recovery control | Deferred optimization |
|---|---:|---:|
| enabled | 0 | 1 |
| layers | 36 | 36 |
| static_keys | 36 | 1 |
| mapped_programs | 36 | 1 |
| logical_calls_per_layer | 1 | 1 |
| checked_vma | 1 | 1 |
| host_transfers | 0 | 0 |

Exactly one full control tuple is required. The known one-program goal mismatch
is accounted for only when both the exact standard reason and the exact tuple
match. No prefix-based failure filtering, tolerance change or inferred mode.
Every other standard failure remains a control failure, including alignment,
capsule/model binding, producer bypass, VMA, finite/replica/rank checks, HBM,
zero-commit and registered norm checks. Missing artifacts remain INCONCLUSIVE.

Additional constraints are an exact full pre-registered source SHA, clean
source-diff hash, exactly one clean terminal, and exact serialized micro/update
norm bits (including signed zero). The output records evaluator/registry
hashes. Source selection must happen before launch, not after inspecting norms.

## Run after an authorized experiment has terminated

From the physical worktree, with the run root and expected source SHA already
recorded in the experiment pre-registration:

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /mnt/disks/tunix-data/venvs/train/bin/python \
  canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/classify_frozenlake_recovery_control.py \
  --root "$recovery_run_root" --docker-exit 0 \
  --anchor-registry canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/gradient_anchors.json \
  --expected-source-commit "$recovery_source_commit" \
  --output "$recovery_new_output"
```

Supply the actual docker exit code, not an assumed zero. The output must be a
new file outside the immutable input run. Existing output files are refused.
Do not edit the launcher to hide its original standard FAIL.

Exit 0 means only `RECOVERY_CONTROL_PASS`; other verdicts exit nonzero. The
one-program optimization remains `NOT_ADMITTED` / `DEFERRED`. Historical
reclassification is analysis, not a fresh-source hardware gate. A fresh control
pass closes only the recovered frozen-capsule baseline sub-gate. It does not
prove full-gradient leaf equality, fresh-rollout Zero-TIM, optimizer health,
convergence, a speedup, another geometry, Pathways or GKE readiness.

## Offline gate

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /mnt/disks/tunix-data/venvs/train/bin/python -m pytest -q -p no:cacheprovider \
  canon-zero-tim/tests/v2_frozenlake_onehost
```

The control tests call the real standard classifier. Negative fixtures cover
all seven tuple fields, missing/duplicate receipts, source/geometry/arm/stage,
norm changes and signed zero, capsule/producer/bypass, VMA, alignment, HBM,
finite/replica/rank/commit errors, incomplete artifacts, failed/duplicate
terminals and overwrite attempts. Valid controls retain standard FAIL; a
one-program candidate cannot select this control contract automatically.
