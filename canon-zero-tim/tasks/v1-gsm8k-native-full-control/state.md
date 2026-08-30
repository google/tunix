# GSM8K Native/mismatch full-control state

- Status: `ATTEMPT 03 REPAIR IMPLEMENTED / PINNED-IMAGE PASS / TARGET NOT RUN`
- Bound worktree: `/home/yuxuan/code_rl_repro/worktrees/m15_apc_attempt17_review_0829`
- Repair base: `2af1197f4d0bb604d7c423f703251fc5187b4594`
- Active phase: `P4 — Explicit-mesh Splash input recovery`
- Objective: expose the registered P56 stock GSM8K trainer as a separately
  named Native/mismatch control for the V1-HP Zero full recipe,
  while retaining identical model, data, seed, command, DP16xTP4 geometry,
  200-step horizon, and W&B project/group.
- Claim ceiling: renderer and offline contract only. No Kubernetes/TPU run,
  convergence claim, or performance comparison is admitted until a separately
  approved target launch completes.
- Mutation policy: implementation, tests, and task records are in scope.
  Commit, push, image publication, Kubernetes/TPU launch, and other remote
  mutation require separate explicit approval.
- Canonical phase: `phases/v1-p1-native-full-control.md` (created after the
  initial task ledger, as required by the phase workflow).

## Arm contract

The control reuses the P33 full scientific command and restart policy, but
uses a dedicated untreated stock profile. Its numerical contract is:

```text
profile=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env
CANON_GSM8K_TRAIN=1
CANON_GSM8K_VANILLA=1
CANON_P32_WORKLOAD=absent
canonical engine overlay=absent
alignment observer=off
P59/V1/system optimization=absent
lm head=stock
mesh=DP16xTP4
max_steps=200
seed=42 (driver constant)
```

All Zero numerical selectors and evidence paths must be absent from the raw
manifest. The resolved profile keeps shared boolean gates at zero and the
entrypoint skips the canonical install/overlay path.
Both arms use W&B project `zero-tim-gsm8k-dp16-tp4` and group
`qwen3-1p7b-dp16-tp4`; distinct run names identify the arms.

## Current result

The stock carrier passed its offline construction gates, then three approved
target attempts advanced through three independent Explicit-mesh admission
errors. Attempt 03 reached the real learner Splash Attention path and failed
before model math because the kernel mask pytree was replicated while
`shard_map.in_specs` required TP/model-sharded leaves.

The local repair reshares only the Splash kernel's dynamic leaves to the
already-declared `manual_sharding_spec` on Explicit meshes. Auto meshes return
the historical kernel object unchanged. A forced eight-device CPU negative
reproduces the exact replicated-versus-model error with a real Splash leaf;
the repaired leaf passes the same `shard_map` and remains byte-identical. The
pinned production image passes 10 Native contracts, 9 Qwen sharding tests,
and 1 optimized-Zero renderer neighbor.

The repair is contained in the current source CL. No post-fix Kubernetes/TPU
target, optimizer commit, convergence/performance comparison, or image
publication has occurred. The claim remains `TARGET NOT RUN` until a fresh
Native Attempt-0 crosses Splash admission and a real optimizer commit.
