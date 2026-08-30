# GSM8K Native/mismatch full-control state

- Status: `ATTEMPT 05 REPAIR IMPLEMENTED / PINNED-IMAGE PASS / TARGET NOT RERUN`
- Bound worktree: `/home/yuxuan/code_rl_repro/worktrees/p58_q4_systemopt_0830`
- Repair base: `98d102eb27fe05fcee327688d0aa6d236b32be4a`
- Active phase: `P5 — Auto/Manual output-sharding legality`
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

Five approved targets successively crossed the earlier admissions. Attempt 04
proved the Splash repair and exposed the Explicit-axis projection contraction;
source then named its output placement. Attempt 05 switched only Native to an
Auto-axis mesh, completed rollout at 5,668.9 tokens/s, and failed in the first
trainer embedder gather because `_activation_out_sharding()` still supplied a
`NamedSharding` containing Auto axes to `.get(out_sharding=...)`, which JAX
forbids.

The local repair returns an output sharding only when every axis named by the
requested spec is Explicit. Auto or Manual named axes return `None`, preserving
compiler inference; Explicit axes retain the Attempt-04 projection/gather
repair. Unknown axis names remain fail-closed. A forced eight-device pinned
image executes 13/13 Qwen sharding tests, including both Attempt-04 projection
tests and two Auto/Manual controls, plus 12/12 Native contracts and one Zero
neighbor.

The first pinned-image invocation revealed that `absltest.main()` preceded the
Attempt-04 test class, so only 11/13 tests executed. The entrypoint now sits at
EOF; the authoritative rerun executes all 13 and ends
`V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS ... auto_out_sharding=2 ...`.
No repaired Kubernetes/TPU target or optimizer commit has run.
