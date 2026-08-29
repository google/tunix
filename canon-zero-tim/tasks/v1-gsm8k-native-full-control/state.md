# GSM8K Native/mismatch full-control state

- Status: `OFFLINE COMPLETE / TARGET NOT RUN`
- Bound worktree: `/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`
- Bound source: `d4128940464054866d466a6cce5adf326f513caf`
- Active phase: `P3 — stock-runtime handoff complete`
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

The first renderer draft was rejected because it selected warning-only
canonical P33 training rather than stock Native training. The corrected
renderer/profile/runtime branch, real `00_env.sh` positive/negative gates,
stock-engine preflight, adjacent suites, and three pinned-image aggregate
gates pass. A fail-fast aggregate render produced four manifests and classified
three optimized Zero carriers plus one stock Native carrier.

No production manifest was rendered through a clean-worktree wrapper because
the implementation is not committed. No TPU/Kubernetes run, server dry-run,
target performance/convergence comparison, commit, push, or image publication
has occurred.
