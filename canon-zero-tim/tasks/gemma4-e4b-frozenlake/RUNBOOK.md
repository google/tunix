# Gemma 4 E4B-IT P1 one-host runbook

## Scope and authority

This runbook executes P1 stock admission only. Each invocation requires the
user's explicit approval for that specific TPU run. It does not authorize a
commit, push, image publication, Kubernetes work, or any P2–P5 action.

## Preconditions

1. Work in
   `/home/yuxuan/code_rl_repro/worktrees/gemma4_e4b_zero_tim_0904`.
2. Read the campaign and repository handoffs plus the canonical branch and
   flag skills.
3. Confirm the selected host is `t1v-n-4a77ebd0-w-0`, the pinned image ID is
   exact, the canonical preflight passes, and no non-system privileged
   container owns the TPU.
4. Confirm the named branch is clean and HEAD is the exact locally reviewed
   P1 commit. The runner refuses a dirty tree; an embedded dirty-tree manifest
   is review evidence only and is never a target-launch identity.
5. Supply the existing checkpoint access token through `HF_TOKEN`; never print
   it or persist it in evidence.
6. Choose a fresh label. Evidence roots are immutable and labels are never
   reused.

If another non-system privileged container or user workload owns the host,
do not preempt or kill it. Monitor that exact live container/process at short
intervals until it reaches an authoritative terminal state or disappears; a
single timeout or failed poll is not evidence that the TPU is free. Invoke the
runner only after the monitor observes release. The runner then repeats the
one-shot exclusion check immediately before creating the run.

The runner enforces these conditions again and hashes every tracked and
non-ignored untracked runtime-tree file before and after execution.
Before Docker can touch the TPU it also writes `launch_preflight.json` and
requires all three launch gates to pass: the resolved-environment truth table,
the positive/negative contract sweep, and an exact manifest intent-diff against
the current reconstructable source tree. A preflight failure consumes the
label, preserves its evidence directory, and does not launch the container.

## Approved execution form

Run only one workload at a time. There must be no shell pipeline after the
launcher.

```bash
cd /home/yuxuan/code_rl_repro/worktrees/gemma4_e4b_zero_tim_0904
bash canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/run_p1_onehost_stock_admission.sh \
  p45 <fresh-p45-label>
```

If and only if the P45 classifier returns PASS, obtain separate approval and
run:

```bash
cd /home/yuxuan/code_rl_repro/worktrees/gemma4_e4b_zero_tim_0904
bash canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/run_p1_onehost_stock_admission.sh \
  m15 <fresh-m15-label>
```

## Evidence layout

Each run writes only beneath:

```text
/mnt/disks/tunix-data/gemma4_e4b_p1/<label>/
```

Required files are `raw.log`, `manifest.json`, `post_manifest.json`,
`launch_preflight.json`, `classification.json`, `SHA256SUMS`, local metric
logs, and the driver header
embedded at the start of `raw.log`. Never delete a failed directory.

## PASS contract

`classification.json` must say PASS and the Docker exit code must be zero.
There must be exactly one runtime, snapshot, tokenizer, dataset, optimizer,
live-weight, shape, rollout, and launch-preflight receipt plus exactly five HBM
receipts.

The hard facts are:

- JAX `0.10.2`, one process, four TPU devices, and DP1xTP4 on both meshes;
- trainer batch geometry exactly `mini/train/logps=1/1/1`; this is required
  even though P1 performs no backward because the real `RLTrainingConfig`
  must construct successfully before rollout admission;
- exact P0 checkpoint, tokenizer/chat-template/token-probe, dataset, source
  tree, and image identities;
- 593 mapped trainer leaves equal 593 live serving leaves bit-for-bit;
- nonempty synchronously materialized optimizer state;
- at least 2 GiB free per device at learner-ready;
- exactly one bounded rollout with nonempty action tokens and a context within
  the workload cap;
- unchanged train/global step and zero backward/optimizer commits;
- byte-identical pre/post runtime manifests.

Any missing receipt, traceback, drift, OOM, timeout, tensor difference, or
nonzero training count is FAIL. Do not weaken a gate to obtain admission.

## Claim ceiling and next phase

P1 PASS proves only one-host stock-model admission and loaded tensor identity.
It does not prove A-B, B-C, backward correctness, optimizer behavior, Zero-TIM,
performance, or convergence. After both P45 and M15 pass, update the external
phase ledger and begin P2 serving-shim design; P2 implementation and any new
target run remain separately reviewed actions.
