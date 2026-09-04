# Gemma 4 E4B-IT FrozenLake handoff

## Current status

P0 identity freeze is complete. P1's isolated, default-off stock-admission
carrier is implemented and passes host and pinned-image CPU gates. No TPU
target has run, so P1 is not yet certified. P2–P5 have not started.
The launch carrier now fails closed on the canonical three-part preflight:
resolved environment, contract sweep, and current-tree intent diff.

Authoritative campaign records live at:

```text
/home/yuxuan/code_rl_repro/tasks/zero_tim_gemma_frozenlake_eval/
```

Read its `goal.md`, `state.md`, `plan.md`, `HANDOFF.md`, current phase file,
and append-only `log.md` before acting. Also read this revision's repository
`AGENTS.md` and canonical branch/flag skills.

## P1 source identity

```text
worktree: /home/yuxuan/code_rl_repro/worktrees/gemma4_e4b_zero_tim_0904
branch:   local/gemma4-e4b-zero-tim-0904
base:     6842edae88b5692c7d4c6ae4ecadfc9e2bf1e411
image:    sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
topology: one host, DP1 x TP4, exactly four local TPU devices
```

The selected payload is the exact P0-frozen Gemma 4 E4B instruction-tuned
safetensors revision. Do not replace it with E4B base, the compatible Orbax
artifact, Gemma 3, or another tokenizer.

## Immediate action

The P1 carrier is committed locally on this named branch and the post-commit
seal is recorded in the external authoritative campaign directory. No push is
approved. Confirm the worktree is clean and use `git rev-parse HEAD` as the
exact source identity, then obtain separate approval for one P45 P1 target
admission. M15 needs a second launch approval after P45 passes. Run them
serially on the named v5p host with fresh labels and follow `RUNBOOK.md`
exactly.

P1 is rollout-only: no canonical Gemma adapter, alignment claim, backward,
optimizer commit, checkpoint, prefix cache, W&B run, Kubernetes launch, or
image publication. Its real trainer configuration is still required to
construct with `mini/train/logps=1/1/1`; the classifier rejects any other
shape. A failure, malformed receipt, or missing receipt stops P1; preserve the
entire evidence directory and repair the first red boundary.

Do not amend, create another commit, or push without separate explicit
approval.

## Required return

For each target return the immutable label/evidence directory, manifest and
post-manifest SHA, classifier verdict, exact source-tree SHA, image digest,
JAX/device/mesh receipt, checkpoint and tokenizer identities, dataset hashes,
all five HBM stages, optimizer-state size, 593/593 live-weight result, shape
ledger, rollout envelope, raw-log SHA, and every failure. Only both workload
admissions passing advances the campaign to P2.
