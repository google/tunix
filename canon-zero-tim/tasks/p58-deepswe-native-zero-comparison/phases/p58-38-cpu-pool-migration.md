# P58.38 — canon head and dedicated sandbox-pool migration

Status: `CPU PASS / PVC AND SANDBOX TARGET GATES NOT RUN`

## Contract

Fresh P58 production and diagnostic renders use one exact infrastructure
pair:

```text
Pathways head: canon-cpu-pool
R2E sandboxes: deepswe-cpu-pool-2
Model claim: haoyugao-cpu-np-pvc (compatibility not yet proved)
```

The renderer, sandbox admission probe, capacity verifier, runtime selector,
and diagnostic preparation scripts reject legacy P58 pool values. Generic
P34/P46 fallbacks remain unchanged outside the P58 workload identity.

The 16Gi proxy/RM hard limits are removed. Historical requests and generous
limits are restored. This deliberately returns the head to Burstable QoS
while keeping its `very-high` PriorityClass; making every request equal to the
old ceiling would reserve roughly 700GB for one Pod and can prevent
scheduling.

## Target gates

1. A production-shaped sandbox Pod must be admitted through
   `multislice-queue`, lose its Kueue scheduling gate, run on an actual
   `deepswe-cpu-pool-2` node, and emit
   `P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only`.
2. The remote operator must confirm ResourceFlavor/quota/autoscaler capacity
   for 128 sandboxes: at least 256 requested CPU and 512Gi requested memory,
   plus overhead.
3. Read PVC/PV topology without mutation, then separately approve a bounded
   read-only mount probe on `canon-cpu-pool`. It must emit
   `P58_HEAD_PVC_PASS scope=canon-head-read-only-mount` after finding
   `Qwen3-4B-Instruct-2507` on `haoyugao-cpu-np-pvc`.
4. Preserve all probe evidence before separately approved exact-Pod cleanup.
5. Only after these infrastructure gates and the complete pinned-image gate
   pass may a fresh K30 JobSet be considered for separate launch approval.

## Local evidence

- P58 renderer: 40/40 PASS.
- Sandbox/PVC probe contracts: 8/8 PASS.
- Checked-VMA ABA renderer/verifier: 4/4 PASS.
- P57 renderer resource regression: 25/25 PASS.
- Pinned dependency-image renderer-to-`00_env.sh`-to-Python environment
  contract: 21/21 PASS.
- Python and Bash syntax: PASS.
- `git diff --check`: PASS.
- The bare-host environment attempt was inconclusive because `metrax` is
  absent; the same gate passed in the pinned dependency image.
- Construction-only K30, sandbox, and PVC YAMLs rendered successfully under
  `/tmp/p58-pool-review.dBHbaN/`. They are not launch artifacts because the
  source worktree is dirty and no matching image was published.

No Kubernetes resource, image, TPU, checkpoint, or optimizer state was
mutated during local validation. Publication does not promote either live
infrastructure gate or authorize K30.

## Rollback

Revert the single P58.38 delivery commit if rollback is required; do not reset
or clean the shared worktree. Historical `cpu-np` evidence remains immutable.
