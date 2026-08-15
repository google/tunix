# P45.3b — long-run host-memory hardening

- Status: local implementation complete; target memory/performance gate pending

## Objective

Prevent a repeat of the p45r5 unobserved 200G host-memory death while
preserving the admitted DP8xTP8 resident optimizer, evaluation cadence,
alignment policy, and numerical program.

## Evidence boundary

P45r5 sustained 47 committed updates before Kubernetes reported the `jax-tpu`
container OOMKilled at its 200G limit. TPU HBM remained healthy. The archived
log is not a complete terminal/RSS record, so it does **not** establish which
host allocation class grew. Raising the limit is operational headroom, not a
root-cause fix.

The generic `RLLearner` evaluation queue is not used by P45. The agentic path
already collects all held-out examples and uses `_last_eval_train_step` to run
evaluation once per policy cadence. This phase freezes that behavior with a
test; it does not change generic evaluation semantics.

## Implementation contract

1. Only manifests produced by `render_p45_frozenlake.py` change the
   `jax-tpu` memory limit from the reviewed base value 200G to 350G. The shared
   base and adjacent P33/P38 manifests remain unchanged.
2. The P45 profile requires host-memory telemetry and a positive committed-step
   GC interval. Other workloads remain unaffected by default.
3. Telemetry reports cgroup current/peak/limit plus process RSS/HWM when the
   platform exposes them. Missing optional files are reported as `null`, not
   treated as zero.
4. Emit `[P45.HOST_MEMORY]` JSON after a held-out evaluation and after each
   completed optimizer update. At the update boundary, release the completed
   rollout/eval batch references before one bounded Python cyclic-GC pass.
5. Do not clear JAX compilation caches, change JIT boundaries, change the
   evaluation dataset/cadence, or add a memory-triggered optimizer fallback.
6. Enable only the already-wired P32 grouped report-window consolidation with
   `CANON_P28_BATCHED_REPORT=1`. Require its `p32_vag_reverse` timing marker.
   The non-grouped evidence and reverse-loop flags remain off until separately
   ported and verified for P32 grouped execution.

## Local gate

- pure helper tests cover cgroup v2 parsing, `memory.max`, `/proc` RSS/HWM,
  disabled mode, positive interval validation, and before/after GC reports;
- evaluation scheduling remains exactly once for one policy step;
- renderer tests require `jax-tpu` limit 350G in both P45 variants and prove
  the checked-in shared base remains 200G;
- P45 profile admission requires telemetry enabled and GC interval one;
- P45 profile admission requires grouped batched-report on and rejects tests
  that silently advertise the two unported grouped optimizations;
- existing checkpoint, TP8 overlay, workload, and alignment gates remain green.

## Target gate

Launch the eval carrier in `new` mode and require:

1. a baseline and at least one eval-complete memory record;
2. one post-GC record for every committed policy step;
3. step 10 creates the sole complete checkpoint and step 11 continues;
4. cgroup current/peak stay below the 350G limit and the post-GC series is
   archived through at least step 11;
5. an explicit `resume` launch restores step 10, syncs rollout weights, and
   commits step 11.
6. warm updates emit finite `p32_vag_reverse` `seconds`, `adjoint`, and
   `accumulate` timings for the live DP8 grouped path.

P45 is not declared memory-stable from a single step. A rising post-GC slope is
diagnostic evidence even if the larger limit prevents OOM; continue the soak
or isolate the growing allocation class before a 450-step durability claim.

## Rollback

Stop using the P45 renderer/profile or revert only this phase's renderer,
profile, learner telemetry, and tests. Checkpoint semantics and the historical
DP16xTP4 paths are independent.
