# Plan

## Outcome

Add a separate Qwen3-4B parity-debug lane without weakening the existing P34
Qwen3-32B production or P43 Qwen3-8B debug contracts. The 64-chip and 256-chip
variants must use the same model, prompt/generation workload, rollout limits,
GRPO algorithm, optimizer placement policy, stage ladder, durable trajectory
format, grouped solve metrics, and postflight semantics. Only physical topology,
DP-local partitioning, worker count, and DP-derived global carrier geometry may
differ.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P44.1 | Shared recipe and explicit topology allowlist | CPU contract test proves normalized 64/256 specs are identical | passed |
| P44.2 | Qwen3-4B TP8 profile and canonical engine shim | Model registry, tensor-shape contract, manifest, and exact-image gates pass | passed |
| P44.3 | Training-path integration and durable evidence parity | Both variants pass environment, CLI, topology, artifact, and stage negative controls | passed |
| P44.4 | One dual-topology renderer, classifiers, and operator runbook | All six bounded JobSets render and normalized manifests differ only by the allowlist | passed |
| P44.5 | Local release and adjacent regressions | P44 plus P43/P39/P34 and overlapping renderer gates pass from a clean diff | passed |
| P44.6 | Target promotion ladder | 64 and 256 each classify rollout-only, one-update, and three-update as PASS | pending |
| P44.7 | Repair the first target failure and preempt the next reviewed runtime faults | Pathways-style 64/256 host placement, single-conversation rollout batching, and 4-prompt x 4-generation logprob execution batching pass fail-closed tests and adjacent release gates | passed |
| P44.8 | Optional direct-attached v5p Qwen3-4B one-host smoke | Hardware/dependency inventory is recorded; when prerequisites exist, real rollout through backward-no-commit passes without changing P34/P39/P44 production contracts | prerequisite-blocked; not run |

## Decisions

- Confirmed: `Qwen/Qwen3-4B` is registered in Tunix with 36 layers, hidden size 2560, intermediate size 9728, 32 attention heads, and 8 KV heads; every TP-sensitive dimension is divisible by 8.
- Confirmed: Current P43 and P34 share the training implementation but are not functionally identical recipes.
- Decision: Use 4 prompts x 4 generations, response length 4096, 5 turns, and the rollout-only/one-update/three-update ladder on both topologies.
- Decision: Keep TP8 for communication-path fidelity even though a smaller TP may be faster for Qwen3-4B.
- Decision: Use device-resident optimizer state and alignment warning-only on both parity-debug variants; retain hard stops for non-finite values, invalid B/C structure, replica failure, and OOM.
- Decision: Treat DP4 versus DP16, local trajectory partitioning, worker count, physical placement, and DP-derived global M as allowed topology differences; do not claim bitwise or performance equivalence.
- Decision: Preserve P34, P39, and P43 defaults and evidence boundaries unchanged.
- Confirmed: `p44r02` reached 256 unique topology devices and failed only at host grouping; the identical one-device IFRT CPU diagnostic is present in earlier successful Pathways runs and is not sufficient evidence of incomplete TPU registration.
- Decision: On Pathways, derive host identity from `(slice_id, logical_task)` rather than the virtual devices' degenerate `process_index`; keep exact host cardinality and host-complete role checks hard-failing on both topologies.
- Decision: Selectively port main commits `38a6fbfc` and `7a15620d`; do not merge main or the workload-reference branches wholesale.
- Decision: A direct-attached Qwen3-4B DP1xTP4 smoke, if runnable, proves only one-host frontend/rollout/trajectory/trainer integration and cannot promote TP8, role separation, DP4/DP16, Pathways, 64/256-chip, or Qwen3-32B claims.
