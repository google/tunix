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
| P44.8 | Optional direct-attached v5p Qwen3-4B one-host smoke | Hardware/dependency inventory is recorded; when prerequisites exist, real rollout through backward-no-commit passes without changing P34/P39/P44 production contracts | superseded by P44.11 after the full model/R2E prerequisites were located |
| P44.9 | Repair the r04 TP8-local SwiGLU feature geometry without changing the BF256 kernel | Model-pinned 4B/32B padding passes exact forward/VJP probes; 8B remains unpadded; unknown widths and missing runtime evidence fail closed | passed; implementation commit `1a058b46` published to the operator branch |
| P44.10 | Repair the r05 Mosaic matmul block geometry | Qwen3-4B uses BN/BK128 plus exact `1216->1280` K/N padding; target-shaped forward and canonical VJP pass on four real v5p devices; missing runtime traces fail closed | local and one-host v5p passed; implementation `29cea119` published; 64/256 target pending |
| P44.11 | Real one-host Qwen3-4B DeepSWE integration | A default-off DP1 x TP4 colocated profile loads the model, executes real Docker R2E rollouts, persists trajectories/solve metrics, runs backward without an optimizer commit, records HBM/placement/state fingerprints, and preserves all production contracts | rollout integration passed; backward executed but `INCONCLUSIVE_NO_SIGNAL`; implementation `29cea119` published; clean-source repeat pending |

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
- Confirmed: `p44r04` reached Qwen3-4B model execution and failed at the SwiGLU wrapper because TP8-local intermediate width `9728/8=1216` is not divisible by the unchanged kernel BF256. Qwen3-32B also has a latent non-divisible TP8-local width `25600/8=3200`; Qwen3-8B width `12288/4=3072` already satisfies BF256.
- Decision: Preserve the BF256 kernel and custom VJP. Admit zero-padding only through exact model-overlay mappings (`1216->1280` for 4B and `3200->3328` for 32B), slice back to the semantic width, and reject every unregistered non-BF256 width. The P44 classifier must observe the 4B runtime feature-padding trace before accepting a stage.
- Confirmed: `p44r05` proved the SwiGLU repair on all 36 layers, then Mosaic rejected the Qwen3-4B matmul `BN64/BK64` block specs because the trailing TPU sublane dimension must be 128-aligned. Merely changing BK is insufficient: gate/up have semantic `N=1216`, while down has semantic `K=1216`.
- Decision: Pin Qwen3-4B matmul BN/BK to 128, zero-pad only registered TP8-local K/N width `1216` to `1280`, slice output N back to its semantic width, and mirror K padding in the canonical VJP. Require both K-padding and N-padding PATHTRACE lines in target classification.
- Confirmed: A privileged immutable-image run on one direct-attached host exposed four TPU v5 devices and passed real Pallas forward plus promoted custom VJP at the r05 target M=4096 for all five unique Qwen3-4B local projection shapes, including both `2560x1216` and `1216x2560`; this is a kernel construction gate, not a rollout, Pathways, TP8, 64/256, or training-stage result.
- Confirmed: The same host also has a complete local Qwen3-4B-Instruct-2507 snapshot, a pinned importable R2E-Gym checkout, cached `R2E-Gym-V1`, a reviewed whitelist, and working Docker access. The earlier P44.8 prerequisite blocker is obsolete; preserve it only as historical inventory.
- Decision: Keep one-host mode separate, default-off, and mutually exclusive with P34/P39/P43/P44 production modes. It admits only Qwen3-4B DP1 x TP4 colocated, one prompt x two generations, response 512, two turns, real Docker R2E, prefix cache off, device-resident optimizer state, and rollout-only/backward-no-commit.
- Decision: A no-commit PASS requires a finite nonzero gradient and unchanged model/reference/optimizer/accumulator state with zero commits. A finite zero gradient caused by an all-zero reward/advantage batch is `INCONCLUSIVE_NO_SIGNAL`, never PASS and never grounds a one-update promotion.
- Confirmed: The actual one-host batch executed real R2E tool actions and the full trainer forward/backward call, but both trajectories hit the bounded context limit with reward zero. Peak HBM was about 35.92 GiB of 95.74 GiB per device and optimizer leaves were device-resident. This proves integration wiring and memory headroom only, not episode completion, learning signal, or target topology behavior.
