# PeftTrainer v2 Migration and v1 Deprecation Plan

## 1. Document Status

- Status: Draft
- Goal: Safely migrate `tunix.sft.peft_trainer.PeftTrainer` from the current v1
  implementation to v2 and eventually remove v1.
- Scope: SFT, CLI, DPO/ORPO, distillation, RL, checkpointing, metrics,
  documentation, and examples.
- Out of scope: Model implementation refactors, training algorithm changes, and
  performance work unrelated to the trainer migration.

## 2. Executive Summary

Do not immediately change `tunix/__init__.py` so that `tunix.PeftTrainer` points
to v2. Most production code still depends on v1, and the RL, DPO, and
distillation trainers directly inherit from it. The safe migration order is:

1. Finalize the target API and promote v2 out of `experimental` into a stable
   module.
2. Complete the lifecycle, metrics, and checkpoint contracts that are still
   missing from v2.
3. Build a shared contract test suite that runs against both v1 and v2.
4. Migrate standalone SFT consumers first, then derived trainers, with RL last.
5. Emit deprecation warnings for at least one release.
6. Switch the default implementation to v2 while retaining an explicit legacy
   fallback for one release.
7. Remove v1 in the next breaking release.

The base engineering estimate is **33–56 engineer-days**. With an approximately
20% contingency for unknowns, plan for **40–67 engineer-days**. With two
engineers familiar with the codebase working in parallel, implementation and
validation should take approximately **5–8 calendar weeks**. The deprecation
bake still requires at least **1–2 release cycles** and cannot be shortened by
adding engineers.

## 3. Estimation Assumptions

- One engineer-day is one focused day by an engineer familiar with JAX/NNX and
  Tunix. It includes implementation, unit tests, PR revisions, and routine
  review feedback.
- Estimates include CPU tests and limited TPU validation, but exclude TPU queue
  time and release waiting periods.
- v2 has already passed an fp32 numerical correctness investigation; this plan
  does not re-estimate the underlying numerical analysis.
- The migration does not redesign RL orchestration or the checkpoint file
  format.
- The estimate assumes `prepare_weight_sync()` can reuse the current RL weight
  synchronization path. A new cross-host transfer implementation would add
  5–10 engineer-days.
- All estimates are ranges. Re-estimate the remaining work at the end of every
  phase as new information becomes available.

## 4. Current Dependency Inventory

### 4.1 Public Entry Points

- `tunix/__init__.py` exports `PeftTrainer` and `TrainingConfig` from
  `tunix.sft.peft_trainer`.
- The CLI, quickstart, metrics documentation, and SFT examples still show the
  v1 import path.

### 4.2 Components That Directly Inherit from v1

- `tunix/rl/trainer.py::Trainer`
- `tunix/sft/dpo/dpo_trainer.py::DPOTrainer`
- `tunix/distillation/distillation_trainer.py::DistillationTrainer`
- `RLTrainingConfig`, `DPOTrainingConfig`, and the distillation config/input
  types also inherit from v1 types.

### 4.3 v2 Contracts That Are Not Yet Complete

- `compile()` is still an empty implementation.
- `restore_checkpoint()` returns an empty dictionary.
- `prepare_weight_sync()` is still an empty implementation.
- After a single direct `train_step()` call, metrics remain in the deferred
  buffer, so the caller cannot immediately retrieve that step's loss from
  either `metrics_logger` or `get_metrics()`.
- v1 and v2 use different type annotations for `checkpointing_options`.

These gaps may not affect the current `train()` happy path, but they prevent v2
from being a dependable public trainer and a complete `AbstractTrainer`
implementation.

## 5. Target Module Layout

Use the following structure during the migration:

```text
tunix/sft/peft_trainer.py          # Public facade; eventually exports v2
tunix/sft/peft_trainer_v2.py       # Stable v2, moved out of experimental
tunix/sft/legacy_peft_trainer.py   # Temporary home for v1
tunix/sft/peft_types.py            # Optional shared Config/Input types
```

In the final state, keep only `peft_trainer.py` as the public entry point.
`peft_trainer_v2.py` and `legacy_peft_trainer.py` are transitional modules and
should not become permanent APIs.

## 6. Phased Plan and Work Estimates

| Phase | Primary outcome | Effort | Expected PRs | Parallelism |
|---|---|---:|---:|---|
| 0. API decisions and baseline | Compatibility scope, migration gates, and module layout agreed | 1–2 days | 1 | Low |
| 1. v2 production readiness | Stable API, metrics, checkpoint, and lifecycle gaps closed | 8–13 days | 3–5 | Medium |
| 2. Shared contract tests | One behavioral suite covers v1 and v2 | 5–8 days | 2–3 | Medium |
| 3. Standalone SFT migration | CLI, examples, docs, and SFT smoke tests use v2 | 3–5 days | 2–3 | High |
| 4. Derived trainer migration | Distillation, DPO/ORPO, and RL all use v2 | 12–21 days | 5–8 | Medium |
| 5. Default switch and deprecation | Public exports use v2 and v1 becomes legacy | 2–4 days | 2–3 | Low |
| 6. Remove v1 | Legacy implementation and compatibility code removed | 2–3 days | 1–2 | Low |
| **Total** |  | **33–56 days** | **16–25** |  |

### 6.1 Phase 0: API Decisions and Migration Baseline

Effort: **1–2 engineer-days**

Tasks:

- Confirm that the final public import path remains
  `tunix.sft.peft_trainer`.
- List the constructor parameters, config fields, hooks, and properties that
  must remain compatible.
- Decide whether both `train()` and the step-level APIs are stable public APIs.
- Define metrics flush, consumption, and close semantics.
- Define forward and rollback compatibility requirements for checkpoints.
- Record performance baselines: compilation count, peak HBM, and steady-state
  step time.
- Freeze v1: accept only critical correctness or security fixes, with no new
  features.

Deliverables:

- An API compatibility matrix.
- A migration gate checklist.
- An agreed module naming and release strategy.

Exit criteria:

- Every consumer knows which behaviors must remain compatible and which changes
  may be breaking.
- Later PRs do not repeatedly reopen the same API decisions.

### 6.2 Phase 1: v2 Production Readiness

Effort: **8–13 engineer-days**

Recommended breakdown:

| Subtask | Effort |
|---|---:|
| Promote v2 to a stable module and organize shared types | 1–2 days |
| Implement or redefine the `compile()` contract | 1–2 days |
| Implement explicit `restore_checkpoint()` and metadata round-tripping | 2–3 days |
| Implement or explicitly limit `prepare_weight_sync()` | 1–2 days |
| Fix first-step, final-step, consumption, and close semantics for metrics | 2–3 days |
| Unify checkpointing options types and compatibility | 1–2 days |
| Audit subclass hooks and repeated `train()` lifecycle behavior | 1–2 days |

Some subtasks can be combined or run in parallel, so the phase estimate is less
than the simple sum of all upper bounds.

Key design requirements:

- No public or abstract method may remain an undocumented `pass` or fake return
  value.
- `close()` must be safely repeatable and guarantee that the final metrics
  buffer is flushed.
- Whether `get_metrics()` consumes data must be documented and tested.
- v2 must be able to restore at least v1 checkpoints.
- v2 Config/Input types must not continue to depend on the legacy module for
  type identity.

Exit criteria:

- `TrainerWorker` can call v2 through the `AbstractTrainer` contract.
- Metrics and checkpoint semantics are defined for both the direct step APIs
  and the `train()` loop.
- No production-migration-blocking stubs remain.

### 6.3 Phase 2: Shared Contract Tests

Effort: **5–8 engineer-days**

Extract common v1/v2 tests into a trainer contract suite and run them against
both implementations through a factory or parameterization:

```python
@parameterized.named_parameters(
    ('v1', legacy_peft_trainer.PeftTrainer),
    ('v2', peft_trainer_v2.PeftTrainer),
)
```

Coverage:

- Standard SFT, LoRA, and full-parameter updates.
- One fused fp32 update compared with v1.
- Multiple fp32 micro-steps followed by one update compared with v1.
- Gradient accumulation cadence and denominator reset.
- Custom loss, auxiliary metrics, and weighted metrics.
- First/final metrics steps, `get_metrics()`, and `close()`.
- Checkpoint save/restore/resume and metadata.
- Restoring a v1 checkpoint with v2.
- Single-device and sharded models, including optimizer-state sharding.
- Hooks, profiler behavior, and trainer reuse.

Numerical gates:

- One fp32 weight update: per-leaf
  `max|v1-v2| / max|v1| <= 1e-6`.
- Require bitwise equality for fused versus split paths only if they are
  designed to execute exactly the same arithmetic.
- Do not require v1/v2 weights to remain close after multiple updates. Compare
  statistical outcomes such as loss and reward curves for long runs.

Exit criteria:

- The contract suite passes against both v1 and v2 on CPU.
- Critical distributed cases pass on TPU.
- The v1/v2 test suites no longer contain large duplicated implementations.

### 6.4 Phase 3: Migrate Standalone SFT Consumers

Effort: **3–5 engineer-days**

Recommended order:

1. `tunix/cli/peft_main.py`
2. SFT examples
3. Quickstart, metrics, and reliability documentation
4. Top-level SFT smoke tests

During the migration, consumers should select v2 explicitly rather than
changing the global top-level export too early:

```python
trainer_cls = peft_trainer_v2.PeftTrainer
```

Validation:

- CPU smoke test.
- At least one real LoRA TPU smoke run.
- Checkpoint resume.
- Peak HBM no higher than v1.
- Steady-state step-time regression within the team-approved threshold; use 5%
  as the default gate unless the team selects another threshold.

Exit criteria:

- The official SFT CLI and examples use v2 by default.
- Each migration PR can be rolled back quickly by restoring one explicit
  import.

### 6.5 Phase 4: Migrate Derived Trainers

Effort: **12–21 engineer-days**

#### Distillation

Effort: **2–3 days**

- Switch the Config, TrainingInput, and Trainer base classes.
- Validate teacher-output preparation timing and evaluation behavior.
- Validate strategy pre/post-processing and `close()`.
- Add one-update, metrics, and checkpoint smoke tests.

#### DPO/ORPO

Effort: **3–5 days**

- Switch the `DPOTrainingConfig` and `DPOTrainer` base classes.
- Verify that the reference model remains unchanged.
- Validate `_prepare_inputs`, loss auxiliary output, and evaluation hooks.
- Run small DPO and ORPO smoke tests.
- Validate both tokenized-input and raw-input paths.

#### RL Trainer / RLCluster

Effort: **7–13 days**

RL is the highest-risk consumer and should migrate last. Validate:

- The `is_managed_externally` lifecycle.
- Restored metadata and global step.
- Independent actor/critic optimizers and checkpoints.
- Sequence packing and caller-driven gradient accumulation.
- Repeated `train()` calls and model offload/reload.
- Custom RL metrics buffers and tqdm metrics.
- At least one smoke test each for GRPO, PPO, and agentic GRPO.
- Trainer close, cluster close, and final metrics flushing.

Exit criteria:

- No production class directly inherits from v1.
- SFT, distillation, DPO/ORPO, and GRPO/PPO all have end-to-end evidence under
  v2.
- RL reward and loss curves show no statistical regression.

### 6.6 Phase 5: Default Switch and Deprecation

Effort: **2–4 engineer-days**, plus at least **1–2 release cycles** of waiting.

#### Release N: Stable Opt-In v2

- Publish a stable v2 import path.
- Emit a user-visible `FutureWarning` from the v1 constructor with
  `stacklevel=2`.
- Point the warning to the migration guide and v2 import path.
- Use v2 throughout documentation, CLI, and examples.
- Add a CI rule that prevents new v1 imports.

#### Release N+1: v2 Becomes the Default

- Point `tunix.sft.peft_trainer.PeftTrainer` and `tunix.PeftTrainer` to v2.
- Allow v1 imports only through `legacy_peft_trainer`.
- Preserve old-checkpoint restore support.
- Publish the migration guide and breaking-change release note.

Rollback:

- Keep the default switch as a standalone, low-code-volume PR.
- If a severe issue appears after release, restore the facade export in a patch
  release without rolling back completed consumer migrations.

Exit criteria:

- A release using v2 by default remains stable for at least one release cycle.
- No unresolved v1 blocker remains.

### 6.7 Phase 6: Remove v1

Effort: **2–3 engineer-days**

- Delete `legacy_peft_trainer.py`.
- Delete backend switches, warnings, and v1-only tests.
- Delete compatibility aliases kept only for v1 type identity.
- Update the final state in the release notes and migration guide.
- Run the full CPU and TPU test suites.

Do not start this phase until every item in the Definition of Done is complete.

## 7. Parallel Execution Recommendation

Recommended split for two engineers:

- Engineer A: v2 lifecycle, metrics, checkpointing, and stable-module
  promotion.
- Engineer B: Contract tests, standalone SFT consumers, and documentation.
- After Phases 1 and 2:
  - Engineer A owns the RL migration.
  - Engineer B owns the distillation and DPO/ORPO migrations.

Critical serial dependency chain:

```text
API decisions
  -> v2 lifecycle/metrics/checkpoint readiness
  -> contract tests
  -> derived trainer migrations
  -> default export switch
  -> release bake
  -> v1 removal
```

## 8. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| v2 deferred-metrics semantics differ from consumer expectations | First or final step metrics may be lost | Finalize flush/get/close contracts in Phase 1 |
| Checkpoint options or state structure changes | Resume and rollback may fail | Add v1-to-v2 checkpoint compatibility tests |
| RL depends on private fields and hooks | Subclass behavior may change silently | Build an RL-specific compatibility matrix |
| Fused/split compilation graphs change | HBM or performance regression | Record HBM, compilation count, and step-time baselines |
| Maintaining two implementations causes drift | Fixes may land in only one version | Freeze v1 features and use shared contract tests |
| Top-level export changes too early | Many consumers fail simultaneously | Change the facade/export only in the final migration stage |
| Expected multi-step weight divergence is treated as a bug | A valid migration is blocked | Apply strict numerical gates only to a single update |
| `prepare_weight_sync` requirements exceed assumptions | RL schedule expands | Decide in Phase 0 whether the existing weight-sync path is reusable |

## 9. Definition of Done

All conditions below must be satisfied before deleting v1:

- Production v1 imports/inheritance are zero, excluding the legacy shim and
  compatibility tests.
- SFT, DPO/ORPO, distillation, and RL all use v2.
- v2 can restore v1 checkpoints and continue with the correct training step and
  optimizer state.
- CPU CI and critical TPU CI remain consistently stable.
- Neither the first nor final metrics step is lost.
- Fused-path peak HBM is no worse than the recorded baseline.
- Steady-state step time meets the team-approved performance gate.
- Official documentation, CLI, and examples no longer show v1.
- v2 has shipped as the default and remained stable for at least one release
  cycle.
- The migration guide, release note, and rollback instructions are complete.

## 10. Recommended First Three PRs

### PR 1: API Compatibility Matrix and Stable Module Skeleton

- Finalize the target API.
- Add the stable v2 module and shared-types structure.
- Do not migrate production consumers yet.

Effort: **1–2 days**.

### PR 2: Metrics and Lifecycle Contract

- Fix first-step and final-step metrics.
- Define `get_metrics()` consumption semantics.
- Make `close()` idempotent and guarantee flushing.
- Add direct-step and train-loop tests.

Effort: **2–3 days**.

### PR 3: Compile and Checkpoint Contract

- Implement `compile()`.
- Implement explicit restore and metadata round-tripping.
- Unify checkpointing options.
- Add a v1-checkpoint-to-v2 restore test.

Effort: **3–5 days**.

Re-estimate the distillation, DPO, and RL migrations after these three PRs are
complete.
