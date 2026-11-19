<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Reliability
## Checkpoint Support

Tunix provides robust checkpointing capabilities to save and resume training
progress, leveraging Orbax as the backend. This includes saving model parameters
(supporting full state or only LoRA parameters for PEFT) and optimizer state.

Checkpointing is managed by [`CheckpointManager`](https://github.com/google/tunix/blob/main/tunix/sft/checkpoint_manager.py#L128)
and integrated into [`PeftTrainer`](https://github.com/google/tunix/blob/main/tunix/sft/peft_trainer.py#L284).
SFT uses `PeftTrainer` directly, while RL uses [`rl.Trainer`](https://github.com/google/tunix/blob/main/tunix/rl/trainer.py#L29),
a subclass of `PeftTrainer`, inside of the [`RLLearner`](https://github.com/google/tunix/blob/main/tunix/rl/rl_learner.py#L54).
Therefore, both SFT and RL share the same checkpointing mechanism. Checkpointing
and restarting are built-in features that require no special setup beyond
configuration. To enable checkpointing, users simply need to set
`checkpoint_root_directory` in `TrainingConfig` (for SFT) or `RLTrainingConfig`
(for RL); if this path is provided, Tunix automatically saves checkpoints and
resumes training from the most recent one if interrupted, restoring model
weights, optimizer state, and training step count. By default, checkpointing is
disabled if `checkpoint_root_directory` is not specified. Users can further
customize checkpointing behavior via `checkpointing_options` in the config.

Users customize background preservation behavior granularly using components
defined inside `checkpointing_options` (via `TunixCheckpointingOptions` in
Python):

*   **Save Decision Policies**: Dictates when to initiate a checkpoint based on
    defined steps or intervals. Supported configurations include
    `FixedIntervalPolicy` and `ContinuousCheckpointingPolicy`. The default is
    `ContinuousCheckpointingPolicy(minimum_interval_secs=180)` (saves every 180
    seconds). See Orbax v1 [`save_decision_policies.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/experimental/v1/_src/training/save_decision_policies.py)
    for the complete interface contracts.
*   **Preservation Policies**: Sets specifications regarding tracking
    checkpoints over bounded timelines (e.g., `LatestN`). The default is
    `LatestN(n=3)` (keeps the latest 3 checkpoints). See Orbax v1
    [`preservation_policies.py`](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/experimental/v1/_src/training/preservation_policies.py)
    for the complete interface contracts.
*   **Step Name Format**: Defines the representation of directory names for step
    checkpoints. The default is `ocp.path.step.standard_name_format()` (uses
    simple integer step names).
*   **Asynchronous Processing**: Manage asynchronous behavior by specifying:
    *   `enable_async_checkpointing`: Whether to use async checkpointing.
        Defaults to `True`. **It is recommended to keep this enabled** to
        prevent the main thread from blocking during training runs while
        checkpoints are written to storage.
    *   `async_options`: In Python, pass
        `ocp.options.AsyncOptions(timeout_secs=1200)` to
        `TunixCheckpointingOptions(async_options=...)` (or set
        `timeout_secs: 1200` directly under `checkpointing_options` in CLI YAML
        configs). Defaults to `1200` seconds.

## Fault Tolerance

Tunix ensures fault tolerance primarily through its checkpointing mechanism,
allowing training to resume after interruptions such as machine restarts or
pre-emptions. For long-running RL steps,
[`SubBatchCheckpointManager`](https://github.com/google/tunix/blob/main/tunix/rl/sub_batch_checkpoint.py#L147)
persists rollouts mid-step so interrupted global steps do not need to regenerate
completed sub-batches. In disaggregated distributed RL (`tunix.dist_rl`),
[`FaultToleranceManager`](https://github.com/google/tunix/blob/main/tunix/dist_rl/orchestrator/fault_tolerance_manager.py#L37)
coordinates worker health monitoring and recovery across rollout and trainer
nodes.

Additionally, to prevent out-of-memory (OOM) errors due to excessive HBM usage,
Tunix includes an [`InflightThrottler`](https://github.com/google/tunix/blob/main/tunix/sft/inflight_throttler.py#L21). This mechanism limits the
number of TPU computations that can be scheduled concurrently, as configured by
`max_inflight_computations` in `TrainingConfig`, thus providing more stable
training runs on memory-constrained hardware.

## Determinism Guarantee

Tunix supports deterministic training runs through careful management of random
number generation and data handling:

*   **Model Initialization**: Models can be initialized with a specific random
    seed (`rng_seed` or `random_seed`) to ensure consistent initial weights
    across runs.
*   **Data Shuffling**: Synchronous RL learners (`GRPOLearner` and `PPOLearner`)
    accept a `data_shuffle_seed` parameter, which ensures that dataset shuffling
    is deterministic (note that `AgenticGRPOLearner` / `AgenticRLLearner`
    streams prompts asynchronously and does not accept `data_shuffle_seed`).
*   **Dropout and Stochastic Layers**: JAX and Flax RNG handling ensures that
    stochastic operations can be made deterministic if RNGs are correctly seeded
    and managed.

By providing explicit seeds for these components, users can ensure
reproducibility of training experiments.
