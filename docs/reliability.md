<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Reliability
## Checkpoint Support

Tunix provides robust checkpointing capabilities to save and resume training
progress, leveraging Orbax as the backend. This includes saving model parameters
(supporting full state or only LoRA parameters for PEFT) and optimizer state.

Checkpointing is managed by [`CheckpointManager`](https://github.com/google/tunix/tree/main/tunix/sft/checkpoint_manager.py?q=symbol:CheckpointManager)
and integrated into [`PeftTrainer`](https://github.com/google/tunix/tree/main/tunix/sft/peft_trainer.py?q=symbol:PeftTrainer).
SFT uses `PeftTrainer` directly, while RL uses [`rl.Trainer`](https://github.com/google/tunix/tree/main/tunix/rl/trainer.py?q=symbol:Trainer),
a subclass of `PeftTrainer`, inside of the [`RLLearner`](https://github.com/google/tunix/tree/main/tunix/rl/rl_learner.py?q=symbol:RLLearner).
Therefore, both SFT and RL share the same checkpointing mechanism. Checkpointing
and restarting are built-in features that require no special setup beyond
configuration. To enable checkpointing, users simply need to set
`checkpoint_root_directory` in `SFTConfig` or `RLConfig`; if this path is
provided, Tunix automatically saves checkpoints and resumes training from the
most recent one if interrupted, restoring model weights, optimizer state, and
training step count. By default, checkpointing is disabled if
`checkpoint_root_directory` is not specified. Users can further customize
checkpointing behavior via `checkpointing_options` in the config.

## Fault Tolerance

Tunix ensures fault tolerance primarily through its checkpointing mechanism,
allowing training to resume after interruptions such as machine restarts or
pre-emptions.

Additionally, to prevent out-of-memory (OOM) errors due to excessive HBM usage,
Tunix includes an [`InflightThrottler`](https://github.com/google/tunix/tree/main/tunix/sft/inflight_throttler.py?q=symbol:InflightThrottler). This mechanism limits the
number of TPU computations that can be scheduled concurrently, as configured by
`max_inflight_computations` in `TrainingConfig`, thus providing more stable
training runs on memory-constrained hardware.

## Determinism Guarantee

Tunix supports deterministic training runs through careful management of random
number generation and data handling:

*   **Model Initialization**: Models can be initialized with a specific random
    seed (`rng_seed` or `random_seed`) to ensure consistent initial weights
    across runs.
*   **Data Shuffling**: RL learners accept a `data_shuffle_seed` parameter,
    which ensures that dataset shuffling is deterministic.
*   **Dropout and Stochastic Layers**: JAX and Flax RNG handling ensures that
    stochastic operations can be made deterministic if RNGs are correctly seeded
    and managed.

By providing explicit seeds for these components, users can ensure
reproducibility of training experiments.
