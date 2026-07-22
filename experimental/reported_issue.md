# DeepSWE Experiment Launch Issues & Observations (2026-07-22)

## 1. Missing Checkpoint & Resumption Logic (train_deepswe_nb.py)
**Severity**: High
**Symptom**: The main training wrapper silently fails to save or resume checkpoints, running completely statelessly from scratch every time it starts.
**Root Cause**: 
Although the script successfully reads the external `--ckpt_dir` and initializes `ocp.CheckpointManagerOptions()` options on line 672, they are explicitly ignored when instantiating the training engine. 
At line 763 onwards, `RLTrainingConfig` has its arguments hardcoded to `None`:
```python
    training_config=rl_cluster_lib.RLTrainingConfig(
        # ...
        checkpoint_root_directory=None,       # Should be CKPT_DIR
        checkpointing_options=None,           # Should be checkpointing_options
    )
```
**Consequence**: If the cluster goes down due to OOM or is preempted, all TPU states/metrics are lost, and progress resets.

---

## 2. Token Generation Throughput vs Sandbox Timeout Imbalance
**Severity**: Medium-High
**Symptom**: Trajectories will inevitably hit a hard killed timeout before reaching `--max_turns=50`.
**Setup & Hyperparameters**:
- Model: `Qwen3-32B`
- Topology: `64 TPUs (FSDP=8, TP=8)`
- `MAX_CONCURRENCY=128` (Parallel sandboxes)
- Python Episode Timeout: `10800` (3 hours)
- R2E-Gym Sandbox Timeout: `activeDeadlineSeconds: 3600` (1 hour) (injected via sed patch)

**Bottleneck Analysis**:
- Based on `loggers.py:271` metrics at `T+30m`, the system handles the 128 sequences by bottlenecking at the generation token boundary.
- **Total Generation Throughput**: `~214.3 tokens/s` (across `126 reqs` currently generating in the KV Cache).
- **Speed per Agent**: `214.3 / 126 = ~1.7 tokens/sec` per agent episode.
- **The Theoretical Ceiling**: Given the `3600` second physical Sandbox timeout, $3600 \times 1.7 = \sim 6,120$ tokens. 
- **The Conflict**: It is physically impossible for the agent to finish an extended `<50 turns>` loop because the physical lifespan of the container will expire as soon as the total tokens exceed ~6000 (which is vastly under the `--max_response_length=32768` budget). The process is heavily token-generation bound, meaning the CPU Sandbox spends most of its 1 hour simply waiting for the TPU rollout engine to trickle tokens to it.

**Suggested Fixes for Next Run**:
- Increase `activeDeadlineSeconds` patch in `relaunch_deepswe_256_mlperf.sh` to `10800` (3 hours) to align with Python `episode_timeout_secs`.
- Decrease `MAX_CONCURRENCY` to `64` to increase average tokens/sec/instance (speed up each agent's individual trajectory response times).
- Fix the `RLTrainingConfig` instantiation.
