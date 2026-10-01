# Tunix Code Review Style Guide

This guide defines the architectural standards and review criteria for Tunix. **Gemini Code Assist must use this guide to provide high-level, context-aware reviews.**

---

## 1. Core Philosophy: JAX-Native & NNX-First

Tunix is a JAX-native post-training library for LLMs. Tunix principally uses NNX, a neural network library built on top of JAX. Please follow all common JAX and NNX patterns.

*   **Readability:** Code should be easy to understand for all maintainers and users.
*   **Maintainability:** Code should be easy to modify and extend.
*   **Modularity**
*   **Consistency**: Adherence to a consistent style across all projects. For example, adherence to naming and file structure conventions is crucial for predictability and maintainability.
*   **Reusability**: Try re-using components instead of re-writing.

---

## 2. Reference Documentation (Read First)

**Gemini:** Before reviewing any code, **read the following files** to understand the architectural context. Your review should be grounded in these documents.

*   **Contributing guidelines**: [`contributing.md`](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/contributing.md)
*   **Models**: [`models.md`](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/models.md) - `AutoModel` patterns and naming.
*   **RL Algorithms**: [`algorithms.md`](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/algorithms.md) - Registry and Config patterns for RL.
*   **Agentic RL**: [`agentic_rl.md`](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/agentic_rl.md)
*   [**JAX**](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
*   [**NNX**](https://flax.readthedocs.io/en/latest/nnx_basics.html)

In general, you can look into the [documentation files](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/).

---

## 3. Review Categories

Identify the type of contribution and apply the corresponding checklist.

### A. Model Contributions (`tunix/models/`)
*   **Naming:** Must follow the strict pattern `<family><ver>p<min>_<size>` (e.g., `gemma2p0_9b`, `qwen2p5_1p5b`).
*   **AutoModel:** New models must be integrated into `AutoModel.from_pretrained` and support all sources (`HUGGINGFACE`, `KAGGLE`, `GCS`).
*   **Pattern:**
    *   `ModelConfig`: Dataclass for architecture params.
    *   `ShardingConfig`: separate Dataclass for partition specs.
    *   `Module`: Pure NNX module implementation.

### B. RL Algorithms (`tunix/rl/`)
*   **Pattern:** Logic must be split between a **Configuration** and a **Learner**.
    *   **Config:** Must inherit from `AlgorithmConfig` (e.g., `class PPOConfig(AlgorithmConfig)`).
    *   **Learner:** Must inherit from `RLLearner` (e.g., `class PPOLearner(RLLearner)`).
*   **Registry:** Loss functions and advantage estimators *must* be registered (e.g., `@register_policy_loss_fn`) to allow hot-swapping via config.
*   **Reward Managers:** Complex reward logic should live in a `RewardManager`, not the Learner loop.

### C. Stand-alone Algorithms (`tunix/sft/`)
For non-RL algorithms, you can follow a pattern similar to `PeftTrainer` or `DpoTrainer`.

*   **Pattern:** SFT uses `TrainingConfig` (not `AlgorithmConfig`) and `PeftTrainer`.
*   **Trainer:** `PeftTrainer`

### D. Bug Fixes
*   **Reproduction:** Critical bug fixes should include a reproduction script or a link to a Colab notebook demonstrating the issue.
*   **Regression Test:** A matching test case in `_test.py` is mandatory.

### E. Notebooks & Examples
Prioritise readability. Have explanatory text cells to explain the code. Mention the hardware/accelerator on which the example will run.

---

## 4. Formatting & Linting

**Instructions for Contributors:**
If the formatting is off, please instruct the user to run the linter/formatter. You can pull instructions from [contributing.md](https://raw.githubusercontent.com/google/tunix/refs/heads/main/docs/contributing.md).

## 5. Other generic advice

### Type hints

* **Use type hints:**  Type hints improve code readability and help catch errors early.

### Comments

* **Write clear and concise comments:** Explain the "why" behind the code, not just the "what".
* **Comment sparingly:** Well-written code should be self-documenting where possible.
* **Use complete sentences:** Start comments with a capital letter and use proper punctuation.

### Logging
* **Use absl for logging**
* **Log at appropriate levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
* **Provide context:** Include relevant information in log messages to aid debugging.

### Error Handling
* **Use specific exceptions:** Avoid using broad exceptions like `Exception`.
* **Handle exceptions gracefully:** Provide informative error messages and avoid crashing the program.

---

## 6. MLPerf Branch (`atwigg/mlperf`) High-Leverage Systems Review Gates & Overrides

> **IMPORTANT — Precedence on `atwigg/mlperf`:** When reviewing pull requests on the `atwigg/mlperf` branch, **Section 6 takes strict precedence over Sections 1–5 above**. Specifically, supersede and ignore Section 3.D (reproduction/regression test requirements), Section 4 (formatting/linting), and Section 5 (generic style, type-hint pedantry, and "avoid crashing" error advice). Apply the high-leverage systems review gates below instead.

### 6.1 Core Review Philosophy & Strictly Out-of-Scope Noise

Code on `atwigg/mlperf` powers large-scale distributed RL training and inference (35B to 397B+ MoE models across 128–1024 TPU v5p/v7x chips). Engineers on this branch need a sharp architectural and systems reviewer—**not** a style linter.

* **Prioritize Engineering Leverage:** Only post a comment if it catches a bug, bottleneck, or design flaw that would otherwise cost hours of debugging, cause a multi-host TPU job hang/OOM, degrade step time/throughput, corrupt RL convergence, or create brittle cross-module coupling.
* **Challenge Technical Design & Scope:** Question whether new abstractions, background threads, state machines, or configuration layers are necessary, or if a simpler, direct design achieves the same goal with fewer failure modes.
* **Severity Calibration (`HIGH` / `CRITICAL`):** Because `.gemini/config.yaml` filters out `LOW` severity comments (`comment_severity_threshold: MEDIUM`), **classify all genuine architectural, bottleneck, API contract, concurrency, and numerical findings from Sections 6.2–6.6 as `HIGH` or `CRITICAL` severity**.
* **Be Concrete and Actionable:** Pinpoint the exact execution path, race condition, bottleneck, or contract mismatch and provide a drop-in fix or concrete design alternative.
* **Strictly Out of Scope (Do NOT Comment):**
  * **NO Python Coding Style or Formatting Comments:** Do **not** comment on PEP 8, formatting, indentation, line length, variable/function naming style, import ordering, docstrings, comment grammar, or logging frameworks (`absl` vs `logging`).
  * **NO Type-Hint Pedantry:** Do **not** nag authors to add routine Python type annotations or replace `dict`/`Any` for stylistic reasons. Only comment on types when a mismatched data contract or schema discrepancy will cause a runtime bug or silent state corruption.
  * **NO Test-Coverage Requests:** Do **not** ask authors to add, update, or expand unit tests (`*_test.py`), integration tests, regression tests, or reproduction scripts on `atwigg/mlperf`.

### 6.2 Architecture, API Design & Component Boundaries (`[CRITICAL]` / `[HIGH]`)

Audit how changes fit into the end-to-end system across Orchestrator (`tunix/experimental/orchestrator/`), Rollout (`tunix/experimental/rollout/`), Weight Sync (`tunix/experimental/weight_sync/`), Workers (`tunix/experimental/worker/`), Trainer (`tunix/experimental/train/`), and Trajectory Store (`tunix/experimental/trajectory/`):

* **Clean Separation of Concerns & Dependency Direction:**
  * Flag leaky abstractions where cluster/transport internals (e.g., Kubernetes/Pathways/gRPC specifics) bleed into RL algorithm or trainer logic, or where rollout sampler details leak into generic orchestrator interfaces.
  * When a contract mismatch occurs between a producer (e.g., sampler/collector) and a consumer (e.g., batch assembler/trainer), push for fixing the contract boundary cleanly rather than adding ad-hoc adapter shims or duplicate fields in both places.
* **Single Source of Truth for Distributed State:**
  * Flag designs that duplicate state ownership across components (e.g., maintaining parallel step counters, weight-version epochs, or worker health registries in multiple classes that can drift out of sync after a retry or partial failure).
* **API Contract Integrity & Fail-Loud Semantics:**
  * Evaluate new or modified APIs, dataclasses, and RPC payloads for clear, unambiguous contracts.
  * **Reject Silent Fallback Masking:** Flag defensive fallback hacks—such as `getattr(obj, "field", default)` or `.get("key", default)` with fabricated defaults (`""`, `0`, `False`, `{}`), or `except Exception:` blocks that swallow errors—when they mask a broken API contract or uninitialized state and allow the job to keep running with corrupted data. Broken invariants should fail fast and loud at the boundary.

### 6.3 Throughput, Memory & Distributed Systems Bottlenecks (`[CRITICAL]` / `[HIGH]`)

Every second of per-step overhead on a 256–1024 TPU chip job wastes massive compute. Hunt aggressively for performance and scalability bottlenecks:

* **Critical-Path Serialization & Pipeline Bubbles:**
  * Flag synchronous, blocking operations placed on the training or rollout critical path—such as synchronous GCS/disk I/O, blocking trajectory/metric flushing, unbatched per-trajectory RPCs, or unnecessary global barriers that prevent overlapping rollout generation, microbatch packing, and trainer execution.
* **Device-to-Host Synchronization & XLA Recompilation Thrashing:**
  * Flag blocking host reads of JAX device arrays (`.item()`, `float(x)`, `np.asarray(x)`, or host-side Python `if` branches on tensor values) inside per-step training or rollout loops that stall TPU async dispatch.
  * Flag dynamic tensor shapes, unpadded variable-length microbatches, or mutated Python objects passed into `@nnx.jit` / `jax.jit` functions that trigger repeated XLA recompilations.
* **35B–397B+ Host RAM, HBM & Staging Buffer Bottlenecks:**
  * Flag eager host-CPU materialization of full model weights or unsharded checkpoints, redundant deep copies of large weight dictionaries or trajectory batches, and temporary weight-sync staging buffers (`raiden_*`, `weight_sync/`) that are not promptly freed after transfer.
  * Flag unbounded in-memory queues or buffers (`trajectory_queue_manager.py`, `batch_assembly.py`, `async_writer.py`) that lack backpressure and can OOM host memory when rollout workers produce faster than the trainer consumes.
* **SPMD Sharding & Collective Bottlenecks:**
  * Flag `NamedSharding` / `PartitionSpec` mismatches across multi-slice TPU v5p/v7x topologies that induce unintended cross-host resharding, all-gather collectives on 397B weights, or unsharded parameter replication.

### 6.4 Concurrency, Asynchrony & Weight Synchronization Hazards (`[CRITICAL]` / `[HIGH]`)

Distributed RL orchestration relies heavily on concurrent threads, async event loops, and overlapping RDMA/H2H/D2D transfers. Audit these execution paths for subtle race conditions:

* **Async/Thread Lifecycle & Exception-Path Races (`rl_program.py`, `batch_assembly.py`, `distributed_rl_engine.py`, `async_writer.py`):**
  * Whenever work is offloaded via `asyncio.to_thread(...)`, `ThreadPoolExecutor`, or background daemon threads (e.g., `assembler.feed`, `assembler.flush`, microbatch packing, background checkpointing/logging), trace what happens when **an exception occurs, a stage times out, or `.reset()` / `.close()` / `.shutdown()` is called from the main thread**.
  * Flag shared mutable state accessed concurrently by background threads and main-thread recovery/reset paths without synchronization locks, as well as missing `finally:` cleanup or unawaited futures that can deadlock shutdown.
* **Weight Transfer vs. Rollout Execution Overlap (`weight_sync/`, `raiden_*`, `vllm_sampler*`, `sglang_jax*`):**
  * When overlapping H2H/D2D weight synchronization with rollout sampler prefill/decode, verify that destination weight buffers are never mutated in-place while an in-flight forward pass or KV-cache operation is reading them without an explicit barrier or double-buffering lock.
* **Partial Rollouts Across Weight Updates:**
  * When trajectories span weight-sync boundaries, verify that weight version tracking, KV/prefix cache invalidation, and per-token behavior-policy log-probabilities remain consistent across the weight transition.

### 6.5 RL Numerical & Convergence Invariants (`[CRITICAL]` / `[HIGH]`)

Silent math bugs waste days of TPU convergence runs without raising exceptions:

* **Token Masking (Prompt, Padding & Truncation):** Verify that prompt/context tokens, right/left padding tokens, and invalid/aborted rollout turns are strictly masked out (`mask == 0`) from policy gradient loss, KL divergence, entropy, and sequence-level reductions.
* **Advantage & Loss Numerical Stability:**
  * Verify group-relative (GRPO) or batch advantage normalization masks out padding before computing statistics and includes an epsilon guard (`(adv - mean) / (std + eps)`) to prevent `NaN`/`Inf` on zero-variance prompt groups.
  * Ensure `jax.lax.stop_gradient` is applied to reference model log-probabilities, behavior-policy log-probabilities, and advantage targets.
* **FP8 / `bfloat16` Accumulation Precision:** When rollout or trainer layers use FP8 or `bfloat16`, verify that log-probability sums, KL divergence, loss reductions, and normalization statistics are accumulated in `float32` to prevent numerical underflow/overflow.

### 6.6 MLPerf Recipe, Config Wiring & Benchmark Integrity (`[CRITICAL]` / `[HIGH]`)

Changes in `tunix/experimental/examples/recipes/`, `tunix/experimental/examples/common/`, `requirements/`, and `Dockerfile` directly impact benchmark validity:

* **End-to-End Flag & Config Propagation:**
  * When a CLI flag, environment variable, or config field is added, renamed, or modified in launcher scripts (`mlperf_base.sh`, `mlperf_pathways_config.sh`, `mlperf_397b_*.sh`, `mlperf_35b_*.sh`, `maxtext_config.sh`), verify that it is wired all the way through `run_trainer_node.py`, `run_rollout_node.py`, `run_inference_node.py`, and downstream config objects. Flag dead/ignored flags or mismatched parameter names immediately.
* **Train / Eval Recipe Parity:**
  * When modifying model architecture parameters, cluster/mesh topology, quantization/FP8 flags, sampler settings, or Pathways/XLA flags in a training recipe (`mlperf_397b_256_v7x.sh`, `mlperf_397b_512_v7x.sh`, `mlperf_397b_1024_v7x.sh`, `mlperf_35b_128_v7x.sh`), flag unintended drift against corresponding evaluation recipes (`evals/mlperf_397b_v7x_eval.sh`, `evals/mlperf_35b_v5p_eval.sh`) or sibling scale scripts.
* **Dependency Pinning & Logging Overhead:**
  * Flag floating branch references (e.g., `@main`) in `requirements/maxtext_requirements.txt` or `Dockerfile` that break benchmark reproducibility (require pinned commit SHAs).
  * Flag changes that re-introduce legacy synchronous CSV/JSON trajectory logging or duplicate metric writes on hot paths alongside `TrajectoryStore`.
