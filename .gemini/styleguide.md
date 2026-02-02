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
