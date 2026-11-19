<!-- DO NOT REMOVE! Placeholder for TOC. -->
# Algorithms

Tunix supports a wide array of SOTA algorithms for RL and SFT. Its modular design
also allows users to easily extend Tunix with custom algorithms, as described
further below.

## Supported Algorithms

* **Supervised Fine-Tuning (SFT) & Preference**

  * **Full-Weight SFT** (`PeftTrainer`) and **Diffusion LLM SFT** (`tunix.sft.diffusion`)
  * **[PEFT](performance.md#peft-with-lora)** (Parameter-Efficient Fine-Tuning with LoRA / QLoRA)
  * **[DPO](https://arxiv.org/abs/2305.18290)** (Direct Preference Optimization, `DPOTrainer`)
  * **[ORPO](https://arxiv.org/abs/2403.07691)** (Odds Ratio Preference Optimization, `ORPOTrainer` with `ORPOTrainingConfig(algorithm="orpo", ...)`)

* **Knowledge Distillation** (`DistillationTrainer` in `tunix.distillation`)

  * **Logit Distillation**: `LogitStrategy`
  * **Feature Distillation**: `FeatureProjectionStrategy`, `FeaturePoolingStrategy`
  * **Attention Distillation**: `AttentionTransferStrategy`, `AttentionProjectionStrategy`

* **Reinforcement Learning (RL)**

  * **[PPO](https://arxiv.org/abs/1707.06347)** (Proximal Policy Optimization, `PPOLearner` / `PPOConfig`)
  * **[GRPO](https://arxiv.org/abs/2402.03300)** (Group Relative Policy Optimization, `GRPOLearner` / `GRPOConfig`)
  * **[GSPO-Token](https://arxiv.org/abs/2507.18071)** (Token-level Group Sequence Policy Optimization, configured via `GRPOConfig(loss_algo="gspo-token")`)
  * **[DAPO](https://arxiv.org/abs/2503.14476)** (Decoupled Clip and Dynamic sAmpling Policy Optimization, `DAPOLearner` / `DAPOConfig`)
  * **[Dr.GRPO](https://arxiv.org/abs/2503.20783)** (Dr. GRPO: GRPO Done Right, `DrGRPOLearner` / `DrGRPOConfig`)
  * **[Agentic RL](agentic_rl.md)** (`AgenticGRPOLearner` / `AgenticRLLearner` for multi-turn tool-using and asynchronous RL rollouts)

## Add a New RL Algorithm

Tunix is designed to be highly extensible. You can introduce new algorithms by
subclassing `AlgorithmConfig` (or its descendants) and implementing a
corresponding Learner.

The system uses a **parallel inheritance** pattern: extending a Configuration
often requires extending a Learner to consume it.

### Class Hierarchy & Interaction Diagram

```text
       CONFIGURATION                            LEARNER (The Engine)
     (Defines Params)                       (Orchestrates Execution)
   +-------------------+                   +-----------------------+
   |  AlgorithmConfig  | <---(binds)--     |       RLLearner       |
   +-------------------+                   +-----------------------+
             ^                                         ^      |
             |                                         |      +---(Uses)---> [Function Registry]
     (Inheritance)                               (Inheritance)                (Loss, Advantage, Reward)
             |                                         |
   +-------------------+                   +-----------------------+
   |    GRPOConfig     | <---(binds)--     |      GRPOLearner      |
   +-------------------+                   +-----------------------+
             ^                                         ^
             |                                         |
   +-------------------+                   +-----------------------+
   |    DAPOConfig     | <---(binds)--     |      DAPOLearner    |
   +-------------------+                   +-----------------------+

```

--------------------------------------------------------------------------------

### 1. Defining the Configuration & Learner

To add a new algorithm, you typically define a config (to hold your params) and
a learner (to use them).

**Step 1: The Configuration** Inherit from `AlgorithmConfig` (or a specific
child like `GRPOConfig` if your algorithm is a variant of it). Use
`__post_init__` to validate your new settings.

```python
@dataclasses.dataclass(slots=True, kw_only=True)
class MyNewAlgoConfig(AlgorithmConfig):
    # 1. Identity
    algo_variant: str = "my_new_algo"

    # 2. Components (References strings in FunctionRegistry)
    advantage_estimator: str = "gae"
    policy_loss_fn: str = "my_custom_loss"
    reward_mode: str = "reward_fn"

    # 3. Custom Hyperparameters
    my_hyperparam: float = 0.5

    def __post_init__(self):
        ...

```

**Step 2: The Learner** Inherit from `RLLearner` (or `GrpoLearner` etc). This is
where you inject specific execution logic, such as modifying the training loop
or injecting custom reward functions.

```python
class MyNewAlgorithmLearner(RLLearner):
    def __init__(self, rl_engine, algo_config: MyNewAlgoConfig, reward_fns, ...):
        # Custom initialization (e.g., adding specific reward shaping)
        if algo_config.my_hyperparam > 0.1:
            reward_fns.append(my_custom_reward_fn)

        super().__init__(
            rl_engine=rl_engine,
            algo_config=algo_config,
            reward_fns=reward_fns,
            ...
        )

```

--------------------------------------------------------------------------------

### 2. Custom Loss & Advantage (The Registry)

Tunix uses a **Function Registry** to manage mathematical components. This
allows you to hot-swap loss functions or advantage estimators in your config
without changing the Learner code.

**How to Register a New Loss Function:** Define your loss function and decorate
it with `@register_policy_loss_fn`.

```python
from flax import nnx
import jax.numpy as jnp
from tunix.rl import common
from tunix.rl.function_registry import register_policy_loss_fn
from tunix.sft import utils as sft_utils


@register_policy_loss_fn("my_custom_loss")
def compute_my_custom_loss(
    model: nnx.Module,
    train_example,
    algo_config,
    pad_id: int,
    eos_id: int,
    **kwargs,
) -> sft_utils.LossOutput:
    """Computes a custom policy gradient loss for Tunix RL learners."""
    graphdef, state = nnx.split(model)
    per_token_logps, _ = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=train_example.prompt_ids,
        completion_tokens=train_example.completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        stop_gradient=False,
        compute_logps_chunk_size=kwargs.get("compute_logps_chunk_size", 0),
    )
    mask = jnp.astype(train_example.completion_mask, jnp.float32)
    advantages = train_example.advantages
    if advantages.ndim == 1:
        advantages = advantages[:, None]

    token_loss = -(per_token_logps * advantages) * mask
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    primary_loss = sft_utils.WeightedMetric(jnp.sum(token_loss), denom)
    return sft_utils.LossOutput(primary_loss=primary_loss, aux_metrics={})
```

**Usage:** Once registered, simply reference it in your config: `policy_loss_fn:
str = "my_custom_loss"`

--------------------------------------------------------------------------------

### 3. Custom Reward Management

Rewards are handled by a **Manager** pattern. The Learner delegates to the
Manager to compute rewards from model output and log the results.

*   **Reward Function:** A simple callable that calculates scores based on
    completion texts (e.g., regex matching, length constraints, keyword
    presence).
*   **Reward Manager:** The orchestrator that calls reward functions, formats
    the output, and handles logging.

**When to use what?**

*   **Simple Case:** Just add a new function to the `reward_fns` list.
*   **Complex Case:** Subclass `SequenceRewardManager` (or
    `AbstractRewardManager` for a completely custom contract) if you need custom
    aggregation (e.g., weighted sums) or specialized logging strategies.

**Example: Custom Manager**

Below is an example of a manager that subclasses `SequenceRewardManager`,
performs custom weighted aggregation over a 2D `(num_prompts, num_reward_fns)`
`np.ndarray`, and registers itself with `@register_reward_manager`.

```python
from typing import Any, Dict, List
import numpy as np
from tunix.rl import function_registry
from tunix.rl import reward_manager


@function_registry.register_reward_manager("my_custom_manager")
class MyCustomRewardManager(reward_manager.SequenceRewardManager):
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Orchestrates reward calculation with custom weighted aggregation."""
        num_prompts = len(prompts)
        num_fns = len(self.reward_fns)
        rewards = np.zeros((num_prompts, num_fns), dtype=np.float64)
        for i, fn in enumerate(self.reward_fns):
            rewards[:, i] = np.asarray(
                fn(prompts=prompts, completions=completions, **kwargs),
                dtype=np.float64,
            )

        # Custom aggregation (e.g., equal or weighted sum across reward functions)
        weights = np.ones(num_fns, dtype=np.float64)
        sum_rewards = rewards @ weights

        # _prepare_log_metrics expects 2D `rewards` and keyword-only `sum_rewards`
        log_metrics = self._prepare_log_metrics(
            prompts,
            completions,
            rewards,
            sum_rewards=sum_rewards,
        )
        return {
            "rewards": sum_rewards,
            "log_metrics": log_metrics,
        }

    def _prepare_log_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: np.ndarray,
        *,
        sum_rewards: np.ndarray,
        env_rewards: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Logs standard summary rewards plus custom intermediate metrics."""
        metrics_to_log = super()._prepare_log_metrics(
            prompts,
            completions,
            rewards,
            sum_rewards=sum_rewards,
            env_rewards=env_rewards,
        )
        metrics_to_log["rewards/score_spread"] = (
            np.ptp(rewards, axis=1) if rewards.shape[1] > 0 else np.zeros(len(prompts)),
            np.mean,
        )
        return metrics_to_log
```
