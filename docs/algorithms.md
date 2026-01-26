
# Algorithms

Tunix supports a wide array of SOTA algorithms for RL and SFT. Its modular design
also allows users to easily extend Tunix with custom algorithms, as described
further below.

## Supported Algorithms

* **Supervised Fine-Tuning (SFT) & Preference**

  * **[PEFT](performance.md#peft-with-lora)** (Parameter-Efficient Fine-Tuning)

  * **[DPO](https://arxiv.org/abs/2305.18290)** (Direct Preference Optimization)
      * **[ORPO](https://arxiv.org/abs/2403.07691)** (Odds ratio Preference Optimization)

* **Reinforcement Learning (RL)**

  * **[PPO](https://arxiv.org/abs/1707.06347)** (Proximal Policy Optimization)
  * **[GRPO](https://arxiv.org/abs/2402.03300)** (Group Relative Policy Optimization)
      * **[GSPO-Token](https://arxiv.org/abs/2507.18071)** (Token-level Group Sequence Policy Optimization)
      * **[DAPO](https://arxiv.org/abs/2503.14476)** (Direct Alignment via Preference Optimization)
      * **[Dr.GRPO](https://arxiv.org/abs/2503.14476)** (Distributionally Robust GRPO)


## Add a New RL Algorithm

Tunix is designed to be highly extensible. You can introduce new algorithms by
subclassing `AlgorithmConfig` (or its descendants) and implementing a
corresponding Learner.

The system uses a **parallel inheritance** pattern: extending a Configuration
often requires extending a Learner to consume it.

### Class Hierarchy & Interaction Diagram

<!-- TODO(b/475597805): Add better formatted diagram. -->

```text
       CONFIGURATION                            LEARNER (The Engine)
     (Defines Params)                       (Orchestrates Execution)
   +-------------------+                   +-----------------------+
   |  AlgorithmConfig  | <---(binds)--|       RLLearner       |
   +-------------------+                   +-----------------------+
             ^                                         ^      |
             |                                         |      +---(Uses)---> [Function Registry]
     (Inheritance)                               (Inheritance)                (Loss, Advantage, Reward)
             |                                         |
   +-------------------+                   +-----------------------+
   |    GRPOConfig     | <---(binds)--|      GRPOLearner      |
   +-------------------+                   +-----------------------+
             ^                                         ^
             |                                         |
   +-------------------+                   +-----------------------+
   |    DAPOConfig     | <---(binds)--|      DAPOLearner    |
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
    reward_manager: str = "sequence-level"

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
    def __init__(self, rl_cluster, algo_config: MyNewAlgoConfig, reward_fns, ...):
        # Custom initialization (e.g., adding specific reward shaping)
        if algo_config.my_hyperparam > 0.1:
            reward_fns.append(my_custom_reward_fn)

        super().__init__(
            rl_cluster=rl_cluster,
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
from tunix.registry import register_policy_loss_fn

@register_policy_loss_fn("my_custom_loss")
def compute_my_custom_loss(log_probs, advantages, **kwargs):
    """
    Args:
        log_probs: Tensor of log probabilities.
        advantages: Tensor of calculated advantages.
    Returns:
        Scalar loss tensor.
    """
    return -torch.mean(log_probs * advantages)

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
*   **Complex Case:** Subclass `AbstractRewardManager` if you need custom
    aggregation (e.g., weighted sums) or specialized logging strategies.

**Example: Custom Manager**

Below is an example of a manager that performs custom aggregation and injects
specific intermediate logs.

```python
class MyCustomRewardManager(AbstractRewardManager):
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> Dict[str, Any]:
        """
        Orchestrates reward calculation.
        """
        # 1. Run all reward functions
        raw_scores = [fn(prompts, completions) for fn in self.reward_fns]

        # 2. Custom Aggregation (e.g., Multi-objective weighted sum)
        final_rewards = self.aggregate_logic(raw_scores)

        # 3. Calculate log metrics
        log_metrics = self._prepare_log_metrics(prompts, completions, raw_scores, final_rewards)

        # 4. Return format required by Learner
        return {
            "rewards": final_rewards,
            "log_metrics": log_metrics
        }

    def _prepare_log_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: np.ndarray,
        sum_rewards: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Logs individual and summed rewards.
        """
        # 1. Standard Logs (prompts, completions, sum, min, max)
        metrics_to_log = super()._prepare_log_metrics(prompts, completions, rewards, sum_rewards)

        # 2. Custom Intermediate Logging
        # User may freely add intermediate reward logging results here.
        # Example: Logging specific components of the reward signal separately
        if hasattr(self, "reward_fns"):
             for i, fn in enumerate(self.reward_fns):
                # Log the specific contribution of each function (e.g. rewards/grammar_score)
                name = getattr(fn, "__name__", f"fn_{i}")
                metrics_to_log[f"rewards/{name}"] = (rewards[:, i], np.mean)

        return metrics_to_log

```
