from typing import Any, Dict, Callable
from tunix.rl.multi_turn.rewards.reward_types import RewardOutput

# ---------- ① Registry ----------
_REGISTRY: Dict[str, Callable[[Dict, str], RewardOutput]] = {}

def register(name: str):
    """Decorator: register the function into the registry"""
    def _wrap(fn):
        _REGISTRY[name] = fn
        return fn
    return _wrap

def get_reward_fn(name: str):
    return _REGISTRY[name]

# ---------- ② Built-in reward strategies ----------

@register("zero")
def zero_reward(task: Dict[str, Any], action: str) -> RewardOutput:
    """Always returns 0 score, used as a placeholder"""
    return RewardOutput(0.0, {})

@register("exact_match")
def exact_match(task: Dict[str, Any], action: str) -> RewardOutput:
    """Returns 1.0 if the answer exactly matches ground_truth, otherwise 0"""
    truth = str(task.get("ground_truth", "")).strip()
    score = 1.0 if action.strip() == truth else 0.0
    return RewardOutput(score, {"exact_match": score})

# ---------- ③ Aggregator (Optional) ----------
def make_linear(weights: Dict[str, float]) -> Callable[[Dict, str], RewardOutput]:
    """
    Linearly combines multiple sub-reward strategies according to weights.
    Example: {"exact_match": 1.0, "zero": 0.0}
    """
    def _fn(task: Dict[str, Any], action: str):
        total, meta = 0.0, {}
        for name, w in weights.items():
            out = get_reward_fn(name)(task, action)
            total += w * out.reward
            meta.update(out.metadata)
        return RewardOutput(total, meta)
    return _fn


# -------- Example Reward Function --------
@register("is_two")
def is_two_reward(task: Dict[str, Any], action: str) -> RewardOutput:
    """Returns 1.0 if the action equals 2 (either number or string), otherwise 0.0"""
    try:
        value = float(action.strip())
        score = 1.0 if value == 2.0 else 0.0
    except ValueError:
        score = 0.0
    return RewardOutput(score, {"is_two": score})
