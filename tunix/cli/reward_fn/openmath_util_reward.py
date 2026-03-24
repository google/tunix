# ~/tunix/tunix/cli/reward_fn/openmath_util_reward.py
from typing import List, Any, Dict
import numpy as np
from tunix.utils import math_rewards as tunix_math_rewards
from absl import logging

def reward_fn(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> np.ndarray:
    """
    Wrapper for tunix.utils.math_rewards.math_reward.
    The 'answer' key is provided by the remapped dataset in grpo_main.py.
    """
    try:
        # The 'answer' key in kwargs comes from the remapped dataset
        rewards = tunix_math_rewards.math_reward(prompts, completions, answer=answer, **kwargs)
        return np.array(rewards, dtype=np.float32)
    except Exception as e:
        logging.exception(f"Error in tunix_math_rewards.math_reward: {e}")
        return np.array([0.0] * len(prompts), dtype=np.float32)


