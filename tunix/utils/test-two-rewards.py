# math rewards test

import math_rewards

import rewards_math_verify

completions = ["\\boxed{1}", "\\boxed{2}", "\\boxed{3}", "\\boxed{4}"]
answers = ["1", "2", "3", "4"]

orig_rewards = math_rewards.math_reward(
    prompts=[""] * len(completions),
    completions=completions,
    answer=answers,
)

verify_rewards = rewards_math_verify.math_reward(
    prompts=[""] * len(completions),
    completions=completions,
    answer=answers,
)

print("Original rewards:", orig_rewards)
print("Verify rewards:", verify_rewards)
# Original rewards: [1.0, 1.0, 1.0, 1.0]
# Verify rewards: [1.0, 1.0, 1.0, 1.0]

completions = ["\\boxed{1}", "\\boxed{2.1}", "3", "\\boxed{4}", "\\boxed{80\\%}", '\\(\\boxed{-\\dfrac{2}{3}}\\).']
answers = ["1", "2", "3", "4", "80\\%", "-\\frac{2}{3}"]
orig_rewards = math_rewards.math_reward(
    prompts=[""] * len(completions),
    completions=completions,
    answer=answers,
)

verify_rewards = rewards_math_verify.math_reward(
    prompts=[""] * len(completions),
    completions=completions,
    answer=answers,
)
print("Original rewards:", orig_rewards)
print("Verify rewards:", verify_rewards)
# Original rewards: [1.0, 0.0, 0.0, 1.0]
# Verify rewards: [1.0, 0.0, 1.0, 1.0]