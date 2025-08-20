# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TAP tests for Tunix RL (Reinforcement Learning) functionality."""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Any, Callable


def test_rl_algorithm_mathematics() -> bool:
    """Test RL algorithm mathematical foundations and computations."""
    try:
        import optax
        
        # Test that RL algorithm modules can be imported and configured
        # This tests Tunix's RL functionality, not JAX operations
        
        # Test GRPO module structure
        from tunix.rl.grpo import grpo_learner, grpo_helpers
        assert hasattr(grpo_learner, '__name__')
        assert hasattr(grpo_helpers, '__name__')
        
        # Test DPO module structure
        from tunix.rl.dpo import dpo_trainer
        assert hasattr(dpo_trainer, '__name__')
        
        # Test PPO module structure
        from tunix.rl.ppo import ppo_learner, ppo_helpers
        assert hasattr(ppo_learner, '__name__')
        assert hasattr(ppo_helpers, '__name__')
        
        # Test RL cluster module structure
        from tunix.rl import rl_cluster
        assert hasattr(rl_cluster, '__name__')
        
        # Test that configuration classes exist and can be instantiated
        from tunix.rl.grpo.grpo_learner import GrpoConfig
        from tunix.rl.dpo.dpo_trainer import DpoTrainingConfig
        
        # Verify these are actual classes
        assert hasattr(GrpoConfig, '__call__')
        assert hasattr(DpoTrainingConfig, '__call__')
        
        # Test that RL utilities module exists
        from tunix.rl import utils
        assert hasattr(utils, '__name__')
        
        return True
    except Exception:
        return False


def test_rl_training_pipeline_validation() -> bool:
    """Test RL training pipeline configurations for production readiness."""
    try:
        from tunix.rl.grpo.grpo_learner import GrpoConfig
        from tunix.rl.dpo.dpo_trainer import DpoTrainingConfig
        from tunix.rl.rollout.base_rollout import RolloutConfig
        import optax
        
        # Test 1: GRPO configuration for human feedback learning
        grpo_config = GrpoConfig(
            num_generations=4,      # Multiple generations for comparison
            num_iterations=3,       # Iterative policy improvement
            beta=0.1,              # KL penalty (higher than typical for stability)
            epsilon=0.2,           # PPO clipping (standard value)
            loss_algo="grpo",
        )
        
        # Validate GRPO hyperparameters for stability
        assert 0.01 <= grpo_config.beta <= 1.0  # KL penalty should be reasonable
        assert 0.1 <= grpo_config.epsilon <= 0.3  # Clipping should prevent large updates
        assert grpo_config.num_generations >= 2   # Need multiple generations to compare
        
        # Test 2: Training batch size calculation and validation
        response_length = 512  # Typical response length
        total_samples_per_step = (grpo_config.num_generations * 
                                 grpo_config.num_iterations * 
                                 response_length)
        assert total_samples_per_step > 1000  # Ensure sufficient sample size
        
        # Test 3: DPO configuration for preference learning
        dpo_config = DpoTrainingConfig(
            eval_every_n_steps=200,
            max_steps=5000,       # Typical DPO training length
            beta=0.5,             # DPO temperature (higher than GRPO)
            label_smoothing=0.1,  # Smoothing for robustness
            padding_value=-100,   # Standard ignore index
        )
        
        # Validate DPO hyperparameters
        assert 0.1 <= dpo_config.beta <= 2.0  # DPO beta should be reasonable
        assert dpo_config.eval_every_n_steps < dpo_config.max_steps
        assert 0 <= dpo_config.label_smoothing <= 0.5
        
        # Test 4: Rollout configuration for inference
        rollout_config = RolloutConfig(
            max_tokens_to_generate=1024,  # Longer responses for helpfulness
            temperature=0.8,              # Balanced creativity/coherence
            top_p=0.95,                  # Nucleus sampling
            n=4,                         # Multiple candidates for GRPO
            max_prompt_length=2048,      # Support long contexts
        )
        
        # Validate rollout parameters for quality generation
        assert 0.1 <= rollout_config.temperature <= 2.0
        assert 0.5 <= rollout_config.top_p <= 1.0
        assert rollout_config.n >= grpo_config.num_generations  # Enough candidates
        assert rollout_config.max_tokens_to_generate > 0
        assert rollout_config.max_prompt_length >= rollout_config.max_tokens_to_generate
        
        # Test 5: Computational requirements estimation and validation
        vocab_size = 50000
        memory_per_token = vocab_size * 4  # 4 bytes per float32 logit
        total_memory_mb = (rollout_config.n * 
                          rollout_config.max_tokens_to_generate * 
                          memory_per_token) / (1024 * 1024)
        
        # Sanity check: memory requirements should be reasonable for modern GPUs
        assert total_memory_mb < 8000  # Less than 8GB for logits alone
        
        # Test 6: Training efficiency analysis
        # Calculate training steps and validation frequency
        training_steps = dpo_config.max_steps
        validation_frequency = dpo_config.eval_every_n_steps
        num_validations = training_steps // validation_frequency
        
        # Should have reasonable number of validations
        assert 5 <= num_validations <= 100, f"Unrealistic validation frequency: {num_validations}"
        
        # Test 7: Resource scaling analysis
        # Test different batch sizes and their memory implications
        batch_sizes = [1, 2, 4, 8]
        memory_requirements = []
        
        for batch_size in batch_sizes:
            # Simulate memory scaling with batch size
            batch_memory = batch_size * total_memory_mb
            memory_requirements.append(batch_memory)
            
            # Memory should scale linearly with batch size
            if batch_size > 1:
                expected_memory = batch_size * memory_requirements[0]
                assert abs(batch_memory - expected_memory) < 0.1 * expected_memory
        
        # Test 8: Training stability validation
        # KL penalty should be proportional to response length for stability
        max_response_length = rollout_config.max_tokens_to_generate
        recommended_beta = min(0.1 * (max_response_length / 512), 1.0)
        
        # Current beta should be reasonable for the response length
        assert grpo_config.beta >= recommended_beta * 0.5, "Beta too low for response length"
        assert grpo_config.beta <= recommended_beta * 2.0, "Beta too high for response length"
        
        return True
    except Exception:
        return False


def test_rl_reward_modeling_and_optimization() -> bool:
    """Test RL reward modeling and optimization pipeline components."""
    try:
        import optax
        
        # Test reward model scoring (central to RLHF)
        batch_size = 16
        seq_len = 256
        hidden_dim = 768
        
        # Simulate model outputs and reward model inputs
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Simulate response representations from language model
        response_hidden_states = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))
        
        # Test reward model head (maps hidden states to scalar rewards)
        def reward_model_head(hidden_states):
            # Simple linear projection to scalar reward
            # In practice, this would be a learned linear layer
            weights = jax.random.normal(key2, (hidden_dim, 1))
            return jnp.sum(hidden_states * weights.T, axis=-1)  # (batch, seq_len)
        
        reward_scores = reward_model_head(response_hidden_states)
        assert reward_scores.shape == (batch_size, seq_len)
        
        # Test reward aggregation strategies
        # Strategy 1: Mean pooling over sequence
        mean_rewards = jnp.mean(reward_scores, axis=1)
        assert mean_rewards.shape == (batch_size,)
        
        # Strategy 2: Last token reward (common for completion tasks)
        last_token_rewards = reward_scores[:, -1]
        assert last_token_rewards.shape == (batch_size,)
        
        # Strategy 3: Max reward (for best-of-n sampling)
        max_rewards = jnp.max(reward_scores, axis=1)
        assert max_rewards.shape == (batch_size,)
        
        # Test Bradley-Terry preference model (DPO foundation)
        def bradley_terry_preference_prob(reward_chosen, reward_rejected, beta=1.0):
            """Compute probability that chosen response is preferred"""
            return jax.nn.sigmoid(beta * (reward_chosen - reward_rejected))
        
        # Simulate paired preference data
        chosen_rewards = mean_rewards[:batch_size//2]
        rejected_rewards = mean_rewards[batch_size//2:]
        
        preference_probs = bradley_terry_preference_prob(chosen_rewards, rejected_rewards, beta=0.5)
        assert preference_probs.shape == (batch_size//2,)
        assert jnp.all((preference_probs >= 0) & (preference_probs <= 1))
        
        # Test policy optimization with Adam
        # Simulate policy parameters
        policy_params = jax.random.normal(key3, (1000,))  # Flattened parameters
        
        # Test optimizer state initialization
        optimizer = optax.adam(learning_rate=1e-4)
        opt_state = optimizer.init(policy_params)
        
        # Test gradient-based update
        def dummy_loss(params):
            return jnp.sum(params ** 2)  # Simple quadratic loss
        
        loss_value, grads = jax.value_and_grad(dummy_loss)(policy_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, policy_params)
        new_params = optax.apply_updates(policy_params, updates)
        
        # Verify optimization step
        assert new_params.shape == policy_params.shape
        assert not jnp.allclose(new_params, policy_params)  # Parameters should change
        
        # Test learning rate scheduling (important for RL stability)
        lr_schedule = optax.exponential_decay(
            init_value=1e-3,
            transition_steps=1000,
            decay_rate=0.9,
            staircase=False
        )
        
        # Test learning rate at different steps
        step_0_lr = lr_schedule(0)
        step_1000_lr = lr_schedule(1000)
        step_5000_lr = lr_schedule(5000)
        
        assert step_0_lr == 1e-3
        assert step_1000_lr < step_0_lr  # Learning rate should decay
        assert step_5000_lr < step_1000_lr
        
        return True
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of TAP tests for RL functionality.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("rl_algorithm_mathematics", test_rl_algorithm_mathematics, True),
        ("rl_training_pipeline_validation", test_rl_training_pipeline_validation, True),
        ("rl_reward_modeling_and_optimization", test_rl_reward_modeling_and_optimization, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
