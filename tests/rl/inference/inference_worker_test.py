# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for inference_worker.py"""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from unittest import mock
from tunix.rl.inference import inference_worker
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)


class MockModel(nnx.Module):
  """Mock model for testing."""
  
  def __init__(self, rngs: nnx.Rngs, output_value: float = 1.0):
    self.output_value = output_value
    self.dense = nnx.Linear(4, 1, rngs=rngs)
  
  def __call__(self, x):
    # Simple mock behavior - return a constant value
    return jnp.full((x.shape[0],), self.output_value)


class InferenceWorkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(42)
    # Sample inputs for testing
    self.prompt_tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    self.completion_tokens = jnp.array([[7, 8, 9], [10, 11, 12]])
    self.pad_id = 0
    self.eos_id = 2

  def test_constructor_with_valid_models(self):
    """Test constructor accepts valid model configurations."""
    # Single models
    models = {
        "critic": MockModel(self.rngs, 0.5),
        "reference": MockModel(self.rngs, 0.7),
        "reward": MockModel(self.rngs, 0.9)
    }
    worker = inference_worker.InferenceWorker(models)
    self.assertIsNotNone(worker)

  def test_constructor_with_multiple_reward_models(self):
    """Test constructor accepts multiple reward models."""
    models = {
        "critic": MockModel(self.rngs, 0.5),
        "reference": MockModel(self.rngs, 0.7),
        "reward": MockModel(self.rngs, 0.9),
        "reward_safety": MockModel(self.rngs, 0.8),
        "reward_helpfulness": MockModel(self.rngs, 0.6),
        "reward_model1": MockModel(self.rngs, 0.4)
    }
    worker = inference_worker.InferenceWorker(models)
    self.assertIsNotNone(worker)

  def test_constructor_rejects_invalid_model_names(self):
    """Test constructor rejects invalid model names."""
    models = {
        "invalid_model": MockModel(self.rngs, 0.5),
        "reward": MockModel(self.rngs, 0.9)
    }
    with self.assertRaises(ValueError) as cm:
      inference_worker.InferenceWorker(models)
    self.assertIn("invalid_model", str(cm.exception))
    self.assertIn("not supported", str(cm.exception))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_rewards_single_model(self, mock_compute_score):
    """Test get_rewards with single reward model."""
    mock_compute_score.return_value = jnp.array([0.9, 0.8])
    
    models = {"reward": MockModel(self.rngs, 0.9)}
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_rewards(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    mock_compute_score.assert_called_once()
    np.testing.assert_array_equal(result, jnp.array([0.9, 0.8]))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_rewards_multiple_models_default(self, mock_compute_score):
    """Test get_rewards with multiple models using default 'reward' model."""
    mock_compute_score.return_value = jnp.array([0.9, 0.8])
    
    models = {
        "reward": MockModel(self.rngs, 0.9),
        "reward_safety": MockModel(self.rngs, 0.8)
    }
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_rewards(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    mock_compute_score.assert_called_once_with(
        models["reward"], self.prompt_tokens, self.completion_tokens, 
        self.pad_id, self.eos_id
    )
    np.testing.assert_array_equal(result, jnp.array([0.9, 0.8]))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_rewards_specific_model(self, mock_compute_score):
    """Test get_rewards with specific reward model name."""
    mock_compute_score.return_value = jnp.array([0.7, 0.6])
    
    models = {
        "reward": MockModel(self.rngs, 0.9),
        "reward_safety": MockModel(self.rngs, 0.7)
    }
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_rewards(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id,
        reward_model_name="reward_safety"
    )
    
    mock_compute_score.assert_called_once_with(
        models["reward_safety"], self.prompt_tokens, self.completion_tokens,
        self.pad_id, self.eos_id
    )
    np.testing.assert_array_equal(result, jnp.array([0.7, 0.6]))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_rewards_fallback_single_model(self, mock_compute_score):
    """Test get_rewards fallback when only one reward model exists."""
    mock_compute_score.return_value = jnp.array([0.8, 0.7])
    
    models = {"reward_safety": MockModel(self.rngs, 0.8)}
    worker = inference_worker.InferenceWorker(models)
    
    # Request non-existent "reward", should fallback to "reward_safety"
    result = worker.get_rewards(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    mock_compute_score.assert_called_once_with(
        models["reward_safety"], self.prompt_tokens, self.completion_tokens,
        self.pad_id, self.eos_id
    )
    np.testing.assert_array_equal(result, jnp.array([0.8, 0.7]))

  def test_get_rewards_no_reward_model(self):
    """Test get_rewards raises error when no reward model exists."""
    models = {"critic": MockModel(self.rngs, 0.5)}
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_rewards(
          self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
      )
    self.assertIn("No reward model is available", str(cm.exception))

  def test_get_rewards_invalid_model_name_multiple_available(self):
    """Test get_rewards error when invalid model name and multiple models exist."""
    models = {
        "reward_safety": MockModel(self.rngs, 0.8),
        "reward_helpfulness": MockModel(self.rngs, 0.7)
    }
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_rewards(
          self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id,
          reward_model_name="nonexistent"
      )
    self.assertIn("nonexistent", str(cm.exception))
    self.assertIn("Available reward models", str(cm.exception))
    self.assertIn("reward_safety", str(cm.exception))
    self.assertIn("reward_helpfulness", str(cm.exception))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_all_rewards(self, mock_compute_score):
    """Test get_all_rewards returns rewards from all models."""
    def side_effect(model, *args):
      if model == models["reward_safety"]:
        return jnp.array([0.8, 0.7])
      elif model == models["reward_helpfulness"]:
        return jnp.array([0.6, 0.5])
      else:
        return jnp.array([0.9, 0.8])
    
    mock_compute_score.side_effect = side_effect
    
    models = {
        "reward": MockModel(self.rngs, 0.9),
        "reward_safety": MockModel(self.rngs, 0.8),
        "reward_helpfulness": MockModel(self.rngs, 0.6)
    }
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_all_rewards(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    self.assertEqual(len(result), 3)
    self.assertIn("reward", result)
    self.assertIn("reward_safety", result)
    self.assertIn("reward_helpfulness", result)
    np.testing.assert_array_equal(result["reward"], jnp.array([0.9, 0.8]))
    np.testing.assert_array_equal(result["reward_safety"], jnp.array([0.8, 0.7]))
    np.testing.assert_array_equal(result["reward_helpfulness"], jnp.array([0.6, 0.5]))

  def test_get_all_rewards_no_reward_models(self):
    """Test get_all_rewards raises error when no reward models exist."""
    models = {"critic": MockModel(self.rngs, 0.5)}
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_all_rewards(
          self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
      )
    self.assertIn("No reward models are available", str(cm.exception))

  def test_get_available_reward_models(self):
    """Test get_available_reward_models returns correct model names."""
    models = {
        "critic": MockModel(self.rngs, 0.5),
        "reward": MockModel(self.rngs, 0.9),
        "reward_safety": MockModel(self.rngs, 0.8),
        "reward_helpfulness": MockModel(self.rngs, 0.6),
        "reference": MockModel(self.rngs, 0.7)
    }
    worker = inference_worker.InferenceWorker(models)
    
    available_models = worker.get_available_reward_models()
    
    expected = ["reward", "reward_safety", "reward_helpfulness"]
    self.assertEqual(set(available_models), set(expected))

  def test_get_available_reward_models_empty(self):
    """Test get_available_reward_models returns empty list when no reward models."""
    models = {"critic": MockModel(self.rngs, 0.5)}
    worker = inference_worker.InferenceWorker(models)
    
    available_models = worker.get_available_reward_models()
    
    self.assertEqual(available_models, [])

  @mock.patch('tunix.rl.common.compute_per_token_logps')
  def test_get_ref_per_token_logps(self, mock_compute_logps):
    """Test get_ref_per_token_logps with reference model."""
    mock_compute_logps.return_value = (jnp.array([0.1, 0.2]), None)
    
    models = {"reference": MockModel(self.rngs, 0.7)}
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_ref_per_token_logps(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    mock_compute_logps.assert_called_once()
    np.testing.assert_array_equal(result, jnp.array([0.1, 0.2]))

  def test_get_ref_per_token_logps_no_reference_model(self):
    """Test get_ref_per_token_logps raises error when no reference model."""
    models = {"reward": MockModel(self.rngs, 0.9)}
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_ref_per_token_logps(
          self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
      )
    self.assertIn("Reference model is not available", str(cm.exception))

  @mock.patch('tunix.rl.common.compute_score')
  def test_get_values(self, mock_compute_score):
    """Test get_values with critic model."""
    mock_compute_score.return_value = jnp.array([0.5, 0.4])
    
    models = {"critic": MockModel(self.rngs, 0.5)}
    worker = inference_worker.InferenceWorker(models)
    
    result = worker.get_values(
        self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
    )
    
    mock_compute_score.assert_called_once()
    np.testing.assert_array_equal(result, jnp.array([0.5, 0.4]))

  def test_get_values_no_critic_model(self):
    """Test get_values raises error when no critic model."""
    models = {"reward": MockModel(self.rngs, 0.9)}
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_values(
          self.prompt_tokens, self.completion_tokens, self.pad_id, self.eos_id
      )
    self.assertIn("Critic model is not available", str(cm.exception))

  def test_get_model(self):
    """Test get_model returns correct model."""
    models = {
        "critic": MockModel(self.rngs, 0.5),
        "reward": MockModel(self.rngs, 0.9)
    }
    worker = inference_worker.InferenceWorker(models)
    
    critic_model = worker.get_model("critic")
    reward_model = worker.get_model("reward")
    
    self.assertEqual(critic_model, models["critic"])
    self.assertEqual(reward_model, models["reward"])

  def test_get_model_invalid_role(self):
    """Test get_model raises error for invalid role."""
    models = {"critic": MockModel(self.rngs, 0.5)}
    worker = inference_worker.InferenceWorker(models)
    
    with self.assertRaises(ValueError) as cm:
      worker.get_model("nonexistent")
    self.assertIn("nonexistent", str(cm.exception))
    self.assertIn("not available", str(cm.exception))

  def test_update_model(self):
    """Test update_model updates model parameters."""
    models = {"reward": MockModel(self.rngs, 0.9)}
    worker = inference_worker.InferenceWorker(models)
    
    # Get original state
    original_state = nnx.state(models["reward"])
    original_kernel = original_state.dense.kernel.value
    
    # Create new state with explicitly different values
    new_model = MockModel(nnx.Rngs(99), 0.5)  # Different seed
    new_state = nnx.state(new_model)
    
    # Manually modify the kernel to ensure it's different
    new_state.dense.kernel.value = jnp.ones_like(original_kernel) * 999.0
    
    # Update the model
    worker.update_model("reward", new_state)
    
    # Verify the update happened
    updated_model = worker.get_model("reward")
    updated_state = nnx.state(updated_model)
    updated_kernel = updated_state.dense.kernel.value

    # Check that the model instance is the same
    self.assertEqual(updated_model, models["reward"])

    # Check that the parameters were actually updated
    self.assertFalse(
        jnp.allclose(original_kernel, updated_kernel),
        "Model parameters should be different after update"
    )

  def test_update_model_invalid_role(self):
    """Test update_model raises error for invalid role."""
    models = {"reward": MockModel(self.rngs, 0.9)}
    worker = inference_worker.InferenceWorker(models)
    
    new_model = MockModel(nnx.Rngs(24), 0.5)
    new_state = nnx.state(new_model)
    
    with self.assertRaises(ValueError) as cm:
      worker.update_model("nonexistent", new_state)
    self.assertIn("nonexistent", str(cm.exception))
    self.assertIn("not available", str(cm.exception))

  @parameterized.named_parameters(
      ("reward_only", {"reward": True}),
      ("reward_with_suffix", {"reward_model1": True, "reward_safety": True}),
      ("mixed_models", {"critic": True, "reference": True, "reward": True, "reward_aux": True}),
  )
  def test_constructor_accepts_valid_reward_naming_patterns(self, model_names):
    """Test constructor accepts various valid reward model naming patterns."""
    models = {}
    for name in model_names:
      models[name] = MockModel(self.rngs, 0.5)
    
    # Should not raise an exception
    worker = inference_worker.InferenceWorker(models)
    self.assertIsNotNone(worker)


if __name__ == "__main__":
  absltest.main()
