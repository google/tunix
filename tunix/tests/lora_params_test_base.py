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

"""Base test class for LoRA merged model saving and loading tests.

This module provides a common base class for testing LoRA merged model
functionality across different model architectures (Qwen3, Gemma3, Llama3,
etc.).
It handles shared test logic including:
- Test setup/teardown (temp directories, config)
- Base model creation
- LoRA application and weight initialization
- Expected merged weight computation
- Common test method templates

Architecture-specific implementations should:
1. Inherit from LoraParamsTestBase
2. Implement abstract methods for checkpoint creation and model loading
3. Optionally override methods for architecture-specific behavior
"""

import abc
import os
import shutil
import tempfile

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import safetensors.numpy as safe_np
from tunix.models import dummy_model_creator

create_dummy_model = dummy_model_creator.create_dummy_model


class LoraParamsTestBase(absltest.TestCase, abc.ABC):
  """Base test class for LoRA merged model saving and loading.

  Subclasses must implement:
  - create_config(): Create model-specific configuration
  - create_checkpoint(): Create safetensors checkpoint from model
  - get_lora_module_path(): Return regex pattern for LoRA target modules
  - get_projection_keys(): Return list of projection keys to verify
  - save_merged_model(): Call model-specific save function
  - create_model_from_checkpoint(): Load model from safetensors (optional)
  """

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()

    # LoRA hyperparameters
    self.rank = 4
    self.alpha = 16
    self.random_seed = 42
    self.lora_scale = 0.01
    self.base_model_scale = 0.02

    # Create temporary directories
    self.temp_dir = tempfile.mkdtemp()
    self.base_checkpoint_dir = os.path.join(self.temp_dir, "base")
    self.merged_output_dir = os.path.join(self.temp_dir, "merged")

    # Create model config (architecture-specific)
    self.config = self.create_config()

  def tearDown(self):
    """Clean up test fixtures."""
    if os.path.exists(self.temp_dir):
      shutil.rmtree(self.temp_dir)
    super().tearDown()

  @abc.abstractmethod
  def create_config(self):
    """Create model-specific configuration.

    Returns:
      Model configuration object with small dimensions for fast testing.
    """
    pass

  @abc.abstractmethod
  def create_checkpoint(self, model) -> str:
    """Extract model weights and save in safetensors format.

    This method should:
    1. Create checkpoint directory if needed
    2. Extract weights from model
    3. Apply architecture-specific transformations
    4. Save to safetensors file
    5. Create minimal config.json

    Args:
      model: Base model to extract weights from.

    Returns:
      Path to the created checkpoint directory.
    """
    pass

  @abc.abstractmethod
  def get_lora_module_path(self) -> str:
    """Get regex pattern for LoRA target modules.

    Returns:
      Regex pattern string for qwix.LoraProvider.
      Example:
      ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    """
    pass

  @abc.abstractmethod
  def get_projection_keys(self, layer_idx: int) -> list[str]:
    """Get list of projection weight keys for a given layer.

    Args:
      layer_idx: Layer index.

    Returns:
      List of safetensors keys for projections in this layer.
      Example: ["model.layers.0.self_attn.q_proj.weight", ...]
    """
    pass

  @abc.abstractmethod
  def save_merged_model(self, lora_model):
    """Save LoRA merged model using architecture-specific function.

    Args:
      lora_model: Model with LoRA applied.
    """
    pass

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    """Load model from safetensors checkpoint (optional).

    Override this method to enable forward pass equivalence testing.

    Args:
      checkpoint_dir: Path to checkpoint directory.

    Returns:
      Loaded model, or None if not implemented.
    """
    return None

  def get_model_class(self):
    """Get model class for dummy model creation.

    Override if needed for architecture-specific model class.

    Returns:
      Model class to use with create_dummy_model.
    """
    # Default implementation - subclasses should override
    raise NotImplementedError("Subclass must implement get_model_class()")

  def _create_base_model(self):
    """Create a base model with deterministic weights."""
    model = create_dummy_model(
        model_class=self.get_model_class(),
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
        random_seed=self.random_seed,
        scale=self.base_model_scale,
    )
    return model

  def _apply_lora_to_model(self, model):
    """Apply LoRA to a model.

    Args:
      model: Base model to apply LoRA to.

    Returns:
      Model with LoRA applied and initialized with random weights.
    """
    # Create LoRA provider
    lora_provider = qwix.LoraProvider(
        module_path=self.get_lora_module_path(),
        rank=self.rank,
        alpha=self.alpha,
    )

    # Get model input and apply LoRA
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(model, lora_provider, **model_input)

    # Initialize LoRA weights
    self._initialize_lora_weights(lora_model)

    return lora_model

  def _initialize_lora_weights(self, lora_model):
    """Initialize LoRA weights with random values.

    Override this method for architecture-specific weight initialization.

    Args:
      lora_model: Model with LoRA applied.
    """
    lora_rng = jax.random.PRNGKey(self.random_seed + 1000)

    for layer in lora_model.layers:
      # Process all attributes in the layer
      for attr_name in ["attn", "mlp"]:
        if not hasattr(layer, attr_name):
          continue

        module = getattr(layer, attr_name)

        # Find all projections in this module
        for proj_name in dir(module):
          proj = getattr(module, proj_name, None)
          if proj is None:
            continue

          # Initialize w_lora_a and w_lora_b (for attention projections)
          if hasattr(proj, "w_lora_a"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.w_lora_a.value = self.lora_scale * jax.random.normal(
                key, proj.w_lora_a.value.shape
            )
          if hasattr(proj, "w_lora_b"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.w_lora_b.value = self.lora_scale * jax.random.normal(
                key, proj.w_lora_b.value.shape
            )

          # Initialize kernel_lora_a and kernel_lora_b (for MLP projections)
          if hasattr(proj, "kernel_lora_a"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.kernel_lora_a.value = self.lora_scale * jax.random.normal(
                key, proj.kernel_lora_a.value.shape
            )
          if hasattr(proj, "kernel_lora_b"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.kernel_lora_b.value = self.lora_scale * jax.random.normal(
                key, proj.kernel_lora_b.value.shape
            )

  def _compute_expected_merged_weight(self, base_weight, lora_a, lora_b):
    """Compute expected merged weight: base + (lora_a @ lora_b) * (alpha/rank).

    Args:
      base_weight: Base weight tensor.
      lora_a: LoRA A matrix.
      lora_b: LoRA B matrix.

    Returns:
      Expected merged weight.
    """
    lora_a = np.array(lora_a)
    lora_b = np.array(lora_b)

    # Handle 3D tensors (flatten to 2D for matmul)
    if lora_a.ndim == 3:
      d0, d1, d2 = lora_a.shape
      lora_a = lora_a.reshape(d0 * d1, d2)
    if lora_b.ndim == 3:
      d0, d1, d2 = lora_b.shape
      lora_b = lora_b.reshape(d0, d1 * d2)

    # Compute LoRA delta
    lora_delta = (lora_a @ lora_b) * (self.alpha / self.rank)

    # Merge with base weight
    expected = np.array(base_weight) + lora_delta.T

    return expected

  # ============================================================================
  # Common test methods
  # ============================================================================

  def test_save_lora_merged_model(self):
    """Test LoRA merged model saving and output file creation."""
    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA
    lora_model = self._apply_lora_to_model(base_model)

    # Verify LoRA was applied
    lora_param_count = 0
    for _, value in nnx.iter_graph(lora_model):
      if isinstance(value, nnx.LoRAParam):
        lora_param_count += 1
    self.assertGreater(
        lora_param_count, 0, "LoRA should be applied to the model"
    )

    # Save merged model
    self.save_merged_model(lora_model)

    # Verify output files exist
    self.assertTrue(os.path.exists(self.merged_output_dir))
    merged_safetensors_path = os.path.join(
        self.merged_output_dir, "model.safetensors"
    )
    self.assertTrue(os.path.exists(merged_safetensors_path))

    # Verify merged state is non-empty
    merged_state = safe_np.load_file(merged_safetensors_path)
    self.assertGreater(len(merged_state), 0)

    # Verify config was copied
    config_path = os.path.join(self.merged_output_dir, "config.json")
    self.assertTrue(os.path.exists(config_path))

  def test_all_projections_merged(self):
    """Test that all projection types are present in merged state."""
    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA and save merged model
    lora_model = self._apply_lora_to_model(base_model)
    self.save_merged_model(lora_model)

    # Load merged state
    merged_state = safe_np.load_file(
        os.path.join(self.merged_output_dir, "model.safetensors")
    )

    # Verify all projection keys exist
    projection_keys = self.get_projection_keys(layer_idx=0)
    for key in projection_keys:
      self.assertIn(key, merged_state, f"Missing projection key: {key}")
      self.assertIsNotNone(merged_state[key])
      self.assertGreater(merged_state[key].size, 0)

  def test_forward_pass_equivalence(self):
    """Test that outputs match between LoRA model and merged-then-reloaded model.

    This test is optional and requires implementing
    create_model_from_checkpoint().
    """
    # Skip if not implemented
    if (
        self.create_model_from_checkpoint.__func__
        is LoraParamsTestBase.create_model_from_checkpoint
    ):
      self.skipTest("create_model_from_checkpoint not implemented")

    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA
    lora_model = self._apply_lora_to_model(base_model)

    # Create test inputs
    input_tokens, positions, attention_mask = self._create_test_inputs()

    # Get output from LoRA model
    lora_output, _ = self._run_forward_pass(
        lora_model, input_tokens, positions, attention_mask
    )

    # Save merged model
    self.save_merged_model(lora_model)

    # Reload merged model
    reloaded_model = self.create_model_from_checkpoint(self.merged_output_dir)

    # Get output from reloaded model
    merged_output, _ = self._run_forward_pass(
        reloaded_model, input_tokens, positions, attention_mask
    )

    # Compare outputs
    np.testing.assert_allclose(
        lora_output,
        merged_output,
        atol=1e-2,
        err_msg=(
            "Forward pass outputs don't match between LoRA model and merged"
            " model"
        ),
    )

  def _create_test_inputs(self):
    """Create test inputs for forward pass.

    Override this method for architecture-specific input requirements.

    Returns:
      Tuple of (input_tokens, positions, attention_mask).
    """
    batch_size = 2
    seq_len = 10

    input_tokens = jax.random.randint(
        jax.random.PRNGKey(123),
        shape=(batch_size, seq_len),
        minval=0,
        maxval=self.config.vocab_size
        if hasattr(self.config, "vocab_size")
        else self.config.num_embed,
    )
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    attention_mask = None  # Override if needed

    return input_tokens, positions, attention_mask

  def _run_forward_pass(self, model, input_tokens, positions, attention_mask):
    """Run forward pass through model.

    Override this method for architecture-specific forward pass parameters.

    Args:
      model: Model to run.
      input_tokens: Input token IDs.
      positions: Position IDs.
      attention_mask: Attention mask (or None).

    Returns:
      Tuple of (output, cache).
    """
    return model(
        input_tokens=input_tokens,
        positions=positions,
        cache=None,
        attention_mask=attention_mask,
    )
