# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
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

from absl.testing import parameterized
from absl.testing import absltest
from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from unittest import mock
from tunix.generate import utils
from tunix.rl import reshard


PartitionSpec = sharding.PartitionSpec
NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh


class MockState:

  def __init__(self, params):
    self.params = params

  def flat_state(self):
    return [(tuple(k.split(".")), v) for k, v in self.params.items()]

  def from_flat_path(self, flat_path):
    new_params = {}
    for keys, param in flat_path:
      new_params[".".join(keys)] = (
          param if hasattr(param, "value") else MockParam(param)
      )
    return MockState(new_params)


class MockParam:

  def __init__(self, value):
    self.value = value

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype

  @property
  def ndim(self):
    return self.value.ndim

  @property
  def sharding(self):
    return self.value.sharding

  def __getitem__(self, item):
    return self.value[item]

  def __array__(self, dtype=None):
    return np.asarray(self.value, dtype=dtype)

  def __jax_array__(self):
    return self.value


class Logprob:

  def __init__(self, logprob, rank=None, decoded_token=None):
    self.logprob = logprob
    self.rank = rank
    self.decoded_token = decoded_token


class UtilsTest(parameterized.TestCase):

  def test_compute_attention_mask(self):
    # Check that the input mask is correctly applied when total sampling steps
    # is lower than the max cache length.
    input_mask = jnp.array([[1, 1, 0, 0, 0], [1, 1, 0, 1, 0]], dtype=jnp.bool_)
    seq_len = 8
    time_step = 4
    attn_mask = utils.compute_attention_masks(time_step, seq_len, input_mask)
    expected_attn_mask = jnp.array(
        [[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())

    # Check that the input mask is correctly applied when total sampling steps
    # is *longer* than the max cache length.
    seq_len = 4
    time_step = 4
    attn_mask = utils.compute_attention_masks(time_step, seq_len, input_mask)
    expected_attn_mask = jnp.array(
        [[0, 1, 1, 1], [0, 1, 0, 1]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array([[0, 1, 1, 0], [1, 1, 1, 0]])
    attn_mask = utils.make_causal_attn_mask(input_mask, 5)
    expected = jnp.array([
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
    ])
    np.testing.assert_array_equal(attn_mask, expected)

  def test_next_power_of_2(self):
    self.assertEqual(utils.next_power_of_2(0), 1)
    self.assertEqual(utils.next_power_of_2(2), 2)
    self.assertEqual(utils.next_power_of_2(3), 4)
    self.assertEqual(utils.next_power_of_2(4), 4)
    self.assertEqual(utils.next_power_of_2(5), 8)

  def test_find_first_non_pad_idx(self):
    data = [
        ([1, 2, 3, 4, 5, 6], 0),
        ([0, 0, 1, 2, 3, 4], 2),
        ([0, 1, 2, 3, 0, 0], 1),
    ]
    for ids, expected in data:
      self.assertEqual(
          utils.find_first_non_pad_idx(jnp.array(ids), 0), expected
      )

  def test_find_first_eos_idx(self):
    data = [
        ([1, 2, 3, 4, 5, -1], 5),
        ([1, 2, -1, 4, -1, 0], 2),
        ([1, 2, 3, 4, 5, 6], 6),
    ]
    for ids, expected in data:
      self.assertEqual(utils.find_first_eos_idx(jnp.array(ids), -1), expected)

  def test_find_last_non_pad_idx(self):
    data = [
        ([1, 2, 3, 4, 5, 6], 5),
        ([1, 2, 3, 0, 0, 0], 2),
        ([0, 1, 2, 3, 0, 0], 3),
    ]
    for ids, expected in data:
      self.assertEqual(utils.find_last_non_pad_idx(jnp.array(ids), 0), expected)

  def test_logprobs_basic_extraction(self):
    token_ids = [271, 567, 15166]
    logprobs = [
        {271: Logprob(-1.71), 198: Logprob(-0.52)},
        {567: Logprob(-0.37)},
        {15166: Logprob(0.0)},
    ]
    expected = [-1.71, -0.37, 0.0]
    self.assertEqual(
        utils.get_logprobs_from_vllm_output(token_ids, logprobs),  # pyrefly: ignore[bad-argument-type]
        expected,
    )

  def test_logprobs_extraction_with_missing_token(self):
    token_ids = [100, 200]
    logprobs = [{101: Logprob(-0.5)}, {200: Logprob(-1.2)}]
    with self.assertRaises(ValueError):
      utils.get_logprobs_from_vllm_output(token_ids, logprobs)  # pyrefly: ignore[bad-argument-type]

  @parameterized.named_parameters(
      ("none_logprobs", [], None),
      ("empty_logprobs", [], []),
      ("list_of_none_logprobs", [1], [None]),
  )
  def test_logprobs_empty_cases(self, token_ids, logprobs):
    self.assertEqual(
        utils.get_logprobs_from_vllm_output(token_ids, logprobs),
        [],
    )

  def test_transfer_state_with_mappings_tranpose_and_sharding_device(self):
    device_count = len(jax.devices())
    if device_count < 2 or device_count % 2 != 0:
      self.skipTest("This example assumes even number of devices >= 2")

    devices_array = np.array(jax.devices()).reshape((device_count // 2, 2))
    mesh = Mesh(devices_array, axis_names=("data", "model"))

    src_sharding = NamedSharding(mesh, PartitionSpec(None, "model"))
    tgt_sharding = NamedSharding(mesh, PartitionSpec("data", "model"))
    src_state = MockState({
        "encoder.layer_0.weight": MockParam(
            jax.device_put(
                jnp.arange(16).reshape(2, 8).astype(jnp.float32),
                device=src_sharding,
            ),
        ),
        "encoder.layer_1.weight": MockParam(
            jax.device_put(
                jnp.arange(16, 32).reshape(2, 8).astype(jnp.float32),
                device=src_sharding,
            ),
        ),
    })
    tgt_state = MockState({
        "decoder.layer_0.weight": MockParam(
            jax.device_put(
                jnp.zeros((8, 2), dtype=jnp.float32), device=tgt_sharding
            ),
        ),
        "encoder.layer_0.weight": MockParam(
            jax.device_put(
                jnp.zeros((8, 2), dtype=jnp.float32), device=tgt_sharding
            ),
        ),
    })
    mappings = {
        "encoder.layer_0.weight": ("decoder.layer_0.weight", None),
        "encoder.layer_1.weight": ("encoder.layer_0.weight", None),
    }
    transpose_keys = {
        "weight": (1, 0),
    }
    hook_fns = {
        "encoder.layer_0.weight": lambda x: x * 2,
    }

    new_tgt_state = utils.transfer_state_with_mappings(
        src_state,
        tgt_state,
        key_mappings=mappings,
        key_mapping_hook_fns=hook_fns,
        transpose_keys=transpose_keys,
        reshard_fn=reshard.reshard_pytree,
    )

    expected_layer_0_weight = jnp.arange(16).reshape(2, 8).T * 2
    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["decoder.layer_0.weight"].value,
            expected_layer_0_weight,
        )
    )
    expected_layer_1_weight = jnp.arange(16, 32).reshape(2, 8).T
    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["encoder.layer_0.weight"].value,
            expected_layer_1_weight,
        )
    )
    self.assertEqual(
        new_tgt_state.params["decoder.layer_0.weight"].sharding, tgt_sharding
    )
    self.assertEqual(
        new_tgt_state.params["encoder.layer_0.weight"].sharding, tgt_sharding
    )

  def test_transfer_state_with_padding(self):
    # Create source module with smaller head dim
    src = MockState(
        {"decoder.layers.5.attn.k_proj": MockParam(jnp.ones((2, 4, 64)))}
    )
    dst = MockState(
        {"decoder.layers.5.attn.k_proj": MockParam(jnp.zeros((2, 4, 128)))}
    )

    mappings = {
        "decoder.layers.5.attn.k_proj": ("decoder.layers.5.attn.k_proj", None),
    }

    new_tgt_state = utils.transfer_state_with_mappings(src, dst, mappings)

    # Validate shape
    self.assertEqual(
        new_tgt_state.params["decoder.layers.5.attn.k_proj"].shape, (2, 4, 128)
    )
    # Validate original values copied correctly
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layers.5.attn.k_proj"][:, :, :64], 1.0
        )
    )
    # Validate padded values are zero
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layers.5.attn.k_proj"][:, :, 64:], 0.0
        )
    )

  def test_transfer_state_with_bias_padding_and_reshape(self):
    """Test rank mismatch, reshape and padding for attention bias."""
    src_key = "layers.0.attn.q_bias"
    src_q_bias = jnp.ones((256,), dtype=jnp.float32)
    src = MockState({src_key: MockParam(src_q_bias)})
    dst = MockState(
        {src_key: MockParam(jnp.zeros((4, 128), dtype=jnp.float32))}
    )

    mappings = {src_key: (src_key, None)}

    result = utils.transfer_state_with_mappings(
        src, dst, mappings, num_kv_heads=2, head_dim=128
    )

    # Verify shape
    self.assertEqual(result.params[src_key].shape, (4, 128))
    # Verify values are repeated correctly
    self.assertTrue(jnp.allclose(result.params[src_key].value, 1.0))

  def test_transfer_state_with_scanned_layers(self):
    """Comprehensive test for scanned layers covering multiple scenarios."""
    num_layers = 3
    embed_dim = 4
    vocab_size = 8
    batch_size = 2

    # Create source state with multiple types of parameters:
    # 1. Scanned weights (layer dim on axis 0)
    # 2. Scanned biases (layer dim on axis 1)
    # 3. Regular embedding - no scanning, direct transfer

    # Scanned weights: shape (num_layers, embed_dim, vocab_size)
    scanned_weights = jnp.stack(
        [
            jnp.full((embed_dim, vocab_size), i + 1, dtype=jnp.float32)
            for i in range(num_layers)
        ],
        axis=0,
    )

    # Scanned biases with layer dim on axis 1:
    # shape (batch_size, num_layers, vocab_size)
    scanned_biases = jnp.stack(
        [
            jnp.full((batch_size, vocab_size), (i + 1) * 10, dtype=jnp.float32)
            for i in range(num_layers)
        ],
        axis=1,
    )

    # Regular parameter (no scanning)
    embedding_weights = jnp.full(
        (vocab_size, embed_dim), 99.0, dtype=jnp.float32
    )

    src_state = MockState({
        "transformer.layers.weight": MockParam(
            scanned_weights
        ),  # Scanned on axis 0
        "transformer.layers.bias": MockParam(
            scanned_biases
        ),  # Scanned on axis 1
        "embedding.weight": MockParam(embedding_weights),  # Regular parameter
    })

    # Create target state with individual layer parameters
    target_params = {
        "embedding.weight": MockParam(
            jnp.zeros((embed_dim, vocab_size), dtype=jnp.float32)
        )
    }

    # Individual layer parameters for scanned weights and biases
    for i in range(num_layers - 1, -1, -1):
      target_params[f"decoder.layer.{i}.weight"] = MockParam(
          jnp.zeros(
              (vocab_size, embed_dim), dtype=jnp.float32
          )  # Transposed shape
      )
      target_params[f"decoder.layer.{i}.bias"] = MockParam(
          jnp.zeros((batch_size, vocab_size), dtype=jnp.float32)
      )

    tgt_state = MockState(target_params)

    # Define mappings for all parameter types
    mappings = {
        # Scanned weight with layer on axis 0, target needs transpose
        "transformer.layers.weight": (
            "decoder.layer.*.weight",
            ("layer", None, None),
        ),
        # Scanned bias with layer on axis 1
        "transformer.layers.bias": (
            "decoder.layer.*.bias",
            (None, "layer", None),
        ),
        # Regular parameter that needs transpose
        "embedding.weight": ("embedding.weight", None),
    }

    # Define transpose operations
    transpose_keys = {"weight": (1, 0)}  # Transpose weight matrices

    # Perform the transfer
    new_tgt_state = utils.transfer_state_with_mappings(
        src_state,
        tgt_state,
        key_mappings=mappings,
        transpose_keys=transpose_keys,
    )

    # Verify scanned weights (axis 0) with transpose
    for layer_idx in range(num_layers):
      layer_key = f"decoder.layer.{layer_idx}.weight"
      transferred = new_tgt_state.params[layer_key]

      self.assertEqual(transferred.shape, (vocab_size, embed_dim))
      self.assertTrue(
          jnp.allclose(
              transferred.value,
              jnp.full(
                  (vocab_size, embed_dim), layer_idx + 1, dtype=jnp.float32
              ),
          ),
          f"Scanned weight layer {layer_idx} mismatch",
      )

    # Verify scanned biases (axis 1) - no transpose
    for layer_idx in range(num_layers):
      layer_key = f"decoder.layer.{layer_idx}.bias"
      transferred = new_tgt_state.params[layer_key]

      # Expected: extract layer from axis 1
      expected = jnp.full(
          (batch_size, vocab_size), (layer_idx + 1) * 10, dtype=jnp.float32
      )

      self.assertEqual(transferred.shape, (batch_size, vocab_size))
      self.assertTrue(
          jnp.allclose(transferred.value, expected),
          f"Scanned bias layer {layer_idx} mismatch",
      )

    # Verify regular parameter with transpose
    transferred_embedding = new_tgt_state.params["embedding.weight"]

    self.assertEqual(transferred_embedding.shape, (embed_dim, vocab_size))
    self.assertTrue(
        jnp.allclose(
            transferred_embedding.value,
            jnp.full((embed_dim, vocab_size), 99.0, dtype=jnp.float32),
        ),
        "Regular parameter with transpose mismatch",
    )

  def test_transfer_state_with_mappings_gemma4(self):
    """Test transfer_state_with_mappings for Gemma4."""
    from tunix.models.gemma4.mapping_vllm_jax import VLLM_JAX_MAPPING

    # Mock source state (Tunix style)
    src_params = {
        "layers.0.attn.q_einsum.w": MockParam(
            jnp.arange(4 * 16 * 8, dtype=jnp.float32).reshape(4, 16, 8)
        ),
        "layers.0.attn.kv_einsum.w": MockParam(
            jnp.arange(2 * 2 * 16 * 8, dtype=jnp.float32).reshape(2, 2, 16, 8)
        ),
        "layers.0.mlp.gate_proj.kernel": MockParam(
            jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)
        ),
        "layers.0.mlp.up_proj.kernel": MockParam(
            jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)
        ),
        "layers.0.moe.gating_einsum": MockParam(
            jnp.arange(4 * 2 * 8 * 16, dtype=jnp.float32).reshape(4, 2, 8, 16)
        ),
        "layers.0.moe.linear": MockParam(
            jnp.arange(4 * 16 * 8, dtype=jnp.float32).reshape(4, 16, 8)
        ),
    }
    src_state = MockState(src_params)

    # Mock destination state (vLLM Jax backend style)
    dst_params = {
        "model.layers.0.self_attn.qkv_proj.weight": MockParam(
            jnp.zeros((16, 64), dtype=jnp.float32)
        ),
        "model.layers.0.mlp.gate_up_proj.weight": MockParam(
            jnp.zeros((16, 64), dtype=jnp.float32)
        ),
        "model.layers.0.experts.kernel_gating_upproj_EDF": MockParam(
            jnp.zeros((4, 2, 8, 16), dtype=jnp.float32)
        ),
        "model.layers.0.experts.kernel_down_proj_EFD": MockParam(
            jnp.zeros((4, 16, 8), dtype=jnp.float32)
        ),
    }
    dst_state = MockState(dst_params)

    # Apply preprocessing if it exists in mapping
    if 'preprocess_src_state' in VLLM_JAX_MAPPING:
      src_state = VLLM_JAX_MAPPING['preprocess_src_state'](src_state)

    key_mappings = VLLM_JAX_MAPPING['to_hf_mappings']
    transpose_keys = VLLM_JAX_MAPPING['to_hf_transpose_keys']

    new_tgt_state = utils.transfer_state_with_mappings(
        src_state,
        dst_state,
        key_mappings=key_mappings,
        transpose_keys=transpose_keys,
    )

    # Assertions
    q_val = jnp.arange(4 * 16 * 8, dtype=jnp.float32).reshape(4, 16, 8)
    kv_val = jnp.arange(2 * 2 * 16 * 8, dtype=jnp.float32).reshape(2, 2, 16, 8)
    k_val = kv_val[0]
    v_val = kv_val[1]

    q_val_t = jnp.reshape(jnp.transpose(q_val, (1, 0, 2)), (16, -1))
    k_val_t = jnp.reshape(jnp.transpose(k_val, (1, 0, 2)), (16, -1))
    v_val_t = jnp.reshape(jnp.transpose(v_val, (1, 0, 2)), (16, -1))

    expected_qkv = jnp.concatenate([q_val_t, k_val_t, v_val_t], axis=-1)

    gate_val = jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)
    up_val = jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32)
    expected_gate_up = jnp.concatenate([gate_val, up_val], axis=-1)

    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["model.layers.0.self_attn.qkv_proj.weight"],
            expected_qkv,
        )
    )
    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["model.layers.0.mlp.gate_up_proj.weight"],
            expected_gate_up,
        )
    )

    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["model.layers.0.experts.kernel_gating_upproj_EDF"],
            src_params["layers.0.moe.gating_einsum"].value,
        )
    )

    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["model.layers.0.experts.kernel_down_proj_EFD"],
            src_params["layers.0.moe.linear"].value,
        )
    )

  def test_verify_state_closeness(self):
    """Test verify_state_closeness function with various scenarios."""

    # Test case 1: Identical states should return True
    identical_params = {
        "layer.0.weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer.0.bias": jnp.array([0.1, 0.2]),
        "layer.1.weight": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
    }
    golden_state_identical = MockState(
        {k: MockParam(v) for k, v in identical_params.items()}
    )
    test_state_identical = MockState(
        {k: MockParam(v) for k, v in identical_params.items()}
    )

    self.assertTrue(
        utils.verify_state_closeness(
            golden_state_identical, test_state_identical
        )
    )

    # Test case 2: States with values within tolerance should return True
    golden_params = {
        "layer.0.weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer.0.bias": jnp.array([0.1, 0.2]),
    }
    close_params = {
        "layer.0.weight": jnp.array(
            [[1.005, 2.003], [3.001, 4.002]]
        ),  # Within default atol=1e-2
        "layer.0.bias": jnp.array([0.105, 0.198]),
    }
    golden_state_close = MockState(
        {k: MockParam(v) for k, v in golden_params.items()}
    )
    test_state_close = MockState(
        {k: MockParam(v) for k, v in close_params.items()}
    )

    self.assertTrue(
        utils.verify_state_closeness(
            golden_state_close, test_state_close, atol=1e-2
        )
    )

    # Test case 3: States with values outside tolerance should return False
    far_params = {
        "layer.0.weight": jnp.array(
            [[1.05, 2.03], [3.01, 4.02]]
        ),  # Outside default atol=1e-2
        "layer.0.bias": jnp.array([0.15, 0.25]),
    }
    test_state_far = MockState({k: MockParam(v) for k, v in far_params.items()})

    self.assertFalse(
        utils.verify_state_closeness(
            golden_state_close, test_state_far, atol=1e-2
        )
    )

    # Test case 4: Different keys should return False
    different_keys_params = {
        "layer.0.weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer.0.different_bias": jnp.array([0.1, 0.2]),  # Different key name
    }
    test_state_diff_keys = MockState(
        {k: MockParam(v) for k, v in different_keys_params.items()}
    )

    self.assertFalse(
        utils.verify_state_closeness(golden_state_close, test_state_diff_keys)
    )

    # Test case 5: Missing keys should return False

    # Missing "layer.0.bias"
    missing_key_params = {"layer.0.weight": jnp.array([[1.0, 2.0], [3.0, 4.0]])}

    test_state_missing = MockState(
        {k: MockParam(v) for k, v in missing_key_params.items()}
    )

    self.assertFalse(
        utils.verify_state_closeness(golden_state_close, test_state_missing)
    )

    # Test case 6: Custom tolerance should work
    custom_tolerance_params = {
        "layer.0.weight": jnp.array(
            [[1.08, 2.07], [3.06, 4.05]]
        ),  # Within atol=0.1
        "layer.0.bias": jnp.array([0.18, 0.27]),
    }
    test_state_custom_tol = MockState(
        {k: MockParam(v) for k, v in custom_tolerance_params.items()}
    )

    # Should fail with default tolerance
    self.assertFalse(
        utils.verify_state_closeness(golden_state_close, test_state_custom_tol)
    )

    # Should pass with custom tolerance
    self.assertTrue(
        utils.verify_state_closeness(
            golden_state_close, test_state_custom_tol, atol=0.1
        )
    )

    # Test case 7: Empty states should return True
    empty_golden = MockState({})
    empty_test = MockState({})

    self.assertTrue(utils.verify_state_closeness(empty_golden, empty_test))

    # Test case 8: Different shapes should return False
    different_shape_params = {
        "layer.0.weight": jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ),  # Different shape
        "layer.0.bias": jnp.array([0.1, 0.2]),
    }
    test_state_diff_shape = MockState(
        {k: MockParam(v) for k, v in different_shape_params.items()}
    )

    self.assertFalse(
        utils.verify_state_closeness(golden_state_close, test_state_diff_shape)
    )

  def test_attention_weight_head_dim_padding(self):
    """Test padding head dimension (last axis) for attention weights."""
    # Source: (num_heads, head_dim=64)
    # Target: (num_heads, head_dim=128)
    src_q_proj = jnp.ones((8, 64), dtype=jnp.float32) * 2.0
    src = MockState({"transformer.layers.0.attn.q_proj": MockParam(src_q_proj)})
    dst = MockState({
        "transformer.layers.0.attn.q_proj": MockParam(
            jnp.zeros((8, 128), dtype=jnp.float32)
        )
    })

    mappings = {
        "transformer.layers.0.attn.q_proj": (
            "transformer.layers.0.attn.q_proj",
            None,
        )
    }

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    # Verify shape
    self.assertEqual(
        result.params["transformer.layers.0.attn.q_proj"].shape, (8, 128)
    )
    # Verify original values preserved
    self.assertTrue(
        jnp.allclose(
            result.params["transformer.layers.0.attn.q_proj"][:, :64], 2.0
        )
    )
    # Verify padded values are zero
    self.assertTrue(
        jnp.allclose(
            result.params["transformer.layers.0.attn.q_proj"][:, 64:], 0.0
        )
    )

  def test_attention_weight_num_heads_repetition(self):
    """Test repeating num_heads dimension (non-last axis) for attention weights."""
    # Source: (num_heads=4, seq_len=16, head_dim=64)
    # Target: (num_heads=8, seq_len=16, head_dim=64)
    src_k_proj = jnp.arange(4 * 16 * 64, dtype=jnp.float32).reshape(4, 16, 64)
    src_key = "base.decoder.layers.3.self_attention.key.kernel"
    dst_key = "model.layers.3.self_attn.k_proj.kernel"

    src = MockState({src_key: MockParam(src_k_proj)})
    dst = MockState(
        {dst_key: MockParam(jnp.zeros((8, 16, 64), dtype=jnp.float32))}
    )

    mappings = {src_key: (dst_key, None)}

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    # Verify shape
    self.assertEqual(result.params[dst_key].shape, (8, 16, 64))

    # Verify that heads are repeated
    self.assertTrue(
        jnp.allclose(result.params[dst_key][::2, ...], src_k_proj, atol=1e-1)
    )

    self.assertTrue(
        jnp.allclose(result.params[dst_key][1::2, ...], src_k_proj, atol=1e-1)
    )

  def test_non_attention_weight_padding_fails(self):
    """Test that padding non-attention weights raises an error."""
    # Try to pad an MLP weight (should fail)
    src_mlp = jnp.ones((256, 64), dtype=jnp.float32)
    src = MockState({"mlp.fc1.weight": MockParam(src_mlp)})
    dst = MockState(
        {"mlp.fc1.weight": MockParam(jnp.zeros((256, 128), dtype=jnp.float32))}
    )

    mappings = {"mlp.fc1.weight": ("mlp.fc1.weight", None)}

    with self.assertRaises(utils.ShapeMismatchError) as context:
      utils.transfer_state_with_mappings(src, dst, mappings)

    self.assertIn(
        "Padding/repetition only supported for attention weights",
        str(context.exception),
    )

  def test_attention_weight_invalid_repeat_factor(self):
    """Test that non-divisible repeat factors raise an error."""
    # Source: (num_heads=3, head_dim=64)
    # Target: (num_heads=8, head_dim=64) - 8 is not divisible by 3
    src = MockState(
        {"attn.k_proj": MockParam(jnp.ones((3, 64), dtype=jnp.float32))}
    )
    dst = MockState(
        {"attn.k_proj": MockParam(jnp.zeros((8, 64), dtype=jnp.float32))}
    )

    mappings = {"attn.k_proj": ("attn.k_proj", None)}

    with self.assertRaises(utils.ShapeMismatchError) as context:
      utils.transfer_state_with_mappings(src, dst, mappings)

    self.assertIn("not divisible", str(context.exception))

  def test_attention_weight_shrinking_fails(self):
    """Test that shrinking dimensions raises an error."""
    # Source: (num_heads=8, head_dim=128)
    # Target: (num_heads=4, head_dim=64) - cannot shrink
    src = MockState(
        {"attn.k_proj": MockParam(jnp.ones((8, 128), dtype=jnp.float32))}
    )
    dst = MockState(
        {"attn.k_proj": MockParam(jnp.zeros((4, 64), dtype=jnp.float32))}
    )

    mappings = {"attn.k_proj": ("attn.k_proj", None)}

    with self.assertRaises(utils.ShapeMismatchError) as context:
      utils.transfer_state_with_mappings(src, dst, mappings)

    self.assertIn("Cannot shrink", str(context.exception))

  def test_various_attention_key_patterns(self):
    """Test that various attention key naming patterns are recognized."""
    attention_keys = [
        "model.layers.0.self_attn.q_proj",
        "encoder.attention.key",
        "attention.value",
        "attention.query.weight",
        "decoder.blocks.3.self_attn.v_proj.kernel",
        "module.attention.o_proj",
    ]

    for key in attention_keys:
      src = MockState({key: MockParam(jnp.ones((4, 64), dtype=jnp.float32))})
      dst = MockState({key: MockParam(jnp.zeros((4, 128), dtype=jnp.float32))})
      mappings = {key: (key, None)}

      # Should not raise an error
      result = utils.transfer_state_with_mappings(src, dst, mappings)
      self.assertEqual(result.params[key].shape, (4, 128))

  def test_transfer_state_directly_simple_transfer(self):
    """Tests direct state transfer with matching structures."""
    src_state = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array([1.0, 2.0]))))
    )
    dst_state = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array([0.0, 0.0]))))
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    np.testing.assert_array_equal(
        dst_state['decoder']['layer0']['weight'][...],
        jnp.array([1.0, 2.0]),
    )

  def test_transfer_state_directly_unwraps_base_and_model(self):
    """Tests unwrapping of 'base' from src and 'model' from dst."""
    # Source has 'base' wrapper
    src_state = nnx.Dict(
        base=nnx.Dict(
            decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array(1.0))))
        )
    )
    # Dest has 'model' wrapper
    dst_state = nnx.Dict(
        model=nnx.Dict(
            decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array(0.0))))
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    np.testing.assert_array_equal(
        dst_state['model']['decoder']['layer0']['weight'][...],
        jnp.array(1.0),
    )

  def test_transfer_state_directly_intersects_keys(self):
    """Tests that only common keys are transferred; extras are ignored."""
    mock_reshard = lambda source, target: source

    # Scenario 1: Source has more keys than destination, with no unique keys in destination
    src_state_more_keys = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.array(1.0))),
            layer1=nnx.Dict(weight=nnx.Param(jnp.array(2.0))),
            layer2=nnx.Dict(weight=nnx.Param(jnp.array(3.0))),  # Extra in src
        )
    )
    dst_state_fewer_keys = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.array(0.0))),
            layer1=nnx.Dict(weight=nnx.Param(jnp.array(0.0))),
        )
    )

    utils.transfer_state_directly(
        src_state_more_keys, dst_state_fewer_keys, reshard_fn=mock_reshard
    )

    np.testing.assert_array_equal(
        dst_state_fewer_keys['decoder']['layer0']['weight'][...],
        jnp.array(1.0),
    )
    np.testing.assert_array_equal(
        dst_state_fewer_keys['decoder']['layer1']['weight'][...],
        jnp.array(2.0),
    )
    self.assertFalse(hasattr(dst_state_fewer_keys['decoder'], 'layer2'))

    # Scenario 2: Destination has more keys than source, with no unique keys in source
    src_state_fewer_keys = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.array(10.0))),
        )
    )
    dst_state_more_keys = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.array(0.0))),
            layer1=nnx.Dict(weight=nnx.Param(jnp.array(20.0))),  # Extra in dst
            layer2=nnx.Dict(weight=nnx.Param(jnp.array(30.0))),  # Extra in dst
        )
    )

    utils.transfer_state_directly(
        src_state_fewer_keys, dst_state_more_keys, reshard_fn=mock_reshard
    )

    np.testing.assert_array_equal(
        dst_state_more_keys['decoder']['layer0']['weight'][...],
        jnp.array(10.0),
    )
    # Extra keys in dst should be preserved
    np.testing.assert_array_equal(
        dst_state_more_keys['decoder']['layer1']['weight'][...],
        jnp.array(20.0),
    )
    np.testing.assert_array_equal(
        dst_state_more_keys['decoder']['layer2']['weight'][...],
        jnp.array(30.0),
    )

    # Original test case: Mixed extras in src and dst
    src_state_mixed = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.array(1.0))),
            layer1=nnx.Dict(weight=nnx.Param(jnp.array(2.0))),  # Extra in src
        ),
        rngs=nnx.Dict(),  # Extra in src
    )
    dst_state_mixed = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array(0.0)))),
        kv_cache=nnx.Dict(),  # Extra in dst
    )

    utils.transfer_state_directly(
        src_state_mixed, dst_state_mixed, reshard_fn=mock_reshard
    )

    # Common key should be updated
    np.testing.assert_array_equal(
        dst_state_mixed['decoder']['layer0']['weight'][...], jnp.array(1.0)
    )
    # Extra key in dst should be preserved
    self.assertIn('kv_cache', dst_state_mixed)
    # Extra key in src ('layer1') should NOT be added to dst.
    self.assertFalse(hasattr(dst_state_mixed['decoder'], 'layer1'))

  def test_transfer_state_directly_with_plain_dicts(self):
    """Tests transfer with plain dicts and various variables."""
    src_state = {
        'decoder': {
            'layer0': {
                    'weight': nnx.Param(jnp.array(1.0)),
                    'some_variable': nnx.Variable(jnp.array([1, 2])),
                },
            'extra_layer': {
                'sub': {
                    'value': nnx.Param(jnp.array(3.0))
                }
            },
        },
        'some_other_variable': nnx.Variable(jnp.array(42)),
    }
    dst_state = {
        'decoder': {
            'layer0': {
                    'weight': nnx.Param(jnp.array(0.0)),
                    'some_variable': nnx.Variable(jnp.array([0, 0])),
                },
            'layer1': {
                'weight': nnx.Param(jnp.array(0.0))
            },  # untouched
            'extra_layer': {
                'sub': {
                    'value': nnx.Param(jnp.array(0.0))
                }
            },
        },
        'some_other_variable': nnx.Variable(jnp.array(0)),
        'untouched_variable': nnx.Variable(jnp.array(-1)),  # untouched
    }

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    np.testing.assert_array_equal(
        dst_state['decoder']['layer0']['weight'][...], jnp.array(1.0)
    )
    np.testing.assert_array_equal(
        dst_state['decoder']['layer0']['some_variable'][...], jnp.array([1, 2])
    )

    np.testing.assert_array_equal(
        dst_state['decoder']['extra_layer']['sub']['value'][...],
        jnp.array(3.0),
    )
    np.testing.assert_array_equal(
        dst_state['some_other_variable'][...], jnp.array(42)
    )

    # Check that layer1 was not touched
    np.testing.assert_array_equal(
        dst_state['decoder']['layer1']['weight'][...], jnp.array(0.0)
    )
    # Check that untouched_variable was not touched
    np.testing.assert_array_equal(
        dst_state['untouched_variable'][...], jnp.array(-1)
    )

  def test_attention_weight_num_heads_repetition_and_rank_alignment(self):
    """Test repeating num_heads dimension (non-last axis) for attention weights."""
    # Source k_proj: (model_dim=16, num_heads=2, head_dim=128)
    # Target k_proj: (model_dim=16, num_heads * head_dim=512)
    src_k_proj = jnp.arange(16 * 2 * 128, dtype=jnp.float32).reshape(16, 2, 128)
    k_src_key = "layers.0.attn.k_proj.w"
    k_dst_key = "layers.0.self_attn.k_proj.w"

    # Source o_proj: (model_dim=16, head_dim=128,num_heads=2)
    # Target o_proj: (model_dim=16, num_heads * head_dim=512)
    src_o_proj = jnp.arange(16 * 128 * 2, dtype=jnp.float32).reshape(16, 128, 2)
    o_src_key = "layers.0.attn.o_proj.w"
    o_dst_key = "layers.0.self_attn.o_proj.w"

    src = MockState({
        k_src_key: MockParam(src_k_proj),
        o_src_key: MockParam(src_o_proj),
    })
    dst = MockState({
        k_dst_key: MockParam(jnp.zeros((16, 512), dtype=jnp.float32)),
        o_dst_key: MockParam(jnp.zeros((16, 512), dtype=jnp.float32)),
    })

    mappings = {
        k_src_key: (k_dst_key, None),
        o_src_key: (o_dst_key, None),
    }

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    # Verify shapes
    self.assertEqual(result.params[k_dst_key].shape, (16, 512))
    self.assertEqual(result.params[o_dst_key].shape, (16, 512))

    # Verify k_proj: heads are repeated on axis 1
    expected_k = jnp.repeat(src_k_proj, 2, axis=1).reshape(16, 512)
    np.testing.assert_allclose(result.params[k_dst_key], expected_k, atol=1e-1)

    # Verify o_proj: heads are repeated on axis 2
    expected_o = jnp.repeat(src_o_proj, 2, axis=2).reshape(16, 512)
    np.testing.assert_allclose(result.params[o_dst_key], expected_o, atol=1e-1)

  def test_transfer_state_with_interleaved_scanned_layers(self):
    """Test transfer with interleaved scanned layers using regex in mappings."""
    num_src_layers = 2
    num_tgt_layers = 4
    embed_dim = 4
    vocab_size = 8

    # Source state: 2 layers
    # src[0] = 1.0 (maps to tgt[0])
    # src[1] = 2.0 (maps to tgt[2])
    src_weights = jnp.stack(
        [
            jnp.full((embed_dim, vocab_size), i + 1, dtype=jnp.float32)
            for i in range(num_src_layers)
        ],
        axis=0,
    )
    src_state = MockState({
        "base.decoder.layers.layers_0.GptOssAttention.query.kernel": MockParam(
            src_weights
        ),
    })

    # Target state: 4 layers, all zeros initially
    target_params = {}
    for i in range(num_tgt_layers):
      target_params[f"decoder.layer.{i}.weight"] = MockParam(
          jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32)
      )
    tgt_state = MockState(target_params)

    # Mappings: Transfer only layers 0 and 2 from source to target
    mappings = {
        "base.decoder.layers.layers_0.GptOssAttention.query.kernel": (
            "decoder.layer.(0|2).weight",
            ("layer", None, None),
        ),
    }
    transpose_keys = {"kernel": (1, 0)}

    new_tgt_state = utils.transfer_state_with_mappings(
        src_state,
        tgt_state,
        key_mappings=mappings,
        transpose_keys=transpose_keys,
    )

    # Verify: Layer 0 and 2 should be transferred and transposed

    # Layer 0 comes from src[0] -> Value 1.0
    expected_layer_0 = jnp.full((vocab_size, embed_dim), 1.0, dtype=jnp.float32)

    # Layer 2 comes from src[1] -> Value 2.0
    expected_layer_2 = jnp.full((vocab_size, embed_dim), 2.0, dtype=jnp.float32)

    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layer.0.weight"].value,
            expected_layer_0,
        ),
        "Interleaved layer 0 mismatch",
    )
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layer.2.weight"].value,
            expected_layer_2,
        ),
        "Interleaved layer 2 mismatch",
    )

    # Layers 1 and 3 should remain zero (not mapped)
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layer.1.weight"].value,
            jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32),
        ),
        "Non-interleaved layer 1 should be zero",
    )
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layer.3.weight"].value,
            jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32),
        ),
        "Non-interleaved layer 3 should be zero",
    )

  def test_transfer_state_with_mappings_syncs_implicit_tied_lm_head(self):
    src = MockState({
        "model.embed_tokens.weight": MockParam(
            jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        ),
    })
    dst = MockState({
        "model.embed.embedding": MockParam(
            jnp.zeros((3, 4), dtype=jnp.float32)
        ),
        "model.lm_head": MockParam(jnp.full((3, 4), -1.0, dtype=jnp.float32)),
    })
    mappings = {
        "model.embed_tokens.weight": ("model.embed.embedding", None),
    }

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    np.testing.assert_array_equal(
        result.params["model.embed.embedding"],
        src.params["model.embed_tokens.weight"].value,
    )
    np.testing.assert_array_equal(
        result.params["model.lm_head"],
        src.params["model.embed_tokens.weight"].value,
    )

  def test_transfer_state_with_mappings_keeps_explicit_lm_head_mapping(self):
    src = MockState({
        "model.embed_tokens.weight": MockParam(
            jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        ),
        "lm_head": MockParam(jnp.full((3, 4), 7.0, dtype=jnp.float32)),
    })
    dst = MockState({
        "model.embed.embedding": MockParam(
            jnp.zeros((3, 4), dtype=jnp.float32)
        ),
        "model.lm_head": MockParam(jnp.full((3, 4), -1.0, dtype=jnp.float32)),
    })
    mappings = {
        "model.embed_tokens.weight": ("model.embed.embedding", None),
        "lm_head": ("model.lm_head", None),
    }

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    np.testing.assert_array_equal(
        result.params["model.embed.embedding"],
        src.params["model.embed_tokens.weight"].value,
    )
    np.testing.assert_array_equal(
        result.params["model.lm_head"],
        src.params["lm_head"].value,
    )

  def test_transfer_state_with_mappings_syncs_when_lm_head_mapping_unused(self):
    src = MockState({
        "model.embed_tokens.weight": MockParam(
            jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        ),
    })
    dst = MockState({
        "model.embed.embedding": MockParam(
            jnp.zeros((3, 4), dtype=jnp.float32)
        ),
        "model.lm_head": MockParam(jnp.full((3, 4), -1.0, dtype=jnp.float32)),
    })
    mappings = {
        "model.embed_tokens.weight": ("model.embed.embedding", None),
        "lm_head": ("model.lm_head", None),
    }

    result = utils.transfer_state_with_mappings(src, dst, mappings)

    np.testing.assert_array_equal(
        result.params["model.embed.embedding"],
        src.params["model.embed_tokens.weight"].value,
    )
    np.testing.assert_array_equal(
        result.params["model.lm_head"],
        src.params["model.embed_tokens.weight"].value,
    )

  def test_transfer_state_directly_scanned_layers(self):
    """Tests transfer from scanned 'layers' in source to 'layers_X' in dest."""
    # Source has 'layers' containing stacked weights (shape (2, ...))
    src_state = nnx.Dict(
        base=nnx.Dict(
            decoder=nnx.Dict(
                layers=nnx.Dict(
                    mlp=nnx.Dict(
                         # Stacked weight for 2 layers: 0->10.0, 1->20.0
                        weight=nnx.Param(jnp.array([10.0, 20.0]))
                    )
                )
            )
        )
    )
    # Dest has 'layers_0', 'layers_1' unrolled
    dst_state = nnx.Dict(
        model=nnx.Dict(
            decoder=nnx.Dict(
                layers_0=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.array(0.0)))),
                layers_1=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.array(0.0)))),
            )
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    np.testing.assert_array_equal(
        dst_state['model']['decoder']['layers_0']['mlp']['weight'][...], # Use [...]
        jnp.array(10.0),
    )
    np.testing.assert_array_equal(
        dst_state['model']['decoder']['layers_1']['mlp']['weight'][...], # Use [...]
        jnp.array(20.0),
    )

  def test_transfer_state_directly_implicit_layers_container(self):
    """Tests transfer when source IS the layers container (GPT-OSS style)."""
    # Use nnx.Dict for consistency
    src_state = nnx.Dict(
        layers=nnx.Dict(
            mlp=nnx.Dict(weight=nnx.Param(jnp.array([100.0, 200.0])))
        )
    )

    dst_state = nnx.Dict(
        layers=nnx.Dict(
            layers_0=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.array(0.0)))),
            layers_1=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.array(0.0)))),
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    np.testing.assert_array_equal(
        dst_state['layers']['layers_0']['mlp']['weight'][...],
        jnp.array(100.0),
    )
    np.testing.assert_array_equal(
        dst_state['layers']['layers_1']['mlp']['weight'][...],
        jnp.array(200.0),
    )

  def test_transfer_state_directly_with_dtype_casting(self):
    """Tests that transfer_state_directly correctly casts dtypes (e.g., f32 to bf16)."""
    # Source state in float32
    src_state = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(
                weight=nnx.Param(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
            ),
            # Scanned layers in float32
            layers=nnx.Dict(
                mlp=nnx.Dict(
                    weight=nnx.Param(jnp.array([[10.0, 11.0], [20.0, 21.0]], dtype=jnp.float32))
                )
            )
        )
    )

    # Destination state in bfloat16
    dst_state = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(
                weight=nnx.Param(jnp.zeros((3,), dtype=jnp.bfloat16))
            ),
            # Unrolled layers in bfloat16
            layers_0=nnx.Dict(
                mlp=nnx.Dict(weight=nnx.Param(jnp.zeros((2,), dtype=jnp.bfloat16)))
            ),
            layers_1=nnx.Dict(
                mlp=nnx.Dict(weight=nnx.Param(jnp.zeros((2,), dtype=jnp.bfloat16)))
            )
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    # Verify direct mapping cast
    self.assertEqual(dst_state['decoder']['layer0']['weight'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        dst_state['decoder']['layer0']['weight'][...],
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16),
        atol=1e-2
    )

    # Verify scanned layer mapping cast
    self.assertEqual(dst_state['decoder']['layers_0']['mlp']['weight'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        dst_state['decoder']['layers_0']['mlp']['weight'][...],
        jnp.array([10.0, 11.0], dtype=jnp.bfloat16),
        atol=1e-2
    )
    self.assertEqual(dst_state['decoder']['layers_1']['mlp']['weight'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        dst_state['decoder']['layers_1']['mlp']['weight'][...],
        jnp.array([20.0, 21.0], dtype=jnp.bfloat16),
        atol=1e-2
    )

  def test_transfer_state_directly_scanned_layers_casting(self):
    """Tests transfer from scanned layers container with dtype casting."""
    # Source has scanned layers in float32
    src_state = nnx.Dict(
        layers=nnx.Dict(
            mlp=nnx.Dict(
                weight=nnx.Param(jnp.array([100.0, 200.0], dtype=jnp.float32))
            )
        )
    )

    # Destination has unrolled layers_X in bfloat16
    dst_state = nnx.Dict(
        layers=nnx.Dict(
            layers_0=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.zeros((), dtype=jnp.bfloat16)))),
            layers_1=nnx.Dict(mlp=nnx.Dict(weight=nnx.Param(jnp.zeros((), dtype=jnp.bfloat16)))),
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    # Verify casting and slicing for implicit layers
    self.assertEqual(dst_state['layers']['layers_0']['mlp']['weight'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        dst_state['layers']['layers_0']['mlp']['weight'][...],
        jnp.array(100.0, dtype=jnp.bfloat16),
        atol=1e-2
    )
    self.assertEqual(dst_state['layers']['layers_1']['mlp']['weight'].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        dst_state['layers']['layers_1']['mlp']['weight'][...],
        jnp.array(200.0, dtype=jnp.bfloat16),
        atol=1e-2
    )

  def test_transfer_state_directly_repeats_kv_heads(self):
    """Tests that direct-match weights are repeated when dst has more heads."""
    src = jnp.array([[1., 2., 3., 4., 5., 6., 7., 8.],
                     [3., 4., 5., 6., 7., 8., 9., 10.]], dtype=jnp.float32)
    src_state = nnx.Dict(
        attn=nnx.Dict(
            kv_weight=nnx.Param(src)
        )
    )
    dst_state = nnx.Dict(
        attn=nnx.Dict(
            kv_weight=nnx.Param(jnp.zeros((4, 8), dtype=jnp.float32))
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    expected = jnp.repeat(src, 2, axis=0)
    np.testing.assert_array_equal(
        dst_state['attn']['kv_weight'][...],
        expected,
    )

  def test_transfer_state_directly_repeats_scanned_kv_heads(self):
    """Tests that scanned weights are tiled when dst has more heads than src."""
    # Source: scanned layers, kv_weight shape [2, 2, 8] (layers=2, heads=2, dim=8)
    src_state = nnx.Dict(
        layers=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(jnp.ones((2, 2, 8), dtype=jnp.float32))
            )
        )
    )
    # Destination: unrolled layers, each with kv_weight shape [4, 8]
    dst_state = nnx.Dict(
        layers_0=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(jnp.zeros((4, 8), dtype=jnp.float32))
            )
        ),
        layers_1=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(jnp.zeros((4, 8), dtype=jnp.float32))
            )
        ),
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    expected = jnp.repeat(jnp.ones((2, 8), dtype=jnp.float32), 2, axis=0)
    np.testing.assert_array_equal(
        dst_state['layers_0']['attn']['kv_weight'][...],
        expected,
    )
    np.testing.assert_array_equal(
        dst_state['layers_1']['attn']['kv_weight'][...],
        expected,
    )

  def test_slice_scanned_param_with_repeatable_target(self):
    """_slice_scanned_param finds scan axis even when post-slice shape needs repeat."""
    # src: (embed=4, layers=3, kv_heads=2, head_dim=8)
    # tgt: (embed=4, kv_heads=4, head_dim=8) — 2x repeat on kv_heads axis
    src = jnp.arange(4 * 3 * 2 * 8, dtype=jnp.float32).reshape(4, 3, 2, 8)
    tgt = jnp.zeros((4, 4, 8), dtype=jnp.float32)
    result = utils._unstack_scanned_param(src, tgt, key_path='test')[1]
    # Should return layer 1 slice: shape (4, 2, 8)
    np.testing.assert_equal(result.shape, (4, 2, 8))
    np.testing.assert_array_equal(result, src[:, 1, :, :])

  def test_transfer_state_directly_scanned_with_repeated_kv_heads(self):
    """Scanned src + KV-head-repeated dst transfer works end-to-end."""
    # src: scanned, shape (2, 2, 4) = (embed, layers, kv_heads*head_dim combined)
    # Use small shapes: embed=4, layers=2, kv_heads=2, head_dim=4
    # scanned param shape: (4, 2, 2, 4)
    layer0 = jnp.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=jnp.float32)  # (2, 4)
    layer1 = jnp.array([[9., 10., 11., 12.], [13., 14., 15., 16.]], dtype=jnp.float32)
    # Stack into scanned shape (2, 2, 4): [layer0, layer1] on axis 0
    scanned = jnp.stack([layer0, layer1], axis=0)  # (2, 2, 4)
    src_state = nnx.Dict(
        layers=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(scanned)
            )
        )
    )
    # dst: unrolled, each layer has (4, 4) — 2x repeat on kv_heads axis
    dst_state = nnx.Dict(
        layers_0=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(jnp.zeros((4, 4), dtype=jnp.float32))
            )
        ),
        layers_1=nnx.Dict(
            attn=nnx.Dict(
                kv_weight=nnx.Param(jnp.zeros((4, 4), dtype=jnp.float32))
            )
        ),
    )
    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard, scan_axis=0)

    expected_0 = jnp.repeat(layer0, 2, axis=0)  # (4, 4)
    expected_1 = jnp.repeat(layer1, 2, axis=0)
    np.testing.assert_array_equal(dst_state['layers_0']['attn']['kv_weight'][...], expected_0)
    np.testing.assert_array_equal(dst_state['layers_1']['attn']['kv_weight'][...], expected_1)

  def test_sglang_jax_1d_kv_bias_alignment(self):
    """Test 1-D KV bias alignment for sglang_jax rollout engine."""
    src_key = "layers.0.attn.k_bias"
    src_k_bias = jnp.arange(128, dtype=jnp.float32)
    src = MockState({src_key: MockParam(src_k_bias)})
    dst = MockState(
        {src_key: MockParam(jnp.zeros((1024,), dtype=jnp.float32))}
    )
    mappings = {src_key: (src_key, None)}

    result = utils.transfer_state_with_mappings(
        src,
        dst,
        mappings,
        rollout_engine="sglang_jax",
        num_kv_heads=1,
        head_dim=128,
    )

    self.assertEqual(result.params[src_key].shape, (1024,))
    expected = jnp.tile(src_k_bias, 8)
    self.assertTrue(jnp.allclose(result.params[src_key].value, expected))

  def test_transfer_state_directly_fuses_moe_weights(self):
    """Tests that wi_0 and wi_1 are fused into wi when target expects it."""
    wi_0_val = jnp.array([[1.0, 2.0], [5.0, 6.0]], dtype=jnp.float32)
    wi_1_val = jnp.array([[3.0, 4.0], [7.0, 8.0]], dtype=jnp.float32)

    src_state = nnx.Dict(
        layers=nnx.Dict(
            wi_0=nnx.Param(wi_0_val),
            wi_1=nnx.Param(wi_1_val),
        )
    )

    dst_state = nnx.Dict(
        layers=nnx.Dict(
            wi=nnx.Param(jnp.zeros((2, 4), dtype=jnp.float32))
        )
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    expected_wi = jnp.concatenate([wi_0_val, wi_1_val], axis=-1)
    np.testing.assert_array_equal(
        dst_state['layers']['wi'][...],
        expected_wi,
    )

  def test_transfer_state_directly_fuses_moe_weights_scanned_to_unrolled(self):
    """Scanned wi_0/wi_1 are unstacked and fused into per-layer wi (unrolled dst).

    Uses the function default `scan_axis=1`, matching MaxText's canonical
    scanned MoE layout `(experts, num_layers, features)`. `experts != num_layers`
    so a regression that prepends `wi_0.shape[0]` (experts) instead of
    `num_layers` will fail the final reshape inside `_interleave_moe_weights`.
    """
    # Layout: [experts=3, num_layers=2, features=2] (scan_axis=1).
    wi_0_val = jnp.array(
        [[[1., 2.], [10., 20.]],
         [[3., 4.], [30., 40.]],
         [[5., 6.], [50., 60.]]],
        dtype=jnp.float32,
    )
    wi_1_val = jnp.array(
        [[[100., 200.], [1000., 2000.]],
         [[300., 400.], [3000., 4000.]],
         [[500., 600.], [5000., 6000.]]],
        dtype=jnp.float32,
    )

    src_state = nnx.Dict(
        layers=nnx.Dict(
            wi_0=nnx.Param(wi_0_val),
            wi_1=nnx.Param(wi_1_val),
        )
    )
    dst_state = nnx.Dict(**{
        'layers_0': nnx.Dict(wi=nnx.Param(jnp.zeros((3, 4), dtype=jnp.float32))),
        'layers_1': nnx.Dict(wi=nnx.Param(jnp.zeros((3, 4), dtype=jnp.float32))),
    })

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    np.testing.assert_array_equal(
        dst_state['layers_0']['wi'][...],
        jnp.concatenate([wi_0_val[:, 0, :], wi_1_val[:, 0, :]], axis=-1),
    )
    np.testing.assert_array_equal(
        dst_state['layers_1']['wi'][...],
        jnp.concatenate([wi_0_val[:, 1, :], wi_1_val[:, 1, :]], axis=-1),
    )

  def test_transfer_state_directly_delete_dst_buffers_no_chunking(self):
    """delete_dst_buffers=True must never pass deleted arrays to reshard_fn."""
    src_val = jnp.array([1.0, 2.0, 3.0])
    src_state = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(src_val)))
    )
    dst_state = nnx.Dict(
        decoder=nnx.Dict(
            layer0=nnx.Dict(weight=nnx.Param(jnp.zeros(3, dtype=jnp.float32)))
        )
    )

    inspected_targets = []

    def reshard_fn(source, target):
      inspected_targets.extend(jax.tree_util.tree_leaves(target))
      return source

    utils.transfer_state_directly(
        src_state, dst_state, reshard_fn=reshard_fn, delete_dst_buffers=True
    )

    self.assertNotEmpty(inspected_targets)
    for leaf in inspected_targets:
      # Pre-fix this would have been a (possibly deleted) jax.Array.
      self.assertIsInstance(
          leaf, (NamedSharding, sharding.SingleDeviceSharding)
      )
    np.testing.assert_array_equal(
        dst_state['decoder']['layer0']['weight'][...], src_val
    )

  def test_transfer_state_directly_delete_dst_buffers_chunked(self):
    """delete_dst_buffers=True works through the chunked path too."""
    src_state = nnx.Dict(
        decoder=nnx.Dict(**{
            f'layer{i}': nnx.Dict(weight=nnx.Param(jnp.array([float(i + 1)])))
            for i in range(4)
        })
    )
    dst_state = nnx.Dict(
        decoder=nnx.Dict(**{
            f'layer{i}': nnx.Dict(weight=nnx.Param(jnp.array([0.0])))
            for i in range(4)
        })
    )

    inspected_targets = []

    def reshard_fn(source, target):
      inspected_targets.extend(jax.tree_util.tree_leaves(target))
      return source

    utils.transfer_state_directly(
        src_state,
        dst_state,
        reshard_fn=reshard_fn,
        delete_dst_buffers=True,
        reshard_chunk_size=2,
    )

    self.assertNotEmpty(inspected_targets)
    for leaf in inspected_targets:
      self.assertIsInstance(
          leaf, (NamedSharding, sharding.SingleDeviceSharding)
      )
    for i in range(4):
      np.testing.assert_array_equal(
          dst_state['decoder'][f'layer{i}']['weight'][...],
          jnp.array([float(i + 1)]),
      )

  def test_transfer_state_directly_delete_dst_buffers_skips_aliased_buffers(
      self,
  ):
    """When src and dst Variables share a backing jax.Array, skip deletion."""
    shared = jnp.array([1.0, 2.0, 3.0])
    src_state = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(shared)))
    )
    # Same backing jax.Array object on both sides — typical of collocated
    # trainer/sampler setups where the rollout state aliases trainer weights.
    dst_state = nnx.Dict(
        decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(shared)))
    )

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(
        src_state,
        dst_state,
        reshard_fn=mock_reshard,
        delete_dst_buffers=True,
    )

    # If deletion misfired the next access raises "Array has been deleted".
    np.testing.assert_array_equal(np.asarray(shared), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        dst_state['decoder']['layer0']['weight'][...], [1.0, 2.0, 3.0]
    )

  def test_transfer_state_directly_delete_dst_buffers_scanned_layers(self):
    """Unstacked-slice targets remain valid after delete_dst_buffers=True."""
    scanned = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    src_state = nnx.Dict(layers=nnx.Dict(weight=nnx.Param(scanned)))
    dst_state = nnx.Dict(**{
        'layers_0': nnx.Dict(
            weight=nnx.Param(jnp.zeros(4, dtype=jnp.float32))
        ),
        'layers_1': nnx.Dict(
            weight=nnx.Param(jnp.zeros(4, dtype=jnp.float32))
        ),
    })

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(
        src_state,
        dst_state,
        reshard_fn=mock_reshard,
        scan_axis=0,
        delete_dst_buffers=True,
    )

    np.testing.assert_array_equal(
        dst_state['layers_0']['weight'][...], scanned[0]
    )
    np.testing.assert_array_equal(
        dst_state['layers_1']['weight'][...], scanned[1]
    )

  def test_transfer_state_directly_fuses_moe_weights_with_padding(self):
    """Tests that wi_0 and wi_1 are fused, padded and interleaved into wi."""
    # Source: wi_0, wi_1 each (2 experts, 2 features)
    wi_0_val = jnp.array([[1.0, 2.0], [5.0, 6.0]], dtype=jnp.float32)
    wi_1_val = jnp.array([[3.0, 4.0], [7.0, 8.0]], dtype=jnp.float32)

    src_state = nnx.Dict(
        layers=nnx.Dict(
            wi_0=nnx.Param(wi_0_val),
            wi_1=nnx.Param(wi_1_val),
        )
    )

    # Target: wi (2 experts, 8 features total -> 4 features each half)
    # To test interleaved sharding, we need n_shards > 1.
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices for sharded padding test.")

    mesh = Mesh(np.array(devices[:2]), axis_names=("model",))
    # Sharding on last axis with 2 shards
    sharding = NamedSharding(mesh, PartitionSpec(None, "model"))

    dst_wi = jax.device_put(jnp.zeros((2, 8), dtype=jnp.float32), sharding)
    dst_state = nnx.Dict(layers=nnx.Dict(wi=nnx.Param(dst_wi)))

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    # Expected:
    # Half 0 (wi_0) padded from 2 to 4 features. Per-shard padding: 1 -> 2.
    # wi_0 local shards: [1.0], [2.0] -> padded: [1.0, 0.0], [2.0, 0.0] -> global: [1.0, 0.0, 2.0, 0.0]
    # Half 1 (wi_1) padded from 2 to 4 features. Per-shard padding: 1 -> 2.
    # wi_1 local shards: [3.0], [4.0] -> padded: [3.0, 0.0], [4.0, 0.0] -> global: [3.0, 0.0, 4.0, 0.0]
    # Interleaved: [half0_shard0, half1_shard0, half0_shard1, half1_shard1]
    # [1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0] for first expert.

    expected_wi = jnp.array(
        [
            [1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0],
            [5.0, 0.0, 7.0, 0.0, 6.0, 0.0, 8.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(dst_state["layers"]["wi"][...], expected_wi)

  def test_transfer_state_directly_fuses_moe_weights_with_padding_scanned(self):
    """Scanned wi_0/wi_1 are fused, padded per-shard, and unstacked per-layer.

    Mirrors `test_transfer_state_directly_fuses_moe_weights_with_padding` but
    with a scanned source under the function default `scan_axis=1` and
    `experts != num_layers` so a regression that prepends `wi_0.shape[0]`
    (experts) instead of `num_layers` will fail the final reshape inside
    `_interleave_moe_weights`.
    """
    # Source: scanned [experts=3, num_layers=2, features=2] with scan_axis=1.
    wi_0_val = jnp.array(
        [[[1., 2.], [10., 20.]],
         [[3., 4.], [30., 40.]],
         [[5., 6.], [50., 60.]]],
        dtype=jnp.float32,
    )
    wi_1_val = jnp.array(
        [[[100., 200.], [1000., 2000.]],
         [[300., 400.], [3000., 4000.]],
         [[500., 600.], [5000., 6000.]]],
        dtype=jnp.float32,
    )

    src_state = nnx.Dict(
        layers=nnx.Dict(
            wi_0=nnx.Param(wi_0_val),
            wi_1=nnx.Param(wi_1_val),
        )
    )

    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices for sharded padding test.")

    mesh = Mesh(np.array(devices[:2]), axis_names=("model",))
    sharding = NamedSharding(mesh, PartitionSpec(None, "model"))

    # Per-layer fused target: (3 experts, 8 features). Two layers, unrolled dst.
    dst_state = nnx.Dict(**{
        'layers_0': nnx.Dict(
            wi=nnx.Param(
                jax.device_put(jnp.zeros((3, 8), dtype=jnp.float32), sharding)
            )
        ),
        'layers_1': nnx.Dict(
            wi=nnx.Param(
                jax.device_put(jnp.zeros((3, 8), dtype=jnp.float32), sharding)
            )
        ),
    })

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    # Per-shard pad-then-interleave on the last axis. For each (expert, layer)
    # the wi_0 features [a, b] pad to [a, 0, b, 0] (2 shards, 1 elt each pads
    # to 2), wi_1 features [c, d] pad to [c, 0, d, 0]; interleaved per shard
    # gives [a, 0, c, 0, b, 0, d, 0].
    expected_layers_0 = jnp.array(
        [
            [1., 0., 100., 0., 2., 0., 200., 0.],
            [3., 0., 300., 0., 4., 0., 400., 0.],
            [5., 0., 500., 0., 6., 0., 600., 0.],
        ],
        dtype=jnp.float32,
    )
    expected_layers_1 = jnp.array(
        [
            [10., 0., 1000., 0., 20., 0., 2000., 0.],
            [30., 0., 3000., 0., 40., 0., 4000., 0.],
            [50., 0., 5000., 0., 60., 0., 6000., 0.],
        ],
        dtype=jnp.float32,
    )

    np.testing.assert_array_equal(
        dst_state["layers_0"]["wi"][...], expected_layers_0
    )
    np.testing.assert_array_equal(
        dst_state["layers_1"]["wi"][...], expected_layers_1
    )

  def test_transfer_state_directly_moe_wi_padding_replicated_source(self):
    """Replicated source -> TP-sharded target produces interleaved global pad.

    The MoE-key path in `_align_per_axis` reshapes the source to expose
    the target's shard structure, tail-pads each chunk under JIT
    (`_jit_zero_pad_axes`), then flattens. The interleaved layout means
    each device receives `[data_chunk, local_pad]` rather than one device
    getting all data and another all padding.
    """
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices for sharded padding test.")

    mesh = Mesh(np.array(devices[:2]), axis_names=("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    sharded = NamedSharding(mesh, PartitionSpec(None, "model"))

    wi_0_val = jnp.array(
        [[1., 2., 3., 4.],
         [5., 6., 7., 8.],
         [9., 10., 11., 12.]],
        dtype=jnp.float32,
    )
    src_wi_0 = jax.device_put(wi_0_val, replicated)

    src_state = nnx.Dict(layers=nnx.Dict(wi_0=nnx.Param(src_wi_0)))
    dst_state = nnx.Dict(layers=nnx.Dict(
        wi_0=nnx.Param(
            jax.device_put(jnp.zeros((3, 8), dtype=jnp.float32), sharded)
        )
    ))

    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(
        src_state, dst_state, reshard_fn=mock_reshard
    )
    result = dst_state["layers"]["wi_0"][...]

    self.assertEqual(result.shape, (3, 8))
    # Each row's 4 source values split into 2 chunks of 2; each chunk gets 2
    # tail zeros, then chunks flatten into the global axis. So
    # [1, 2, 3, 4] -> [[1, 2], [3, 4]] -> [[1, 2, 0, 0], [3, 4, 0, 0]]
    # -> [1, 2, 0, 0, 3, 4, 0, 0]. Device 0 sees [1, 2, 0, 0]; device 1
    # sees [3, 4, 0, 0] — no data shifts across the shard boundary.
    expected = jnp.array(
        [[1., 2., 0., 0., 3., 4., 0., 0.],
         [5., 6., 0., 0., 7., 8., 0., 0.],
         [9., 10., 0., 0., 11., 12., 0., 0.]],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(result, expected)

  def test_transfer_state_directly_moe_wi_padding_unscanned_separate_replicated_src(
      self,
  ):
    """End-to-end: unscanned replicated wi_0/wi_1 -> unscanned padded sharded wi_0/wi_1.

    The bug this whole refactor targets: a replicated trainer source that
    needs MoE-dim padding for a TP-sharded inference target. Pre-fix this
    silently produced under-padded arrays after `nnx.update`. The new
    interleaved global pad keeps each device's slice aligned with its
    portion of the source data.
    """
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices for sharded padding test.")
    mesh = Mesh(np.array(devices[:2]), axis_names=("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    sharded = NamedSharding(mesh, PartitionSpec(None, "model"))

    wi_0_val = jnp.array(
        [[1., 2., 3., 4.],
         [5., 6., 7., 8.],
         [9., 10., 11., 12.]],
        dtype=jnp.float32,
    )
    src_state = nnx.Dict(layers=nnx.Dict(
        wi_0=nnx.Param(jax.device_put(wi_0_val, replicated)),
    ))
    dst_state = nnx.Dict(layers=nnx.Dict(
        wi_0=nnx.Param(
            jax.device_put(jnp.zeros((3, 8), dtype=jnp.float32), sharded)
        ),
    ))
    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    # Interleaved: each row's 4 values split into 2 chunks of 2, each chunk
    # gets 2 trailing zeros, then chunks flatten — so device 0 owns
    # `[a, b, 0, 0]` and device 1 owns `[c, d, 0, 0]` for each row.
    expected = jnp.array(
        [[1., 2., 0., 0., 3., 4., 0., 0.],
         [5., 6., 0., 0., 7., 8., 0., 0.],
         [9., 10., 0., 0., 11., 12., 0., 0.]],
        dtype=jnp.float32,
    )
    self.assertEqual(dst_state["layers"]["wi_0"][...].shape, (3, 8))
    np.testing.assert_array_equal(dst_state["layers"]["wi_0"][...], expected)

  def test_transfer_state_directly_moe_wi_padding_scanned_separate_replicated_src(
      self,
  ):
    """Scanned replicated wi_0/wi_1 -> unrolled padded sharded non-fused wi_0/wi_1.

    Exercises the bulk-align-and-unstack path on a scanned source that's
    replicated relative to the target's TP sharding. Verifies per-layer
    shapes are correctly padded after `nnx.update`.
    """
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Need at least 2 devices for sharded padding test.")
    mesh = Mesh(np.array(devices[:2]), axis_names=("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    sharded = NamedSharding(mesh, PartitionSpec(None, "model"))

    # Scanned replicated source: (experts=3, num_layers=2, features=4) at scan_axis=1.
    wi_0_unsharded = jnp.array(
        [[[1., 2., 3., 4.], [10., 20., 30., 40.]],
         [[5., 6., 7., 8.], [50., 60., 70., 80.]],
         [[9., 10., 11., 12.], [90., 100., 110., 120.]]],
        dtype=jnp.float32,
    )
    src_state = nnx.Dict(layers=nnx.Dict(
        wi_0=nnx.Param(jax.device_put(wi_0_unsharded, replicated)),
    ))

    def _zeros():
      return jax.device_put(jnp.zeros((3, 8), dtype=jnp.float32), sharded)

    dst_state = nnx.Dict(**{
        "layers_0": nnx.Dict(wi_0=nnx.Param(_zeros())),
        "layers_1": nnx.Dict(wi_0=nnx.Param(_zeros())),
    })
    mock_reshard = lambda source, target: source
    utils.transfer_state_directly(src_state, dst_state, reshard_fn=mock_reshard)

    # Replicated source -> sharded target: per-axis pad goes through the
    # global interleaved path. For each unrolled layer slice the 4 source
    # features split into 2 chunks of 2; each chunk gets 2 trailing zeros,
    # so the per-row layout becomes [a, b, 0, 0, c, d, 0, 0] — aligned with
    # the target's 2-shard split on the last axis.
    expected_layer_0 = jnp.array(
        [[1., 2., 0., 0., 3., 4., 0., 0.],
         [5., 6., 0., 0., 7., 8., 0., 0.],
         [9., 10., 0., 0., 11., 12., 0., 0.]],
        dtype=jnp.float32,
    )
    expected_layer_1 = jnp.array(
        [[10., 20., 0., 0., 30., 40., 0., 0.],
         [50., 60., 0., 0., 70., 80., 0., 0.],
         [90., 100., 0., 0., 110., 120., 0., 0.]],
        dtype=jnp.float32,
    )
    self.assertEqual(dst_state["layers_0"]["wi_0"][...].shape, (3, 8))
    self.assertEqual(dst_state["layers_1"]["wi_0"][...].shape, (3, 8))
    np.testing.assert_array_equal(
        dst_state["layers_0"]["wi_0"][...], expected_layer_0
    )
    np.testing.assert_array_equal(
        dst_state["layers_1"]["wi_0"][...], expected_layer_1
    )

  def test_align_per_axis_attention_pure_repeat(self):
    """Attention key path → pure `repeat` on every mismatched axis.

    KV-head expansion and head_dim padding never co-occur on a single
    tensor in production, so `_align_per_axis` no longer composes both:
    attention keys take the repeat-only path. This pins that contract.
    """
    src = jnp.array(
        [[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=jnp.float32
    )  # shape (2, 4) — kv_heads=2, head_dim=4
    tgt_shape = (8, 4)  # 4x KV-head expansion only.
    result = utils._align_per_axis(
        src, tgt_shape, tgt_sharding=None, key_path="layers.0.attn.q_proj"
    )
    self.assertEqual(result.shape, tgt_shape)
    expected = jnp.repeat(src, 4, axis=0)
    np.testing.assert_array_equal(np.asarray(result), expected)

  def test_align_per_axis_non_repeatable_non_moe_raises(self):
    """Non-MoE key with a non-integer-multiple mismatch raises.

    The simplified single-mode aligner no longer falls back to zero-pad
    for attention head_dim mismatches; if a non-MoE tensor has an axis
    that isn't an integer multiple, it's a configuration error.
    """
    src = jnp.zeros((2, 3), dtype=jnp.float32)
    with self.assertRaises(utils.ShapeMismatchError):
      utils._align_per_axis(
          src, (8, 4), tgt_sharding=None, key_path="layers.0.attn.q_proj"
      )

  def test_align_per_axis_moe_two_axes_zero_pad(self):
    """MoE `wi` style: multiple mismatched axes, all classified as zero_pad.

    Pre-refactor `_align_to_model_shape` only padded the *last* mismatched
    axis on MoE keys (and warned). The new per-axis aligner handles every
    mismatched axis. This test exercises a synthetic case to pin that down.
    """
    src = jnp.array(
        [[[1., 2.], [3., 4.]]],  # (1, 2, 2)
        dtype=jnp.float32,
    )
    result = utils._align_per_axis(
        src, tgt_shape=(2, 2, 4), tgt_sharding=None, key_path="layers.0.wi"
    )
    self.assertEqual(result.shape, (2, 2, 4))
    expected = jnp.pad(src, ((0, 1), (0, 0), (0, 2)))
    np.testing.assert_array_equal(np.asarray(result), expected)


class ResolveParallelismSizesTest(parameterized.TestCase):

  def _make_mesh(self, total_devices):
    """Returns a mock mesh with the given total device count."""
    mesh = mock.MagicMock()
    mesh.shape = {"axis": total_devices}
    return mesh

  @parameterized.named_parameters(
      ("tp_and_dp_inferred_no_ep", 8, -1, -1, 1, 8, 1, 1),
      ("tp_and_dp_inferred_with_ep", 8, -1, -1, 2, 4, 1, 2),
      ("tp_inferred_with_ep", 8, -1, 2, 2, 2, 2, 2),
      ("dp_inferred_with_ep", 8, 2, -1, 2, 2, 2, 2),
      ("all_explicit", 8, 4, 2, 1, 4, 2, 1),
  )
  def test_resolve_parallelism_sizes(
      self,
      total_devices,
      tp_in,
      dp_in,
      ep_in,
      expected_tp,
      expected_dp,
      expected_ep,
  ):
    mesh = self._make_mesh(total_devices)
    tp, dp, ep = utils.resolve_parallelism_sizes(
        mesh=mesh,
        tensor_parallel_size=tp_in,
        data_parallel_size=dp_in,
        expert_parallel_size=ep_in,
    )
    self.assertEqual(tp, expected_tp)
    self.assertEqual(dp, expected_dp)
    self.assertEqual(ep, expected_ep)

  def test_resolve_parallelism_sizes_indivisible_ep_raises(self):
    mesh = self._make_mesh(8)
    with self.assertRaisesRegex(ValueError, "expert_parallel_size"):
      utils.resolve_parallelism_sizes(mesh=mesh, expert_parallel_size=3)


if __name__ == "__main__":
  absltest.main()
