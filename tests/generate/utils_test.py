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
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
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
      new_params[".".join(keys)] = param.value
    return MockState(new_params)


class MockParam:

  def __init__(self, value):
    self.value = value


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
        utils.get_logprobs_from_vllm_output(token_ids, logprobs),
        expected,
    )

  def test_logprobs_extraction_with_missing_token(self):
    token_ids = [100, 200]
    logprobs = [{101: Logprob(-0.5)}, {200: Logprob(-1.2)}]
    with self.assertRaises(ValueError):
      utils.get_logprobs_from_vllm_output(token_ids, logprobs)

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
    assert device_count % 2 == 0, "This example assumes even number of devices"

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
            new_tgt_state.params["decoder.layer_0.weight"],
            expected_layer_0_weight,
        )
    )
    expected_layer_1_weight = jnp.arange(16, 32).reshape(2, 8).T
    self.assertTrue(
        jnp.array_equal(
            new_tgt_state.params["encoder.layer_0.weight"],
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
        {"decoder.layers.5.attn.o_proj": MockParam(jnp.ones((2, 4, 64)))}
    )
    dst = MockState(
        {"decoder.layers.5.attn.o_proj": MockParam(jnp.zeros((2, 4, 128)))}
    )

    mappings = {
        "decoder.layers.5.attn.o_proj": ("decoder.layers.5.attn.o_proj", None),
    }

    new_tgt_state = utils.transfer_state_with_mappings(src, dst, mappings)

    # Validate shape
    self.assertEqual(
        new_tgt_state.params["decoder.layers.5.attn.o_proj"].shape, (2, 4, 128)
    )
    # Validate original values copied correctly
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layers.5.attn.o_proj"][:, :, :64], 1.0
        )
    )
    # Validate padded values are zero
    self.assertTrue(
        jnp.allclose(
            new_tgt_state.params["decoder.layers.5.attn.o_proj"][:, :, 64:], 0.0
        )
    )

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
              transferred,
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
          jnp.allclose(transferred, expected),
          f"Scanned bias layer {layer_idx} mismatch",
      )

    # Verify regular parameter with transpose
    transferred_embedding = new_tgt_state.params["embedding.weight"]

    self.assertEqual(transferred_embedding.shape, (embed_dim, vocab_size))
    self.assertTrue(
        jnp.allclose(
            transferred_embedding,
            jnp.full((embed_dim, vocab_size), 99.0, dtype=jnp.float32),
        ),
        "Regular parameter with transpose mismatch",
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
