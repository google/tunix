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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.sft import utils
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)
jax.config.update("jax_default_matmul_precision", "highest")

def _compute_loss(*args, **kwargs):
  out = getattr(common, "aggregate_loss")(*args, **kwargs)
  return out.compute()


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "kl",
          "kl",
          np.array([
              [
                  [0.38486493, 0.7469206, 0.3495195, 0.5621129],
                  [-0.4474684, -0.13095665, 0.46064317, -0.19887352],
              ],
              [
                  [0.43108296, 0.53564644, -0.25296474, 0.44137287],
                  [-0.06834459, -0.12115264, -0.61533415, 0.15468943],
              ],
          ]),
      ),
      (
          "mse_kl",
          "mse_kl",
          np.array([
              [
                  [0.07406051, 0.27894518, 0.06108194, 0.15798548],
                  [0.10011399, 0.00857482, 0.10609607, 0.01977534],
              ],
              [
                  [0.09291626, 0.14345856, 0.03199558, 0.09740501],
                  [0.00233549, 0.00733898, 0.18931806, 0.01196441],
              ],
          ]),
      ),
      (
          "low_var_kl",
          "low_var_kl",
          np.array([
              [
                  [0.0654075, 0.220744, 0.0545462, 0.1321163],
                  [0.1168784, 0.0089617, 0.0915209, 0.0211542],
              ],
              [
                  [0.080888, 0.1209372, 0.0348731, 0.0845257],
                  [0.0023897, 0.0076445, 0.2349406, 0.0113707],
              ],
          ]),
      ),
  )
  def test_compute_kl_divergence(self, method, expected_value):
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    per_token_logps = jax.random.uniform(k1, shape=(2, 2, 4))
    ref_per_token_logps = jax.random.uniform(k2, shape=(2, 2, 4))
    kl_divergence = common.compute_kl_divergence(
        per_token_logps, ref_per_token_logps, method=method
    )
    np.testing.assert_allclose(
        kl_divergence, expected_value, atol=1e-5, rtol=1e-2
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="no_seq_packing",
          prompt_tokens=np.array([[1, 2], [3, 4]]),
          completion_tokens=np.array([[5, 6, 7], [8, 9, 10]]),
          segment_ids=None,
          segment_positions=None,
          expected_shape=(2, 5),
          expected_scores=np.array(
              [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]], dtype=np.float32
          ),
      ),
      dict(
          testcase_name="with_seq_packing",
          prompt_tokens=np.zeros((2, 0), dtype=np.int32),
          completion_tokens=np.array([[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]]),
          segment_ids=np.array(
              [[1, 1, 2, 2, 2], [1, 1, 2, 2, 2]], dtype=np.int32
          ),
          segment_positions=np.array(
              [[0, 1, 0, 1, 2], [0, 1, 0, 1, 2]], dtype=np.int32
          ),
          expected_shape=(2, 5),
          expected_scores=np.array(
              [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]], dtype=np.float32
          ),
      ),
  )
  def test_compute_score(
      self,
      prompt_tokens,
      completion_tokens,
      segment_ids,
      segment_positions,
      expected_shape,
      expected_scores,
  ):
    class MockModel(nnx.Module):

      def __call__(
          self,
          input_ids,
          positions,
          attention_mask=None,
          segment_ids=None,
          cache=None,
      ):
        return input_ids.astype(jnp.float32)[..., None], None

    model = MockModel()

    scores = common.compute_score(
        model,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
    )

    self.assertEqual(scores.shape, expected_shape)
    np.testing.assert_allclose(scores, expected_scores)

  def test_selective_log_softmax(self):
    rng = jax.random.PRNGKey(0)
    logits = jax.random.uniform(rng, shape=(2, 4, 8))
    input_ids = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    per_token_logps = common.selective_log_softmax(logits, input_ids)
    jitted_per_token_logps = jax.jit(common.selective_log_softmax)(
        logits, input_ids
    )
    expected_value = jnp.array([
        [-2.242679, -2.2733693, -2.1024966, -1.9994389],
        [-2.0603075, -2.4863663, -1.9176172, -2.0206313],
    ])
    np.testing.assert_allclose(
        per_token_logps, expected_value, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        per_token_logps, jitted_per_token_logps, rtol=1e-05, atol=1e-05
    )

  def test_get_per_token_logps(self):
    rng = jax.random.PRNGKey(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    input_tokens = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    positions = jnp.ones((2, 4))
    attn_mask = common.make_causal_attn_mask(positions)
    per_token_logps = common.get_per_token_logps(
        model, input_tokens, positions, attn_mask, logits_to_keep=2
    )
    np.testing.assert_allclose(
        per_token_logps,
        np.array([[-5.7448483, -5.937829], [-4.222273, -4.41953]]),
        rtol=1e-02,
        atol=1e-03,
    )

  def test_process_ids_raises_value_error(self):
    prompt_tokens = jnp.array([[1, 2], [3, 4]])
    completion_tokens = jnp.array([[5, 6], [7, 8]])
    segment_ids = jnp.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    with self.assertRaisesRegex(
        ValueError,
        "segment_positions must be explicitly provided for packed sequences.",
    ):
      common.process_ids(
          prompt_tokens,
          completion_tokens,
          pad_id=0,
          eos_id=-1,
          segment_ids=segment_ids,
          segment_positions=None,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="normal",
          prompt_tokens=np.array([[1, 2, 3, 4], [0, 0, 1, 2], [0, 1, 2, 3]]),
          completion_tokens=np.array(
              [[10, 11, -1, 12], [10, 11, 12, 13], [10, 11, 12, -1]]
          ),
          segment_ids=None,
          segment_positions=None,
          expected_logps=np.array([
              [-5.876301, -8.700251, -5.046069, -5.788748],
              [-6.071025, -7.5328417, -5.9712567, -4.653783],
              [-6.039485, -8.264197, -6.2771187, -4.767109],
          ]),
      ),
      dict(
          testcase_name="seq-packed-single-item",
          prompt_tokens=np.zeros((3, 0), dtype=np.int32),
          completion_tokens=np.array([
              [1, 2, 3, 4, 10, 11, -1, 12],
              [0, 0, 1, 2, 10, 11, 12, 13],
              [0, 1, 2, 3, 10, 11, 12, -1],
          ]),
          segment_ids=np.ones((3, 8), dtype=np.int32),
          segment_positions=np.tile(np.arange(8), (3, 1)),
          expected_logps=np.array([
              [
                  0.0,
                  -7.3199797,
                  -6.8320303,
                  -5.6091313,
                  -5.876301,
                  -8.700251,
                  -5.0460696,
                  -5.788748,
              ],
              [
                  0.0,
                  -6.4536085,
                  -5.5156517,
                  -7.103587,
                  -6.0710244,
                  -7.5328417,
                  -5.971257,
                  -4.653783,
              ],
              [
                  0.0,
                  -5.789238,
                  -7.7057056,
                  -6.7916627,
                  -6.0394855,
                  -8.264197,
                  -6.2771187,
                  -4.7671094,
              ],
          ]),
      ),
      dict(
          testcase_name="seq-packed-multi-item",
          prompt_tokens=np.zeros((2, 0), dtype=np.int32),
          completion_tokens=np.array([
              [1, 2, 3, 4, 10, 11, -1, 12, 0, 0, 1, 2, 10, 11, 12, 13],
              [0, 1, 2, 3, 10, 11, 12, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          ]),
          segment_ids=np.array([
              [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          ]),
          segment_positions=np.array([
              [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
              [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
          ]),
          # NOTE: Expected logprobs diverge from single-item values because
          # floating-point reductions in XLA compound differently when changing
          # batch size from 3 to 2 and sequence length from 8 to 16.
          expected_logps=np.array([
              [
                  0.0,
                  -7.255163,
                  -6.413455,
                  -5.682157,
                  -5.83097,
                  -8.132578,
                  -4.8891325,
                  -5.7902822,
                  -6.452383,
                  -6.524351,
                  -5.778284,
                  -7.255163,
                  -6.245493,
                  -8.132578,
                  -6.025977,
                  -4.6675467,
              ],
              [
                  0.0,
                  -4.070095,
                  -7.792082,
                  -6.3780885,
                  -6.312748,
                  -6.536421,
                  -6.0986547,
                  -5.62961,
                  -5.558264,
                  -6.595858,
                  -6.595858,
                  -6.595858,
                  -6.595858,
                  -6.595858,
                  -6.595858,
                  -6.595858,
              ],
          ]),
      ),
  )
  def test_compute_per_token_logps(
      self,
      prompt_tokens,
      completion_tokens,
      segment_ids,
      segment_positions,
      expected_logps,
  ):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    per_token_logps = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        return_entropy=False,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
    )

    np.testing.assert_allclose(
        per_token_logps, expected_logps, atol=1e-4, rtol=1e-4
    )

    _, entropy = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        return_entropy=True,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
    )
    np.testing.assert_equal(entropy.shape, expected_logps.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="normal",
          prompt_tokens=np.array([[1, 2, 3, 4], [0, 0, 1, 2], [0, 1, 2, 3]]),
          completion_tokens=np.array(
              [[10, 11, -1, 12], [10, 11, 12, 13], [10, 11, 12, -1]]
          ),
          segment_ids=None,
          segment_positions=None,
          temperature=1.0,
      ),
      dict(
          testcase_name="seq-packed-single-item",
          prompt_tokens=np.zeros((3, 0), dtype=np.int32),
          completion_tokens=np.array([
              [1, 2, 3, 4, 10, 11, -1, 12],
              [0, 0, 1, 2, 10, 11, 12, 13],
              [0, 1, 2, 3, 10, 11, 12, -1],
          ]),
          segment_ids=np.ones((3, 8), dtype=np.int32),
          segment_positions=np.tile(np.arange(8), (3, 1)),
          temperature=0.7,
      ),
      dict(
          testcase_name="seq-packed-multi-item",
          prompt_tokens=np.zeros((2, 0), dtype=np.int32),
          completion_tokens=np.array([
              [1, 2, 3, 4, 10, 11, -1, 12, 0, 0, 1, 2, 10, 11, 12, 13],
              [0, 1, 2, 3, 10, 11, 12, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          ]),
          segment_ids=np.array([
              [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          ]),
          segment_positions=np.array([
              [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
              [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
          ]),
          temperature=1.5,
      ),
  )
  def test_chunked_compute_per_token_logps(
      self,
      prompt_tokens,
      completion_tokens,
      segment_ids,
      segment_positions,
      temperature,
  ):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    for return_entropy in [True, False]:
      res = common.compute_per_token_logps(
          graphdef,
          state,
          prompt_tokens,
          completion_tokens,
          pad_id=0,
          eos_id=-1,
          return_entropy=return_entropy,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          temperature=temperature,
          chunk_size=0,
      )
      if return_entropy:
        expected_logps, expected_entropy = res
      else:
        expected_logps = res

      for chunk_size in [2, 4, 8, 16]:
        chunked_res = common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens,
            completion_tokens,
            pad_id=0,
            eos_id=-1,
            return_entropy=return_entropy,
            segment_ids=segment_ids,
            segment_positions=segment_positions,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        if return_entropy:
          chunked_logps, chunked_entropy = chunked_res
        else:
          chunked_logps = chunked_res

        np.testing.assert_allclose(
            chunked_logps, expected_logps, atol=1e-5, rtol=1e-5
        )
        if return_entropy:
          # NB: skip first one since it's a padded value.
          np.testing.assert_allclose(
              chunked_entropy[..., 1:],
              expected_entropy[..., 1:],
              atol=1e-5,
              rtol=1e-5,
          )

  def test_np_make_completion_mask(self):
    completion_ids = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 0],
            [1, 2, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    np_completion_mask = common.np_make_completion_mask(completion_ids)
    expected_value = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    np.testing.assert_allclose(np_completion_mask, expected_value)

  def test_make_completion_mask(self):
    completion_ids = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 0],
            [1, 2, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    completion_mask = common.make_completion_mask(completion_ids)
    expected_value = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    np.testing.assert_allclose(completion_mask, expected_value)

  def test_pad_to_length(self):
    x = jnp.ones((2, 4))
    padded_x = common.pad_to_length(x, target_length=5)
    self.assertEqual(padded_x.shape, (5, 4))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, pad_value=1, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 10)
    padded_x = common.pad_to_length(x, target_length=3, axis=-1)
    np.testing.assert_array_equal(padded_x, x)
    padded_x = common.pad_to_length(x, target_length=5, left=True, axis=-1)
    np.testing.assert_array_equal(
        padded_x, jnp.array([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="token_mean",
          loss_agg_mode="token-mean",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={},
          expected_loss=(0.1 + 0.2 + 0.4 + 0.5 + 0.6) / 5.0,
      ),
      dict(
          testcase_name="sequence_mean_token_mean",
          loss_agg_mode="sequence-mean-token-mean",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={},
          expected_loss=((0.1 + 0.2) / 2 + (0.4 + 0.5 + 0.6) / 3) / 2,
      ),
      dict(
          testcase_name="sequence_mean_token_scale",
          loss_agg_mode="sequence-mean-token-scale",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={},
          expected_loss=((0.1 + 0.2) / 3 + (0.4 + 0.5 + 0.6) / 3) / 2,
      ),
      dict(
          testcase_name="sequence_mean_token_scale_custom",
          loss_agg_mode="sequence-mean-token-scale",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={"norm": 3.14},
          expected_loss=((0.1 + 0.2) / 3.14 + (0.4 + 0.5 + 0.6) / 3.14) / 2,
      ),
      dict(
          testcase_name="sequence_mean_token_sum_norm_default",
          loss_agg_mode="sequence-mean-token-sum-norm",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={},
          expected_loss=(0.1 + 0.2 + 0.4 + 0.5 + 0.6) / 2.0,
      ),
      dict(
          testcase_name="sequence_mean_token_sum_norm_custom",
          loss_agg_mode="sequence-mean-token-sum-norm",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={"norm": 4.0},
          expected_loss=(0.1 + 0.2 + 0.4 + 0.5 + 0.6) / 4.0,
      ),
      dict(
          testcase_name="token_mean_zero_mask",
          loss_agg_mode="token-mean",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[0, 0], [0, 0]],
          kwargs={},
          expected_loss=0.0,
      ),
      dict(
          testcase_name="sequence_mean_token_mean_zero_mask",
          loss_agg_mode="sequence-mean-token-mean",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[0, 0], [0, 0]],
          kwargs={},
          expected_loss=0.0,
      ),
      dict(
          testcase_name="sequence_mean_token_sum_norm_zero_mask",
          loss_agg_mode="sequence-mean-token-sum-norm",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[0, 0], [0, 0]],
          kwargs={"norm": 4.0},
          expected_loss=0.0,
      ),
      dict(
          testcase_name="seq_mean_token_sum",
          loss_agg_mode="seq-mean-token-sum",
          per_token_loss_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          completion_mask_list=[[1, 1, 0], [1, 1, 1]],
          kwargs={},
          expected_loss=0.9,
      ),
      dict(
          testcase_name="seq_mean_token_sum_zero_mask",
          loss_agg_mode="seq-mean-token-sum",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[0, 0], [0, 0]],
          kwargs={},
          expected_loss=0.0,
      ),
      dict(
          testcase_name="seq_mean_token_sum_partial_zero_mask",
          loss_agg_mode="seq-mean-token-sum",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[1, 1], [0, 0]],
          kwargs={},
          expected_loss=0.3,
      ),
      dict(
          testcase_name="sequence_mean_token_sum_norm_partial_zero_mask",
          loss_agg_mode="sequence-mean-token-sum-norm",
          per_token_loss_list=[[0.1, 0.2], [0.3, 0.4]],
          completion_mask_list=[[1, 1], [0, 0]],
          kwargs={"norm": 4.0},
          expected_loss=(0.1 + 0.2) / 4.0,
      ),
  )
  def test_aggregate_loss_values(
      self,
      loss_agg_mode,
      per_token_loss_list,
      completion_mask_list,
      kwargs,
      expected_loss,
  ):
    per_token_loss = jnp.array(per_token_loss_list)
    completion_mask = jnp.array(completion_mask_list)
    actual_loss = _compute_loss(
        per_token_loss, completion_mask, loss_agg_mode, **kwargs
    )
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ("token_mean", "token-mean", {}),
      ("seq_mean_token_mean", "sequence-mean-token-mean", {}),
      ("seq_mean_token_scale", "sequence-mean-token-scale", {"norm": 5.0}),
      ("seq_mean_token_sum", "seq-mean-token-sum", {}),
      ("seq_mean_token_sum_norm", "sequence-mean-token-sum-norm", {"norm": 3.0}),
  )
  def test_reduced_equals_unreduced_compute(self, loss_agg_mode, kwargs):
    # reduced_loss_agg (eager scalar) must equal aggregate_loss(...).compute()
    # (deferred WeightedMetric) for every mode, including empty rows and
    # varying lengths. This ties the two independent aggregations together so
    # they cannot silently diverge.
    per_token_loss = jnp.array([
        [0.5, 1.5, 2.5, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
    ])
    completion_mask = jnp.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ])
    reduced = common.reduced_loss_agg(
        per_token_loss, completion_mask, loss_agg_mode, **kwargs
    )
    unreduced = _compute_loss(
        per_token_loss, completion_mask, loss_agg_mode, **kwargs
    )
    np.testing.assert_allclose(reduced, unreduced, rtol=1e-6, atol=1e-6)

  def test_mean_of_means_matches_legacy_np_mean(self):
    # mean_of_means computes each WeightedMetric then averages — exactly what
    # the old pipeline did (pre-compute to scalar, then np.mean). No regression.
    metrics = [
        utils.WeightedMetric(jnp.array(6.0), jnp.array(3.0)),  # -> 2.0
        utils.WeightedMetric(jnp.array(4.0), jnp.array(1.0)),  # -> 4.0
    ]
    legacy = np.mean([float(m.compute()) for m in metrics])
    self.assertAlmostEqual(float(common.mean_of_means(metrics)), legacy, places=6)
    self.assertAlmostEqual(float(common.mean_of_means(metrics)), 3.0, places=6)

  def test_mean_of_means_is_np_mean_for_scalars(self):
    # Safe drop-in for np.mean on plain scalars.
    self.assertAlmostEqual(
        float(common.mean_of_means([2.0, 4.0])), 3.0, places=6
    )

  def test_global_weighted_mean_is_sum_over_sum(self):
    # Global sum(S)/sum(d), and it DIVERGES from mean_of_means when the
    # denominator varies across micro-batches.
    metrics = [
        utils.WeightedMetric(jnp.array(10.0), jnp.array(4.0)),  # mean 2.5
        utils.WeightedMetric(jnp.array(2.0), jnp.array(1.0)),   # mean 2.0
    ]
    self.assertAlmostEqual(
        common.global_weighted_mean(metrics), 12.0 / 5.0, places=6
    )
    self.assertNotAlmostEqual(
        common.global_weighted_mean(metrics),
        float(common.mean_of_means(metrics)),  # 2.25
        places=3,
    )

  def test_global_weighted_mean_zero_denominator(self):
    metrics = [utils.WeightedMetric(jnp.array(0.0), jnp.array(0.0))]
    self.assertEqual(common.global_weighted_mean(metrics), 0.0)

  def test_invalid_mode(self):
    with self.assertRaisesRegex(
        ValueError, "Unsupported loss aggregation mode"
    ):
      _compute_loss(jnp.ones((2, 2)), jnp.ones((2, 2)), "invalid-mode")

  @parameterized.named_parameters(
      dict(
          testcase_name="norm_zero_token_sum_norm",
          norm_val=0,
          loss_agg_mode="sequence-mean-token-sum-norm",
      ),
      dict(
          testcase_name="norm_negative_token_sum_norm",
          norm_val=-1.0,
          loss_agg_mode="sequence-mean-token-sum-norm",
      ),
      dict(
          testcase_name="norm_string_token_sum_norm",
          norm_val="abc",
          loss_agg_mode="sequence-mean-token-sum-norm",
      ),
      dict(
          testcase_name="norm_zero_token_scale",
          norm_val=0,
          loss_agg_mode="sequence-mean-token-scale",
      ),
      dict(
          testcase_name="norm_negative_token_scale",
          norm_val=-1.0,
          loss_agg_mode="sequence-mean-token-scale",
      ),
      dict(
          testcase_name="norm_string_token_scale",
          norm_val="abc",
          loss_agg_mode="sequence-mean-token-scale",
      ),
  )
  def test_invalid_norm(self, norm_val, loss_agg_mode):
    with self.assertRaisesRegex(ValueError, "Invalid 'norm' value"):
      _compute_loss(
          jnp.ones((2, 2)),
          jnp.ones((2, 2)),
          loss_agg_mode,
          norm=norm_val,
      )

  def test_compute_kl_divergence_bf16(self):
    per_token_logps = jnp.array([-10.0, -1.0, 0.0], dtype=jnp.bfloat16)
    ref_per_token_logps = jnp.array([-1.0, -10.0, 0.0], dtype=jnp.bfloat16)

    kl = common.compute_kl_divergence(
        per_token_logps, ref_per_token_logps, method="low_var_kl"
    )
    self.assertEqual(kl.dtype, jnp.float32)
    expected_kl = common.compute_kl_divergence(
        per_token_logps.astype(jnp.float32),
        ref_per_token_logps.astype(jnp.float32),
        method="low_var_kl",
    )
    np.testing.assert_allclose(kl, expected_kl, rtol=1e-3)

  @parameterized.named_parameters(
      ("kl", "kl"),
      ("mse_kl", "mse_kl"),
      ("low_var_kl", "low_var_kl"),
  )
  def test_compute_kl_divergence_output_clamp_default_is_no_op(self, method):
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    per_token_logps = jax.random.uniform(k1, shape=(2, 2, 4))
    ref_per_token_logps = jax.random.uniform(k2, shape=(2, 2, 4))
    baseline = common.compute_kl_divergence(
        per_token_logps, ref_per_token_logps, method=method
    )
    with_explicit_none = common.compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        method=method,
        clamp_value=None,
    )
    np.testing.assert_array_equal(baseline, with_explicit_none)

  @parameterized.named_parameters(
      ("kl", "kl"),
      ("mse_kl", "mse_kl"),
      ("low_var_kl", "low_var_kl"),
  )
  def test_compute_kl_divergence_output_clamp_caps_outliers(self, method):
    per_token_logps = jnp.array([-50.0, 0.0, 50.0], dtype=jnp.float32)
    ref_per_token_logps = jnp.array([50.0, 0.0, -50.0], dtype=jnp.float32)
    clamp = 10.0
    kl = common.compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        method=method,
        clamp_value=clamp,
    )
    self.assertTrue(bool(jnp.all(kl >= -clamp)))
    self.assertTrue(bool(jnp.all(kl <= clamp)))

  def test_compute_kl_divergence_output_clamp_passes_through_in_range(self):
    per_token_logps = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    ref_per_token_logps = jnp.array([0.4, 0.5, 0.6], dtype=jnp.float32)
    unclamped = common.compute_kl_divergence(
        per_token_logps, ref_per_token_logps, method="kl"
    )
    clamped = common.compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        method="kl",
        clamp_value=10000.0,
    )
    np.testing.assert_array_equal(clamped, unclamped)

  def test_compute_kl_divergence_output_clamp_symmetric(self):
    per_token_logps = jnp.array([100.0], dtype=jnp.float32)
    ref_per_token_logps = jnp.array([-100.0], dtype=jnp.float32)
    pos = common.compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        method="kl",
        clamp_value=5.0,
    )
    neg = common.compute_kl_divergence(
        ref_per_token_logps,
        per_token_logps,
        method="kl",
        clamp_value=5.0,
    )
    np.testing.assert_array_equal(pos, jnp.array([5.0]))
    np.testing.assert_array_equal(neg, jnp.array([-5.0]))

  def test_aggregate_loss_bf16(self):
    per_token_loss = jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16)
    completion_mask = jnp.array([1, 1, 0], dtype=jnp.int32)

    loss = _compute_loss(
        per_token_loss, completion_mask, loss_agg_mode="token-mean"
    )
    self.assertEqual(loss.dtype, jnp.float32)
    self.assertAlmostEqual(loss, 1.5, places=5)

  def test_model_call_contains(self):
    class ModelWithSegIds:
      def __call__(self, x, segment_ids=None):
        pass

    class ModelWithoutSegIds:
      def __call__(self, x):
        pass

    class WrapperWithKwargs:
      def __init__(self, transformer):
        self.transformer = transformer

      def __call__(self, *args, **kwargs):
        pass

    class ModelWithKwargsOnly:
      def __call__(self, **kwargs):
        pass

    # 1. Test raw models
    self.assertTrue(
        common.model_call_contains(ModelWithSegIds(), "segment_ids")
    )
    self.assertFalse(
        common.model_call_contains(ModelWithoutSegIds(), "segment_ids")
    )

    # 2. Test wrapped models (looks at model.transformer)
    self.assertTrue(
        common.model_call_contains(
            WrapperWithKwargs(ModelWithSegIds()), "segment_ids"
        )
    )
    self.assertFalse(
        common.model_call_contains(
            WrapperWithKwargs(ModelWithoutSegIds()), "segment_ids"
        )
    )

    # 3. Test models with **kwargs
    self.assertTrue(
        common.model_call_contains(ModelWithKwargsOnly(), "segment_ids")
    )

  def test_compute_score_with_wrapped_model_without_segment_ids(self):
    """Test that segment ids are not passed when a model doesn't accept segment_ids.

    This test reproduces the GemmaWithScoreHead scenario where Gemma doesn't
    accept segment_ids.
    """

    class ModelWithoutSegIds(nnx.Module):
      def __call__(self, x, positions=None, cache=None, attention_mask=None):
        return jnp.zeros((*x.shape, 1))

    class ScoreHeadWrapper(nnx.Module):
      def __init__(self, transformer):
        self.transformer = transformer

      def __call__(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    wrapped_model = ScoreHeadWrapper(ModelWithoutSegIds())
    prompt_tokens = jnp.array([[1, 2]])
    completion_tokens = jnp.array([[3, 4]])

    # Verifies score is computed cleanly without raising
    # TypeError for segment_ids
    scores = common.compute_score(
        wrapped_model, prompt_tokens, completion_tokens, pad_id=0, eos_id=-1
    )
    self.assertEqual(scores.shape, (1, 4))

  def test_compute_per_token_logps_segment_ids_fallback(self):
    """Verifies compute_per_token_logps falls back cleanly if model doesn't support segment_ids."""

    class ModelWithoutSegIds(nnx.Module):
      def __call__(self, x, positions=None, cache=None, attention_mask=None):
        return jnp.zeros((*x.shape, 10)), cache

    model = ModelWithoutSegIds()
    graphdef, state = nnx.split(model)

    prompt_tokens = jnp.array([[1, 2]])
    completion_tokens = jnp.array([[3, 4]])

    # Packed mode on model without segment_ids should fall back
    # gracefully without TypeError
    logps_packed = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        segment_ids=jnp.ones((1, 4), dtype=jnp.int32),
        segment_positions=jnp.arange(4, dtype=jnp.int32),
    )
    self.assertEqual(logps_packed.shape, (1, 4))

  def test_compute_per_token_logps_segment_ids_supported(self):
    """Verifies compute_per_token_logps passes segment_ids to model when supplied and supported."""

    class ModelWithSegIds(nnx.Module):
      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        # Segment_ids should be passed when supplied and supported
        if segment_ids is None:
          raise ValueError("segment_ids should be passed when supported.")
        return jnp.zeros((*x.shape, 10)), cache

    model = ModelWithSegIds()
    graphdef, state = nnx.split(model)

    prompt_tokens = jnp.array([[1, 2]])
    completion_tokens = jnp.array([[3, 4]])

    # Verifies segment_ids are passed when supported and supplied
    logps_packed = common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        segment_ids=jnp.ones((1, 4), dtype=jnp.int32),
        segment_positions=jnp.arange(4, dtype=jnp.int32),
    )
    self.assertEqual(logps_packed.shape, (1, 4))

if __name__ == "__main__":
  absltest.main()
