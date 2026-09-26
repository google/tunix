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

  def test_packed_logps_match_unpacked_per_segment(self):
    """Packed segment-aware per-token logps == unpacked per-row logps.

    Packs `num_seq` sequences into a single row and checks that
    `compute_per_token_logps` with `segment_ids` reproduces, token-for-token,
    the logps computed with each sequence on its own row. This is the core
    correctness property of pack-first log-probs: the segment-aware forward must
    not let one packed sequence leak into another. Uses a self-contained
    segment-aware toy (attention confined to same-segment positions) so the
    check runs on CPU with no real model. The first token of each packed segment
    is a cross-segment-boundary prediction (masked out downstream by
    `completion_mask`), so only positions `t >= 1` within each segment are
    compared.
    """

    class _SegAwareToy(nnx.Module):
      """Tiny model whose attention is confined to same-segment positions."""

      def __init__(self, *, vocab, dim, rngs):
        self.emb = nnx.Embed(vocab, dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=dim,
            qkv_features=dim,
            use_bias=False,
            decode=False,
            rngs=rngs,
        )
        self.head = nnx.Linear(dim, vocab, rngs=rngs)

      def __call__(
          self,
          x,
          segment_ids=None,
          positions=None,
          cache=None,
          attention_mask=None,
      ):
        h = self.emb(x)
        if segment_ids is not None:
          same_seg = segment_ids[:, :, None] == segment_ids[:, None, :]
          h = self.attn(h, mask=same_seg[:, None, :, :]) + h
        else:
          h = self.attn(h) + h
        return self.head(h), cache

    model = _SegAwareToy(vocab=16, dim=16, rngs=nnx.Rngs(42))
    graphdef, state = nnx.split(model)
    num_seq, seq_len = 3, 4
    tokens = (
        np.random.default_rng(7)
        .integers(1, 16, size=(num_seq, seq_len))
        .astype(np.int32)
    )

    def _logps(completion_tokens, segment_ids, segment_positions):
      return np.asarray(
          common.compute_per_token_logps(
              graphdef,
              state,
              jnp.zeros((completion_tokens.shape[0], 0), dtype=jnp.int32),
              jnp.asarray(completion_tokens, jnp.int32),
              pad_id=0,
              eos_id=-1,
              segment_ids=jnp.asarray(segment_ids, jnp.int32),
              segment_positions=jnp.asarray(segment_positions, jnp.int32),
          )
      )

    # Unpacked: [num_seq, seq_len], one segment per row.
    unpacked = _logps(
        tokens,
        np.ones((num_seq, seq_len), np.int32),
        np.broadcast_to(np.arange(seq_len, dtype=np.int32), (num_seq, seq_len)),
    )

    # Packed: [1, num_seq * seq_len], one row holding num_seq segments.
    packed = _logps(
        tokens.reshape(1, -1),
        np.concatenate(
            [np.full(seq_len, i + 1, np.int32) for i in range(num_seq)]
        )[None, :],
        np.concatenate(
            [np.arange(seq_len, dtype=np.int32) for _ in range(num_seq)]
        )[None, :],
    ).reshape(num_seq, seq_len)

    np.testing.assert_allclose(
        packed[:, 1:], unpacked[:, 1:], atol=1e-4, rtol=1e-4
    )


_ROUTING_LAYERS = 2
_ROUTING_TOP_K = 2
_UNSET = common.UNSET_ROUTED_EXPERT


def _routing(length, fill):
  """`[length, num_layers, top_k]` routing where every slot holds `fill`."""
  return np.full(
      (length, _ROUTING_LAYERS, _ROUTING_TOP_K), fill, dtype=np.int32
  )


class AlignRoutedExpertsTest(parameterized.TestCase):
  """Rollout routing must be padded exactly like the token ids it describes."""

  def test_prompt_right_aligned_and_completion_left_aligned(self):
    """Prompts are left-padded and completions right-padded, so routing must

    follow, or every replayed expert lands on the wrong token.
    """
    prompt_len, completion_len = 2, 3
    prompt_width, completion_width = 5, 6
    routed = np.concatenate(
        [_routing(prompt_len, 7), _routing(completion_len, 9)], axis=0
    )

    out = common.align_routed_experts(
        [routed],
        completion_lengths=[completion_len],
        prompt_width=prompt_width,
        completion_width=completion_width,
    )

    self.assertEqual(out.dtype, np.int16)
    self.assertEqual(
        out.shape,
        (1, prompt_width + completion_width, _ROUTING_LAYERS, _ROUTING_TOP_K),
    )
    row = out[0]
    np.testing.assert_array_equal(row[: prompt_width - prompt_len], _UNSET)
    np.testing.assert_array_equal(
        row[prompt_width - prompt_len : prompt_width], 7
    )
    np.testing.assert_array_equal(
        row[prompt_width : prompt_width + completion_len], 9
    )
    np.testing.assert_array_equal(row[prompt_width + completion_len :], _UNSET)

  def test_batches_rows_in_order(self):
    """Row i's routing must stay on row i, or rows train on each other's."""
    rows = [
        np.concatenate([_routing(1, i), _routing(2, i)], axis=0)
        for i in range(3)
    ]
    out = common.align_routed_experts(
        rows, completion_lengths=[2] * 3, prompt_width=1, completion_width=2
    )
    self.assertEqual(out.shape[0], 3)
    for i in range(3):
      np.testing.assert_array_equal(out[i], i)

  @parameterized.named_parameters(
      ("nothing_captured", None),
      ("empty", []),
      ("partially_captured", "partial"),
  )
  def test_replay_is_all_or_nothing(self, rows):
    """A half-replayed batch would mix replayed and freshly routed rows."""
    if rows == "partial":
      rows = [np.concatenate([_routing(1, 1), _routing(2, 1)], axis=0), None]
    lengths = [2] * (len(rows) if rows else 0)
    self.assertIsNone(
        common.align_routed_experts(
            rows,
            completion_lengths=lengths,
            prompt_width=1,
            completion_width=2,
        )
    )

  def test_overlong_prompt_keeps_the_tail(self):
    """Left padding keeps the last prompt tokens, so routing must too."""
    prompt = np.concatenate([_routing(1, 5), _routing(1, 6)], axis=0)
    routed = np.concatenate([prompt, _routing(1, 9)], axis=0)
    out = common.align_routed_experts(
        [routed], completion_lengths=[1], prompt_width=1, completion_width=1
    )
    # Prompt row 5 is dropped; 6 survives.
    np.testing.assert_array_equal(out[0, 0], 6)
    np.testing.assert_array_equal(out[0, 1], 9)

  def test_wrong_rank_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "length, num_layers, top_k"):
      common.align_routed_experts(
          [np.zeros((3, _ROUTING_TOP_K), dtype=np.int32)],
          completion_lengths=[2],
          prompt_width=1,
          completion_width=2,
      )

  def test_upcasts_int16_routed_experts_to_int32_on_trainer_end(self):
    """Trainer receives int16 routing over IPC and upcasts to int32 on device."""
    captured = {}

    class DummyModel(nnx.Module):

      def __call__(
          self,
          x,
          positions=None,
          attention_mask=None,
          cache=None,
          forced_routed_experts=None,
          **kwargs,
      ):
        del positions, attention_mask, cache, kwargs
        captured["forced_routed_experts"] = forced_routed_experts
        return jnp.zeros((x.shape[0], x.shape[1], 16), dtype=jnp.float32), None

    model = DummyModel()
    graphdef, state = nnx.split(model)
    routed_int16 = jnp.ones((1, 4, 2, 2), dtype=jnp.int16)
    common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=jnp.ones((1, 2), dtype=jnp.int32),
        completion_tokens=jnp.ones((1, 2), dtype=jnp.int32),
        pad_id=0,
        eos_id=1,
        routed_experts=routed_int16,
    )
    self.assertIn("forced_routed_experts", captured)
    self.assertEqual(captured["forced_routed_experts"].dtype, jnp.int32)


class SamplerTrainerAgreementTest(parameterized.TestCase):

  def test_returns_empty_when_logps_missing(self):
    mask = jnp.ones((2, 3), dtype=jnp.int32)
    logps = jnp.zeros((2, 3), dtype=jnp.float32)
    metrics, weights, filtered_mask = common.sampler_trainer_agreement(
        None, logps, mask
    )
    self.assertEqual((metrics, weights), ({}, None))
    np.testing.assert_array_equal(filtered_mask, mask)
    metrics, weights, filtered_mask = common.sampler_trainer_agreement(
        logps, None, mask
    )
    self.assertEqual((metrics, weights), ({}, None))
    np.testing.assert_array_equal(filtered_mask, mask)

  @parameterized.named_parameters(
      ("trainer_shape_mismatch", (2, 3), (2, 4), (2, 3)),
      ("mask_shape_mismatch", (2, 3), (2, 3), (2, 4)),
      ("broadcastable_trainer_shape_mismatch", (2, 1), (2, 3), (2, 3)),
      ("broadcastable_mask_shape_mismatch", (2, 3), (2, 3), (1, 3)),
  )
  def test_shape_mismatch_raises_value_error(
      self, rollout_shape, trainer_shape, mask_shape
  ):
    rollout = jnp.zeros(rollout_shape, dtype=jnp.float32)
    trainer = jnp.zeros(trainer_shape, dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      common.sampler_trainer_agreement(rollout, trainer, mask)

  def test_none_completion_mask_raises_value_error(self):
    logps = jnp.zeros((2, 3), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "Completion mask"):
      common.sampler_trainer_agreement(logps, logps, None)

  def test_identical_logps_report_agreement(self):
    logps = jnp.array(
        [[-0.1, -0.5, -2.0], [-0.3, -1.0, -0.2]], dtype=jnp.float32
    )
    mask = jnp.ones((2, 3), dtype=jnp.int32)
    metrics, weights, filtered_mask = common.sampler_trainer_agreement(
        logps, logps, mask
    )
    self.assertIsNone(weights)
    np.testing.assert_array_equal(filtered_mask, mask)
    self.assertAlmostEqual(metrics["sampler_trainer/logp_diff_mean"][0], 0.0)
    self.assertAlmostEqual(metrics["sampler_trainer/logp_diff_max"][0], 0.0)
    self.assertAlmostEqual(metrics["sampler_trainer/prob_diff_mean"][0], 0.0)
    self.assertAlmostEqual(
        metrics["sampler_trainer/probs_pearson_corr"][0], 1.0, places=4
    )
    # No importance-sampling metrics unless sampler_is == "token".
    self.assertNotIn("sampler_is/weight_mean", metrics)

  def test_logp_diff_uses_only_masked_positions(self):
    rollout = jnp.array([[-0.1, -0.5, -2.0]], dtype=jnp.float32)
    trainer = jnp.array([[-0.1, -0.5, -5.0]], dtype=jnp.float32)
    # Mask out the divergent third token; the metric must ignore it.
    mask = jnp.array([[1, 1, 0]], dtype=jnp.int32)
    metrics, _, _ = common.sampler_trainer_agreement(rollout, trainer, mask)
    self.assertAlmostEqual(metrics["sampler_trainer/logp_diff_mean"][0], 0.0)
    self.assertAlmostEqual(metrics["sampler_trainer/logp_diff_max"][0], 0.0)

  def test_token_importance_sampling_weights(self):
    rollout = jnp.array([[-1.0, -1.0, -1.0]], dtype=jnp.float32)
    # trainer - rollout = log([1, 3, 10]) -> exp(log_ratio) = [1, 3, 10].
    trainer = rollout + jnp.log(
        jnp.array([[1.0, 3.0, 10.0]], dtype=jnp.float32)
    )
    mask = jnp.ones((1, 3), dtype=jnp.int32)
    metrics, weights, filtered_mask = common.sampler_trainer_agreement(
        rollout, trainer, mask, sampler_is="token", sampler_is_threshold=2.0
    )
    np.testing.assert_array_equal(filtered_mask, mask)
    self.assertIsNotNone(weights)
    # Clamped at threshold 2.0: [1, 2, 2].
    np.testing.assert_allclose(
        np.asarray(weights), np.array([[1.0, 2.0, 2.0]]), rtol=1e-5
    )
    self.assertIn("sampler_is/weight_mean", metrics)
    self.assertAlmostEqual(metrics["sampler_is/weight_max"][0], 2.0, places=5)
    # Two of three positions (3x, 10x) exceed the threshold.
    self.assertAlmostEqual(
        metrics["sampler_is/frac_clipped_at_threshold"][0], 2.0 / 3.0, places=5
    )

  def test_seq_logprob_error_threshold_unpacked_masks_divergent_sequence(self):
    # Seq 0: |diff| = 0.1 -> exp(0.1) = 1.105 <= 2.0 (kept)
    # Seq 1: |diff| = 1.0 -> exp(1.0) = 2.718 > 2.0 (masked out)
    rollout = jnp.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], dtype=jnp.float32)
    trainer = jnp.array([[-0.9, -1.1, -1.0], [-2.0, -2.0, -2.0]], dtype=jnp.float32)
    mask = jnp.ones((2, 3), dtype=jnp.int32)
    metrics, weights, filtered_mask = common.sampler_trainer_agreement(
        rollout,
        trainer,
        mask,
        sampler_is="token",
        sampler_is_threshold=2.0,
        seq_logprob_error_threshold=2.0,
    )
    self.assertIsNotNone(filtered_mask)
    np.testing.assert_array_equal(
        np.asarray(filtered_mask),
        np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int32),
    )
    # TIS weights on the rejected sequence must also be zeroed out.
    np.testing.assert_array_equal(np.asarray(weights)[1], np.zeros(3))
    self.assertAlmostEqual(
        metrics["sampler_trainer/seq_error_masked_frac"][0], 0.5, places=5
    )
    self.assertAlmostEqual(
        metrics["sampler_trainer/seq_error_masked_count"][0], 1.0, places=5
    )

  def test_seq_logprob_error_threshold_packed_masks_only_divergent_segment(self):
    # Single packed row with 2 segments (seg 1: tokens 0..1, seg 2: tokens 2..3, pad: token 4)
    # Seg 1 has |diff| = 0.0 -> mult_err = 1.0 <= 2.0 (kept)
    # Seg 2 has |diff| = 1.2 -> mult_err = exp(1.2) = 3.32 > 2.0 (masked out)
    rollout = jnp.array([[-1.0, -1.0, -1.0, -1.0, 0.0]], dtype=jnp.float32)
    trainer = jnp.array([[-1.0, -1.0, -2.2, -2.2, 0.0]], dtype=jnp.float32)
    mask = jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32)
    segment_ids = jnp.array([[1, 1, 2, 2, 0]], dtype=jnp.int32)
    metrics, _, filtered_mask = common.sampler_trainer_agreement(
        rollout,
        trainer,
        mask,
        seq_logprob_error_threshold=2.0,
        segment_ids=segment_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(filtered_mask),
        np.array([[1, 1, 0, 0, 0]], dtype=np.int32),
    )
    self.assertAlmostEqual(
        metrics["sampler_trainer/seq_error_masked_frac"][0], 0.5, places=5
    )

  def test_seq_logprob_error_threshold_packed_all_padding_and_negative_ids(self):
    # Degenerate all-padding row (with 0 and -1 pad segment IDs) should not
    # fail or wrap negative indices in take_along_axis.
    rollout = jnp.array([[-1.0, -1.0, -1.0]], dtype=jnp.float32)
    trainer = jnp.array([[-1.0, -1.0, -1.0]], dtype=jnp.float32)
    mask = jnp.array([[0, 0, 0]], dtype=jnp.int32)
    segment_ids = jnp.array([[-1, 0, -1]], dtype=jnp.int32)
    metrics, _, filtered_mask = common.sampler_trainer_agreement(
        rollout,
        trainer,
        mask,
        seq_logprob_error_threshold=2.0,
        segment_ids=segment_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(filtered_mask),
        np.array([[0, 0, 0]], dtype=np.int32),
    )
    self.assertAlmostEqual(
        metrics["sampler_trainer/seq_error_masked_frac"][0], 0.0, places=5
    )

  def test_low_variance_identical_logps_preserve_unit_pearson(self):
    # Converged policies often emit high-confidence tokens in [0.998, 1.000]
    # where sigma_p ~ 5e-4 (sigma_p^4 ~ 6e-14 < 1e-12). Pearson correlation must
    # still report 1.0 rather than collapsing due to a premature variance clamp.
    logps = jnp.array(
        [[-0.0020, -0.0015, -0.0010, -0.0005, -0.0012, -0.0018]],
        dtype=jnp.float32,
    )
    mask = jnp.ones_like(logps, dtype=jnp.int32)
    metrics, _, _ = common.sampler_trainer_agreement(logps, logps, mask)
    self.assertAlmostEqual(
        metrics["sampler_trainer/probs_pearson_corr"][0], 1.0, places=5
    )

  def test_multi_microbatch_weighted_agreement_reduction(self):
    # Microbatch 1: high-variance batch with strong agreement (r ~ 1.0).
    mb1_r = jnp.array([[-0.1, -0.5, -1.2, -2.0]], dtype=jnp.float32)
    mb1_t = jnp.array([[-0.1, -0.5, -1.2, -2.0]], dtype=jnp.float32)
    mb1_m = jnp.ones_like(mb1_r, dtype=jnp.int32)
    # Microbatch 2: constant probabilities (zero variance -> cov_sum=0, std_sum=0).
    mb2_r = jnp.array([[-0.001, -0.001, -0.001, -0.001]], dtype=jnp.float32)
    mb2_t = jnp.array([[-0.001, -0.001, -0.001, -0.001]], dtype=jnp.float32)
    mb2_m = jnp.ones_like(mb2_r, dtype=jnp.int32)

    m1, _, _ = common.compute_sampler_trainer_agreement_jax(mb1_r, mb1_t, mb1_m)
    m2, _, _ = common.compute_sampler_trainer_agreement_jax(mb2_r, mb2_t, mb2_m)
    p1, op = m1["sampler_trainer/probs_pearson_corr"]
    p2, _ = m2["sampler_trainer/probs_pearson_corr"]
    # Zero-variance microbatch has std_sum=0 and should not drag the step
    # correlation down to 0.5 when reduced with weighted_metric_mean.
    self.assertAlmostEqual(op([p1, p2]), 1.0, places=5)

  def test_chan_parallel_pearson_equivalence_with_global_batch(self):
    """Proves Chan's parallel covariance/variance formula equals global Pearson."""
    rng = np.random.default_rng(12345)

    def _microbatch_chan_stats(rollout_logps, trainer_logps, comp_mask):
      """Per-microbatch sufficient statistics (n, mean_x, mean_y, C_xy, C_xx, C_yy)."""
      mf = np.asarray(comp_mask, dtype=np.float64).reshape(-1)
      rp = np.exp(np.asarray(rollout_logps, dtype=np.float64)).reshape(-1)
      tp = np.exp(np.asarray(trainer_logps, dtype=np.float64)).reshape(-1)
      n = mf.sum()
      denom = max(n, 1.0)
      mean_x = (rp * mf).sum() / denom
      mean_y = (tp * mf).sum() / denom
      dx = (rp - mean_x) * mf
      dy = (tp - mean_y) * mf
      c_xy = (dx * dy).sum()
      c_xx = (dx * dx).sum()
      c_yy = (dy * dy).sum()
      return n, mean_x, mean_y, c_xy, c_xx, c_yy

    def _reduce_chan_pearson(stats_list):
      """Reduces per-microbatch Chan stats via the Law of Total Covariance."""
      ns = np.array([s[0] for s in stats_list], dtype=np.float64)
      mean_xs = np.array([s[1] for s in stats_list], dtype=np.float64)
      mean_ys = np.array([s[2] for s in stats_list], dtype=np.float64)
      c_xys = np.array([s[3] for s in stats_list], dtype=np.float64)
      c_xxs = np.array([s[4] for s in stats_list], dtype=np.float64)
      c_yys = np.array([s[5] for s in stats_list], dtype=np.float64)

      total_n = ns.sum()
      if total_n == 0:
        return 0.0
      global_mean_x = (ns * mean_xs).sum() / total_n
      global_mean_y = (ns * mean_ys).sum() / total_n

      total_c_xy = c_xys.sum() + (
          ns * (mean_xs - global_mean_x) * (mean_ys - global_mean_y)
      ).sum()
      total_c_xx = c_xxs.sum() + (ns * (mean_xs - global_mean_x) ** 2).sum()
      total_c_yy = c_yys.sum() + (ns * (mean_ys - global_mean_y) ** 2).sum()
      return float(
          total_c_xy / np.sqrt(max(total_c_xx * total_c_yy, 1e-24))
      )

    # Case 1: 8 heterogeneous microbatches with different prompt means, lengths,
    # and masks (including an all-masked empty microbatch and a constant microbatch).
    microbatches = []
    for m, prompt_base_logp in enumerate(
        [-0.002, -0.8, -0.05, -1.5, -0.001, -0.3, -2.2, -0.15]
    ):
      b, l = 4, 16
      if m == 4:
        # Constant low-entropy microbatch (zero within-microbatch variance)
        r_logps = np.full((b, l), prompt_base_logp, dtype=np.float32)
        t_logps = r_logps + np.float32(1e-4)
        mask = np.ones((b, l), dtype=np.int32)
      elif m == 6:
        # Empty (all-masked) microbatch
        r_logps = rng.uniform(-2.0, -0.01, size=(b, l)).astype(np.float32)
        t_logps = r_logps + rng.normal(0.0, 0.02, size=(b, l)).astype(np.float32)
        mask = np.zeros((b, l), dtype=np.int32)
      else:
        r_logps = (
            prompt_base_logp
            + rng.normal(0.0, 0.15, size=(b, l))
        ).clip(-5.0, -1e-4).astype(np.float32)
        t_logps = (
            r_logps + rng.normal(0.0, 0.01, size=(b, l)).astype(np.float32)
        ).clip(-5.0, -1e-4)
        mask = (rng.uniform(0.0, 1.0, size=(b, l)) > 0.25).astype(np.int32)
      microbatches.append((r_logps, t_logps, mask))

    # Compute Chan's parallel Pearson correlation across the 8 microbatches.
    stats_list = [
        _microbatch_chan_stats(r, t, m) for r, t, m in microbatches
    ]
    chan_pearson = _reduce_chan_pearson(stats_list)

    # Compute the ground-truth global Pearson correlation on the single
    # concatenated batch of all 8 * 4 = 32 sequences.
    global_r = np.concatenate([r for r, _, _ in microbatches], axis=0)
    global_t = np.concatenate([t for _, t, _ in microbatches], axis=0)
    global_m = np.concatenate([m for _, _, m in microbatches], axis=0)
    global_metrics, _, _ = common.sampler_trainer_agreement(
        global_r, global_t, global_m
    )
    global_pearson = global_metrics["sampler_trainer/probs_pearson_corr"][0]

    np.testing.assert_allclose(chan_pearson, global_pearson, rtol=1e-7, atol=1e-7)

    # Also verify the production JAX implementation (`compute_sampler_trainer_agreement_jax`
    # reduced via `utils.weighted_metric_mean` / `common.global_weighted_mean`)
    # matches the concatenated global batch correlation within float32 tolerance.
    mb_jax_metrics = [
        common.compute_sampler_trainer_agreement_jax(
            jnp.asarray(r), jnp.asarray(t), jnp.asarray(m)
        )[0]["sampler_trainer/probs_pearson_corr"][0]
        for r, t, m in microbatches
    ]
    prod_chan_pearson = utils.weighted_metric_mean(mb_jax_metrics)
    np.testing.assert_allclose(
        prod_chan_pearson, global_pearson, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        common.global_weighted_mean(mb_jax_metrics),
        global_pearson,
        rtol=1e-7,
        atol=1e-7,
    )

    # Case 2: Regime where within-microbatch variance is near zero (each prompt
    # is deterministic with tiny bf16 noise), while between-microbatch variance
    # across prompts is large. Within-only correlation collapses, whereas Chan's
    # parallel formula recovers the exact global correlation (~0.9999).
    mb_easy_r = np.full((4, 8), -0.001, dtype=np.float32)
    mb_easy_t = mb_easy_r + np.array(
        [[1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4]] * 4,
        dtype=np.float32,
    )
    mb_hard_r = np.full((4, 8), -0.700, dtype=np.float32)
    mb_hard_t = mb_hard_r + np.array(
        [[1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4, 1e-4, -1e-4]] * 4,
        dtype=np.float32,
    )
    mask_4x8 = np.ones((4, 8), dtype=np.int32)
    chan_two_prompt = _reduce_chan_pearson([
        _microbatch_chan_stats(mb_easy_r, mb_easy_t, mask_4x8),
        _microbatch_chan_stats(mb_hard_r, mb_hard_t, mask_4x8),
    ])
    concat_metrics, _, _ = common.sampler_trainer_agreement(
        np.concatenate([mb_easy_r, mb_hard_r], axis=0),
        np.concatenate([mb_easy_t, mb_hard_t], axis=0),
        np.concatenate([mask_4x8, mask_4x8], axis=0),
    )
    np.testing.assert_allclose(
        chan_two_prompt,
        concat_metrics["sampler_trainer/probs_pearson_corr"][0],
        rtol=1e-7,
        atol=1e-7,
    )
    self.assertGreater(chan_two_prompt, 0.9999)

    prod_two_prompt = common.global_weighted_mean([
        common.compute_sampler_trainer_agreement_jax(
            jnp.asarray(mb_easy_r),
            jnp.asarray(mb_easy_t),
            jnp.asarray(mask_4x8),
        )[0]["sampler_trainer/probs_pearson_corr"][0],
        common.compute_sampler_trainer_agreement_jax(
            jnp.asarray(mb_hard_r),
            jnp.asarray(mb_hard_t),
            jnp.asarray(mask_4x8),
        )[0]["sampler_trainer/probs_pearson_corr"][0],
    ])
    np.testing.assert_allclose(
        prod_two_prompt,
        concat_metrics["sampler_trainer/probs_pearson_corr"][0],
        rtol=1e-7,
        atol=1e-7,
    )
    self.assertGreater(prod_two_prompt, 0.9999)


class ProcessIdsTokenMaskTest(absltest.TestCase):

  def test_explicit_token_mask_marks_validity_independent_of_pad_id(self):
    prompt = jnp.array([[0, 5]])
    completion = jnp.array([[6, 0, 7, 0]])  # a real token equal to pad id (0)
    explicit = jnp.array([[1, 1, 1, 1, 1, 0]])
    _, positions, attention, segments = common.process_ids(
        prompt, completion, 0, 255, token_mask=explicit
    )
    np.testing.assert_array_equal(positions, [[0, 1, 2, 3, 4, 4]])
    np.testing.assert_array_equal(segments, explicit)
    expected = np.tril(np.ones((6, 6), dtype=bool))
    expected[:, 5] = False
    np.testing.assert_array_equal(attention[0], expected)
    # Without token_mask the legacy pad-id rule is unchanged.
    _, old_positions, _, old_segments = common.process_ids(
        prompt, completion, 0, 255
    )
    np.testing.assert_array_equal(old_positions, [[0, 0, 1, 1, 2, 2]])
    np.testing.assert_array_equal(old_segments, [[0, 1, 1, 0, 1, 0]])
    with self.assertRaises(ValueError):  # shape must cover prompt+completion
      common.process_ids(
          jnp.array([[1, 2]]),
          jnp.array([[3, 4]]),
          0,
          255,
          token_mask=np.ones((1, 3), bool),
      )
    with self.assertRaises(ValueError):  # exclusive with packed segments
      common.process_ids(
          jnp.array([[1, 2]]),
          jnp.array([[3, 4]]),
          0,
          255,
          token_mask=np.ones((1, 4), bool),
          segment_ids=np.ones((1, 4), np.int32),
          segment_positions=np.arange(4)[None],
      )


if __name__ == "__main__":
  absltest.main()
