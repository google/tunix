"""Pinned-image controls for P57 processed-B target identity and values."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.layers.jax.sample.sampling_metadata import (
    TPUSupportedSamplingMetadata,
)
from tpu_inference.runner import p57_stock_prompt_observer as observer


class _Request:

  def __init__(self, tokens, *, computed=0):
    self._tokens = tuple(tokens)
    self.num_computed_tokens = computed
    self.num_prompt_tokens = len(tokens)

  def get_token_id(self, index):
    return self._tokens[index]


class P57StockPromptObserverTest(unittest.TestCase):

  def test_absolute_targets_do_not_roll_across_dp_or_request_boundaries(self):
    requests = {
        "a": _Request([11, 12, 13]),
        "b": _Request([21, 22]),
        "c": _Request([31, 32, 33, 34]),
    }
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"a": 3, "b": 2, "c": 4}
    )
    targets = observer._expand_absolute_target_ids(
        requests,
        scheduler,
        {0: ["a", "b"], 1: ["c"]},
        dp_size=2,
        padded_num_tokens=12,
    )
    np.testing.assert_array_equal(
        targets,
        np.array([12, 13, 0, 22, 0, 0, 32, 33, 34, 0, 0, 0], np.int32),
    )
    packed = np.array([11, 12, 13, 21, 22, 0, 31, 32, 33, 34, 0, 0])
    self.assertNotEqual(int(np.roll(packed, -1)[2]), int(targets[2]))
    self.assertNotEqual(int(np.roll(packed, -1)[5]), int(targets[5]))

  def test_temperature_transform_matches_direct_definition(self):
    logits = jnp.array([[1.0, 2.0, -0.5]], dtype=jnp.float32)
    metadata = TPUSupportedSamplingMetadata(
        temperature=jnp.array([0.7], dtype=jnp.float32),
        top_k=jnp.array([-1], dtype=jnp.int32),
        top_p=jnp.array([1.0], dtype=jnp.float32),
        do_sampling=True,
        logprobs=True,
    )
    actual = np.asarray(observer._process_prompt_logits(logits, metadata))
    np.testing.assert_array_equal(actual, np.asarray(logits / 0.7))

  def test_end_to_end_tensors_keep_target_identity_and_processed_value(self):
    request = _Request([10, 20, 30, 40])
    input_batch = SimpleNamespace(
        req_id_to_index={"r": 0},
        temperature_cpu=np.array([0.7], np.float32),
        top_k_cpu=np.array([-1], np.int32),
        top_p_cpu=np.array([1.0], np.float32),
        num_prompt_logprobs={"r": 0},
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={"r": 4})
    logits = jnp.arange(4 * 64, dtype=jnp.float32).reshape(4, 64) / 100
    mesh = Mesh(np.array(jax.devices()), ("data",))
    with jax.set_mesh(mesh):
      output = observer.compute_processed_prompt_logprobs(
          mesh=mesh,
          full_logits=logits,
          input_batch=input_batch,
          requests={"r": request},
          scheduler_output=scheduler,
          req_ids_dp={0: ["r"]},
          dp_size=1,
          max_logprobs=1,
      )
    np.testing.assert_array_equal(
        np.asarray(output.tensors.logprob_token_ids)[:, 0],
        np.array([20, 30, 40, 0], np.int32),
    )
    expected = jax.nn.log_softmax(logits / 0.7, axis=-1)
    np.testing.assert_array_equal(
        np.asarray(output.tensors.logprobs)[:3, 0],
        np.asarray(expected)[np.arange(3), np.array([20, 30, 40])],
    )
    self.assertEqual(len(output.req_snaps), 1)
    self.assertEqual(output.req_snaps[0].num_logits, 3)
    self.assertTrue(output.req_snaps[0].is_last_chunk)


if __name__ == "__main__":
  unittest.main()
