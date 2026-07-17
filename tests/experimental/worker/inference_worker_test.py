# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the frozen-model InferenceWorker wrapper."""

import cloudpickle
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.worker import inference_worker as inference_lib


class _StubCore:
  """Deterministic, row-independent stand-in for the real inference core."""

  def __init__(self):
    self.calls = 0

  def get_ref_per_token_logps(
      self, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0
  ):
    del prompt_tokens, pad_id, eos_id
    self.calls += 1
    return jnp.asarray(completion_tokens, dtype=jnp.float32) * temperature

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del prompt_tokens, pad_id, eos_id
    self.calls += 1
    return jnp.asarray(completion_tokens, dtype=jnp.float32).sum(axis=1)


def _worker(core=None):
  return inference_lib.InferenceWorker(
      core if core is not None else _StubCore(),
      pad_id=0,
      eos_id=1,
      model_version=3,
  )


def _logprobs_request(batch=4, **overrides):
  kwargs = dict(
      request_id="r1",
      prompt_tokens=np.ones((batch, 2), dtype=np.int32),
      completion_tokens=np.arange(batch * 3, dtype=np.int32).reshape(batch, 3),
      temperature=1.0,
  )
  kwargs.update(overrides)
  return datatypes.LogprobsRequest(**kwargs)


class InferenceWorkerTest(absltest.TestCase):

  def test_compute_logprobs_shape_and_echo(self):
    result = _worker().compute_logprobs(_logprobs_request(batch=4))

    self.assertEqual(result.request_id, "r1")
    self.assertEqual(result.model_version, 3)
    self.assertEqual(result.per_token_logps.shape, (4, 3))
    self.assertEqual(result.per_token_logps.dtype, np.float32)

  def test_compute_logprobs_applies_temperature(self):
    req = _logprobs_request(batch=2, temperature=2.0)

    result = _worker().compute_logprobs(req)

    np.testing.assert_allclose(
        result.per_token_logps, req.completion_tokens.astype(np.float32) * 2.0
    )

  def test_microbatching_matches_single_pass(self):
    core_single = _StubCore()
    single = _worker(core_single).compute_logprobs(_logprobs_request(batch=6))

    core_chunked = _StubCore()
    chunked = _worker(core_chunked).compute_logprobs(
        _logprobs_request(batch=6, micro_batch_size=2)
    )

    np.testing.assert_array_equal(
        single.per_token_logps, chunked.per_token_logps
    )
    self.assertEqual(core_single.calls, 1)  # single pass
    self.assertEqual(core_chunked.calls, 3)  # 6 rows / 2 = 3 micro-batches

  def test_compute_logprobs_rejects_non_reference_role(self):
    with self.assertRaises(NotImplementedError):
      _worker().compute_logprobs(_logprobs_request(model_role="policy"))

  def test_score_shape_and_echo(self):
    req = datatypes.ScoreRequest(
        request_id="s1",
        prompt_tokens=np.ones((3, 2), dtype=np.int32),
        completion_tokens=np.arange(9, dtype=np.int32).reshape(3, 3),
    )

    result = _worker().score(req)

    self.assertEqual(result.request_id, "s1")
    self.assertEqual(result.model_version, 3)
    self.assertEqual(result.scores.shape, (3,))
    self.assertEqual(result.scores.dtype, np.float32)

  def test_start_stop_lifecycle(self):
    worker = _worker()
    self.assertFalse(worker._is_running)
    worker.start()
    self.assertTrue(worker._is_running)
    worker.stop()
    self.assertFalse(worker._is_running)

  def test_result_dtos_are_wire_safe_and_round_trip(self):
    result = _worker().compute_logprobs(_logprobs_request(batch=2))

    datatypes.validate_wire_safe(result)  # numpy result must pass the guard
    restored = cloudpickle.loads(cloudpickle.dumps(result))

    np.testing.assert_array_equal(
        restored.per_token_logps, result.per_token_logps
    )


if __name__ == "__main__":
  absltest.main()
