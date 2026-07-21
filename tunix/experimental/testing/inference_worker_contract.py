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

"""Reusable contract suite (and a stub core) for InferenceWorker implementations.

`StubReferenceScoringCore` is a deterministic, row-independent,
temperature-dependent stand-in for a real reference/reward scoring core, so the
suite can run without a model. Mix `InferenceWorkerContractSuite` into an
`absltest.TestCase` and implement `make_worker()`; the shared tests pin the
scoring contract (shapes, echo, frozen-reference-only, temperature plumbing,
worker-internal micro-batching, wire-safe results). Exact log-prob values vs a
real reference model are a separate golden check, not part of this suite.
"""

import cloudpickle
import jax.numpy as jnp
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.common import rpc_utils


class StubReferenceScoringCore:
  """Deterministic, row-independent, temperature-dependent scoring core."""

  def get_ref_per_token_logps(
      self, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0
  ):
    del prompt_tokens, pad_id, eos_id
    return jnp.asarray(completion_tokens, dtype=jnp.float32) * temperature

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del prompt_tokens, pad_id, eos_id
    return jnp.asarray(completion_tokens, dtype=jnp.float32).sum(axis=1)


class InferenceWorkerContractSuite:
  """Contract tests shared across InferenceWorker implementations."""

  def make_worker(self):
    raise NotImplementedError("Subclasses must provide make_worker().")

  def _logprobs_request(
      self, batch: int = 4, temperature: float = 1.0, **overrides
  ) -> datatypes.LogprobsRequest:
    kwargs = dict(
        request_id="req-lp",
        prompt_tokens=np.ones((batch, 2), dtype=np.int32),
        completion_tokens=np.arange(batch * 3, dtype=np.int32).reshape(batch, 3),
        temperature=temperature,
    )
    kwargs.update(overrides)
    return datatypes.LogprobsRequest(**kwargs)

  def test_lifecycle_and_info(self):
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    self.assertEqual(worker.health().state, "READY")
    self.assertIn("inference", worker.info().roles)
    worker.stop()
    self.assertEqual(worker.health().state, "STOPPED")

  def test_compute_logprobs_shape_and_echo(self):
    worker = self.make_worker()
    result = worker.compute_logprobs(self._logprobs_request(batch=4))
    self.assertEqual(result.request_id, "req-lp")
    self.assertEqual(result.per_token_logps.shape, (4, 3))
    self.assertEqual(result.per_token_logps.dtype, np.float32)

  def test_compute_logprobs_rejects_non_reference_role(self):
    worker = self.make_worker()
    with self.assertRaises(NotImplementedError):
      worker.compute_logprobs(self._logprobs_request(model_role="policy"))

  def test_compute_logprobs_honors_temperature(self):
    worker = self.make_worker()
    at_one = worker.compute_logprobs(
        self._logprobs_request(batch=2, temperature=1.0)
    )
    at_two = worker.compute_logprobs(
        self._logprobs_request(batch=2, temperature=2.0)
    )
    self.assertFalse(
        np.allclose(at_one.per_token_logps, at_two.per_token_logps)
    )

  def test_microbatching_matches_single_pass(self):
    worker = self.make_worker()
    single = worker.compute_logprobs(self._logprobs_request(batch=6))
    chunked = worker.compute_logprobs(
        self._logprobs_request(batch=6, micro_batch_size=2)
    )
    np.testing.assert_array_equal(
        single.per_token_logps, chunked.per_token_logps
    )

  def test_score_shape_and_echo(self):
    worker = self.make_worker()
    request = datatypes.ScoreRequest(
        request_id="req-score",
        prompt_tokens=np.ones((3, 2), dtype=np.int32),
        completion_tokens=np.arange(9, dtype=np.int32).reshape(3, 3),
    )
    result = worker.score(request)
    self.assertEqual(result.request_id, "req-score")
    self.assertEqual(result.scores.shape, (3,))
    self.assertEqual(result.scores.dtype, np.float32)

  def test_result_is_wire_safe_and_round_trips(self):
    worker = self.make_worker()
    result = worker.compute_logprobs(self._logprobs_request(batch=2))
    rpc_utils.validate_wire_safe(result)
    restored = cloudpickle.loads(cloudpickle.dumps(result))
    np.testing.assert_array_equal(
        restored.per_token_logps, result.per_token_logps
    )
