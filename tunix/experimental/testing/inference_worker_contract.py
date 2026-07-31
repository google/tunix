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

"""Reusable contract suite (and a stub core) for inference-worker implementations.

`StubReferenceScoringCore` is a deterministic, row-independent,
temperature-dependent stand-in for a real scoring core, so the suite runs
without a model. Mix `InferenceWorkerContractSuite` into an
`absltest.TestCase` and implement `make_worker()`; the tests pin the scoring
contract an orchestrator relies on: response shapes, request-id echo,
frozen-reference-only routing, temperature plumbing, chunking that does not
change results, failures reported in-band rather than raised, and responses
that survive the wire.

Exact log-prob values against a real reference model are a separate golden
check, not part of this suite.
"""

from typing import Any

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
  """Contract tests shared across inference-worker implementations."""

  def make_worker(self, chunk_size: int | None = None) -> Any:
    """Returns a fresh inference worker backed by a deterministic core."""
    raise NotImplementedError("Subclasses must provide make_worker().")

  def _logprobs_request(
      self, batch: int = 4, temperature: float = 1.0, **overrides
  ) -> datatypes.LogprobsRequest:
    kwargs: dict[str, Any] = dict(
        request_id="req-lp",
        prompt_tokens=np.ones((batch, 2), dtype=np.int32),
        completion_tokens=np.arange(batch * 3, dtype=np.int32).reshape(
            batch, 3
        ),
        temperature=temperature,
    )
    kwargs.update(overrides)
    return datatypes.LogprobsRequest(**kwargs)

  def test_reports_its_role_to_the_control_plane(self):
    worker = self.make_worker()
    self.assertIn("inference", worker.info().roles)

  def test_lifecycle_reaches_ready_then_stopped(self):
    worker = self.make_worker()

    worker.initialize()
    worker.start()
    self.assertEqual(
        worker.heartbeat().state, datatypes.WorkerState.READY
    )

    worker.stop()
    self.assertEqual(
        worker.heartbeat().state, datatypes.WorkerState.STOPPED
    )

  def test_scoring_echoes_the_request_and_shapes_the_response(self):
    worker = self.make_worker()

    response = worker.compute_logps(self._logprobs_request(batch=4))

    self.assertIsNone(response.error)
    self.assertEqual(response.request_id, "req-lp")
    self.assertEqual(response.per_token_logps.shape, (4, 3))
    self.assertEqual(response.per_token_logps.dtype, np.float32)

  def test_scoring_a_non_reference_role_fails_in_band(self):
    worker = self.make_worker()

    response = worker.compute_logps(
        self._logprobs_request(model_role="policy")
    )

    self.assertIsNotNone(response.error)
    self.assertEqual(response.request_id, "req-lp")

  def test_scoring_honors_temperature(self):
    worker = self.make_worker()

    at_one = worker.compute_logps(
        self._logprobs_request(batch=2, temperature=1.0)
    )
    at_two = worker.compute_logps(
        self._logprobs_request(batch=2, temperature=2.0)
    )

    self.assertFalse(
        np.allclose(at_one.per_token_logps, at_two.per_token_logps)
    )

  def test_non_positive_temperature_fails_in_band(self):
    worker = self.make_worker()

    response = worker.compute_logps(self._logprobs_request(temperature=0.0))

    self.assertIsNotNone(response.error)

  def test_chunking_does_not_change_the_result(self):
    single = self.make_worker().compute_logps(self._logprobs_request(batch=6))
    chunked = self.make_worker(chunk_size=2).compute_logps(
        self._logprobs_request(batch=6)
    )

    np.testing.assert_array_equal(
        single.per_token_logps, chunked.per_token_logps
    )

  def test_response_is_wire_safe_and_round_trips(self):
    worker = self.make_worker()

    response = worker.compute_logps(self._logprobs_request(batch=2))

    rpc_utils.validate_wire_safe(response)
    restored = cloudpickle.loads(cloudpickle.dumps(response))
    np.testing.assert_array_equal(
        restored.per_token_logps, response.per_token_logps
    )
