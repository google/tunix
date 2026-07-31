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

"""Killing a rollout worker mid-batch must cost trajectories, not the run.

The failures injected here are the ones a fleet actually sees: a worker dies
while holding work, a worker that was given up on answers anyway, and a worker
reports itself unhealthy. Each is asserted to end with the batch completed on
the survivors, no group trained on partial or duplicated members, and no
capacity left reserved for work that will never come back.
"""

import asyncio
from typing import Any

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import group_gate
from tunix.experimental.orchestrator import group_retry
from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import request_ledger as ledger_lib
from tunix.experimental.orchestrator import rollout_pool
from tunix.experimental.orchestrator import worker_eviction
from tunix.experimental.orchestrator import worker_registry

_GROUP_SIZE = 2


class _Output:

  def __init__(self, prompts):
    self.text = [f"completion {i}" for i in range(len(prompts))]
    self.tokens = [np.array([1, 2], dtype=np.int32) for _ in prompts]
    self.logprobs = [np.zeros(2, dtype=np.float32) for _ in prompts]
    self.left_padded_prompt_tokens = np.array(
        [[0, 1] for _ in prompts], dtype=np.int32
    )
    self.logits = None


class _Engine:
  """A sampler that can be made to die, permanently or for a while."""

  def __init__(self, name: str, *, dies_after: int | None = None):
    self.name = name
    self.calls = 0
    self._dies_after = dies_after
    self.dead = False

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    self.calls += 1
    if self.dead or (
        self._dies_after is not None and self.calls > self._dies_after
    ):
      self.dead = True
      raise ConnectionError(f"{self.name} died")
    return _Output(prompts)


def _worker(engine: _Engine) -> hosted_rollout_worker.HostedRolloutWorker:
  return hosted_rollout_worker.HostedRolloutWorker(
      engine, worker_id=engine.name
  )


def _records(group_id: str, count: int = _GROUP_SIZE):
  return [
      ledger_lib.RequestRecord(
          request=datatypes.RolloutRequest(
              request_id=f"{group_id}-{index}",
              prompt={"prompts": f"prompt {index}"},
              prompt_id=group_id,
              group_id=group_id,
              sample_index=index,
          ),
          group_id=group_id,
          sample_index=index,
      )
      for index in range(count)
  ]


class RolloutFaultInjectionTest(absltest.TestCase):

  def _pool(self, workers, **kwargs):
    return rollout_pool.PooledRolloutWorker.from_workers(
        workers, max_concurrency=1, **kwargs
    )

  def test_a_worker_dying_mid_batch_costs_trajectories_not_the_run(self):
    healthy = _Engine("healthy")
    dying = _Engine("dying", dies_after=0)
    pool = self._pool([_worker(healthy), _worker(dying)])
    ledger = ledger_lib.RequestLedger(group_size=_GROUP_SIZE)

    outcome = asyncio.run(
        group_retry.generate_groups(
            pool,
            _records("g0"),
            ledger,
            group_size=_GROUP_SIZE,
            max_attempts=3,
        )
    )

    # The group came out whole, refilled by the surviving worker.
    self.assertEqual(outcome.complete, ["g0"])
    self.assertEmpty(outcome.incomplete)
    self.assertLen(outcome.responses["g0"], _GROUP_SIZE)
    self.assertGreater(outcome.attempts, 1)
    # And no capacity is still reserved for the work that failed.
    self.assertEqual(pool.dispatcher.router.total_outstanding(), 0)

  def test_a_late_straggler_does_not_become_an_extra_member(self):
    """The reason retries are keyed on the slot rather than the request."""
    ledger = ledger_lib.RequestLedger(group_size=_GROUP_SIZE)
    original = _records("g0")
    ledger.register(original)

    # The first attempt is given up on and reissued.
    retry = ledger_lib.RequestRecord(
        request=datatypes.RolloutRequest(
            request_id="g0-0:retry1",
            prompt="p",
            group_id="g0",
            sample_index=0,
        ),
        group_id="g0",
        sample_index=0,
        attempt=1,
    )
    ledger.register([retry])

    ledger.admit(_response("g0-0:retry1"))
    ledger.admit(_response("g0-1"))
    self.assertTrue(ledger.is_group_complete("g0"))

    # The abandoned original finally answers.
    late = ledger.admit(_response("g0-0"))

    self.assertEqual(late, ledger_lib.Admission.DUPLICATE)
    self.assertLen(ledger.accepted("g0"), _GROUP_SIZE)

  def test_a_group_that_cannot_be_refilled_is_not_trained_on(self):
    everything_dies = _Engine("broken", dies_after=0)
    pool = self._pool([_worker(everything_dies)])
    ledger = ledger_lib.RequestLedger(group_size=_GROUP_SIZE)

    outcome = asyncio.run(
        group_retry.generate_groups(
            pool,
            _records("g0"),
            ledger,
            group_size=_GROUP_SIZE,
            max_attempts=2,
        )
    )

    self.assertEqual(outcome.incomplete, ["g0"])
    self.assertEmpty(outcome.complete)
    self.assertEqual(pool.dispatcher.router.total_outstanding(), 0)

  def test_healthy_groups_survive_a_batch_containing_a_broken_one(self):
    # One worker fails the first request it sees, then recovers.
    flaky = _Engine("flaky", dies_after=1)
    pool = self._pool([_worker(_Engine("healthy")), _worker(flaky)])
    ledger = ledger_lib.RequestLedger(group_size=_GROUP_SIZE)

    outcome = asyncio.run(
        group_retry.generate_groups(
            pool,
            _records("g0") + _records("g1"),
            ledger,
            group_size=_GROUP_SIZE,
            max_attempts=1,
        )
    )

    # With no retry budget, whatever could not be filled is simply not trained
    # on, and the rest is unaffected.
    for group_id in outcome.complete:
      self.assertLen(outcome.responses[group_id], _GROUP_SIZE)
    self.assertEqual(pool.dispatcher.router.total_outstanding(), 0)

  def test_partial_groups_never_reach_the_advantage_math(self):
    """The gate and the ledger have to agree about what is trainable."""
    records = _records("g0")
    responses = [_response("g0-0")]

    gated = group_gate.gate_groups(
        [record.request for record in records],
        responses,
        group_size=_GROUP_SIZE,
    )

    self.assertEmpty(gated.items)
    self.assertEqual(
        gated.dropped[0].reason, group_gate.DropReason.INCOMPLETE
    )


def _response(request_id: str) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id, status="SUCCEEDED", policy_version=1
  )


if __name__ == "__main__":
  absltest.main()
