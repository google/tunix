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

"""Detecting a failed worker only matters if something acts on it."""

import numpy as np
from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import rollout_pool
from tunix.experimental.orchestrator import worker_eviction
from tunix.experimental.orchestrator import worker_registry


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

  def __init__(self, name: str):
    self.name = name

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    return _Output(prompts)


def _worker(engine: _Engine) -> hosted_rollout_worker.HostedRolloutWorker:
  return hosted_rollout_worker.HostedRolloutWorker(
      engine, worker_id=engine.name
  )


class EvictionTest(absltest.TestCase):

  def _fleet(self, *workers):
    registry = worker_registry.WorkerRegistry()
    for worker in workers:
      registry.register(worker)
    pool = rollout_pool.PooledRolloutWorker.from_workers(
        list(workers), max_concurrency=1
    )
    return registry, pool

  def test_a_worker_reporting_an_error_leaves_the_fleet(self):
    healthy, broken = _worker(_Engine("healthy")), _worker(_Engine("broken"))
    registry, pool = self._fleet(healthy, broken)
    evictor = worker_eviction.WorkerEvictor(registry, pool=pool)

    evictions = evictor.evict_unhealthy({
        "healthy": datatypes.HealthReport(state=datatypes.WorkerState.READY),
        "broken": datatypes.HealthReport(
            state=datatypes.WorkerState.ERROR, last_error="unreachable"
        ),
    })

    self.assertEqual([e.worker_id for e in evictions], ["broken"])
    self.assertTrue(evictions[0].removed_from_dispatch)
    self.assertEqual(registry.worker_ids(), ["healthy"])
    self.assertLen(pool.dispatcher.actors, 1)

  def test_a_worker_stuck_in_a_state_leaves_the_fleet(self):
    stuck, healthy = _worker(_Engine("stuck")), _worker(_Engine("healthy"))
    registry, pool = self._fleet(stuck, healthy)
    evictor = worker_eviction.WorkerEvictor(registry, pool=pool)

    class _Overdue:
      worker_id = "stuck"
      state = datatypes.WorkerState.INITIALIZING
      elapsed_s = 1000.0
      deadline_s = 60.0

    evictions = evictor.evict_unhealthy({}, overdue=[_Overdue()])

    self.assertEqual([e.worker_id for e in evictions], ["stuck"])
    self.assertEqual(
        evictions[0].reason, worker_eviction.EvictionReason.OVERDUE_IN_STATE
    )

  def test_a_worker_is_not_evicted_twice_for_two_reasons(self):
    registry, pool = self._fleet(_worker(_Engine("bad")), _worker(_Engine("ok")))
    evictor = worker_eviction.WorkerEvictor(registry, pool=pool)

    class _Overdue:
      worker_id = "bad"
      state = datatypes.WorkerState.SYNCING
      elapsed_s = 1000.0
      deadline_s = 60.0

    evictions = evictor.evict_unhealthy(
        {
            "bad": datatypes.HealthReport(
                state=datatypes.WorkerState.ERROR, last_error="boom"
            )
        },
        overdue=[_Overdue()],
    )

    self.assertLen(evictions, 1)

  def test_the_last_worker_is_kept_even_when_unhealthy(self):
    """An empty rotation turns every later request into a wait for nothing."""
    only = _worker(_Engine("only"))
    registry, pool = self._fleet(only)
    evictor = worker_eviction.WorkerEvictor(registry, pool=pool)

    evictor.evict_unhealthy({
        "only": datatypes.HealthReport(
            state=datatypes.WorkerState.ERROR, last_error="boom"
        )
    })

    self.assertLen(pool.dispatcher.actors, 1)

  def test_evicting_an_unknown_worker_is_harmless(self):
    registry, pool = self._fleet(_worker(_Engine("r0")), _worker(_Engine("r1")))
    evictor = worker_eviction.WorkerEvictor(registry, pool=pool)

    self.assertIsNone(evictor.evict("stranger"))


if __name__ == "__main__":
  absltest.main()
