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

"""Tests that the fleet wires the control plane to the orchestrator's handles."""

import asyncio
import types

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import inprocess_workers
from tunix.experimental.orchestrator import worker_fleet
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import rollout_worker


class _FakeCluster:
  """Records the primitive calls the handles delegate to."""

  def __init__(self):
    self.global_steps = 0
    self.generate_calls = []
    self.update_actor_calls = []
    self.sync_calls = 0
    self.ref_calls = []
    self.actor_calls = []
    self.cluster_config = types.SimpleNamespace(
        training_config=types.SimpleNamespace(
            compute_logps_micro_batch_size=2
        )
    )

  def generate(self, *args):
    self.generate_calls.append(args)
    return "GEN"

  def update_actor(self, *args):
    self.update_actor_calls.append(args)

  def sync_weights(self):
    self.sync_calls += 1

  def get_ref_per_token_logps(self, **kwargs):
    self.ref_calls.append(kwargs)
    return "REF"

  def get_actor_per_token_logps(self, **kwargs):
    self.actor_calls.append(kwargs)
    return "ACTOR"


class WorkerFleetTest(absltest.TestCase):

  def test_in_process_fleet_registers_a_worker_per_role(self):
    fleet = worker_fleet.WorkerFleet.in_process(_FakeCluster())
    self.assertEqual(fleet.registry.roles(), {"trainer", "rollout", "inference"})
    for role in ("trainer", "rollout", "inference"):
      self.assertLen(fleet.registry.group(role).members(), 1)
    # Weight sync is an action, not a managed resource.
    self.assertNotIn("weight_sync", fleet.registry.roles())

  def test_handles_are_workers_the_control_plane_can_manage(self):
    fleet = worker_fleet.WorkerFleet.in_process(_FakeCluster())
    for handle in (fleet.trainer, fleet.rollout, fleet.inference):
      self.assertIsInstance(handle, abstract_worker.Worker)
      self.assertIsInstance(handle.info(), datatypes.WorkerInfo)
      self.assertIsInstance(handle.heartbeat(), datatypes.HealthReport)

  def test_bring_up_then_shutdown_moves_every_worker(self):
    fleet = worker_fleet.WorkerFleet.in_process(_FakeCluster())
    fleet.bring_up()
    reports = fleet.poll_health()
    self.assertLen(reports, 3)
    for report in reports.values():
      self.assertEqual(report.state, datatypes.WorkerState.READY)
    self.assertEmpty(fleet.overdue())

    fleet.shutdown()
    for report in fleet.poll_health().values():
      self.assertEqual(report.state, datatypes.WorkerState.STOPPED)

  def test_built_cluster_routes_every_primitive_to_the_handles(self):
    base = _FakeCluster()
    fleet = worker_fleet.WorkerFleet.in_process(base)
    cluster = fleet.build_cluster(base)

    cluster.generate(["p"])
    cluster.update_actor(["chunk"], None, False)
    cluster.sync_weights()
    cluster.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2)
    cluster.get_actor_per_token_logps("p", "c", pad_id=0, eos_id=2)

    # Each primitive reached the backing cluster *through* its handle.
    self.assertLen(base.generate_calls, 1)
    self.assertEqual(base.update_actor_calls, [(["chunk"], None, False)])
    self.assertEqual(base.sync_calls, 1)
    self.assertLen(base.ref_calls, 1)
    self.assertLen(base.actor_calls, 1)
    # Handles supply the micro-batch size from cluster config.
    self.assertEqual(base.ref_calls[0]["micro_batch_size"], 2)

  def test_fleet_accepts_custom_handles(self):
    # Any object satisfying the handle contracts works; only Workers are
    # registered for lifecycle/health.
    class _PlainTrainer:

      def train(self, chunks, eval_ds, skip_jit):
        del chunks, eval_ds, skip_jit

    fleet = worker_fleet.WorkerFleet(trainer=_PlainTrainer())
    self.assertEmpty(fleet.registry.roles())
    self.assertIsNotNone(fleet.trainer)


class RolloutFleetPoolTest(absltest.TestCase):
  """A fleet holding several rollout workers manages and balances all of them."""

  class _TrajectoryWorker(rollout_worker.RolloutWorker):
    """Minimal worker speaking the per-trajectory rollout contract."""

    def __init__(self, worker_id: str):
      super().__init__(worker_id=worker_id)
      self.seen = []

    async def generate(self, request, on_complete=None):
      del on_complete
      self.seen.append(request.prompt_id)
      return datatypes.RolloutResponse(
          status="SUCCEEDED", metadata={"served_by": self.worker_id}
      )

  def test_registers_every_rollout_worker_in_the_pool(self):
    workers = [self._TrajectoryWorker(f"rollout_{i}") for i in range(3)]
    fleet = worker_fleet.WorkerFleet(rollout=workers)

    self.assertLen(fleet.registry.group("rollout").members(), 3)
    self.assertLen(fleet.rollout_workers, 3)
    # Whole-batch generation still targets a single worker.
    self.assertIs(fleet.rollout, workers[0])

  def test_pool_spreads_requests_over_the_registered_workers(self):
    workers = [self._TrajectoryWorker(f"rollout_{i}") for i in range(3)]
    fleet = worker_fleet.WorkerFleet(rollout=workers, rollout_max_concurrency=2)

    pool = fleet.rollout_pool
    self.assertEqual(pool.max_in_flight, 6)

    requests = [
        datatypes.RolloutRequest(prompt=f"p{i}", prompt_id=f"p{i}")
        for i in range(9)
    ]
    responses = asyncio.run(pool.generate(requests))

    self.assertLen(responses, 9)
    self.assertTrue(all(r.status == "SUCCEEDED" for r in responses))
    # Every worker took a share rather than one absorbing the batch.
    for worker in workers:
      self.assertNotEmpty(worker.seen)

  def test_fleet_without_rollout_workers_has_no_pool(self):
    fleet = worker_fleet.WorkerFleet()
    self.assertIsNone(fleet.rollout_pool)
    self.assertIsNone(fleet.rollout)


class InProcessHandleTest(absltest.TestCase):

  def test_trainer_handle_trains_and_scores(self):
    base = _FakeCluster()
    handle = inprocess_workers.InProcessTrainerWorker(base)
    handle.train(["c"], None, False)
    self.assertEqual(base.update_actor_calls, [(["c"], None, False)])
    self.assertEqual(
        handle.per_token_logps("p", "c", pad_id=0, eos_id=2), "ACTOR"
    )

  def test_rollout_handle_generates(self):
    base = _FakeCluster()
    handle = inprocess_workers.InProcessRolloutWorker(base)
    self.assertEqual(handle.generate(["p"]), "GEN")
    self.assertLen(base.generate_calls, 1)

  def test_inference_handle_scores_the_reference_model(self):
    base = _FakeCluster()
    handle = inprocess_workers.InProcessInferenceWorker(base)
    self.assertEqual(
        handle.per_token_logps("p", "c", pad_id=0, eos_id=2), "REF"
    )
    self.assertEqual(base.ref_calls[0]["prompt_tokens"], "p")

  def test_weight_sync_handle_syncs(self):
    base = _FakeCluster()
    inprocess_workers.InProcessWeightSync(base).sync()
    self.assertEqual(base.sync_calls, 1)


if __name__ == "__main__":
  absltest.main()
