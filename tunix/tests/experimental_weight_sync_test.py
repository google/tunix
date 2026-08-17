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

"""Unit tests for experimental Raiden weight sync consensus and adapters."""

import asyncio
from typing import Any, Mapping, Sequence
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import weight_sync
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.rollout import manager as rollout_manager
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import rollout_worker
from tunix.experimental.worker import trainer_worker


class MockWeightSyncSource(weight_sync.WeightSyncSource):
  """Mock implementation of WeightSyncSource for testing."""

  def __init__(self, worker_id: str = "mock-trainer-0"):
    self._worker_id = worker_id
    self.prepared = False
    self.released = False

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )

  async def prepare_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Sequence[weight_sync.WorkUnitMetadata]:
    del sync_request, kwargs
    self.prepared = True
    unit_id = weight_sync.WorkUnitId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    var = weight_sync.TensorMetadata(
        name="layer.weight",
        shape=(16, 16),
        mesh_shape=(1, 1),
        layout=(1, 0),
        item_size=4,
        layer_idx=0,
        sharding_spec=("", ""),
    )
    return [
        weight_sync.WorkUnitMetadata(
            unit=unit_id,
            shards=("127.0.0.1:29500",),
            control_plane_rpc_address="127.0.0.1:29500",
            mesh_shape=(1,),
            mesh_axes=("fsdp",),
            variables=(var,),
        )
    ]

  async def release_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> None:
    del sync_request, kwargs
    self.released = True


class MockWeightSyncDestination(weight_sync.WeightSyncDestination):
  """Mock implementation of WeightSyncDestination for testing."""

  def __init__(self, worker_id: str = "mock-rollout-0"):
    self._worker_id = worker_id
    self.bound = False
    self.prepared = False
    self.synced = False
    self.committed = False
    self.aborted = False
    self.tracker = weight_sync_coordinator.WorkerRoundTracker()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"rollout"})
    )

  async def bind_weight_sync(self, **kwargs: Any) -> None:
    del kwargs
    self.bound = True

  async def get_weight_sync_metadata(
      self, **kwargs: Any
  ) -> Sequence[weight_sync.WorkUnitMetadata]:
    del kwargs
    unit_id = weight_sync.WorkUnitId(
        job_name="rollout", job_replica_id="0", data_name="weights"
    )
    var = weight_sync.TensorMetadata(
        name="layer.weight",
        shape=(16, 16),
        mesh_shape=(1, 1),
        layout=(1, 0),
        item_size=4,
        layer_idx=0,
        sharding_spec=("", ""),
    )
    return [
        weight_sync.WorkUnitMetadata(
            unit=unit_id,
            shards=("127.0.0.1:29600",),
            control_plane_rpc_address="127.0.0.1:29600",
            mesh_shape=(1,),
            mesh_axes=("tp",),
            variables=(var,),
        )
    ]

  async def pre_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    del kwargs
    if self.tracker.admit(sync_request, "prepared"):
      self.prepared = True
      self.tracker.complete(sync_request, "prepared")
    return True

  async def weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    del kwargs
    if self.tracker.admit(sync_request, "h2d_done"):
      self.synced = True
      self.tracker.complete(sync_request, "h2d_done")
    return True

  async def post_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    del kwargs
    if self.tracker.admit(sync_request, "committed"):
      self.committed = True
      self.tracker.complete(sync_request, "committed")
    return True

  async def abort_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    del kwargs
    if self.tracker.admit(sync_request, "aborted"):
      self.aborted = True
      self.tracker.complete(sync_request, "aborted")
    return True

  async def get_weight_sync_status(self, **kwargs: Any) -> Mapping[str, Any]:
    del kwargs
    return self.tracker.report()


class MockHandler(weight_sync.WeightSyncHandler):
  """Mock transport handler for consensus coordinator tests."""

  def __init__(self, should_succeed: bool = True):
    self.should_succeed = should_succeed
    self.registered_units: list[weight_sync.WorkUnitMetadata] = []

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    self.registered_units.append(metadata)

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: str | None = None,
      generation: int | None = None,
  ) -> weight_sync.TransferResult:
    del src_units, dst_units, generation
    return weight_sync.TransferResult(
        req_id=req_id or "test_req",
        success=self.should_succeed,
        message="OK" if self.should_succeed else "Simulated DMA failure",
    )


class ExperimentalWeightSyncTest(unittest.IsolatedAsyncioTestCase):
  """Unit and integration tests for experimental weight synchronization."""

  async def test_worker_round_tracker_lifecycle(self):
    tracker = weight_sync_coordinator.WorkerRoundTracker()
    req1 = datatypes.WeightSyncRequest(
        policy_version=1,
        extra_config={"req_id": "req-1", "uuid": 1},
    )

    self.assertTrue(tracker.admit(req1, "prepared"))
    tracker.complete(req1, "prepared")

    self.assertTrue(tracker.admit(req1, "h2d_done"))
    tracker.complete(req1, "h2d_done")

    self.assertTrue(tracker.admit(req1, "committed"))
    tracker.complete(req1, "committed")

    # Stale version rejection
    req_stale = datatypes.WeightSyncRequest(
        policy_version=1,
        extra_config={"req_id": "req-1", "uuid": 0},
    )
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      tracker.admit(req_stale, "prepared")

    # Next generation admission
    req2 = datatypes.WeightSyncRequest(
        policy_version=2,
        extra_config={"req_id": "req-2", "uuid": 2},
    )
    self.assertTrue(tracker.admit(req2, "prepared"))

  async def test_coordinator_consensus_success(self):
    registry = worker_registry.WorkerRegistry()
    src = MockWeightSyncSource()
    dst = MockWeightSyncDestination()
    registry.register(src)
    registry.register(dst)

    handler = MockHandler(should_succeed=True)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=handler,
        source_role="trainer",
        destination_role="rollout",
    )

    result = await coordinator.sync(policy_version=1)
    self.assertEqual(result.state, weight_sync_coordinator.RoundState.COMMITTED)
    self.assertEqual(result.policy_version, 1)
    self.assertTrue(src.prepared)
    self.assertTrue(src.released)
    self.assertTrue(dst.bound)
    self.assertTrue(dst.prepared)
    self.assertTrue(dst.synced)
    self.assertTrue(dst.committed)
    self.assertFalse(dst.aborted)

  async def test_coordinator_abort_on_transfer_failure(self):
    registry = worker_registry.WorkerRegistry()
    src = MockWeightSyncSource()
    dst = MockWeightSyncDestination()
    registry.register(src)
    registry.register(dst)

    handler = MockHandler(should_succeed=False)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=handler,
        source_role="trainer",
        destination_role="rollout",
    )

    with self.assertRaises(weight_sync_coordinator.WeightSyncError):
      await coordinator.sync(policy_version=1)

    self.assertTrue(src.released)
    self.assertTrue(dst.aborted)
    self.assertFalse(dst.committed)

  async def test_legacy_vllm_sampler_adapter_weight_sync(self):
    mock_sampler = mock.MagicMock()
    mock_sampler.transformer_state = None
    mock_sampler.mesh = None
    adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
        server_id="test-sampler",
        vllm_sampler=mock_sampler,
    )

    # Bind and get metadata
    await adapter.bind_weight_sync()
    meta = await adapter.get_weight_sync_metadata()
    self.assertEqual(len(meta), 1)
    self.assertEqual(meta[0].unit.job_name, "test-sampler")

    # Full round
    sync_req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "test-1", "uuid": 1}
    )
    self.assertTrue(await adapter.pre_weight_sync(sync_req))
    self.assertTrue(await adapter.weight_sync(sync_req))
    self.assertTrue(await adapter.post_weight_sync(sync_req))

    status = await adapter.get_weight_sync_status()
    self.assertEqual(status.get("policy_version"), 1)

  async def test_rollout_manager_weight_sync(self):
    mock_sampler = mock.MagicMock()
    mock_sampler.transformer_state = None
    mock_sampler.mesh = None
    adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
        server_id="mgr-sampler",
        vllm_sampler=mock_sampler,
    )
    manager = rollout_manager.RolloutManager(
        sampler=adapter,
        tokenizer=mock.MagicMock(),
        chat_parser=mock.MagicMock(),
    )

    await manager.bind_weight_sync()
    meta = await manager.get_weight_sync_metadata()
    self.assertEqual(len(meta), 1)

    sync_req = datatypes.WeightSyncRequest(
        policy_version=5, extra_config={"req_id": "test-5", "uuid": 5}
    )
    await manager.pre_weight_sync(sync_req)
    v = await manager.weight_sync(sync_req)
    self.assertEqual(v, 5)
    await manager.post_weight_sync(sync_req)

    status = await manager.get_weight_sync_status()
    self.assertEqual(status.get("policy_version"), 5)

  async def test_trainer_worker_weight_sync(self):
    mock_trainer = mock.MagicMock(spec=abstract_trainer.AbstractTrainer)
    mock_trainer.policy_version = 0
    mock_trainer.prepare_weight_sync.return_value = [
        weight_sync.WorkUnitMetadata(
            unit=weight_sync.WorkUnitId("train", "0", "weights"),
            shards=("127.0.0.1:29500",),
            control_plane_rpc_address="127.0.0.1:29500",
            mesh_shape=(1,),
            mesh_axes=("fsdp",),
            variables=(),
        )
    ]
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: mock_trainer
    )

    res = worker.prepare_weight_sync(
        datatypes.WeightSyncRequest(policy_version=1)
    )
    self.assertEqual(len(res), 1)
    self.assertEqual(worker.state, datatypes.WorkerState.SYNCING)

    worker.release_weight_sync()
    self.assertEqual(worker.state, datatypes.WorkerState.READY)
    mock_trainer.release_weight_sync.assert_called_once()

  async def test_distributed_rl_engine_sync_weights_coordination(self):
    src = MockWeightSyncSource()
    dst = MockWeightSyncDestination()
    handler = MockHandler(should_succeed=True)

    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[dst],
        trainer_workers={datatypes.Role.ACTOR: src},
        inference_workers={},
        weight_sync_handler=handler,
    )

    new_version = await engine.sync_weights(role=datatypes.Role.ACTOR)
    self.assertEqual(new_version, 1)
    self.assertEqual(engine.policy_version, 1)
    self.assertTrue(src.prepared)
    self.assertTrue(dst.committed)


if __name__ == "__main__":
  unittest.main()
