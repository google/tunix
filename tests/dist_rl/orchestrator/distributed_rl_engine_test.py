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

"""Unit tests for DistributedRLEngine and WorkerPoolBalancer."""

import asyncio
import collections
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import fault_tolerance_manager as fault_tolerance_manager_lib
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import rl_engine_interface
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.testing import worker_registry_testing
from tunix.dist_rl.weight_sync import weight_sync_coordinator as weight_sync_coordinator_lib
from tunix.dist_rl.worker import remote_execution
from tunix.rl import algo_core
from tunix.rl import algorithm_config


class MockActorHandle(mock.MagicMock):
  """A smart mock for ActorHandle that routes asubmit/dispatch_task to logical methods.

  This allows tests to write clean assertions like
  `worker.generate.assert_called_once()` while preserving the strict ActorHandle
  type requirements of the engine.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(spec=remote_execution.ActorHandle, *args, **kwargs)
    self._dispatched_request_ids: collections.deque[str] = collections.deque()

    # Ensure all mocked methods return awaitables by default
    async def _poll_impl(
        timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S,
    ):
      val = self.poll_responses.return_value
      if val is None or val == [] or isinstance(val, mock.MagicMock):
        await asyncio.sleep(min(timeout_s, 0.05))
        return None
      self.poll_responses.return_value = None
      req_id = (
          self._dispatched_request_ids.popleft()
          if self._dispatched_request_ids
          else ""
      )
      if isinstance(val, remote_execution.ExecutionResponse):
        if not val.request_id and req_id:
          val.request_id = req_id
        return val
      return remote_execution.ExecutionResponse(request_id=req_id, result=val)

    self.generate = mock.AsyncMock()
    self.poll_responses = mock.AsyncMock(
        return_value=None, side_effect=_poll_impl
    )
    self.weight_sync = mock.AsyncMock()
    self.fwd_bwd = mock.AsyncMock()
    self.update = mock.AsyncMock(return_value=(1, {}))
    self.prepare_weight_sync = mock.AsyncMock()
    self.release_weight_sync = mock.AsyncMock()
    self.pre_weight_sync = mock.AsyncMock()
    self.post_weight_sync = mock.AsyncMock()
    self.abort_weight_sync = mock.AsyncMock()
    self.score = mock.AsyncMock()
    self.per_token_logps = mock.AsyncMock()
    self.save_checkpoint = mock.AsyncMock()
    self.restore_checkpoint = mock.AsyncMock()
    self.get_metrics = mock.AsyncMock(return_value={})
    self.get_target_state = mock.AsyncMock(return_value={"params": 1})
    self.set_target_state = mock.AsyncMock()
    self.fetch_staged = mock.AsyncMock(return_value={})
    self.release_staged = mock.AsyncMock(return_value=0)
    self.clear_staged = mock.AsyncMock(return_value=0)
    self.with_loss_fn = mock.MagicMock()
    self.with_gen_model_input_fn = mock.MagicMock()
    self.compile = mock.MagicMock()

  def submit(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return method(*args, **kwargs)

  async def asubmit(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return await method(*args, **kwargs)

  async def dispatch_task(
      self,
      request_id: str | None = None,
      method_name: str | None = None,
      *args,
      **kwargs,
  ):
    if (
        method_name is None
        and request_id is not None
        and hasattr(self, request_id)
    ):
      method_name = request_id
    if request_id is not None:
      self._dispatched_request_ids.append(request_id)
    method = getattr(self, method_name)
    await method(*args, **kwargs)
    return request_id


class _FakeSyncResult:
  """Minimal result object exposing `policy_version` and `workers`."""

  def __init__(self, policy_version: int, workers=()):
    self.policy_version = policy_version
    self.workers = tuple(workers)


class _FakeWeightSyncCoordinator:
  """Records sync calls and echoes (or forces) the resulting policy version."""

  def __init__(self, forced_version: int | None = None, workers=()):
    self.calls: list[int] = []
    self.kwargs_history: list[dict[str, object]] = []
    self._forced_version = forced_version
    self._workers = tuple(workers)

  async def sync(self, policy_version: int = 0, **kwargs):
    self.calls.append(policy_version)
    self.kwargs_history.append(kwargs)
    version = (
        policy_version if self._forced_version is None else self._forced_version
    )
    return _FakeSyncResult(version, workers=self._workers)


def _make_engine(
    *,
    registry: worker_registry.WorkerRegistry | None = None,
    rollout_workers=(),
    trainer_workers=None,
    inference_workers=None,
    weight_sync_coordinator=None,
    fault_tolerance_config: datatypes.FaultToleranceConfig | None = None,
) -> distributed_rl_engine.DistributedRLEngine:
  registry = (
      registry if registry is not None else worker_registry.WorkerRegistry()
  )
  for idx, handle in enumerate(rollout_workers):
    registry.register_handle(
        f"rollout-{idx}", [datatypes.Role.ROLLOUT], handle
    )
  for role, handle in (trainer_workers or {}).items():
    registry.register_handle(f"trainer-{role}", [role], handle)
  for role, handle in (inference_workers or {}).items():
    registry.register_handle(f"inference-{role}", [role], handle)
  for wid in registry.worker_ids():
    worker_registry_testing.set_status(
        registry, wid, worker_registry.AdmissionStatus.ACTIVE
    )
  ftm = fault_tolerance_manager_lib.FaultToleranceManager(
      registry, config=fault_tolerance_config
  )
  return distributed_rl_engine.DistributedRLEngine(
      registry=registry,
      weight_sync_coordinator=weight_sync_coordinator,
      fault_tolerance_manager=ftm,
  )


class DistributedRLEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_rollout_1 = MockActorHandle()
    self.mock_rollout_2 = MockActorHandle()
    self.mock_actor = MockActorHandle()
    self.mock_ref = MockActorHandle()

    self.engine = _make_engine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
        fault_tolerance_config=datatypes.FaultToleranceConfig(enabled=True),
    )

  def test_generate_load_balances_across_rollout_workers(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="p1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      resp2 = datatypes.RolloutResponse(
          request_id="r2",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="p2",
              group_index=0,
              traj={
                  "trajectory_reward": 2.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )

      self.mock_rollout_1.generate.return_value = [resp1]
      self.mock_rollout_2.generate.return_value = [resp2]

      results = await self.engine.generate([
          {"prompt": "p1", "prompt_id": "p1"},
          {"prompt": "p2", "prompt_id": "p2"},
      ])
      self.assertLen(results, 2)
      rewards = {res.traj["trajectory_reward"] for res in results}
      self.assertEqual(rewards, {1.0, 2.0})

      # Which prompt lands on which worker is decided by hashing its traj_id,
      # so assert on the set of dispatched requests rather than on per-worker
      # identity. Both workers get one request because these two traj_ids
      # happen to hash to different buckets at N=2.
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)
      dispatched = [
          req
          for mock in (self.mock_rollout_1, self.mock_rollout_2)
          for req in mock.generate.call_args.kwargs["requests"]
      ]
      self.assertCountEqual([req.prompt for req in dispatched], ["p1", "p2"])
      self.assertCountEqual(
          [req.prompt_id for req in dispatched], ["p1", "p2"]
      )

    asyncio.run(_run())

  def test_generate_uses_explicit_generation_args(self):
    async def _run():
      resp = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="p1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.generate.return_value = [resp]
      results = await self.engine.generate(
          [{
              "prompt": "p1",
              "prompt_id": "prompt_1",
          }],
          generation_args=datatypes.GenerationArgs(
              max_generation_steps=8,
              temperature=0.5,
              return_logprobs=False,
          ),
      )
      self.assertLen(results, 1)
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      req = self.mock_rollout_1.generate.call_args.kwargs["requests"][0]
      self.assertEqual(
          req.generation_kwargs,
          {
              "max_generation_steps": 8,
              "temperature": 0.5,
              "return_logprobs": False,
          },
      )

    asyncio.run(_run())

  def test_generate_rejects_legacy_generation_kwargs(self):
    async def _run():
      with self.assertRaisesRegex(TypeError, "GenerationArgs"):
        await self.engine.generate(
            [{"prompt": "p1", "prompt_id": "prompt_1"}], temperature=0.5
        )

    asyncio.run(_run())

  def test_generate_routes_rollout_requests(self):
    async def _run():
      request = datatypes.RolloutRequest(
          request_id="r1",
          prompt="p1",
          prompt_id="prompt_1",
          generation_kwargs={"max_generation_steps": 8},
      )
      resp = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.generate.return_value = [resp]

      results = await self.engine.generate([request])

      self.assertLen(results, 1)
      self.mock_rollout_1.generate.assert_called_once()
      self.assertEqual(
          self.mock_rollout_1.generate.call_args.kwargs["requests"], [request]
      )

    asyncio.run(_run())

  def test_poll_rollouts_aggregates_worker_responses(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="p1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.poll_responses.return_value = [resp1]
      self.mock_rollout_2.poll_responses.return_value = []

      await self.engine.dispatch_rollouts(
          [
              {"prompt": "p1", "prompt_id": "p1"},
              {"prompt": "p2", "prompt_id": "p2"},
          ],
          num_generations=2,
      )
      results = await self.engine.poll_rollouts(timeout_s=0.1)
      self.assertEqual(len(results), 1)
      self.assertEqual(results[0].traj["trajectory_reward"], 1.0)

      self.mock_rollout_1.poll_responses.assert_called()

    asyncio.run(_run())

  def test_train_step_routes_to_actor(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = {"loss": 0.5}
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)
      mock_payload.metadata = {"lineage_id": "batch_1"}

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
      )
      self.assertEqual(res, {"loss": 0.5})

      self.mock_actor.fwd_bwd.assert_called_once()
      call_kwargs = self.mock_actor.fwd_bwd.call_args.kwargs
      self.assertIn("request", call_kwargs)
      req = call_kwargs["request"]
      self.assertIsInstance(req, datatypes.TrainRequest)
      self.assertIs(req.payload, mock_payload)
      self.assertEqual(req.metadata, {"lineage_id": "batch_1"})
      self.mock_actor.update.assert_not_called()

    asyncio.run(_run())

  def test_train_step_applies_optimizer_on_last_microbatch(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = datatypes.Response(
          metadata={"queued": True}
      )
      self.mock_actor.update.return_value = (3, {"loss": 0.25})
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)
      mock_payload.metadata = {}

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=True,
      )
      self.assertEqual(
          res,
          {
              "fwd_bwd": datatypes.Response(metadata={"queued": True}),
              "updated": True,
              "train_step": 3,
              "accumulated": True,
              "metrics": {"loss": 0.25},
          },
      )

      self.mock_actor.fwd_bwd.assert_called_once()
      call_kwargs = self.mock_actor.fwd_bwd.call_args.kwargs
      self.assertIn("request", call_kwargs)
      req = call_kwargs["request"]
      self.assertIsInstance(req, datatypes.TrainRequest)
      self.assertIs(req.payload, mock_payload)
      self.mock_actor.update.assert_called_once_with()

    asyncio.run(_run())

  def test_save_checkpoint_delegates_to_trainer_worker(self):
    async def _run():
      self.mock_actor.save_checkpoint.return_value = datatypes.Response(
          metadata={"checkpoint_saved": True}
      )
      metadata = {"step": 5, "policy_version": 2}
      res = await self.engine.save_checkpoint(
          role=datatypes.Role.ACTOR, metadata=metadata
      )
      self.assertEqual(
          res, datatypes.Response(metadata={"checkpoint_saved": True})
      )
      self.mock_actor.save_checkpoint.assert_called_once_with(
          metadata=metadata, step=None, force=False
      )

    asyncio.run(_run())

  def test_save_checkpoint_propagates_step_and_force(self):
    async def _run():
      self.mock_actor.save_checkpoint.return_value = datatypes.Response(
          metadata={"checkpoint_saved": True}
      )
      metadata = {"policy_version": 2}
      res = await self.engine.save_checkpoint(
          role=datatypes.Role.ACTOR,
          metadata=metadata,
          step=10,
          force=True,
      )
      self.assertEqual(
          res, datatypes.Response(metadata={"checkpoint_saved": True})
      )
      self.mock_actor.save_checkpoint.assert_called_once_with(
          metadata=metadata, step=10, force=True
      )

    asyncio.run(_run())

  def test_save_checkpoint_rejects_undeclared_options(self):
    async def _run():
      with self.assertRaises(TypeError):
        await self.engine.save_checkpoint(save_only_lora_params=True)
      self.mock_actor.save_checkpoint.assert_not_called()

    asyncio.run(_run())

  def test_train_step_rejects_undeclared_options(self):
    async def _run():
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)
      with self.assertRaises(TypeError):
        await self.engine.train_step(  # pylint: disable=unexpected-keyword-arg
            mock_payload, skip_jit=True
        )
      self.mock_actor.fwd_bwd.assert_not_called()

    asyncio.run(_run())

  def test_save_checkpoint_propagates_step_kwarg_without_metadata(self):
    async def _run():
      self.mock_actor.save_checkpoint.return_value = datatypes.Response(
          metadata={"checkpoint_saved": True}
      )
      res = await self.engine.save_checkpoint(step=42, force=True)
      self.assertEqual(
          res, datatypes.Response(metadata={"checkpoint_saved": True})
      )
      self.mock_actor.save_checkpoint.assert_called_once_with(
          metadata=None, step=42, force=True
      )

    asyncio.run(_run())

  def test_sync_weights_accepts_explicit_policy_version(self):
    async def _run():
      coordinator = _FakeWeightSyncCoordinator()
      engine = self._engine_with_coordinator(coordinator)
      version = await engine.sync_weights(policy_version=5)
      self.assertEqual(version, 5)
      self.assertEqual(coordinator.calls, [5])

    asyncio.run(_run())

  def test_sync_weights_accepts_role_and_target_roles(self):
    async def _run():
      coordinator = _FakeWeightSyncCoordinator()
      engine = self._engine_with_coordinator(coordinator)

      version = await engine.sync_weights(
          role=datatypes.Role.CRITIC,
          target_roles=[datatypes.Role.ACTOR],
      )
      self.assertEqual(version, 1)
      self.assertEqual(coordinator.calls, [1])

    asyncio.run(_run())

  def _engine_with_coordinator(self, coordinator):
    return _make_engine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
        weight_sync_coordinator=coordinator,
    )

  def test_resume_from_checkpoint_returns_step_without_syncing_rollouts(self):
    async def _run():
      self.mock_actor.restore_checkpoint.return_value = {
          "step": 3,
          "policy_version": 3,
      }
      coordinator = _FakeWeightSyncCoordinator(forced_version=3)
      engine = self._engine_with_coordinator(coordinator)

      result = await engine.resume_from_checkpoint(role=datatypes.Role.ACTOR)

      self.assertEqual(result, 3)
      # Engine aligns its own policy version; rollout sync is handled by
      # `prepare_rollout_policy`.
      self.assertEqual(engine._policy_version, 3)
      self.assertEqual(coordinator.calls, [])
      self.mock_actor.restore_checkpoint.assert_called_once_with(step=None)
      self.mock_rollout_1.get_target_state.assert_not_called()

    asyncio.run(_run())

  def test_resume_uses_global_step_when_optimizer_step_is_different(self):

    async def _run():
      self.mock_actor.restore_checkpoint.return_value = {
          "step": 12,
          "global_step": 3,
          "policy_version": 3,
      }
      coordinator = _FakeWeightSyncCoordinator(forced_version=3)
      engine = self._engine_with_coordinator(coordinator)
      result = await engine.resume_from_checkpoint()
      self.assertEqual(result, 3)
      self.assertEqual(engine._policy_version, 3)
      self.assertEqual(coordinator.calls, [])

    asyncio.run(_run())

  def test_resume_from_checkpoint_uses_step_boundary_policy_version(self):
    async def _run():
      # Recorded mid-step policy_version is ignored in favor of the step.
      self.mock_actor.restore_checkpoint.return_value = {
          "step": 3,
          "policy_version": 2,
      }
      coordinator = _FakeWeightSyncCoordinator(forced_version=3)
      engine = self._engine_with_coordinator(coordinator)

      result = await engine.resume_from_checkpoint()

      self.assertEqual(result, 3)
      self.assertEqual(engine._policy_version, 3)
      self.assertEqual(coordinator.calls, [])

    asyncio.run(_run())

  def test_resume_from_checkpoint_exposes_restored_checkpoint_metadata_and_off_policy_version(
      self,
  ):
    async def _run():
      ckpt_meta = {
          "step": 3,
          "global_step": 3,
          "policy_version": 5,
          "committed_prompt_ids": ["prompt_0", "prompt_3"],
          "skipped_prompt_ids": ["prompt_1"],
      }
      self.mock_actor.restore_checkpoint.return_value = ckpt_meta
      coordinator = _FakeWeightSyncCoordinator(forced_version=5)
      engine = self._engine_with_coordinator(coordinator)

      result = await engine.resume_from_checkpoint()

      self.assertEqual(result, 3)
      self.assertEqual(engine._policy_version, 5)
      self.assertEqual(coordinator.calls, [])
      self.assertEqual(engine.restored_checkpoint_metadata, ckpt_meta)

    asyncio.run(_run())

  def test_resume_from_checkpoint_no_checkpoint_does_not_resync(self):
    async def _run():
      self.mock_actor.restore_checkpoint.return_value = {"step": 0}
      coordinator = _FakeWeightSyncCoordinator()
      engine = self._engine_with_coordinator(coordinator)

      result = await engine.resume_from_checkpoint()

      self.assertEqual(result, 0)
      self.assertEqual(coordinator.calls, [])

    asyncio.run(_run())

  def test_resume_from_checkpoint_tolerates_bad_metadata(self):
    for bad_value in (None, "not-a-dict", {"step": "bogus"}):
      with self.subTest(bad_value=bad_value):

        async def _run(bad_value=bad_value):
          self.mock_actor.restore_checkpoint.return_value = bad_value
          coordinator = _FakeWeightSyncCoordinator()
          engine = self._engine_with_coordinator(coordinator)

          result = await engine.resume_from_checkpoint()

          self.assertEqual(result, 0)
          self.assertEqual(coordinator.calls, [])

        asyncio.run(_run())

  def test_train_step_sends_only_the_request_to_fwd_bwd(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = {"loss": 0.5}
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
      )
      self.assertEqual(res, {"loss": 0.5})
      self.mock_actor.fwd_bwd.assert_called_once()
      call_kwargs = self.mock_actor.fwd_bwd.call_args.kwargs
      self.assertEqual(set(call_kwargs), {"request"})
      self.assertIsInstance(call_kwargs["request"], datatypes.TrainRequest)
      self.assertIs(call_kwargs["request"].payload, mock_payload)

    asyncio.run(_run())

  def test_per_token_logps_propagates_optional_kwargs(self):
    async def _run():
      self.mock_ref.per_token_logps.return_value = [0.1, 0.2]
      res = await self.engine.per_token_logps(
          datatypes.Role.REFERENCE,
          items="test_items",
          chunk_size=8,
          custom_opt=True,
      )
      self.assertEqual(res, [0.1, 0.2])
      self.mock_ref.per_token_logps.assert_called_once_with(
          items="test_items",
          chunk_size=8,
          custom_opt=True,
      )

    asyncio.run(_run())

  def test_score_propagates_optional_kwargs(self):
    async def _run():
      self.mock_ref.score.return_value = [1.0, 2.0]
      res = await self.engine.score(
          datatypes.Role.REFERENCE,
          items=["i1", "i2"],
          normalize=True,
      )
      self.assertEqual(res, [1.0, 2.0])
      self.mock_ref.score.assert_called_once_with(
          items=["i1", "i2"],
          normalize=True,
      )

    asyncio.run(_run())

  def test_get_metrics_propagates_optional_kwargs(self):
    async def _run():
      self.mock_actor.get_metrics.return_value = {"metric_a": 1.0}
      res = await self.engine.get_metrics(datatypes.Role.ACTOR, reset=True)
      self.assertEqual(res, {"metric_a": 1.0})
      self.mock_actor.get_metrics.assert_called_once_with(reset=True)

    asyncio.run(_run())

  def test_save_checkpoint_raises_on_missing_worker(self):
    async def _run():
      with self.assertRaises(ValueError):
        await self.engine.save_checkpoint(role=datatypes.Role.CRITIC)

    asyncio.run(_run())

  def test_sync_weights_delegates_to_coordinator(self):
    async def _run():
      class _FakeResult:
        policy_version = 7
        workers = ()

      class _FakeCoordinator:

        def __init__(self):
          self.calls = []

        async def sync(self, policy_version=0, **kwargs):
          del kwargs
          self.calls.append(policy_version)
          _FakeResult.policy_version = policy_version
          return _FakeResult

      coordinator = _FakeCoordinator()
      engine = _make_engine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
          inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
          weight_sync_coordinator=coordinator,
      )
      self.assertEqual(await engine.sync_weights(), 1)
      self.assertEqual(await engine.sync_weights(), 2)
      self.assertEqual(coordinator.calls, [1, 2])

    asyncio.run(_run())

  def test_prepare_rollout_policy_sets_target_state_and_bootstraps_sync(self):
    async def _run():
      class _FakeResult:
        policy_version = 0
        workers = ()

      class _FakeCoordinator:

        def __init__(self):
          self.calls = []

        async def sync(self, policy_version=0, **kwargs):
          del kwargs
          self.calls.append(policy_version)
          _FakeResult.policy_version = policy_version
          return _FakeResult

      coordinator = _FakeCoordinator()
      engine = _make_engine(
          rollout_workers=[self.mock_rollout_1],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
          inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
          weight_sync_coordinator=coordinator,
      )

      version = await engine.prepare_rollout_policy()

      self.assertEqual(version, 0)
      self.mock_rollout_1.get_target_state.assert_called_once_with()
      self.mock_actor.set_target_state.assert_called_once_with(
          target_state={"params": 1}
      )
      self.assertEqual(coordinator.calls, [0])

    asyncio.run(_run())

  def test_maybe_configure_trainer_target_state_no_trainer_or_rollout_workers(
      self,
  ):
    async def _run():
      engine_no_rollout = _make_engine(
          rollout_workers=[],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
      )
      await engine_no_rollout._maybe_configure_trainer_target_state(
          datatypes.Role.ACTOR
      )
      self.mock_actor.set_target_state.assert_not_called()

      # Role with no registered trainer
      await self.engine._maybe_configure_trainer_target_state(
          datatypes.Role.CRITIC
      )
      self.mock_rollout_1.get_target_state.assert_not_called()

    asyncio.run(_run())

  def test_maybe_configure_trainer_target_state_uses_pending_weight_sync_worker(
      self,
  ):
    async def _run():
      engine = _make_engine(
          rollout_workers=[self.mock_rollout_1],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
      )
      worker_registry_testing.set_status(
          engine._registry,
          "rollout-0",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      self.assertEmpty(engine._rollout_handles())

      await engine._maybe_configure_trainer_target_state(datatypes.Role.ACTOR)

      self.mock_rollout_1.get_target_state.assert_called_once_with()
      self.mock_actor.set_target_state.assert_called_once_with(
          target_state={"params": 1}
      )

    asyncio.run(_run())

  def test_maybe_configure_trainer_target_state_tolerates_attribute_error(self):
    async def _run():
      # Direct AttributeError
      self.mock_rollout_1.get_target_state.side_effect = AttributeError(
          "Worker has no method get_target_state"
      )
      await self.engine._maybe_configure_trainer_target_state(
          datatypes.Role.ACTOR
      )
      self.mock_actor.set_target_state.assert_not_called()

      # Remote RuntimeError wrapping an AttributeError
      self.mock_rollout_1.get_target_state.side_effect = RuntimeError(
          "Actor method call failed with AttributeError: 'Worker' object has no"
          " attribute 'get_target_state'"
      )
      await self.engine._maybe_configure_trainer_target_state(
          datatypes.Role.ACTOR
      )
      self.mock_actor.set_target_state.assert_not_called()

    asyncio.run(_run())

  def test_maybe_configure_trainer_target_state_raises_unrelated_runtime_error(
      self,
  ):
    async def _run():
      self.mock_rollout_1.get_target_state.side_effect = RuntimeError(
          "Worker connection timed out"
      )
      with self.assertRaisesRegex(
          RuntimeError, "Worker connection timed out"
      ):
        await self.engine._maybe_configure_trainer_target_state(
            datatypes.Role.ACTOR
        )

    asyncio.run(_run())

  def test_sync_weights_requires_a_coordinator(self):
    async def _run():
      with self.assertRaises(RuntimeError):
        await self.engine.sync_weights()

    asyncio.run(_run())

  def test_dispatch_spreads_one_group_across_workers(self):
    async def _run():
      # A single prompt: every request shares one prompt_id, which used to pin
      # the whole group to a single worker.
      await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}], num_generations=4
      )

      dispatched_1 = [
          req
          for call in self.mock_rollout_1.generate.call_args_list
          for req in call.kwargs["requests"]
      ]
      dispatched_2 = [
          req
          for call in self.mock_rollout_2.generate.call_args_list
          for req in call.kwargs["requests"]
      ]

      # Per-prompt least-loaded striping spreads the 4 trajectories across both
      # active workers.
      self.assertLen(dispatched_1, 2)
      self.assertLen(dispatched_2, 2)

      # No trajectory is dropped or duplicated.
      self.assertEqual(
          sorted(
              r.group_index for r in (*dispatched_1, *dispatched_2)
          ),
          [0, 1, 2, 3],
      )

    asyncio.run(_run())

  def test_dispatch_routes_same_trajectory_to_same_worker(self):
    async def _run():
      # Within a single dispatch_rollout_requests call, duplicate traj_id
      # entries bucket into the same trajectory unit and land on the same
      # worker.
      def _req(request_id: str) -> datatypes.RolloutRequest:
        return datatypes.RolloutRequest(
            request_id=request_id,
            prompt="p1",
            prompt_id="1",
            group_index=0,
        )

      req_ids = await self.engine.dispatch_rollout_requests(
          [_req("first"), _req("retry")]
      )
      self.assertEqual(req_ids, ["first", "retry"])

      dispatched_1 = [
          req
          for call in self.mock_rollout_1.generate.call_args_list
          for req in call.kwargs["requests"]
      ]
      dispatched_2 = [
          req
          for call in self.mock_rollout_2.generate.call_args_list
          for req in call.kwargs["requests"]
      ]

      # Both requests in the batch resolve to the same least-loaded worker.
      landed = dispatched_1 or dispatched_2
      self.assertEqual(
          [r.request_id for r in landed], ["first", "retry"]
      )
      self.assertEmpty(dispatched_2 if dispatched_1 else dispatched_1)

    asyncio.run(_run())

  def test_dispatch_rollouts_delegates_to_dispatch_rollout_requests(self):
    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="1",
          prompt="p1",
          prompt_id="1",
      )
      req2 = datatypes.RolloutRequest(
          request_id="2",
          prompt="p2",
          prompt_id="2",
      )

      req_ids = await self.engine.dispatch_rollouts([req1, req2])
      self.assertEqual(req_ids, ["1", "2"])

      # Delegation check: both requests must reach active rollout workers.
      dispatched = [
          req
          for mock in (self.mock_rollout_1, self.mock_rollout_2)
          for call in mock.generate.call_args_list
          for req in call.kwargs["requests"]
      ]
      self.assertCountEqual([r.request_id for r in dispatched], ["1", "2"])

    asyncio.run(_run())

  def test_dispatch_rollouts_expands_num_generations(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [
              {"prompt": "p1", "prompt_id": "p1"},
              {"prompt": "p2", "prompt_id": "p2"},
          ],
          num_generations=3,
          policy_version=5,
      )
      self.assertLen(req_ids, 6)

      # 2 calls to generate (1 per worker in pool via prefix hash / round-robin)
      total_dispatched = 0
      for mock_w in (self.mock_rollout_1, self.mock_rollout_2):
        for call in mock_w.generate.call_args_list:
          reqs = call.kwargs["requests"]
          total_dispatched += len(reqs)
          for r in reqs:
            self.assertEqual(r.target_policy_version, 5)
            self.assertIn(r.group_index, (0, 1, 2))
            self.assertIn(r.metadata["group_index"], (0, 1, 2))
      self.assertEqual(total_dispatched, 6)

    asyncio.run(_run())

  def test_dispatch_rollouts_auto_extracts_prompt_and_group_ids(self):
    async def _run():
      dict_item = {
          "prompt": "Solve math",
          "prompt_id": "math_1",
          "generation_kwargs": {"max_generation_steps": 32},
          "metadata": {
              "env_config": {"gold_answer": "42"},
          },
      }
      req_ids = await self.engine.dispatch_rollouts(
          [dict_item], num_generations=2, policy_version=1
      )
      self.assertLen(req_ids, 2)

      all_dispatched = []
      for mock_w in (self.mock_rollout_1, self.mock_rollout_2):
        for c in mock_w.generate.call_args_list:
          all_dispatched.extend(c.kwargs["requests"])

      self.assertLen(all_dispatched, 2)
      group_indices = {r.group_index for r in all_dispatched}
      self.assertEqual(group_indices, {0, 1})
      self.assertTrue(all(r.prompt_id == "math_1" for r in all_dispatched))
      self.assertTrue(all(r.prompt == "Solve math" for r in all_dispatched))
      self.assertTrue(
          all(
              r.generation_kwargs == {"max_generation_steps": 32}
              for r in all_dispatched
          )
      )
      self.assertEqual(
          {r.metadata["group_index"] for r in all_dispatched},
          {0, 1},
      )
      self.assertTrue(
          all(r.metadata["num_generations"] == 2 for r in all_dispatched)
      )
      self.assertEqual(
          {r.metadata["env_config"]["group_index"] for r in all_dispatched},
          {0, 1},
      )
      self.assertTrue(
          all(
              r.metadata["env_config"]["num_generations"] == 2
              for r in all_dispatched
          )
      )
      self.assertTrue(
          all(
              r.metadata["env_config"]["policy_version"] == 1
              for r in all_dispatched
          )
      )

    asyncio.run(_run())

  def test_build_rollout_requests_deep_injects_env_config_for_mappings(self):
    original_env_config = {"env_name": "math_arena", "timeout_s": 30}
    prompts = [{
        "prompt": "Solve 2+2",
        "prompt_id": "math_p1",
        "metadata": {"env_config": original_env_config},
    }]
    requests = self.engine._build_rollout_requests(
        prompts, num_generations=3, policy_version=4
    )
    self.assertLen(requests, 3)

    for idx, req in enumerate(requests):
      self.assertEqual(req.group_index, idx)
      self.assertEqual(req.target_policy_version, 4)
      self.assertEqual(req.metadata["group_index"], idx)
      self.assertEqual(req.metadata["num_generations"], 3)
      # Verify env_config deep injection
      env_cfg = req.metadata["env_config"]
      self.assertEqual(env_cfg["env_name"], "math_arena")
      self.assertEqual(env_cfg["timeout_s"], 30)
      self.assertEqual(env_cfg["group_index"], idx)
      self.assertEqual(env_cfg["num_generations"], 3)
      self.assertEqual(env_cfg["policy_version"], 4)

    # Verify original env_config was not mutated in place
    self.assertEqual(
        original_env_config, {"env_name": "math_arena", "timeout_s": 30}
    )

  def test_build_rollout_requests_handles_non_mapping_env_config(self):
    # 1. env_config is a string
    requests_str = self.engine._build_rollout_requests(
        [{
            "prompt": "p1",
            "prompt_id": "p1",
            "metadata": {"env_config": "env_v1"},
        }],
        num_generations=2,
        policy_version=1,
    )
    self.assertLen(requests_str, 2)
    self.assertEqual(requests_str[0].metadata["env_config"], "env_v1")

    # 2. env_config is None
    requests_none = self.engine._build_rollout_requests(
        [{
            "prompt": "p2",
            "prompt_id": "p2",
            "metadata": {"env_config": None},
        }],
        num_generations=1,
        policy_version=1,
    )
    self.assertLen(requests_none, 1)
    self.assertIsNone(requests_none[0].metadata["env_config"])

    # 3. env_config is omitted
    requests_omitted = self.engine._build_rollout_requests(
        [{"prompt": "p3", "prompt_id": "p3"}],
        num_generations=1,
        policy_version=1,
    )
    self.assertLen(requests_omitted, 1)
    self.assertNotIn("env_config", requests_omitted[0].metadata)

  def test_dispatch_rollouts_passes_generation_args_and_route_metadata(self):
    async def _run():
      gen_args = datatypes.GenerationArgs(
          temperature=0.7, max_generation_steps=128
      )
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1", "max_response_length": 512}],
          num_generations=1,
          generation_args=gen_args,
          route_metadata={"custom_key": "custom_value"},
      )
      self.assertLen(req_ids, 1)

      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched = mock_call.kwargs["requests"][0]
      self.assertEqual(
          dispatched.generation_kwargs,
          {
              "temperature": 0.7,
              "max_generation_steps": 128,
          },
      )
      self.assertEqual(dispatched.max_response_length, 512)
      # route_metadata has no fixed schema; any key is merged through verbatim.
      self.assertEqual(dispatched.metadata["custom_key"], "custom_value")

    asyncio.run(_run())

  def test_dispatch_rollouts_generates_deterministic_request_ids(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "Hello", "prompt_id": "p_123"}],
          num_generations=2,
          policy_version=3,
      )
      self.assertEqual(req_ids, ["req_p_123_g0_v3", "req_p_123_g1_v3"])

    asyncio.run(_run())

  def test_dispatch_rollouts_handles_none_metadata(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}],
          num_generations=1,
          metadata=None,
          route_metadata=None,
      )
      self.assertLen(req_ids, 1)

    asyncio.run(_run())

  def test_dispatch_rollouts_raises_without_prompt_id(self):
    async def _run():
      with self.assertRaisesRegex(ValueError, "lacks 'prompt_id'"):
        await self.engine.dispatch_rollouts(["raw_prompt_without_id"])

    asyncio.run(_run())

  def test_dispatch_rollouts_auto_stamps_lineage_context(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "Hello world", "prompt_id": "prompt_42"}],
          num_generations=2,
          policy_version=5,
      )
      self.assertLen(req_ids, 2)

      requests = []
      for call in self.mock_rollout_1.generate.call_args_list:
        requests.extend(call.kwargs.get("requests", []))
      for call in self.mock_rollout_2.generate.call_args_list:
        requests.extend(call.kwargs.get("requests", []))
      self.assertLen(requests, 2)
      requests.sort(key=lambda r: r.group_index)

      req0 = requests[0]
      self.assertIn("lineage", req0.metadata)
      ctx0 = req0.metadata["lineage"]
      self.assertIsInstance(ctx0, lineage.LineageContext)
      self.assertEqual(ctx0.tracking_id, "traj_prompt_42_g0")
      self.assertEqual(ctx0.parent_tracking_ids, ["prompt_42"])
      self.assertLen(ctx0.events, 1)
      self.assertEqual(ctx0.events[0].component, "engine.dispatch")
      self.assertEqual(ctx0.events[0].operation, "rollout")
      self.assertEqual(ctx0.events[0].attributes["policy_version"], 5)
      self.assertEqual(ctx0.events[0].attributes["group_index"], 0)

      req1 = requests[1]
      ctx1 = req1.metadata["lineage"]
      self.assertEqual(ctx1.tracking_id, "traj_prompt_42_g1")
      self.assertEqual(ctx1.parent_tracking_ids, ["prompt_42"])
      self.assertEqual(ctx1.events[0].attributes["group_index"], 1)

    asyncio.run(_run())

  def test_dispatch_rollouts_stamps_task_name_from_prompt(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [
              {
                  "prompt": "Solve 2+2",
                  "prompt_id": "p_gsm",
                  "task_name": "gsm8k",
              },
              {
                  "prompt": "Fix repo",
                  "prompt_id": "p_swe",
                  "task_name": "deepswe",
              },
              {"prompt": "No task", "prompt_id": "p_unset"},
          ],
          num_generations=2,
          policy_version=2,
      )
      self.assertLen(req_ids, 6)

      requests = []
      for call in self.mock_rollout_1.generate.call_args_list:
        requests.extend(call.kwargs.get("requests", []))
      for call in self.mock_rollout_2.generate.call_args_list:
        requests.extend(call.kwargs.get("requests", []))
      by_prompt: dict[str, list[datatypes.RolloutRequest]] = {}
      for req in requests:
        by_prompt.setdefault(str(req.prompt_id), []).append(req)

      self.assertLen(by_prompt["p_gsm"], 2)
      for req in by_prompt["p_gsm"]:
        gsm_ctx = req.metadata["lineage"]
        self.assertEqual(gsm_ctx.task_name, "gsm8k")
        self.assertNotIn("task_name", gsm_ctx.events[0].attributes)

      self.assertLen(by_prompt["p_swe"], 2)
      for req in by_prompt["p_swe"]:
        swe_ctx = req.metadata["lineage"]
        self.assertEqual(swe_ctx.task_name, "deepswe")
        self.assertNotIn("task_name", swe_ctx.events[0].attributes)

      self.assertLen(by_prompt["p_unset"], 2)
      for req in by_prompt["p_unset"]:
        unset_ctx = req.metadata["lineage"]
        self.assertEqual(unset_ctx.task_name, "")
        self.assertNotIn("task_name", unset_ctx.events[0].attributes)

    asyncio.run(_run())

  def test_dispatch_rollouts_rejects_non_string_task_name(self):
    async def _run():
      with self.assertRaisesRegex(TypeError, "non-string 'task_name'"):
        await self.engine.dispatch_rollouts(
            [{"prompt": "p", "prompt_id": "p1", "task_name": 123}]
        )

    asyncio.run(_run())

  def test_poll_rollouts_forwards_lineage_to_trajectory_item(self):
    async def _run():
      ctx = lineage.LineageContext(
          tracking_id="traj_p1_g0", parent_tracking_ids=["p1"]
      )
      ctx.add_event("engine.dispatch", "rollout")
      ctx.add_event("worker.rollout", "generate", {"worker_id": "rollout_0"})

      resp = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="p1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.5,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
              metadata={"lineage": ctx},
          ),
          metadata={"lineage": ctx},
      )
      self.mock_rollout_1.poll_responses.return_value = [resp]
      self.mock_rollout_2.poll_responses.return_value = []

      await self.engine.dispatch_rollouts(
          [
              {"prompt": "p1", "prompt_id": "p1"},
              {"prompt": "p2", "prompt_id": "p2"},
          ],
          num_generations=2,
      )
      items = await self.engine.poll_rollouts()
      self.assertLen(items, 1)
      item = items[0]
      self.assertIn("lineage", item.metadata)
      self.assertIs(item.metadata["lineage"], ctx)
      self.assertEqual(item.metadata["lineage"].tracking_id, "traj_p1_g0")
      self.assertLen(item.metadata["lineage"].events, 2)

    asyncio.run(_run())

  def test_dispatch_rollout_requests_preserves_existing_lineage(self):
    async def _run():
      custom_ctx = lineage.LineageContext(
          tracking_id="custom_traj_id_99", parent_tracking_ids=["custom_parent"]
      )
      custom_ctx.add_event("custom.system", "custom_op")

      req = datatypes.RolloutRequest(
          request_id="req_custom",
          prompt="test prompt",
          prompt_id="prompt_custom",
          metadata={"lineage": custom_ctx},
      )
      await self.engine.dispatch_rollout_requests([req])

      self.assertIs(req.metadata["lineage"], custom_ctx)
      self.assertEqual(req.metadata["lineage"].tracking_id, "custom_traj_id_99")
      self.assertLen(req.metadata["lineage"].events, 1)
      self.assertEqual(
          req.metadata["lineage"].events[0].component, "custom.system"
      )

    asyncio.run(_run())

  def test_dispatch_rollout_requests_handles_none_metadata(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="req_none_meta",
          prompt="test prompt",
          prompt_id="p_none",
          metadata=None,
      )
      await self.engine.dispatch_rollout_requests([req])

      self.assertIsNotNone(req.metadata)
      self.assertIn("lineage", req.metadata)
      self.assertEqual(req.metadata["lineage"].tracking_id, "traj_p_none_g0")
      self.assertEqual(req.metadata["lineage"].parent_tracking_ids, ["p_none"])

    asyncio.run(_run())

  def test_dispatch_rollout_requests_coerces_int_prompt_id_to_str(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="req_int_id",
          prompt="test prompt",
          prompt_id=42,  # pyrefly: ignore[bad-argument-type]
          metadata={},
      )
      await self.engine.dispatch_rollout_requests([req])

      ctx = req.metadata["lineage"]
      self.assertEqual(ctx.tracking_id, "traj_42_g0")
      self.assertEqual(ctx.parent_tracking_ids, ["42"])
      self.assertIsInstance(ctx.parent_tracking_ids[0], str)

    asyncio.run(_run())

  def test_dispatch_rollout_requests_raises_without_prompt_id(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="req_no_prompt_id",
          prompt="test prompt",
          prompt_id="",
          metadata={},
      )
      with self.assertRaisesRegex(ValueError, "lacks 'prompt_id'"):
        await self.engine.dispatch_rollout_requests([req])

    asyncio.run(_run())

  def test_generate_stamps_lineage_context_for_rollout_requests(self):
    async def _run():
      req = datatypes.RolloutRequest(
          request_id="r1",
          prompt="p1",
          prompt_id="prompt_1",
          group_index=0,
          target_policy_version=3,
          metadata={},
      )
      resp = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.generate.return_value = [resp]
      self.mock_rollout_2.generate.return_value = [resp]

      await self.engine.generate([req])

      self.assertIn("lineage", req.metadata)
      ctx = req.metadata["lineage"]
      self.assertIsInstance(ctx, lineage.LineageContext)
      self.assertEqual(ctx.tracking_id, "traj_prompt_1_g0")
      self.assertEqual(ctx.parent_tracking_ids, ["prompt_1"])
      self.assertLen(ctx.events, 1)
      self.assertEqual(ctx.events[0].component, "engine.dispatch")
      self.assertEqual(ctx.events[0].operation, "rollout")
      self.assertEqual(ctx.events[0].attributes["policy_version"], 3)
      self.assertEqual(ctx.events[0].attributes["group_index"], 0)

    asyncio.run(_run())

  def test_generate_stamps_lineage_context_for_mapping_prompts(self):
    async def _run():
      p = {
          "prompt": "hello",
          "prompt_id": "prompt_dict_1",
          "group_index": 2,
          "policy_version": 4,
          "metadata": {},
      }
      resp = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_dict_1",
              group_index=2,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.generate.return_value = [resp]
      self.mock_rollout_2.generate.return_value = [resp]

      await self.engine.generate([p])

      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched_req = mock_call.kwargs["requests"][0]
      self.assertEqual(dispatched_req.prompt_id, "prompt_dict_1")
      self.assertEqual(dispatched_req.group_index, 0)
      self.assertIn("lineage", dispatched_req.metadata)
      ctx = dispatched_req.metadata["lineage"]
      self.assertEqual(ctx.tracking_id, "traj_prompt_dict_1_g0")
      self.assertEqual(ctx.parent_tracking_ids, ["prompt_dict_1"])
      self.assertEqual(ctx.events[0].component, "engine.dispatch")
      self.assertEqual(ctx.events[0].operation, "rollout")
      self.assertEqual(ctx.events[0].attributes["policy_version"], 0)
      self.assertEqual(ctx.events[0].attributes["group_index"], 0)

    asyncio.run(_run())

  def test_response_to_trajectory_item_with_rollout_response(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_p_extra_g1", parent_tracking_ids=["p_extra"]
    )
    traj_item = datatypes.TrajectoryItem(
        prompt_id="p_extra",
        group_index=1,
        policy_version=2,
        traj={
            "trajectory_reward": 2.0,
            "status": datatypes.TrajectoryStatus.SUCCEEDED,
        },
        prompt_tokens=np.array([10], dtype=np.int32),
        completion_tokens=np.array([1, 2], dtype=np.int32),
        action_mask=np.array([1.0, 1.0], dtype=np.float32),
        metadata={"lineage": ctx},
    )
    resp = datatypes.RolloutResponse(
        request_id="req_1",
        status="COMPLETED",
        payload=traj_item,
        metadata={"lineage": ctx},
    )

    item = distributed_rl_engine._response_to_trajectory_item(resp)
    self.assertIs(item, traj_item)
    self.assertEqual(item.prompt_id, "p_extra")
    self.assertEqual(item.group_index, 1)
    self.assertEqual(item.policy_version, 2)
    self.assertIn("lineage", item.metadata)
    self.assertIs(item.metadata["lineage"], ctx)
    np.testing.assert_array_equal(
        item.completion_tokens, np.array([1, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        item.action_mask, np.array([1, 1], dtype=np.float32)
    )
    self.assertIsNone(getattr(item, "logprobs", None))

  def test_response_to_trajectory_item_collects_aligned_logps(self):
    traj_item = datatypes.TrajectoryItem(
        prompt_id="p_lp",
        group_index=0,
        policy_version=1,
        traj={"trajectory_reward": 1.0},
        prompt_tokens=np.array([10], dtype=np.int32),
        completion_tokens=np.array([1, 2, 3], dtype=np.int32),
        action_mask=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        logprobs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
    )
    resp = datatypes.RolloutResponse(
        request_id="req_lp_ok",
        status="COMPLETED",
        payload=traj_item,
    )
    item = distributed_rl_engine._response_to_trajectory_item(resp)
    np.testing.assert_allclose(
        item.logprobs,
        np.array([-0.1, -0.2, -0.3], dtype=np.float32),
    )

  def test_response_to_trajectory_item_with_error(self):
    err = datatypes.ErrorInfo(error_type="Timeout", message="Timed out")
    resp = datatypes.RolloutResponse(
        request_id="req_err",
        status="ERROR",
        error=err,
        metadata={"prompt_id": "p_err", "group_index": 1},
    )
    item = distributed_rl_engine._response_to_trajectory_item(resp)
    self.assertEqual(item.prompt_id, "p_err")
    self.assertEqual(item.group_index, 1)
    self.assertIsInstance(item.traj, dict)
    self.assertEqual(item.traj["status"], datatypes.TrajectoryStatus.FAILED)
    self.assertEqual(item.status, datatypes.TrajectoryStatus.FAILED)
    self.assertIn("Timed out", item.metadata["error"])

  def test_response_to_trajectory_item_none_payload_raises(self):
    resp = datatypes.RolloutResponse(
        request_id="req_none",
        status="COMPLETED",
        payload=None,
    )
    with self.assertRaises(ValueError):
      distributed_rl_engine._response_to_trajectory_item(resp)

  def test_response_to_trajectory_item_rejects_unsupported_type(self):
    with self.assertRaises(TypeError):
      distributed_rl_engine._response_to_trajectory_item("invalid_type")

  def test_train_step_generates_unique_request_ids_across_microbatches(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = {"loss": 0.5}
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)
      mock_payload.metadata = {}

      req_ids = []
      for _ in range(3):
        await self.engine.train_step(
            mock_payload,
            role=datatypes.Role.ACTOR,
            accumulate_gradients=True,
            apply_optimizer=False,
        )
        req = self.mock_actor.fwd_bwd.call_args.kwargs["request"]
        req_ids.append(req.request_id)

      self.assertLen(req_ids, 3)
      self.assertLen(set(req_ids), 3)
      for req_id in req_ids:
        self.assertTrue(req_id.startswith("train_req_"))

    asyncio.run(_run())

  def test_dispatch_rollouts_handles_integer_prompt_id_zero(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "Test 0", "prompt_id": 0}],
          num_generations=1,
          policy_version=0,
      )
      self.assertEqual(req_ids, ["req_0_g0_v0"])
      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched = mock_call.kwargs["requests"][0]
      self.assertEqual(dispatched.prompt_id, "0")
      self.assertEqual(dispatched.traj_id, "traj_0_g0")
      self.assertEqual(dispatched.metadata["lineage"].tracking_id, "traj_0_g0")
      self.assertEqual(
          dispatched.metadata["lineage"].parent_tracking_ids, ["0"]
      )

    asyncio.run(_run())

  def test_response_to_trajectory_item_handles_error_response(self):
    resp = datatypes.RolloutResponse(
        request_id="req_err",
        status="FAILED",
        error=datatypes.ErrorInfo(
            error_type="WorkerTimeout",
            message="timed out",
        ),
        payload=None,
    )
    item = distributed_rl_engine._response_to_trajectory_item(resp)
    self.assertIsInstance(item.traj, dict)
    self.assertEqual(item.traj["status"], datatypes.TrajectoryStatus.FAILED)
    self.assertEqual(item.status, datatypes.TrajectoryStatus.FAILED)
    self.assertIn("WorkerTimeout", item.metadata.get("error", ""))

  def test_response_to_trajectory_item_raises_when_payload_none_without_error(
      self,
  ):
    resp = datatypes.RolloutResponse(
        request_id="req_none",
        status="COMPLETED",
        payload=None,
    )
    with self.assertRaisesRegex(ValueError, "RolloutResponse payload is None"):
      distributed_rl_engine._response_to_trajectory_item(resp)

  def test_response_to_trajectory_item_rejects_trajectory_item(self):
    item = datatypes.TrajectoryItem(
        prompt_id="direct_item", group_index=0, traj={}
    )
    with self.assertRaises(TypeError):
      distributed_rl_engine._response_to_trajectory_item(item)

  def test_configure_worker_actor_configures_loss_and_gen_model_input_fn(self):
    algo_cfg = algorithm_config.GRPOConfig(
        beta=0.0,
        use_rollout_logps=True,
        routed_experts_encode_mode="auto",
    )
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    mock_assembler.pad_id = 10
    mock_dummy = mock.MagicMock()
    mock_assembler.build_dummy_payload.return_value = mock_dummy
    parent = mock.MagicMock()
    parent.attach_mock(self.mock_actor.with_loss_fn, "with_loss_fn")
    parent.attach_mock(
        self.mock_actor.with_gen_model_input_fn, "with_gen_model_input_fn"
    )
    parent.attach_mock(self.mock_actor.compile, "compile")

    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock_assembler,
        eos_id=20,
    )

    self.mock_actor.with_loss_fn.assert_called_once_with(
        algo_core.grpo_loss_fn, has_aux=True
    )
    self.mock_actor.with_gen_model_input_fn.assert_called_once()
    gen_fn = self.mock_actor.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "algo_config": algo_cfg,
            "pad_id": 10,
            "eos_id": 20,
        },
    )
    mock_assembler.build_dummy_payload.assert_called_once_with(
        with_old_per_token_logps=True,
        with_ref_per_token_logps=False,
        with_sampler_is_weights=False,
        sampler_agreement_applied=False,
        with_routed_experts=False,
        routed_experts_encode_mode="auto",
    )
    self.mock_actor.compile.assert_called_once_with(mock_dummy)
    # Both setters clear the trainer JIT cache; compile must come last.
    self.assertEqual(
        [c[0] for c in parent.mock_calls],
        ["with_loss_fn", "with_gen_model_input_fn", "compile"],
    )

  def test_configure_worker_actor_forwards_non_fused_sampler_agreement_flags(
      self,
  ):
    algo_cfg = algorithm_config.GRPOConfig(
        beta=0.04,
        use_rollout_logps=True,
        sampler_is="token",
        sampler_rs=None,
    )
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    mock_assembler.pad_id = 0

    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock_assembler,
        can_fuse_agreement_in_loss=False,
    )

    mock_assembler.build_dummy_payload.assert_called_once_with(
        with_old_per_token_logps=True,
        with_ref_per_token_logps=True,
        with_sampler_is_weights=True,
        sampler_agreement_applied=True,
        with_routed_experts=False,
        routed_experts_encode_mode="auto",
    )

  def test_configure_worker_actor_builds_routed_experts_dummy_from_assembler(
      self,
  ):
    algo_cfg = algorithm_config.GRPOConfig(
        beta=0.0,
        use_rollout_logps=True,
        sampler_is=None,
        sampler_rs=None,
        routed_experts_encode_mode="uint10",
    )
    algo_cfg.temperature = 1.0
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=4,
        pad_id=0,
        num_generations=2,
        mini_batch_size=1,
    )

    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=assembler,
        return_routed_experts=True,
    )

    self.mock_actor.compile.assert_called_once()
    compiled_payload = self.mock_actor.compile.call_args.args[0]
    self.assertIsNotNone(compiled_payload.routed_experts)
    self.assertEqual(compiled_payload.routed_experts.shape, (2, 8, 0, 0))
    self.assertEqual(compiled_payload.routed_experts.dtype, np.int16)
    self.assertIn("routed_experts_encode_fn", compiled_payload.metadata)
    self.assertEqual(
        compiled_payload.metadata["routed_experts_encode_fn"].keywords["mode"],
        "uint10",
    )

  def test_configure_worker_actor_precompile_false_skips_compile(self):
    algo_cfg = algorithm_config.GRPOConfig(
        beta=0.0,
        use_rollout_logps=True,
    )
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    mock_assembler.pad_id = 0

    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock_assembler,
        precompile=False,
    )

    self.mock_actor.with_loss_fn.assert_called_once()
    self.mock_actor.with_gen_model_input_fn.assert_called_once()
    mock_assembler.build_dummy_payload.assert_not_called()
    self.mock_actor.compile.assert_not_called()

  def test_preformed_rollout_requests_inherit_generation_args(self):
    async def _run():
      preformed = datatypes.RolloutRequest(
          request_id="req_pre_0",
          prompt="p2",
          prompt_id="p2",
          generation_kwargs={"top_p": 0.9},
      )
      await self.engine.dispatch_rollouts(
          [preformed],
          num_generations=1,
          generation_args=datatypes.GenerationArgs(temperature=0.85),
      )
      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched = mock_call.kwargs["requests"][0]
      self.assertEqual(
          dispatched.generation_kwargs,
          {"top_p": 0.9, "temperature": 0.85},
      )

    asyncio.run(_run())

  def test_configure_worker_actor_raises_when_algo_none(self):
    with self.assertRaisesRegex(ValueError, "algo_config is required"):
      self.engine.configure_worker(
          role=datatypes.Role.ACTOR,
          algo_config=None,
          assembler=mock.MagicMock(),
      )

  def test_configure_worker_actor_raises_when_assembler_none(self):
    with self.assertRaisesRegex(ValueError, "assembler is required"):
      self.engine.configure_worker(
          role=datatypes.Role.ACTOR,
          algo_config=mock.MagicMock(),
          assembler=None,
      )

  def test_configure_worker_critic_configures_loss_and_gen_model_input_fn(self):
    mock_critic = MockActorHandle()
    algo_cfg = algorithm_config.PPOConfig()
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    mock_assembler.pad_id = 5

    engine = _make_engine(
        rollout_workers=[self.mock_rollout_1],
        trainer_workers={datatypes.Role.CRITIC: mock_critic},
    )
    with self.assertRaisesRegex(ValueError, "requires a PPOConfig"):
      engine.configure_worker(
          role=datatypes.Role.CRITIC,
          algo_config=algorithm_config.GRPOConfig(),
          assembler=mock_assembler,
      )
    engine.configure_worker(
        role=datatypes.Role.CRITIC,
        algo_config=algo_cfg,
        assembler=mock_assembler,
        eos_id=6,
    )

    mock_critic.with_loss_fn.assert_called_once_with(
        algo_core.ppo_value_loss_fn, has_aux=True
    )
    mock_critic.with_gen_model_input_fn.assert_called_once()
    gen_fn = mock_critic.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "clip_range_value": algo_cfg.clip_range_value,
            "pad_id": 5,
            "eos_id": 6,
        },
    )

  def test_configure_worker_fallback_pad_and_eos_kwargs(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0

    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock.MagicMock(
            spec=["pad_id", "build_dummy_payload"], pad_id=0
        ),
        pad_id=42,
        eos_id=43,
    )

    gen_fn = self.mock_actor.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "algo_config": algo_cfg,
            "pad_id": 42,
            "eos_id": 43,
        },
    )

  def test_configure_worker_raises_on_missing_worker(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0
    with self.assertRaises(ValueError):
      self.engine.configure_worker(
          role=datatypes.Role.CRITIC,
          algo_config=algo_cfg,
          assembler=mock.MagicMock(),
      )

  def test_configure_worker_rollout_and_reference(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    self.engine.configure_worker(
        role=datatypes.Role.ROLLOUT,
        algo_config=algo_cfg,
        assembler=mock_assembler,
    )
    self.engine.configure_worker(
        role=datatypes.Role.REFERENCE,
        algo_config=algo_cfg,
        assembler=mock_assembler,
    )

    engine_no_workers = _make_engine(
        rollout_workers=[],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
    )
    with self.assertRaises(ValueError):
      engine_no_workers.configure_worker(
          role=datatypes.Role.ROLLOUT,
          algo_config=algo_cfg,
          assembler=mock_assembler,
      )
    with self.assertRaises(ValueError):
      engine_no_workers.configure_worker(
          role=datatypes.Role.REFERENCE,
          algo_config=algo_cfg,
          assembler=mock_assembler,
      )

  def test_configure_worker_unsupported_role(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock()
    with self.assertRaises(ValueError):
      self.engine.configure_worker(
          role="unsupported_role",
          algo_config=algo_cfg,
          assembler=mock_assembler,
      )

  def test_distributed_rl_engine_implements_protocol(self):
    self.assertIsInstance(self.engine, rl_engine_interface.AbstractRLEngine)

  def test_configure_worker_default_role_is_actor(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0
    mock_assembler = mock.MagicMock(pad_id=1)

    self.engine.configure_worker(
        algo_config=algo_cfg,
        assembler=mock_assembler,
        eos_id=2,
    )

    self.mock_actor.with_loss_fn.assert_called_once_with(
        algo_core.grpo_loss_fn, has_aux=True
    )
    self.mock_actor.with_gen_model_input_fn.assert_called_once()
    gen_fn = self.mock_actor.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "algo_config": algo_cfg,
            "pad_id": 1,
            "eos_id": 2,
        },
    )

  def test_configure_worker_actor_raises_when_no_actor_worker(self):
    engine = _make_engine(
        rollout_workers=[self.mock_rollout_1],
        trainer_workers={},
    )
    with self.assertRaisesRegex(
        ValueError, "No trainer worker registered for role actor"
    ):
      engine.configure_worker(
          role=datatypes.Role.ACTOR,
          algo_config=mock.MagicMock(),
          assembler=mock.MagicMock(),
      )

  def test_configure_worker_pad_and_eos_defaults(self):
    algo_cfg = algorithm_config.GRPOConfig()
    algo_cfg.temperature = 1.0

    # 1. Zero defaults when assembler defines pad_id=0 and kwargs omit pad/eos
    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock.MagicMock(
            spec=["pad_id", "build_dummy_payload"], pad_id=0
        ),
    )
    gen_fn = self.mock_actor.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "algo_config": algo_cfg,
            "pad_id": 0,
            "eos_id": 0,
        },
    )

    # 2. eos_id defaults to pad_id when only pad_id is set on assembler
    mock_assembler_pad_only = mock.MagicMock(
        spec=["pad_id", "build_dummy_payload"]
    )
    mock_assembler_pad_only.pad_id = 7
    self.engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=algo_cfg,
        assembler=mock_assembler_pad_only,
    )
    gen_fn = self.mock_actor.with_gen_model_input_fn.call_args.args[0]
    self.assertEqual(
        gen_fn("batch"),
        {
            "train_example": "batch",
            "algo_config": algo_cfg,
            "pad_id": 7,
            "eos_id": 7,
        },
    )

  def test_poll_rollouts_does_not_block_on_slow_or_idle_worker(self):
    async def _run():
      resp1 = remote_execution.ExecutionResponse(
          request_id="req_p1_g0_v0",
          result=datatypes.RolloutResponse(
              request_id="req_p1_g0_v0",
              status="COMPLETED",
              payload=datatypes.TrajectoryItem(
                  prompt_id="p1",
                  group_index=0,
                  traj={
                      "trajectory_reward": 1.0,
                      "status": datatypes.TrajectoryStatus.SUCCEEDED,
                  },
              ),
          ),
      )

      fast_returned = False

      async def _fast_poll(timeout_s=50.0):
        nonlocal fast_returned
        if not fast_returned and self.mock_rollout_1._dispatched_request_ids:
          fast_returned = True
          await asyncio.sleep(0.01)
          resp1.request_id = (
              self.mock_rollout_1._dispatched_request_ids.popleft()
          )
          return resp1
        await asyncio.sleep(timeout_s)
        return None

      async def _slow_poll(timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

      self.mock_rollout_1.poll_responses.side_effect = _fast_poll
      self.mock_rollout_2.poll_responses.side_effect = _slow_poll

      await self.engine.dispatch_rollouts(
          [
              {"prompt": "p1", "prompt_id": "p1"},
              {"prompt": "p2", "prompt_id": "p2"},
          ],
          num_generations=2,
          policy_version=0,
      )

      loop = asyncio.get_running_loop()
      start = loop.time()
      results = await self.engine.poll_rollouts(timeout_s=5.0)
      elapsed = loop.time() - start

      self.assertGreaterEqual(len(results), 1)
      self.assertEqual(results[0].prompt_id, "p1")
      self.assertLess(elapsed, 1.0)

    asyncio.run(_run())

  def test_response_to_trajectory_item_error_preserves_metadata_and_policy_version(
      self,
  ):
    resp = datatypes.RolloutResponse(
        request_id="req_p1_g2",
        status="ERROR",
        error=datatypes.ErrorInfo(
            error_type="TrajectoryError",
            message="episode crashed",
        ),
        payload=None,
        metadata={
            "prompt_id": "p1",
            "group_index": 2,
            "batch_idx": 3,
            "prompt_idx": 7,
            "intra_batch_idx": 1,
            "policy_version": 5,
        },
    )
    item = distributed_rl_engine._response_to_trajectory_item(resp)
    self.assertEqual(item.prompt_id, "p1")
    self.assertEqual(item.group_index, 2)
    self.assertFalse(item.is_valid)
    self.assertEqual(item.policy_version, 5)
    self.assertEqual(item.metadata["batch_idx"], 3)
    self.assertEqual(item.metadata["prompt_idx"], 7)
    self.assertEqual(item.metadata["intra_batch_idx"], 1)
    self.assertIn("old_logprobs", item.traj)

  def test_rollout_session_dynamically_reflects_late_registration(self):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      # Construct engine BEFORE any rollout worker is registered in registry
      engine = distributed_rl_engine.DistributedRLEngine(registry=registry)

      late_rollout = MockActorHandle()
      registry.register_handle(
          "rollout-late", [datatypes.Role.ROLLOUT], late_rollout
      )
      worker_registry_testing.set_status(
          registry, "rollout-late", worker_registry.AdmissionStatus.ACTIVE
      )

      req_ids = await engine.dispatch_rollouts(
          [{"prompt": "p_late", "prompt_id": "p_late"}],
          num_generations=1,
          policy_version=0,
      )
      self.assertEqual(req_ids, ["req_p_late_g0_v0"])
      late_rollout.generate.assert_awaited_once()
      await engine.close()

    asyncio.run(_run())

  def test_poll_rollouts_propagates_worker_error_loudly(self):
    async def _run():
      async def _failing_poll(timeout_s=50.0):
        del timeout_s
        raise RuntimeError("Rollout worker process crashed with OOM")

      self.mock_rollout_1.poll_responses.side_effect = _failing_poll
      self.mock_rollout_2.poll_responses.side_effect = _failing_poll

      await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}],
          num_generations=1,
          policy_version=0,
      )

      with self.assertRaisesRegex(
          RuntimeError, "Rollout worker process crashed with OOM"
      ):
        await self.engine.poll_rollouts(timeout_s=1.0)
      self.assertEmpty(self.engine._registry.handles(datatypes.Role.ROLLOUT))

    asyncio.run(_run())

  def test_poll_rollouts_evicts_failed_worker_from_registry_and_retries_on_survivor(
      self,
  ):
    async def _run():
      async def _crashing_poll(timeout_s=50.0):
        del timeout_s
        await asyncio.sleep(0.005)
        raise RuntimeError("rollout-0 crashed mid-poll")

      resp = datatypes.RolloutResponse(
          request_id="req_prompt_1_g0_v0",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.poll_responses.side_effect = _crashing_poll
      self.mock_rollout_2.poll_responses.return_value = [resp]

      # "prompt_1" routes initially to mock_rollout_1; when mock_rollout_1's
      # poll_responses raises RuntimeError, TaskSession evicts mock_rollout_1
      # from WorkerRegistry and retries the request on mock_rollout_2.
      await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "prompt_1"}],
          num_generations=1,
          policy_version=0,
      )
      results = await self.engine.poll_rollouts(timeout_s=1.0)
      await asyncio.sleep(0)

      self.assertLen(results, 1)
      self.assertEqual(results[0].prompt_id, "prompt_1")
      self.assertEqual(
          self.engine._registry.handles(datatypes.Role.ROLLOUT),
          [self.mock_rollout_2],
      )
      self.assertNotIn("rollout-0", self.engine._registry)
      self.mock_rollout_1.sync_close.assert_called_once()
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)
      await self.engine.close()

    asyncio.run(_run())

  def test_poll_rollouts_transient_poll_budget_exhaustion_evicts_and_retries(
      self,
  ):
    async def _run():
      # A deleted pod surfaces as gRPC UNAVAILABLE, which `poll_responses`
      # maps to `TransientTransportError`: eviction only happens once the
      # engine-configured `poll_retry_budget_s` is exhausted.
      engine = _make_engine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=True, poll_retry_budget_s=0.05
          ),
      )
      self.assertEqual(engine._rollout_session._poll_retry_budget_s, 0.05)

      async def _unavailable_poll(timeout_s=50.0):
        del timeout_s
        await asyncio.sleep(0.005)
        raise remote_execution.TransientTransportError(
            "UNAVAILABLE: rollout-0 pod deleted"
        )

      resp = datatypes.RolloutResponse(
          request_id="req_prompt_1_g0_v0",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_1.poll_responses.side_effect = _unavailable_poll
      self.mock_rollout_2.poll_responses.return_value = [resp]

      await engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "prompt_1"}],
          num_generations=1,
          policy_version=0,
      )
      results = await engine.poll_rollouts(timeout_s=2.0)
      await asyncio.sleep(0)

      self.assertLen(results, 1)
      self.assertEqual(results[0].prompt_id, "prompt_1")
      self.assertEqual(
          engine._registry.handles(datatypes.Role.ROLLOUT),
          [self.mock_rollout_2],
      )
      self.assertNotIn("rollout-0", engine._registry)
      self.mock_rollout_1.sync_close.assert_called_once()
      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollouts_evicts_failed_worker_and_retries_on_survivor(
      self,
  ):
    async def _run():
      self.mock_rollout_1.generate.side_effect = ConnectionError(
          "rollout-0 connection refused on dispatch"
      )
      resp1 = datatypes.RolloutResponse(
          request_id="req_prompt_1_g0_v0",
          status="COMPLETED",
          payload=datatypes.TrajectoryItem(
              prompt_id="prompt_1",
              group_index=0,
              traj={
                  "trajectory_reward": 1.0,
                  "status": datatypes.TrajectoryStatus.SUCCEEDED,
              },
          ),
      )
      self.mock_rollout_2.poll_responses.return_value = [resp1]

      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "prompt_1"}],
          num_generations=1,
          policy_version=0,
      )
      await asyncio.sleep(0)

      self.assertEqual(req_ids, ["req_prompt_1_g0_v0"])
      self.assertEqual(
          self.engine._registry.handles(datatypes.Role.ROLLOUT),
          [self.mock_rollout_2],
      )
      self.assertNotIn("rollout-0", self.engine._registry)
      self.mock_rollout_1.sync_close.assert_called_once()
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)

      results = await self.engine.poll_rollouts(timeout_s=0.5)
      self.assertLen(results, 1)
      self.assertEqual(results[0].prompt_id, "prompt_1")
      await self.engine.close()

    asyncio.run(_run())

  def test_fault_tolerance_disabled_does_not_evict_or_retry(self):
    async def _run():
      engine = _make_engine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=False, max_task_retries=3
          ),
      )
      self.mock_rollout_1.generate.side_effect = ConnectionError(
          "rollout-0 connection refused on dispatch"
      )

      with self.assertRaisesRegex(
          ConnectionError, "rollout-0 connection refused on dispatch"
      ):
        await engine.dispatch_rollouts(
            [{"prompt": "p1", "prompt_id": "prompt_1"}],
            num_generations=1,
            policy_version=0,
        )

      # With enabled=False, rollout-0 is not evicted and mock_rollout_2 is not
      # retried.
      self.assertIn("rollout-0", engine._registry)
      self.mock_rollout_1.sync_close.assert_not_called()
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      self.assertEqual(self.mock_rollout_2.generate.call_count, 0)
      self.assertEqual(engine._rollout_session._poll_retry_budget_s, 300.0)
      self.assertEqual(engine._rollout_session._max_task_retries, 0)
      self.assertIsNone(engine._rollout_session._on_worker_unreachable)
      await engine.close()

      self.assertEqual(
          datatypes._FT_OFF_DEFAULT_POLL_RETRY_BUDGET_S,
          distributed_rl_engine.task_session.DEFAULT_POLL_RETRY_BUDGET_S,
      )
      disabled_custom_budget_engine = _make_engine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=False, poll_retry_budget_s=15.0
          ),
      )
      self.assertEqual(
          disabled_custom_budget_engine._rollout_session._poll_retry_budget_s,
          15.0,
      )
      self.assertEqual(
          disabled_custom_budget_engine._rollout_session._max_task_retries, 0
      )
      self.assertIsNone(
          disabled_custom_budget_engine._rollout_session._on_worker_unreachable
      )
      await disabled_custom_budget_engine.close()

      self.assertEqual(self.engine._rollout_session._poll_retry_budget_s, 60.0)
      self.assertEqual(self.engine._rollout_session._max_task_retries, 1)
      self.assertEqual(
          self.engine._rollout_session._on_worker_unreachable,
          self.engine._fault_tolerance_manager.on_worker_unreachable,
      )

    asyncio.run(_run())

  def test_sync_weights_includes_pending_weight_sync_and_promotes_committed_workers(
      self,
  ):
    async def _run():
      mock_rollout_3 = MockActorHandle()
      mock_rollout_rejoined = MockActorHandle()
      old_aba_handle = MockActorHandle()
      new_aba_handle = MockActorHandle()
      coordinator = _FakeWeightSyncCoordinator(
          workers=(
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-0",
                  phase="committed",
                  handle=self.mock_rollout_1,
              ),
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-1",
                  phase="committed",
                  handle=self.mock_rollout_2,
              ),
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-2", phase="aborted", handle=mock_rollout_3
              ),
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-rejoined",
                  phase="committed",
                  handle=mock_rollout_rejoined,
              ),
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-pending-aba",
                  phase="committed",
                  handle=old_aba_handle,
              ),
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-evicted",
                  phase="committed",
                  handle=MockActorHandle(),
              ),
          )
      )
      engine = _make_engine(
          rollout_workers=[
              self.mock_rollout_1,
              self.mock_rollout_2,
              mock_rollout_3,
          ],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
          weight_sync_coordinator=coordinator,
          fault_tolerance_config=datatypes.FaultToleranceConfig(enabled=True),
      )
      # Register rollout-rejoined in REGISTERED status so we can verify that
      # REGISTERED workers are not promoted by sync_weights().
      engine._registry.register_handle(
          "rollout-rejoined",
          [datatypes.Role.ROLLOUT],
          mock_rollout_rejoined,
      )
      # Simulate ABA on "rollout-pending-aba": the round committed on
      # old_aba_handle, but new_aba_handle is now registered under the same
      # worker_id in PENDING_WEIGHT_SYNC without having synced weights.
      engine._registry.register_handle(
          "rollout-pending-aba",
          [datatypes.Role.ROLLOUT],
          new_aba_handle,
      )
      worker_registry_testing.set_status(
          engine._registry,
          "rollout-pending-aba",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      worker_registry_testing.set_status(
          engine._registry,
          "rollout-1",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      worker_registry_testing.set_status(
          engine._registry,
          "rollout-2",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      self.assertEqual(engine._rollout_handles(), [self.mock_rollout_1])

      version = await engine.sync_weights(policy_version=4)
      self.assertEqual(version, 4)
      self.assertEqual(
          coordinator.kwargs_history[-1].get("destination_statuses"),
          (
              worker_registry.AdmissionStatus.ACTIVE,
              worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
          ),
      )
      self.assertEqual(
          engine._registry.status("rollout-1"),
          worker_registry.AdmissionStatus.ACTIVE,
      )
      self.assertEqual(
          engine._registry.status("rollout-2"),
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      # A worker in REGISTERED, ABA-replaced, or evicted must NOT be promoted
      # to ACTIVE.
      self.assertEqual(
          engine._registry.status("rollout-rejoined"),
          worker_registry.AdmissionStatus.REGISTERED,
      )
      self.assertEqual(
          engine._registry.status("rollout-pending-aba"),
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      self.assertNotIn("rollout-evicted", engine._registry)
      self.assertEqual(
          engine._rollout_handles(),
          [self.mock_rollout_1, self.mock_rollout_2],
      )
      await engine.close()

    asyncio.run(_run())

  def test_engine_without_fault_tolerance_manager_uses_default_ft_config(
      self,
  ):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      registry.register_handle(
          "rollout-0", [datatypes.Role.ROLLOUT], self.mock_rollout_1
      )
      worker_registry_testing.set_status(
          registry, "rollout-0", worker_registry.AdmissionStatus.ACTIVE
      )
      registry.register_handle(
          "rollout-1",
          [datatypes.Role.ROLLOUT],
          self.mock_rollout_2,
      )
      worker_registry_testing.set_status(
          registry,
          "rollout-1",
          worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      )
      coordinator = _FakeWeightSyncCoordinator(
          workers=(
              weight_sync_coordinator_lib.WorkerRoundReport(
                  worker_id="rollout-1",
                  phase="committed",
                  handle=self.mock_rollout_2,
              ),
          )
      )
      engine = distributed_rl_engine.DistributedRLEngine(
          registry=registry,
          weight_sync_coordinator=coordinator,
      )
      self.assertIsNotNone(engine._fault_tolerance_manager)
      self.assertTrue(engine._fault_tolerance_manager.config.enabled)
      self.assertEqual(
          engine._fault_tolerance_manager.config.min_active_rollout_ratio, 0.5
      )
      self.assertEqual(engine._rollout_session._poll_retry_budget_s, 60.0)
      self.assertEqual(engine._rollout_session._max_task_retries, 1)
      self.assertIsNotNone(engine._rollout_session._on_worker_unreachable)

      await engine.sync_weights(policy_version=2)
      self.assertEqual(
          coordinator.kwargs_history[-1].get("destination_statuses"),
          (
              worker_registry.AdmissionStatus.ACTIVE,
              worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
          ),
      )
      self.assertEqual(
          registry.status("rollout-1"),
          worker_registry.AdmissionStatus.ACTIVE,
      )
      await engine.close()

    asyncio.run(_run())

  def test_train_step_overlaps_final_fwd_bwd_and_gates_update_on_prior_fwd_bwd(
      self,
  ):
    """Verifies final MB overlaps fwd_bwd with prior MB and awaits prior before update."""
    events: list[str] = []
    mb0_fwd_bwd_entered = asyncio.Event()
    mb1_fwd_bwd_finished = asyncio.Event()
    release_mb0_fwd_bwd = asyncio.Event()

    async def _fwd_bwd(request: datatypes.TrainRequest):
      mb_idx = request.metadata["mb_idx"]
      events.append(f"fwd_bwd_start_{mb_idx}")
      if mb_idx == 0:
        mb0_fwd_bwd_entered.set()
        await asyncio.wait_for(release_mb0_fwd_bwd.wait(), timeout=5.0)
      events.append(f"fwd_bwd_end_{mb_idx}")
      if mb_idx == 1:
        mb1_fwd_bwd_finished.set()
      return {"mb_idx": mb_idx}

    async def _update():
      events.append("update")
      return 1, {"loss": 0.1}

    self.mock_actor.fwd_bwd.side_effect = _fwd_bwd
    self.mock_actor.update.side_effect = _update

    async def _run():
      p0 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros((1, 2), dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.zeros((1, 2), dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          metadata={"mb_idx": 0},
      )
      p1 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros((1, 2), dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.zeros((1, 2), dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          metadata={"mb_idx": 1},
      )
      t0 = asyncio.create_task(
          self.engine.train_step(
              p0,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=False,
          )
      )
      await asyncio.wait_for(mb0_fwd_bwd_entered.wait(), timeout=5.0)
      t1 = asyncio.create_task(
          self.engine.train_step(
              p1,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=True,
          )
      )
      # mb_1's fwd_bwd runs and finishes while mb_0's fwd_bwd is still running.
      await asyncio.wait_for(mb1_fwd_bwd_finished.wait(), timeout=5.0)
      await asyncio.sleep(0.01)
      # update must NOT be called while mb_0's fwd_bwd is still in flight!
      self.mock_actor.update.assert_not_called()

      release_mb0_fwd_bwd.set()
      res0, res1 = await asyncio.gather(t0, t1)
      self.assertEqual(res0, {"mb_idx": 0})
      self.assertEqual(
          res1,
          {
              "fwd_bwd": {"mb_idx": 1},
              "updated": True,
              "train_step": 1,
              "accumulated": True,
              "metrics": {"loss": 0.1},
          },
      )
      self.assertEmpty(self.engine._inflight_fwd_bwd[datatypes.Role.ACTOR])

    asyncio.run(_run())
    self.assertEqual(
        events,
        [
            "fwd_bwd_start_0",
            "fwd_bwd_start_1",
            "fwd_bwd_end_1",
            "fwd_bwd_end_0",
            "update",
        ],
    )

  def test_train_step_skips_update_when_prior_inflight_fwd_bwd_fails(self):
    """Verifies failing prior fwd_bwd aborts final train_step before update."""
    mb0_entered = asyncio.Event()
    mb1_finished = asyncio.Event()

    async def _fwd_bwd(request: datatypes.TrainRequest):
      mb_idx = request.metadata["mb_idx"]
      if mb_idx == 0:
        mb0_entered.set()
        await asyncio.wait_for(mb1_finished.wait(), timeout=5.0)
        raise RuntimeError("corrupted bulk stripe on mb_0")
      loop = asyncio.get_running_loop()
      loop.call_soon(mb1_finished.set)
      return {"mb_idx": mb_idx}

    self.mock_actor.fwd_bwd.side_effect = _fwd_bwd

    async def _run():
      p0 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros((1, 2), dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.zeros((1, 2), dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          metadata={"mb_idx": 0},
      )
      p1 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros((1, 2), dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.zeros((1, 2), dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          metadata={"mb_idx": 1},
      )
      t0 = asyncio.create_task(
          self.engine.train_step(
              p0,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=False,
          )
      )
      await asyncio.wait_for(mb0_entered.wait(), timeout=5.0)
      t1 = asyncio.create_task(
          self.engine.train_step(
              p1,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=True,
          )
      )
      with self.assertRaisesRegex(
          RuntimeError, "corrupted bulk stripe on mb_0"
      ):
        await t0
      with self.assertRaisesRegex(
          RuntimeError, "corrupted bulk stripe on mb_0"
      ):
        await t1
      self.mock_actor.update.assert_not_called()
      self.assertEmpty(self.engine._inflight_fwd_bwd[datatypes.Role.ACTOR])

    asyncio.run(_run())

  def test_train_step_skips_update_when_prior_fails_early_or_cancels(self):
    """Verifies early failure or cancellation of prior fwd_bwd blocks update."""
    mb0_entered = asyncio.Event()

    async def _fwd_bwd(request: datatypes.TrainRequest):
      mb_idx = request.metadata["mb_idx"]
      if mb_idx == 0:
        raise RuntimeError("early mb_0 failure")
      if mb_idx == 2:
        mb0_entered.set()
        await asyncio.sleep(10.0)
      return {"mb_idx": mb_idx}

    self.mock_actor.fwd_bwd.side_effect = _fwd_bwd

    async def _run():
      def _payload(idx: int) -> datatypes.RLTrainerPayload:
        return datatypes.RLTrainerPayload(
            prompt_ids=np.zeros((1, 2), dtype=np.int32),
            prompt_mask=np.ones((1, 2), dtype=np.float32),
            completion_ids=np.zeros((1, 2), dtype=np.int32),
            completion_mask=np.ones((1, 2), dtype=np.float32),
            advantages=np.ones((1, 2), dtype=np.float32),
            metadata={"mb_idx": idx},
        )

      # Case 1: t0 fails completely BEFORE t1 (apply_optimizer=True) starts.
      with self.assertRaisesRegex(RuntimeError, "early mb_0 failure"):
        await self.engine.train_step(
            _payload(0),
            role=datatypes.Role.ACTOR,
            accumulate_gradients=True,
            apply_optimizer=False,
        )
      with self.assertRaisesRegex(RuntimeError, "early mb_0 failure"):
        await self.engine.train_step(
            _payload(1),
            role=datatypes.Role.ACTOR,
            accumulate_gradients=True,
            apply_optimizer=True,
        )
      self.mock_actor.update.assert_not_called()
      self.assertEmpty(self.engine._inflight_fwd_bwd[datatypes.Role.ACTOR])

      # Case 2: prior fwd_bwd is cancelled while final train_step is in flight;
      # final train_step raises RuntimeError (not raw CancelledError).
      t_cancel = asyncio.create_task(
          self.engine.train_step(
              _payload(2),
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=False,
          )
      )
      await asyncio.wait_for(mb0_entered.wait(), timeout=5.0)
      t_final = asyncio.create_task(
          self.engine.train_step(
              _payload(3),
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=True,
          )
      )
      t_cancel.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await t_cancel
      with self.assertRaisesRegex(
          RuntimeError, "Prior in-flight fwd_bwd was cancelled"
      ):
        await t_final
      self.mock_actor.update.assert_not_called()
      self.assertEmpty(self.engine._inflight_fwd_bwd[datatypes.Role.ACTOR])

    asyncio.run(_run())

  def test_fault_tolerance_metrics_reports_fleet_gauges_and_counters(self):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      for idx in range(3):
        bad_handle = MockActorHandle()
        bad_handle.generate.side_effect = ConnectionError("dispatch failed")
        registry.register_handle(
            f"bad-{idx}", [datatypes.Role.ROLLOUT], bad_handle
        )
        worker_registry_testing.set_status(
            registry, f"bad-{idx}", worker_registry.AdmissionStatus.ACTIVE
        )

      registry.register_handle(
          "rollout-0", [datatypes.Role.ROLLOUT], self.mock_rollout_1
      )
      worker_registry_testing.set_status(
          registry, "rollout-0", worker_registry.AdmissionStatus.ACTIVE
      )

      promoted_reports = []
      for idx in range(4):
        h_promote = MockActorHandle()
        wid = f"rollout-promote-{idx}"
        registry.register_handle(wid, [datatypes.Role.ROLLOUT], h_promote)
        worker_registry_testing.set_status(
            registry, wid, worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC
        )
        promoted_reports.append(
            weight_sync_coordinator_lib.WorkerRoundReport(
                worker_id=wid, phase="committed", handle=h_promote
            )
        )

      coordinator = _FakeWeightSyncCoordinator(workers=promoted_reports)
      mock_lifecycle = mock.MagicMock(
          spec=lifecycle.LifecycleDriver,
          bring_up_failures_total=6,
      )
      ftm = fault_tolerance_manager_lib.FaultToleranceManager(
          registry,
          config=datatypes.FaultToleranceConfig(
              enabled=True, max_task_retries=1
          ),
          lifecycle_driver=mock_lifecycle,
      )
      engine = distributed_rl_engine.DistributedRLEngine(
          registry=registry,
          weight_sync_coordinator=coordinator,
          fault_tolerance_manager=ftm,
      )

      # Drive TaskSession counters with `route_key=0` (selects lexicographically
      # first ACTIVE worker):
      # 1) `req_fail`: `bad-0` fails (eviction #1) -> retries on `bad-1`
      #    (retry #1), which also fails (eviction #2) -> terminal failure #1.
      with self.assertRaises(ConnectionError):
        await engine._rollout_session.submit(
            "req_fail", "generate", requests=[], route_key=0
        )
      # 2) `req_ok`: `bad-2` fails (eviction #3) -> retries on `rollout-0`
      #    (retry #2), which accepts dispatch.
      await engine._rollout_session.submit(
          "req_ok", "generate", requests=[], route_key=0
      )

      await engine.sync_weights(policy_version=1)

      for idx in range(7):
        wid = f"rollout-pending-{idx}"
        registry.register_handle(
            wid, [datatypes.Role.ROLLOUT], MockActorHandle()
        )
        worker_registry_testing.set_status(
            registry, wid, worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC
        )
      for idx in range(8):
        registry.register_handle(
            f"rollout-registered-{idx}",
            [datatypes.Role.ROLLOUT],
            MockActorHandle(),
        )

      self.assertEqual(
          engine.fault_tolerance_metrics(),
          {
              "ft/rollout/task_unrecovered_total": 1.0,
              "ft/rollout/task_retries_total": 2.0,
              "ft/evictions_total/task_session": 3.0,
              "ft/rollout/evicted_release_skips_total": 0.0,
              "ft/rejoin/promoted_total": 4.0,
              "ft/fleet/rollout_active": 5.0,
              "ft/fleet/rollout_active_ratio": 1.0,
              "ft/rejoin/bring_up_failures_total": 6.0,
              "ft/fleet/rollout_pending_weight_sync": 7.0,
              "ft/fleet/rollout_registered": 8.0,
          },
      )
      await engine.close()

      minimal_engine = distributed_rl_engine.DistributedRLEngine(
          registry=worker_registry.WorkerRegistry()
      )
      self.assertEqual(
          minimal_engine.fault_tolerance_metrics(),
          {
              "ft/fleet/rollout_registered": 0.0,
              "ft/fleet/rollout_pending_weight_sync": 0.0,
              "ft/fleet/rollout_active": 0.0,
              "ft/fleet/rollout_active_ratio": 1.0,
              "ft/evictions_total/task_session": 0.0,
              "ft/rollout/evicted_release_skips_total": 0.0,
              "ft/rollout/task_retries_total": 0.0,
              "ft/rollout/task_unrecovered_total": 0.0,
              "ft/rejoin/promoted_total": 0.0,
              "ft/rejoin/bring_up_failures_total": 0.0,
          },
      )
      await minimal_engine.close()

    asyncio.run(_run())

  def test_engine_enforces_rollout_capacity_floor_across_init_poll_and_sync(
      self,
  ):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      w0, w1 = MockActorHandle(), MockActorHandle()
      for idx, handle in enumerate((w0, w1)):
        wid = f"rollout-{idx}"
        registry.register_handle(wid, [datatypes.Role.ROLLOUT], handle)
        worker_registry_testing.set_status(
            registry, wid, worker_registry.AdmissionStatus.ACTIVE
        )

      w1_q: asyncio.Queue[remote_execution.ExecutionResponse] = asyncio.Queue()

      async def _w1_generate(*, requests):
        req = requests[0]
        w1_q.put_nowait(
            remote_execution.ExecutionResponse(
                request_id=req.request_id,
                result=[
                    datatypes.RolloutResponse(
                        request_id=req.request_id,
                        status="COMPLETED",
                        payload=datatypes.TrajectoryItem(
                            prompt_id=str(req.prompt_id),
                            group_index=req.group_index,
                            traj={
                                "trajectory_reward": 1.0,
                                "status": datatypes.TrajectoryStatus.SUCCEEDED,
                            },
                        ),
                    )
                ],
            )
        )

      async def _w1_poll(timeout_s=50.0):
        del timeout_s
        return await w1_q.get()

      w1.generate.side_effect = _w1_generate
      w1.poll_responses.side_effect = _w1_poll
      w0.poll_responses.side_effect = ConnectionError("w0 died on first poll")

      now = [0.0]
      ftm = fault_tolerance_manager_lib.FaultToleranceManager(
          registry,
          config=datatypes.FaultToleranceConfig(
              enabled=True,
              max_task_retries=1,
              min_active_rollout_ratio=1.0,
              min_active_rollout_grace_s=10.0,
          ),
      )
      coordinator = _FakeWeightSyncCoordinator(workers=[])
      with mock.patch.object(
          fault_tolerance_manager_lib, "_monotonic", side_effect=lambda: now[0]
      ):
        engine = distributed_rl_engine.DistributedRLEngine(
            registry=registry,
            weight_sync_coordinator=coordinator,
            fault_tolerance_manager=ftm,
        )

        # Dispatch 2 requests (1 to w0, 1 to w1). w0 is evicted during the first
        # poll_rollouts() call; because __init__ seeded peak=2, poll_rollouts()
        # observes 1/2 < 1.0 and starts the grace timer at t=0.0.
        await engine.dispatch_rollout_requests([
            datatypes.RolloutRequest(
                request_id=f"req_{g}",
                prompt="p",
                prompt_id="p0",
                group_index=g,
            )
            for g in range(2)
        ])
        completed: list[datatypes.TrajectoryItem] = []
        while len(completed) < 2:
          completed.extend(await engine.poll_rollouts(timeout_s=1.0))
        self.assertLen(completed, 2)
        self.assertEqual(
            engine.fault_tolerance_metrics()["ft/fleet/rollout_active_ratio"],
            0.5,
        )

        # Once the 10s grace window elapses without rejoin, sync_weights()
        # raises RolloutCapacityError with the 1/2 active baseline intact.
        now[0] = 10.0
        with self.assertRaisesRegex(
            fault_tolerance_manager_lib.RolloutCapacityError,
            r"for 10\.0s \(grace 10\.0s\): 1/2 active \(0\.500 < 1\.000\)",
        ):
          await engine.sync_weights(policy_version=1)
        await engine.close()

    asyncio.run(_run())

  def _bind_completion_queues(self, workers: list[MockActorHandle]):
    queues = [
        asyncio.Queue[remote_execution.ExecutionResponse]() for _ in workers
    ]
    for handle, response_q in zip(workers, queues, strict=True):

      async def _poll(timeout_s=50.0, q=response_q):
        del timeout_s
        return await q.get()

      handle.poll_responses.side_effect = _poll

    def _enqueue_completions(worker_idx: int, count: int) -> None:
      handle = workers[worker_idx]
      for _ in range(count):
        req_id = handle._dispatched_request_ids.popleft()
        queues[worker_idx].put_nowait(
            remote_execution.ExecutionResponse(
                request_id=req_id,
                result=[
                    datatypes.RolloutResponse(
                        request_id=req_id,
                        status="COMPLETED",
                        payload=datatypes.TrajectoryItem(
                            prompt_id="done",
                            group_index=0,
                            traj={
                                "trajectory_reward": 1.0,
                                "status": datatypes.TrajectoryStatus.SUCCEEDED,
                            },
                        ),
                    )
                ],
            )
        )

    return _enqueue_completions

  def test_dispatch_rollout_requests_stripes_onto_least_loaded_workers(self):
    async def _run():
      workers = [MockActorHandle() for _ in range(4)]
      enqueue_completions = self._bind_completion_queues(workers)
      w0, w1, w2, w3 = workers
      engine = _make_engine(rollout_workers=workers)

      # Dispatch two 16-request groups so loads start at [8, 8, 8, 8], then
      # complete 0 on w0, 8 on w1, 4 on w2, and 8 on w3 via poll_rollouts() to
      # establish skewed active loads [8, 0, 4, 0] through public APIs.
      for seed_idx in range(2):
        await engine.dispatch_rollout_requests([
            datatypes.RolloutRequest(
                prompt=f"seed_{seed_idx}",
                prompt_id=f"seed_prompt_{seed_idx}",
                group_index=g,
            )
            for g in range(16)
        ])
      enqueue_completions(1, 8)
      enqueue_completions(2, 4)
      enqueue_completions(3, 8)
      drained = 0
      while drained < 20:
        batch = await engine.poll_rollouts(timeout_s=1.0)
        drained += len(batch)
      self.assertEqual(drained, 20)
      for handle in workers:
        handle.generate.reset_mock()

      reqs_p0 = [
          datatypes.RolloutRequest(
              prompt="p0",
              prompt_id="deepswe_0",
              group_index=g,
          )
          for g in range(16)
      ]
      await engine.dispatch_rollout_requests(reqs_p0)
      # Active loads started at [8, 0, 4, 0]:
      # - Stripe 0 (g=0..3) -> w1 (load 0 -> 4, rr_cursor=2)
      # - Stripe 1 (g=4..7) -> w3 (load 0 -> 4, rr_cursor=0)
      # - Stripe 2 (g=8..11) -> w1 (load 4 -> 8, tied [8,4,4,4], rr_cursor=0)
      # - Stripe 3 (g=12..15) -> w2 (load 4 -> 8, tied [8,8,4,4], rr_cursor=2)
      self.assertEqual(w0.generate.call_count, 0)
      self.assertEqual(w1.generate.call_count, 8)
      self.assertEqual(w2.generate.call_count, 4)
      self.assertEqual(w3.generate.call_count, 4)

      w1_group_indices = [
          call.kwargs["requests"][0].group_index
          for call in w1.generate.call_args_list
      ]
      w2_group_indices = [
          call.kwargs["requests"][0].group_index
          for call in w2.generate.call_args_list
      ]
      w3_group_indices = [
          call.kwargs["requests"][0].group_index
          for call in w3.generate.call_args_list
      ]
      self.assertEqual(w1_group_indices, [0, 1, 2, 3, 8, 9, 10, 11])
      self.assertEqual(w3_group_indices, [4, 5, 6, 7])
      self.assertEqual(w2_group_indices, [12, 13, 14, 15])

      # Active loads are now [8, 8, 8, 4]; with G=4 on 4 workers, eff_stripe=1,
      # so all 4 trajectories route to w3 (4 -> 5 -> 6 -> 7 -> 8).
      reqs_p1 = [
          datatypes.RolloutRequest(
              prompt="p1",
              prompt_id="deepswe_1",
              group_index=g,
          )
          for g in range(4)
      ]
      await engine.dispatch_rollout_requests(reqs_p1)
      self.assertEqual(w3.generate.call_count, 8)
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_handles_remainder_stripe(self):
    async def _run():
      w0, w1 = MockActorHandle(), MockActorHandle()
      engine = _make_engine(rollout_workers=[w0, w1])
      # G=10 on 2 workers -> eff_stripe = min(4, 10 // 2) = 4:
      # stripes [0, 1, 2, 3] -> w0 (load 4), [4, 5, 6, 7] -> w1 (load 4),
      # and remainder [8, 9] -> w0 (load 6).
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_rem",
              prompt_id="prompt_rem",
              group_index=g,
          )
          for g in range(10)
      ])
      w0_groups = [
          call.kwargs["requests"][0].group_index
          for call in w0.generate.call_args_list
      ]
      w1_groups = [
          call.kwargs["requests"][0].group_index
          for call in w1.generate.call_args_list
      ]
      self.assertEqual(w0_groups, [0, 1, 2, 3, 8, 9])
      self.assertEqual(w1_groups, [4, 5, 6, 7])

      # A follow-up G=2 prompt (eff_stripe=1) sends both requests to w1 (load
      # 4 -> 5 -> 6) to balance w0's load of 6.
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_next",
              prompt_id="prompt_next",
              group_index=g,
          )
          for g in range(2)
      ])
      self.assertEqual(w0.generate.call_count, 6)
      self.assertEqual(w1.generate.call_count, 6)
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_load_balances_small_groups(self):
    async def _run():
      pool4 = [MockActorHandle() for _ in range(4)]
      engine = _make_engine(rollout_workers=pool4)
      for p_idx in range(2):
        await engine.dispatch_rollout_requests([
            datatypes.RolloutRequest(
                prompt=f"small_{p_idx}",
                prompt_id=f"small_prompt_{p_idx}",
                group_index=g,
            )
            for g in range(2)
        ])
      self.assertEqual([w.generate.call_count for w in pool4], [1, 1, 1, 1])
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_persists_tie_break_cursor_across_completions(
      self,
  ):
    async def _run():
      workers = [MockActorHandle() for _ in range(4)]
      enqueue_completions = self._bind_completion_queues(workers)
      engine = _make_engine(rollout_workers=workers)

      # Step 1: G=2 on 4 idle workers (eff_stripe=1) routes to w0 and w1,
      # leaving _last_routed_actor=w1 (next cursor index=2 -> w2).
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_a",
              prompt_id="prompt_a",
              group_index=g,
          )
          for g in range(2)
      ])
      self.assertEqual([w.generate.call_count for w in workers], [1, 1, 0, 0])

      # Complete both tasks so active loads return to a 4-way tie [0, 0, 0, 0].
      enqueue_completions(0, 1)
      enqueue_completions(1, 1)
      drained = 0
      while drained < 2:
        drained += len(await engine.poll_rollouts(timeout_s=1.0))
      self.assertEqual(drained, 2)

      # Step 2: G=1 with tied loads [0, 0, 0, 0] must resume at w2 (not reset
      # to w0).
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_b",
              prompt_id="prompt_b",
              group_index=0,
          )
      ])
      self.assertEqual([w.generate.call_count for w in workers], [1, 1, 1, 0])

      # Complete w2's task (loads [0, 0, 0, 0]) and verify a G=2 prompt wraps
      # around onto w3 (g=0) and w0 (g=1).
      enqueue_completions(2, 1)
      self.assertLen(await engine.poll_rollouts(timeout_s=1.0), 1)
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_c",
              prompt_id="prompt_c",
              group_index=g,
          )
          for g in range(2)
      ])
      self.assertEqual([w.generate.call_count for w in workers], [2, 1, 1, 1])
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_tie_break_cursor_tracks_actor_identity_across_eviction(
      self,
  ):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      workers = [MockActorHandle() for _ in range(4)]
      enqueue_completions = self._bind_completion_queues(workers)
      w0, w1, w2, w3 = workers
      engine = _make_engine(registry=registry, rollout_workers=workers)

      # Route G=2 onto w0 and w1 (_last_routed_actor=w1), drain both tasks, and
      # evict rollout-0 so live actors shift to [w1, w2, w3] at loads [0, 0, 0].
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_init",
              prompt_id="prompt_init",
              group_index=g,
          )
          for g in range(2)
      ])
      enqueue_completions(0, 1)
      enqueue_completions(1, 1)
      drained = 0
      while drained < 2:
        drained += len(await engine.poll_rollouts(timeout_s=1.0))
      self.assertEqual(drained, 2)
      registry.unregister("rollout-0")

      # Identity-based cursor resumes immediately after w1 onto w2 (rather than
      # skipping w2 via a stale positional index=2 -> w3).
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt="p_after_evict",
              prompt_id="prompt_after_evict",
              group_index=0,
          )
      ])
      self.assertEqual(
          [w.generate.call_count for w in (w0, w1, w2, w3)], [1, 1, 1, 0]
      )
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_spreads_single_trajectory_prompts_evenly(
      self,
  ):
    async def _run():
      # 16 distinct G=1 prompts in one batch on 8 idle workers must spread
      # evenly ([2] * 8) rather than packing 4 unrelated prompts onto 0..3.
      pool8 = [MockActorHandle() for _ in range(8)]
      engine = _make_engine(rollout_workers=pool8)
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              prompt=f"p_{idx}",
              prompt_id=f"prompt_{idx}",
              group_index=0,
          )
          for idx in range(16)
      ])
      self.assertEqual([w.generate.call_count for w in pool8], [2] * 8)
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_never_straddles_prompt_boundaries(self):
    async def _run():
      # 3 prompts with G=6 on 2 workers (_ROLLOUT_PROMPT_STRIPE_SIZE=4) must
      # compute eff_stripe = min(4, 6 // 2) = 3 per prompt so stripes never
      # straddle distinct prompt_ids.
      w0, w1 = MockActorHandle(), MockActorHandle()
      engine = _make_engine(rollout_workers=[w0, w1])
      multi_reqs: list[datatypes.RolloutRequest] = []
      for pid in ("alpha", "beta", "gamma"):
        for g in range(6):
          multi_reqs.append(
              datatypes.RolloutRequest(
                  prompt=f"text_{pid}",
                  prompt_id=pid,
                  group_index=g,
              )
          )
      await engine.dispatch_rollout_requests(multi_reqs)
      w0_pairs = [
          (
              call.kwargs["requests"][0].prompt_id,
              call.kwargs["requests"][0].group_index,
          )
          for call in w0.generate.call_args_list
      ]
      w1_pairs = [
          (
              call.kwargs["requests"][0].prompt_id,
              call.kwargs["requests"][0].group_index,
          )
          for call in w1.generate.call_args_list
      ]
      self.assertEqual(
          w0_pairs,
          [
              ("alpha", 0),
              ("alpha", 1),
              ("alpha", 2),
              ("beta", 0),
              ("beta", 1),
              ("beta", 2),
              ("gamma", 0),
              ("gamma", 1),
              ("gamma", 2),
          ],
      )
      self.assertEqual(
          w1_pairs,
          [
              ("alpha", 3),
              ("alpha", 4),
              ("alpha", 5),
              ("beta", 3),
              ("beta", 4),
              ("beta", 5),
              ("gamma", 3),
              ("gamma", 4),
              ("gamma", 5),
          ],
      )
      await engine.close()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_eviction_spreads_retries_across_survivors(
      self,
  ):
    async def _run():
      trigger_crash = asyncio.Event()
      registry = worker_registry.WorkerRegistry()
      workers = [MockActorHandle() for _ in range(4)]
      w0, w1, w2, w3 = workers
      survivor_queues: dict[
          MockActorHandle, asyncio.Queue[remote_execution.ExecutionResponse]
      ] = {
          w1: asyncio.Queue(),
          w2: asyncio.Queue(),
          w3: asyncio.Queue(),
      }

      for survivor, response_q in survivor_queues.items():

        def _bind_survivor(
            handle: MockActorHandle,
            q: asyncio.Queue[remote_execution.ExecutionResponse],
        ) -> None:
          async def _generate(*, requests):
            req = requests[0]
            q.put_nowait(
                remote_execution.ExecutionResponse(
                    request_id=req.request_id,
                    result=[
                        datatypes.RolloutResponse(
                            request_id=req.request_id,
                            status="COMPLETED",
                            payload=datatypes.TrajectoryItem(
                                prompt_id=str(req.prompt_id),
                                group_index=req.group_index,
                                traj={
                                    "trajectory_reward": 1.0,
                                    "status": (
                                        datatypes.TrajectoryStatus.SUCCEEDED
                                    ),
                                },
                            ),
                        )
                    ],
                )
            )

          async def _poll(timeout_s=50.0):
            del timeout_s
            return await q.get()

          handle.generate.side_effect = _generate
          handle.poll_responses.side_effect = _poll

        _bind_survivor(survivor, response_q)

      async def _crashing_poll(timeout_s=50.0):
        del timeout_s
        await trigger_crash.wait()
        raise ConnectionError("rollout-0 crashed with active striped tasks")

      w0.poll_responses.side_effect = _crashing_poll

      engine = _make_engine(
          registry=registry,
          rollout_workers=workers,
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=True, max_task_retries=1
          ),
      )

      # G=32 on 4 workers places 8 trajectories on each worker (w0 receives
      # g=0..3 and g=16..19 via preferred_actor=w0 while keeping
      # route_key=req.traj_id).
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              request_id=f"req_evict_g{g}",
              prompt="p_evict",
              prompt_id="prompt_evict",
              group_index=g,
          )
          for g in range(32)
      ])
      self.assertEqual(w0.generate.call_count, 8)
      before_eviction_counts = [w.generate.call_count for w in (w1, w2, w3)]
      self.assertEqual(before_eviction_counts, [8, 8, 8])

      trigger_crash.set()
      completed: list[datatypes.TrajectoryItem] = []
      while len(completed) < 32:
        completed.extend(await engine.poll_rollouts(timeout_s=1.0))

      self.assertLen(completed, 32)
      self.assertEqual(
          sorted(item.group_index for item in completed), list(range(32))
      )
      self.assertNotIn("rollout-0", registry)
      retried_per_survivor = [
          w.generate.call_count - 8 for w in (w1, w2, w3)
      ]
      self.assertEqual(sum(retried_per_survivor), 8)
      # Retries hash distinct traj_id strings across all 3 survivors instead of
      # collapsing all 8 orphaned tasks onto a single k % (N - 1) survivor.
      self.assertTrue(all(count > 0 for count in retried_per_survivor))
      await engine.close()

    asyncio.run(_run())

  def test_routed_experts_spec_validation_and_staging_liveness(self):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      w0, w1 = MockActorHandle(), MockActorHandle()
      registry.register_handle(
          "rollout-0",
          [datatypes.Role.ROLLOUT],
          w0,
          resources={"routed_experts_shape": None, "staging_max_bytes": 1024},
      )
      registry.register_handle(
          "rollout-1",
          [datatypes.Role.ROLLOUT],
          w1,
          resources={"routed_experts_shape": (4, 8), "staging_max_bytes": 1024},
      )
      for wid in ("rollout-0", "rollout-1"):
        worker_registry_testing.set_status(
            registry, wid, worker_registry.AdmissionStatus.ACTIVE
        )
      coord = _FakeWeightSyncCoordinator()
      engine = _make_engine(
          registry=registry,
          weight_sync_coordinator=coord,
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=True, min_active_rollout_ratio=0.5
          ),
      )

      # 1. None shape fails fast.
      with self.assertRaisesRegex(ValueError, "routed_experts_shape=None"):
        engine.routed_experts_spec(encode_mode="uint8")

      # 2. Disagreed shapes across workers fail fast.
      registry.info("rollout-0").resources["routed_experts_shape"] = (4, 4)
      with self.assertRaisesRegex(ValueError, "disagreeing with"):
        engine.routed_experts_spec(encode_mode="uint8")

      # 3. Insufficient total worst-case slots fails fast without mutating
      # engine state (`_staging_worst_case_bytes` stays None):
      # worst_case_bytes = (8 + 8) * (4 * 8 * 1) = 512 -> 2 slots/worker (4).
      registry.info("rollout-0").resources["routed_experts_shape"] = (4, 8)
      spec = engine.routed_experts_spec(encode_mode="uint8")
      self.assertEqual(
          spec,
          batch_assembly.RoutedExpertsSpec(
              num_layers=4, top_k=8, encoding="uint8"
          ),
      )
      with self.assertRaisesRegex(RuntimeError, "below required deadlock-free"):
        engine.configure_staging_liveness(
            worst_case_bytes=512,
            required_slots=6,
        )
      self.assertIsNone(engine._staging_worst_case_bytes)
      self.assertEqual(engine._required_staging_slots, 0)

      # 4. Valid capacity succeeds, pool shrink below `required_slots` at
      # `sync_weights` fails fast, and `clear_all_staged_routed_experts` resets
      # the liveness bound so a subsequent non-Relay run on the same engine is
      # unaffected.
      engine.configure_staging_liveness(
          worst_case_bytes=512,
          required_slots=4,
      )
      await engine.sync_weights(min_retained_policy_version=-1)
      self.assertEqual(
          coord.kwargs_history[-1]["min_retained_policy_version"], -1
      )
      registry.unregister("rollout-1")
      with self.assertRaisesRegex(RuntimeError, "below required deadlock-free"):
        await engine.sync_weights(min_retained_policy_version=0)
      await engine.clear_all_staged_routed_experts()
      self.assertIsNone(engine._staging_worst_case_bytes)
      self.assertEqual(engine._required_staging_slots, 0)
      await engine.sync_weights()
      await engine.close()

    asyncio.run(_run())

  def test_occupancy_aware_routing_and_inline_generate(
      self,
  ):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      w0, w1 = MockActorHandle(), MockActorHandle()
      # worst_case_bytes = (5 + 5) * (2 * 5 * 1) = 100 -> 4 slots per worker.
      for idx, handle in enumerate((w0, w1)):
        registry.register_handle(
            f"rollout-{idx}",
            [datatypes.Role.ROLLOUT],
            handle,
            resources={
                "routed_experts_shape": (2, 5),
                "staging_max_bytes": 400,
            },
        )
        worker_registry_testing.set_status(
            registry, f"rollout-{idx}", worker_registry.AdmissionStatus.ACTIVE
        )
      engine = _make_engine(
          registry=registry,
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=True, min_active_rollout_ratio=0.5
          ),
      )
      engine.routed_experts_spec(encode_mode="uint8")
      engine.configure_staging_liveness(
          worst_case_bytes=100,
          required_slots=6,
      )

      # Blocking standalone `generate()` must keep `stage_routed_experts=False`.
      w0.generate.return_value = [
          datatypes.RolloutResponse(
              request_id="r_inline",
              status="COMPLETED",
              payload=datatypes.TrajectoryItem(
                  prompt_id="p_inline",
                  group_index=0,
                  traj={
                      "trajectory_reward": 1.0,
                      "status": datatypes.TrajectoryStatus.SUCCEEDED,
                  },
              ),
          )
      ]
      await engine.generate(
          [{"prompt": "hi", "prompt_id": "p_inline"}],
          generation_args=datatypes.GenerationArgs(return_routed_experts=True),
      )
      inline_req = w0.generate.call_args.kwargs["requests"][0]
      self.assertFalse(inline_req.stage_routed_experts)
      w0.generate.reset_mock()

      # Drive `rollout-0` to 3/4 unreleased staged refs via real `poll_rollouts`
      # (`_unreleased_staged_refs["rollout-0"] += 1` per received ref).
      seed_refs = [
          datatypes.RoutedExpertsRef(
              worker_id="rollout-0", instance_id="inst-0", ref_id=10 + g
          )
          for g in range(3)
      ]
      w0.poll_responses.return_value = [
          datatypes.RolloutResponse(
              request_id=f"seed_{g}",
              status="COMPLETED",
              payload=datatypes.TrajectoryItem(
                  prompt_id="p_seed",
                  group_index=g,
                  traj={
                      "trajectory_reward": 1.0,
                      "status": datatypes.TrajectoryStatus.SUCCEEDED,
                  },
                  routed_experts_ref=seed_refs[g],
              ),
          )
          for g in range(3)
      ]
      await engine.dispatch_rollout_requests([
          datatypes.RolloutRequest(
              request_id="req_p_seed_g0_v0",
              prompt="seed",
              prompt_id="p_seed",
              group_index=0,
              stage_routed_experts=True,
              max_response_length=5,
          )
      ])
      polled = await engine.poll_rollouts(timeout_s=0.1)
      self.assertLen(polled, 3)
      self.assertEqual(engine._unreleased_staged_refs["rollout-0"], 3)
      w0.generate.reset_mock()
      w1.generate.reset_mock()

      await engine.dispatch_rollouts(
          [{"prompt": "p_skew", "prompt_id": "p_skew"}],
          num_generations=4,
          stage_routed_experts=True,
          max_response_length=5,
      )
      w0_reqs = [
          call.kwargs["requests"][0] for call in w0.generate.call_args_list
      ]
      w1_reqs = [
          call.kwargs["requests"][0] for call in w1.generate.call_args_list
      ]
      # rollout-0 had only 1 free slot (3/4), rollout-1 had 4 free slots (0/4):
      # rollout-1 absorbs >= 3 of the 4 requests without overflowing rollout-0!
      self.assertLessEqual(len(w0_reqs), 1)
      self.assertGreaterEqual(len(w1_reqs), 3)
      for req in (*w0_reqs, *w1_reqs):
        self.assertTrue(req.stage_routed_experts)
        self.assertEqual(req.max_response_length, 5)

      # Drain in-flight skew requests and release the 3 staged refs on
      # `rollout-0` -> occupancy returns to 0 and balanced routing is restored.
      await engine._rollout_session.abandon_in_flight()
      await engine.release_routed_experts(seed_refs)
      self.assertEmpty(engine._unreleased_staged_refs)
      w0.generate.reset_mock()
      w1.generate.reset_mock()
      await engine.dispatch_rollouts(
          [{"prompt": "p_balanced", "prompt_id": "p_balanced"}],
          num_generations=4,
          stage_routed_experts=True,
          max_response_length=5,
      )
      self.assertEqual(w0.generate.call_count, 2)
      self.assertEqual(w1.generate.call_count, 2)

      # Post-poll `_check_staging_liveness()` raises once pool slots (4) drop
      # below `required_slots` (6).
      registry.unregister("rollout-1")
      with self.assertRaisesRegex(RuntimeError, "below required deadlock-free"):
        await engine.poll_rollouts(timeout_s=0.05)
      await engine.close()

    asyncio.run(_run())

  def test_fetch_release_and_clear_staged_routed_experts_contracts(self):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      w0, w1 = MockActorHandle(), MockActorHandle()
      for idx, handle in enumerate((w0, w1)):
        registry.register_handle(
            f"rollout-{idx}",
            [datatypes.Role.ROLLOUT],
            handle,
            resources={
                "routed_experts_shape": (2, 2),
                "staging_max_bytes": 4096,
            },
        )
        worker_registry_testing.set_status(
            registry, f"rollout-{idx}", worker_registry.AdmissionStatus.ACTIVE
        )
      engine = _make_engine(
          registry=registry,
          fault_tolerance_config=datatypes.FaultToleranceConfig(
              enabled=True, min_active_rollout_ratio=0.5
          ),
      )

      ref0 = datatypes.RoutedExpertsRef(
          worker_id="rollout-0", instance_id="inst-0", ref_id=101
      )
      ref1 = datatypes.RoutedExpertsRef(
          worker_id="rollout-1", instance_id="inst-1", ref_id=202
      )
      arr0 = np.ones((4, 2, 2), dtype=np.uint8)
      arr1 = np.full((4, 2, 2), 2, dtype=np.uint8)
      w0.fetch_staged.return_value = {101: arr0}
      w1.fetch_staged.return_value = {202: arr1}

      # Releasing an uncounted ref fails fast before counter mutation or RPC.
      with self.assertRaisesRegex(RuntimeError, "only 0 unreleased ref"):
        await engine.release_routed_experts([ref0])
      w0.release_staged.assert_not_awaited()

      engine._unreleased_staged_refs["rollout-0"] = 1
      engine._unreleased_staged_refs["rollout-1"] = 1

      fetched = await engine.fetch_routed_experts([ref0, ref1])
      np.testing.assert_array_equal(fetched[ref0], arr0)
      np.testing.assert_array_equal(fetched[ref1], arr1)

      # Evicting rollout-1 causes `fetch_routed_experts` to fail fast, while
      # `release_routed_experts` skips only the `KeyError` worker with a warning
      # and counter bump, still releasing `ref0` on `rollout-0`.
      registry.unregister("rollout-1")
      with self.assertRaisesRegex(RuntimeError, "was evicted before staged"):
        await engine.fetch_routed_experts([ref1])

      await engine.release_routed_experts([ref0, ref1])
      w0.release_staged.assert_awaited_once_with(ref_ids=[101])
      self.assertEqual(engine.evicted_release_skips_total, 1)
      self.assertEqual(
          engine.fault_tolerance_metrics()[
              "ft/rollout/evicted_release_skips_total"
          ],
          1.0,
      )
      self.assertEmpty(engine._unreleased_staged_refs)

      # Re-releasing already-released `ref0` fails fast!
      with self.assertRaisesRegex(RuntimeError, "only 0 unreleased ref"):
        await engine.release_routed_experts([ref0])

      # Non-KeyError RPC failures on `release_staged` MUST propagate!
      engine._unreleased_staged_refs["rollout-0"] = 1
      w0.release_staged.side_effect = RuntimeError("gRPC INTERNAL release fail")
      with self.assertRaisesRegex(RuntimeError, "gRPC INTERNAL release fail"):
        await engine.release_routed_experts([ref0])

      # `clear_all_staged_routed_experts` clears active workers, resets
      # counters, and bumps the rollout request-id epoch (`_e1`).
      engine._unreleased_staged_refs["rollout-0"] = 5
      await engine.clear_all_staged_routed_experts()
      w0.clear_staged.assert_awaited_once()
      self.assertEmpty(engine._unreleased_staged_refs)
      req_ids = await engine.dispatch_rollouts(
          [{"prompt": "hi", "prompt_id": "p_next_run"}],
          num_generations=1,
          policy_version=0,
      )
      self.assertEqual(req_ids, ["req_p_next_run_g0_v0_e1"])
      await engine.close()

    asyncio.run(_run())

  def test_configure_staging_liveness_rejects_zero_slot_worker_and_supports_overcommit(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    w0, w1 = MockActorHandle(), MockActorHandle()
    registry.register_handle(
        "rollout-0",
        [datatypes.Role.ROLLOUT],
        w0,
        resources={"staging_max_bytes": 1000},
    )
    registry.register_handle(
        "rollout-1",
        [datatypes.Role.ROLLOUT],
        w1,
        resources={"staging_max_bytes": 5000},
    )
    for wid in ("rollout-0", "rollout-1"):
      worker_registry_testing.set_status(
          registry, wid, worker_registry.AdmissionStatus.ACTIVE
      )
    engine = distributed_rl_engine.DistributedRLEngine(registry=registry)

    # rollout-0 has 0 physical slots when worst_case_bytes=2000; must fail even
    # though rollout-1 alone has 2 slots.
    with self.assertRaisesRegex(RuntimeError, "0 physical staging slots"):
      engine.configure_staging_liveness(worst_case_bytes=2000, required_slots=2)

    # With worst_case_bytes=1000, rollout-0 has 1 slot and rollout-1 has 5
    # slots (6 total). overcommit_factor=1.5 scales orchestrator slots to
    # int(1*1.5) + int(5*1.5) = 1 + 7 = 8.
    with self.assertRaisesRegex(RuntimeError, "only 6 worst-case"):
      engine.configure_staging_liveness(
          worst_case_bytes=1000, required_slots=8, overcommit_factor=1.0
      )
    engine.configure_staging_liveness(
        worst_case_bytes=1000, required_slots=8, overcommit_factor=1.5
    )

    async def _run():
      # Over-budget request max_response_length is rejected when staging.
      with self.assertRaisesRegex(
          ValueError, "RolloutRequest.max_response_length must be in"
      ):
        await engine.dispatch_rollouts(
            [{"prompt": "hi", "prompt_id": "p_bad", "max_response_length": 9}],
            num_generations=1,
            policy_version=0,
            stage_routed_experts=True,
            max_response_length=8,
        )
      await engine.close()

    asyncio.run(_run())

  def test_release_and_clear_skip_transport_error_under_fault_tolerance(self):
    async def _run():
      registry = worker_registry.WorkerRegistry()
      w0 = MockActorHandle()
      registry.register_handle("rollout-0", [datatypes.Role.ROLLOUT], w0)
      worker_registry_testing.set_status(
          registry, "rollout-0", worker_registry.AdmissionStatus.ACTIVE
      )
      ftm = fault_tolerance_manager_lib.FaultToleranceManager(
          registry,
          config=datatypes.FaultToleranceConfig(enabled=True),
      )
      engine = distributed_rl_engine.DistributedRLEngine(
          registry=registry, fault_tolerance_manager=ftm
      )
      engine._unreleased_staged_refs["rollout-0"] = 1
      w0.release_staged.side_effect = ConnectionError("worker died mid-release")
      await engine.release_routed_experts([
          datatypes.RoutedExpertsRef(
              worker_id="rollout-0", instance_id="inst-0", ref_id=1
          )
      ])
      self.assertEqual(engine.evicted_release_skips_total, 1)
      self.assertEqual(
          engine.fault_tolerance_metrics()[
              "ft/rollout/evicted_release_skips_total"
          ],
          1.0,
      )
      self.assertEmpty(engine._unreleased_staged_refs)

      w0.clear_staged.side_effect = ConnectionError("worker died mid-clear")
      await engine.clear_all_staged_routed_experts()
      await engine.close()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
