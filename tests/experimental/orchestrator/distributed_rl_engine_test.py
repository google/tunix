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
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.worker import remote_execution


class MockActorHandle(mock.MagicMock):
  """A smart mock for ActorHandle that routes asubmit/dispatch_task to logical methods.

  This allows tests to write clean assertions like
  `worker.generate.assert_called_once()` while preserving the strict ActorHandle
  type requirements of the engine.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(spec=remote_execution.ActorHandle, *args, **kwargs)
    # Ensure all mocked methods return awaitables by default
    self.generate = mock.AsyncMock()
    self.poll_responses = mock.AsyncMock()
    self.weight_sync = mock.AsyncMock()
    self.fwd_bwd = mock.AsyncMock()
    self.update = mock.AsyncMock()
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

  async def asubmit(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return await method(*args, **kwargs)

  async def dispatch_task(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return await method(*args, **kwargs)


class _RecordingRouter:
  """Callable router stub recording `(method_name, kwargs)` per routing call.

  Deliberately defines no `generate` attribute so RoutingActorPool takes the
  plain-callable `(actors, method_name, args, kwargs)` path — the same one
  remote_scheduler_router.RemoteSchedulerRouter relies on.
  """

  def __init__(self, pick_index: int = 0):
    self.pick_index = pick_index
    self.calls = []
    self.weights_synced_count = 0

  def __call__(self, actors, method_name, args, kwargs):
    self.calls.append((method_name, dict(kwargs or {})))
    return actors[self.pick_index]

  def notify_weights_synced(self):
    self.weights_synced_count += 1


class DistributedRLEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_rollout_1 = MockActorHandle()
    self.mock_rollout_2 = MockActorHandle()
    self.mock_actor = MockActorHandle()
    self.mock_ref = MockActorHandle()

    self.engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
    )

  def test_generate_load_balances_across_rollout_workers(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      resp2 = datatypes.RolloutResponse(
          request_id="r2", status="COMPLETED", env_reward=2.0
      )

      self.mock_rollout_1.generate.return_value = [resp1]
      self.mock_rollout_2.generate.return_value = [resp2]

      results = await self.engine.generate([
          {"prompt": "p1", "prompt_id": "p1", "metadata": {"prefix_hash": 0}},
          {"prompt": "p2", "prompt_id": "p2", "metadata": {"prefix_hash": 1}},
      ])
      self.assertLen(results, 2)
      rewards = {res.traj.reward for res in results}
      self.assertEqual(rewards, {1.0, 2.0})

      # Verify underlying logical methods were called correctly
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      p1 = self.mock_rollout_1.generate.call_args.kwargs["requests"][0]
      self.assertEqual(p1.prompt, "p1")
      self.assertEqual(p1.prompt_id, "p1")

      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)
      p2 = self.mock_rollout_2.generate.call_args.kwargs["requests"][0]
      self.assertEqual(p2.prompt, "p2")
      self.assertEqual(p2.prompt_id, "p2")

    asyncio.run(_run())

  def test_generate_uses_explicit_generation_args(self):
    async def _run():
      resp = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      self.mock_rollout_1.generate.return_value = [resp]
      results = await self.engine.generate(
          [{
              "prompt": "p1",
              "prompt_id": "prompt_1",
              "metadata": {"prefix_hash": 0},
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
          metadata={"prefix_hash": 0},
      )
      resp = datatypes.RolloutResponse(
          request_id="r1",
          prompt_id="prompt_1",
          status="COMPLETED",
          env_reward=1.0,
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
          env_reward=1.0,
      )
      self.mock_rollout_1.poll_responses.return_value = [resp1]
      self.mock_rollout_2.poll_responses.return_value = []

      results = await self.engine.poll_rollouts(timeout_s=0.1)
      self.assertEqual(len(results), 1)
      self.assertEqual(results[0].traj.reward, 1.0)

      self.mock_rollout_1.poll_responses.assert_called_once_with(timeout_s=0.1)
      self.mock_rollout_2.poll_responses.assert_called_once_with(timeout_s=0.1)

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
      self.mock_actor.update.return_value = 3
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
      self.assertEqual(res, datatypes.Response(metadata={"checkpoint_saved": True}))
      self.mock_actor.save_checkpoint.assert_called_once_with(
          metadata=metadata
      )

    asyncio.run(_run())

  def test_save_checkpoint_propagates_step_and_optional_kwargs(self):
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
          save_only_lora_params=True,
          custom_flag="custom_val",
      )
      self.assertEqual(
          res, datatypes.Response(metadata={"checkpoint_saved": True})
      )
      self.mock_actor.save_checkpoint.assert_called_once_with(
          metadata=metadata,
          step=10,
          force=True,
          save_only_lora_params=True,
          custom_flag="custom_val",
      )

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

  def test_train_step_propagates_optional_kwargs(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = {"loss": 0.5}
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
          custom_arg="test_arg",
      )
      self.assertEqual(res, {"loss": 0.5})
      self.mock_actor.fwd_bwd.assert_called_once()
      call_kwargs = self.mock_actor.fwd_bwd.call_args.kwargs
      self.assertEqual(call_kwargs["skip_jit"], False)
      self.assertEqual(call_kwargs["custom_arg"], "test_arg")
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
      res = await self.engine.get_metrics(
          datatypes.Role.ACTOR, reset=True
      )
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

      class _FakeCoordinator:

        def __init__(self):
          self.calls = []

        async def sync(self, policy_version=0, **kwargs):
          self.calls.append(policy_version)
          _FakeResult.policy_version = policy_version
          return _FakeResult

      coordinator = _FakeCoordinator()
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
          inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
          weight_sync_coordinator=coordinator,
      )
      self.assertEqual(await engine.sync_weights(), 1)
      self.assertEqual(await engine.sync_weights(), 2)
      self.assertEqual(coordinator.calls, [1, 2])

    asyncio.run(_run())

  def test_sync_weights_requires_a_coordinator(self):
    async def _run():
      with self.assertRaises(RuntimeError):
        await self.engine.sync_weights()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_with_prefix_routing(self):
    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="1",
          prompt="p1",
          prompt_id="1",
          metadata={"prefix_hash": 0},
      )
      req2 = datatypes.RolloutRequest(
          request_id="2",
          prompt="p2",
          prompt_id="2",
          metadata={"prefix_hash": 1},
      )

      req_ids = await self.engine.dispatch_rollout_requests([req1, req2])
      self.assertEqual(req_ids, ["1", "2"])

      # Due to deterministic hash logic, req1 -> rollout_1 and req2 -> rollout_2
      self.mock_rollout_1.generate.assert_called_once()
      dispatched_req1 = self.mock_rollout_1.generate.call_args.kwargs[
          "requests"
      ][0]
      self.assertEqual(dispatched_req1.request_id, "1")

      self.mock_rollout_2.generate.assert_called_once()
      dispatched_req2 = self.mock_rollout_2.generate.call_args.kwargs[
          "requests"
      ][0]
      self.assertEqual(dispatched_req2.request_id, "2")

    asyncio.run(_run())

  def test_dispatch_rollouts_delegates_to_dispatch_rollout_requests(self):
    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="1",
          prompt="p1",
          prompt_id="1",
          metadata={"prefix_hash": 0},
      )
      req2 = datatypes.RolloutRequest(
          request_id="2",
          prompt="p2",
          prompt_id="2",
          metadata={"prefix_hash": 1},
      )

      req_ids = await self.engine.dispatch_rollouts([req1, req2])
      self.assertEqual(req_ids, ["1", "2"])

      self.mock_rollout_1.generate.assert_called_once()
      self.mock_rollout_2.generate.assert_called_once()

    asyncio.run(_run())

  def test_dispatch_rollouts_expands_group_size(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [
              {"prompt": "p1", "prompt_id": "p1"},
              {"prompt": "p2", "prompt_id": "p2"},
          ],
          group_size=3,
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
          [dict_item], group_size=2, policy_version=1
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
      self.assertTrue(all(r.metadata["group_size"] == 2 for r in all_dispatched))
      self.assertEqual(
          {r.metadata["env_config"]["group_index"] for r in all_dispatched},
          {0, 1},
      )
      self.assertTrue(
          all(
              r.metadata["env_config"]["group_size"] == 2
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
    prompts = [
        {
            "prompt": "Solve 2+2",
            "prompt_id": "math_p1",
            "metadata": {"env_config": original_env_config},
        }
    ]
    requests = self.engine._build_rollout_requests(
        prompts, group_size=3, policy_version=4
    )
    self.assertLen(requests, 3)

    for idx, req in enumerate(requests):
      self.assertEqual(req.group_index, idx)
      self.assertEqual(req.target_policy_version, 4)
      self.assertEqual(req.metadata["group_index"], idx)
      self.assertEqual(req.metadata["group_size"], 3)
      # Verify env_config deep injection
      env_cfg = req.metadata["env_config"]
      self.assertEqual(env_cfg["env_name"], "math_arena")
      self.assertEqual(env_cfg["timeout_s"], 30)
      self.assertEqual(env_cfg["group_index"], idx)
      self.assertEqual(env_cfg["group_size"], 3)
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
        group_size=2,
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
        group_size=1,
        policy_version=1,
    )
    self.assertLen(requests_none, 1)
    self.assertIsNone(requests_none[0].metadata["env_config"])


    # 3. env_config is omitted
    requests_omitted = self.engine._build_rollout_requests(
        [{"prompt": "p3", "prompt_id": "p3"}],
        group_size=1,
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
          [{"prompt": "p1", "prompt_id": "p1"}],
          group_size=1,
          generation_args=gen_args,
          route_metadata={"prefix_hash": "cache_key_1"},
      )
      self.assertLen(req_ids, 1)

      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched = mock_call.kwargs["requests"][0]
      self.assertEqual(
          dispatched.generation_kwargs,
          {"temperature": 0.7, "max_generation_steps": 128},
      )
      self.assertEqual(dispatched.metadata["prefix_hash"], "cache_key_1")

    asyncio.run(_run())

  def test_dispatch_rollouts_generates_deterministic_request_ids(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "Hello", "prompt_id": "p_123"}],
          group_size=2,
          policy_version=3,
      )
      self.assertEqual(req_ids, ["req_p_123_g0_v3", "req_p_123_g1_v3"])

    asyncio.run(_run())

  def test_dispatch_rollouts_handles_none_metadata(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}],
          group_size=1,
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

  def _engine_with_router(self, router):
    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
        router=router,
    )

  def test_dispatch_rollouts_passes_routing_hints_to_router(self):
    router = _RecordingRouter(pick_index=1)
    engine = self._engine_with_router(router)

    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="r1",
          prompt="p1",
          prompt_id="prompt_1",
          metadata={"prefix_hash": 7},
      )
      req2 = datatypes.RolloutRequest(
          request_id="r2",
          prompt="p2",
          prompt_id="prompt_2",
      )

      await engine.dispatch_rollouts([req1, req2])

      self.assertEqual(
          router.calls,
          [
              (
                  "generate",
                  {"route_key": 7, "request_id": "r1", "prompt": "p1"},
              ),
              (
                  "generate",
                  {"route_key": "prompt_2", "request_id": "r2", "prompt": "p2"},
              ),
          ],
      )
      # Both dispatches must land on the router-picked worker.
      self.mock_rollout_1.generate.assert_not_called()
      self.assertEqual(self.mock_rollout_2.generate.call_count, 2)

    asyncio.run(_run())

  def test_generate_passes_request_routing_hints_to_router(self):
    router = _RecordingRouter(pick_index=0)
    engine = self._engine_with_router(router)

    async def _run():
      request = datatypes.RolloutRequest(
          request_id="r1",
          prompt="p1",
          prompt_id="prompt_1",
          metadata={"prefix_hash": 3},
      )
      resp = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      self.mock_rollout_1.generate.return_value = [resp]

      results = await engine.generate([request])

      self.assertLen(results, 1)
      self.assertEqual(
          router.calls,
          [("generate", {"route_key": 3, "request_id": "r1", "prompt": "p1"})],
      )
      self.mock_rollout_1.generate.assert_called_once()
      self.mock_rollout_2.generate.assert_not_called()

    asyncio.run(_run())

  def test_generate_passes_raw_prompt_routing_hints_to_router(self):
    router = _RecordingRouter(pick_index=1)
    engine = self._engine_with_router(router)

    async def _run():
      resp = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      self.mock_rollout_2.generate.return_value = [resp]

      results = await engine.generate(
          ["p1"], route_metadata={"prefix_hash": "h1"}
      )

      self.assertLen(results, 1)
      self.assertEqual(
          router.calls,
          [("generate", {"route_key": "h1", "prompt": "p1"})],
      )
      self.mock_rollout_2.generate.assert_called_once_with(prompts=["p1"])
      self.mock_rollout_1.generate.assert_not_called()

    asyncio.run(_run())

  def test_sync_weights_notifies_router_once(self):
    router = _RecordingRouter()
    coordinator = mock.MagicMock()
    coordinator.sync = mock.AsyncMock(
        return_value=mock.MagicMock(policy_version=42)
    )
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
        router=router,
        weight_sync_coordinator=coordinator,
    )

    async def _run():
      ver = await engine.sync_weights(role=datatypes.Role.ACTOR)

      self.assertEqual(ver, 42)
      coordinator.sync.assert_awaited_once()
      self.assertEqual(router.weights_synced_count, 1)

    asyncio.run(_run())

  def test_sync_weights_failure_skips_router_notification(self):
    router = _RecordingRouter()
    coordinator = mock.MagicMock()
    coordinator.sync = mock.AsyncMock(
        side_effect=RuntimeError("transfer failed")
    )
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
        router=router,
        weight_sync_coordinator=coordinator,
    )

    async def _run():
      with self.assertRaisesRegex(RuntimeError, "transfer failed"):
        await engine.sync_weights(role=datatypes.Role.ACTOR)

      self.assertEqual(router.weights_synced_count, 0)

    asyncio.run(_run())

  def test_dispatch_rollouts_auto_stamps_lineage_context(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "Hello world", "prompt_id": "prompt_42"}],
          group_size=2,
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

  def test_poll_rollouts_forwards_lineage_to_trajectory_item(self):
    async def _run():
      ctx = lineage.LineageContext(
          tracking_id="traj_p1_g0", parent_tracking_ids=["p1"]
      )
      ctx.add_event("engine.dispatch", "rollout")
      ctx.add_event("worker.rollout", "generate", {"worker_id": "rollout_0"})

      resp = datatypes.RolloutResponse(
          request_id="r1",
          prompt_id="p1",
          group_index=0,
          status="COMPLETED",
          env_reward=1.5,
          metadata={"lineage": ctx},
      )
      self.mock_rollout_1.poll_responses.return_value = [resp]
      self.mock_rollout_2.poll_responses.return_value = []

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
          prompt_id="prompt_1",
          status="COMPLETED",
          env_reward=1.0,
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
          prompt_id="prompt_dict_1",
          group_index=2,
          status="COMPLETED",
          env_reward=1.0,
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
    resp = datatypes.RolloutResponse(
        request_id="req_1",
        prompt_id="p_extra",
        group_index=1,
        policy_version=2,
        status="COMPLETED",
        env_reward=2.0,
        prompt_tokens=np.array([10], dtype=np.int32),
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=np.array([1, 2], dtype=np.int32),
                loss_mask=np.array([1, 1], dtype=np.float32),
            )
        ],
        metadata={"lineage": ctx},
    )

    item = distributed_rl_engine._response_to_trajectory_item(resp)
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
          group_size=1,
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

  def test_response_to_trajectory_item_with_dict_segments(self):
    ctx = lineage.LineageContext(
        tracking_id="traj_p_dict_g0", parent_tracking_ids=["p_dict"]
    )
    resp = datatypes.RolloutResponse(
        request_id="req_dict_seg",
        prompt_id="p_dict",
        group_index=0,
        policy_version=1,
        status="COMPLETED",
        env_reward=1.0,
        prompt_tokens=np.array([10], dtype=np.int32),
        segments=[
            {
                "source": "assistant",
                "tokens": [1, 2],
                "loss_mask": [1.0, 1.0],
            },
            {
                "source": "env",
                "tokens": [3],
            },
        ],  # pyrefly: ignore[bad-argument-type]
        metadata={"lineage": ctx},
    )

    item = distributed_rl_engine._response_to_trajectory_item(resp)
    self.assertEqual(item.prompt_id, "p_dict")
    self.assertEqual(item.group_index, 0)
    self.assertIn("lineage", item.metadata)
    np.testing.assert_array_equal(
        item.completion_tokens, np.array([1, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        item.action_mask, np.array([1, 1], dtype=np.float32)
    )

  def test_response_to_trajectory_item_with_dict_segments_missing_loss_mask(
      self,
  ):
    resp = datatypes.RolloutResponse(
        request_id="req_dict_no_mask",
        prompt_id="p_dict",
        group_index=0,
        policy_version=1,
        status="COMPLETED",
        segments=[{
            "source": "assistant",
            "tokens": [5, 6],
        }],  # pyrefly: ignore[bad-argument-type]
    )
    item = distributed_rl_engine._response_to_trajectory_item(resp)
    np.testing.assert_array_equal(
        item.completion_tokens, np.array([5, 6], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        item.action_mask, np.array([1.0, 1.0], dtype=np.float32)
    )

  def test_poll_rollouts_deserializes_dict_responses_with_dict_segments(self):
    async def _run():
      raw_dict_response = {
          "request_id": "r_deserialized",
          "prompt_id": "p_raw",
          "group_index": 0,
          "policy_version": 1,
          "status": "COMPLETED",
          "env_reward": 2.5,
          "prompt_tokens": [10, 20],
          "segments": [{
              "source": "assistant",
              "tokens": [30, 40],
              "loss_mask": [1.0, 1.0],
          }],
          "metadata": {},
      }
      self.mock_rollout_1.poll_responses.return_value = [raw_dict_response]
      self.mock_rollout_2.poll_responses.return_value = []

      items = await self.engine.poll_rollouts()
      self.assertLen(items, 1)
      item = items[0]
      self.assertEqual(item.prompt_id, "p_raw")
      self.assertEqual(item.traj.reward, 2.5)
      np.testing.assert_array_equal(
          item.prompt_tokens, np.array([10, 20], dtype=np.int32)
      )
      np.testing.assert_array_equal(
          item.completion_tokens, np.array([30, 40], dtype=np.int32)
      )
      np.testing.assert_array_equal(
          item.action_mask, np.array([1.0, 1.0], dtype=np.float32)
      )

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
