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

import asyncio
import builtins
from collections.abc import Sequence
from typing import Any
from unittest import mock

from absl.testing import absltest
import metrax.logging as metrax_logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.worker import remote_execution
from tunix.rl import common as rl_common
from tunix.sft import metrics_logger as metrics_logger_lib
from tunix.sft import utils as sft_utils


class _MockWorkerHandle(mock.MagicMock):
  """Mock remote worker handle (used for rollout and trainer workers).

  Simulates remote ActorHandle execution:
  - Rollout responses: `responses` is a FIFO queue of batches
    (`list[list[RolloutResponse]]`). `poll_responses()` pops and returns the
    next batch inside an `ExecutionResponse`. When `responses` is empty (all
    queued rollouts consumed), it returns `None` to emulate an idle long-polling
    worker awaiting new dispatch requests.
  - Trainer execution: `fwd_bwd`, `update`, and `get_metrics` are handled via
    `asubmit()`.
  """

  def __init__(self, role: str = "rollout", *args: Any, **kwargs: Any):
    super().__init__(spec=remote_execution.ActorHandle, *args, **kwargs)
    self.role = role
    self.responses: list[list[datatypes.RolloutResponse]] = []
    self.metrics_buffer: exp_metrics.MetricsBuffer | None = None
    self.train_step_count: int = 0
    self.dispatched_requests: list[Any] = []

  async def dispatch_task(
      self,
      request_id: str | None = None,
      method_name: str | None = None,
      *args: Any,
      **kwargs: Any,
  ) -> str:
    self.dispatched_requests.append((request_id, method_name, args, kwargs))
    if method_name == "generate":
      reqs = kwargs.get("requests", [])
      if reqs:
        resps = [
            _create_rollout_response(
                request_id=req.request_id,
                prompt_id=req.prompt_id,
                group_index=req.group_index,
                policy_version=req.target_policy_version,
                reward=1.0 + float(req.group_index or 0),
            )
            for req in reqs
        ]
        self.responses.append(resps)
    return request_id or "task_ack"

  async def poll_responses(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> Any:
    """Pops queued rollout responses, or returns None if no responses are ready."""
    del timeout_s
    if self.responses:
      items = self.responses.pop(0)
      return remote_execution.ExecutionResponse(request_id="poll", result=items)
    await asyncio.sleep(0.01)
    return None

  async def asubmit(
      self, method_name: str | None = None, *args: Any, **kwargs: Any
  ) -> Any:
    if method_name == "fwd_bwd":
      return datatypes.Response(request_id="step", metadata={"loss": 0.5})
    elif method_name == "update":
      self.train_step_count += 1
      return self.train_step_count
    elif method_name == "get_metrics":
      return self.metrics_buffer
    elif method_name == "generate":
      if self.responses:
        return self.responses.pop(0)
      return []
    return None


def _create_rollout_response(
    request_id: str,
    prompt_id: str,
    group_index: int = 0,
    policy_version: int = 0,
    reward: float = 1.0,
) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id,
      prompt_id=prompt_id,
      group_index=group_index,
      status="COMPLETED",
      env_reward=reward,
      policy_version=policy_version,
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=np.array([3, 4], dtype=np.int32),
              loss_mask=np.array([1, 1], dtype=np.int32),
          )
      ],
      metadata={},
  )


def _make_trajectory_group(
    prompt_id: str = "prompt_0",
    group_size: int = 2,
    reward: float = 1.0,
) -> list[datatypes.TrajectoryItem]:
  return [
      distributed_rl_engine._response_to_trajectory_item(
          _create_rollout_response(
              f"req_{prompt_id}_{idx}",
              prompt_id,
              group_index=idx,
              reward=reward,
          )
      )
      for idx in range(group_size)
  ]


def _set_mock_poll_batches(
    mock_engine: mock.MagicMock,
    *batches: Sequence[datatypes.TrajectoryItem],
) -> None:
  call_idx = 0
  batch_list = list(batches)

  async def _mock_poll(timeout_s=0.1):
    del timeout_s
    nonlocal call_idx
    if call_idx < len(batch_list):
      res = list(batch_list[call_idx])
      call_idx += 1
      return res
    await asyncio.sleep(0.01)
    return []

  mock_engine.poll_rollouts.side_effect = _mock_poll


class RLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock(
        spec=distributed_rl_engine.DistributedRLEngine
    )
    self.mock_engine.dispatch_rollouts = mock.AsyncMock()
    self.mock_engine.train_step = mock.AsyncMock(return_value="step_done")
    self.mock_engine.save_checkpoint = mock.AsyncMock(
        return_value={"checkpoint_saved": True}
    )

    async def _mock_poll(*args, **kwargs):
      del args, kwargs
      await asyncio.sleep(0.01)
      return []

    self.mock_engine.sync_weights = mock.AsyncMock(return_value=1)
    self.mock_engine.get_metrics = mock.AsyncMock(return_value=None)
    self.mock_engine.poll_rollouts = mock.AsyncMock(side_effect=_mock_poll)
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.group_size = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_turns = 1
    self.mock_algo.max_packed_len = 16
    self.mock_algo.requires_reference_kl = False

    mock_payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.full(4, 1.0, dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    self.mock_algo.create_trainer_payloads.return_value = [
        mock_payload,
        mock_payload,
    ]
    self.assembler = batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=16
    )

  def tearDown(self):
    super().tearDown()
    try:
      import jax._src.monitoring as jax_monitoring  # pyrefly: ignore[import-error]

      jax_monitoring._scalar_listeners.clear()
    except Exception:
      pass

  def _create_program(
      self,
      dataset: Any = ("prompt_0",),
      max_steps: int | None = 1,
      reward_fns: Any = None,
      **kwargs: Any,
  ) -> rl_program.StandardRLProgram:
    program = rl_program.StandardRLProgram(
        dataset=dataset,
        max_steps=max_steps,
        algo=self.mock_algo,
        reward_fns=reward_fns if reward_fns is not None else [lambda x: 1.0],
        assembler=self.assembler,
        **kwargs,
    )
    return program

  def test_dataset_exhausted_before_max_steps(self):
    async def _run():
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group(prompt_id="p0", group_size=2),
          [],
      )

      p = self._create_program(
          dataset=(
              "p0",
          ),  # Just 1 prompt. Dispatches 2 rollouts since group_size=2.
          group_size=2,
          mini_batch_size=1,
          max_steps=10,
      )

      # Since group size is 2, it dispatches 2 rollouts.
      # These 2 rollouts will form 1 group.
      # Train stage needs 1 minibatches = 1 group per step.
      # Step 0 will process 1 group.
      # Step 1 will ask for a group, but dataset is exhausted and dispatch loop finished!
      # It should cleanly break and exit run_async!

      await p.run_async(engine=self.mock_engine)
      self.assertEqual(p.step, 1)

    asyncio.run(_run())

  def test_initialization(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
        reward_fns=[lambda x: 1.0],
        assembler=self.assembler,
    )
    self.assertEqual(program.step, 0)
    self.assertEqual(program.group_size, 2)
    self.assertEqual(program.mini_batch_size, 1)
    self.assertIsNotNone(program.raw_q)
    self.assertIsNotNone(program.scored_q)

  def test_run_async_four_stages_with_long_polling(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])

      begin_steps = []
      end_steps = []

      def on_begin(step):
        begin_steps.append(step)

      def on_end(step, result):
        end_steps.append((step, result))

      program = self._create_program(
          dataset=["prompt_data_0"],
          max_steps=1,
          on_step_begin=on_begin,
          on_step_end=on_end,
      )

      await program.run_async(self.mock_engine)

      self.assertEqual(program.step, 1)
      self.assertEqual(begin_steps, [0])
      self.assertEqual(end_steps, [(0, "step_done")])
      self.mock_engine.dispatch_rollouts.assert_called_once_with(
          [{"prompt": "prompt_data_0", "prompt_id": "prompt_0"}],
          group_size=2,
          policy_version=0,
      )
      self.mock_engine.train_step.assert_called_once()
      self.mock_engine.save_checkpoint.assert_called_once_with(
          role=datatypes.Role.ACTOR,
          metadata={
              "step": 1,
              "policy_version": 0,
              "num_rollouts": 2,
              "num_microbatches": 1,
          },
      )
      self.mock_engine.sync_weights.assert_called_once_with(
          role=datatypes.Role.ACTOR
      )
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.num_rollouts, 2)
      self.assertEqual(program.last_step_result.num_microbatches, 1)
      self.assertEqual(program.last_step_result.reward_mean, 1.0)
      self.assertEqual(program.last_step_result.policy_version, 1)
      self.assertEqual(program.last_step_result.train_result, "step_done")

    asyncio.run(_run())

  def test_step_can_skip_weight_sync(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(sync_weights=False)

      await program.run_async(self.mock_engine)

      self.assertEqual(program.step, 1)
      self.mock_engine.save_checkpoint.assert_called_once()
      self.mock_engine.sync_weights.assert_not_called()
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.policy_version, 1)

    asyncio.run(_run())

  def test_checkpoint_called_before_sync_weights(self):
    async def _run():
      call_order = []

      async def mock_save_checkpoint(*args, **kwargs):
        del args, kwargs
        call_order.append("save_checkpoint")
        return {"checkpoint_saved": True}

      async def mock_sync_weights(*args, **kwargs):
        del args, kwargs
        call_order.append("sync_weights")
        return 1

      self.mock_engine.save_checkpoint.side_effect = mock_save_checkpoint
      self.mock_engine.sync_weights.side_effect = mock_sync_weights

      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(sync_weights=True)

      await program.run_async(self.mock_engine, num_steps=1)

      self.assertEqual(call_order, ["save_checkpoint", "sync_weights"])

    asyncio.run(_run())

  def test_zero_staleness_dispatches_only_one_minibatch_ahead(self):
    async def _run():
      dispatched = []

      async def mock_dispatch(prompts, **kwargs):
        dispatched.append((prompts[0], kwargs["policy_version"]))
        return [f"{prompts[0]}_{kwargs['policy_version']}"]

      self.mock_engine.dispatch_rollouts.side_effect = mock_dispatch

      program = rl_program.StandardRLProgram(
          dataset=["prompt_0", "prompt_1"],
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
          max_staleness=0,
      )
      program.engine = self.mock_engine

      program._dispatch_capacity = asyncio.Semaphore(1)
      dispatch_task = asyncio.create_task(program.rollout_dispatch_stage())

      for _ in range(50):
        if dispatched:
          break
        await asyncio.sleep(0.01)

      self.assertEqual(
          dispatched,
          [({"prompt": "prompt_0", "prompt_id": "prompt_0"}, 0)],
      )

      await asyncio.sleep(0.1)
      self.assertEqual(
          dispatched,
          [({"prompt": "prompt_0", "prompt_id": "prompt_0"}, 0)],
      )

      program.policy_version = 1
      program._dispatch_capacity.release()
      await asyncio.wait_for(dispatch_task, timeout=1.0)
      self.assertEqual(
          dispatched,
          [
              ({"prompt": "prompt_0", "prompt_id": "prompt_0"}, 0),
              ({"prompt": "prompt_1", "prompt_id": "prompt_1"}, 1),
          ],
      )

    asyncio.run(_run())

  def test_train_stage_updates_only_on_last_microbatch(self):
    class TwoMicrobatchAssembler:

      def pack(self, items):
        del items
        return ["microbatch_0", "microbatch_1"]

    async def _run():
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=TwoMicrobatchAssembler(),
          sync_weights=False,
      )
      program.engine = self.mock_engine

      for group_index in range(2):
        item = datatypes.TrajectoryItem(
            group_index=group_index,
            prompt_id="prompt_0",
            start_step=0,
            traj=datatypes.Trajectory(reward=1.0),
        )
        item.payload = self.mock_algo.create_trainer_payloads.return_value[
            group_index
        ]
        await program.scored_q.put(item)

      program._dispatch_capacity = asyncio.Semaphore(1)
      await program.train_stage()

      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      self.assertEqual(
          [
              call.kwargs["apply_optimizer"]
              for call in self.mock_engine.train_step.call_args_list
          ],
          [False, True],
      )
      self.mock_engine.sync_weights.assert_not_called()

    asyncio.run(_run())

  def test_stage_exception_aborts_queue_and_propagates(self):
    class FailingProgram(rl_program.StandardRLProgram):

      async def rollout_dispatch_stage(self, train_dataset=None):
        del train_dataset
        raise RuntimeError("Rollout worker cluster down!")

    async def _run():
      prog = FailingProgram(
          dataset=["prompt"],
          algo=self.mock_algo,
          assembler=self.assembler,
      )
      with self.assertRaises(RuntimeError) as cm:
        await prog.run_async(self.mock_engine)
      self.assertIn("Rollout worker cluster down!", str(cm.exception))

    asyncio.run(_run())

  def test_run_synchronous_entry_point(self):
    _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
    program = self._create_program(
        reward_fns=[lambda x: 2.0], dataset=["sync_prompt"]
    )

    program.run(self.mock_engine)

    self.assertEqual(program.step, 1)
    self.assertIsNotNone(program.last_step_result)
    self.assertEqual(program.last_step_result.num_rollouts, 2)
    self.assertEqual(program.last_step_result.reward_mean, 2.0)

  def test_run_with_existing_running_loop(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(dataset=["async_prompt"])

      program.run(self.mock_engine)
      self.assertIsNotNone(program._bg_task)
      await program._bg_task
      self.assertEqual(program.step, 1)

    asyncio.run(_run())

  def test_missing_dataset_raises_value_error(self):
    async def _run():
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          assembler=self.assembler,
      )
      program.engine = self.mock_engine
      with self.assertRaises(ValueError) as cm:
        await program.run_async(self.mock_engine)
      self.assertIn("requires a dataset", str(cm.exception))

    asyncio.run(_run())

  def test_prompt_dictionary_id_and_group_extraction(self):
    async def _run():
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group(prompt_id="custom_p0"),
      )
      dict_item = {
          "prompt_id": "custom_p0",
          "data": "test",
      }
      program = self._create_program(dataset=[dict_item])

      await program.run_async(self.mock_engine)

      self.mock_engine.dispatch_rollouts.assert_called_once_with(
          [dict_item],
          group_size=2,
          policy_version=0,
      )

    asyncio.run(_run())

  def test_prompt_id_and_group_index_propagation_end_to_end(self):
    """Verifies that prompt_id and group_index are automatically built and propagated:

    1. Raw dataset string prompt (without manual prompt_id) -> RLProgram builds prompt_0
    2. Engine dispatch -> assigns group_index (0, 1) and creates deterministic request_ids
    3. Rollout worker -> creates RolloutResponse inheriting prompt_id and group_index
    4. Queue manager -> groups by prompt_0, delivers complete group
    5. Reward function -> receives items with prompt_id='prompt_0' and group_index=(0, 1)
    6. Batch assembly -> receives reconstructed items with prompt_id and group_index preserved
    7. Trainer step -> executed with batch
    """
    async def _run():
      mock_rollout = _MockWorkerHandle(role=datatypes.Role.ROLLOUT)
      mock_actor = _MockWorkerHandle(role=datatypes.Role.ACTOR)
      mock_coordinator = mock.MagicMock()
      mock_coordinator.sync = mock.AsyncMock(
          return_value=mock.MagicMock(policy_version=1)
      )
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[mock_rollout],
          trainer_workers={datatypes.Role.ACTOR: mock_actor},
          weight_sync_coordinator=mock_coordinator,
      )

      # 1. Raw prompt dataset (prompt_id not preconfigured; built automatically by program)
      dataset = ["What is 2+2?"]

      # 2. Track items observed in reward_fn
      observed_in_reward = []
      def tracking_reward_fn(it: datatypes.TrajectoryItem) -> float:
        observed_in_reward.append({
            "prompt_id": it.prompt_id,
            "group_index": it.group_index,
        })
        return 1.0

      # 3. Track items passed to algo.create_trainer_payloads
      passed_to_algo = []
      def tracking_create_payloads(step_items, **kwargs):
        del kwargs
        for it in step_items:
          passed_to_algo.append({
              "prompt_id": it.prompt_id,
              "group_index": it.group_index,
          })
        mock_p = datatypes.RLTrainerPayload(
            token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
            token_mask=np.array([0, 0, 1, 1], dtype=np.float32),
            loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
            advantages=np.full(4, 1.0, dtype=np.float32),
            action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        )
        return [mock_p, mock_p]

      self.mock_algo.create_trainer_payloads = tracking_create_payloads

      program = self._create_program(
          dataset=dataset,
          reward_fns=[tracking_reward_fn],
          group_size=2,
          mini_batch_size=1,
          max_steps=1,
      )

      await program.run_async(engine)

      # 4. Verify RolloutRequests dispatched to worker
      self.assertEqual(len(mock_rollout.dispatched_requests), 2)
      req_0 = mock_rollout.dispatched_requests[0][3]["requests"][0]
      req_1 = mock_rollout.dispatched_requests[1][3]["requests"][0]
      self.assertEqual(req_0.prompt_id, "prompt_0")
      self.assertEqual(req_0.group_index, 0)
      self.assertEqual(req_0.request_id, "req_prompt_0_g0_v0")
      self.assertEqual(req_1.prompt_id, "prompt_0")
      self.assertEqual(req_1.group_index, 1)
      self.assertEqual(req_1.request_id, "req_prompt_0_g1_v0")

      # 5. Verify items observed in reward_fn
      self.assertEqual(
          observed_in_reward,
          [
              {"prompt_id": "prompt_0", "group_index": 0},
              {"prompt_id": "prompt_0", "group_index": 1},
          ],
      )

      # 6. Verify items passed to algo.create_trainer_payloads
      self.assertEqual(
          passed_to_algo,
          [
              {"prompt_id": "prompt_0", "group_index": 0},
              {"prompt_id": "prompt_0", "group_index": 1},
          ],
      )

      # 7. Verify trainer step executed
      self.assertEqual(mock_actor.train_step_count, 1)

    asyncio.run(_run())

  def test_multi_group_mini_batch_gradient_accumulation(self):
    async def _run():
      self.mock_algo.mini_batch_size = 2
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group("prompt_0"),
          _make_trajectory_group("prompt_1"),
      )
      program = self._create_program(dataset=["p0", "p1"])

      await program.run_async(self.mock_engine)

      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      calls = self.mock_engine.train_step.call_args_list
      # First group: accumulate_gradients=True, apply_optimizer=False
      self.assertTrue(calls[0].kwargs["accumulate_gradients"])
      self.assertFalse(calls[0].kwargs["apply_optimizer"])
      # Second group: accumulate_gradients=True, apply_optimizer=True
      self.assertTrue(calls[1].kwargs["accumulate_gradients"])
      self.assertTrue(calls[1].kwargs["apply_optimizer"])
      self.assertEqual(program.last_step_result.num_rollouts, 4)
      self.assertEqual(program.last_step_result.num_microbatches, 2)

    asyncio.run(_run())

  def test_reference_kl_logprobs_scoring_in_train_stage(self):
    async def _run():
      self.mock_algo.requires_reference_kl = True
      mock_train_example = rl_common.TrainExample(
          prompt_ids=np.array([[1, 2]], dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.array([[3, 4]], dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
      self.assembler.pack = mock.MagicMock(return_value=[mock_train_example])
      self.mock_engine.per_token_logps = mock.AsyncMock(
          return_value=np.array([[-0.1, -0.2]], dtype=np.float32)
      )

      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(dataset=["prompt_0"])

      await program.run_async(self.mock_engine)

      self.mock_engine.per_token_logps.assert_called_once_with(
          datatypes.Role.REFERENCE, items=mock_train_example
      )
      self.assertEqual(program.step, 1)

    asyncio.run(_run())

  def test_reference_kl_raises_type_error_for_invalid_microbatch(self):
    async def _run():
      self.mock_algo.requires_reference_kl = True
      # Returning a raw dict instead of TrainExample
      self.assembler.pack = mock.MagicMock(return_value=[{"raw": "batch"}])
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(dataset=["prompt_0"])

      with self.assertRaises(TypeError) as cm:
        await program.run_async(self.mock_engine)
      self.assertIn("Reference KL requires an assembler", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_handles_early_dispatch_completion(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program()
      await program.run_async(self.mock_engine)
      self.assertEqual(program.step, 1)

    asyncio.run(_run())

  def test_run_async_propagates_train_stage_exception(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      self.mock_engine.train_step.side_effect = RuntimeError(
          "Training worker OOM"
      )
      program = self._create_program(max_steps=1)

      with self.assertRaises(RuntimeError) as cm:
        await program.run_async(self.mock_engine)
      self.assertIn("Training worker OOM", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_propagates_save_checkpoint_exception_and_skips_weight_sync(
      self,
  ):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      self.mock_engine.save_checkpoint.side_effect = RuntimeError(
          "Checkpoint save failed: disk full"
      )
      self.mock_engine.sync_weights = mock.AsyncMock()

      end_steps = []

      def on_end(step, result):
        end_steps.append((step, result))

      program = self._create_program(
          max_steps=1,
          sync_weights=True,
          on_step_end=on_end,
      )

      with self.assertRaises(RuntimeError) as cm:
        await program.run_async(self.mock_engine)

      self.assertIn("Checkpoint save failed: disk full", str(cm.exception))
      self.mock_engine.save_checkpoint.assert_called_once()
      self.mock_engine.sync_weights.assert_not_called()
      self.assertEqual(program.step, 0)
      self.assertIsNone(program.last_step_result)
      self.assertEmpty(end_steps)

    asyncio.run(_run())

  def test_run_async_save_checkpoint_io_error(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      self.mock_engine.save_checkpoint.side_effect = IOError(
          "Storage quota exceeded"
      )
      self.mock_engine.sync_weights = mock.AsyncMock()

      program = self._create_program(max_steps=1, sync_weights=True)

      with self.assertRaises(IOError) as cm:
        await program.run_async(self.mock_engine)

      self.assertIn("Storage quota exceeded", str(cm.exception))
      self.mock_engine.sync_weights.assert_not_called()
      self.assertEqual(program.step, 0)

    asyncio.run(_run())


  def test_run_async_propagates_critique_stage_exception(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())

      def failing_reward_fn(_):
        raise ValueError("Reward model computation failed")

      program = self._create_program(reward_fns=[failing_reward_fn])

      with self.assertRaises(ValueError) as cm:
        await program.run_async(self.mock_engine)
      self.assertIn("Reward model computation failed", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_cancels_background_stages_on_external_cancellation(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine)  # Yields empty and sleeps
      program = self._create_program()

      task = asyncio.create_task(program.run_async(self.mock_engine))
      await asyncio.sleep(0.02)
      task.cancel()

      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(_run())

  def test_metrics_logging_full_pipeline(self):
    async def _run():
      mock_buffer = exp_metrics.MetricsBuffer(
          id=0,
          scalar_metrics={
              "loss": 0.5,
              "learning_rate": 1e-4,
              "grad_norm": 0.25,
          },
          weighted_metrics={
              "kl": sft_utils.WeightedMetric(
                  unreduced_sum=np.array(0.04), denominator=np.array(2.0)
              ),
          },
          mode="train",
      )
      rollout_worker = _MockWorkerHandle(role="rollout")
      rollout_worker.responses = [
          [
              _create_rollout_response(
                  "req_0",
                  "prompt_data_0",
                  group_index=0,
                  reward=2.5,
              ),
              _create_rollout_response(
                  "req_1",
                  "prompt_data_0",
                  group_index=1,
                  reward=2.5,
              ),
          ],
      ]
      trainer_worker = _MockWorkerHandle(role="trainer")
      trainer_worker.metrics_buffer = mock_buffer

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[rollout_worker],
          trainer_workers={datatypes.Role.ACTOR: trainer_worker},
      )

      program = self._create_program(
          dataset=["prompt_data_0"], reward_fns=[], sync_weights=False
      )
      await program.run_async(engine, max_steps=1)

      logger = program.metrics_logger
      self.assertIsNotNone(logger)

      # 1. Trainer Metrics (retrieved from TrainerWorker.get_metrics)
      self.assertTrue(logger.metric_exists("", "trainer/loss", "train"))
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/loss", "train"), 0.5
      )
      self.assertTrue(logger.metric_exists("", "trainer/perplexity", "train"))
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/perplexity", "train"),
          float(np.exp(0.5)),
          places=5,
      )
      self.assertTrue(
          logger.metric_exists("", "trainer/learning_rate", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/learning_rate", "train"), 1e-4
      )
      self.assertTrue(logger.metric_exists("", "trainer/grad_norm", "train"))
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/grad_norm", "train"), 0.25
      )
      self.assertTrue(logger.metric_exists("", "trainer/kl", "train"))
      self.assertAlmostEqual(logger.get_metric("", "trainer/kl", "train"), 0.02)

      # 2. Reward Metrics
      self.assertTrue(logger.metric_exists("", "rewards/mean", "train"))
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/mean", "train"), 2.5
      )
      self.assertTrue(logger.metric_exists("", "rewards/std", "train"))
      self.assertAlmostEqual(logger.get_metric("", "rewards/std", "train"), 0.0)
      self.assertTrue(logger.metric_exists("", "rewards/min", "train"))
      self.assertAlmostEqual(logger.get_metric("", "rewards/min", "train"), 2.5)
      self.assertTrue(logger.metric_exists("", "rewards/max", "train"))
      self.assertAlmostEqual(logger.get_metric("", "rewards/max", "train"), 2.5)
      self.assertTrue(logger.metric_exists("", "rewards/sum", "train"))
      self.assertAlmostEqual(logger.get_metric("", "rewards/sum", "train"), 5.0)

      # 3. Rollout Metrics (collected from RolloutWorker responses)
      self.assertTrue(
          logger.metric_exists("", "rollout/prompt_length_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/prompt_length_mean", "train"), 2.0
      )
      self.assertTrue(
          logger.metric_exists("", "rollout/completion_length_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/completion_length_mean", "train"), 2.0
      )
      self.assertTrue(
          logger.metric_exists("", "rollout/total_tokens_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/total_tokens_mean", "train"), 4.0
      )

      self.assertTrue(logger.metric_exists("", "rollout/success_rate", "train"))
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/success_rate", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "rollout/staleness_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_mean", "train"), 0.0
      )
      self.assertTrue(
          logger.metric_exists("", "rollout/staleness_max", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_max", "train"), 0.0
      )
      self.assertTrue(
          logger.metric_exists("", "rollout/staleness_min", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_min", "train"), 0.0
      )

      # 4. Orchestrator Metrics
      self.assertTrue(
          logger.metric_exists("", "orchestrator/policy_version", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "orchestrator/policy_version", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "orchestrator/num_rollouts", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "orchestrator/num_rollouts", "train"), 2.0
      )
      self.assertTrue(
          logger.metric_exists("", "orchestrator/num_microbatches", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "orchestrator/num_microbatches", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "orchestrator/step_time_sec", "train")
      )

    asyncio.run(_run())

  def test_engine_get_metrics_across_roles(self):
    async def _run():
      actor_worker = _MockWorkerHandle(role="actor")
      actor_worker.metrics_buffer = exp_metrics.MetricsBuffer(
          id=1, scalar_metrics={"loss": 0.4}
      )
      critic_worker = _MockWorkerHandle(role="critic")
      critic_worker.metrics_buffer = exp_metrics.MetricsBuffer(
          id=1, scalar_metrics={"vf_loss": 0.15}
      )
      ref_worker = _MockWorkerHandle(role="reference")
      ref_worker.metrics_buffer = {"throughput": 120.0}
      rollout_worker_1 = _MockWorkerHandle(role="rollout")
      rollout_worker_1.metrics_buffer = {"rollouts_completed": 10}
      rollout_worker_2 = _MockWorkerHandle(role="rollout")
      rollout_worker_2.metrics_buffer = {"rollouts_completed": 12}

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[rollout_worker_1, rollout_worker_2],
          trainer_workers={
              datatypes.Role.ACTOR: actor_worker,
              datatypes.Role.CRITIC: critic_worker,
          },
          inference_workers={
              datatypes.Role.REFERENCE: ref_worker,
          },
      )

      # 1. Trainer Role (Actor)
      actor_metrics = await engine.get_metrics(role=datatypes.Role.ACTOR)
      self.assertEqual(actor_metrics.scalar_metrics["loss"], 0.4)

      # 2. Trainer Role (Critic)
      critic_metrics = await engine.get_metrics(role=datatypes.Role.CRITIC)
      self.assertEqual(critic_metrics.scalar_metrics["vf_loss"], 0.15)

      # 3. Inference Role (Reference)
      ref_metrics = await engine.get_metrics(role=datatypes.Role.REFERENCE)
      self.assertEqual(ref_metrics["throughput"], 120.0)

      # 4. Rollout Role (Aggregated over all rollout workers)
      rollout_metrics = await engine.get_metrics(role=datatypes.Role.ROLLOUT)
      self.assertLen(rollout_metrics, 2)
      self.assertEqual(
          rollout_metrics,
          [{"rollouts_completed": 10}, {"rollouts_completed": 12}],
      )

    asyncio.run(_run())

  def test_metrics_logging_with_prefix_and_eval_mode(self):
    async def _run():
      _set_mock_poll_batches(
          self.mock_engine, _make_trajectory_group(reward=3.0), []
      )
      self.mock_engine.train_step.return_value = {
          "updated": True,
      }
      self.mock_engine.get_metrics.return_value = {"loss": 0.2}

      program = self._create_program(
          dataset=["prompt_0"],
          reward_fns=[],
          metrics_prefix="actor_mesh",
          mode=metrics_logger_lib.Mode.EVAL,
      )
      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertTrue(
          logger.metric_exists("actor_mesh", "rewards/mean", "eval")
      )
      self.assertAlmostEqual(
          logger.get_metric("actor_mesh", "rewards/mean", "eval"), 3.0
      )
      self.assertTrue(
          logger.metric_exists("actor_mesh", "trainer/loss", "eval")
      )
      self.assertAlmostEqual(
          logger.get_metric("actor_mesh", "trainer/loss", "eval"), 0.2
      )

    asyncio.run(_run())

  def test_program_close_flushes_metrics_logger(self):
    program = self._create_program()
    internal_logger = program.metrics_logger
    internal_logger.close = mock.MagicMock()
    program.close()
    internal_logger.close.assert_called_once()

  def test_distributed_engine_train_step_and_get_metrics(self):
    async def _run():
      mock_worker = mock.MagicMock()
      mock_worker.asubmit.side_effect = lambda method, *args, **kwargs: {
          "fwd_bwd": "fwd_bwd_done",
          "update": 1,
          "get_metrics": exp_metrics.MetricsBuffer(
              id=1, scalar_metrics={"loss": 0.1}
          ),
      }[method]

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[],
          trainer_workers={datatypes.Role.ACTOR: mock_worker},
      )
      payload = datatypes.RLTrainerPayload(
          token_ids=np.array([1, 2], dtype=np.int32),
          token_mask=np.array([1, 1], dtype=np.float32),
          loss_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
          action_mask=np.array([1, 1], dtype=np.float32),
      )
      res = await engine.train_step(
          payload, role=datatypes.Role.ACTOR, apply_optimizer=True
      )
      self.assertTrue(res["updated"])
      self.assertNotIn("metrics", res)

      metrics = await engine.get_metrics(role=datatypes.Role.ACTOR)
      self.assertEqual(metrics.scalar_metrics["loss"], 0.1)

    asyncio.run(_run())

  def test_staleness_computation_with_nonzero_policy_version(self):
    async def _run():
      resp_v3_0 = _create_rollout_response(
          "req_0", "prompt_0", group_index=0, policy_version=3
      )
      resp_v3_1 = _create_rollout_response(
          "req_1", "prompt_0", group_index=1, policy_version=3
      )
      _set_mock_poll_batches(
          self.mock_engine,
          [
              distributed_rl_engine._response_to_trajectory_item(resp_v3_0),
              distributed_rl_engine._response_to_trajectory_item(resp_v3_1),
          ],
          [],
      )
      program = self._create_program(
          dataset=["prompt_0"], reward_fns=[], max_staleness=2
      )
      program.policy_version = 5

      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertTrue(
          logger.metric_exists("", "rollout/staleness_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_mean", "train"), 2.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_max", "train"), 2.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/staleness_min", "train"), 2.0
      )

    asyncio.run(_run())

  def test_token_mask_and_loss_mask_fallback(self):
    async def _run():
      payload_0 = datatypes.RLTrainerPayload(
          token_ids=np.arange(10, dtype=np.int32),
          token_mask=np.ones(10, dtype=np.float32),
          loss_mask=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
          advantages=np.ones(10, dtype=np.float32),
          action_mask=np.ones(10, dtype=np.float32),
      )
      payload_1 = datatypes.RLTrainerPayload(
          token_ids=np.arange(10, dtype=np.int32),
          token_mask=np.ones(10, dtype=np.float32),
          loss_mask=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
          advantages=np.ones(10, dtype=np.float32),
          action_mask=np.ones(10, dtype=np.float32),
      )
      traj_item_0 = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj=datatypes.Trajectory(reward=1.0),
      )
      traj_item_0.payload = payload_0
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj=datatypes.Trajectory(reward=1.0),
      )
      traj_item_1.payload = payload_1
      self.mock_algo.create_trainer_payloads.return_value = [
          payload_0,
          payload_1,
      ]

      _set_mock_poll_batches(self.mock_engine, [traj_item_0, traj_item_1], [])
      program = self._create_program(dataset=["prompt_0"], reward_fns=[])

      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/prompt_length_mean", "train"), 4.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/completion_length_mean", "train"), 6.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/total_tokens_mean", "train"), 10.0
      )

    asyncio.run(_run())

  def test_rollouts_without_status_omits_success_rate(self):
    async def _run():
      traj_item_0 = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj=datatypes.Trajectory(reward=1.0, status=None),
      )
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj=datatypes.Trajectory(reward=1.0, status=None),
      )
      _set_mock_poll_batches(self.mock_engine, [traj_item_0, traj_item_1], [])
      program = self._create_program(dataset=["prompt_0"], reward_fns=[])

      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertFalse(
          logger.metric_exists("", "rollout/success_rate", "train")
      )

    asyncio.run(_run())

  def test_nested_dict_metrics_buffer_ingestion(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])
      self.mock_engine.train_step.return_value = {
          "updated": True,
      }
      self.mock_engine.get_metrics.return_value = {
          "scalar_metrics": {"loss": 0.35, "learning_rate": 5e-5},
          "weighted_metrics": {"kl": 0.01},
      }
      program = self._create_program(dataset=["prompt_0"], reward_fns=[])

      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/loss", "train"), 0.35
      )
      self.assertAlmostEqual(
          logger.get_metric("", "trainer/learning_rate", "train"), 5e-5
      )
      self.assertAlmostEqual(logger.get_metric("", "trainer/kl", "train"), 0.01)

    asyncio.run(_run())

  def test_engine_get_metrics_empty_workers_raises_value_error(self):
    async def _run():
      actor_worker = _MockWorkerHandle(role="actor")
      actor_worker.metrics_buffer = exp_metrics.MetricsBuffer(
          id=1, scalar_metrics={"loss": 0.25}
      )

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[],
          trainer_workers={datatypes.Role.ACTOR: actor_worker},
      )

      # Querying empty rollout workers raises ValueError
      with self.assertRaises(ValueError) as ctx:
        await engine.get_metrics(role=datatypes.Role.ROLLOUT)
      self.assertIn("No rollout workers registered", str(ctx.exception))

      # Querying unregistered role raises ValueError
      with self.assertRaises(ValueError) as ctx:
        await engine.get_metrics(role=datatypes.Role.CRITIC)
      self.assertIn("No worker registered for role", str(ctx.exception))

    asyncio.run(_run())

  def test_rollouts_with_steps_logs_turns_mean(self):
    async def _run():
      mock_step = datatypes.Step()
      traj_item_0 = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj=datatypes.Trajectory(reward=1.0, steps=[mock_step, mock_step]),
      )
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj=datatypes.Trajectory(
              reward=1.0, steps=[mock_step, mock_step, mock_step, mock_step]
          ),
      )
      _set_mock_poll_batches(self.mock_engine, [traj_item_0, traj_item_1], [])
      program = self._create_program(dataset=["prompt_0"], reward_fns=[])

      await program.run_async(self.mock_engine)

      logger = program.metrics_logger
      self.assertTrue(
          logger.metric_exists("", "rollout/num_turns_mean", "train")
      )
      # (2 + 4) / 2 = 3.0
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/num_turns_mean", "train"), 3.0
      )

    asyncio.run(_run())

  def test_extract_scalar_compute_failure_returns_none(self):
    class FailingMetric:

      def compute(self):
        raise RuntimeError("Metric compute failed")

    self.assertIsNone(rl_program._extract_scalar(FailingMetric()))

  def test_generate_mock_dashboard_events(self):
    async def _run():
      log_dir = "/tmp/tunix_rl_dashboard_demo"
      options = metrics_logger_lib.MetricsLoggerOptions(
          log_dir=log_dir, flush_every_n_steps=1
      )

      batches = []
      for step_idx in range(1, 21):
        r1 = _create_rollout_response(
            f"req_{step_idx}_0",
            "p0",
            group_index=0,
            reward=float(1.5 + 2.5 * (1.0 - np.exp(-step_idx / 6.0))),
            policy_version=max(0, step_idx - 1),
        )
        r2 = _create_rollout_response(
            f"req_{step_idx}_1",
            "p0",
            group_index=1,
            reward=float(1.5 + 2.5 * (1.0 - np.exp(-step_idx / 6.0))),
            policy_version=max(0, step_idx - 1),
        )
        batches.append([
            distributed_rl_engine._response_to_trajectory_item(r1),
            distributed_rl_engine._response_to_trajectory_item(r2),
        ])
      _set_mock_poll_batches(self.mock_engine, *batches)

      def _mock_train_step(payload, role=None, apply_optimizer=True, **kwargs):
        del payload, role, kwargs
        return {
            "updated": apply_optimizer,
        }

      def _mock_get_metrics(role=datatypes.Role.ACTOR, **kwargs):
        del role, kwargs
        step_num = self.mock_engine.train_step.call_count
        loss = float(0.3 + 1.8 * np.exp(-step_num / 5.0))
        lr = float(1e-4 * max(0.1, 1.0 - (step_num / 20.0)))
        grad_norm = float(0.2 + 0.8 * np.exp(-step_num / 8.0))
        return exp_metrics.MetricsBuffer(
            id=step_num,
            scalar_metrics={
                "loss": loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
            },
            weighted_metrics={"kl": float(0.01 + 0.02 * (step_num / 20.0))},
        )

      self.mock_engine.train_step.side_effect = _mock_train_step
      self.mock_engine.get_metrics.side_effect = _mock_get_metrics
      program = self._create_program(
          dataset=["p0"] * 20,
          reward_fns=[],
          max_steps=20,
          metrics_logging_options=options,
          sync_weights=False,
      )
      await program.run_async(self.mock_engine)
      program.close()
      self.assertEqual(program.step, 20)

    asyncio.run(_run())

  def test_program_logs_to_wandb_backend(self):
    async def _run():
      mock_wandb = mock.Mock()
      mock_wandb.run = mock.Mock()
      mock_wandb.run.url = "https://wandb.ai/my-org/my-project/runs/mock123"

      real_import = builtins.__import__

      def _mock_import(name, *args, **kwargs):
        if name == "wandb":
          return mock_wandb
        return real_import(name, *args, **kwargs)

      with mock.patch("jax.process_index", return_value=0), mock.patch(
          "builtins.__import__", side_effect=_mock_import
      ):
        wandb_backend = metrax_logging.WandbBackend(
            project="test-rl-project", name="rl-run-1"
        )
        options = metrics_logger_lib.MetricsLoggerOptions(
            log_dir="/tmp/test_wandb_dir",
            backend_kwargs={"custom_backend": [lambda: wandb_backend]},
        )

        r1 = _create_rollout_response(
            "req_0", "p0", group_index=0, reward=2.5, policy_version=0
        )
        r2 = _create_rollout_response(
            "req_1", "p0", group_index=1, reward=1.5, policy_version=0
        )
        _set_mock_poll_batches(
            self.mock_engine,
            [
                distributed_rl_engine._response_to_trajectory_item(r1),
                distributed_rl_engine._response_to_trajectory_item(r2),
            ],
        )

        self.mock_engine.train_step.return_value = {
            "updated": True,
        }
        self.mock_engine.get_metrics.return_value = exp_metrics.MetricsBuffer(
            id=1,
            scalar_metrics={"loss": 0.42, "learning_rate": 1e-4},
        )

        program = self._create_program(
            dataset=["p0"],
            reward_fns=[],
            metrics_logging_options=options,
            sync_weights=False,
        )
        await program.run_async(self.mock_engine)
        program.close()

        # Verify wandb initialization
        mock_wandb.init.assert_called_once_with(
            project="test-rl-project", name="rl-run-1", anonymous="allow"
        )

        # Verify wandb logged the RL scalar metrics
        logged_dicts = [call.args[0] for call in mock_wandb.log.call_args_list]
        logged_keys = {k for d in logged_dicts for k in d.keys()}

        self.assertIn("train/trainer/loss", logged_keys)
        self.assertIn("train/trainer/learning_rate", logged_keys)
        self.assertIn("train/rewards/mean", logged_keys)
        self.assertIn("train/rollout/prompt_length_mean", logged_keys)
        self.assertIn("train/rollout/staleness_mean", logged_keys)
        self.assertIn("train/orchestrator/policy_version", logged_keys)

        # Verify wandb.finish was called on close
        mock_wandb.finish.assert_called_once()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
