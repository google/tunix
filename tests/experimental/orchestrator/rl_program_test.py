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
import dataclasses
import types
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
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.worker import remote_execution
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
  traj_item = datatypes.TrajectoryItem(
      prompt_id=prompt_id,
      group_index=group_index,
      start_step=0,
      traj={
          "trajectory_reward": reward,
          "status": datatypes.TrajectoryStatus.SUCCEEDED,
      },
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      completion_tokens=np.array([3, 4], dtype=np.int32),
      action_mask=np.array([1, 1], dtype=np.float32),
      policy_version=policy_version,
      metadata={"prompt_id": prompt_id, "group_index": group_index},
  )
  return datatypes.RolloutResponse(
      request_id=request_id,
      status="COMPLETED",
      payload=traj_item,
      metadata={},
  )


def _make_trajectory_group(
    prompt_id: str = "prompt_0",
    num_generations: int = 2,
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
      for idx in range(num_generations)
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
    self.mock_engine.restore_checkpoint = mock.AsyncMock(
        return_value={"step": 0}
    )
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=0)

    async def _mock_poll(*args, **kwargs):
      del args, kwargs
      await asyncio.sleep(0.01)
      return []

    async def _mock_sync_weights(*args, policy_version=None, **kwargs):
      del args, kwargs
      return 1 if policy_version is None else policy_version

    self.mock_engine.sync_weights = mock.AsyncMock(
        side_effect=_mock_sync_weights
    )
    self.mock_engine.prepare_rollout_policy = mock.AsyncMock(return_value=0)
    self.mock_engine.sync_weights = mock.AsyncMock(return_value=1)
    self.mock_engine.get_metrics = mock.AsyncMock(return_value=None)
    self.mock_engine.poll_rollouts = mock.AsyncMock(side_effect=_mock_poll)
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.num_generations = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_turns = 1
    self.mock_algo.max_packed_len = 16
    self.mock_algo.max_response_length = 1024
    self.mock_algo.requires_reference_kl = False
    self.mock_algo.algo_config = types.SimpleNamespace(
        temperature=None,
        use_rollout_logps=True,
    )

    mock_payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.array([1, 1], dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.array([1, 1], dtype=np.float32),
        advantages=np.array([1.0, 1.0], dtype=np.float32),
    )
    self.mock_algo.create_trainer_payloads.return_value = [
        mock_payload,
        mock_payload,
    ]
    self.assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=2,
        mini_batch_size=4,
        max_packed_len=16,
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
      assembler: Any = None,
      **kwargs: Any,
  ) -> rl_program.StandardRLProgram:
    program = rl_program.StandardRLProgram(
        dataset=dataset,
        max_steps=max_steps,
        algo=self.mock_algo,
        reward_fns=reward_fns if reward_fns is not None else [lambda *_: 1.0],
        assembler=assembler if assembler is not None else self.assembler,
        **kwargs,
    )
    # Hold the dispatch window open. Most tests run `rollout_dispatch_stage`
    # with no trainer to advance `_next_batch`, so the real gate would park the
    # dispatcher `max_staleness` batches in and never let it reach the end of
    # the dataset. The window itself is covered by the tests named for it,
    # which build their programs directly.
    program._wait_for_dispatch_window = mock.AsyncMock()
    return program

  def test_dataset_exhausted_before_max_steps(self):
    async def _run():
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group(prompt_id="p0", num_generations=2),
          [],
      )

      p = self._create_program(
          dataset=(
              "p0",
          ),  # Just 1 prompt. Dispatches 2 rollouts since num_generations=2.
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
        reward_fns=[lambda *_: 1.0],
        assembler=self.assembler,
    )
    self.assertEqual(program.step, 0)
    self.assertEqual(program.num_generations, 2)
    self.assertEqual(program.mini_batch_size, 1)
    self.assertEqual(program.full_batch_size, 1)
    self.assertIsNotNone(program.raw_q)
    self.assertIsNotNone(program.scored_q)

  def test_default_assembler_inherits_algo_train_micro_batch_size(self):
    self.mock_algo.train_micro_batch_size = 2
    program = rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
    )
    self.assertIsInstance(
        program.assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(program.assembler.batch_size, 2)

  def test_program_inherits_num_generations_and_mini_batch_size_from_algo(self):
    self.mock_algo.num_generations = 5
    self.mock_algo.mini_batch_size = 3
    program = rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
    )
    self.assertEqual(program.num_generations, 5)
    self.assertEqual(program.mini_batch_size, 3)

  def test_unexpected_num_generations_argument_raises_type_error(self):
    with self.assertRaises(TypeError):
      rl_program.StandardRLProgram(  # pyrefly: ignore[unexpected-keyword-arg]
          dataset=["prompt_1"],
          algo=self.mock_algo,
          num_generations=4,
      )

  def test_unexpected_mini_batch_size_argument_raises_type_error(self):
    with self.assertRaises(TypeError):
      rl_program.StandardRLProgram(  # pyrefly: ignore[unexpected-keyword-arg]
          dataset=["prompt_1"],
          algo=self.mock_algo,
          mini_batch_size=3,
      )

  def test_program_use_rollout_logps_matching(self):
    self.mock_algo.algo_config = types.SimpleNamespace(
        temperature=0.7,
        use_rollout_logps=True,
    )
    program = rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
        generation_args=datatypes.GenerationArgs(
            temperature=0.7,
            return_logprobs=True,
        ),
    )
    self.assertTrue(program.generation_args.return_logprobs)
    self.assertTrue(self.mock_algo.algo_config.use_rollout_logps)

  def test_program_use_rollout_logps_missing_in_generation_args_inherits_from_algo(
      self,
  ):
    self.mock_algo.algo_config = types.SimpleNamespace(
        temperature=0.7,
        use_rollout_logps=True,
    )
    program = rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
    )
    self.assertTrue(program.generation_args.return_logprobs)
    self.assertTrue(self.mock_algo.algo_config.use_rollout_logps)

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
      self.mock_engine.prepare_rollout_policy.assert_called_once_with(
          role=datatypes.Role.ACTOR,
          sync_weights=True,
          policy_version=0,
      )
      self.mock_engine.dispatch_rollouts.assert_called_once_with(
          [{
              "prompt": "prompt_data_0",
              "prompt_id": "prompt_0",
              "max_response_length": 1024,
              "metadata": {
                  "prompt_idx": 0,
                  "batch_idx": 0,
                  "intra_batch_idx": 0,
              },
          }],
          num_generations=2,
          policy_version=0,
          exact_token_continuity=True,
          generation_args=datatypes.GenerationArgs(
              return_logprobs=True,
          ),
      )
      self.mock_engine.train_step.assert_called_once()
      self.mock_engine.save_checkpoint.assert_called_once_with(
          role=datatypes.Role.ACTOR,
          metadata={
              "step": 1,
              "global_step": 1,
              "next_batch_idx": 1,
              "policy_version": 1,
              "num_rollouts": 2,
              "num_microbatches": 1,
          },
          overwrite=False,
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
      self.mock_engine.prepare_rollout_policy.assert_not_called()
      self.mock_engine.sync_weights.assert_not_called()
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.policy_version, 0)

    asyncio.run(_run())

  def test_policy_version_incremented_after_weight_sync(self):
    async def _run():
      self.mock_engine.sync_weights = mock.AsyncMock(return_value=None)
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group("prompt_0"),
          _make_trajectory_group("prompt_1"),
      )
      program = self._create_program(
          dataset=["prompt_data_0", "prompt_data_1"],
          max_steps=2,
          sync_weights=True,
      )

      self.assertEqual(program.policy_version, 0)
      await program.run_async(self.mock_engine)

      self.assertEqual(self.mock_engine.sync_weights.call_count, 2)
      self.assertEqual(program.policy_version, 2)
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.policy_version, 2)

    asyncio.run(_run())

  def test_policy_version_updated_with_explicit_version_after_weight_sync(self):
    async def _run():
      self.mock_engine.sync_weights = mock.AsyncMock(return_value=5)
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(sync_weights=True)

      self.assertEqual(program.policy_version, 0)
      await program.run_async(self.mock_engine, num_steps=1)

      self.mock_engine.sync_weights.assert_called_once_with(
          role=datatypes.Role.ACTOR
      )
      self.assertEqual(program.policy_version, 5)
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.policy_version, 5)

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

  def test_resume_sets_step_and_policy_version_from_engine(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=3)
    program = self._create_program(dataset=["p0"], max_steps=5)

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()

    asyncio.run(_run())
    self.assertEqual(program.step, 3)
    self.assertEqual(program.policy_version, 3)

  def test_resume_forwards_role_and_resync_flag_when_sync_enabled(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=3)
    program = self._create_program(
        dataset=["p0"], max_steps=5, sync_weights=True
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()

    asyncio.run(_run())
    self.mock_engine.resume_from_checkpoint.assert_called_once_with(
        role=datatypes.Role.ACTOR, resync_rollout_weights=True, step=None
    )

  def test_resume_forwards_resync_disabled_when_sync_weights_false(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=2)
    program = self._create_program(
        dataset=["p0"], max_steps=5, sync_weights=False
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()

    asyncio.run(_run())
    self.mock_engine.resume_from_checkpoint.assert_called_once_with(
        role=datatypes.Role.ACTOR, resync_rollout_weights=False, step=None
    )
    self.assertEqual(program.step, 2)

  def test_resume_forwards_configured_checkpoint_restore_step(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=2)
    program = self._create_program(
        dataset=["p0"],
        max_steps=5,
        sync_weights=True,
        checkpoint_restore_step=2,
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()

    asyncio.run(_run())
    self.mock_engine.resume_from_checkpoint.assert_called_once_with(
        role=datatypes.Role.ACTOR, resync_rollout_weights=True, step=2
    )
    self.assertEqual(program.step, 2)

  def test_save_checkpoint_forwards_checkpoint_overwrite(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])
      program = self._create_program(
          dataset=["prompt_0"],
          max_steps=1,
          checkpoint_overwrite=True,
      )
      await program.run_async(self.mock_engine)
      self.mock_engine.save_checkpoint.assert_called_once_with(
          role=datatypes.Role.ACTOR,
          metadata={
              "step": 1,
              "global_step": 1,
              "policy_version": 1,
              "num_rollouts": 2,
              "num_microbatches": 1,
          },
          overwrite=True,
      )

    asyncio.run(_run())

  def test_resume_skips_already_consumed_dataset_prefix(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=3)
    dataset = [f"p{i}" for i in range(5)]
    program = self._create_program(
        dataset=dataset,
        max_steps=5,
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()
      await program.rollout_dispatch_stage()

    asyncio.run(_run())
    dispatched = [
        call.args[0][0]["prompt_id"]
        for call in self.mock_engine.dispatch_rollouts.call_args_list
    ]
    self.assertEqual(
        dispatched,
        [
            "prompt_3",
            "prompt_4",
        ],
    )

  def test_resume_skip_uses_full_batch_size(self):
    self.mock_algo.mini_batch_size = 2
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=1)
    dataset = [f"p{i}" for i in range(8)]
    program = self._create_program(
        dataset=dataset,
        max_steps=2,
        batch_size=4,
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()
      await program.rollout_dispatch_stage()

    asyncio.run(_run())
    dispatched = [
        call.args[0][0]["prompt_id"]
        for call in self.mock_engine.dispatch_rollouts.call_args_list
    ]
    self.assertEqual(
        dispatched, ["prompt_4", "prompt_5", "prompt_6", "prompt_7"]
    )

  def test_fresh_run_does_not_skip_dataset(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=0)
    dataset = [f"p{i}" for i in range(5)]
    program = self._create_program(
        dataset=dataset,
        max_steps=5,
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()
      await program.rollout_dispatch_stage()

    asyncio.run(_run())
    self.assertEqual(program.step, 0)
    self.assertEqual(self.mock_engine.dispatch_rollouts.call_count, 5)

  def test_resume_runs_before_first_dispatch(self):
    call_order = []
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=1)

    async def _resume(*args, **kwargs):
      del args, kwargs
      call_order.append("resume_from_checkpoint")
      return 1

    async def _dispatch(*args, **kwargs):
      del args, kwargs
      call_order.append("dispatch_rollouts")

    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(
        side_effect=_resume
    )
    self.mock_engine.dispatch_rollouts = mock.AsyncMock(side_effect=_dispatch)
    program = self._create_program(
        dataset=["p0", "p1", "p2"], max_steps=3, sync_weights=True
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()
      await program.rollout_dispatch_stage()

    asyncio.run(_run())
    self.assertEqual(call_order[0], "resume_from_checkpoint")
    self.assertIn("dispatch_rollouts", call_order)

  def _dispatched_prompts(self) -> list[Any]:
    return [
        call.args[0][0]
        for call in self.mock_engine.dispatch_rollouts.call_args_list
    ]

  def _run_dispatch(self, program: rl_program.StandardRLProgram) -> None:
    async def _run():
      program.engine = self.mock_engine
      await program.rollout_dispatch_stage()

    asyncio.run(_run())

  def test_dispatch_tags_dataset_coordinates(self):
    program = self._create_program(
        dataset=[f"p{i}" for i in range(5)], max_steps=3, batch_size=2
    )

    self._run_dispatch(program)

    coordinates = [
        (
            prompt["metadata"]["prompt_idx"],
            prompt["metadata"]["batch_idx"],
            prompt["metadata"]["intra_batch_idx"],
        )
        for prompt in self._dispatched_prompts()
    ]
    self.assertEqual(
        coordinates, [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1), (4, 2, 0)]
    )

  def test_dispatch_coordinates_follow_full_batch_size_not_mini_batch(self):
    self.mock_algo.mini_batch_size = 2
    program = self._create_program(
        dataset=[f"p{i}" for i in range(4)], max_steps=1, batch_size=4
    )

    self._run_dispatch(program)

    self.assertEqual(
        [prompt["metadata"]["batch_idx"] for prompt in
         self._dispatched_prompts()],
        [0, 0, 0, 0],
    )

  def test_dispatch_merges_coordinates_into_existing_metadata(self):
    program = self._create_program(
        dataset=[{"prompt": "p0", "metadata": {"env_config": {"id": 7}}}],
        max_steps=1,
    )

    self._run_dispatch(program)

    metadata = self._dispatched_prompts()[0]["metadata"]
    self.assertEqual(metadata["env_config"], {"id": 7})
    self.assertEqual(metadata["prompt_idx"], 0)

  def test_dispatch_coordinates_override_stale_dataset_tags(self):
    program = self._create_program(
        dataset=[{"prompt": "p0", "metadata": {"batch_idx": 99}}],
        max_steps=1,
    )

    self._run_dispatch(program)

    self.assertEqual(self._dispatched_prompts()[0]["metadata"]["batch_idx"], 0)

  def test_dispatch_does_not_mutate_dataset_items(self):
    dataset_item = {"prompt": "p0"}
    program = self._create_program(dataset=[dataset_item], max_steps=1)

    self._run_dispatch(program)

    self.assertEqual(dataset_item, {"prompt": "p0"})
    self.assertIn("metadata", self._dispatched_prompts()[0])

  def test_dispatch_tags_object_prompt_items_without_mutating_them(self):
    prompt_item = types.SimpleNamespace(
        prompt="p0", prompt_id="custom_0", metadata={"env_config": {"id": 7}}
    )
    program = self._create_program(dataset=[prompt_item], max_steps=1)

    self._run_dispatch(program)

    dispatched = self._dispatched_prompts()[0]
    self.assertEqual(dispatched.prompt_id, "custom_0")
    self.assertEqual(dispatched.metadata["batch_idx"], 0)
    self.assertEqual(dispatched.metadata["env_config"], {"id": 7})
    self.assertEqual(prompt_item.metadata, {"env_config": {"id": 7}})

  def test_dispatch_tags_frozen_dataclass_prompt_items(self):
    @dataclasses.dataclass(frozen=True)
    class _FrozenPrompt:
      prompt: str
      prompt_id: str
      metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    prompt_item = _FrozenPrompt(prompt="p0", prompt_id="frozen_0")
    program = self._create_program(dataset=[prompt_item], max_steps=1)

    self._run_dispatch(program)

    dispatched = self._dispatched_prompts()[0]
    self.assertEqual(dispatched.prompt_id, "frozen_0")
    self.assertEqual(dispatched.metadata["batch_idx"], 0)
    self.assertEmpty(prompt_item.metadata)

  def test_dispatch_dispatches_untagged_when_metadata_cannot_be_set(self):
    class _SlottedPrompt:
      __slots__ = ("prompt", "prompt_id")

      def __init__(self):
        self.prompt = "p0"
        self.prompt_id = "slotted_0"

    prompt_item = _SlottedPrompt()
    program = self._create_program(dataset=[prompt_item], max_steps=1)

    with self.assertLogs(level="WARNING") as logs:
      self._run_dispatch(program)

    self.assertIs(self._dispatched_prompts()[0], prompt_item)
    self.assertIn("_SlottedPrompt", "".join(logs.output))

  def test_resumed_dispatch_tags_continue_from_dataset_position(self):
    self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=1)
    program = self._create_program(
        dataset=[f"p{i}" for i in range(4)], max_steps=2, batch_size=2
    )

    async def _run():
      program.engine = self.mock_engine
      await program._resume_from_checkpoint()
      await program.rollout_dispatch_stage()

    asyncio.run(_run())

    coordinates = [
        (
            prompt["metadata"]["prompt_idx"],
            prompt["metadata"]["batch_idx"],
            prompt["metadata"]["intra_batch_idx"],
        )
        for prompt in self._dispatched_prompts()
    ]
    self.assertEqual(coordinates, [(2, 1, 0), (3, 1, 1)])

  def test_dataset_coordinates_reach_rollout_requests(self):
    mock_rollout = _MockWorkerHandle(role=datatypes.Role.ROLLOUT)
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[mock_rollout],
        trainer_workers={
            datatypes.Role.ACTOR: _MockWorkerHandle(role=datatypes.Role.ACTOR)
        },
        weight_sync_coordinator=mock.MagicMock(),
    )
    program = self._create_program(
        dataset=["a", "b", "c"], max_steps=2, batch_size=2
    )

    async def _run():
      program.engine = engine
      await program.rollout_dispatch_stage()

    asyncio.run(_run())

    requests = [
        call[3]["requests"][0] for call in mock_rollout.dispatched_requests
    ]
    # num_generations=2, so each prompt fans out to two requests that must
    # carry identical coordinates.
    self.assertEqual(
        [(req.metadata["batch_idx"], req.metadata["intra_batch_idx"])
         for req in requests],
        [(0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0)],
    )
    self.assertEqual(
        [req.metadata["prompt_idx"] for req in requests], [0, 0, 1, 1, 2, 2]
    )

  def test_zero_staleness_dispatches_only_one_batch_ahead(self):
    async def _run():
      dispatched = []

      async def mock_dispatch(prompts, **kwargs):
        dispatched.append((prompts[0], kwargs["policy_version"]))
        return [f"{prompts[0]}_{kwargs['policy_version']}"]

      self.mock_engine.dispatch_rollouts.side_effect = mock_dispatch

      program = rl_program.StandardRLProgram(
          dataset=["prompt_0", "prompt_1"],
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=self.assembler,
          max_staleness=0,
      )
      program.engine = self.mock_engine

      dispatch_task = asyncio.create_task(program.rollout_dispatch_stage())

      for _ in range(50):
        if dispatched:
          break
        await asyncio.sleep(0.01)

      expected_p0 = {
          "prompt": "prompt_0",
          "prompt_id": "prompt_0",
          "max_response_length": 1024,
          "metadata": {
              "prompt_idx": 0,
              "batch_idx": 0,
              "intra_batch_idx": 0,
          },
      }
      expected_p1 = {
          "prompt": "prompt_1",
          "prompt_id": "prompt_1",
          "max_response_length": 1024,
          "metadata": {
              "prompt_idx": 1,
              "batch_idx": 1,
              "intra_batch_idx": 0,
          },
      }
      self.assertEqual(
          dispatched,
          [(expected_p0, 0)],
      )

      await asyncio.sleep(0.1)
      self.assertEqual(
          dispatched,
          [(expected_p0, 0)],
      )

      program.policy_version = 1
      program._step = 1
      program._release_window()
      await asyncio.wait_for(dispatch_task, timeout=1.0)
      self.assertEqual(
          dispatched,
          [
              (expected_p0, 0),
              (expected_p1, 1),
          ],
      )

    asyncio.run(_run())

  def _window_program(self, **kwargs: Any) -> rl_program.StandardRLProgram:
    """A program whose full batch is two single-rollout groups."""
    self.mock_algo.num_generations = 1
    self.mock_algo.mini_batch_size = 1
    program = rl_program.StandardRLProgram(
        dataset=[],
        max_steps=1,
        batch_size=2,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=batch_assembly.PaddedBatchAssembler(
            batch_size=1,
            max_prompt_length=4,
            max_response_length=4,
            pad_id=0,
            num_generations=1,
            mini_batch_size=1,
        ),
        sync_weights=False,
        **kwargs,
    )
    program.engine = self.mock_engine
    return program

  def _scored_item(self, group_idx: int) -> datatypes.TrajectoryItem:
    item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id=f"prompt_{group_idx}",
        start_step=0,
        traj={"trajectory_reward": 1.0},
    )
    item.payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.array([1.0, 1.0], dtype=np.float32),
        advantages=np.array([1.0, 1.0], dtype=np.float32),
    )
    return item

  def test_dispatch_window_admits_max_staleness_batches_ahead(self):
    async def _run():
      dispatched = []

      async def mock_dispatch(prompts, **kwargs):
        del kwargs
        dispatched.append(prompts[0]["metadata"]["batch_idx"])
        return ["rollout"]

      self.mock_engine.dispatch_rollouts.side_effect = mock_dispatch

      program = rl_program.StandardRLProgram(
          dataset=[f"prompt_{i}" for i in range(8)],
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=self.assembler,
          batch_size=2,
          max_staleness=1,
      )
      program.engine = self.mock_engine

      dispatch_task = asyncio.create_task(program.rollout_dispatch_stage())
      for _ in range(50):
        if len(dispatched) >= 4:
          break
        await asyncio.sleep(0.01)

      # `_step` is 0, so batches 0 and 1 are inside the window and batch 2 is
      # not. The dispatcher parks mid-dataset rather than running to the end.
      await asyncio.sleep(0.05)
      self.assertEqual(dispatched, [0, 0, 1, 1])

      program._step = 1
      program._release_window()
      for _ in range(50):
        if len(dispatched) >= 6:
          break
        await asyncio.sleep(0.01)

      await asyncio.sleep(0.05)
      self.assertEqual(dispatched, [0, 0, 1, 1, 2, 2])

      dispatch_task.cancel()
      await asyncio.gather(dispatch_task, return_exceptions=True)

    asyncio.run(_run())

  def test_window_advances_a_whole_batch_when_a_group_goes_missing(self):
    async def _run():
      program = self._window_program()

      # Only one of the batch's two prompts yielded a group the trainer could
      # consume: the other was dropped by the filter, or its rollouts never
      # completed.
      await program.scored_q.put(self._scored_item(0))
      await program.scored_q.close()

      await program.train_stage()

      # The window's lower edge moves by a batch, not by the number of groups
      # that happened to arrive, so a lost group cannot narrow it.
      self.assertEqual(program._step, 1)
      self.assertTrue(program._window_release.is_set())

    asyncio.run(_run())

  def test_train_stage_updates_only_on_last_microbatch(self):
    class TwoMicrobatchAssembler:
      num_generations: int = 1
      mini_batch_size: int = 1

      def feed(self, items):
        del items
        return [
            batch_assembly.AssembledBatch(
                payload="microbatch_0", is_final_batch=False
            ),
            batch_assembly.AssembledBatch(
                payload="microbatch_1", is_final_batch=True
            ),
        ]

      def flush(self):
        return []

      def reset(self):
        pass

      def pack(self, items):
        del items
        return ["microbatch_0", "microbatch_1"]

    async def _run():
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=TwoMicrobatchAssembler(),
          sync_weights=False,
      )
      program.engine = self.mock_engine

      for group_index in range(2):
        item = datatypes.TrajectoryItem(
            group_index=group_index,
            prompt_id="prompt_0",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = self.mock_algo.create_trainer_payloads.return_value[
            group_index
        ]
        await program.scored_q.put(item)

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

  def test_train_stage_streaming_padded_batch_assembler(self):
    async def _run():
      self.mock_algo.mini_batch_size = 4
      padded_assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=4,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=2,
          mini_batch_size=4,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=padded_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(group_idx, item_idx):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.array([1, 2], dtype=np.int32),
            prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
            completion_ids=np.array([3, 4], dtype=np.int32),
            completion_mask=np.array([1.0, 1.0], dtype=np.float32),
            advantages=np.array([1.0, 1.0], dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=item_idx,
            prompt_id=f"prompt_{group_idx}",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Enqueue 4 groups of 2 rollouts each = 8 rollouts
      for g in range(4):
        for i in range(2):
          await program.scored_q.put(_make_item(g, i))

      await program.train_stage()

      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      self.assertEqual(
          [
              call.kwargs["apply_optimizer"]
              for call in self.mock_engine.train_step.call_args_list
          ],
          [False, True],
      )
      self.assertEqual(program.last_step_result.num_microbatches, 2)
      self.assertEqual(program.last_step_result.num_rollouts, 8)

    asyncio.run(_run())

  def test_full_batch_contains_multiple_optimizer_updates_and_one_sync(self):
    async def _run():
      self.mock_algo.num_generations = 2
      self.mock_algo.mini_batch_size = 2
      self.mock_engine.train_step.side_effect = [
          "queued",
          {"train_step": 1},
          "queued",
          {"train_step": 2},
      ]
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=2,
          mini_batch_size=2,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          batch_size=4,
          reward_fns=[lambda *_: 1.0],
          assembler=assembler,
          sync_weights=True,
      )
      program.engine = self.mock_engine

      for group_idx in range(4):
        for item_idx in range(2):
          payload = datatypes.RLTrainerPayload(
              prompt_ids=np.array([1, 2], dtype=np.int32),
              prompt_mask=np.ones(2, dtype=np.float32),
              completion_ids=np.array([3, 4], dtype=np.int32),
              completion_mask=np.ones(2, dtype=np.float32),
              advantages=np.ones(2, dtype=np.float32),
          )
          item = datatypes.TrajectoryItem(
              group_index=item_idx,
              prompt_id=f"prompt_{group_idx}",
              start_step=0,
              traj={"trajectory_reward": 1.0},
          )
          item.payload = payload
          await program.scored_q.put(item)

      await program.train_stage()

      self.assertEqual(
          [
              call.kwargs["apply_optimizer"]
              for call in self.mock_engine.train_step.call_args_list
          ],
          [False, True, False, True],
      )
      self.mock_engine.sync_weights.assert_called_once_with(
          role=datatypes.Role.ACTOR
      )
      self.mock_engine.save_checkpoint.assert_called_once()
      checkpoint_metadata = self.mock_engine.save_checkpoint.call_args.kwargs[
          "metadata"
      ]
      self.assertEqual(checkpoint_metadata["step"], 2)
      self.assertEqual(checkpoint_metadata["global_step"], 1)
      self.assertEqual(checkpoint_metadata["policy_version"], 1)
      self.assertEqual(program.last_step_result.num_rollouts, 8)
      self.assertEqual(program.last_step_result.num_microbatches, 4)

    asyncio.run(_run())

  def test_full_batch_size_must_be_divisible_by_mini_batch_size(self):
    self.mock_algo.mini_batch_size = 2
    with self.assertRaisesRegex(ValueError, "batch_size must be divisible"):
      self._create_program(batch_size=3)

  def test_train_stage_mid_step_dataset_exhaustion_flushes_and_saves_checkpoint(
      self,
  ):
    async def _run():
      self.mock_algo.num_generations = 1
      self.mock_algo.mini_batch_size = 4
      padded_assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=4,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=1,
          mini_batch_size=4,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=padded_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(group_idx):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.array([1, 2], dtype=np.int32),
            prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
            completion_ids=np.array([3, 4], dtype=np.int32),
            completion_mask=np.array([1.0, 1.0], dtype=np.float32),
            advantages=np.array([1.0, 1.0], dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=0,
            prompt_id=f"prompt_{group_idx}",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Only 3 groups available (3 rollouts) instead of mini_batch_size=4 (4 rollouts)
      # Since batch_size=4, feed() buffers all 3 without emitting.
      for g in range(3):
        await program.scored_q.put(_make_item(g))
      await program.scored_q.close()

      await program.train_stage()

      # Flushed and trained the partial microbatch with apply_optimizer=True
      self.assertEqual(self.mock_engine.train_step.call_count, 1)
      self.assertTrue(
          self.mock_engine.train_step.call_args_list[0].kwargs[
              "apply_optimizer"
          ]
      )
      # Checkpoint and metrics are executed on the flushed batch
      self.mock_engine.get_metrics.assert_called_once_with(
          role=datatypes.Role.ACTOR
      )
      self.mock_engine.save_checkpoint.assert_called_once()
      self.assertEqual(program.last_step_result.num_microbatches, 1)
      self.assertEqual(program.last_step_result.num_rollouts, 3)

    asyncio.run(_run())

  def test_train_stage_mid_step_dataset_exhaustion_on_mini_batch_boundary_saves_checkpoint(
      self,
  ):
    async def _run():
      self.mock_algo.num_generations = 1
      self.mock_algo.mini_batch_size = 2
      padded_assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=1,
          mini_batch_size=2,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          batch_size=4,
          reward_fns=[lambda *_: 1.0],
          assembler=padded_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(group_idx):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.array([1, 2], dtype=np.int32),
            prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
            completion_ids=np.array([3, 4], dtype=np.int32),
            completion_mask=np.array([1.0, 1.0], dtype=np.float32),
            advantages=np.array([1.0, 1.0], dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=0,
            prompt_id=f"prompt_{group_idx}",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # 2 groups available = exactly 1 mini-batch (mini_batch_size=2).
      # full_batch_size=4, so groups_consumed (2) < full_batch_size (4).
      # On the next iteration, queue is empty, flush() returns [], and
      # save_checkpoint must still be called.
      for g in range(2):
        await program.scored_q.put(_make_item(g))
      await program.scored_q.close()

      await program.train_stage()

      self.mock_engine.train_step.assert_called_once()
      self.assertTrue(
          self.mock_engine.train_step.call_args.kwargs["apply_optimizer"]
      )
      self.mock_engine.save_checkpoint.assert_called_once()
      checkpoint_metadata = self.mock_engine.save_checkpoint.call_args.kwargs[
          "metadata"
      ]
      self.assertEqual(checkpoint_metadata["step"], 1)
      self.assertEqual(checkpoint_metadata["global_step"], 1)
      self.assertEqual(checkpoint_metadata["num_rollouts"], 2)
      self.assertEqual(checkpoint_metadata["num_microbatches"], 1)

    asyncio.run(_run())

  def test_train_stage_save_checkpoint_handles_none_train_step(self):
    async def _run():
      self.mock_algo.num_generations = 1
      self.mock_algo.mini_batch_size = 1
      self.mock_engine.train_step.return_value = {"train_step": None}
      program = self._create_program(batch_size=1)
      program.engine = self.mock_engine

      payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1.0, 1.0], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
      )
      item = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          traj={"trajectory_reward": 1.0},
      )
      item.payload = payload
      await program.scored_q.put(item)
      await program.scored_q.close()

      await program.train_stage()

      self.mock_engine.save_checkpoint.assert_called_once()
      checkpoint_metadata = self.mock_engine.save_checkpoint.call_args.kwargs[
          "metadata"
      ]
      self.assertEqual(checkpoint_metadata["step"], 1)

    asyncio.run(_run())

  def test_train_stage_sequence_packed_final_batch_broken_down_into_multiple_microbatches(
      self,
  ):
    async def _run():
      self.mock_algo.num_generations = 3
      self.mock_algo.mini_batch_size = 1
      packed_assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=1,
          max_packed_len=16,
          pad_id=0,
          num_generations=3,
          mini_batch_size=1,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=packed_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(item_idx):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.array([1, 2, 3], dtype=np.int32),
            prompt_mask=np.ones(3, dtype=np.float32),
            completion_ids=np.array([4, 5, 6, 7], dtype=np.int32),
            completion_mask=np.ones(4, dtype=np.float32),
            advantages=np.ones(4, dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=item_idx,
            prompt_id="prompt_0",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Enqueue 1 prompt group with 3 rollouts of length 7 each (total 21 tokens > 16)
      for i in range(3):
        await program.scored_q.put(_make_item(i))

      await program.train_stage()

      # Must break down into 2 microbatches: [apply_optimizer=False] and [apply_optimizer=True]
      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      self.assertEqual(
          [
              call.kwargs["apply_optimizer"]
              for call in self.mock_engine.train_step.call_args_list
          ],
          [False, True],
      )
      self.assertEqual(program.last_step_result.num_microbatches, 2)
      self.assertEqual(program.last_step_result.num_rollouts, 3)

    asyncio.run(_run())

  def test_train_stage_streaming_sequence_packed_across_input_batch_boundaries(
      self,
  ):
    async def _run():
      self.mock_algo.num_generations = 2
      self.mock_algo.mini_batch_size = 2
      packed_assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=1,
          max_packed_len=16,
          pad_id=0,
          num_generations=2,
          mini_batch_size=2,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=packed_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(group_idx, item_idx, length):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.full(1, group_idx + 1, dtype=np.int32),
            prompt_mask=np.ones(1, dtype=np.float32),
            completion_ids=np.full(length - 1, group_idx + 1, dtype=np.int32),
            completion_mask=np.ones(length - 1, dtype=np.float32),
            advantages=np.ones(length - 1, dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=item_idx,
            prompt_id=f"prompt_{group_idx}",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Group 0: 2 items of 2 tokens = 4 tokens
      for i in range(2):
        await program.scored_q.put(_make_item(0, i, length=2))
      # Group 1: 2 items of 3 tokens = 6 tokens
      for i in range(2):
        await program.scored_q.put(_make_item(1, i, length=3))

      await program.train_stage()

      # Combined into 1 packed sequence (4 + 6 = 10 tokens <= 16)
      self.assertEqual(self.mock_engine.train_step.call_count, 1)
      self.assertTrue(
          self.mock_engine.train_step.call_args_list[0].kwargs[
              "apply_optimizer"
          ]
      )
      self.assertEqual(program.last_step_result.num_microbatches, 1)
      self.assertEqual(program.last_step_result.num_rollouts, 4)

    asyncio.run(_run())

  def test_train_stage_sequence_packed_with_batch_size_greater_than_one(self):
    async def _run():
      self.mock_algo.train_micro_batch_size = 2
      self.mock_algo.num_generations = 2
      self.mock_algo.mini_batch_size = 2
      packed_assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=2,
          max_packed_len=16,
          pad_id=0,
          num_generations=2,
          mini_batch_size=2,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=packed_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(prompt_id, idx, length):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.full(1, idx + 1, dtype=np.int32),
            prompt_mask=np.ones(1, dtype=np.float32),
            completion_ids=np.full(length - 1, idx + 1, dtype=np.int32),
            completion_mask=np.ones(length - 1, dtype=np.float32),
            advantages=np.ones(length - 1, dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=idx,
            prompt_id=prompt_id,
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Each item has 10 tokens. Since 10 + 10 = 20 > 16, each item requires its own bin.
      # 4 items -> 4 bins.
      # With batch_size=2, these 4 bins form 2 microbatches of shape [2, 16].
      for i in range(2):
        await program.scored_q.put(_make_item("prompt_0", i, length=10))
      for i in range(2):
        await program.scored_q.put(_make_item("prompt_1", i, length=10))

      await program.train_stage()

      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      calls = self.mock_engine.train_step.call_args_list
      # Microbatch 0: batch_size=2, apply_optimizer=False
      mb0 = calls[0].args[0]
      self.assertEqual(mb0.completion_ids.shape, (2, 16))
      self.assertFalse(calls[0].kwargs["apply_optimizer"])
      self.assertTrue(calls[0].kwargs["accumulate_gradients"])
      # Microbatch 1: batch_size=2, apply_optimizer=True
      mb1 = calls[1].args[0]
      self.assertEqual(mb1.completion_ids.shape, (2, 16))
      self.assertTrue(calls[1].kwargs["apply_optimizer"])
      self.assertTrue(calls[1].kwargs["accumulate_gradients"])

      self.assertEqual(program.last_step_result.num_microbatches, 2)
      self.assertEqual(program.last_step_result.num_rollouts, 4)

    asyncio.run(_run())

  def test_train_stage_sequence_packed_pads_trailing_microbatch_to_batch_size(
      self,
  ):
    async def _run():
      self.mock_algo.train_micro_batch_size = 2
      self.mock_algo.num_generations = 3
      self.mock_algo.mini_batch_size = 1
      packed_assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=2,
          max_packed_len=16,
          pad_id=0,
          num_generations=3,
          mini_batch_size=1,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=packed_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(idx, length):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.full(1, idx + 1, dtype=np.int32),
            prompt_mask=np.ones(1, dtype=np.float32),
            completion_ids=np.full(length - 1, idx + 1, dtype=np.int32),
            completion_mask=np.ones(length - 1, dtype=np.float32),
            advantages=np.ones(length - 1, dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=idx,
            prompt_id="prompt_0",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # 3 items of 10 tokens each. Each requires its own bin.
      # 3 bins with batch_size=2 produces:
      # Microbatch 0: 2 bins (shape [2, 16])
      # Microbatch 1: 1 bin + 1 zero-padded trailing row (shape [2, 16])
      for i in range(3):
        await program.scored_q.put(_make_item(i, length=10))

      await program.train_stage()

      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      calls = self.mock_engine.train_step.call_args_list
      mb0 = calls[0].args[0]
      mb1 = calls[1].args[0]
      self.assertEqual(mb0.completion_ids.shape, (2, 16))
      self.assertEqual(mb1.completion_ids.shape, (2, 16))

      # Trailing row in mb1 is zero-padded
      self.assertTrue(np.all(mb1.segment_ids[1] == 0))
      self.assertTrue(np.all(mb1.completion_mask[1] == 0.0))
      self.assertTrue(np.all(mb1.completion_ids[1] == 0))

      self.assertEqual(program.last_step_result.num_microbatches, 2)
      self.assertEqual(program.last_step_result.num_rollouts, 3)

    asyncio.run(_run())

  def test_train_stage_logs_prompt_ids(self):
    class TwoMicrobatchAssembler:
      num_generations: int = 2
      mini_batch_size: int = 1
      groups_per_assembly_batch: int = 1

      @property
      def assembly_batch_size(self) -> int:
        return self.groups_per_assembly_batch * self.num_generations

      def feed(self, items):
        del items
        return [
            batch_assembly.AssembledBatch(
                payload="microbatch_0",
                is_final_batch=False,
                trajectory_ids=("traj_prompt_0_g0",),
            ),
            batch_assembly.AssembledBatch(
                payload="microbatch_1",
                is_final_batch=True,
                trajectory_ids=("traj_prompt_0_g1",),
            ),
        ]

      def flush(self):
        return []

      def reset(self):
        pass

      def pack(self, items):
        del items
        return ["microbatch_0", "microbatch_1"]

    async def _run():
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=TwoMicrobatchAssembler(),
          sync_weights=False,
      )
      program.engine = self.mock_engine

      for group_index in range(2):
        item = datatypes.TrajectoryItem(
            group_index=group_index,
            prompt_id="prompt_0",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = self.mock_algo.create_trainer_payloads.return_value[
            group_index
        ]
        await program.scored_q.put(item)

      with self.assertLogs(level="INFO") as logs:
        await program.train_stage()

      self.assertTrue(
          any(
              "Packed 1 trajectories into microbatch: [traj_prompt_0_g0]" in log
              for log in logs.output
          )
      )
      self.assertTrue(
          any(
              "Packed 1 trajectories into microbatch: [traj_prompt_0_g1]" in log
              for log in logs.output
          )
      )

    asyncio.run(_run())

  def test_train_stage_logs_multi_group_packed_microbatch(self):
    async def _run():
      self.mock_algo.num_generations = 2
      self.mock_algo.mini_batch_size = 2
      padded_assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=4,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=2,
          mini_batch_size=2,
      )
      program = rl_program.StandardRLProgram(
          dataset=[],
          max_steps=1,
          algo=self.mock_algo,
          reward_fns=[lambda *_: 1.0],
          assembler=padded_assembler,
          sync_weights=False,
      )
      program.engine = self.mock_engine

      def _make_item(group_idx, item_idx):
        payload = datatypes.RLTrainerPayload(
            prompt_ids=np.array([1, 2], dtype=np.int32),
            prompt_mask=np.array([1.0, 1.0], dtype=np.float32),
            completion_ids=np.array([3, 4], dtype=np.int32),
            completion_mask=np.array([1.0, 1.0], dtype=np.float32),
            advantages=np.array([1.0, 1.0], dtype=np.float32),
        )
        item = datatypes.TrajectoryItem(
            group_index=item_idx,
            prompt_id=f"prompt_{group_idx}",
            start_step=0,
            traj={"trajectory_reward": 1.0},
        )
        item.payload = payload
        return item

      # Group 0: 2 items
      for i in range(2):
        await program.scored_q.put(_make_item(0, i))
      # Group 1: 2 items
      for i in range(2):
        await program.scored_q.put(_make_item(1, i))

      with self.assertLogs(level="INFO") as logs:
        await program.train_stage()

      self.assertTrue(
          any(
              "Packed 4 trajectories into microbatch: [traj_prompt_0_g0,"
              " traj_prompt_0_g1, traj_prompt_1_g0, traj_prompt_1_g1]"
              in log
              for log in logs.output
          )
      )

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
        reward_fns=[lambda *_: 2.0], dataset=["sync_prompt"]
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
          [{
              **dict_item,
              "max_response_length": 1024,
              "metadata": {
                  "prompt_idx": 0,
                  "batch_idx": 0,
                  "intra_batch_idx": 0,
              },
          }],
          num_generations=2,
          policy_version=0,
          exact_token_continuity=True,
          generation_args=datatypes.GenerationArgs(
              return_logprobs=True,
          ),
      )

    asyncio.run(_run())

  def test_prompt_id_and_group_index_propagation_end_to_end(self):
    """Verifies that prompt_id and group_index are automatically built and propagated:

    1. Raw dataset string prompt (without manual prompt_id) -> RLProgram builds
    prompt_0
    2. Engine dispatch -> assigns group_index (0, 1) and creates deterministic
    request_ids
    3. Rollout worker -> creates RolloutResponse inheriting prompt_id and
    group_index
    4. Queue manager -> groups by prompt_0, delivers complete group
    5. Reward function -> receives items with prompt_id='prompt_0' and
    group_index=(0, 1)
    6. Batch assembly -> receives reconstructed items with prompt_id and
    group_index preserved
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

      def tracking_reward_fn(
          unused_completion: str, metadata: dict[str, Any]
      ) -> float:
        observed_in_reward.append({
            "prompt_id": metadata["prompt_id"],
            "group_index": metadata["group_index"],
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
            prompt_ids=np.array([1, 2], dtype=np.int32),
            prompt_mask=np.array([1, 1], dtype=np.float32),
            completion_ids=np.array([3, 4], dtype=np.int32),
            completion_mask=np.array([1, 1], dtype=np.float32),
            advantages=np.array([1.0, 1.0], dtype=np.float32),
        )
        return [mock_p, mock_p]

      self.mock_algo.create_trainer_payloads = tracking_create_payloads

      program = self._create_program(
          dataset=dataset,
          reward_fns=[tracking_reward_fn],
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
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=1,
          num_generations=2,
          mini_batch_size=2,
          max_packed_len=8,
      )
      program = self._create_program(dataset=["p0", "p1"], assembler=assembler)

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

  def test_multi_group_sequence_packed_with_batch_size_greater_than_one(self):
    async def _run():
      self.mock_algo.train_micro_batch_size = 2
      self.mock_algo.mini_batch_size = 2
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group("prompt_0"),
          _make_trajectory_group("prompt_1"),
      )
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=2,
          num_generations=2,
          mini_batch_size=2,
          max_packed_len=8,
      )
      program = self._create_program(dataset=["p0", "p1"], assembler=assembler)

      await program.run_async(self.mock_engine)

      # 2 groups of 8 tokens each form 2 bins. With batch_size=2, they form 1 microbatch of shape [2, 8].
      self.assertEqual(self.mock_engine.train_step.call_count, 1)
      calls = self.mock_engine.train_step.call_args_list
      self.assertEqual(calls[0].args[0].completion_ids.shape, (2, 8))
      self.assertTrue(calls[0].kwargs["accumulate_gradients"])
      self.assertTrue(calls[0].kwargs["apply_optimizer"])
      self.assertEqual(program.last_step_result.num_rollouts, 4)
      self.assertEqual(program.last_step_result.num_microbatches, 1)

    asyncio.run(_run())

  def test_reference_kl_logprobs_scoring_in_train_stage(self):
    async def _run():
      self.mock_algo.requires_reference_kl = True
      mock_payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([[1, 2]], dtype=np.int32),
          prompt_mask=np.ones((1, 2), dtype=np.float32),
          completion_ids=np.array([[3, 4]], dtype=np.int32),
          completion_mask=np.ones((1, 2), dtype=np.float32),
          advantages=np.ones((1, 2), dtype=np.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
      self.assembler.feed = mock.MagicMock(
          return_value=[
              batch_assembly.AssembledBatch(
                  payload=mock_payload,
                  is_final_batch=True,
                  trajectory_ids=(),
              )
          ]
      )
      self.mock_engine.per_token_logps = mock.AsyncMock(
          return_value=np.array([[-0.1, -0.2]], dtype=np.float32)
      )

      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group())
      program = self._create_program(dataset=["prompt_0"])

      await program.run_async(self.mock_engine)

      self.mock_engine.per_token_logps.assert_called_once_with(
          datatypes.Role.REFERENCE, items=mock_payload
      )
      self.assertEqual(program.step, 1)

    asyncio.run(_run())

  def test_reference_kl_raises_type_error_for_invalid_microbatch(self):
    async def _run():
      self.mock_algo.requires_reference_kl = True
      # Returning a raw dict instead of RLTrainerPayload
      self.assembler.feed = mock.MagicMock(
          return_value=[
              batch_assembly.AssembledBatch(
                  payload={"raw": "batch"},  # pyrefly: ignore[bad-argument-type]
                  is_final_batch=True,
                  trajectory_ids=(),
              )
          ]
      )
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

      def failing_reward_fn(*_):
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

      # Advantage Metrics
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/mean", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/max", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/max", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/min", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/min", "train"), 1.0
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/std", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/std", "train"), 0.0
      )
      self.assertAlmostEqual(program.last_step_result.advantage_mean, 1.0)
      self.assertAlmostEqual(program.last_step_result.advantage_std, 0.0)

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
      self.mock_engine.sync_weights.assert_not_called()
      self.assertTrue(
          logger.metric_exists("", "orchestrator/policy_version", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "orchestrator/policy_version", "train"), 0.0
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

  def test_advantage_metrics_logging(self):
    async def _run():
      rollout_worker = _MockWorkerHandle(role="rollout")
      rollout_worker.responses = [
          [
              _create_rollout_response(
                  "req_0", "prompt_0", group_index=0, reward=1.0
              ),
              _create_rollout_response(
                  "req_1", "prompt_0", group_index=1, reward=2.0
              ),
          ],
      ]
      trainer_worker = _MockWorkerHandle(role="trainer")
      trainer_worker.metrics_buffer = exp_metrics.MetricsBuffer(
          id=1, scalar_metrics={"loss": 0.1}
      )
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[rollout_worker],
          trainer_workers={datatypes.Role.ACTOR: trainer_worker},
      )
      payload_0 = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.full(2, 1.5, dtype=np.float32),
      )
      payload_1 = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([5, 6], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.full(2, -0.5, dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.return_value = [
          payload_0,
          payload_1,
      ]

      program = self._create_program(
          dataset=["prompt_0"], reward_fns=[], sync_weights=False
      )
      await program.run_async(engine, max_steps=1)

      logger = program.metrics_logger
      self.assertIsNotNone(logger)
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/mean", "train"), 0.5
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/max", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/max", "train"), 1.5
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/min", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/min", "train"), -0.5
      )
      self.assertTrue(
          logger.metric_exists("", "rewards/advantage/std", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rewards/advantage/std", "train"), 1.0
      )
      self.assertAlmostEqual(program.last_step_result.advantage_mean, 0.5)
      self.assertAlmostEqual(program.last_step_result.advantage_std, 1.0)

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
          prompt_ids=np.array([1], dtype=np.int32),
          prompt_mask=np.array([1], dtype=np.float32),
          completion_ids=np.array([2], dtype=np.int32),
          completion_mask=np.array([1], dtype=np.float32),
          advantages=np.array([1.0], dtype=np.float32),
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
          prompt_ids=np.arange(4, dtype=np.int32),
          prompt_mask=np.ones(4, dtype=np.float32),
          completion_ids=np.arange(6, dtype=np.int32),
          completion_mask=np.ones(6, dtype=np.float32),
          advantages=np.ones(6, dtype=np.float32),
      )
      payload_1 = datatypes.RLTrainerPayload(
          prompt_ids=np.arange(4, dtype=np.int32),
          prompt_mask=np.ones(4, dtype=np.float32),
          completion_ids=np.arange(6, dtype=np.int32),
          completion_mask=np.ones(6, dtype=np.float32),
          advantages=np.ones(6, dtype=np.float32),
      )
      traj_item_0 = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj={"trajectory_reward": 1.0},
      )
      traj_item_0.payload = payload_0
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj={"trajectory_reward": 1.0},
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

  def test_sequence_packed_payload_metric_computation(self):
    async def _run():
      # Sequence-packed payload where prompts are masked with 0 in completion_mask
      # and identified via segment_ids > 0.
      payload_0 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros(0, dtype=np.int32),
          prompt_mask=np.zeros(0, dtype=np.float32),
          completion_ids=np.arange(7, dtype=np.int32),
          completion_mask=np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32),
          segment_ids=np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.int32),
          advantages=np.ones(7, dtype=np.float32),
      )
      payload_1 = datatypes.RLTrainerPayload(
          prompt_ids=np.zeros(0, dtype=np.int32),
          prompt_mask=np.zeros(0, dtype=np.float32),
          completion_ids=np.arange(7, dtype=np.int32),
          completion_mask=np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32),
          segment_ids=np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.int32),
          advantages=np.ones(7, dtype=np.float32),
      )
      traj_item_0 = datatypes.TrajectoryItem(
          group_index=0,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj={"trajectory_reward": 1.0},
      )
      traj_item_0.payload = payload_0
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=None,
          completion_tokens=None,
          traj={"trajectory_reward": 1.0},
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
          logger.get_metric("", "rollout/prompt_length_mean", "train"), 2.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/completion_length_mean", "train"), 3.0
      )
      self.assertAlmostEqual(
          logger.get_metric("", "rollout/total_tokens_mean", "train"), 5.0
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
          traj={"trajectory_reward": 1.0, "status": None},
      )
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj={"trajectory_reward": 1.0, "status": None},
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
          traj={"trajectory_reward": 1.0, "steps": [mock_step, mock_step]},
      )
      traj_item_1 = datatypes.TrajectoryItem(
          group_index=1,
          prompt_id="prompt_0",
          start_step=0,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          traj={
              "trajectory_reward": 1.0,
              "steps": [mock_step, mock_step, mock_step, mock_step],
          },
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

      with mock.patch("jax.process_index", return_value=0), mock.patch.dict(
          "sys.modules", {"wandb": mock_wandb}
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
        self.assertIn("train/rewards/advantage/abs_mean", logged_keys)
        self.assertIn("train/rewards/advantage/nonzero_frac", logged_keys)
        self.assertIn("train/rollout/prompt_length_mean", logged_keys)
        self.assertIn("train/rollout/staleness_mean", logged_keys)
        self.assertIn("train/orchestrator/policy_version", logged_keys)

        # Verify wandb.finish was called on close
        mock_wandb.finish.assert_called_once()

    asyncio.run(_run())

  def test_pipelined_multi_prompt_microbatch_execution(self):
    async def _run():
      self.mock_algo.num_generations = 2
      self.mock_algo.mini_batch_size = 4
      mock_payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.side_effect = (
          lambda group, **kwargs: [mock_payload] * len(group)
      )

      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=4,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=2,
          mini_batch_size=4,
      )

      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group("prompt_0", num_generations=2),
          _make_trajectory_group("prompt_1", num_generations=2),
          _make_trajectory_group("prompt_2", num_generations=2),
          _make_trajectory_group("prompt_3", num_generations=2),
      )

      program = self._create_program(
          dataset=["p0", "p1", "p2", "p3"],
          assembler=assembler,
          sync_weights=False,
      )

      await program.run_async(self.mock_engine)

      # 4 groups of 2 rollouts = 8 rollouts total.
      # train_micro_batch_size = 4 rollouts (2 groups per microbatch).
      # Total microbatches = 2.
      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      calls = self.mock_engine.train_step.call_args_list

      # First microbatch (groups 0 & 1): accumulate_gradients=True,
      # apply_optimizer=False
      self.assertTrue(calls[0].kwargs["accumulate_gradients"])
      self.assertFalse(calls[0].kwargs["apply_optimizer"])
      # Second microbatch (groups 2 & 3): accumulate_gradients=True,
      # apply_optimizer=True
      self.assertTrue(calls[1].kwargs["accumulate_gradients"])
      self.assertTrue(calls[1].kwargs["apply_optimizer"])

      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.num_rollouts, 8)
      self.assertEqual(program.last_step_result.num_microbatches, 2)

    asyncio.run(_run())

  def test_pipelined_sub_prompt_microbatch_execution(self):
    async def _run():
      self.mock_algo.num_generations = 4
      self.mock_algo.mini_batch_size = 1
      mock_payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.side_effect = (
          lambda group, **kwargs: [mock_payload] * len(group)
      )

      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=4,
          max_response_length=4,
          pad_id=0,
          num_generations=4,
          mini_batch_size=1,
      )

      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group("prompt_0", num_generations=4),
      )

      program = self._create_program(
          dataset=["p0"],
          assembler=assembler,
          sync_weights=False,
      )

      await program.run_async(self.mock_engine)

      # 1 group of 4 rollouts = 4 rollouts total.
      # train_micro_batch_size = 2 rollouts (2 microbatches for the group).
      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      calls = self.mock_engine.train_step.call_args_list

      # First microbatch: accumulate_gradients=True, apply_optimizer=False
      self.assertTrue(calls[0].kwargs["accumulate_gradients"])
      self.assertFalse(calls[0].kwargs["apply_optimizer"])
      # Second microbatch: accumulate_gradients=True, apply_optimizer=True
      self.assertTrue(calls[1].kwargs["accumulate_gradients"])
      self.assertTrue(calls[1].kwargs["apply_optimizer"])

      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.num_rollouts, 4)
      self.assertEqual(program.last_step_result.num_microbatches, 2)

    asyncio.run(_run())

  def test_non_positive_batch_dimensions_rejected(self):
    self.mock_algo.mini_batch_size = 0
    with self.assertRaisesRegex(
        ValueError, "mini_batch_size and num_generations must be positive"
    ):
      self._create_program()

    self.mock_algo.mini_batch_size = 1
    self.mock_algo.num_generations = 0
    with self.assertRaisesRegex(
        ValueError, "mini_batch_size and num_generations must be positive"
    ):
      self._create_program()

  def test_program_passes_generation_args_to_dispatch_rollouts(self):
    async def _run():
      self.mock_algo.max_response_length = 512
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group(prompt_id="p0", num_generations=2),
          [],
      )
      gen_args = datatypes.GenerationArgs(
          max_generation_steps=128,
          temperature=0.7,
          top_p=0.9,
          return_logprobs=True,
      )
      p = self._create_program(
          dataset=("p0",),
          generation_args=gen_args,
          sync_weights=False,
      )
      await p.run_async(self.mock_engine)
      self.mock_engine.dispatch_rollouts.assert_called_once()
      args, kwargs = self.mock_engine.dispatch_rollouts.call_args
      expected_gen_args = datatypes.GenerationArgs(
          max_generation_steps=128,
          temperature=0.7,
          top_p=0.9,
          return_logprobs=True,
      )
      self.assertEqual(kwargs.get("generation_args"), expected_gen_args)
      self.assertEqual(args[0][0].get("max_response_length"), 512)

    asyncio.run(_run())

  def test_program_passes_algo_max_response_length_when_gen_args_none(self):
    async def _run():
      self.mock_algo.max_response_length = 512
      _set_mock_poll_batches(
          self.mock_engine,
          _make_trajectory_group(prompt_id="p0", num_generations=2),
          [],
      )
      p = self._create_program(
          dataset=("p0",),
          generation_args=None,
          sync_weights=False,
      )
      await p.run_async(self.mock_engine)
      self.mock_engine.dispatch_rollouts.assert_called_once()
      args, kwargs = self.mock_engine.dispatch_rollouts.call_args
      expected_gen_args = datatypes.GenerationArgs(
          return_logprobs=True,
      )
      self.assertEqual(kwargs.get("generation_args"), expected_gen_args)
      self.assertEqual(args[0][0].get("max_response_length"), 512)

    asyncio.run(_run())

  def test_program_temperature_matching_sets_algo_config(self):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_algo.num_generations = 2
    mock_algo.mini_batch_size = 1
    mock_algo.max_turns = 1
    mock_algo.max_packed_len = 16
    mock_algo.max_response_length = 1024
    mock_algo.requires_reference_kl = False
    mock_algo.algo_config = mock.MagicMock(
        temperature=0.8, use_rollout_logps=None
    )

    gen_args = datatypes.GenerationArgs(temperature=0.8)
    rl_program.StandardRLProgram(
        dataset=("p0",),
        algo=mock_algo,
        generation_args=gen_args,
    )
    self.assertEqual(mock_algo.algo_config.temperature, 0.8)

  def test_program_temperature_missing_in_generation_args_leaves_none(
      self,
  ):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_algo.num_generations = 2
    mock_algo.mini_batch_size = 1
    mock_algo.max_turns = 1
    mock_algo.max_packed_len = 16
    mock_algo.max_response_length = 1024
    mock_algo.requires_reference_kl = False
    mock_algo.algo_config = mock.MagicMock(
        temperature=0.8, use_rollout_logps=None
    )

    program = rl_program.StandardRLProgram(
        dataset=("p0",),
        algo=mock_algo,
    )
    self.assertIsNone(program.generation_args.temperature)

  def test_program_temperature_missing_in_algo_config_propagates_from_generation_args(
      self,
  ):
    mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_algo.num_generations = 2
    mock_algo.mini_batch_size = 1
    mock_algo.max_turns = 1
    mock_algo.max_packed_len = 16
    mock_algo.max_response_length = 1024
    mock_algo.requires_reference_kl = False
    mock_algo.algo_config = mock.MagicMock(
        temperature=None, use_rollout_logps=None
    )

    gen_args = datatypes.GenerationArgs(temperature=0.8)
    rl_program.StandardRLProgram(
        dataset=("p0",),
        algo=mock_algo,
        generation_args=gen_args,
    )
    self.assertEqual(mock_algo.algo_config.temperature, 0.8)

  def test_run_async_auto_configures_worker_on_engine(self):
    async def _run():
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])
      mock_assembler = mock.MagicMock()
      mock_assembler.pad_id = 11
      mock_assembler.eos_id = 22
      mock_assembler.assembly_batch_size = 2
      mock_assembler.groups_per_assembly_batch = 1
      mock_assembler.pack.return_value = [
          mock.MagicMock(spec=datatypes.RLTrainerPayload)
      ]

      program = self._create_program(
          dataset=["prompt_data_0"],
          max_steps=1,
          assembler=mock_assembler,
      )
      await program.run_async(self.mock_engine)

      self.mock_engine.configure_worker.assert_called_once_with(
          role=datatypes.Role.ACTOR,
          algo=self.mock_algo,
          assembler=mock_assembler,
      )

    asyncio.run(_run())

  def test_run_async_configures_worker_before_prepare_rollout_policy(self):
    async def _run():
      call_order = []
      self.mock_engine.configure_worker.side_effect = (
          lambda **kwargs: call_order.append("configure_worker")
      )
      self.mock_engine.prepare_rollout_policy = mock.AsyncMock(
          side_effect=lambda **kwargs: call_order.append(
              "prepare_rollout_policy"
          )
      )
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])

      program = self._create_program(
          dataset=["prompt_data_0"],
          max_steps=1,
          sync_weights=True,
      )
      await program.run_async(self.mock_engine)

      self.assertEqual(
          call_order, ["configure_worker", "prepare_rollout_policy"]
      )

    asyncio.run(_run())

  def test_run_async_propagates_configure_worker_failure(self):
    async def _run():
      self.mock_engine.configure_worker.side_effect = ValueError(
          "Worker configuration failed"
      )

      program = self._create_program(
          dataset=["prompt_data_0"],
          max_steps=1,
      )
      with self.assertRaisesRegex(ValueError, "Worker configuration failed"):
        await program.run_async(self.mock_engine)

      self.mock_engine.prepare_rollout_policy.assert_not_called()
      self.mock_engine.dispatch_rollouts.assert_not_called()

    asyncio.run(_run())

  def test_program_creates_sequence_packed_assembler(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=None,
        batch_config=batch_assembly.BatchConfig(
            max_seq_token_per_tpu=1024,
            max_segments_per_packed_row=4,
            trainer_fsdp=2,
            trainer_dp=2,
        ),
    )
    self.assertIsInstance(
        program.assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(program.assembler.max_packed_len, 1024)
    self.assertEqual(program.assembler.batch_size, 4)
    self.assertEqual(program.assembler.max_segments_per_packed_row, 4)

  def test_program_creates_padded_assembler(self):
    self.mock_algo.max_response_length = 2048
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=None,
        batch_config=batch_assembly.BatchConfig(
            max_prompt_length=128,
            pad_id=5,
        ),
    )
    self.assertIsInstance(
        program.assembler, batch_assembly.PaddedBatchAssembler
    )
    self.assertEqual(program.assembler.max_prompt_length, 128)
    self.assertEqual(program.assembler.max_response_length, 2048)
    self.assertEqual(program.assembler.pad_id, 5)

    program_override = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=None,
        batch_config=batch_assembly.BatchConfig(
            max_prompt_length=128,
            max_response_length=512,
            pad_id=5,
        ),
    )
    self.assertEqual(program_override.assembler.max_response_length, 512)

  def test_program_falls_back_to_algo_config_max_response_length(self):
    self.mock_algo.max_response_length = None
    self.mock_algo.algo_config.max_response_length = 768
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
    )
    self.assertEqual(program.max_response_length, 768)
    self.assertEqual(program.batch_config.max_response_length, 768)

  def test_program_creates_default_assembler(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=None,
    )
    self.assertIsInstance(
        program.assembler, batch_assembly.SequencePackedBatchAssembler
    )
    self.assertEqual(program.assembler.max_packed_len, 8192)

  # --- Sampler/trainer agreement (Phase 4) ---

  def test_sampler_trainer_agreement_skipped_when_use_rollout_logps_false(self):
    """When use_rollout_logps is False, agreement is skipped even with old logps."""

    async def _run():
      self.mock_algo.algo_config.use_rollout_logps = False
      payload_with_old_logps = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
          old_per_token_logps=np.array([-0.5, -0.2], dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.return_value = [
          payload_with_old_logps,
          payload_with_old_logps,
      ]
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])
      program = self._create_program(dataset=["prompt_data_0"])
      await program.run_async(self.mock_engine)
      self.mock_engine.per_token_logps.assert_not_called()

    asyncio.run(_run())

  def test_sampler_trainer_agreement_triggered_in_train_stage(self):
    """When use_rollout_logps is True and old_per_token_logps present, agreement runs."""

    async def _run():
      self.mock_algo.algo_config.use_rollout_logps = True
      payload_with_old_logps = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.array([1, 1], dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.array([1, 1], dtype=np.float32),
          advantages=np.array([1.0, 1.0], dtype=np.float32),
          old_per_token_logps=np.array([-0.5, -0.2], dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.return_value = [
          payload_with_old_logps,
          payload_with_old_logps,
      ]

      async def _fake_per_token_logps(role, *, items, **kwargs):
        del role, kwargs
        return datatypes.LogprobsResponse(
            per_token_logps=np.full_like(
                items.completion_tokens, -0.5, dtype=np.float32
            ),
            model_version=1,
        )

      self.mock_engine.per_token_logps = mock.AsyncMock(
          side_effect=_fake_per_token_logps
      )
      _set_mock_poll_batches(self.mock_engine, _make_trajectory_group(), [])
      program = self._create_program(dataset=["prompt_data_0"])
      await program.run_async(self.mock_engine)
      self.mock_engine.per_token_logps.assert_awaited()
      logger = program.metrics_logger
      self.assertTrue(
          logger.metric_exists("", "sampler_trainer/logp_diff_mean", "train")
      )
      self.assertAlmostEqual(
          logger.get_metric("", "sampler_trainer/logp_diff_mean", "train"),
          0.15,
          places=5,
      )

    asyncio.run(_run())

  def test_apply_sampler_trainer_agreement_records_metrics(self):
    """Helper scores actor logps and records namespaced agreement metrics."""

    async def _run():
      self.mock_algo.algo_config.use_rollout_logps = True
      self.mock_algo.algo_config.sampler_is = None
      self.mock_algo.algo_config.sampler_is_threshold = 2.0
      program = self._create_program()
      trainer_logps = np.array([[-0.5, -1.0, -0.2]], dtype=np.float32)
      program.engine = mock.MagicMock()
      program.engine.per_token_logps = mock.AsyncMock(
          return_value=datatypes.LogprobsResponse(
              per_token_logps=trainer_logps, model_version=1
          )
      )
      batch = datatypes.RLTrainerPayload(
          prompt_ids=np.array([[1, 2]], dtype=np.int32),
          prompt_mask=np.array([[1, 1]], dtype=np.float32),
          completion_ids=np.array([[3, 4, 5]], dtype=np.int32),
          completion_mask=np.array([[1, 1, 1]], dtype=np.float32),
          advantages=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
          old_per_token_logps=np.array([[-0.4, -1.2, -0.1]], dtype=np.float32),
      )
      acc: dict[str, Any] = {}
      out = await program._apply_sampler_trainer_agreement(batch, acc)

      program.engine.per_token_logps.assert_awaited_once()
      await_args = program.engine.per_token_logps.await_args
      self.assertEqual(await_args.args[0], datatypes.Role.ACTOR)
      req = await_args.kwargs["items"]
      self.assertIsInstance(req, datatypes.LogprobsRequest)
      self.assertEqual(req.model_role, "actor")
      self.assertEqual(req.pad_id, program.batch_config.pad_id)
      self.assertIn("sampler_trainer/logp_diff_mean", acc)
      _, diff_mean_vals = acc["sampler_trainer/logp_diff_mean"]
      self.assertAlmostEqual(diff_mean_vals[0], 0.4 / 3, places=5)
      self.assertIn("sampler_trainer/probs_pearson_corr", acc)
      # sampler_is is None -> no batch mutation, no TIS weights.
      self.assertIs(out, batch)
      self.assertIsNone(out.sampler_is_weights)

    asyncio.run(_run())

  def test_apply_sampler_trainer_agreement_token_is_feeds_weights(self):
    """With sampler_is='token' the helper feeds TIS weights and trainer logps."""

    async def _run():
      self.mock_algo.algo_config.sampler_is = "token"
      self.mock_algo.algo_config.sampler_is_threshold = 2.0
      program = self._create_program()
      self.assertEqual(program.sampler_is, "token")
      self.assertEqual(program.sampler_is_threshold, 2.0)
      trainer_logps = np.array([[-0.5, -1.0, -0.2]], dtype=np.float32)
      program.engine = mock.MagicMock()
      program.engine.per_token_logps = mock.AsyncMock(
          return_value=datatypes.LogprobsResponse(
              per_token_logps=trainer_logps, model_version=1
          )
      )
      batch = datatypes.RLTrainerPayload(
          prompt_ids=np.array([[1, 2]], dtype=np.int32),
          prompt_mask=np.array([[1, 1]], dtype=np.float32),
          completion_ids=np.array([[3, 4, 5]], dtype=np.int32),
          completion_mask=np.array([[1, 1, 1]], dtype=np.float32),
          advantages=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
          old_per_token_logps=np.array([[-0.4, -1.2, -0.1]], dtype=np.float32),
      )
      acc: dict[str, Any] = {}
      out = await program._apply_sampler_trainer_agreement(batch, acc)

      self.assertIsNotNone(out.sampler_is_weights)
      # old_per_token_logps is overwritten with the trainer logps.
      np.testing.assert_allclose(
          np.asarray(out.old_per_token_logps), trainer_logps
      )
      self.assertIn("sampler_is/weight_mean", acc)
      self.assertIn("sampler_is/frac_clipped_at_threshold", acc)

    asyncio.run(_run())

  def test_apply_sampler_trainer_agreement_seq_logprob_error_threshold(self):
    """Sequences exceeding seq_logprob_error_threshold have completion_mask zeroed."""

    async def _run():
      self.mock_algo.algo_config.sampler_is = None
      self.mock_algo.algo_config.seq_logprob_error_threshold = 2.0
      program = self._create_program()
      self.assertEqual(program.seq_logprob_error_threshold, 2.0)
      trainer_logps = np.array(
          [[-0.5, -0.5, -0.5], [-2.0, -2.0, -2.0]], dtype=np.float32
      )
      program.engine = mock.MagicMock()
      program.engine.per_token_logps = mock.AsyncMock(
          return_value=datatypes.LogprobsResponse(
              per_token_logps=trainer_logps, model_version=1
          )
      )
      batch = datatypes.RLTrainerPayload(
          prompt_ids=np.array([[1, 2], [1, 2]], dtype=np.int32),
          prompt_mask=np.array([[1, 1], [1, 1]], dtype=np.float32),
          completion_ids=np.array([[3, 4, 5], [3, 4, 5]], dtype=np.int32),
          completion_mask=np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32),
          advantages=np.array(
              [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
          ),
          old_per_token_logps=np.array(
              [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]], dtype=np.float32
          ),
      )
      acc: dict[str, Any] = {}
      out = await program._apply_sampler_trainer_agreement(batch, acc)

      np.testing.assert_array_equal(
          np.asarray(out.completion_mask),
          np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32),
      )
      np.testing.assert_allclose(
          np.asarray(out.old_per_token_logps), trainer_logps
      )
      self.assertIn("sampler_trainer/seq_error_masked_frac", acc)
      self.assertAlmostEqual(
          acc["sampler_trainer/seq_error_masked_frac"][1][0], 0.5, places=5
      )

    asyncio.run(_run())

  def test_collect_and_log_step_metrics_logs_sampler_agreement(self):
    """Accumulated agreement metrics are reduced by their agg fn and logged."""
    program = self._create_program()
    program._collect_and_log_step_metrics(
        all_step_items=[],
        step_rewards=[],
        step_advantages=[],
        step_result=None,
        trainer_metrics=None,
        num_rollouts=0,
        num_microbatches=0,
        step_time_sec=0.0,
        consumed_policy_version=0,
        log_step=0,
        sampler_agreement={
            "sampler_trainer/logp_diff_mean": (np.mean, [0.2, 0.4]),
            "sampler_trainer/logp_diff_max": (np.max, [0.5, 0.9]),
        },
    )
    logger = program.metrics_logger
    self.assertTrue(
        logger.metric_exists("", "sampler_trainer/logp_diff_mean", "train")
    )
    self.assertAlmostEqual(
        logger.get_metric("", "sampler_trainer/logp_diff_mean", "train"), 0.3
    )
    self.assertAlmostEqual(
        logger.get_metric("", "sampler_trainer/logp_diff_max", "train"), 0.9
    )

  def test_trajectory_logger_initialization(self):
    with mock.patch("tunix.utils.trajectory_logger.AsyncTrajectoryLogger") as mock_logger_cls:
      mock_logger_inst = mock.MagicMock()
      mock_logger_cls.return_value = mock_logger_inst

      # Case 1: derived from metrics_logging_options.log_dir
      program1 = rl_program.StandardRLProgram(
          dataset=["prompt_0"],
          max_steps=1,
          algo=self.mock_algo,
          metrics_logging_options=metrics_logger_lib.MetricsLoggerOptions(
              log_dir="/tmp/metrics_dir"
          ),
      )
      mock_logger_cls.assert_called_with("/tmp/metrics_dir/trajectories")
      self.assertIs(program1.trajectory_logger, mock_logger_inst)

      # Case 2: explicit trajectory_log_dir
      mock_logger_cls.reset_mock()
      program2 = rl_program.StandardRLProgram(
          dataset=["prompt_0"],
          max_steps=1,
          algo=self.mock_algo,
          trajectory_log_dir="/custom/trajectories",
      )
      mock_logger_cls.assert_called_with("/custom/trajectories")
      self.assertIs(program2.trajectory_logger, mock_logger_inst)

      # Case 3: disabled when no log_dir provided
      program3 = rl_program.StandardRLProgram(
          dataset=["prompt_0"],
          max_steps=1,
          algo=self.mock_algo,
      )
      self.assertIsNone(program3.trajectory_logger)

  def _scoring_item(self, group_index, *, masked=False, failed=False):
    """A Token-mode trajectory as the collector hands it to critique."""
    return datatypes.TrajectoryItem(
        prompt_id="p0",
        group_index=group_index,
        start_step=0,
        traj={
            "status": (
                datatypes.TrajectoryStatus.FAILED
                if failed
                else datatypes.TrajectoryStatus.SUCCEEDED
            ),
            "trajectory_reward": 0.0,
            "conversation_text": [{"role": "assistant", "content": "answer"}],
            "conversation_masks": (
                np.zeros(2, dtype=np.float32)
                if masked
                else np.ones(2, dtype=np.float32)
            ),
        },
        metadata={"group_index": group_index},
    )

  def test_critique_stage_skips_reward_fns_for_masked_items(self):
    async def _run():
      scored_group_indices = []

      def reward_fn(completion, metadata):
        del completion
        scored_group_indices.append(metadata["group_index"])
        return 1.0

      program = self._create_program(reward_fns=[reward_fn])
      program.engine = self.mock_engine
      await program.raw_q.put(self._scoring_item(0))
      await program.raw_q.put(self._scoring_item(1, masked=True))
      await program.raw_q.close()

      await program.critique_stage()

      # The masked rollout is scored 0.0 outright: running the reward function
      # over its truncated conversation would grade a broken trajectory.
      self.assertEqual(scored_group_indices, [0])
      rewards = self.mock_algo.create_trainer_payloads.call_args.kwargs[
          "rewards"
      ]
      self.assertEqual(rewards, [1.0, 0.0])
      program.close()

    asyncio.run(_run())

  def test_critique_stage_skips_reward_fns_for_failed_items(self):
    async def _run():
      reward_fn = mock.MagicMock(return_value=1.0)
      program = self._create_program(reward_fns=[reward_fn])
      program.engine = self.mock_engine
      await program.raw_q.put(self._scoring_item(0))
      await program.raw_q.put(self._scoring_item(1, failed=True))
      await program.raw_q.close()

      await program.critique_stage()

      reward_fn.assert_called_once()
      rewards = self.mock_algo.create_trainer_payloads.call_args.kwargs[
          "rewards"
      ]
      self.assertEqual(rewards, [1.0, 0.0])
      program.close()

    asyncio.run(_run())

  def test_critique_stage_evaluates_unmasked_max_steps_and_skips_timeout(
      self,
  ):
    async def _run():
      reward_fn = mock.MagicMock(return_value=0.75)
      program = self._create_program(reward_fns=[reward_fn])
      program.engine = self.mock_engine
      unmasked_max_steps_item = datatypes.TrajectoryItem(
          prompt_id="p0",
          group_index=0,
          traj={
              "conversation_text": [
                  {"role": "assistant", "content": "answer"},
              ],
              "status": datatypes.TrajectoryStatus.MAX_STEPS_REACHED,
              "prompt_tokens": np.array([1, 2], dtype=np.int32),
              "conversation_tokens": np.array([3, 4], dtype=np.int32),
              "conversation_masks": np.ones(2, dtype=np.float32),
          },
          metadata={"group_index": 0},
      )
      unmasked_timeout_item = datatypes.TrajectoryItem(
          prompt_id="p0",
          group_index=1,
          traj={
              "conversation_text": [
                  {"role": "assistant", "content": "answer"},
              ],
              "status": datatypes.TrajectoryStatus.TIMEOUT,
              "prompt_tokens": np.array([1, 2], dtype=np.int32),
              "conversation_tokens": np.array([3, 4], dtype=np.int32),
              "conversation_masks": np.ones(2, dtype=np.float32),
          },
          metadata={"group_index": 1},
      )
      await program.raw_q.put(unmasked_max_steps_item)
      await program.raw_q.put(unmasked_timeout_item)
      await program.raw_q.close()

      await program.critique_stage()

      # MAX_STEPS_REACHED with non-zero masks (overlong_filter=False) is valid
      # and scored, whereas TIMEOUT is always invalid and skipped.
      self.assertEqual(reward_fn.call_count, 1)
      rewards = self.mock_algo.create_trainer_payloads.call_args.kwargs[
          "rewards"
      ]
      self.assertEqual(rewards, [0.75, 0.0])
      program.close()

    asyncio.run(_run())

  def _masked_metrics_items(self):
    """One healthy and one masked-out item, each carrying a trainer payload."""
    items = []
    for group_index, mask in enumerate([np.ones(2), np.zeros(2)]):
      item = self._scoring_item(group_index, masked=not mask.any())
      item.payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.ones(2, dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=mask.astype(np.float32),
          advantages=np.zeros(2, dtype=np.float32),
      )
      items.append(item)
    return items

  def test_collect_and_log_step_metrics_logs_invalid_trajectory_frac(self):
    program = self._create_program()

    program._collect_and_log_step_metrics(
        all_step_items=self._masked_metrics_items(),
        step_rewards=[1.0, 0.0],
        step_advantages=[0.5, 0.0],
        step_result=None,
        trainer_metrics=None,
        num_rollouts=2,
        num_microbatches=1,
        step_time_sec=0.0,
        consumed_policy_version=0,
        log_step=0,
    )

    logger = program.metrics_logger
    self.assertAlmostEqual(
        logger.get_metric("rollout", "invalid_trajectory_frac", "train"), 0.5
    )
    # Reward metrics reflect only valid trajectories.
    self.assertAlmostEqual(logger.get_metric("rewards", "mean", "train"), 1.0)
    self.assertFalse(logger.metric_exists("rewards", "valid_mean", "train"))
    program.close()

  def test_collect_and_log_step_metrics_skips_rewards_when_all_invalid(self):
    program = self._create_program()
    all_invalid_items = [
        self._scoring_item(0, masked=True),
        self._scoring_item(1, failed=True),
    ]

    program._collect_and_log_step_metrics(
        all_step_items=all_invalid_items,
        step_rewards=[0.0, 0.0],
        step_advantages=[0.0, 0.0],
        step_result=None,
        trainer_metrics=None,
        num_rollouts=2,
        num_microbatches=1,
        step_time_sec=0.0,
        consumed_policy_version=0,
        log_step=0,
    )

    logger = program.metrics_logger
    self.assertAlmostEqual(
        logger.get_metric("rollout", "invalid_trajectory_frac", "train"), 1.0
    )
    self.assertFalse(logger.metric_exists("rewards", "mean", "train"))
    program.close()

  def test_critique_stage_preserves_is_valid_for_degenerate_group_survivor(
      self,
  ):
    async def _run():
      # Simulate a degenerate group (1 valid + 1 masked) where both trainer
      # payloads receive zeroed completion_mask.
      zero_payload = datatypes.RLTrainerPayload(
          prompt_ids=np.array([1, 2], dtype=np.int32),
          prompt_mask=np.ones(2, dtype=np.float32),
          completion_ids=np.array([3, 4], dtype=np.int32),
          completion_mask=np.zeros(2, dtype=np.float32),
          advantages=np.zeros(2, dtype=np.float32),
      )
      self.mock_algo.create_trainer_payloads.return_value = [
          zero_payload,
          zero_payload,
      ]
      program = self._create_program(reward_fns=[lambda c, m: 1.0])
      program.engine = self.mock_engine
      await program.raw_q.put(self._scoring_item(0))
      await program.raw_q.put(self._scoring_item(1, masked=True))
      await program.raw_q.close()

      await program.critique_stage()
      scored_group = await program.scored_q.get_group()

      self.assertTrue(scored_group[0].is_valid)
      self.assertFalse(scored_group[1].is_valid)

      program._collect_and_log_step_metrics(
          all_step_items=scored_group,
          step_rewards=[1.0, 0.0],
          step_advantages=[0.0, 0.0],
          step_result=None,
          trainer_metrics=None,
          num_rollouts=2,
          num_microbatches=1,
          step_time_sec=0.0,
          consumed_policy_version=0,
          log_step=0,
      )
      logger = program.metrics_logger
      self.assertAlmostEqual(
          logger.get_metric("rollout", "invalid_trajectory_frac", "train"), 0.5
      )
      self.assertAlmostEqual(
          logger.get_metric("rewards", "mean", "train"), 1.0
      )
      program.close()

    asyncio.run(_run())

  def test_log_consumed_trajectories(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        trajectory_log_dir="/tmp/trajectories",
    )
    mock_traj_logger = mock.MagicMock()
    program.trajectory_logger = mock_traj_logger

    # A real Token-mode trajectory dict, matching what the collector produces.
    # A MagicMock here would make the reward assertion pass for the wrong
    # reason, since float(MagicMock()) happens to return 1.0.
    traj = {
        "status": datatypes.TrajectoryStatus.SUCCEEDED,
        "trajectory_reward": 1.0,
        "conversation_text": [
            {"role": "user", "content": "Q: What is 2+2?\nA:"},
            {"role": "assistant", "content": "4"},
        ],
    }
    item = datatypes.TrajectoryItem(
        traj_id="traj_1",
        prompt_id="prompt_1",
        group_index=0,
        policy_version=2,
        prompt_tokens=[1, 2],
        completion_tokens=[3, 4],
        metadata={
            "question": "What is 2+2?",
            "prompt": "Q: What is 2+2?\nA:",
            "gold_answer": "4",
        },
        traj=traj,
    )

    program._log_consumed_trajectories(
        [item],
        log_step=3,
        consumed_policy_version=3,
    )

    mock_traj_logger.log_item_async.assert_called_once()
    row = mock_traj_logger.log_item_async.call_args[0][0]
    self.assertEqual(row["global_step"], 3)
    self.assertEqual(row["consumed_policy_version"], 3)
    self.assertEqual(row["prompt_id"], "prompt_1")
    self.assertEqual(row["group_index"], 0)
    self.assertEqual(row["rollout_policy_version"], 2)
    self.assertEqual(row["status"], "SUCCEEDED")
    self.assertEqual(row["reward"], 1.0)
    self.assertEqual(row["question"], "What is 2+2?")
    self.assertEqual(row["prompt"], "Q: What is 2+2?\nA:")
    # Only the assistant turn, never the prompt that precedes it.
    self.assertEqual(row["completion"], "4")
    self.assertEqual(row["gold_answer"], "4")
    self.assertEqual(row["prompt_tokens"], [1, 2])
    self.assertEqual(row["completion_tokens"], [3, 4])
    self.assertEqual(row["metadata"], item.metadata)
    self.assertEqual(row["trajectory"], traj)

    program.close()
    mock_traj_logger.stop.assert_called_once()

  def test_log_consumed_trajectories_reward_distinction(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        trajectory_log_dir="/tmp/trajectories",
    )
    mock_traj_logger = mock.MagicMock()
    program.trajectory_logger = mock_traj_logger

    # Case 1: Actual zero reward (should log 0.0)
    zero_reward_item = datatypes.TrajectoryItem(
        prompt_id="p_zero",
        group_index=0,
        traj={"status": "FAILED", "trajectory_reward": 0.0},
    )
    # Case 2: Reward not available / None (should log None, not 0.0)
    none_reward_item = datatypes.TrajectoryItem(
        prompt_id="p_none",
        group_index=1,
        traj={"status": "RUNNING"},
    )
    # Case 3: No traj object at all (should log None, not 0.0)
    no_traj_item = datatypes.TrajectoryItem(
        prompt_id="p_notraj",
        group_index=2,
        traj=None,
    )

    program._log_consumed_trajectories(
        [zero_reward_item, none_reward_item, no_traj_item],
        log_step=1,
        consumed_policy_version=1,
    )

    self.assertEqual(mock_traj_logger.log_item_async.call_count, 3)
    rows = [
        call[0][0] for call in mock_traj_logger.log_item_async.call_args_list
    ]

    self.assertEqual(rows[0]["reward"], 0.0)
    self.assertIsNone(rows[1]["reward"])
    self.assertIsNone(rows[2]["reward"])

    program.close()

  def test_log_consumed_trajectories_multi_turn_includes_env_messages(self):
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=self.mock_algo,
        trajectory_log_dir="/tmp/trajectories",
    )
    mock_traj_logger = mock.MagicMock()
    program.trajectory_logger = mock_traj_logger

    traj = {
        "status": datatypes.TrajectoryStatus.SUCCEEDED,
        "trajectory_reward": 1.0,
        "conversation_text": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix bug in foo.py"},
            {"role": "assistant", "content": "ls -l"},
            {"role": "user", "content": "foo.py bar.py"},
            {"role": "assistant", "content": "done"},
        ],
    }
    item = datatypes.TrajectoryItem(
        prompt_id="prompt_multi",
        group_index=0,
        traj=traj,
    )

    program._log_consumed_trajectories(
        [item], log_step=1, consumed_policy_version=1
    )

    mock_traj_logger.log_item_async.assert_called_once()
    row = mock_traj_logger.log_item_async.call_args[0][0]
    expected_completion = (
        "[assistant]: ls -l\n[environment]: foo.py bar.py\n[assistant]: done"
    )
    self.assertEqual(row["completion"], expected_completion)
    program.close()

  def test_invoke_reward_fn_passes_concatenated_assistant_text_and_metadata(
      self,
  ):
    item = datatypes.TrajectoryItem(
        prompt_id="p_math",
        group_index=3,
        traj={
            "conversation_text": [
                {"role": "system", "content": "you are a math tutor"},
                {
                    "role": "user",
                    "content": (
                        "Put your reasoning inside"
                        " <reasoning>...</reasoning> tags and your answer"
                        " inside <answer>\\boxed{}</answer> tags."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "<reasoning>2+2=4.</reasoning>",
                },
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "<answer>\\boxed{4}</answer>"},
            ]
        },
        metadata={"gold_answer": "4", "prompt_id": "p_math", "group_index": 3},
    )
    captured = {}

    def reward_fn(completion: str, metadata: dict[str, Any]) -> float:
      captured["completion"] = completion
      captured["metadata"] = dict(metadata)
      return 1.0

    score = rl_program._invoke_reward_fn(reward_fn, item)
    self.assertEqual(score, 1.0)
    self.assertEqual(
        captured["completion"],
        "<reasoning>2+2=4.</reasoning><answer>\\boxed{4}</answer>",
    )
    self.assertEqual(
        captured["metadata"],
        {"gold_answer": "4", "prompt_id": "p_math", "group_index": 3},
    )


def _traj_item(tokens, clipped=None, raw_length=None):
  """Builds a TrajectoryItem, optionally with collector annotations."""
  traj = {"conversation_tokens": np.asarray(tokens, dtype=np.int32)}
  metadata = {}
  if clipped is not None:
    metadata["clipped"] = clipped
    metadata["raw_length"] = (
        raw_length if raw_length is not None else len(tokens)
    )
  return datatypes.TrajectoryItem(
      prompt_id="p", group_index=0, traj=traj, metadata=metadata
  )


class GenerationMetricsTest(absltest.TestCase):
  """Covers `_generation_metrics`, the per-prompt-group accounting."""

  def test_uses_collector_annotations(self):
    # The collector scored these against the budget actually enforced for the
    # request, which may differ from this program's default. Whatever it
    # decided is authoritative here; nothing is recomputed.
    metrics = rl_program._generation_metrics([[
        _traj_item([1, 2, 3], clipped=True, raw_length=3),
        _traj_item([1, 2], clipped=False, raw_length=2),
    ]])

    self.assertEqual(metrics["generation/completions/clip_ratio"], 0.5)
    self.assertEqual(metrics["generation/completions/mean_raw_length"], 2.5)

  def test_ignores_annotations_in_traj_dict(self):
    # Enforces Zero-Redundancy contract: annotations must be read strictly
    # from item.metadata, not item.traj.
    item_with_traj_only = datatypes.TrajectoryItem(
        prompt_id="p",
        group_index=0,
        traj={
            "conversation_tokens": np.array([1, 2, 3], dtype=np.int32),
            "clipped": True,
            "raw_length": 3,
        },
        metadata={},
    )
    self.assertEmpty(rl_program._generation_metrics([[item_with_traj_only]]))

  def test_zero_length_rollout_stays_in_denominator(self):
    # A rollout that ran and produced nothing is still a rollout: the agentic
    # learner divides by the whole group, so dropping it would inflate
    # clip_ratio (1.0 here instead of 0.5).
    metrics = rl_program._generation_metrics([[
        _traj_item([1, 2, 3, 4], clipped=True, raw_length=4),
        _traj_item([], clipped=False, raw_length=0),
    ]])

    self.assertEqual(metrics["generation/completions/clip_ratio"], 0.5)
    self.assertEqual(metrics["generation/completions/min_raw_length"], 0.0)

  def test_unannotated_items_do_not_join_the_denominator(self):
    # Mixed groups are not expected in practice, but an unannotated rollout
    # must never be counted as "not clipped" by omission.
    metrics = rl_program._generation_metrics([[
        _traj_item([1, 2, 3, 4], clipped=True, raw_length=4),
        _traj_item([1, 2, 3, 4]),
    ]])

    self.assertEqual(metrics["generation/completions/clip_ratio"], 1.0)
    self.assertEqual(metrics["generation/completions/mean_raw_length"], 4.0)

  def test_half_annotated_item_is_skipped_not_raised(self):
    # A producer that wrote one key and not the other is a bug, but it must
    # not take down the training step: this is only a metric.
    half = _traj_item([1, 2, 3, 4], clipped=True, raw_length=4)
    del half.metadata["raw_length"]

    metrics = rl_program._generation_metrics(
        [[half, _traj_item([1, 2], clipped=False, raw_length=2)]]
    )

    self.assertEqual(metrics["generation/completions/clip_ratio"], 0.0)
    self.assertEqual(metrics["generation/completions/mean_raw_length"], 2.0)

  def test_returns_empty_without_annotations(self):
    payload_only = datatypes.TrajectoryItem(prompt_id="p", traj={})
    non_dict_traj = datatypes.TrajectoryItem(prompt_id="p", traj=object())

    self.assertEmpty(rl_program._generation_metrics([]))
    self.assertEmpty(rl_program._generation_metrics([[payload_only]]))
    self.assertEmpty(rl_program._generation_metrics([[non_dict_traj]]))
    self.assertEmpty(rl_program._generation_metrics([[_traj_item([1, 2])]]))

  def test_accepts_numpy_scalar_annotations(self):
    # The collector casts these, but the trajectory may round-trip through
    # numpy on the way here.
    metrics = rl_program._generation_metrics(
        [[_traj_item([1, 2], clipped=np.True_, raw_length=np.int64(2))]]
    )

    self.assertIsInstance(metrics["generation/completions/clip_ratio"], float)
    self.assertEqual(metrics["generation/completions/clip_ratio"], 1.0)
    self.assertEqual(metrics["generation/completions/mean_raw_length"], 2.0)

  def test_each_metric_uses_its_own_reduce_op(self):
    metrics = rl_program._generation_metrics([
        [
            _traj_item([], clipped=True, raw_length=8),
            _traj_item([], clipped=False, raw_length=12),
        ],
        [
            _traj_item([], clipped=False, raw_length=10),
            _traj_item([], clipped=False, raw_length=30),
        ],
    ])

    self.assertEqual(
        metrics,
        {
            "generation/completions/clip_ratio": 0.25,
            "generation/completions/mean_raw_length": 15.0,
            "generation/completions/max_raw_length": 30.0,
            "generation/completions/min_raw_length": 8.0,
        },
    )

  def test_unequal_groups_average_per_group_not_per_rollout(self):
    # Group A: 1 of 4 clipped. Group B: 2 of 2 clipped. Averaging the group
    # ratios gives 0.625; pooling rollouts would give 0.5. The agentic learner
    # averages group ratios, so this pins that behavior.
    group_a = [
        _traj_item([], clipped=c, raw_length=10)
        for c in (True, False, False, False)
    ]
    group_b = [_traj_item([], clipped=True, raw_length=10) for _ in range(2)]

    metrics = rl_program._generation_metrics([group_a, group_b])

    self.assertEqual(metrics["generation/completions/clip_ratio"], 0.625)


class GenerationMetricsLoggingTest(absltest.TestCase):
  """Covers how the computed metrics reach the metrics logger."""

  def _log_metrics(self, metrics):
    algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    algo.num_generations = 2
    algo.mini_batch_size = 1
    algo.max_turns = 1
    algo.max_packed_len = 16
    algo.max_response_length = 1024
    algo.requires_reference_kl = False
    algo.algo_config = types.SimpleNamespace(
        temperature=None,
        use_rollout_logps=True,
    )
    program = rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        max_steps=1,
        algo=algo,
        reward_fns=[lambda *_: 1.0],
    )
    program.metrics_logger = mock.MagicMock()
    program._collect_and_log_step_metrics(
        all_step_items=[],
        step_rewards=[],
        generation_metrics=metrics,
        num_rollouts=0,
        num_microbatches=0,
        step_time_sec=0.0,
        consumed_policy_version=0,
        log_step=0,
    )
    return {
        call.args[1]: call.args[2]
        for call in program.metrics_logger.log.call_args_list
    }

  def test_logs_the_names_generation_metrics_produced(self):
    # The names are the ones `_generation_metrics` returns; this pins that
    # nothing rewrites or re-prefixes them on the way to the logger.
    computed = rl_program._generation_metrics(
        [[_traj_item([1, 2], clipped=True, raw_length=2)]]
    )

    logged = self._log_metrics(computed)

    self.assertEqual(logged["generation/completions/clip_ratio"], 1.0)
    self.assertEqual(logged["generation/completions/max_raw_length"], 2.0)

  def test_no_metrics_logs_no_generation_metrics(self):
    logged = self._log_metrics({})

    self.assertEmpty(
        [k for k in logged if k.startswith("generation/completions/")]
    )


class StandardRLProgramTrajectoryStoreTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.num_generations = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_packed_len = 16
    self.mock_algo.max_response_length = 1024
    self.assembler = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=2,
        mini_batch_size=1,
        max_packed_len=16,
    )

  def _create_program(self, **kwargs) -> rl_program.StandardRLProgram:
    return rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        algo=self.mock_algo,
        reward_fns=[lambda *_: 1.0],
        assembler=self.assembler,
        **kwargs,
    )

  def test_no_trajectory_store_by_default(self):
    program = self._create_program()
    self.assertIsNone(program.trajectory_store)
    program.close()

  def test_holds_the_instance_it_was_given(self):
    store = in_memory_store.InMemoryTrajectoryStore()
    program = self._create_program(trajectory_store=store)
    self.assertIs(program.trajectory_store, store)
    program.close()

  def test_close_does_not_close_an_injected_store(self):
    store = mock.MagicMock(spec=in_memory_store.InMemoryTrajectoryStore)
    program = self._create_program(trajectory_store=store)
    program.close()
    store.close.assert_not_called()

  def test_close_without_a_store_does_not_raise(self):
    program = self._create_program()
    program.close()


class StandardRLProgramPromptBatchOrderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.num_generations = 1
    self.mock_algo.mini_batch_size = 2
    self.mock_algo.train_micro_batch_size = 2
    self.mock_algo.max_packed_len = 16
    self.mock_algo.max_response_length = 16
    self.mock_algo.requires_reference_kl = False
    self.mock_algo.algo_config = mock.MagicMock(
        temperature=1.0,
        use_rollout_logps=False,
        sampler_is=None,
        sampler_is_threshold=2.0,
    )
    self.mock_engine = mock.AsyncMock(
        spec=rl_program.rl_engine_interface.AbstractRLEngine
    )
    self.mock_engine.train_step.return_value = {"loss": 0.1}
    self.mock_engine.get_metrics.return_value = {"loss": 0.1}
    self.mock_engine.sync_weights.return_value = None

  def _make_scored_item(
      self,
      prompt_id: str,
      batch_idx: int,
      reward: float = 1.0,
      policy_version: int = 0,
  ) -> datatypes.TrajectoryItem:
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.array([1, 1], dtype=np.int32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.array([1, 1], dtype=np.int32),
        advantages=np.array([reward, reward], dtype=np.float32),
        metadata={"prompt_id": prompt_id, "batch_idx": batch_idx},
    )
    item = datatypes.TrajectoryItem(
        prompt_id=prompt_id,
        group_index=0,
        start_step=0,
        policy_version=policy_version,
        traj={"trajectory_reward": reward},
        metadata={"batch_idx": batch_idx},
    )
    item.payload = payload  # pyrefly: ignore[missing-attribute]
    return item

  def test_defaults_to_arrival_and_prompt_batch_builds_ordered_queue(self):
    default_prog = rl_program.StandardRLProgram(
        algo=self.mock_algo,
        dataset=["p0", "p1"],
    )
    self.assertEqual(
        default_prog.group_order,
        rl_program.trajectory_queue_manager.GroupOrder.ARRIVAL,
    )
    self.assertIsInstance(
        default_prog.scored_q,
        rl_program.trajectory_queue_manager.TrajectoryQueueManager,
    )
    default_prog.close()

    ordered_prog = rl_program.StandardRLProgram(
        algo=self.mock_algo,
        dataset=["p0", "p1"],
        batch_size=2,
        group_order="prompt_batch",
    )
    self.assertEqual(
        ordered_prog.group_order,
        rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
    )
    self.assertIsInstance(
        ordered_prog.raw_q,
        rl_program.trajectory_queue_manager.TrajectoryQueueManager,
    )
    self.assertIsInstance(
        ordered_prog.scored_q,
        rl_program.trajectory_queue_manager.BatchOrderedQueueManager,
    )
    self.assertEqual(ordered_prog.scored_q.full_batch_size, 2)
    ordered_prog.close()

  def test_train_stage_consumes_batches_in_prompt_batch_order(self):
    async def _run():
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          dataset=["p0", "p1", "p2", "p3"],
          batch_size=2,
          group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
      )
      program.engine = self.mock_engine

      # Batch 1 arrives before batch 0.
      await program.scored_q.put(self._make_scored_item("p2", batch_idx=1))
      await program.scored_q.put(self._make_scored_item("p3", batch_idx=1))
      await program.scored_q.put(self._make_scored_item("p0", batch_idx=0))
      await program.scored_q.put(self._make_scored_item("p1", batch_idx=0))
      await program.scored_q.close()

      trained_batches: list[list[str]] = []

      async def _record_train_step(batch, **_):
        trained_batches.append(
            [m["prompt_id"] for m in batch.metadata.get("items", [])]
            if "items" in batch.metadata
            else [batch.metadata.get("traj_id", "")]
        )
        return {"loss": 0.1}

      self.mock_engine.train_step.side_effect = _record_train_step
      committed_Order: list[tuple[int, float]] = []
      program.on_step_end = lambda step, _: committed_Order.append(
          (step, program.last_step_result.reward_mean)
      )

      await program.train_stage()
      self.assertEqual(program.step, 2)
      self.assertEqual(program._next_batch, 2)
      self.assertEqual(self.mock_engine.train_step.call_count, 2)
      program.close()

    asyncio.run(_run())

  def test_short_batch_flushes_and_commits_without_waiting_for_next_batch(self):
    async def _run():
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          dataset=["p0"],
          batch_size=2,
          max_staleness=0,
          group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
      )
      program.engine = self.mock_engine
      assert isinstance(
          program.scored_q, rl_program.trajectory_queue_manager.BatchOrderedQueueManager
      )
      program.scored_q.expect(0, num_groups=1)
      await program.scored_q.put(
          self._make_scored_item("p0", batch_idx=0, reward=2.5)
      )
      await program.scored_q.close()

      await program.train_stage()
      self.assertEqual(program.step, 1)
      self.assertEqual(program._next_batch, 1)
      self.assertAlmostEqual(program.last_step_result.reward_mean, 2.5)
      self.assertTrue(program._window_release.is_set())
      program.close()

    asyncio.run(_run())

  def test_cursor_advance_updates_next_batch_and_releases_window(self):
    program = rl_program.StandardRLProgram(
        algo=self.mock_algo,
        dataset=["p0", "p1", "p2"],
        batch_size=2,
        max_staleness=1,
        group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
    )
    assert isinstance(
        program.scored_q, rl_program.trajectory_queue_manager.BatchOrderedQueueManager
    )
    self.assertEqual(program._next_batch, 0)
    self.assertFalse(program._window_release.is_set())

    program.scored_q.skip(0, 2)
    self.assertEqual(program._next_batch, 2)
    self.assertTrue(program._window_release.is_set())
    program.close()

  def test_raw_q_staleness_drop_seals_short_batch_and_skips_hole_in_scored_q(
      self,
  ):
    async def _run():
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          dataset=["p0", "p1", "p2", "p3"],
          batch_size=2,
          max_staleness=1,
          group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
      )
      program.engine = self.mock_engine
      program.policy_version = 5

      def _make_raw_item(prompt_id: str, batch_idx: int, policy_version: int):
        return datatypes.TrajectoryItem(
            prompt_id=prompt_id,
            group_index=0,
            start_step=0,
            policy_version=policy_version,
            traj={"trajectory_reward": 1.0},
            metadata={"batch_idx": batch_idx},
        )

      # Batch 0: both groups are stale (policy_version=1, staleness=4 > 1) -> hole!
      await program.raw_q.put(_make_raw_item("p0", batch_idx=0, policy_version=1))
      await program.raw_q.put(_make_raw_item("p1", batch_idx=0, policy_version=1))

      # Hole in batch 0 immediately advances scored_q.next_batch_idx to 1 and releases window.
      self.assertEqual(program._next_batch, 1)
      self.assertTrue(program._window_release.is_set())

      # Batch 1: p2 is dropped as stale by raw_q; p3 survives and reaches scored_q.
      await program.raw_q.put(_make_raw_item("p2", batch_idx=1, policy_version=1))
      await program.scored_q.put(
          self._make_scored_item(
              "p3", batch_idx=1, reward=3.0, policy_version=5
          )
      )
      await program.scored_q.close()

      await program.train_stage()
      self.assertEqual(program.step, 1)
      self.assertEqual(program._next_batch, 2)
      self.assertAlmostEqual(program.last_step_result.reward_mean, 3.0)
      checkpoint_metadata = self.mock_engine.save_checkpoint.call_args.kwargs[
          "metadata"
      ]
      self.assertEqual(checkpoint_metadata["global_step"], 1)
      self.assertEqual(checkpoint_metadata["next_batch_idx"], 2)
      program.close()

    asyncio.run(_run())

  def test_scored_q_staleness_filter_drops_group_that_aged_during_critique(
      self,
  ):
    async def _run():
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          dataset=["p0", "p1"],
          batch_size=2,
          max_staleness=1,
          group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
      )
      program.engine = self.mock_engine
      program.policy_version = 5

      # p0 aged past max_staleness while in critique_stage (policy_version=2 < 4)
      # and is dropped at scored_q.put(); p1 (policy_version=4) is admitted.
      await program.scored_q.put(
          self._make_scored_item(
              "p0", batch_idx=0, reward=1.0, policy_version=2
          )
      )
      await program.scored_q.put(
          self._make_scored_item(
              "p1", batch_idx=0, reward=4.0, policy_version=4
          )
      )
      await program.scored_q.close()

      await program.train_stage()
      self.assertEqual(program.step, 1)
      self.assertEqual(program._next_batch, 1)
      self.assertAlmostEqual(program.last_step_result.reward_mean, 4.0)
      program.close()

    asyncio.run(_run())

  def test_resume_uses_next_batch_idx_to_skip_scored_q_and_seek_dataset(self):
    async def _run():
      dataset = [f"p{i}" for i in range(6)]
      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          dataset=dataset,
          batch_size=2,
          max_staleness=1,
          group_order=rl_program.trajectory_queue_manager.GroupOrder.PROMPT_BATCH,
      )
      # Checkpoint after 1 trained step where batch 0 was a hole and batch 1
      # was trained -> step=1, next_batch_idx=2.
      self.mock_engine.resume_from_checkpoint = mock.AsyncMock(return_value=1)
      self.mock_engine.restored_next_batch_idx = 2
      program.engine = self.mock_engine

      await program._resume_from_checkpoint()
      self.assertEqual(program.step, 1)
      self.assertEqual(program.policy_version, 1)
      self.assertEqual(program._next_batch, 2)

      # Dispatch stage skips batches 0 and 1 (4 prompts: p0..p3) and starts at
      # batch 2 (p4, p5).
      await program.rollout_dispatch_stage()
      dispatched_coords = [
          call.args[0][0]["metadata"]
          for call in self.mock_engine.dispatch_rollouts.call_args_list
      ]
      self.assertEqual(
          [c["prompt_idx"] for c in dispatched_coords],
          [4, 5],
      )
      self.assertEqual(
          [c["batch_idx"] for c in dispatched_coords],
          [2, 2],
      )
      program.close()

    asyncio.run(_run())


class ExtractScalarTest(absltest.TestCase):

  def test_extract_scalar_with_compute(self):
    class _MockMetric:

      def __init__(self, val):
        self.val = val

      def compute(self):
        return self.val

    m = _MockMetric(np.array([1.0, 5.0, 3.0]))
    self.assertEqual(
        rl_program._extract_scalar(m, "mult_prob_error_max"), 5.0
    )
    self.assertEqual(
        rl_program._extract_scalar(m, "mult_prob_error_min"), 1.0
    )
    self.assertAlmostEqual(
        rl_program._extract_scalar(m, "mult_prob_error_mean"), 3.0
    )
    self.assertAlmostEqual(
        rl_program._extract_scalar(m, "sample_mask/mult_prob_error/max"), 5.0
    )
    self.assertAlmostEqual(
        rl_program._extract_scalar(m, "sample_mask/mult_prob_error/min"), 1.0
    )

  def test_extract_scalar_single_element(self):
    class _MockScalarMetric:

      def compute(self):
        return np.array([4.2])

    self.assertAlmostEqual(
        rl_program._extract_scalar(_MockScalarMetric()), 4.2
    )
    self.assertAlmostEqual(
        rl_program._extract_scalar(3.14), 3.14
    )


if __name__ == "__main__":
  absltest.main()

