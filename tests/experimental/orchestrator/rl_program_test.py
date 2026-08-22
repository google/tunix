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
from collections.abc import Sequence
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_program
from tunix.rl import common as rl_common


def _create_rollout_response(
    request_id: str,
    prompt_id: str,
    group_id: str,
    pair_index: int = 0,
    policy_version: int = 0,
    reward: float = 1.0,
) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id,
      prompt_id=prompt_id,
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
      metadata={
          "group_id": group_id,
          "pair_index": pair_index,
      },
  )


class RLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock(
        spec=distributed_rl_engine.DistributedRLEngine
    )
    self.mock_engine.dispatch_rollouts = mock.AsyncMock()
    self.mock_engine.train_step = mock.AsyncMock(return_value="step_done")

    async def _mock_poll(*args, **kwargs):
      del args, kwargs
      await asyncio.sleep(0.01)
      return []

    self.mock_engine.sync_weights = mock.AsyncMock(return_value=1)
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

  def _make_trajectory_group(
      self,
      prompt_id: str = "prompt_0",
      group_id: str = "group_0",
      group_size: int = 2,
      reward: float = 1.0,
  ) -> list[datatypes.TrajectoryItem]:
    return [
        distributed_rl_engine._response_to_trajectory_item(
            _create_rollout_response(
                f"req_{prompt_id}_{idx}",
                prompt_id,
                group_id,
                pair_index=idx,
                reward=reward,
            )
        )
        for idx in range(group_size)
    ]

  def _set_mock_poll_batches(
      self, *batches: Sequence[datatypes.TrajectoryItem]
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

    self.mock_engine.poll_rollouts.side_effect = _mock_poll

  def _create_program(
      self,
      dataset: Any = ("prompt_0",),
      reward_fns: Any = None,
      **kwargs: Any,
  ) -> rl_program.StandardRLProgram:
    return rl_program.StandardRLProgram(
        dataset=dataset,
        algo=self.mock_algo,
        reward_fns=reward_fns if reward_fns is not None else [lambda x: 1.0],
        assembler=self.assembler,
        **kwargs,
    )

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
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      begin_steps = []
      end_steps = []

      def on_begin(step):
        begin_steps.append(step)

      def on_end(step, result):
        end_steps.append((step, result))

      program = rl_program.StandardRLProgram(
          dataset=["prompt_data_0"],
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
          on_step_begin=on_begin,
          on_step_end=on_end,
      )

      await program.run_async(self.mock_engine, num_steps=1)

      self.assertEqual(program.step, 1)
      self.assertEqual(begin_steps, [0])
      self.assertEqual(end_steps, [(1, "step_done")])
      self.mock_engine.dispatch_rollouts.assert_called_once_with(
          [{"prompt": "prompt_data_0", "prompt_id": "prompt_0"}],
          group_size=2,
          policy_version=0,
      )
      self.mock_engine.train_step.assert_called_once()
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
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
          sync_weights=False,
      )

      await program.run_async(
          self.mock_engine, train_dataset=["override_prompt"], num_steps=1
      )

      self.assertEqual(program.step, 1)
      self.mock_engine.sync_weights.assert_not_called()
      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.policy_version, 1)

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

      dispatch_task = asyncio.create_task(
          program.rollout_dispatch_stage(self.mock_engine)
      )

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
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=TwoMicrobatchAssembler(),
          sync_weights=False,
      )

      for pair_index in range(2):
        item = datatypes.TrajectoryItem(
            pair_index=pair_index,
            group_id="group_0",
            start_step=0,
            traj=datatypes.Trajectory(reward=1.0),
        )
        item.payload = self.mock_algo.create_trainer_payloads.return_value[
            pair_index
        ]
        await program.scored_q.put(item)

      await program.train_stage(self.mock_engine, num_steps=1)

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

      async def rollout_dispatch_stage(self, engine, train_dataset=None):
        del engine, train_dataset
        raise RuntimeError("Rollout worker cluster down!")

    async def _run():
      prog = FailingProgram(
          dataset=["prompt"],
          algo=self.mock_algo,
          assembler=self.assembler,
      )
      with self.assertRaises(RuntimeError) as cm:
        await prog.run_async(self.mock_engine, num_steps=1)
      self.assertIn("Rollout worker cluster down!", str(cm.exception))

    asyncio.run(_run())


  def test_run_synchronous_entry_point(self):
    poll_results = [
        [
            distributed_rl_engine._response_to_trajectory_item(
                _create_rollout_response(
                    "req_0_0", "prompt_0", "group_0", pair_index=0
                )
            ),
            distributed_rl_engine._response_to_trajectory_item(
                _create_rollout_response(
                    "req_0_1", "prompt_0", "group_0", pair_index=1
                )
            ),
        ],
    ]
    call_idx = 0

    async def mock_poll(timeout_s=0.1):
      del timeout_s
      nonlocal call_idx
      if call_idx < len(poll_results):
        res = poll_results[call_idx]
        call_idx += 1
        return res
      await asyncio.sleep(0.01)
      return []

    self.mock_engine.poll_rollouts.side_effect = mock_poll

    program = rl_program.StandardRLProgram(
        algo=self.mock_algo,
        reward_fns=[lambda x: 2.0],
        assembler=self.assembler,
    )

    program.run(
        self.mock_engine, train_dataset=["sync_prompt"], num_steps=1
    )

    self.assertEqual(program.step, 1)
    self.assertIsNotNone(program.last_step_result)
    self.assertEqual(program.last_step_result.num_rollouts, 2)
    self.assertEqual(program.last_step_result.reward_mean, 1.0)

  def test_run_with_existing_running_loop(self):
    async def _run():
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
      )

      program.run(
          self.mock_engine, train_dataset=["async_prompt"], num_steps=1
      )
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
      with self.assertRaises(ValueError) as cm:
        await program.run_async(self.mock_engine, num_steps=1)
      self.assertIn("requires a dataset", str(cm.exception))

    asyncio.run(_run())

  def test_prompt_dictionary_id_and_group_extraction(self):
    async def _run():
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "custom_p0", "custom_g0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "custom_p0", "custom_g0", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
      )

      dict_item = {
          "prompt_id": "custom_p0",
          "group_id": "custom_g0",
          "data": "test",
      }
      await program.run_async(
          self.mock_engine, train_dataset=[dict_item], num_steps=1
      )

      self.mock_engine.dispatch_rollouts.assert_called_once_with(
          [dict_item],
          group_size=2,
          policy_version=0,
      )

    asyncio.run(_run())

  def test_multi_group_mini_batch_gradient_accumulation(self):
    async def _run():
      self.mock_algo.mini_batch_size = 2
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_1_0", "prompt_1", "group_1", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_1_1", "prompt_1", "group_1", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
      )

      await program.run_async(
          self.mock_engine, train_dataset=["p0", "p1"], num_steps=1
      )

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

      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
      )

      await program.run_async(
          self.mock_engine, train_dataset=["prompt_0"], num_steps=1
      )

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

      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_0", "prompt_0", "group_0", pair_index=0
                  )
              ),
              distributed_rl_engine._response_to_trajectory_item(
                  _create_rollout_response(
                      "req_0_1", "prompt_0", "group_0", pair_index=1
                  )
              ),
          ],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        del timeout_s
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      program = rl_program.StandardRLProgram(
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
      )

      with self.assertRaises(TypeError) as cm:
        await program.run_async(
            self.mock_engine, train_dataset=["prompt_0"], num_steps=1
        )
      self.assertIn("Reference KL requires an assembler", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_handles_early_dispatch_completion(self):
    async def _run():
      self._set_mock_poll_batches(self._make_trajectory_group())
      program = self._create_program()
      await program.run_async(self.mock_engine, num_steps=1)
      self.assertEqual(program.step, 1)

    asyncio.run(_run())

  def test_run_async_propagates_train_stage_exception(self):
    async def _run():
      self._set_mock_poll_batches(self._make_trajectory_group())
      self.mock_engine.train_step.side_effect = RuntimeError(
          "Training worker OOM"
      )
      program = self._create_program()

      with self.assertRaises(RuntimeError) as cm:
        await program.run_async(self.mock_engine, num_steps=1)
      self.assertIn("Training worker OOM", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_propagates_critique_stage_exception(self):
    async def _run():
      self._set_mock_poll_batches(self._make_trajectory_group())
      def failing_reward_fn(_):
        raise ValueError("Reward model computation failed")

      program = self._create_program(reward_fns=[failing_reward_fn])

      with self.assertRaises(ValueError) as cm:
        await program.run_async(self.mock_engine, num_steps=1)
      self.assertIn("Reward model computation failed", str(cm.exception))

    asyncio.run(_run())

  def test_run_async_cancels_background_stages_on_external_cancellation(self):
    async def _run():
      self._set_mock_poll_batches()  # Yields empty and sleeps
      program = self._create_program()

      task = asyncio.create_task(
          program.run_async(self.mock_engine, num_steps=5)
      )
      await asyncio.sleep(0.02)
      task.cancel()

      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
