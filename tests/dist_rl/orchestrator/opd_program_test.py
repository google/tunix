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

"""CPU integration tests for OPD through critique, GRPO and batch assembly."""

import asyncio
from collections.abc import Sequence
import unittest
from unittest import mock

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.distillation import on_policy
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_program
from tunix.rl import algorithm_config


def _group(rewards=(0.0, 0.0), *, second_valid=True):
  """Two responses, including a non-assistant token in the first response."""
  items = []
  for index, (tokens, mask, old_logps) in enumerate((
      ([11, 12, 13], [1, 0, 1], [-1.0, -2.0, -3.0]),
      ([21, 22], [1, 1], [-2.0, -3.0]),
  )):
    items.append(
        datatypes.TrajectoryItem(
            prompt_id="p0",
            group_index=index,
            is_valid=index == 0 or second_valid,
            traj={
                "prompt_tokens": np.array([1, 2], dtype=np.int32),
                "conversation_tokens": np.array(tokens, dtype=np.int32),
                "conversation_masks": np.array(mask, dtype=np.float32),
                "old_logprobs": np.array(old_logps, dtype=np.float32),
                "trajectory_reward": rewards[index],
                "status": datatypes.TrajectoryStatus.SUCCEEDED,
                "conversation_text": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "policy_version": 0,
            },
        )
    )
  return items


class _RecordingTeacher:
  """Async teacher with known token scores and no model or network."""

  def __init__(self):
    self.calls = []

  async def score(
      self, payloads: Sequence[datatypes.RLTrainerPayload]
  ) -> Sequence[np.ndarray]:
    self.calls.append(tuple(payloads))
    return [
        np.array([-0.5, -100.0, -4.0], dtype=np.float32),
        np.array([-3.0, -2.0], dtype=np.float32),
    ]


class OPDProgramTest(unittest.IsolatedAsyncioTestCase):

  def _program(
      self,
      *,
      opd_config=None,
      teacher_scorer=None,
      packed=False,
      num_generations=2,
      algorithm_options=None,
      generation_args=None,
      max_staleness=0,
      sync_weights=True,
  ):
    options = dict(
        num_generations=num_generations,
        beta=0.0,
        use_rollout_logps=True,
        reward_mode="env",
    )
    options.update(algorithm_options or {})
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(**options),
        mini_batch_size=1,
        train_micro_batch_size=2,
        max_packed_len=16,
        max_response_length=4,
    )
    if packed:
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=1,
          num_generations=num_generations,
          mini_batch_size=1,
          max_packed_len=16,
      )
    else:
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          num_generations=num_generations,
          mini_batch_size=1,
          max_prompt_length=3,
          max_response_length=4,
          pad_id=0,
      )
    with mock.patch.object(rl_program, "MetricsLogger"):
      program = rl_program.StandardRLProgram(
          algo=adapter,
          assembler=assembler,
          generation_args=generation_args
          or datatypes.GenerationArgs(temperature=1.0, top_p=1.0, top_k=0),
          max_staleness=max_staleness,
          sync_weights=sync_weights,
          opd_config=opd_config,
          teacher_scorer=teacher_scorer,
      )
    self.addCleanup(program.close)
    return program

  async def _enqueue(self, program, group):
    program.engine = mock.Mock()
    for item in group:
      await program.raw_q.put(item)
    await program.raw_q.close()

  async def _critique(self, program, group):
    await self._enqueue(program, group)
    await program.critique_stage()
    scored = await program.scored_q.get()
    self.assertIsNotNone(scored)
    self.assertIsNone(await program.scored_q.get())
    return scored

  async def test_pure_opd_survives_real_padded_and_packed_assemblers(self):
    expected = {11: 0.5, 13: -1.0, 21: -1.0, 22: 1.0}
    for packed in (False, True):
      with self.subTest(packed=packed):
        teacher = _RecordingTeacher()
        program = self._program(
            opd_config=on_policy.OnPolicyDistillationConfig(),
            teacher_scorer=teacher,
            packed=packed,
        )
        scored = await self._critique(program, _group())
        self.assertEqual(scored.rewards, (0.0, 0.0))
        self.assertEqual(len(scored.payloads), 2)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(len(teacher.calls[0]), 2)
        np.testing.assert_array_equal(teacher.calls[0][0].prompt_ids, [1, 2])
        np.testing.assert_array_equal(
            teacher.calls[0][0].completion_ids, [11, 12, 13]
        )
        np.testing.assert_array_equal(teacher.calls[0][0].advantages, 0.0)
        np.testing.assert_array_equal(
            scored.payloads[0].advantages, [0.5, 0.0, -1.0]
        )
        batches = list(program.assembler.feed(scored.payloads))
        batches.extend(program.assembler.flush())
        self.assertTrue(batches)
        observed = {}
        for assembled in batches:
          payload = assembled.payload
          mask = np.asarray(payload.completion_mask, dtype=bool)
          advantages = np.asarray(payload.advantages)
          np.testing.assert_array_equal(advantages[~mask], 0.0)
          for token, advantage in zip(
              np.asarray(payload.completion_ids)[mask], advantages[mask]
          ):
            observed[int(token)] = float(advantage)
        self.assertEqual(observed, expected)

  async def test_task_rewards_remain_telemetry_in_pure_opd(self):
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=_RecordingTeacher(),
    )
    scored = await self._critique(program, _group(rewards=(3.0, 9.0)))
    self.assertEqual(scored.rewards, (3.0, 9.0))
    self.assertEqual(
        tuple(item.traj["trajectory_reward"] for item in scored.items),
        (3.0, 9.0),
    )
    np.testing.assert_array_equal(
        scored.payloads[0].advantages, [0.5, 0.0, -1.0]
    )

  async def test_zero_coefficient_scales_task_without_teacher_call(self):
    baseline = await self._critique(self._program(), _group((3.0, 9.0)))
    teacher = mock.Mock()
    teacher.score = mock.AsyncMock(side_effect=AssertionError("unused"))
    for scorer in (None, teacher):
      with self.subTest(with_scorer=scorer is not None):
        program = self._program(
            opd_config=on_policy.OnPolicyDistillationConfig(
                coefficient=0.0, task_reward_coefficient=0.25
            ),
            teacher_scorer=scorer,
        )
        scored = await self._critique(program, _group((3.0, 9.0)))
        self.assertEqual(scored.rewards, baseline.rewards)
        for actual, original in zip(scored.payloads, baseline.payloads):
          mask = np.asarray(original.completion_mask, dtype=bool)
          np.testing.assert_allclose(
              np.asarray(actual.advantages)[mask],
              0.25 * np.asarray(original.advantages)[mask],
          )
          np.testing.assert_array_equal(
              actual.completion_mask, original.completion_mask
          )
    teacher.score.assert_not_awaited()

  async def test_task_and_teacher_advantages_are_combined_after_grpo(self):
    baseline = await self._critique(self._program(), _group((3.0, 9.0)))
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(
            coefficient=0.5, task_reward_coefficient=0.25
        ),
        teacher_scorer=_RecordingTeacher(),
    )
    scored = await self._critique(program, _group((3.0, 9.0)))
    guidance = (np.array([0.5, 0.0, -1.0]), np.array([-1.0, 1.0]))
    for actual, original, signal in zip(
        scored.payloads, baseline.payloads, guidance
    ):
      mask = np.asarray(actual.completion_mask, dtype=bool)
      np.testing.assert_allclose(
          np.asarray(actual.advantages)[mask],
          0.25 * np.asarray(original.advantages)[mask] + 0.5 * signal[mask],
      )

  async def test_opd_cannot_reactivate_degenerate_grpo_group(self):
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=_RecordingTeacher(),
    )
    scored = await self._critique(program, _group(second_valid=False))
    self.assertEqual(len(scored.payloads), 2)
    for payload in scored.payloads:
      np.testing.assert_array_equal(payload.completion_mask, 0.0)
      np.testing.assert_array_equal(payload.advantages, 0.0)

  async def test_teacher_failure_propagates_and_closes_scored_queue(self):
    teacher = mock.Mock()
    teacher.score = mock.AsyncMock(side_effect=RuntimeError("teacher offline"))
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=teacher,
    )
    await self._enqueue(program, _group())
    with self.assertRaisesRegex(RuntimeError, "teacher offline"):
      await program.critique_stage()
    self.assertIsNone(await program.scored_q.get())

  async def test_teacher_response_alignment_failure_is_not_swallowed(self):
    teacher = mock.Mock()
    teacher.score = mock.AsyncMock(return_value=[np.zeros(1, np.float32)])
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=teacher,
    )
    await self._enqueue(program, _group())
    with self.assertRaises(ValueError):
      await program.critique_stage()
    self.assertIsNone(await program.scored_q.get())

  async def test_teacher_is_awaited_without_blocking_other_tasks(self):
    started, release = asyncio.Event(), asyncio.Event()
    teacher = _RecordingTeacher()
    original_score = teacher.score

    async def gated_score(payloads):
      started.set()
      await release.wait()
      return await original_score(payloads)

    teacher.score = gated_score
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=teacher,
    )
    await self._enqueue(program, _group())
    critique = asyncio.create_task(program.critique_stage())
    try:
      await asyncio.wait_for(started.wait(), timeout=5.0)
      self.assertFalse(critique.done())
      await asyncio.sleep(0)
      release.set()
      await asyncio.wait_for(critique, timeout=5.0)
      self.assertEqual(len(teacher.calls), 1)
      self.assertIsNotNone(await program.scored_q.get())
    finally:
      release.set()
      if not critique.done():
        critique.cancel()
      await asyncio.gather(critique, return_exceptions=True)

  def test_rejects_unsupported_opd_sampling_and_schedule(self):
    invalid = (
        {
            "generation_args": datatypes.GenerationArgs(
                temperature=0.8, top_p=1.0, top_k=0
            )
        },
        {
            "generation_args": datatypes.GenerationArgs(
                temperature=1.0, top_p=0.9, top_k=0
            )
        },
        {
            "generation_args": datatypes.GenerationArgs(
                temperature=1.0, top_p=1.0, top_k=32
            )
        },
        {
            "generation_args": datatypes.GenerationArgs(
                temperature=1.0, top_p=1.0
            )
        },
        {"algorithm_options": {"use_rollout_logps": False}},
        {"algorithm_options": {"exact_token_continuity": False}},
        {"max_staleness": 1},
        {"sync_weights": False},
        {"num_generations": 1},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises(ValueError):
          self._program(
              opd_config=on_policy.OnPolicyDistillationConfig(),
              teacher_scorer=_RecordingTeacher(),
              **options,
          )

  async def test_rejects_per_prompt_distribution_overrides_before_dispatch(
      self,
  ):
    for overrides in (
        {"temperature": 0.8},
        {"top_p": 0.9},
        {"top_k": 4},
        {"return_logprobs": False},
        {"return_logprobs": None},
        {"return_logprobs": 0},
        {"require_unfiltered": False},
        {"require_unfiltered": None},
        {"require_unfiltered": 0},
        {"require_unfiltered": 1},
        {"forbidden_tokens": [11]},
        {"bad_words": ["filtered"]},
        {"pad_output": True},
        {"echo": True},
        {"beam_size": 2},
        {"beam_size": 1},
        {"min_tokens": 1},
        {"structured_outputs": {}},
        {"guided_decoding": {}},
        {"best_of": 2},
        {"n": 2},
        {"multi_sampling": 2},
        {"images": ["image.png"]},
        {"audios": [[0.1, 0.2]]},
    ):
      with self.subTest(overrides=overrides):
        program = self._program(
            opd_config=on_policy.OnPolicyDistillationConfig(),
            teacher_scorer=_RecordingTeacher(),
        )
        program.dataset = [{
            "prompt_id": "override",
            "prompt": "question",
            "generation_kwargs": overrides,
        }]
        program.engine = mock.Mock()
        program.engine.dispatch_rollouts = mock.AsyncMock()
        with self.assertRaises(ValueError):
          await program.rollout_dispatch_stage()
        program.engine.dispatch_rollouts.assert_not_awaited()

  async def test_opd_enforces_unfiltered_sampling_and_preserves_normal_rl(self):
    for enabled in (False, True):
      for incoming_flag in (None, False, True):
        with self.subTest(enabled=enabled, incoming_flag=incoming_flag):
          program = self._program(
              opd_config=(
                  on_policy.OnPolicyDistillationConfig() if enabled else None
              ),
              teacher_scorer=_RecordingTeacher() if enabled else None,
              generation_args=datatypes.GenerationArgs(
                  temperature=1.0,
                  top_p=1.0,
                  top_k=0,
                  require_unfiltered=incoming_flag,
              ),
          )
          program.dataset = ["question"]
          program.engine = mock.Mock()
          program.engine.dispatch_rollouts = mock.AsyncMock()
          await program.rollout_dispatch_stage()
          dispatched = program.engine.dispatch_rollouts.call_args.kwargs[
              "generation_args"
          ]
          self.assertIs(
              dispatched.require_unfiltered, True if enabled else incoming_flag
          )
          if not enabled and incoming_flag is None:
            self.assertNotIn("require_unfiltered", dispatched.as_kwargs())

  async def test_per_prompt_neutral_settings_keep_unfiltered_contract(self):
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=_RecordingTeacher(),
    )
    program.dataset = [{
        "prompt_id": "neutral",
        "prompt": "question",
        "generation_kwargs": {
            "require_unfiltered": True,
            "min_tokens": 0,
            "bad_words": [],
            "best_of": 1,
            "n": 1,
            "multi_sampling": 1,
        },
    }]
    program.engine = mock.Mock()
    program.engine.dispatch_rollouts = mock.AsyncMock()
    await program.rollout_dispatch_stage()
    program.engine.dispatch_rollouts.assert_awaited_once()

  async def test_rejects_context_truncation_before_contacting_teacher(self):
    for field in ("prompt", "completion", "packed"):
      with self.subTest(field=field):
        teacher = _RecordingTeacher()
        program = self._program(
            opd_config=on_policy.OnPolicyDistillationConfig(),
            teacher_scorer=teacher,
            packed=field == "packed",
        )
        group = _group()
        for item in group:
          if field in ("prompt", "packed"):
            item.traj["prompt_tokens"] = np.arange(
                20 if field == "packed" else 4, dtype=np.int32
            )
          else:
            item.traj["conversation_tokens"] = np.arange(5, dtype=np.int32)
            item.traj["conversation_masks"] = np.ones(5, dtype=np.float32)
            item.traj["old_logprobs"] = np.full(5, -1, dtype=np.float32)
        await self._enqueue(program, group)
        with self.assertRaisesRegex(ValueError, "truncat|packing budget"):
          await program.critique_stage()
        self.assertEqual(teacher.calls, [])

  async def test_attention_validity_retains_masked_tool_and_real_pad_tokens(
      self,
  ):
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=_RecordingTeacher(),
    )
    group = _group()
    # The zero is an actual context token, not a padded array element. The
    # middle completion is a tool observation with zero policy-loss weight.
    group[0].traj["prompt_tokens"] = np.array([1, 0], dtype=np.int32)
    group[0].traj["conversation_tokens"] = np.array([11, 0, 13], dtype=np.int32)
    scored = await self._critique(program, group)
    batches = list(program.assembler.feed(scored.payloads))
    batches.extend(program.assembler.flush())
    payload = program._with_opd_attention_mask(
        batches[0].payload, batches[0].padding_stats
    )
    np.testing.assert_array_equal(payload.prompt_mask[0], [0, 1, 1])
    np.testing.assert_array_equal(payload.completion_mask[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(
        payload.completion_attention_mask[0], [True, True, True, False]
    )
    np.testing.assert_array_equal(
        payload.completion_attention_mask[1], [True, True, False, False]
    )

    program.engine = mock.Mock()
    program.engine.per_token_logps = mock.AsyncMock(
        return_value=datatypes.LogprobsResponse(
            per_token_logps=np.full(payload.completion_ids.shape, -1.0)
        )
    )
    await program._apply_sampler_trainer_agreement(payload, {})
    request = program.engine.per_token_logps.call_args.kwargs["items"]
    np.testing.assert_array_equal(request.token_mask[0], [0, 1, 1, 1, 1, 1, 0])

  async def test_opd_metrics_weight_tokens_across_unequal_groups(self):
    program = self._program(
        opd_config=on_policy.OnPolicyDistillationConfig(),
        teacher_scorer=_RecordingTeacher(),
    )
    scored = await self._critique(program, _group())
    values, reduce_fn = scored.reward_log_metrics["opd/reverse_kl"]
    # This real group has four assistant tokens with total reverse KL 0.5.
    # A second, shorter group contributes total 3 over two tokens.
    self.assertAlmostEqual(float(reduce_fn(values)), 0.125)
    self.assertAlmostEqual(
        float(reduce_fn(np.concatenate([values, [[3.0, 2.0]]]))),
        3.5 / 6.0,
    )

  def test_rejects_missing_config_or_required_teacher(self):
    with self.assertRaises(ValueError):
      self._program(teacher_scorer=_RecordingTeacher())
    with self.assertRaises(ValueError):
      self._program(opd_config=on_policy.OnPolicyDistillationConfig())

  def test_rejects_non_grpo_adapter(self):
    with self.assertRaises(ValueError):
      rl_program.StandardRLProgram(
          algo=algorithm_adapter.PPOAdapter(num_generations=2),
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, top_p=1.0, top_k=0
          ),
          opd_config=on_policy.OnPolicyDistillationConfig(),
          teacher_scorer=_RecordingTeacher(),
      )


if __name__ == "__main__":
  unittest.main()
