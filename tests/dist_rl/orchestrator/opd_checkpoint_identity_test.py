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

"""OPD checkpoint identity through real program initialization and stages."""

import copy
import unittest
from unittest import mock

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.distillation import identity
from tunix.dist_rl.distillation import on_policy
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_program
from tunix.rl import algorithm_config


class OPDCheckpointIdentityTest(unittest.IsolatedAsyncioTestCase):

  def _program(self, *, enabled=True, coefficient=1.0, declared=True, top_k=0):
    self.scorer = mock.Mock()
    self.scorer.score = mock.AsyncMock(
        return_value=[np.array([-0.5]), np.array([-1.5])]
    )
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(
            num_generations=2, beta=0, use_rollout_logps=True, reward_mode="env"
        ),
        mini_batch_size=1,
        train_micro_batch_size=2,
        max_response_length=2,
    )
    with mock.patch.object(rl_program, "MetricsLogger"):
      program = rl_program.StandardRLProgram(
          algo=adapter,
          dataset=["question"],
          max_steps=1,
          assembler=batch_assembly.PaddedBatchAssembler(
              batch_size=2,
              num_generations=2,
              mini_batch_size=1,
              max_prompt_length=2,
              max_response_length=2,
              pad_id=0,
          ),
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, top_p=1.0, top_k=top_k
          ),
          opd_config=(
              on_policy.OnPolicyDistillationConfig(coefficient=coefficient)
              if enabled
              else None
          ),
          teacher_scorer=self.scorer if enabled and coefficient > 0 else None,
          opd_teacher_identity=(
              identity.TeacherIdentity(
                  declared_revision="sha256:teacher-v1",
                  tokenizer_fingerprint="tokenizer-v1",
              )
              if declared and enabled
              else None
          ),
      )
    self.addCleanup(program.close)
    return program

  def _engine(self, *, step=3, metadata=None):
    engine = mock.Mock()
    engine.resume_from_checkpoint = mock.AsyncMock(return_value=step)
    engine.restored_checkpoint_metadata = metadata
    engine.prepare_rollout_policy = mock.AsyncMock()
    engine.dispatch_rollouts = mock.AsyncMock()
    engine.save_checkpoint = mock.AsyncMock()
    engine.train_step = mock.AsyncMock(return_value={"train_step": 1})
    engine.get_metrics = mock.AsyncMock(return_value=None)
    engine.sync_weights = mock.AsyncMock(return_value=1)
    return engine

  async def test_matches_resolved_generation_contract(self):
    program = self._program()
    self.assertIs(program._opd_manifest["sampling"]["return_logprobs"], True)
    self.assertIs(program.generation_args.return_logprobs, True)
    self.assertIs(program._opd_manifest["sampling"]["require_unfiltered"], True)
    metadata = identity.checkpoint_metadata(program._opd_manifest)
    metadata["policy_version"] = 4
    program.engine = self._engine(metadata=metadata)
    await program._resume_from_checkpoint()
    self.assertEqual(program.step, 3)
    self.assertEqual(program.policy_version, 4)

  async def test_mismatch_rejected_before_dispatch_or_teacher_request(self):
    for field, value in (
        (("coefficient",), 0.25),
        (("task_reward_coefficient",), 1.0),
        (("sampling", "return_logprobs"), False),
        (("teacher", "declared_revision"), "sha256:teacher-v2"),
        (("teacher", "tokenizer_fingerprint"), "tokenizer-v2"),
    ):
      with self.subTest(field=field):
        program = self._program()
        stored = copy.deepcopy(program._opd_manifest)
        target = stored
        for part in field[:-1]:
          target = target[part]
        target[field[-1]] = value
        engine = self._engine(metadata=identity.checkpoint_metadata(stored))
        with self.assertRaisesRegex(ValueError, "objective differs"):
          await program.run_async(engine)
        self.assertEqual(program.step, 0)
        engine.configure_worker.assert_not_called()
        engine.prepare_rollout_policy.assert_not_awaited()
        engine.dispatch_rollouts.assert_not_awaited()
        self.scorer.score.assert_not_awaited()

  async def test_legacy_checkpoint_cannot_silently_enable_opd(self):
    program = self._program()
    program.engine = self._engine(metadata={"step": 3})
    with self.assertRaisesRegex(ValueError, "no opd_manifest"):
      await program._resume_from_checkpoint()

  async def test_opd_checkpoint_cannot_silently_disable_opd(self):
    enabled = self._program()
    for step in (0, 3):
      with self.subTest(step=step):
        disabled = self._program(enabled=False)
        disabled.engine = self._engine(
            step=step,
            metadata=identity.checkpoint_metadata(enabled._opd_manifest),
        )
        with self.assertRaisesRegex(ValueError, "OPD disabled"):
          await disabled._resume_from_checkpoint()

  async def test_incomplete_identity_allows_fresh_run_but_not_resume(self):
    program = self._program(declared=False)
    program.engine = self._engine(step=0, metadata=None)
    await program._resume_from_checkpoint()
    program.engine = self._engine(
        metadata=identity.checkpoint_metadata(program._opd_manifest)
    )
    with self.assertRaisesRegex(ValueError, "declared teacher revision"):
      await program._resume_from_checkpoint()

  async def test_zero_coefficient_resumes_without_teacher_identity(self):
    program = self._program(coefficient=0, declared=False)
    program.engine = self._engine(
        metadata=identity.checkpoint_metadata(program._opd_manifest)
    )
    await program._resume_from_checkpoint()
    self.assertEqual(program.step, 3)

  async def test_equivalent_top_k_configuration_can_resume(self):
    original = self._program(top_k=-1)
    program = self._program(top_k=0)
    program.engine = self._engine(
        metadata=identity.checkpoint_metadata(original._opd_manifest)
    )
    await program._resume_from_checkpoint()

  async def test_real_critique_and_train_checkpoint_records_manifest(self):
    for enabled in (False, True):
      with self.subTest(enabled=enabled):
        program = self._program(enabled=enabled)
        program.engine = self._engine(step=0)
        for index in range(2):
          await program.raw_q.put(
              datatypes.TrajectoryItem(
                  prompt_id="p0",
                  group_index=index,
                  traj={
                      "prompt_tokens": np.array([1], dtype=np.int32),
                      "conversation_tokens": np.array([2], dtype=np.int32),
                      "conversation_masks": np.array([1], dtype=np.float32),
                      "old_logprobs": np.array([-1], dtype=np.float32),
                      "trajectory_reward": 0,
                      "status": datatypes.TrajectoryStatus.SUCCEEDED,
                      "conversation_text": [
                          {"role": "user", "content": "question"},
                          {"role": "assistant", "content": "answer"},
                      ],
                  },
              )
          )
        await program.raw_q.close()
        await program.critique_stage()
        await program.train_stage()
        program.engine.save_checkpoint.assert_awaited_once()
        metadata = program.engine.save_checkpoint.call_args.kwargs["metadata"]
        self.assertEqual(identity.METADATA_KEY in metadata, enabled)
        if enabled:
          identity.validate_resume_manifest(program._opd_manifest, metadata)
          self.scorer.score.assert_awaited_once()


if __name__ == "__main__":
  unittest.main()
