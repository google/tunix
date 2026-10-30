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

"""GSM8K OPD configuration and teacher construction without worker processes."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.distillation import teacher
from tunix.dist_rl.examples.math_gsm8k_dist import run_gsm8k_dist_grpo as entry


class OnPolicyDistillationCliTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ, {}, clear=True))

  def _args(self, *extra):
    return entry._parse_args([
        "--opd_teacher_url=http://teacher:30001",
        "--opd_teacher_tokenizer_path=/snapshots/fixed teacher",
        "--weight_sync_mode=raiden",
        *extra,
    ])

  def test_disabled_path_does_not_load_teacher_resources(self):
    args = entry._parse_args([])
    entry._validate_opd_args(args)
    with mock.patch.object(entry.AutoTokenizer, "from_pretrained") as load:
      with mock.patch.object(teacher, "SGLangTeacherScorer") as scorer:
        self.assertEqual(entry._opd_program_kwargs(args, object()), {})
    load.assert_not_called()
    scorer.assert_not_called()

  def test_requires_explicit_teacher_tokenizer(self):
    args = entry._parse_args(["--opd_teacher_url=http://teacher:30001"])
    with self.assertRaisesRegex(ValueError, "teacher_tokenizer_path"):
      entry._validate_opd_args(args)

  @parameterized.parameters(
      ("--temperature=0.7", "temperature"),
      ("--top_p=0.9", "top_p"),
      ("--top_k=10", "top_k"),
      ("--top_k=-2", "top_k"),
      ("--no-use_rollout_logps", "use_rollout_logps"),
      ("--max_staleness=1", "max_staleness"),
      ("--in_flight_weight_updates=true", "in_flight_weight_updates"),
      ("--weight_sync_mode=none", "weight_sync_mode"),
      ("--weight_sync_mode=fallback", "weight_sync_mode"),
  )
  def test_rejects_unsupported_behavior(self, flag, message):
    with self.assertRaisesRegex(ValueError, message):
      entry._validate_opd_args(self._args(flag))

  @parameterized.parameters(-1, 0)
  def test_accepts_disabled_top_k_and_independent_reference_beta(self, top_k):
    entry._validate_opd_args(self._args(f"--top_k={top_k}", "--beta=0.04"))

  def test_teacher_settings_reach_the_scorer_and_program(self):
    student_tokenizer = mock.Mock()
    configured_scorer = mock.Mock(tokenizer_fingerprint="tokenizer-v1")
    teacher_tokenizer = mock.Mock(pad_token_id=None, eos_token="<eos>")
    args = self._args(
        "--opd_coefficient=0.5",
        "--opd_task_reward_coefficient=1",
        "--opd_teacher_timeout_seconds=12.5",
        "--opd_max_concurrency=3",
    )
    with mock.patch.object(
        entry.AutoTokenizer, "from_pretrained", return_value=teacher_tokenizer
    ) as load:
      with mock.patch.object(
          teacher, "SGLangTeacherScorer", return_value=configured_scorer
      ) as scorer:
        kwargs = entry._opd_program_kwargs(args, student_tokenizer)

    load.assert_called_once_with(
        "/snapshots/fixed teacher", trust_remote_code=True
    )
    scorer.assert_called_once_with(
        "http://teacher:30001",
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        timeout_seconds=12.5,
        max_concurrency=3,
    )
    self.assertEqual(teacher_tokenizer.pad_token, "<eos>")
    self.assertEqual(kwargs["opd_config"].coefficient, 0.5)
    self.assertEqual(kwargs["opd_config"].task_reward_coefficient, 1)
    self.assertIs(kwargs["teacher_scorer"], configured_scorer)

  def test_incompatible_teacher_stops_configuration(self):
    with mock.patch.object(entry.AutoTokenizer, "from_pretrained"):
      with mock.patch.object(
          teacher, "SGLangTeacherScorer", side_effect=ValueError("token IDs")
      ):
        with self.assertRaisesRegex(ValueError, "token IDs"):
          entry._opd_program_kwargs(self._args(), object())


if __name__ == "__main__":
  absltest.main()
