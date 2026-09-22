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

"""Unit tests for launcher and orchestrator config wiring."""

import os
from unittest import mock

from absl.testing import absltest
import tunix
from tunix.experimental.orchestrator import rl_program


def _launcher_text(example_dir: str = "math_gsm8k_dist") -> str:
  path = os.path.join(
      os.path.dirname(os.path.abspath(tunix.__file__)),
      "experimental",
      "examples",
      example_dir,
      "launcher.sh",
  )
  with open(path) as f:
    return f.read()


def _cmd_block(name: str, example_dir: str = "math_gsm8k_dist") -> str:
  text = _launcher_text(example_dir)
  start = text.index(f"{name}=(")
  return text[start : text.index("\n    )", start)]


def _metrics_summary(trainer_metrics):
  program = object.__new__(rl_program.StandardRLProgram)
  program.metrics_logger = mock.Mock()
  program.metrics_prefix = "test"
  program.mode = rl_program.Mode.TRAIN
  program.policy_version = 0
  return rl_program.StandardRLProgram._collect_and_log_step_metrics(  # pylint: disable=protected-access
      program,
      all_step_items=[],
      step_rewards=[1.0],
      step_advantages=[0.0],
      trainer_metrics=trainer_metrics,
      num_rollouts=1,
      num_microbatches=1,
      step_time_sec=1.0,
      consumed_policy_version=0,
      log_step=0,
  )


class SamplingParamsTest(absltest.TestCase):
  """Tests for generation sampling parameters in the launcher."""

  def test_sampling_params_are_bound_to_their_variables(self):
    for example_dir in ("math_gsm8k_dist", "deepswe_dist"):
      text = _launcher_text(example_dir)
      for flag, var in (
          ("--temperature", "TEMPERATURE"),
          ("--top_p", "TOP_P"),
          ("--top_k", "TOP_K"),
      ):
        self.assertIn(
            f'{flag}="${var}"',
            text,
            f"{example_dir}: {flag} is not bound to ${var}",
        )

  def test_inference_node_receives_the_sampling_temperature(self):
    for example_dir in ("math_gsm8k_dist", "deepswe_dist"):
      self.assertIn(
          '--temperature="$TEMPERATURE"',
          _cmd_block("INFERENCE_CMD", example_dir),
      )


class GradNormMetricTest(absltest.TestCase):
  """Tests for grad_norm in the train-step metrics summary."""

  def test_grad_norm_is_returned_in_the_summary(self):
    summary = _metrics_summary({"loss": 0.5, "grad_norm": 1.25})
    self.assertEqual(summary["grad_norm_val"], 1.25)

  def test_grad_norm_is_none_when_the_trainer_does_not_report_it(self):
    summary = _metrics_summary({"loss": 0.5})
    self.assertIsNone(summary["grad_norm_val"])


if __name__ == "__main__":
  absltest.main()
