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

"""Unit tests for DeepSWE evaluation utilities."""

import asyncio
import logging
from unittest import mock

from absl.testing import absltest
from examples.deepswe import deepswe_utils
from examples.deepswe import eval_utils
from examples.deepswe import swe_env
from tunix.rl.agentic.agents import agent_types


class EvalUtilsTest(absltest.TestCase):

  def test_str2bool(self):
    self.assertTrue(deepswe_utils.str2bool("true"))
    self.assertTrue(deepswe_utils.str2bool("1"))
    self.assertFalse(deepswe_utils.str2bool("false"))
    self.assertFalse(deepswe_utils.str2bool("0"))
    self.assertTrue(deepswe_utils.str2bool(True))

  def test_is_prompt_overflow_error(self):
    self.assertTrue(
        eval_utils._is_prompt_overflow_error(
            ValueError("Prompt too long before sampler call")
        )
    )
    self.assertTrue(
        eval_utils._is_prompt_overflow_error(
            ValueError("input_tokens=33000 exceeds max_model_len=32768")
        )
    )
    self.assertFalse(
        eval_utils._is_prompt_overflow_error(
            ValueError("input_tokens=100 without model len error")
        )
    )

  def test_estimate_pass_at_k(self):
    self.assertIsNone(eval_utils._estimate_pass_at_k(n=2, c=1, k=4))
    self.assertAlmostEqual(eval_utils._estimate_pass_at_k(n=4, c=4, k=1), 1.0)
    self.assertAlmostEqual(eval_utils._estimate_pass_at_k(n=4, c=0, k=1), 0.0)
    self.assertAlmostEqual(eval_utils._estimate_pass_at_k(n=4, c=2, k=1), 0.5)
    self.assertAlmostEqual(eval_utils._estimate_pass_at_k(n=4, c=1, k=4), 1.0)

  def test_compute_pass_at_k_logs_unique_instances_and_rollouts(self):
    results = [
        {"instance_id": "inst_1", "reward": 1.0, "num_steps": 3, "status": "OK"},
        {"instance_id": "inst_1", "reward": 0.0, "num_steps": 5, "status": "OK"},
        {"instance_id": "inst_2", "reward": 0.0, "num_steps": 4, "status": "OK"},
        {"instance_id": "inst_2", "reward": 0.0, "num_steps": 4, "status": "OK"},
    ]
    mock_logger = mock.MagicMock(spec=logging.Logger)
    eval_utils.compute_pass_at_k(results, ks=(1, 2), logger=mock_logger)
    logged_messages = [
        call.args[0] % call.args[1:] if len(call.args) > 1 else call.args[0]
        for call in mock_logger.info.call_args_list
    ]
    self.assertTrue(
        any("Total instances:  2 (rollouts: 4)" in m for m in logged_messages)
    )
    self.assertTrue(
        any("Resolved:         1 (rollouts: 1)" in m for m in logged_messages)
    )

  def test_normalize_entry(self):
    raw = {
        "repo_name": "django",
        "commit_hash": "1234567890abcdef",
        "docker_image": "namanjain12/django_final:v1",
        "hints": ["hint1", "hint2"],
    }
    norm = swe_env._normalize_entry(
        raw, docker_image_prefix="us-central1-docker.pkg.dev/proj/repo"
    )
    self.assertEqual(norm["instance_id"], "django__12345678")
    self.assertEqual(
        norm["docker_image"],
        "us-central1-docker.pkg.dev/proj/repo/django_final:v1",
    )
    self.assertEqual(norm["hints"], '["hint1", "hint2"]')

  def test_eval_trajectory_collect_engine_prompt_overflow(self):
    mock_agent = mock.MagicMock()
    step = agent_types.Step(observation="obs")
    mock_agent.trajectory = agent_types.Trajectory(steps=[step])
    mock_env = mock.MagicMock()
    mock_env.extra_kwargs = {"pair_index": 0}
    mock_env.entry = {"instance_id": "inst_0"}

    class OverflowEngine(eval_utils.EvalTrajectoryCollectEngine):
      skip_final_reward_on_overflow = True

    engine = OverflowEngine(
        agent=mock_agent,
        env=mock_env,
        model_call=mock.MagicMock(),
    )
    with mock.patch.object(
        eval_utils.trajectory_collect_engine.TrajectoryCollectEngine,
        "_one_step",
        side_effect=eval_utils.PromptTooLongError("too long"),
    ):
      done = asyncio.run(engine._one_step())

    self.assertTrue(done)
    self.assertEqual(
        mock_agent.trajectory.status,
        agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
    )
    self.assertTrue(step.done)
    traj = engine.compute_trajectory_reward()
    self.assertEqual(traj.reward, 0.0)


if __name__ == "__main__":
  absltest.main()
