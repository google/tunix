"""Unit tests for DeepsweRLProgram."""

import sys
import types
import unittest
from unittest import mock

# Mock heavyweight Kubernetes/OpenAI agent components before imports
mock_env = types.ModuleType("swe_env")
mock_env.SWEEnv = mock.MagicMock()
mock_env._init_global_fleet = mock.MagicMock()

mock_agent = types.ModuleType("swe_agent")
mock_agent.SWEAgent = mock.MagicMock()

sys.modules["tunix.oss.examples.deepswe.swe_env"] = mock_env
sys.modules["tunix.oss.examples.deepswe.swe_agent"] = mock_agent
sys.modules["examples.deepswe.swe_env"] = mock_env
sys.modules["examples.deepswe.swe_agent"] = mock_agent

swe_env = mock_env
swe_agent = mock_agent

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.deepswe_dist.deepswe_rl_program import DeepsweRLProgram  # pylint: disable=g-import-not-at-top


class TestDeepsweRLProgram(unittest.IsolatedAsyncioTestCase):
  """Tests DeepsweRLProgram multi-turn episode logic."""

  async def test_run_agent_sandbox_loop(self):
    # 1. Setup Mock Agent
    mock_agent_instance = mock.MagicMock()
    mock_agent_instance.build_prompt.return_value = "System Instruction Prompt"
    swe_agent.SWEAgent.return_value = mock_agent_instance

    # 2. Setup Mock Environment
    mock_env_instance = mock.MagicMock()
    mock_env_instance.reset.return_value = ("Initial Observation", 0.0, False)

    # Mock env.step explicitly returning 4-tuples (obs, reward, done, info)
    def mock_step(action):
      if "submit" in action:
        return ("Shell Output 2", 1.0, True, {})
      return ("Shell Output 1", 0.0, False, {})

    mock_env_instance.step = mock_step
    swe_env.SWEEnv.return_value = mock_env_instance

    # 3. Initialize Program
    dataset = iter([{"instance_id": "test_app_1"}])
    program = DeepsweRLProgram(
        dataset=dataset, algo=mock.MagicMock(), max_steps=1
    )

    # Mock engine async generate
    async def mock_generate(chat_history):
      if "Shell Output 1" in chat_history[0]:
        return [datatypes.TrajectoryItem(traj="submit")]
      return [datatypes.TrajectoryItem(traj="cat app.py")]

    program.engine = mock.MagicMock()
    program.engine.generate = mock_generate

    # 4. Run loop
    chat_history, reward = await program._run_agent_sandbox_loop(
        {"task": "mock"}
    )

    # 5. Verify interaction sequence and output
    self.assertEqual(reward, 1.0)
    self.assertIn("cat app.py", chat_history)
    self.assertIn("submit", chat_history)
    self.assertIn("Initial Observation", chat_history)
    self.assertIn("Shell Output 1", chat_history)


if __name__ == "__main__":
  unittest.main()

