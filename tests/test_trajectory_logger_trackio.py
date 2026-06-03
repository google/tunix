# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tunix"
    / "utils"
    / "trajectory_logger.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "trajectory_logger", _MODULE_PATH
)
trajectory_logger = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = trajectory_logger
_SPEC.loader.exec_module(trajectory_logger)


class TrackioTrajectoryLoggerTest(unittest.TestCase):

  def test_log_rollouts_creates_trackio_traces(self):
    run = mock.MagicMock()
    trace = mock.MagicMock(
        side_effect=lambda messages, metadata: {
            "messages": messages,
            "metadata": metadata,
        }
    )
    fake_trackio = types.SimpleNamespace(
        init=mock.MagicMock(return_value=run),
        Trace=trace,
    )

    with mock.patch.dict(sys.modules, {"trackio": fake_trackio}):
      logger = trajectory_logger.AsyncTrajectoryLogger(
          backends=[
              trajectory_logger.TrackioTrajectoryLogBackend(
                  project="tunix-smoke",
                  run_name="run-1",
                  max_traces_per_step=2,
              )
          ],
      )
      logger.log_rollouts(
          prompts=["What is 2 + 2?"],
          completions=["4", "Four."],
          rewards=np.array([1.0, 0.8]),
          advantages=np.array([[0.1, 0.2], [0.3, 0.4]]),
          mode="train",
          step=7,
      )
      logger.stop()

    fake_trackio.init.assert_called_once_with(
        project="tunix-smoke", name="run-1"
    )
    self.assertEqual(trace.call_count, 2)
    self.assertEqual(
        trace.call_args_list[0].kwargs["messages"],
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
    )
    self.assertEqual(trace.call_args_list[0].kwargs["metadata"]["step"], 7)
    self.assertEqual(
        trace.call_args_list[1].kwargs["metadata"]["advantages"], [0.3, 0.4]
    )
    run.log.assert_called_once_with(
        {"rollout/traces": [mock.ANY, mock.ANY]}, step=7
    )

  def test_from_config_adds_trackio_backend(self):
    config = types.SimpleNamespace(
        trackio_project="tunix-smoke",
        trackio_run_name="run-1",
        trackio_trace_key="rollout/traces",
        trackio_max_traces_per_step=2,
        trackio_init_kwargs={},
    )

    logger = trajectory_logger.AsyncTrajectoryLogger.from_config(config=config)

    self.assertTrue(logger.has_backends)
    logger.stop()

  def test_log_messages_uses_existing_chat_messages(self):
    run = mock.MagicMock()
    trace = mock.MagicMock(return_value="trace")
    fake_trackio = types.SimpleNamespace(
        init=mock.MagicMock(return_value=run),
        Trace=trace,
    )
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
    ]

    with mock.patch.dict(sys.modules, {"trackio": fake_trackio}):
      logger = trajectory_logger.AsyncTrajectoryLogger(
          backends=[
              trajectory_logger.TrackioTrajectoryLogBackend(
                  project="tunix-smoke",
                  max_traces_per_step=1,
              )
          ],
      )
      logger.log_messages(
          messages_list=[messages],
          metadata_list=[{"trajectory_reward": 0.9}],
          trace_key="agentic/trajectories",
          step=3,
      )
      logger.stop()

    trace.assert_called_once_with(
        messages=messages,
        metadata={
            "mode": "train",
            "sample_index": 0,
            "step": 3,
            "trajectory_reward": 0.9,
        },
    )
    run.log.assert_called_once_with({"agentic/trajectories": ["trace"]}, step=3)

  def test_direct_trackio_backend_still_logs_rollouts(self):
    run = mock.MagicMock()
    trace = mock.MagicMock(
        side_effect=lambda messages, metadata: {
            "messages": messages,
            "metadata": metadata,
        }
    )
    fake_trackio = types.SimpleNamespace(
        init=mock.MagicMock(return_value=run),
        Trace=trace,
    )

    with mock.patch.dict(sys.modules, {"trackio": fake_trackio}):
      logger = trajectory_logger.TrackioTrajectoryLogBackend(
          project="tunix-smoke",
          run_name="run-1",
          max_traces_per_step=2,
      )
      logger.log_rollouts(
          prompts=["What is 2 + 2?"],
          completions=["4", "Four."],
          rewards=np.array([1.0, 0.8]),
          advantages=np.array([[0.1, 0.2], [0.3, 0.4]]),
          mode="train",
          step=7,
      )

    fake_trackio.init.assert_called_once_with(
        project="tunix-smoke", name="run-1"
    )
    self.assertEqual(trace.call_count, 2)
    self.assertEqual(
        trace.call_args_list[0].kwargs["messages"],
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
    )
    self.assertEqual(trace.call_args_list[0].kwargs["metadata"]["step"], 7)
    self.assertEqual(
        trace.call_args_list[1].kwargs["metadata"]["advantages"], [0.3, 0.4]
    )
    run.log.assert_called_once_with(
        {"rollout/traces": [mock.ANY, mock.ANY]}, step=7
    )


if __name__ == "__main__":
  unittest.main()
