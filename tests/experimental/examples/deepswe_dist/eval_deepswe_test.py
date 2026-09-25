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

"""Tests for standalone distributed DeepSWE evaluation."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tunix.experimental.examples.deepswe_dist import eval_deepswe


class EvalDeepSWETest(unittest.TestCase):

  def test_defaults_match_agentic_eval(self):
    args = eval_deepswe.parse_args([])
    self.assertEqual(args.model_id, "Qwen/Qwen3-32B")
    self.assertEqual(args.dataset_name, "R2E-Gym/SWE-Bench-Verified")
    self.assertEqual(args.dataset_split, "test")
    self.assertEqual(args.max_steps, 30)
    self.assertEqual(args.max_model_len, 32768)
    self.assertEqual(args.max_response_length, 8192)
    self.assertEqual(args.max_concurrent, 256)
    self.assertEqual(args.timeout, 600)
    self.assertFalse(args.enable_guard)
    self.assertFalse(args.disable_thinking)

  def test_request_uses_one_greedy_trajectory_and_agentic_limits(self):
    args = eval_deepswe.parse_args([])
    entry = {
        "instance_id": "task-1",
        "docker_image": "image-1",
        "problem_statement": "Fix the test",
    }

    request = eval_deepswe.request_fields(args, entry, 7)

    self.assertEqual(request["group_index"], 0)
    self.assertEqual(request["prompt"], "Fix the test")
    self.assertEqual(request["max_turns"], 30)
    self.assertEqual(request["max_response_length"], 32512)
    self.assertEqual(request["generation_kwargs"], {
        "max_generation_steps": 8192,
        "temperature": 0.0,
        "return_logprobs": False,
    })
    self.assertFalse(request["metadata"]["exact_token_continuity"])
    self.assertTrue(request["metadata"]["record_episode_summary"])
    self.assertEqual(request["metadata"]["env_config"]["entry"], entry)

  def test_result_and_pass_at_one_include_failures(self):
    successful = eval_deepswe.compact_result(SimpleNamespace(
        error=None,
        payload=SimpleNamespace(
            traj={
                "trajectory_reward": 1.0,
                "status": "SUCCEEDED",
                "episode_summary": {
                    "num_steps": 3,
                    "guard_blocked_steps": 1,
                    "guard_reasons": ["invalid_action"],
                },
            },
        ),
    ))
    failed = eval_deepswe.compact_result(SimpleNamespace(
        error=SimpleNamespace(message="worker failed"), payload=None,
    ))
    overflow = eval_deepswe.compact_result(SimpleNamespace(
        error=SimpleNamespace(message="Prompt too long for max_model_len"),
        payload=None,
    ))

    summary = eval_deepswe.summarize([successful, failed, overflow], 3)

    self.assertEqual(summary["pass_at_1"], 1 / 3)
    self.assertEqual(summary["errors"], 1)
    self.assertEqual(summary["guard_blocks"], 1)
    self.assertEqual(overflow["status"], "MAX_CONTEXT_LIMIT_REACHED")
    self.assertIsNone(overflow["error"])

  def test_optional_guard_blocks_invalid_function_calls(self):
    from tunix.experimental.examples.deepswe_dist import deepswe
    from tunix.experimental.rl.agentic import registry

    name = eval_deepswe.register_guarded_env(deepswe)
    env = object.__new__(registry.ENV_REGISTRY.get(name))
    env.guard = mock.Mock()
    env.guard._parse_action.return_value = (None, None)

    result = env._step_impl("no tool call")

    self.assertTrue(result.info["guard_blocked"])
    self.assertEqual(result.info["guard_reason"], "missing_function_call")

  def test_bounded_dispatch_records_out_of_order_completions(self):
    args = eval_deepswe.parse_args(["--max_concurrent", "2"])
    entries = [
        {"instance_id": f"task-{i}", "docker_image": f"image-{i}"}
        for i in range(3)
    ]
    responses = ["eval_1", "eval_0", "eval_2"]

    class FakeHandle:

      def __init__(self):
        self.pending = set()
        self.peak = 0

      async def dispatch_task(self, request_id, method, fields):
        self.pending.add(request_id)
        self.peak = max(self.peak, len(self.pending))
        self.assert_fields.append((method, fields))
        return request_id

      async def poll_responses(self, timeout_s):
        del timeout_s
        request_id = responses.pop(0)
        self.pending.remove(request_id)
        return SimpleNamespace(
            request_id=request_id,
            unwrap=lambda: {
                "reward": 0.0, "num_steps": 1, "status": "SUCCEEDED",
                "guard_blocked_steps": 0, "guard_reasons": [], "error": None,
            },
        )

    handle = FakeHandle()
    handle.assert_fields = []
    rows = []
    asyncio.run(eval_deepswe.collect_worker(
        handle, iter(enumerate(entries)), 2, args, rows.append
    ))

    self.assertEqual(handle.peak, 2)
    self.assertEqual([row["instance_id"] for row in rows],
                     ["task-1", "task-0", "task-2"])
    self.assertTrue(all(method == "evaluate"
                        for method, _ in handle.assert_fields))

  def test_controller_writes_agentic_style_results(self):
    args = eval_deepswe.parse_args(["--tasks_limit", "1"])
    entry = {"instance_id": "task-1", "docker_image": "image-1"}

    class FakeHandle:

      async def asubmit(self, method):
        self_method = method
        assert self_method == "evaluation_info"
        return eval_deepswe.model_profile(args)

      async def dispatch_task(self, request_id, method, fields):
        del method, fields
        return request_id

      async def poll_responses(self, timeout_s):
        del timeout_s
        return SimpleNamespace(
            request_id="eval_0",
            unwrap=lambda: {
                "reward": 1.0, "num_steps": 2, "status": "SUCCEEDED",
                "guard_blocked_steps": 0, "guard_reasons": [], "error": None,
            },
        )

      async def close(self):
        pass

    with tempfile.TemporaryDirectory() as temp_dir:
      args.output_dir = temp_dir
      with mock.patch.object(eval_deepswe, "load_entries", return_value=[entry]):
        with mock.patch(
            "tunix.experimental.worker.remote_execution.ActorHandle.from_address",
            return_value=FakeHandle(),
        ):
          summary = asyncio.run(eval_deepswe.run_controller(args))
      results = list(Path(temp_dir).glob("*.jsonl"))
      self.assertEqual(len(results), 1)
      row = json.loads(results[0].read_text().strip())
      self.assertEqual(row["instance_id"], "task-1")
      self.assertEqual(row["docker_image"], "image-1")
      self.assertEqual(summary["pass_at_1"], 1.0)
      self.assertTrue(results[0].with_suffix(".summary.json").exists())


if __name__ == "__main__":
  unittest.main()
