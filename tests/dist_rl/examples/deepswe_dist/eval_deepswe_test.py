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
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from etils import epath
from tunix.dist_rl.examples.deepswe_dist import eval_deepswe


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
    from tunix.dist_rl.common import registry

    class BaseEnv:

      def _step_impl(self, action):
        raise AssertionError(f"Guard should block {action!r}")

    # The real DeepSWE env needs r2egym; the guard only wraps _step_impl.
    with mock.patch.dict(registry.ENV_REGISTRY._registry):
      name = eval_deepswe.register_guarded_env(
          SimpleNamespace(DeepSWEEnv=BaseEnv)
      )
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
            "tunix.dist_rl.worker.remote_execution.ActorHandle.from_address",
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

  def test_sampling_flags_and_validation(self):
    args = eval_deepswe.parse_args([
        "--num_rollouts_per_instance",
        "4",
        "--temperature",
        "0.6",
        "--top_p",
        "0.95",
        "--top_k",
        "20",
    ])
    entry = {"instance_id": "task-1", "docker_image": "image-1"}

    request = eval_deepswe.request_fields(args, entry, 3, 2)

    self.assertEqual(request["request_id"], "eval_3_2")
    self.assertEqual(request["prompt_id"], "eval_3")
    self.assertEqual(request["group_index"], 2)
    self.assertNotIn("generation_kwargs", request)
    profile = eval_deepswe.model_profile(args)
    self.assertEqual(
        (profile["temperature"], profile["top_p"], profile["top_k"]),
        (0.6, 0.95, 20),
    )
    for bad in (
        ["--num_rollouts_per_instance", "0"],
        ["--temperature", "-1"],
        ["--top_p", "0"],
        ["--top_k", "0"],
    ):
      with self.subTest(bad=bad), self.assertRaises(SystemExit):
        eval_deepswe.parse_args(bad)

  def test_multiple_rollouts_require_positive_temperature(self) -> None:
    greedy = eval_deepswe.parse_args(["--num_rollouts_per_instance", "1"])
    self.assertEqual(greedy.num_rollouts_per_instance, 1)
    self.assertEqual(greedy.temperature, 0.0)

    sampled = eval_deepswe.parse_args([
        "--num_rollouts_per_instance",
        "2",
        "--temperature",
        "0.6",
    ])
    self.assertEqual(sampled.num_rollouts_per_instance, 2)
    self.assertEqual(sampled.temperature, 0.6)

    for bad in (
        ["--num_rollouts_per_instance", "2"],
        ["--num_rollouts_per_instance", "2", "--temperature", "0"],
    ):
      stderr = io.StringIO()
      with (
          self.subTest(bad=bad),
          contextlib.redirect_stderr(stderr),
          self.assertRaises(SystemExit) as ctx,
      ):
        eval_deepswe.parse_args(bad)
      self.assertEqual(ctx.exception.code, 2)
      self.assertIn(
          "--num_rollouts_per_instance > 1 requires --temperature > 0",
          stderr.getvalue(),
      )

  def test_pass_at_k_counts_failed_and_missing_attempts(self):
    def row(pair_index, reward, error=None):
      return {
          "pair_index": pair_index,
          "instance_id": "same-id",
          "reward": reward,
          "num_steps": 1,
          "status": "ERROR" if error else "SUCCEEDED",
          "guard_blocked_steps": 0,
          "guard_reasons": [],
          "error": error,
      }

    # Task 0: 1 of 2 resolved; task 1: 0 of 2 (one failed); task 2: missing.
    # Duplicate instance ids must not merge tasks.
    rows = [
        row(0, 1.0),
        row(0, 0.0),
        row(1, 0.0),
        row(1, 0.0, error="sandbox failed"),
    ]

    summary = eval_deepswe.summarize(rows, 3, attempts=2)

    self.assertEqual(summary["total_attempts"], 6)
    self.assertEqual(summary["completed_attempts"], 4)
    self.assertEqual(summary["pass_at_1"], 1 / 6)
    self.assertEqual(summary["pass_at_k"], {"2": 1 / 3})
    self.assertEqual(summary["errors"], 1)
    self.assertNotIn("pass_at_k", eval_deepswe.summarize(rows, 3))

  def test_each_task_dispatches_all_attempts(self):
    args = eval_deepswe.parse_args([
        "--num_rollouts_per_instance",
        "2",
        "--temperature",
        "0.6",
        "--max_concurrent",
        "3",
    ])
    entries = [
        {"instance_id": f"task-{i}", "docker_image": f"image-{i}"}
        for i in range(2)
    ]

    class FakeHandle:

      def __init__(self):
        self.pending = []
        self.dispatched = []

      async def dispatch_task(self, request_id, method, fields):
        del method
        self.pending.append(request_id)
        self.dispatched.append((request_id, fields["group_index"]))
        return request_id

      async def poll_responses(self, timeout_s):
        del timeout_s
        request_id = self.pending.pop()
        return SimpleNamespace(
            request_id=request_id,
            unwrap=lambda: {
                "reward": 1.0,
                "num_steps": 1,
                "status": "SUCCEEDED",
                "guard_blocked_steps": 0,
                "guard_reasons": [],
                "error": None,
            },
        )

    handle = FakeHandle()
    rows = []
    asyncio.run(
        eval_deepswe.collect_worker(
            handle, iter(enumerate(entries)), 3, args, rows.append
        )
    )

    self.assertEqual(
        sorted(handle.dispatched),
        [
            ("eval_0_0", 0),
            ("eval_0_1", 1),
            ("eval_1_0", 0),
            ("eval_1_1", 1),
        ],
    )
    self.assertEqual(
        sorted((row["instance_id"], row["attempt"]) for row in rows),
        [("task-0", 0), ("task-0", 1), ("task-1", 0), ("task-1", 1)],
    )
    self.assertEqual(
        eval_deepswe.summarize(rows, 2, 2)["pass_at_k"], {"2": 1.0}
    )

  def test_write_helpers_persist_attempts_config_and_summary(self):
    row = {"request_id": "eval_0_1", "instance_id": "task-0", "reward": 1.0}
    with tempfile.TemporaryDirectory() as temp_dir, contextlib.chdir(temp_dir):
      for run_dir, out in (
          (epath.Path(temp_dir) / "out" / "run", Path(temp_dir) / "out"),
          (epath.Path("eval_results") / "run", Path(temp_dir) / "eval_results"),
      ):
        with self.subTest(run_dir=str(run_dir)):
          eval_deepswe.write_json(
              run_dir / "config.json", {"num_rollouts_per_instance": 2}
          )
          eval_deepswe.write_json(run_dir / "attempts" / "eval_0_1.json", row)
          eval_deepswe.write_results(run_dir, [row], {"pass_at_1": 1.0})

          self.assertEqual(
              json.loads(
                  (out / "run/attempts/eval_0_1.json").read_text(
                      encoding="utf-8"
                  )
              ),
              row,
          )
          self.assertEqual(
              json.loads((out / "run/config.json").read_text(encoding="utf-8")),
              {"num_rollouts_per_instance": 2},
          )
          self.assertEqual(
              json.loads((out / "run.jsonl").read_text(encoding="utf-8")), row
          )
          self.assertEqual(
              json.loads(
                  (out / "run.summary.json").read_text(encoding="utf-8")
              ),
              {"pass_at_1": 1.0},
          )

  def test_write_helpers_support_gcs_epath(self):
    run_dir = epath.Path("gs://bkt/out") / "run"
    row = {"request_id": "eval_0", "reward": 0.0}
    summary = {"pass_at_1": 0.0}
    path_cls = type(run_dir)

    with (
        mock.patch.object(path_cls, "mkdir", autospec=True) as mock_mkdir,
        mock.patch.object(path_cls, "write_text", autospec=True) as mock_write,
    ):
      eval_deepswe.write_json(run_dir / "attempts" / "eval_0.json", row)
      eval_deepswe.write_results(run_dir, [row], summary)

    mock_mkdir.assert_not_called()
    self.assertEqual(
        mock_write.call_args_list,
        [
            mock.call(
                epath.Path("gs://bkt/out/run/attempts/eval_0.json"),
                json.dumps(row, indent=2) + "\n",
                encoding="utf-8",
            ),
            mock.call(
                epath.Path("gs://bkt/out/run.jsonl"),
                json.dumps(row) + "\n",
                encoding="utf-8",
            ),
            mock.call(
                epath.Path("gs://bkt/out/run.summary.json"),
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            ),
        ],
    )

  def test_controller_survives_attempt_write_oserror(self) -> None:
    args = eval_deepswe.parse_args(["--tasks_limit", "2"])
    entries: list[dict[str, object]] = [
        {"instance_id": "task-0", "docker_image": "image-0"},
        {"instance_id": "task-1", "docker_image": "image-1"},
    ]
    replies = iter([
        SimpleNamespace(
            request_id="eval_0",
            unwrap=lambda: {
                "reward": 1.0,
                "num_steps": 2,
                "status": "SUCCEEDED",
                "guard_blocked_steps": 0,
                "guard_reasons": [],
                "error": None,
            },
        ),
        SimpleNamespace(
            request_id="eval_1",
            unwrap=lambda: {
                "reward": 0.0,
                "num_steps": 3,
                "status": "SUCCEEDED",
                "guard_blocked_steps": 0,
                "guard_reasons": [],
                "error": None,
            },
        ),
    ])
    handle = mock.AsyncMock()
    handle.asubmit.return_value = eval_deepswe.model_profile(args)
    handle.dispatch_task.side_effect = (
        lambda request_id, method, fields: request_id
    )
    handle.poll_responses.side_effect = lambda timeout_s: next(replies)
    real_write_json = eval_deepswe.write_json

    def flaky_write_json(path: epath.PathLike, value: object) -> None:
      if epath.Path(path).name == "eval_0.json":
        raise OSError("transient GCS error")
      real_write_json(path, value)

    with tempfile.TemporaryDirectory() as temp_dir:
      args.output_dir = temp_dir
      with (
          mock.patch.object(eval_deepswe, "load_entries", return_value=entries),
          mock.patch(
              "tunix.dist_rl.worker.remote_execution.ActorHandle.from_address",
              return_value=handle,
          ),
          mock.patch.object(
              eval_deepswe,
              "write_json",
              autospec=True,
              side_effect=flaky_write_json,
          ),
          self.assertLogs(level="INFO") as logs,
      ):
        summary = asyncio.run(eval_deepswe.run_controller(args))

      self.assertTrue(
          any(
              record.levelname == "WARNING"
              and "eval_0" in record.getMessage()
              and "transient GCS error" in record.getMessage()
              for record in logs.records
          )
      )
      results = list(Path(temp_dir).glob("*.jsonl"))
      self.assertEqual(len(results), 1)
      run_dir = Path(temp_dir) / results[0].stem
      self.assertTrue((run_dir / "config.json").exists())
      self.assertFalse((run_dir / "attempts" / "eval_0.json").exists())
      self.assertTrue((run_dir / "attempts" / "eval_1.json").exists())
      written_rows = [
          json.loads(line)
          for line in results[0].read_text(encoding="utf-8").splitlines()
      ]
      self.assertEqual(
          [row["instance_id"] for row in written_rows], ["task-0", "task-1"]
      )
      saved_summary = json.loads(
          results[0].with_suffix(".summary.json").read_text(encoding="utf-8")
      )
      self.assertEqual(summary["completed_attempts"], 2)
      self.assertEqual(summary["resolved"], 1)
      self.assertEqual(summary["pass_at_1"], 0.5)
      self.assertIsNone(summary["fatal_error"])
      self.assertEqual(saved_summary, summary)


if __name__ == "__main__":
  unittest.main()
