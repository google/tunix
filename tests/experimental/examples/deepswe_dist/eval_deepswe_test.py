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

"""CPU tests, including real distributed DispatchTask/PollResponses RPC.

Run directly with absl-py, cloudpickle, grpcio, fsspec and PyYAML installed.
The model stack (JAX/vLLM/MaxText) is not needed for these contract tests.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "tunix/experimental/examples/deepswe_dist"


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


eval_lib = load("deepswe_eval_under_test", RECIPE / "eval_deepswe.py")


class EvalTest(unittest.TestCase):

  def args(self, *extra):
    return eval_lib.parse_args(
        ["--model_absolute_path", "gs://models/0/items", *extra]
    )

  def entry(self):
    return {
        "instance_id": "repo/issue",
        "problem_statement": "fix bug",
        "docker_image": "image",
    }

  def test_checkpoint_is_required_and_never_dummy(self):
    with self.assertRaises(SystemExit):
      eval_lib.parse_args([])
    config = eval_lib.maxtext_config(self.args())
    self.assertEqual(config["load_parameters_path"], "gs://models/0/items")
    self.assertFalse(config["scan_layers"])
    self.assertTrue(config["checkpoint_storage_use_ocdbt"])
    self.assertFalse(config["checkpoint_storage_use_zarr3"])

  def test_context_limits_and_per_attempt_seeds(self):
    a = self.args()
    first = eval_lib.request_fields(a, self.entry(), 0, 0)
    second = eval_lib.request_fields(a, self.entry(), 0, 1)
    self.assertNotEqual(
        first["generation_kwargs"]["seed"], second["generation_kwargs"]["seed"]
    )
    self.assertNotEqual(first["request_id"], second["request_id"])
    self.assertIsNone(first["max_response_length"])
    self.assertEqual(first["generation_kwargs"]["max_generation_steps"], 12288)
    self.assertEqual(first["metadata"]["max_context_length"], 14336)
    self.assertEqual(first["metadata"]["max_model_length"], 16384)
    with self.assertRaises(SystemExit):
      self.args("--max_context_limit", "16384")

  def test_missing_and_failed_attempts_are_not_dropped(self):
    rows = [
        dict(
            instance_id="a",
            attempt=0,
            reward=1,
            resolved=True,
            status="SUCCEEDED",
        ),
        dict(
            instance_id="a",
            attempt=1,
            reward=0,
            resolved=False,
            status="ERROR",
            error="pod failed",
        ),
        dict(
            instance_id="b",
            attempt=0,
            reward=0,
            resolved=False,
            status="MAX_STEPS_REACHED",
        ),
    ]
    summary = eval_lib.summarize(rows, ["a", "b"], 4)
    self.assertEqual(summary["expected_attempts"], 8)
    self.assertEqual(summary["missing_attempts"], 5)
    self.assertEqual(summary["error_attempts"], 1)
    self.assertEqual(summary["avg_at_k"], 1 / 8)
    self.assertEqual(summary["pass_at_k"], {"1": 1 / 8, "4": 1 / 2})
    self.assertFalse(summary["complete"])
    with self.assertRaisesRegex(ValueError, "duplicate"):
      eval_lib.summarize(rows + [rows[0]], ["a", "b"], 4)

  def test_reward_comes_from_trajectory_not_completion_status(self):
    response = types.SimpleNamespace(
        error=None,
        status="COMPLETED",
        payload=types.SimpleNamespace(
            traj={"trajectory_reward": 0, "status": "SUCCEEDED"}
        ),
    )
    self.assertFalse(eval_lib.compact_result(response)["resolved"])
    response.payload.traj["trajectory_reward"] = 1
    self.assertTrue(eval_lib.compact_result(response)["resolved"])
    response.error = "infrastructure failure"
    self.assertFalse(eval_lib.compact_result(response)["resolved"])

  def test_individual_results_are_persisted(self):
    with tempfile.TemporaryDirectory() as directory:
      writer = eval_lib.ResultWriter(directory)
      row = dict(request_id="eval_0_0", instance_id="../unsafe-id", reward=0)
      writer.record(row)
      self.assertEqual(
          json.loads((Path(directory) / "attempts/eval_0_0.json").read_text()),
          row,
      )

  def test_jobset_topology_secret_and_eval_arguments(self):
    import yaml

    manifest_path = RECIPE / "eval_qwen35_gke.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    self.assertEqual(
        manifest["metadata"]["name"], "haoyu-deepswe-qwen35-dist-eval"
    )
    jobs = manifest["spec"]["replicatedJobs"]
    self.assertEqual(jobs[1]["template"]["spec"]["parallelism"], 16)
    pod = jobs[0]["template"]["spec"]["template"]["spec"]
    checkout = pod["initContainers"][0]
    self.assertEqual(checkout["name"], "tunix-source")
    self.assertIn("sanbao/trellis_eval", checkout["args"][0])
    container = pod["containers"][0]
    self.assertEqual(container["workingDir"], "/workspace/tunix")
    env = {item["name"]: item for item in container["env"]}
    self.assertEqual(env["PYTHONPATH"]["value"], "/workspace/tunix")
    self.assertEqual(
        env["JAX_BACKEND_TARGET"]["value"], "grpc://$(PATHWAYS_HEAD):29000"
    )
    self.assertIn("secretKeyRef", env["SANDBOX_SESSION_KEY"]["valueFrom"])
    self.assertEqual(
        env["DOCKER_IMAGE_PREFIX"]["value"],
        env["IMAGE_REWRITE_PREFIX"]["value"],
    )
    self.assertNotIn("WANDB_API_KEY", env)
    # The manifest command is accepted by the actual eval CLI.
    parsed = eval_lib.parse_args(container["args"])
    self.assertEqual(parsed.mesh_fsdp * parsed.mesh_tp, 64)
    self.assertEqual(parsed.model_id, "Qwen/Qwen3.5-35B-A3B")
    self.assertEqual(
        manifest["spec"]["successPolicy"]["targetReplicatedJobs"],
        ["pathways-head"],
    )

  def test_load_entries_normalizes_r2e_gym_easy_format_and_rewrites_image(self):
    raw_rows = [
        {
            "repo_name": "sympy",
            "commit_hash": "abcdef1234",
            "problem_statement": "fix integral",
            "docker_image": "namanjain12/sympy_final:abcdef1234",
        }
    ]
    from tunix.experimental.examples.deepswe_dist import deepswe

    with mock.patch.object(
        deepswe, "load_deepswe_dataset", return_value=raw_rows
    ):
      a = self.args(
          "--docker_image_prefix",
          "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix",
          "--scaffold",
          "r2egym",
      )
      entries = eval_lib.load_entries(a)
      self.assertEqual(len(entries), 1)
      self.assertEqual(entries[0]["instance_id"], "sympy__abcdef12")
      self.assertEqual(
          entries[0]["docker_image"],
          "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/sympy_final:abcdef1234",
      )
      fields = eval_lib.request_fields(a, entries[0], 0, 0)
      self.assertTrue(
          fields["metadata"]["env_config"]["enforce_xml_function_check"]
      )
      a_oh = self.args("--scaffold", "openhands")
      fields_oh = eval_lib.request_fields(a_oh, entries[0], 0, 0)
      self.assertFalse(
          fields_oh["metadata"]["env_config"]["enforce_xml_function_check"]
      )


class RpcTest(unittest.IsolatedAsyncioTestCase):

  async def test_real_grpc_out_of_order_bounded_dispatch_and_failure(self):
    remote = load(
        "eval_remote_execution_under_test",
        ROOT / "tunix/experimental/worker/remote_execution.py",
    )

    class Worker:
      active = 0
      peak = 0

      async def evaluate(self, fields):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
          attempt = fields["group_index"]
          await asyncio.sleep(0.03 if attempt == 0 else 0.001)
          if attempt == 2:
            raise RuntimeError("sandbox failed")
          return dict(reward=1, resolved=True, status="SUCCEEDED", error=None)
        finally:
          self.active -= 1

    worker = Worker()
    server = remote.GrpcRemoteExecutionServer(worker)
    with socket.socket() as reservation:
      reservation.bind(("127.0.0.1", 0))
      port = reservation.getsockname()[1]
    await server.start_serving_async(port)
    handle = remote.ActorHandle.from_address(f"grpc://127.0.0.1:{port}")
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    entry = {
        "instance_id": "a",
        "problem_statement": "fix",
        "docker_image": "img",
    }
    jobs = iter(eval_lib.request_fields(a, entry, 0, i) for i in range(4))
    written = []
    try:
      rows = await eval_lib.evaluate_worker(handle, jobs, 2, 10, written.append)
    finally:
      await handle.close()
      await server.stop_serving()
    self.assertEqual(len(rows), 4)
    self.assertEqual(len(written), 4)
    self.assertLessEqual(worker.peak, 2)
    self.assertEqual(rows[0]["attempt"], 1)
    summary = eval_lib.summarize(rows, ["a"], 4)
    self.assertEqual(summary["error_attempts"], 1)
    self.assertEqual(summary["avg_at_k"], 0.75)

  async def test_dispatch_error_and_unknown_response_fail_fast(self):
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    entry = {
        "instance_id": "a",
        "problem_statement": "fix",
        "docker_image": "img",
    }
    fields = eval_lib.request_fields(a, entry, 0, 0)
    handle = types.SimpleNamespace(
        dispatch_task=mock.AsyncMock(return_value="wrong-ack"),
        poll_responses=mock.AsyncMock(),
    )
    with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
      await eval_lib.evaluate_worker(
          handle, iter([fields]), 1, 10, lambda r: None
      )
    handle.dispatch_task.return_value = fields["request_id"]
    handle.poll_responses.return_value = types.SimpleNamespace(
        request_id="stale"
    )
    with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
      await eval_lib.evaluate_worker(
          handle, iter([fields]), 1, 10, lambda r: None
      )

  async def test_missing_reply_has_deadline(self):
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    fields = eval_lib.request_fields(
        a, {"instance_id": "a", "problem_statement": "p"}, 0, 0
    )
    handle = types.SimpleNamespace(
        dispatch_task=mock.AsyncMock(return_value=fields["request_id"]),
        poll_responses=mock.AsyncMock(return_value=None),
    )
    with mock.patch.object(eval_lib.time, "monotonic", side_effect=[0, 20]):
      with self.assertRaises(TimeoutError):
        await eval_lib.evaluate_worker(
            handle, iter([fields]), 1, 10, lambda r: None
        )


if __name__ == "__main__":
  unittest.main()
