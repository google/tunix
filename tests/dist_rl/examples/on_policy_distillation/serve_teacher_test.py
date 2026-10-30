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

"""Teacher launcher contracts using HTTP metadata, without model weights or GPUs."""

import asyncio
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import signal
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from tunix.dist_rl.examples.on_policy_distillation import models
from tunix.dist_rl.examples.on_policy_distillation import serve_teacher


def _manifest():
  return {
      "teacher": {
          "repo_id": models.TEACHER_MODEL_ID,
          "revision": models.TEACHER_REVISION,
          "snapshot_path": "/local/fixed/teacher",
          "tokenizer_path": "/local/fixed/teacher",
          "declared_revision": (
              f"hf:{models.TEACHER_MODEL_ID}@{models.TEACHER_REVISION}"
          ),
      },
      "tokenizer_fingerprint": "f" * 64,
  }


def _model_info():
  # Fields are from SGLang's official model_info/get_model_info handler.
  return {
      "model_path": "/local/fixed/teacher",
      "tokenizer_path": "/local/fixed/teacher",
      "served_model_name": serve_teacher.expected_served_model_name(
          _manifest()
      ),
      "weight_version": models.TEACHER_REVISION,
      "is_generation": True,
      "load_format": "safetensors",
      "context_length": 8192,
  }


class TeacherServiceTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    self.requests = []
    self.info = _model_info()
    self.healthy = True
    self.legacy = False
    owner = self

    class Handler(BaseHTTPRequestHandler):

      def log_message(self, *args):
        pass

      def do_GET(self):
        owner.requests.append(self.path)
        if self.path == "/health_generate":
          code, body = (200 if owner.healthy else 503), b"healthy"
        elif self.path == "/model_info" and owner.legacy:
          code, body = 404, b"not found"
        elif self.path in ("/model_info", "/get_model_info"):
          code, body = 200, json.dumps(owner.info).encode()
        else:
          code, body = 404, b"not found"
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self.thread.start()
    self.config = serve_teacher.TeacherServerConfig(
        port=self.server.server_port,
        startup_timeout_seconds=1,
        poll_interval_seconds=0.005,
    )
    self.process = SimpleNamespace(poll=lambda: None, returncode=None)

  async def asyncTearDown(self):
    await asyncio.to_thread(self.server.shutdown)
    self.server.server_close()
    self.thread.join(timeout=2)

  async def test_readiness_checks_health_and_real_metadata_fields(self):
    info = await serve_teacher.wait_for_teacher(
        self.process, _manifest(), self.config
    )
    self.assertEqual(info["weight_version"], models.TEACHER_REVISION)
    self.assertEqual(self.requests, ["/health_generate", "/model_info"])

  async def test_legacy_metadata_route_and_optional_load_format(self):
    self.legacy = True
    self.info.pop("load_format")
    await serve_teacher.verify_teacher_service(
        self.config.base_url, _manifest(), min_context_length=5120
    )
    self.assertIn("/get_model_info", self.requests)

  async def test_remote_service_paths_can_differ_but_owned_launcher_paths_cannot(
      self,
  ):
    self.info["model_path"] = "/gpu-host/cache/snapshot"
    self.info["tokenizer_path"] = "/gpu-host/cache/snapshot"
    await serve_teacher.verify_teacher_service(
        self.config.base_url, _manifest(), min_context_length=5120
    )
    with self.assertRaisesRegex(ValueError, "model_path"):
      await serve_teacher.wait_for_teacher(
          self.process, _manifest(), self.config
      )

  async def test_mismatched_or_missing_stable_identity_is_rejected(self):
    for name, value in (
        ("served_model_name", "Qwen/Qwen3-32B"),
        ("weight_version", "main"),
        ("is_generation", False),
        ("context_length", 4096),
        ("load_format", "dummy"),
    ):
      with self.subTest(name=name):
        self.info = _model_info()
        self.info[name] = value
        with self.assertRaises(ValueError):
          await serve_teacher.verify_teacher_service(
              self.config.base_url, _manifest(), min_context_length=5120
          )
    self.info = _model_info()
    self.info.pop("weight_version")
    with self.assertRaisesRegex(ValueError, "weight_version"):
      await serve_teacher.verify_teacher_service(
          self.config.base_url, _manifest()
      )

  async def test_startup_deadline_and_child_exit_are_failures(self):
    self.healthy = False
    config = serve_teacher.TeacherServerConfig(
        port=self.server.server_port,
        startup_timeout_seconds=0.05,
        poll_interval_seconds=0.001,
    )
    with self.assertRaises(TimeoutError):
      await serve_teacher.wait_for_teacher(self.process, _manifest(), config)
    exited = SimpleNamespace(poll=lambda: 9, returncode=9)
    with self.assertRaisesRegex(RuntimeError, "code 9"):
      await serve_teacher.wait_for_teacher(exited, _manifest(), self.config)


class TeacherLauncherTest(unittest.TestCase):

  def test_command_fixes_identity_and_disallows_remote_model_code(self):
    config = serve_teacher.TeacherServerConfig(
        port=31001, tensor_parallel_size=4, cuda_visible_devices="0,1,2,3"
    )
    command = serve_teacher.build_command(_manifest(), config)
    for name, value in (
        ("--model-path", "/local/fixed/teacher"),
        ("--revision", models.TEACHER_REVISION),
        ("--weight-version", models.TEACHER_REVISION),
        ("--load-format", "safetensors"),
        ("--dtype", "bfloat16"),
        ("--tp", "4"),
        ("--port", "31001"),
    ):
      self.assertEqual(command[command.index(name) + 1], value)
    self.assertNotIn("--trust-remote-code", command)

  def test_cleanup_targets_owned_group_even_if_direct_child_already_exited(
      self,
  ):
    child = mock.Mock(pid=12345, returncode=1)
    child.poll.return_value = 1
    with mock.patch.object(serve_teacher.os, "killpg") as kill:
      serve_teacher.stop_owned_process(child)
    self.assertEqual(
        kill.call_args_list,
        [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)],
    )

  def test_cleanup_escalates_only_its_own_group_after_timeout(self):
    child = mock.Mock(pid=12345)
    child.wait.side_effect = [subprocess.TimeoutExpired("teacher", 10), 0]
    with mock.patch.object(serve_teacher.os, "killpg") as kill:
      serve_teacher.stop_owned_process(child)
    kill.assert_any_call(12345, signal.SIGKILL)

  def test_failed_preflight_stops_owned_child_and_preserves_offline_model_paths(
      self,
  ):
    child = mock.Mock(pid=12345)
    with (
        mock.patch.object(models, "load_manifest", return_value=_manifest()),
        mock.patch.object(
            models,
            "_read_json",
            return_value={"max_position_embeddings": 40960},
        ),
        mock.patch.object(serve_teacher, "_assert_port_available"),
        mock.patch.object(
            serve_teacher,
            "wait_for_teacher",
            new=mock.AsyncMock(side_effect=ValueError("bad identity")),
        ),
        mock.patch.object(
            serve_teacher.subprocess, "Popen", return_value=child
        ) as launch,
        mock.patch.object(serve_teacher, "stop_owned_process") as stop,
    ):
      with self.assertRaisesRegex(ValueError, "bad identity"):
        asyncio.run(
            serve_teacher.serve_teacher(
                "models.json", serve_teacher.TeacherServerConfig()
            )
        )
    stop.assert_called_once_with(child)
    self.assertIs(launch.call_args.kwargs["start_new_session"], True)
    self.assertEqual(launch.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
    self.assertNotIn("shell", launch.call_args.kwargs)


if __name__ == "__main__":
  unittest.main()
