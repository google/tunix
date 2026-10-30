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

"""Launch and supervise the fixed SGLang OPD teacher in the foreground.

The current environment must have a compatible SGLang installation. Only this
launcher's child process group is terminated on failure, interruption, or exit.
Readiness checks verify server declarations, not remote weight attestation.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from typing import Any

import httpx
from tunix.dist_rl.examples.on_policy_distillation import models


@dataclasses.dataclass(frozen=True, kw_only=True)
class TeacherServerConfig:
  host: str = "127.0.0.1"
  port: int = 30001
  tensor_parallel_size: int = 2
  mem_fraction_static: float = 0.8
  context_length: int = 8192
  chunked_prefill_size: int = 4096
  startup_timeout_seconds: float = 900.0
  poll_interval_seconds: float = 2.0
  cuda_visible_devices: str | None = None

  def __post_init__(self) -> None:
    if not self.host or any(c in self.host for c in "/?#"):
      raise ValueError(
          "host must be a bind address without a URL scheme or path."
      )
    if (
        isinstance(self.port, bool)
        or not isinstance(self.port, int)
        or not 1 <= self.port <= 65535
    ):
      raise ValueError("port must be an integer between 1 and 65535.")
    for name in (
        "tensor_parallel_size",
        "context_length",
        "chunked_prefill_size",
    ):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if (
        not math.isfinite(self.mem_fraction_static)
        or not 0 < self.mem_fraction_static < 1
    ):
      raise ValueError(
          "mem_fraction_static must be finite and between 0 and 1."
      )
    for name in ("startup_timeout_seconds", "poll_interval_seconds"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")

  @property
  def probe_host(self) -> str:
    return {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)

  @property
  def base_url(self) -> str:
    host = self.probe_host
    return (
        f"http://[{host}]:{self.port}"
        if ":" in host
        else f"http://{host}:{self.port}"
    )


def build_command(
    manifest: dict[str, Any], config: TeacherServerConfig
) -> list[str]:
  """Builds explicit model arguments; arbitrary SGLang overrides are excluded."""
  record = manifest["teacher"]
  return [
      sys.executable,
      "-m",
      "sglang.launch_server",
      "--model-path",
      record["snapshot_path"],
      "--tokenizer-path",
      record["tokenizer_path"],
      "--revision",
      models.TEACHER_REVISION,
      "--served-model-name",
      expected_served_model_name(manifest),
      "--weight-version",
      models.TEACHER_REVISION,
      "--load-format",
      "safetensors",
      "--dtype",
      "bfloat16",
      "--host",
      config.host,
      "--port",
      str(config.port),
      "--tp",
      str(config.tensor_parallel_size),
      "--mem-fraction-static",
      str(config.mem_fraction_static),
      "--context-length",
      str(config.context_length),
      "--chunked-prefill-size",
      str(config.chunked_prefill_size),
  ]


def expected_served_model_name(manifest: dict[str, Any]) -> str:
  """Returns the fixed, revision-qualified name required of the teacher."""
  record = manifest.get("teacher", {})
  if (
      record.get("repo_id") != models.TEACHER_MODEL_ID
      or record.get("revision") != models.TEACHER_REVISION
  ):
    raise ValueError("Teacher identity differs from the fixed recipe.")
  return f"{models.TEACHER_MODEL_ID}@{models.TEACHER_REVISION}"


def validate_model_info(
    info: Any,
    manifest: dict[str, Any],
    *,
    min_context_length: int,
    check_local_paths: bool = False,
) -> None:
  """Checks the identity and scoring capacity declared by the owned server."""
  record = manifest["teacher"]
  expected = {
      "served_model_name": expected_served_model_name(manifest),
      "weight_version": models.TEACHER_REVISION,
      "is_generation": True,
  }
  if not isinstance(info, dict):
    raise ValueError("Teacher model metadata must be a JSON object.")
  if check_local_paths:
    expected.update(
        model_path=record["snapshot_path"],
        tokenizer_path=record["tokenizer_path"],
    )
  for name, value in expected.items():
    actual = info.get(name)
    if name in ("model_path", "tokenizer_path") and isinstance(actual, str):
      actual = str(Path(actual).expanduser().resolve())
    if actual != value:
      raise ValueError(f"Teacher model metadata mismatch for {name}.")
  if "load_format" in info and info["load_format"] != "safetensors":
    raise ValueError("Teacher model metadata reports a different load_format.")
  context = info.get("context_length")
  if (
      isinstance(context, bool)
      or not isinstance(context, int)
      or context < min_context_length
  ):
    raise ValueError(
        "Teacher metadata reports an insufficient or unknown context_length."
    )


async def _fetch_teacher_info(client: httpx.AsyncClient, base_url: str) -> Any:
  base_url = base_url.rstrip("/")
  health = await client.get(base_url + "/health_generate")
  health.raise_for_status()
  response = await client.get(base_url + "/model_info")
  if response.status_code == 404:
    response = await client.get(base_url + "/get_model_info")
  response.raise_for_status()
  return response.json()


async def verify_teacher_service(
    base_url: str,
    manifest: dict[str, Any],
    timeout_seconds: float = 60.0,
    *,
    min_context_length: int = 1,
) -> dict[str, Any]:
  """Verifies remote service declarations without assuming shared local paths.

  This is a preflight check, not proof of the remote checkpoint's contents.
  Unlike startup polling, the existing service must already be healthy.
  """
  if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be finite and positive.")
  if (
      isinstance(min_context_length, bool)
      or not isinstance(min_context_length, int)
      or min_context_length < 1
  ):
    raise ValueError("min_context_length must be a positive integer.")
  async with asyncio.timeout(timeout_seconds):
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
      info = await _fetch_teacher_info(client, base_url)
  validate_model_info(info, manifest, min_context_length=min_context_length)
  return info


async def wait_for_teacher(
    process: subprocess.Popen,
    manifest: dict[str, Any],
    config: TeacherServerConfig,
) -> dict[str, Any]:
  """Waits for health and matching metadata within one total startup deadline."""
  last_error = "No healthy response received."
  try:
    async with asyncio.timeout(config.startup_timeout_seconds):
      async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        while True:
          if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited during startup with code {process.returncode}."
                " Check its logs and installed SGLang compatibility."
            )
          try:
            info = await _fetch_teacher_info(client, config.base_url)
          except httpx.HTTPError as exc:
            last_error = str(exc)
          else:
            validate_model_info(
                info,
                manifest,
                min_context_length=config.context_length,
                check_local_paths=True,
            )
            if process.poll() is not None:
              raise RuntimeError(
                  "SGLang exited before readiness could be established."
              )
            return info
          await asyncio.sleep(config.poll_interval_seconds)
  except TimeoutError as exc:
    raise TimeoutError(
        f"Teacher startup deadline exceeded: {last_error}"
    ) from exc


def _assert_port_available(config: TeacherServerConfig) -> None:
  family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
  with socket.socket(family, socket.SOCK_STREAM) as probe:
    try:
      probe.bind((config.host, config.port))
    except OSError as exc:
      raise RuntimeError(
          "Teacher bind address is unavailable; no existing process will be"
          " stopped."
      ) from exc


def stop_owned_process(process: subprocess.Popen) -> None:
  """Stops only the new session created by this launcher, including workers."""
  try:
    os.killpg(process.pid, signal.SIGTERM)
  except ProcessLookupError:
    process.wait(timeout=10)
    return
  try:
    process.wait(timeout=10)
  except subprocess.TimeoutExpired:
    pass
  # A launcher can exit while its GPU workers remain in the owned group.
  # Kill any survivors even if the direct child has already been reaped.
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait(timeout=10)


async def serve_teacher(
    manifest_path: str | Path, config: TeacherServerConfig
) -> None:
  """Runs the teacher until its owned process exits or the caller interrupts."""
  manifest = models.load_manifest(manifest_path)
  teacher_config = models._read_json(
      Path(manifest["teacher"]["snapshot_path"]) / "config.json"
  )
  max_context = teacher_config.get("max_position_embeddings")
  if not isinstance(max_context, int) or config.context_length > max_context:
    raise ValueError(
        "Requested context_length exceeds the pinned teacher configuration."
    )
  _assert_port_available(config)
  command = build_command(manifest, config)
  env = os.environ.copy()
  env["HF_HUB_OFFLINE"] = "1"
  env["TRANSFORMERS_OFFLINE"] = "1"
  if config.cuda_visible_devices is not None:
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
  process = subprocess.Popen(command, env=env, start_new_session=True)
  try:
    info = await wait_for_teacher(process, manifest, config)
    print(
        json.dumps(
            {
                "status": "ready",
                "teacher_url": config.base_url,
                "teacher_revision": manifest["teacher"]["declared_revision"],
                "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
                "context_length": info["context_length"],
                "pid": process.pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while process.poll() is None:
      await asyncio.sleep(1.0)
    if process.returncode:
      raise RuntimeError(
          f"SGLang teacher exited with code {process.returncode}."
      )
  finally:
    stop_owned_process(process)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models-manifest", required=True)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=30001)
  parser.add_argument("--tensor-parallel-size", type=int, default=2)
  parser.add_argument("--mem-fraction-static", type=float, default=0.8)
  parser.add_argument("--context-length", type=int, default=8192)
  parser.add_argument("--chunked-prefill-size", type=int, default=4096)
  parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
  parser.add_argument("--cuda-visible-devices")
  args = parser.parse_args(argv)
  config = TeacherServerConfig(**{
      key: value
      for key, value in vars(args).items()
      if key != "models_manifest"
  })

  def interrupt(signum, frame):
    del signum, frame
    raise KeyboardInterrupt

  previous = signal.signal(signal.SIGTERM, interrupt)
  try:
    asyncio.run(serve_teacher(args.models_manifest, config))
  finally:
    signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
  main()
