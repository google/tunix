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

"""Rollout worker script that hosts a vLLM server and serves generation requests."""

import argparse
import asyncio
import logging
import os
import pickle
import socket
import subprocess
import threading
import time
from typing import Callable, Sequence
import urllib.request

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.rollout import sampler
from tunix.experimental.rollout import stubs
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker


def run_vllm_server(
    model_name: str, on_serve: Callable[[str], None] | None = None
) -> None:
  """Spawns a local vLLM OpenAI-compatible server subprocess and runs a callback once healthy.

  Args:
    model_name: The name or path of the model to serve with vLLM.
    on_serve: Callback function invoked with the server address once healthy.
  """
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    vllm_port = s.getsockname()[1]

  cmd = ["vllm", "serve", model_name, f"--port={vllm_port}", "--disable-uvicorn-access-log"]

  logging.info("Starting vLLM OpenAI server subprocess: %s", " ".join(cmd))

  env = os.environ.copy()

  if "JAX_PLATFORMS" not in env:
    env["JAX_PLATFORMS"] = "tpu"
  if "cpu" not in env["JAX_PLATFORMS"]:
    env["JAX_PLATFORMS"] += ",cpu"

  env.setdefault("VLLM_NO_USAGE_STATS", "1")
  env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
  env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
  env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  env.setdefault("SKIP_JAX_PRECOMPILE", "1")
  env.setdefault("MODEL_IMPL_TYPE", "vllm")
  # env.setdefault("PYTHONUNBUFFERED", "1")
  # env.setdefault("HF_HUB_OFFLINE", "1")

  server_proc = subprocess.Popen(
      cmd,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
  )

  def _log_reader() -> None:
    """Reads stdout from the vLLM server process and forwards to logging."""
    if server_proc.stdout:
      for line in server_proc.stdout:
        logging.info("[vLLM-server] %s", line.rstrip())

  reader_thread = threading.Thread(target=_log_reader, daemon=True)
  reader_thread.start()

  vllm_addr = f"http://localhost:{vllm_port}"

  try:
    health_url = f"{vllm_addr}/health"
    logging.info("Waiting for server to be healthy at %s...", health_url)
    server_ready = False
    for i in range(300):
      if server_proc.poll() is not None:
        raise RuntimeError(
            "vLLM server subprocess exited early with return code"
            f" {server_proc.returncode}"
        )
      time.sleep(1)
      try:
        with urllib.request.urlopen(health_url, timeout=1) as resp:
          if resp.status == 200:
            logging.info("Server is healthy! (after %ds)", i + 1)
            server_ready = True
            break
      except Exception:
        pass

    if not server_ready:
      raise RuntimeError(
          "vLLM online server failed to become healthy in 300s on port"
          f" {vllm_port}"
      )

    if on_serve:
      on_serve(vllm_addr)
  finally:
    logging.info("Terminating vLLM OpenAI server subprocess...")
    server_proc.terminate()
    try:
      server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
      server_proc.kill()
      server_proc.wait()
    logging.info("Server terminated.")


def main(argv: Sequence[str], context: ProcessContext | None) -> None:
  """Main entry point for the rollout worker process.

  Args:
    argv: Command-line arguments.
    context: Process context for IPC discovery and distributed execution.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_name",
      type=str,
      default="Qwen/Qwen2.5-0.5B",
      help="Name or path of the model to serve.",
  )
  parser.add_argument(
      "--worker_id",
      type=str,
      default="",
      help="Unique identifier for this rollout worker.",
  )
  parser.add_argument(
      "--service_port",
      type=int,
      default=11111,
      help="Port for the gRPC remote execution server.",
  )
  args = parser.parse_args(argv)

  # Suppress noisy HTTP request logs from openai and httpx.
  logging.getLogger("openai").setLevel(logging.WARNING)
  logging.getLogger("httpx").setLevel(logging.WARNING)

  def rollout_worker_main(vllm_addr: str) -> None:
    """Initializes the sampler and starts the gRPC remote execution server.

    Args:
      vllm_addr: Base HTTP address of the running local vLLM server.
    """
    sampler_server = sampler.RaidenSamplerServer(
        server_id="vllm-0",
        openai_base_url=f"{vllm_addr}/v1",
        node_id=1,
        weight_sync=stubs.MockWeightSynchronizer(),
        model=args.model_name,
    )
    sampler_server.initialize()
    sampler_client = sampler.LocalSamplerClient(local_server=sampler_server)

    worker_service = rollout_worker.RolloutWorkerService(
        worker_id=args.worker_id,
        sampler_client=sampler_client,
        env_pool=stubs.StubEnvironmentPool(pool_size=1),
        agent_factory=stubs.StubAgent,
    )

    async def execution_server_main() -> None:
      """Starts the remote execution server and registers with discovery."""
      server = remote_execution.GrpcRemoteExecutionServer(worker_service)
      await server.start_serving_async(args.service_port)

      if context and context.ipc and context.ipc.discovery:
        context.ipc.discovery.register(
            metadata=pickle.dumps({
                "service_type": "rollout",
                "service_port": args.service_port,
                "worker_id": args.worker_id,
                "model_name": args.model_name,
            })
        )

      logging.info("rollout worker is ready at port %d.", args.service_port)
      try:
        while True:
          await asyncio.sleep(1)
      except asyncio.CancelledError:
        pass
      finally:
        await server.stop_serving()

    asyncio.run(execution_server_main())

  run_vllm_server(model_name=args.model_name, on_serve=rollout_worker_main)
