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
import pickle
from typing import Sequence

from transformers import AutoTokenizer  # pylint: disable=g-importing-member
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.distributed.runtime.context import ProcessContext  # pylint: disable=g-importing-member
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import legacy_vllm_sampler_adapter as legacy_sampler_lib
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker
from tunix.generate import vllm_sampler


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

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  config = vllm_sampler.VllmConfig(engine_kwargs={"model": args.model_name})
  sampler_server = legacy_sampler_lib.LegacyVllmSamplerAdapter(  # pyrefly: ignore[bad-instantiation]
      server_id="vllm-0",
      tokenizer=tokenizer,
      config=config,
  )
  sampler_server.initialize()

  worker_service = rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      sampler=sampler_server,
      env_pool=mocks.MockEnvironmentPool(
          pool_size=1, env_factory=registry.ENV_REGISTRY.get("mock_env")
      ),
      agent_factory=registry.AGENT_REGISTRY.get("mock_agent"),
      tokenizer=tokenizer,
      chat_parser=mocks.MockChatParser(),
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
