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

"""DeepSWE rollout worker runner for the experimental distributed demo."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import sys
from typing import Any

from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.examples.math_gsm8k_dist import run_rollout_node


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _parse_deepswe_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      "--scaffold", choices=("r2egym", "sweagent"), default="r2egym"
  )
  parser.add_argument("--use_fn_calling", action="store_true")
  parser.add_argument("--format_model_response", action="store_true")
  parser.add_argument("--max_concurrency", type=int, default=64)
  return parser.parse_known_args(argv)


def _configure_deepswe_rollout(worker_service: Any, args: argparse.Namespace) -> None:
  agent_config = {
      "use_fn_calling": args.use_fn_calling,
      "format_model_response": args.format_model_response,
      "scaffold": args.scaffold,
  }
  config = worker_service.config
  config.env_name = deepswe.DEEPSWE_ENV_NAME
  config.agent_name = deepswe.DEEPSWE_AGENT_NAME
  config.agent_config = agent_config
  worker_service.manager.config.env_name = deepswe.DEEPSWE_ENV_NAME
  worker_service.manager.config.agent_name = deepswe.DEEPSWE_AGENT_NAME
  worker_service.manager.config.agent_config = agent_config
  worker_service.manager.max_concurrency = args.max_concurrency
  logging.info(
      "Configured rollout worker for DeepSWE env=%s agent=%s scaffold=%s.",
      config.env_name,
      config.agent_name,
      args.scaffold,
  )


def main(argv: list[str], context: Any = None) -> None:
  if not (context and context.ipc and context.ipc.discovery):
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  deepswe_args, base_argv = _parse_deepswe_args(argv)
  args = run_rollout_node._parse_args(base_argv)  # pylint: disable=protected-access
  logging.basicConfig(
      level=logging.DEBUG if args.debug else logging.INFO,
      format="%(asctime)s - [DeepSWERolloutNode] %(message)s",
      force=True,
  )
  logging.info("Parsed args: %s", args)
  logging.info("Parsed DeepSWE rollout args: %s", deepswe_args)

  if context and args.sampler != "vllm":
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if args.maxtext_model_name:
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer: Any = AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  async def grpc_server_main() -> None:
    logging.info("Creating DeepSWE rollout worker service...")
    # Reuse the distributed GSM8K worker factories so this first DeepSWE demo
    # stays on the same trainer/rollout worker surface.
    # pylint: disable=protected-access
    if args.sampler == "vanilla":
      worker_service = run_rollout_node._create_vanilla_worker(args, tokenizer)
    else:
      worker_service = run_rollout_node._create_vllm_worker(args, tokenizer)
    # pylint: enable=protected-access
    _configure_deepswe_rollout(worker_service, deepswe_args)

    from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
        remote_execution,
    )

    logging.info("Creating rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving DeepSWE rollout worker on port %d.", args.port)

    if args.sampler != "vanilla":
      logging.info("Eagerly starting sampler engine...")
      await worker_service.sampler.start()
      logging.info("Sampler engine started.")
      if hasattr(worker_service.sampler, "bind_weight_sync"):
        logging.info("Eagerly warming up weight sync...")
        await worker_service.sampler.bind_weight_sync()
        logging.info("Weight sync warmed up.")

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "rollout",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("DeepSWE rollout worker is registered.")

    try:
      while True:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
      pass
    finally:
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])