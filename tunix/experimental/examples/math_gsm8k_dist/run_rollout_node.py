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

"""vLLM rollout worker process runner for the distributed GRPO demo."""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--tpu_chips", type=str, default="2,3")
  parser.add_argument(
      "--tpu_chips_per_host_bounds",
      type=str,
      default=os.getenv("TPU_CHIPS_PER_HOST_BOUNDS", ""),
      help="Optional 3D TPU chip bounds for this worker, e.g. 1,2,1.",
  )
  parser.add_argument(
      "--tpu_host_bounds",
      type=str,
      default=os.getenv("TPU_HOST_BOUNDS", "1,1,1"),
      help="Optional 3D TPU host bounds for this worker.",
  )
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", ""))
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  return parser.parse_args()


def _parse_tpu_chips(tpu_chips: str) -> list[str]:
  chips = [chip.strip() for chip in tpu_chips.split(",") if chip.strip()]
  if not chips:
    raise ValueError("--tpu_chips must contain at least one chip id.")
  return chips


def _default_chips_per_host_bounds(num_chips: int) -> str:
  if num_chips == 1:
    return "1,1,1"
  if num_chips == 2:
    return "1,2,1"
  return ""


def _set_libtpu_init_arg(name: str, value: str) -> None:
  prefix = f"--{name}="
  existing_args = [
      arg
      for arg in os.environ.get("LIBTPU_INIT_ARGS", "").split()
      if not arg.startswith(prefix)
  ]
  existing_args.append(f"{prefix}{value}")
  os.environ["LIBTPU_INIT_ARGS"] = " ".join(existing_args).strip()


def _configure_logging() -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )


def _configure_tpu_visibility(parsed_args: argparse.Namespace) -> None:
  chips = _parse_tpu_chips(parsed_args.tpu_chips)
  visible_devices = ",".join(chips)
  os.environ.setdefault("JAX_PLATFORMS", "tpu")
  os.environ["TPU_VISIBLE_DEVICES"] = visible_devices
  os.environ["TPU_VISIBLE_CHIPS"] = visible_devices

  chips_per_host_bounds = (
      parsed_args.tpu_chips_per_host_bounds
      or _default_chips_per_host_bounds(len(chips))
  )
  if not chips_per_host_bounds:
    return
  host_bounds = parsed_args.tpu_host_bounds or "1,1,1"
  os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = chips_per_host_bounds
  os.environ["TPU_HOST_BOUNDS"] = host_bounds
  _set_libtpu_init_arg(
      "deepsea_chips_per_host_bounds", chips_per_host_bounds
  )
  _set_libtpu_init_arg("deepsea_host_bounds", host_bounds)


args = _parse_args()
_configure_tpu_visibility(args)
_configure_logging()
logging.info("Parsed args: %s", args)
logging.info("Pre-import JAX_PLATFORMS=%s", os.getenv("JAX_PLATFORMS"))
logging.info(
    "Pre-import TPU_VISIBLE_DEVICES=%s", os.getenv("TPU_VISIBLE_DEVICES")
)
logging.info(
    "Pre-import TPU_CHIPS_PER_HOST_BOUNDS=%s",
    os.getenv("TPU_CHIPS_PER_HOST_BOUNDS"),
)
logging.info("Pre-import TPU_HOST_BOUNDS=%s", os.getenv("TPU_HOST_BOUNDS"))
logging.info("Pre-import LIBTPU_INIT_ARGS=%s", os.getenv("LIBTPU_INIT_ARGS"))

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)
logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

logging.info("Importing JAX and rollout dependencies...")
import jax  # pylint: disable=g-import-not-at-top
from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

from tunix.experimental.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top
from tunix.experimental.rollout import legacy_vllm_sampler_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import rollout_worker  # pylint: disable=g-import-not-at-top
from tunix.generate import mappings as mappings_lib  # pylint: disable=g-import-not-at-top
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib  # pylint: disable=g-import-not-at-top
from tunix.generate import vllm_sampler  # pylint: disable=g-import-not-at-top
from tunix.models.qwen3 import mapping_vllm_jax  # pylint: disable=g-import-not-at-top
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib  # pylint: disable=g-import-not-at-top
logging.info("Finished importing rollout dependencies.")


def _create_rollout_mesh() -> Mesh:
  shape = (1, jax.device_count())
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("dp", "tp"))


def _create_vllm_worker(tokenizer):
  logging.info("Creating vLLM mapping config...")
  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = args.model_dir or args.model_id
  rollout_mesh = _create_rollout_mesh()
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      jax.device_count(),
      max_model_len,
  )
  vllm_config = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=jax.device_count(),
      data_parallel_size=1,
      return_logprobs=True,
      lora_config=(
          {
              "max_lora_rank": args.lora_rank,
              "max_loras": 1,
          }
          if args.use_lora
          else None
      ),
      mapping_config=mapping_config,
      engine_kwargs={
          "model": vllm_model,
          "max_model_len": max_model_len,
      },
  )
  sampler_adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
      server_id="vllm-rollout-0",
      tokenizer=tokenizer,
      config=vllm_config,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
  config = rollout_worker.RolloutConfig(
      sampler_type="legacy_vllm",
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      rollout_vllm_model_version=vllm_model,
      env_name=gsm8k.GSM8K_ENV_NAME,
      agent_name=gsm8k.GSM8K_AGENT_NAME,
  )
  return rollout_worker.RolloutWorker(
      worker_id="vllm-rollout-0",
      config=config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def main() -> None:
  logging.info("Initializing JAX on TPU chips: %s using vLLM.", args.tpu_chips)
  logging.info("TPU_VISIBLE_DEVICES=%s", os.getenv("TPU_VISIBLE_DEVICES"))
  logging.info(
      "TPU_CHIPS_PER_HOST_BOUNDS=%s", os.getenv("TPU_CHIPS_PER_HOST_BOUNDS")
  )
  logging.info("TPU_HOST_BOUNDS=%s", os.getenv("TPU_HOST_BOUNDS"))
  logging.info("LIBTPU_INIT_ARGS=%s", os.getenv("LIBTPU_INIT_ARGS"))
  logging.info("Visible devices: %s", jax.devices())

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  logging.info("Creating rollout worker service...")
  worker_service = _create_vllm_worker(tokenizer)
  logging.info("Creating rollout gRPC server...")
  server = remote_execution.GrpcRemoteExecutionServer(worker_service)
  logging.info("Serving vLLM rollout worker on port %d.", args.port)
  server.start_serving(args.port)


if __name__ == "__main__":
  main()
