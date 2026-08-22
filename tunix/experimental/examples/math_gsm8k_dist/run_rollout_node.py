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
import asyncio
import logging
import os
import pickle
import sys

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer
from tunix.experimental.examples.math_gsm8k_dist import gsm8k
from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker
from tunix.generate import mappings as mappings_lib
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
from tunix.generate import vllm_sampler
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  parser.add_argument("--tensor_parallel_size", type=int, default=4)
  parser.add_argument(
      "--sampler_type",
      type=str,
      default="vllm",
      choices=["vllm", "legacy_vllm"],
  )
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      default="",
      help=(
          "MaxText model_name (e.g. qwen3-0.6b). When set, the vLLM sampler"
          " loads the model natively as MaxText's own module tree"
          " (maxtext_vllm_adapter's MaxTextForCausalLM) instead of"
          " tpu-inference's own reimplementation, so trainer and rollout"
          " share identical Raiden weight-sync tensor names."
      ),
  )
  return parser.parse_args(argv)


def _create_rollout_mesh() -> Mesh:
  shape = (1, jax.device_count())
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))


def _create_vllm_worker(args, tokenizer):
  vllm_model = (
      args.model_dir
      if (
          args.model_dir
          and os.path.exists(args.model_dir)
          and any(os.scandir(args.model_dir))
      )
      else args.model_id
  )
  max_model_len = args.max_prompt_length + args.max_response_length

  if getattr(args, "sampler_type", "vllm") == "vllm":
    from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
    from tunix.experimental.rollout import vllm_sampler_adapter  # pylint: disable=g-import-not-at-top

    tp_size = args.tensor_parallel_size
    logging.info(
        "Creating vLLM RLVllmSampler config for model=%s tp_size=%d "
        "max_model_len=%d...",
        vllm_model,
        tp_size,
        max_model_len,
    )
    engine_kwargs = dict(
        model=vllm_model,
        tokenizer=args.tokenizer_path or vllm_model,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=args.use_lora,
        max_lora_rank=args.lora_rank if args.use_lora else None,
        max_loras=1 if args.use_lora else None,
    )
    if args.maxtext_model_name:
      logging.info(
          "Loading MaxText model %r natively via maxtext_vllm_adapter's"
          " MaxTextForCausalLM (architectures override).",
          args.maxtext_model_name,
      )
      engine_kwargs["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
      engine_kwargs["additional_config"] = {
          "maxtext_config": {"model_name": args.maxtext_model_name}
      }
    engine_args = AsyncEngineArgs(**engine_kwargs)
    sampler_adapter = vllm_sampler_adapter.VllmSamplerAdapter(
        server_id=args.worker_id,
        engine_args=engine_args,
        model_name=vllm_model,
    )
    rollout_config = rollout_worker.RolloutConfig(
        sampler_type="vllm",
        max_prompt_length=args.max_prompt_length,
        max_tokens_to_generate=args.max_response_length,
        temperature=1.0,
        top_p=1.0,
        return_logprobs=True,
        rollout_vllm_model_version=vllm_model,
    )
  else:
    logging.info("Creating vLLM mapping config...")
    mapping_config = mappings_lib.MappingConfig(
        lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
    )
    rollout_mesh = _create_rollout_mesh()
    logging.info(
        "Creating legacy vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
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
        server_id=args.worker_id,
        tokenizer=tokenizer,
        config=vllm_config,
    )
    rollout_config = rollout_worker.RolloutConfig(
        sampler_type="legacy_vllm",
        max_prompt_length=args.max_prompt_length,
        max_tokens_to_generate=args.max_response_length,
        temperature=1.0,
        top_p=1.0,
        return_logprobs=True,
        rollout_vllm_model_version=vllm_model,
    )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=rollout_config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  logging.info("Parsed args: %s", args)

  if context and getattr(args, "sampler_type", "vllm") != "vllm":
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  if args.maxtext_model_name:
    # configs/inference/vllm.yml's logical_axis_rules reference a 7-axis mesh
    # (data/attn_dp/attn_dp_expert/expert/model/dcp/pcp). tpu-inference only
    # builds that mesh under NEW_MODEL_DESIGN (tpu_runner.py's _init_mesh);
    # its default path builds a 2-axis (data, model) mesh instead, which is
    # missing every axis the MaxText adapter's rules reference and crashes
    # model construction (nnx.eval_shape) before weight sync is even reached.
    # Matches the reference MaxText+RL integration test
    # (single_host_train_rl_test.py), which sets this for the same reason.
    # Must be set before any vllm/tpu_inference import reads it, hence here
    # rather than inside _create_vllm_worker.
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)


  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  async def grpc_server_main() -> None:
    logging.info("Creating rollout worker service...")
    worker_service = _create_vllm_worker(args, tokenizer)

    logging.info("Creating rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving vLLM rollout worker on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "rollout",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Rollout worker is registered.")

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
