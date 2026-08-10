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

"""Reference inference worker process runner for the distributed GRPO demo."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer

from tunix.experimental.worker import inference_worker as exp_inference_worker
from tunix.experimental.worker import remote_execution
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl.inference import inference_worker as rl_inference_worker


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_MODEL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, "artifacts", "qwen3_dist_gsm8k", "models"
)


def _has_direct_safetensors(model_path: Path) -> bool:
  return any(model_path.glob("*.safetensors"))


def _ensure_model_dir(model_dir: str, model_id: str) -> str:
  if not model_dir:
    raise ValueError("--model_dir is required for reference model weights.")
  model_path = Path(model_dir).expanduser()
  if model_path.exists() and not model_path.is_dir():
    raise ValueError(f"--model_dir must be a directory. Got: {model_dir}")
  if _has_direct_safetensors(model_path):
    return str(model_path)

  logging.info(
      "No direct safetensors found in %s. Downloading %s before importing JAX.",
      model_path,
      model_id,
  )
  model_path.mkdir(parents=True, exist_ok=True)
  from tunix.oss import utils as oss_utils

  oss_utils.hf_pipeline(model_id, str(model_path))
  if _has_direct_safetensors(model_path):
    return str(model_path)
  raise ValueError(
      "Download completed, but no '*.safetensors' files were found directly "
      f"in --model_dir: {model_path}"
  )

def _qwen3_config(model_name: str) -> qwen3_model_lib.ModelConfig:
  normalized = model_name.lower().replace("_", "-")
  if "1.7b" in normalized or "1p7b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  elif "32b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_32b()
  else:
    raise ValueError(f"Unsupported demo model_name: {model_name!r}")
  config.shd_config = qwen3_model_lib.ShardingConfig.get_default_sharding()
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  return config


def _create_mesh(args) -> Mesh:
  shape = (args.mesh_fsdp, args.mesh_tp)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "Inference mesh dimensions must multiply to visible JAX device count. "
        f"Got shape={shape}, devices={jax.device_count()}."
    )
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))


class _MeshBoundReferenceCore:
  """Runs the shared RL inference core under this worker's mesh."""

  def __init__(self, core: rl_inference_worker.InferenceWorker, mesh: Mesh):
    self._core = core
    self._mesh = mesh

  def get_ref_per_token_logps(self, *args, **kwargs) -> Any:
    with self._mesh:
      return self._core.get_ref_per_token_logps(*args, **kwargs)

  def get_rewards(self, *args, **kwargs) -> Any:
    with self._mesh:
      return self._core.get_rewards(*args, **kwargs)


def main(argv: list[str], context: Any = None) -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [InferenceNode] %(message)s",
      force=True,
  )

  parser = argparse.ArgumentParser(description="Reference inference worker")
  parser.add_argument("--port", type=int, default=20002)
  parser.add_argument("--worker_id", type=str, default="reference-inference-0")
  parser.add_argument("--tpu_chips", type=str, default="0,1")
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
  parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir",
      type=str,
      default=os.getenv(
          "MODEL_DIR",
          os.getenv("MODEL_DOWNLOAD_DIR", DEFAULT_MODEL_DOWNLOAD_DIR),
      ),
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--mesh_fsdp", type=int, default=2)
  parser.add_argument("--mesh_tp", type=int, default=1)
  parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

  # 1. parse args
  args = parser.parse_args(argv)
  logging.info("Parsed args: %s", args)

  # 2. init JAX, import PATH
  if context:
    context.jax.initialize()
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  args.model_dir = _ensure_model_dir(args.model_dir, args.model_id)
  logging.info("Prepared reference safetensors directory: %s", args.model_dir)

  # 3. load tokenizer
  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  # 4. create worker
  logging.info("Creating inference mesh...")
  mesh = _create_mesh(args)
  logging.info("Inference mesh: %s", mesh)

  logging.info("Loading frozen reference model...")
  with mesh:
    reference_model = qwen3_params_lib.create_model_from_safe_tensors(
        args.model_dir,
        _qwen3_config(args.model_name),
        mesh,
        dtype=jnp.bfloat16,
    )
    core = rl_inference_worker.InferenceWorker({"reference": reference_model})

  worker_service = exp_inference_worker.InferenceWorker(
      _MeshBoundReferenceCore(core, mesh),
      worker_id=args.worker_id,
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=args.compute_logps_micro_batch_size,
  )

  # 5. start rpc endpoint
  async def grpc_server_main() -> None:
    logging.info("Starting inference gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving reference inference worker on port %d.", args.port)

    # 6. register
    if context and context.ipc and context.ipc.discovery:
      context.ipc.discovery.register(
          metadata=pickle.dumps({
              "service_type": "inference",
              "service_port": args.port,
              "worker_id": args.worker_id,
              "model_id": args.model_id,
          })
      )
      logging.info("Inference worker is registered.")
    else:
      logging.info("Inference worker is not registered: process context doesn't support.")

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
