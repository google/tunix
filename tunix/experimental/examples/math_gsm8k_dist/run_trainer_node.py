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

"""Trainer worker process runner for the experimental distributed GRPO demo."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import optax
from tunix.cli.utils import model as model_utils
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_MODEL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, "artifacts", "qwen3_dist_gsm8k", "models"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="JAX trainer worker process")
  parser.add_argument("--port", type=int, default=20000)
  parser.add_argument("--worker_id", type=str, default="trainer-0")
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
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--mini_batch_size", type=int, default=1)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)
  parser.add_argument("--compute_logps_chunk_size", type=int, default=0)
  parser.add_argument("--eval_every_n_steps", type=int, default=1000000)
  parser.add_argument("--learning_rate", type=float, default=2.0e-7)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  return parser.parse_args(argv)


def _nested_safetensors_dirs(model_dir: Path) -> list[str]:
  candidates: dict[str, int] = {}
  model_depth = len(model_dir.parts)
  for root, dirnames, files in os.walk(model_dir):
    root_path = Path(root)
    if len(root_path.parts) - model_depth >= 5:
      dirnames[:] = []
    safetensors_count = sum(
        1 for file_name in files if file_name.endswith(".safetensors")
    )
    if safetensors_count and root_path != model_dir:
      candidates[str(root_path)] = safetensors_count
    if len(candidates) >= 20:
      dirnames[:] = []
      break
  return [
      f"{path} ({count} safetensors)"
      for path, count in sorted(candidates.items())
  ]


def _has_direct_safetensors(model_path: Path) -> bool:
  return any(model_path.glob("*.safetensors"))


def _ensure_model_dir_for_trainer(model_dir: str, model_id: str) -> str:
  if not model_dir:
    raise ValueError(
        "--model_dir is required for JAX trainer weights. Set MODEL_DIR or pass "
        "--model_dir=/path/to/local/qwen3/safetensors."
    )

  model_path = Path(model_dir).expanduser()
  if model_path.exists() and not model_path.is_dir():
    raise ValueError(
        "--model_dir must point to an existing local directory. "
        f"Got: {model_dir}"
    )

  if _has_direct_safetensors(model_path):
    return str(model_path)

  logging.info(
      "No direct safetensors found in %s. Downloading %s before importing JAX.",
      model_path,
      model_id,
  )
  nested_dirs = _nested_safetensors_dirs(model_path)
  if nested_dirs:
    logging.info(
        "Nested safetensors candidates were found, but the trainer loader "
        "expects direct shards:\n  %s",
        "\n  ".join(nested_dirs),
    )
  model_path.mkdir(parents=True, exist_ok=True)
  from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

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
        "Trainer mesh dimensions must multiply to visible JAX device count. "
        f"Got shape={shape}, devices={jax.device_count()}."
    )
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))


def _load_qwen3(args, mesh: Mesh, *, lora: bool):
  if not args.model_dir:
    raise ValueError(
        "--model_dir is required for JAX trainer weights. Set MODEL_DIR or pass "
        "--model_dir=/path/to/local/qwen3/safetensors."
    )
  config = _qwen3_config(args.model_name)
  model = qwen3_params_lib.create_model_from_safe_tensors(
      args.model_dir, config, mesh, dtype=jnp.bfloat16
  )
  if not lora:
    return model
  lora_config = {
      "module_path": (
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      "rank": args.lora_rank,
      "alpha": args.lora_alpha,
  }
  return model_utils.apply_lora_to_model(model, mesh=mesh, lora_config=lora_config)


class _MeshBoundTrainer:
  """Binds generic PeftTrainer v2 calls to this worker's JAX mesh."""

  def __init__(self, trainer: peft_trainer_v2.PeftTrainer, mesh: Mesh):
    self._trainer = trainer
    self._mesh = mesh

  def __getattr__(self, name: str) -> Any:
    return getattr(self._trainer, name)

  def fwd_bwd(self, *args, **kwargs) -> None:
    with self._mesh:
      self._trainer.fwd_bwd(*args, **kwargs)

  def update(self, **kwargs) -> int:
    with self._mesh:
      return self._trainer.update(**kwargs)

  def eval_step(self, *args, **kwargs) -> None:
    with self._mesh:
      self._trainer.eval_step(*args, **kwargs)

  @contextlib.contextmanager
  def eval_context(self):
    with self._mesh:
      with self._trainer.eval_context():
        yield

  def compile(self, *args, **kwargs) -> None:
    with self._mesh:
      self._trainer.compile(*args, **kwargs)

  def prepare_weight_sync(self, **kwargs) -> Any:
    with self._mesh:
      return self._trainer.prepare_weight_sync(**kwargs)

  def close(self) -> None:
    with self._mesh:
      self._trainer.close()


def _create_trainer(
    args,
    actor_model: Any,
    training_config: peft_trainer_v2.TrainingConfig,
    mesh: Mesh,
) -> _MeshBoundTrainer:
  with mesh:
    trainer = peft_trainer_v2.PeftTrainer(
        actor_model,
        optax.adamw(learning_rate=args.learning_rate),
        training_config,
    )
  return _MeshBoundTrainer(trainer, mesh)


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [TrainerNode] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  logging.info("Parsed args: %s", args)

  if context:
    context.jax.initialize()
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  args.model_dir = _ensure_model_dir_for_trainer(args.model_dir, args.model_id)
  logging.info("Prepared trainer safetensors directory: %s", args.model_dir)

  logging.info("Creating trainer mesh...")
  mesh = _create_mesh(args)
  logging.info("Trainer mesh: %s", mesh)

  logging.info("Loading actor model with use_lora=%s...", args.use_lora)
  actor_model = _load_qwen3(args, mesh, lora=args.use_lora)

  logging.info("Building PeftTrainer v2 config...")
  if args.train_micro_batch_size <= 0:
    raise ValueError("--train_micro_batch_size must be positive.")
  grad_accumulation_steps = max(
      1, math.ceil(args.mini_batch_size / args.train_micro_batch_size)
  )
  training_config = peft_trainer_v2.TrainingConfig(
      eval_every_n_steps=args.eval_every_n_steps,
      gradient_accumulation_steps=grad_accumulation_steps,
      metrics_prefix="actor",
      pbar_description="Actor Training",
      data_sharding_axis=("fsdp",),
  )
  logging.info(
      "PeftTrainer v2 gradient_accumulation_steps=%d.",
      grad_accumulation_steps,
  )

  logging.info("Creating generic TrainerWorker and gRPC server...")
  worker_service = trainer_worker.TrainerWorker(
      trainer_factory=lambda: _create_trainer(
          args, actor_model, training_config, mesh
      ),
      worker_id=args.worker_id,
  )

  async def grpc_server_main() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving trainer worker on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "trainer",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Trainer worker is registered.")

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
