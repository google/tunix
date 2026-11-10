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

"""Reference inference worker process runner shared by distributed RL examples."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

from transformers import AutoTokenizer
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.worker import inference_worker as exp_inference_worker
from tunix.dist_rl.worker import remote_execution
from tunix.models import automodel
from tunix.oss import utils as oss_utils
from tunix.rl.inference import inference_worker as rl_inference_worker

if TYPE_CHECKING:
  from jax.sharding import Mesh

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_MODEL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, "artifacts", "distributed_examples", "models"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Reference inference worker")
  parser.add_argument("--port", type=int, default=20002)
  parser.add_argument("--worker_id", type=str, default="reference-inference-0")
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
  parser.add_argument(
      "--model_dtype",
      choices=("bfloat16", "float32"),
      default=os.getenv("MODEL_DTYPE", "float32"),
      help=(
          "Data type for the reference model parameters and computation"
          " (e.g. 'bfloat16', 'float32')."
      ),
  )
  parser.add_argument("--mesh_fsdp", type=int, default=2)
  parser.add_argument("--mesh_tp", type=int, default=1)
  parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument(
      "--run_id",
      type=str,
      default=os.getenv("RUN_ID", ""),
      help="Run identifier shared between orchestrator and inference workers.",
  )
  return parser.parse_args(argv)


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
  oss_utils.hf_pipeline(model_id, str(model_path))
  if _has_direct_safetensors(model_path):
    return str(model_path)
  raise ValueError(
      "Download completed, but no '*.safetensors' files were found directly "
      f"in --model_dir: {model_path}"
  )


def _create_mesh(args: argparse.Namespace) -> Mesh:
  """Creates the 2D `(fsdp, tp)` JAX device mesh for inference."""
  import jax  # pylint: disable=g-import-not-at-top
  from jax import sharding as jax_sharding  # pylint: disable=g-import-not-at-top
  from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top

  shape = (args.mesh_fsdp, args.mesh_tp)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "Inference mesh dimensions must multiply to visible JAX device count. "
        f"Got shape={shape}, devices={jax.device_count()}."
    )
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return jax_sharding.Mesh(devices, axis_names=("fsdp", "tp"))


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [InferenceNode] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  logging.info("Parsed args: %s", args)

  otel_setup.setup_metrics(
      default_service_name="tunix-inference",
      resource_attributes={
          "tunix.worker_id": args.worker_id,
          "tunix.role": "inference",
          "tunix.run_id": args.run_id,
      },
  )

  if context:
    context.jax.initialize()
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  args.model_dir = _ensure_model_dir(args.model_dir, args.model_id)
  logging.info("Prepared reference safetensors directory: %s", args.model_dir)

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer: Any = AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  logging.info("Creating inference mesh...")
  mesh = _create_mesh(args)
  logging.info("Inference mesh: %s", mesh)

  logging.info("Loading frozen reference model...")
  with mesh:
    reference_model, _ = automodel.AutoModel.from_pretrained(
        model_id=args.model_name,
        mesh=mesh,
        model_path=args.model_dir,
        dtype=args.model_dtype,
        load_dtype=args.model_dtype,
    )
    core = rl_inference_worker.InferenceWorker({"reference": reference_model})

  worker_service = exp_inference_worker.InferenceWorker(
      core,
      worker_id=args.worker_id,
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=args.compute_logps_micro_batch_size,
      execution_context=mesh,
  )

  asyncio.run(
      remote_execution.serve_worker(
          worker_service,
          port=args.port,
          discovery=context.ipc.discovery,
          metadata={
              "service_type": "inference",
              "service_port": args.port,
              "worker_id": args.worker_id,
          },
      )
  )


if __name__ == "__main__":
  main(sys.argv[1:])
