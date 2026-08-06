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

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_MODEL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, "artifacts", "qwen3_dist_gsm8k", "models"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Reference inference worker")
  parser.add_argument("--port", type=int, default=20002)
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
      format="%(asctime)s - [InferenceNode] %(message)s",
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
  from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

  oss_utils.hf_pipeline(model_id, str(model_path))
  if _has_direct_safetensors(model_path):
    return str(model_path)
  raise ValueError(
      "Download completed, but no '*.safetensors' files were found directly "
      f"in --model_dir: {model_path}"
  )


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
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)
logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)
args.model_dir = _ensure_model_dir(args.model_dir, args.model_id)
logging.info("Prepared reference safetensors directory: %s", args.model_dir)

logging.info("Importing JAX and inference dependencies...")
import jax  # pylint: disable=g-import-not-at-top
from jax import numpy as jnp  # pylint: disable=g-import-not-at-top
from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

from tunix.experimental.worker import inference_worker as exp_inference_worker  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.models.qwen3 import model as qwen3_model_lib  # pylint: disable=g-import-not-at-top
from tunix.models.qwen3 import params as qwen3_params_lib  # pylint: disable=g-import-not-at-top
from tunix.rl.inference import inference_worker as rl_inference_worker  # pylint: disable=g-import-not-at-top
logging.info("Finished importing inference dependencies.")


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


def _create_mesh() -> Mesh:
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


def main() -> None:
  logging.info("Initializing JAX on TPU chips: %s", args.tpu_chips)
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
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  logging.info("Creating inference mesh...")
  mesh = _create_mesh()
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
      worker_id="reference-inference-0",
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=args.compute_logps_micro_batch_size,
  )
  server = remote_execution.GrpcRemoteExecutionServer(worker_service)
  logging.info("Serving reference inference worker on port %d.", args.port)
  server.start_serving(args.port)


if __name__ == "__main__":
  main()
