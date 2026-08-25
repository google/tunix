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

from flax import nnx
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
  parser.add_argument("--mesh_expert", type=int, default=1)
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
  parser.add_argument(
      "--trainer_backend",
      choices=("peft", "maxtext"),
      default="peft",
      help=(
          "peft runs tunix's PeftTrainer on a tunix Qwen3; maxtext runs "
          "MaxTextTrainingEngine, which builds its own model and optimizer."
      ),
  )
  parser.add_argument(
      "--log_train_steps",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Log every fwd_bwd/update with the recorded loss and the element-wise change "
          "in a slice of parameters, so a run shows whether weights actually moved."
      ),
  )
  parser.add_argument("--maxtext_model_name", type=str, default="qwen3-0.6b")
  parser.add_argument(
      "--maxtext_padded_moe_mlp_dim",
      type=int,
      default=0,
      help=(
          "Explicit padded_base_moe_mlp_dim override for MoE models. The"
          " rollout side (maxtext_vllm_adapter.generate_maxtext_config) pads"
          " moe_intermediate_size up automatically to satisfy the GMM"
          " kernel's tile-alignment requirement whenever it doesn't divide"
          " evenly by rollout tensor_parallel_size * 2 * num_lanes; the"
          " trainer never applies this padding on its own, so Raiden's"
          " weight-sync preflight fails on a global-shape mismatch for"
          " every MoE MLP tensor. Must match whatever the rollout side"
          " actually computes for its own tensor_parallel_size -- 0 leaves"
          " padding off (correct for dense models, or a rollout TP where no"
          " padding is needed)."
      ),
  )
  parser.add_argument(
      "--maxtext_load_parameters_path",
      type=str,
      default=os.getenv("MAXTEXT_CKPT", ""),
      help="Orbax params-only checkpoint for the MaxText trainer, e.g. gs://...",
  )
  parser.add_argument(
      "--maxtext_warmup_steps_fraction",
      type=float,
      default=0.0,
      help=(
          "MaxText's LR schedule warms up linearly from zero, so the default 0.0 is "
          "what makes step 0 a real update instead of a silent no-op."
      ),
  )
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
  if "0.6b" in normalized or "0p6b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_0p6b()
  elif "1.7b" in normalized or "1p7b" in normalized:
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


def _maxtext_modules():
  """Imports MaxText lazily; some installs nest it under maxtext.src.maxtext."""
  try:
    from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
    from maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
    from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  except ImportError:  # pragma: no cover - layout-dependent
    from maxtext.src.maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
    from maxtext.src.maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
    from maxtext.src.maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  return pyconfig, maxtext_engine, maxtext_utils


def _tokenizer_pad_id(args) -> int:
  """Resolves the pad token id the MaxText adapter masks with.

  Must match how run_gsm8k_dist_grpo.py derives it -- a mismatch silently
  corrupts trainer log-probs rather than raising.
  """
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def _build_maxtext_config(args, num_devices: int) -> Any:
  """Builds the MaxText HyperParameters the training engine runs on."""
  pyconfig, _, _ = _maxtext_modules()

  # MaxText shards the batch dim over fsdp; fail early instead of deep in the first step.
  if args.train_micro_batch_size % args.mesh_fsdp:
    raise ValueError(
        f"--train_micro_batch_size={args.train_micro_batch_size} must be a multiple of "
        f"--mesh_fsdp={args.mesh_fsdp}; MaxText shards the batch dimension across it."
    )
  per_device_batch_size = args.train_micro_batch_size / num_devices

  base_yml = os.path.join(
      os.path.dirname(os.path.abspath(pyconfig.__file__)), "base.yml"
  )
  if not os.path.exists(base_yml):
    raise FileNotFoundError(f"MaxText base.yml not found at {base_yml}")
  output_dir = os.path.join(REPO_ROOT, "artifacts", "qwen3_dist_gsm8k", "maxtext")

  enable_checkpointing = bool(args.maxtext_load_parameters_path)
  argv = [
      "run_trainer_node.py",
      base_yml,
      f"model_name={args.maxtext_model_name}",
      f"run_name={args.worker_id}",
      f"base_output_directory={output_dir}",
      f"enable_checkpointing={enable_checkpointing}",
  ]
  if args.maxtext_load_parameters_path:
    argv.append(f"load_parameters_path={args.maxtext_load_parameters_path}")
  argv.extend([
      # checkpoint is scanned; weight-sync mapping converts to unscanned for rollout
      #
      # Confirmed NOT the cause of the native "Push range out of bounds"
      # weight-sync error (tracked upstream with the Raiden team): the same
      # error reproduces identically with scan_layers=False (genuine
      # per-layer arrays straight from nnx.state(model), no raiden_unscan.py
      # transform involved at all). Left at True since that's required for
      # loading a scanned checkpoint anyway (MaxTextTrainingEngine's restore
      # path has no scan->unscan remap, unlike the rollout/vLLM adapter's).
      "scan_layers=True",
      # with load_parameters_path set, from_pretrained must not fall back to HF conversion
      "convert_checkpoint_if_possible=False",
      # MaxText defaults jax_cache_dir to "~/jax_cache" (base.yml), which
      # pyconfig.py wires straight into JAX's persistent compilation cache.
      # The trainer's main container has no volume mount, so that cache lands
      # on the pod's writable overlay -- on a TPU worker node's local disk,
      # which can be as small as ~44GB allocatable ephemeral-storage. A 30B-A3B
      # MoE compile (48 scanned layers, padded MoE MLP dim) produced enough
      # cached executables there to trip a kubelet ephemeral-storage eviction
      # mid-run. Each pod is fresh anyway (no cross-run reuse benefit), so
      # disable the persistent cache entirely rather than trying to size a
      # volume for it.
      "jax_cache_dir=",
      "skip_jax_distributed_system=True",  # tunix already did this
      f"per_device_batch_size={per_device_batch_size}",
      # tunix owns gradient accumulation, so MaxText must not also accumulate
      "gradient_accumulation_steps=1",
      f"max_target_length={args.max_prompt_length + args.max_response_length}",
      # flash/splash attention both dispatch to the same Pallas splash kernel,
      # which requires q_block_size (512) to divide q_seq_len exactly; it
      # doesn't for max_prompt_length + max_response_length = 640. Confirmed
      # by testing "flash" directly: ValueError: q_block_size=512 should
      # divide q_seq_len=640. dot_product is the only kernel that works at
      # this sequence length without also changing the block size.
      "attention=dot_product",
      "use_tokamax_gmm=true",
      "use_gmm_v2=true",
      f"ici_fsdp_parallelism={args.mesh_fsdp}",
      *(
          [f"padded_base_moe_mlp_dim={args.maxtext_padded_moe_mlp_dim}"]
          if args.maxtext_padded_moe_mlp_dim
          else []
      ),
      f"ici_tensor_parallelism={args.mesh_tp}",
      f"ici_expert_parallelism={args.mesh_expert}",
      f"learning_rate={args.learning_rate}",
      f"warmup_steps_fraction={args.maxtext_warmup_steps_fraction}",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "init_weights_seed=42",
  ])
  logging.info("MaxText config argv: %s", argv)
  return pyconfig.initialize(argv)


def _create_maxtext_mesh(maxtext_config) -> Mesh:
  """Builds the mesh MaxText's own sharding annotations are written against."""
  _, _, maxtext_utils = _maxtext_modules()
  devices = maxtext_utils.create_device_mesh(maxtext_config)
  return Mesh(devices, maxtext_config.mesh_axes)


class _MeshBoundTrainer:
  """Binds generic PeftTrainer v2 calls to this worker's JAX mesh."""

  def __init__(
      self,
      trainer: peft_trainer_v2.PeftTrainer,
      mesh: Mesh,
      log_train_steps: bool = True,
  ):
    self._trainer = trainer
    self._mesh = mesh
    self._log_train_steps = log_train_steps
    self._fwd_bwd_calls = 0

  def __getattr__(self, name: str) -> Any:
    return getattr(self._trainer, name)

  def _recorded_loss(self) -> float | None:
    """Reads the loss the trainer just recorded without clearing its buffer."""
    getter = getattr(self._trainer, "get_metrics", None)
    if getter is None:
      return None
    try:
      buffer = getter(clear_cache=False)
    except TypeError:
      return None
    for bucket in (
        getattr(buffer, "weighted_metrics", None) or {},
        getattr(buffer, "scalar_metrics", None) or {},
    ):
      for key, value in bucket.items():
        if key.split("/")[-1] == "loss":
          value = value.compute() if hasattr(value, "compute") else value
          return float(jnp.asarray(value).reshape(-1)[0])
    return None

  def fwd_bwd(self, *args, **kwargs) -> None:
    with self._mesh:
      self._trainer.fwd_bwd(*args, **kwargs)
    if self._log_train_steps:
      self._fwd_bwd_calls += 1
      logging.info(
          "[train] fwd_bwd #%d done: micro_steps=%s has_accumulated_grads=%s",
          self._fwd_bwd_calls,
          getattr(self._trainer, "micro_step_count", "n/a"),
          getattr(self._trainer, "has_accumulated_grads", "n/a"),
      )

  def update(self, **kwargs) -> int:
    with self._mesh:
      train_step = self._trainer.update(**kwargs)
    if self._log_train_steps:
      loss = self._recorded_loss()
      logging.info(
          "[train] update -> train_step=%s loss=%s",
          train_step,
          "unrecorded" if loss is None else f"{loss:.6f}",
      )
    return train_step

  def eval_step(self, *args, **kwargs) -> None:
    with self._mesh:
      self._trainer.eval_step(*args, **kwargs)

  @contextlib.contextmanager
  def eval_context(self):
    with self._mesh:
      with self._trainer.eval_context():
        yield

  def compile(self, *args, **kwargs) -> None:
    # dummy_data=None is fine here: PeftTrainer.compile is a no-op, and the engine
    # defers to the first fwd_bwd anyway, so a synthetic payload would just risk
    # diverging from the real one (e.g. --beta != 0 adds ref_per_token_logps).
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
  return _MeshBoundTrainer(trainer, mesh, log_train_steps=args.log_train_steps)


def _create_maxtext_trainer(
    args,
    maxtext_config: Any,
    mesh: Mesh,
    pad_id: int,
) -> _MeshBoundTrainer:
  """Builds MaxTextTrainingEngine as the trainer behind TrainerWorker.

  A trainer swap, not a model swap: the engine builds its own model and optimizer,
  so no tunix model, optax transform, or PeftTrainer TrainingConfig is involved.
  """
  _, maxtext_engine, _ = _maxtext_modules()

  with mesh:
    engine = maxtext_engine.MaxTextTrainingEngine(
        maxtext_config,
        mesh=mesh,
        # grpo_loss_fn calls the model with tunix's signature
        wrap_with_tunix_adapter=True,
        tokenizer_pad_id=pad_id,
    )

  model_type = type(engine.model).__name__
  logging.info("MaxText engine model: %s (pad_id=%d)", model_type, pad_id)
  if model_type != "TunixMaxTextAdapter":
    raise RuntimeError(
        f"Expected the engine's model to be TunixMaxTextAdapter, got {model_type}."
    )
  _log_param_shapes(engine.model)

  return _MeshBoundTrainer(
      engine,
      mesh,
      log_train_steps=args.log_train_steps,
  )


def _log_param_shapes(model: Any) -> None:
  """Logs a scanned-layer parameter shape as a sanity check that a checkpoint loaded."""
  flat = nnx.to_pure_dict(nnx.state(model, nnx.Param))

  def walk(node, path=""):
    if isinstance(node, dict):
      for key, value in node.items():
        yield from walk(value, f"{path}.{key}" if path else str(key))
    elif hasattr(node, "shape"):
      yield path, node.shape

  shapes = dict(walk(flat))
  for name, shape in shapes.items():
    if "wi_0" in name or "query" in name:
      logging.info("MaxText param %s shape=%s", name, shape)
  logging.info("MaxText model has %d parameter arrays.", len(shapes))


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

  if args.train_micro_batch_size <= 0:
    raise ValueError("--train_micro_batch_size must be positive.")

  # The HF safetensors directory is needed for the tokenizer on both paths, and for the
  # actor weights on the peft path only.
  if args.trainer_backend != "maxtext" or args.maxtext_load_parameters_path:
    args.model_dir = _ensure_model_dir_for_trainer(args.model_dir, args.model_id)
    logging.info("Prepared trainer safetensors directory: %s", args.model_dir)

  if args.trainer_backend == "maxtext":
    logging.info("Trainer backend: MaxTextTrainingEngine.")
    pad_id = _tokenizer_pad_id(args)
    maxtext_config = _build_maxtext_config(args, jax.device_count())
    logging.info("Creating MaxText device mesh...")
    mesh = _create_maxtext_mesh(maxtext_config)
    logging.info("Trainer mesh: %s", mesh)
    trainer_factory = lambda: _create_maxtext_trainer(  # pylint: disable=g-long-lambda
        args, maxtext_config, mesh, pad_id
    )
  else:
    logging.info("Trainer backend: PeftTrainer v2.")
    logging.info("Creating trainer mesh...")
    mesh = _create_mesh(args)
    logging.info("Trainer mesh: %s", mesh)

    logging.info("Loading actor model with use_lora=%s...", args.use_lora)
    actor_model = _load_qwen3(args, mesh, lora=args.use_lora)

    logging.info("Building PeftTrainer v2 config...")
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
    trainer_factory = lambda: _create_trainer(  # pylint: disable=g-long-lambda
        args, actor_model, training_config, mesh
    )

  logging.info("Creating generic TrainerWorker and gRPC server...")
  worker_service = trainer_worker.TrainerWorker(
      trainer_factory=trainer_factory,
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
