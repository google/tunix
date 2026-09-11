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

"""Trainer worker process runner shared by distributed RL examples."""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import logging
import math
import os
from pathlib import Path
import pickle
import signal
import sys
from typing import Any

from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from orbax import checkpoint as ocp
from tunix.cli import config as cli_config
from tunix.cli.utils import model as model_utils
from tunix.experimental.examples.common import models
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker
from tunix.utils import maxtext_utils

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_MODEL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, "artifacts", "distributed_examples", "models"
)


_OPTIMIZER_ARG_PREFIX = "optimizer_"


def _optimizer_config_from_args(args) -> dict[str, Any]:
  """Collects the `--optimizer_*` flags into a cli optimizer config dict.

  Every flag named `--optimizer_<key>` becomes `<key>` in the returned dict,
  which mirrors the `optimizer_config` block of `tunix/cli/base_config.yaml`
  and is the keyword mapping `tunix.cli.config.create_optimizer` consumes:
  `opt_type` picks the `optax` optimizer, `schedule_type` picks the
  `optax.schedules` function, and the remaining keys are forwarded to whichever
  of the two declares them. Keys that neither declares are ignored, and flags
  left unset are omitted so that the `optax` defaults apply.

  Args:
    args: Parsed CLI namespace.

  Returns:
    The optimizer config dict.
  """
  return {
      name.removeprefix(_OPTIMIZER_ARG_PREFIX): value
      for name, value in vars(args).items()
      if name.startswith(_OPTIMIZER_ARG_PREFIX) and value is not None
  }


def _build_optimizer(args):
  """Builds the actor optimizer from the `--optimizer_*` CLI flags.

  Defaults reproduce the previous bare optax.adamw with a constant learning
  rate (no clipping, optax defaults b2=0.999 / weight_decay=0.0).

  The optimizer is built by `tunix.cli.config.create_optimizer` from the flag
  dict, so the runner and the YAML driven cli share one implementation:
  `--optimizer_opt_type` picks the `optax` optimizer and
  `--optimizer_schedule_type` picks the `optax.schedules` function.

  Gradient clipping is not an optimizer keyword but a separate `optax`
  transformation, so it is declared by `--optimizer_opt_chain_type` (e.g.
  `clip_by_global_norm`) and `--optimizer_chain_kwargs` (e.g.
  `{'max_norm': 1.0}`), which `create_optimizer` chains ahead of the optimizer.

  Args:
    args: Parsed CLI namespace.

  Returns:
    The optimizer, preceded by the chained transformation when configured.
  """
  return cli_config.create_optimizer(
      _optimizer_config_from_args(args), "run_trainer_node"
  )


def _str2bool(v: str | bool) -> bool:
  """Converts string representations of booleans to bool."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _parse_mapping(v: str | dict[str, Any]) -> dict[str, Any]:
  """Parses a mapping literal passed on the command line.

  Both JSON (`{"max_norm": 1.0}`) and Python (`{'max_norm': 1.0}`) literals are
  accepted, because nesting double quotes inside the launcher startup commands
  requires several levels of shell escaping.

  Args:
    v: The mapping literal, or an already parsed mapping.

  Returns:
    The mapping, empty when `v` is empty.

  Raises:
    argparse.ArgumentTypeError: If `v` is not a mapping literal.
  """
  if isinstance(v, dict):
    return v
  if not v:
    return {}
  try:
    parsed = ast.literal_eval(v)
  except (ValueError, SyntaxError) as e:
    raise argparse.ArgumentTypeError(
        f"Expected a mapping literal, got {v!r}: {e}"
    ) from e
  if not isinstance(parsed, dict):
    raise argparse.ArgumentTypeError(
        f"Expected a mapping literal, got {type(parsed).__name__} from {v!r}."
    )
  return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments for trainer worker process."""
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
  parser.add_argument(
      "--mini_batch_size",
      type=int,
      default=1,
      help="Number of prompt groups per optimizer update.",
  )
  parser.add_argument(
      "--num_generations",
      type=int,
      default=1,
      help="Number of rollout trajectories generated per prompt group.",
  )
  parser.add_argument(
      "--train_micro_batch_size",
      type=int,
      default=1,
      help="Number of trajectories per forward/backward microbatch.",
  )
  parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)
  parser.add_argument("--compute_logps_chunk_size", type=int, default=0)
  parser.add_argument("--eval_every_n_steps", type=int, default=1000000)
  parser.add_argument(
      "--optimizer_opt_type",
      type=str,
      default="adamw",
      help="Name of the `optax` optimizer to build (e.g. adamw, sgd).",
  )
  parser.add_argument(
      "--optimizer_learning_rate",
      "--learning_rate",
      dest="optimizer_learning_rate",
      type=float,
      default=2.0e-7,
      help=(
          "Constant learning rate. Ignored when --optimizer_schedule_type"
          " builds a schedule."
      ),
  )
  parser.add_argument(
      "--optimizer_opt_chain_type",
      type=str,
      default=None,
      help=(
          "Name of an `optax` gradient transformation chained ahead of the"
          " optimizer (e.g. clip_by_global_norm). Unset leaves the optimizer"
          " unchained."
      ),
  )
  parser.add_argument(
      "--optimizer_chain_kwargs",
      type=_parse_mapping,
      default={},
      help=(
          "Arguments of --optimizer_opt_chain_type as a mapping literal, e.g."
          " {'max_norm': 1.0} for clip_by_global_norm."
      ),
  )
  parser.add_argument(
      "--optimizer_b1",
      "--adam_b1",
      dest="optimizer_b1",
      type=float,
      default=0.9,
  )
  parser.add_argument(
      "--optimizer_b2",
      "--adam_b2",
      dest="optimizer_b2",
      type=float,
      default=0.999,
  )
  parser.add_argument(
      "--optimizer_weight_decay",
      "--weight_decay",
      dest="optimizer_weight_decay",
      type=float,
      default=0.0,
  )
  parser.add_argument(
      "--optimizer_eps",
      "--adam_eps",
      dest="optimizer_eps",
      type=float,
      default=1e-8,
      help="AdamW epsilon.",
  )
  parser.add_argument(
      "--optimizer_schedule_type",
      "--schedule_type",
      dest="optimizer_schedule_type",
      type=str,
      default="",
      help=(
          "Name of an `optax.schedules` function used to build the actor LR"
          " (e.g. constant_schedule, warmup_cosine_decay_schedule). Empty"
          " keeps the constant --optimizer_learning_rate. Only applies to"
          " --trainer_backend=tunix; the maxtext backend builds its own"
          " optimizer."
      ),
  )
  parser.add_argument(
      "--optimizer_value",
      dest="optimizer_value",
      type=float,
      default=None,
      help="Learning rate of constant_schedule.",
  )
  parser.add_argument(
      "--optimizer_init_value",
      dest="optimizer_init_value",
      type=float,
      default=None,
      help="Learning rate the warmup starts from.",
  )
  parser.add_argument(
      "--optimizer_peak_value",
      dest="optimizer_peak_value",
      type=float,
      default=None,
      help="Learning rate the warmup ends at, where the decay starts.",
  )
  parser.add_argument(
      "--optimizer_end_value",
      dest="optimizer_end_value",
      type=float,
      default=None,
      help="Learning rate the decay ends at.",
  )
  parser.add_argument(
      "--optimizer_warmup_steps",
      "--warmup_steps",
      dest="optimizer_warmup_steps",
      type=int,
      default=None,
      help="Steps to warm up from init_value to peak_value over.",
  )
  parser.add_argument(
      "--optimizer_decay_steps",
      "--lr_decay_steps",
      dest="optimizer_decay_steps",
      type=int,
      default=None,
      help=(
          "Total steps of the schedule, warmup included, after which the"
          " learning rate stays at end_value."
      ),
  )
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=64)
  parser.add_argument("--lora_alpha", type=float, default=64.0)
  parser.add_argument("--checkpoint_save_interval_steps", type=int, default=1)
  parser.add_argument("--checkpoint_max_to_keep", type=int, default=10)
  parser.add_argument(
      "--checkpoint_root_directory",
      type=str,
      default=os.getenv(
          "CHECKPOINT_ROOT_DIRECTORY",
          os.path.join(REPO_ROOT, "checkpoints"),
      ),
  )
  parser.add_argument(
      "--sampler_type",
      type=str,
      choices=("inprocess_vllm", "vllm", "vanilla"),
      default="inprocess_vllm",
      help="Sampler type for the trainer to use.",
  )
  parser.add_argument(
      "--trainer_backend",
      choices=("tunix", "maxtext"),
      default="tunix",
      help="tunix runs Tunix's PeftTrainer; maxtext runs MaxTextTrainingEngine",
  )
  parser.add_argument("--maxtext_model_name", type=str, default="qwen3-0.6b")
  parser.add_argument(
      "--maxtext_padded_moe_mlp_dim",
      type=int,
      default=0,
      help=(
          "Explicit padded_base_moe_mlp_dim override to match rollout TP"
          " tile-alignment padding for MoE models."
      ),
  )
  parser.add_argument(
      "--maxtext_ckpt_path",
      type=str,
      default=os.getenv("MAXTEXT_CKPT", ""),
      help=(
          "Orbax params-only checkpoint for the MaxText trainer, e.g. gs://..."
      ),
  )
  parser.add_argument(
      "--maxtext_output_directory",
      type=str,
      default=os.getenv(
          "MAXTEXT_OUTPUT_DIR",
          os.path.join(REPO_ROOT, "artifacts", "math_gsm8k_dist", "maxtext"),
      ),
      help="Base directory for MaxText trainer outputs.",
  )
  parser.add_argument(
      "--maxtext_warmup_steps_fraction",
      type=float,
      default=0.0,
      help=(
          "Warmup fraction for MaxText LR schedule (0.0 enables updates from"
          " step 0)."
      ),
  )
  parser.add_argument(
      "--rollout_mesh_tp",
      type=int,
      default=0,
      help="Rollout TP degree to align MaxText MoE MLP dimensions with.",
  )
  parser.add_argument(
      "--prefuse_moe_weights",
      type=_str2bool,
      default=False,
      nargs="?",
      const=True,
      help=(
          "Prefuse MoE weights (w0/w1). Off for the trainer."
      ),
  )
  parser.add_argument(
      "--use_weight_converter",
      type=_str2bool,
      default=True,
      nargs="?",
      const=True,
      help="Use weight converter for MaxText weight synchronization.",
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging for the trainer worker.",
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
        "--model_dir is required for JAX trainer weights. Set MODEL_DIR or pass"
        " --model_dir=/path/to/local/qwen3/safetensors."
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


def _create_mesh(args) -> Mesh:
  shape = (args.mesh_fsdp, args.mesh_tp)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "Trainer mesh dimensions must multiply to visible JAX device count. "
        f"Got shape={shape}, devices={jax.device_count()}."
    )
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))


def _load_actor_model(args, mesh: Mesh, *, lora: bool):
  if not args.model_dir:
    raise ValueError(
        "--model_dir is required for JAX trainer weights. Set MODEL_DIR or pass"
        " --model_dir=/path/to/local/safetensors."
    )
  model = models.create_model(args.model_name, args.model_dir, mesh)
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
  return model_utils.apply_lora_to_model(
      model, mesh=mesh, lora_config=lora_config
  )


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

  def save_checkpoint(self, metadata: Any = None, **kwargs) -> None:
    with self._mesh:
      self._trainer.save_checkpoint(metadata, **kwargs)

  def restore_checkpoint(self, **kwargs) -> Any:
    with self._mesh:
      return self._trainer.restore_checkpoint(**kwargs)

  def close(self) -> None:
    with self._mesh:
      self._trainer.close()


def _checkpointing_options(args) -> Any:
  """Builds the Orbax options; `save_interval_steps=0` means "never save".

  Orbax's FixedIntervalPolicy takes `step % save_interval_steps`, so 0 has to
  become `read_only` rather than being passed through. Restoring is a separate
  path and keeps working, which is what makes 0 usable for short smoke tests.
  """
  if args.checkpoint_save_interval_steps < 0:
    raise ValueError(
        "checkpoint_save_interval_steps must be non-negative, got"
        f" {args.checkpoint_save_interval_steps}."
    )
  if args.checkpoint_save_interval_steps == 0:
    logging.info(
        "checkpoint_save_interval_steps=0; checkpoint saving is disabled "
        "(restore is unaffected)."
    )
    return ocp.CheckpointManagerOptions(read_only=True)
  return ocp.CheckpointManagerOptions(
      save_interval_steps=args.checkpoint_save_interval_steps,
      max_to_keep=args.checkpoint_max_to_keep,
  )


def _create_maxtext_trainer_factory(args) -> Any:
  """Creates the trainer factory function for MaxText's MaxTextTrainingEngine."""
  logging.info("Trainer backend: MaxText's MaxTextTrainingEngine.")
  pad_id = maxtext_utils.get_tokenizer_pad_id(
      args.model_id, args.tokenizer_path, args.model_dir
  )
  checkpointing_options = _checkpointing_options(args)
  grad_accumulation_steps = max(
      1, math.ceil(args.mini_batch_size / args.train_micro_batch_size)
  )
  if args.optimizer_schedule_type:
    logging.warning(
        "--optimizer_schedule_type=%s is ignored by the maxtext backend, which"
        " builds its own optimizer from --optimizer_learning_rate and"
        " --maxtext_warmup_steps_fraction.",
        args.optimizer_schedule_type,
    )
  extra_cfg_kwargs = {}
  import inspect  # pylint: disable=g-import-not-at-top
  sig = inspect.signature(maxtext_utils.build_maxtext_config)
  for k, v in [
      ("rollout_mesh_tp", args.rollout_mesh_tp),
      ("prefuse_moe_weights", args.prefuse_moe_weights),
      ("use_weight_converter", args.use_weight_converter),
  ]:
    if k in sig.parameters:
      extra_cfg_kwargs[k] = v

  maxtext_config = maxtext_utils.build_maxtext_config(
      model_name=args.maxtext_model_name,
      worker_id=args.worker_id,
      train_micro_batch_size=args.train_micro_batch_size,
      mesh_fsdp=args.mesh_fsdp,
      mesh_tp=args.mesh_tp,
      mesh_expert=args.mesh_expert,
      num_devices=jax.device_count(),
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      learning_rate=args.optimizer_learning_rate,
      warmup_steps_fraction=args.maxtext_warmup_steps_fraction,
      load_parameters_path=args.maxtext_ckpt_path,
      padded_moe_mlp_dim=args.maxtext_padded_moe_mlp_dim,
      base_output_directory=args.maxtext_output_directory,
      gradient_accumulation_steps=grad_accumulation_steps,
      checkpointing_options=checkpointing_options,
      **extra_cfg_kwargs,
  )
  logging.info("Creating MaxText device mesh...")
  mesh = maxtext_utils.create_maxtext_mesh(maxtext_config)
  logging.info("Trainer mesh: %s", mesh)

  def _factory():
    engine = maxtext_utils.create_maxtext_engine(
        maxtext_config,
        mesh=mesh,
        tokenizer_pad_id=pad_id,
        wrap_with_tunix_adapter=True,
    )
    return _MeshBoundTrainer(engine, mesh)

  return _factory


def _gradient_accumulation_steps(args: argparse.Namespace) -> int:
  if args.mini_batch_size <= 0:
    raise ValueError("--mini_batch_size must be positive.")
  if args.num_generations <= 0:
    raise ValueError("--num_generations must be positive.")
  if args.train_micro_batch_size <= 0:
    raise ValueError("--train_micro_batch_size must be positive.")
  update_trajectories = args.mini_batch_size * args.num_generations
  if update_trajectories % args.train_micro_batch_size != 0:
    raise ValueError(
        "--mini_batch_size * --num_generations must be divisible by "
        "--train_micro_batch_size; got "
        f"mini_batch_size={args.mini_batch_size}, "
        f"num_generations={args.num_generations}, "
        f"train_micro_batch_size={args.train_micro_batch_size}."
    )
  return update_trajectories // args.train_micro_batch_size


def _create_tunix_trainer_factory(args) -> Any:
  """Creates the trainer factory function for Tunix's PeftTrainer."""
  logging.info("Trainer backend: Tunix's PeftTrainer.")
  grad_accumulation_steps = _gradient_accumulation_steps(args)
  update_trajectories = args.mini_batch_size * args.num_generations

  args.model_dir = _ensure_model_dir_for_trainer(args.model_dir, args.model_id)
  logging.info("Prepared trainer safetensors directory: %s", args.model_dir)

  logging.info("Creating trainer mesh...")
  mesh = _create_mesh(args)
  logging.info("Trainer mesh: %s", mesh)

  logging.info("Loading actor model with use_lora=%s...", args.use_lora)
  actor_model = _load_actor_model(args, mesh, lora=args.use_lora)

  logging.info("Building PeftTrainer v2 config...")
  checkpointing_options = _checkpointing_options(args)
  training_config = peft_trainer_v2.TrainingConfig(
      eval_every_n_steps=args.eval_every_n_steps,
      gradient_accumulation_steps=grad_accumulation_steps,
      metrics_prefix="actor",
      pbar_description="Actor Training",
      data_sharding_axis=("fsdp",),
      checkpointing_options=checkpointing_options,
      checkpoint_root_directory=args.checkpoint_root_directory,
      # The orchestrator owns resume: it calls restore_checkpoint() explicitly.
      # Orchestrator needs to realign its step/policy_version from the returned
      # metadata.
      resume_from_checkpoint_on_init=False,
  )
  logging.info(
      "PeftTrainer v2 gradient_accumulation_steps=%d "
      "(mini_batch_size=%d prompt groups, num_generations=%d, "
      "update_trajectories=%d, train_micro_batch_size=%d).",
      grad_accumulation_steps,
      args.mini_batch_size,
      args.num_generations,
      update_trajectories,
      args.train_micro_batch_size,
  )

  def _factory():
    with mesh:
      trainer = peft_trainer_v2.PeftTrainer(
          actor_model,
          _build_optimizer(args),
          training_config,
          sampler_type=args.sampler_type,
      )
    return _MeshBoundTrainer(trainer, mesh)

  return _factory


def _create_trainer_factory(args) -> Any:
  """Creates the trainer factory function based on args.trainer_backend."""
  if args.trainer_backend == "maxtext":
    return _create_maxtext_trainer_factory(args)
  return _create_tunix_trainer_factory(args)


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
  if args.mini_batch_size <= 0:
    raise ValueError("--mini_batch_size must be positive.")
  if args.num_generations <= 0:
    raise ValueError("--num_generations must be positive.")
  if args.optimizer_chain_kwargs and not args.optimizer_opt_chain_type:
    raise ValueError(
        "--optimizer_chain_kwargs is only used by"
        " --optimizer_opt_chain_type, which is not set."
    )

  logging.info("Creating generic TrainerWorker and gRPC server...")
  trainer_factory = _create_trainer_factory(args)
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
    # Shut down gracefully on SIGTERM/SIGINT so that TrainerWorker.stop() ->
    # PeftTrainerV2.close() runs and blocks until every in-flight async ops is
    # finished.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.add_signal_handler(sig, stop_event.set)
      except NotImplementedError:
        pass

    try:
      await stop_event.wait()
    except asyncio.CancelledError:
      pass
    finally:
      logging.info("Draining trainer worker...")
      try:
        worker_service.stop()
        logging.info("Trainer worker drained.")
      except Exception:
        logging.exception("Failed to drain trainer worker cleanly.")
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])
