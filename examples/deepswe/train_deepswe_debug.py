"""Debug DeepSWE GRPO recipe for Qwen3 models.

This recipe follows the same high-level layout as:

1. logging / runtime setup
2. argparse + recipe defaults
3. Kubernetes / R2E-Gym setup
4. mesh construction
5. dataset loading
6. tokenizer / model loading
7. checkpoint + metrics + optimizer
8. rollout + RL cluster
9. GRPO trainer
10. training

It keeps the current DeepSWE environment, dataset, and split-mesh topology from
``train_deepswe_nb.py``, but moves them into a cleaner debug entrypoint.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import logging
import math
import os
import signal
import sys
from typing import Any

from absl import logging as absl_logging

if "--pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
  sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "100000000000")

faulthandler.register(signal.SIGINT, all_threads=True)

# ====== Logging Configuration ======
absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")
print("Logging configured at INFO level.")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKDIR = os.getcwd()
if os.path.exists(os.path.join(WORKDIR, "tunix")):
  WORKSPACE_ROOT = WORKDIR
else:
  WORKSPACE_ROOT = REPO_ROOT

for root in [
    os.path.dirname(__file__),
    REPO_ROOT,
    WORKSPACE_ROOT,
    os.path.join(WORKSPACE_ROOT, "tunix"),
    os.path.join(WORKSPACE_ROOT, "pathways-utils"),
    os.path.join(WORKSPACE_ROOT, "r2egym"),
]:
  if root not in sys.path:
    sys.path.insert(0, root)

_DISTRIBUTED_INITIALIZED = False
try:
  import tunix  # pytype: disable=import-error  # noqa: F401
except Exception:
  pass

try:
  import r2egym  # pytype: disable=import-error  # noqa: F401
except Exception:
  pass

try:
  import pathwaysutils  # pytype: disable=import-error

  pathwaysutils.initialize()
  _DISTRIBUTED_INITIALIZED = True
except Exception:
  pass

import grain
from flax import nnx
from huggingface_hub import snapshot_download
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
import optax
from orbax import checkpoint as ocp
from transformers import AutoTokenizer

if not _DISTRIBUTED_INITIALIZED:
  try:
    jax.distributed.initialize()
  except Exception as exc:
    print(f"jax.distributed.initialize() skipped: {exc}")

print("jax devices: ", jax.devices())

from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_utils
from tunix.models.automodel import call_model_config
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils

import deepswe_data
from r2egym_runtime_patch import apply_repoenv_kubernetes_watch_patch
from swe_agent import SWE_SYSTEM_PROMPT
from swe_agent import SWEAgent
from swe_env import SWEEnv

VALID_STATUS_NAMES = [status.name for status in agent_types.TrajectoryStatus]


def _str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  normalized = value.strip().lower()
  if normalized in {"1", "true", "t", "yes", "y", "on"}:
    return True
  if normalized in {"0", "false", "f", "no", "n", "off"}:
    return False
  raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def create_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Train DeepSWE with a debug GRPO entrypoint."
  )

  parser.add_argument("--model_version", type=str, default="Qwen3-32B")
  parser.add_argument("--models_base_dir", type=str, default="models")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--node_selector_val", type=str, default="deepswe-cpu-pool")
  parser.add_argument("--image_pull_secret", type=str, default="dockerhub-pro")
  parser.add_argument("--dataset_name", type=str, default="R2E-Gym/R2E-Gym-Subset")
  parser.add_argument("--dataset_split", type=str, default="train")
  parser.add_argument("--dataset_cache_dir", type=str, default="dataset_cache")

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--mini_batch_size", type=int, default=8)
  parser.add_argument("--train_fraction", type=float, default=1.0)
  parser.add_argument("--max_steps", type=int, default=None)
  parser.add_argument("--eval_every_n_steps", type=int, default=10)
  parser.add_argument("--num_epochs", type=int, default=1000)

  parser.add_argument("--train_with_lora", type=_str_to_bool, default=False)
  parser.add_argument("--rank", type=int, default=64)
  parser.add_argument(
      "--lora_rank",
      dest="rank",
      type=int,
      default=argparse.SUPPRESS,
      help=argparse.SUPPRESS,
  )
  parser.add_argument("--alpha", type=float, default=64.0)
  parser.add_argument(
      "--lora_alpha",
      dest="alpha",
      type=float,
      default=argparse.SUPPRESS,
      help=argparse.SUPPRESS,
  )

  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--num_iterations", type=int, default=1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument("--epsilon_high", type=float, default=0.28)
  parser.add_argument(
      "--loss_algo",
      type=str,
      default="grpo",
      choices=["grpo", "gspo-token"],
      help="'grpo' (per-token PPO) or 'gspo-token' (sequence-mean IS).",
  )
  parser.add_argument("--off_policy_steps", type=int, default=0)

  parser.add_argument("--max_prompt_length", type=int, default=4096)
  parser.add_argument("--max_response_length", type=int, default=32768)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument(
      "--rollout_engine",
      type=str,
      default="vllm",
      choices=["vllm", "vanilla", "sglang_jax"],
  )
  parser.add_argument(
      "--vllm_utilization",
      type=float,
      default=0.6,
  )
  parser.add_argument(
      "--rollout_vllm_hbm_utilization",
      dest="vllm_utilization",
      type=float,
      default=argparse.SUPPRESS,
      help=argparse.SUPPRESS,
  )
  parser.add_argument("--rollout_vllm_max_num_seqs", type=int, default=None)
  parser.add_argument(
      "--rollout_vllm_max_num_batched_tokens", type=int, default=None
  )

  parser.add_argument("--learning_rate", type=float, default=1e-6)
  parser.add_argument("--b1", type=float, default=0.9)
  parser.add_argument("--b2", type=float, default=0.999)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--optimizer_offload", type=_str_to_bool, default=False)

  parser.add_argument("--ckpt_dir", type=str, default="/tmp/cp/deepswe_ckpt/01")
  parser.add_argument("--max_to_keep", type=int, default=4)
  parser.add_argument("--save_interval_steps", type=int, default=10)
  parser.add_argument("--metrics_logger_dir", type=str, default=None)

  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--rollout_micro_batch_size", type=int, default=1)
  parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

  parser.add_argument("--max_turns", type=int, default=50)
  parser.add_argument("--episode_timeout_secs", type=int, default=5400)
  parser.add_argument("--per_turn_timeout_secs", type=int, default=None)
  parser.add_argument("--max_concurrency", type=int, default=64)
  parser.add_argument(
      "--scaffold",
      type=str,
      default="r2egym",
      choices=["r2egym", "sweagent"],
  )

  parser.add_argument("--rollout_mesh_fsdp", type=int, default=None)
  parser.add_argument("--rollout_mesh_tp", type=int, default=None)
  parser.add_argument("--train_mesh_fsdp", type=int, default=None)
  parser.add_argument("--train_mesh_sp", type=int, default=None)
  parser.add_argument("--train_mesh_tp", type=int, default=None)
  parser.add_argument("--rollout_split_fraction", type=float, default=0.5)

  parser.add_argument(
      "--filter_statuses",
      type=str,
      nargs="+",
      default=None,
      choices=VALID_STATUS_NAMES,
  )
  parser.add_argument(
      "--loss_agg_mode", type=str, default="seq-mean-token-sum"
  )
  parser.add_argument(
      "--kl_loss_mode",
      type=str,
      default="low_var_kl",
  )
  parser.add_argument("--advantage_estimator", type=str, default="rloo")
  parser.add_argument("--use_rollout_logps", type=_str_to_bool, default=False)
  parser.add_argument(
      "--sampler_is",
      type=str,
      default="none",
      choices=["none", "token"],
  )
  parser.add_argument("--sampler_is_threshold", type=float, default=2.0)
  parser.add_argument(
      "--degenerate_group_masking", type=_str_to_bool, default=False
  )

  parser.add_argument("--enable_remat", type=_str_to_bool, default=True)
  parser.add_argument(
      "--remat_policy",
      type=str,
      default="decoder",
      choices=["block", "decoder"],
  )
  parser.add_argument("--use_flash_attention", type=_str_to_bool, default=True)
  parser.add_argument("--flash_attention_block_size", type=int, default=1024)
  parser.add_argument("--do_mem_profiling", type=_str_to_bool, default=False)
  parser.add_argument(
      "--logging_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
  )
  parser.add_argument(
      "--dtype",
      type=str,
      default="bfloat16",
      choices=["bfloat16", "float16", "float32"],
  )
  parser.add_argument(
      "--param_dtype",
      type=str,
      default="float32",
      choices=["bfloat16", "float16", "float32"],
  )
  return parser


def configure_kubernetes(args: argparse.Namespace) -> None:
  os.environ.setdefault("KUBECONFIG", os.path.expanduser("~/.kube/config"))
  os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
  os.environ["NODE_SELECTOR_VAL"] = args.node_selector_val
  if args.image_pull_secret:
    os.environ["IMAGE_PULL_SECRET"] = args.image_pull_secret

  logging.info(
      "Using Kubernetes node selector: %s=%s imagePullSecret=%s",
      os.environ["NODE_SELECTOR_KEY"],
      os.environ["NODE_SELECTOR_VAL"],
      os.environ.get("IMAGE_PULL_SECRET"),
  )

  try:
    k8s_config.load_kube_config()
    client.CoreV1Api().list_namespace(timeout_seconds=5)
    logging.info("Kubernetes connection verified.")
  except Exception as exc:
    logging.warning("Kubernetes config loading failed: %r", exc)

  apply_repoenv_kubernetes_watch_patch()


def _resolve_model_id(model_version: str) -> str:
  return model_version if "/" in model_version else f"Qwen/{model_version}"


def _resolve_model_dir(args: argparse.Namespace) -> str:
  model_name = args.model_version.split("/")[-1]
  base_dir = args.models_base_dir
  if not os.path.isabs(base_dir):
    base_dir = os.path.join(WORKSPACE_ROOT, base_dir)
  return os.path.join(base_dir, model_name)


def ensure_model_downloaded(model_id: str, model_path: str) -> None:
  if os.path.isdir(model_path) and any(
      filename.endswith(".safetensors") for filename in os.listdir(model_path)
  ):
    return
  os.makedirs(model_path, exist_ok=True)
  snapshot_download(
      repo_id=model_id,
      local_dir=model_path,
      local_dir_use_symlinks=False,
  )


def configure_model(
    args: argparse.Namespace,
    *,
    enable_sp: bool,
) -> tuple[Any, Any, Any]:
  dtype_map = {
      "bfloat16": jnp.bfloat16,
      "float16": jnp.float16,
      "float32": jnp.float32,
  }
  model_dtype = dtype_map[args.dtype]
  param_dtype = dtype_map[args.param_dtype]

  config = call_model_config(args.model_version.split("/")[-1])
  if args.enable_remat:
    remat_policy_map = {
        "block": qwen3_model_lib.RematConfig.BLOCK,
        "decoder": qwen3_model_lib.RematConfig.DECODER,
    }
    config.remat_config = remat_policy_map[args.remat_policy]
  else:
    config.remat_config = qwen3_model_lib.RematConfig.NONE

  config.dtype = model_dtype
  config.param_dtype = param_dtype
  if args.use_flash_attention:
    config.use_flash_attention = True
    config.flash_attention_block_size = args.flash_attention_block_size
  if enable_sp:
    config.shd_config = qwen3_model_lib.ShardingConfig.get_default_sharding(
        enable_sp=True
    )
  return config, model_dtype, param_dtype


def create_meshes(
    args: argparse.Namespace,
    *,
    num_kv_heads: int,
) -> tuple[Mesh, Mesh, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
  devices = jax.devices()
  total_devices = len(devices)

  rollout_fsdp = args.rollout_mesh_fsdp
  rollout_tp = args.rollout_mesh_tp
  if rollout_fsdp is not None or rollout_tp is not None:
    rollout_dims = []
    if rollout_fsdp is not None:
      rollout_dims.append(("fsdp", rollout_fsdp))
    if rollout_tp is not None:
      rollout_dims.append(("tp", rollout_tp))
  else:
    num_rollout_devices = int(total_devices * args.rollout_split_fraction)
    if num_rollout_devices <= 0 or num_rollout_devices >= total_devices:
      raise ValueError(
          "rollout_split_fraction must leave at least one device for both"
          f" rollout and train. Got fraction={args.rollout_split_fraction},"
          f" total_devices={total_devices}."
      )
    rollout_tp = int(np.gcd(num_rollout_devices, num_kv_heads))
    rollout_tp = max(rollout_tp, 1)
    rollout_fsdp = num_rollout_devices // rollout_tp
    rollout_dims = [("fsdp", rollout_fsdp), ("tp", rollout_tp)]
  num_rollout_devices = int(np.prod([size for _, size in rollout_dims]))

  train_fsdp = args.train_mesh_fsdp
  train_sp = args.train_mesh_sp
  train_tp = args.train_mesh_tp
  if any(value is not None for value in (train_fsdp, train_sp, train_tp)):
    train_dims = [("fsdp", train_fsdp if train_fsdp is not None else 1)]
    if train_sp is not None:
      train_dims.append(("sp", train_sp))
    train_dims.append(("tp", train_tp if train_tp is not None else 1))
  else:
    num_train_devices = total_devices - num_rollout_devices
    if num_train_devices <= 0:
      raise ValueError(
          f"No devices left for training. total_devices={total_devices},"
          f" num_rollout_devices={num_rollout_devices}."
      )
    train_fsdp = int(
        np.gcd(num_train_devices, args.train_micro_batch_size * args.num_generations)
    )
    train_fsdp = max(train_fsdp, 1)
    train_tp = num_train_devices // train_fsdp
    train_dims = [("fsdp", train_fsdp), ("tp", train_tp)]
  num_train_devices = int(np.prod([size for _, size in train_dims]))

  if num_rollout_devices + num_train_devices > total_devices:
    raise ValueError(
        f"Requested {num_rollout_devices} rollout devices +"
        f" {num_train_devices} train devices, but only {total_devices}"
        " devices are available."
    )

  rollout_shape = tuple(size for _, size in rollout_dims)
  rollout_axis_names = tuple(name for name, _ in rollout_dims)
  train_shape = tuple(size for _, size in train_dims)
  train_axis_names = tuple(name for name, _ in train_dims)

  rollout_devices = np.array(devices[:num_rollout_devices]).reshape(rollout_shape)
  train_devices = np.array(
      devices[num_rollout_devices : num_rollout_devices + num_train_devices]
  ).reshape(train_shape)

  rollout_mesh = Mesh(rollout_devices, axis_names=rollout_axis_names)
  train_mesh = Mesh(train_devices, axis_names=train_axis_names)
  logging.info("Rollout mesh dims=%s shape=%s", rollout_dims, rollout_mesh.shape)
  logging.info("Train mesh dims=%s shape=%s", train_dims, train_mesh.shape)
  return rollout_mesh, train_mesh, tuple(rollout_dims), tuple(train_dims)


def create_train_dataset(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> grain.MapDataset:
  raw_dataset = deepswe_data.create_dataset(
      dataset_name=args.dataset_name,
      dataset_split=args.dataset_split,
      cache_dir=args.dataset_cache_dir,
      shuffle=True,
      seed=args.seed,
  )
  dataset, _ = data_lib.post_init_dataset(
      raw_dataset,
      tokenizer,
      batch_size=args.batch_size,
      num_batches=None,
      max_prompt_length=args.max_prompt_length,
      fraction=args.train_fraction,
      num_epochs=args.num_epochs,
      prompt_key="problem_statement",
      custom_batch_fn=deepswe_data.batch_fn,
  )
  return dataset


def maybe_apply_lora(
    model: nnx.Module,
    mesh: Mesh,
    *,
    enabled: bool,
    rank: int,
    alpha: float,
) -> nnx.Module:
  if not enabled:
    return model
  lora_config = {
      "module_path": (
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      "rank": rank,
      "alpha": alpha,
  }
  return model_utils.apply_lora_to_model(model, mesh=mesh, lora_config=lora_config)


def put_model_on_device(model: nnx.Module) -> nnx.Module:
  graph_def, state = nnx.split(model)
  state = rl_utils.put_params_on_memory_kind(state, "device")
  return nnx.merge(graph_def, state)


def create_reference_and_actor(
    *,
    args: argparse.Namespace,
    model_path: str,
    config: Any,
    train_mesh: Mesh,
    model_dtype: Any,
    param_dtype: Any,
) -> tuple[nnx.Module, nnx.Module]:
  reference = qwen3_params_lib.create_model_from_safe_tensors(
      model_path, config, train_mesh, dtype=model_dtype
  )
  actor_base = qwen3_params_lib.create_model_from_safe_tensors(
      model_path, config, train_mesh, dtype=param_dtype
  )
  reference = put_model_on_device(reference)
  actor = maybe_apply_lora(
      actor_base,
      train_mesh,
      enabled=args.train_with_lora,
      rank=args.rank,
      alpha=args.alpha,
  )
  actor = put_model_on_device(actor)
  return reference, actor


def resolve_episode_timeout(args: argparse.Namespace) -> int:
  if args.per_turn_timeout_secs is not None:
    return args.per_turn_timeout_secs * args.max_turns
  return args.episode_timeout_secs


def create_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
  optimizer = optax.adamw(
      learning_rate=args.learning_rate,
      b1=args.b1,
      b2=args.b2,
      weight_decay=args.weight_decay,
  )
  if args.max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.max_grad_norm),
        optimizer,
    )
  return optimizer


def create_checkpointing_options(
    args: argparse.Namespace,
) -> ocp.CheckpointManagerOptions | None:
  if not args.ckpt_dir:
    return None
  os.makedirs(args.ckpt_dir, exist_ok=True)
  return ocp.CheckpointManagerOptions(
      save_interval_steps=args.save_interval_steps,
      max_to_keep=args.max_to_keep,
  )


def create_metrics_logging_options(
    args: argparse.Namespace,
) -> metrics_logger.MetricsLoggerOptions | None:
  if not args.metrics_logger_dir:
    return None
  os.makedirs(args.metrics_logger_dir, exist_ok=True)
  return metrics_logger.MetricsLoggerOptions(
      log_dir=args.metrics_logger_dir,
      project_name="tunix-deepswe",
      flush_every_n_steps=2,
  )


def resolve_kv_cache_size(args: argparse.Namespace) -> int:
  return args.max_prompt_length + args.max_response_length + 128


def resolve_vllm_rollout_limits(args: argparse.Namespace) -> tuple[int, int]:
  vllm_max_num_seqs = (
      args.rollout_vllm_max_num_seqs
      if args.rollout_vllm_max_num_seqs is not None
      else 1024
  )
  vllm_max_num_batched_tokens = (
      args.rollout_vllm_max_num_batched_tokens
      if args.rollout_vllm_max_num_batched_tokens is not None
      else 8192
  )
  return vllm_max_num_seqs, vllm_max_num_batched_tokens


def initialize_wandb(
    args: argparse.Namespace,
    *,
    num_devices: int,
    rollout_dims: tuple[tuple[str, int], ...],
    train_dims: tuple[tuple[str, int], ...],
) -> None:
  try:
    import datetime
    import wandb  # pytype: disable=import-error

    settings = wandb.Settings(console="off")
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rollout_dim_map = dict(rollout_dims)
    train_dim_map = dict(train_dims)
    vllm_max_num_seqs, vllm_max_batched_tokens = resolve_vllm_rollout_limits(
        args
    )
    wandb_config = {
        **vars(args),
        "kv_cache_size": resolve_kv_cache_size(args),
        "vllm_max_num_seqs": vllm_max_num_seqs,
        "vllm_max_batched_tokens": vllm_max_batched_tokens,
        "filter_statuses": args.filter_statuses,
        "degenerate_group_masking": args.degenerate_group_masking,
        "num_devices": num_devices,
        "rollout_mesh_fsdp": rollout_dim_map.get("fsdp"),
        "rollout_mesh_tp": rollout_dim_map.get("tp"),
        "train_mesh_fsdp": train_dim_map.get("fsdp"),
        "train_mesh_sp": train_dim_map.get("sp"),
        "train_mesh_tp": train_dim_map.get("tp"),
    }
    wandb.init(
        project="tunix-deepswe",
        name=run_name,
        config=wandb_config,
        settings=settings,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"W&B initialization failed with error: {exc}")


def create_rollout_config(
    *,
    args: argparse.Namespace,
    model_path: str,
    tokenizer: AutoTokenizer,
    rollout_mesh: Mesh,
) -> base_rollout.RolloutConfig:
  kv_cache_size = resolve_kv_cache_size(args)
  vllm_max_num_seqs, vllm_max_num_batched_tokens = (
      resolve_vllm_rollout_limits(args)
  )
  eos_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

  base_rollout_dict = {
      "max_prompt_length": args.max_prompt_length,
      "kv_cache_size": kv_cache_size,
      "temperature": args.temperature,
      "top_p": args.top_p,
      "top_k": args.top_k,
      "eos_tokens": eos_tokens,
      "return_logprobs": args.use_rollout_logps,
      "max_tokens_to_generate": args.max_response_length,
  }

  if args.rollout_engine == "sglang_jax":
    rollout_kwargs = {
        "rollout_sglang_jax_model_version": model_path,
        "rollout_sglang_jax_mem_fraction_static": 0.9,
        "rollout_sglang_jax_init_with_random_weights": True,
        "rollout_sglang_jax_disable_radix_cache": False,
        "rollout_sglang_jax_enable_deterministic_sampling": False,
        "rollout_sglang_jax_chunked_prefill_size": 2048,
        "rollout_sglang_jax_max_running_requests": args.max_concurrency,
        "rollout_sglang_jax_page_size": 128,
    }
    return base_rollout.RolloutConfig(**base_rollout_dict, **rollout_kwargs)

  if args.rollout_engine == "vllm":
    rollout_kwargs = {
        "rollout_vllm_model_version": model_path,
        "rollout_vllm_hbm_utilization": args.vllm_utilization,
        "rollout_vllm_tpu_backend_type": "jax",
        "rollout_vllm_server_mode": True,
        "rollout_vllm_async_scheduling": False,
        "rollout_vllm_init_with_random_weights": True,
        "tensor_parallel_size": rollout_mesh.shape.get("tp", 1),
        "data_parallel_size": rollout_mesh.shape.get("fsdp", 1),
        "rollout_vllm_max_num_seqs": vllm_max_num_seqs,
        "rollout_vllm_max_num_batched_tokens": vllm_max_num_batched_tokens,
        "rollout_vllm_kwargs": {
            "kv_cache_metrics": True,
            "disable_log_stats": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "dtype": "bfloat16",
        },
    }
    if args.train_with_lora:
      rollout_kwargs["rollout_vllm_lora_config"] = {
          "max_lora_rank": args.rank,
      }
    return base_rollout.RolloutConfig(**base_rollout_dict, **rollout_kwargs)

  return base_rollout.RolloutConfig(**base_rollout_dict)


def create_cluster_config(
    *,
    args: argparse.Namespace,
    rollout_mesh: Mesh,
    train_mesh: Mesh,
    optimizer: optax.GradientTransformation,
    metrics_logging_options: metrics_logger.MetricsLoggerOptions | None,
    checkpointing_options: ocp.CheckpointManagerOptions | None,
    rollout_config: base_rollout.RolloutConfig,
) -> rl_cluster_lib.ClusterConfig:
  return rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: train_mesh,
          rl_cluster_lib.Role.REFERENCE: train_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine=args.rollout_engine,
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=args.eval_every_n_steps,
          max_steps=args.max_steps,
          mini_batch_size=args.mini_batch_size,
          train_micro_batch_size=args.train_micro_batch_size,
          compute_logps_micro_batch_size=args.compute_logps_micro_batch_size,
          rollout_micro_batch_size=args.rollout_micro_batch_size,
          metrics_logging_options=metrics_logging_options,
          # Match train_deepswe_nb.py: DeepSWE disables RL trainer checkpoints.
          checkpoint_root_directory=None,
          checkpointing_options=None,
      ),
      rollout_config=rollout_config,
  )


def create_grpo_config(args: argparse.Namespace) -> agentic_grpo_learner.GRPOConfig:
  filter_statuses = (
      {agent_types.TrajectoryStatus[name] for name in args.filter_statuses}
      if args.filter_statuses is not None
      else None
  )
  return agentic_grpo_learner.GRPOConfig(
      num_generations=args.num_generations,
      num_iterations=args.num_iterations,
      max_response_length=args.max_response_length,
      beta=args.beta,
      epsilon=args.epsilon,
      epsilon_high=args.epsilon_high,
      system_prompt=SWE_SYSTEM_PROMPT,
      max_concurrency=args.max_concurrency,
      off_policy_steps=args.off_policy_steps,
      episode_timeout=resolve_episode_timeout(args),
      overlong_filter=True,
      filter_statuses=filter_statuses,
      loss_agg_mode=args.loss_agg_mode,
      kl_loss_mode=args.kl_loss_mode,
      loss_algo=args.loss_algo,
      sampler_is=None if args.sampler_is == "none" else args.sampler_is,
      sampler_is_threshold=args.sampler_is_threshold,
      advantage_estimator=args.advantage_estimator,
      use_rollout_logps=args.use_rollout_logps,
      degenerate_group_masking=args.degenerate_group_masking,
  )


def shutdown_rollout_runtime(rl_cluster: rl_cluster_lib.RLCluster) -> None:
  rollout = getattr(rl_cluster, "rollout", None)
  if rollout is not None:
    for method_name in ("close", "stop", "shutdown"):
      method = getattr(rollout, method_name, None)
      if callable(method):
        try:
          method()
        except Exception:
          logging.exception("Failed to %s rollout runtime during teardown.", method_name)
        break
  gc.collect()
  try:
    jax.clear_caches()
  except Exception:
    logging.exception("Failed to clear JAX caches during teardown.")


def main() -> None:
  args, _ = create_arg_parser().parse_known_args()
  log_level = getattr(logging, args.logging_level.upper())
  logging.getLogger().setLevel(log_level)
  logging.getLogger("absl").setLevel(log_level)
  absl_logging.set_verbosity(getattr(absl_logging, args.logging_level.upper()))
  absl_logging.set_stderrthreshold(args.logging_level.lower())
  if args.optimizer_offload:
    logging.warning(
        "optimizer_offload is not wired in this recipe yet and will be ignored."
    )
  configure_kubernetes(args)

  model_id = _resolve_model_id(args.model_version)
  model_path = _resolve_model_dir(args)
  ensure_model_downloaded(model_id, model_path)

  base_model_config = call_model_config(args.model_version.split("/")[-1])
  rollout_mesh, train_mesh, rollout_dims, train_dims = create_meshes(
      args, num_kv_heads=base_model_config.num_kv_heads
  )
  enable_sp = any(name == "sp" for name, _ in train_dims)
  model_config, model_dtype, param_dtype = configure_model(
      args, enable_sp=enable_sp
  )
  initialize_wandb(
      args,
      num_devices=len(jax.devices()),
      rollout_dims=rollout_dims,
      train_dims=train_dims,
  )

  tokenizer = AutoTokenizer.from_pretrained(
      model_path,
      local_files_only=True,
      trust_remote_code=True,
  )
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  train_dataset = create_train_dataset(args, tokenizer)
  sft_utils.show_hbm_usage("Done with loading datasets")

  reference, actor = create_reference_and_actor(
      args=args,
      model_path=model_path,
      config=model_config,
      train_mesh=train_mesh,
      model_dtype=model_dtype,
      param_dtype=param_dtype,
  )
  sft_utils.show_hbm_usage("after loading qwen_reference / qwen_actor")

  optimizer = create_optimizer(args)
  checkpointing_options = None
  metrics_logging_options = create_metrics_logging_options(args)
  rollout_config = create_rollout_config(
      args=args,
      model_path=model_path,
      tokenizer=tokenizer,
      rollout_mesh=rollout_mesh,
  )
  cluster_config = create_cluster_config(
      args=args,
      rollout_mesh=rollout_mesh,
      train_mesh=train_mesh,
      optimizer=optimizer,
      metrics_logging_options=metrics_logging_options,
      checkpointing_options=checkpointing_options,
      rollout_config=rollout_config,
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  sft_utils.show_hbm_usage("after RLCluster creation")

  grpo_trainer = agentic_grpo_learner.GRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=None,
      agent_class=SWEAgent,
      agent_kwargs={},
      env_class=SWEEnv,
      env_kwargs={
          "max_steps": args.max_turns,
          "scaffold": args.scaffold,
      },
      algo_config=create_grpo_config(args),
      chat_parser=chat_parser,
  )
  sft_utils.show_hbm_usage("after GRPOLearner creation")

  try:
    logging.info("Starting DeepSWE training...")
    grpo_trainer.train(train_dataset=train_dataset)
  finally:
    shutdown_rollout_runtime(rl_cluster)


if __name__ == "__main__":
  main()
