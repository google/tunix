# %%
# [WIP] Reproduction of [DeepSWE](https://www.together.ai/blog/deepswe)
# with Multi-turn Agentic framework.

# %%
import argparse
import faulthandler
from functools import partial
import logging
import os
import signal
import sys

from absl import logging as absl_logging
from flax import nnx
import grain
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kubernetes import client, config as k8s_config
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from transformers import AutoTokenizer
from tunix.cli.utils import data as data_lib
from examples.deepswe import deepswe_data
from examples.deepswe import wandb_step_backend
from tunix.rl.agentic.agents import agent_types
from tunix.utils import compat
import vllm  # pytype: disable=import-error

faulthandler.register(signal.SIGINT, all_threads=True)


def _parse_bool(value: str) -> bool:
  if value.lower() in ("1", "true", "yes", "on"):
    return True
  if value.lower() in ("0", "false", "no", "off"):
    return False
  raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(
    description="DeepSWE Training with Multi-turn Agentic Framework"
)

# General Config
parser.add_argument("--models_base_dir", type=str, default="models")
parser.add_argument(
    "--model_source",
    type=str,
    default="huggingface",
    choices=["huggingface", "maxtext"],
)
parser.add_argument(
    "--model_absolute_path",
    type=str,
    default=None,
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_version", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--node_selector_val", type=str, default="deepswe-cpu-pool")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="R2E-Gym/R2E-Gym-Subset")
parser.add_argument("--dataset_revision", type=str, default="2e8108ff942f24fcb5686badfaf7f9a8808566d5")
parser.add_argument("--dataset_split", type=str, default="train")
parser.add_argument("--gold_whitelist", type=str, default=None)
parser.add_argument(
    "--shuffle",
    type=_parse_bool,
    default=True,
    help=(
        "Shuffle the task list (seeded by --seed). Set false to keep an"
        " ordered curriculum such as the MLPerf easy-curriculum dataset."
    ),
)

parser.add_argument("--tpu_topology", type=str, default=None)


# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mini_batch_size", type=int, default=8)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--eval_every_n_steps", type=int, default=1000000)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--enable_remat", type=_parse_bool, default=True)
parser.add_argument(
    "--remat_policy",
    type=str,
    default="decoder",
    choices=["block", "decoder"],
    help=(
        "Remat policy when enable_remat is True: 'block' remats the attention"
        " block, 'decoder' remats the full decoder layer."
    ),
)

# LoRA
# LoRA Config
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--alpha", type=float, default=64.0)
parser.add_argument("--train_with_lora", type=_parse_bool, default=False)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=16)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)
parser.add_argument("--off_policy_steps", type=int, default=0)
parser.add_argument("--score_centering", type=_parse_bool, default=False)
parser.add_argument("--score_centering_top_k", type=int, default=32)
parser.add_argument("--score_centering_eps", type=float, default=1e-6)
parser.add_argument("--exact_token_continuity", type=_parse_bool, default=False)
parser.add_argument(
    "--sampler_is", choices=["none", "token", "seq-mask-tis"], default="none"
)
parser.add_argument("--sampler_is_threshold", type=float, default=2.0)
# seq-mask-tis keep-band on the per-sequence geometric-mean
# trainer/sampler ratio (MLPerf: 0.999 / 1.002).
parser.add_argument(
    "--truncated_importance_sampling_ratio_min", type=float, default=None
)
parser.add_argument(
    "--truncated_importance_sampling_ratio", type=float, default=None
)
# Masks sequences whose mean multiplicative trainer/sampler probability error
# exceeds this value (MLPerf: 2.0).
parser.add_argument("--seq_logprob_error_threshold", type=float, default=None)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=4096)
parser.add_argument("--max_response_length", type=int, default=16384)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--rollout_engine", type=str, default="vllm")
parser.add_argument("--vllm_utilization", type=float, default=0.6)
parser.add_argument(
    "--vllm_reshard_chunk_size",
    type=int,
    default=None,
    help="Number of flat keys to reshard at a time. None for single-call.",
)
parser.add_argument(
    "--max_num_batched_tokens",
    type=int,
    default=20480,
    help="Max number of tokens to be processed in parallel by vLLM.",
)

# Optimizer Config
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.99)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument(
    "--optimizer_offload",
    type=_parse_bool,
    default=False,
    help="Whether to offload optimizer states to CPU (pinned host memory).",
)  # not supported yet


# Checkpointing
parser.add_argument("--ckpt_dir", type=str, default="artifacts/deepswe_agentic_q4_clean/checkpoints")
parser.add_argument("--max_to_keep", type=int, default=2)
parser.add_argument("--save_interval_steps", type=int, default=1)

# Microbatch Sizes
parser.add_argument("--train_micro_batch_size", type=int, default=2)
parser.add_argument("--rollout_micro_batch_size", type=int, default=1)
parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

# DeepSWE Agentic Specifics
parser.add_argument("--max_turns", type=int, default=50)
parser.add_argument("--episode_timeout_secs", type=int, default=4800)
parser.add_argument("--step_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--reward_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--max_concurrency", type=int, default=32)
parser.add_argument("--env_backend", choices=["docker", "kubernetes"], default="docker")
parser.add_argument("--scaffold", type=str, default="r2egym")
parser.add_argument("--action_compat_mode", type=str, default="q4_r2egym_xml_v2")
parser.add_argument(
    "--use_agent_sandbox",
    action="store_true",
    help="Whether to use Kubernetes Agent Sandbox runtime instead of local Docker socket.",
)

parser.add_argument(
    "--overlong_filter",
    type=_parse_bool,
    default=False,
    help="Whether to filter out trajectories that exceed length limits",
)

# Mesh / Topology Config Override
parser.add_argument(
    "--rollout_mesh_fsdp",
    type=int,
    default=1,
    help="Optional override for rollout mesh FSDP dimension.",
)
parser.add_argument(
    "--rollout_mesh_tp",
    type=int,
    default=2,
    help="Optional override for rollout mesh TP dimension.",
)
parser.add_argument(
    "--train_mesh_fsdp",
    type=int,
    default=1,
    help="Optional override for train mesh FSDP dimension.",
)
parser.add_argument(
    "--train_mesh_tp",
    type=int,
    default=2,
    help="Optional override for train mesh TP dimension.",
)
parser.add_argument(
    "--train_mesh_sp",
    type=int,
    default=None,
    help="Optional override for train mesh SP dimension.",
)

parser.add_argument(
    "--rollout_split_fraction",
    type=float,
    default=0.5,
    help=(
        "Fraction of total devices to allocate to the rollout mesh. Default is"
        " 0.5 (1:1 ratio)."
    ),
)


VALID_STATUS_NAMES = [status.name for status in agent_types.TrajectoryStatus]

parser.add_argument(
    "--filter_statuses",
    type=str,
    nargs="+",
    default=None,  # Set default to None
    choices=VALID_STATUS_NAMES,
    help=(
        "List of trajectory statuses to filter out. Valid statuses:"
        f" {VALID_STATUS_NAMES}. Defaults to None."
    ),
)

parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-scale"
)
parser.add_argument("--advantage_estimator", type=str, default="rloo")
parser.add_argument(
    "--use_rollout_logps",
    type=_parse_bool,
    default=True,
    help=(
        "Whether to use rollout-cached logprobs as old policy logps. "
        "Default is False to recompute old logps on the actor side. "
    ),
)


# Other
parser.add_argument("--do_mem_profiling", type=_parse_bool, default=False)

parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float16", "float32"],  # Restrict to valid inputs
    help="Data type for the model activations(e.g., bfloat16, float32)",
)
parser.add_argument(
    "--param_dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float16", "float32"],  # Restrict to valid inputs
    help="Data type for the model weights (e.g., bfloat16, float32)",
)


parser.add_argument("--use_flash_attention", type=_parse_bool, default=True)
parser.add_argument("--flash_attention_block_size", type=int, default=None)
parser.add_argument("--target_accuracy", type=float, default=0.69)
parser.add_argument("--rcp_logging", action="store_true", default=False)
parser.add_argument(
    "--metric_logger_dir",
    type=str,
    default="artifacts/deepswe_agentic_q4_clean/events",
)
parser.add_argument(
    "--logging_level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level for the script and relevant libraries.",
)

args, _ = parser.parse_known_args()

# Register MaxText vLLM adapter if using a MaxText model
if args.model_source == "maxtext":
  try:
    from maxtext.integration.vllm import maxtext_vllm_adapter  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    maxtext_vllm_adapter.register()
    logging.info("Successfully registered MaxTextForCausalLM model with vLLM.")
  except ImportError as e:
    logging.warning("Could not import maxtext_vllm_adapter: %s", e)

MODEL_VERSION = args.model_version
NODE_SELECTOR_VAL = args.node_selector_val


# Monkeypatch r2egym DockerRuntime to dynamically configure Kubernetes nodeSelector.
# This is required because r2egym hardcodes the CPU nodepool name (using
# Karpenter bigcpu-standby), which does not exist in our GKE cluster. We
# override it here to match the nodepool configured via the
# --node_selector_val flag.
def patch_kubernetes_runtime():
  try:
    from r2egym.agenthub.runtime.docker import DockerRuntime
    import os

    original_start_kubernetes_pod = DockerRuntime._start_kubernetes_pod

    def patched_start_kubernetes_pod(
        self, docker_image, command, pod_name, **docker_kwargs
    ):
      original_create_namespaced_pod = self.client.create_namespaced_pod

      def patched_create_namespaced_pod(*args, **kwargs):
        body = kwargs.get("body")
        if body and "spec" in body:
          key = os.environ.get(
              "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
          )
          val = os.environ.get("NODE_SELECTOR_VAL", "cpu-np")
          body["spec"]["nodeSelector"] = {key: val}
          print(f"[Monkeypatch] Overrode nodeSelector to {key}={val}")
        return original_create_namespaced_pod(*args, **kwargs)

      self.client.create_namespaced_pod = patched_create_namespaced_pod
      try:
        return original_start_kubernetes_pod(
            self, docker_image, command, pod_name, **docker_kwargs
        )
      finally:
        self.client.create_namespaced_pod = original_create_namespaced_pod

    DockerRuntime._start_kubernetes_pod = patched_start_kubernetes_pod
    print(
        "[Monkeypatch] Successfully patched DockerRuntime._start_kubernetes_pod"
    )
  except Exception as e:
    print(f"[Monkeypatch] Failed to patch DockerRuntime: {e}")


patch_kubernetes_runtime()


# Orbax (0.12.x) calls storage.buckets.get to detect hierarchical-namespace
# (HNS) buckets when it lists checkpoint steps (i.e. on resume) and after it
# deletes old checkpoints (max_to_keep rotation). Service accounts with only
# object-level access (e.g. roles/storage.objectAdmin) get a 403 there, which
# makes every resume fail. Treat a failed probe as "not HNS": the flat-namespace
# code path only needs object list/get/delete.
def patch_orbax_gcs_hns_probe():
  try:
    from orbax.checkpoint._src.path import gcs_utils as ocp_gcs_utils

    original_is_hns = ocp_gcs_utils.is_hierarchical_namespace_enabled
    if getattr(original_is_hns, "_deepswe_patched", False):
      return
    failed_buckets = set()

    def safe_is_hns(path):
      bucket = None
      if str(path).startswith("gs://"):
        try:
          bucket = ocp_gcs_utils.parse_gcs_path(path)[0]
        except Exception:  # pylint: disable=broad-exception-caught
          bucket = None
      if bucket is not None and bucket in failed_buckets:
        return False
      try:
        return original_is_hns(path)
      except Exception as e:  # pylint: disable=broad-exception-caught
        if bucket is not None:
          failed_buckets.add(bucket)
        print(
            f"[Monkeypatch] orbax HNS probe failed for bucket {bucket!r}"
            f" ({type(e).__name__}: {str(e)[:160]}); assuming a"
            " flat-namespace bucket"
        )
        return False

    safe_is_hns._deepswe_patched = True
    ocp_gcs_utils.is_hierarchical_namespace_enabled = safe_is_hns
    print(
        "[Monkeypatch] Patched orbax gcs_utils.is_hierarchical_namespace_enabled"
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"[Monkeypatch] Failed to patch orbax HNS probe: {e}")


patch_orbax_gcs_hns_probe()

# ====== Logging Configuration ======
# 1. Force absl to use python logging
absl_logging.use_python_logging()

# 2. Configure the root logger
log_level = getattr(logging, args.logging_level.upper())
logging.basicConfig(
    stream=sys.stdout,
    level=log_level,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(log_level)
logging.getLogger("absl").setLevel(log_level)

# 4. Set absl verbosity so they actually print
absl_logging.set_verbosity(getattr(absl_logging, args.logging_level.upper()))
absl_logging.set_stderrthreshold(args.logging_level.lower())

# %%
# ==========================================
# 1. Path Setup
# ==========================================

# Use the current working directory as ROOT folder
workdir = os.getcwd()
tunix_root = os.path.join(workdir, "tunix")
pathways_root = os.path.join(workdir, "pathways-utils")
r2egym_root = os.path.join(workdir, "r2egym")

for root in [workdir, pathways_root, r2egym_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# Verification
import tunix
import r2egym  # pytype: disable=import-error

if os.getenv("JAX_PLATFORMS") == "proxy":
  import pathwaysutils

  pathwaysutils.initialize()


# %%
# ==========================================
# 2. Imports from Custom Modules
# ==========================================
from tunix.models.qwen3 import params as params_lib
from tunix.models.qwen3 import model as model_lib
from tunix.sft import utils as sft_utils
from tunix.sft import metrics_logger
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix import PerfMetricsConfig
from tunix.perf.experimental.export import PerfMetricsExport
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from tunix.utils import mllog_utils  # pytype: disable=missing-module-attribute,import-error
try:
  from examples.deepswe import swe_agent
  from examples.deepswe import swe_env
except ImportError:
  from examples.deepswe import swe_agent  # pytype: disable=import-error
  from examples.deepswe import swe_env  # pytype: disable=import-error

if args.rcp_logging:
  mllog_utils.init_start(args)

# %%
# ==========================================
# 3. Environment Configuration
# ==========================================
DATASET_CACHE = os.getenv(
    "DATASET_CACHE", os.path.join(workdir, "dataset_cache")
)
os.makedirs(DATASET_CACHE, exist_ok=True)

os.environ["KUBECONFIG"] = "~/.kube/config"
os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
os.environ["NODE_SELECTOR_VAL"] = (
    NODE_SELECTOR_VAL  # NB: change based on your node pool name
)
print(
    "Using Kubernetes node selector:"
    f" {os.environ['NODE_SELECTOR_KEY']}={os.environ['NODE_SELECTOR_VAL']}"
)


# Kubernetes Setup
try:
  k8s_config.load_kube_config()
  k8s_client = client.CoreV1Api()
  # k8s_client.list_namespace(timeout_seconds=5)
except Exception as e:
  print(f"Warning: Kubernetes config loading failed: {e}")


# %%
# ==========================================
# 4. Model & Training Hyperparameters
# ==========================================
MODEL_SOURCE = args.model_source
MODEL_ABSOLUTE_PATH = args.model_absolute_path

if MODEL_ABSOLUTE_PATH:
  MODEL_PATH = MODEL_ABSOLUTE_PATH
  print(f"Using model from absolute path: {MODEL_PATH}")
else:
  MODELS_BASE_DIR = os.path.join(workdir, args.models_base_dir)
  MODEL_PATH = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)

  print(f"Looking for local model at: {MODEL_PATH}...")

  # Check if directory exists and is not empty
  if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print(f"Model not found locally. Starting download to {MODEL_PATH}...")
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Requires full HF repository ID (e.g. "Qwen/Qwen3-32B").
    hf_repo_id = (
        f"Qwen/{MODEL_VERSION}"
        if "qwen" in MODEL_VERSION.lower() and "/" not in MODEL_VERSION
        else MODEL_VERSION
    )
    snapshot_download(  # pyrefly: ignore[no-matching-overload]
        repo_id=hf_repo_id,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )
    print("Download complete!")
  else:
    print(f"✅ Found existing local model at {MODEL_PATH}")

# ====== Data ======
TRAIN_FRACTION = args.train_fraction

# ====== Reproducibility ======
SEED = args.seed

# ====== LoRA ======
RANK = args.rank
ALPHA = args.alpha
TRAIN_WITH_LORA = args.train_with_lora

# ====== Sharding ======
# MESH = [(4, 2), ("fsdp", "tp")]


# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_GENERATIONS = args.num_generations  # This corresponds to `G` in Algorithm 1

# === other GRPO configs ===
NUM_ITERATIONS = args.num_iterations
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high
OFF_POLICY_STEPS = args.off_policy_steps

# ====== Training ======
DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "int32": jnp.int32,
}
DTYPE = DTYPE_MAP[args.dtype]
PARAM_DTYPE = DTYPE_MAP[args.param_dtype]
USE_FLASH_ATTENTION = args.use_flash_attention
FLASH_ATTENTION_BLOCK_SIZE = args.flash_attention_block_size
ENABLE_REMAT = args.enable_remat
REMAT_POLICY = args.remat_policy
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size


COMPUTE_LOGPS_MICRO_BATCH_SIZE = args.compute_logps_micro_batch_size
TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
ROLLOUT_MICRO_BATCH_SIZE = args.rollout_micro_batch_size

EVAL_EVERY_N_STEPS = args.eval_every_n_steps
NUM_EPOCHS = args.num_epochs

# Number of training steps.
MAX_STEPS = args.max_steps

# Max turns in mult-agent interaction (set to 1 for single-turn)
MAX_TURNS = args.max_turns
EPISODE_TIMEOUT_SECS = args.episode_timeout_secs
STEP_TIMEOUT_SECS = args.step_timeout_secs
REWARD_TIMEOUT_SECS = args.reward_timeout_secs

MAX_CONCURRENCY = args.max_concurrency
USE_AGENT_SANDBOX = args.use_agent_sandbox
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
print(f"kv_cache_size (Capped): {KV_CACHE_SIZE}")
# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = args.learning_rate
B1 = args.b1
B2 = args.b2
WEIGHT_DECAY = args.weight_decay
# WARMUP_STEPS = int(args.warmup_ratio * MAX_STEPS)
MAX_GRAD_NORM = args.max_grad_norm
OPTIMIZER_OFFLOAD = args.optimizer_offload

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = args.save_interval_steps
MAX_TO_KEEP = args.max_to_keep
DO_MEM_PROFILING = args.do_mem_profiling

# ====== Rollout ======
ROLLOUT_ENGINE = args.rollout_engine
CKPT_DIR = (
    args.ckpt_dir
    if args.ckpt_dir and args.ckpt_dir.lower() not in ("none", "null")
    else None
)


def latest_committed_checkpoint_step(actor_ckpt_dir: str) -> int:
  """Returns the latest finalized orbax step under `actor_ckpt_dir`, or 0.

  Mirrors what the actor trainer will restore: integer step directories, which
  on GCS must also contain orbax's `commit_success.txt` marker.
  """
  from etils import epath  # pylint: disable=g-import-not-at-top

  root = epath.Path(actor_ckpt_dir)
  if not root.exists():
    return 0
  needs_marker = str(actor_ckpt_dir).startswith("gs://")
  steps = [
      int(p.name)
      for p in root.iterdir()
      if p.name.isdigit()
      and (not needs_marker or (p / "commit_success.txt").exists())
  ]
  return max(steps, default=0)


RESUME_STEP = (
    latest_committed_checkpoint_step(os.path.join(CKPT_DIR, "actor"))
    if CKPT_DIR
    else 0
)
print(f"[resume] latest committed checkpoint step: {RESUME_STEP}", flush=True)


# Agentic rollout dispatches one prompt at a time; vLLM still admits the
# distributed recipe's 32 concurrent generations.
VLLM_MAX_NUM_SEQS = MAX_CONCURRENCY

VLLM_UTILIZATION = args.vllm_utilization
VLLM_RESHARD_CHUNK_SIZE = args.vllm_reshard_chunk_size

# Max number of tokens to be processed in parallel by vllm.
VLLM_MAX_BATCHED_TOKENS = args.max_num_batched_tokens
print(f"vllm_max_batched_tokens: {VLLM_MAX_BATCHED_TOKENS}")

OVERLONG_FILTER = args.overlong_filter
FILTER_STATUSES = (
    {agent_types.TrajectoryStatus[name] for name in args.filter_statuses}
    if args.filter_statuses is not None
    else None
)
LOSS_AGG_MODE = args.loss_agg_mode
ADVANTAGE_ESTIMATOR = args.advantage_estimator
USE_ROLLOUT_LOGPS = args.use_rollout_logps
if args.score_centering:
  if ROLLOUT_ENGINE != "vllm":
    raise ValueError("Score Centering requires --rollout_engine vllm.")
  if not USE_ROLLOUT_LOGPS:
    raise ValueError("Score Centering requires --use_rollout_logps true.")
  if args.top_p != 1.0 or args.top_k != 0:
    raise ValueError("Score Centering requires --top_p 1.0 --top_k 0.")
  if args.score_centering_top_k < 2:
    raise ValueError("Score Centering requires --score_centering_top_k >= 2.")
RCP_LOGGING = args.rcp_logging


# %%
# ==========================================
# 5. Tokenizer & Dataset Preparation (No JAX)
# ==========================================
tokenizer_path = MODEL_PATH
local_files_only = True
if MODEL_SOURCE == "maxtext" and MODEL_PATH.startswith("gs://"):
  tokenizer_path = MODEL_VERSION
  local_files_only = False
  print(f"Loading tokenizer from HF Hub: {tokenizer_path}")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=local_files_only, trust_remote_code=True
)

chat_parser = template_parser.QwenChatTemplateParser(tokenizer)

print("Loading Dataset...")

dataset = deepswe_data.load_clean_dataset(
    dataset_name=args.dataset_name,
    dataset_revision=args.dataset_revision,
    dataset_split=args.dataset_split,
    dataset_path=args.dataset_path or "",
    gold_whitelist=args.gold_whitelist or "",
    cache_dir=DATASET_CACHE,
    shuffle=args.shuffle,
    seed=SEED,
)
if len(dataset) * NUM_EPOCHS < MAX_STEPS * BATCH_SIZE:
  raise ValueError(
      "Clean dataset does not cover the requested run: "
      f"{len(dataset)} tasks x {NUM_EPOCHS} epochs < "
      f"{MAX_STEPS} steps x {BATCH_SIZE} prompts"
  )
grain_dataset = grain.MapDataset.source(dataset)  # pyrefly: ignore[bad-argument-type]


train_dataset, _ = data_lib.post_init_dataset(
    grain_dataset,
    tokenizer,  # pyrefly: ignore[bad-argument-type]
    batch_size=BATCH_SIZE,
    num_batches=None,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    prompt_key="problem_statement",
    custom_batch_fn=deepswe_data.batch_fn,
)


class SkippedBatchesIterator:
  """Answers the learner's resume fast-forward without touching the dataset.

  On resume the learner skips `global_steps` batches by calling next() on the
  iterator it is given. We already advanced the underlying dataset past those
  batches (before the sandbox prewarmer saw them), so the first `num_skipped`
  calls just return None.
  """

  def __init__(self, inner, num_skipped: int):
    self._inner = inner
    self._remaining = num_skipped

  def __iter__(self):
    return self

  def __next__(self):
    if self._remaining > 0:
      self._remaining -= 1
      return None
    return next(self._inner)


if RESUME_STEP:
  # Fast-forward here so the sandbox prewarmer starts at the resumed batch
  # instead of warming (and immediately tearing down) pools for every skipped
  # batch inside the learner's skip loop.
  train_dataset = iter(train_dataset)
  for skipped in range(RESUME_STEP):
    try:
      next(train_dataset)
    except StopIteration:
      print(f"[resume] dataset exhausted after {skipped} batches", flush=True)
      break
  print(f"[resume] fast-forwarded the dataset by {RESUME_STEP} batches", flush=True)

fleet = None
if USE_AGENT_SANDBOX:
  fleet = swe_env._init_global_fleet(
      tasks=dataset,
      max_concurrency=MAX_CONCURRENCY,
      num_generations=NUM_GENERATIONS,
      batch_size=MINI_BATCH_SIZE,
  )
  train_dataset = swe_env.PrewarmDatasetIterator(
      train_dataset,
      fleet=fleet,
      num_generations=NUM_GENERATIONS,
      batch_size=MINI_BATCH_SIZE,
  )

if RESUME_STEP:
  train_dataset = SkippedBatchesIterator(train_dataset, RESUME_STEP)


# %%
# ==========================================
# 6. JAX Device & Mesh Setup
# ==========================================
import jax
import jax.numpy as jnp
from tunix.models.automodel import AutoModel
from tunix.models.automodel import ModelSource
from tunix.models.automodel import call_model_config

config = call_model_config(MODEL_VERSION.rsplit("/", 1)[-1].lower())

if ENABLE_REMAT:
  _REMAT_POLICY_MAP = {
      "block": model_lib.RematConfig.BLOCK,
      "decoder": model_lib.RematConfig.DECODER,
  }
  config.remat_config = _REMAT_POLICY_MAP[REMAT_POLICY]

if DTYPE is not None:
  config.dtype = DTYPE

if USE_FLASH_ATTENTION:
  config.use_flash_attention = USE_FLASH_ATTENTION
  if FLASH_ATTENTION_BLOCK_SIZE is not None:
    config.flash_attention_block_size = FLASH_ATTENTION_BLOCK_SIZE

devices = jax.devices()
total_devices = len(devices)

# 1. Resolve Rollout Mesh Dimensions
# Each explicitly-provided dim becomes an axis in the mesh; unspecified dims are
# dropped (not defaulted to 1), so passing only --rollout_mesh_fsdp yields a 1D mesh.
# If nothing is provided, fall back to the split-fraction heuristic (2D: fsdp+tp).
tp_axis_name = "tensor" if MODEL_SOURCE == "maxtext" else "tp"

rollout_fsdp = args.rollout_mesh_fsdp
rollout_tp = args.rollout_mesh_tp
if rollout_fsdp is not None or rollout_tp is not None:
  rollout_dims = []
  if rollout_fsdp is not None:
    rollout_dims.append(("fsdp", rollout_fsdp))
  if rollout_tp is not None:
    rollout_dims.append((tp_axis_name, rollout_tp))
else:
  num_rollout_devices = int(total_devices * args.rollout_split_fraction)
  rollout_tp = int(np.gcd(num_rollout_devices, config.num_kv_heads))
  rollout_fsdp = num_rollout_devices // rollout_tp
  rollout_dims = [("fsdp", rollout_fsdp), (tp_axis_name, rollout_tp)]
num_rollout_devices = int(np.prod([d for _, d in rollout_dims]))

# 2. Resolve Train Mesh Dimensions
# Same rule: each provided dim becomes an axis; unspecified dims are dropped.
# Supports fsdp-only, fsdp+sp, fsdp+tp, fsdp+sp+tp, etc. If nothing is provided,
# fall back to leftover devices (2D: fsdp+tp).
train_fsdp = args.train_mesh_fsdp
train_sp = args.train_mesh_sp
train_tp = args.train_mesh_tp
if any(v is not None for v in (train_fsdp, train_sp, train_tp)):
  train_dims = []
  train_dims.append(("fsdp", train_fsdp if train_fsdp is not None else 1))
  if train_sp is not None:
    train_dims.append(("sp", train_sp))
  train_dims.append((tp_axis_name, train_tp if train_tp is not None else 1))
else:
  num_train_devices = total_devices - num_rollout_devices
  train_fsdp = int(
      np.gcd(num_train_devices, TRAIN_MICRO_BATCH_SIZE * NUM_GENERATIONS)
  )
  train_tp = num_train_devices // train_fsdp
  train_dims = [("fsdp", train_fsdp), (tp_axis_name, train_tp)]
num_train_devices = int(np.prod([d for _, d in train_dims]))

# 3. Sanity Check
if num_rollout_devices + num_train_devices > total_devices:
  raise ValueError(
      f"Requested {num_rollout_devices} rollout devices + {num_train_devices} "
      f"train devices, but cluster only has {total_devices} available."
  )

# 4. Route to Meshes
rollout_axis_names = tuple(name for name, _ in rollout_dims)
rollout_shape = tuple(d for _, d in rollout_dims)
train_axis_names = tuple(name for name, _ in train_dims)
train_shape = tuple(d for _, d in train_dims)

rollout_devices = np.array(devices[:num_rollout_devices]).reshape(rollout_shape)
train_devices = np.array(
    devices[num_rollout_devices : num_rollout_devices + num_train_devices]
).reshape(train_shape)

rollout_mesh = Mesh(rollout_devices, axis_names=rollout_axis_names)
train_mesh = Mesh(train_devices, axis_names=train_axis_names)


print(
    f"*** Rollout Mesh *** | dims: {rollout_dims} | Shape: {rollout_mesh.shape}"
)
print(f"*** Train Mesh *** | dims: {train_dims} | Shape: {train_mesh.shape}")

if train_sp is not None:
  config.shd_config = model_lib.ShardingConfig.get_default_sharding(
      enable_sp=True
  )

# %%
# ==========================================
# 7. Model Initialization
# ==========================================

if MODEL_SOURCE == "maxtext":
  qwen_reference, _ = AutoModel.from_pretrained(
      model_id=MODEL_VERSION,
      mesh=train_mesh,
      model_source=ModelSource.MAXTEXT,
      model_path=MODEL_PATH,
      enable_checkpointing=True,
      allow_split_physical_axes=True,
      scan_layers=False,
  )
else:
  qwen_reference = params_lib.create_model_from_safe_tensors(
      MODEL_PATH, config, mesh=train_mesh, dtype=PARAM_DTYPE  # pyrefly: ignore[bad-argument-type]
  )


def get_lora_model(base_model, model_mesh):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with compat.set_mesh(model_mesh):
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


if TRAIN_WITH_LORA:
  qwen_actor = get_lora_model(qwen_reference, train_mesh)
else:
  graph_def, params = nnx.split(qwen_reference)
  qwen_actor = nnx.merge(
      graph_def,
      jax.tree.map(jnp.copy, params),
  )

sft_utils.show_hbm_usage()

# %%

# %%
# ==========================================
# 8. Optimizer & Checkpointing
# ==========================================
if CKPT_DIR:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
  )
else:
  checkpointing_options = None


metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=args.metric_logger_dir,
    project_name=os.getenv("WANDB_PROJECT", "trellis-deepswe"),
    run_name=os.getenv("WANDB_RUN_NAME", "deepswe-agentic-q4-clean"),
    flush_every_n_steps=1,
    backend_kwargs={
        "custom_backend": [
            partial(
                metrics_logger.TensorboardBackend,
                log_dir=args.metric_logger_dir,
                flush_every_n_steps=1,
            ),
            partial(
                wandb_step_backend.TrainingStepWandbBackend,
                project=os.getenv("WANDB_PROJECT", "trellis-deepswe"),
                name=os.getenv("WANDB_RUN_NAME", "deepswe-agentic-q4-clean"),
                config=vars(args),
            ),
        ]
    },
)

optimizer = optax.schedules.inject_hyperparams(optax.adamw)(
    learning_rate=LEARNING_RATE, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY
)

if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )


# %%
# ==========================================
# 9. RL Cluster Setup
# ==========================================

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": KV_CACHE_SIZE,
    # The TPU JAX vLLM backend rejects per-request seeds. Set the engine seed
    # below while keeping the dataset and model seed at 42.
    "seed": None if ROLLOUT_ENGINE == "vllm" else SEED,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "eos_tokens": [tokenizer.encode("<|im_end|>")[0]],  # pyrefly: ignore[missing-attribute]
    "return_logprobs": USE_ROLLOUT_LOGPS,
    # Set before RLEngine creates VllmSampler and its max_logprobs limit.
    "num_logprobs": args.score_centering_top_k if args.score_centering else 1,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
}

sglang_jax_rollout_dict = {
    "rollout_sglang_jax_model_version": MODEL_PATH,  # Uses local absolute path
    "rollout_sglang_jax_mem_fraction_static": 0.9,
    "rollout_sglang_jax_init_with_random_weights": True,
    "rollout_sglang_jax_disable_radix_cache": False,
    "rollout_sglang_jax_enable_deterministic_sampling": False,
    "rollout_sglang_jax_chunked_prefill_size": 2048,
    "rollout_sglang_jax_max_running_requests": MAX_CONCURRENCY,
    "rollout_sglang_jax_page_size": 128,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": (
        tokenizer_path if MODEL_SOURCE == "maxtext" else MODEL_PATH
    ),
    "rollout_vllm_hbm_utilization": VLLM_UTILIZATION,
    "rollout_vllm_reshard_chunk_size": VLLM_RESHARD_CHUNK_SIZE,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": True,
    "tensor_parallel_size": rollout_mesh.shape.get(tp_axis_name, 1),
    "data_parallel_size": rollout_mesh.shape.get("fsdp", 1),
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "seed": SEED,
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "tokenizer": tokenizer_path,
    },
}

if MODEL_SOURCE == "maxtext":
  vllm_rollout_dict["rollout_vllm_kwargs"]["hf_overrides"] = {
      "architectures": ["MaxTextForCausalLM"]
  }
  vllm_rollout_dict["rollout_vllm_additional_config"] = {
      "maxtext_config": {
          "model_name": MODEL_VERSION.lower().split("/")[-1],
          "model_call_mode": "inference",
          "enable_dp_attention": False,
          "allow_split_physical_axes": True,
          "log_config": False,
          "weight_dtype": "bfloat16",
          "prefuse_moe_weights": True,
          "attention": "vllm_rpa",
      }
  }
  # Force no-op mappings for weight sync if both trainer and sampler use MaxText
  if hasattr(qwen_reference, "use_no_op_mappings"):
    qwen_reference.use_no_op_mappings = True  # pyrefly: ignore[missing-attribute]
  if hasattr(qwen_actor, "use_no_op_mappings"):
    qwen_actor.use_no_op_mappings = True  # pyrefly: ignore[missing-attribute]
    logging.info("Forced use_no_op_mappings=True on actor/reference models.")


if ROLLOUT_ENGINE == "sglang_jax":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **sglang_jax_rollout_dict
  )
elif ROLLOUT_ENGINE == "vllm":
  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
  # Currently, vllm does not support LoRA properly.
  if TRAIN_WITH_LORA:
    vllm_rollout_dict["rollout_vllm_lora_config"] = {
        "max_lora_rank": RANK,
    }
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")


def filter_logical_rules(rules, mesh):
  """Filters logical sharding rules to keep only physical axes present in mesh."""
  valid_axes = set(mesh.shape.keys())
  filtered_rules = []
  for logical_axis, physical_axes in rules:
    if isinstance(physical_axes, (list, tuple)):
      new_phys = [ax for ax in physical_axes if ax in valid_axes]
      filtered_rules.append((logical_axis, tuple(new_phys)))
    else:
      if physical_axes in valid_axes:
        filtered_rules.append((logical_axis, physical_axes))
      else:
        filtered_rules.append((logical_axis, ()))
  return tuple(filtered_rules)


role_to_logical_axis_rule = None
logical_rules = getattr(
    getattr(getattr(qwen_reference, "base", None), "config", None),
    "logical_axis_rules",
    None,
)
if logical_rules:
  print(f"Configuring role_to_logical_axis_rule with: {logical_rules}")
  role_to_logical_axis_rule = {
      rl_engine_lib.Role.ACTOR: filter_logical_rules(
          logical_rules, train_mesh
      ),
      rl_engine_lib.Role.REFERENCE: filter_logical_rules(
          logical_rules, train_mesh
      ),
      rl_engine_lib.Role.ROLLOUT: filter_logical_rules(
          logical_rules, rollout_mesh
      ),
  }

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: train_mesh,
        rl_engine_lib.Role.REFERENCE: train_mesh,
        rl_engine_lib.Role.ROLLOUT: rollout_mesh,
    },
    role_to_logical_axis_rule=role_to_logical_axis_rule,
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        train_trajectory_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        compute_logps_trajectory_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
        compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
        rollout_micro_batch_size=ROLLOUT_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        # optimizer_offload=OPTIMIZER_OFFLOAD,
    ),
    rollout_config=rollout_engine_config,
)
sft_utils.show_hbm_usage()

RLClusterCls = getattr(
    rl_engine_lib, "RLCluster", getattr(rl_engine_lib, "RLEngine", None)
)
rl_engine = RLClusterCls(  # pytype: disable=not-callable
    actor=qwen_actor,
    reference=qwen_reference,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

if RCP_LOGGING:
  rl_engine.with_external_metrics_logger(
      mllog_utils.create_rcp_metrics_logger(args, rl_engine=rl_engine)
  )
  mllog_utils.init_print(
      args,
      train_dataset=dataset,
      rollout_mesh=rollout_mesh,
      train_mesh=train_mesh,
      total_devices=total_devices,
  )

# %%
# ==========================================
# 10. Learner & Agent Setup
# ==========================================

config_kwargs = {
    "num_generations": NUM_GENERATIONS,
    "num_iterations": NUM_ITERATIONS,
    "max_response_length": MAX_RESPONSE_LENGTH,
    "beta": BETA,
    "epsilon": EPSILON,
    "system_prompt": swe_agent.SWE_SYSTEM_PROMPT,
    "max_concurrency": MAX_CONCURRENCY,
    "epsilon_high": EPSILON_HIGH,
    "off_policy_steps": OFF_POLICY_STEPS,
    "episode_timeout": EPISODE_TIMEOUT_SECS,
    "overlong_filter": OVERLONG_FILTER,
    "filter_statuses": FILTER_STATUSES,
    "loss_agg_mode": LOSS_AGG_MODE,
    "advantage_estimator": ADVANTAGE_ESTIMATOR,
    "use_rollout_logps": USE_ROLLOUT_LOGPS,
    "score_centering": args.score_centering,
    "score_centering_top_k": args.score_centering_top_k,
    "score_centering_eps": args.score_centering_eps,
    "exact_token_continuity": args.exact_token_continuity,
    "sampler_is": None if args.sampler_is == "none" else args.sampler_is,
    "sampler_is_threshold": args.sampler_is_threshold,
    "truncated_importance_sampling_ratio_min": (
        args.truncated_importance_sampling_ratio_min
    ),
    "truncated_importance_sampling_ratio": (
        args.truncated_importance_sampling_ratio
    ),
    "seq_logprob_error_threshold": args.seq_logprob_error_threshold,
}

grpo_config = agentic_grpo_learner.GRPOConfig(**config_kwargs)

agentic_grpo_learner = agentic_grpo_learner.GRPOLearner(
    rl_engine,
    reward_fns=None,
    agent_class=swe_agent.SWEAgent,
    agent_kwargs={
        "scaffold": args.scaffold,
        "action_compat_mode": args.action_compat_mode,
    },
    env_class=swe_env.SWEEnv,
    env_kwargs={
        "max_steps": MAX_TURNS,
        "step_timeout": STEP_TIMEOUT_SECS,
        "reward_timeout": REWARD_TIMEOUT_SECS,
        "backend": args.env_backend,
        "scaffold": args.scaffold,
        "use_agent_sandbox": USE_AGENT_SANDBOX,
        "fleet": fleet,
    },
    algo_config=grpo_config,
    chat_parser=chat_parser,
)


if RCP_LOGGING:
  mllog_utils.train_start(args)

if rl_engine.global_steps != RESUME_STEP:
  raise RuntimeError(
      f"Restored global step {rl_engine.global_steps} does not match the"
      f" latest committed checkpoint step {RESUME_STEP} used to fast-forward"
      " the dataset."
  )

print("Starting training...", flush=True)
agentic_grpo_learner.train(train_dataset=train_dataset)

if RCP_LOGGING:
  mllog_utils.train_stop(args)


# %%
