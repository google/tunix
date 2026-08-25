"""Agentic FrozenLake GRPO recipe for Qwen3-8B on a single TPU host.

Targets v5p-8 / v6e-4 -class hosts where actor, reference, and rollout share
a single mesh. Hyperparameters are exposed via argparse; the rollout backend
is selected via the ``ROLLOUT_ENGINE`` environment variable ("vllm" or
"vanilla", default "vllm").
"""

import contextlib
import json
import logging
import math
import os
import sys
from typing import List

from absl import logging as absl_logging
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
import qwix

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

from tunix.models.qwen3 import params as params_lib
from tunix.models.qwen3 import model as model_lib
from tunix.oss import utils as oss_utils
from tunix.sft import metrics_logger
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import dp_workloads
from tunix.rl import frozenlake_checkpoint
from tunix.rl.rollout import base_rollout
from tunix.sft import checkpoint_manager as checkpoint_manager_lib
from tunix.sft import checkpoint_options as checkpoint_options_lib
from tunix.sft import utils as sft_utils
from tunix.cli.utils import data as data_lib
# The A1b/A2 contract and A3 adapter preflights exit before constructing the
# learner and intentionally have no FrozenLake environment dependency.  Keep
# the normal path's eager imports unchanged so a real L3 run still fails
# immediately when its environment package is absent.
CANON_P3_APC_BOUNDARY_REPORT = os.getenv(
    "CANON_P3_APC_BOUNDARY_REPORT", ""
)
_CANON_PRELEARNER_ONLY = (
    os.getenv("CANON_L3_CONTRACT_ONLY", "") == "1"
    or os.getenv("CANON_L3_A3_ONLY", "") == "1"
    or os.getenv("CANON_P28_G3_ONLY", "") == "1"
    or os.getenv("CANON_P28_G4_ONLY", "") == "1"
    or os.getenv("CANON_P28_G5_ONLY", "") == "1"
    or os.getenv("CANON_P38_FROZENLAKE_REPLAY", "") == "1"
    or bool(CANON_P3_APC_BOUNDARY_REPORT)
)
if not _CANON_PRELEARNER_ONLY:
  from examples.frozenlake.agent import FrozenLakeAgent
  from examples.frozenlake.env import FrozenLakeEnv
  from examples.frozenlake import data as frozenlake_data
  from examples.frozenlake import p57_workloads
else:
  FrozenLakeAgent = None
  FrozenLakeEnv = None

_DISTRIBUTED_INITIALIZED = False
try:
  import pathwaysutils
  pathwaysutils.initialize()
  _DISTRIBUTED_INITIALIZED = True
except Exception:
  pass

if not _DISTRIBUTED_INITIALIZED:
  # Multi-host TPU (e.g. v5p-16, v6e-16+) needs jax.distributed.initialize()
  # for Orbax checkpoint barrier sync. Single-host slices are auto-detected,
  # so this is a no-op there.
  try:
    jax.distributed.initialize()
  except Exception as exc:
    print(f"jax.distributed.initialize() skipped: {exc}")

print("jax devices: ", jax.devices())

# %%
import argparse

arg_parser = argparse.ArgumentParser(
    description="Train FrozenLake on Qwen3-8B (single-host TPU)."
)
# Effective on-policy batch is `batch_size * num_generations` per global step.
# Tuned together with `num_generations=8` to keep per-step rollout latency
# manageable on a single host while preserving enough samples per prompt for
# the GRPO group-mean baseline.
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--mini_batch_size", type=int, default=64)
arg_parser.add_argument(
    "--train_trajectory_micro_batch_size", type=int, default=None
)
arg_parser.add_argument("--mesh_dp", type=int, default=None)
arg_parser.add_argument("--mesh_tp", type=int, default=None)
arg_parser.add_argument("--max_steps", type=int, default=None)
arg_parser.add_argument("--learning_rate", type=float, default=1e-6)
arg_parser.add_argument("--b1", type=float, default=0.9)
# AdamW second-moment decay (β2). Lower than the AdamW default (0.999) so the
# second-moment estimate adapts faster to the non-stationary gradient
# distribution of RL fine-tuning.
arg_parser.add_argument("--b2", type=float, default=0.95)
arg_parser.add_argument("--weight_decay", type=float, default=0.0)
arg_parser.add_argument("--num_batches", type=int, default=150)
arg_parser.add_argument("--num_generations", type=int, default=8)
arg_parser.add_argument("--beta", type=float, default=0.0)
# GSPO-token defaults: tight clip ratios because the importance ratio is
# sequence-mean (much lower variance than per-token PPO), so a wider clip would
# rarely bind. Override via --epsilon/--epsilon_high for PPO-style runs.
arg_parser.add_argument("--epsilon", type=float, default=0.003)
arg_parser.add_argument("--epsilon_high", type=float, default=0.005)
arg_parser.add_argument(
    "--loss_algo", type=str, default="gspo-token",
    help="'grpo' (per-token PPO) or 'gspo-token' (sequence-mean IS).",
)
arg_parser.add_argument("--max_prompt_length", type=int, default=2048)
arg_parser.add_argument("--max_response_length", type=int, default=2048)
arg_parser.add_argument("--temperature", type=float, default=0.7)
# No top_p / top_k filter at rollout time. The processed_logprobs returned by
# the rollout engine apply log_softmax over the filtered logit set; if filters
# are active, the rollout's denominator covers only those tokens while the
# trainer recompute uses the full vocabulary, biasing the sampler-trainer
# logprob diff by ~log(vocab / k) per position even when both forward passes
# agree exactly. Disabling the filters at rollout keeps the two distributions
# comparable; exploration can be controlled via temperature.
arg_parser.add_argument("--top_p", type=float, default=1.0)
arg_parser.add_argument("--top_k", type=int, default=0)
# Concurrent rollout threads are global. The vLLM limits below are per data
# parallel rank, so their global capacity is multiplied by mesh DP. Keep this
# distinction explicit: passing a global limit as a per-rank limit silently
# expands the TPU precompile shapes by the DP width.
arg_parser.add_argument("--max_concurrency", type=int, default=256)
arg_parser.add_argument("--vllm_max_num_seqs", type=int, default=64)
arg_parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None)
arg_parser.add_argument("--env_max_steps", type=int, default=8)
arg_parser.add_argument("--num_test_batches", type=int, default=2)
arg_parser.add_argument("--eval_every_n_steps", type=int, default=10)
arg_parser.add_argument(
    "--evaluation_only",
    action="store_true",
    help="Restore one P57 checkpoint and run held-out rollout without training.",
)
arg_parser.add_argument("--p57_workload_candidate", type=str, default="")
arg_parser.add_argument("--p57_data_split", type=str, default="")
arg_parser.add_argument(
    "--p57_calibration_mode", choices=("", "stochastic"), default=""
)
arg_parser.add_argument(
    "--p57_calibration_recipes", type=str, default=""
)
arg_parser.add_argument("--shuffle_data", type=bool, default=True)
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
)
arg_parser.add_argument(
    "--kl_loss_mode", type=str, default="low_var_kl"
)
# Advantage estimator. "rloo" (leave-one-out baseline) has smaller-magnitude
# advantages than "grpo" (z-score with /std), which interacts gently with very
# tight PPO clip ratios. "grpo" is the registry default; switch via CLI.
arg_parser.add_argument(
    "--advantage_estimator", type=str, default="rloo",
    help="'grpo' (z-score) or 'rloo' (leave-one-out baseline).",
)
arg_parser.add_argument(
    "--sampler_is",
    choices=("token", "none"),
    default="token",
    help=(
        "Sampler/trainer importance-sampling correction. 'none' keeps "
        "rollout logprobs as the old-policy denominator without TIS weights."
    ),
)
args, _ = arg_parser.parse_known_args()

CANON_P57_RUN_KIND = os.getenv("CANON_P57_RUN_KIND", "")
CANON_P57_INFERENCE_REGIME = os.getenv("CANON_P57_INFERENCE_REGIME", "")
CANON_P57_TIM_ARM = os.getenv("CANON_P57_TIM_ARM", "")
_CANON_VLLM_PREFIX_CACHE_RAW = os.getenv(
    "CANON_VLLM_ENABLE_PREFIX_CACHING", "0"
)
if _CANON_VLLM_PREFIX_CACHE_RAW not in ("", "0", "1"):
  raise ValueError(
      "CANON_VLLM_ENABLE_PREFIX_CACHING must be absent, empty, 0, or 1"
  )
CANON_VLLM_ENABLE_PREFIX_CACHING = (
    _CANON_VLLM_PREFIX_CACHE_RAW == "1"
)
CANON_P57_EVALUATION = CANON_P57_RUN_KIND == "eval"
CANON_P57_CALIBRATION = CANON_P57_RUN_KIND == "calibration"
CANON_P57_STOCK_TRAIN = (
    CANON_P57_RUN_KIND == "train"
    and CANON_P57_TIM_ARM in ("mismatch", "is")
    and CANON_P57_INFERENCE_REGIME == "stock-fast"
)
CANON_P57_STOCK_EVAL = (
    CANON_P57_RUN_KIND == "eval"
    and CANON_P57_TIM_ARM in ("mismatch", "is")
    and CANON_P57_INFERENCE_REGIME == "stock-fast"
)
CANON_P57_NO_UPDATE = CANON_P57_EVALUATION or CANON_P57_CALIBRATION
CANON_P57_WORKLOAD_CANDIDATE = os.getenv(
    "CANON_P57_WORKLOAD_CANDIDATE", ""
)
CANON_P57_DATA_SPLIT = os.getenv("CANON_P57_DATA_SPLIT", "")
CANON_P57_CALIBRATION_MODE = os.getenv("CANON_P57_CALIBRATION_MODE", "")
CANON_P57_CALIBRATION_RECIPES = tuple(
    value for value in os.getenv(
        "CANON_P57_CALIBRATION_RECIPES", ""
    ).split(",") if value
)
if args.evaluation_only != CANON_P57_NO_UPDATE:
  raise ValueError(
      "--evaluation_only must agree with P57 eval/calibration run kinds"
  )
if CANON_P57_NO_UPDATE and os.getenv("CANON_PROFILE_FILE", "") != (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
):
  raise ValueError("isolated FrozenLake evaluation/calibration requires P57")
if (
    args.p57_workload_candidate != CANON_P57_WORKLOAD_CANDIDATE
    or args.p57_data_split != CANON_P57_DATA_SPLIT
):
  raise ValueError("P57 workload CLI arguments must match their signed env fields")
if bool(CANON_P57_WORKLOAD_CANDIDATE) != bool(CANON_P57_DATA_SPLIT):
  raise ValueError("P57 workload candidate and data split must be set together")
if CANON_P57_CALIBRATION:
  expected_recipes = ("m10", "m15", "m20")
  if (
      CANON_P57_WORKLOAD_CANDIDATE
      or CANON_P57_DATA_SPLIT
      or CANON_P57_CALIBRATION_MODE != "stochastic"
      or CANON_P57_CALIBRATION_RECIPES != expected_recipes
      or args.p57_calibration_mode != CANON_P57_CALIBRATION_MODE
      or tuple(value for value in args.p57_calibration_recipes.split(",") if value)
      != expected_recipes
  ):
    raise ValueError(
        "P57 calibration requires the exact signed "
        "m10,m15,m20 stochastic inventory and no single candidate"
    )
elif args.p57_calibration_mode or args.p57_calibration_recipes:
  raise ValueError("P57 calibration CLI arguments require run kind calibration")
if CANON_P57_WORKLOAD_CANDIDATE:
  if os.getenv("CANON_PROFILE_FILE", "") not in (
      "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env",
      "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
      "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env",
  ):
    raise ValueError("materialized P57 workloads require the P57 profile")
  p57_workload_spec = p57_workloads.candidate(
      CANON_P57_WORKLOAD_CANDIDATE
  )
  p57_workloads.validate_split(CANON_P57_DATA_SPLIT)
  if args.env_max_steps != p57_workload_spec.max_turns:
    raise ValueError(
        "P57 candidate max-turn contract drifted: "
        f"candidate={p57_workload_spec.name} "
        f"expected={p57_workload_spec.max_turns} got={args.env_max_steps}"
    )
else:
  p57_workload_spec = None

CANON_L3 = os.getenv("CANON_FROZENLAKE_L3", "") == "1"
CANON_P27 = os.getenv("CANON_FROZENLAKE_P27", "") == "1"
CANON_CONTRACT_ONLY = os.getenv("CANON_L3_CONTRACT_ONLY", "") == "1"
CANON_A3_ONLY = os.getenv("CANON_L3_A3_ONLY", "") == "1"
CANON_P28_G3_ONLY = os.getenv("CANON_P28_G3_ONLY", "") == "1"
CANON_P28_G4_ONLY = os.getenv("CANON_P28_G4_ONLY", "") == "1"
CANON_P28_G5_ONLY = os.getenv("CANON_P28_G5_ONLY", "") == "1"
CANON_P28_G5C_ONLY = os.getenv("CANON_P28_G5C_ONLY", "") == "1"
CANON_P28_G6_UPDATE = os.getenv("CANON_P28_G6_UPDATE", "") == "1"
CANON_P38_FROZENLAKE_REPLAY = (
    os.getenv("CANON_P38_FROZENLAKE_REPLAY", "") == "1"
    or bool(CANON_P3_APC_BOUNDARY_REPORT)
)
CANON_P29_FULL_TRAIN = os.getenv("CANON_P29_FULL_TRAIN", "") == "1"
CANON_P31_CONVERGENCE = os.getenv("CANON_P31_CONVERGENCE", "") == "1"
_P32_WORKLOAD_NAME = os.getenv("CANON_P32_WORKLOAD", "")
if _P32_WORKLOAD_NAME and not _P32_WORKLOAD_NAME.startswith("frozenlake"):
  raise ValueError(
      "FrozenLake recipe cannot run a different P32 workload: "
      f"{_P32_WORKLOAD_NAME!r}"
  )
CANON_P32_WORKLOAD = bool(_P32_WORKLOAD_NAME)
CANON_P33_SHORT_ALIGNMENT = (
    os.getenv("CANON_P33_SHORT_ALIGNMENT", "0") == "1"
)
CANON_P38_PRECHECK_ONLY = (
    os.getenv("CANON_P38_PRECHECK_ONLY", "0") == "1"
)
CANON_ALIGNMENT_TRAIN_MODE = dp_workloads.requires_alignment_train_mode(
    os.environ
)
CANON_P33_DISABLE_EVAL = os.getenv("CANON_P33_DISABLE_EVAL", "") == "1"
P32_WORKLOAD = dp_workloads.active_workload() if CANON_P32_WORKLOAD else None
P57_STOCK_FAST_ATTESTATION = None
if CANON_P32_WORKLOAD:
  if CANON_P57_CALIBRATION:
    P57_STOCK_FAST_ATTESTATION = (
        dp_workloads.validate_p57_stock_fast_environment(
            P32_WORKLOAD, os.environ
        )
    )
    if CANON_L3:
      raise ValueError("P57 stock-fast calibration forbids canonical L3")
    CANON_P33_ENABLE_EVAL = False
  elif CANON_P57_STOCK_TRAIN:
    P57_STOCK_FAST_ATTESTATION = (
        dp_workloads.validate_p57_stock_train_environment(
            P32_WORKLOAD, os.environ
        )
    )
    if CANON_L3:
      raise ValueError("P57 stock training forbids canonical L3")
    CANON_P33_ENABLE_EVAL = dp_workloads.frozenlake_evaluation_enabled(
        os.environ
    )
  elif CANON_P57_STOCK_EVAL:
    P57_STOCK_FAST_ATTESTATION = (
        dp_workloads.validate_p57_stock_eval_environment(
            P32_WORKLOAD, os.environ
        )
    )
    if CANON_L3:
      raise ValueError("P57 stock evaluation forbids canonical L3")
    CANON_P33_ENABLE_EVAL = False
  else:
    dp_workloads.validate_environment(
        P32_WORKLOAD, require_reduction_admission=True
    )
    if not CANON_L3:
      raise ValueError("canonical DP FrozenLake requires CANON_FROZENLAKE_L3=1")
    CANON_P33_ENABLE_EVAL = dp_workloads.frozenlake_evaluation_enabled(
        os.environ
    )
else:
  CANON_P33_ENABLE_EVAL = False
CANON_OPTIMIZER_PLACEMENT = dp_workloads.canonical_optimizer_placement(
    os.environ, require_explicit=CANON_P32_WORKLOAD
)
CANON_P30_OPT_STATE_OFFLOAD = (
    CANON_OPTIMIZER_PLACEMENT == "pinned-host-offload"
)
prelearner_modes = {
    "contract-only": CANON_CONTRACT_ONLY,
    "A3-only": CANON_A3_ONLY,
    "P28-G3-only": CANON_P28_G3_ONLY,
    "P28-G4-only": CANON_P28_G4_ONLY,
    "P28-G5-only": CANON_P28_G5_ONLY,
    "P38-FrozenLake-replay": CANON_P38_FROZENLAKE_REPLAY,
}
active_prelearner_modes = [
    name for name, active in prelearner_modes.items() if active
]
if len(active_prelearner_modes) > 1:
  raise ValueError(
      "canonical prelearner modes are mutually exclusive: "
      f"{active_prelearner_modes}"
  )
if active_prelearner_modes and not CANON_L3:
  raise ValueError(
      "canonical prelearner modes require CANON_FROZENLAKE_L3=1"
  )
if CANON_P27 and not CANON_L3:
  raise ValueError("CANON_FROZENLAKE_P27=1 requires CANON_FROZENLAKE_L3=1")
if CANON_P27 and active_prelearner_modes:
  raise ValueError("P27 cannot be combined with prelearner-only modes")
if CANON_P28_G5C_ONLY and (CANON_P27 or active_prelearner_modes):
  raise ValueError("P28 G5c cannot be combined with P27/prelearner modes")
if CANON_P28_G5C_ONLY and not CANON_L3:
  raise ValueError("P28 G5c requires CANON_FROZENLAKE_L3=1")
if CANON_P28_G6_UPDATE and not CANON_P27:
  raise ValueError("P28 G6 requires CANON_FROZENLAKE_P27=1")
if CANON_P28_G6_UPDATE and os.getenv(
    "CANON_ALIGNMENT_UPDATE_CANARY", ""
) != ("0" if CANON_ALIGNMENT_TRAIN_MODE else "1"):
  raise ValueError("P28 G6 requires update-canary mode")
if CANON_P31_CONVERGENCE:
  required_p31 = {
      "CANON_FROZENLAKE_L3": CANON_L3,
      "CANON_FROZENLAKE_P27": CANON_P27,
      "CANON_P28_G6_UPDATE": CANON_P28_G6_UPDATE,
      "CANON_P29_FULL_TRAIN": CANON_P29_FULL_TRAIN,
      "CANON_ALIGNMENT_TRAIN": os.getenv("CANON_ALIGNMENT_TRAIN") == "1",
  }
  missing_p31 = [name for name, active in required_p31.items() if not active]
  if missing_p31:
    raise ValueError(f"P31 convergence prerequisites missing: {missing_p31}")
  if os.getenv("CANON_ALIGNMENT_UPDATE_CANARY", "") != "0":
    raise ValueError("P31 requires train mode, not update-canary mode")
if CANON_P29_FULL_TRAIN and not CANON_P28_G6_UPDATE:
  raise ValueError("P29 full train requires the attested P28 G6 update path")
if CANON_P30_OPT_STATE_OFFLOAD and not CANON_P29_FULL_TRAIN:
  raise ValueError("P30 optimizer offload requires the P29 full-train path")
if (
    CANON_P28_G3_ONLY
    or CANON_P28_G4_ONLY
    or CANON_P28_G5_ONLY
    or CANON_P28_G5C_ONLY
    or (CANON_P28_G6_UPDATE and not CANON_ALIGNMENT_TRAIN_MODE)
):
  expected_p28_geometry = {
      "batch_size": (args.batch_size, 4),
      "mini_batch_size": (args.mini_batch_size, 4),
      "num_generations": (args.num_generations, 2),
      "max_prompt_length": (args.max_prompt_length, 2048),
      "max_response_length": (args.max_response_length, 64),
      "max_concurrency": (args.max_concurrency, 2),
  }
  wrong_p28_geometry = {
      name: got
      for name, (got, expected) in expected_p28_geometry.items()
      if got != expected
  }
  if wrong_p28_geometry:
    raise ValueError(f"P28 frozen geometry mismatch: {wrong_p28_geometry}")

TRAIN_FRACTION = 1.0
SEED = args.seed

# ====== Sharding ======
# Single shared mesh across actor / reference / rollout. The signed local path
# remains pure TP. The default-off P32 path uses a distinct DP axis so model
# and optimizer leaves remain replicated over data parallel ranks.
if args.mesh_dp is not None:
  mesh_tp = args.mesh_tp or (jax.device_count() // args.mesh_dp)
  SHARED_MESH_SHAPE = (args.mesh_dp, mesh_tp)
  SHARED_MESH_AXIS_NAMES = ("dp", "tp")
else:
  SHARED_MESH_SHAPE = (1, jax.device_count())
  SHARED_MESH_AXIS_NAMES = ("fsdp", "tp")

# ====== GRPO ======
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_GENERATIONS = args.num_generations

# vLLM (if used). Concurrent sequence count and batched-token budget for the
# rollout engine. Set to roughly twice ``max_concurrency`` so the rollout has
# some headroom without provisioning a huge unused KV-cache pool — on a
# shared trainer+rollout mesh that KV-cache pool consumes HBM that the
# trainer needs at peak (logits + activations + optimizer state).
VLLM_MAX_NUM_SEQS = args.vllm_max_num_seqs
if args.vllm_max_num_batched_tokens is not None:
  VLLM_MAX_BATCHED_TOKENS = args.vllm_max_num_batched_tokens
elif CANON_P32_WORKLOAD or CANON_L3:
  VLLM_MAX_BATCHED_TOKENS = 256
else:
  VLLM_MAX_BATCHED_TOKENS = VLLM_MAX_NUM_SEQS * 4 * 1024 // 8

NUM_ITERATIONS = 1
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

# ====== Training ======
# Gradient checkpointing on the transformer decoder block. Recomputes
# activations during backward pass instead of holding them in memory across
# the forward; reduces peak HBM by ~num_layers × activation_size at the cost
# of one extra forward pass per backward.
ENABLE_REMAT = True
# Flash attention on the trainer forward path. The pallas splash kernel
# computes only the causal mask kernel-side; per-batch padding has to flow
# in via per-position segment ids. The model now plumbs segment ids derived
# from the non-pad mask into splash, so left-padded prompts no longer
# contaminate real-token attention outputs.
ENABLE_FLASH_ATTENTION = True
ENABLE_MIX_PRECISION = True
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches
if CANON_P32_WORKLOAD:
  assert P32_WORKLOAD is not None
  # P57 uses one complete eight-row prompt group in every run kind.  In
  # particular, isolated evaluation retains trainer-side rescore, whose
  # caller-global row axis is sharded over DP8.  Keep this tied to the same
  # registry consumed by the renderer instead of maintaining a second
  # evaluation-only literal here.
  expected_generations = (
      p57_workloads.GENERATIONS_PER_PROMPT if CANON_P57_RUN_KIND else 8
  )
  if CANON_P57_CALIBRATION:
    # All recipes share one physical envelope. Their smaller preregistered
    # context caps are applied by the offline classifier to observed lengths,
    # so an engine-side truncation cannot make a recipe look artificially easy.
    expected_prompt_length = 16_384
    expected_response_length = 16_384
    expected_env_steps = max(
        spec.max_turns for spec in p57_workloads.RECIPES.values()
    )
    expected_temperature = 0.7
  elif p57_workload_spec is not None:
    expected_prompt_length = 4096
    expected_response_length = p57_workload_spec.context_hard_cap - 4096
    expected_env_steps = p57_workload_spec.max_turns
    expected_temperature = 0.0 if CANON_P57_EVALUATION else 0.7
  else:
    expected_prompt_length = 4096
    expected_response_length = 512 if CANON_P33_SHORT_ALIGNMENT else 2048
    expected_env_steps = (
        p57_workload_spec.max_turns
        if p57_workload_spec is not None
        else (2 if CANON_P33_SHORT_ALIGNMENT else 5)
    )
    expected_temperature = 0.0 if CANON_P57_EVALUATION else 0.7
  # The P38 serving-capture job is a pre-backward diagnostic. It consumes four
  # prompt groups so the first diagnostic batch is 4 x 8 = 32 trajectories,
  # exactly divisible by DP16. The dataset/global batch remains 32 prompts and
  # every non-P38 training profile retains mini_batch_size=32.
  expected_mini_batch_size = 4 if CANON_P38_PRECHECK_ONLY else 32
  dp_workloads.validate_frozenlake_max_concurrency(
      P32_WORKLOAD, args.max_concurrency, os.environ
  )
  expected_geometry = {
      "batch_size": (BATCH_SIZE, 32),
      "mini_batch_size": (MINI_BATCH_SIZE, expected_mini_batch_size),
      "num_batches": (NUM_BATCHES, 150),
      "num_generations": (
          NUM_GENERATIONS,
          expected_generations,
      ),
      "max_prompt_length": (MAX_PROMPT_LENGTH, expected_prompt_length),
      "max_response_length": (
          MAX_RESPONSE_LENGTH,
          expected_response_length,
      ),
      "vllm_max_num_seqs": (
          VLLM_MAX_NUM_SEQS,
          P32_WORKLOAD.local_trajectories,
      ),
      "vllm_max_num_batched_tokens": (
          VLLM_MAX_BATCHED_TOKENS,
          P32_WORKLOAD.local_m,
      ),
      "env_max_steps": (args.env_max_steps, expected_env_steps),
      "learning_rate": (args.learning_rate, 1e-6),
      "b1": (args.b1, 0.9),
      "b2": (args.b2, 0.95),
      "weight_decay": (args.weight_decay, 0.0),
      "beta": (args.beta, 0.0),
      "epsilon": (args.epsilon, 0.003),
      "epsilon_high": (args.epsilon_high, 0.005),
      "temperature": (
          args.temperature,
          expected_temperature,
      ),
      "top_p": (args.top_p, 1.0),
      "top_k": (args.top_k, 0),
      "loss_algo": (args.loss_algo, "gspo-token"),
      "advantage_estimator": (args.advantage_estimator, "rloo"),
      "sampler_is": (
          args.sampler_is,
          (
              "token"
              if CANON_P57_TIM_ARM == "is"
              else "none"
              if CANON_P57_RUN_KIND in ("train", "eval")
              else "token"
          ),
      ),
      "seed": (SEED, 42),
      "mesh": (
          SHARED_MESH_SHAPE,
          (P32_WORKLOAD.dp_size, P32_WORKLOAD.tp_size),
      ),
      "trajectory_micro": (
          args.train_trajectory_micro_batch_size,
          P32_WORKLOAD.dp_size,
      ),
  }
  wrong_geometry = {
      name: got
      for name, (got, expected) in expected_geometry.items()
      if got != expected
  }
  if wrong_geometry:
    raise ValueError(f"P32 FrozenLake geometry mismatch: {wrong_geometry}")
elif CANON_P31_CONVERGENCE:
  expected_geometry = {
      "batch_size": (BATCH_SIZE, 4),
      "mini_batch_size": (MINI_BATCH_SIZE, 4),
      "num_batches": (NUM_BATCHES, 150),
      "num_generations": (NUM_GENERATIONS, 8),
      "max_prompt_length": (MAX_PROMPT_LENGTH, 4096),
      "max_response_length": (MAX_RESPONSE_LENGTH, 2048),
      "max_concurrency": (args.max_concurrency, 8),
      "vllm_max_num_seqs": (VLLM_MAX_NUM_SEQS, 16),
      "env_max_steps": (args.env_max_steps, 5),
      "num_test_batches": (args.num_test_batches, 25),
      "eval_every_n_steps": (args.eval_every_n_steps, 25),
      "learning_rate": (args.learning_rate, 1e-6),
      "b1": (args.b1, 0.9),
      "b2": (args.b2, 0.95),
      "weight_decay": (args.weight_decay, 0.0),
      "beta": (args.beta, 0.0),
      "epsilon": (args.epsilon, 0.003),
      "epsilon_high": (args.epsilon_high, 0.005),
      "temperature": (args.temperature, 0.7),
      "top_p": (args.top_p, 1.0),
      "top_k": (args.top_k, 0),
      "loss_algo": (args.loss_algo, "gspo-token"),
      "advantage_estimator": (args.advantage_estimator, "rloo"),
  }
  wrong_geometry = {
      name: got
      for name, (got, expected) in expected_geometry.items()
      if got != expected
  }
  if wrong_geometry:
    raise ValueError(f"P31 frozen geometry mismatch: {wrong_geometry}")
  if os.getenv("CANON_P31_ENABLE_EVAL", "") not in ("0", "1"):
    raise ValueError("P31 requires explicit CANON_P31_ENABLE_EVAL=0/1")
elif CANON_P27:
  expected_geometry = {
      "batch_size": (BATCH_SIZE, 4),
      "mini_batch_size": (MINI_BATCH_SIZE, 4),
      "num_generations": (NUM_GENERATIONS, 2),
      "max_prompt_length": (MAX_PROMPT_LENGTH, 2048),
      "max_response_length": (MAX_RESPONSE_LENGTH, 64),
      "max_concurrency": (args.max_concurrency, 2),
  }
  wrong_geometry = {
      name: got
      for name, (got, expected) in expected_geometry.items()
      if got != expected
  }
  if wrong_geometry:
    raise ValueError(f"P27 frozen geometry mismatch: {wrong_geometry}")
TRAJECTORY_MINI_BATCH_SIZE = (
    MINI_BATCH_SIZE * NUM_GENERATIONS
    if CANON_P32_WORKLOAD or CANON_P27
    else None
)
_P27_TRAJECTORY_MICRO_RAW = os.getenv("CANON_P27_TRAJECTORY_MICRO", "")
if _P27_TRAJECTORY_MICRO_RAW and not CANON_P27:
  raise ValueError("CANON_P27_TRAJECTORY_MICRO requires CANON_FROZENLAKE_P27=1")
if CANON_P32_WORKLOAD:
  TRAIN_TRAJECTORY_MICRO_BATCH_SIZE = args.train_trajectory_micro_batch_size
  if CANON_P38_PRECHECK_ONLY:
    p38_trajectories = MINI_BATCH_SIZE * NUM_GENERATIONS
    p38_units = BATCH_SIZE // MINI_BATCH_SIZE
    if (
        P32_WORKLOAD is None
        or P32_WORKLOAD.name != "frozenlake"
        or p38_trajectories != 32
        or p38_units != 8
        or p38_units * p38_trajectories != 256
        or p38_trajectories % P32_WORKLOAD.dp_size
    ):
      raise ValueError(
          "P38 diagnostic requires eight 4-prompt units x 8 generations "
          "= 256 covered trajectories, with every 32-trajectory unit "
          "divisible by FrozenLake DP16"
      )
    print(
        "[CANON_P38] DIAGNOSTIC_BATCH_CONTRACT "
        f"global_prompts={BATCH_SIZE} unit_prompts={MINI_BATCH_SIZE} "
        f"units={p38_units} generations={NUM_GENERATIONS} "
        f"unit_trajectories={p38_trajectories} "
        f"covered_trajectories={p38_units * p38_trajectories} "
        f"dp={P32_WORKLOAD.dp_size} verdict=PASS",
        flush=True,
    )
elif CANON_P27:
  try:
    TRAIN_TRAJECTORY_MICRO_BATCH_SIZE = int(
        _P27_TRAJECTORY_MICRO_RAW or "2"
    )
  except ValueError as exc:
    raise ValueError("CANON_P27_TRAJECTORY_MICRO must be 1 or 2") from exc
  if TRAIN_TRAJECTORY_MICRO_BATCH_SIZE not in (1, 2):
    raise ValueError("CANON_P27_TRAJECTORY_MICRO must be 1 or 2")
else:
  TRAIN_TRAJECTORY_MICRO_BATCH_SIZE = None
# Held-out eval pool size in batches. The frozenlake test set ships with 100
# prompts; NUM_TEST_BATCHES * BATCH_SIZE should be >= 100 to cover one full
# pass per eval. With the default BATCH_SIZE=64, NUM_TEST_BATCHES=2 is
# sufficient. Eval wall-time scales linearly with NUM_TEST_BATCHES *
# BATCH_SIZE * num_generations.
NUM_TEST_BATCHES = args.num_test_batches

EVAL_EVERY_N_STEPS = args.eval_every_n_steps
NUM_EPOCHS = 3
# P21.3 L3 is a one-batch, no-update release gate, not a convergence run.
# Keeping this decision inside the default-off L3 branch prevents the three
# dataset epochs from silently producing three alignment records.
if CANON_P57_RUN_KIND:
  try:
    MAX_STEPS = int(os.environ["CANON_P57_EXPECTED_UPDATES"])
  except (KeyError, ValueError) as exc:
    raise ValueError("P57 expected-update horizon must be an integer") from exc
  if MAX_STEPS <= 0 or args.max_steps != MAX_STEPS:
    raise ValueError(
        "P57 command/horizon mismatch: "
        f"signed={MAX_STEPS} command={args.max_steps}"
    )
  if CANON_P57_RUN_KIND == "train":
    try:
      P57_STOP_AFTER_STEP = int(
          os.environ.get("CANON_P57_STOP_AFTER_STEP", str(MAX_STEPS))
      )
    except ValueError as exc:
      raise ValueError("P57 segment stop must be an integer") from exc
    if (
        P57_STOP_AFTER_STEP
        not in (50, 100, 150, 200, 250, 300, 350, 400, 450)
        or P57_STOP_AFTER_STEP > MAX_STEPS
    ):
      raise ValueError(
          "P57 segment stop must be a registered 50-step boundary within "
          "the signed horizon"
      )
  else:
    P57_STOP_AFTER_STEP = MAX_STEPS
elif CANON_P32_WORKLOAD:
  MAX_STEPS = dp_workloads.requested_max_steps(P32_WORKLOAD)
  if args.max_steps != MAX_STEPS:
    raise ValueError(
        "P33 FrozenLake --max_steps does not match CANON_P33_RUN_STAGE: "
        f"expected {MAX_STEPS}, found {args.max_steps}"
    )
elif CANON_P31_CONVERGENCE:
  try:
    MAX_STEPS = int(os.getenv("CANON_P31_MAX_STEPS", "450"))
  except ValueError as exc:
    raise ValueError("CANON_P31_MAX_STEPS must be an integer") from exc
  if MAX_STEPS < 1 or MAX_STEPS > 450:
    raise ValueError("CANON_P31_MAX_STEPS must be in [1, 450]")
else:
  MAX_STEPS = (
      NUM_BATCHES
      if CANON_P27
      else 1
      if CANON_L3
      else int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
  )
if not CANON_P57_RUN_KIND:
  P57_STOP_AFTER_STEP = MAX_STEPS

MAX_CONCURRENCY = args.max_concurrency
OFF_POLICY_STEPS = 0
MODEL_DTYPE = jnp.bfloat16

LEARNING_RATE = args.learning_rate
B1 = args.b1
B2 = args.b2
WEIGHT_DECAY = args.weight_decay
# Linear warmup over WARMUP_STEPS steps before the LR schedule begins decaying.
# 0 means start at the peak LR from step 1; this is the typical setting for
# fine-tuning RL from an already-pretrained policy. Set to a positive integer
# (e.g. ``int(0.05 * MAX_STEPS)``) only if you observe early-training
# instability from full-LR updates against a stale reference.
WARMUP_STEPS = 0
# Global-norm gradient clip. The asymmetric ratio clip and (optional) truncated
# importance-sampling correction already bound individual per-token
# contributions, so an additional tight global clip is unnecessary. The high
# threshold here effectively disables clipping while keeping a safety net
# against numerical explosions; lower it (e.g. ``1.0``) if a particular
# recipe exhibits unstable grad norms.
MAX_GRAD_NORM = 100.0

# ====== Checkpoint saving ======
P45_CHECKPOINT = frozenlake_checkpoint.from_env(os.environ)
if _P32_WORKLOAD_NAME == "frozenlake-dp8-tp8":
  frozenlake_checkpoint.require_p45(P45_CHECKPOINT, os.environ)
elif P45_CHECKPOINT.enabled:
  raise ValueError(
      "the GCS FrozenLake checkpoint contract is isolated to P45 DP8xTP8"
  )
SAVE_INTERVAL_STEPS = P45_CHECKPOINT.interval or 10**9
MAX_TO_KEEP = P45_CHECKPOINT.max_to_keep or 1

if P45_CHECKPOINT.enabled:
  P45_CHECKPOINT_CONTRACT = frozenlake_checkpoint.build_contract(
      P45_CHECKPOINT,
      {
          "source_commit": os.getenv("CANON_EXPECT_COMMIT", ""),
          "profile": os.getenv("CANON_PROFILE", ""),
          "workload": _P32_WORKLOAD_NAME,
          "model_version": "Qwen/Qwen3-8B",
          "model_dir_name": os.getenv("CANON_MODEL_DIR_NAME", ""),
          "mesh_dp": SHARED_MESH_SHAPE[0],
          "mesh_tp": SHARED_MESH_SHAPE[1],
          "batch_size": BATCH_SIZE,
          "mini_batch_size": MINI_BATCH_SIZE,
          "trajectory_mini_batch_size": TRAJECTORY_MINI_BATCH_SIZE,
          "train_trajectory_micro_batch_size": (
              TRAIN_TRAJECTORY_MICRO_BATCH_SIZE
          ),
          "num_generations": NUM_GENERATIONS,
          "num_iterations": NUM_ITERATIONS,
          "max_prompt_length": MAX_PROMPT_LENGTH,
          "max_response_length": MAX_RESPONSE_LENGTH,
          "max_concurrency": MAX_CONCURRENCY,
          "env_max_steps": args.env_max_steps,
          "learning_rate": LEARNING_RATE,
          "b1": B1,
          "b2": B2,
          "weight_decay": WEIGHT_DECAY,
          "beta": BETA,
          "epsilon": EPSILON,
          "epsilon_high": EPSILON_HIGH,
          "loss_algo": args.loss_algo,
          "loss_agg_mode": args.loss_agg_mode,
          "kl_loss_mode": args.kl_loss_mode,
          "advantage_estimator": args.advantage_estimator,
          "seed": SEED,
          "shuffle_data": bool(args.shuffle_data),
          "eval_enabled": CANON_P33_ENABLE_EVAL,
          "eval_every_n_steps": EVAL_EVERY_N_STEPS,
          "max_steps": MAX_STEPS,
          "optimizer_placement": CANON_OPTIMIZER_PLACEMENT,
          "p57_tim_arm": os.getenv("CANON_P57_TIM_ARM", ""),
          "p57_fixed_lm_head": os.getenv("CANON_P38_FIXED_LM_HEAD", "0"),
          "p57_workload_candidate": CANON_P57_WORKLOAD_CANDIDATE,
          "p57_data_split": CANON_P57_DATA_SPLIT,
      },
  )
  os.environ["CANON_CHECKPOINT_CONTRACT_JSON"] = (
      frozenlake_checkpoint.contract_json(P45_CHECKPOINT_CONTRACT)
  )
else:
  P45_CHECKPOINT_CONTRACT = {}

# ====== Rollout ======
ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vllm")  # "vanilla" | "vllm"
if CANON_L3:
  required_l3_env = {
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ENGINE_MODULE_C": "1",
      "CANON_RPA_VJP2": "1",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
  }
  bad_l3_env = {
      key: os.getenv(key)
      for key, expected in required_l3_env.items()
      if os.getenv(key) != expected
  }
  if bad_l3_env:
    raise ValueError(
        "canonical FrozenLake L3 environment is incomplete: "
        f"{bad_l3_env}"
    )
  alignment_modes = {
      "gate-only": os.getenv("CANON_ALIGNMENT_GATE_ONLY") == "1",
      "update-canary": (
          os.getenv("CANON_ALIGNMENT_UPDATE_CANARY") == "1"
      ),
      "train": os.getenv("CANON_ALIGNMENT_TRAIN") == "1",
  }
  active_modes = [name for name, active in alignment_modes.items() if active]
  if CANON_ALIGNMENT_TRAIN_MODE:
    if active_modes != ["train"]:
      raise ValueError(
          "canonical training workload requires alignment train mode, got "
          f"{active_modes}"
      )
  elif CANON_P27:
    if active_modes not in (["gate-only"], ["update-canary"]):
      raise ValueError(
          "P27 bounded gates require gate-only or update-canary mode, got "
          f"{active_modes}"
      )
  elif active_modes != ["gate-only"]:
    raise ValueError(
        "legacy canonical FrozenLake L3 requires gate-only mode, got "
        f"{active_modes}"
    )
  if ROLLOUT_ENGINE != "vllm":
    raise ValueError("canonical FrozenLake L3 requires ROLLOUT_ENGINE=vllm")
  # The adapter maps the outer training batch with ``jax.lax.map`` and invokes
  # the real engine ``model_fn`` once per sequence.  VJP2's static sequence
  # capacity therefore describes one model_fn call, not the outer batch size.
  expected_vjp2_max_seqs = 1
  if os.getenv("CANON_VJP2_MAX_SEQS") != str(expected_vjp2_max_seqs):
    raise ValueError(
        "canonical FrozenLake L3 maps the outer batch one sequence per "
        "engine model_fn call and requires CANON_VJP2_MAX_SEQS=1: "
        f"expected {expected_vjp2_max_seqs}, got "
        f"{os.getenv('CANON_VJP2_MAX_SEQS')!r}"
    )
  if os.getenv("CANON_PALLAS_LOGSOFTMAX") not in ("0", "1"):
    raise ValueError(
        "canonical FrozenLake L3 requires an explicit "
        "CANON_PALLAS_LOGSOFTMAX=0/1 causal arm"
    )

# ====== Paths ======
MODEL_VERSION = "Qwen/Qwen3-8B"
MODEL_DOWNLOAD_DIR = os.getenv("MODEL_DOWNLOAD_DIR", "/tmp/models/Qwen3-8B")
DATA_DIR = os.getenv("FROZENLAKE_DATA_DIR", "/tmp/data/frozenlake")
P57_BASE_DATA_DIR = DATA_DIR

# P45 uses a stable GCS campaign path. Other FrozenLake carriers retain the
# historical checkpoint-disabled default.
CKPT_DIR = P45_CHECKPOINT.directory
TB_LOG_DIR = "/tmp/tunix-tb/frozenlake"

if CKPT_DIR:
  checkpointing_options = checkpoint_options_lib.TunixCheckpointingOptions(
      save_decision_policy=(
          ocp.training.save_decision_policies.FixedIntervalPolicy(
              SAVE_INTERVAL_STEPS
          )
      ),
      preservation_policy=frozenlake_checkpoint.build_preservation_policy(
          P45_CHECKPOINT
      ),
      save_on_close=False,
  )
  checkpoint_probe = checkpoint_manager_lib.CheckpointManager(
      root_directory=os.path.join(CKPT_DIR, "actor"),
      options=checkpointing_options,
  )
  latest_checkpoint_step = checkpoint_probe.latest_step()
  checkpoint_probe.close()
  frozenlake_checkpoint.validate_latest(
      P45_CHECKPOINT, latest_checkpoint_step
  )
  print(
      "[P45.CHECKPOINT] PREFLIGHT "
      f"mode={P45_CHECKPOINT.mode} root={CKPT_DIR} "
      f"latest={latest_checkpoint_step if latest_checkpoint_step is not None else 'none'} "
      f"interval={SAVE_INTERVAL_STEPS} max_to_keep={MAX_TO_KEEP} "
      f"milestone_interval={P45_CHECKPOINT.milestone_interval}",
      flush=True,
  )
else:
  checkpointing_options = None


# ====== Build the single shared mesh ======
if jax.device_count() < math.prod(SHARED_MESH_SHAPE):
  raise ValueError(
      f"Expected at least {math.prod(SHARED_MESH_SHAPE)} devices for mesh "
      f"{SHARED_MESH_SHAPE}, got {jax.device_count()}."
  )

if CANON_P32_WORKLOAD:
  assert P32_WORKLOAD is not None
  if (
      args.mesh_dp != P32_WORKLOAD.dp_size
      or args.mesh_tp != P32_WORKLOAD.tp_size
  ):
    raise ValueError(
        "canonical FrozenLake DP workload requires "
        f"--mesh_dp={P32_WORKLOAD.dp_size} "
        f"--mesh_tp={P32_WORKLOAD.tp_size}"
    )
  shared_mesh = dp_workloads.create_mesh(jax.devices(), P32_WORKLOAD)
else:
  shared_device_list = jax._src.mesh_utils.create_device_mesh(
      SHARED_MESH_SHAPE, jax.devices()[: math.prod(SHARED_MESH_SHAPE)]
  )
  shared_mesh = jax.sharding.Mesh(
      shared_device_list,
      axis_names=SHARED_MESH_AXIS_NAMES,
      axis_types=(jax.sharding.AxisType.Auto,) * len(SHARED_MESH_SHAPE),
  )
print(f"shared_mesh.devices.shape={shared_mesh.devices.shape}")

# ====== Data ======
import pandas as pd
import datasets as datasets_lib
import transformers

try:
  from google.cloud import storage  # noqa: F401  (ensures gcsfs deps load on GCS)
except Exception:
  pass
import fsspec

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

if CANON_P57_WORKLOAD_CANDIDATE:
  DATA_DIR = os.path.join(
      DATA_DIR,
      "p57",
      CANON_P57_WORKLOAD_CANDIDATE,
      CANON_P57_DATA_SPLIT,
  )
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.parquet")
P57_DATASET_ATTESTATION: dict[str, str] = {}
P57_CALIBRATION_ATTESTATION: dict[str, str] = {}


def create_datasets(
    train_ds_path: str = TRAIN_DATA_PATH,
    test_ds_path: str = TEST_DATA_PATH,
):
  global P57_DATASET_ATTESTATION
  data_dir = os.path.dirname(train_ds_path)
  os.makedirs(data_dir, exist_ok=True)
  if not os.path.exists(train_ds_path) or not os.path.exists(test_ds_path):
    if CANON_P57_WORKLOAD_CANDIDATE:
      train_data, test_data = p57_workloads.materialize_dataset_pair(
          CANON_P57_WORKLOAD_CANDIDATE,
          CANON_P57_DATA_SPLIT,
          train_count=10_000,
          eval_count=100,
      )
    else:
      train_seeds, train_sizes, train_ps = frozenlake_data.generate_dataset_parameters(
          10000, random_seed=42
      )
      train_data = [
          frozenlake_data.get_frozenlake_dict(s, sz, p)
          for s, sz, p in zip(train_seeds, train_sizes, train_ps)
      ]
      test_seeds, test_sizes, test_ps = frozenlake_data.generate_dataset_parameters(
          100, random_seed=123
      )
      test_data = [
          frozenlake_data.get_frozenlake_dict(s, sz, p)
          for s, sz, p in zip(test_seeds, test_sizes, test_ps)
      ]
    frozenlake_data.save_dataset(train_data, train_ds_path)
    frozenlake_data.save_dataset(test_data, test_ds_path)

  with fsspec.open(train_ds_path, "rb") as train_f, fsspec.open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_parquet(train_f)
    test_df = pd.read_parquet(test_f)

  if CANON_P57_WORKLOAD_CANDIDATE:
    P57_DATASET_ATTESTATION = {
        "train_sha256": p57_workloads.attest_records(
            train_df.to_dict("records"),
            CANON_P57_WORKLOAD_CANDIDATE,
            CANON_P57_DATA_SPLIT,
            "train",
            expected_count=10_000,
        ),
        "eval_sha256": p57_workloads.attest_records(
            test_df.to_dict("records"),
            CANON_P57_WORKLOAD_CANDIDATE,
            CANON_P57_DATA_SPLIT,
            "eval",
            expected_count=100,
        ),
    }
    print(
        "[P57.DATASET] MATERIALIZED_PASS "
        f"candidate={CANON_P57_WORKLOAD_CANDIDATE} "
        f"split={CANON_P57_DATA_SPLIT} train_rows=10000 eval_rows=100 "
        f"train_sha256={P57_DATASET_ATTESTATION['train_sha256']} "
        f"eval_sha256={P57_DATASET_ATTESTATION['eval_sha256']}",
        flush=True,
    )
  elif CANON_P57_RUN_KIND:
    P57_DATASET_ATTESTATION = {
        "train_sha256": p57_workloads.attest_p45_records(
            train_df.to_dict("records"), "train", expected_count=10_000
        ),
        "eval_sha256": p57_workloads.attest_p45_records(
            test_df.to_dict("records"), "eval", expected_count=100
        ),
    }
    print(
        "[P57.DATASET] MATERIALIZED_PASS "
        "candidate=p45 split=legacy train_rows=10000 eval_rows=100 "
        f"train_sha256={P57_DATASET_ATTESTATION['train_sha256']} "
        f"eval_sha256={P57_DATASET_ATTESTATION['eval_sha256']}",
        flush=True,
    )

  train_ds = grain.MapDataset.source(
      frozenlake_data.add_empty_prompt_column(Dataset.from_pandas(train_df))
  )
  test_ds = grain.MapDataset.source(
      frozenlake_data.add_empty_prompt_column(Dataset.from_pandas(test_df))
  )
  if args.shuffle_data:
    train_ds = train_ds.shuffle(SEED)
    test_ds = test_ds.shuffle(SEED)
  return train_ds, test_ds


def create_calibration_datasets():
  """Loads the three frozen held-out inventories without creating train data."""
  global P57_CALIBRATION_ATTESTATION
  result = {}
  for recipe_name in CANON_P57_CALIBRATION_RECIPES:
    eval_path = os.path.join(
        P57_BASE_DATA_DIR, "p57", recipe_name, "calibration", "test.parquet"
    )
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    if not os.path.exists(eval_path):
      rows = p57_workloads.materialize_records(
          recipe_name, "calibration", "eval", 100
      )
      frozenlake_data.save_dataset(rows, eval_path)
    with fsspec.open(eval_path, "rb") as eval_file:
      frame = pd.read_parquet(eval_file)
    rows = frame.to_dict("records")
    dataset_sha = p57_workloads.attest_records(
        rows,
        recipe_name,
        "calibration",
        "eval",
        expected_count=100,
    )
    P57_CALIBRATION_ATTESTATION[recipe_name] = dataset_sha
    result[recipe_name] = grain.MapDataset.source(
        frozenlake_data.add_empty_prompt_column(Dataset.from_pandas(frame))
    )
    print(
        "[P57.CALIBRATION.DATASET] MATERIALIZED_PASS "
        f"recipe={recipe_name} rows=100 eval_sha256={dataset_sha}",
        flush=True,
    )
  return result


tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
# Disable Qwen3 thinking mode. The agent prompt already requests explicit
# step-by-step reasoning; with thinking enabled the model writes hundreds of
# ``<think>...</think>`` tokens per turn and exhausts the response budget
# before producing an action.
chat_parser = parser.QwenChatTemplateParser(tokenizer, enable_thinking=False)

if CANON_CONTRACT_ONLY or CANON_A3_ONLY or CANON_P38_FROZENLAKE_REPLAY:
  # A1b/A2 inventory needs the real model and rollout runner, not an RL batch.
  # Skipping dataset I/O keeps this preflight independent of FrozenLake data.
  train_dataset = test_dataset = None
  calibration_datasets = {}
  print("[CANON_L3] contract-only: dataset I/O skipped", flush=True)
elif CANON_P57_CALIBRATION:
  train_dataset = test_dataset = None
  calibration_datasets = create_calibration_datasets()
  calibration_datasets = {
      recipe_name: data_lib.post_init_dataset(
          dataset,
          tokenizer,
          batch_size=BATCH_SIZE,
          num_batches=NUM_TEST_BATCHES,
          max_prompt_length=MAX_PROMPT_LENGTH,
      )[0]
      for recipe_name, dataset in calibration_datasets.items()
  }
else:
  calibration_datasets = {}
  train_dataset, test_dataset = create_datasets()
  if CANON_P31_CONVERGENCE or CANON_P32_WORKLOAD:
    train_rows = len(train_dataset)
    test_rows = len(test_dataset)
    selected_train_rows = min(NUM_BATCHES * BATCH_SIZE, train_rows)
    available_updates = (
        selected_train_rows * NUM_EPOCHS // BATCH_SIZE
    )
    if (train_rows, test_rows) != (10_000, 100):
      raise ValueError(
          "P31 dataset contract requires train/test=10000/100, got "
          f"{train_rows}/{test_rows}"
      )
    if available_updates != 450 or MAX_STEPS > available_updates:
      raise ValueError(
          "P31 update capacity mismatch: "
          f"available={available_updates} requested={MAX_STEPS}"
      )
    print(
        "[CANON_FROZENLAKE_P31] DATA_CONTRACT "
        f"train_rows={train_rows} test_rows={test_rows} "
        f"selected_train_rows={selected_train_rows} epochs={NUM_EPOCHS} "
        f"available_updates={available_updates} requested_updates={MAX_STEPS}",
        flush=True,
    )
  train_dataset, val_dataset = data_lib.post_init_dataset(
      train_dataset,
      tokenizer,
      batch_size=BATCH_SIZE,
      num_batches=NUM_BATCHES,
      max_prompt_length=MAX_PROMPT_LENGTH,
      fraction=TRAIN_FRACTION,
      num_epochs=NUM_EPOCHS,
  )
  if (
      not CANON_P32_WORKLOAD
      or CANON_P33_ENABLE_EVAL
      or CANON_P57_EVALUATION
  ):
    test_dataset, _ = data_lib.post_init_dataset(
        test_dataset,
        tokenizer,
        batch_size=BATCH_SIZE,
        num_batches=NUM_TEST_BATCHES,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )
  else:
    # Keep the existing default-off P33 arm independent of evaluation cost.
    test_dataset = None

show_hbm_usage = sft_utils.show_hbm_usage
show_hbm_usage("Done with loading datasets")

# ====== Download + load model ======
# Download safetensors from HF if not present locally.
if not os.path.isdir(MODEL_DOWNLOAD_DIR) or not any(
    f.endswith(".safetensors") for f in os.listdir(MODEL_DOWNLOAD_DIR)
):
  os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
  oss_utils.hf_pipeline(MODEL_VERSION, MODEL_DOWNLOAD_DIR)

config = model_lib.ModelConfig.qwen3_8b()
if CANON_P32_WORKLOAD:
  dp_workloads.configure_replicated_parameter_sharding(config)
if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.DECODER
if ENABLE_FLASH_ATTENTION:
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
if ENABLE_MIX_PRECISION:
  config.dtype = jnp.bfloat16

# Reference: keep bf16 storage (frozen, never updated -> HBM savings safe).
qwen_ref = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, shared_mesh, dtype=MODEL_DTYPE
)
show_hbm_usage("after loading qwen_ref")

# Actor: storage MUST be fp32. At LR=1e-6 with typical weight magnitudes
# ~1e-2, Adam updates are ~1e-6, well below bf16 ULP (~7.8e-5). bf16 storage
# silently rounds every update to zero in optax.apply_updates, so the policy
# never moves. Forward compute can still be bf16 via config.dtype.
qwen_actor = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, shared_mesh, dtype=jnp.float32
)
show_hbm_usage("after loading qwen_actor")

# ====== Checkpoint + metrics + optimizer ======
wandb_config = vars(args)
wandb_config.update({
    "WARMUP_STEPS": WARMUP_STEPS,
    "num_steps": MAX_STEPS,
    "rollout_engine": ROLLOUT_ENGINE,
    "model_id": MODEL_VERSION,
    "mesh_shape": SHARED_MESH_SHAPE,
})
metrics_logging_options = None
if CANON_P29_FULL_TRAIN or not (
    CANON_L3 or CANON_CONTRACT_ONLY or CANON_A3_ONLY
):
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=os.getenv("CANON_P29_LOG_DIR", TB_LOG_DIR),
      project_name=os.getenv(
          "CANON_WANDB_PROJECT"
          if CANON_P32_WORKLOAD
          else "CANON_P29_WANDB_PROJECT",
          "tunix-frozenlake",
      ),
      run_name=os.getenv(
          "CANON_WANDB_RUN_NAME"
          if CANON_P32_WORKLOAD
          else "CANON_P29_WANDB_RUN_NAME",
          "",
      ),
      flush_every_n_steps=1,
      backend_kwargs={
          "wandb": {
              "config": wandb_config,
              **(
                  {"group": os.environ["CANON_WANDB_GROUP"]}
                  if CANON_P32_WORKLOAD
                  else {}
              ),
          }
      },
  )

_FL_STATELESS_OPTIMIZER = os.getenv("FL_STATELESS_OPTIMIZER", "")
if _FL_STATELESS_OPTIMIZER not in ("", "0", "1"):
  raise ValueError("FL_STATELESS_OPTIMIZER must be absent, empty, 0, or 1")
if _FL_STATELESS_OPTIMIZER == "1" and not CANON_P38_PRECHECK_ONLY:
  raise ValueError("FL_STATELESS_OPTIMIZER=1 is restricted to P38 gate-only")
if _FL_STATELESS_OPTIMIZER == "1":
  # P38 gate-only performs zero optimizer commits.  Avoid allocating Adam
  # moments that do not fit beside Qwen3-8B and vLLM on a 32-GiB v4 chip.
  optimizer = optax.sgd(learning_rate=LEARNING_RATE)
else:
  optimizer = optax.adamw(
      learning_rate=LEARNING_RATE,
      b1=B1,
      b2=B2,
      weight_decay=WEIGHT_DECAY,
  )
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# ====== Rollout + RL cluster ======
print("Shared mesh:", shared_mesh)

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "return_logprobs": True,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": MODEL_VERSION,
    # Fraction of per-chip HBM that the rollout engine pre-allocates for KV
    # cache + model weights. On a shared trainer+rollout mesh this directly
    # competes with the trainer's peak (logits + activations + optimizer
    # state). Sized to fit the actual KV-cache need at our max_num_seqs and
    # max_seq_len rather than the vLLM default. Once vLLM-TPU gains support
    # for sleep/wake_up, this can be relaxed since the KV pool can be
    # offloaded to host RAM during train_step.
    "rollout_vllm_hbm_utilization": float(
        os.getenv("FL_VLLM_HBM_UTIL", "0.20")
    ),
    "rollout_vllm_tpu_backend_type": "jax",
    # AgenticRLLearner requires the in-process continuous-batching driver.
    # Canonical C accesses that driver's live model runner through the same
    # VllmSampler._model_runner contract as batch-inference mode.
    "rollout_vllm_server_mode": True,
    # Async scheduling adds an extra in-flight step that can race weight sync;
    # disable it under engine-disagg so each rollout completes before the next
    # train step starts.
    "rollout_vllm_async_scheduling": False,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": SHARED_MESH_SHAPE[1],
    "data_parallel_size": SHARED_MESH_SHAPE[0],
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": CANON_VLLM_ENABLE_PREFIX_CACHING,
        "dtype": "bfloat16",
        **({"seed": 0} if CANON_P57_RUN_KIND else {}),
    },
}

if CANON_P57_RUN_KIND:
  if SEED != 42:
    raise ValueError(f"P57 experiment seed drifted: {SEED} != 42")
  print(
      f"[P57.SEED] CONTRACT_PASS data_shuffle_seed={SEED} "
      "vllm_global_seed=0 per_request_seed=unsupported",
      flush=True,
  )
canon_continue_decode = os.environ.get("CANON_CONTINUE_DECODE", "")
if canon_continue_decode:
  if (
      not canon_continue_decode.isdigit()
      or not 1 <= int(canon_continue_decode) <= 64
  ):
    raise ValueError(
        "CANON_CONTINUE_DECODE must be an integer in [1, 64], got "
        f"{canon_continue_decode!r}"
    )
  vllm_rollout_dict["rollout_vllm_additional_config"] = {
      "enable_continue_decode": True,
      "max_decode_steps": int(canon_continue_decode),
  }
  print(
      "[P57.CONTINUE_DECODE] on-device decode loop enabled "
      f"max_decode_steps={canon_continue_decode}",
      flush=True,
  )
print(
    "[P3_APC_CONFIG] "
    f"enabled={int(CANON_VLLM_ENABLE_PREFIX_CACHING)} "
    "workload=frozenlake reader=train_frozenlake_qwen3",
    flush=True,
)

if ROLLOUT_ENGINE == "vllm":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: shared_mesh,
        rl_cluster_lib.Role.REFERENCE: shared_mesh,
        rl_cluster_lib.Role.ROLLOUT: shared_mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    # Keep actor weights resident on device. With ``delete_dst_buffers=True``
    # the vLLM weight-sync path frees old buffers before re-allocating, so the
    # host-offload workaround previously used to relieve HBM pressure during
    # sync is no longer necessary on this hardware.
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=P57_STOP_AFTER_STEP,
        mini_batch_size=MINI_BATCH_SIZE,
        # Memory-shaping micro-batch for forward+backward. The optimizer sees
        # ``mini_batch_size`` sequences per gradient update; under the hood the
        # trainer iterates the merged rollout buffer in chunks of
        # ``train_micro_batch_size`` and accumulates gradients across
        # ``mini_batch_size // train_micro_batch_size`` chunks before stepping.
        # Reducing this lowers peak HBM (the lm_head logits tensor
        # ``[micro_batch * num_gen * seq_len, vocab/TP]`` in fp32 is the
        # dominant allocation on small TPU slices) at the cost of more
        # micro-step launches per optimizer update. It does NOT change the
        # effective optimizer batch size or training dynamics.
        # Keep the execution micro-batch tied to the requested mini-batch.
        # Stock/release remains 4; the default-off C0 compile discriminator
        # sets both to 1 so one legal two-generation GRPO pair is one JIT call.
        train_micro_batch_size=MINI_BATCH_SIZE,
        trajectory_mini_batch_size=TRAJECTORY_MINI_BATCH_SIZE,
        train_trajectory_micro_batch_size=(
            TRAIN_TRAJECTORY_MICRO_BATCH_SIZE
        ),
        compute_logps_micro_batch_size=MINI_BATCH_SIZE,
        optimizer_offload=CANON_P30_OPT_STATE_OFFLOAD,
        data_sharding_axis=("dp",) if CANON_P32_WORKLOAD else ("fsdp",),
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        checkpoint_restore_step=(
            int(os.environ["CANON_P57_EVAL_CHECKPOINT_STEP"])
            if CANON_P57_EVALUATION
            and int(os.environ["CANON_P57_EVAL_CHECKPOINT_STEP"]) > 0
            else None
        ),
        precomputed_gradient_checkpointing_contract=(
            frozenlake_checkpoint.SCHEMA if P45_CHECKPOINT.enabled else None
        ),
    ),
    rollout_config=rollout_engine_config,
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    system_prompt="",
    max_concurrency=MAX_CONCURRENCY,
    off_policy_steps=OFF_POLICY_STEPS,
    loss_agg_mode=args.loss_agg_mode,
    kl_loss_mode=args.kl_loss_mode,
    loss_algo=args.loss_algo,
    # The default preserves the existing FrozenLake token-TIS recipe. P57
    # explicitly passes ``none`` to both arms so rollout S_decode remains the
    # old-policy denominator and no TIM-aware correction weights enter loss.
    sampler_is=None if args.sampler_is == "none" else args.sampler_is,
    sampler_is_threshold=2.0,
    use_rollout_logps=True,
    advantage_estimator=args.advantage_estimator,
)

perf_config = None
canon_perf_trace_dir = os.environ.get("CANON_PERF_TRACE_DIR", "")
if canon_perf_trace_dir:
  # Official tunix.perf v2 semantic spans.  Empty/unset remains the existing
  # NoopTracer path; Phase3 profile runs set this only as instrumentation.
  from tunix.perf import metrics as perf_metrics_lib  # pylint: disable=g-import-not-at-top
  from tunix.perf.experimental import export as perf_export_lib  # pylint: disable=g-import-not-at-top
  from tunix.perf import profile_window as perf_profile_window  # pylint: disable=g-import-not-at-top

  perfetto_exporter = perf_export_lib.PerfMetricsExport.from_cluster_config(
      cluster_config=cluster_config,
      trace_dir=canon_perf_trace_dir,
  )
  perfetto_target_step = int(
      os.environ.get("CANON_PERF_TRACE_EXPORT_STEP", "") or "2"
  )
  perf_config = perf_metrics_lib.PerfMetricsConfig(
      custom_export_fn_v2=perf_profile_window.single_step_export_fn(
          perfetto_exporter.export_metrics,
          target_step=perfetto_target_step,
      )
  )

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen_actor,
    reference=qwen_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
    perf_config=perf_config,
)
show_hbm_usage("after RLCluster creation")
if P45_CHECKPOINT.enabled:
  restored_checkpoint_step = rl_cluster.actor_trainer.restored_global_step()
  restored_optimizer = (
      rl_cluster.actor_trainer.checkpoint_manager.last_restore_had_optimizer
  )
  restored_metadata = rl_cluster.actor_trainer._restored_custom_metadata
  if CANON_P57_EVALUATION:
    frozenlake_checkpoint.validate_p57_evaluation_restored(
        P45_CHECKPOINT,
        restored_step=restored_checkpoint_step,
        metadata=restored_metadata,
        env=os.environ,
    )
  else:
    frozenlake_checkpoint.validate_restored(
        P45_CHECKPOINT,
        restored_step=restored_checkpoint_step,
        optimizer_restored=restored_optimizer,
        metadata=restored_metadata,
        expected_contract=P45_CHECKPOINT_CONTRACT,
    )
  if rl_cluster.actor_trainer.train_steps != restored_checkpoint_step:
    raise ValueError(
        "P45 trainer/global checkpoint step mismatch: "
        f"trainer={rl_cluster.actor_trainer.train_steps} "
        f"global={restored_checkpoint_step}"
    )
  if P45_CHECKPOINT.mode == "new":
    print("[P45.CHECKPOINT] NEW_PASS latest=none", flush=True)
  else:
    print(
        "[P45.CHECKPOINT] RESTORE_PASS "
        f"step={restored_checkpoint_step} optimizer_state=1 "
        "contract_match=1",
        flush=True,
    )
  if CANON_P57_STOCK_TRAIN:
    if P57_STOP_AFTER_STEP <= restored_checkpoint_step:
      raise ValueError(
          "P57 stock segment must advance beyond the restored checkpoint: "
          f"restored={restored_checkpoint_step} stop={P57_STOP_AFTER_STEP}"
      )
    print(
        "[P57.STOCK] SEGMENT_PREFLIGHT "
        f"restored={restored_checkpoint_step} "
        f"stop_after={P57_STOP_AFTER_STEP} horizon={MAX_STEPS} "
        f"checkpoint_interval={SAVE_INTERVAL_STEPS} "
        f"max_to_keep={MAX_TO_KEEP}",
        flush=True,
    )
if CANON_P32_WORKLOAD:
  wandb_attestation = dp_workloads.require_online_wandb_run(P32_WORKLOAD)
  print(
      f"[CANON_P33_WANDB] ONLINE_RUN_PASS {wandb_attestation}",
      flush=True,
  )
if CANON_L3:
  contract = rl_cluster.rollout.canonical_engine_contract_attestation()
  print(f"[CANON_L3] engine contract admitted: {contract}", flush=True)
  if CANON_CONTRACT_ONLY:
    print("[CANON_L3] CONTRACT_ONLY_PASS", flush=True)
    raise SystemExit(0)
  if CANON_A3_ONLY:
    import p21_l30_a3_gate

    p21_l30_a3_gate.run(
        actor=qwen_actor,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
    )
    print("[CANON_L3] A3_ONLY_PASS", flush=True)
    raise SystemExit(0)
  if CANON_P28_G3_ONLY:
    rl_cluster.rollout.run_p28_segmented_forward_gate()
    print("[P28.G3] FORWARD_ONLY_PASS no_backward=1 no_optimizer=1", flush=True)
    raise SystemExit(0)
  if CANON_P28_G4_ONLY:
    layer_index = int(os.getenv("CANON_P28_G4_LAYER_INDEX", "0"))
    rl_cluster.rollout.run_p28_block_vjp_gate(layer_index=layer_index)
    print("[P28.G4] BLOCK_VJP_ONLY_PASS no_optimizer=1", flush=True)
    raise SystemExit(0)
  if CANON_P28_G5_ONLY:
    rl_cluster.rollout.run_p28_full_chain_gate()
    print(
        "[P28.G5B] CHAIN_ONLY_PASS no_loss=1 no_optimizer=1",
        flush=True,
    )
    raise SystemExit(0)
  if CANON_P3_APC_BOUNDARY_REPORT:
    if not CANON_P38_PRECHECK_ONLY:
      raise ValueError("P3 APC boundary probe requires gate-only mode")
    if os.path.exists(CANON_P3_APC_BOUNDARY_REPORT):
      raise FileExistsError(
          "refusing to overwrite P3 APC boundary report: "
          f"{CANON_P3_APC_BOUNDARY_REPORT}"
      )
    weight_attestation = rl_cluster.attest_actor_anchor_matches_engine()
    if weight_attestation.get("equal") is not True:
      raise ValueError(
          "P3 APC boundary probe requires bitwise-equal actor and engine weights"
      )
    report = rl_cluster.rollout.run_p3_apc_boundary_probe()
    report["weight_attestation"] = {
        "equal": True,
        "mapped_leaves": int(weight_attestation["mapped_leaves"]),
        "live_leaves": int(weight_attestation["live_leaves"]),
        "total_elements": int(weight_attestation["total_elements"]),
        "mismatch_indices": list(weight_attestation["mismatch_indices"]),
        "mesh_device_ids": list(weight_attestation["mesh_device_ids"]),
    }
    os.makedirs(
        os.path.dirname(CANON_P3_APC_BOUNDARY_REPORT) or ".", exist_ok=True
    )
    with open(
        CANON_P3_APC_BOUNDARY_REPORT, "x", encoding="utf-8"
    ) as report_file:
      json.dump(report, report_file, sort_keys=True, indent=2)
      report_file.write("\n")
    print(
        "[P3_APC_BOUNDARY] COMPLETE "
        f"report={CANON_P3_APC_BOUNDARY_REPORT} "
        f"cases={len(report['cases'])} backward=0 optimizer_commits=0",
        flush=True,
    )
    raise SystemExit(0)
  if CANON_P38_FROZENLAKE_REPLAY:
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

    capsule_path = os.getenv("CANON_P38_CAPSULE_INPUT", "")
    report_path = os.getenv("CANON_P38_REPLAY_REPORT", "")
    if not capsule_path or not report_path:
      raise ValueError(
          "P38 FrozenLake replay requires CANON_P38_CAPSULE_INPUT and "
          "CANON_P38_REPLAY_REPORT"
      )
    if os.path.exists(report_path):
      raise FileExistsError(
          f"refusing to overwrite P38 replay report: {report_path}"
      )
    weight_attestation = rl_cluster.attest_actor_anchor_matches_engine()
    if weight_attestation.get("equal") is not True:
      raise ValueError(
          "P38 FrozenLake replay requires bitwise-equal actor and engine weights"
      )
    report = canonical_forward.require_registered().run_p38_frozenlake_causal_replay(
        capsule_path=capsule_path,
        row_index=int(os.getenv("CANON_P38_CAPSULE_ROW_INDEX", "0")),
        temperature=TEMPERATURE,
    )
    report["weight_attestation"] = {
        "equal": True,
        "mapped_leaves": int(weight_attestation["mapped_leaves"]),
        "live_leaves": int(weight_attestation["live_leaves"]),
        "total_elements": int(weight_attestation["total_elements"]),
        "mismatch_indices": list(weight_attestation["mismatch_indices"]),
        "mesh_device_ids": list(weight_attestation["mesh_device_ids"]),
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "x", encoding="utf-8") as report_file:
      json.dump(report, report_file, sort_keys=True, indent=2)
      report_file.write("\n")
    print(
        "[CANON_P38_REPLAY] COMPLETE "
        f"report={report_path} classification={report['classification']} "
        "no_backward=1 optimizer_commits=0",
        flush=True,
    )
    raise SystemExit(0)


_metric_call_idx = 0


def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, completions, advantages, kwargs
  global _metric_call_idx
  _metric_call_idx += 1
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()
  solve_partial = (~solve_all) and (~solve_none)
  solve_ratio = (rewards > 0.1).mean()
  reward_mean = float(rewards.mean())
  reward_max = float(rewards.max())
  absl_logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx, len(rewards), float(solve_ratio), reward_mean,
      reward_max, int(solve_all), int(solve_none),
  )
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    agent_class=FrozenLakeAgent,
    agent_kwargs={"use_multistep_prompt": True},
    env_class=FrozenLakeEnv,
    env_kwargs={"max_steps": args.env_max_steps},
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
)
show_hbm_usage("after GRPOLearner creation")
if CANON_P57_STOCK_TRAIN:
  print(
      "[P57.STOCK] TRAIN_RUNTIME_PASS "
      f"regime=stock-fast arm={CANON_P57_TIM_ARM} canonical_bundle=off "
      "observer=warning-only processed_b=observer-only",
      flush=True,
  )

rollout_weight_sync = None
if P45_CHECKPOINT.mode == "resume" or CANON_P57_NO_UPDATE:
  if grpo_trainer.rl_cluster.global_steps != restored_checkpoint_step:
    raise ValueError(
        "P45 learner did not adopt the restored global step: "
        f"learner={grpo_trainer.rl_cluster.global_steps} "
        f"restored={restored_checkpoint_step}"
    )
  rollout_weight_sync = frozenlake_checkpoint.sync_rollout_for_no_update(
      grpo_trainer,
      stock_fast=P57_STOCK_FAST_ATTESTATION is not None,
  )
  if P57_STOCK_FAST_ATTESTATION is not None:
    print(
        "[P57.STOCK_FAST] ROLLOUT_SYNC_PASS "
        f"step={restored_checkpoint_step} transport=update_params "
        "exact_weight_attestation=unavailable-by-design",
        flush=True,
    )
  else:
    print(
        "[P45.CHECKPOINT] ROLLOUT_SYNC_PASS "
        f"step={restored_checkpoint_step} weights_equal=1 "
        f"reason={'evaluation' if CANON_P57_EVALUATION else 'resume'}",
        flush=True,
    )

if CANON_P32_WORKLOAD:
  if CANON_P33_ENABLE_EVAL:
    if test_dataset is None:
      raise ValueError("canonical FrozenLake evaluation dataset is missing")
    training_eval_dataset = test_dataset
    print(
        "[CANON_" "P33_EVAL] ENABLED workload=frozenlake "
        f"cadence={EVAL_EVERY_N_STEPS} held_out_rows=100 generations=8",
        flush=True,
    )
  else:
    training_eval_dataset = None
    print(
        "[CANON_P33_EVAL] DISABLED workload=frozenlake",
        flush=True,
    )
elif (
    CANON_P31_CONVERGENCE
    and os.getenv("CANON_P31_ENABLE_EVAL", "") == "1"
):
  training_eval_dataset = test_dataset
elif CANON_L3:
  training_eval_dataset = None
else:
  training_eval_dataset = test_dataset

if CANON_P57_CALIBRATION:
  output_path = os.getenv("CANON_P57_CALIBRATION_OUTPUT", "")
  if not output_path or not os.path.isabs(output_path):
    raise ValueError("P57 calibration output must be an absolute path")
  if os.path.exists(output_path):
    raise FileExistsError(
        f"refusing to overwrite P57 calibration output: {output_path}"
    )
  starting_train_steps = rl_cluster.actor_trainer.train_steps
  starting_global_steps = rl_cluster.global_steps
  results = {}
  try:
    for recipe_name in CANON_P57_CALIBRATION_RECIPES:
      spec = p57_workloads.recipe(recipe_name)
      grpo_trainer.env_kwargs["max_steps"] = spec.max_turns
      print(
          "[CANON_P57_CALIBRATION] RECIPE_START "
          f"mode={CANON_P57_CALIBRATION_MODE} recipe={recipe_name} "
          f"max_turns={spec.max_turns} context_cap={spec.context_hard_cap}",
          flush=True,
      )
      results[recipe_name] = {
          "recipe": {
              "name": spec.name,
              "min_grid_side": spec.min_grid_side,
              "max_grid_side": spec.max_grid_side,
              "max_turns": spec.max_turns,
              "context_hard_cap": spec.context_hard_cap,
              "frozen_probability": spec.frozen_probability,
              "eligible": spec.eligible,
          },
          "dataset_eval_sha256": P57_CALIBRATION_ATTESTATION[recipe_name],
          **grpo_trainer.rollout_only_evaluate(
              calibration_datasets[recipe_name], policy_step=0
          ),
      }
      print(
          "[CANON_P57_CALIBRATION] RECIPE_COMPLETE "
          f"mode={CANON_P57_CALIBRATION_MODE} recipe={recipe_name} "
          f"trajectories={results[recipe_name]['trajectories']} "
          f"wall_seconds={results[recipe_name]['wall_seconds']:.3f}",
          flush=True,
      )
    if (
        rl_cluster.actor_trainer.train_steps != starting_train_steps
        or rl_cluster.global_steps != starting_global_steps
    ):
      raise RuntimeError("P57 calibration mutated training state")
    record = {
        "schema": "p57-frozenlake-stock-rollout-calibration-v2",
        "arm": "mismatch",
        "inference_regime": CANON_P57_INFERENCE_REGIME,
        "zero_tim_off_attestation": P57_STOCK_FAST_ATTESTATION,
        "rollout_weight_sync": rollout_weight_sync,
        "fixed_lm_head": os.getenv("CANON_P38_FIXED_LM_HEAD", "0"),
        "source_commit": os.getenv("CANON_EXPECT_COMMIT", ""),
        "mode": CANON_P57_CALIBRATION_MODE,
        "temperature": TEMPERATURE,
        "generations": NUM_GENERATIONS,
        "seed": SEED,
        "recipe_order": list(CANON_P57_CALIBRATION_RECIPES),
        "physical_max_prompt_length": MAX_PROMPT_LENGTH,
        "physical_max_response_length": MAX_RESPONSE_LENGTH,
        "train_steps_before": starting_train_steps,
        "train_steps_after": rl_cluster.actor_trainer.train_steps,
        "global_steps_before": starting_global_steps,
        "global_steps_after": rl_cluster.global_steps,
        "backward_calls": 0,
        "optimizer_commits": 0,
        "checkpoint_writes": 0,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as output_file:
      json.dump(record, output_file, indent=2, sort_keys=True)
      output_file.write("\n")
    print(
        "[CANON_P57_CALIBRATION_JSON] "
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "[CANON_P57_CALIBRATION] COMPLETE "
        f"mode={CANON_P57_CALIBRATION_MODE} "
        f"inference_regime={CANON_P57_INFERENCE_REGIME} "
        f"recipes={len(CANON_P57_CALIBRATION_RECIPES)} "
        f"trajectories={sum(value['trajectories'] for value in results.values())} "
        "backward=0 optimizer_commits=0 checkpoint_writes=0",
        flush=True,
    )
  finally:
    rl_cluster.close()
  raise SystemExit(0)

if CANON_P57_EVALUATION:
  if test_dataset is None:
    raise ValueError("P57 isolated evaluation dataset is missing")
  output_path = os.getenv("CANON_P57_EVAL_OUTPUT", "")
  if not output_path or not os.path.isabs(output_path):
    raise ValueError("P57 isolated evaluation output must be an absolute path")
  if os.path.exists(output_path):
    raise FileExistsError(
        f"refusing to overwrite P57 evaluation output: {output_path}"
    )
  try:
    evaluation = grpo_trainer.evaluate_only(
        test_dataset,
        policy_step=restored_checkpoint_step,
    )
    record = {
        "schema": "p57-frozenlake-isolated-evaluation-v1",
        "arm": os.getenv("CANON_P57_TIM_ARM", ""),
        "fixed_lm_head": os.getenv("CANON_P38_FIXED_LM_HEAD", "0"),
        "source_commit": os.getenv("CANON_EXPECT_COMMIT", ""),
        "expected_updates": int(os.environ["CANON_P57_EXPECTED_UPDATES"]),
        "checkpoint_step": int(
            os.environ["CANON_P57_EVAL_CHECKPOINT_STEP"]
        ),
        "checkpoint_tag": P45_CHECKPOINT.tag,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "held_out_rows": 100,
        "workload_candidate": CANON_P57_WORKLOAD_CANDIDATE,
        "data_split": CANON_P57_DATA_SPLIT,
        "dataset_eval_sha256": P57_DATASET_ATTESTATION.get(
            "eval_sha256", ""
        ),
        **evaluation,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as output_file:
      json.dump(record, output_file, indent=2, sort_keys=True)
      output_file.write("\n")
    print(
        "[CANON_P57_EVAL_JSON] "
        + json.dumps(record, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "[CANON_P57_EVAL] COMPLETE "
        f"arm={record['arm']} step={record['policy_step']} "
        f"prompts={record['prompts']} generations={record['generations']} "
        f"rewards={record['n']} solve={record['solve']:.6f} "
        "backward=0 optimizer_commits=0 checkpoint_writes=0",
        flush=True,
    )
  finally:
    rl_cluster.close()
  raise SystemExit(0)

grpo_trainer.train(
    train_dataset,
    eval_dataset=training_eval_dataset,
)
if CANON_P57_STOCK_TRAIN:
  completed_step = int(grpo_trainer.rl_cluster.actor_trainer.train_steps)
  durable_step = (
      grpo_trainer.rl_cluster.actor_trainer.checkpoint_manager.latest_step()
  )
  if completed_step != P57_STOP_AFTER_STEP or durable_step != completed_step:
    raise RuntimeError(
        "P57 stock segment did not close on its durable boundary: "
        f"completed={completed_step} durable={durable_step} "
        f"expected={P57_STOP_AFTER_STEP}"
    )
  next_action = "complete" if completed_step == MAX_STEPS else "isolated-eval"
  print(
      "[P57.STOCK] SEGMENT_COMPLETE "
      f"step={completed_step} durable_checkpoint={durable_step} "
      f"horizon={MAX_STEPS} next_action={next_action}",
      flush=True,
  )
if CANON_P31_CONVERGENCE:
  print(
      "[CANON_FROZENLAKE_P31] TRAINING_DONE "
      f"max_steps={MAX_STEPS} trajectory_mini={TRAJECTORY_MINI_BATCH_SIZE} "
      f"trajectory_micro={TRAIN_TRAJECTORY_MICRO_BATCH_SIZE} "
      f"generations={NUM_GENERATIONS} env_max_steps={args.env_max_steps}",
      flush=True,
  )
elif CANON_P27:
  print(
      "[CANON_FROZENLAKE_P27] TRAINING_DONE "
      f"max_steps={MAX_STEPS} trajectory_mini={TRAJECTORY_MINI_BATCH_SIZE} "
      f"trajectory_micro={TRAIN_TRAJECTORY_MICRO_BATCH_SIZE}",
      flush=True,
  )
elif CANON_L3:
  print("[CANON_L3] FULL_GATE_PASS", flush=True)
