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
from orbax import checkpoint as ocp
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
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils
from tunix.cli.utils import data as data_lib
# The A1b/A2 contract and A3 adapter preflights exit before constructing the
# learner and intentionally have no FrozenLake environment dependency.  Keep
# the normal path's eager imports unchanged so a real L3 run still fails
# immediately when its environment package is absent.
_CANON_PRELEARNER_ONLY = (
    os.getenv("CANON_L3_CONTRACT_ONLY", "") == "1"
    or os.getenv("CANON_L3_A3_ONLY", "") == "1"
    or os.getenv("CANON_P28_G3_ONLY", "") == "1"
    or os.getenv("CANON_P28_G4_ONLY", "") == "1"
    or os.getenv("CANON_P28_G5_ONLY", "") == "1"
    or os.getenv("CANON_P32_DP2TP2_INIT_ONLY", "") == "1"
)
if not _CANON_PRELEARNER_ONLY:
  from examples.frozenlake.agent import FrozenLakeAgent
  from examples.frozenlake.env import FrozenLakeEnv
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
# Concurrent rollout threads. Should stay at or below the vLLM engine's
# `max_num_seqs` (default 64) plus a small backlog; pushing it much higher
# pegs the KV cache at 100% and forces chunked-prefill to interleave with
# decode, which makes the sampler's logits diverge from the trainer's
# recomputation (visible as a large `sampler_trainer/train/logp_diff_mean`)
# and noticeably degrades steady-state reward. Keep ~4x `max_num_seqs` so the
# engine has work queued without saturating the cache.
arg_parser.add_argument("--max_concurrency", type=int, default=256)
arg_parser.add_argument("--vllm_max_num_seqs", type=int, default=64)
arg_parser.add_argument("--env_max_steps", type=int, default=8)
arg_parser.add_argument("--num_test_batches", type=int, default=2)
arg_parser.add_argument("--eval_every_n_steps", type=int, default=10)
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
args, _ = arg_parser.parse_known_args()

CANON_L3 = os.getenv("CANON_FROZENLAKE_L3", "") == "1"
CANON_P27 = os.getenv("CANON_FROZENLAKE_P27", "") == "1"
CANON_CONTRACT_ONLY = os.getenv("CANON_L3_CONTRACT_ONLY", "") == "1"
CANON_A3_ONLY = os.getenv("CANON_L3_A3_ONLY", "") == "1"
CANON_P28_G3_ONLY = os.getenv("CANON_P28_G3_ONLY", "") == "1"
CANON_P28_G4_ONLY = os.getenv("CANON_P28_G4_ONLY", "") == "1"
CANON_P28_G5_ONLY = os.getenv("CANON_P28_G5_ONLY", "") == "1"
CANON_P28_G5C_ONLY = os.getenv("CANON_P28_G5C_ONLY", "") == "1"
CANON_P28_G6_UPDATE = os.getenv("CANON_P28_G6_UPDATE", "") == "1"
CANON_P29_FULL_TRAIN = os.getenv("CANON_P29_FULL_TRAIN", "") == "1"
CANON_P31_CONVERGENCE = os.getenv("CANON_P31_CONVERGENCE", "") == "1"
CANON_P32_DP2TP2 = os.getenv("CANON_P32_DP2TP2", "") == "1"
CANON_P32_DP2TP2_INIT_ONLY = (
    os.getenv("CANON_P32_DP2TP2_INIT_ONLY", "") == "1"
)
CANON_P32_DP2TP2_FORWARD_ONLY = (
    os.getenv("CANON_P32_DP2TP2_FORWARD_ONLY", "") == "1"
)
CANON_P32_DP2TP2_BACKWARD_NO_COMMIT = (
    os.getenv("CANON_P32_DP2TP2_BACKWARD_NO_COMMIT", "") == "1"
)
CANON_P32_D3A_SPLIT_OUTER_JIT = (
    os.getenv("CANON_P32_D3A_SPLIT_OUTER_JIT", "") == "1"
)
CANON_P32_D3B_SEGMENTED = (
    os.getenv("CANON_P32_D3B_SEGMENTED", "") == "1"
)
CANON_P30_OPT_STATE_OFFLOAD = (
    os.getenv("CANON_P30_OPT_STATE_OFFLOAD", "") == "1"
)
prelearner_modes = {
    "contract-only": CANON_CONTRACT_ONLY,
    "A3-only": CANON_A3_ONLY,
    "P28-G3-only": CANON_P28_G3_ONLY,
    "P28-G4-only": CANON_P28_G4_ONLY,
    "P28-G5-only": CANON_P28_G5_ONLY,
    "P32-DP2TP2-init-only": CANON_P32_DP2TP2_INIT_ONLY,
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
if CANON_P32_DP2TP2_INIT_ONLY and not CANON_P32_DP2TP2:
  raise ValueError("P32 DP2xTP2 init-only requires CANON_P32_DP2TP2=1")
if CANON_P32_DP2TP2_FORWARD_ONLY and not CANON_P32_DP2TP2:
  raise ValueError("P32 DP2xTP2 forward-only requires CANON_P32_DP2TP2=1")
if CANON_P32_DP2TP2_BACKWARD_NO_COMMIT and not CANON_P32_DP2TP2:
  raise ValueError(
      "P32 DP2xTP2 backward-no-commit requires CANON_P32_DP2TP2=1"
  )
if (
    CANON_P32_D3A_SPLIT_OUTER_JIT
    and not CANON_P32_DP2TP2_BACKWARD_NO_COMMIT
):
  raise ValueError(
      "P32 D3a split-outer-JIT mode requires backward-no-commit"
  )
if CANON_P32_D3B_SEGMENTED and not CANON_P32_DP2TP2_BACKWARD_NO_COMMIT:
  raise ValueError("P32 D3b0 segmented mode requires backward-no-commit")
if CANON_P32_D3B_SEGMENTED and CANON_P32_D3A_SPLIT_OUTER_JIT:
  raise ValueError("P32 D3b0 and the rejected D3a split-JIT mode are exclusive")
if (
    CANON_P32_DP2TP2_FORWARD_ONLY
    and CANON_P32_DP2TP2_BACKWARD_NO_COMMIT
):
  raise ValueError(
      "P32 DP2xTP2 forward-only and backward-no-commit are exclusive"
  )
if (
    CANON_P32_DP2TP2_FORWARD_ONLY
    and os.getenv("CANON_P32_D1_ADMITTED", "") != "1"
):
  raise ValueError("P32 DP2xTP2 forward-only requires D1 admission")
if CANON_P32_DP2TP2_BACKWARD_NO_COMMIT and (
    os.getenv("CANON_P32_D1_ADMITTED", "") != "1"
    or os.getenv("CANON_P32_D2_ADMITTED", "") != "1"
):
  raise ValueError(
      "P32 DP2xTP2 backward-no-commit requires D1 and D2 admission"
  )
if CANON_P32_DP2TP2:
  from tunix.rl import dp_training

  _P32_CONTRACT = dp_training.DPTrainingContract(
      dp_size=2,
      tp_size=2,
      global_prompts=4,
      num_generations=8,
      local_trajectories=16,
  )
  _P32_CONTRACT.validate()
  if not CANON_L3:
    raise ValueError("P32 DP2xTP2 requires CANON_FROZENLAKE_L3=1")
  if jax.device_count() != _P32_CONTRACT.total_devices:
    raise ValueError(
        "P32 DP2xTP2 requires exactly four devices: "
        f"got {jax.device_count()}"
    )
  if os.getenv("MIN_TOKEN_BUCKET") != "512":
    raise ValueError("P32 DP2xTP2 requires MIN_TOKEN_BUCKET=512")
  if os.getenv("CANON_LOGPROB_M") != "512":
    raise ValueError("P32 DP2xTP2 requires CANON_LOGPROB_M=512")
  if (
      not CANON_P32_DP2TP2_INIT_ONLY
      and not CANON_P32_DP2TP2_FORWARD_ONLY
      and not CANON_P32_DP2TP2_BACKWARD_NO_COMMIT
      and os.getenv("CANON_P32_TRAIN_ADMITTED", "") != "1"
  ):
    raise ValueError(
        "P32 DP2xTP2 training is not admitted; run the init-only gate first"
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
) != ("0" if CANON_P31_CONVERGENCE else "1"):
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
if CANON_P30_OPT_STATE_OFFLOAD and not (
    CANON_P29_FULL_TRAIN or CANON_P32_DP2TP2
):
  raise ValueError(
      "optimizer offload requires the P29 or P32 memory-bounded path"
  )
if (
    CANON_P28_G3_ONLY
    or CANON_P28_G4_ONLY
    or CANON_P28_G5_ONLY
    or CANON_P28_G5C_ONLY
    or (CANON_P28_G6_UPDATE and not CANON_P31_CONVERGENCE)
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
# Single shared mesh across actor / reference / rollout. Pure tensor-parallel
# (fsdp=1) so the rollout sampler's batch=1 prefill is not split across an
# fsdp axis.
if CANON_P32_DP2TP2:
  SHARED_MESH_SHAPE = (2, 2)
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
VLLM_MAX_BATCHED_TOKENS = (
    256 if CANON_L3 else VLLM_MAX_NUM_SEQS * 4 * 1024 // 8
)

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
if CANON_P31_CONVERGENCE:
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
    if CANON_P27 or CANON_P32_DP2TP2 else None
)
_P27_TRAJECTORY_MICRO_RAW = os.getenv("CANON_P27_TRAJECTORY_MICRO", "")
if _P27_TRAJECTORY_MICRO_RAW and not CANON_P27:
  raise ValueError("CANON_P27_TRAJECTORY_MICRO requires CANON_FROZENLAKE_P27=1")
if CANON_P32_DP2TP2:
  if _P27_TRAJECTORY_MICRO_RAW:
    raise ValueError(
        "P32 DP2xTP2 derives its local trajectory batch; "
        "CANON_P27_TRAJECTORY_MICRO must be unset"
    )
  TRAIN_TRAJECTORY_MICRO_BATCH_SIZE = 32
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
if CANON_P31_CONVERGENCE:
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
SAVE_INTERVAL_STEPS = 10**9  # effectively disabled; set CKPT_DIR + lower this to enable
MAX_TO_KEEP = 1

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
      "forward-only": os.getenv("CANON_ALIGNMENT_FORWARD_ONLY") == "1",
      "gate-only": os.getenv("CANON_ALIGNMENT_GATE_ONLY") == "1",
      "update-canary": (
          os.getenv("CANON_ALIGNMENT_UPDATE_CANARY") == "1"
      ),
      "train": os.getenv("CANON_ALIGNMENT_TRAIN") == "1",
  }
  active_modes = [name for name, active in alignment_modes.items() if active]
  if CANON_P32_DP2TP2_FORWARD_ONLY:
    if active_modes != ["forward-only"]:
      raise ValueError(
        f"P32 D2 requires alignment forward-only mode, got {active_modes}"
      )
  elif CANON_P32_DP2TP2_BACKWARD_NO_COMMIT:
    if active_modes != ["gate-only"]:
      raise ValueError(
          "P32 D3a requires alignment gate-only mode, got "
          f"{active_modes}"
      )
  elif CANON_P31_CONVERGENCE:
    if active_modes != ["train"]:
      raise ValueError(
          f"P31 requires alignment train mode, got {active_modes}"
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

# Checkpointing is opt-in: set CKPT_DIR to a writable path to enable.
CKPT_DIR = None
TB_LOG_DIR = "/tmp/tunix-tb/frozenlake"


# ====== Build the single shared mesh ======
if jax.device_count() < math.prod(SHARED_MESH_SHAPE):
  raise ValueError(
      f"Expected at least {math.prod(SHARED_MESH_SHAPE)} devices for mesh "
      f"{SHARED_MESH_SHAPE}, got {jax.device_count()}."
  )

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

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.parquet")


def create_datasets(
    train_ds_path: str = TRAIN_DATA_PATH,
    test_ds_path: str = TEST_DATA_PATH,
):
  with fsspec.open(train_ds_path, "rb") as train_f, fsspec.open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_parquet(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df)
  test_ds = Dataset.from_pandas(test_df)
  if args.shuffle_data:
    train_ds = train_ds.shuffle(SEED)
    test_ds = test_ds.shuffle(SEED)

  def process_item(item):
    item["prompts"] = ""
    return item

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds


tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
# Disable Qwen3 thinking mode. The agent prompt already requests explicit
# step-by-step reasoning; with thinking enabled the model writes hundreds of
# ``<think>...</think>`` tokens per turn and exhausts the response budget
# before producing an action.
chat_parser = parser.QwenChatTemplateParser(tokenizer, enable_thinking=False)

if CANON_CONTRACT_ONLY or CANON_A3_ONLY:
  # A1b/A2 inventory needs the real model and rollout runner, not an RL batch.
  # Skipping dataset I/O keeps this preflight independent of FrozenLake data.
  train_dataset = test_dataset = None
  print("[CANON_L3] contract-only: dataset I/O skipped", flush=True)
else:
  train_dataset, test_dataset = create_datasets()
  if CANON_P31_CONVERGENCE:
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
  test_dataset, _ = data_lib.post_init_dataset(
      test_dataset,
      tokenizer,
      batch_size=BATCH_SIZE,
      num_batches=NUM_TEST_BATCHES,
      max_prompt_length=MAX_PROMPT_LENGTH,
  )

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
if CANON_P32_DP2TP2:
  config.shd_config = model_lib.ShardingConfig.get_data_parallel_sharding(
      "dp"
  )
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
if CKPT_DIR:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
  )
else:
  checkpointing_options = None

wandb_config = vars(args)
wandb_config.update({
    "WARMUP_STEPS": WARMUP_STEPS,
    "num_steps": MAX_STEPS,
    "rollout_engine": ROLLOUT_ENGINE,
    "model_id": MODEL_VERSION,
    "mesh_shape": SHARED_MESH_SHAPE,
})
metrics_logging_options = None
if CANON_P32_DP2TP2 or CANON_P29_FULL_TRAIN or not (
    CANON_L3 or CANON_CONTRACT_ONLY or CANON_A3_ONLY
):
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=os.getenv(
          "CANON_P32_LOG_DIR",
          os.getenv("CANON_P29_LOG_DIR", TB_LOG_DIR),
      ),
      project_name=os.getenv(
          "CANON_P32_WANDB_PROJECT",
          os.getenv("CANON_P29_WANDB_PROJECT", "tunix-frozenlake"),
      ),
      run_name=os.getenv(
          "CANON_P32_WANDB_RUN_NAME",
          os.getenv("CANON_P29_WANDB_RUN_NAME", ""),
      ),
      flush_every_n_steps=1,
      backend_kwargs={"wandb": {"config": wandb_config}},
  )

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
    "rollout_vllm_hbm_utilization": (
        0.40 if CANON_P32_DP2TP2 else 0.20
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
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
    },
}

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
        max_steps=MAX_STEPS,
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
        data_sharding_axis=("dp",) if CANON_P32_DP2TP2 else ("fsdp",),
        optimizer_offload=CANON_P30_OPT_STATE_OFFLOAD,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
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
    # Per-token truncated importance-sampling correction. Switches the policy
    # loss to use the trainer's start-of-step recomputed logp as
    # ``old_per_token_logps`` and applies a detached per-token weight
    # ``min(exp(trainer_logp - sampler_logp), threshold)`` to the pg loss.
    # Recommended for multi-turn agentic rollouts where residual numerical
    # drift between sampler and trainer can produce occasional outlier
    # importance ratios.
    sampler_is="token",
    sampler_is_threshold=2.0,
    use_rollout_logps=True,
    advantage_estimator=args.advantage_estimator,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen_actor,
    reference=qwen_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
show_hbm_usage("after RLCluster creation")
if CANON_P32_DP2TP2_INIT_ONLY:
  actor_trainer = rl_cluster.actor_trainer

  def _p32_inventory(state, label):
    specs = nnx.get_partition_spec(state)
    summary = dp_training.validate_dp_replicated_partition_specs(
        specs, label=label
    )
    arrays = [
        value for value in jax.tree.leaves(state)
        if isinstance(value, jax.Array)
    ]
    actual_violations = [
        index for index, value in enumerate(arrays)
        if "dp" in repr(getattr(value.sharding, "spec", None))
    ]
    if actual_violations:
      raise ValueError(
          f"{label} arrays are sharded over dp: {actual_violations[:8]}"
      )
    return {
        **summary,
        "arrays": len(arrays),
        "logical_bytes": sum(
            int(value.size * value.dtype.itemsize) for value in arrays
        ),
        "memory_kinds": sorted({
            value.sharding.memory_kind for value in arrays
        }),
    }

  p32_inventory = {
      "model": _p32_inventory(
          nnx.state(actor_trainer.model, nnx.Param), "model"
      ),
      "optimizer": _p32_inventory(
          nnx.state(actor_trainer.optimizer, nnx.optimizer.OptState),
          "optimizer",
      ),
      "accumulator": _p32_inventory(
          nnx.state(actor_trainer.grad_accumulator), "accumulator"
      ),
  }
  p32_mesh_ids = [int(device.id) for device in shared_mesh.devices.flat]
  if actor_trainer.train_steps != 0:
    raise ValueError(
        f"P32 init-only mutated train_steps: {actor_trainer.train_steps}"
    )
  p32_memory = []
  for device in jax.local_devices():
    stats = device.memory_stats()
    if not stats or "bytes_in_use" not in stats or "bytes_limit" not in stats:
      raise ValueError(f"P32 init-only missing HBM stats for {device}")
    p32_memory.append({
        "device_id": int(device.id),
        "bytes_in_use": int(stats["bytes_in_use"]),
        "bytes_limit": int(stats["bytes_limit"]),
    })
  p32_resident_limit = 80 * 1024**3
  if max(item["bytes_in_use"] for item in p32_memory) >= p32_resident_limit:
    raise ValueError(
        "P32 init-only resident HBM exceeded the registered limit: "
        f"measurements={p32_memory} limit={p32_resident_limit}"
    )
  p32_cleanup = dp_training.detach_jax_vllm_cleanup_finalizer(
      rl_cluster.rollout
  )
  p32_report = {
      "status": "pass",
      "mesh": {
          "shape": list(SHARED_MESH_SHAPE),
          "axes": list(SHARED_MESH_AXIS_NAMES),
          "device_ids": p32_mesh_ids,
      },
      "batch": {
          "global_prompts": _P32_CONTRACT.global_prompts,
          "local_prompts": _P32_CONTRACT.local_prompts,
          "num_generations": _P32_CONTRACT.num_generations,
          "global_trajectories": _P32_CONTRACT.global_trajectories,
          "local_trajectories": _P32_CONTRACT.local_trajectories,
      },
      "inventory": p32_inventory,
      "hbm": p32_memory,
      "resident_limit_bytes": p32_resident_limit,
      "optimizer_commits": 0,
      "cleanup": p32_cleanup,
      "rollout_hbm_utilization": 0.40,
      "m_contract": {"global": 512, "local_per_dp_rank": 256},
      "wandb": {
          "enabled": metrics_logging_options is not None,
          "project": os.getenv(
              "CANON_P32_WANDB_PROJECT", "tunix-frozenlake"
          ),
          "run_name": os.getenv("CANON_P32_WANDB_RUN_NAME", ""),
      },
  }
  print(
      f"[P32.D1] JSON {json.dumps(p32_report, sort_keys=True)}",
      flush=True,
  )
  raise SystemExit(0)
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

# Pass test_dataset as the eval set so the learner runs held-out rollouts
# every EVAL_EVERY_N_STEPS and logs `eval/...` metrics (including
# trajectory_reward → solve rate) separately from train metrics.
grpo_trainer.train(
    train_dataset,
    # P21.3's gate-only trainer rejects evaluation so a second batch cannot
    # consume or reorder the pending host sidecar.
    eval_dataset=(
        test_dataset
        if CANON_P31_CONVERGENCE
        and os.getenv("CANON_P31_ENABLE_EVAL", "") == "1"
        else None
        if CANON_L3
        else test_dataset
    ),
)
if CANON_P32_DP2TP2_FORWARD_ONLY:
  p32_cleanup = dp_training.detach_jax_vllm_cleanup_finalizer(
      rl_cluster.rollout
  )
  rl_cluster.close()
  print(
      "[P32.D2] FORWARD_ONLY_DONE "
      f"global_trajectories={_P32_CONTRACT.global_trajectories} "
      f"local_trajectories={_P32_CONTRACT.local_trajectories} "
      f"cleanup={p32_cleanup} backward=0 commits=0",
      flush=True,
  )
elif CANON_P32_DP2TP2_BACKWARD_NO_COMMIT:
  p32_d3_marker = "[P32.D3B0]" if CANON_P32_D3B_SEGMENTED else "[P32.D3A]"
  p32_d3_label = "D3b0" if CANON_P32_D3B_SEGMENTED else "D3a"
  p32_memory = []
  for device in jax.local_devices():
    stats = device.memory_stats()
    required = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    if not stats or any(name not in stats for name in required):
      raise ValueError(f"P32 {p32_d3_label} missing HBM stats for {device}")
    p32_memory.append({
        "device_id": int(device.id),
        **{name: int(stats[name]) for name in required},
    })
  p32_resident_limit = 80 * 1024**3
  if max(item["bytes_in_use"] for item in p32_memory) >= p32_resident_limit:
    raise ValueError(
        f"P32 {p32_d3_label} resident HBM exceeded the registered limit: "
        f"measurements={p32_memory} limit={p32_resident_limit}"
    )
  if any(
      item["peak_bytes_in_use"] >= item["bytes_limit"]
      for item in p32_memory
  ):
    raise ValueError(
        f"P32 {p32_d3_label} peak HBM reached the physical limit: "
        f"measurements={p32_memory}"
    )
  p32_cleanup = dp_training.detach_jax_vllm_cleanup_finalizer(
      rl_cluster.rollout
  )
  rl_cluster.close()
  print(
      f"{p32_d3_marker} JSON "
      + json.dumps({
          "status": "pass",
          "hbm": p32_memory,
          "resident_limit_bytes": p32_resident_limit,
          "global_trajectories": _P32_CONTRACT.global_trajectories,
          "local_trajectories": _P32_CONTRACT.local_trajectories,
          "backward_executed": True,
          "optimizer_commits": 0,
          "cleanup": p32_cleanup,
      }, sort_keys=True),
      flush=True,
  )
  print(
      f"{p32_d3_marker} BACKWARD_NO_COMMIT_DONE "
      "global_trajectories=32 local_trajectories=16 "
      "backward=1 commits=0",
      flush=True,
  )
elif CANON_P31_CONVERGENCE:
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
