"""Agentic FrozenLake GRPO recipe for Gemma4-2B on a single TPU host.

Designed for v5p-4 / v6e-8 -class hosts where actor, reference, and rollout
share a single mesh. Configuration is env-driven so the same image runs
unchanged on any spot VM:

  HF_TOKEN              Hugging Face token for model download.
  WANDB_API_KEY         Wandb API key (auto-picked-up by wandb lib).
  WANDB_PROJECT         Wandb project name (default "tunix-frozenlake").
  WANDB_RUN_NAME        Wandb run name (default uses timestamp).
  MODEL_DOWNLOAD_DIR    Local dir for HF safetensors (default
                        /tmp/models/Gemma4-2B).
  DATA_DIR              Local or gs:// dir holding train.parquet / test.parquet
                        (default gs://tunix/data/Frozenlake).
  CKPT_DIR              Output checkpoint dir. Checkpointing is opt-in; if
                        unset, no checkpoints are written.
  TB_LOG_DIR            TensorBoard log dir (default /tmp/tunix-tb/frozenlake).
  SHARED_MESH_SHAPE     Override the (fsdp, tp) mesh shape. Defaults to
                        (1, jax.device_count()) (pure tensor parallel).
  ROLLOUT_ENGINE        "vanilla" | "vllm"  (default "vllm" — the disaggregated
                        vLLM server avoids the trace-context issues of running
                        the in-process sampler under REMAT and offers higher
                        throughput at full concurrency).
"""

try:
  import gymnasium
except ImportError:
  import subprocess
  import sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])

import contextlib
import datetime
import logging
import math
import os
import sys
from typing import List

from absl import logging as absl_logging
from flax import nnx
import grain
import huggingface_hub
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

from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.models.gemma4 import model as model_lib
from tunix.oss import utils as oss_utils
from tunix.sft import metrics_logger
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils
from tunix.cli.utils import data as data_lib
from examples.frozenlake.agent import FrozenLakeAgent
from examples.frozenlake.env import FrozenLakeEnv

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


def _str_to_bool(value):
  if isinstance(value, bool):
    return value

  normalized_value = value.lower()
  if normalized_value in ("true", "t", "1", "yes", "y"):
    return True
  if normalized_value in ("false", "f", "0", "no", "n"):
    return False

  raise argparse.ArgumentTypeError(
      "expected one of: true, false, 1, 0, yes, no"
  )


arg_parser = argparse.ArgumentParser(
    description="Train FrozenLake on Gemma4-2B (single-host TPU)."
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
# Concurrent rollout threads. Higher values let more trajectories generate
# in parallel; the vLLM engine batches them efficiently as long as KV-cache
# headroom remains. With engine-disagg rollout (vLLM server) there is no
# pjit-dispatch race with the trainer, so all `full_batch_size *
# num_generations` trajectories can be in flight at once. A high cap also lets
# every multi-turn agent step its env without waiting for a previous wave to
# drain. Drop only if KV cache saturates or generation throughput regresses.
arg_parser.add_argument("--max_concurrency", type=int, default=512)
arg_parser.add_argument("--shuffle_data", type=_str_to_bool, default=True)
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

TRAIN_FRACTION = 1.0
SEED = args.seed

# ====== Sharding ======
# Default: v5p-8 → 8 chips, pure TP (fsdp=1) because rollout sampler prefills
# batch=1 sequences and fsdp>1 + splash kernel mismatch.
# Override SHARED_MESH_SHAPE via env for other slice sizes (e.g. v6e-4 → 1,4).
# _mesh_env = os.getenv("SHARED_MESH_SHAPE")
# if _mesh_env:
#   SHARED_MESH_SHAPE = tuple(int(x) for x in _mesh_env.split(","))
# else:
ROLLOUT_MESH_SHAPE = (2, jax.device_count()//2)
TRAINER_MESH_SHAPE = (jax.device_count(), 1)
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
VLLM_MAX_NUM_SEQS = 32
VLLM_MAX_BATCHED_TOKENS = VLLM_MAX_NUM_SEQS * 2 * 1024 // 8

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
# Held-out eval pool size in batches. The frozenlake test set ships with 100
# prompts; with BATCH_SIZE=8 a value of 13 covers one full pass per eval.
# Each eval pass runs NUM_TEST_BATCHES * BATCH_SIZE prompts * num_generations
# rollouts, so eval wall-time scales linearly. If you change BATCH_SIZE,
# adjust this so that NUM_TEST_BATCHES * BATCH_SIZE >= test set size to
# evaluate the full held-out set once per eval.
NUM_TEST_BATCHES = 2

EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 3
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

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
SAVE_INTERVAL_STEPS = 5
MAX_TO_KEEP = 50

# ====== Rollout ======
ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vllm")  # "vanilla" | "vllm"

# ====== Paths (env-driven so the same image runs anywhere) ======
MODEL_VERSION = "google/gemma-4-E2B-it"
MODEL_DOWNLOAD_DIR = huggingface_hub.snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
DATA_DIR = "gs://tunix/data/Frozenlake"

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Checkpointing is opt-in: set CKPT_DIR to enable, otherwise nothing is written.
# Orbax's CheckpointManager force-saves the first step regardless of the
# configured save_interval_steps, so a large interval alone does not disable it.
CKPT_DIR = os.getenv("CKPT_DIR") or None
TB_LOG_DIR = "gs://linchai-bucket-dev/tensorboard/grpo"


# ====== Build the single shared mesh ======
if jax.device_count() < math.prod(ROLLOUT_MESH_SHAPE):
  raise ValueError(
      f"Expected at least {math.prod(ROLLOUT_MESH_SHAPE)} devices for mesh "
      f"{ROLLOUT_MESH_SHAPE}, got {jax.device_count()}."
  )

rollout_device_list = jax._src.mesh_utils.create_device_mesh(
    ROLLOUT_MESH_SHAPE, jax.devices()[: math.prod(ROLLOUT_MESH_SHAPE)]
)
rollout_mesh = jax.sharding.Mesh(
    rollout_device_list,
    axis_names=SHARED_MESH_AXIS_NAMES,
    axis_types=(jax.sharding.AxisType.Auto,) * len(ROLLOUT_MESH_SHAPE),
)
print(f"rollout_mesh.devices.shape={rollout_mesh.devices.shape}")

trainer_device_list = jax._src.mesh_utils.create_device_mesh(
    TRAINER_MESH_SHAPE, jax.devices()[: math.prod(TRAINER_MESH_SHAPE)]
)
trainer_mesh = jax.sharding.Mesh(
    trainer_device_list,
    axis_names=SHARED_MESH_AXIS_NAMES,
    axis_types=(jax.sharding.AxisType.Auto,) * len(TRAINER_MESH_SHAPE),
)
print(f"trainer_mesh.devices.shape={trainer_mesh.devices.shape}")

# ====== Data ======
import pandas as pd
import datasets as datasets_lib
import transformers

try:
  from google.cloud import storage  # noqa: F401 (ensures loading on GCS)
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
# Disable Gemma4 thinking mode. The agent prompt already requests explicit
# step-by-step reasoning; with thinking enabled the model writes hundreds of
# ``<|channel>..<channel|>`` tokens per turn and exhausts the response budget
# before producing an action.
chat_parser = parser.Gemma4ChatTemplateParser(tokenizer, enable_thinking=False)

train_dataset, test_dataset = create_datasets()
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

config = model_lib.ModelConfig.gemma4_e2b()
if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.DECODER
if ENABLE_FLASH_ATTENTION:
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.use_sliding_window_kv_cache = False
if ENABLE_MIX_PRECISION:
  config.dtype = jnp.bfloat16

# Reference: keep bf16 storage (frozen, never updated -> HBM savings safe).
gemma4_ref = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, trainer_mesh, dtype=MODEL_DTYPE
)
show_hbm_usage("after loading gemma4_ref")

# Actor: storage MUST be fp32. At LR=1e-6 with typical weight magnitudes
# ~1e-2, Adam updates are ~1e-6, well below bf16 ULP (~7.8e-5). bf16 storage
# silently rounds every update to zero in optax.apply_updates, so the policy
# never moves. Forward compute can still be bf16 via config.dtype.
gemma4_actor = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, trainer_mesh, dtype=jnp.float32
)
show_hbm_usage("after loading gemma4_actor")

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
    "mesh_shape": TRAINER_MESH_SHAPE,
})
# Tunix's WandbBackend already forwards project_name+run_name; don't put `name`
# in wandb_kwargs (would collide with `name=run_name` kwarg in metrics_logger).
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_LOG_DIR,
    project_name=os.getenv("WANDB_PROJECT", "tunix-frozenlake"),
    run_name=os.getenv("WANDB_RUN_NAME", ""),
    flush_every_n_steps=1,
    backend_kwargs={"wandb": {"config": wandb_config,}},
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
print("Rollout mesh:", rollout_mesh)
print("Trainer mesh:", trainer_mesh)

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
    "rollout_vllm_hbm_utilization": 0.2,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    # Async scheduling adds an extra in-flight step that can race weight sync;
    # disable it under engine-disagg so each rollout completes before the next
    # train step starts.
    "rollout_vllm_async_scheduling": False,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": ROLLOUT_MESH_SHAPE[1],
    "data_parallel_size": ROLLOUT_MESH_SHAPE[0],
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
        "hf_overrides": {
            "final_logit_softcapping": 30.0,
            "text_config": {
                "final_logit_softcapping": 30.0,
            },
            "architectures": ["Gemma4ForCausalLM"],
        },
    },
    "rollout_vllm_sampling_kwargs": {
        "skip_special_tokens": False,
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
        rl_cluster_lib.Role.ACTOR: trainer_mesh,
        rl_cluster_lib.Role.REFERENCE: trainer_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
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
        # ``[micro_batch, seq_len, vocab/TP]`` is the dominant allocation) at
        # the cost of more micro-step launches per optimizer update. It does
        # NOT change the effective optimizer batch size or training dynamics.
        # Memory-shaping micro-batch for forward+backward. Forward+backward
        # produces an ``[micro_batch * num_gen * seq_len, vocab/TP]`` logits
        # tensor in fp32; on small TPU slices this is the dominant allocation.
        # With agentic outer-loop chunking applied below, each outer iter
        # invokes the trainer ``mini_batch_size // train_micro_batch_size``
        # times, so the optimizer still sees a ``mini_batch_size`` gradient
        # per update.
        train_micro_batch_size=2,
        compute_logps_micro_batch_size=2,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        compute_logps_chunk_size=2048,
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
    advantage_estimator=args.advantage_estimator,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=gemma4_actor,
    reference=gemma4_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
show_hbm_usage("after RLCluster creation")


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
    env_kwargs={"max_steps": 8},
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
)
show_hbm_usage("after GRPOLearner creation")

# Pass test_dataset as the eval set so the learner runs held-out rollouts
# every EVAL_EVERY_N_STEPS and logs `eval/...` metrics (including
# trajectory_reward → solve rate) separately from train metrics.
grpo_trainer.train(train_dataset, eval_dataset=test_dataset)
