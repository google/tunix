"""Agentic FrozenLake GRPO recipe for Qwen3-8B on a single TPU host.

Targets v5p-8 / v6e-4 -class hosts where actor, reference, and rollout share
a single mesh. Hyperparameters are exposed via argparse; the rollout backend
is selected via the ``ROLLOUT_ENGINE`` environment variable ("vllm" or
"vanilla", default "vllm").
"""

import contextlib
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
from tunix.perf import export as perf_export
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental import export as perf_export_v2
from tunix.sft import metrics_logger
from tunix.sft import profiler as profiler_lib
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
arg_parser.add_argument("--num_epochs", type=int, default=3)
# Mesh (fsdp, tp). mesh_tp=0 means "all remaining devices" -> the historical
# pure-TP default (1, device_count). Set e.g. --mesh_fsdp 2 --mesh_tp 2 for a
# (2,2) mesh; the vLLM rollout's dp/tp follow the mesh automatically below.
arg_parser.add_argument("--mesh_fsdp", type=int, default=1)
arg_parser.add_argument("--mesh_tp", type=int, default=0)
# Memory-shaping micro-batches (see RLTrainingConfig comment below).
arg_parser.add_argument("--train_micro_batch_size", type=int, default=4)
arg_parser.add_argument("--compute_logps_micro_batch_size", type=int, default=4)
# Sequence packing (None = OFF). Budget tokens per packed row; the segment cap
# defaults to the budget-derived safe bound when unset.
arg_parser.add_argument("--max_seq_token_per_tpu", type=int, default=None)
arg_parser.add_argument("--max_segments_per_packed_row", type=int, default=None)
# wandb run name (e.g. cl3_frozenlake_pack) so pack/unpack arms are comparable.
arg_parser.add_argument("--run_name", type=str, default="")
# vLLM rollout HBM fraction. 0.20 fits the 64-chip swe-evaluation runs; on a
# 4-chip host the rollout must also hold ~8GB/chip of Qwen3-8B weights inside
# this budget, so bump it (e.g. 0.3) if vLLM OOMs at startup.
arg_parser.add_argument(
    "--rollout_vllm_hbm_utilization", type=float, default=0.20
)
# Chunked logp computation: materialize logits [chunk, vocab/TP] instead of
# the full [tokens, vocab/TP] (0 = off). At compute_logps_micro_batch_size=4
# (= 32 seqs x 4096 = 65.5k tokens/chip after fsdp) the full fp32 logits are
# ~20GB (x2 with the log_softmax intermediate) and OOM the inference worker
# on a 4-chip host; chunk 2048 caps that at ~0.6GB. Qwen3 supports the
# required skip_lm_head / compute_final_logits (model.py).
arg_parser.add_argument("--compute_logps_chunk_size", type=int, default=2048)
# Profiler (off by default): set --profiler_log_dir to enable xprof.
arg_parser.add_argument("--profiler_log_dir", type=str, default=None)
arg_parser.add_argument("--profiler_skip_steps", type=int, default=5)
arg_parser.add_argument("--profiler_steps", type=int, default=3)
# RL perf tracing: v1 feeds aggregate span metrics into the metrics stream;
# v2 writes a Perfetto trace to --perf_trace_dir (ui.perfetto.dev).
arg_parser.add_argument("--enable_perf_v1", action="store_true")
arg_parser.add_argument("--enable_perf_v2", action="store_true")
arg_parser.add_argument("--perf_trace_dir", type=str, default=None)
args, _ = arg_parser.parse_known_args()

TRAIN_FRACTION = 1.0
SEED = args.seed

# ====== Sharding ======
# Single shared mesh across actor / reference / rollout. Default is pure
# tensor-parallel (fsdp=1) so the VANILLA sampler's batch=1 prefill is not
# split across an fsdp axis; with the vLLM engine fsdp>1 is fine (rollout
# parallelism follows the mesh via tensor/data_parallel_size below) and gives
# pack_size = fsdp for sequence packing.
SHARED_MESH_SHAPE = (
    args.mesh_fsdp,
    args.mesh_tp or jax.device_count() // args.mesh_fsdp,
)
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
VLLM_MAX_NUM_SEQS = 64
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
# Held-out eval pool size in batches. The frozenlake test set ships with 100
# prompts; NUM_TEST_BATCHES * BATCH_SIZE should be >= 100 to cover one full
# pass per eval (ceil keeps that true for any batch size; equals the old
# hardcoded 2 at the default BATCH_SIZE=64). Eval wall-time scales linearly
# with NUM_TEST_BATCHES * BATCH_SIZE * num_generations.
NUM_TEST_BATCHES = -(-100 // BATCH_SIZE)

EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = args.num_epochs
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# ====== Sequence packing + profiling ======
MAX_SEQ_TOKEN_PER_TPU = args.max_seq_token_per_tpu
MAX_SEGMENTS_PER_PACKED_ROW = args.max_segments_per_packed_row
PROFILER_OPTIONS = (
    profiler_lib.ProfilerOptions(
        log_dir=args.profiler_log_dir,
        skip_first_n_steps=args.profiler_skip_steps,
        profiler_steps=args.profiler_steps,
    )
    if args.profiler_log_dir
    else None
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

# ====== Paths ======
MODEL_VERSION = "Qwen/Qwen3-8B"
MODEL_DOWNLOAD_DIR = "/tmp/models/Qwen3-8B"
DATA_DIR = "/tmp/data/frozenlake"

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

# ====== Download + load model ======
# Download safetensors from HF if not present locally.
if not os.path.isdir(MODEL_DOWNLOAD_DIR) or not any(
    f.endswith(".safetensors") for f in os.listdir(MODEL_DOWNLOAD_DIR)
):
  os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
  oss_utils.hf_pipeline(MODEL_VERSION, MODEL_DOWNLOAD_DIR)

config = model_lib.ModelConfig.qwen3_8b()
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
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_LOG_DIR,
    project_name="tunix-frozenlake",
    run_name=args.run_name,
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
    "rollout_vllm_hbm_utilization": args.rollout_vllm_hbm_utilization,
    "rollout_vllm_tpu_backend_type": "jax",
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
        train_micro_batch_size=args.train_micro_batch_size,
        compute_logps_micro_batch_size=args.compute_logps_micro_batch_size,
        compute_logps_chunk_size=args.compute_logps_chunk_size,
        # Sequence packing: None = OFF (the historical unpacked path).
        max_seq_token_per_tpu=MAX_SEQ_TOKEN_PER_TPU,
        max_segments_per_packed_row=MAX_SEGMENTS_PER_PACKED_ROW,
        profiler_options=PROFILER_OPTIONS,
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
    advantage_estimator=args.advantage_estimator,
)

# Lightweight RL perf tracing over the whole run (complements the short xprof
# window). v1 feeds aggregate span metrics into the metrics stream; v2 writes
# a Perfetto trace to --perf_trace_dir for ui.perfetto.dev.
perf_config = (
    perf_metrics.PerfMetricsConfig()
    if (args.enable_perf_v1 or args.enable_perf_v2)
    else None
)
if args.enable_perf_v1:
  perf_config.custom_export_fn = (
      perf_export.PerfMetricsExport.from_cluster_config(cluster_config)
  )
if args.enable_perf_v2:
  perf_config.custom_export_fn_v2 = (
      perf_export_v2.PerfMetricsExport.from_cluster_config(
          cluster_config=cluster_config,
          trace_dir=args.perf_trace_dir,
      ).export_metrics
  )

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen_actor,
    reference=qwen_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
    perf_config=perf_config,
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
