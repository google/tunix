#!/usr/bin/env python
"""DeepSWE evaluation with deepscaler-style task-level parallelism.

This script runs SWE evaluation trajectories and uses RolloutOrchestrator to
parallelize tasks across a TPU cluster for MaxText models (e.g. Qwen3.5-35B-A3B)
with configurable JAX/vLLM sharding meshes.
"""

import argparse
import asyncio
import gc
import os
import time

from examples.deepswe import deepswe_utils

deepswe_utils.setup_runtime_environment()

import datasets as datasets_lib
from examples.deepswe import eval_utils
import jax
from jax.sharding import Mesh
import numpy as np
from transformers import AutoTokenizer
from maxtext.integration.vllm import maxtext_vllm_adapter
import swe_env
from swe_agent import SWEAgent

from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.rl.agentic.parser.chat_template_parser import parser

str2bool = deepswe_utils.str2bool


# ========================== Argument Parsing ==========================

parser_cli = argparse.ArgumentParser(
    description="DeepSWE Evaluation with task-level parallelism"
)
parser_cli.add_argument(
    "--model_version",
    type=str,
    default=os.getenv("MODEL_VERSION", "Qwen/Qwen3-32B"),
    help="Model identifier",
)
parser_cli.add_argument(
    "--model_absolute_path",
    type=str,
    default=os.getenv("MODEL_ABSOLUTE_PATH", os.getenv("MODEL_PATH", None)),
    help="Absolute model path (GCS bucket or local directory)",
)
parser_cli.add_argument(
    "--dataset_path",
    type=str,
    default=os.getenv("DATASET_PATH", None),
    help="Dataset path",
)
parser_cli.add_argument(
    "--dataset_split",
    type=str,
    default=os.getenv("DATASET_SPLIT", "test"),
    help="Dataset split",
)
parser_cli.add_argument(
    "--max_steps",
    type=int,
    default=int(os.getenv("MAX_STEPS", "30")),
    help="Max agent steps per trajectory",
)
parser_cli.add_argument(
    "--max_model_len",
    type=int,
    default=int(os.getenv("MAX_MODEL_LEN", "32768")),
    help="Max context length of the LLM",
)
parser_cli.add_argument(
    "--max_response_length",
    type=int,
    default=int(
        os.getenv(
            "MAX_RESPONSE_LENGTH", os.getenv("MAX_GENERATION_STEPS", "4096")
        )
    ),
    help="Max response tokens per generation",
)
parser_cli.add_argument(
    "--max_concurrent",
    type=int,
    default=int(os.getenv("MAX_CONCURRENT", "128")),
    help="Max concurrent tasks/trajectories",
)
parser_cli.add_argument(
    "--timeout",
    type=float,
    default=float(os.getenv("TIMEOUT", "3600")),
    help="Timeout in seconds for a single trajectory",
)
parser_cli.add_argument(
    "--eval_max_examples",
    type=int,
    default=(
        int(os.environ["EVAL_MAX_EXAMPLES"])
        if os.getenv("EVAL_MAX_EXAMPLES")
        else None
    ),
    help=(
        "Optionally limit total examples loaded for evaluation. Defaults to"
        " None, i.e. evaluate the whole dataset."
    ),
)
parser_cli.add_argument(
    "--max_context_limit",
    type=int,
    default=None,
    help="Max context token limit before terminating",
)
parser_cli.add_argument(
    "--vllm_utilization",
    type=float,
    default=float(
        os.getenv("VLLM_UTILIZATION", os.getenv("VLLM_HBM_UTILIZATION", "0.85"))
    ),
    help="HBM utilization ratio for vLLM",
)
parser_cli.add_argument(
    "--vllm_max_num_seqs",
    type=int,
    default=int(os.getenv("VLLM_MAX_NUM_SEQS", "128")),
)
parser_cli.add_argument(
    "--vllm_max_batched_tokens",
    type=int,
    default=int(os.getenv("VLLM_MAX_BATCHED_TOKENS", "165888")),
)
parser_cli.add_argument(
    "--vllm_reshard_chunk_size",
    type=int,
    default=int(os.getenv("VLLM_RESHARD_CHUNK_SIZE", "1")),
    help="Reshard chunk size for transferring weights into VllmSampler (1 for sequential chunked sync)",
)
parser_cli.add_argument(
    "--use_ocdbt_with_pathways",
    type=str2bool,
    default=os.getenv("USE_OCDBT_WITH_PATHWAYS", "true").lower() == "true",
    help=(
        "Whether to load model weights via from_pretrained and transfer"
        " chunked into VllmSampler. Required for OCDBT checkpoints on Pathways"
        " runtime to prevent host proxy OOM."
    ),
)
parser_cli.add_argument(
    "--mesh_fsdp",
    type=int,
    default=int(os.getenv("MESH_FSDP", "4")),
    help="FSDP dimension size of the mesh",
)
parser_cli.add_argument(
    "--mesh_tp",
    type=int,
    default=int(os.getenv("MESH_TP", "4")),
    help="TP dimension size of the mesh",
)
parser_cli.add_argument(
    "--scan_layers",
    type=str2bool,
    default=os.getenv("SCAN_LAYERS", "true").lower() == "true",
    help="Whether to scan layers for MaxText models",
)
parser_cli.add_argument(
    "--allow_split_physical_axes",
    type=str2bool,
    default=os.getenv("ALLOW_SPLIT_PHYSICAL_AXES", "true").lower() == "true",
    help=(
        "Whether to allow splitting physical axes in MaxText (can cause"
        " performance hit)"
    ),
)
parser_cli.add_argument(
    "--checkpoint_storage_concurrent_gb",
    type=int,
    default=int(os.getenv("CHECKPOINT_STORAGE_CONCURRENT_GB", "32")),
    help="Concurrent GB limit for Orbax checkpoint storage restore",
)
parser_cli.add_argument(
    "--checkpoint_storage_use_ocdbt",
    type=str2bool,
    default=os.getenv("CHECKPOINT_STORAGE_USE_OCDBT", "true").lower() == "true",
    help="Whether the MaxText checkpoint uses OCDBT storage",
)
parser_cli.add_argument(
    "--checkpoint_storage_use_zarr3",
    type=str2bool,
    default=os.getenv("CHECKPOINT_STORAGE_USE_ZARR3", "false").lower()
    == "true",
    help="Whether the MaxText checkpoint uses Zarr3 storage",
)
parser_cli.add_argument(
    "--weight_dtype",
    type=str,
    default=os.getenv("WEIGHT_DTYPE", "bfloat16"),
    help="Weight data type for MaxText model",
)
parser_cli.add_argument(
    "--prefuse_moe_weights",
    type=str2bool,
    default=os.getenv("PREFUSE_MOE_WEIGHTS", "true").lower() == "true",
    help="Whether to prefuse MoE weights in MaxText",
)
parser_cli.add_argument(
    "--maxtext_attention",
    type=str,
    default=os.getenv("MAXTEXT_ATTENTION", "vllm_rpa"),
    help="Attention backend for MaxText in vLLM",
)
parser_cli.add_argument(
    "--enable_continue_decode",
    type=str2bool,
    default=os.getenv("ENABLE_CONTINUE_DECODE", "false").lower() == "true",
    help="Whether to enable continue decode in vLLM",
)
parser_cli.add_argument(
    "--enable_prefix_caching",
    type=str2bool,
    default=os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true",
    help="Whether to enable vLLM prefix caching",
)
parser_cli.add_argument(
    "--temperature",
    type=float,
    default=float(os.getenv("TEMPERATURE", "0.0")),
    help="Sampling temperature (0 for greedy)",
)
parser_cli.add_argument(
    "--top_p",
    type=float,
    default=(float(os.environ["TOP_P"]) if os.getenv("TOP_P") else None),
    help="Nucleus sampling probability",
)
parser_cli.add_argument(
    "--top_k",
    type=int,
    default=(int(os.environ["TOP_K"]) if os.getenv("TOP_K") else None),
    help="Top-k sampling cutoff",
)
parser_cli.add_argument(
    "--num_rollouts_per_instance",
    type=int,
    default=int(os.getenv("NUM_ROLLOUTS_PER_INSTANCE", "1")),
    help="Trajectories to sample per instance (>1 enables pass@k)",
)
parser_cli.add_argument(
    "--scaffold",
    type=str,
    default=os.getenv("SCAFFOLD", "r2egym"),
    choices=["r2egym", "sweagent", "openhands"],
    help="Agent scaffold/sandbox toolset",
)
parser_cli.add_argument(
    "--step_timeout_secs",
    type=int,
    default=int(os.getenv("STEP_TIMEOUT_SECS", str(30 * 60))),
    help="Timeout for a single environment step",
)
parser_cli.add_argument(
    "--reward_timeout_secs",
    type=int,
    default=int(os.getenv("REWARD_TIMEOUT_SECS", str(30 * 60))),
    help="Timeout for reward computation",
)
parser_cli.add_argument(
    "--docker_image_prefix",
    type=str,
    default=os.getenv("DOCKER_IMAGE_PREFIX", None),
    help=(
        "Optional registry prefix to replace the dataset image repo with (e.g."
        " us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix)"
    ),
)
parser_cli.add_argument(
    "--node_selector_val",
    type=str,
    default=os.getenv("NODE_SELECTOR_VAL", "cpu-np"),
    help="Node selector value for GKE nodepool",
)
parser_cli.add_argument(
    "--output_dir",
    type=str,
    default=os.getenv(
        "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "eval_results")
    ),
    help="Output directory for results",
)
parser_cli.add_argument(
    "--logging_level",
    type=str,
    default=os.getenv("LOGGING_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser_cli.add_argument(
    "--use_agent_sandbox",
    type=str2bool,
    nargs="?",
    const=True,
    default=os.getenv("USE_AGENT_SANDBOX", "false").lower() == "true",
    help=(
        "Whether to use Kubernetes Agent Sandbox runtime instead of local"
        " Docker socket."
    ),
)
parser_cli.add_argument(
    "--max_warmpool_replicas",
    "--max_warmpool_size",
    dest="max_warmpool_replicas",
    type=int,
    default=(
        int(os.getenv("MAX_WARMPOOL_SIZE", os.getenv("MAX_WARMPOOL_REPLICAS")))
        if os.getenv("MAX_WARMPOOL_SIZE") or os.getenv("MAX_WARMPOOL_REPLICAS")
        else None
    ),
    help=(
        "Max warmpool replicas per task/image. Defaults to"
        " num_rollouts_per_instance."
    ),
)

args, _ = parser_cli.parse_known_args()

DATASET_PATH = args.dataset_path
DATASET_SPLIT = args.dataset_split

MODEL_VERSION = args.model_version
MODEL_SOURCE = "maxtext"
MODEL_PATH = args.model_absolute_path or os.path.join(
    "/scratch/models/", MODEL_VERSION
)

MAX_STEPS = args.max_steps
MAX_MODEL_LEN = args.max_model_len
MAX_RESPONSE_LENGTH = args.max_response_length
MAX_CONCURRENT = args.max_concurrent
TIMEOUT = args.timeout
EVAL_MAX_EXAMPLES = args.eval_max_examples
MAX_CONTEXT_LIMIT = (
    args.max_context_limit
    if args.max_context_limit is not None
    else max(1, MAX_MODEL_LEN - 256)
)


VLLM_HBM_UTILIZATION = args.vllm_utilization
VLLM_MAX_NUM_SEQS = args.vllm_max_num_seqs
VLLM_MAX_BATCHED_TOKENS = args.vllm_max_batched_tokens
VLLM_RESHARD_CHUNK_SIZE = args.vllm_reshard_chunk_size
USE_OCDBT_WITH_PATHWAYS = args.use_ocdbt_with_pathways

MESH_FSDP = args.mesh_fsdp
MESH_TP = args.mesh_tp
SCAN_LAYERS = args.scan_layers
ALLOW_SPLIT_PHYSICAL_AXES = args.allow_split_physical_axes
CHECKPOINT_STORAGE_CONCURRENT_GB = args.checkpoint_storage_concurrent_gb
WEIGHT_DTYPE = args.weight_dtype
PREFUSE_MOE_WEIGHTS = args.prefuse_moe_weights
MAXTEXT_ATTENTION = args.maxtext_attention
ENABLE_CONTINUE_DECODE = args.enable_continue_decode
ENABLE_PREFIX_CACHING = args.enable_prefix_caching
CHECKPOINT_STORAGE_USE_OCDBT = args.checkpoint_storage_use_ocdbt
CHECKPOINT_STORAGE_USE_ZARR3 = args.checkpoint_storage_use_zarr3

# Longest prompt MaxText must be able to prefill, i.e. the context window
# minus the tokens reserved for the response.
MAX_PREFILL_LENGTH = max(1, MAX_MODEL_LEN - MAX_RESPONSE_LENGTH)

TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_ROLLOUTS_PER_INSTANCE = args.num_rollouts_per_instance

SCAFFOLD = args.scaffold
STEP_TIMEOUT_SECS = args.step_timeout_secs
REWARD_TIMEOUT_SECS = args.reward_timeout_secs
DOCKER_IMAGE_PREFIX = args.docker_image_prefix

NODE_SELECTOR_VAL = args.node_selector_val
OUTPUT_DIR = args.output_dir
USE_AGENT_SANDBOX = args.use_agent_sandbox
MAX_WARMPOOL_REPLICAS = args.max_warmpool_replicas

# ========================== Logging ==========================

logger = deepswe_utils.setup_logging(level=args.logging_level)

# ========================== JAX / Pathways ==========================

logger.info("JAX backend initialized.")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Register MaxText vLLM adapter
maxtext_vllm_adapter.register()
logger.info("Successfully registered MaxTextForCausalLM model with vLLM.")

# ========================== Dataset ==========================

logger.info(
    "Loading dataset %s split=%s ...",
    DATASET_PATH or "R2E-Gym/R2E-Gym-Subset",
    DATASET_SPLIT,
)
if args.dataset_path:
  dataset = datasets_lib.load_from_disk(args.dataset_path)
  if isinstance(dataset, datasets_lib.DatasetDict):
    dataset = dataset[DATASET_SPLIT]
else:
  dataset = datasets_lib.load_dataset(
      "R2E-Gym/R2E-Gym-Subset",
      split=DATASET_SPLIT,
  )

entries = [
    swe_env._normalize_entry(e, docker_image_prefix=DOCKER_IMAGE_PREFIX)
    for e in dataset
    if "docker_image" in e
]
if EVAL_MAX_EXAMPLES:
  entries = entries[:EVAL_MAX_EXAMPLES]

unique_images = set(e["docker_image"] for e in entries)
logger.info(
    "Loaded %d instances (%d unique Docker images)",
    len(entries),
    len(unique_images),
)

# ========================== Kubernetes ==========================

deepswe_utils.setup_kubernetes_config(
    node_selector_val=NODE_SELECTOR_VAL,
    logger=logger,
)

fleet = None
if USE_AGENT_SANDBOX:
  fleet = swe_env._init_global_fleet(
      tasks=entries,
      max_concurrency=MAX_CONCURRENT,
      num_generations=NUM_ROLLOUTS_PER_INSTANCE,
      batch_size=min(MAX_CONCURRENT, 8),
      max_warmpool_replicas=MAX_WARMPOOL_REPLICAS,
      scaffold=SCAFFOLD,
  )
  logger.info('[Main] Starting warmpools for planned tasks on K8s...')
  fleet.start_warmpools(wait=False)


# ========================== Model & Mesh ==========================

# Tokenizer Setup
if not MODEL_PATH.startswith("gs://") and os.path.isdir(MODEL_PATH):
  tokenizer_path = MODEL_PATH
  local_files_only = True
  logger.info("Loading tokenizer from local directory: %s", tokenizer_path)
else:
  tokenizer_path = (
      MODEL_VERSION if "/" in MODEL_VERSION else f"Qwen/{MODEL_VERSION}"
  )
  local_files_only = False
  logger.info("Loading tokenizer from HF Hub: %s", tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=local_files_only, trust_remote_code=True
)
tokenizer_for_agentic = tok_adapter.TokenizerAdapter(tokenizer)
chat_parser = parser.QwenChatTemplateParser(tokenizer)

# The r2egym scaffold terminates every action with `</function>`; stopping
# there matches the training rollouts and avoids generating past the action.
STOP_STRINGS = ["</function>", "<|im_end|>", "<|endoftext|>"]
STOP_TOKEN_IDS = [
    tokenizer.encode("<|im_end|>")[0],
    tokenizer.encode("<|endoftext|>")[0],
]

# Mesh Setup
devices = jax.devices()
total_mesh_devices = MESH_FSDP * MESH_TP
if total_mesh_devices > len(devices):
  raise ValueError(
      f"Requested mesh FSDP={MESH_FSDP} * TP={MESH_TP} "
      f"({total_mesh_devices} devices) exceeds the {len(devices)} available "
      "devices. Set --mesh_fsdp/--mesh_tp to match the topology."
  )

mesh_devices = np.array(devices[:total_mesh_devices]).reshape(
    MESH_FSDP, MESH_TP
)
mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))
logger.info(
    "Using mesh shape fsdp=%d tp=%d (total_devices=%d, used_devices=%d)",
    mesh.shape["fsdp"],
    mesh.shape["tp"],
    len(devices),
    total_mesh_devices,
)

deepswe_utils.configure_orbax_ocdbt_handler(MODEL_PATH, logger=logger)

# ========================== Sampler ==========================

logger.info("Creating VllmSampler ...")

try:
  import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_v3_kernel

  _orig_get_default_block_sizes = rpa_v3_kernel.get_default_block_sizes

  def _safe_get_default_block_sizes(*args, **kwargs):
    res = _orig_get_default_block_sizes(*args, **kwargs)
    if isinstance(res, dict):
      if "bkv_sz" in res and res["bkv_sz"] > 1024:
        res["bkv_sz"] = 1024
      if "bkv_csz" in res and res["bkv_csz"] > 512:
        res["bkv_csz"] = 512
    return res

  rpa_v3_kernel.get_default_block_sizes = _safe_get_default_block_sizes
  logger.info(
      "Patched tpu_inference RPA v3 get_default_block_sizes: capped"
      " bkv_sz<=1024, bkv_csz<=512 to prevent TensorCoreSequencer overflow."
  )
except Exception as e:
  logger.warning("Could not patch RPA v3 get_default_block_sizes: %s", e)

# Patch tpu_inference GDN v3 wrapper.fused_conv1d_gdn:
# Replace Stage 1 (`config.GDNMode.BATCHED`, which compiles to `HLO: fused_conv1d_gdn_batched.1`
# and halts TensorCoreSequencer at 0x2d9f because `seq_tile_size=4, window_size=1` causes
# `wait_in`/`wait_out` in `memory_ref.py` to slice `vmem_ref.at[0, pl.ds(0, dma_size)]` by up to 4x
# across axis 1 of size 1) with a 100% pure-JAX XLA vectorized einsum decode step (`_pure_jax_gdn_batched_stage`),
# while preserving Stage 2 (`config.GDNMode.PER_SEQ`, where `seq_tile_size=1` so `dma_size` never
# exceeds axis 1 and chunked MXU matmul runs prefill at >44,000 tokens/s).
try:
  import functools
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from tpu_inference.kernels.gdn.v3 import config as gdn_v3_config
  from tpu_inference.kernels.gdn.v3 import memory_ref as gdn_v3_memory_ref
  from tpu_inference.kernels.gdn.v3 import metadata as gdn_v3_metadata
  from tpu_inference.kernels.gdn.v3 import wrapper as gdn_v3_wrapper

  def _pure_jax_gdn_batched_stage(
      qkv_2d: jax.Array,          # [batch_size, dim] float32
      b_2d: jax.Array,            # [batch_size, n_v] float32
      a_2d: jax.Array,            # [batch_size, n_v] float32
      conv_state_3d: jax.Array,   # [num_blocks, kernel_size - 1, dim] float32
      recurrent_state: jax.Array, # [num_blocks, n_v, d_k, d_v]
      conv_weight_raw: jax.Array, # [dim, 1, kernel_size]
      conv_bias_1d: jax.Array | None, # [dim] or None
      a_log: jax.Array,           # [n_v]
      dt_bias: jax.Array,         # [n_v]
      distribution: jax.Array,    # [3]
      seq_lens: jax.Array,        # [num_seqs]
      state_indices: jax.Array,   # [num_seqs]
      read_state_indices: jax.Array, # [num_seqs]
      padded_batch_size: int,
      act_out_dtype: jnp.dtype,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pure-JAX vectorized Stage 1 (BATCHED 1-token decode for sequences 0..distribution[0])."""
    max_decode = min(qkv_2d.shape[0], state_indices.shape[0])
    dim = qkv_2d.shape[1]
    k_minus_1 = kernel_size - 1
    num_decode = distribution[0]
    seq_arange = jnp.arange(max_decode, dtype=jnp.int32)
    seq_lens_d = seq_lens[:max_decode]
    state_indices_d = state_indices[:max_decode]
    read_state_indices_d = read_state_indices[:max_decode]
    valid_mask = (seq_arange < num_decode) & (seq_lens_d > 0)
    has_init = valid_mask & (seq_lens_d > 1)

    safe_read_idx = jnp.clip(read_state_indices_d, 0, conv_state_3d.shape[0] - 1)
    # 1. Gather conv_state: [max_decode, k_minus_1, dim]
    init_conv = conv_state_3d[safe_read_idx]
    init_conv = jnp.where(has_init[:, None, None], init_conv, 0.0)

    # Gather the `max_decode` decode tokens from the front of `qkv_2d`: [max_decode, 1, dim]
    x_decode = qkv_2d[:max_decode, None, :]
    window = jnp.concatenate([init_conv, x_decode], axis=1)  # [num_seqs, kernel_size, dim]

    # Depthwise conv1d + SiLU
    w = jnp.squeeze(conv_weight_raw, axis=1).astype(jnp.float32)  # [dim, kernel_size]
    conv_out = jnp.einsum("skd,dk->sd", window, w, precision=jax.lax.Precision.HIGHEST)
    if conv_bias_1d is not None:
      conv_out = conv_out + conv_bias_1d.astype(jnp.float32)[None, :]
    conv_activated = jax.nn.silu(conv_out)  # [num_seqs, dim]

    # Scatter updated conv_state (`window[:, 1:, :]`) for valid decode slots
    new_conv_per_seq = window[:, 1:, :]  # [max_decode, k_minus_1, dim]
    write_indices = jnp.where(valid_mask, state_indices_d, -1)
    out_conv_3d = conv_state_3d.at[write_indices].set(
        new_conv_per_seq,
        mode="drop",
        wrap_negative_indices=False,
    )

    # 2. Vectorized 1-token Gated Delta Rule over `max_decode` decode tokens
    key_dim = n_kq * d_k
    q_all, k_all, v_all = jnp.split(conv_activated, [key_dim, 2 * key_dim], axis=-1)
    q_s = q_all.reshape(max_decode, n_kq, d_k)
    k_s = k_all.reshape(max_decode, n_kq, d_k)
    v_s = v_all.reshape(max_decode, n_v, d_v)

    if n_v != n_kq:
      repeats = n_v // n_kq
      q_s = jnp.repeat(q_s, repeats, axis=1)
      k_s = jnp.repeat(k_s, repeats, axis=1)

    q_s = q_s * jax.lax.rsqrt(jnp.sum(q_s * q_s, axis=-1, keepdims=True) + 1e-6) * (d_k ** -0.5)
    k_s = k_s * jax.lax.rsqrt(jnp.sum(k_s * k_s, axis=-1, keepdims=True) + 1e-6)

    a_s = a_2d[:max_decode].astype(jnp.float32)
    b_s = b_2d[:max_decode].astype(jnp.float32)
    g_s = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
        a_s + dt_bias.astype(jnp.float32)[None, :]
    )
    beta_s = jax.nn.sigmoid(b_s)

    init_rec = recurrent_state[safe_read_idx].astype(jnp.float32)  # [num_seqs, n_v, d_k, d_v]
    init_rec = jnp.where(has_init[:, None, None, None], init_rec, 0.0)

    s_decayed = init_rec * jnp.exp(g_s)[:, :, None, None]
    kv_mem = jnp.einsum("shkv,shk->shv", s_decayed, k_s, precision=jax.lax.Precision.HIGHEST)
    delta = (v_s - kv_mem) * beta_s[:, :, None]
    new_rec = s_decayed + jnp.einsum("shk,shv->shkv", k_s, delta, precision=jax.lax.Precision.HIGHEST)
    out_s = jnp.einsum("shkv,shk->shv", new_rec, q_s, precision=jax.lax.Precision.HIGHEST)
    out_s = jnp.where(valid_mask[:, None, None], out_s, 0.0).astype(act_out_dtype)

    out_recurrent_state = recurrent_state.at[write_indices].set(
        new_rec.astype(recurrent_state.dtype),
        mode="drop",
        wrap_negative_indices=False,
    )

    # Construct `out_act` of shape `(padded_batch_size, n_v, d_v)` for Stage 2 (`PER_SEQ`)
    out_act = jnp.zeros((padded_batch_size, n_v, d_v), dtype=act_out_dtype)
    out_act = out_act.at[:max_decode].set(out_s)
    out_conv_4d = out_conv_3d.reshape(-1, k_minus_1, 1, dim)
    return out_act, out_conv_4d, out_recurrent_state

  @functools.partial(
      jax.jit,
      donate_argnames=("conv_state", "recurrent_state"),
      static_argnames=(
          "n_kq",
          "n_v",
          "d_k",
          "d_v",
          "kernel_size",
          "num_spec_tokens",
          "decode_tile_size",
          "mixed_tile_size",
          "zero_initialize_out",
          "compute_precision",
      ),
  )
  def _hybrid_fused_conv1d_gdn(
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: jax.Array | None,
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      read_state_indices: jax.Array,
      read_offsets: jax.Array | None = None,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      num_spec_tokens: int = 0,
      zero_initialize_out: bool = True,
      compute_precision: jnp.dtype = jnp.float32.dtype,
      decode_tile_size: int = 4,
      mixed_tile_size: int = 64,
  ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    act_out_dtype = qkv.dtype
    conv_out_dtype = conv_state.dtype
    recurrent_out_dtype = recurrent_state.dtype

    qkv_f32 = qkv.astype(jnp.float32)
    b_f32 = b.astype(jnp.float32)
    a_f32 = a.astype(jnp.float32)
    conv_state_f32 = conv_state.astype(jnp.float32)

    del read_offsets, num_spec_tokens, zero_initialize_out, decode_tile_size
    batch_size, dim = qkv_f32.shape
    read_state_indices = read_state_indices.astype(state_indices.dtype)
    act_in_dtype = qkv_f32.dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    mixed_tile_size = min(mixed_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    # Stage 1: Pure-JAX vectorized decode for sequences 0..distribution[0]
    out_act, out_conv_state, out_recurrent_state = _pure_jax_gdn_batched_stage(
        qkv_2d=qkv_f32,
        b_2d=b_f32,
        a_2d=a_f32,
        conv_state_3d=conv_state_f32,
        recurrent_state=recurrent_state,
        conv_weight_raw=conv_weight,
        conv_bias_1d=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        distribution=distribution,
        seq_lens=seq_lens,
        state_indices=state_indices,
        read_state_indices=read_state_indices,
        padded_batch_size=padded_batch_size,
        act_out_dtype=act_out_dtype,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    # Stage 2: Pallas PER_SEQ chunked-matmul for prefill sequences distribution[0]..distribution[-1]
    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    qkv_pad = jnp.pad(qkv_f32, ((0, batch_padding_size), (0, 0))).reshape(padded_batch_size, 1, -1)
    b_pad = jnp.pad(b_f32, ((0, batch_padding_size), (0, num_v_padding_size))).reshape(padded_batch_size, 1, -1)
    a_pad = jnp.pad(a_f32, ((0, batch_padding_size), (0, num_v_padding_size))).reshape(padded_batch_size, 1, -1)

    conv_state_shape = conv_state.shape
    conv_weight_swapped = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias_f32 = conv_bias.astype(jnp.float32) if conv_bias is not None else None

    conv_weights = gdn_v3_memory_ref.ConvWeightsRef(weight=conv_weight_swapped, bias=conv_bias_f32)
    gdn_weights = gdn_v3_memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = gdn_v3_memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    cfg = gdn_v3_config.GDNConfig(
        mode=gdn_v3_config.GDNMode.PER_SEQ,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=mixed_tile_size,
        window_size=1,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=gdn_v3_config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=out_recurrent_state.dtype,
            conv_state=out_conv_state.dtype,
        ),
    )
    metadata_obj = gdn_v3_metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        start_seq=distribution[0],
        end_seq=distribution[-1],
        read_indices=read_state_indices,
    )
    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
    input_output_aliases = {
        len(metadata_obj) + 3: 1,
        len(metadata_obj) + 4: 2,
        len(metadata_obj) + 5: 0,
    }
    out_act, out_conv_state, out_recurrent_state = pl.pallas_call(
        functools.partial(gdn_v3_wrapper.outer_kernel, cfg=cfg),
        out_shape=(out_act, out_conv_state, out_recurrent_state),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
    )(
        metadata_obj,
        qkv_pad,
        b_pad,
        a_pad,
        out_conv_state,
        out_recurrent_state,
        out_act,
        weights,
    )

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    out_conv_state = out_conv_state.astype(conv_out_dtype).reshape(conv_state_shape)
    out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)
    return (out_conv_state, out_recurrent_state), out_act

  gdn_v3_wrapper.fused_conv1d_gdn = _hybrid_fused_conv1d_gdn
  logger.info(
      "Patched tpu_inference GDN v3 wrapper.fused_conv1d_gdn with hybrid"
      " pure-JAX Stage-1 BATCHED decode + Pallas Stage-2 PER_SEQ prefill."
  )
except Exception as e:
  logger.warning("Could not patch GDN v3 fused_conv1d_gdn: %s", e)

from tunix.generate import mappings
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

maxtext_cfg = {
    "model_name": MODEL_VERSION.lower().split("/")[-1],
    "model_call_mode": "inference",
    # vLLM inference requires unrolled layers for PagedAttention KV cache indexing (see maxtext vllm.yml).
    # The base checkpoint remains scanned; weight_converter automatically unrolls scanned layers into vLLM.
    "scan_layers": False,
    "enable_dp_attention": False,
    "allow_split_physical_axes": ALLOW_SPLIT_PHYSICAL_AXES,
    "log_config": False,
    "weight_dtype": WEIGHT_DTYPE,
    "prefuse_moe_weights": PREFUSE_MOE_WEIGHTS,
    "attention": MAXTEXT_ATTENTION,
    "remat_policy": "none",
    "max_target_length": MAX_MODEL_LEN,
    "max_prefill_predict_length": MAX_PREFILL_LENGTH,
    "checkpoint_storage_use_ocdbt": CHECKPOINT_STORAGE_USE_OCDBT,
    "checkpoint_storage_use_zarr3": CHECKPOINT_STORAGE_USE_ZARR3,
    "checkpoint_storage_concurrent_gb": (
        CHECKPOINT_STORAGE_CONCURRENT_GB
    ),
}
if not USE_OCDBT_WITH_PATHWAYS:
  maxtext_cfg["load_parameters_path"] = MODEL_PATH

additional_config = {
    "enable_continue_decode": ENABLE_CONTINUE_DECODE,
    "maxtext_config": maxtext_cfg,
}


mapping_config = mappings.MappingConfig()
engine_kwargs = {
    "model": tokenizer_path,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": VLLM_MAX_NUM_SEQS,
    "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "enable_prefix_caching": ENABLE_PREFIX_CACHING,
    "async_scheduling": True,
    "kv_cache_metrics": True,
    "disable_log_stats": False,
    "tokenizer": tokenizer_path,
    "hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
    "dtype": "bfloat16",
    "enable_expert_parallel": False,
}

# Must be set here rather than at the call site: `VllmSampler.__call__`
# forwards unknown kwargs via `setattr` and swallows failures. Stop strings
# additionally require `detokenize=True`, which the sampler otherwise
# hardcodes to False.
sampling_kwargs = {
    "stop": STOP_STRINGS,
    "stop_token_ids": STOP_TOKEN_IDS,
    "detokenize": True,
}

vllm_config = VllmConfig(
    mesh=mesh,
    hbm_utilization=VLLM_HBM_UTILIZATION,
    init_with_random_weights=USE_OCDBT_WITH_PATHWAYS,
    tpu_backend_type="jax",
    server_mode=True,
    tensor_parallel_size=mesh.shape["tp"],
    data_parallel_size=mesh.shape["fsdp"],
    mapping_config=mapping_config,
    additional_config=additional_config,
    reshard_chunk_size=VLLM_RESHARD_CHUNK_SIZE
    if VLLM_RESHARD_CHUNK_SIZE > 0
    else None,
    engine_kwargs=engine_kwargs,
    sampling_kwargs=sampling_kwargs,
)

# NOTE(b/365549911): Why we use a two-step weight transfer instead of direct loading:
# - Issue: vLLM's direct checkpoint loader transfers all weights at once (e.g. ~70 GB
#   for Qwen3.5 35B). Without native OCDBT direct-read support in Pathways, the
#   gRPC proxy buffers all layers simultaneously and OOMs (>70 GB).
# - Solution: We load the model in MaxText first, then stream weights into vLLM
#   one tensor at a time (reshard_chunk_size=1) via `sampler.update_params()`. This
#   drops proxy memory usage to < 2 GiB and prevents OOM.
if USE_OCDBT_WITH_PATHWAYS:
  from flax import nnx
  from maxtext.configs import pyconfig
  from maxtext import model_creation_utils

  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
  model_name_slug = MODEL_VERSION.lower().split("/")[-1]

  logger.info("Initializing base model weights from %s (scan_layers=%s)...", MODEL_PATH, SCAN_LAYERS)
  trainer_config = pyconfig.initialize(
      [
          "",
          base_yml,
          "num_slices=1",
          f"model_name={model_name_slug}",
          f"load_parameters_path={MODEL_PATH}",
          f"ici_fsdp_parallelism={MESH_FSDP}",
          f"ici_tensor_parallelism={MESH_TP}",
          f"scan_layers={SCAN_LAYERS}",
          f"max_target_length={MAX_MODEL_LEN}",
          f"max_prefill_predict_length={MAX_PREFILL_LENGTH}",
          "remat_policy=none",
          f"dtype={WEIGHT_DTYPE}",
          f"attention={'flash' if MAXTEXT_ATTENTION == 'flash' else 'dot_product'}",
          f"prefuse_moe_weights={PREFUSE_MOE_WEIGHTS}",
          f"checkpoint_storage_use_ocdbt={CHECKPOINT_STORAGE_USE_OCDBT}",
          f"checkpoint_storage_use_zarr3={CHECKPOINT_STORAGE_USE_ZARR3}",
          f"checkpoint_storage_concurrent_gb={CHECKPOINT_STORAGE_CONCURRENT_GB}",
          f"allow_split_physical_axes={ALLOW_SPLIT_PHYSICAL_AXES}",
          "skip_jax_distributed_system=True",
          "load_checkpoint_only_once=True",
          "use_standalone_converter=False",
          "log_config=False",
      ],
      vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
  )

  model, _ = model_creation_utils.from_pretrained(
      trainer_config,
      devices=devices[:total_mesh_devices],
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=tokenizer.pad_token_id,
  )

  logger.info("Initializing VllmSampler with dummy weights (use_ocdbt_with_pathways chunked sync enabled)... ")
  sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)

  model_state = nnx.state(model)
  logger.info("Transferring model weights to VllmSampler with reshard_chunk_size=%s...", VLLM_RESHARD_CHUNK_SIZE)
  sampler.update_params(model_state)
  logger.info("Weight transfer complete. Freeing temporary model weights...")
  del model, model_state
  gc.collect()
else:
  logger.info(
      "Initializing VllmSampler directly with checkpoint from %s ...",
      MODEL_PATH,
  )
  sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)
  logger.info(
      "VllmSampler successfully initialized directly with model weights."
  )

# ========================== Model Call ==========================

# `eos_tokens` is not a parameter of `VllmSampler.__call__`; it would land in
# **kwargs and be silently dropped by the `setattr(SamplingParams, ...)` path.
# For vLLM, stop tokens are configured via `VllmConfig.sampling_kwargs` instead.
SAMPLER_CALL_KWARGS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
}

model_call = eval_utils.create_model_call(
    sampler=sampler,
    tokenizer=tokenizer,
    chat_parser=chat_parser,
    max_response_length=MAX_RESPONSE_LENGTH,
    max_context_limit=MAX_CONTEXT_LIMIT,
    sampler_kwargs=SAMPLER_CALL_KWARGS,
    logger=logger,
)


# ========================== Evaluation ==========================

EvalTrajectoryCollectEngine = eval_utils.EvalTrajectoryCollectEngine


def pairs_generator():
  """Yield NUM_ROLLOUTS_PER_INSTANCE trajectory tasks per dataset entry."""
  for pair_index in range(len(entries) * NUM_ROLLOUTS_PER_INSTANCE):
    entry = entries[pair_index // NUM_ROLLOUTS_PER_INSTANCE]
    agent = SWEAgent(scaffold=SCAFFOLD)
    env = eval_utils.LoggedSWEEnv(
        entry=entry,
        max_steps=MAX_STEPS,
        pair_index=pair_index,
        group_id=pair_index,
        scaffold=SCAFFOLD,
        step_timeout=STEP_TIMEOUT_SECS,
        reward_timeout=REWARD_TIMEOUT_SECS,
        use_agent_sandbox=USE_AGENT_SANDBOX,
        fleet=fleet,
        enforce_xml_function_check=True,
        clip_obs_len=12000,
    )
    yield agent, env


# ========================== Main ==========================

if __name__ == "__main__":
  logger.info(
      "Starting deepscaler-style evaluation: model=%s (source=%s), %d"
      " instances, max_concurrent=%d, max_steps=%d, mesh=(%d, %d)",
      MODEL_VERSION,
      MODEL_SOURCE,
      len(entries),
      MAX_CONCURRENT,
      MAX_STEPS,
      MESH_FSDP,
      MESH_TP,
  )

  try:
    eval_results = asyncio.run(
        eval_utils.run_evaluation(
            entries=entries,
            pairs_stream=pairs_generator(),
            model_call=model_call,
            tokenizer_for_agentic=tokenizer_for_agentic,
            chat_parser=chat_parser,
            timeout=TIMEOUT,
            max_concurrent=MAX_CONCURRENT,
            output_dir=OUTPUT_DIR,
            engine_cls=EvalTrajectoryCollectEngine,
            logger=logger,
            num_rollouts_per_instance=NUM_ROLLOUTS_PER_INSTANCE,
            max_response_length=MAX_RESPONSE_LENGTH,
        )
    )
    eval_utils.compute_pass_at_k(eval_results, logger=logger)
    eval_utils.save_results(
        eval_results,
        entries=entries,
        output_dir=OUTPUT_DIR,
        filename_prefix=f"eval_{MODEL_VERSION.replace('/', '_')}",
        logger=logger,
    )
  finally:
    if USE_AGENT_SANDBOX and fleet is not None:
      logger.info(
          "[Main] Explicitly tearing down SandboxFleet on clean exit..."
      )
      fleet.teardown()
