"""Qwen3-1.7B GSM8K GRPO demo with VTC-style prompt and reward.

This is a Tunix-native, single-turn agentic GRPO demo derived from the
agentic GSM8K notebook flow. It is designed to mirror the visible behavior of
the standalone NeMo GSM8K/VTC demo as closely as the current Tunix agentic
stack allows:

- dataset: GSM8K train/test
- prompt: VTC-style user prompt with <reasoning> and <answer> tags
- reward: exact 0.0 / 0.1 / 0.5 / 1.0 VTC reward table
- algorithm: GRPO (group mean + std normalization), not RLOO
- training sampling: temperature=1.0, top_p=1.0
- evaluation sampling: greedy via a separate eval rollout config

Known gaps:
- Validation cadence matches eval_every_n_steps, but there is no explicit
  val_at_end hook here.
- Tunix does not expose the same KL variant label as the target config, so this
  demo uses the standard KL penalty with beta=0.04.

Run from repo root:

  python examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py
"""

from __future__ import annotations

import argparse
import importlib.resources as stdlib_importlib_resources
import os
import re
import sys

# Disable pathways subslice check by appending it to sys.argv before JAX/absl parse it.
if "--pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
  sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

os.environ["VLLM_TPU_RPA_VERSION"] = "2" 
os.environ["DISABLE_MOSAIC_ATTN"] = "1"

import time
from typing import Any

from absl import flags

try:
  import importlib_resources  # pytype: disable=import-error  # noqa: F401
except ModuleNotFoundError:
  sys.modules["importlib_resources"] = stdlib_importlib_resources

try:
  import vllm
  from vllm.sampling_params import SamplingParams
  vllm.SamplingParams = SamplingParams
  del SamplingParams
  del vllm
except ImportError:
  pass

import grain
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
import tensorflow_datasets.text.gsm8k  # pylint: disable=unused-import
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
workdir = os.getcwd()
if os.path.exists(os.path.join(workdir, "tunix")):
  workspace_root = workdir
else:
  workspace_root = os.path.dirname(REPO_ROOT)

tunix_root = os.path.join(workspace_root, "tunix")
pathways_root = os.path.join(workspace_root, "pathways-utils")
r2egym_root = os.path.join(workspace_root, "r2egym")

for root in [REPO_ROOT, workspace_root, tunix_root, pathways_root, r2egym_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

try:
  import tunix  # pytype: disable=import-error  # noqa: F401
  import pathwaysutils  # pytype: disable=import-error
  import r2egym  # pytype: disable=import-error  # noqa: F401
except ImportError:
  pathwaysutils = None

if pathwaysutils is not None and os.getenv("JAX_PLATFORMS", None) == "proxy":
  pathwaysutils.initialize()

from tunix.cli.utils import model as model_utils
from tunix.models.automodel import call_model_config
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.utils import script_utils


# ------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------
MODEL_NAME = "Qwen3-1.7B"
MODEL_ID = f"Qwen/{MODEL_NAME}"
SEED = 42

NUM_PROMPTS_PER_STEP = 16
NUM_GENERATIONS = 8
TRAIN_GLOBAL_BATCH_SIZE = NUM_PROMPTS_PER_STEP * NUM_GENERATIONS  # 128
TRAIN_MICRO_BATCH_SIZE = 1  # one prompt group at a time

MAX_STEPS = 200
NUM_EPOCHS = 1000
EVAL_EVERY_N_STEPS = 50
EVAL_BATCH_SIZE = 128

BETA = 0.04
EPSILON = 0.2
LEARNING_RATE = 2.0e-7
WEIGHT_DECAY = 0.01
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1.0e-8
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 50
LR_DECAY_STEPS = 500

MAX_TOTAL_SEQUENCE_LENGTH = 1024
MAX_INPUT_SEQUENCE_LENGTH = 1024
MAX_NEW_TOKENS = 1024
MAX_PROMPT_LENGTH = MAX_INPUT_SEQUENCE_LENGTH
MAX_GENERATION_LENGTH = MAX_NEW_TOKENS
KV_CACHE_SIZE = MAX_INPUT_SEQUENCE_LENGTH + MAX_NEW_TOKENS + 256
DEFAULT_VLLM_HBM_UTILIZATION = 0.6
DEFAULT_VLLM_MAX_NUM_SEQS = NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = 65536

TRAIN_TEMPERATURE = 1.0
TRAIN_TOP_P = 1.0
TRAIN_TOP_K = None

EVAL_TEMPERATURE = 1e-4
EVAL_TOP_P = 1.0
EVAL_TOP_K = 1

USE_LORA = False
LORA_RANK = 64
LORA_ALPHA = 64.0

ARTIFACT_ROOT = os.path.join(REPO_ROOT, "artifacts", "qwen3_grpo_gsm8k_vtc")
TFDS_DATA_DIR = os.path.join(ARTIFACT_ROOT, "data")
MODEL_DOWNLOAD_DIR = os.path.join(ARTIFACT_ROOT, "models")
INTERMEDIATE_CKPT_DIR = os.path.join(ARTIFACT_ROOT, "intermediate_ckpt")
CHECKPOINT_ROOT = os.path.join(ARTIFACT_ROOT, "checkpoints", str(int(time.time())))
LOG_DIR = os.path.join(ARTIFACT_ROOT, "logs")

for path in [
    TFDS_DATA_DIR,
    MODEL_DOWNLOAD_DIR,
    INTERMEDIATE_CKPT_DIR,
    CHECKPOINT_ROOT,
    LOG_DIR,
]:
  os.makedirs(path, exist_ok=True)


VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def build_prompt(question: str) -> str:
  return VTC_PROMPT_TEMPLATE.format(question)


def _as_text(value: Any) -> str:
  return value if isinstance(value, str) else value.decode("utf-8")


def build_gsm8k_dataset(
    *,
    split: str,
    seed: int,
    batch_size: int,
    data_dir: str,
    shuffle: bool,
) -> grain.MapDataset:
  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  dataset = dataset.map(
      lambda x: {
          "prompts": build_prompt(_as_text(x["question"])),
          "question": _as_text(x["question"]),
          "answer": extract_hash_answer(_as_text(x["answer"])),
      }
  )
  return dataset.batch(batch_size)


def extract_boxed_answer(text: str) -> str | None:
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text

  boxed = []
  stack = []
  for i, ch in enumerate(content):
    if ch == "{":
      stack.append(i)
    elif ch == "}":
      if not stack:
        continue
      open_idx = stack.pop()
      if content[:open_idx].endswith(r"\boxed"):
        boxed.append(content[open_idx + 1 : i].strip())
  if boxed:
    return boxed[-1]

  fallback = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if fallback:
    return fallback.group(1).strip()
  return None


def is_vtc_format_correct(text: str) -> bool:
  has_reasoning = text.count("</reasoning>") == 1
  has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
  reasoning_end = text.find("</reasoning>")
  answer_open = text.find("<answer>")
  answer_close = text.find("</answer>")
  ordered = (
      has_reasoning
      and has_answer
      and reasoning_end != -1
      and answer_open != -1
      and answer_close != -1
      and reasoning_end < answer_open < answer_close
  )
  return has_reasoning and has_answer and ordered


def normalize_answer(text: str | None) -> str | None:
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def _normalize_example_value(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return _normalize_example_value(flat[0])
    return [_normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return value


def normalize_single_example(example: dict[str, Any]) -> dict[str, Any]:
  return {key: _normalize_example_value(value) for key, value in example.items()}


def vtc_reward(prompts, completions, answer, **kwargs):
  del prompts, kwargs
  scores = []
  for completion, gold in zip(completions, answer):
    format_ok = is_vtc_format_correct(completion)
    pred = normalize_answer(extract_boxed_answer(completion))
    true = normalize_answer(gold)
    answer_ok = pred is not None and true is not None and pred == true

    if format_ok and answer_ok:
      score = 1.0
    elif format_ok and not answer_ok:
      score = 0.1
    elif not format_ok and answer_ok:
      score = 0.5
    else:
      score = 0.0
    scores.append(score)
  return scores


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Run the Qwen3-1.7B GSM8K VTC GRPO demo with vLLM rollout engine "
          "on a separate mesh."
      )
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=16,
      help="Global batch size (number of prompts per step).",
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      default=200,
      help="Maximum number of training steps.",
  )
  parser.add_argument(
      "--train_micro_batch_size",
      type=int,
      default=1,
      help="Micro batch size for training.",
  )
  parser.add_argument(
      "--max_response_length",
      type=int,
      default=1024,
      help="Maximum length of the generated response.",
  )
  parser.add_argument(
      "--rollout_mesh_fsdp",
      type=int,
      default=None,
      help="Optional override for rollout mesh FSDP dimension.",
  )
  parser.add_argument(
      "--rollout_mesh_tp",
      type=int,
      default=None,
      help="Optional override for rollout mesh TP dimension.",
  )
  parser.add_argument(
      "--train_mesh_fsdp",
      type=int,
      default=None,
      help="Optional override for train mesh FSDP dimension.",
  )
  parser.add_argument(
      "--train_mesh_tp",
      type=int,
      default=None,
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
          "Fraction of total devices to allocate to rollout. Default is 0.5 "
          "which splits devices evenly between Rollout and Train."
      ),
  )
  parser.add_argument(
      "--rollout_vllm_hbm_utilization",
      type=float,
      default=DEFAULT_VLLM_HBM_UTILIZATION,
      help="HBM utilization target for the vLLM rollout backend.",
  )
  parser.add_argument(
      "--rollout_vllm_max_num_seqs",
      type=int,
      default=None,
      help="Maximum concurrent sequences for the vLLM rollout backend.",
  )
  parser.add_argument(
      "--rollout_vllm_max_num_batched_tokens",
      type=int,
      default=None,
      help="Maximum batched tokens for the vLLM rollout backend.",
  )
  args, _ = parser.parse_known_args()
  return args


def maybe_apply_lora(model: nnx.Module, mesh: Mesh) -> nnx.Module:
  if not USE_LORA:
    graph_def, params = nnx.split(model)
    return nnx.merge(graph_def, jax.tree.map(jnp.copy, params))

  lora_config = {
      "module_path": (
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      "rank": LORA_RANK,
      "alpha": LORA_ALPHA,
  }
  return model_utils.apply_lora_to_model(model, mesh=mesh, lora_config=lora_config)


def put_model_on_device(model: nnx.Module) -> nnx.Module:
  graph_def, state = nnx.split(model)
  state = rl_utils.put_params_on_memory_kind(state, "device")
  return nnx.merge(graph_def, state)


def create_reference_and_actor(mesh: Mesh) -> tuple[nnx.Module, nnx.Module, str]:
  model_config = {
      "model_name": MODEL_NAME,
      "model_source": "huggingface",
      "model_id": MODEL_ID,
      "model_download_path": MODEL_DOWNLOAD_DIR,
      "intermediate_ckpt_dir": INTERMEDIATE_CKPT_DIR,
      "model_display": False,
      "use_flash_attention": False,
      "flash_attention_block_size": 1024,
      "dtype": jnp.bfloat16,
  }
  tokenizer_config = {
      "tokenizer_path": MODEL_ID,
      "tokenizer_type": "hf",
      "add_bos": False,
      "add_eos": False,
  }
  reference, tokenizer_path = model_utils.create_model(
      model_config, tokenizer_config, mesh
  )
  reference = put_model_on_device(reference)
  actor = maybe_apply_lora(reference, mesh)
  actor = put_model_on_device(actor)
  return reference, actor, tokenizer_path


def create_optimizer() -> optax.GradientTransformation:
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=LEARNING_RATE,
      warmup_steps=WARMUP_STEPS,
      decay_steps=LR_DECAY_STEPS,
      end_value=0.0,
  )
  optimizer = optax.adamw(
      learning_rate=lr_schedule,
      b1=ADAM_B1,
      b2=ADAM_B2,
      eps=ADAM_EPS,
      weight_decay=WEIGHT_DECAY,
  )
  return optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optimizer)


class VTCQwenChatTemplateParser(chat_parser_lib.QwenChatTemplateParser):
  """Qwen parser variant that omits the blank/default system turn."""

  def _handle_first_message(self, messages):
    del messages
    return ""

  def _parse_system(self, content: str) -> str:
    if not content:
      return ""
    return super()._parse_system(content)


class VTCGRPOLearner(GRPOLearner):
  """Demo-local learner that normalizes TFDS string payloads to Python str."""

  def _create_agent_env_pair(self, single_example, group_id: int, pair_index: int):
    normalized_example = normalize_single_example(single_example)
    return super()._create_agent_env_pair(
        normalized_example, group_id=group_id, pair_index=pair_index
    )


def main():
  args = parse_args()

  global MAX_STEPS, TRAIN_MICRO_BATCH_SIZE, MAX_NEW_TOKENS, MAX_GENERATION_LENGTH, KV_CACHE_SIZE, NUM_PROMPTS_PER_STEP, TRAIN_GLOBAL_BATCH_SIZE
  MAX_STEPS = args.max_steps
  NUM_PROMPTS_PER_STEP = args.batch_size
  TRAIN_GLOBAL_BATCH_SIZE = NUM_PROMPTS_PER_STEP * NUM_GENERATIONS
  TRAIN_MICRO_BATCH_SIZE = args.train_micro_batch_size
  MAX_NEW_TOKENS = args.max_response_length
  MAX_GENERATION_LENGTH = args.max_response_length
  KV_CACHE_SIZE = MAX_INPUT_SEQUENCE_LENGTH + MAX_NEW_TOKENS + 256

  config = call_model_config(MODEL_NAME)
  devices = jax.devices()
  total_devices = len(devices)

  # 1. Resolve Mesh Dimensions
  if args.rollout_split_fraction == 1.0:
    # Use Shared Mesh: All devices for both Rollout and Train
    print(f"Using Shared Mesh for Rollout and Train (100% of {total_devices} devices).")
    num_rollout_devices = total_devices
    num_train_devices = total_devices
    
    tp_size = int(np.gcd(total_devices, config.num_kv_heads))
    fsdp_size = total_devices // tp_size
    
    rollout_dims = [("fsdp", fsdp_size), ("tp", tp_size)]
    train_dims = [("fsdp", fsdp_size), ("tp", tp_size)]
    
    rollout_axis_names = tuple(name for name, _ in rollout_dims)
    rollout_shape = tuple(d for _, d in rollout_dims)
    train_axis_names = tuple(name for name, _ in train_dims)
    train_shape = tuple(d for _, d in train_dims)

    rollout_devices = np.array(devices).reshape(rollout_shape)
    train_devices = np.array(devices).reshape(train_shape)

    rollout_mesh = Mesh(rollout_devices, axis_names=rollout_axis_names)
    train_mesh = Mesh(train_devices, axis_names=train_axis_names)

  else:
    # Use Separate Meshes
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
            "rollout_split_fraction must allocate at least 1 rollout device and "
            "leave at least 1 train device."
        )
      rollout_tp = int(np.gcd(num_rollout_devices, config.num_kv_heads))
      rollout_fsdp = num_rollout_devices // rollout_tp
      rollout_dims = [("fsdp", rollout_fsdp), ("tp", rollout_tp)]
    num_rollout_devices = int(np.prod([d for _, d in rollout_dims]))

    train_fsdp = args.train_mesh_fsdp
    train_sp = args.train_mesh_sp
    train_tp = args.train_mesh_tp
    if any(v is not None for v in (train_fsdp, train_sp, train_tp)):
      train_dims = []
      train_dims.append(("fsdp", train_fsdp if train_fsdp is not None else 1))
      if train_sp is not None:
        train_dims.append(("sp", train_sp))
      train_dims.append(("tp", train_tp if train_tp is not None else 1))
    else:
      num_train_devices = total_devices - num_rollout_devices
      if num_train_devices <= 0:
        raise ValueError("Separate rollout mesh resolution left no train devices.")
      train_tp = int(np.gcd(num_train_devices, config.num_kv_heads))
      train_fsdp = num_train_devices // train_tp
      train_dims = [("fsdp", train_fsdp), ("tp", train_tp)]
    num_train_devices = int(np.prod([d for _, d in train_dims]))

    if num_rollout_devices + num_train_devices > total_devices:
      raise ValueError("Requested devices exceed total available devices.")

    rollout_axis_names = tuple(name for name, _ in rollout_dims)
    rollout_shape = tuple(d for _, d in rollout_dims)
    train_axis_names = tuple(name for name, _ in train_dims)
    train_shape = tuple(d for _, d in train_dims)

    rollout_devices = np.array(devices[:num_rollout_devices]).reshape(rollout_shape)
    train_devices = np.array(devices[num_rollout_devices : num_rollout_devices + num_train_devices]).reshape(train_shape)

    rollout_mesh = Mesh(rollout_devices, axis_names=rollout_axis_names)
    train_mesh = Mesh(train_devices, axis_names=train_axis_names)

  train_dataset = build_gsm8k_dataset(
      split="train",
      seed=SEED,
      batch_size=NUM_PROMPTS_PER_STEP,
      data_dir=TFDS_DATA_DIR,
      shuffle=True,
  ).repeat(NUM_EPOCHS)
  eval_dataset = build_gsm8k_dataset(
      split="test",
      seed=SEED,
      batch_size=EVAL_BATCH_SIZE,
      data_dir=TFDS_DATA_DIR,
      shuffle=False,
  )

  reference, actor, tokenizer_path = create_reference_and_actor(train_mesh)
  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path,
      token=os.getenv("HF_TOKEN"),
      trust_remote_code=True,
  )
  chat_parser = VTCQwenChatTemplateParser(tokenizer)
  qwen_eos_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=MAX_STEPS,
      max_to_keep=1,
  )
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=LOG_DIR,
      flush_every_n_steps=10,
  )

  rollout_config_base = dict(
      max_tokens_to_generate=MAX_GENERATION_LENGTH,
      max_prompt_length=MAX_PROMPT_LENGTH,
      kv_cache_size=KV_CACHE_SIZE,
      eos_tokens=qwen_eos_tokens,
  )
  train_rollout_kwargs = dict(
      temperature=TRAIN_TEMPERATURE,
      top_p=TRAIN_TOP_P,
      top_k=TRAIN_TOP_K,
  )
  eval_rollout_kwargs = dict(
      temperature=EVAL_TEMPERATURE,
      top_p=EVAL_TOP_P,
      top_k=EVAL_TOP_K,
  )
  vllm_max_num_seqs = args.rollout_vllm_max_num_seqs if args.rollout_vllm_max_num_seqs is not None else (NUM_PROMPTS_PER_STEP * NUM_GENERATIONS)
  vllm_max_batched_tokens = args.rollout_vllm_max_num_batched_tokens if args.rollout_vllm_max_num_batched_tokens is not None else ((vllm_max_num_seqs * KV_CACHE_SIZE) // 8)

  vllm_rollout_kwargs = dict(
      rollout_vllm_model_version=MODEL_ID,
      rollout_vllm_hbm_utilization=args.rollout_vllm_hbm_utilization,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_server_mode=True,
      rollout_vllm_async_scheduling=True,
      tensor_parallel_size=rollout_mesh.shape.get("tp", 1),
      data_parallel_size=rollout_mesh.shape.get("fsdp", 1),
      rollout_vllm_max_num_seqs=vllm_max_num_seqs,
      rollout_vllm_max_num_batched_tokens=vllm_max_batched_tokens,
      rollout_vllm_kwargs={
          "kv_cache_metrics": True,
          "disable_log_stats": False,
          "enable_prefix_caching": True,
          "max_model_len": KV_CACHE_SIZE,
          "tpu_rpa_version": 2,
          "disable_mosaic_attn": True,
      },
  )
  train_rollout_kwargs.update(vllm_rollout_kwargs)
  eval_rollout_kwargs.update(vllm_rollout_kwargs)

  train_rollout_config = base_rollout.RolloutConfig(
      **rollout_config_base,
      **train_rollout_kwargs,
  )
  eval_rollout_config = base_rollout.RolloutConfig(
      **rollout_config_base,
      **eval_rollout_kwargs,
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: train_mesh,
          rl_cluster_lib.Role.REFERENCE: train_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine="vllm",
      # Force explicit host<->device materialization for vanilla rollout.
      # This avoids mixed host/device memory spaces inside sampler gather ops on
      # TPU, which can happen with shared-mesh Qwen3 demo setup.
      #
      # Setting to False to avoid gather memory space mismatch error on TPU,
      # where weights are on host and indices on device.
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=create_optimizer(),
          eval_every_n_steps=EVAL_EVERY_N_STEPS,
          max_steps=MAX_STEPS,
          metrics_logging_options=metrics_logging_options,
          checkpoint_root_directory=CHECKPOINT_ROOT,
          checkpointing_options=checkpointing_options,
          train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
          mini_batch_size=TRAIN_GLOBAL_BATCH_SIZE // NUM_GENERATIONS,
      ),
      rollout_config={
          rl_cluster_lib.Mode.TRAIN: train_rollout_config,
          rl_cluster_lib.Mode.EVAL: eval_rollout_config,
      },
  )

  grpo_config = GRPOConfig(
      num_generations=NUM_GENERATIONS,
      eval_num_generations=1,
      num_iterations=1,
      beta=BETA,
      kl_loss_mode="kl",
      epsilon=EPSILON,
      epsilon_high=EPSILON,
      advantage_estimator="grpo",
      degenerate_group_masking=False,
      use_rollout_logps=False,
      system_prompt="",
      max_concurrency=1024,
      loss_agg_mode="sequence-mean-token-mean",
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  trainer = VTCGRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=[vtc_reward],
      algo_config=grpo_config,
      chat_parser=chat_parser,
  )

  print(f"Devices: {jax.device_count()}")
  print(f"Train mesh: {train_mesh} | dims: {train_dims}")
  print(f"Rollout mesh: {rollout_mesh} | dims: {rollout_dims}")
  print(f"Artifact root: {ARTIFACT_ROOT}")
  print(f"Checkpoint dir: {CHECKPOINT_ROOT}")
  print(
      "Config summary:",
      {
          "model": MODEL_ID,
          "rollout_engine": "vllm",
          "separate_rollout_mesh": True,
          "shared_mesh_fsdp": None,
          "shared_mesh_tp": None,
          "rollout_split_fraction": args.rollout_split_fraction,
          "rollout_mesh_fsdp": args.rollout_mesh_fsdp,
          "rollout_mesh_tp": args.rollout_mesh_tp,
          "train_mesh_fsdp": args.train_mesh_fsdp,
          "train_mesh_sp": args.train_mesh_sp,
          "train_mesh_tp": args.train_mesh_tp,
          "batch_size": NUM_PROMPTS_PER_STEP,
          "num_generations": NUM_GENERATIONS,
          "eval_num_generations": 1,
          "mini_batch_groups": TRAIN_GLOBAL_BATCH_SIZE // NUM_GENERATIONS,
          "max_steps": MAX_STEPS,
          "max_total_sequence_length": MAX_TOTAL_SEQUENCE_LENGTH,
          "train_temperature": TRAIN_TEMPERATURE,
          "eval_temperature": EVAL_TEMPERATURE,
          "rollout_vllm_hbm_utilization": args.rollout_vllm_hbm_utilization,
          "rollout_vllm_max_num_seqs": args.rollout_vllm_max_num_seqs,
          "rollout_vllm_max_num_batched_tokens": (
              args.rollout_vllm_max_num_batched_tokens
          ),
      },
  )

  with script_utils.profile_and_capture_log(
      "qwen3_grpo_gsm8k_vtc_demo", enable_profile=False
  ):
    trainer.train(train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
  main()
