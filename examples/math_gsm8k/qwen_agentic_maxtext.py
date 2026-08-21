# %%
# [WIP] Reproduction of [DeepSWE](https://www.together.ai/blog/deepswe)
# with Multi-turn Agentic framework.

# %%
import argparse
import faulthandler
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

os.environ["VLLM_TPU_RPA_VERSION"] = "2"
os.environ["DISABLE_MOSAIC_ATTN"] = "1"
import signal
import sys

from absl import logging as absl_logging
import datasets as datasets_lib
from flax import nnx
import grain
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from transformers import AutoTokenizer
from pydantic import ValidationError
from tunix.cli.utils import data as data_lib
from tunix.rl.agentic.agents import agent_types
from tunix.utils import compat
import vllm  # pytype: disable=import-error

faulthandler.register(signal.SIGINT, all_threads=True)

Dataset = datasets_lib.Dataset


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ==========================================
# 0. Argument Parsing
# ==========================================
parser = argparse.ArgumentParser(
    description="DeepSWE Training with Multi-turn Agentic Framework"
)
parser.add_argument("--scan_layers", type=str2bool, default=False)

# General Config
parser.add_argument("--models_base_dir", type=str, default="models")
parser.add_argument(
    "--model_source",
    type=str,
    default="maxtext",
    choices=["huggingface", "maxtext"],
)
parser.add_argument(
    "--model_absolute_path",
    type=str,
    default="gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_version", type=str, default="Qwen3.5-35B-A3B")
parser.add_argument("--node_selector_val", type=str, default="deepswe-cpu-pool")
parser.add_argument("--dataset_path", type=str, default=None)

parser.add_argument("--tpu_topology", type=str, default=None)


# Data & Training Flow
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mini_batch_size", type=int, default=8)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--eval_every_n_steps", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--enable_remat", type=bool, default=True)
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
parser.add_argument("--train_with_lora", type=bool, default=False)

# GRPO Config
parser.add_argument("--num_generations", type=int, default=8)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--epsilon_high", type=float, default=0.28)
parser.add_argument("--off_policy_steps", type=int, default=0)

# Rollout Config
parser.add_argument("--max_prompt_length", type=int, default=4096)
parser.add_argument("--max_response_length", type=int, default=8192)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--rollout_engine", type=str, default="vllm")
parser.add_argument("--vllm_utilization", type=float, default=0.4)
parser.add_argument(
    "--vllm_reshard_chunk_size",
    type=int,
    default=None,
    help="Number of flat keys to reshard at a time. None for single-call.",
)
parser.add_argument(
    "--max_num_batched_tokens",
    type=int,
    default=8192,
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
    type=bool,
    default=False,
    help="Whether to offload optimizer states to CPU (pinned host memory).",
)  # not supported yet


# Checkpointing
parser.add_argument("--ckpt_dir", type=str, default="/tmp/cp/deepswe_ckpt/01")
parser.add_argument("--max_to_keep", type=int, default=4)
parser.add_argument("--save_interval_steps", type=int, default=500)
parser.add_argument("--checkpoint_storage_concurrent_gb", type=int, default=96)
parser.add_argument(
    "--checkpoint_storage_use_ocdbt", type=str2bool, default=True
)
parser.add_argument(
    "--checkpoint_storage_use_zarr3", type=str2bool, default=True
)


# Microbatch Sizes
parser.add_argument("--train_micro_batch_size", type=int, default=1)
parser.add_argument("--rollout_micro_batch_size", type=int, default=1)
parser.add_argument("--compute_logps_micro_batch_size", type=int, default=1)

# DeepSWE Agentic Specifics
parser.add_argument("--max_turns", type=int, default=1)
parser.add_argument("--per_turn_timeout_secs", type=int, default=300)
parser.add_argument("--episode_timeout_secs", type=int, default=3 * 60 * 60)
parser.add_argument("--step_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--reward_timeout_secs", type=int, default=30 * 60)
parser.add_argument("--max_concurrency", type=int, default=200)

parser.add_argument(
    "--overlong_filter",
    type=bool,
    default=True,
    help="Whether to filter out trajectories that exceed length limits",
)

# Mesh / Topology Config Override
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
    type=bool,
    default=False,
    help=(
        "Whether to use rollout-cached logprobs as old policy logps. "
        "Default is False to recompute old logps on the actor side. "
    ),
)


# Other
parser.add_argument("--do_mem_profiling", type=bool, default=False)

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
    default="float32",
    choices=["bfloat16", "float16", "float32"],  # Restrict to valid inputs
    help="Data type for the model weights (e.g., bfloat16, float32)",
)


parser.add_argument("--use_flash_attention", type=bool, default=True)
parser.add_argument("--flash_attention_block_size", type=int, default=1024)
parser.add_argument("--metric_logger_dir", type=str, default=None)
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

for root in [workdir, pathways_root]:
  if root not in sys.path:
    sys.path.insert(0, root)

# Verification
try:
  import tunix
  import pathwaysutils

  print("✅ tunix pathways-utils are successfully mapped.")
except ImportError as e:
  print(f"❌ Still missing a module: {e}")


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
from tunix.rl.grpo import grpo_learner
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic.parser.chat_template_parser import parser as template_parser
from tunix import PerfMetricsConfig
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental.export import PerfMetricsExport
from tunix.rl.agentic.rewards.reward_types import RewardOutput

# %%

import tensorflow_datasets as tfds
import tensorflow_datasets.text.gsm8k
import re


def _as_text(value: str) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, (bytes, np.bytes_)):
    return (
        value.tobytes().decode("utf-8")
        if isinstance(value, np.bytes_)
        else value.decode("utf-8")
    )
  if isinstance(value, (list, tuple, np.ndarray)):
    flat = np.array(value).reshape(-1)
    return _as_text(flat[0]) if len(flat) > 0 else ""
  return str(value)


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


GSM8K_SYSTEM_PROMPT = (
    "You are given a problem. Think about the problem and provide your"
    " reasoning. Place it between <reasoning> and </reasoning>. Then, provide"
    " the final answer (i.e., just one numerical value) between <answer> and"
    " </answer>."
)


def build_prompt(question: str) -> str:
  return f"{GSM8K_SYSTEM_PROMPT}\n\nProblem: {question}\n\n"


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


def create_datasets() -> tuple[grain.MapDataset, grain.MapDataset]:
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
  return train_dataset, eval_dataset


def _normalize_example_value(value: Any) -> Any:
  if isinstance(value, (list, tuple)):
    if len(value) == 1:
      return _normalize_example_value(value[0])
    return [_normalize_example_value(v) for v in value]
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
  return {
      key: _normalize_example_value(value) for key, value in example.items()
  }


# ====== Reward + Metrics ======
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
  if answer_blocks and answer_blocks[-1].strip():
    return answer_blocks[-1].strip()
  return None


def is_vtc_format_correct(text: str) -> bool:
  has_reasoning = ("<reasoning>" in text and "</reasoning>" in text) or (
      "<think>" in text and "</think>" in text
  )
  has_boxed = (r"\boxed" in text) or (
      "<answer>" in text and "</answer>" in text
  )
  return bool(has_reasoning and has_boxed)


def normalize_answer(text: str | None) -> str | None:
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def _vtc_completion_outcome(
    completion: str, gold: Any
) -> tuple[float, bool, bool, bool]:
  format_ok = is_vtc_format_correct(completion)
  pred = normalize_answer(extract_boxed_answer(completion))
  true = normalize_answer(_normalize_example_value(gold))
  answer_ok = pred is not None and true is not None and pred == true
  extracted_ok = pred is not None

  if format_ok and answer_ok:
    score = 1.0
  elif format_ok and not answer_ok:
    score = 0.1
  elif not format_ok and answer_ok:
    score = 0.5
  else:
    score = 0.0
  return score, format_ok, answer_ok, extracted_ok


from types import SimpleNamespace
from maxtext.trainers.post_train.rl import utils_rl

maxtext_tmvp_config = SimpleNamespace(
    reward_exact_answer=1.0,
    reward_white_space_format_match=1.0,
    reward_exact_format_match=0.1,
    reward_partial_format_match=0.0,
    penalty_incorrect_format=0.0,
    penalty_incorrect_answer=0.0,
    math_verify_timeout=120,
    math_verify_num_procs=None,
    reasoning_start_token="<reasoning>",
    reasoning_end_token="</reasoning>",
    reasoning_start_token_in_prompt=False,
    solution_start_token="<answer>",
    solution_end_token="</answer>",
    debug=False,
)


def match_format_exactly(prompts, completions, **kwargs):
  del kwargs
  p_list = [_as_text(p) for p in prompts]
  c_list = [_as_text(c) for c in completions]
  return utils_rl.match_format_exactly(
      p_list, c_list, tmvp_config=maxtext_tmvp_config
  )


def match_format_approximately(prompts, completions, **kwargs):
  del kwargs
  p_list = [_as_text(p) for p in prompts]
  c_list = [_as_text(c) for c in completions]
  return utils_rl.match_format_approximately(
      p_list, c_list, tmvp_config=maxtext_tmvp_config
  )


def check_numbers(prompts, completions, answer=None, **kwargs):
  if answer is None:
    answer = kwargs.get("answer", [])
  p_list = [_as_text(p) for p in prompts]
  c_list = [_as_text(c) for c in completions]
  q_list = [_as_text(q) for q in kwargs.get("question", p_list)]

  encoded_answers = []
  for a in answer:
    if isinstance(a, str):
      try:
        parsed = json.loads(a)
        if isinstance(parsed, list):
          encoded_answers.append(a)
        else:
          encoded_answers.append(json.dumps([str(parsed)]))
      except Exception:
        encoded_answers.append(json.dumps([a]))
    elif isinstance(a, (list, tuple, np.ndarray)):
      encoded_answers.append(json.dumps([_as_text(x) for x in a]))
    else:
      encoded_answers.append(json.dumps([_as_text(a)]))

  return utils_rl.check_numbers(
      p_list,
      c_list,
      answer=encoded_answers,
      tmvp_config=maxtext_tmvp_config,
      question=q_list,
  )


def gsm8k_batch_reward_fn(prompts, completions, answer=None, **kwargs):
  r_fmt = match_format_exactly(prompts, completions, **kwargs)
  r_app = match_format_approximately(prompts, completions, **kwargs)
  r_num = check_numbers(prompts, completions, answer=answer, **kwargs)
  return [float(f + a + n) for f, a, n in zip(r_fmt, r_app, r_num)]


def vtc_env_reward(task, action):
  gold = task.get("answer")
  prompt = task.get("prompts", task.get("question", ""))
  completion = action.action if hasattr(action, "action") else action
  comp_str = _as_text(completion)
  p_str = _as_text(prompt)
  q_str = _as_text(task.get("question", p_str))
  r = gsm8k_batch_reward_fn(
      [p_str], [comp_str], answer=[gold], question=[q_str]
  )
  return float(r[0])


_metric_call_idx = 0
_traj_logger = None


def get_traj_logger():
  global _traj_logger
  if _traj_logger is None and args.metric_logger_dir:
    try:
      from tunix.utils import trajectory_logger

      _traj_logger = trajectory_logger.AsyncTrajectoryLogger(
          args.metric_logger_dir
      )
    except Exception as e:
      absl_logging.warning(f"Failed to initialize AsyncTrajectoryLogger: {e}")
  return _traj_logger


def vtc_metric_fn(
    prompts, completions, rewards, advantages, answer=None, **kwargs
):
  del advantages, kwargs
  global _metric_call_idx
  _metric_call_idx += 1

  rewards_arr = np.asarray(rewards, dtype=np.float32)
  solve_all = bool(np.all(rewards_arr > 0.1))
  solve_none = bool(np.all(np.isclose(rewards_arr, 0.0)))
  solve_partial = (not solve_all) and (not solve_none)
  solve_ratio = float(np.mean(rewards_arr > 0.1))
  reward_mean = float(rewards_arr.mean())
  reward_max = float(rewards_arr.max())

  absl_logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx,
      len(rewards_arr),
      solve_ratio,
      reward_mean,
      reward_max,
      int(solve_all),
      int(solve_none),
  )

  # Print live sample trajectories to stdout for inspection
  if completions is not None and len(completions) > 0:
    num_samples_to_log = min(2, len(completions))
    for i in range(num_samples_to_log):
      p_str = (
          str(prompts[i]) if prompts is not None and i < len(prompts) else ""
      )
      c_str = str(completions[i])
      g_str = str(answer[i]) if answer is not None and i < len(answer) else ""
      r_val = float(rewards[i]) if i < len(rewards) else 0.0
      preview = c_str[:1200]
      if len(c_str) > 1500:
        preview += "\n... [TRUNCATED] ...\n" + c_str[-300:]
      absl_logging.info(
          "\n"
          + "=" * 80
          + f"\n[SAMPLE TRAJECTORY {i+1}/{len(completions)}] Call:"
          f" {_metric_call_idx} | Reward: {r_val:.2f} | Length: {len(c_str)}"
          f" chars | Gold: {g_str}\n"
          + f"PROMPT:\n{p_str[-300:]}\n"
          + "-" * 40
          + f"\nCOMPLETION:\n{preview}\n"
          + "=" * 80
      )

  # Log all trajectories to CSV in GCS
  logger = get_traj_logger()
  if logger is not None and completions is not None and len(completions) > 0:
    for i in range(len(completions)):
      p_str = (
          str(prompts[i]) if prompts is not None and i < len(prompts) else ""
      )
      c_str = str(completions[i])
      g_str = str(answer[i]) if answer is not None and i < len(answer) else ""
      r_val = float(rewards[i]) if i < len(rewards) else 0.0
      logger.log_item_async({
          "call_idx": _metric_call_idx,
          "sample_idx": i,
          "reward": r_val,
          "prompt": p_str,
          "completion": c_str,
          "gold": g_str,
          "completion_len": len(c_str),
      })

  return {
      "gsm8k/reward_mean": (reward_mean, np.mean),
      "gsm8k/reward_max": (reward_max, np.max),
      "gsm8k/solve_all": (1 if solve_all else 0, np.mean),
      "gsm8k/solve_none": (1 if solve_none else 0, np.mean),
      "gsm8k/solve_partial": (1 if solve_partial else 0, np.mean),
      "gsm8k/solve_ratio": (solve_ratio, np.mean),
  }


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
    snapshot_download(  # pyrefly: ignore[no-matching-overload]
        repo_id=MODEL_VERSION,
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
PER_TURN_TIMEOUT_SECS = args.per_turn_timeout_secs
EPISODE_TIMEOUT_SECS = args.episode_timeout_secs
STEP_TIMEOUT_SECS = args.step_timeout_secs
REWARD_TIMEOUT_SECS = args.reward_timeout_secs

MAX_CONCURRENCY = args.max_concurrency
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128
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


# Max number of sequences to be processed in parallel by vllm.
VLLM_MAX_NUM_SEQS = ROLLOUT_MICRO_BATCH_SIZE * NUM_GENERATIONS

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


# %%
# ==========================================
# 5. Tokenizer & Dataset Preparation
# ==========================================
tokenizer_path = (
    MODEL_VERSION
    if MODEL_VERSION.startswith("Qwen/")
    else f"Qwen/{MODEL_VERSION}"
)
print(f"Loading tokenizer from HF Hub: {tokenizer_path}")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, local_files_only=False, trust_remote_code=True
)

chat_parser = template_parser.QwenChatTemplateParser(tokenizer)


print("Loading GSM8K Dataset...")

train_dataset = build_gsm8k_dataset(
    split="train",
    seed=SEED,
    batch_size=BATCH_SIZE,
    data_dir="/tmp/gsm8k_data",
    shuffle=True,
).repeat(NUM_EPOCHS)


# %%
# ==========================================
# 6. JAX Device, Config & Mesh Setup (MaxText)
# ==========================================
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig, types
from maxtext.utils import model_creation_utils, maxtext_utils

devices = jax.devices()
total_devices = len(devices)

# 1. Resolve Rollout Mesh Dimensions
rollout_fsdp = args.rollout_mesh_fsdp
rollout_tp = args.rollout_mesh_tp
if rollout_fsdp is None and rollout_tp is None:
  num_rollout_devices = int(total_devices * args.rollout_split_fraction)
  rollout_tp = 2
  rollout_fsdp = num_rollout_devices // rollout_tp
else:
  rollout_fsdp = rollout_fsdp if rollout_fsdp is not None else 1
  rollout_tp = rollout_tp if rollout_tp is not None else 1
  num_rollout_devices = rollout_fsdp * rollout_tp

# 2. Resolve Train Mesh Dimensions
train_fsdp = args.train_mesh_fsdp
train_tp = args.train_mesh_tp
if train_fsdp is None and train_tp is None:
  num_train_devices = total_devices - num_rollout_devices
  train_tp = 2
  train_fsdp = num_train_devices // train_tp
else:
  train_fsdp = train_fsdp if train_fsdp is not None else 1
  train_tp = train_tp if train_tp is not None else 1
  num_train_devices = train_fsdp * train_tp

if num_rollout_devices + num_train_devices > total_devices:
  raise ValueError(
      f"Requested {num_rollout_devices} rollout devices + {num_train_devices} "
      f"train devices, but cluster only has {total_devices} available."
  )

base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
vllm_yml = os.path.join(
    os.path.dirname(pyconfig.__file__), "inference", "vllm.yml"
)

maxtext_remat_policy = "none"
if args.enable_remat:
  if args.remat_policy in ("decoder", "full"):
    maxtext_remat_policy = "full"
  elif args.remat_policy in ("block", "minimal"):
    maxtext_remat_policy = "minimal"
  else:
    maxtext_remat_policy = args.remat_policy

trainer_config = pyconfig.initialize(
    [
        "",
        base_yml,
        f"model_name=qwen3.5-35b-a3b",
        f"load_parameters_path={MODEL_PATH}",
        f"ici_fsdp_parallelism={train_fsdp}",
        f"ici_tensor_parallelism={train_tp}",
        f"scan_layers={args.scan_layers}",
        f"max_target_length={KV_CACHE_SIZE}",
        f"max_prefill_predict_length={MAX_PROMPT_LENGTH}",
        f"remat_policy={maxtext_remat_policy}",
        f"dtype={args.dtype}",
        f"checkpoint_storage_use_ocdbt={args.checkpoint_storage_use_ocdbt}",
        f"checkpoint_storage_use_zarr3={args.checkpoint_storage_use_zarr3}",
        f"checkpoint_storage_concurrent_gb={args.checkpoint_storage_concurrent_gb}",
        "skip_jax_distributed_system=True",
        "load_checkpoint_only_once=True",
        "use_standalone_converter=False",
        "log_config=False",
    ],
    vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
)

sampler_config = pyconfig.initialize(
    [
        "",
        vllm_yml,
        f"model_name=qwen3.5-35b-a3b",
        f"ici_data_parallelism={rollout_fsdp}",
        f"ici_tensor_parallelism={rollout_tp}",
        f"rollout_tensor_parallelism={rollout_tp}",
        f"rollout_data_parallelism={rollout_fsdp}",
        f"max_target_length={KV_CACHE_SIZE}",
        f"max_prefill_predict_length={MAX_PROMPT_LENGTH}",
        f"dtype={args.dtype}",
        "skip_jax_distributed_system=True",
        "remat_policy=none",
        "use_standalone_converter=False",
        "log_config=False",
    ],
    config_class=types.RLConfig,
    vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]},
)

sampler_devices = devices[:num_rollout_devices]
trainer_devices = devices[
    num_rollout_devices : num_rollout_devices + num_train_devices
]

# %%
# ==========================================
# 7. Model Initialization via MaxText
# ==========================================

from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import type_handler_registry

# Ensure standard ArrayHandler is used for OCDBT base model restore
type_handler_registry.register_type_handler(
    jax.Array, jax_array_handlers.ArrayHandler(), override=True
)

(
    qwen_reference,
    reference_mesh,
    qwen_actor,
    actor_mesh,
    rollout_mesh,
) = model_creation_utils.create_models_and_meshes(
    trainer_config=trainer_config,
    sampler_config=sampler_config,
    trainer_devices=trainer_devices,
    sampler_devices=sampler_devices,
    tokenizer_pad_id=tokenizer.pad_token_id,
)

train_mesh = actor_mesh

print(f"*** Rollout Mesh *** | Shape: {rollout_mesh.shape}")
print(f"*** Train Mesh *** | Shape: {train_mesh.shape}")

if TRAIN_WITH_LORA:

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

  qwen_actor = get_lora_model(qwen_actor, train_mesh)

if hasattr(qwen_reference, "use_no_op_mappings"):
  qwen_reference.use_no_op_mappings = False
if hasattr(qwen_actor, "use_no_op_mappings"):
  qwen_actor.use_no_op_mappings = False

sft_utils.show_hbm_usage()

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
    log_dir=args.metric_logger_dir, flush_every_n_steps=2
)

optimizer = optax.schedules.inject_hyperparams(optax.adamw)(
    learning_rate=LEARNING_RATE,
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)

if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(MAX_GRAD_NORM),
      optimizer,
  )

# %%
# ==========================================
# 9. Rollout Engine Setup (vLLM)
# ==========================================
base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "kv_cache_size": KV_CACHE_SIZE,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": tokenizer_path,
    "rollout_vllm_hbm_utilization": VLLM_UTILIZATION,
    "rollout_vllm_reshard_chunk_size": VLLM_RESHARD_CHUNK_SIZE,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": True,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": rollout_mesh.shape.get("model", 1),
    "data_parallel_size": rollout_mesh.shape.get("data", 1),
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "tokenizer": tokenizer_path,
        "dtype": "bfloat16",
        "enable_expert_parallel": False,
        "hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
    },
    "rollout_mapping_config": {},
    "rollout_vllm_additional_config": {
        "maxtext_config": {
            "model_name": "qwen3.5-35b-a3b",
            "model_call_mode": "inference",
            "attention": "vllm_rpa",
            "allow_split_physical_axes": True,
            "log_config": False,
            "weight_dtype": "bfloat16",
            "prefuse_moe_weights": True,
            "remat_policy": "none",
            "enable_dp_attention": False,
            "vllm_hf_overrides": {"architectures": ["MaxTextForCausalLM"]},
        }
    },
    "rollout_vllm_sampling_kwargs": {
        "stop": ["</answer>", "<|im_end|>", "<|endoftext|>"],
        "stop_token_ids": [
            tokenizer.encode("<|im_end|>")[0],
            tokenizer.encode("<|endoftext|>")[0],
        ],
        "detokenize": True,
    },
}

if ROLLOUT_ENGINE == "vllm":
  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
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

from maxtext.integration.vllm.maxtext_vllm_adapter import adapter

_orig_generate_maxtext_config = adapter.generate_maxtext_config


def _generate_maxtext_config_with_no_remat(vllm_config_param):
  if "maxtext_config" not in vllm_config_param.additional_config:
    vllm_config_param.additional_config["maxtext_config"] = {}
  vllm_config_param.additional_config["maxtext_config"]["remat_policy"] = "none"
  return _orig_generate_maxtext_config(vllm_config_param)


adapter.generate_maxtext_config = _generate_maxtext_config_with_no_remat

role_to_logical_axis_rule = {
    rl_engine_lib.Role.ACTOR: trainer_config.logical_axis_rules,
    rl_engine_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
    rl_engine_lib.Role.ROLLOUT: sampler_config.logical_axis_rules,
}

import functools
from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout

rollout_engine_arg = functools.partial(
    MaxTextVllmRollout, maxtext_config=sampler_config
)

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: actor_mesh,
        rl_engine_lib.Role.REFERENCE: reference_mesh,
        rl_engine_lib.Role.ROLLOUT: rollout_mesh,
    },
    role_to_logical_axis_rule=role_to_logical_axis_rule,
    rollout_engine=rollout_engine_arg,
    offload_to_cpu=False,
    training_config=rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
        rollout_micro_batch_size=ROLLOUT_MICRO_BATCH_SIZE,
        metrics_logging_options=metrics_logging_options,
        perf_metrics_options=perf_metrics.PerfMetricsOptions(),
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=rollout_engine_config,
)
sft_utils.show_hbm_usage()

try:
  rl_engine = rl_engine_lib.RLEngine(
      actor=qwen_actor,
      reference=qwen_reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
except ValidationError as e:
  print("Failed to initialize RLEngine due to ValidationError:", flush=True)
  import pprint

  pprint.pprint(e.errors())
  raise

# %%
# ==========================================
# 10. Learner & Trainer Setup
# ==========================================

config_kwargs = {
    "num_generations": NUM_GENERATIONS,
    "num_iterations": NUM_ITERATIONS,
    "beta": BETA,
    "epsilon": EPSILON,
    "loss_agg_mode": LOSS_AGG_MODE,
    "advantage_estimator": ADVANTAGE_ESTIMATOR,
    "use_rollout_logps": USE_ROLLOUT_LOGPS,
    "max_concurrency": MAX_CONCURRENCY,
    "max_response_length": MAX_RESPONSE_LENGTH,
}
if FILTER_STATUSES:
  config_kwargs["filter_statuses"] = FILTER_STATUSES
if OVERLONG_FILTER is not None:
  config_kwargs["overlong_filter"] = OVERLONG_FILTER

grpo_config = agentic_grpo_learner.GRPOConfig(**config_kwargs)

trainer = agentic_grpo_learner.GRPOLearner(
    rl_engine=rl_engine,
    algo_config=grpo_config,
    chat_parser=chat_parser,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_numbers,
    ],
    metric_fns=[vtc_metric_fn],
)


try:
  import datetime
  import wandb  # pytype: disable=import-error

  settings = wandb.Settings(console="off")
  run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  wandb_config = {
      **vars(args),
      # Derived values not present in args
      "kv_cache_size": KV_CACHE_SIZE,
      "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
      "vllm_max_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
      # Stringify set so wandb can serialize it
      "filter_statuses": (
          [s.name for s in FILTER_STATUSES] if FILTER_STATUSES else None
      ),
      # Mesh topology
      "num_devices": len(devices),
      "rollout_mesh_fsdp": rollout_fsdp,
      "rollout_mesh_tp": rollout_tp,
      "train_mesh_fsdp": train_fsdp,
      "train_mesh_sp": train_sp,
      "train_mesh_tp": train_tp,
      "checkpoint_root_directory": CKPT_DIR,
      "save_interval_steps": SAVE_INTERVAL_STEPS,
      "max_to_keep": MAX_TO_KEEP,
  }
  if wandb.run is None:
    wandb.init(
        project="tunix", name=run_name, config=wandb_config, settings=settings
    )
except Exception as e:
  print(f"W&B initialization failed with error: {e}")


# %%
print("Syncing initial checkpoint weights to rollout workers...", flush=True)
rl_engine.sync_weights()

if (
    CKPT_DIR
    and "proxy" in os.getenv("JAX_PLATFORMS", "")
    and os.getenv("ENABLE_PATHWAYS_PERSISTENCE", "")
):
  import orbax.checkpoint.pathways as ocp_pathways

  print(
      "Registering Pathways persistence handlers for training checkpoints...",
      flush=True,
  )
  ocp_pathways.register_type_handlers(
      checkpointing_impl=ocp_pathways.CheckpointingImpl.PERSISTENCE
  )

print("Starting training...", flush=True)
trainer.train(train_dataset=train_dataset)


# %%
