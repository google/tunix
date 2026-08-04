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

"""Orchestrator script for distributed GSM8K GRPO training. Runs on CPU."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Any

from absl import logging as absl_logging

# Force CPU for Orchestrator JAX
os.environ["JAX_PLATFORMS"] = "cpu"

import grain
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

# Setup paths to import tunix
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.cli.utils import model as model_utils
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils

from tunix.experimental.orchestrator import orchestrator_rl_engine
from tunix.experimental.orchestrator import grpc_worker_proxies
from tunix.experimental.worker import remote_execution


# ====== Argparse ======
arg_parser = argparse.ArgumentParser(
    description="Distributed Orchestrator for Qwen3-1.7B on GSM8K with GRPO."
)
arg_parser.add_argument("--batch_size", type=int, default=4)
arg_parser.add_argument("--mini_batch_size", type=int, default=2)
arg_parser.add_argument("--max_steps", type=int, default=20)
arg_parser.add_argument("--max_response_length", type=int, default=512)
arg_parser.add_argument("--trainer_addr", type=str, default="localhost:20000")
arg_parser.add_argument("--rollout_addr", type=str, default="localhost:20001")
args, _ = arg_parser.parse_known_args()

# ====== Recipe Defaults ======
MODEL_NAME = "Qwen3-1.7B"
MODEL_ID = f"Qwen/{MODEL_NAME}"
SEED = 42

NUM_PROMPTS_PER_STEP = args.batch_size
NUM_GENERATIONS = 8
MINI_BATCH_SIZE = args.mini_batch_size
TRAIN_MICRO_BATCH_SIZE = 1
COMPUTE_LOGPS_MICRO_BATCH_SIZE = 1

MAX_STEPS = args.max_steps
NUM_EPOCHS = 1
EVAL_EVERY_N_STEPS = 100
EVAL_BATCH_SIZE = 16
EVAL_AT_START = False
EVAL_AT_END = False

BETA = 0.04
EPSILON = 0.2
KL_LOSS_MODE = "mse_kl"
LEARNING_RATE = 2.0e-7
WEIGHT_DECAY = 0.01
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1.0e-8
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 5
LR_DECAY_STEPS = 50

MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = args.max_response_length
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128

TRAIN_TEMPERATURE = 1.0
TRAIN_TOP_P = 1.0
TRAIN_TOP_K = None
EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0
EVAL_TOP_K = 1

USE_LORA = True  # MUST use LoRA for gRPC weight sync to fit in memory
LORA_RANK = 64
LORA_ALPHA = 64.0
MODEL_DTYPE = jnp.bfloat16

ARTIFACT_ROOT = os.path.join(REPO_ROOT, "artifacts", "qwen3_grpo_gsm8k_dist")
TFDS_DATA_DIR = os.path.join(ARTIFACT_ROOT, "data")
os.makedirs(TFDS_DATA_DIR, exist_ok=True)

VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""

# ====== Data Loader ======
def _as_text(value: Any) -> str:
  return value if isinstance(value, str) else value.decode("utf-8")

def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()

def build_prompt(question: str) -> str:
  return VTC_PROMPT_TEMPLATE.format(question)

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
  return None

def is_vtc_format_correct(text: str) -> bool:
  has_reasoning = text.count("</reasoning>") == 1
  has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
  reasoning_end = text.find("</reasoning>")
  answer_open = text.find("<answer>")
  answer_close = text.find("</answer>")
  return (
      has_reasoning
      and has_answer
      and reasoning_end != -1
      and answer_open != -1
      and answer_close != -1
      and reasoning_end < answer_open < answer_close
  )

def normalize_answer(text: str | None) -> str | None:
  if text is None:
    return None
  return str(text).replace(",", "").strip()

def _vtc_completion_outcome(
    completion: str, gold: Any
) -> tuple[float, bool, bool, bool]:
  format_ok = is_vtc_format_correct(completion)
  pred = normalize_answer(extract_boxed_answer(completion))
  true = normalize_answer(gold)
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

def vtc_env_reward(task, action):
  gold = task.get("answer")
  completion = action.action if hasattr(action, "action") else action
  score, _, _, _ = _vtc_completion_outcome(completion, gold)
  return score

def vtc_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs):
  del prompts, completions, advantages, answer, kwargs
  rewards = np.asarray(rewards, dtype=np.float32)
  solve_ratio = float(np.mean(rewards > 0.1))
  reward_mean = float(rewards.mean())
  logging.info("[Orchestrator Metric] solve_ratio=%.3f, reward_mean=%.3f", solve_ratio, reward_mean)
  return {
      "rewards/solve_ratio": (solve_ratio, np.mean),
      "rewards/reward_mean": (reward_mean, np.mean),
  }


class VTCGRPOLearner(GRPOLearner):

  def _create_agent_env_pair(self, single_example, group_id: int, pair_index: int):
    # Normalize tfds byte arrays/ndarrays to standard strings for gRPC
    normalized = {
        "prompts": _as_text(single_example["prompts"]),
        "question": _as_text(single_example["question"]),
        "answer": _as_text(single_example["answer"]),
    }
    return super()._create_agent_env_pair(normalized, group_id=group_id, pair_index=pair_index)


def main():
  absl_logging.use_python_logging()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s - [Orchestrator] %(message)s")
  
  # Connect to remote worker processes via gRPC
  logging.info("Connecting to Trainer at grpc://%s", args.trainer_addr)
  trainer_handle = remote_execution.ActorHandle.from_address(f"grpc://{args.trainer_addr}")
  
  logging.info("Connecting to Rollout at grpc://%s", args.rollout_addr)
  rollout_handle = remote_execution.ActorHandle.from_address(f"grpc://{args.rollout_addr}")

  # 1. Initialize tokenizer and local dummy model on CPU
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
  
  # Mesh is a dummy on CPU orchestrator
  mesh = Mesh(jax.devices()[:1], ("data",))
  config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  # Instantiate dummy model to obtain parameter structure (no weights loaded)
  dummy_actor = qwen3_model_lib.Qwen3(config, rngs=nnx.Rngs(0))

  # 2. Local configurations (mirrors trainer/rollout settings)
  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.REFERENCE: mesh,
          rl_engine_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          eval_every_n_steps=EVAL_EVERY_N_STEPS,
          max_steps=MAX_STEPS,
          mini_batch_size=MINI_BATCH_SIZE,
          train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
          compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_prompt_length=MAX_PROMPT_LENGTH,
          max_tokens_to_generate=MAX_RESPONSE_LENGTH,
          return_logprobs=True,
          kv_cache_size=KV_CACHE_SIZE,
          temperature=TRAIN_TEMPERATURE,
          top_p=TRAIN_TOP_P,
      ),
  )

  # 3. Create dummy base engine and wrap with OrchestratorRLEngine
  base_dummy = rl_engine_lib.RLEngine(
      actor=dummy_actor,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  # Proxies route calls from OrchestratorRLEngine to remote gRPC nodes
  trainer_proxy = grpc_worker_proxies.GrpcTrainerWorkerProxy(trainer_handle)
  inference_proxy = grpc_worker_proxies.GrpcInferenceWorkerProxy(trainer_handle)
  rollout_proxy = grpc_worker_proxies.GrpcRolloutWorkerProxy(rollout_handle, cluster_config)
  weight_sync_proxy = grpc_worker_proxies.GrpcWeightSyncProxy(trainer_handle, rollout_handle)

  cluster = orchestrator_rl_engine.OrchestratorRLEngine(
      base=base_dummy,
      trainer_worker=trainer_proxy,
      rollout_worker=rollout_proxy,
      inference_worker=inference_proxy, # Reference scoring runs on Trainer TPU
      weight_sync=weight_sync_proxy,
  )
  
  # Patch actor_trainer with proxy so with_loss_fn() is sent over gRPC
  cluster.actor_trainer = grpc_worker_proxies.RemoteActorTrainerProxy(trainer_handle)

  # 4. GRPOLearner initialization
  grpo_config = GRPOConfig(
      num_generations=NUM_GENERATIONS,
      num_iterations=1,
      beta=BETA,
      kl_loss_mode=KL_LOSS_MODE,
      epsilon=EPSILON,
      epsilon_high=EPSILON,
      advantage_estimator="grpo",
      degenerate_group_masking=False,
      use_rollout_logps=False,
      system_prompt="",
      max_response_length=MAX_RESPONSE_LENGTH,
      loss_agg_mode="sequence-mean-token-mean",
  )

  grpo_trainer = VTCGRPOLearner(
      rl_engine=cluster,
      algo_config=grpo_config,
      chat_parser=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True), # Use tokenizer directly as mock chat template formatter
      metric_fns=[vtc_metric_fn],
      env_kwargs={"reward_fn": vtc_env_reward},
  )

  # Load datasets
  logging.info("Loading gsm8k dataset...")
  train_dataset = build_gsm8k_dataset(
      split="train",
      seed=SEED,
      batch_size=NUM_PROMPTS_PER_STEP,
      data_dir=TFDS_DATA_DIR,
      shuffle=True,
  ).repeat(NUM_EPOCHS)
  
  # Run the training loop!
  logging.info("Starting distributed training...")
  grpo_trainer.train(train_dataset)
  logging.info("Training finished successfully.")

if __name__ == "__main__":
  main()
