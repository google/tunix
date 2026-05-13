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
- This uses Tunix's vanilla rollout backend for portability.
- Validation cadence matches eval_every_n_steps, but there is no explicit
  val_at_end hook here.
- Tunix does not expose the same KL variant label as the target config, so this
  demo uses the standard KL penalty with beta=0.04.

Run from repo root:

  python examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

import grain
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.cli.utils import model as model_utils
from tunix.rl import rl_cluster as rl_cluster_lib
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

NUM_PROMPTS_PER_STEP = 4
NUM_GENERATIONS = 8
TRAIN_GLOBAL_BATCH_SIZE = 16  # 2 prompt groups x 8 generations/group
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
KV_CACHE_SIZE = MAX_INPUT_SEQUENCE_LENGTH + MAX_NEW_TOKENS

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
TRAIN_DATA_DIR = os.path.join(ARTIFACT_ROOT, "data", "train")
TEST_DATA_DIR = os.path.join(ARTIFACT_ROOT, "data", "test")
MODEL_DOWNLOAD_DIR = os.path.join(ARTIFACT_ROOT, "models")
INTERMEDIATE_CKPT_DIR = os.path.join(ARTIFACT_ROOT, "intermediate_ckpt")
CHECKPOINT_ROOT = os.path.join(ARTIFACT_ROOT, "checkpoints", str(int(time.time())))
LOG_DIR = os.path.join(ARTIFACT_ROOT, "logs")

for path in [
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
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


def build_mesh() -> Mesh:
  devices = np.array(jax.devices())
  if devices.size == 0:
    raise ValueError("No JAX devices found.")
  return Mesh(devices.reshape((devices.size,)), axis_names=("fsdp",))


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


def create_reference_and_actor(mesh: Mesh) -> tuple[nnx.Module, nnx.Module, str]:
  model_config = {
      "model_name": MODEL_NAME,
      "model_source": "hf",
      "model_id": MODEL_ID,
      "model_download_path": MODEL_DOWNLOAD_DIR,
      "intermediate_ckpt_dir": INTERMEDIATE_CKPT_DIR,
      "model_display": False,
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
  actor = maybe_apply_lora(reference, mesh)
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


def main():
  mesh = build_mesh()

  train_dataset = build_gsm8k_dataset(
      split="train",
      seed=SEED,
      batch_size=NUM_PROMPTS_PER_STEP,
      data_dir=TRAIN_DATA_DIR,
      shuffle=True,
  ).repeat(NUM_EPOCHS)
  eval_dataset = build_gsm8k_dataset(
      split="test",
      seed=SEED,
      batch_size=EVAL_BATCH_SIZE,
      data_dir=TEST_DATA_DIR,
      shuffle=False,
  )

  reference, actor, tokenizer_path = create_reference_and_actor(mesh)
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

  train_rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_GENERATION_LENGTH,
      max_prompt_length=MAX_PROMPT_LENGTH,
      kv_cache_size=KV_CACHE_SIZE,
      temperature=TRAIN_TEMPERATURE,
      top_p=TRAIN_TOP_P,
      top_k=TRAIN_TOP_K,
      eos_tokens=qwen_eos_tokens,
  )
  eval_rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_GENERATION_LENGTH,
      max_prompt_length=MAX_PROMPT_LENGTH,
      kv_cache_size=KV_CACHE_SIZE,
      temperature=EVAL_TEMPERATURE,
      top_p=EVAL_TOP_P,
      top_k=EVAL_TOP_K,
      eos_tokens=qwen_eos_tokens,
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
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
      max_concurrency=8,
      loss_agg_mode="sequence-mean-token-mean",
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  trainer = GRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=[vtc_reward],
      algo_config=grpo_config,
      chat_parser=chat_parser,
  )

  print(f"Devices: {jax.device_count()} | Mesh: {mesh}")
  print(f"Artifact root: {ARTIFACT_ROOT}")
  print(f"Checkpoint dir: {CHECKPOINT_ROOT}")
  print(
      "Config summary:",
      {
          "model": MODEL_ID,
          "batch_size": NUM_PROMPTS_PER_STEP,
          "num_generations": NUM_GENERATIONS,
          "eval_num_generations": 1,
          "mini_batch_groups": TRAIN_GLOBAL_BATCH_SIZE // NUM_GENERATIONS,
          "max_steps": MAX_STEPS,
          "max_total_sequence_length": MAX_TOTAL_SEQUENCE_LENGTH,
          "train_temperature": TRAIN_TEMPERATURE,
          "eval_temperature": EVAL_TEMPERATURE,
      },
  )

  with script_utils.profile_and_capture_log(
      "qwen3_grpo_gsm8k_vtc_demo", enable_profile=False
  ):
    trainer.train(train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
  main()
