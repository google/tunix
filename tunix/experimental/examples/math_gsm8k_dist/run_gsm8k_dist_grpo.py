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

"""CPU orchestrator for a minimal distributed GSM8K-style GRPO chain demo.

This script intentionally drives remote worker primitives directly:
  1. lifecycle handshake with trainer and rollout workers,
  2. rollout generation on the rollout worker,
  3. TrainExample assembly on the CPU orchestrator,
  4. trainer-side GRPO loss configuration and actor update,
  5. optional LoRA weight sync back to the rollout worker.

It is a plumbing demo, not a full-quality GSM8K training recipe.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.rl import function_registry  # pylint: disable=g-import-not-at-top
from tunix.rl.agentic import agentic_grpo_learner  # pylint: disable=g-import-not-at-top
from tunix.rl.agentic import utils as agentic_utils  # pylint: disable=g-import-not-at-top


PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags.

Problem: {question}
<reasoning>
"""

DEMO_TASKS = (
    ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she babysat for 3 hours. How much did she earn?", "36"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take?", "3"),
    ("Betty is saving money for a wallet which costs $100. She has $15 saved. How much more does she need?", "85"),
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Minimal distributed Qwen3 GRPO chain demo on remote workers."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=2,
      help="Number of prompt groups per step; trajectories=batch_size*num_generations.",
  )
  parser.add_argument("--num_generations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--trainer_addr", type=str, default="localhost:20000")
  parser.add_argument("--rollout_addr", type=str, default="localhost:20001")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--reward_mode",
      choices=("synthetic", "exact"),
      default="synthetic",
      help="synthetic proves the distributed chain without relying on model quality.",
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--sync_lora_weights", action="store_true")
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  return parser.parse_args()


def _connect(addr: str, timeout_s: float) -> remote_execution.ActorHandle:
  return remote_execution.GrpcRemoteActorHandle(
      target_address=f"grpc://{addr}", rpc_timeout_s=timeout_s
  )


def _build_prompts(batch_size: int, num_generations: int) -> tuple[list[str], list[str]]:
  prompts = []
  gold_answers = []
  for i in range(batch_size):
    question, answer = DEMO_TASKS[i % len(DEMO_TASKS)]
    prompt = PROMPT_TEMPLATE.format(question=question)
    for _ in range(num_generations):
      prompts.append(prompt)
      gold_answers.append(answer)
  return prompts, gold_answers


def _extract_answer(text: str) -> str | None:
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text
  boxed = re.search(r"\\boxed\s*\{([^{}]+)\}", content)
  if boxed:
    return boxed.group(1).strip().replace(",", "")
  numeric = re.findall(r"-?\d+(?:\.\d+)?", content)
  return numeric[-1].replace(",", "") if numeric else None


def _compute_rewards(
    completions: list[str],
    gold_answers: list[str],
    *,
    mode: str,
    num_generations: int,
) -> np.ndarray:
  if mode == "synthetic":
    per_group = np.linspace(0.0, 1.0, num_generations, dtype=np.float32)
    return np.tile(per_group, len(completions) // num_generations)

  rewards = []
  for completion, gold in zip(completions, gold_answers):
    rewards.append(1.0 if _extract_answer(completion) == gold else 0.0)
  return np.asarray(rewards, dtype=np.float32)


def _encode(tokenizer: Any, text: str) -> list[int]:
  try:
    return tokenizer.encode(text, add_special_tokens=False)
  except TypeError:
    return tokenizer.encode(text)


def _build_train_example(
    *,
    tokenizer: Any,
    prompts: list[str],
    completion_tokens: list[Any],
    rewards: np.ndarray,
    num_generations: int,
    max_prompt_length: int,
    max_response_length: int,
    pad_id: int,
) -> agentic_grpo_learner.TrainExample:
  padded_prompt_ids = []
  padded_completion_ids = []
  padded_completion_masks = []
  for prompt, tokens in zip(prompts, completion_tokens):
    prompt_ids = _encode(tokenizer, prompt)
    completion_ids = np.asarray(tokens, dtype=np.int32).tolist()
    padded_prompt, padded_completion, _ = agentic_utils.pad_prompt_and_completion(
        prompt_ids,
        completion_ids,
        max_prompt_length,
        max_response_length,
        pad_id,
    )
    padded_prompt_ids.append(padded_prompt)
    padded_completion_ids.append(padded_completion)
    padded_completion_masks.append((padded_completion != pad_id).astype(np.int32))

  advantage_fn = function_registry.get_advantage_estimator("grpo")
  advantages = advantage_fn(rewards, num_generations=num_generations)
  return agentic_grpo_learner.TrainExample(
      prompt_ids=np.asarray(padded_prompt_ids, dtype=np.int32),
      prompt_mask=np.asarray(padded_prompt_ids) != pad_id,
      completion_ids=np.asarray(padded_completion_ids, dtype=np.int32),
      completion_mask=np.asarray(padded_completion_masks, dtype=np.int32),
      advantages=np.asarray(advantages, dtype=np.float32),
      ref_per_token_logps=None,
      old_per_token_logps=None,
      policy_version=np.zeros((len(prompts),), dtype=np.int32),
  )


def _slice_or_none(value: Any, start: int, end: int) -> Any:
  if value is None:
    return None
  return value[start:end]


def _split_train_example(
    example: agentic_grpo_learner.TrainExample,
    micro_batch_size: int,
) -> list[agentic_grpo_learner.TrainExample]:
  batch = int(example.prompt_ids.shape[0])
  if micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  chunks = []
  for start in range(0, batch, micro_batch_size):
    end = min(start + micro_batch_size, batch)
    chunks.append(
        example.replace(
            prompt_ids=example.prompt_ids[start:end],
            prompt_mask=example.prompt_mask[start:end],
            completion_ids=example.completion_ids[start:end],
            completion_mask=example.completion_mask[start:end],
            advantages=example.advantages[start:end],
            ref_per_token_logps=_slice_or_none(
                example.ref_per_token_logps, start, end
            ),
            old_per_token_logps=_slice_or_none(
                example.old_per_token_logps, start, end
            ),
            policy_version=_slice_or_none(example.policy_version, start, end),
            sampler_is_weights=_slice_or_none(
                example.sampler_is_weights, start, end
            ),
        )
    )
  return chunks


def _maybe_add_ref_logps(
    trainer_handle: remote_execution.ActorHandle,
    example: agentic_grpo_learner.TrainExample,
    *,
    beta: float,
    pad_id: int,
    eos_id: int,
) -> agentic_grpo_learner.TrainExample:
  if beta == 0.0:
    return example
  ref_logps = trainer_handle.submit(
      "reference_logps",
      example.prompt_ids,
      example.completion_ids,
      pad_id,
      eos_id,
  )
  return example.replace(ref_per_token_logps=ref_logps)


def _log_lifecycle(name: str, handle: remote_execution.ActorHandle) -> None:
  init_resp = handle.submit("initialize")
  start_resp = handle.submit("start")
  info = handle.submit("info")
  heartbeat = handle.submit("heartbeat")
  logging.info("%s initialize: %s", name, getattr(init_resp, "metadata", init_resp))
  logging.info("%s start: %s", name, getattr(start_resp, "metadata", start_resp))
  logging.info("%s info: %s", name, info)
  logging.info("%s heartbeat: %s", name, heartbeat)


def main() -> None:
  args = _parse_args()
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")

  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - [Orchestrator] %(message)s"
  )
  logging.info("Orchestrator JAX backend: %s", jax.default_backend())

  tokenizer_path = args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  trainer_handle = _connect(args.trainer_addr, args.rpc_timeout_s)
  rollout_handle = _connect(args.rollout_addr, args.rpc_timeout_s)

  _log_lifecycle("trainer", trainer_handle)
  _log_lifecycle("rollout", rollout_handle)
  logging.info("Configuring trainer-side GRPO loss.")
  trainer_handle.submit(
      "configure_grpo_loss",
      num_generations=args.num_generations,
      max_response_length=args.max_response_length,
      beta=args.beta,
      epsilon=args.epsilon,
      temperature=args.temperature,
  )
  logging.info("Trainer-side GRPO loss configured.")

  top_k = None if args.top_k < 0 else args.top_k
  for step in range(args.max_steps):
    prompts, gold_answers = _build_prompts(args.batch_size, args.num_generations)
    logging.info(
        "Step %d: requesting %d trajectories from rollout worker.",
        step,
        len(prompts),
    )
    rollout_output = rollout_handle.submit(
        "generate",
        prompts,
        max_generation_steps=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )
    logging.info(
        "Step %d: rollout returned %d completions.",
        step,
        len(rollout_output.text),
    )
    rewards = _compute_rewards(
        rollout_output.text,
        gold_answers,
        mode=args.reward_mode,
        num_generations=args.num_generations,
    )
    train_example = _build_train_example(
        tokenizer=tokenizer,
        prompts=prompts,
        completion_tokens=rollout_output.tokens,
        rewards=rewards,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        pad_id=pad_id,
    )
    train_example = _maybe_add_ref_logps(
        trainer_handle,
        train_example,
        beta=args.beta,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    chunks = _split_train_example(train_example, args.train_micro_batch_size)
    logging.info(
        "Step %d: reward_mean=%.3f reward_std=%.3f chunks=%d.",
        step,
        float(rewards.mean()),
        float(rewards.std()),
        len(chunks),
    )
    for chunk in chunks:
      trainer_handle.submit("fwd_bwd", chunk)
    train_step = trainer_handle.submit("update", eval_ds=None, skip_jit=False)
    logging.info(
        "Step %d: trainer update finished at train_step=%s.", step, train_step
    )

    if args.sync_lora_weights:
      logging.info("Step %d: syncing LoRA weights to rollout worker.", step)
      trainer_handle.submit("prepare_weight_sync")
      lora_weights = trainer_handle.submit("get_lora_weights")
      rollout_handle.submit("pre_weight_sync")
      rollout_handle.submit("weight_sync", lora_weights)
      rollout_handle.submit("post_weight_sync")
    else:
      logging.info(
          "Step %d: running rollout weight-sync barrier without LoRA weights.",
          step,
      )
      rollout_handle.submit("pre_weight_sync")
      rollout_handle.submit("weight_sync")
      rollout_handle.submit("post_weight_sync")
    logging.info("Step %d: distributed train+sync chain completed.", step)

  if args.stop_workers_on_exit:
    rollout_handle.submit("stop")
    trainer_handle.submit("stop")

  logging.info("Distributed GRPO chain demo finished successfully.")


if __name__ == "__main__":
  main()
