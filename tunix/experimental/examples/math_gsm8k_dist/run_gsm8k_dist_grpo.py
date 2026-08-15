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

"""CPU control-plane for a minimal Orchestrator V2 GSM8K GRPO demo.

The TPU worker processes host the expensive pieces:
  1. a TrainerWorker backed by experimental PeftTrainer V2,
  2. a vLLM RolloutWorker,
  3. optionally an InferenceWorker for frozen reference log-probs.

This process only owns Orchestrator V2 control flow. It registers remote worker
handles with ClusterOrchestrator, configures the GRPO loss on the trainer worker,
and executes SyncRLProgram through ClusterOrchestrator.run_program().
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import dataclasses
import functools
import json
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top


BUILTIN_TASKS = (
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in "
        "April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 3 "
        "hours. How much did she earn?",
        "36",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts of fiber does it take?",
        "3",
    ),
    (
        "Betty is saving money for a wallet which costs $100. She has $15 "
        "saved. How much more does she need?",
        "85",
    ),
)

# PeftTrainer V2 does not yet export rollout-ready weight-sync payloads, so this
# demo intentionally keeps rollout weights fixed while validating the E2E loop.
ENABLE_WEIGHT_SYNC = False


@dataclasses.dataclass(frozen=True)
class GSM8KExample:
  prompt: str
  question: str
  answer: str


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Minimal Orchestrator V2 Qwen3 GSM8K GRPO demo."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=2,
      help="Number of prompt groups per step.",
  )
  parser.add_argument("--num_generations", type=int, default=4)
  parser.add_argument("--max_steps", type=int, default=10)
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--trainer_addr", type=str, default="localhost:20000")
  parser.add_argument("--rollout_addr", type=str, default="localhost:20001")
  parser.add_argument(
      "--inference_addr",
      type=str,
      default="",
      help=(
          "Optional reference InferenceWorker address. Required when --beta is "
          "non-zero because KL scoring needs a reference worker."
      ),
  )
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--dataset_source",
      choices=("huggingface", "local_jsonl", "builtin"),
      default="huggingface",
      help=(
          "Where to load GSM8K prompts from. 'huggingface' matches the math "
          "agent recipe; 'builtin' is only for offline smoke tests."
      ),
  )
  parser.add_argument("--dataset_name", type=str, default="openai/gsm8k:main")
  parser.add_argument("--dataset_split", type=str, default="train")
  parser.add_argument("--dataset_path", type=str, default="")
  parser.add_argument("--dataset_seed", type=int, default=42)
  parser.add_argument("--prompt_key", type=str, default="question")
  parser.add_argument("--answer_key", type=str, default="answer")
  parser.add_argument(
      "--reward_mode",
      choices=("env", "math_gsm8k", "exact", "synthetic"),
      default="env",
      help=(
          "'env' uses the registered GSM8KEnv reward; 'math_gsm8k' "
          "recomputes the same math GSM8K reward in the orchestrator; "
          "'synthetic' is debug-only."
      ),
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  return parser.parse_args()


def _connect(addr: str, timeout_s: float) -> remote_execution.ActorHandle:
  return remote_execution.ActorHandle.from_address(
      f"grpc://{addr}", rpc_timeout_s=timeout_s
  )


def _parse_hf_dataset_name(dataset_name: str) -> tuple[str, str | None]:
  if ":" in dataset_name:
    name, config_name = dataset_name.split(":", maxsplit=1)
    return name, config_name or None
  if "/" in dataset_name:
    return dataset_name, "default"
  return dataset_name, None


def _to_text(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return str(value)


def _extract_dataset_answer(answer_value: Any) -> str:
  answer_text = _to_text(answer_value)
  answer = gsm8k.extract_hash_answer(answer_text)
  if answer is not None:
    return answer
  normalized = gsm8k.normalize_answer(answer_text)
  return normalized or answer_text


def _load_builtin_examples() -> list[GSM8KExample]:
  return [
      GSM8KExample(
          prompt=gsm8k.build_prompt(question),
          question=question,
          answer=answer,
      )
      for question, answer in BUILTIN_TASKS
  ]


def _load_local_jsonl_examples(args: argparse.Namespace) -> list[GSM8KExample]:
  if not args.dataset_path:
    raise ValueError("--dataset_path is required for --dataset_source=local_jsonl")
  examples = []
  with open(args.dataset_path, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      question = _to_text(row[args.prompt_key])
      answer = _extract_dataset_answer(row[args.answer_key])
      examples.append(
          GSM8KExample(
              prompt=gsm8k.build_prompt(question),
              question=question,
              answer=answer,
          )
      )
  return examples


def _load_huggingface_examples(args: argparse.Namespace) -> list[GSM8KExample]:
  import datasets as hf_datasets  # pylint: disable=g-import-not-at-top

  dataset_name, config_name = _parse_hf_dataset_name(args.dataset_name)
  logging.info(
      "Loading GSM8K dataset from Hugging Face: name=%s config=%s split=%s",
      dataset_name,
      config_name,
      args.dataset_split,
  )
  dataset = hf_datasets.load_dataset(
      dataset_name,
      config_name,
      split=args.dataset_split,
  )
  dataset = dataset.shuffle(seed=args.dataset_seed)
  examples = []
  for row in dataset:
    question = _to_text(row[args.prompt_key])
    answer = _extract_dataset_answer(row[args.answer_key])
    examples.append(
        GSM8KExample(
            prompt=gsm8k.build_prompt(question),
            question=question,
            answer=answer,
        )
    )
  return examples


def _load_examples(args: argparse.Namespace) -> list[GSM8KExample]:
  if args.dataset_source == "builtin":
    examples = _load_builtin_examples()
  elif args.dataset_source == "local_jsonl":
    examples = _load_local_jsonl_examples(args)
  elif args.dataset_source == "huggingface":
    examples = _load_huggingface_examples(args)
  else:
    raise ValueError(f"Unsupported dataset_source: {args.dataset_source}")
  if not examples:
    raise ValueError("GSM8K demo dataset is empty.")
  logging.info(
      "Loaded %d GSM8K examples from %s; first answer=%s",
      len(examples),
      args.dataset_source,
      examples[0].answer,
  )
  return examples


def _select_step_examples(
    examples: Sequence[GSM8KExample],
    *,
    step: int,
    batch_size: int,
) -> list[GSM8KExample]:
  start = step * batch_size
  return [examples[(start + i) % len(examples)] for i in range(batch_size)]


def _item_env_reward(item: datatypes.TrajectoryItem) -> float:
  traj = getattr(item, "traj", None)
  reward = getattr(traj, "reward", None)
  if reward is None and item.metadata:
    reward = item.metadata.get("reward")
  return float(reward or 0.0)


def _make_reward_fn(mode: str, num_generations: int):
  """Creates the per-trajectory reward function used by SyncRLProgram."""

  def reward_fn(item: datatypes.TrajectoryItem) -> float:
    metadata = dict(item.metadata or {})
    if mode == "synthetic":
      pair_index = int(metadata.get("pair_index", item.pair_index))
      return pair_index / max(num_generations - 1, 1)
    if mode == "env":
      return _item_env_reward(item)

    text = str(metadata.get("text", ""))
    gold_answer = metadata.get("gold_answer")
    score, _, answer_ok, _ = gsm8k.score_completion(text, gold_answer)
    if mode == "exact":
      return 1.0 if answer_ok else 0.0
    return score

  return reward_fn


def _grpo_model_input(
    train_example: Any,
    *,
    algo_config: Any,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
  """Maps a TrainExample microbatch to algo_core.grpo_loss_fn kwargs."""
  return {
      "train_example": train_example,
      "algo_config": algo_config,
      "pad_id": pad_id,
      "eos_id": eos_id,
  }


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  algo = algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      mini_batch_size=args.batch_size * args.num_generations,
      max_packed_len=args.max_prompt_length + args.max_response_length,
      clip_epsilon=args.epsilon,
      beta_kl=args.beta,
  )
  return algo


def _build_grpo_config(args: argparse.Namespace) -> Any:
  return SimpleNamespace(
      beta=args.beta,
      epsilon=args.epsilon,
      loss_algo="grpo",
      loss_agg_mode="sequence-mean-token-mean",
      temperature=args.temperature,
      kl_loss_mode="mse_kl",
      kl_clamp_value=None,
  )


def _configure_trainer_loss(
    trainer_handle: remote_execution.ActorHandle,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    grpo_config: Any,
    pad_id: int,
    eos_id: int,
) -> None:
  logging.info("Configuring trainer-side GRPO loss via TrainerWorker RPC.")
  trainer_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
  trainer_handle.submit(
      "with_gen_model_input_fn",
      functools.partial(
          _grpo_model_input,
          algo_config=grpo_config,
          pad_id=pad_id,
          eos_id=eos_id,
      ),
  )


def _register_workers(
    args: argparse.Namespace,
    *,
    cluster: orchestrator.ClusterOrchestrator,
    trainer_handle: remote_execution.ActorHandle,
    rollout_handle: remote_execution.ActorHandle,
    inference_handle: remote_execution.ActorHandle | None,
) -> None:
  """Registers gRPC-backed workers in the Orchestrator V2 registry."""
  cluster.register_worker_handle(
      worker_id="trainer-0",
      roles=[datatypes.Role.ACTOR],
      handle=trainer_handle,
      resources={"address": args.trainer_addr},
  )
  cluster.register_worker_handle(
      worker_id="rollout-0",
      roles=[datatypes.Role.ROLLOUT],
      handle=rollout_handle,
      resources={"address": args.rollout_addr},
  )
  if inference_handle is not None:
    cluster.register_worker_handle(
        worker_id="reference-0",
        roles=[datatypes.Role.REFERENCE],
        handle=inference_handle,
        resources={"address": args.inference_addr},
    )


def _build_step_requests(
    *,
    step: int,
    examples: Sequence[GSM8KExample],
    num_generations: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> list[datatypes.RolloutRequest]:
  requests = []
  for prompt_idx, example in enumerate(examples):
    prompt_id = f"step_{step}_prompt_{prompt_idx}"
    for generation_idx in range(num_generations):
      requests.append(
          datatypes.RolloutRequest(
              request_id=f"{prompt_id}_gen_{generation_idx}",
              prompt=example.prompt,
              prompt_id=prompt_id,
              group_offset_id=str(generation_idx),
              target_policy_version=step,
              generation_kwargs={
                  "max_generation_steps": max_response_length,
                  "temperature": temperature,
                  "top_p": top_p,
                  "top_k": top_k,
                  "return_logprobs": True,
              },
              metadata={
                  "group_id": prompt_id,
                  "pair_index": generation_idx,
                  "gold_answer": example.answer,
                  "question": example.question,
                  "prefix_hash": prompt_id,
                  "env_config": {
                      "prompt": example.prompt,
                      "gold_answer": example.answer,
                      "group_id": prompt_id,
                      "pair_index": generation_idx,
                      "policy_version": step,
                      "max_steps": 1,
                  },
              },
          )
      )
  return requests


def _iter_request_batches(
    args: argparse.Namespace,
    examples: Sequence[GSM8KExample],
) -> Iterator[list[datatypes.RolloutRequest]]:
  top_k = None if args.top_k < 0 else args.top_k
  for step in range(args.max_steps):
    step_examples = _select_step_examples(
        examples, step=step, batch_size=args.batch_size
    )
    yield _build_step_requests(
        step=step,
        examples=step_examples,
        num_generations=args.num_generations,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )


def main() -> None:
  args = _parse_args()
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.beta != 0.0 and not args.inference_addr:
    raise ValueError(
        "--inference_addr is required when --beta is non-zero because "
        "reference KL scoring must route to an InferenceWorker."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [OrchestratorV2] %(message)s",
      force=True,
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())
  logging.info("Weight sync enabled: %s", ENABLE_WEIGHT_SYNC)
  logging.info(
      "Demo training plan: steps=%d prompt_groups_per_step=%d "
      "num_generations=%d trajectories_per_step=%d reward_mode=%s",
      args.max_steps,
      args.batch_size,
      args.num_generations,
      args.batch_size * args.num_generations,
      args.reward_mode,
  )
  trajectories_per_step = args.batch_size * args.num_generations
  microbatches_per_step = (
      trajectories_per_step + args.train_micro_batch_size - 1
  ) // args.train_micro_batch_size

  tokenizer_path = args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  trainer_handle = _connect(args.trainer_addr, args.rpc_timeout_s)
  rollout_handle = _connect(args.rollout_addr, args.rpc_timeout_s)
  inference_handle = (
      _connect(args.inference_addr, args.rpc_timeout_s)
      if args.inference_addr
      else None
  )

  algo = _build_algo(args)
  grpo_config = _build_grpo_config(args)
  examples = _load_examples(args)
  _configure_trainer_loss(
      trainer_handle,
      algo=algo,
      grpo_config=grpo_config,
      pad_id=pad_id,
      eos_id=eos_id,
  )

  cluster = orchestrator.ClusterOrchestrator()
  _register_workers(
      args,
      cluster=cluster,
      trainer_handle=trainer_handle,
      rollout_handle=rollout_handle,
      inference_handle=inference_handle,
  )
  logging.info("Registered Orchestrator V2 workers: %s", cluster.worker_infos())

  program = rl_program.SyncRLProgram(
      algo=algo,
      reward_fns=[_make_reward_fn(args.reward_mode, args.num_generations)],
      assembler=batch_assembly.GRPOTrainExampleAssembler(
          batch_size=args.train_micro_batch_size,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          pad_id=pad_id,
      ),
      sync_weights=ENABLE_WEIGHT_SYNC,
      on_step_begin=lambda step: logging.info(
          "GRPO step %d/%d starting.",
          step + 1,
          args.max_steps,
      ),
      on_step_end=lambda step, result: logging.info(
          "GRPO step %d/%d finished: policy_version=%d "
          "expected_rollouts=%d expected_microbatches=%d train_result=%s.",
          step,
          args.max_steps,
          step,
          trajectories_per_step,
          microbatches_per_step,
          result,
      ),
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator.")
    cluster.bring_up_workers(dummy_data=None)
    logging.info("Running SyncRLProgram through ClusterOrchestrator.run_program.")
    cluster.run_program(
        program=program,
        train_dataset=_iter_request_batches(args, examples),
        num_steps=args.max_steps,
        bring_up=False,
    )
  finally:
    if args.stop_workers_on_exit:
      cluster.shutdown()
    else:
      cluster.monitor.close()

  result = program.last_step_result
  if result is not None:
    logging.info(
        "Final step summary: step=%d policy_version=%d rollouts=%d "
        "microbatches=%d reward_mean=%.3f reward_std=%.3f.",
        result.step,
        result.policy_version,
        result.num_rollouts,
        result.num_microbatches,
        result.reward_mean,
        result.reward_std,
    )
  logging.info("Distributed GSM8K GRPO Orchestrator V2 demo finished.")


if __name__ == "__main__":
  main()
