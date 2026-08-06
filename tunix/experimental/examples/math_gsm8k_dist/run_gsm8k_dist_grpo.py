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

This script intentionally drives the training loop through RLOrchestrator:
  1. lifecycle handshake with trainer and rollout workers,
  2. remote-only OrchestratorRLEngine over gRPC worker proxies,
  3. rollout generation on the rollout worker,
  4. TrainExample assembly through the GRPO algorithm adapter,
  5. trainer-side GRPO loss configuration and actor update,
  6. optional LoRA weight sync back to the rollout worker.

It is a plumbing demo, not a full-quality GSM8K training recipe.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from types import SimpleNamespace
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

from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import grpo_loop  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import grpc_worker_proxies  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator_rl_engine  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_orchestrator  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.rl import rl_cluster as rl_cluster_lib  # pylint: disable=g-import-not-at-top
from tunix.rl.agentic import agentic_grpo_learner  # pylint: disable=g-import-not-at-top
from tunix.rl.rollout import base_rollout  # pylint: disable=g-import-not-at-top


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
  parser.add_argument(
      "--inference_addr",
      type=str,
      default="",
      help=(
          "Optional reference/reward InferenceWorker address. Required when "
          "--beta is non-zero or force reference KL scoring is enabled."
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


def _build_prompt_groups(batch_size: int) -> tuple[list[str], list[str]]:
  prompts = []
  gold_answers = []
  for i in range(batch_size):
    question, answer = DEMO_TASKS[i % len(DEMO_TASKS)]
    prompt = PROMPT_TEMPLATE.format(question=question)
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
    prompts: list[str],
    completions: list[str],
    gold_answers: list[str],
    *,
    mode: str,
    **kwargs,
) -> np.ndarray:
  del prompts
  if mode == "synthetic":
    num_generations = int(kwargs["num_generations"])
    per_group = np.linspace(0.0, 1.0, num_generations, dtype=np.float32)
    return np.tile(per_group, len(completions) // num_generations)

  rewards = []
  for completion, gold in zip(completions, gold_answers):
    rewards.append(1.0 if _extract_answer(completion) == gold else 0.0)
  return np.asarray(rewards, dtype=np.float32)


def _build_cluster_config(args: argparse.Namespace) -> Any:
  top_k = None if args.top_k < 0 else args.top_k
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=top_k,
      return_logprobs=True,
  )
  return SimpleNamespace(
      training_config=SimpleNamespace(
          train_micro_batch_size=args.train_micro_batch_size,
          compute_logps_micro_batch_size=args.train_micro_batch_size,
          compute_logps_chunk_size=0,
      ),
      rollout_config={
          rl_cluster_lib.Mode.TRAIN: rollout_config,
          rl_cluster_lib.Mode.EVAL: rollout_config,
      },
  )


def _build_orchestrator(
    args: argparse.Namespace,
    tokenizer: Any,
    trainer_handle: remote_execution.ActorHandle,
    rollout_handle: remote_execution.ActorHandle,
    inference_handle: remote_execution.ActorHandle | None,
    pad_id: int,
    eos_id: int,
) -> rl_orchestrator.RLOrchestrator:
  cluster_config = _build_cluster_config(args)
  trainer_proxy = grpc_worker_proxies.GrpcTrainerWorkerProxy(trainer_handle)
  rollout_proxy = grpc_worker_proxies.GrpcRolloutWorkerProxy(
      rollout_handle, cluster_config
  )
  inference_proxy = (
      grpc_worker_proxies.GrpcInferenceWorkerProxy(inference_handle)
      if inference_handle is not None
      else None
  )
  weight_sync_proxy = grpc_worker_proxies.GrpcWeightSyncProxy(
      trainer_handle,
      rollout_handle,
      sync_lora_weights=args.sync_lora_weights,
  )
  cluster = orchestrator_rl_engine.OrchestratorRLEngine(
      trainer_worker=trainer_proxy,
      rollout_worker=rollout_proxy,
      inference_worker=inference_proxy,
      weight_sync=weight_sync_proxy,
      cluster_config=cluster_config,
      actor_trainer=grpc_worker_proxies.RemoteActorTrainerProxy(trainer_handle),
      tokenizer=tokenizer,
      pad_id=pad_id,
      eos_id=eos_id,
  )
  grpo_config = agentic_grpo_learner.GRPOConfig(
      num_generations=args.num_generations,
      num_iterations=1,
      beta=args.beta,
      kl_loss_mode="mse_kl",
      epsilon=args.epsilon,
      max_response_length=args.max_response_length,
      use_rollout_logps=True,
  )
  grpo_config.temperature = args.temperature
  return rl_orchestrator.RLOrchestrator(
      cluster, algorithm_adapter.GRPOAdapter(grpo_config)
  )


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
  if args.beta != 0.0 and not args.inference_addr:
    raise ValueError(
        "--inference_addr is required when --beta is non-zero because "
        "reference KL scoring must route to an InferenceWorker."
    )

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
  inference_handle = (
      _connect(args.inference_addr, args.rpc_timeout_s)
      if args.inference_addr
      else None
  )

  _log_lifecycle("trainer", trainer_handle)
  _log_lifecycle("rollout", rollout_handle)
  if inference_handle is not None:
    _log_lifecycle("inference", inference_handle)
  orchestrator = _build_orchestrator(
      args,
      tokenizer,
      trainer_handle,
      rollout_handle,
      inference_handle,
      pad_id,
      eos_id,
  )
  logging.info("Creating formal GRPO loop on top of RLOrchestrator.")
  loop = grpo_loop.GRPOLoop(
      orchestrator,
      reward_fn=_compute_rewards,
      tokenizer=tokenizer,
      num_generations=args.num_generations,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      train_micro_batch_size=args.train_micro_batch_size,
      pad_id=pad_id,
      eos_id=eos_id,
      sync_weights=True,
  )
  logging.info("Trainer-side GRPO objective configured.")

  for step in range(args.max_steps):
    prompt_groups, gold_answers = _build_prompt_groups(args.batch_size)
    logging.info(
        "Step %d: requesting %d prompt groups / %d trajectories.",
        step,
        len(prompt_groups),
        len(prompt_groups) * args.num_generations,
    )
    result = loop.train_step(
        prompt_groups,
        reward_kwargs={
            "gold_answers": gold_answers,
            "mode": args.reward_mode,
            "num_generations": args.num_generations,
        },
        step=step,
        eval_ds=None,
        skip_jit=False,
    )
    logging.info(
        "Step %d: reward_mean=%.3f reward_std=%.3f chunks=%d.",
        result.step,
        result.reward_mean,
        result.reward_std,
        result.num_chunks,
    )
    logging.info(
        "Step %d: trainer update finished at train_step=%s.",
        result.step,
        result.train_step,
    )
    logging.info(
        "Step %d: distributed train+sync chain completed at global_step=%d.",
        result.step,
        result.global_step,
    )

  if args.stop_workers_on_exit:
    rollout_handle.submit("stop")
    trainer_handle.submit("stop")

  logging.info("Distributed GRPO chain demo finished successfully.")


if __name__ == "__main__":
  main()
