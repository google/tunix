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

"""CPU orchestrator for a minimal distributed GSM8K-style GRPO v2 demo.

This entrypoint is intentionally centered on Orchestrator V2:
  1. remote trainer / rollout gRPC handles are registered as v2 Workers,
  2. ClusterOrchestrator owns lifecycle, registry, and engine creation,
  3. DistributedRLEngine routes rollout, trainer, and weight-sync calls,
  4. RLProgram coordinates rollout -> reward -> GRPO payload -> train -> sync.

The demo is a plumbing proof for one trainer process plus one vLLM rollout
process. It is not a full-quality GSM8K recipe.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
import re
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
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator as orchestrator_v2  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import remote_worker  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.rl import algo_core  # pylint: disable=g-import-not-at-top
from tunix.rl import common as rl_common  # pylint: disable=g-import-not-at-top


PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags.

Problem: {question}
<reasoning>
"""

DEMO_TASKS = (
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April "
        "and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 3 "
        "hours. How much did she earn?",
        "36",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
        "many bolts of fiber does it take?",
        "3",
    ),
    (
        "Betty is saving money for a wallet which costs $100. She has $15 "
        "saved. How much more does she need?",
        "85",
    ),
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Minimal distributed Qwen3 GRPO v2 chain demo."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=2,
      help="Number of prompt groups per step.",
  )
  parser.add_argument("--num_generations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--compute_logps_chunk_size", type=int, default=0)
  parser.add_argument("--trainer_addr", type=str, default="localhost:20000")
  parser.add_argument("--rollout_addr", type=str, default="localhost:20001")
  parser.add_argument(
      "--inference_addr",
      type=str,
      default="",
      help=(
          "Optional reference InferenceWorker address. Required when --beta is "
          "non-zero because reference KL scoring must be available."
      ),
  )
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument("--kl_loss_mode", type=str, default="mse_kl")
  parser.add_argument(
      "--loss_agg_mode",
      type=str,
      default="sequence-mean-token-mean",
  )
  parser.add_argument(
      "--reward_mode",
      choices=("synthetic", "exact"),
      default="synthetic",
      help="synthetic proves the distributed chain without model-quality noise.",
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
    prompts.append(PROMPT_TEMPLATE.format(question=question))
    gold_answers.append(answer)
  return prompts, gold_answers


def _build_rollout_requests(
    *,
    step: int,
    prompts: Sequence[str],
    gold_answers: Sequence[str],
    num_generations: int,
    policy_version: int,
) -> list[datatypes.RolloutRequest]:
  requests = []
  for prompt_idx, (prompt, gold) in enumerate(zip(prompts, gold_answers)):
    prompt_id = f"step_{step}_prompt_{prompt_idx}"
    for gen_idx in range(num_generations):
      requests.append(
          datatypes.RolloutRequest(
              request_id=f"{prompt_id}_gen_{gen_idx}",
              prompt=prompt,
              prompt_id=prompt_id,
              group_offset_id=str(gen_idx),
              target_policy_version=policy_version,
              metadata={
                  "group_id": prompt_id,
                  "pair_index": gen_idx,
                  "gold_answer": gold,
                  "prompt_text": prompt,
              },
          )
      )
  return requests


def _extract_answer(text: str) -> str | None:
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text
  boxed = re.search(r"\\boxed\s*\{([^{}]+)\}", content)
  if boxed:
    return boxed.group(1).strip().replace(",", "")
  numeric = re.findall(r"-?\d+(?:\.\d+)?", content)
  return numeric[-1].replace(",", "") if numeric else None


def _build_reward_fn(args: argparse.Namespace):
  def _reward(item: datatypes.TrajectoryItem) -> float:
    pair_index = int(getattr(item, "pair_index", 0))
    if args.reward_mode == "synthetic":
      denom = max(1, args.num_generations - 1)
      return float(pair_index / denom)

    metadata = dict(getattr(item, "metadata", {}) or {})
    completion_text = str(metadata.get("text", ""))
    gold_answer = str(metadata.get("gold_answer", ""))
    return 1.0 if _extract_answer(completion_text) == gold_answer else 0.0

  return _reward


def _configure_trainer_grpo_loss(
    *,
    trainer_handle: remote_execution.ActorHandle,
    args: argparse.Namespace,
    pad_id: int,
    eos_id: int,
) -> None:
  config = SimpleNamespace(
      beta=args.beta,
      epsilon=args.epsilon,
      epsilon_high=args.epsilon,
      epsilon_c=None,
      loss_algo="grpo",
      loss_agg_mode=args.loss_agg_mode,
      temperature=args.temperature,
      kl_loss_mode=args.kl_loss_mode,
      kl_clamp_value=None,
  )

  def loss_fn(model: Any, train_example: rl_common.TrainExample):
    return algo_core.grpo_loss_fn(
        model,
        train_example,
        algo_config=config,
        pad_id=pad_id,
        eos_id=eos_id,
        compute_logps_chunk_size=args.compute_logps_chunk_size,
    )

  trainer_handle.submit("with_loss_fn", loss_fn, True)
  trainer_handle.submit(
      "with_gen_model_input_fn", lambda train_example: {
          "train_example": train_example
      }
  )
  logging.info("Configured trainer-side GRPO loss for PeftTrainer v2.")


def _register_remote_workers(
    *,
    orch: orchestrator_v2.ClusterOrchestrator,
    trainer_handle: remote_execution.ActorHandle,
    rollout_handle: remote_execution.ActorHandle,
    inference_handle: remote_execution.ActorHandle | None,
    args: argparse.Namespace,
    pad_id: int,
    eos_id: int,
) -> None:
  common_kwargs = dict(
      pad_id=pad_id,
      eos_id=eos_id,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      temperature=args.temperature,
  )
  trainer = remote_worker.RemoteActorWorker(
      worker_id="remote-trainer-actor",
      roles=(datatypes.Role.ACTOR,),
      handle=trainer_handle,
      **common_kwargs,
  )
  rollout = remote_worker.RemoteActorWorker(
      worker_id="remote-vllm-rollout",
      roles=(datatypes.Role.ROLLOUT,),
      handle=rollout_handle,
      **common_kwargs,
  )
  orch.register_worker(trainer)
  orch.register_worker(rollout)
  if inference_handle is not None:
    reference = remote_worker.RemoteActorWorker(
        worker_id="remote-reference",
        roles=(datatypes.Role.REFERENCE,),
        handle=inference_handle,
        **common_kwargs,
    )
    orch.register_worker(reference)


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
      level=logging.INFO, format="%(asctime)s - [OrchestratorV2] %(message)s"
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

  _configure_trainer_grpo_loss(
      trainer_handle=trainer_handle,
      args=args,
      pad_id=pad_id,
      eos_id=eos_id,
  )

  algo = algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      batch_size_groups=args.batch_size,
      max_packed_len=args.max_prompt_length + args.max_response_length,
      clip_epsilon=args.epsilon,
      beta_kl=args.beta,
  )
  algo.requires_reference_kl = args.beta != 0.0
  assembler = batch_assembly.GRPOTrainExampleAssembler(
      batch_size=args.train_micro_batch_size,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      pad_id=pad_id,
  )

  orch = orchestrator_v2.ClusterOrchestrator()
  _register_remote_workers(
      orch=orch,
      trainer_handle=trainer_handle,
      rollout_handle=rollout_handle,
      inference_handle=inference_handle,
      args=args,
      pad_id=pad_id,
      eos_id=eos_id,
  )
  logging.info("Registered v2 workers with roles: %s", sorted(orch.registry.roles()))
  orch.bring_up_workers(dummy_data=None)
  if orch.engine is None:
    raise RuntimeError("ClusterOrchestrator failed to create a v2 engine.")

  program = rl_program.RLProgram(
      engine=orch.engine,
      algo=algo,
      reward_fns=[_build_reward_fn(args)],
      assembler=assembler,
      sync_weights=args.sync_lora_weights,
  )

  top_k = None if args.top_k < 0 else args.top_k
  for step in range(args.max_steps):
    prompt_groups, gold_answers = _build_prompt_groups(args.batch_size)
    rollout_requests = _build_rollout_requests(
        step=step,
        prompts=prompt_groups,
        gold_answers=gold_answers,
        num_generations=args.num_generations,
        policy_version=program.policy_version,
    )
    logging.info(
        "Step %d: requesting %d prompt groups / %d trajectories.",
        step,
        len(prompt_groups),
        len(rollout_requests),
    )
    train_result = program.step_once(
        rollout_requests,
        max_generation_steps=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )
    result = program.last_step_result
    if result is None:
      raise RuntimeError("RLProgram did not report step stats.")
    logging.info(
        "Step %d: reward_mean=%.3f reward_std=%.3f microbatches=%d.",
        result.step,
        result.reward_mean,
        result.reward_std,
        result.num_microbatches,
    )
    logging.info(
        "Step %d: trainer result=%s policy_version=%d.",
        result.step,
        train_result,
        result.policy_version,
    )

  if args.stop_workers_on_exit:
    orch.shutdown()

  logging.info("Distributed GRPO v2 chain demo finished successfully.")


if __name__ == "__main__":
  main()
