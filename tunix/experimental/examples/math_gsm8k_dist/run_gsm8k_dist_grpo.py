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
import functools
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
from tunix.experimental.orchestrator import orchestrator  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import abstract_worker  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top


PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags.

Problem: {question}
<reasoning>
"""

DEMO_TASKS = (
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
      "--reward_mode",
      choices=("synthetic", "exact"),
      default="synthetic",
      help="synthetic proves the distributed chain without relying on quality.",
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  return parser.parse_args()


def _connect(addr: str, timeout_s: float) -> remote_execution.ActorHandle:
  return remote_execution.ActorHandle.from_address(
      f"grpc://{addr}", rpc_timeout_s=timeout_s
  )


def _role_names(roles: Sequence[datatypes.Role | str]) -> frozenset[str]:
  return frozenset(
      role.value if isinstance(role, datatypes.Role) else role
      for role in roles
  )


class _RemoteWorkerRef(abstract_worker.Worker):
  """Local Orchestrator registry entry for a remote_execution actor handle."""

  def __init__(
      self,
      *,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      resources: dict[str, Any] | None = None,
  ):
    self._handle = handle
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=_role_names(roles),
        resources={"remote": True, **dict(resources or {})},
    )
    self._state = datatypes.WorkerState.PENDING

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def initialize(self) -> datatypes.Response:
    self.state = datatypes.WorkerState.INITIALIZING
    try:
      response = self._handle.submit("initialize")
    except Exception:
      self.state = datatypes.WorkerState.ERROR
      raise
    self.state = datatypes.WorkerState.READY
    return response

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    if self.state == datatypes.WorkerState.PENDING:
      self.initialize()
    self.state = datatypes.WorkerState.COMPILING
    try:
      response = self._handle.submit("compile", dummy_data)
    except Exception:
      self.state = datatypes.WorkerState.ERROR
      raise
    self.state = datatypes.WorkerState.READY
    return response

  def start(self) -> datatypes.Response:
    if self.state == datatypes.WorkerState.PENDING:
      self.initialize()
    return self._handle.submit("start")

  def stop(self) -> datatypes.Response:
    response = self._handle.submit("stop")
    self.state = datatypes.WorkerState.STOPPED
    return response

  def heartbeat(self) -> datatypes.HealthReport:
    return self._handle.submit("heartbeat")

  def submit(self, method_name: str | None = None, *args: Any, **kwargs: Any):
    return self._handle.submit(method_name, *args, **kwargs)

  async def asubmit(
      self, method_name: str | None = None, *args: Any, **kwargs: Any
  ):
    return await self._handle.asubmit(method_name, *args, **kwargs)

  async def dispatch_task(
      self,
      request_id: str | None = None,
      method_name: str | None = None,
      *args: Any,
      **kwargs: Any,
  ) -> str:
    return await self._handle.dispatch_task(
        request_id, method_name, *args, **kwargs
    )

  async def poll_responses(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> remote_execution.ExecutionResponse | None:
    return await self._handle.poll_responses(timeout_s=timeout_s)


def _build_prompt_groups(batch_size: int) -> tuple[list[str], list[str]]:
  prompts = []
  gold_answers = []
  for i in range(batch_size):
    question, answer = DEMO_TASKS[i % len(DEMO_TASKS)]
    prompts.append(PROMPT_TEMPLATE.format(question=question))
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


def _make_reward_fn(mode: str, num_generations: int):
  """Creates the per-trajectory reward function used by SyncRLProgram."""

  def reward_fn(item: datatypes.TrajectoryItem) -> float:
    metadata = dict(item.metadata or {})
    if mode == "synthetic":
      pair_index = int(metadata.get("pair_index", item.pair_index))
      return pair_index / max(num_generations - 1, 1)

    text = str(metadata.get("text", ""))
    gold_answer = metadata.get("gold_answer")
    return 1.0 if gold_answer and _extract_answer(text) == gold_answer else 0.0

  return reward_fn


def _grpo_model_input(
    train_example: Any,
    *,
    algo_config: Any,
    pad_id: int,
    eos_id: int,
    compute_logps_chunk_size: int,
) -> dict[str, Any]:
  """Maps a TrainExample microbatch to algo_core.grpo_loss_fn kwargs."""
  return {
      "train_example": train_example,
      "algo_config": algo_config,
      "pad_id": pad_id,
      "eos_id": eos_id,
      "compute_logps_chunk_size": compute_logps_chunk_size,
  }


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  algo = algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      mini_batch_size=args.batch_size,
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
          compute_logps_chunk_size=0,
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
  cluster.register_worker(
      _RemoteWorkerRef(
          worker_id="trainer-0",
          roles=[datatypes.Role.ACTOR],
          handle=trainer_handle,
          resources={"address": args.trainer_addr},
      )
  )
  cluster.register_worker(
      _RemoteWorkerRef(
          worker_id="rollout-0",
          roles=[datatypes.Role.ROLLOUT],
          handle=rollout_handle,
          resources={"address": args.rollout_addr},
      )
  )
  if inference_handle is not None:
    cluster.register_worker(
        _RemoteWorkerRef(
            worker_id="reference-0",
            roles=[datatypes.Role.REFERENCE],
            handle=inference_handle,
            resources={"address": args.inference_addr},
        )
    )


def _build_step_requests(
    *,
    step: int,
    batch_size: int,
    num_generations: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> list[datatypes.RolloutRequest]:
  prompts, gold_answers = _build_prompt_groups(batch_size)
  requests = []
  for prompt_idx, (prompt, gold_answer) in enumerate(zip(prompts, gold_answers)):
    prompt_id = f"step_{step}_prompt_{prompt_idx}"
    for generation_idx in range(num_generations):
      requests.append(
          datatypes.RolloutRequest(
              request_id=f"{prompt_id}_gen_{generation_idx}",
              prompt=prompt,
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
                  "gold_answer": gold_answer,
                  "prefix_hash": prompt_id,
              },
          )
      )
  return requests


def _iter_request_batches(
    args: argparse.Namespace,
) -> Iterator[list[datatypes.RolloutRequest]]:
  top_k = None if args.top_k < 0 else args.top_k
  for step in range(args.max_steps):
    yield _build_step_requests(
        step=step,
        batch_size=args.batch_size,
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
  if args.beta != 0.0 and not args.inference_addr:
    raise ValueError(
        "--inference_addr is required when --beta is non-zero because "
        "reference KL scoring must route to an InferenceWorker."
    )

  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - [OrchestratorV2] %(message)s"
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())

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
  logging.info("Registered Orchestrator V2 workers: %s", cluster.registry.infos())

  program = rl_program.SyncRLProgram(
      algo=algo,
      reward_fns=[_make_reward_fn(args.reward_mode, args.num_generations)],
      assembler=batch_assembly.GRPOTrainExampleAssembler(
          batch_size=args.train_micro_batch_size,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          pad_id=pad_id,
      ),
      sync_weights=False,
      on_step_begin=lambda step: logging.info("GRPO step %d starting.", step),
      on_step_end=lambda step, result: logging.info(
          "GRPO advanced to policy_version=%d train_result=%s.", step, result
      ),
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator.")
    cluster.bring_up_workers(dummy_data=None)
    logging.info("Running SyncRLProgram through ClusterOrchestrator.run_program.")
    cluster.run_program(
        program=program,
        train_dataset=_iter_request_batches(args),
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
