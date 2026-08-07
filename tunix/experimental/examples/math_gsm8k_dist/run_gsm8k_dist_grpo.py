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
import asyncio
from collections.abc import Sequence
import dataclasses
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

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator as orchestrator_v2  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import abstract_worker  # pylint: disable=g-import-not-at-top
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


@dataclasses.dataclass(frozen=True)
class StepResult:
  step: int
  policy_version: int
  num_rollouts: int
  num_microbatches: int
  reward_mean: float
  reward_std: float
  train_result: Any


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


def _run_sync(value: Any) -> Any:
  if not asyncio.iscoroutine(value):
    return value
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None
  if loop and loop.is_running():
    return value
  return asyncio.run(value)


def _role_names(roles: Sequence[datatypes.Role | str]) -> frozenset[str]:
  return frozenset(role.value if isinstance(role, datatypes.Role) else role
                   for role in roles)


def _left_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  arr = np.asarray(values, dtype=np.int32).reshape(-1)[-length:]
  out = np.full(length, pad_id, dtype=np.int32)
  mask = np.zeros(length, dtype=np.float32)
  if arr.size:
    out[-arr.size:] = arr
    mask[-arr.size:] = 1.0
  return out, mask


def _right_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_value: float | int = 0,
    dtype: Any = np.int32,
) -> tuple[np.ndarray, np.ndarray]:
  arr = np.asarray(values, dtype=dtype).reshape(-1)[:length]
  out = np.full(length, pad_value, dtype=dtype)
  mask = np.zeros(length, dtype=np.float32)
  if arr.size:
    out[:arr.size] = arr
    mask[:arr.size] = 1.0
  return out, mask


def _completion_aligned(
    values: Any | None,
    completion_len: int,
    max_response_length: int,
    *,
    fill_value: float = 0.0,
    prompt_len: int | None = None,
    full_completion_len: int | None = None,
) -> np.ndarray:
  if values is None:
    arr = np.full(completion_len, fill_value, dtype=np.float32)
  else:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1:
      arr = np.full(completion_len, float(arr[0]), dtype=np.float32)
    elif (
        prompt_len is not None
        and full_completion_len is not None
        and arr.size == prompt_len + full_completion_len
    ):
      arr = arr[prompt_len:prompt_len + full_completion_len]
    elif full_completion_len is not None and arr.size >= full_completion_len:
      arr = arr[:full_completion_len]
    elif arr.size >= completion_len:
      arr = arr[:completion_len]
    else:
      arr = np.pad(arr, (0, completion_len - arr.size), constant_values=0.0)
    arr = arr[:completion_len]
  out, _ = _right_pad(
      arr,
      max_response_length,
      pad_value=0.0,
      dtype=np.float32,
  )
  return out


class _RemoteWorkerHandle(abstract_worker.Worker):
  """Registers a remote gRPC ActorHandle as an Orchestrator V2 Worker."""

  def __init__(
      self,
      *,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      pad_id: int,
      eos_id: int,
      max_prompt_length: int,
      max_response_length: int,
      temperature: float,
  ):
    self._worker_id = worker_id
    self._roles = _role_names(roles)
    self._handle = handle
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._max_prompt_length = max_prompt_length
    self._max_response_length = max_response_length
    self._temperature = temperature

  def _submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    logging.info("[%s] %s", self._worker_id, method_name)
    return self._handle.submit(method_name, *args, **kwargs)

  def initialize(self) -> datatypes.Response:
    return self._submit("initialize")

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    return self._submit("compile", dummy_data)

  def start(self) -> datatypes.Response:
    return self._submit("start")

  def stop(self) -> datatypes.Response:
    return self._submit("stop")

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=self._roles,
        resources={"remote": True},
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return self._submit("heartbeat")

  def submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    return self._handle.submit(method_name, *args, **kwargs)

  async def asubmit(
      self, method_name: str, *args: Any, **kwargs: Any
  ) -> Any:
    if method_name == "prepare_weight_sync":
      return await self._prepare_weight_sync(*args, **kwargs)
    if method_name == "per_token_logps":
      return await self._remote_reference_logps(**kwargs)
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
  ) -> Any:
    return await self._handle.poll_responses(timeout_s=timeout_s)

  async def _prepare_weight_sync(self, *args: Any, **kwargs: Any) -> Any:
    metadata = await self._handle.asubmit(
        "prepare_weight_sync", *args, **kwargs
    )
    weights = await self._handle.asubmit("get_lora_weights")
    policy_version = 0
    if isinstance(metadata, datatypes.Response):
      policy_version = int(metadata.metadata.get("policy_version", 0))
    return SimpleNamespace(
        weights=weights,
        metadata=metadata,
        new_policy_version=policy_version + 1,
    )

  async def _remote_reference_logps(
      self,
      items: Sequence[datatypes.TrajectoryItem],
      **kwargs: Any,
  ) -> np.ndarray:
    prompt_rows = []
    completion_rows = []
    for item in items:
      prompt, _ = _left_pad(
          item.prompt_tokens if item.prompt_tokens is not None else np.zeros(0),
          self._max_prompt_length,
          pad_id=self._pad_id,
      )
      completion, _ = _right_pad(
          item.completion_tokens
          if item.completion_tokens is not None
          else np.zeros(0),
          self._max_response_length,
          pad_value=self._pad_id,
          dtype=np.int32,
      )
      prompt_rows.append(prompt)
      completion_rows.append(completion)

    req = datatypes.LogprobsRequest(
        request_id="reference_logps",
        prompt_tokens=np.stack(prompt_rows),
        completion_tokens=np.stack(completion_rows),
        temperature=float(kwargs.get("temperature", self._temperature)),
        model_role="reference",
    )
    resp = await self._handle.asubmit("compute_logps", req=req)
    if getattr(resp, "error", None) is not None:
      raise RuntimeError(resp.error.message)
    return np.asarray(resp.per_token_logps, dtype=np.float32)


class _GRPOTrainExampleAssembler:
  """Pads v2 RLTrainerPayloads into GRPO TrainExample microbatches."""

  def __init__(
      self,
      *,
      batch_size: int,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
  ):
    if batch_size <= 0:
      raise ValueError("train microbatch size must be positive.")
    self.batch_size = batch_size
    self.max_prompt_length = max_prompt_length
    self.max_response_length = max_response_length
    self.pad_id = pad_id

  def pack(
      self, items: Sequence[datatypes.RLTrainerPayload]
  ) -> list[rl_common.TrainExample]:
    item_list = list(items)
    if not item_list:
      return []

    microbatches = []
    for start in range(0, len(item_list), self.batch_size):
      chunk = item_list[start:start + self.batch_size]
      microbatches.append(self._pack_chunk(chunk))
    return microbatches

  def _pack_chunk(
      self, chunk: Sequence[datatypes.RLTrainerPayload]
  ) -> rl_common.TrainExample:
    prompt_ids = []
    prompt_mask = []
    completion_ids = []
    completion_mask = []
    advantages = []
    ref_logps = []
    old_logps = []
    has_ref_logps = any(x.ref_per_token_logps is not None for x in chunk)
    has_old_logps = any(x.old_per_token_logps is not None for x in chunk)

    for item in chunk:
      p = np.asarray(item.prompt_ids, dtype=np.int32).reshape(-1)
      c_full = np.asarray(item.completion_ids, dtype=np.int32).reshape(-1)
      c_mask_src = (
          np.asarray(item.completion_mask, dtype=np.float32).reshape(-1)
          if item.completion_mask is not None
          else np.ones(c_full.shape, dtype=np.float32)
      )
      c = c_full[:self.max_response_length]
      c_mask_src = c_mask_src[:c.size]

      p_ids, p_mask = _left_pad(
          p, self.max_prompt_length, pad_id=self.pad_id
      )
      c_ids, c_default_mask = _right_pad(
          c,
          self.max_response_length,
          pad_value=self.pad_id,
          dtype=np.int32,
      )
      c_mask = np.zeros(self.max_response_length, dtype=np.float32)
      if c_mask_src.size:
        c_mask[:c_mask_src.size] = c_mask_src
      else:
        c_mask = c_default_mask

      prompt_ids.append(p_ids)
      prompt_mask.append(p_mask)
      completion_ids.append(c_ids)
      completion_mask.append(c_mask)

      adv_src = item.advantages
      if adv_src is not None:
        adv_arr = np.asarray(adv_src, dtype=np.float32).reshape(-1)
      else:
        adv_arr = None
      advantages.append(
          _completion_aligned(
              adv_arr,
              c.size,
              self.max_response_length,
              fill_value=0.0,
              prompt_len=p.size,
              full_completion_len=c_full.size,
          )
      )

      if has_ref_logps:
        ref_logps.append(
            _completion_aligned(
                item.ref_per_token_logps,
                c.size,
                self.max_response_length,
                full_completion_len=c_full.size,
            )
        )
      if has_old_logps:
        old_logps.append(
            _completion_aligned(
                item.old_per_token_logps,
                c.size,
                self.max_response_length,
                full_completion_len=c_full.size,
            )
        )

    while len(prompt_ids) < self.batch_size:
      prompt_ids.append(np.full(self.max_prompt_length, self.pad_id, np.int32))
      prompt_mask.append(np.zeros(self.max_prompt_length, dtype=np.float32))
      completion_ids.append(
          np.full(self.max_response_length, self.pad_id, np.int32)
      )
      completion_mask.append(
          np.zeros(self.max_response_length, dtype=np.float32)
      )
      advantages.append(np.zeros(self.max_response_length, dtype=np.float32))
      if has_ref_logps:
        ref_logps.append(np.zeros(self.max_response_length, dtype=np.float32))
      if has_old_logps:
        old_logps.append(np.zeros(self.max_response_length, dtype=np.float32))

    return rl_common.TrainExample(
        prompt_ids=np.stack(prompt_ids),
        prompt_mask=np.stack(prompt_mask),
        completion_ids=np.stack(completion_ids),
        completion_mask=np.stack(completion_mask),
        advantages=np.stack(advantages),
        ref_per_token_logps=np.stack(ref_logps) if has_ref_logps else None,
        old_per_token_logps=np.stack(old_logps) if has_old_logps else None,
    )


class _GRPODemoProgram(rl_program.RLProgram):
  """RLProgram variant with correct microbatch accumulation boundaries."""

  def __init__(self, *args: Any, sync_weights: bool, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._sync_weights = sync_weights

  def step_once(
      self,
      prompts: Sequence[datatypes.RolloutRequest],
      **kwargs: Any,
  ) -> StepResult:
    current_step = self.policy_version
    if self.on_step_begin:
      self.on_step_begin(current_step)

    rollouts = _run_sync(self.engine.generate(prompts=prompts, **kwargs))
    rollouts = sorted(
        rollouts,
        key=lambda item: (
            getattr(item, "group_id", ""),
            int(getattr(item, "pair_index", 0)),
        ),
    )
    rewards = [
        float(sum(fn(item) for fn in self.reward_fns))
        if self.reward_fns
        else float(getattr(item.traj, "reward", 0.0))
        for item in rollouts
    ]

    ref_logps = None
    if getattr(self.algo, "requires_reference_kl", False):
      ref_logps = _run_sync(
          self.engine.per_token_logps(
              datatypes.Role.REFERENCE,
              items=rollouts,
              temperature=kwargs.get("temperature", 1.0),
          )
      )

    trainer_payloads = self.algo.create_trainer_payloads(
        rollouts, rewards=rewards, ref_logps=ref_logps
    )
    microbatches = self.assembler.pack(trainer_payloads)
    if not microbatches:
      raise RuntimeError("No trainer microbatches were assembled.")

    step_result = None
    for index, batch in enumerate(microbatches):
      is_last = index == len(microbatches) - 1
      step_result = _run_sync(
          self.engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=is_last,
          )
      )

    if self._sync_weights:
      new_version = _run_sync(
          self.engine.sync_weights(role=datatypes.Role.ACTOR)
      )
      if isinstance(new_version, int) and new_version > current_step:
        self.policy_version = new_version
      else:
        self.policy_version = current_step + 1
    else:
      self.policy_version = current_step + 1

    result = StepResult(
        step=current_step,
        policy_version=self.policy_version,
        num_rollouts=len(rollouts),
        num_microbatches=len(microbatches),
        reward_mean=float(np.mean(rewards)) if rewards else 0.0,
        reward_std=float(np.std(rewards)) if rewards else 0.0,
        train_result=step_result,
    )
    if self.on_step_end:
      self.on_step_end(self.policy_version, result)
    return result


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
  trainer = _RemoteWorkerHandle(
      worker_id="remote-trainer-actor",
      roles=(datatypes.Role.ACTOR,),
      handle=trainer_handle,
      **common_kwargs,
  )
  rollout = _RemoteWorkerHandle(
      worker_id="remote-vllm-rollout",
      roles=(datatypes.Role.ROLLOUT,),
      handle=rollout_handle,
      **common_kwargs,
  )
  orch.register_worker(trainer)
  orch.register_worker(rollout)
  if inference_handle is not None:
    reference = _RemoteWorkerHandle(
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
  assembler = _GRPOTrainExampleAssembler(
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

  program = _GRPODemoProgram(
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
    result = program.step_once(
        rollout_requests,
        max_generation_steps=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )
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
        result.train_result,
        result.policy_version,
    )

  if args.stop_workers_on_exit:
    orch.shutdown()

  logging.info("Distributed GRPO v2 chain demo finished successfully.")


if __name__ == "__main__":
  main()
