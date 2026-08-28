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
and executes StandardRLProgram through ClusterOrchestrator.run_program().
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent import futures
import functools
import logging
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import grain  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
# pylint: disable-next=unused-import,g-import-not-at-top
import tensorflow_datasets.text.gsm8k
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.math_gsm8k_dist import (  # pylint: disable=g-import-not-at-top
    gsm8k,
)
from tunix.experimental.orchestrator import (  # pylint: disable=g-import-not-at-top
    algorithm_adapter,
)
from tunix.experimental.orchestrator import (  # pylint: disable=g-import-not-at-top
    batch_assembly,
)
from tunix.experimental.orchestrator import (  # pylint: disable=g-import-not-at-top
    orchestrator,
)
from tunix.experimental.orchestrator import (  # pylint: disable=g-import-not-at-top
    rl_program,
)
from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
    remote_execution,
)
from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=g-import-not-at-top



def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 Qwen3 GSM8K GRPO demo."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=4,
      help="Number of prompt groups per rollout batch.",
  )
  parser.add_argument("--mini_batch_size", type=int, default=2)
  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument(
      "--train_max_response_length",
      type=int,
      default=0,
      help=(
          "Static completion length used for trainer/logprob batches. Defaults "
          "to max_response_length; the launcher pads this for MaxText splash "
          "attention when needed."
      ),
  )
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.04)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--offpolicy",
      "--max_staleness",
      dest="max_staleness",
      type=int,
      default=0,
      help=(
          "Maximum policy-version lag accepted by the async rollout queue. "
          "0 means queue-level on-policy training."
      ),
  )
  parser.add_argument(
      "--weight_sync_backend",
      choices=("raiden", "no-op", "none"),
      default="none",
      help=(
          "Enable post-update weight sync coordinator using the given backend. "
          "'none' disables weight synchronization."
      ),
  )
  parser.add_argument(
      "--reward_mode",
      choices=("env", "exact"),
      default="env",
      help=(
          "env uses the rollout environment reward; exact recomputes the same "
          "GSM8K reward in the orchestrator from returned trajectory text."
      ),
  )
  parser.add_argument(
      "--tfds_data_dir",
      type=str,
      default=os.getenv("TFDS_DATA_DIR", "/tmp/gsm8k_data"),
  )
  parser.add_argument("--tfds_split", type=str, default="train")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--shuffle", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.getenv("LOG_DIR", "/tmp/trellis_gsm8k"),
      help="Directory for local event logging (TensorBoard/CLU).",
  )
  parser.add_argument(
      "--wandb_project",
      type=str,
      default=os.getenv("WANDB_PROJECT", "trellis-gsm8k"),
      help="W&B project name.",
  )
  parser.add_argument(
      "--wandb_run_name",
      type=str,
      default=os.getenv("WANDB_RUN_NAME", ""),
      help="W&B run name. Defaults to timestamp-based name if unset.",
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--inference_addr", type=str, default="")
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  return parser.parse_args(argv)


def _connect(addr: str, timeout_s: float) -> remote_execution.ActorHandle:
  return remote_execution.ActorHandle.from_address(
      f"grpc://{addr}", rpc_timeout_s=timeout_s
  )


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


def _as_text(value: Any) -> str:
  normalized = _normalize_example_value(value)
  return normalized if isinstance(normalized, str) else str(normalized)


def _build_gsm8k_dataset(args: argparse.Namespace) -> grain.MapDataset:
  """Loads the real GSM8K split and maps examples to prompt/answer records."""
  logging.info(
      "Loading GSM8K TFDS split=%s data_dir=%s shuffle=%s seed=%d.",
      args.tfds_split,
      args.tfds_data_dir,
      args.shuffle,
      args.seed,
  )
  data = tfds.data_source(
      "gsm8k",
      split=args.tfds_split,
      data_dir=args.tfds_data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )
  dataset = grain.MapDataset.source(data)
  if args.shuffle:
    dataset = dataset.shuffle(seed=args.seed)
  return dataset.map(
      lambda x: {
          "prompts": gsm8k.build_prompt(_as_text(x["question"])),
          "question": _as_text(x["question"]),
          "answer": gsm8k.extract_hash_answer(_as_text(x["answer"])),
      }
  )


def _make_reward_fn(mode: str):
  """Creates the optional orchestrator-side reward function."""
  if mode == "env":
    return None

  def reward_fn(item: datatypes.TrajectoryItem) -> float:
    metadata = dict(item.metadata or {})
    text = str(metadata.get("text", ""))
    reward, _ = gsm8k.score_gsm8k_completion(
        text, metadata.get("answer", metadata.get("gold_answer"))
    )
    return reward

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
      mini_batch_size=args.mini_batch_size,
      max_packed_len=args.max_prompt_length + args.train_max_response_length,
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
    trainer_addr: str,
    rollout_handle: remote_execution.ActorHandle,
    rollout_addr: str,
    inference_handle: remote_execution.ActorHandle | None,
    inference_addr: str | None,
) -> None:
  """Registers gRPC-backed workers in the Orchestrator V2 registry."""
  cluster.register_worker_handle(
      worker_id="trainer-0",
      roles=[datatypes.Role.ACTOR],
      handle=trainer_handle,
      resources={"address": trainer_addr},
  )
  cluster.register_worker_handle(
      worker_id="rollout-0",
      roles=[datatypes.Role.ROLLOUT],
      handle=rollout_handle,
      resources={"address": rollout_addr},
  )
  if inference_handle is not None:
    cluster.register_worker_handle(
        worker_id="reference-0",
        roles=[datatypes.Role.REFERENCE],
        handle=inference_handle,
        resources={"address": inference_addr},
    )


def _build_prompt_item(
    *,
    example: dict[str, Any],
    prompt_idx: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> dict[str, Any]:
  prompt = _as_text(example["prompts"])
  question = _as_text(example["question"])
  answer = _normalize_example_value(example["answer"])
  prompt_id = f"prompt_{prompt_idx}"
  return {
      "prompt": prompt,
      "prompt_id": prompt_id,
      "group_id": prompt_id,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": {
          "answer": answer,
          "question": question,
          "prefix_hash": prompt_id,
          "env_config": {
              "prompt": prompt,
              "question": question,
              "answer": answer,
              "group_id": prompt_id,
              "max_steps": 1,
          },
      },
  }


def _iter_prompt_items(
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
  top_k = None if args.top_k < 0 else args.top_k
  dataset = _build_gsm8k_dataset(args)
  dataset_size = len(dataset)
  if dataset_size == 0:
    raise ValueError("GSM8K dataset is empty.")
  for prompt_idx in range(args.max_steps * args.batch_size):
    example = dataset[prompt_idx % dataset_size]
    yield _build_prompt_item(
        example=example,
        prompt_idx=prompt_idx,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [Orchestrator] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.mini_batch_size <= 0:
    raise ValueError("mini_batch_size must be positive.")
  if args.batch_size % args.mini_batch_size != 0:
    raise ValueError(
        "mini_batch_size must divide batch_size to match the "
        "qwen3_grpo_demo.py recipe semantics. Got "
        f"batch_size={args.batch_size}, mini_batch_size={args.mini_batch_size}."
    )
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  if args.train_max_response_length <= 0:
    args.train_max_response_length = args.max_response_length
  if args.train_max_response_length < args.max_response_length:
    raise ValueError(
        "train_max_response_length must be >= max_response_length so generated "
        "tokens are not truncated before training."
    )
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")

  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - [OrchestratorV2] %(message)s"
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())
  logging.info(
      "Async rollout max_staleness=%d (0 means queue-level on-policy).",
      args.max_staleness,
  )
  logging.info("Weight sync enabled: %s", args.sync_weights)

  tokenizer_path = args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

  trainer_addr_future = futures.Future()
  rollout_addr_future = futures.Future()
  inference_addr_future = futures.Future()

  def accept_worker(hostname: str, _: int, metadata: bytes) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    service_address = f"{hostname}:{md['service_port']}"
    worker_id = md["worker_id"]

    logging.info(
        "discovered %s service %s at %s",
        service_type,
        worker_id,
        service_address,
    )

    match service_type:
      case "trainer":
        trainer_addr_future.set_result(service_address)
      case "rollout":
        rollout_addr_future.set_result(service_address)
      case "inference":
        inference_addr_future.set_result(service_address)
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

  assert context and context.ipc and context.ipc.discovery
  context.ipc.discovery.on_register(accept_worker)

  logging.info("Waiting for workers to connect...")
  trainer_addr = trainer_addr_future.result()
  trainer_handle = _connect(trainer_addr, args.rpc_timeout_s)
  rollout_addr = rollout_addr_future.result()
  rollout_handle = _connect(rollout_addr, args.rpc_timeout_s)
  inference_addr = None
  inference_handle = None
  if args.beta != 0.0:
    inference_addr = (
        args.inference_addr
        if args.inference_addr
        else inference_addr_future.result(timeout=args.rpc_timeout_s)
    )
    inference_handle = _connect(inference_addr, args.rpc_timeout_s)

  algo = _build_algo(args)
  grpo_config = _build_grpo_config(args)
  _configure_trainer_loss(
      trainer_handle,
      algo=algo,
      grpo_config=grpo_config,
      pad_id=pad_id,
      eos_id=eos_id,
  )

  weight_sync_backend = None if args.weight_sync_backend == "none" else args.weight_sync_backend
  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_coordinator=_make_weight_sync_coordinator(
          trainer_handle, rollout_handle
      )
  )

  _register_workers(
      args,
      cluster=cluster,
      trainer_handle=trainer_handle,
      trainer_addr=trainer_addr,
      rollout_handle=rollout_handle,
      rollout_addr=rollout_addr,
      inference_handle=inference_handle,
      inference_addr=inference_addr,
  )
  logging.info("Registered Orchestrator V2 workers: %s", cluster.worker_infos())

  metrics_logging_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=1,
      backend_kwargs={
          "wandb": {
              "config": vars(args),
          }
      },
  )

  reward_fn = _make_reward_fn(args.reward_mode)
  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=_iter_prompt_items(args),
      max_steps=args.max_steps,
      reward_fns=[reward_fn] if reward_fn is not None else None,
      assembler=batch_assembly.GRPOTrainExampleAssembler(
          batch_size=args.train_micro_batch_size,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.train_max_response_length,
          pad_id=pad_id,
      ),
      metrics_logging_options=metrics_logging_options,
      max_staleness=args.max_staleness,
      sync_weights=True,
      on_step_begin=lambda step: logging.info(
          "Async GRPO step %d starting.", step
      ),
      on_step_end=lambda step, result: logging.info(
          "Async GRPO advanced to policy_version=%d train_result=%s.",
          step,
          result,
      ),
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator.")
    cluster.bring_up_workers(dummy_data=None)
    logging.info(
        "Running StandardRLProgram through ClusterOrchestrator.run_program."
    )
    cluster.run_program(
        program=program,
        bring_up=False,
    )
  finally:
    program.close()
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
  main(sys.argv[1:])
