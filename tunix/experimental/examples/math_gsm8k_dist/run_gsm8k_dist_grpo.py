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

try:
  import tensorflow_datasets.text.gsm8k  # pylint: disable=unused-import
except (ImportError, ModuleNotFoundError):
  pass
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
from tunix.experimental.orchestrator import remote_scheduler_router  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=g-import-not-at-top


def _parse_weight_sync_mode(value: str) -> str:
  mode = value.lower()
  if mode in ("noop", "no-op"):
    return "fallback"
  if mode not in ("none", "fallback", "raiden"):
    raise argparse.ArgumentTypeError(
        "weight_sync_mode must be one of: none, fallback, raiden"
    )
  return mode


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 Qwen3 GSM8K GRPO demo."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=4,
      help="Number of prompt groups per step.",
  )
  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--trainer_addr", type=str, default="localhost:20000")
  parser.add_argument(
      "--rollout_addr",
      type=str,
      default="localhost:20001",
      help="Comma-separated RolloutWorker addresses (one worker per address).",
  )
  parser.add_argument(
      "--scheduler_url",
      type=str,
      default="",
      help=(
          "Optional py-inference-scheduler sidecar URL (e.g."
          " http://localhost:8100). When set, rollout requests are routed by"
          " the sidecar instead of prefix-hash/round-robin."
      ),
  )
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
  parser.add_argument(
      "--beta",
      type=float,
      default=0.0,
      help=(
          "KL coefficient. Set to 0.04 with a reference inference worker to "
          "match the Qwen3 GSM8K recipe."
      ),
  )
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
      "--weight_sync_mode",
      type=_parse_weight_sync_mode,
      default=_parse_weight_sync_mode(os.getenv("WEIGHT_SYNC_MODE", "none")),
      help=(
          "Weight synchronization mode. 'none' disables post-update sync, "
          "'raiden' uses Raiden, and 'fallback' runs protocol-only sync."
      ),
  )
  parser.add_argument(
      "--reward_mode",
      choices=("env", "exact"),
      default="env",
      help=(
          "env uses rollout environment rewards; exact recomputes the same "
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
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging and print full sampler responses.",
  )
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
  logging.info("GSM8K dataset loaded successfully: %d examples.", len(dataset))
  return dataset.map(
      lambda x: {
          "prompts": gsm8k.build_prompt(_as_text(x["question"])),
          "question": _as_text(x["question"]),
          "answer": gsm8k.extract_hash_answer(_as_text(x["answer"])),
      }
  )


def _make_reward_fn(mode: str, debug: bool = False):
  """Creates the optional orchestrator-side reward function."""
  if mode == "env":
    return None

  def reward_fn(item: datatypes.TrajectoryItem) -> float:
    metadata = dict(item.metadata or {})
    text = str(metadata.get("text", ""))
    reward, _ = gsm8k.score_gsm8k_completion(
        text, metadata.get("answer", metadata.get("gold_answer"))
    )
    if debug:
      prompt_id = metadata.get("prompt_id", getattr(item, "group_id", "unknown"))
      gold_answer = metadata.get("gold_answer")
      logging.debug(
          "[Orchestrator] Sampler response for %s:\n"
          "[Sampled Response] ---\n%s\n--- [End Response] ---\n"
          "Gold Answer: %s, Extracted Answer: %s",
          prompt_id,
          text,
          gold_answer,
          gsm8k.extract_boxed_answer(text),
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
      # StandardRLProgram consumes this many prompt groups per trainer update.
      mini_batch_size=args.batch_size,
      max_packed_len=args.max_prompt_length + args.max_response_length,
      clip_epsilon=args.epsilon,
      beta_kl=args.beta,
  )
  return algo


def _get_config_attr(config: Any, key: str, default: Any = None) -> Any:
  if config is None:
    return default
  if isinstance(config, dict):
    return config.get(key, default)
  return getattr(config, key, default)


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
  beta = _get_config_attr(grpo_config, "beta", "N/A")
  epsilon = _get_config_attr(grpo_config, "epsilon", "N/A")
  loss_algo = _get_config_attr(grpo_config, "loss_algo", "N/A")
  logging.info(
      "Configuring trainer-side GRPO loss via TrainerWorker RPC (beta=%s, "
      "epsilon=%s, loss_algo=%s).",
      beta,
      epsilon,
      loss_algo,
  )
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
    rollout_handles: Sequence[tuple[str, remote_execution.ActorHandle]],
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
  for idx, (addr, handle) in enumerate(rollout_handles):
    cluster.register_worker(
        _RemoteWorkerRef(
            worker_id=f"rollout-{idx}",
            roles=[datatypes.Role.ROLLOUT],
            handle=handle,
            resources={"address": addr},
        )
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
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": {
          "answer": answer,
          "gold_answer": answer,
          "question": question,
          "prefix_hash": prompt_id,
          "env_config": {
              "prompt": prompt,
              "prompts": prompt,
              "question": question,
              "answer": answer,
              "gold_answer": answer,
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
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")

  logging.info("=== Starting Distributed GSM8K GRPO Orchestrator ===")
  logging.info(
      "Configuration: model_id=%s, batch_size=%d (prompt groups), "
      "num_generations=%d (%d rollouts/step), max_steps=%d, "
      "train_micro_batch_size=%d, beta=%.4f, epsilon=%.2f, reward_mode=%s, "
      "max_staleness=%d, weight_sync_mode=%s.",
      args.model_id,
      args.batch_size,
      args.num_generations,
      args.batch_size * args.num_generations,
      args.max_steps,
      args.train_micro_batch_size,
      args.beta,
      args.epsilon,
      args.reward_mode,
      args.max_staleness,
      args.weight_sync_mode,
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())
  logging.info(
      "Dataset: GSM8K split=%s data_dir=%s reward_mode=%s.",
      args.tfds_split,
      args.tfds_data_dir,
      args.reward_mode,
  )

  tokenizer_path = args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
  logging.info(
      "Loaded tokenizer from %s (vocab_size=%d, pad_id=%d, eos_id=%d).",
      tokenizer_path,
      len(tokenizer),
      pad_id,
      eos_id,
  )

  trainer_addr_future = futures.Future()
  rollout_addr_future = futures.Future()
  inference_addr_future = futures.Future()

  def accept_worker(hostname: str, _: int, metadata: bytes) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    service_address = f"{hostname}:{md['service_port']}"
    worker_id = md["worker_id"]

    logging.info(
        "Discovered %s service (%s) at %s.",
        service_type,
        worker_id,
        service_address,
    )

    match service_type:
      case "trainer":
        if not trainer_addr_future.done():
          trainer_addr_future.set_result(service_address)
      case "rollout":
        if not rollout_addr_future.done():
          rollout_addr_future.set_result(service_address)
      case "inference":
        if not inference_addr_future.done():
          inference_addr_future.set_result(service_address)
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

  assert context and context.ipc and context.ipc.discovery
  context.ipc.discovery.on_register(accept_worker)

  logging.info("Waiting for workers to register via discovery service...")
  trainer_addr = trainer_addr_future.result()
  trainer_handle = _connect(trainer_addr, args.rpc_timeout_s)
  rollout_addr = rollout_addr_future.result()
  rollout_addrs = [a.strip() for a in rollout_addr.split(",") if a.strip()]
  rollout_handles = [
      (addr, _connect(addr, args.rpc_timeout_s)) for addr in rollout_addrs
  ]
  inference_addr = None
  inference_handle = None
  if args.beta != 0.0:
    inference_addr = (
        args.inference_addr
        if args.inference_addr
        else inference_addr_future.result(timeout=args.rpc_timeout_s)
    )
    inference_handle = _connect(inference_addr, args.rpc_timeout_s)

  logging.info(
      "Connected to all required workers: Trainer=%s, Rollout=%s%s.",
      trainer_addr,
      rollout_addr,
      f", Inference={inference_addr}" if inference_addr else "",
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

  rollout_router = None
  if args.scheduler_url:
    logging.info(
        "Routing rollout requests via scheduler sidecar at %s.",
        args.scheduler_url,
    )
    rollout_router = remote_scheduler_router.RemoteSchedulerRouter(
        args.scheduler_url, target_model=args.model_id
    )

  cluster = orchestrator.ClusterOrchestrator(rollout_router=rollout_router, weight_sync_mode=args.weight_sync_mode)
  _register_workers(
      args,
      cluster=cluster,
      trainer_handle=trainer_handle,
      trainer_addr=trainer_addr,
      rollout_handles=rollout_handles,
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

  reward_fn = _make_reward_fn(args.reward_mode, debug=args.debug)
  reward_fns = [reward_fn] if reward_fn is not None else []
  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=_iter_prompt_items(args),
      max_steps=args.max_steps,
      reward_fns=reward_fns,
      assembler=batch_assembly.GRPOTrainExampleAssembler(
          batch_size=args.train_micro_batch_size,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          pad_id=pad_id,
      ),
      metrics_logging_options=metrics_logging_options,
      max_staleness=args.max_staleness,
      sync_weights=(args.weight_sync_mode != "none"),
      on_step_begin=lambda step: logging.info(
          ">>> Step %d starting | Policy Version: %d",
          step,
          step,
      ),
      on_step_end=lambda step, result: logging.info(
          "<<< Step %d finished | Advanced to Policy Version: %d",
          step,
          step + 1,
      ),
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator...")
    cluster.bring_up_workers(dummy_data=None)
    logging.info(
        "Cluster workers ready: %s. Starting StandardRLProgram execution...",
        [w.worker_id for w in cluster.worker_infos()],
    )
    cluster.run_program(
        program=program,
        num_steps=args.max_steps,
        bring_up=False,
    )
  finally:
    if rollout_router is not None:
      rollout_router.stop()
    program.close()
    if args.stop_workers_on_exit:
      logging.info("Shutting down cluster workers...")
      cluster.shutdown()
    else:
      cluster.monitor.close()

  result = program.last_step_result
  if result is not None:
    logging.info(
        "=== GRPO Training Finished Successfully ===\n"
        "  Final step: %d\n"
        "  Final policy version: %d\n"
        "  Total rollouts: %d\n"
        "  Total microbatches: %d\n"
        "  Final step reward: mean=%.4f, std=%.4f",
        result.step,
        result.policy_version,
        result.num_rollouts,
        result.num_microbatches,
        result.reward_mean,
        result.reward_std,
    )
  else:
    logging.info("=== GRPO Training Finished (No step results) ===")


if __name__ == "__main__":
  main(sys.argv[1:])
