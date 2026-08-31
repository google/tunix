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

"""CPU control-plane for a minimal Orchestrator V2 DeepSWE GRPO demo.

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
import threading
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import grain  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top

from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from examples.deepswe import swe_env  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.deepswe_dist.deepswe_rl_program import DeepsweRLProgram  # pylint: disable=g-import-not-at-top
from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top

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
      description="Orchestrator V2 Qwen3 DeepSWE GRPO demo."
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
          "match the Qwen3 DeepSWE recipe."
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
          "DeepSWE reward in the orchestrator from returned trajectory text."
      ),
  )
  parser.add_argument(
      "--tfds_data_dir",
      type=str,
      default=os.getenv("TFDS_DATA_DIR", "/tmp/deepswe_data"),
  )
  parser.add_argument("--tfds_split", type=str, default="train")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--shuffle", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument("--inference_addr", type=str, default="")
  parser.add_argument(
      "--num_rollout_workers",
      type=int,
      default=1,
      help=(
          "Number of independent rollout replicas (distinct worker_id) to"
          " wait for and register, e.g. 2 for data-parallel rollout across"
          " two single-host TPU slices. Multiple pods of the SAME replica"
          " (one multihost rollout jobset) share one worker_id and are not"
          " counted separately -- see accept_worker."
      ),
  )
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

class _CoordinatorWorkerShim:
  """Presents a remote ActorHandle as a coordinator-protocol worker."""

  def __init__(self, handle, worker_id, roles):
    self._handle = handle
    self._worker_id = worker_id
    self._roles = frozenset(roles)

  def info(self):
    return datatypes.WorkerInfo(worker_id=self._worker_id, roles=self._roles)

  async def prepare_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("prepare_weight_sync", *args, **kwargs)

  async def release_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("release_weight_sync", *args, **kwargs)

  async def bind_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("bind_weight_sync", *args, **kwargs)

  async def get_weight_sync_metadata(self, *args, **kwargs):
    return await self._handle.asubmit("get_weight_sync_metadata", *args, **kwargs)

  async def pre_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("pre_weight_sync", *args, **kwargs)

  async def weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("weight_sync", *args, **kwargs)

  async def post_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("post_weight_sync", *args, **kwargs)

  async def abort_weight_sync(self, *args, **kwargs):
    return await self._handle.asubmit("abort_weight_sync", *args, **kwargs)

  async def get_weight_sync_status(self, *args, **kwargs):
    return await self._handle.asubmit("get_weight_sync_status", *args, **kwargs)

def _make_weight_sync_coordinator(trainer_handle, rollout_handles, mode: str):
  """Builds the weight sync coordinator over the configured transport."""
  if mode == "none":
    return None
  from tunix.experimental.orchestrator import weight_sync  # pylint: disable=g-import-not-at-top
  from tunix.experimental.orchestrator import weight_sync_coordinator  # pylint: disable=g-import-not-at-top
  from tunix.experimental.orchestrator import worker_registry as registry_lib  # pylint: disable=g-import-not-at-top

  class _NullHandler(weight_sync.WeightSyncHandler):
    """Runs every phase without moving bytes."""

    def register_work_unit(self, metadata):
      del metadata

    def transfer(self, src_units, dst_units, req_id=None, generation=None):
      del src_units, dst_units, generation
      return weight_sync.TransferResult(req_id=req_id or "", success=True)

  registry = registry_lib.WorkerRegistry()
  registry.register(
      _CoordinatorWorkerShim(trainer_handle, "trainer-0", {"trainer"})
  )
  # WeightSyncCoordinator._destinations() already fans out to every worker
  # registered under the rollout role, so registering N handles here is
  # enough to broadcast weight sync to N independent rollout replicas -- no
  # coordinator/registry changes needed.
  for i, handle in enumerate(rollout_handles):
    registry.register(
        _CoordinatorWorkerShim(handle, f"rollout-{i}", {"rollout"})
    )
# TODO: standardize a handeler registry seperate from the worker registry.
  if mode == "raiden":
    from tunix.experimental.orchestrator import raiden_handler  # pylint: disable=g-import-not-at-top

    handler = raiden_handler.RaidenHandler(
        transfer_options=raiden_handler.make_host_staged_transfer_options()
    )
    logging.info(
        "Raiden weight sync enabled; controller on port %d.", handler.port
    )
  elif mode in ("noop", "fallback"):
    handler = _NullHandler()
    logging.info("Weight sync running protocol-only; no bytes move.")
  else:
    raise ValueError(f"Unknown weight sync mode: {mode!r}")
  return weight_sync_coordinator.WeightSyncCoordinator(
      registry, handler, controller_id="deepswe-demo"
  )

def _register_workers(
    args: argparse.Namespace,
    *,
    cluster: orchestrator.ClusterOrchestrator,
    trainer_handle: remote_execution.ActorHandle,
    trainer_addr: str,
    rollout_handles: list[remote_execution.ActorHandle],
    rollout_addrs: list[str],
    inference_handle: remote_execution.ActorHandle | None,
    inference_addr: str | None,
) -> None:
  """Registers gRPC-backed workers in the Orchestrator V2 registry."""
  cluster.register_worker_handle(
      worker_id="trainer-0",
      roles=[datatypes.Role.ACTOR, "trainer"],
      handle=trainer_handle,
      resources={"address": trainer_addr},
  )
  # ClusterOrchestrator._get_actor_handles(ROLLOUT) collects every worker
  # registered under this role, and DistributedRLEngine's RoutingActorPool
  # load-balances across all of them -- registering N handles here is
  # enough for N-way data-parallel rollout, no engine/pool changes needed.
  for i, (handle, addr) in enumerate(zip(rollout_handles, rollout_addrs)):
    cluster.register_worker_handle(
        worker_id=f"rollout-{i}",
        roles=[datatypes.Role.ROLLOUT, "rollout"],
        handle=handle,
        resources={"address": addr},
    )
  if inference_handle is not None:
    cluster.register_worker_handle(
        worker_id="reference-0",
        roles=[datatypes.Role.REFERENCE],
        handle=inference_handle,
        resources={"address": inference_addr},
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

  logging.info("=== Starting Distributed DeepSWE GRPO Orchestrator ===")
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
      "Async rollout max_staleness=%d (0 means queue-level on-policy).",
      args.max_staleness,
  )
  logging.info(
      "Dataset: DeepSWE split=%s data_dir=%s reward_mode=%s.",
      args.tfds_split,
      args.tfds_data_dir,
      args.reward_mode,
  )
  logging.info("Weight sync mode: %s", args.weight_sync_mode)

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

  from examples.deepswe import deepswe_data
  grain_dataset = deepswe_data.create_dataset(
      dataset_name=args.tfds_data_dir or "R2E-Gym/R2E-Gym-V1",
      dataset_split=args.tfds_split or "train",
      seed=args.seed,
  )
  dataset = list(grain_dataset)
  swe_env._init_global_fleet(tasks=dataset, max_concurrency=args.num_rollout_workers)

  trainer_addr_future = futures.Future()
  inference_addr_future = futures.Future()
  # Keyed by worker_id, not a single Future: a multihost rollout jobset has
  # one pod per host all sharing one worker_id (that jobset's own
  # JAX-distributed mesh coordinates the rest internally, so only one
  # address per worker_id is kept), but N independent rollout replicas
  # (e.g. two single-host TPU slices for data-parallel rollout) register
  # under N distinct worker_ids and must all be kept and connected.
  rollout_addrs_by_worker: dict[str, str] = {}
  rollout_workers_ready = threading.Event()

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

    # A multihost trainer/rollout jobset has one pod per host, and every pod
    # registers here independently. Only the first registration per
    # worker_id is used to drive the RPC connection (that pod's own
    # JAX-distributed mesh coordinates the rest of its jobset internally),
    # so later registrations under an already-resolved worker_id are
    # expected and must be no-ops rather than overwrite/crash.
    match service_type:
      case "trainer":
        if not trainer_addr_future.done():
          trainer_addr_future.set_result(service_address)
      case "rollout":
        if worker_id not in rollout_addrs_by_worker:
          rollout_addrs_by_worker[worker_id] = service_address
          if len(rollout_addrs_by_worker) >= args.num_rollout_workers:
            rollout_workers_ready.set()
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
  rollout_workers_ready.wait(timeout=args.rpc_timeout_s)
  if len(rollout_addrs_by_worker) < args.num_rollout_workers:
    raise TimeoutError(
        f"only {len(rollout_addrs_by_worker)}/{args.num_rollout_workers}"
        " rollout workers registered within rpc_timeout_s"
        f" ({args.rpc_timeout_s}s): {sorted(rollout_addrs_by_worker)}"
    )
  # Sorted for a deterministic worker_id -> handle order across runs.
  rollout_entries = sorted(rollout_addrs_by_worker.items())
  rollout_handles = [
      _connect(addr, args.rpc_timeout_s) for _, addr in rollout_entries
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

  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_coordinator=_make_weight_sync_coordinator(
          trainer_handle, rollout_handles, args.weight_sync_mode
      )
  )

  _register_workers(
      args,
      cluster=cluster,
      trainer_handle=trainer_handle,
      trainer_addr=trainer_addr,
      rollout_handles=rollout_handles,
      rollout_addrs=[addr for _, addr in rollout_entries],
      inference_handle=inference_handle,
      inference_addr=inference_addr,
  )
  logging.info("Registered Orchestrator V2 workers: %s", cluster.worker_infos())

  program = DeepsweRLProgram(dataset=iter(dataset), algo=algo, max_steps=args.max_steps)

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
