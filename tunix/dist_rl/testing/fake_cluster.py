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

"""Test-only single-machine real-gRPC cluster harness for Tunix Orchestrator."""

from collections.abc import Callable, Sequence
import concurrent.futures
import contextlib
import dataclasses
import pickle
import threading
from typing import Any

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import test_utils
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import orchestrator
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.testing import fault_injection
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import rollout_worker as rollout_worker_lib
from tunix.dist_rl.worker import trainer_worker as trainer_worker_lib
from tunix.rl import algorithm_config


class RecordingFaultHandler(weight_sync.WeightSyncHandler):
  """Recording WeightSyncHandler supporting fault injection and call inspection."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self.registered_work_units: list[weight_sync.WorkUnitMetadata] = []
    self.transfers: list[
        tuple[
            list[weight_sync.WorkUnitId],
            list[weight_sync.WorkUnitId],
            str,
        ]
    ] = []
    self.transfer_gate: fault_injection.FaultGate | None = None
    self.fail_on_transfer: Exception | None = None

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    """Registers a single work unit as called by WeightSyncCoordinator."""
    with self._lock:
      self.registered_work_units.append(metadata)

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: str | None = None,
      generation: int | None = None,
      timeout_s: float | None = None,
  ) -> weight_sync.TransferResult:
    """Moves weights and records the transfer call under lock."""
    del generation
    with self._lock:
      self.transfers.append((list(src_units), list(dst_units), req_id or ""))

    if self.transfer_gate is not None:
      self.transfer_gate.wait_sync(
          timeout=timeout_s if timeout_s is not None else 30.0
      )

    if self.fail_on_transfer is not None:
      raise self.fail_on_transfer

    return weight_sync.TransferResult(req_id=req_id or "", success=True)


class _DaemonFuture(concurrent.futures.Future[Any]):
  """A Future executed on a dedicated daemon thread so hung tasks don't block teardown."""

  def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    super().__init__()
    self._thread = threading.Thread(
        target=self._worker,
        args=(fn, args, kwargs),
        daemon=True,
        name="orchestrator-daemon-runner",
    )
    self._thread.start()

  def _worker(
      self,
      fn: Callable[..., Any],
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ) -> None:
    try:
      res = fn(*args, **kwargs)
      self.set_result(res)
    except BaseException as e:
      self.set_exception(e)


class FakeClusterHarness:
  """Single-machine distributed cluster testbed with real gRPC servers."""

  def __init__(
      self,
      *,
      num_rollout_workers: int = 2,
      weight_sync_mode: str = "fallback",
      weight_sync_handler: weight_sync.WeightSyncHandler | None = None,
      weight_sync_timeouts: weight_sync_coordinator.PhaseTimeouts | None = None,
      rpc_timeout_s: float = 30.0,
      poll_retry_budget_s: float = 30.0,
      fault_tolerance_config: datatypes.FaultToleranceConfig | None = None,
      enable_bulk_transport: bool = False,
      partial_rollout: bool = False,
      drain_timeout_s: float = 5.0,
      max_concurrency: int = 8,
      compile_error_workers: set[str] | None = None,
  ) -> None:
    self.num_rollout_workers = num_rollout_workers
    self.weight_sync_mode = weight_sync_mode
    self.weight_sync_handler = weight_sync_handler
    self.weight_sync_timeouts = weight_sync_timeouts
    self.rpc_timeout_s = rpc_timeout_s
    self.poll_retry_budget_s = poll_retry_budget_s
    if fault_tolerance_config is None:
      self.fault_tolerance_config = datatypes.FaultToleranceConfig(
          poll_retry_budget_s=self.poll_retry_budget_s
      )
    elif fault_tolerance_config.poll_retry_budget_s is None:
      self.fault_tolerance_config = dataclasses.replace(
          fault_tolerance_config,
          poll_retry_budget_s=self.poll_retry_budget_s,
      )
    else:
      self.fault_tolerance_config = fault_tolerance_config
      self.poll_retry_budget_s = fault_tolerance_config.poll_retry_budget_s
    self.enable_bulk_transport = enable_bulk_transport
    self.partial_rollout = partial_rollout
    self.drain_timeout_s = drain_timeout_s
    self.max_concurrency = max_concurrency
    self.compile_error_workers = compile_error_workers or set()

    self.injector = fault_injection.FaultInjector()
    self.orchestrator = orchestrator.ClusterOrchestrator(
        weight_sync_mode=self.weight_sync_mode,
        weight_sync_handler=self.weight_sync_handler,
        weight_sync_timeouts=self.weight_sync_timeouts,
        fault_tolerance_config=self.fault_tolerance_config,
    )

    self.servers: dict[str, fault_injection.RunningWorkerServer] = {}
    self.rollout_workers: dict[str, rollout_worker_lib.RolloutWorker] = {}
    self.samplers: dict[str, test_utils.MockBaseSamplerImpl] = {}

    try:
      # Bring up 1 TrainerWorker
      self._create_and_register_trainer_worker("trainer_0", override=False)

      # Bring up num_rollout_workers RolloutWorkers
      for i in range(num_rollout_workers):
        wid = f"rollout_{i}"
        self._create_and_register_rollout_worker(wid, override=False)
    except BaseException:
      self.close()
      raise

  def _create_and_register_trainer_worker(
      self,
      wid: str = "trainer_0",
      port: int = 0,
      *,
      override: bool = False,
      re_register: bool = True,
  ) -> int:
    self.fake_trainer = test_utils.FakeTrainer(
        trainer_id=wid,
        compile_error=(
            RuntimeError(f"Simulated compile failure for {wid}")
            if wid in self.compile_error_workers
            else None
        ),
    )
    self.trainer_worker = trainer_worker_lib.TrainerWorker(
        lambda: self.fake_trainer,
        worker_id=wid,
    )
    server = fault_injection.FaultInjectingServer(
        self.trainer_worker,
        worker_id=wid,
        injector=self.injector,
        enable_bulk_transport=self.enable_bulk_transport,
    )
    running_server = fault_injection.RunningWorkerServer(server, port=port)
    bound_port = running_server.start()
    self.servers[wid] = running_server
    if re_register:
      self.register_worker_from_discovery(
          wid, "trainer", bound_port, override=override
      )
    return bound_port

  def _create_and_register_rollout_worker(
      self,
      wid: str,
      port: int = 0,
      *,
      override: bool = False,
      re_register: bool = True,
  ) -> int:
    sampler = test_utils.MockBaseSamplerImpl(
        sampler_name=wid, default_delay=0.0, server_id=wid
    )
    self.samplers[wid] = sampler
    env_pool = test_utils.MockEnvironmentPool(pool_size=32, default_delay=0.0)
    worker = rollout_worker_lib.RolloutWorker(
        worker_id=wid,
        sampler=sampler,
        env_pool=env_pool,
        agent_factory=test_utils.MockAgent,
        tokenizer=test_utils.MockTokenizer(),
        chat_parser=test_utils.MockChatParser(),
        config=rollout_worker_lib.RolloutConfig(
            partial_rollout=self.partial_rollout,
            drain_timeout_s=self.drain_timeout_s,
        ),
        max_concurrency=self.max_concurrency,
    )
    self.rollout_workers[wid] = worker
    server = fault_injection.FaultInjectingServer(
        worker,
        worker_id=wid,
        injector=self.injector,
        enable_bulk_transport=self.enable_bulk_transport,
    )
    running_server = fault_injection.RunningWorkerServer(server, port=port)
    bound_port = running_server.start()
    self.servers[wid] = running_server
    if re_register:
      self.register_worker_from_discovery(
          wid, "rollout", bound_port, override=override
      )
    return bound_port

  def register_worker_from_discovery(
      self,
      worker_id: str,
      service_type: str,
      port: int,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Encodes discovery metadata and registers worker with WorkerRegistry."""
    metadata = pickle.dumps({
        "service_type": service_type,
        "service_port": port,
        "worker_id": worker_id,
    })
    if override:
      role = (
          datatypes.Role.ACTOR
          if service_type == "trainer"
          else datatypes.Role.ROLLOUT
      )
      return self.orchestrator.registry.register_handle(
          worker_id=worker_id,
          roles=[role],
          handle=remote_execution.ActorHandle.from_address(
              f"grpc://localhost:{port}",
              rpc_timeout_s=self.rpc_timeout_s,
          ),
          resources={"address": f"localhost:{port}"},
          override=True,
      )
    return self.orchestrator.registry.register_from_hostname(
        "localhost",
        port,
        metadata,
        rpc_timeout_s=self.rpc_timeout_s,
    )

  def stop_worker_server(self, worker_id: str, grace: float = 0.0) -> None:
    """Stops the background RunningWorkerServer for a worker."""
    server = self.servers.get(worker_id)
    if server is not None:
      server.stop(grace=grace)

  def restart_worker(
      self,
      worker_id: str,
      *,
      same_port: bool = False,
      override: bool = True,
      re_register: bool = True,
  ) -> int:
    """Restarts a worker with fresh instance and server, optionally re-registering."""
    old_server = self.servers[worker_id]
    old_port = old_server.bound_port
    old_server.stop(grace=0.0)
    target_port = old_port if same_port else 0

    if worker_id == "trainer_0":
      return self._create_and_register_trainer_worker(
          worker_id,
          port=target_port,
          override=override,
          re_register=re_register,
      )
    else:
      return self._create_and_register_rollout_worker(
          worker_id,
          port=target_port,
          override=override,
          re_register=re_register,
      )

  def build_prompts(
      self, num_prompts: int = 2, *, min_turns: int = 0
  ) -> list[dict[str, Any]]:
    """Builds synthetic prompt payloads for testing."""
    return [
        {
            "prompt": f"Prompt {i}",
            "prompt_id": f"prompt_{i}",
            "generation_kwargs": {"max_tokens": 8, "min_turns": min_turns},
        }
        for i in range(num_prompts)
    ]

  def build_program(
      self,
      *,
      num_prompts: int = 2,
      max_steps: int = 1,
      num_generations: int = 2,
      mini_batch_size: int = 2,
      batch_size: int | None = None,
      max_staleness: int = 0,
      sync_weights: bool = True,
      precompile_trainer: bool = False,
      assembler_type: str = "padded",
      group_order: (
          rl_program.GroupOrder | str
      ) = rl_program.GroupOrder.TRAJECTORY_COMPLETION,
      prompts: Sequence[dict[str, Any]] | None = None,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      generation_args: datatypes.GenerationArgs | None = None,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
  ) -> rl_program.StandardRLProgram:
    """Builds a StandardRLProgram wired to GRPO and batch assemblers."""
    algo_cfg = algorithm_config.GRPOConfig(
        num_generations=num_generations, epsilon=0.2, beta=0.0
    )
    algo = algorithm_adapter.GRPOAdapter(
        algo_config=algo_cfg,
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=1,
        max_packed_len=64,
        max_response_length=32,
    )
    effective_batch_size = (
        batch_size if batch_size is not None else mini_batch_size
    )
    if assembler_type == "packed":
      assembler: batch_assembly.BatchAssembler = (
          batch_assembly.SequencePackedBatchAssembler(
              batch_size=effective_batch_size,
              num_generations=num_generations,
              mini_batch_size=mini_batch_size,
              max_packed_len=64,
              pad_id=0,
          )
      )
    else:
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=effective_batch_size,
          max_prompt_length=32,
          max_response_length=32,
          pad_id=0,
          num_generations=num_generations,
          mini_batch_size=mini_batch_size,
      )

    if reward_fns is None:
      reward_fns = [
          lambda prompts, completions, group_index, **kwargs: [
              1.0 if g % 2 == 0 else -1.0 for g in group_index
          ]
      ]

    dataset = (
        prompts
        if prompts is not None
        else self.build_prompts(num_prompts=num_prompts)
    )
    gen_args = (
        generation_args
        if generation_args is not None
        else datatypes.GenerationArgs(temperature=1.0)
    )
    return rl_program.StandardRLProgram(
        algo=algo,
        dataset=dataset,
        max_steps=max_steps,
        reward_fns=reward_fns,
        assembler=assembler,
        generation_args=gen_args,
        max_staleness=max_staleness,
        sync_weights=sync_weights,
        precompile_trainer=precompile_trainer,
        group_order=group_order,
        on_step_begin=on_step_begin,
        on_step_end=on_step_end,
    )

  def start_program(
      self, program: rl_program.StandardRLProgram, *, bring_up: bool = True
  ) -> concurrent.futures.Future[None]:
    """Starts orchestrator program execution in a background daemon thread."""
    return _DaemonFuture(self.orchestrator.run, program, bring_up=bring_up)

  def close(self) -> None:
    """Strict teardown releasing gates, closing orchestrator, and stopping servers."""
    # Step 1: Release all gates so no worker or orchestrator is blocked on FaultGate
    self.injector.release_all_gates()
    if (
        isinstance(self.weight_sync_handler, RecordingFaultHandler)
        and self.weight_sync_handler.transfer_gate is not None
    ):
      self.weight_sync_handler.transfer_gate.release()

    # Step 2: Shutdown orchestrator & registry
    with contextlib.suppress(Exception):
      self.orchestrator.shutdown()
    with contextlib.suppress(Exception):
      self.orchestrator.registry.close()

    # Step 3: Stop all running worker servers
    for server in self.servers.values():
      with contextlib.suppress(Exception):
        server.stop(grace=0.1)

  def __enter__(self) -> "FakeClusterHarness":
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()
