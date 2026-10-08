# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only multi-process integration test for distributed RL with asymmetric sharding and Raiden weight sync."""

import os
import sys

# Virtualize 4 CPU devices on host platform before JAX initializes
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import asyncio
import contextlib
import functools
import logging
import signal
import socket
import time
from types import SimpleNamespace
from typing import Any, Iterator, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import portpicker

# copybara:strip_begin
import multiprocessing
# copybara:strip_end

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.weight_sync import raiden_handler
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.experimental.weight_sync import raiden_weight_sync_delegate
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution as remote_lib
from tunix.experimental.worker import rollout_worker as rollout_worker_lib
from tunix.experimental.worker import trainer_worker as trainer_worker_lib
from tunix.generate import sampler as vanilla_sampler_lib
from tunix.generate import tokenizer_adapter
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.experimental.rl.agentic import registry
from tunix.rl import algorithm_config
from tunix.tests import test_common


# ==============================================================================
# Synthetic Environment and Agent
# ==============================================================================

TOY_ENV_NAME = "cpu_rl_toy_env"
TOY_AGENT_NAME = "cpu_rl_toy_agent"


@registry.register_env(TOY_ENV_NAME)
class ToyEnv(base_environment.BaseTaskEnv):
  """Deterministic, single-turn toy environment for testing."""

  def __init__(self, task: dict[str, Any] | None = None, **kwargs: Any):
    self.poison = bool(kwargs.pop("poison", False))
    self.poison_group_index = kwargs.pop("poison_group_index", None)
    super().__init__(task=task or {"prompt": "hello"}, max_steps=1, **kwargs)
    self.step_count = 0
    self.final_reward_fn = None
    self.group_index = kwargs.get("group_index", 0)

  def _maybe_raise_poison(self) -> None:
    if self.poison or (
        self.poison_group_index is not None
        and self.group_index == self.poison_group_index
    ):
      raise RuntimeError(
          f"Poison prompt failure for group_index={self.group_index}"
      )

  def _reward(self) -> float:
    return 1.0 if self.group_index % 2 == 0 else -1.0

  def reset(
      self, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[str, dict[str, Any]]:
    del seed, options
    self._maybe_raise_poison()
    self.step_count = 0
    return "hello", {}

  def step(self, action: Any) -> tuple[str, float, bool, dict[str, Any]]:
    del action
    self._maybe_raise_poison()
    self.step_count += 1
    return "world", self._reward(), True, {"step": self.step_count}

  def _initial_observation(self) -> str:
    self._maybe_raise_poison()
    return "hello"

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    del action
    self._maybe_raise_poison()
    return base_environment.EnvStepResult(
        observation="world",
        reward=self._reward(),
        done=True,
        info={},
    )


@registry.register_agent(TOY_AGENT_NAME)
class ToyAgent(base_agent.ConversationAgentBase):
  """Fast single-turn agent that tracks messages and trajectory."""

  def __init__(self, system_prompt: str = "ToyAgent"):
    super().__init__(system_prompt=system_prompt)

  def update_from_model(
      self, response: str, **kwargs: Any
  ) -> agent_types.Action:
    del kwargs
    action = agent_types.Action(action=response)
    step = agent_types.Step(
        model_response=response,
        thought="",
        action=action,
    )
    self._trajectory.steps.append(step)
    self._messages.append({"role": "assistant", "content": response})
    return action

  def get_current_step(self) -> agent_types.Step | None:
    return self._trajectory.steps[-1] if self._trajectory.steps else None


class ChatMockVocab(test_common.MockVocab):
  """Mock vocabulary supporting chat template formatting and standard token attributes."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.pad_token_id = 0
    self.bos_token_id = 1
    self.eos_token_id = 2

  def encode(self, text: str, **kwargs: Any) -> list[int]:
    return self.EncodeAsIds(text)

  def apply_chat_template(
      self,
      messages: Sequence[Any],
      tokenize: bool = False,
      add_generation_prompt: bool = False,
  ) -> str:
    del tokenize
    parts = []
    for msg in messages:
      if isinstance(msg, dict):
        parts.append(str(msg.get("content", "")))
      else:
        parts.append(str(msg))
    text = " ".join(parts)
    if add_generation_prompt:
      text += " <assistant>"
    return text


class MockChatParser(chat_parser_lib.DefaultChatTemplateParser):
  """Mock chat parser that formats messages into strings for the sampler."""

  def __init__(self, tokenizer: Any = None):
    super().__init__(tokenizer=tokenizer)
    self.assistant_token = "<assistant>"
    self._call_count = 0

  def parse(
      self,
      messages: Sequence[Any],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    del is_first_msg
    if not messages:
      return ""
    if isinstance(messages, str):
      return messages
    if hasattr(self.tokenizer, "apply_chat_template"):
      return self.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=add_generation_prompt,
      )
    parts = []
    for msg in messages:
      if isinstance(msg, dict):
        parts.append(str(msg.get("content", "")))
      else:
        parts.append(str(msg))
    text = " ".join(parts)
    if add_generation_prompt:
      text += f" {self.assistant_token}"
    return text

  def update_assistant_end_tokens(
      self, tokens: Any
  ) -> tuple[np.ndarray, int]:
    return np.asarray(tokens, dtype=np.int32).copy(), 0


# ==============================================================================
# Asymmetric Sharding Setup for ToyTransformer
# ==============================================================================

def build_toy_config() -> test_common.ModelConfig:
  return test_common.ModelConfig(
      vocab_size=32,
      num_layers=2,
      dtype=jnp.float32,
  )


def create_trainer_sharded_model(
    config: test_common.ModelConfig, rngs: nnx.Rngs, mesh: jax.sharding.Mesh
) -> test_common.ToyTransformer:
  """Shards model across 1D FSDP mesh: row-partitioned parameters."""
  model = test_common.ToyTransformer(config, rngs=rngs)
  state = nnx.state(model)

  def shard1(x):
    if x.ndim >= 2 and x.shape[0] % 4 == 0:
      spec = jax.sharding.PartitionSpec("fsdp", None)
    elif x.ndim >= 1 and x.shape[0] % 4 == 0:
      spec = jax.sharding.PartitionSpec("fsdp")
    else:
      spec = jax.sharding.PartitionSpec()
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

  sharded_state = jax.tree.map(shard1, state)
  nnx.update(model, sharded_state)
  return model


def create_rollout_sharded_model(
    config: test_common.ModelConfig, rngs: nnx.Rngs, mesh: jax.sharding.Mesh
) -> test_common.ToyTransformer:
  """Shards model across 1D TP mesh: column/head-partitioned parameters."""
  model = test_common.ToyTransformer(config, rngs=rngs)
  state = nnx.state(model)

  def shard2(x):
    if x.ndim >= 2 and x.shape[1] % 4 == 0:
      spec = jax.sharding.PartitionSpec(None, "tp")
    elif x.ndim >= 1 and x.shape[0] % 4 == 0:
      spec = jax.sharding.PartitionSpec("tp")
    else:
      spec = jax.sharding.PartitionSpec()
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

  sharded_state = jax.tree.map(shard2, state)
  nnx.update(model, sharded_state)
  return model


# ==============================================================================
# Multi-Process Worker Harness
# ==============================================================================

def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
  """Polls a TCP socket until it is open and accepting connections."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(0.2)
      if s.connect_ex((host, port)) == 0:
        return True
    time.sleep(0.05)
  return False


@contextlib.contextmanager
def running_multiprocess_worker(
    worker_fn: Any,
    *worker_args: Any,
) -> Iterator[
    Tuple[remote_lib.GrpcRemoteActorHandle, int, Any, Any]
]:
  """Spawns an isolated child process running GrpcRemoteExecutionServer over TCP."""
  # copybara:strip_begin
  ctx = multiprocessing.get_context("forkserver")
  # copybara:strip_end_and_replace_begin
  # ctx = multiprocessing.get_context("spawn")
  # copybara:replace_end

  port = portpicker.pick_unused_port()
  ready_event = ctx.Event()
  stop_event = ctx.Event()

  process = ctx.Process(
      target=worker_fn,
      args=(port, ready_event, stop_event, *worker_args),
      daemon=True,
  )
  process.start()

  handle = None
  try:
    if not ready_event.wait(timeout=180.0) or not _wait_for_port(
        "localhost", port, timeout=60.0
    ):
      if process.is_alive():
        process.terminate()
      raise RuntimeError(
          f"Worker process (PID {process.pid}) failed to start serving on port"
          f" {port} within timeout."
      )

    handle = remote_lib.GrpcRemoteActorHandle(
        target_address=f"grpc://localhost:{port}"
    )
    yield handle, port, process, stop_event
  finally:
    if handle is not None:
      try:
        asyncio.run(handle.close())
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    stop_event.set()
    process.join(timeout=3.0)
    if process.is_alive():
      logging.warning("Worker process PID %d hung; terminating.", process.pid)
      process.terminate()
      process.join(timeout=2.0)


# ==============================================================================
# Worker Process Server Functions
# ==============================================================================

def _run_mock_worker_process(
    port: int,
    ready_event: Any,
    stop_event: Any,
    worker_id: str,
    roles: set[str],
) -> None:
  """Runs a MockWorker in an isolated process for lifecycle verification."""
  worker = mock_worker.MockWorker(worker_id=worker_id, roles=roles)
  server = remote_lib.GrpcRemoteExecutionServer(worker)

  async def _serve() -> None:
    await server.start_serving_async(port=port)
    ready_event.set()
    while not stop_event.is_set():
      await asyncio.sleep(0.05)
    await server.stop_serving(grace=0.2)

  try:
    asyncio.run(_serve())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Mock worker crashed: %s", e)


def _read_device_array(arr: Any) -> np.ndarray:
  """Reads an array from device memory, bypassing JAX host-side caching."""
  if hasattr(arr, "_npy_value"):
    try:
      object.__setattr__(arr, "_npy_value", None)
    except Exception:
      pass
  try:
    fresh = jax.jit(lambda a: a + 0)(arr)
    return np.array(fresh)
  except Exception:
    return np.array(arr)


class InspectableTrainerWorker(trainer_worker_lib.TrainerWorker):
  """Exposes inspectable weights over RPC for test verification."""

  def get_weights(self) -> dict[str, np.ndarray]:
    state = nnx.state(self._trainer.model)
    names, arrays = raiden_synchronizer._filter_bindable(
        *raiden_synchronizer.flatten_weights(state)
    )
    return {name: _read_device_array(arr) for name, arr in zip(names, arrays)}

  def get_pid(self) -> int:
    return os.getpid()


class InspectableRolloutWorker(rollout_worker_lib.RolloutWorker):
  """Exposes inspectable weights over RPC for test verification."""

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self._generate_count = 0

  async def generate(self, requests, on_complete=None):
    req_list = list(requests) if isinstance(requests, Sequence) else [requests]
    self._generate_count += len(req_list)
    for i, req in enumerate(req_list):
      if hasattr(req, "generation_kwargs") and req.generation_kwargs is not None:
        req.generation_kwargs = dict(req.generation_kwargs)
        req.generation_kwargs["temperature"] = 1.0
        req.generation_kwargs["seed"] = (
            getattr(req, "group_index", 0) or 0
        ) * 100 + i + 1
    return await super().generate(requests, on_complete=on_complete)

  def get_generate_count(self) -> int:
    return self._generate_count

  def get_weights(self) -> dict[str, np.ndarray]:
    adapter = getattr(self.manager, "sampler", None)
    delegate = getattr(adapter, "raiden_sync_delegate", None)
    if (
        delegate is not None
        and delegate._synchronizers
        and delegate._synchronizers[0].arrays
    ):
      ws = delegate._synchronizers[0]
      return {name: _read_device_array(arr) for name, arr in zip(ws.names, ws.arrays)}
    sampler = getattr(self.sampler, "sampler", self.sampler)
    state = getattr(sampler, "transformer_state", None)
    if state is None:
      state = nnx.state(getattr(sampler, "transformer", sampler))
    names, arrays = raiden_synchronizer._filter_bindable(
        *raiden_synchronizer.flatten_weights(state)
    )
    return {name: _read_device_array(arr) for name, arr in zip(names, arrays)}

  def get_pid(self) -> int:
    return os.getpid()


def _run_trainer_worker_process(
    port: int,
    ready_event: Any,
    stop_event: Any,
) -> None:
  """Runs PeftTrainer in an isolated child process with 1D FSDP sharding."""
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  os.environ["JAX_PLATFORMS"] = "cpu"

  devices = np.array(jax.devices())
  mesh = jax.sharding.Mesh(devices, ("fsdp",))
  config = build_toy_config()

  with jax.set_mesh(mesh):
    model = create_trainer_sharded_model(config, nnx.Rngs(0), mesh)
    model(jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32))
    trainer = peft_trainer_v2.PeftTrainer(
        model=model,
        optimizer=optax.sgd(learning_rate=0.5),
        training_config=peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=100,
            max_steps=10,
            gradient_accumulation_steps=4,
        ),
        sampler_type="vanilla",
    )

  worker = InspectableTrainerWorker(
      trainer_factory=lambda: trainer, worker_id="trainer-0"
  )
  server = remote_lib.GrpcRemoteExecutionServer(worker)

  async def _serve() -> None:
    await server.start_serving_async(port=port)
    ready_event.set()
    while not stop_event.is_set():
      await asyncio.sleep(0.05)
    await server.stop_serving(grace=0.2)

  try:
    asyncio.run(_serve())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Trainer process crashed: %s", e)


def _run_rollout_worker_process(
    port: int,
    ready_event: Any,
    stop_event: Any,
    worker_id: str = "rollout-0",
) -> None:
  """Runs RolloutWorker with VanillaSamplerAdapter in an isolated process with 1D TP sharding."""
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  os.environ["JAX_PLATFORMS"] = "cpu"

  devices = np.array(jax.devices())
  mesh = jax.sharding.Mesh(devices, ("tp",))
  config = build_toy_config()

  with jax.set_mesh(mesh):
    model = create_rollout_sharded_model(config, nnx.Rngs(1), mesh)
    # Forward dummy input to initialize internal variables
    model(jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32))

  vocab = ChatMockVocab()
  tokenizer = tokenizer_adapter.TokenizerAdapter(vocab)
  chat_parser = MockChatParser(vocab)

  rollout_cfg = rollout_worker_lib.RolloutConfig(
      sampler_type="vanilla",
      weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
      max_prompt_length=8,
      max_tokens_to_generate=8,
      env_name=TOY_ENV_NAME,
      agent_name=TOY_AGENT_NAME,
  )
  worker_offset = 1 if worker_id == "rollout-0" else 7

  class _DiversifyingVanillaSamplerAdapter(
      vanilla_sampler_adapter.VanillaSamplerAdapter
  ):
    """Test adapter that guarantees distinct sampled tokens across group rollouts."""

    def __init__(self, *args: Any, **kwargs: Any):
      super().__init__(*args, **kwargs)
      self._sample_count = 0

    async def sample(self, sampling_requests: Any, **kwargs: Any) -> Any:
      res = await super().sample(sampling_requests, **kwargs)
      items = res if isinstance(res, list) else [res]
      for item in items:
        self._sample_count += 1
        if len(item.token_ids) > 0:
          item.token_ids = item.token_ids.copy()
          item.token_ids[0] = int(
              (item.token_ids[0] + self._sample_count * worker_offset) % 32
          )
      return res

  delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
      server_id=worker_id
  )
  sampler_adapter = _DiversifyingVanillaSamplerAdapter(
      server_id=worker_id,
      config=rollout_cfg,
      raiden_sync_delegate=delegate,
  )
  sampler_adapter.sampler = vanilla_sampler_lib.Sampler(
      transformer=model,
      tokenizer=vocab,
      cache_config=vanilla_sampler_lib.CacheConfig(
          cache_size=64,
          num_layers=config.num_layers,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
      ),
  )

  worker = InspectableRolloutWorker(
      worker_id=worker_id,
      config=rollout_cfg,
      sampler=sampler_adapter,
      tokenizer=tokenizer,
      chat_parser=chat_parser,
      max_concurrency=4,
  )
  server = remote_lib.GrpcRemoteExecutionServer(worker)

  async def _serve() -> None:
    await server.start_serving_async(port=port)
    ready_event.set()
    while not stop_event.is_set():
      await asyncio.sleep(0.05)
    await server.stop_serving(grace=0.2)

  try:
    asyncio.run(_serve())
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Rollout process crashed: %s", e)


# ==============================================================================
# Integration Test Suite
# ==============================================================================

class CpuDistributedRLIntegrationTest(parameterized.TestCase):
  """Comprehensive multi-process distributed RL integration test suite on CPU."""

  def test_m1_multiprocess_worker_rpc_lifecycle(self):
    """Milestone 1: Verifies 3-process isolation, gRPC connectivity, and lifecycle hooks."""
    with running_multiprocess_worker(
        _run_mock_worker_process, "trainer-mock", {"trainer"}
    ) as (t_handle, _, t_proc, _):
      with running_multiprocess_worker(
          _run_mock_worker_process, "rollout-mock", {"rollout"}
      ) as (r_handle, _, r_proc, _):
        # 1. Assert process isolation (all 3 PIDs distinct)
        current_pid = os.getpid()
        self.assertNotEqual(t_proc.pid, current_pid)
        self.assertNotEqual(r_proc.pid, current_pid)
        self.assertNotEqual(t_proc.pid, r_proc.pid)

        # 2. Assert RPC invocation and worker identity
        t_info = t_handle.submit("info")
        self.assertEqual(t_info.worker_id, "trainer-mock")
        self.assertIn("trainer", t_info.roles)

        r_info = r_handle.submit("info")
        self.assertEqual(r_info.worker_id, "rollout-mock")
        self.assertIn("rollout", r_info.roles)

        # 3. Assert ClusterOrchestrator bring-up and shutdown lifecycle
        cluster = orchestrator.ClusterOrchestrator()
        cluster.register_worker_handle(
            worker_id="trainer-mock",
            roles=[datatypes.Role.ACTOR],
            handle=t_handle,
        )
        cluster.register_worker_handle(
            worker_id="rollout-mock",
            roles=[datatypes.Role.ROLLOUT],
            handle=r_handle,
        )

        cluster.bring_up_workers()
        # Verify both workers transitioned through start
        t_heartbeat = t_handle.submit("heartbeat")
        r_heartbeat = r_handle.submit("heartbeat")
        self.assertEqual(t_heartbeat.state, datatypes.WorkerState.READY)
        self.assertEqual(r_heartbeat.state, datatypes.WorkerState.READY)

        cluster.shutdown()

  def test_m2_asymmetric_sharding_raiden_sync(self):
    """Milestone 2: Verifies standalone Raiden P2P weight transfer between 1D FSDP and 1D TP."""
    with running_multiprocess_worker(_run_trainer_worker_process) as (
        t_handle,
        _,
        t_proc,
        _,
    ):
      with running_multiprocess_worker(_run_rollout_worker_process) as (
          r_handle,
          _,
          r_proc,
          _,
      ):
        # Verify distinct worker PIDs
        self.assertEqual(t_handle.submit("get_pid"), t_proc.pid)
        self.assertEqual(r_handle.submit("get_pid"), r_proc.pid)

        # Check initial weights are different (different RNG seeds)
        init_t = t_handle.submit("get_weights")
        init_r = r_handle.submit("get_weights")
        self.assertEqual(len(init_t), len(init_r))
        self.assertFalse(
            all(np.allclose(init_t[k], init_r[k]) for k in init_t),
            "Initial weights unexpectedly identical between Trainer and Rollout.",
        )

        # Execute Raiden synchronization across processes via ClusterOrchestrator engine
        cluster = orchestrator.ClusterOrchestrator(weight_sync_mode="raiden")
        cluster.register_worker_handle(
            worker_id="trainer-0",
            roles=[datatypes.Role.ACTOR],
            handle=t_handle,
        )
        cluster.register_worker_handle(
            worker_id="rollout-0",
            roles=[datatypes.Role.ROLLOUT],
            handle=r_handle,
        )
        engine = cluster._create_engine()
        sync_ok = asyncio.run(engine.sync_weights())
        self.assertTrue(sync_ok)

        # Verify Rollout weights now match Trainer weights tensor-for-tensor
        synced_r = r_handle.submit("get_weights")
        self.assertEqual(len(synced_r), len(init_t))
        for k in init_t:
          self.assertIn(k, synced_r)
          np.testing.assert_allclose(
              synced_r[k],
              init_t[k],
              atol=1e-4,
              rtol=1e-4,
              err_msg=f"Mismatch at tensor {k} after Raiden sync.",
          )
        cluster.shutdown()

  @parameterized.named_parameters(
      dict(
          testcase_name="padded_batch_assembler_reward_fn",
          assembler_type="padded",
          reward_source="reward_fn",
      ),
      dict(
          testcase_name="sequence_packed_batch_assembler_reward_fn",
          assembler_type="packed",
          reward_source="reward_fn",
      ),
      dict(
          testcase_name="padded_batch_assembler_env_reward",
          assembler_type="padded",
          reward_source="env",
      ),
      dict(
          testcase_name="sequence_packed_batch_assembler_env_reward",
          assembler_type="packed",
          reward_source="env",
      ),
  )
  def test_m4_end_to_end_distributed_rl_workflow(
      self, assembler_type: str, reward_source: str = "reward_fn"
  ):
    """Milestone 4: Full end-to-end distributed RL loop with batch_size=2, trainer update, and Raiden sync."""

    with running_multiprocess_worker(_run_trainer_worker_process) as (
        t_handle,
        _,
        t_proc,
        _,
    ):
      with running_multiprocess_worker(_run_rollout_worker_process) as (
          r_handle,
          _,
          r_proc,
          _,
      ):
        # 1. Initialize ClusterOrchestrator
        cluster = orchestrator.ClusterOrchestrator(weight_sync_mode="raiden")
        cluster.register_worker_handle(
            worker_id="trainer-0",
            roles=[datatypes.Role.ACTOR],
            handle=t_handle,
        )
        cluster.register_worker_handle(
            worker_id="rollout-0",
            roles=[datatypes.Role.ROLLOUT],
            handle=r_handle,
        )

        # 2. Configure Loss Function on Trainer
        algo_config = algorithm_config.GRPOConfig(
            num_generations=2,
            epsilon=0.2,
            beta=0.0,
            temperature=1.0,
        )
        algo = algorithm_adapter.GRPOAdapter(
            algo_config=algo_config,
            mini_batch_size=2,
            train_micro_batch_size=1,
            max_packed_len=16,
            max_response_length=8,
        )

        t_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
        t_handle.submit(
            "with_gen_model_input_fn",
            algo.build_gen_model_input_fn(pad_id=0, eos_id=2),
        )

        # 3. Capture Initial Pre-Training Weights
        initial_t_weights = t_handle.submit("get_weights")

        # 4. Define 2 Prompt Items for batch_size=2 (4 rollouts total)
        prompt_items = [
            {
                "prompt": "prompt_alpha",
                "prompt_id": "prompt_0",
                "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
            },
            {
                "prompt": "prompt_beta",
                "prompt_id": "prompt_1",
                "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
            },
        ]

        # Alternating mock rewards ensure non-zero group variance in advantages
        def mock_reward_fn(
            unused_completion: str,
            metadata: dict[str, Any],
        ) -> float:
          return (
              1.0 if metadata.get("group_index", 0) % 2 == 0 else -1.0
          )

        reward_fns = [mock_reward_fn] if reward_source == "reward_fn" else None

        if assembler_type == "padded":
          assembler = batch_assembly.PaddedBatchAssembler(
              batch_size=1,
              num_generations=algo.num_generations,
              mini_batch_size=algo.mini_batch_size,
              max_prompt_length=16,
              max_response_length=16,
              pad_id=0,
          )
        elif assembler_type == "packed":
          assembler = batch_assembly.SequencePackedBatchAssembler(
              batch_size=1,
              num_generations=algo.num_generations,
              mini_batch_size=algo.mini_batch_size,
              max_packed_len=32,
              pad_id=0,
          )
        else:
          raise ValueError(f"Unknown assembler_type: {assembler_type}")

        program = rl_program.StandardRLProgram(
            algo=algo,
            dataset=prompt_items,
            max_steps=1,
            reward_fns=reward_fns,
            assembler=assembler,
            sync_weights=True,
        )

        # 5. Run Program Under Orchestrator Supervision
        cluster.run(program, bring_up=True)

        # 6. Assertions:
        self.assertIsNotNone(program.last_step_result)
        self.assertAlmostEqual(program.last_step_result.reward_mean, 0.0)
        self.assertAlmostEqual(program.last_step_result.reward_std, 1.0)
        self.assertNotIn("", program.metrics_logger._metrics)
        self.assertAlmostEqual(
            program.metrics_logger.get_metric("rewards", "mean", "train"), 0.0
        )
        self.assertTrue(
            program.metrics_logger.metric_exists(
                "rollout", "prompts/mean_length", "train"
            )
        )
        self.assertTrue(
            program.metrics_logger.metric_exists(
                "rollout", "completions/mean_length", "train"
            )
        )
        self.assertTrue(
            program.metrics_logger.metric_exists(
                "perf", "global_step_time", "train"
            )
        )

        # A. Verify Trainer weights actually changed (gradient accumulation + update occurred)
        post_t_weights = t_handle.submit("get_weights")
        diffs = {
            k: float(np.max(np.abs(initial_t_weights[k] - post_t_weights[k])))
            for k in initial_t_weights
        }
        for k, v in diffs.items():
          logging.info(
              "TRAINER_WEIGHT_DIFF %s: max_diff=%g, initial_mean=%g,"
              " post_mean=%g",
              k,
              v,
              float(np.mean(initial_t_weights[k])),
              float(np.mean(post_t_weights[k])),
          )
        has_updated = any(diff > 1e-5 for diff in diffs.values())
        self.assertTrue(
            has_updated,
            f"Trainer weights failed to update after RL training step: {diffs}",
        )

        # B. Verify Raiden moved weights from Trainer (1D FSDP) to Rollout (1D TP)
        post_r_weights = r_handle.submit("get_weights")
        self.assertEqual(len(post_r_weights), len(post_t_weights))
        for k in post_t_weights:
          self.assertIn(k, post_r_weights)
          np.testing.assert_allclose(
              post_r_weights[k],
              post_t_weights[k],
              atol=1e-4,
              rtol=1e-4,
              err_msg=f"Rollout weight mismatch at tensor {k} after Raiden sync.",
          )

        program.close()
        cluster.shutdown()

  def test_m5_multi_rollout_workers_end_to_end_training_step(self):
    """Milestone 5: E2E distributed RL loop across MULTIPLE rollout workers via PoolExecutionSession."""
    with running_multiprocess_worker(_run_trainer_worker_process) as (
        t_handle,
        _,
        _,
        _,
    ):
      with running_multiprocess_worker(
          _run_rollout_worker_process, "rollout-0"
      ) as (r0_handle, _, _, _):
        with running_multiprocess_worker(
            _run_rollout_worker_process, "rollout-1"
        ) as (r1_handle, _, _, _):
          cluster = orchestrator.ClusterOrchestrator(weight_sync_mode="raiden")
          cluster.register_worker_handle(
              worker_id="trainer-0",
              roles=[datatypes.Role.ACTOR],
              handle=t_handle,
          )
          cluster.register_worker_handle(
              worker_id="rollout-0",
              roles=[datatypes.Role.ROLLOUT],
              handle=r0_handle,
          )
          cluster.register_worker_handle(
              worker_id="rollout-1",
              roles=[datatypes.Role.ROLLOUT],
              handle=r1_handle,
          )

          algo_config = algorithm_config.GRPOConfig(
              num_generations=2,
              epsilon=0.2,
              beta=0.0,
              temperature=1.0,
          )
          algo = algorithm_adapter.GRPOAdapter(
              algo_config=algo_config,
              mini_batch_size=2,
              train_micro_batch_size=1,
              max_packed_len=16,
              max_response_length=8,
          )

          t_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
          t_handle.submit(
              "with_gen_model_input_fn",
              algo.build_gen_model_input_fn(pad_id=0, eos_id=2),
          )

          initial_t_weights = t_handle.submit("get_weights")

          prompt_items = [
              {
                  "prompt": "prompt_alpha",
                  "prompt_id": "prompt_0",
                  "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
              },
              {
                  "prompt": "prompt_beta",
                  "prompt_id": "prompt_1",
                  "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
              },
          ]

          def mock_reward_fn(
              unused_completion: str,
              metadata: dict[str, Any],
          ) -> float:
            return 1.0 if metadata.get("group_index", 0) % 2 == 0 else -1.0

          assembler = batch_assembly.PaddedBatchAssembler(
              batch_size=1,
              num_generations=algo.num_generations,
              mini_batch_size=algo.mini_batch_size,
              max_prompt_length=16,
              max_response_length=16,
              pad_id=0,
          )

          program = rl_program.StandardRLProgram(
              algo=algo,
              dataset=prompt_items,
              max_steps=1,
              reward_fns=[mock_reward_fn],
              assembler=assembler,
              sync_weights=True,
          )

          cluster.run(program, bring_up=True)

          # 1. Verify both rollout workers generated trajectories via PoolExecutionSession
          r0_count = r0_handle.submit("get_generate_count")
          r1_count = r1_handle.submit("get_generate_count")
          self.assertGreater(r0_count, 0)
          self.assertGreater(r1_count, 0)
          self.assertEqual(r0_count + r1_count, 4)

          # 2. Verify training step completed and updated Trainer weights
          self.assertIsNotNone(program.last_step_result)
          post_t_weights = t_handle.submit("get_weights")
          has_updated = any(
              float(np.max(np.abs(initial_t_weights[k] - post_t_weights[k])))
              > 1e-5
              for k in initial_t_weights
          )
          self.assertTrue(has_updated)

          # 3. Verify both Rollout workers synced updated weights via Raiden
          for wid, r_handle in (("rollout-0", r0_handle), ("rollout-1", r1_handle)):
            post_r_weights = r_handle.submit("get_weights")
            for k in post_t_weights:
              np.testing.assert_allclose(
                  post_r_weights[k],
                  post_t_weights[k],
                  atol=1e-4,
                  rtol=1e-4,
                  err_msg=f"{wid} weight mismatch at tensor {k} after Raiden sync.",
              )

          program.close()
          cluster.shutdown()

  def test_m6_rollout_worker_crash_eviction_and_rejoin(self):
    """Milestone 6: Rollout worker process crash, automatic eviction, re-dispatch, and dynamic rejoin with Raiden weight sync."""
    with running_multiprocess_worker(_run_trainer_worker_process) as (
        t_handle,
        _,
        _,
        _,
    ):
      with running_multiprocess_worker(
          _run_rollout_worker_process, "rollout-0"
      ) as (r0_handle, _, r0_proc, _):
        with running_multiprocess_worker(
            _run_rollout_worker_process, "rollout-1"
        ) as (r1_handle, _, r1_proc, _):
          cluster = orchestrator.ClusterOrchestrator(weight_sync_mode="raiden")
          cluster.register_worker_handle(
              worker_id="trainer-0",
              roles=[datatypes.Role.ACTOR],
              handle=t_handle,
          )
          cluster.register_worker_handle(
              worker_id="rollout-0",
              roles=[datatypes.Role.ROLLOUT],
              handle=r0_handle,
          )
          cluster.register_worker_handle(
              worker_id="rollout-1",
              roles=[datatypes.Role.ROLLOUT],
              handle=r1_handle,
          )

          algo_config = algorithm_config.GRPOConfig(
              num_generations=2,
              epsilon=0.2,
              beta=0.0,
              temperature=1.0,
          )
          algo = algorithm_adapter.GRPOAdapter(
              algo_config=algo_config,
              mini_batch_size=2,
              train_micro_batch_size=1,
              max_packed_len=16,
              max_response_length=8,
          )

          t_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
          t_handle.submit(
              "with_gen_model_input_fn",
              algo.build_gen_model_input_fn(pad_id=0, eos_id=2),
          )

          cluster.bring_up_workers()
          assert cluster.engine is not None

          # 1. Bootstrap initial weights (policy_version=0) on both workers.
          asyncio.run(cluster.engine.prepare_rollout_policy())
          self.assertEqual(
              cluster.registry.state("rollout-0"),
              worker_registry.MembershipState.ACTIVE,
          )
          self.assertEqual(
              cluster.registry.state("rollout-1"),
              worker_registry.MembershipState.ACTIVE,
          )

          prompt_items = [
              {
                  "prompt": "prompt_alpha",
                  "prompt_id": "prompt_0",
                  "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
              },
              {
                  "prompt": "prompt_beta",
                  "prompt_id": "prompt_1",
                  "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
              },
          ]

          # 2. Kill rollout-1 process abruptly (SIGKILL) and verify
          # fault-tolerant dispatch + polling evicts rollout-1 and retries on
          # rollout-0.
          r1_proc.kill()
          r1_proc.join(timeout=5.0)

          async def _dispatch_and_drain(
              policy_version: int = 0,
              prompts: list[dict[str, Any]] | None = None,
          ):
            assert cluster.engine is not None
            await cluster.engine.dispatch_rollouts(
                prompts if prompts is not None else prompt_items,
                num_generations=2,
                policy_version=policy_version,
            )
            collected = []
            deadline = time.monotonic() + 30.0
            while len(collected) < 4 and time.monotonic() < deadline:
              batch = await cluster.engine.poll_rollouts(timeout_s=2.0)
              collected.extend(batch)
            return collected

          items = asyncio.run(_dispatch_and_drain(policy_version=0))
          self.assertLen(items, 4)
          for item in items:
            self.assertIn(
                item.status,
                (datatypes.TrajectoryStatus.SUCCEEDED, "SUCCEEDED"),
            )
          self.assertEqual(
              cluster.registry.state("rollout-1"),
              worker_registry.MembershipState.EVICTED,
          )
          self.assertEqual(
              cluster.registry.state("rollout-0"),
              worker_registry.MembershipState.ACTIVE,
          )
          self.assertEqual(
              cluster.worker_handles(datatypes.Role.ROLLOUT), [r0_handle]
          )

          # 3. Spawn a replacement rollout-1 process and dynamically re-register
          # it.
          with running_multiprocess_worker(
              _run_rollout_worker_process, "rollout-1"
          ) as (r1_new_handle, _, r1_new_proc, _):
            cluster.register_worker_handle(
                worker_id="rollout-1",
                roles=[datatypes.Role.ROLLOUT],
                handle=r1_new_handle,
            )
            cluster.wait_for_pending_bring_ups(timeout=60.0)
            self.assertEqual(
                cluster.registry.state("rollout-1"),
                worker_registry.MembershipState.PENDING_WEIGHT_SYNC,
            )

            # 4. Run a full training step with Raiden weight sync; verify
            # rollout-1 is promoted to ACTIVE and receives the updated weights.
            def mock_reward_fn(
                unused_completion: str,
                metadata: dict[str, Any],
            ) -> float:
              return 1.0 if metadata.get("group_index", 0) % 2 == 0 else -1.0

            assembler = batch_assembly.PaddedBatchAssembler(
                batch_size=1,
                num_generations=algo.num_generations,
                mini_batch_size=algo.mini_batch_size,
                max_prompt_length=16,
                max_response_length=16,
                pad_id=0,
            )
            program = rl_program.StandardRLProgram(
                algo=algo,
                dataset=prompt_items,
                max_steps=1,
                reward_fns=[mock_reward_fn],
                assembler=assembler,
                sync_weights=True,
            )
            cluster.run(program, bring_up=False)

            self.assertEqual(
                cluster.registry.state("rollout-1"),
                worker_registry.MembershipState.ACTIVE,
            )
            post_t_weights = t_handle.submit("get_weights")
            for wid, r_handle in (
                ("rollout-0", r0_handle),
                ("rollout-1", r1_new_handle),
            ):
              post_r_weights = r_handle.submit("get_weights")
              for k in post_t_weights:
                np.testing.assert_allclose(
                    post_r_weights[k],
                    post_t_weights[k],
                    atol=1e-4,
                    rtol=1e-4,
                    err_msg=(
                        f"{wid} weight mismatch at tensor {k} after rejoin"
                        " Raiden sync."
                    ),
                )

            # 5. Simulate a silent TPU/vLLM wedge on rollout-1 via SIGSTOP
            # (process stays alive and TCP socket stays open, but stops
            # responding) and verify rollout_task_timeout_s evicts rollout-1
            # and retries its in-flight rollouts on rollout-0.
            wedge_prompts = [
                {
                    "prompt": "prompt_alpha",
                    "prompt_id": "prompt_w0",
                    "max_turns": 1,
                    "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
                },
                {
                    "prompt": "prompt_beta",
                    "prompt_id": "prompt_w1",
                    "max_turns": 1,
                    "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
                },
            ]
            cluster.engine.set_rollout_task_timeout_s(5.0)
            os.kill(r1_new_proc.pid, signal.SIGSTOP)
            try:
              wedge_items = asyncio.run(
                  _dispatch_and_drain(
                      policy_version=cluster.engine.policy_version,
                      prompts=wedge_prompts,
                  )
              )
              self.assertLen(wedge_items, 4)
              for item in wedge_items:
                self.assertIn(
                    item.status,
                    (datatypes.TrajectoryStatus.SUCCEEDED, "SUCCEEDED"),
                )
              self.assertEqual(
                  cluster.registry.state("rollout-1"),
                  worker_registry.MembershipState.EVICTED,
              )
              self.assertEqual(
                  cluster.registry.state("rollout-0"),
                  worker_registry.MembershipState.ACTIVE,
              )
              self.assertEqual(
                  cluster.worker_handles(datatypes.Role.ROLLOUT), [r0_handle]
              )
            finally:
              os.kill(r1_new_proc.pid, signal.SIGCONT)

            # 6. S5: Poison prompt / trajectory failure isolation.
            # Dispatch a prompt whose group_index=1 triggers an environment
            # failure alongside a healthy prompt, and verify that the failed
            # trajectory returns as a FAILED placeholder with preserved
            # prompt_id/group_index/policy_version while healthy trajectories
            # succeed and rollout-0 stays ACTIVE.
            poison_prompts = [
                {
                    "prompt": "prompt_alpha",
                    "prompt_id": "prompt_poison",
                    "max_turns": 1,
                    "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
                    "metadata": {"env_config": {"poison_group_index": 1}},
                },
                {
                    "prompt": "prompt_beta",
                    "prompt_id": "prompt_clean",
                    "max_turns": 1,
                    "generation_kwargs": {"max_tokens": 4, "temperature": 1.0},
                },
            ]
            s5_items = asyncio.run(
                _dispatch_and_drain(
                    policy_version=cluster.engine.policy_version,
                    prompts=poison_prompts,
                )
            )
            self.assertLen(s5_items, 4)
            failed_items = [
                it
                for it in s5_items
                if it.status in (datatypes.TrajectoryStatus.FAILED, "FAILED")
            ]
            succeeded_items = [
                it
                for it in s5_items
                if it.status
                in (datatypes.TrajectoryStatus.SUCCEEDED, "SUCCEEDED")
            ]
            self.assertLen(failed_items, 1)
            self.assertLen(succeeded_items, 3)
            self.assertEqual(failed_items[0].prompt_id, "prompt_poison")
            self.assertEqual(failed_items[0].group_index, 1)
            self.assertEqual(
                failed_items[0].policy_version, cluster.engine.policy_version
            )
            self.assertFalse(failed_items[0].is_valid)
            self.assertGreaterEqual(
                cluster.engine.fault_tolerance_metrics[
                    "terminal_failed_trajectories_total"
                ],
                1,
            )
            self.assertEqual(
                cluster.registry.state("rollout-0"),
                worker_registry.MembershipState.ACTIVE,
            )

            # 7. S6: Total rollout worker loss -> bounded wait ->
            # NoHealthyRolloutWorkersError without hanging.
            cluster.engine.set_max_zero_worker_wait_s(0.2)
            r0_proc.kill()
            r0_proc.join(timeout=5.0)
            with self.assertRaises(datatypes.NoHealthyRolloutWorkersError):
              asyncio.run(
                  _dispatch_and_drain(
                      policy_version=cluster.engine.policy_version,
                      prompts=wedge_prompts,
                  )
              )
            self.assertGreater(
                cluster.engine.fault_tolerance_metrics["zero_worker_seconds"],
                0.0,
            )

            program.close()
            cluster.shutdown()


if __name__ == "__main__":
  # copybara:strip_begin
  absltest.main()
  # copybara:strip_end_and_replace_begin
  # absltest.main()
  # copybara:replace_end
