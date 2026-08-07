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

"""Mock Environment, Sampler, and Agent implementations for testing."""

import asyncio
import dataclasses
import functools
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import unittest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import remote_execution
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment


class MockEnvironment(base_environment.BaseTaskEnv):
  """Mock Environment simulating Kubernetes pod interactions with latency."""

  def __init__(self, env_id: str = "mock_env_01", delay_seconds: float = 0.05):
    super().__init__()
    self.env_id = env_id
    self.delay_seconds = delay_seconds
    self.step_count = 0

  def reset(
      self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
  ) -> Tuple[Any, Dict[str, Any]]:
    self.step_count = 0
    obs = f"[{self.env_id}] Environment reset ready. Initial observation."
    return obs, {}

  def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    self.step_count += 1
    action_str = str(action)
    if "FINAL_ANSWER" in action_str or self.step_count >= 4:
      return (
          f"[{self.env_id}] Task solved!",
          1.0,
          True,
          {"latency": self.delay_seconds, "steps": self.step_count},
      )
    return (
        (
            f"[{self.env_id}] Step {self.step_count} executed. Obs for:"
            f" {action_str}"
        ),
        0.1,
        False,
        {"latency": self.delay_seconds, "steps": self.step_count},
    )

  def close(self) -> None:
    pass


registry.ENV_REGISTRY.register("mock_env")(MockEnvironment)


class MockEnvironmentPool:
  """A mock environment pool for rollout collection."""

  def __init__(
      self,
      pool_size: int = 10,
      default_delay: float = 0.05,
      env_factory: Any = None,
  ) -> None:
    self.pool_size = pool_size
    self.default_delay = default_delay
    env_cls = env_factory or registry.ENV_REGISTRY.get("mock_env")
    self._envs = [env_cls(f"env_{i}", default_delay) for i in range(pool_size)]

  def acquire_env(
      self, config: Optional[Dict[str, Any]] = None
  ) -> MockEnvironment:
    """Acquires a mock environment instance from the pool."""
    delay = (
        config.get("delay_seconds", self.default_delay)
        if config
        else self.default_delay
    )
    if self._envs:
      env = self._envs.pop()
      env.delay_seconds = delay
      return env
    return MockEnvironment("env_dynamic", delay)

  def release_env(self, env: MockEnvironment) -> None:
    """Releases a mock environment instance back into the pool."""
    self._envs.append(env)


class MockTokenizer:
  """Mock tokenizer returning simple ASCII ordinal token IDs."""

  def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
    del add_special_tokens
    return [ord(c) % 1000 for c in text] or [101]

  def dedup_bos_ids(self, tokens: List[int]) -> List[int]:
    return tokens


class MockChatParser:
  """Mock chat parser formatting role messages for tokenization."""

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    del add_generation_prompt, is_first_msg
    return " ".join(m.get("content", "") for m in messages)

  def update_assistant_end_tokens(
      self, tokens: np.ndarray
  ) -> Tuple[np.ndarray, int]:
    return tokens, 0


class MockBaseSamplerImpl(sampler_lib.VanillaSamplerAdapter):
  """Mock BaseSamplerImpl simulating LLM generation and Raiden KV transfer."""

  def __init__(
      self,
      sampler_name: str = "mock",
      default_delay: float = 0.05,
      server_id: str = "mock_server",
      **kwargs,
  ):
    super().__init__(server_id=server_id or sampler_name, **kwargs)
    self.sampler_name = sampler_name
    self.default_delay = default_delay
    self.migration_history: List[Dict[str, Any]] = []
    self._turn_counters: Dict[str, int] = {}
    self.current_policy_version = 0
    self.tokenizer = MockTokenizer()
    self.chat_parser = MockChatParser()

  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
          | Any
      ) = None,
      **kwargs,
  ) -> Any:
    """Simulates LLM inference latency and agentic turn responses."""
    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")
    delay = kwargs.get("delay_seconds", self.default_delay)
    await asyncio.sleep(delay)

    req_id_str = "default"
    if (
        hasattr(sampling_requests, "request_id")
        and sampling_requests.request_id
    ):
      req_id_str = str(sampling_requests.request_id)
    turn = self._turn_counters.get(req_id_str, 0)
    self._turn_counters[req_id_str] = turn + 1

    min_turns = kwargs.get("min_turns", 1)
    if turn >= min_turns or kwargs.get("force_finish", False):
      ans = kwargs.get("answer", f"result_for_{req_id_str}")
      txt = f"FINAL_ANSWER: [{self.sampler_name}] {ans}"
    else:
      txt = f"TOOL_CALL: search(query='turn {turn} for {req_id_str}')"
    tokens = np.array([101, 102], dtype=np.int32)
    return base_sampler_lib.SamplingResponse(
        text=txt,
        token_ids=tokens,
        logprobs=np.zeros_like(tokens, dtype=np.float32),
    )

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Simulates Raiden P2P KV-cache transfer across TPU slices."""
    del kwargs
    await asyncio.sleep(0.01)
    self.migration_history.append({
        "source": source_server_id,
        "target": target_server_id,
        "tokens_transferred": len(token_ids),
    })
    return True

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    metadata = kwargs.pop("metadata", sync_request)
    return getattr(metadata, "new_policy_version", None)

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    metadata = kwargs.pop("metadata", sync_request)
    self.current_policy_version = metadata.new_policy_version
    return self.current_policy_version


class MockAgent(base_agent.ConversationAgentBase):
  """Mock Agent tracking turn history and memory from environment observations."""

  def __init__(self, system_prompt: str = "Mock Agent"):
    super().__init__(system_prompt=system_prompt)
    self.history: List[Tuple[Any, float]] = []

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    action = agent_types.Action(action=response)
    step = agent_types.Step(
        model_response=response,
        thought="",
        action=action,
    )
    self._trajectory.steps.append(step)
    self._messages.append({"role": "assistant", "content": response})
    return action


registry.AGENT_REGISTRY.register("mock_agent")(MockAgent)


class MockTrainer:
  """Mock Trainer service running on Trainer TPU slice."""

  def __init__(self, trainer_id: str = "trainer_01"):
    self.trainer_id = trainer_id
    self.current_step = 0
    self.current_policy_version = 1

  def train_step(self, batch: Any) -> int:
    """Simulates a forward-backward training step."""
    del batch
    self.current_step += 1
    self.current_policy_version += 1
    return self.current_policy_version

  def push_weights(self, peer_addresses: Optional[List[str]] = None) -> int:
    """Unimplemented weight synchronization API."""
    del peer_addresses
    raise NotImplementedError("Weight sync API is unimplemented.")


TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]


class MockGlobalOrchestrator:
  """TOP-LEVEL DRIVER NODE IN DISTRIBUTED-RL.

  Coordinates the RL training loop across distributed RolloutWorker
  actor handles (Phase 1: Rollout generation via ActorPool) and Trainer
  instances (Phase 2: Training & Phase 3: Weight Synchronization).
  """

  def __init__(
      self,
      orchestrator_id: str,
      rollout_actors: Optional[
          Sequence[Union[remote_execution.ActorHandle, Any]]
      ] = None,
      trainer_actor: Optional[Union[remote_execution.ActorHandle, Any]] = None,
      **kwargs,
  ):
    self.orchestrator_id = orchestrator_id
    actors = rollout_actors or kwargs.get("rollout_workers", [])
    self.actor_handles: List[remote_execution.ActorHandle] = []
    for w in actors:
      if isinstance(w, str):
        self.actor_handles.append(remote_execution.ActorHandle.from_address(w))
      elif isinstance(w, remote_execution.ActorHandle):
        self.actor_handles.append(w)
      elif (
          hasattr(w, "actor_handle") and getattr(w, "actor_handle") is not None
      ):
        self.actor_handles.append(getattr(w, "actor_handle"))
      else:
        raise TypeError(
            "Expected ActorHandle, string URI, or object exposing"
            f" actor_handle, got {type(w)}"
        )

    trainer = trainer_actor or kwargs.get("trainer_worker")
    if isinstance(trainer, str):
      self.trainer_handle: Optional[remote_execution.ActorHandle] = (
          remote_execution.ActorHandle.from_address(trainer)
      )
    elif isinstance(trainer, remote_execution.ActorHandle):
      self.trainer_handle = trainer
    else:
      self.trainer_handle = getattr(trainer, "actor_handle", None)
    self.worker_pool = remote_execution.RoutingActorPool(self.actor_handles)

  def start_workers(self) -> None:
    """Starts all managed rollout worker instances via actor handles."""
    for handle in self.actor_handles:
      try:
        handle.submit("start")
      except (AttributeError, RuntimeError):
        pass

  def stop_workers(self) -> None:
    """Stops all managed rollout worker instances via actor handles."""
    for handle in self.actor_handles:
      try:
        handle.submit("stop")
      except (AttributeError, RuntimeError):
        pass

  async def collect_rollout_batch(
      self,
      requests: Sequence[datatypes.RolloutRequest],
      group_size: int = 1,
  ) -> List[TrajectoryOrError]:
    """Dispatches requests via ActorPool and collects out-of-order results."""
    if not self.actor_handles:
      raise ValueError(
          "No valid ActorHandles available in MockGlobalOrchestrator."
      )
    if group_size < 1:
      raise ValueError(f"group_size must be at least 1, got {group_size}")

    fanned_out_requests = []
    for req in requests:
      for g_idx in range(group_size):
        gid = str(g_idx) if group_size > 1 else req.group_offset_id
        fanned_out_requests.append(
            dataclasses.replace(req, group_offset_id=gid)
        )
    tasks: List[Tuple[str, str, Sequence[Any], Dict[str, Any]]] = [
        (req.request_id or "req", "generate", (req,), {})
        for req in fanned_out_requests
    ]
    trajectories: List[TrajectoryOrError] = []
    async for traj in self.worker_pool.as_completed_stream(tasks):
      trajectories.append(traj)

    return trajectories

  async def run_training_step(
      self, trajectories: Sequence[TrajectoryOrError]
  ) -> int:
    """Executes Phase 2 training step via trainer actor handle."""
    valid_trajs = [
        t for t in trajectories if isinstance(t, trajectory_lib.Trajectory)
    ]
    if self.trainer_handle:
      return await self.trainer_handle.asubmit("train_step", valid_trajs)
    return 1

  def synchronize_weights(
      self, metadata: datatypes.WeightSyncMetadata
  ) -> List[int]:
    """Executes Phase 3 weight synchronization across rollout handles."""
    if self.trainer_handle:
      self.trainer_handle.submit(
          "push_weights", metadata.source_endpoints or []
      )

    versions: List[int] = []
    for handle in self.actor_handles:
      handle.submit("pre_weight_sync", metadata)
    for handle in self.actor_handles:
      v = handle.submit("weight_sync", metadata)
      versions.append(v)
    return versions


def is_tpu_available() -> bool:
  """Checks whether physical TPU hardware (XLA/PjRt TPU devices) is available without prematurely locking PjRt."""
  if (
      os.path.exists("/dev/accel0")
      or "TPU_NAME" in os.environ
      or os.environ.get("UNITTEST_ON_FORGE") == "1"
  ):
    return True
  try:
    import importlib  # pylint: disable=g-import-not-at-top

    jax = importlib.import_module("jax")
    if jax.default_backend() == "tpu" or any(
        getattr(d, "device_kind", "") == "TPU" for d in jax.devices()
    ):
      return True
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return False


def tpu_only(func: Callable[..., Any]) -> Callable[..., Any]:
  """Test decorator that skips the decorated test method if physical TPU hardware is not available."""

  @functools.wraps(func)
  def wrapper(*args, **kwargs) -> Any:
    if not is_tpu_available():
      raise unittest.SkipTest(
          "Skipping test: requires physical TPU hardware / XLA TPU devices."
      )
    return func(*args, **kwargs)

  return wrapper
