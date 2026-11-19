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

"""Mock and fake implementations for Tunix distributed RL tests and benchmarks."""

import asyncio
import collections
import dataclasses
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import unittest
import zlib
import numpy as np
from tunix.common import configs
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import registry
from tunix.dist_rl.rollout import sampler as base_sampler_lib
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment


class MockEnvironment(base_environment.BaseTaskEnv):
  """Mock Environment simulating Kubernetes pod interactions with latency."""

  def __init__(
      self,
      env_id: str = "mock_env_01",
      delay_seconds: float = 0.05,
      max_steps: int | None = None,
      **kwargs: Any,
  ):
    del kwargs
    super().__init__()
    self.env_id = env_id
    self.delay_seconds = delay_seconds
    if max_steps is not None:
      self.max_steps = max_steps
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


_MOCK_ROUTED_SLAB_ROWS = 4096
_MAX_TURN_COUNTER_ENTRIES = 4096


class TokenizerLike(Protocol):
  """Minimal tokenizer protocol consumed by `MockBaseSamplerImpl`."""

  def encode(self, text: str) -> Sequence[int]:
    ...


def _build_routed_slab(
    shape_spec: tuple[int, int] | None, num_experts: int
) -> np.ndarray | None:
  """Pre-allocates a deterministic pseudo-random int16 MoE routing slab."""
  if shape_spec is None:
    return None
  num_layers, top_k = int(shape_spec[0]), int(shape_spec[1])
  if num_layers <= 0 or top_k <= 0 or num_experts <= 0:
    raise ValueError(
        "routed_experts_shape_spec and num_experts must be positive, got"
        f" shape={shape_spec!r}, num_experts={num_experts}."
    )
  rng = np.random.default_rng(seed=20261110)
  return rng.integers(
      0,
      num_experts,
      size=(_MOCK_ROUTED_SLAB_ROWS * 2, num_layers, top_k),
      dtype=np.int16,
  )


class MockBaseSamplerImpl(base_sampler_lib.Sampler):
  """Mock Sampler simulating LLM generation and Raiden KV transfer."""

  def __init__(
      self,
      sampler_name: str = "mock",
      default_delay: float = 0.05,
      server_id: str = "mock_server",
      *,
      response_tokens_per_turn: int = 2,
      min_turns: int = 1,
      routed_experts_shape_spec: tuple[int, int] | None = None,
      num_experts: int = 256,
      eos_token_id: int | None = None,
      tokenizer: TokenizerLike | None = None,
      **kwargs,
  ):
    self.server_id = server_id or sampler_name
    self.config = kwargs.pop("config", None) or configs.RolloutConfig()
    self.sampler_name = sampler_name
    self.default_delay = float(default_delay)
    self.response_tokens_per_turn = int(response_tokens_per_turn)
    self.min_turns = max(0, int(min_turns))
    self._routed_experts_shape = (
        (int(routed_experts_shape_spec[0]), int(routed_experts_shape_spec[1]))
        if routed_experts_shape_spec is not None
        else None
    )
    self.num_experts = int(num_experts)
    self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None
    self._routed_slab = _build_routed_slab(
        self._routed_experts_shape, self.num_experts
    )
    self.migration_history: List[Dict[str, Any]] = []
    self.sample_calls: int = 0
    self._turn_counters: collections.OrderedDict[str, int] = (
        collections.OrderedDict()
    )
    self.current_policy_version: int = 0
    self.staging_policy_version: int | None = None
    self.fail_request_ids: set[str] = set()
    self.tokenizer: TokenizerLike = (
        tokenizer if tokenizer is not None else MockTokenizer()
    )
    self.chat_parser = MockChatParser()

  @property
  def policy_version(self) -> int:
    return self.current_policy_version

  def routed_experts_shape(self) -> tuple[int, int] | None:
    """Returns `(num_layers, top_k)` for `Sampler.routed_experts_shape()`."""
    return self._routed_experts_shape

  def initialize(self) -> None:
    pass

  async def start(self, **kwargs) -> Any:
    del kwargs
    return True

  async def stop(self, **kwargs) -> Any:
    del kwargs
    return True

  async def pause(self, **kwargs) -> Any:
    del kwargs
    return True

  async def resume(self, **kwargs) -> Any:
    del kwargs
    return True

  async def get_mesh(self, **kwargs) -> Any:
    del kwargs
    return None

  def get_target_state(self) -> Any:
    return None

  async def get_weight_sync_metadata(
      self, **kwargs
  ) -> list[weight_sync.WorkUnitMetadata]:
    del kwargs
    return [
        weight_sync.WorkUnitMetadata(
            unit=weight_sync.WorkUnitId(
                job_name=self.server_id, job_replica_id="0"
            ),
            shards=("127.0.0.1:20001",),
            control_plane_rpc_address="127.0.0.1:20501",
            global_shape=(4,),
            mesh_shape=(1,),
            layout=(0,),
            item_size=4,
        )
    ]

  async def bind_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del sync_request, kwargs
    return None

  async def post_weight_sync(
      self,
      sync_request: (
          datatypes.WeightSyncRequest | datatypes.WeightSyncMetadata | None
      ) = None,
      **kwargs,
  ) -> Any:
    del kwargs
    if self.staging_policy_version is not None:
      self.current_policy_version = self.staging_policy_version
      self.staging_policy_version = None
    elif isinstance(sync_request, datatypes.WeightSyncRequest):
      self.current_policy_version = int(sync_request.policy_version)
    return True

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del sync_request, kwargs
    self.staging_policy_version = None
    return True

  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    del req_id, kwargs
    return "SUCCESS"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    del kwargs
    return base_sampler_lib.LoadInfo()

  def _build_turn_tokens(
      self,
      *,
      max_tokens: int | None,
      is_final_turn: bool,
  ) -> np.ndarray:
    """Builds generated token IDs for one assistant turn."""
    target_len = self.response_tokens_per_turn
    if max_tokens is not None and max_tokens > 0:
      target_len = min(target_len, int(max_tokens))
    tokens = 101 + (np.arange(target_len, dtype=np.int32) % 799)
    if is_final_turn and self.eos_token_id is not None and tokens.size > 0:
      tokens[-1] = self.eos_token_id
    return tokens

  def _resolve_prompt_ids(self, req: Any) -> np.ndarray:
    prompt = (
        req.prompt if isinstance(req, base_sampler_lib.SamplingRequest) else req
    )
    if isinstance(prompt, np.ndarray):
      return np.asarray(prompt, dtype=np.int32).reshape(-1)
    return np.asarray(self.tokenizer.encode(str(prompt)), dtype=np.int32)

  def _maybe_build_routed(
      self,
      *,
      req_id_str: str,
      turn: int,
      prompt_len: int,
      gen_len: int,
      sp: base_sampler_lib.SamplingParams | None,
  ) -> np.ndarray | None:
    slab = self._routed_slab
    if slab is None or sp is None or not sp.return_routed_experts:
      return None
    prompt_start = int(sp.routed_experts_prompt_start)
    if prompt_start < 0 or prompt_start > prompt_len:
      raise ValueError(
          "routed_experts_prompt_start must satisfy"
          f" 0 <= prompt_start <= prompt_len ({prompt_len}), got"
          f" {prompt_start}."
      )
    routed_len = (prompt_len - prompt_start) + gen_len
    if routed_len <= 0:
      return np.empty((0, *slab.shape[1:]), dtype=np.int16)
    key = f"{req_id_str}:{turn}".encode("utf-8")
    offset = zlib.crc32(key) % _MOCK_ROUTED_SLAB_ROWS
    window = slab[offset : offset + _MOCK_ROUTED_SLAB_ROWS]
    if routed_len <= _MOCK_ROUTED_SLAB_ROWS:
      return np.array(window[:routed_len], copy=True)
    out = np.empty((routed_len, *slab.shape[1:]), dtype=np.int16)
    for pos in range(0, routed_len, _MOCK_ROUTED_SLAB_ROWS):
      chunk = min(_MOCK_ROUTED_SLAB_ROWS, routed_len - pos)
      out[pos : pos + chunk] = window[:chunk]
    return out

  def _sample_single(
      self, req: Any, base_kwargs: Dict[str, Any]
  ) -> base_sampler_lib.SamplingResponse:
    """Builds a single `SamplingResponse` for `req`."""
    is_req = isinstance(req, base_sampler_lib.SamplingRequest)
    request_kwargs = {
        **base_kwargs,
        **(req.metadata if is_req and req.metadata else {}),
    }
    req_id_str = str(req.request_id) if is_req and req.request_id else "default"
    if request_kwargs.get("raise_in_sample"):
      raise RuntimeError(str(request_kwargs["raise_in_sample"]))
    if req_id_str in self.fail_request_ids:
      raise RuntimeError(f"Simulated failure for request_id={req_id_str}")

    sp = req.sampling_params if is_req else None
    prompt_ids = self._resolve_prompt_ids(req)
    self.sample_calls += 1
    turn = self._turn_counters.pop(req_id_str, 0)
    min_turns = int(request_kwargs.get("min_turns", self.min_turns))
    is_final = turn >= min_turns or bool(request_kwargs.get("force_finish"))
    if not is_final:
      self._turn_counters[req_id_str] = turn + 1
      if len(self._turn_counters) > _MAX_TURN_COUNTER_ENTRIES:
        self._turn_counters.popitem(last=False)
    ans = request_kwargs.get("answer", f"result_for_{req_id_str}")
    txt = (
        f"FINAL_ANSWER: [{self.sampler_name}] {ans}"
        if is_final
        else f"TOOL_CALL: search(query='turn {turn} for {req_id_str}')"
    )
    tokens = self._build_turn_tokens(
        max_tokens=sp.max_tokens if sp is not None else None,
        is_final_turn=is_final,
    )
    routed_experts = self._maybe_build_routed(
        req_id_str=req_id_str,
        turn=turn,
        prompt_len=int(prompt_ids.size),
        gen_len=int(tokens.size),
        sp=sp,
    )
    return base_sampler_lib.SamplingResponse(
        request_id=req_id_str,
        text=txt,
        prompt_token_ids=prompt_ids,
        token_ids=tokens,
        logprobs=np.zeros_like(tokens, dtype=np.float32),
        routed_experts=routed_experts,
    )

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
    if delay > 0:
      await asyncio.sleep(delay)

    is_sequence = isinstance(sampling_requests, (list, tuple))
    requests = list(sampling_requests) if is_sequence else [sampling_requests]
    responses = [self._sample_single(req, kwargs) for req in requests]
    return responses if is_sequence else responses[0]

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

  async def pre_weight_sync(
      self,
      sync_request: (
          datatypes.WeightSyncRequest | datatypes.WeightSyncMetadata | None
      ) = None,
      **kwargs,
  ) -> Any:
    self.staging_policy_version = None
    if isinstance(sync_request, datatypes.WeightSyncRequest):
      return int(sync_request.policy_version)
    metadata = kwargs.pop("metadata", sync_request)
    if isinstance(metadata, datatypes.WeightSyncMetadata):
      return metadata.new_policy_version
    return None

  async def weight_sync(
      self,
      sync_request: (
          datatypes.WeightSyncRequest | datatypes.WeightSyncMetadata | None
      ) = None,
      **kwargs,
  ) -> Any:
    if isinstance(sync_request, datatypes.WeightSyncRequest):
      self.staging_policy_version = int(sync_request.policy_version)
      return self.staging_policy_version
    metadata = kwargs.pop("metadata", sync_request)
    if isinstance(metadata, datatypes.WeightSyncMetadata):
      self.current_policy_version = metadata.new_policy_version
    return self.current_policy_version


@dataclasses.dataclass(frozen=True)
class FakeMoeModelConfig:
  """Minimal MoE model config satisfying `resolve_dummy_routed_experts`."""

  num_layers: int
  num_experts_per_tok: int
  num_experts: int


class FakeTrainer(abstract_trainer.AbstractTrainer):
  """Pure CPU / zero XLA model compilation implementation of AbstractTrainer."""

  def __init__(
      self,
      config: Any = None,
      *,
      trainer_id: str = "trainer_0",
      policy_version: int = 0,
      step_count: int = 0,
      compile_error: Exception | None = None,
      fwd_bwd_sleep_s: float = 0.0,
      expected_routed_experts_ndim: int | None = None,
      routed_experts_shape_spec: tuple[int, int] | None = None,
      num_experts: int = 256,
      synthetic_fwd_only: bool = False,
  ) -> None:
    self.config = config
    # Feeds the `trainer._config` fallback in
    # `abstract_trainer.resolve_dummy_routed_experts`.
    self._config: FakeMoeModelConfig | None = None
    if routed_experts_shape_spec is not None:
      self._config = FakeMoeModelConfig(
          num_layers=int(routed_experts_shape_spec[0]),
          num_experts_per_tok=int(routed_experts_shape_spec[1]),
          num_experts=int(num_experts),
      )
    self.trainer_id = trainer_id
    self.policy_version = policy_version
    self.step_count = step_count
    self.compile_error = compile_error
    self._fwd_only_fn: Callable[..., Any] | None = None
    self.fwd_bwd_sleep_s = float(fwd_bwd_sleep_s)
    self.expected_routed_experts_ndim = expected_routed_experts_ndim
    self.synthetic_fwd_only = bool(synthetic_fwd_only)
    self.prepare_calls: int = 0
    self.release_calls: int = 0
    self.fwd_bwd_calls: int = 0
    self.fwd_only_calls: int = 0
    self.total_tokens: int = 0
    self.total_token_slots: int = 0
    self.total_routed_experts_bytes: int = 0
    self.last_batch_shape: tuple[int, int] | None = None
    self.last_routed_experts_shape: tuple[int, ...] | None = None
    self.compiled_routed_experts_shape: tuple[int, ...] | None = None

  def compile(self, dummy_data: datatypes.TrainerPayload | None = None) -> None:
    if self.compile_error is not None:
      raise self.compile_error
    if (
        isinstance(dummy_data, datatypes.RLTrainerPayload)
        and dummy_data.routed_experts is not None
    ):
      self.compiled_routed_experts_shape = tuple(
          int(d) for d in np.asarray(dummy_data.routed_experts).shape
      )

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "FakeTrainer":
    del loss_fn, has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "FakeTrainer":
    del gen_model_input_fn
    return self

  def _record_rl_tokens(self, payload: datatypes.RLTrainerPayload) -> None:
    if payload.completion_ids is None:
      return
    comp = np.asarray(payload.completion_ids)
    prompt_cols = 0
    if payload.prompt_ids is not None:
      prompt_arr = np.asarray(payload.prompt_ids)
      if prompt_arr.ndim == 2:
        prompt_cols = int(prompt_arr.shape[1])
    total_cols = prompt_cols + int(comp.shape[1])
    self.last_batch_shape = (int(comp.shape[0]), total_cols)
    self.total_token_slots += int(comp.shape[0]) * total_cols
    if payload.segment_ids is not None:
      self.total_tokens += int(np.count_nonzero(payload.segment_ids))
      return
    valid_prompt = (
        int(np.sum(np.asarray(payload.prompt_mask) > 0))
        if payload.prompt_mask is not None
        else int(comp.shape[0]) * prompt_cols
    )
    valid_comp = (
        int(np.sum(np.asarray(payload.completion_mask) > 0))
        if payload.completion_mask is not None
        else int(comp.size)
    )
    self.total_tokens += valid_prompt + valid_comp

  def _record_rl_routed(self, routed: np.ndarray | None) -> None:
    if self.expected_routed_experts_ndim is not None and routed is None:
      raise ValueError("FakeTrainer expected non-None payload.routed_experts.")
    if routed is None:
      return
    routed_arr = np.asarray(routed)
    if (
        self.expected_routed_experts_ndim is not None
        and routed_arr.ndim != self.expected_routed_experts_ndim
    ):
      raise ValueError(
          "Expected payload.routed_experts.ndim =="
          f" {self.expected_routed_experts_ndim}, got {routed_arr.ndim}"
          f" (shape={routed_arr.shape})."
      )
    self.last_routed_experts_shape = tuple(int(d) for d in routed_arr.shape)
    self.total_routed_experts_bytes += int(routed_arr.nbytes)

  def fwd_bwd(self, payload: datatypes.TrainerPayload) -> None:
    if isinstance(payload, datatypes.RLTrainerPayload):
      self._record_rl_tokens(payload)
      self._record_rl_routed(payload.routed_experts)
    self.fwd_bwd_calls += 1
    if self.fwd_bwd_sleep_s > 0:
      time.sleep(self.fwd_bwd_sleep_s)

  def update(self) -> int:
    self.step_count += 1
    self.policy_version += 1
    return self.step_count

  def eval_step(self, payload: datatypes.TrainerPayload) -> None:
    del payload

  def with_fwd_only_fn(self, fn: Callable[..., Any]) -> "FakeTrainer":
    self._fwd_only_fn = fn
    return self

  def fwd_only(self, payload: datatypes.TrainerPayload) -> Any:
    self.fwd_only_calls += 1
    if (
        self.synthetic_fwd_only
        and isinstance(payload, datatypes.LogprobsScoringPayload)
    ):
      self._record_rl_routed(payload.routed_experts)
      return np.zeros(
          np.asarray(payload.completion_ids).shape, dtype=np.float32
      )
    if self._fwd_only_fn is None:
      raise ValueError("FakeTrainer.fwd_only needs with_fwd_only_fn() first.")
    return self._fwd_only_fn(None, payload)

  def save_checkpoint(
      self,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> None:
    del metadata, step, force

  def restore_checkpoint(self, *, step: int | None = None) -> dict[str, Any]:
    del step
    return {}

  # Return type kept as `Any` for Pyrefly override compatibility with
  # `AbstractTrainer.get_metrics() -> datatypes.MetricsBuffer`.
  def get_metrics(self) -> Any:
    return {
        "loss": 0.25,
        "mock_trainer/fwd_bwd_calls": self.fwd_bwd_calls,
        "mock_trainer/routed_experts_bytes": self.total_routed_experts_bytes,
    }

  def prepare_weight_sync(self, *, sync_request: Any = None) -> Any:
    del sync_request
    self.prepare_calls += 1
    return [
        weight_sync.WorkUnitMetadata(
            unit=weight_sync.WorkUnitId(
                job_name=self.trainer_id, job_replica_id="0"
            ),
            shards=("127.0.0.1:20000",),
            control_plane_rpc_address="127.0.0.1:20500",
            global_shape=(4,),
            mesh_shape=(1,),
            layout=(0,),
            item_size=4,
        )
    ]

  def release_weight_sync(
      self,
      *,
      sync_request: datatypes.WeightSyncRequest | None = None,
  ) -> None:
    del sync_request
    self.release_calls += 1

  def close(self) -> None:
    pass


class MockAgent(base_agent.ConversationAgentBase):
  """Mock Agent tracking turn history and memory from environment observations."""

  def __init__(self, system_prompt: str = "Mock Agent", **kwargs: Any):
    del kwargs
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
  actor handles (Phase 1: Rollout generation via TaskSession) and Trainer
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
      num_generations: int = 1,
  ) -> List[TrajectoryOrError]:
    """Dispatches requests via TaskSession and collects out-of-order results."""
    if not self.actor_handles:
      raise ValueError(
          "No valid ActorHandles available in MockGlobalOrchestrator."
      )
    if num_generations < 1:
      raise ValueError(
          f"num_generations must be at least 1, got {num_generations}"
      )

    fanned_out_requests = []
    for req in requests:
      if num_generations > 1:
        for group_index in range(num_generations):
          fanned_out_requests.append(
              dataclasses.replace(req, group_index=group_index)
          )
      else:
        fanned_out_requests.append(req)
    trajectories: List[TrajectoryOrError] = []
    async with task_session.TaskSession(self.actor_handles) as session:
      for req in fanned_out_requests:
        await session.submit(req.request_id or "req", "generate", req)
      async for traj, exc in session.as_completed():
        if exc is not None:
          raise exc
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
