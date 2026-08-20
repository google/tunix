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

"""Top-level RolloutWorker abstractions (Service vs Client Driver)."""

import dataclasses
from typing import Any, AsyncIterator, Callable, List, Optional, Sequence, Union
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import abstract_worker
from tunix.rl.rollout import base_rollout


@dataclasses.dataclass
class RolloutConfig(base_rollout.RolloutConfig):
  """Rollout configuration extending base RolloutConfig with sampler choice and registry options.

  Attributes:
    sampler_type: Type of sampler adapter to construct ("vanilla",
      "legacy_vllm", "vllm").
    env_name: Registered name of environment class in ENV_REGISTRY.
    agent_name: Registered name of agent class in AGENT_REGISTRY.
    env_config: Configuration dictionary passed to environment constructor.
    agent_config: Configuration dictionary passed to agent constructor.
  """

  sampler_type: str = "vanilla"
  env_name: str = ""
  agent_name: str = ""
  env_config: dict[str, Any] = dataclasses.field(default_factory=dict)
  agent_config: dict[str, Any] = dataclasses.field(default_factory=dict)


TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]

WorkerState = datatypes.WorkerState


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops
  locally on its remote CPU host.
  """

  def __init__(
      self,
      worker_id: str,
      config: Optional[RolloutConfig] = None,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
  ):
    super().__init__()
    self.worker_id = worker_id
    self.config = config
    self._policy_version = 0
    self._state = datatypes.WorkerState.PENDING
    self._sync_round = {"req_id": None, "uuid": 0, "phase": "idle"}
    if tokenizer is None or chat_parser is None:
      raise ValueError(
          "RolloutWorker requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )
    self.manager = manager_lib.RolloutManager(
        config=config,
        sampler=sampler,
        env_pool=env_pool,
        agent_factory=agent_factory,
        max_concurrency=max_concurrency,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )

  @property
  def sampler(self) -> sampler_lib.Sampler:
    return self.manager.sampler

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset({"rollout"}),
        resources={
            "sampler": type(self.sampler).__name__,
            "policy_version": self._policy_version,
        },
    )

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    self.sampler.initialize()
    try:
      return datatypes.Response(
          metadata={
              "worker_id": self.worker_id,
              "state": self.state.value,
              "policy_version": self._policy_version,
          }
      )
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    if self.state == WorkerState.PENDING:
      self.initialize()
    return datatypes.Response(
        metadata={
            "worker_id": self.worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version,
        }
    )

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    self.manager.cancel_all()
    return datatypes.Response()

  def pause(self) -> datatypes.Response:
    self.manager.pause_all()
    return datatypes.Response()

  def resume(self) -> datatypes.Response:
    self.manager.resume_all()
    return datatypes.Response()

  def _infer_shapes(self) -> Any:
    return None

  def _compile_with_shapes(self, abstract_state: Any) -> None:
    pass

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version,
        inflight=len(self.manager._active_tasks),  # pylint: disable=protected-access
        queue_depth=self.manager._completed_queue.qsize(),  # pylint: disable=protected-access
    )

  def _left_pad_prompt_token_ids(
      self, prompt_token_ids: Sequence[np.ndarray]
  ) -> np.ndarray:
    pad_id = getattr(self.manager.tokenizer, "pad_token_id", None)
    if pad_id is None:
      pad_id = getattr(self.manager.tokenizer, "eos_token_id", 0) or 0
    configured_len = (
        getattr(self.config, "max_prompt_length", 0) if self.config else 0
    )
    max_len = max([1, configured_len] + [len(ids) for ids in prompt_token_ids])
    padded = np.full((len(prompt_token_ids), max_len), pad_id, dtype=np.int32)
    for i, ids in enumerate(prompt_token_ids):
      if ids.size:
        padded[i, -min(ids.size, max_len) :] = ids[-max_len:]
    return padded

  def _as_sampling_response_list(
      self, responses: Any
  ) -> list[sampler_lib.SamplingResponse]:
    if isinstance(responses, (list, tuple)):
      return list(responses)
    return [responses]

  async def sample_prompts(
      self,
      prompts: str | Sequence[str],
      *,
      max_generation_steps: int | None = None,
      temperature: float | None = None,
      top_p: float | None = None,
      top_k: int | None = None,
      seed: int | None = None,
      return_logprobs: bool = True,
  ) -> base_rollout.RolloutOutput:
    """Direct single-turn prompt sampling path using the worker's Sampler."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompt_list:
      return base_rollout.RolloutOutput(
          text=[],
          logits=None,
          tokens=[],
          left_padded_prompt_tokens=np.zeros((0, 1), dtype=np.int32),
          logprobs=[] if return_logprobs else None,
      )

    config = self.config or base_rollout.RolloutConfig()
    sampling_params = sampler_lib.SamplingParams(
        max_tokens=(
            max_generation_steps
            if max_generation_steps is not None
            else config.max_tokens_to_generate
        ),
        temperature=temperature if temperature is not None else config.temperature,
        top_p=top_p if top_p is not None else config.top_p,
        top_k=top_k if top_k is not None else config.top_k,
        seed=seed if seed is not None else config.seed,  # pyrefly: ignore[bad-argument-type]
        return_logprobs=return_logprobs,
    )
    requests = [
        sampler_lib.SamplingRequest(
            request_id=f"{self.worker_id}_sample_{i}",
            prompt=prompt,
            sampling_params=sampling_params,
        )
        for i, prompt in enumerate(prompt_list)
    ]
    responses = self._as_sampling_response_list(
        await self.sampler.sample(requests)
    )
    if len(responses) != len(prompt_list):
      raise RuntimeError(
          f"Sampler returned {len(responses)} responses for"
          f" {len(prompt_list)} prompts."
      )
    prompt_token_ids = [
        np.asarray(response.prompt_token_ids, dtype=np.int32).reshape(-1)
        for response in responses
    ]

    logprobs: list[np.ndarray] | None = None
    if return_logprobs:
      logprobs = []
      for response in responses:
        assert response.logprobs is not None
        logprobs.append(response.logprobs)

    return base_rollout.RolloutOutput(
        text=[response.text for response in responses],
        logits=None,
        tokens=[response.token_ids for response in responses],
        left_padded_prompt_tokens=self._left_pad_prompt_token_ids(
            prompt_token_ids
        ),
        logprobs=logprobs,
    )

  def _to_rollout_response(
      self,
      item: Any,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
      policy_version: int = 0,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, datatypes.RolloutResponse):
      return item
    if isinstance(item, trajectory_lib.TrajectoryError):
      return datatypes.RolloutResponse(
          request_id=request_id
          or getattr(item, "trajectory_id", "")
          or getattr(item, "prompt_id", ""),
          status="ERROR",
          error=item.error_message,  # pyrefly: ignore[bad-argument-type]
          prompt_tokens=(
              prompt_tokens
              if prompt_tokens is not None
              else np.zeros(0, dtype=np.int32)
          ),
          policy_version=policy_version,
      )
    if isinstance(item, trajectory_lib.Trajectory):
      req_id = request_id or getattr(item, "trajectory_id", "default")
      extra = getattr(item, "extra", None)
      extra = extra if isinstance(extra, dict) else {}
      if prompt_tokens is None:
        prompt_tokens = np.asarray(
            extra.get("prompt_tokens", np.zeros(0, dtype=np.int32)),
            dtype=np.int32,
        )
      response = datatypes.RolloutResponse.from_trajectory(
          request_id=req_id,
          traj=item,  # pyrefly: ignore[bad-argument-type]
          prompt_tokens=prompt_tokens,
          policy_version=policy_version,
      )
      response.prompt_id = str(extra.get("prompt_id", response.prompt_id))
      response.env_reward = float(extra.get("reward", response.env_reward))
      response.metadata.update(
          {k: v for k, v in extra.items() if k != "prompt_tokens"}
      )
      return response
    return item

  def _sampling_to_rollout_response(
      self,
      request: datatypes.RolloutRequest,
      text: str,
      prompt_tokens: Any,
      token_ids: Any,
      logprobs: Any | None,
  ) -> datatypes.RolloutResponse:
    """Builds the v2 rollout DTO for the direct single-turn sampler path."""
    completion_tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    completion_logps = (
        np.asarray(logprobs, dtype=np.float32).reshape(-1)
        if logprobs is not None
        else None
    )
    if (
        completion_logps is not None
        and completion_logps.shape != completion_tokens.shape
    ):
      completion_logps = None
    prompt_token_arr = np.asarray(prompt_tokens, dtype=np.int32).reshape(-1)
    if prompt_token_arr.size == 0:
      raise RuntimeError(
          "Sampler response is missing prompt_token_ids for "
          f"{request.request_id or request.traj_id}."
      )
    metadata = dict(request.metadata or {})
    metadata.setdefault("text", text)
    return datatypes.RolloutResponse(
        request_id=request.request_id or request.traj_id,
        prompt_id=request.prompt_id,
        status="COMPLETED",
        prompt_tokens=prompt_token_arr,
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=completion_tokens,
                loss_mask=np.ones(completion_tokens.shape, dtype=np.float32),
                logps=completion_logps,
            )
        ],
        env_reward=0.0,
        policy_version=self._policy_version,
        metadata=metadata,
    )

  async def _generate_rollout_requests_direct(
      self,
      requests: Sequence[datatypes.RolloutRequest],
      **generation_kwargs,
  ) -> list[datatypes.RolloutResponse]:
    """Runs RolloutRequest batches through the direct string sampler."""
    config = self.config or base_rollout.RolloutConfig()
    sampling_requests = []
    for req in requests:
      sample_kwargs = dict(req.generation_kwargs)
      sample_kwargs.update(generation_kwargs)
      sampling_requests.append(
          sampler_lib.SamplingRequest(
              request_id=req.request_id or req.traj_id,
              prompt=req.prompt,
              metadata=sample_kwargs,
              sampling_params=sampler_lib.SamplingParams(
                  max_tokens=sample_kwargs.get(
                      "max_generation_steps", config.max_tokens_to_generate
                  ),
                  temperature=sample_kwargs.get("temperature", config.temperature),
                  top_p=sample_kwargs.get("top_p", config.top_p),
                  top_k=sample_kwargs.get("top_k", config.top_k),
                  seed=sample_kwargs.get("seed", config.seed),
                  return_logprobs=sample_kwargs.get("return_logprobs", True),
              ),
          )
      )
    responses = self._as_sampling_response_list(
        await self.sampler.sample(sampling_requests)
    )
    if len(responses) != len(requests):
      raise RuntimeError(
          f"Sampler returned {len(responses)} responses for"
          f" {len(requests)} rollout requests."
      )
    return [
        self._sampling_to_rollout_response(
            request=req,
            text=responses[i].text,
            prompt_tokens=responses[i].prompt_token_ids,
            token_ids=responses[i].token_ids,
            logprobs=responses[i].logprobs,
        )
        for i, req in enumerate(requests)
    ]

  async def generate(
      self,
      requests: (
          datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest] | Any
      ) = None,
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]] = None,
      prompts: Any = None,
      **generation_kwargs,
  ) -> datatypes.RolloutResponse | List[datatypes.RolloutResponse] | Any:
    """Coroutine method for single or batched generate requests."""
    if requests is None:
      requests = prompts
    if requests is None:
      raise ValueError("generate requires `requests` or v2 `prompts`.")
    if isinstance(requests, str) or (
        isinstance(requests, (list, tuple))
        and all(isinstance(req, str) for req in requests)
    ):
      return await self.sample_prompts(requests, **generation_kwargs)  # pyrefly: ignore[bad-argument-type]

    cb = None
    if on_complete is not None:
      cb = lambda item: on_complete(self._to_rollout_response(item))
    res = await self.manager.generate(requests, on_complete=cb)
    if isinstance(res, (list, tuple)):
      return [self._to_rollout_response(r) for r in res]
    return self._to_rollout_response(res)

  async def pop_next_completed(self) -> datatypes.RolloutResponse | Any:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    res = await self.manager.pop_next_completed()
    return self._to_rollout_response(res)

  async def as_completed_stream(
      self,
  ) -> AsyncIterator[datatypes.RolloutResponse | Any]:
    """Async stream yielding completed trajectories or errors strictly out-of-order."""
    async for res in self.manager.as_completed_stream():
      yield self._to_rollout_response(res)

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Quiesces the worker; it stays SYNCING until post or abort."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.SYNCING
    self._record_round(sync_request, "idle")
    result = await self.manager.pre_weight_sync(sync_request, **kwargs)
    self._record_round(sync_request, "prepared")
    return result

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Materializes the received weights; the worker stays SYNCING."""
    if self.state == WorkerState.PENDING:
      self.initialize()
    self.state = WorkerState.SYNCING
    metadata = kwargs.pop("metadata", None)
    request = sync_request if sync_request is not None else metadata
    result = await self.manager.weight_sync(request, **kwargs)
    self._policy_version += 1
    self._record_round(request, "h2d_done")
    return result

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Publishes the new weights and resumes serving."""
    result = await self.manager.post_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "committed")
    return result

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the destination-side transport via the manager."""
    return await self.manager.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sampler's transport metadata via the manager."""
    return await self.manager.get_weight_sync_metadata(**kwargs)

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Discards the round and resumes serving the previous weights."""
    self.manager.resume_all()
    self.manager.reopen_admission()
    self.state = WorkerState.READY
    self._record_round(sync_request, "aborted")
    return None

  async def get_weight_sync_status(self, **kwargs) -> Any:
    """Returns this worker's view of the current weight sync round."""
    return dict(self._sync_round, policy_version=self._policy_version)

  def _record_round(self, sync_request: Any, phase: str) -> None:
    extra = getattr(sync_request, "extra_config", None) or {}
    if extra.get("req_id") is not None:
      self._sync_round["req_id"] = extra.get("req_id")
      self._sync_round["uuid"] = extra.get("uuid", 0)
    self._sync_round["phase"] = phase
