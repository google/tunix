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

"""Remote ActorHandle-backed Worker adapters for Orchestrator V2."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


def _role_names(roles: Sequence[datatypes.Role | str]) -> frozenset[str]:
  return frozenset(
      role.value if isinstance(role, datatypes.Role) else role
      for role in roles
  )


def _left_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_id: int,
) -> np.ndarray:
  arr = np.asarray(values, dtype=np.int32).reshape(-1)[-length:]
  out = np.full(length, pad_id, dtype=np.int32)
  if arr.size:
    out[-arr.size:] = arr
  return out


def _right_pad(
    values: np.ndarray,
    length: int,
    *,
    pad_id: int,
) -> np.ndarray:
  arr = np.asarray(values, dtype=np.int32).reshape(-1)[:length]
  out = np.full(length, pad_id, dtype=np.int32)
  if arr.size:
    out[:arr.size] = arr
  return out


class RemoteActorWorker(abstract_worker.Worker):
  """Registers a remote ActorHandle as an Orchestrator V2 Worker.

  This adapter keeps ClusterOrchestrator's registry and lifecycle APIs uniform
  for local workers and gRPC-backed workers. It also bridges the current frozen
  InferenceWorker wire method (`compute_logps(LogprobsRequest)`) into the v2
  engine's role-oriented `per_token_logps(items=...)` call.
  """

  def __init__(
      self,
      *,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      pad_id: int | None = None,
      eos_id: int | None = None,
      max_prompt_length: int | None = None,
      max_response_length: int | None = None,
      temperature: float = 1.0,
      resources: dict[str, Any] | None = None,
  ):
    self._worker_id = worker_id
    self._roles = _role_names(roles)
    self._handle = handle
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._max_prompt_length = max_prompt_length
    self._max_response_length = max_response_length
    self._temperature = temperature
    self._resources = dict(resources or {})
    self._state = datatypes.WorkerState.PENDING

  def _submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    logging.info("[%s] %s", self._worker_id, method_name)
    return self._handle.submit(method_name, *args, **kwargs)

  def initialize(self) -> datatypes.Response:
    self.state = datatypes.WorkerState.INITIALIZING
    response = self._submit("initialize")
    self.state = datatypes.WorkerState.READY
    return response

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    if self.state == datatypes.WorkerState.PENDING:
      self.initialize()
    self.state = datatypes.WorkerState.COMPILING
    response = self._submit("compile", dummy_data)
    self.state = datatypes.WorkerState.READY
    return response

  def start(self) -> datatypes.Response:
    if self.state == datatypes.WorkerState.PENDING:
      self.initialize()
    return self._submit("start")

  def stop(self) -> datatypes.Response:
    response = self._submit("stop")
    self.state = datatypes.WorkerState.STOPPED
    return response

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=self._roles,
        resources={"remote": True, **self._resources},
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
    if (
        method_name == "per_token_logps"
        and datatypes.Role.REFERENCE.value in self._roles
    ):
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
    if isinstance(metadata, datatypes.WeightSyncMetadata):
      return metadata
    weights = await self._handle.asubmit("get_lora_weights")
    policy_version = getattr(metadata, "new_policy_version", None)
    if policy_version is None and isinstance(metadata, datatypes.Response):
      policy_version = int(metadata.metadata.get("policy_version", 0)) + 1
    return SimpleNamespace(
        weights=weights,
        metadata=metadata,
        new_policy_version=int(policy_version or 1),
    )

  def _require_logps_config(self) -> tuple[int, int, int, int]:
    pad_id = self._pad_id
    eos_id = self._eos_id
    max_prompt_length = self._max_prompt_length
    max_response_length = self._max_response_length

    missing = []
    if pad_id is None:
      missing.append("pad_id")
    if eos_id is None:
      missing.append("eos_id")
    if max_prompt_length is None:
      missing.append("max_prompt_length")
    if max_response_length is None:
      missing.append("max_response_length")

    if missing:
      raise ValueError(
          "Remote reference logps require configuration for: "
          + ", ".join(missing)
      )
    assert pad_id is not None
    assert eos_id is not None
    assert max_prompt_length is not None
    assert max_response_length is not None
    return (
        pad_id,
        eos_id,
        max_prompt_length,
        max_response_length,
    )

  async def _remote_reference_logps(
      self,
      items: Sequence[datatypes.TrajectoryItem],
      **kwargs: Any,
  ) -> np.ndarray:
    pad_id, _, max_prompt_length, max_response_length = (
        self._require_logps_config()
    )
    prompt_rows = []
    completion_rows = []
    for item in items:
      prompt_rows.append(
          _left_pad(
              item.prompt_tokens
              if item.prompt_tokens is not None
              else np.zeros(0),
              max_prompt_length,
              pad_id=pad_id,
          )
      )
      completion_rows.append(
          _right_pad(
              item.completion_tokens
              if item.completion_tokens is not None
              else np.zeros(0),
              max_response_length,
              pad_id=pad_id,
          )
      )

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
