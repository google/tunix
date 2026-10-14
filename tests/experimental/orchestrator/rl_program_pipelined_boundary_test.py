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

"""Tests for StandardRLProgram's pipelined step boundary.

`_train_loop` is one loop for both step boundaries. Every scenario runs
through it in three modes -- flags off, `pipeline_step_boundary`, and that
plus `stage_next_microbatch` -- and through a frozen, byte-identical copy of
the loop as it was before the pipelined boundary (`_BaseLoopProgram`:
atwigg/mlperf as of google/tunix#2719's fused final update and deferred
get_metrics), against the same trainer double. Flags off must be that loop
exactly: the same trainer calls, the same program events (feeds, flushes,
step callbacks) in the same order, and the same metrics. The pipelined modes
must be indistinguishable from it to the trainer: the same calls with the
same inputs in the same order, each micro-batch computed on the same weights,
the same checkpoints, and the same logged metric values. Separate tests show
that the pipelined modes do change *when* the orchestrator does its work, and
that every check here fails on a deliberately wrong program.

Two trainer doubles cover the two regimes of the fused last call. `_Trainer`
cannot confirm its update, like MaxText at tunix's pin today: the call
returns at the update's dispatch and get_metrics stays ahead of the
checkpoint save. `_TokenTrainer` can (`last_update_token`, maxtext#5571): the
call returns once the update landed and the base loop defers get_metrics past
the weight sync and commit. There a planned stage makes the last call return
at the dispatch and confirms the update itself, so a staged boundary is the
base loop's plus the one stage call; a planned stage that finds nothing
packed in time falls back to get_metrics ahead of the save, which the
relaxed check (`_assert_equivalent_but_metrics_fetch`) allows and nothing
else.

The trainer double runs behind a real `TrainerWorker`, on its own thread and
event loop, like the remote gRPC server: methods run one at a time and block
that loop while they run, and a request's transfer time is spent on that loop
before the method runs. Its device keeps the MaxText engine's rules: at most
two computations in flight, run in dispatch order; `get_metrics` waits for the
update and returns only the newest step's metrics buffer; `save_checkpoint`
and `prepare_weight_sync` wait for every computation in flight.
"""

import asyncio
import collections
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import hashlib
import inspect
import os
import random
import subprocess
import threading
import time
import types
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tunix
from tunix.experimental.common import datatypes
from tunix.experimental.common import logging_utils
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker as trainer_worker_lib


NUM_GENERATIONS = 2
ROW_TOKENS = 64

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _payload(seed: int, *, logps: bool) -> datatypes.RLTrainerPayload:
  """An unbatched trajectory whose content is unique to `seed`."""
  rng = np.random.default_rng(seed)
  p_len = int(rng.integers(3, 9))
  c_len = int(rng.integers(6, 22))
  completion_mask = (rng.random(c_len) > 0.2).astype(np.float32)
  completion_mask[0] = 1.0
  return datatypes.RLTrainerPayload(
      prompt_ids=rng.integers(1, 1000, p_len).astype(np.int32),
      prompt_mask=np.ones(p_len, dtype=np.float32),
      completion_ids=rng.integers(1, 1000, c_len).astype(np.int32),
      completion_mask=completion_mask,
      advantages=np.full(c_len, rng.normal(), dtype=np.float32),
      old_per_token_logps=(
          -rng.random(c_len).astype(np.float32) if logps else None
      ),
      routed_experts=rng.integers(0, 64, (p_len + c_len, 2, 2)).astype(
          np.int16
      ),
  )


def _item(
    prompt_idx: int,
    gen: int,
    *,
    full_batch_size: int,
    logps: bool,
) -> datatypes.TrajectoryItem:
  """A scored trajectory as `critique_stage` puts it on `scored_q`."""
  seed = 1000 * prompt_idx + gen
  payload = _payload(seed, logps=logps)
  item = datatypes.TrajectoryItem(
      prompt_id=f"prompt_{prompt_idx}",
      group_index=gen,
      start_step=0,
      traj={
          "trajectory_reward": float((seed * 7919) % 5) / 4.0,
          "status": datatypes.TrajectoryStatus.SUCCEEDED,
          "steps": [0] * (1 + seed % 3),
      },
      is_valid=True,
      prompt_tokens=np.asarray(payload.prompt_ids),
      completion_tokens=np.asarray(payload.completion_ids),
      policy_version=0,
      metadata={
          "prompt_idx": prompt_idx,
          "batch_idx": prompt_idx // full_batch_size,
          "intra_batch_idx": prompt_idx % full_batch_size,
      },
  )
  item.payload = payload  # pyrefly: ignore[missing-attribute]
  return item


def _groups(
    num_prompts: int, *, full_batch_size: int, logps: bool
) -> list[list[datatypes.TrajectoryItem]]:
  return [
      [
          _item(p, g, full_batch_size=full_batch_size, logps=logps)
          for g in range(NUM_GENERATIONS)
      ]
      for p in range(num_prompts)
  ]


def _fingerprint(payload: Any) -> str:
  """Content hash of a micro-batch payload as the trainer receives it."""
  h = hashlib.sha256()
  for field in dataclasses.fields(payload):
    value = getattr(payload, field.name)
    if field.name == "metadata":
      value = (value or {}).get("trajectory_ids")
      h.update(repr(value).encode())
      continue
    if value is None:
      h.update(f"{field.name}=None;".encode())
      continue
    arr = np.ascontiguousarray(np.asarray(value))
    h.update(f"{field.name}:{arr.dtype}:{arr.shape};".encode())
    h.update(arr.tobytes())
  return h.hexdigest()[:16]


def _loss(fingerprint: str, weights_version: int) -> float:
  """A loss that depends on both the data and the weights it is computed on."""
  digest = hashlib.sha256(f"{fingerprint}@{weights_version}".encode())
  return int(digest.hexdigest()[:8], 16) / 2.0**32


# ---------------------------------------------------------------------------
# Trainer double
# ---------------------------------------------------------------------------


class _HostStagedSync:
  """A synchronizer whose transfer reads a host copy (background sync is ok)."""

  staged_on_host = True

  def release_buffers(self) -> int:
    return 0


@dataclasses.dataclass
class _Costs:
  """Seconds of trainer work; all zero for the equivalence tests."""

  device_s: float = 0.0  # one micro-batch's forward/backward on the device
  update_s: float = 0.0  # the optimizer update on the device
  stage_s: float = 0.0  # prepare_weight_sync after the device drains
  host_s: float = 0.0  # moving one micro-batch's data to the trainer
  metrics_s: float = 0.0  # get_metrics once the update has landed


class _Trainer:
  """MaxText-engine-shaped trainer double (see the module docstring)."""

  def __init__(self, log: list[Any], costs: _Costs, max_inflight: int = 2):
    self.log = log
    self.costs = costs
    self.max_inflight = max_inflight
    self.policy_version = 0
    self.checkpoint_dir = "/ckpt"
    self._weight_sync = _HostStagedSync()
    self.weights_version = 0
    self._accumulated: list[str] = []
    self._buffers: list[tuple[int, list[Any]]] = []
    self._inflight: collections.deque[float] = collections.deque()
    self._device_free_at = 0.0
    self._update_lands_at: float | None = None
    # (kind, label, start, end) on the device, monotonic clock.
    self.device: list[tuple[str, Any, float, float]] = []
    # Wall-clock ms at which each update finished on the device.
    self.update_landed_ms: list[float] = []
    # What each `wait_for_device` waited, in seconds.
    self.device_waits: list[float] = []

  def _wait_until(self, t: float) -> None:
    delay = t - time.monotonic()
    if delay > 0:
      time.sleep(delay)

  def _dispatch(self, kind: str, label: Any, seconds: float) -> None:
    # InflightThrottler.wait_for_next, then an in-order device queue.
    if len(self._inflight) >= self.max_inflight:
      self._wait_until(self._inflight.popleft())
    start = max(time.monotonic(), self._device_free_at)
    end = start + seconds
    self._device_free_at = end
    self._inflight.append(end)
    self.device.append((kind, label, start, end))

  def _wait_for_all(self) -> None:
    self._wait_until(self._device_free_at)
    self._inflight.clear()

  def _step_buffer(self) -> list[Any]:
    step = self.weights_version
    if not self._buffers or self._buffers[-1][0] != step:
      self._buffers.append((step, []))
    return self._buffers[-1][1]

  def prepare_fwd_bwd_input(self, payload: Any) -> Any:
    """What `TrainerWorker.stage_fwd_bwd` holds; here the payload itself."""
    self.log.append(("stage", _fingerprint(payload), self.weights_version))
    return payload

  def wait_for_device(self) -> float:
    """MaxText's `wait_for_device`: waits for the device, leaves the queue."""
    start = time.monotonic()
    self._wait_until(self._device_free_at)
    waited = time.monotonic() - start
    self.device_waits.append(waited)
    return waited

  def fwd_bwd(self, payload: Any, **kwargs: Any) -> None:
    del kwargs
    fingerprint = _fingerprint(payload)
    self._dispatch("fwd_bwd", fingerprint, self.costs.device_s)
    self.log.append(("fwd_bwd", fingerprint, self.weights_version))
    self._accumulated.append(fingerprint)
    self._step_buffer().append(_loss(fingerprint, self.weights_version))

  def update(self, **kwargs: Any) -> int:
    del kwargs
    self._dispatch("update", self.weights_version + 1, self.costs.update_s)
    end = self._device_free_at
    self._update_lands_at = end
    self.update_landed_ms.append(
        (time.time() + (end - time.monotonic())) * 1000.0
    )
    norm = _loss("|".join(self._accumulated), self.weights_version)
    self._step_buffer().append(("grad_norm", norm))
    self.weights_version += 1
    self.log.append(("update", self.weights_version, tuple(self._accumulated)))
    self._accumulated = []
    return self.weights_version

  def get_metrics(self) -> exp_metrics.MetricsBuffer:
    if self._update_lands_at is not None:
      self._wait_until(self._update_lands_at)
    time.sleep(self.costs.metrics_s)
    buffers, self._buffers = self._buffers, []
    if not buffers:
      return exp_metrics.MetricsBuffer(id=-1)
    step, values = buffers[-1]
    # Like `MaxTextTrainingEngine.get_metrics`: the newest buffer only.
    self.log.append(("get_metrics", step, len(buffers)))
    losses = [v for v in values if not isinstance(v, tuple)]
    norms = [v[1] for v in values if isinstance(v, tuple)]
    scalars: dict[str, Any] = {}
    if losses:
      scalars["loss"] = np.asarray(losses, dtype=np.float64)
    if norms:
      scalars["grad_norm"] = np.asarray(norms, dtype=np.float64)
    return exp_metrics.MetricsBuffer(id=step, scalar_metrics=scalars)

  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    self._wait_for_all()
    self.log.append((
        "save",
        tuple(sorted((metadata or {}).items())),
        tuple(sorted(kwargs.items())),
        self.weights_version,
        tuple(self._accumulated),
        tuple(step for step, _ in self._buffers),
    ))

  def prepare_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    del sync_request, kwargs
    self._wait_for_all()
    time.sleep(self.costs.stage_s)
    self.log.append(
        ("prepare_weight_sync", self.weights_version, tuple(self._accumulated))
    )
    return [{"unit": "trainer"}]

  def release_weight_sync(self, **kwargs: Any) -> Any:
    del kwargs
    self.log.append(("release_weight_sync",))
    return True

  def close(self) -> None:
    pass


class _Worker(trainer_worker_lib.TrainerWorker):
  """TrainerWorker whose actor log-probs come from the trainer double.

  They depend on the weights they are computed on, so a micro-batch rescored
  (`use_rollout_logps`) on the wrong weights trains on different inputs.
  """

  def per_token_logps(self, items, **kwargs):
    del kwargs
    trainer = self._trainer
    completion = np.asarray(items.completion_tokens)
    version = trainer.weights_version
    trainer.log.append(("per_token_logps", version, completion.shape))
    logps = -((completion.astype(np.float32) * (version + 1)) % 97) / 97.0
    return datatypes.LogprobsResponse(
        request_id=items.request_id,
        per_token_logps=logps,
        model_version=version,
    )


class _OldTrainerWorker(_Worker):
  """A trainer image from before staging.

  No `stage_fwd_bwd` / `fwd_bwd_staged`, and an `info()` without the staging
  capabilities, as `TrainerWorker` was at the base commit.
  """

  stage_fwd_bwd = None
  fwd_bwd_staged = None

  def info(self) -> datatypes.WorkerInfo:
    info = super().info()
    resources = {
        k: v
        for k, v in info.resources.items()
        if k not in ("stage_fwd_bwd", "reports_device_wait")
    }
    return dataclasses.replace(info, resources=resources)


class _NoWaitTrainer(_Trainer):
  """A trainer that can hold a staged micro-batch but cannot say whether the
  transfer hid behind its device (no `wait_for_device`)."""

  wait_for_device = None


class _UpdateToken:
  """An output of one update, as MaxText's `last_update_token` (its gradient
  norm): `jax.block_until_ready` on it returns once that update has landed."""

  def __init__(self, trainer: "_TokenTrainer", lands_at: float, version: int):
    self._trainer = trainer
    self._lands_at = lands_at
    self._version = version

  def block_until_ready(self) -> "_UpdateToken":
    self._trainer._wait_until(self._lands_at)
    self._trainer.log.append(("token_wait", self._version))
    return self


class _TokenTrainer(_Trainer):
  """A trainer that can confirm its update has landed (`last_update_token`).

  So a fused last call waits for the update and says so (`update_ready`), and
  the base loop defers get_metrics past the weight sync and commit
  (google/tunix#2719). The double without it is the trainer of today's MaxText
  pin, where the fused call returns at the update's dispatch.
  """

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.last_update_token: _UpdateToken | None = None

  def update(self, **kwargs: Any) -> int:
    version = super().update(**kwargs)
    assert self._update_lands_at is not None
    self.last_update_token = _UpdateToken(self, self._update_lands_at, version)
    return version


class _Reference:
  """Reference policy: static log-probs from the micro-batch content."""

  def per_token_logps(self, items, **kwargs):
    del kwargs
    completion = np.asarray(items.completion_ids)
    return datatypes.LogprobsResponse(
        request_id="ref",
        per_token_logps=-(completion.astype(np.float32) % 13) / 13.0,
        model_version=0,
    )


class _RemoteHandle(remote_execution.ActorHandle):
  """Serves a worker on its own thread and event loop, like a remote server."""

  def __init__(
      self,
      worker: Any,
      transfer_s: Callable[[str, dict[str, Any]], float] = lambda m, k: 0.0,
  ):
    self._server = remote_execution.InProcessRemoteExecutionServer(worker)
    self._transfer_s = transfer_s
    self._loop = asyncio.new_event_loop()
    self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
    self._thread.start()

  async def _execute(self, request: remote_execution.ExecutionRequest) -> Any:
    delay = self._transfer_s(request.method_name, request.kwargs)
    if delay:
      # On the server loop: a method blocking it also holds this up.
      await asyncio.sleep(delay)
    return await self._server.execute_request(request)

  async def asubmit(self, method_name: str | None = None, *args, **kwargs):
    request = remote_execution.ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    future = asyncio.run_coroutine_threadsafe(
        self._execute(request), self._loop
    )
    return (await asyncio.wrap_future(future)).unwrap()

  def submit(self, method_name: str | None = None, *args, **kwargs):
    request = remote_execution.ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    future = asyncio.run_coroutine_threadsafe(
        self._execute(request), self._loop
    )
    return future.result().unwrap()

  async def dispatch_task(self, request_id=None, method_name=None, *a, **k):
    raise NotImplementedError

  async def poll_responses(self, timeout_s=0.0):
    raise NotImplementedError

  def close(self) -> None:
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._thread.join(timeout=10)
    self._loop.close()


class _Coordinator:
  """Weight-sync coordinator double: destination phases, then the source."""

  def __init__(self, trainer: _RemoteHandle, dest_s: float, round_s: float):
    self._trainer = trainer
    self._dest_s = dest_s
    self._round_s = round_s
    self.in_flight = False
    self.poisoned = None

  def has_pending_destinations(self) -> bool:
    return False

  async def sync(self, policy_version, source_staged=None, only_pending=False):
    del only_pending
    extra = (
        {weight_sync.RELEASE_SOURCE_AFTER_STAGE: True}
        if source_staged is not None
        else {}
    )
    request = types.SimpleNamespace(extra_config=extra)
    await asyncio.sleep(self._dest_s)  # bind + metadata on the rollouts
    await self._trainer.asubmit("prepare_weight_sync", sync_request=request)
    if source_staged is not None:
      source_staged.set()
    await asyncio.sleep(self._round_s)
    await self._trainer.asubmit("release_weight_sync", sync_request=request)
    return types.SimpleNamespace(policy_version=policy_version)


class _MetricsRecorder:

  def __init__(self):
    self.records: list[tuple[str, str, float, int]] = []

  def log(self, prefix, name, value, mode, step):
    del mode
    self.records.append((prefix, name, float(np.asarray(value)), int(step)))

  def close(self):
    pass


class _CountingAssembler:
  """Delegates to a real assembler and records each feed and flush."""

  def __init__(self, inner: Any, log: list[Any], pack_s: float = 0.0):
    self._inner = inner
    self._log = log
    self._pack_s = pack_s
    self.num_generations = inner.num_generations
    self.mini_batch_size = inner.mini_batch_size
    self.pad_id = getattr(inner, "pad_id", 0)

  def __setattr__(self, name, value):
    if name in ("num_generations", "mini_batch_size") and "_inner" in vars(
        self
    ):
      setattr(self._inner, name, value)
    super().__setattr__(name, value)

  def feed(self, items):
    out = self._inner.feed(items)
    ids = tuple(
        sorted({getattr(p, "metadata", {}).get("traj_id", "") for p in items})
    )
    self._log.append(("feed", ids))
    if out and self._pack_s:
      time.sleep(self._pack_s * len(out))
    return out

  def flush(self):
    out = self._inner.flush()
    self._log.append(("flush", len(out)))
    if out and self._pack_s:
      time.sleep(self._pack_s * len(out))
    return out

  def reset(self, **kwargs):
    self._inner.reset(**kwargs)


# ---------------------------------------------------------------------------
# The reference: the loop as it was before the pipelined boundary
# ---------------------------------------------------------------------------

# Module globals the frozen loop below refers to, bound to rl_program's own.
_extract_reward = rl_program._extract_reward
_generation_metrics = rl_program._generation_metrics
RLStepResult = rl_program.RLStepResult
Mode = rl_program.Mode


def _release_host_memory() -> None:
  # Looked up at call time, so a test patching rl_program's is honoured.
  rl_program._release_host_memory()


# sha256 of the base's `StandardRLProgram._train_loop` source, which the copy
# below reproduces byte for byte: atwigg/mlperf from c52127238 (the merge of
# google/tunix#2719: the fused final update and the deferred get_metrics)
# through 21b62f2bb, where `_train_loop` is unchanged.
_BASE_TRAIN_LOOP_SHA256 = (
    "37032f00118fcb4ef7d9019ac6aec1066c9f9d4ad7d8a03d9000c576c514f1dd"
)


class _BaseLoopProgram(rl_program.StandardRLProgram):
  """The program with `_train_loop` exactly as before the pipelined boundary.

  The reference every run of the unified loop is compared with: its
  `_train_loop` is atwigg/mlperf's at c52127238 (google/tunix#2719) through
  21b62f2bb, verbatim (`BaseLoopOracleTest` pins its hash).
  """

  # A verbatim copy, pinned by its hash: never reformat it.
  # fmt: off
  async def _train_loop(self) -> None:
    assert self.engine is not None

    while self.max_steps is None or self._step < self.max_steps:
      current_step = self._step
      step_start_time = time.monotonic()
      consumed_policy_version = self.policy_version

      uncommitted_groups = []
      step_result = None
      trainer_metrics = None
      step_sampler_agreement: dict[str, tuple[Any, list[float]]] = {}
      step_rewards = []
      step_advantages = []
      num_microbatches = 0
      step_padding_stats: list[batch_assembly.PaddingStats] = []
      step_packing_time_sec = 0.0
      num_rollouts = 0
      all_step_items = []
      scored_items = []
      groups_consumed = 0
      checkpoint_saved = False
      final_minibatch_completed = False
      pending_metrics_fetch = False
      policy_training_time = 0.0
      exposed_generation_time = 0.0
      weight_sync_time = 0.0
      # The end-of-step get_metrics() deferred past the weight sync and commit,
      # kept out of policy_training_time because it no longer runs in training.
      metrics_fetch_time = 0.0

      current_batch_idx: int | None = None

      async def _await_pending_train() -> None:
        nonlocal step_result, policy_training_time
        task, self._pending_train = self._pending_train, None
        if task is not None:
          step_result, elapsed = await task
          policy_training_time += elapsed

      async def _maybe_save_checkpoint() -> None:
        nonlocal checkpoint_saved
        ckpt_ts_ms = time.time_ns() // 1_000_000
        self.last_step_timestamp_ms = ckpt_ts_ms
        optimizer_step = self.step + 1
        if (
            isinstance(step_result, dict)
            and step_result.get("train_step") is not None
        ):
          optimizer_step = int(step_result["train_step"])
        if (
            self.val_start_step is not None
            and optimizer_step < self.val_start_step
        ):
          checkpoint_saved = True
          return
        if isinstance(
            self.scored_q,
            trajectory_queue_manager.BatchOrderedQueueManager,
        ):
          next_batch_idx = (
              self.scored_q.next_batch_after(current_batch_idx)
              if current_batch_idx is not None
              else self.scored_q.next_batch_idx
          )
        else:
          next_batch_idx = self.step + 1
        save_kwargs = {}
        interval = self.checkpoint_optimizer_interval_steps
        if interval == 0 or (
            optimizer_step % interval != 0
            and (self.max_steps is None or self.step + 1 < self.max_steps)
        ):
          # Sent only when skipping, so an interval of 1 leaves the request,
          # and trainers that predate the option, untouched.
          save_kwargs["save_optimizer_state"] = False
        save_resp = await self.engine.save_checkpoint(
            role=datatypes.Role.ACTOR,
            metadata={
                "step": optimizer_step,
                "global_step": self.step + 1,
                "next_batch_idx": next_batch_idx,
                "policy_version": self.policy_version + 1,
                "num_rollouts": num_rollouts,
                "num_microbatches": num_microbatches,
            },
            **save_kwargs,
        )
        checkpoint_saved = True
        if self.on_checkpoint_saved is not None:
          # `TrainerWorker.save_checkpoint` always returns the saved path in
          # `Response.metadata["checkpoint_path"]`. The callback runs in a
          # worker thread (it may do blocking file/GCS I/O) and is awaited, so
          # calls never overlap and its exceptions still propagate.
          ckpt_info: dict[str, Any] = {
              "step": optimizer_step,
              "timestamp_ms": ckpt_ts_ms,
              "checkpoint_path": save_resp.metadata["checkpoint_path"],
          }
          await asyncio.to_thread(self.on_checkpoint_saved, ckpt_info)

      while groups_consumed < self.full_batch_size:
        _t_gen = time.monotonic()
        if isinstance(
            self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
        ):
          ordered = await self.scored_q.get_ordered_group(
              batch_idx=current_batch_idx
          )
          if ordered is None:
            scored_items = []
          else:
            current_batch_idx, scored_items = ordered
        else:
          scored_items = await self.scored_q.get_group_batch(num_groups=1)
        # TODO(sanbao): Track accurate generation time by matching rollout
        # requests/responses or reading from the rollout worker. Currently this
        # measures time spent waiting on the scored queue (which includes
        # rollout + reward calculation), as MLPerf does not strictly validate
        # per-component timing accuracy.
        exposed_generation_time += time.monotonic() - _t_gen
        if not scored_items:
          packing_start_time = time.perf_counter()
          assembled_batches = list(await self._assemble(self.assembler.flush))
          step_packing_time_sec += time.perf_counter() - packing_start_time
        else:
          if groups_consumed == 0 and self.on_step_begin:
            self.on_step_begin(current_step)

          groups_consumed += 1
          uncommitted_groups.append(scored_items)
          all_step_items.extend(scored_items)
          num_rollouts += len(scored_items)
          for item in scored_items:
            step_rewards.append(_extract_reward(item))
            payload = getattr(item, "payload", None)
            if (
                item.is_valid
                and payload is not None
                and payload.advantages is not None
                and np.asarray(payload.advantages).size > 0
            ):
              step_advantages.append(float(np.mean(payload.advantages)))

          payloads = []
          for item in scored_items:
            payload = getattr(item, "payload", None)
            if isinstance(payload, datatypes.RLTrainerPayload):
              payload = dataclasses.replace(
                  payload,
                  metadata={
                      **payload.metadata,
                      "traj_id": item.traj_id,
                  },
              )
            payloads.append(payload)
          packing_start_time = time.perf_counter()
          assembled_batches = list(
              await self._assemble(
                  self.assembler.feed,
                  payloads,  # pyrefly: ignore[bad-argument-type]
              )
          )
          step_packing_time_sec += time.perf_counter() - packing_start_time
          # `self.assembler` has now copied or packed `routed_experts`; release
          # the per-trajectory references on `payloads` and `scored_items`
          # (which remain in `all_step_items` until step-end metric logging) so
          # unbatched routing arrays do not coexist with packed microbatches and
          # weight sync. Lightweight 1D mask fields on `item.payload` are kept
          # for `_collect_and_log_step_metrics`.
          payloads.clear()
          for item in scored_items:
            item.routed_experts = None
            if item.metadata is not None:
              item.metadata.pop("routed_experts", None)
            if isinstance(item.traj, dict):
              item.traj.pop("routed_experts", None)
            item_payload = getattr(item, "payload", None)
            if (
                isinstance(item_payload, datatypes.RLTrainerPayload)
                and item_payload.routed_experts is not None
            ):
              item.payload = dataclasses.replace(
                  item_payload, routed_experts=None
              )

        while assembled_batches:
          mb = assembled_batches.pop(0)
          is_final_batch = mb.is_final_batch
          batch = mb.payload
          if getattr(self.algo, "requires_reference_kl", False):
            if not isinstance(batch, datatypes.RLTrainerPayload):
              raise TypeError(
                  "Reference KL requires an assembler that returns "
                  "datatypes.RLTrainerPayload microbatches; got "
                  f"{type(batch).__name__}."
              )
            # The reference model is static and on its own worker, so this
            # overlaps the in-flight actor train_step.
            ref_logps = await self.engine.per_token_logps(
                datatypes.Role.REFERENCE, items=batch
            )
            batch = batch_assembly.with_ref_per_token_logps(batch, ref_logps)
          algo_config = getattr(self.algo, "algo_config", None)
          if (
              isinstance(batch, datatypes.RLTrainerPayload)
              and batch.old_per_token_logps is not None
              and algo_config is not None
              and algo_config.use_rollout_logps
          ):
            await _await_pending_train()
            batch = await self._apply_sampler_trainer_agreement(
                batch, step_sampler_agreement
            )

          num_microbatches += 1
          if mb.padding_stats is not None:
            step_padding_stats.append(mb.padding_stats)
            logging.info(
                "Packed %d trajectories into microbatch: %s (padding_ratio=%.3f,"
                " row_imbalance=%.3f)",
                len(mb.trajectory_ids),
                logging_utils.summarize_list(list(mb.trajectory_ids)),
                mb.padding_stats.padding_ratio,
                mb.padding_stats.row_imbalance,
            )
          else:
            logging.info(
                "Packed %d trajectories into microbatch: %s",
                len(mb.trajectory_ids),
                logging_utils.summarize_list(list(mb.trajectory_ids)),
            )
          mb = None
          if not self._first_train_payload_logged:
            self._first_train_payload_logged = True
            logging.info(
                "First trainer micro-batch: %s",
                batch_assembly.describe_payload(batch),
            )
          await _await_pending_train()
          train_step = self._timed_train_step(
              batch, apply_optimizer=is_final_batch
          )
          batch = None
          if self.pipeline_train_microbatches and not is_final_batch:
            self._pending_train = asyncio.create_task(train_step)
            continue
          step_result, elapsed = await train_step
          policy_training_time += elapsed
          if is_final_batch:
            # Save only at a resumable full-batch boundary. An optimizer step
            # can occur earlier when a full batch contains multiple mini
            # batches, but the dataset resume cursor advances in full batches.
            full_batch_complete = (
                groups_consumed >= self.full_batch_size or not scored_items
            )
            # The checkpoint is stamped as the weight-update time and saved
            # next, so the update must have landed on device first. Unless the
            # trainer has said it did (`update_ready`), fetching the metrics is
            # what waits for it, so the fetch stays ahead of the save, counted
            # as training time, as with a separate update RPC.
            pending_metrics_fetch = full_batch_complete and bool(
                isinstance(step_result, dict)
                and step_result.get("update_ready")
            )
            if not pending_metrics_fetch:
              _t_metrics = time.monotonic()
              trainer_metrics = await self.engine.get_metrics(
                  role=datatypes.Role.ACTOR
              )
              policy_training_time += time.monotonic() - _t_metrics
            final_minibatch_completed = True
            if self.sync_weights and self.async_weight_sync:
              self._round_due = True
            # TODO(tunix-dev): Configurable checkpointing frequency. Today we
            # checkpoint at the same frequency as the weight update.
            # TODO(tunix-dev): For now any failures in save_checkpoint will
            # abort the entire program. Make it configurable on whether to fail
            # or continue.
            if full_batch_complete:
              await _maybe_save_checkpoint()

        if not scored_items:
          if not checkpoint_saved and final_minibatch_completed:
            await _maybe_save_checkpoint()
          break

      await _await_pending_train()
      if not all_step_items:
        logging.info(
            "Dataset exhausted at step %d before max_steps.", current_step
        )
        break

      if self.sync_weights and self.async_weight_sync:
        # Only the time the trainer is actually held: the previous round, if
        # it outlived this step, plus this round's source snapshot.
        _t_sync = time.monotonic()
        await self._start_background_sync()
        weight_sync_time = time.monotonic() - _t_sync
      elif self.sync_weights:
        _t_sync = time.monotonic()
        new_version = await self.engine.sync_weights(role=datatypes.Role.ACTOR)
        weight_sync_time = time.monotonic() - _t_sync
        self.policy_version = (
            new_version if new_version is not None else self.policy_version + 1
        )

      # Before `commit()`, which will eventually take ownership of the groups.
      generation_metrics = _generation_metrics(uncommitted_groups)
      logging.info(
          "[pipeline] COMMIT step=%d batch_idx=%s consumed_version=%d"
          " rollout_versions=%s groups=%s",
          current_step,
          current_batch_idx,
          consumed_policy_version,
          [getattr(g[0], "policy_version", 0) for g in uncommitted_groups if g],
          [
              (
                  getattr(g[0], "prompt_id", ""),
                  (getattr(g[0], "metadata", None) or {}).get("batch_idx"),
              )
              for g in uncommitted_groups
              if g
          ],
      )
      if isinstance(
          self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
      ):
        assert current_batch_idx is not None
        self.scored_q.commit_batch(current_batch_idx)
      else:
        self.scored_q.commit(current_step, groups=uncommitted_groups)
      # The commit may have opened the dispatch window. Yield so the
      # dispatcher refills the just-synced rollout workers now, rather than
      # after this step's bookkeeping and the next step's first packing.
      await asyncio.sleep(0)

      if pending_metrics_fetch:
        _t_metrics = time.monotonic()
        trainer_metrics = await self.engine.get_metrics(
            role=datatypes.Role.ACTOR
        )
        metrics_fetch_time += time.monotonic() - _t_metrics

      step_time_sec = time.monotonic() - step_start_time

      filtered_groups = await self._drain_filtered_groups()

      metrics_summary = self._collect_and_log_step_metrics(
          all_step_items=all_step_items,
          step_rewards=step_rewards,
          step_advantages=step_advantages,
          generation_metrics=generation_metrics,
          step_result=step_result,
          trainer_metrics=trainer_metrics,
          num_rollouts=num_rollouts,
          num_microbatches=num_microbatches,
          padding_stats=step_padding_stats,
          packing_time_sec=step_packing_time_sec,
          step_time_sec=step_time_sec,
          consumed_policy_version=consumed_policy_version,
          log_step=current_step,
          sampler_agreement=step_sampler_agreement,
          policy_training_time=policy_training_time,
          exposed_generation_time=exposed_generation_time,
          weight_sync_time=weight_sync_time,
          metrics_fetch_time=metrics_fetch_time,
          filtered_groups=filtered_groups,
      )
      self._log_consumed_trajectories(
          all_step_items,
          log_step=current_step,
          consumed_policy_version=consumed_policy_version,
      )
      filtered_groups.clear()
      all_step_items.clear()
      uncommitted_groups.clear()
      scored_items.clear()

      self.last_step_result = RLStepResult(
          step=current_step,
          policy_version=self.policy_version,
          num_rollouts=num_rollouts,
          num_microbatches=num_microbatches,
          reward_mean=metrics_summary["reward_mean"],
          reward_std=metrics_summary["reward_std"],
          advantage_mean=metrics_summary["advantage_mean"],
          advantage_std=metrics_summary["advantage_std"],
          train_result=step_result,
      )

      loss_val = metrics_summary["loss_val"]
      perplexity_val = metrics_summary["perplexity_val"]
      grad_norm_val = metrics_summary["grad_norm_val"]
      if self.mode == Mode.TRAIN:
        logging.info(
            "Train step %d - loss: %s - reward_mean: %.4f - advantage_mean:"
            " %.4f - grad_norm: %s - perplexity: %s - step_time: %.2fs",
            current_step,
            f"{loss_val:.4f}" if loss_val is not None else "N/A",
            metrics_summary["reward_mean"],
            metrics_summary["advantage_mean"],
            f"{grad_norm_val:.4g}" if grad_norm_val is not None else "N/A",
            f"{perplexity_val:.4f}" if perplexity_val is not None else "N/A",
            step_time_sec,
        )

      if self.on_step_end:
        self.on_step_end(current_step, step_result)
      self._step += 1
      # Strictly after the increment: the dispatcher re-reads `_next_batch` on
      # waking, and waking it on the old value would park it again with no one
      # left to set the event.
      self._release_window()
      _release_host_memory()
  # fmt: on


# ---------------------------------------------------------------------------
# Running one scenario
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Scenario:
  order: str = "arrival"
  num_prompts: int = 12  # groups available in the queue
  full_batch_size: int = 4
  mini_batch_size: int = 4
  max_steps: int | None = 3
  async_sync: bool = True
  sync_weights: bool = True
  pipeline_microbatches: bool = True
  logps: bool = False  # use_rollout_logps with old logps: rescored per mb
  reference_kl: bool = False
  val_start_step: int | None = None
  # Seconds between groups reaching the queue; 0 puts them all up front.
  arrival_gap_s: float = 0.0
  # Batches (PROMPT_BATCH) declared shorter than full: batch_idx -> groups.
  short_batches: dict[int, int] = dataclasses.field(default_factory=dict)
  costs: _Costs = dataclasses.field(default_factory=_Costs)
  dest_s: float = 0.0
  round_s: float = 0.0
  pack_s: float = 0.0
  tail_s: float = 0.0  # on_step_end blocks the orchestrator this long


@dataclasses.dataclass
class _Run:
  log: list[Any]
  metrics: list[tuple[str, str, float, int]]
  checkpoints: list[dict[str, Any]]
  last_step_result: Any
  trainer: _Trainer
  program: rl_program.StandardRLProgram


def _new_program(
    sc: _Scenario,
    *,
    pipelined: bool,
    staged: bool,
    program_cls: type[rl_program.StandardRLProgram],
    log: list[Any],
    checkpoints: list[dict[str, Any]],
    program_kwargs: dict[str, Any] | None = None,
) -> rl_program.StandardRLProgram:
  """The program for `sc`; `program_kwargs` override or add constructor args."""
  # The program refuses use_rollout_logps=False with several updates per
  # step. Without old logps in the payloads nothing is rescored either way.
  use_rollout_logps = sc.logps or sc.full_batch_size != sc.mini_batch_size
  algo = types.SimpleNamespace(
      num_generations=NUM_GENERATIONS,
      mini_batch_size=sc.mini_batch_size,
      train_micro_batch_size=1,
      max_response_length=1024,
      requires_reference_kl=sc.reference_kl,
      use_rollout_logps=use_rollout_logps,
      algo_config=types.SimpleNamespace(
          temperature=None,
          use_rollout_logps=use_rollout_logps,
          sampler_is=None,
          # Set, so rescoring writes the trainer's logps into the batch: a
          # micro-batch rescored on the wrong weights then trains differently.
          seq_logprob_error_threshold=100.0 if sc.logps else None,
      ),
  )
  inner = batch_assembly.SequencePackedBatchAssembler(
      batch_size=2,
      num_generations=NUM_GENERATIONS,
      mini_batch_size=sc.mini_batch_size,
      max_packed_len=ROW_TOKENS,
      segment_align_multiple=1,
  )

  def _on_step_end(step, result):
    del result
    log.append(("on_step_end", step))
    if sc.tail_s:
      time.sleep(sc.tail_s)

  kwargs: dict[str, Any] = dict(
      algo=algo,
      dataset=None,
      max_steps=sc.max_steps,
      assembler=_CountingAssembler(inner, log, sc.pack_s),
      batch_size=sc.full_batch_size,
      max_staleness=0,
      sync_weights=sc.sync_weights,
      async_weight_sync=sc.async_sync,
      pipeline_train_microbatches=sc.pipeline_microbatches,
      group_order=sc.order,
      val_start_step=sc.val_start_step,
      on_step_begin=lambda step: log.append(("on_step_begin", step)),
      on_step_end=_on_step_end,
      on_checkpoint_saved=checkpoints.append,
      pipeline_step_boundary=pipelined,
      stage_next_microbatch=staged,
  )
  kwargs.update(program_kwargs or {})
  program = program_cls(**kwargs)
  program.metrics_logger = _MetricsRecorder()  # pyrefly: ignore
  return program


async def _feed_queue(
    program: rl_program.StandardRLProgram, sc: _Scenario
) -> None:
  groups = _groups(
      sc.num_prompts, full_batch_size=sc.full_batch_size, logps=sc.logps
  )
  if sc.order == "prompt_batch":
    for batch_idx, num_groups in sc.short_batches.items():
      program.scored_q.expect(batch_idx, num_groups)
  for idx, group in enumerate(groups):
    batch_idx = idx // sc.full_batch_size
    intra = idx % sc.full_batch_size
    if intra >= sc.short_batches.get(batch_idx, sc.full_batch_size):
      continue
    for item in group:
      await program.scored_q.put(item)
    if sc.arrival_gap_s:
      await asyncio.sleep(sc.arrival_gap_s)
  await program.scored_q.close()


def _run(
    sc: _Scenario,
    *,
    pipelined: bool,
    staged: bool = False,
    program_cls: type[
        rl_program.StandardRLProgram
    ] = rl_program.StandardRLProgram,
    errors: list[BaseException] | None = None,
    worker_cls: type[_Worker] = _Worker,
    trainer_cls: type[_Trainer] = _Trainer,
) -> _Run:
  """Runs `sc` to the end; with `errors`, records a failure instead of raising."""
  log: list[Any] = []
  checkpoints: list[dict[str, Any]] = []
  trainer = trainer_cls(log, sc.costs)
  host_s = sc.costs.host_s

  def transfer_s(method: str, kwargs: dict[str, Any]) -> float:
    del kwargs
    # Only the calls carrying a micro-batch move data.
    return host_s if method in ("fwd_bwd", "stage_fwd_bwd") else 0.0

  worker = worker_cls(lambda: trainer)
  worker.initialize()
  handle = _RemoteHandle(worker, transfer_s)
  ref_handle = _RemoteHandle(_Reference())
  coordinator = _Coordinator(handle, sc.dest_s, sc.round_s)
  engine = distributed_rl_engine.DistributedRLEngine(
      rollout_workers=[],
      trainer_workers={datatypes.Role.ACTOR: handle},
      inference_workers={datatypes.Role.REFERENCE: ref_handle},
      weight_sync_coordinator=coordinator,
  )
  program = _new_program(
      sc,
      pipelined=pipelined,
      staged=staged,
      program_cls=program_cls,
      log=log,
      checkpoints=checkpoints,
  )

  async def _main():
    program.engine = engine
    feeder = asyncio.create_task(_feed_queue(program, sc))
    try:
      await asyncio.wait_for(program.train_stage(), timeout=120)
    finally:
      feeder.cancel()

  try:
    asyncio.run(_main())
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if errors is None:
      raise
    errors.append(exc)
  finally:
    handle.close()
    ref_handle.close()
  return _Run(
      log=log,
      metrics=program.metrics_logger.records,  # pyrefly: ignore
      checkpoints=checkpoints,
      last_step_result=program.last_step_result,
      trainer=trainer,
      program=program,
  )


# Logged under their timing names; these measure the change, so they differ.
_TIMING_METRICS = (
    "step_time",
    "global_step_time",
    "policy_training_time",
    "exposed_generation_time",
    "weight_sync_time",
    # google/tunix#2719's deferred fetch; always 0 without a confirmed update.
    "metrics_fetch_time",
    "time_sec_total",
    "time_sec_mean",
)
# Program-side events whose place relative to trainer calls is what changes,
# a data-only transfer, the background round's release (timed by the round,
# not the program), and a wait for an update that already ran (by the fused
# call, or by the stage right after it).
_NOT_TRAINER_STATE = (
    "on_step_end",
    "on_step_begin",
    "feed",
    "flush",
    "stage",
    "release_weight_sync",
    "token_wait",
)


def _trainer_calls(log: Sequence[Any]) -> list[Any]:
  return [e for e in log if e[0] not in _NOT_TRAINER_STATE]


def _data_metrics(records):
  return [r for r in records if not any(t in r[1] for t in _TIMING_METRICS)]


def _assert_equivalent(test: absltest.TestCase, ref: _Run, new: _Run) -> None:
  """The trainer, the checkpoints and the metrics cannot tell the runs apart."""
  test.assertEqual(_trainer_calls(ref.log), _trainer_calls(new.log))
  _assert_same_outputs(test, ref, new)


def _assert_equivalent_but_metrics_fetch(
    test: absltest.TestCase, ref: _Run, new: _Run
) -> None:
  """`_assert_equivalent`, except where get_metrics sits within a boundary.

  The same calls otherwise, the same fetches (each returning the same step's
  buffer, one buffer each), each at its boundary, and the same outputs.
  """
  test.assertEqual(
      _without_metrics_fetch(ref.log), _without_metrics_fetch(new.log)
  )
  test.assertEqual(
      [e for e in ref.log if e[0] == "get_metrics"],
      [e for e in new.log if e[0] == "get_metrics"],
  )
  _assert_metrics_fetched_at_the_boundary(test, new)
  _assert_same_outputs(test, ref, new)


def _assert_same_outputs(test: absltest.TestCase, ref: _Run, new: _Run) -> None:
  """The same checkpoints, logged metric values and last step result."""
  test.assertEqual(
      [(c["step"], c["checkpoint_path"]) for c in ref.checkpoints],
      [(c["step"], c["checkpoint_path"]) for c in new.checkpoints],
  )
  test.assertEqual(_data_metrics(ref.metrics), _data_metrics(new.metrics))
  a, b = ref.last_step_result, new.last_step_result
  test.assertEqual(
      (a.step, a.policy_version, a.num_rollouts, a.num_microbatches),
      (b.step, b.policy_version, b.num_rollouts, b.num_microbatches),
  )
  test.assertEqual((a.reward_mean, a.reward_std), (b.reward_mean, b.reward_std))
  test.assertEqual(
      (a.advantage_mean, a.advantage_std), (b.advantage_mean, b.advantage_std)
  )


def _check_equivalent(ref: _Run, new: _Run) -> bool:
  """`_assert_equivalent` as a predicate, for the negative controls."""
  probe = absltest.TestCase()
  try:
    _assert_equivalent(probe, ref, new)
  except AssertionError:
    return False
  return True


def _microbatches_per_step(run: _Run) -> list[int]:
  """Micro-batches each logged step trained, from its own step metrics."""
  per_step = {
      r[3]: int(r[2])
      for r in run.metrics
      if r[0] == "orchestrator" and r[1] == "num_microbatches"
  }
  return [per_step[k] for k in sorted(per_step)]


def _boundaries(run: _Run) -> list[tuple[bool, bool]]:
  """Per step boundary k: (step k+1's first micro-batch reached the trainer
  before step k's on_step_end, that micro-batch was staged ahead).

  Step k+1's first micro-batch is found by counting each step's micro-batches,
  so this holds for any number of updates per step.
  """
  log = run.log
  fwd = [i for i, e in enumerate(log) if e[0] == "fwd_bwd"]
  out = []
  first_of_next = 0
  for k, n in enumerate(_microbatches_per_step(run)):
    first_of_next += n
    if first_of_next >= len(fwd) or ("on_step_end", k) not in log:
      break
    first = fwd[first_of_next]
    end = log.index(("on_step_end", k))
    update = max(i for i in range(first) if log[i][0] == "update")
    staged = any(
        e[0] == "stage" and e[1] == log[first][1] for e in log[update:first]
    )
    out.append((first < end, staged))
  return out


# What the program, not the trainer, does; each comes from one thread at a
# time in program order, so their sequence is deterministic.
_PROGRAM_EVENTS = ("feed", "flush", "on_step_begin", "on_step_end")


def _program_events(log: Sequence[Any]) -> list[Any]:
  return [e for e in log if e[0] in _PROGRAM_EVENTS]


def _assert_is_the_base_loop(
    test: absltest.TestCase, ref: _Run, new: _Run
) -> None:
  """Flags off: the old loop exactly -- calls, program events, metrics."""
  _assert_equivalent(test, ref, new)
  test.assertEqual(_program_events(ref.log), _program_events(new.log))
  # Every metric, timing ones included, under the same name at the same step.
  test.assertEqual(
      [(r[0], r[1], r[3]) for r in ref.metrics],
      [(r[0], r[1], r[3]) for r in new.metrics],
  )
  test.assertNotIn("stage", [e[0] for e in new.log])
  test.assertEqual(_boundaries(new), [(False, False)] * len(_boundaries(new)))
  test.assertIsNone(new.program._intake_task)


_REFERENCE_RUNS: dict[tuple[str, type[_Trainer]], _Run] = {}


def _reference(name: str, trainer_cls: type[_Trainer] = _Trainer) -> _Run:
  """The base loop's run of `_SCENARIOS[name]`, computed once per trainer."""
  key = (name, trainer_cls)
  if key not in _REFERENCE_RUNS:
    _REFERENCE_RUNS[key] = _run(
        _SCENARIOS[name],
        pipelined=False,
        program_cls=_BaseLoopProgram,
        trainer_cls=trainer_cls,
    )
  return _REFERENCE_RUNS[key]


def _without_metrics_fetch(log: Sequence[Any]) -> list[Any]:
  """Trainer calls without get_metrics, and each save without the metrics
  buffers still pending on the trainer.

  What may differ when a stage was planned but found nothing packed in time
  against a trainer that confirms its updates: that step's update is then not
  confirmed, so get_metrics() is its barrier again, ahead of the save (as
  without a confirmed update), instead of after the commit. MaxText writes
  the same completed-step checkpoint either way: it serializes undrained
  metric buffers only for an intra-step checkpoint (`save_checkpoint`,
  maxtext 853d655).
  """
  return [
      e[:5] if e[0] == "save" else e
      for e in _trainer_calls(log)
      if e[0] != "get_metrics"
  ]


def _assert_metrics_fetched_at_the_boundary(
    test: absltest.TestCase, run: _Run
) -> None:
  """Each get_metrics returns step k's buffer after update k, before step k+1.

  The rule the deferral and the pipelined boundary must both keep: MaxText
  returns only the newest step buffer, so the fetch must follow the update
  that closes it and precede the next step's first micro-batch.
  """
  calls = _trainer_calls(run.log)
  fetches = 0
  for i, event in enumerate(calls):
    if event[0] != "get_metrics":
      continue
    fetches += 1
    step = event[1]
    updates = [e for e in calls[:i] if e[0] == "update"]
    test.assertTrue(updates and updates[-1][1] == step + 1, (i, calls))
    test.assertFalse(
        any(e[0] == "fwd_bwd" and e[2] == step + 1 for e in calls[:i]),
        (i, calls),
    )
  # One fetch per update (a run that trained nothing fetches nothing).
  test.assertEqual(fetches, sum(1 for e in calls if e[0] == "update"), calls)


def _fetches_after_sync(run: _Run) -> list[bool]:
  """Per get_metrics: whether a weight sync staged since the update before it.

  True is the base loop's deferred fetch (google/tunix#2719), taken when the
  trainer confirmed the update; False is the fetch ahead of the save.
  """
  calls = _trainer_calls(run.log)
  out = []
  for i, event in enumerate(calls):
    if event[0] != "get_metrics":
      continue
    last_update = max(j for j in range(i) if calls[j][0] == "update")
    out.append(any(e[0] == "prepare_weight_sync" for e in calls[last_update:i]))
  return out


def _staged_boundaries_confirm_the_update(
    test: absltest.TestCase, run: _Run
) -> int:
  """At each stage: the last call returned at the update's dispatch, and the
  stage waited for that update. Returns the number of stages."""
  log = run.log
  stages = [i for i, e in enumerate(log) if e[0] == "stage"]
  for s in stages:
    update = max(i for i in range(s) if log[i][0] == "update")
    version = log[update][1]
    test.assertNotIn(
        ("token_wait", version), log[update:s], f"waited before the stage {log}"
    )
    nxt = next(
        (
            i
            for i in range(s + 1, len(log))
            if log[i][0] not in _NOT_TRAINER_STATE
        ),
        len(log),
    )
    test.assertIn(("token_wait", version), log[s:nxt], log)
  return len(stages)


def _assert_weights_follow_updates(test: absltest.TestCase, run: _Run) -> None:
  """Each micro-batch of step k is computed on the weights after update k."""
  version = 0
  saw = 0
  for event in run.log:
    if event[0] == "update":
      version = event[1]
    elif event[0] == "fwd_bwd":
      test.assertEqual(
          event[2], version, f"fwd_bwd {event[1]} on stale weights"
      )
      saw += 1
  test.assertGreater(saw, 0)


# ---------------------------------------------------------------------------
# Broken programs: each must be caught by a check below.
# ---------------------------------------------------------------------------


class _SendsBeforeBoundary(rl_program.StandardRLProgram):
  """WRONG: sends step k+1's first micro-batch right after step k's update.

  That is, right after step k's last train call, which runs the update: before
  step k's metrics fetch, checkpoint and weight-sync staging, so step k's
  metrics buffer is clobbered and the sync and checkpoint run with step k+1's
  gradients half accumulated. It hooks in where a stage would, at every
  boundary the first version of this loop offered a stage at (a full batch,
  the flag, a step taken ahead), so it is not narrowed by `_stage_planned`.
  """

  async def _stage_planned(self, st, *, rescored):
    del rescored
    return self.stage_next_microbatch and st is not None

  async def _maybe_stage_next_microbatch(self, st, step_result):
    task = await rl_program.StandardRLProgram._send_first_microbatch_ahead(
        self, st
    )
    if task is not None:
      await asyncio.wait({task})

  async def _send_first_microbatch_ahead(self, st):
    return None


class _PrepacksWrongBatch(rl_program.StandardRLProgram):
  """WRONG: takes the next step's groups from one batch too far ahead."""

  async def _prepack(self, st):
    if st.groups_consumed == 0 and st.current_batch_idx is not None:
      st.current_batch_idx += 1
    await super()._prepack(st)


class _NeverLatches(rl_program.StandardRLProgram):
  """Staging that never switches itself off (every stage counts as hidden)."""

  def _staging_hidden(self, device_wait_s):
    return True


class _MinGapLatch(rl_program.StandardRLProgram):
  """WRONG, the latch as first built (review finding F1).

  Stops staging once a stage takes 0.9x the SHORTEST gap between the step's
  train calls returning. A step starts on a device the weight sync drained and
  MaxText queues two computations, so with three or more calls per step that
  gap is the transfer time, not the device time, and it latches off although
  the trainer is device-bound.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._returns: list[float] = []
    self.cadences: list[float | None] = []

  async def _timed_train_step(self, batch, **kwargs):
    result = await super()._timed_train_step(batch, **kwargs)
    self._returns.append(time.monotonic())
    return result

  def _staging_hidden(self, device_wait_s):
    return True  # Only the min-gap rule below decides.

  async def _maybe_stage_next_microbatch(self, st, step_result):
    returns, self._returns = sorted(self._returns), []
    gaps = [b - a for a, b in zip(returns, returns[1:])]
    cadence = min(gaps) if gaps else None
    staged_before = st.staged is not None
    start = time.monotonic()
    update_ready = await super()._maybe_stage_next_microbatch(st, step_result)
    stage_s = time.monotonic() - start
    if st.staged is not None and not staged_before:
      self.cadences.append(cadence)
      if cadence is not None and stage_s >= 0.9 * cadence:
        self._staging_latched_off = True
    return update_ready


class _StampsBeforeUpdateLands(rl_program.StandardRLProgram):
  """WRONG: stamps the checkpoint when the update is dispatched, not landed.

  It takes the stamp when the step's last train call returns. That call runs
  the update; against a trainer that cannot confirm the update (no
  `last_update_token`) it returns at the update's dispatch.
  """

  async def _timed_train_step(self, batch, **kwargs):
    result = await super()._timed_train_step(batch, **kwargs)
    if kwargs.get("apply_optimizer"):
      self._early_stamp_ms = time.time_ns() // 1_000_000
    return result

  async def _save_step_checkpoint(self, st, step_result):
    real_time_ns = time.time_ns
    stamp = getattr(self, "_early_stamp_ms", None)
    if stamp is None:
      return await super()._save_step_checkpoint(st, step_result)
    try:
      time.time_ns = lambda: stamp * 1_000_000
      return await super()._save_step_checkpoint(st, step_result)
    finally:
      time.time_ns = real_time_ns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# The trainer-bound regime: a micro-batch's device time well above the time
# to pack one (tr-s3: 15.11 s against ~2.1 s), so step k+1's first micro-batch
# is packed, and staged, before step k's boundary ends and the pipelined paths
# are the ones taken. (`zero_costs` and `groups_arrive_late` cover the rest.)
_BUSY = _Costs(device_s=0.15, update_s=0.01, host_s=0.02)

_SCENARIOS = dict(
    arrival=_Scenario(order="arrival", costs=_BUSY),
    prompt_batch=_Scenario(order="prompt_batch", costs=_BUSY),
    zero_costs=_Scenario(order="prompt_batch"),
    no_microbatch_pipelining=_Scenario(
        order="prompt_batch", pipeline_microbatches=False, costs=_BUSY
    ),
    blocking_sync=_Scenario(
        order="prompt_batch", async_sync=False, costs=_BUSY
    ),
    no_sync=_Scenario(order="arrival", sync_weights=False, costs=_BUSY),
    two_updates_per_step=_Scenario(
        order="prompt_batch",
        full_batch_size=4,
        mini_batch_size=2,
        costs=_BUSY,
    ),
    checkpoints_every_step=_Scenario(
        order="prompt_batch", val_start_step=1, max_steps=3, costs=_BUSY
    ),
    # Steps 0-1 unscored, steps 2-3 write scored checkpoints (MLPerf-like).
    checkpoints_from_step_3=_Scenario(
        order="prompt_batch",
        num_prompts=16,
        val_start_step=3,
        max_steps=4,
        costs=_BUSY,
    ),
    short_last_batch=_Scenario(
        order="prompt_batch",
        num_prompts=12,
        short_batches={2: 3},
        val_start_step=2,
        costs=_BUSY,
    ),
    dataset_runs_out=_Scenario(
        order="arrival", num_prompts=10, max_steps=None, costs=_BUSY
    ),
    reference_kl=_Scenario(
        order="prompt_batch", reference_kl=True, costs=_BUSY
    ),
    rescored_on_live_weights=_Scenario(
        order="prompt_batch", logps=True, costs=_BUSY
    ),
    groups_arrive_late=_Scenario(order="prompt_batch", arrival_gap_s=0.03),
)

# Scenarios in which step k+1's first micro-batch is ready ahead of time and
# is neither its step's last nor rescored, so it is sent (and staged) early.
_SENT_AHEAD = (
    "arrival",
    "prompt_batch",
    "no_microbatch_pipelining",
    "blocking_sync",
    "no_sync",
    "checkpoints_every_step",
    "checkpoints_from_step_3",
    "reference_kl",
)


def _writes_scored_checkpoint(sc: _Scenario, step: int) -> bool:
  """Step `step` (0-based, one update per step) writes a scored checkpoint."""
  return sc.val_start_step is not None and step + 1 >= sc.val_start_step


class BaseLoopOracleTest(absltest.TestCase):
  """The reference really is the loop as it was before this change."""

  def test_the_frozen_loop_is_the_base_commits_byte_for_byte(self):
    source = inspect.getsource(_BaseLoopProgram._train_loop)
    self.assertEqual(
        hashlib.sha256(source.encode()).hexdigest(), _BASE_TRAIN_LOOP_SHA256
    )

  def test_the_pin_catches_an_edited_copy(self):
    source = inspect.getsource(_BaseLoopProgram._train_loop)
    edited = source.replace("groups_consumed += 1", "groups_consumed += 2", 1)
    self.assertNotEqual(edited, source)
    self.assertNotEqual(
        hashlib.sha256(edited.encode()).hexdigest(), _BASE_TRAIN_LOOP_SHA256
    )

  def test_the_unified_loop_is_not_the_frozen_one(self):
    # Otherwise every comparison below would be a loop against itself.
    self.assertIsNot(
        _BaseLoopProgram._train_loop, rl_program.StandardRLProgram._train_loop
    )
    self.assertNotEqual(
        inspect.getsource(_BaseLoopProgram._train_loop),
        inspect.getsource(rl_program.StandardRLProgram._train_loop),
    )


_MODES = {
    "off": dict(pipelined=False, staged=False),
    "unstaged": dict(pipelined=True, staged=False),
    "staged": dict(pipelined=True, staged=True),
}


class PipelinedBoundaryEquivalenceTest(parameterized.TestCase):
  """The unified loop trains exactly what the base loop trains."""

  @parameterized.named_parameters(
      (f"{name}_{mode}", name, mode) for name in _SCENARIOS for mode in _MODES
  )
  def test_trainer_cannot_tell_it_from_the_base_loop(self, name, mode):
    sc = _SCENARIOS[name]
    ref = _reference(name)
    new = _run(sc, **_MODES[mode])
    staged = _MODES[mode]["staged"]
    _assert_equivalent(self, ref, new)
    _assert_weights_follow_updates(self, new)
    # One set of step metrics per step, from one get_metrics that returned a
    # single (that step's) buffer.
    steps = [r[3] for r in new.metrics if r[1] == "mean" and r[0] == "rewards"]
    self.assertEqual(steps, sorted(set(steps)))
    self.assertTrue(
        all(e[2] == 1 for e in new.log if e[0] == "get_metrics"), new.log
    )
    self.assertIsNone(new.program._pending_train)
    self.assertTrue(
        new.program._intake_task is None or new.program._intake_task.done()
    )
    if mode == "off":
      # The default path is the old loop, program events and all.
      _assert_is_the_base_loop(self, ref, new)
      return
    if name in _SENT_AHEAD:
      # The pipelined path really ran, at every boundary; the reference never
      # sends a step's micro-batch before the previous step's bookkeeping.
      # Staging is skipped after a step that writes a scored checkpoint.
      new_b, ref_b = _boundaries(new), _boundaries(ref)
      self.assertNotEmpty(new_b)
      expected = [
          (True, staged and not _writes_scored_checkpoint(sc, k))
          for k in range(len(new_b))
      ]
      self.assertEqual(new_b, expected, new.log)
      self.assertEqual(ref_b, [(False, False)] * len(ref_b))
    if not staged:
      self.assertNotIn("stage", [e[0] for e in new.log])


class PipelinedBoundaryChecksFailOnWrongProgramsTest(absltest.TestCase):
  """Rule 7: each gate above is shown to fire on a deliberately wrong input."""

  def test_equivalence_check_catches_sending_before_the_boundary(self):
    sc = _SCENARIOS["prompt_batch"]
    ref = _reference("prompt_batch")
    bad = _run(
        sc, pipelined=True, staged=True, program_cls=_SendsBeforeBoundary
    )
    self.assertFalse(_check_equivalent(ref, bad))
    # And how: step k+1's first fwd_bwd reached the trainer before step k's
    # get_metrics, which then returned (and dropped) the wrong buffers.
    calls = _trainer_calls(bad.log)
    first_update = next(i for i, e in enumerate(calls) if e[0] == "update")
    self.assertEqual(calls[first_update + 1][0], "fwd_bwd")
    self.assertTrue(any(e[0] == "get_metrics" and e[2] > 1 for e in calls))

  def test_equivalence_check_catches_prepacking_the_wrong_batch(self):
    sc = dataclasses.replace(_SCENARIOS["prompt_batch"], num_prompts=20)
    ref = _run(sc, pipelined=False, program_cls=_BaseLoopProgram)
    bad = _run(sc, pipelined=True, program_cls=_PrepacksWrongBatch)
    self.assertFalse(_check_equivalent(ref, bad))

  def test_equivalence_check_catches_one_swapped_trainer_call(self):
    sc = _SCENARIOS["arrival"]
    ref = _reference("arrival")
    new = _run(sc, pipelined=True)
    self.assertTrue(_check_equivalent(ref, new))
    calls = new.log
    i = next(k for k, e in enumerate(calls) if e[0] == "update")
    j = next(k for k, e in enumerate(calls) if k > i and e[0] == "fwd_bwd")
    swapped = list(calls)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    self.assertFalse(
        _check_equivalent(ref, dataclasses.replace(new, log=swapped))
    )

  def test_weights_check_catches_a_stale_micro_batch(self):
    sc = _SCENARIOS["arrival"]
    run = _run(sc, pipelined=True)
    stale = [
        (e[0], e[1], e[2] - 1) if e[0] == "fwd_bwd" and e[2] > 0 else e
        for e in run.log
    ]
    with self.assertRaises(AssertionError):
      _assert_weights_follow_updates(self, dataclasses.replace(run, log=stale))

  def test_base_loop_check_catches_a_reordered_boundary(self):
    # The pipelined mode trains the same thing but moves the next step's
    # feeds ahead of this step's on_step_end: the strict flags-off check must
    # see that, or it would pass a default path that silently pipelines.
    sc = _SCENARIOS["prompt_batch"]
    ref = _reference("prompt_batch")
    pipelined = _run(sc, pipelined=True)
    self.assertTrue(_check_equivalent(ref, pipelined))
    probe = absltest.TestCase()
    with self.assertRaises(AssertionError):
      _assert_is_the_base_loop(probe, ref, pipelined)
    self.assertNotEqual(
        _program_events(ref.log), _program_events(pipelined.log)
    )


class PipelinedBoundaryOrderingTest(absltest.TestCase):
  """What the pipelined loop does change: when the orchestrator works."""

  def _index(self, log, pred):
    return next(i for i, e in enumerate(log) if pred(e))

  def test_next_step_is_sent_before_this_steps_bookkeeping(self):
    sc = _SCENARIOS["prompt_batch"]
    for pipelined, expect_before in ((False, False), (True, True)):
      run = _run(sc, pipelined=pipelined)
      log = run.log
      # Step 0's first fwd_bwd after its sync staged belongs to step 1.
      staged_0 = self._index(log, lambda e: e[0] == "prepare_weight_sync")
      first_1 = (
          self._index(log[staged_0:], lambda e: e[0] == "fwd_bwd") + staged_0
      )
      end_0 = self._index(log, lambda e: e == ("on_step_end", 0))
      self.assertEqual(
          first_1 < end_0,
          expect_before,
          f"pipelined={pipelined}: step 1's first fwd_bwd at {first_1},"
          f" on_step_end(0) at {end_0}",
      )
      # Callback order is unchanged: on_step_end(0) before on_step_begin(1).
      self.assertLess(
          end_0, self._index(log, lambda e: e == ("on_step_begin", 1))
      )

  def test_next_steps_groups_are_packed_while_this_step_still_trains(self):
    sc = _SCENARIOS["prompt_batch"]
    for pipelined in (False, True):
      run = _run(sc, pipelined=pipelined)
      log = run.log
      feeds = [i for i, e in enumerate(log) if e[0] == "feed"]
      # Step 1's first group is the 5th feed (4 groups per step).
      first_feed_1 = feeds[sc.full_batch_size]
      update_0 = self._index(log, lambda e: e[0] == "update")
      self.assertEqual(first_feed_1 < update_0, pipelined, log)

  def test_nothing_is_taken_ahead_past_the_last_step(self):
    sc = dataclasses.replace(_SCENARIOS["arrival"], num_prompts=40, max_steps=2)
    run = _run(sc, pipelined=True)
    feeds = [e for e in run.log if e[0] == "feed"]
    self.assertLen(feeds, 2 * sc.full_batch_size)
    self.assertIsNone(run.program._intake_task)


class _RecordingPlans(rl_program.StandardRLProgram):
  """Records each `_stage_planned` decision and what the stage then reported."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.plans: list[bool] = []
    self.stage_results: list[bool] = []

  async def _stage_planned(self, st, *, rescored):
    planned = await super()._stage_planned(st, rescored=rescored)
    self.plans.append(planned)
    return planned

  async def _maybe_stage_next_microbatch(self, st, step_result):
    update_ready = await super()._maybe_stage_next_microbatch(st, step_result)
    self.stage_results.append(update_ready)
    return update_ready


class TokenTrainerRegimeTest(absltest.TestCase):
  """The double that confirms its update really puts the base loop in the
  deferred-fetch regime; the double that cannot keeps it out (two-sided)."""

  def test_only_the_token_trainer_defers_the_fetch(self):
    with_token = _reference("prompt_batch", _TokenTrainer)
    without = _reference("prompt_batch")
    self.assertEqual(
        _fetches_after_sync(with_token),
        [True] * len(_fetches_after_sync(with_token)),
    )
    self.assertNotEmpty(_fetches_after_sync(with_token))
    self.assertEqual(
        _fetches_after_sync(without),
        [False] * len(_fetches_after_sync(without)),
    )
    # The fused call waited for each update (one token wait per update).
    updates = [e for e in with_token.log if e[0] == "update"]
    waits = [e for e in with_token.log if e[0] == "token_wait"]
    self.assertLen(waits, len(updates))
    self.assertNotIn("token_wait", [e[0] for e in without.log])


class TokenTrainerEquivalenceTest(parameterized.TestCase):
  """Against a trainer that confirms its update (MaxText with maxtext#5571's
  `last_update_token`), where the base loop defers get_metrics past the
  weight sync and commit. A planned stage then makes the step's last call
  return at the update's dispatch and confirms the update itself, so a staged
  boundary keeps the deferred fetch."""

  @parameterized.named_parameters(
      (f"{name}_{mode}", name, mode) for name in _SCENARIOS for mode in _MODES
  )
  def test_trainer_cannot_tell_it_from_the_base_loop(self, name, mode):
    sc = _SCENARIOS[name]
    ref = _reference(name, _TokenTrainer)
    new = _run(sc, **_MODES[mode], trainer_cls=_TokenTrainer)
    _assert_weights_follow_updates(self, new)
    _assert_metrics_fetched_at_the_boundary(self, new)
    self.assertIsNone(new.program._pending_train)
    if mode == "off":
      _assert_is_the_base_loop(self, ref, new)
      return
    if mode == "unstaged" or name in _SENT_AHEAD:
      # Nothing planned that did not happen: the base loop's calls exactly,
      # deferred fetch included.
      _assert_equivalent(self, ref, new)
    else:
      _assert_equivalent_but_metrics_fetch(self, ref, new)
    stages = _staged_boundaries_confirm_the_update(self, new)
    if mode == "staged" and name in _SENT_AHEAD:
      expected = sum(
          not _writes_scored_checkpoint(sc, k)
          for k in range(len(_boundaries(new)))
      )
      self.assertEqual(stages, expected, new.log)
    if mode == "unstaged":
      self.assertEqual(stages, 0)


class TokenTrainerChecksFailOnWrongProgramsTest(absltest.TestCase):
  """Rule 7 for the token-trainer checks, the relaxed one above all."""

  def test_the_relaxed_check_still_catches_the_wrong_programs(self):
    sc = _SCENARIOS["prompt_batch"]
    ref = _reference("prompt_batch", _TokenTrainer)
    sends = _run(
        sc,
        pipelined=True,
        staged=True,
        program_cls=_SendsBeforeBoundary,
        trainer_cls=_TokenTrainer,
    )
    with self.assertRaises(AssertionError):
      _assert_equivalent_but_metrics_fetch(absltest.TestCase(), ref, sends)
    wide = dataclasses.replace(sc, num_prompts=20)
    wide_ref = _run(
        wide,
        pipelined=False,
        program_cls=_BaseLoopProgram,
        trainer_cls=_TokenTrainer,
    )
    wrong_batch = _run(
        wide,
        pipelined=True,
        program_cls=_PrepacksWrongBatch,
        trainer_cls=_TokenTrainer,
    )
    with self.assertRaises(AssertionError):
      _assert_equivalent_but_metrics_fetch(
          absltest.TestCase(), wide_ref, wrong_batch
      )

  def test_the_relaxed_check_allows_only_a_fetch_moved_within_its_boundary(
      self,
  ):
    run = _reference("prompt_batch", _TokenTrainer)
    log = list(run.log)
    fetch = next(i for i, e in enumerate(log) if e[0] == "get_metrics")
    save = max(i for i in range(fetch) if log[i][0] == "save")
    # Allowed: the fetch ahead of the save, the order without a confirmed
    # update (the save then has no buffer pending).
    ahead = log[:save] + [log[fetch]] + log[save:fetch] + log[fetch + 1 :]
    ahead[save + 1] = ahead[save + 1][:5] + ((),)
    moved = dataclasses.replace(run, log=ahead)
    self.assertFalse(_check_equivalent(run, moved))
    _assert_equivalent_but_metrics_fetch(self, run, moved)
    # Not allowed: the fetch after the next step's first micro-batch.
    nxt = next(i for i in range(fetch + 1, len(log)) if log[i][0] == "fwd_bwd")
    late = (
        log[:fetch] + log[fetch + 1 : nxt + 1] + [log[fetch]] + log[nxt + 1 :]
    )
    with self.assertRaises(AssertionError):
      _assert_equivalent_but_metrics_fetch(
          absltest.TestCase(), run, dataclasses.replace(run, log=late)
      )


class TokenTrainerStagingTest(absltest.TestCase):
  """How a stage and the deferred fetch combine against a token trainer."""

  _BOUNDARY_EVENTS = frozenset({
      "update",
      "token_wait",
      "stage",
      "save",
      "prepare_weight_sync",
      "get_metrics",
      "fwd_bwd",
  })

  def _boundary_calls(self, run: _Run, step: int) -> list[str]:
    """What reaches the trainer from step `step`'s update to step+1's first
    fwd_bwd (the background round's release left out)."""
    calls = run.log
    updates = [i for i, e in enumerate(calls) if e[0] == "update"]
    start = updates[step]
    end = next(
        i for i in range(start + 1, len(calls)) if calls[i][0] == "fwd_bwd"
    )
    return [
        e[0] for e in calls[start : end + 1] if e[0] in self._BOUNDARY_EVENTS
    ]

  def test_a_staged_boundary_keeps_the_base_loops_order(self):
    sc = _SCENARIOS["prompt_batch"]
    base = _reference("prompt_batch", _TokenTrainer)
    staged = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_TokenTrainer,
        program_cls=_RecordingPlans,
    )
    self.assertEqual(
        self._boundary_calls(base, 0),
        [
            "update",
            "token_wait",
            "save",
            "prepare_weight_sync",
            "get_metrics",
            "fwd_bwd",
        ],
    )
    self.assertEqual(
        self._boundary_calls(staged, 0),
        [
            "update",
            "stage",
            "token_wait",
            "save",
            "prepare_weight_sync",
            "get_metrics",
            "fwd_bwd",
        ],
    )
    self.assertTrue(staged.program.stage_results[0])

  def test_a_planned_stage_that_finds_nothing_packed_fetches_first(self):
    # Packing one micro-batch (0.25 s) outlasts the last call (~20 ms), so the
    # stage is planned but nothing is packed when the call returns.
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"],
        pack_s=0.25,
        costs=_Costs(device_s=0.01, update_s=0.005, host_s=0.005),
    )
    run = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_TokenTrainer,
        program_cls=_RecordingPlans,
    )
    self.assertNotIn("stage", [e[0] for e in run.log])
    self.assertTrue(any(run.program.plans), run.program.plans)
    # The update was not confirmed (the call returned at its dispatch and no
    # stage waited for it), so the fetch is its barrier again, ahead of the
    # save: the order the base loop uses without a confirmed update.
    self.assertEqual(
        self._boundary_calls(run, 0),
        ["update", "get_metrics", "save", "prepare_weight_sync", "fwd_bwd"],
    )
    self.assertEqual(_fetches_after_sync(run)[0], False)
    _assert_equivalent_but_metrics_fetch(
        self, _run(sc, pipelined=False, trainer_cls=_TokenTrainer), run
    )
    # Control: unstaged, the same scenario keeps the deferred fetch.
    unstaged = _run(sc, pipelined=True, trainer_cls=_TokenTrainer)
    self.assertEqual(_fetches_after_sync(unstaged)[0], True)

  def test_without_a_token_a_plan_changes_nothing_the_trainer_sees(self):
    # Today's MaxText pin: the fused call never waits, so whether a stage was
    # planned, made or not, the calls are the base loop's.
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"],
        pack_s=0.25,
        costs=_Costs(device_s=0.01, update_s=0.005, host_s=0.005),
    )
    ref = _run(sc, pipelined=False, program_cls=_BaseLoopProgram)
    run = _run(sc, pipelined=True, staged=True, program_cls=_RecordingPlans)
    self.assertTrue(any(run.program.plans))
    _assert_equivalent(self, ref, run)


class CheckpointStampTest(absltest.TestCase):
  """The MLPerf rule: a checkpoint is stamped after its last weight update."""

  def _assert_stamped_after_update(self, run: _Run) -> None:
    self.assertNotEmpty(run.checkpoints)
    for ckpt in run.checkpoints:
      landed = run.trainer.update_landed_ms[ckpt["step"] - 1]
      self.assertGreaterEqual(ckpt["timestamp_ms"], int(landed))

  def test_stamp_follows_the_landed_update(self):
    costs = _Costs(device_s=0.03, update_s=0.05)
    sc = dataclasses.replace(_SCENARIOS["checkpoints_every_step"], costs=costs)
    for pipelined, staged in ((False, False), (True, False), (True, True)):
      self._assert_stamped_after_update(
          _run(sc, pipelined=pipelined, staged=staged)
      )

  def test_stamp_check_catches_a_stamp_taken_at_dispatch(self):
    costs = _Costs(device_s=0.03, update_s=0.05)
    sc = dataclasses.replace(_SCENARIOS["checkpoints_every_step"], costs=costs)
    bad = _run(
        sc, pipelined=True, staged=True, program_cls=_StampsBeforeUpdateLands
    )
    with self.assertRaises(AssertionError):
      self._assert_stamped_after_update(bad)

  def test_stamp_follows_the_landed_update_with_a_token_trainer(self):
    # Checkpoints every step and a stage planned at every boundary but the
    # last, whose update the stage, not the last call, confirms.
    costs = _Costs(device_s=0.03, update_s=0.05)
    sc = dataclasses.replace(_SCENARIOS["prompt_batch"], costs=costs)
    for pipelined, staged in ((False, False), (True, False), (True, True)):
      run = _run(
          sc, pipelined=pipelined, staged=staged, trainer_cls=_TokenTrainer
      )
      self._assert_stamped_after_update(run)
      if staged:
        self.assertGreater(_staged_boundaries_confirm_the_update(self, run), 0)

  def test_a_planned_stage_makes_the_last_call_return_at_dispatch(self):
    # Two-sided: stamping as the last call returns is right while that call
    # waits for the update (the base loop against a token trainer), and early
    # once a planned stage has it return at the update's dispatch -- which is
    # why the program then takes the stamp after the stage's confirmation.
    costs = _Costs(device_s=0.03, update_s=0.05)
    sc = dataclasses.replace(_SCENARIOS["prompt_batch"], costs=costs)
    waits = _run(
        sc,
        pipelined=False,
        trainer_cls=_TokenTrainer,
        program_cls=_StampsBeforeUpdateLands,
    )
    self._assert_stamped_after_update(waits)
    at_dispatch = _run(
        sc,
        pipelined=True,
        staged=True,
        trainer_cls=_TokenTrainer,
        program_cls=_StampsBeforeUpdateLands,
    )
    self.assertIn("stage", [e[0] for e in at_dispatch.log])
    with self.assertRaises(AssertionError):
      self._assert_stamped_after_update(at_dispatch)

  def _stamp_delays_s(self, run: _Run) -> dict[int, float]:
    """Seconds from each scored checkpoint's update landing to its stamp."""
    return {
        c["step"]: (
            (c["timestamp_ms"] - run.trainer.update_landed_ms[c["step"] - 1])
            / 1000.0
        )
        for c in run.checkpoints
    }

  def test_staging_never_delays_a_scored_stamp_when_feed_bound(self):
    # Feed-bound: moving a micro-batch (0.3 s) outlasts the device's work left
    # after the update is dispatched (0.06 s), as with several trainer copies
    # fed through one controller. Steps 2-3 write scored checkpoints.
    costs = _Costs(device_s=0.05, update_s=0.01, host_s=0.3)
    sc = dataclasses.replace(_SCENARIOS["checkpoints_from_step_3"], costs=costs)

    class _StagesEveryStep(rl_program.StandardRLProgram):
      """WRONG: stages even after a step that writes a scored checkpoint."""

      # Without this the feed-bound latch would stop staging after one step.
      def _staging_hidden(self, device_wait_s):
        return True

      def _writes_scored_checkpoint(self, step_result):
        return False

    unstaged = self._stamp_delays_s(_run(sc, pipelined=True))
    staged_run = _run(sc, pipelined=True, staged=True)
    staged = self._stamp_delays_s(staged_run)
    bad = self._stamp_delays_s(
        _run(sc, pipelined=True, staged=True, program_cls=_StagesEveryStep)
    )
    msg = f"unstaged={unstaged} staged={staged} stages_every_step={bad}"
    print(f"STAMP_DELAYS_S {msg}")
    self.assertEqual(sorted(staged), [3, 4], msg)
    # Staging still ran where nothing is scored ...
    self.assertTrue(any(e[0] == "stage" for e in staged_run.log), msg)
    for step in (3, 4):
      # ... and the scored stamps are as prompt as without staging,
      self.assertLess(staged[step], 0.1, msg)
      self.assertLess(abs(staged[step] - unstaged[step]), 0.1, msg)
    # while staging after the scored step 3 would have pushed its stamp back
    # by the transfer's excess over the device's remaining work.
    self.assertGreater(bad[3], 0.15, msg)


def _stages(run: _Run) -> int:
  return sum(1 for e in run.log if e[0] == "stage")


def _latch_scenario(costs: _Costs, **kwargs: Any) -> _Scenario:
  """12 groups per step: several train calls per step, as at tr-s3 (8-9)."""
  base = dict(
      order="prompt_batch",
      num_prompts=48,
      full_batch_size=12,
      mini_batch_size=12,
      max_steps=4,
      costs=costs,
      dest_s=1.64 * UNIT,
      pack_s=2.1 * UNIT,
      tail_s=2.7 * UNIT,
  )
  base.update(kwargs)
  return _Scenario(**base)


def _trs3_costs() -> _Costs:
  """tr-s3's MEASURED trainer costs, one production second = UNIT."""
  return _Costs(
      device_s=15.11 * UNIT,
      update_s=1.0 * UNIT,
      stage_s=4.45 * UNIT,
      host_s=9.53 * UNIT,
      metrics_s=0.5 * UNIT,
  )


_FEED_BOUND = _Costs(device_s=0.05, update_s=0.01, host_s=0.3)


class StagingLatchTest(absltest.TestCase):
  """Staging switches itself off when the trainer is fed slower than it computes.

  The latch reads how long the trainer still waited for its device once a
  staged micro-batch was in place (`device_wait_s`). Review finding F1: the
  latch as first built compared the stage with the shortest gap between the
  step's train calls returning, which is the transfer time whenever a step
  has three or more calls, so it switched staging off on a device-bound
  trainer (`_MinGapLatch` below reproduces it).
  """

  def test_feed_bound_trainer_stops_staging_after_one_step(self):
    # Moving a micro-batch (0.3 s) outlasts computing one (0.05 s): the stage
    # finds the device idle.
    sc = _Scenario(
        order="prompt_batch",
        num_prompts=20,
        max_steps=5,
        costs=_FEED_BOUND,
    )
    run = _run(sc, pipelined=True, staged=True)
    self.assertEqual(_stages(run), 1, run.log)
    self.assertTrue(run.program._staging_latched_off)

    # Negative control: without the latch it stages at all 4 boundaries.
    unlatched = _run(sc, pipelined=True, staged=True, program_cls=_NeverLatches)
    self.assertEqual(_stages(unlatched), 4, unlatched.log)
    self.assertFalse(unlatched.program._staging_latched_off)
    # Either way the trainer trains the same thing.
    _assert_equivalent(
        self, _run(sc, pipelined=False, program_cls=_BaseLoopProgram), run
    )

  def test_trainer_bound_trainer_keeps_staging(self):
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"], num_prompts=20, max_steps=5
    )
    run = _run(sc, pipelined=True, staged=True)
    self.assertEqual(_stages(run), 4, run.log)
    self.assertFalse(run.program._staging_latched_off)

  def test_trainer_bound_at_trs3_ratios_keeps_staging(self):
    # PRE-REGISTERED (review P1/P2, inverted for the fix): device 15.11,
    # transfer 9.53 units, >= 3 calls per step. All 3 boundaries stage, and
    # each stage finds ~D + update - T = 6.6 units of device work left.
    sc = _latch_scenario(_trs3_costs())
    with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
      run = _run(sc, pipelined=True, staged=True)
    waits = [w / UNIT for w in run.trainer.device_waits]
    msg = f"device waits (units) {waits}"
    print(f"TRS3_LATCH {msg} microbatches/step {_microbatches_per_step(run)}")
    self.assertGreaterEqual(min(_microbatches_per_step(run)), 3, run.log)
    self.assertEqual(_stages(run), 3, msg)
    self.assertFalse(run.program._staging_latched_off, msg)
    self.assertLen(waits, 3, msg)
    for wait in waits:
      self.assertBetween(wait, 3.0, 12.0, msg)
    _assert_equivalent(
        self, _run(sc, pipelined=False, program_cls=_BaseLoopProgram), run
    )

  def test_the_first_latch_tripped_at_trs3_ratios(self):
    # Negative control (review P1): the min-gap latch, same scenario, stages
    # once and switches off, the gap it read being the transfer, not D.
    sc = _latch_scenario(_trs3_costs())
    with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
      bad = _run(sc, pipelined=True, staged=True, program_cls=_MinGapLatch)
    cadences = [c / UNIT for c in bad.program.cadences if c is not None]
    msg = f"cadences (units) {cadences}"
    self.assertEqual(_stages(bad), 1, msg)
    self.assertTrue(bad.program._staging_latched_off)
    self.assertBetween(cadences[0], 8.0, 11.5, msg)

  def test_lane_busy_costs_with_three_or_more_calls_keep_staging(self):
    # Review P3, inverted: D/T = 7.5, >= 3 calls per step.
    sc = _latch_scenario(_BUSY, dest_s=0.0, pack_s=0.0, tail_s=0.0)
    run = _run(sc, pipelined=True, staged=True)
    self.assertGreaterEqual(min(_microbatches_per_step(run)), 3, run.log)
    self.assertEqual(_stages(run), 3, run.trainer.device_waits)
    self.assertFalse(run.program._staging_latched_off)
    # And the first latch did trip here.
    bad = _run(sc, pipelined=True, staged=True, program_cls=_MinGapLatch)
    self.assertEqual(_stages(bad), 1)

  def test_feed_bound_with_three_or_more_calls_latches_after_one_stage(self):
    sc = _latch_scenario(
        _FEED_BOUND,
        num_prompts=60,
        max_steps=5,
        dest_s=0.0,
        pack_s=0.0,
        tail_s=0.0,
    )
    run = _run(sc, pipelined=True, staged=True)
    self.assertGreaterEqual(min(_microbatches_per_step(run)), 3, run.log)
    self.assertEqual(_stages(run), 1, run.trainer.device_waits)
    self.assertTrue(run.program._staging_latched_off)
    self.assertLess(
        run.trainer.device_waits[0],
        rl_program.StandardRLProgram._STAGING_HIDDEN_WAIT_S,
    )
    unlatched = _run(sc, pipelined=True, staged=True, program_cls=_NeverLatches)
    self.assertEqual(_stages(unlatched), 4)

  def test_device_wait_reaches_the_latch(self):
    # What the latch reads is what the trainer measured, stage for stage.
    class _Recording(rl_program.StandardRLProgram):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[float | None] = []

      def _staging_hidden(self, device_wait_s):
        self.seen.append(device_wait_s)
        return super()._staging_hidden(device_wait_s)

    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"], num_prompts=20, max_steps=5
    )
    run = _run(sc, pipelined=True, staged=True, program_cls=_Recording)
    self.assertEqual(run.program.seen, run.trainer.device_waits)
    self.assertLen(run.program.seen, 4)


_SUPPORT_SCENARIO = dataclasses.replace(
    _SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4
)


class StagingSupportTest(absltest.TestCase):
  """Staging runs only on a trainer that supports it (review finding F4)."""

  def test_an_older_trainer_image_gets_the_boundary_unstaged(self):
    sc = _SUPPORT_SCENARIO
    run = _run(sc, pipelined=True, staged=True, worker_cls=_OldTrainerWorker)
    self.assertEqual(run.program.step, sc.max_steps)
    self.assertNotIn("stage", [e[0] for e in run.log])
    self.assertFalse(run.program._staging_supported)
    self.assertTrue(run.program._staging_latched_off)
    self.assertEqual(
        _boundaries(run), [(True, False)] * (sc.max_steps - 1), run.log
    )
    _assert_equivalent(
        self, _run(sc, pipelined=False, program_cls=_BaseLoopProgram), run
    )

  def test_without_the_check_an_older_trainer_image_fails_the_run(self):
    # Negative control, the review's F1 measurement: staging without asking.
    class _SkipsTheCheck(rl_program.StandardRLProgram):

      async def _trainer_can_stage(self):
        return True

    errors: list[BaseException] = []
    run = _run(
        _SUPPORT_SCENARIO,
        pipelined=True,
        staged=True,
        worker_cls=_OldTrainerWorker,
        program_cls=_SkipsTheCheck,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIn("stage_fwd_bwd", str(errors[0]))
    self.assertNotIn(("on_step_end", 0), run.log)

  def test_a_trainer_that_cannot_report_its_device_is_not_staged(self):
    sc = _SUPPORT_SCENARIO
    run = _run(sc, pipelined=True, staged=True, trainer_cls=_NoWaitTrainer)
    self.assertNotIn("stage", [e[0] for e in run.log])
    self.assertFalse(run.program._staging_supported)
    self.assertEqual(run.program.step, sc.max_steps)
    _assert_equivalent(
        self, _run(sc, pipelined=False, program_cls=_BaseLoopProgram), run
    )

  def test_the_trainer_is_asked_once(self):
    calls = []
    real_info = _Worker.info

    def _counting_info(self):
      calls.append(1)
      return real_info(self)

    with mock.patch.object(_Worker, "info", _counting_info):
      run = _run(_SUPPORT_SCENARIO, pipelined=True, staged=True)
    self.assertLen(calls, 1)
    self.assertEqual(_stages(run), 3)
    self.assertTrue(run.program._staging_supported)

  def test_unstaged_and_off_never_ask(self):
    calls = []
    real_info = _Worker.info

    def _counting_info(self):
      calls.append(1)
      return real_info(self)

    with mock.patch.object(_Worker, "info", _counting_info):
      _run(_SUPPORT_SCENARIO, pipelined=True, staged=False)
      _run(_SUPPORT_SCENARIO, pipelined=False)
    self.assertEmpty(calls)

  def test_a_stage_that_fails_on_the_trainer_is_fatal(self):
    # The trainer ran the stage and it raised: the run ends with that error,
    # as it would for a train call (past its refusals, stage_fwd_bwd puts the
    # worker in ERROR, so no checkpoint could follow). A stage that fails on
    # the way, or in the reference call before it, falls back to sending the
    # micro-batch whole instead (rl_program_pipelined_boundary_robustness).
    real = _Worker.stage_fwd_bwd
    calls = {"n": 0}

    def _flaky(self, request, **kwargs):
      calls["n"] += 1
      if calls["n"] == 1:
        raise ConnectionError("raised by the worker's stage_fwd_bwd")
      return real(self, request, **kwargs)

    errors: list[BaseException] = []
    with mock.patch.object(_Worker, "stage_fwd_bwd", _flaky):
      run = _run(_SUPPORT_SCENARIO, pipelined=True, staged=True, errors=errors)
    self.assertLen(errors, 1)
    self.assertIn("raised by the worker's stage_fwd_bwd", str(errors[0]))
    self.assertEqual(run.program.step, 0)


def _advance_over_holes_eq(self):
  """`_advance_over_holes` before the fix: passes a hole only at serving == next."""
  while (
      self._serving_batch_idx == self._next_batch_idx
      and self._next_batch_idx in self._sealed
      and self._admitted.get(self._next_batch_idx, 0) == 0
  ):
    hole = self._next_batch_idx
    self._drop_batch_state(hole)
    self._advance_to(hole + 1)


class _CursorRecordingMixin:
  """Records (step, next_batch_idx, _next_batch) at every train call.

  Labelled by the step the call trains (updates before it), not by
  `self.step`, which the pipelined loop increments after the next step's
  first call is already on the trainer.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cursor: list[tuple[int, int, int]] = []
    self._updates_done = 0

  async def _timed_train_step(self, batch, **kwargs):
    self.cursor.append(
        (self._updates_done, self.scored_q.next_batch_idx, self._next_batch)
    )
    result = await super()._timed_train_step(batch, **kwargs)
    if kwargs.get("apply_optimizer"):
      self._updates_done += 1
    return result


class _CursorRecording(_CursorRecordingMixin, rl_program.StandardRLProgram):
  pass


class _BaseCursorRecording(_CursorRecordingMixin, _BaseLoopProgram):
  pass


def _hole_scenario(short_batches: dict[int, int]) -> _Scenario:
  return dataclasses.replace(
      _SCENARIOS["prompt_batch"],
      num_prompts=16,
      max_steps=3,
      short_batches=short_batches,
  )


def _cursor_during_step(run: _Run, step: int) -> list[int]:
  return sorted({c[1] for c in run.program.cursor if c[0] == step})


class HoleCursorTest(absltest.TestCase):
  """PROMPT_BATCH, a short batch k and then a hole at k+1 (review F3).

  The pipelined boundary reads batch k+1 by index before batch k commits.
  When batch k was short, step k's own last read had already moved the
  serving cursor to k+1, and that read moves it on to k+2 over the hole.
  `_advance_over_holes` then had to accept a serving cursor past
  `next_batch_idx`, or `next_batch_idx` lagged one batch for all of step k+1.
  """

  def _runs(self, sc: _Scenario) -> tuple[_Run, _Run, _Run, _Run]:
    ref = _run(sc, pipelined=False, program_cls=_BaseCursorRecording)
    off = _run(sc, pipelined=False, program_cls=_CursorRecording)
    new = _run(sc, pipelined=True, program_cls=_CursorRecording)
    staged = _run(sc, pipelined=True, staged=True, program_cls=_CursorRecording)
    return ref, off, new, staged

  def test_short_batch_then_hole_keeps_the_dispatch_window(self):
    sc = _hole_scenario({0: 3, 1: 0})
    ref, off, new, staged = self._runs(sc)
    self.assertEqual(_cursor_during_step(ref, 1), [2])
    for run in (off, new, staged):
      _assert_equivalent(self, ref, run)
      self.assertEqual(_cursor_during_step(run, 1), [2], run.program.cursor)
      # The hole's state is dropped, not left behind.
      self.assertNotIn(1, run.program.scored_q._sealed)

  def test_the_cursor_check_catches_the_old_hole_rule(self):
    sc = _hole_scenario({0: 3, 1: 0})
    with mock.patch.object(
        trajectory_queue_manager.BatchOrderedQueueManager,
        "_advance_over_holes",
        _advance_over_holes_eq,
    ):
      ref, off, new, staged = self._runs(sc)
    # The loop without the pipelined boundary never needed the fix ...
    self.assertEqual(_cursor_during_step(ref, 1), [2])
    self.assertEqual(_cursor_during_step(off, 1), [2])
    # ... the pipelined one lagged a batch and left the hole behind,
    self.assertEqual(_cursor_during_step(new, 1), [1])
    self.assertEqual(_cursor_during_step(staged, 1), [1])
    self.assertIn(1, new.program.scored_q._sealed)
    # though it trained the same thing (why trainer equivalence missed it).
    _assert_equivalent(self, ref, new)

  def test_full_batch_then_hole_agrees_with_either_rule(self):
    sc = _hole_scenario({1: 0})
    for rule in (None, _advance_over_holes_eq):
      patch = (
          mock.patch.object(
              trajectory_queue_manager.BatchOrderedQueueManager,
              "_advance_over_holes",
              rule,
          )
          if rule is not None
          else contextlib.nullcontext()
      )
      with patch:
        runs = self._runs(sc)
      for run in runs:
        self.assertEqual(_cursor_during_step(run, 1), [2], run.program.cursor)


class _StepTimes(rl_program.StandardRLProgram):
  """Records when each step's step_time ends and when its on_step_end runs."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.ends: list[float] = []
    self.callbacks: list[float] = []
    inner = self.on_step_end

    def _on_step_end(step, result):
      self.callbacks.append(time.monotonic())
      if inner is not None:
        inner(step, result)

    self.on_step_end = _on_step_end

  async def _drain_filtered_groups(self):
    # Called right after the step's step_time is taken.
    self.ends.append(time.monotonic())
    return await super()._drain_filtered_groups()


class StepTimeTest(absltest.TestCase):
  """What `step_time` (and the RCP `train_step_time` built from it) spans.

  Review finding F5 said consecutive steps' step_time now overlap. They do
  not: step k ends at its commit, before step k+1's first micro-batch is
  sent, which is where step k+1 now starts. What changes is that step k's
  deferred bookkeeping falls inside step k+1's time instead of between two
  steps.
  """

  def test_steps_never_overlap_and_the_next_starts_at_its_first_send(self):
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4, tail_s=0.05
    )
    for pipelined in (False, True):
      run = _run(sc, pipelined=pipelined, program_cls=_StepTimes)
      times = {
          r[3]: r[2]
          for r in run.metrics
          if r[0] == "orchestrator" and r[1] == "step_time_sec"
      }
      ends, callbacks = run.program.ends, run.program.callbacks
      self.assertLen(ends, sc.max_steps)
      starts = [ends[k] - times[k] for k in range(len(ends))]
      for k in range(1, len(ends)):
        with self.subTest(pipelined=pipelined, step=k):
          self.assertGreaterEqual(starts[k], ends[k - 1])
          if pipelined:
            self.assertLess(starts[k], callbacks[k - 1])
          else:
            self.assertGreaterEqual(starts[k], callbacks[k - 1])


class StepLogTest(absltest.TestCase):
  """Step k's `Train step` line waits for step k+1's first call (review F7).

  So a `[pipeline] SENT_AHEAD step=k` line, logged once that call is sent,
  marks the step in the log before its metrics line, for monitoring keyed on
  step lines. Flags off logs none.
  """

  def _lines(self, pipelined: bool) -> list[str]:
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4
    )
    with self.assertLogs(level="INFO") as logs:
      _run(sc, pipelined=pipelined)
    return [r.getMessage() for r in logs.records]

  def test_sent_ahead_precedes_each_deferred_step_line(self):
    lines = self._lines(pipelined=True)
    for k in range(3):
      sent = next(
          i
          for i, m in enumerate(lines)
          if m.startswith(f"[pipeline] SENT_AHEAD step={k} ")
      )
      step_line = next(
          i for i, m in enumerate(lines) if m.startswith(f"Train step {k} ")
      )
      self.assertLess(sent, step_line)

  def test_flags_off_logs_no_sent_ahead(self):
    lines = self._lines(pipelined=False)
    self.assertTrue(any(m.startswith("Train step 0 ") for m in lines))
    self.assertFalse(any("SENT_AHEAD" in m for m in lines))


class _LibraryDefaults(rl_program.StandardRLProgram):
  """Drops the two new kwargs, so the library defaults apply."""

  def __init__(
      self, *args, pipeline_step_boundary=None, stage_next_microbatch=None, **kw
  ):
    del pipeline_step_boundary, stage_next_microbatch
    super().__init__(*args, **kw)


def _boom(*args, **kwargs):
  del args, kwargs
  raise AssertionError("a pipelined-boundary path ran")


class DefaultOffTest(absltest.TestCase):
  """Off by default, and off means the old loop (review D1/D2)."""

  def test_library_defaults_take_none_of_the_new_paths(self):
    sc = dataclasses.replace(
        _SCENARIOS["prompt_batch"], num_prompts=16, max_steps=4
    )
    patches = dict(
        _prepack=_boom,
        _send_first_microbatch_ahead=_boom,
        _stage_planned=_boom,
        _maybe_stage_next_microbatch=_boom,
        _trainer_can_stage=_boom,
    )
    with mock.patch.multiple(rl_program.StandardRLProgram, **patches):
      default = _run(
          sc, pipelined=True, staged=True, program_cls=_LibraryDefaults
      )
    self.assertFalse(default.program.pipeline_step_boundary)
    self.assertFalse(default.program.stage_next_microbatch)
    _assert_is_the_base_loop(
        self, _run(sc, pipelined=False, program_cls=_BaseLoopProgram), default
    )
    # Two-sided: with the flags on, the same patches fire.
    errors: list[BaseException] = []
    with mock.patch.multiple(rl_program.StandardRLProgram, **patches):
      _run(sc, pipelined=True, staged=True, errors=errors)
    self.assertLen(errors, 1)
    self.assertIn("pipelined-boundary path ran", str(errors[0]))

  def test_argparse_defaults_are_off(self):
    args = run_deepswe_dist._parse_args([])
    self.assertFalse(args.pipeline_step_boundary)
    self.assertFalse(args.stage_next_microbatch)


def _recipe_guard() -> str:
  """The STAGE_NEXT_MICROBATCH guard block of mlperf_base.sh."""
  path = os.path.join(
      os.path.dirname(os.path.abspath(tunix.__file__)),
      "experimental",
      "examples",
      "recipes",
      "mlperf_base.sh",
  )
  with open(path) as f:
    text = f.read()
  start = text.index("export PIPELINE_STEP_BOUNDARY=")
  end_marker = "unset _pipeline_step_boundary_on _stage_next_microbatch_on\n"
  return text[start : text.index(end_marker, start) + len(end_marker)]


# The guard as first built (review finding F6): only the literal "true".
_FIRST_RECIPE_GUARD = """export PIPELINE_STEP_BOUNDARY=${PIPELINE_STEP_BOUNDARY:-false}
export STAGE_NEXT_MICROBATCH=${STAGE_NEXT_MICROBATCH:-false}
if [[ "${STAGE_NEXT_MICROBATCH}" == "true" && "${PIPELINE_STEP_BOUNDARY}" != "true" ]]; then
  echo "STAGE_NEXT_MICROBATCH=true requires PIPELINE_STEP_BOUNDARY=true" >&2
  exit 1
fi
"""


def _guard_accepts(guard: str, boundary: str | None, stage: str | None) -> bool:
  env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
  if boundary is not None:
    env["PIPELINE_STEP_BOUNDARY"] = boundary
  if stage is not None:
    env["STAGE_NEXT_MICROBATCH"] = stage
  done = subprocess.run(
      ["bash", "-c", guard], env=env, capture_output=True, check=False
  )
  return done.returncode == 0


class RecipeGuardTest(absltest.TestCase):
  """mlperf_base.sh refuses staging without the boundary, in the launcher's
  own truthiness (true, True or 1 is on)."""

  # (PIPELINE_STEP_BOUNDARY, STAGE_NEXT_MICROBATCH, accepted)
  _CASES = (
      (None, None, True),
      ("true", "true", True),
      ("1", "true", True),
      ("True", "1", True),
      ("true", "True", True),
      ("false", "false", True),
      ("1", "0", True),
      ("false", "true", False),
      ("0", "1", False),
      (None, "True", False),
  )

  def test_the_guard_uses_the_launchers_truthiness(self):
    guard = _recipe_guard()
    for boundary, stage, accepted in self._CASES:
      with self.subTest(boundary=boundary, stage=stage):
        self.assertEqual(_guard_accepts(guard, boundary, stage), accepted)

  def test_the_first_guard_refused_a_valid_combination(self):
    # Negative control: the literal comparison refused boundary=1 stage=true.
    self.assertFalse(_guard_accepts(_FIRST_RECIPE_GUARD, "1", "true"))
    self.assertTrue(_guard_accepts(_recipe_guard(), "1", "true"))


# ---------------------------------------------------------------------------
# Randomized scenarios (from the review's fuzz test, against the base loop).
# ---------------------------------------------------------------------------

_NUM_DRAWS = 30


def _draw(seed: int) -> _Scenario:
  """A scenario across the harness's parameter space, including pairings the
  hand-picked scenarios never make (holes with short batches, two updates per
  step with rescoring, blocking sync with late arrivals, random costs)."""
  rng = random.Random(seed)
  order = rng.choice(["arrival", "prompt_batch"])
  full, mini = rng.choice([(4, 4), (4, 2), (6, 3), (3, 3), (2, 2), (6, 6)])
  num_batches = rng.randint(2, 5)
  num_prompts = full * num_batches - rng.choice([0, 0, 1])
  short = {}
  if order == "prompt_batch":
    for b in range(num_batches):
      r = rng.random()
      if r < 0.15:
        short[b] = 0  # a hole
      elif r < 0.35:
        short[b] = rng.randint(1, full - 1) if full > 1 else 1
  max_steps = rng.choice([None, 1, 2, 3, num_batches])
  sync = rng.choice(["async", "async", "blocking", "none"])
  logps = rng.random() < 0.25
  costs = _Costs(
      device_s=rng.choice([0.0, 0.004, 0.02]),
      update_s=rng.choice([0.0, 0.002, 0.01]),
      host_s=rng.choice([0.0, 0.003, 0.03]),
      metrics_s=rng.choice([0.0, 0.002]),
      stage_s=rng.choice([0.0, 0.003]),
  )
  val_start = rng.choice([None, None, 1, 2, 3])
  return _Scenario(
      order=order,
      num_prompts=num_prompts,
      full_batch_size=full,
      mini_batch_size=mini,
      max_steps=max_steps,
      async_sync=(sync == "async"),
      sync_weights=(sync != "none"),
      pipeline_microbatches=rng.random() < 0.7,
      logps=logps,
      reference_kl=rng.random() < 0.2,
      val_start_step=val_start,
      arrival_gap_s=rng.choice([0.0, 0.0, 0.004, 0.015]),
      short_batches=short,
      costs=costs,
      dest_s=rng.choice([0.0, 0.003]),
      round_s=rng.choice([0.0, 0.005]),
      pack_s=rng.choice([0.0, 0.003]),
      tail_s=rng.choice([0.0, 0.004]),
  )


def _draw_trainer_bound(seed: int) -> _Scenario:
  """Draws in which step k+1's first micro-batch is usually ready early."""
  rng = random.Random(10_000 + seed)
  sc = _draw(10_000 + seed)
  num_batches = rng.randint(3, 5)
  return dataclasses.replace(
      sc,
      num_prompts=sc.full_batch_size * num_batches,
      max_steps=rng.choice([3, None, num_batches]),
      costs=_Costs(
          device_s=rng.choice([0.02, 0.03]),
          update_s=rng.choice([0.002, 0.01]),
          host_s=rng.choice([0.002, 0.006]),
          metrics_s=rng.choice([0.0, 0.002]),
          stage_s=rng.choice([0.0, 0.003]),
      ),
      arrival_gap_s=rng.choice([0.0, 0.0, 0.002]),
      logps=rng.random() < 0.15,
  )


class _Counting(rl_program.StandardRLProgram):
  """Counts how often the sent-ahead path actually ran."""

  sent_ahead = 0

  async def _send_first_microbatch_ahead(self, st):
    task = await super()._send_first_microbatch_ahead(st)
    if task is not None:
      _Counting.sent_ahead += 1  # On the base class, also from subclasses.
    return task


class _CountingNoLatch(_Counting):

  def _staging_hidden(self, device_wait_s):
    return True


def _assert_matches_base_loop(
    test: absltest.TestCase, ref: _Run, new: _Run, mode: str
) -> None:
  _assert_equivalent(test, ref, new)
  if any(e[0] == "fwd_bwd" for e in new.log):
    _assert_weights_follow_updates(test, new)
  test.assertIsNone(new.program._pending_train)
  test.assertTrue(
      new.program._intake_task is None or new.program._intake_task.done()
  )
  if mode == "off":
    _assert_is_the_base_loop(test, ref, new)
  if not _MODES[mode]["staged"]:
    test.assertNotIn("stage", [e[0] for e in new.log])


class RandomizedEquivalenceTest(parameterized.TestCase):
  """PRE-REGISTERED (review): all draws pass in every mode."""

  @parameterized.parameters(*range(_NUM_DRAWS))
  def test_unified_loop_matches_the_base_loop(self, seed):
    sc = _draw(seed)
    ref = _run(sc, pipelined=False, program_cls=_BaseLoopProgram)
    for mode, kwargs in _MODES.items():
      new = _run(sc, **kwargs)
      with self.subTest(mode=mode, scenario=repr(sc)):
        _assert_matches_base_loop(self, ref, new, mode)


class TokenRandomizedEquivalenceTest(parameterized.TestCase):
  """The same draws against a trainer that confirms its update.

  Flags off and unstaged must be the base loop's calls exactly. Staged, a
  draw may plan a stage that finds nothing packed in time (random costs,
  late groups), so the relaxed check applies; every stage that did happen
  must have confirmed the update after the last call returned at dispatch.
  """

  @parameterized.parameters(*range(_NUM_DRAWS))
  def test_unified_loop_matches_the_base_loop(self, seed):
    sc = _draw(seed)
    ref = _run(
        sc,
        pipelined=False,
        program_cls=_BaseLoopProgram,
        trainer_cls=_TokenTrainer,
    )
    for mode, kwargs in _MODES.items():
      new = _run(sc, **kwargs, trainer_cls=_TokenTrainer)
      with self.subTest(mode=mode, scenario=repr(sc)):
        if mode == "staged":
          _assert_equivalent_but_metrics_fetch(self, ref, new)
          _staged_boundaries_confirm_the_update(self, new)
          if any(e[0] == "fwd_bwd" for e in new.log):
            _assert_weights_follow_updates(self, new)
          self.assertIsNone(new.program._pending_train)
        else:
          _assert_matches_base_loop(self, ref, new, mode)


class TokenRandomizedCoverageTest(absltest.TestCase):
  """Rule 7 on the token draws: they reach both staged outcomes."""

  def test_draws_stage_and_fall_back(self):
    staged_draws = fallback_draws = 0
    for seed in range(_NUM_DRAWS):
      sc = _draw_trainer_bound(seed)
      ref = _run(
          sc,
          pipelined=False,
          program_cls=_BaseLoopProgram,
          trainer_cls=_TokenTrainer,
      )
      new = _run(
          sc,
          pipelined=True,
          staged=True,
          program_cls=_CountingNoLatch,
          trainer_cls=_TokenTrainer,
      )
      _assert_equivalent_but_metrics_fetch(self, ref, new)
      staged_draws += _staged_boundaries_confirm_the_update(self, new) > 0
      fallback_draws += not _check_equivalent(ref, new)
    print(
        f"TOKEN_COVERAGE staged_draws={staged_draws}"
        f" fallback_draws={fallback_draws}"
    )
    self.assertGreater(staged_draws, 0)
    self.assertGreater(fallback_draws, 0)


class RandomizedChecksAreNotVacuousTest(absltest.TestCase):
  """Rule 7 on the draws: they reach the new paths and catch wrong programs.

  Review result, kept: the wrong programs were caught in 6/30 and 5/30 draws
  (prediction >= 5 each, HIT); coverage of the new paths is printed only (its
  predictions missed: zero device cost, max_steps 1 and the latch keep many
  draws off them). `BiasedRandomizedEquivalenceTest` covers that gap.
  """

  def test_new_paths_run_and_wrong_programs_are_caught(self):
    sent_draws = staged_draws = 0
    caught = {"sends_before_boundary": 0, "prepacks_wrong_batch": 0}
    for seed in range(_NUM_DRAWS):
      sc = _draw(seed)
      _Counting.sent_ahead = 0
      run = _run(sc, pipelined=True, staged=True, program_cls=_Counting)
      sent_draws += _Counting.sent_ahead > 0
      staged_draws += any(e[0] == "stage" for e in run.log)
      ref = _run(sc, pipelined=False, program_cls=_BaseLoopProgram)
      for name, cls, staged in (
          ("sends_before_boundary", _SendsBeforeBoundary, True),
          ("prepacks_wrong_batch", _PrepacksWrongBatch, False),
      ):
        errors: list[BaseException] = []
        bad = _run(
            sc, pipelined=True, staged=staged, program_cls=cls, errors=errors
        )
        if errors or not _check_equivalent(ref, bad):
          caught[name] += 1
    print(
        f"FUZZ_COVERAGE sent_ahead_draws={sent_draws} "
        f"staged_draws={staged_draws} caught={caught}"
    )
    self.assertGreaterEqual(caught["sends_before_boundary"], 5)
    self.assertGreaterEqual(caught["prepacks_wrong_batch"], 5)


class BiasedRandomizedEquivalenceTest(absltest.TestCase):
  """Trainer-bound draws with the latch disabled, so the new paths run."""

  def test_trainer_bound_draws(self):
    sent_draws = staged_draws = 0
    for seed in range(_NUM_DRAWS):
      sc = _draw_trainer_bound(seed)
      ref = _run(sc, pipelined=False, program_cls=_BaseLoopProgram)
      for mode in ("unstaged", "staged"):
        _Counting.sent_ahead = 0
        new = _run(sc, **_MODES[mode], program_cls=_CountingNoLatch)
        with self.subTest(seed=seed, mode=mode, scenario=repr(sc)):
          _assert_matches_base_loop(self, ref, new, mode)
        if mode == "staged":
          sent_draws += _Counting.sent_ahead > 0
          staged_draws += any(e[0] == "stage" for e in new.log)
    print(
        f"BIASED_COVERAGE sent_ahead_draws={sent_draws}"
        f" staged_draws={staged_draws}"
    )
    self.assertGreater(sent_draws, 0)
    self.assertGreater(staged_draws, 0)


class FailureHandlingTest(absltest.TestCase):

  def test_stage_next_microbatch_needs_the_pipelined_boundary(self):
    with self.assertRaisesRegex(ValueError, "requires pipeline_step_boundary"):
      rl_program.StandardRLProgram(
          algo=types.SimpleNamespace(  # pyrefly: ignore[bad-argument-type]
              num_generations=2,
              mini_batch_size=1,
              algo_config=types.SimpleNamespace(temperature=None),
          ),
          stage_next_microbatch=True,
      )

  def test_a_failed_prepack_surfaces_after_the_steps_bookkeeping(self):

    class _FailsTakingAhead(rl_program.StandardRLProgram):

      async def _prepack(self, st):
        raise RuntimeError("queue broke while taking ahead")

    errors: list[BaseException] = []
    run = _run(
        _SCENARIOS["prompt_batch"],
        pipelined=True,
        program_cls=_FailsTakingAhead,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIn("queue broke while taking ahead", str(errors[0]))
    # Step 0 finished, bookkeeping included, before step 1 raised; nothing of
    # step 1 reached the trainer.
    self.assertIn(("on_step_end", 0), run.log)
    self.assertEqual(run.program.step, 1)
    self.assertNotIn(("on_step_begin", 1), run.log)
    self.assertLen([e for e in run.log if e[0] == "update"], 1)
    self.assertIsNone(run.program._intake_task)

  def test_a_failed_early_train_call_surfaces_in_the_next_step(self):

    class _Boom(Exception):
      pass

    calls = {"n": 0}

    class _FailsSecondStepsFirstCall(rl_program.StandardRLProgram):

      async def _timed_train_step(self, batch, **kwargs):
        if (
            self.step == 0
            and calls["n"] == 1
            and not kwargs.get("apply_optimizer")
        ):
          raise _Boom("trainer rejected the micro-batch sent ahead")
        if self.step == 0 and kwargs.get("apply_optimizer"):
          calls["n"] = 1
        return await super()._timed_train_step(batch, **kwargs)

    errors: list[BaseException] = []
    run = _run(
        _SCENARIOS["prompt_batch"],
        pipelined=True,
        program_cls=_FailsSecondStepsFirstCall,
        errors=errors,
    )
    self.assertLen(errors, 1)
    self.assertIsInstance(errors[0], _Boom)
    # Step 0's bookkeeping still ran before the failure surfaced.
    self.assertIn(("on_step_end", 0), run.log)


# ---------------------------------------------------------------------------
# Timing, with the trainer's measured costs scaled down (fakes, not a TPU).
# ---------------------------------------------------------------------------


# One "second" of the real system. Costs below are tr-s3's (MEASURED, see the
# report): device 15.11 s per micro-batch, update ~1 s, host part 9.53 s,
# get_metrics ~0.5 s once the update landed, destination bind+metadata 1.64 s,
# trainer prepare_weight_sync 4.45 s, step bookkeeping 2.7 s, packing the
# first micro-batch ~4 s.
def _run_with_dispatcher(
    sc: _Scenario,
    *,
    pipelined: bool,
    staged: bool = False,
    program_cls: type[
        rl_program.StandardRLProgram
    ] = rl_program.StandardRLProgram,
    first_stagger_s: float = 0.0,
    stagger_s: float = 0.0,
    rollout_s: float = 0.01,
) -> tuple[_Run, dict[int, float], dict[int, float]]:
  """Runs `sc` with the real dispatcher (`rollout_dispatch_stage`) in front.

  The engine's `dispatch_rollouts` stands in for the rollouts and critique:
  each dispatched prompt's group reaches `scored_q` `rollout_s` later, stamped
  with policy version `max(0, batch_idx - 1)`, which `max_staleness` 1 always
  admits. (The version it was dispatched at would depend on whether step 0 had
  synced by then, i.e. on machine load.) With `max_staleness` 1, batch 1 is
  inside the dispatch window from the start, so with a blocking sync only the
  start-up stagger (google/tunix#2845) holds it.

  Returns:
    The run, and in seconds after the stagger's own reference point (batch 0
    reaching the dispatch window, `_initial_dispatch_t0`; batch 0's first
    dispatch when there is no stagger): when each prompt batch's first prompt
    was dispatched, and when each step ended.
  """
  log: list[Any] = []
  checkpoints: list[dict[str, Any]] = []
  trainer = _Trainer(log, sc.costs)
  host_s = sc.costs.host_s

  def transfer_s(method: str, kwargs: dict[str, Any]) -> float:
    del kwargs
    return host_s if method in ("fwd_bwd", "stage_fwd_bwd") else 0.0

  worker = _Worker(lambda: trainer)
  worker.initialize()
  handle = _RemoteHandle(worker, transfer_s)
  engine = distributed_rl_engine.DistributedRLEngine(
      rollout_workers=[],
      trainer_workers={datatypes.Role.ACTOR: handle},
      weight_sync_coordinator=_Coordinator(handle, sc.dest_s, sc.round_s),
  )
  program = _new_program(
      sc,
      pipelined=pipelined,
      staged=staged,
      program_cls=program_cls,
      log=log,
      checkpoints=checkpoints,
      program_kwargs=dict(
          dataset=[f"task_{i}" for i in range(sc.num_prompts)],
          max_staleness=1,
          initial_dispatch_stagger_s=stagger_s,
          initial_dispatch_first_stagger_s=first_stagger_s,
      ),
  )
  first_dispatch: dict[int, float] = {}
  step_end: dict[int, float] = {}
  deliveries: set[asyncio.Task[None]] = set()
  on_step_end = program.on_step_end
  assert on_step_end is not None

  def _timed_on_step_end(step: int, result: Any) -> None:
    step_end[step] = time.monotonic()
    on_step_end(step, result)

  program.on_step_end = _timed_on_step_end

  async def _deliver(prompt_idx: int) -> None:
    await asyncio.sleep(rollout_s)
    batch_idx = prompt_idx // sc.full_batch_size
    for gen in range(NUM_GENERATIONS):
      item = _item(
          prompt_idx, gen, full_batch_size=sc.full_batch_size, logps=sc.logps
      )
      item.policy_version = max(0, batch_idx - 1)
      await program.scored_q.put(item)

  async def _dispatch_rollouts(prompts, **kwargs):
    del kwargs
    for prompt in prompts:
      coordinates = prompt["metadata"]
      first_dispatch.setdefault(coordinates["batch_idx"], time.monotonic())
      task = asyncio.create_task(_deliver(coordinates["prompt_idx"]))
      deliveries.add(task)
      task.add_done_callback(deliveries.discard)
    return [p["prompt_id"] for p in prompts]

  engine.dispatch_rollouts = _dispatch_rollouts  # pyrefly: ignore

  async def _main():
    program.engine = engine
    dispatcher = asyncio.create_task(program.rollout_dispatch_stage())
    try:
      await asyncio.wait_for(program.train_stage(), timeout=120)
    finally:
      dispatcher.cancel()
      for task in list(deliveries):
        task.cancel()

  # As in BoundaryTimingTest: gc.collect() at a step's end can stall the event
  # loop, and with it the dispatcher, for tens of ms.
  try:
    with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
      asyncio.run(_main())
  finally:
    handle.close()
  # The stagger counts from batch 0 reaching the window, a thread hop before
  # its first dispatch; measured from the dispatch, a loaded box makes batch 1
  # look early.
  t0 = program._initial_dispatch_t0  # pylint: disable=protected-access
  if t0 is None:
    t0 = first_dispatch[0]
  run = _Run(
      log=log,
      metrics=program.metrics_logger.records,  # pyrefly: ignore
      checkpoints=checkpoints,
      last_step_result=program.last_step_result,
      trainer=trainer,
      program=program,
  )
  return (
      run,
      {b: t - t0 for b, t in first_dispatch.items()},
      {s: t - t0 for s, t in step_end.items()},
  )


class StaggeredDispatchTest(absltest.TestCase):
  """google/tunix#2845's start-up stagger, with the real dispatcher in front.

  PRE-REGISTERED, written before the first run: batch 1 is held for the first
  gap in every mode; when step 0 is over before batch 1 goes out, nothing is
  taken ahead at boundary 0 (step 1's groups do not exist yet, so the prepack
  waits on the held batch without holding up step 0 or the dispatcher); a
  later boundary is pipelined; and the trainer cannot tell any mode from the
  base loop. Control: without the stagger batch 1 goes out at once.

  Revised after the first cmcs runs, where the box ran at load ~170 on 160
  cores: the first version asserted batch 1 goes out within 0.3 s of the gap
  and boundary 0 is never taken ahead. Both assumed step 0 ends well within
  the gap; under that load it did not, in every mode including the base loop
  (MEASURED: base 1.10-1.29 s, off up to 2.08 s), and with a background sync
  the dispatch window then rightly waits for step 0's round. Now the sync
  blocks (so only the stagger holds batch 1), the lower bound is exact, the
  upper bound allows for a loaded box, and the boundary-0 claim is checked
  whenever its precondition (step 0 ended before batch 1 went out) holds.
  """

  _SC = _Scenario(
      order="prompt_batch",
      num_prompts=12,
      max_steps=3,
      costs=_BUSY,
      async_sync=False,
  )
  _FIRST_S = 1.0

  def test_the_held_batch_and_the_pipelined_boundary(self):
    ref, ref_due, _ = _run_with_dispatcher(
        self._SC,
        pipelined=False,
        program_cls=_BaseLoopProgram,
        first_stagger_s=self._FIRST_S,
    )
    self.assertGreaterEqual(ref_due[1], self._FIRST_S - 0.001, ref_due)
    for mode, kwargs in _MODES.items():
      new, due, step_end = _run_with_dispatcher(
          self._SC, **kwargs, first_stagger_s=self._FIRST_S
      )
      boundaries = _boundaries(new)
      print(
          f"STAGGER_DUE_S base={ref_due[1]:.3f} {mode}={due[1]:.3f}"
          f" step0_end={step_end[0]:.3f} boundaries={boundaries}"
      )
      with self.subTest(mode=mode):
        # Held for the gap (to the timer's resolution), and not much longer
        # even on a loaded box.
        self.assertGreaterEqual(due[1], self._FIRST_S - 0.001, due)
        self.assertLess(due[1], self._FIRST_S + 0.5, due)
        _assert_equivalent(self, ref, new)
        _assert_weights_follow_updates(self, new)
        self.assertLen(boundaries, 2)
        if step_end[0] < due[1]:
          # Step 0 was over before its successor's groups existed: the
          # prepack had nothing to take, and waited without holding it up.
          self.assertEqual(boundaries[0], (False, False), (step_end, due))
        self.assertEqual(any(b[0] for b in boundaries), kwargs["pipelined"])
    _, no_stagger, _ = _run_with_dispatcher(
        self._SC, pipelined=True, staged=True
    )
    self.assertLess(no_stagger[1], 0.5, no_stagger)


UNIT = 0.02
T_DEV, T_UPD, T_HOST, T_METRICS = 15.0, 1.0, 9.5, 0.5
T_DEST, T_STAGE, T_TAIL, T_PACK = 1.6, 4.5, 2.7, 4.0
# PRE-REGISTERED, written before the first run: device idle from step k's
# update landing to step k+1's first micro-batch starting, in those units.
PRED_TODAY = T_METRICS + T_DEST + T_STAGE + T_TAIL + T_PACK + T_HOST  # 22.8
PRED_PIPELINED = T_METRICS + T_DEST + T_STAGE + T_HOST  # 16.1
PRED_STAGED = T_METRICS + T_DEST + T_STAGE  # 6.6
# Absolute gaps carry the harness's own fixed latency (thread hops, logging:
# ~1 unit when first measured); the differences cancel it.
TOLERANCE = 3.0
DIFF_TOLERANCE = 2.0


def _boundary_gaps(
    *,
    pipelined: bool,
    staged: bool,
    trainer_cls: type[_Trainer] = _Trainer,
    program_cls: type[
        rl_program.StandardRLProgram
    ] = rl_program.StandardRLProgram,
) -> tuple[list[float], _Run]:
  """Device idle from each update landing to the next micro-batch, in units."""
  sc = _Scenario(
      order="prompt_batch",
      num_prompts=12,
      full_batch_size=6,
      mini_batch_size=6,
      max_steps=2,
      costs=_Costs(
          device_s=T_DEV * UNIT,
          update_s=T_UPD * UNIT,
          stage_s=T_STAGE * UNIT,
          host_s=T_HOST * UNIT,
          metrics_s=T_METRICS * UNIT,
      ),
      dest_s=T_DEST * UNIT,
      round_s=0.0,
      pack_s=T_PACK * UNIT,
      tail_s=T_TAIL * UNIT,
  )
  # gc.collect() costs a test process tens of ms, noise at this scale; the
  # production cost of releasing host memory is part of T_TAIL.
  with mock.patch.object(rl_program, "_release_host_memory", lambda: None):
    run = _run(
        sc,
        pipelined=pipelined,
        staged=staged,
        trainer_cls=trainer_cls,
        program_cls=program_cls,
    )
  device = run.trainer.device
  gaps = []
  for i, (kind, _, _, end) in enumerate(device):
    if kind == "update" and i + 1 < len(device):
      gaps.append((device[i + 1][2] - end) / UNIT)
  return gaps, run


class BoundaryTimingTest(absltest.TestCase):
  """The device's idle gap at the step boundary, in each mode."""

  def _gaps(self, *, pipelined: bool, staged: bool) -> list[float]:
    return _boundary_gaps(pipelined=pipelined, staged=staged)[0]

  def test_gap_today_pipelined_and_staged_match_the_prediction(self):
    today = self._gaps(pipelined=False, staged=False)
    pipelined = self._gaps(pipelined=True, staged=False)
    staged = self._gaps(pipelined=True, staged=True)
    msg = f"today={today} pipelined={pipelined} staged={staged}"
    print(f"BOUNDARY_GAP_UNITS {msg}")
    self.assertLen(today, 1, msg)
    # Positive control: the double reproduces today's boundary.
    self.assertAlmostEqual(today[0], PRED_TODAY, delta=TOLERANCE, msg=msg)
    self.assertAlmostEqual(
        pipelined[0], PRED_PIPELINED, delta=TOLERANCE, msg=msg
    )
    self.assertAlmostEqual(staged[0], PRED_STAGED, delta=TOLERANCE, msg=msg)
    # What each mode removes: bookkeeping + packing, then also the transfer.
    self.assertAlmostEqual(
        today[0] - pipelined[0],
        PRED_TODAY - PRED_PIPELINED,
        delta=DIFF_TOLERANCE,
        msg=msg,
    )
    self.assertAlmostEqual(
        today[0] - staged[0],
        PRED_TODAY - PRED_STAGED,
        delta=DIFF_TOLERANCE,
        msg=msg,
    )


class _WaitsForTheUpdateBeforeStaging(rl_program.StandardRLProgram):
  """WRONG: stages after a last call that waited for the update.

  What hooking the stage in after google/tunix#2719's fused call, without
  having that call return at the update's dispatch, would do once the trainer
  can confirm its update: the stage then starts on an idle device.
  """

  async def _timed_train_step(self, batch, **kwargs):
    kwargs.pop("wait_for_update", None)
    return await super()._timed_train_step(batch, **kwargs)


# PRE-REGISTERED, written before the first run. Against a token trainer the
# fused call returns once its update landed and get_metrics moves past the
# sync and commit: the gap is reordered, not changed, so today and pipelined
# are as above. Staged, the last call returns at the update's dispatch and
# the stage confirms the update, so the transfer hides as before.
PRED_TODAY_TOKEN = PRED_TODAY  # 22.8
PRED_PIPELINED_TOKEN = PRED_PIPELINED  # 16.1
PRED_STAGED_TOKEN = PRED_STAGED  # 6.6
# Control (`_WaitsForTheUpdateBeforeStaging`): the stage starts on an idle
# device, so its transfer is exposed again (= pipelined) and the latch, seeing
# no device wait, switches staging off.
PRED_STAGED_AFTER_A_WAIT = PRED_PIPELINED  # 16.1


class TokenBoundaryTimingTest(absltest.TestCase):
  """The same gaps against a trainer that confirms its update."""

  def test_gaps_match_the_prediction_and_the_control_loses_the_staging(self):
    today, _ = _boundary_gaps(
        pipelined=False, staged=False, trainer_cls=_TokenTrainer
    )
    pipelined, _ = _boundary_gaps(
        pipelined=True, staged=False, trainer_cls=_TokenTrainer
    )
    staged, staged_run = _boundary_gaps(
        pipelined=True, staged=True, trainer_cls=_TokenTrainer
    )
    after_wait, after_wait_run = _boundary_gaps(
        pipelined=True,
        staged=True,
        trainer_cls=_TokenTrainer,
        program_cls=_WaitsForTheUpdateBeforeStaging,
    )
    msg = (
        f"today={today} pipelined={pipelined} staged={staged}"
        f" stage_after_a_wait={after_wait}"
        f" device_waits={staged_run.trainer.device_waits}"
        f"/{after_wait_run.trainer.device_waits}"
    )
    print(f"TOKEN_BOUNDARY_GAP_UNITS {msg}")
    self.assertLen(today, 1, msg)
    self.assertAlmostEqual(today[0], PRED_TODAY_TOKEN, delta=TOLERANCE, msg=msg)
    self.assertAlmostEqual(
        pipelined[0], PRED_PIPELINED_TOKEN, delta=TOLERANCE, msg=msg
    )
    self.assertAlmostEqual(
        staged[0], PRED_STAGED_TOKEN, delta=TOLERANCE, msg=msg
    )
    self.assertAlmostEqual(
        after_wait[0], PRED_STAGED_AFTER_A_WAIT, delta=TOLERANCE, msg=msg
    )
    # Both staged runs did stage; only the control found the device idle.
    self.assertEqual(_stages(staged_run), 1, msg)
    self.assertEqual(_stages(after_wait_run), 1, msg)
    self.assertFalse(staged_run.program._staging_latched_off, msg)
    self.assertTrue(after_wait_run.program._staging_latched_off, msg)


if __name__ == "__main__":
  absltest.main()
