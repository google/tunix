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

"""TorchTitan Trainer Worker and Rank-0 Follower Loop."""

from __future__ import annotations

from collections.abc import Mapping
import concurrent.futures
import contextlib
import json
import logging
import shutil
import tempfile
import threading
import time
from typing import Any

import cloudpickle
from etils import epath
import numpy as np
import torch
import torch.distributed as dist
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import basic_handler
from tunix.dist_rl.worker import abstract_worker
from tunix.experimental.torchtitan.engine import TorchTitanEngine
from tunix.rl import algorithm_config

WorkerState = datatypes.WorkerState


def slice_payload_for_rank(
    payload: datatypes.RLTrainerPayload,
    rank: int,
    world_size: int,
) -> datatypes.RLTrainerPayload:
  """Slices batch axis 0 of RLTrainerPayload for a specific DP rank."""
  batch_size = payload.prompt_ids.shape[0]
  if batch_size % world_size != 0:
    raise ValueError(
        f"Batch size {batch_size} is not divisible by world_size {world_size}."
    )

  chunk_size = batch_size // world_size
  start = rank * chunk_size
  end = (rank + 1) * chunk_size

  def _slice_required(arr: np.ndarray) -> np.ndarray:
    return arr[start:end]

  def _slice_optional(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
      return None
    return arr[start:end]

  return datatypes.RLTrainerPayload(
      prompt_ids=_slice_required(payload.prompt_ids),
      prompt_mask=_slice_required(payload.prompt_mask),
      completion_ids=_slice_required(payload.completion_ids),
      completion_mask=_slice_required(payload.completion_mask),
      advantages=_slice_required(payload.advantages),
      segment_ids=_slice_optional(payload.segment_ids),
      segment_positions=_slice_optional(payload.segment_positions),
      ref_per_token_logps=_slice_optional(payload.ref_per_token_logps),
      old_per_token_logps=_slice_optional(payload.old_per_token_logps),
      sampler_is_weights=_slice_optional(payload.sampler_is_weights),
      routed_experts=_slice_optional(payload.routed_experts),
      returns=_slice_optional(payload.returns),
      old_values=_slice_optional(payload.old_values),
      num_segments=payload.num_segments,
      sampler_agreement_applied=payload.sampler_agreement_applied,
      metadata=payload.metadata,
  )


def tensors_to_raw_payloads(
    hf_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, basic_handler.RawTensorPayload]:
  """Converts a dict of CPU tensors into RawTensorPayloads."""
  payloads: dict[str, basic_handler.RawTensorPayload] = {}
  for name, tensor in hf_state_dict.items():
    t = tensor.contiguous()
    if t.dtype == torch.bfloat16:
      payloads[name] = basic_handler.RawTensorPayload(
          buffer=t.view(torch.uint16).numpy(),
          shape=tuple(t.shape),
          dtype="bf16",
      )
    else:
      arr = t.numpy()
      payloads[name] = basic_handler.RawTensorPayload(
          buffer=arr,
          shape=tuple(t.shape),
          dtype=str(arr.dtype),
      )
  return payloads


def _copy_tree_parallel(
    src_dir: epath.Path,
    dst_dir: epath.Path,
    *,
    exclude_names: frozenset[str] = frozenset(),
) -> None:
  """Recursively copies files from src_dir to dst_dir in parallel using epath streaming."""
  dst_dir.mkdir(parents=True, exist_ok=True)
  copy_tasks: list[tuple[epath.Path, epath.Path]] = []

  def _collect(s: epath.Path, d: epath.Path, is_top: bool) -> None:
    for entry in s.iterdir():
      if is_top and entry.name in exclude_names:
        continue
      target = d / entry.name
      if entry.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        _collect(entry, target, False)
      else:
        copy_tasks.append((entry, target))

  _collect(src_dir, dst_dir, True)
  if not copy_tasks:
    return

  def _do_copy(task: tuple[epath.Path, epath.Path]) -> None:
    src, dst = task
    src.copy(dst, overwrite=True)

  max_workers = min(8, len(copy_tasks))
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    list(executor.map(_do_copy, copy_tasks))


def _remove_tree_epath(dir_path: epath.Path) -> None:
  """Recursively deletes a directory tree via epath, unlinking dotfiles first for remote filesystems."""
  if not dir_path.exists():
    return
  for child in dir_path.iterdir():
    if child.is_dir():
      _remove_tree_epath(child)
    else:
      child.unlink(missing_ok=True)
  dir_path.rmtree(missing_ok=True)


def _prune_old_checkpoints(
    base_path: epath.Path, *, max_to_keep: int | None, current_step: int
) -> None:
  """Prunes stale uncommitted directories and committed checkpoints beyond max_to_keep."""
  if not base_path.exists():
    return
  committed: list[tuple[int, epath.Path]] = []
  uncommitted: list[tuple[int, epath.Path]] = []
  for d in base_path.iterdir():
    if not d.is_dir() or not d.name.startswith("step_"):
      continue
    try:
      step_idx = int(d.name.split("_", 1)[1])
    except (IndexError, ValueError):
      continue
    if (d / "metadata.json").exists() and (d / ".metadata").exists():
      committed.append((step_idx, d))
    elif step_idx < current_step:
      uncommitted.append((step_idx, d))

  for _, d in uncommitted:
    _remove_tree_epath(d)

  if (
      max_to_keep is not None
      and max_to_keep > 0
      and len(committed) > max_to_keep
  ):
    committed.sort(key=lambda x: x[0])
    for _, d in committed[:-max_to_keep]:
      _remove_tree_epath(d)


class TorchTitanTrainerWorker(abstract_worker.Worker):
  """Trainer worker running on Rank 0 of a multi-process TorchTitan cluster."""

  def __init__(
      self,
      engine: TorchTitanEngine,
      control_pg: dist.ProcessGroup | None = None,
      dp_pg: dist.ProcessGroup | None = None,
      worker_id: str = "trainer_0",
      weight_sync_exporter: basic_handler.BasicSourceExporter | None = None,
  ):
    super().__init__()
    self._engine = engine
    self._control_pg = control_pg
    self._dp_pg = dp_pg
    self._worker_id = worker_id
    self._state = WorkerState.PENDING
    self._step_lock = threading.Lock()
    self._spmd_broken = False
    self._weight_sync_exporter = weight_sync_exporter
    self._latest_metrics: dict[str, float] = {}

  @property
  def worker_id(self) -> str:
    return self._worker_id

  @abstract_worker.Worker.state.getter
  def state(self) -> WorkerState:
    return self._state

  @state.setter
  def state(self, new_state: WorkerState) -> None:
    self._state = new_state

  def _policy_version(self) -> int:
    return self._engine.policy_version

  def _check_spmd_healthy(self) -> None:
    if self._spmd_broken:
      raise RuntimeError(
          "TorchTitan worker SPMD process group is broken or desynchronized"
          " from an earlier collective failure or abort; restart the worker."
      )

  @contextlib.contextmanager
  def _spmd_collective(self):
    try:
      yield
    except Exception:
      self._spmd_broken = True
      self.state = WorkerState.ERROR
      raise

  def _response(self, **metadata: Any) -> datatypes.Response:
    return datatypes.Response(
        metadata={
            "worker_id": self._worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version(),
            **metadata,
        }
    )

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        instance_id=self.instance_id,
        roles=frozenset({"trainer", "actor", "weight_sync"}),
        resources={
            "trainer": "TorchTitanEngine",
            "policy_version": self._policy_version(),
        },
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version(),
    )

  @abstract_worker.offload_to_thread
  def start(self) -> datatypes.Response:
    self._state = WorkerState.READY
    return self._response(started=True)

  @abstract_worker.offload_to_thread
  def initialize(self) -> datatypes.Response:
    self._state = WorkerState.READY
    return self._response(initialized=True, ready=True)

  @abstract_worker.offload_to_thread
  def set_target_state(self, target_state: Any) -> datatypes.Response:
    """Accepts rollout target-state skeleton for contract parity."""
    del target_state
    return self._response(target_state_configured=True)

  @abstract_worker.offload_to_thread
  def with_loss_fn(
      self,
      loss_fn: str | algorithm_config.PolicyLossFn,
      *,
      algo_config: algorithm_config.AlgorithmConfig,
      role: datatypes.Role = datatypes.Role.ACTOR,
      pad_id: int = 0,
      eos_id: int = 0,
      fuse_agreement_in_loss: bool = False,
      return_routed_experts: bool = False,
  ) -> datatypes.Response:
    """Validates and installs a named or custom PyTorch policy loss function."""
    with self._step_lock:
      self._check_spmd_healthy()
      self._engine.with_loss_fn(
          loss_fn,
          algo_config=algo_config,
          role=role,
          pad_id=pad_id,
          eos_id=eos_id,
          fuse_agreement_in_loss=fuse_agreement_in_loss,
          return_routed_experts=return_routed_experts,
      )
      loss_wire = (
          ("name", loss_fn)
          if isinstance(loss_fn, str)
          else ("bytes", cloudpickle.dumps(loss_fn))
      )
      if self._control_pg is not None:
        with self._spmd_collective():
          dist.broadcast_object_list(
              [[
                  "with_loss_fn",
                  loss_wire,
                  algo_config,
                  role,
                  pad_id,
                  eos_id,
                  fuse_agreement_in_loss,
                  return_routed_experts,
              ]],
              src=0,
              group=self._control_pg,
          )
      return self._response(loss_fn_configured=True)

  @abstract_worker.offload_to_thread
  def compile(self, dummy_payload: Any = None) -> datatypes.Response:
    with self._step_lock:
      self._check_spmd_healthy()
      with self._spmd_collective():
        if self._control_pg is not None:
          dist.broadcast_object_list(
              [["compile", dummy_payload]], src=0, group=self._control_pg
          )
        self._engine.compile(dummy_payload)
      return self._response(compiled=True)

  @abstract_worker.offload_to_thread
  def fwd_bwd(
      self,
      request: datatypes.TrainRequest,
  ) -> datatypes.Response:
    with self._step_lock:
      self._check_spmd_healthy()
      payload = request.payload
      if not isinstance(payload, datatypes.RLTrainerPayload):
        raise TypeError(
            "TorchTitanTrainerWorker requires RLTrainerPayload, got"
            f" {type(payload)}"
        )
      if np.any((payload.prompt_mask > 0).sum(axis=1) <= 0):
        raise ValueError(
            "Every RL prompt must have >= 1 valid token; found row with"
            " prompt_len <= 0."
        )
      world_size = (
          dist.get_world_size(self._dp_pg) if self._dp_pg is not None else 1
      )
      if self._control_pg is not None and world_size > 1:
        slices = [
            slice_payload_for_rank(payload, r, world_size)
            for r in range(world_size)
        ]
        local_payload = slices[0]
      else:
        slices = None
        local_payload = payload

      with self._spmd_collective():
        if slices is not None:
          dist.broadcast_object_list(
              [["fwd_bwd", slices]], src=0, group=self._control_pg
          )
        metrics = self._engine.fwd_bwd(local_payload, dp_group=self._dp_pg)
      meta = dict(request.metadata) if request.metadata else {}
      resp = self._response(queued=True, fwd_bwd=True, metrics=metrics, **meta)
      resp.request_id = request.request_id
      return resp

  @abstract_worker.offload_to_thread
  def update(self) -> tuple[int, dict[str, float]]:
    with self._step_lock:
      self._check_spmd_healthy()
      with self._spmd_collective():
        if self._control_pg is not None:
          dist.broadcast_object_list(
              [["update"]], src=0, group=self._control_pg
          )
        train_step, metrics = self._engine.update(dp_group=self._dp_pg)
      self._latest_metrics = dict(metrics)
      return train_step, metrics

  def get_metrics(self) -> dict[str, float]:
    m = self._latest_metrics
    self._latest_metrics = {}
    return m

  @abstract_worker.offload_to_thread
  def per_token_logps(
      self,
      items: datatypes.LogprobsRequest,
      **kwargs: Any,
  ) -> datatypes.LogprobsResponse:
    del kwargs
    if items.pad_id is None:
      raise ValueError("LogprobsRequest requires pad_id.")
    if items.eos_id is None:
      raise ValueError("LogprobsRequest requires eos_id.")
    prompt = np.asarray(items.prompt_tokens)
    completion = np.asarray(items.completion_tokens)
    batch_size = prompt.shape[0]
    if batch_size == 0:
      return datatypes.LogprobsResponse(
          request_id=items.request_id,
          per_token_logps=np.zeros(
              (0, completion.shape[1] if completion.ndim > 1 else 0),
              dtype=np.float32,
          ),
          model_version=self._policy_version(),
      )

    pad_id = int(items.pad_id)
    if np.any((prompt != pad_id).sum(axis=1) <= 0):
      raise ValueError(
          "Every RL prompt must have >= 1 valid token; found row with"
          " prompt_len <= 0."
      )
    temperature = (
        float(items.temperature) if items.temperature is not None else 1.0
    )

    world_size = (
        dist.get_world_size(self._dp_pg) if self._dp_pg is not None else 1
    )

    with self._step_lock:
      self._check_spmd_healthy()
      if self._control_pg is not None and world_size > 1:
        padded_bs = ((batch_size + world_size - 1) // world_size) * world_size
        per_rank = padded_bs // world_size
        if padded_bs > batch_size:
          pad_count = padded_bs - batch_size
          dummy_p = np.tile(prompt[:1], (pad_count, 1))
          dummy_c = np.tile(completion[:1], (pad_count, 1))
          padded_p = np.concatenate([prompt, dummy_p], axis=0)
          padded_c = np.concatenate([completion, dummy_c], axis=0)
        else:
          padded_p = prompt
          padded_c = completion

        slices_p = [
            padded_p[r * per_rank : (r + 1) * per_rank]
            for r in range(world_size)
        ]
        slices_c = [
            padded_c[r * per_rank : (r + 1) * per_rank]
            for r in range(world_size)
        ]

        with self._spmd_collective():
          dist.broadcast_object_list(
              [["per_token_logps", slices_p, slices_c, pad_id, temperature]],
              src=0,
              group=self._control_pg,
          )
          local_logps = self._engine.per_token_logps(
              prompt_tokens=slices_p[0],
              completion_tokens=slices_c[0],
              pad_id=pad_id,
              temperature=temperature,
          )
          gathered: list[Any] = [None] * world_size
          dist.gather_object(
              local_logps, gathered, dst=0, group=self._control_pg
          )
        all_logps = np.concatenate(gathered, axis=0)[:batch_size]
        return datatypes.LogprobsResponse(
            request_id=items.request_id,
            per_token_logps=all_logps,
            model_version=self._policy_version(),
        )

      with self._spmd_collective():
        result = self._engine.per_token_logps(
            prompt_tokens=prompt,
            completion_tokens=completion,
            pad_id=pad_id,
            temperature=temperature,
        )
      return datatypes.LogprobsResponse(
          request_id=items.request_id,
          per_token_logps=result,
          model_version=self._policy_version(),
      )

  @abstract_worker.offload_to_thread
  def save_checkpoint(
      self,
      metadata: Mapping[str, Any] | None = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> datatypes.Response:
    if step is not None:
      step_num = int(step)
    elif isinstance(metadata, Mapping) and "step" in metadata:
      step_num = int(metadata["step"])
    else:
      step_num = self._engine.train_step

    checkpoint_dir = self._engine.config.checkpoint_dir
    save_interval = self._engine.config.save_interval_steps
    if not force and (save_interval <= 0 or step_num % save_interval != 0):
      return self._response(checkpoint_saved=False, step=step_num)
    if not checkpoint_dir:
      return self._response(checkpoint_saved=False, step=step_num)

    with self._step_lock:
      self._check_spmd_healthy()
      local_dcp_dir = tempfile.mkdtemp(
          prefix=f"torchtitan_dcp_save_{step_num}_"
      )
      try:
        with self._spmd_collective():
          if self._control_pg is not None:
            dist.broadcast_object_list(
                [["save_checkpoint", local_dcp_dir]],
                src=0,
                group=self._control_pg,
            )
          lr_sched_sd = self._engine.save_dcp_checkpoint(
              local_dcp_dir, process_group=self._control_pg
          )
          if self._control_pg is not None:
            dist.barrier(group=self._control_pg)

        step_path = epath.Path(checkpoint_dir) / f"step_{step_num}"
        meta_file = step_path / "metadata.json"
        if step_path.exists():
          _remove_tree_epath(step_path)
        _copy_tree_parallel(epath.Path(local_dcp_dir), step_path)

        saved_meta: dict[str, Any] = (
            dict(metadata) if isinstance(metadata, Mapping) else {}
        )
        saved_meta["step"] = step_num
        if "policy_version" not in saved_meta:
          saved_meta["policy_version"] = self._policy_version()
        saved_meta["lr_scheduler_state"] = lr_sched_sd

        # Write metadata.json LAST as commit point
        meta_file.write_text(json.dumps(saved_meta))
        _prune_old_checkpoints(
            epath.Path(checkpoint_dir),
            max_to_keep=self._engine.config.max_to_keep,
            current_step=step_num,
        )
      finally:
        shutil.rmtree(local_dcp_dir, ignore_errors=True)

    return self._response(checkpoint_saved=True, step=step_num)

  @abstract_worker.offload_to_thread
  def restore_checkpoint(
      self, step: int | None = None
  ) -> dict[str, Any] | None:
    checkpoint_dir = self._engine.config.checkpoint_dir
    if not checkpoint_dir:
      return None

    base_path = epath.Path(checkpoint_dir)
    if not base_path.exists():
      return None

    if step is not None:
      step_path = base_path / f"step_{step}"
      if (
          not step_path.exists()
          or not (step_path / "metadata.json").exists()
          or not (step_path / ".metadata").exists()
      ):
        return None
    else:
      step_dirs = []
      for d in base_path.iterdir():
        if (
            d.is_dir()
            and d.name.startswith("step_")
            and (d / "metadata.json").exists()
            and (d / ".metadata").exists()
        ):
          try:
            step_dirs.append((int(d.name.split("_")[1]), d))
          except (IndexError, ValueError):
            pass
      if not step_dirs:
        return None
      step_dirs.sort(key=lambda x: x[0])
      step_path = step_dirs[-1][1]

    raw_meta = json.loads((step_path / "metadata.json").read_text())
    restored_opt_step = int(raw_meta["step"])
    restored_pv = int(raw_meta["policy_version"])
    lr_sched_sd = (
        raw_meta["lr_scheduler_state"]
        if "lr_scheduler_state" in raw_meta
        else None
    )
    public_meta = {
        k: v for k, v in raw_meta.items() if k != "lr_scheduler_state"
    }

    local_dcp_dir = tempfile.mkdtemp(prefix="torchtitan_dcp_restore_")
    try:
      _copy_tree_parallel(
          step_path,
          epath.Path(local_dcp_dir),
          exclude_names=frozenset({"metadata.json"}),
      )

      with self._step_lock:
        self._check_spmd_healthy()
        with self._spmd_collective():
          if self._control_pg is not None:
            dist.broadcast_object_list(
                [[
                    "restore_checkpoint",
                    local_dcp_dir,
                    restored_opt_step,
                    restored_pv,
                    lr_sched_sd,
                ]],
                src=0,
                group=self._control_pg,
            )
          self._engine.load_dcp_checkpoint(
              local_dcp_dir,
              train_step=restored_opt_step,
              policy_version=restored_pv,
              lr_scheduler_state=lr_sched_sd,
              process_group=self._control_pg,
          )
          if self._control_pg is not None:
            dist.barrier(group=self._control_pg)
    finally:
      shutil.rmtree(local_dcp_dir, ignore_errors=True)

    return public_meta

  @abstract_worker.offload_to_thread
  def prepare_weight_sync(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> Any:
    t0 = time.perf_counter()
    target_pv: int | None = (
        int(sync_request.policy_version) if sync_request is not None else None
    )
    with self._step_lock:
      self._check_spmd_healthy()
      self.state = WorkerState.SYNCING
      try:
        with self._spmd_collective():
          if self._control_pg is not None:
            dist.broadcast_object_list(
                [["prepare_weight_sync", target_pv]],
                src=0,
                group=self._control_pg,
            )
          hf_sd = self._engine.export_hf_state_dict(is_rank_zero=True)
          if target_pv is not None:
            self._engine.policy_version = target_pv
        raw_payloads = tensors_to_raw_payloads(hf_sd)
        elapsed = time.perf_counter() - t0
        num_tensors = len(raw_payloads)
        total_bytes = sum(p.buffer.nbytes for p in raw_payloads.values())
        logging.info(
            "[BASIC] exported %d tensors (%.2f MB) to CPU RAM in %.3fs"
            " (policy_version=%d)",
            num_tensors,
            total_bytes / (1024 * 1024),
            elapsed,
            self._policy_version(),
        )
        if self._weight_sync_exporter is None and sync_request is not None:
          self._weight_sync_exporter = basic_handler.BasicSourceExporter(
              job_name=self._worker_id
          )
        if self._weight_sync_exporter is not None and sync_request is not None:
          return self._weight_sync_exporter.prepare_weight_sync(
              sync_request, raw_payloads
          )
        return raw_payloads
      except Exception:
        self.state = WorkerState.ERROR
        raise

  @abstract_worker.offload_to_thread
  def release_weight_sync(self, sync_request: Any = None) -> None:
    with self._step_lock:
      if self._weight_sync_exporter is not None and sync_request is not None:
        self._weight_sync_exporter.release_weight_sync(sync_request)
      if self.state == WorkerState.SYNCING:
        self.state = WorkerState.READY

  @abstract_worker.offload_to_thread
  def stop(self, *, abort: bool = False) -> datatypes.Response:
    if self.state == WorkerState.STOPPED:
      return self._response(stopped=True, abort=abort)
    try:
      if abort:
        self._spmd_broken = True
        if self._weight_sync_exporter is not None:
          try:
            self._weight_sync_exporter.close()
          except Exception:  # pylint: disable=broad-exception-caught
            pass
          self._weight_sync_exporter = None
        self.state = WorkerState.STOPPED
        return self._response(stopped=True, abort=True)

      with self._step_lock:
        if self._control_pg is not None:
          try:
            dist.broadcast_object_list(
                [["stop"]], src=0, group=self._control_pg
            )
          except Exception:  # pylint: disable=broad-exception-caught
            pass
        if self._weight_sync_exporter is not None:
          self._weight_sync_exporter.close()
          self._weight_sync_exporter = None
        self.state = WorkerState.STOPPED
        return self._response(stopped=True, abort=False)
    finally:
      self._notify_stopped(abort)


def run_follower_loop(
    engine: TorchTitanEngine,
    control_pg: dist.ProcessGroup,
    dp_pg: dist.ProcessGroup,
    rank: int,
) -> None:
  """Follower loop running on rank 1..N-1 listening to Rank 0 commands."""
  while True:
    msg: list[Any] = [None]
    dist.broadcast_object_list(msg, src=0, group=control_pg)
    cmd_data = msg[0]

    if not isinstance(cmd_data, list) or not cmd_data:
      raise ValueError(f"Expected non-empty command list, got {cmd_data!r}")

    cmd = cmd_data[0]

    if cmd == "stop":
      break
    elif cmd == "with_loss_fn":
      (
          loss_wire,
          algo_config,
          role,
          pad_id,
          eos_id,
          fuse_agreement_in_loss,
          return_routed_experts,
      ) = cmd_data[1:]
      wire_kind, wire_val = loss_wire
      if wire_kind == "name":
        loss_fn = wire_val
      elif wire_kind == "bytes":
        loss_fn = cloudpickle.loads(  # pylint: disable=g-unsafe-pickle-load
            wire_val
        )
      else:
        raise ValueError(f"Unknown with_loss_fn wire kind: {wire_kind!r}")
      engine.with_loss_fn(
          loss_fn,
          algo_config=algo_config,
          role=role,
          pad_id=pad_id,
          eos_id=eos_id,
          fuse_agreement_in_loss=fuse_agreement_in_loss,
          return_routed_experts=return_routed_experts,
      )
    elif cmd == "compile":
      dummy_payload = cmd_data[1]
      engine.compile(dummy_payload)
    elif cmd == "fwd_bwd":
      slices = cmd_data[1]
      local_slice = slices[rank]
      engine.fwd_bwd(local_slice, dp_group=dp_pg)
    elif cmd == "update":
      engine.update(dp_group=dp_pg)
    elif cmd == "per_token_logps":
      slices_p, slices_c, pad_id, temperature = cmd_data[1:]
      local_logps = engine.per_token_logps(
          prompt_tokens=slices_p[rank],
          completion_tokens=slices_c[rank],
          pad_id=pad_id,
          temperature=temperature,
      )
      dist.gather_object(local_logps, None, dst=0, group=control_pg)
    elif cmd == "prepare_weight_sync":
      target_pv = cmd_data[1]
      engine.export_hf_state_dict(is_rank_zero=False)
      if target_pv is not None:
        engine.policy_version = target_pv
    elif cmd == "save_checkpoint":
      local_dcp_dir = cmd_data[1]
      engine.save_dcp_checkpoint(local_dcp_dir, process_group=control_pg)
      dist.barrier(group=control_pg)
    elif cmd == "restore_checkpoint":
      local_dcp_dir, restored_opt_step, restored_pv, lr_sched_sd = cmd_data[1:]
      engine.load_dcp_checkpoint(
          local_dcp_dir,
          train_step=restored_opt_step,
          policy_version=restored_pv,
          lr_scheduler_state=lr_sched_sd,
          process_group=control_pg,
      )
      dist.barrier(group=control_pg)
    else:
      raise ValueError(f"Unknown command received in follower loop: {cmd!r}")
