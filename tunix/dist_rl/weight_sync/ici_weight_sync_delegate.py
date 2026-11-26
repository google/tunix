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

"""Worker-side endpoints of the ICI weight transport.

`IciSourceSynchronizer` is the trainer side and `IciWeightSyncDelegate` the
rollout side. They mirror `RaidenSynchronizer` and `RaidenWeightSyncDelegate`
closely enough to be swapped in where those are used, and add the two calls
`IciHandler` fans out to every participant: `check` (validate, no
collectives) and `transfer` (enter the collectives).

Both sides name leaves with `weight_utils.param_key`, as the Raiden path
does, so the trainer's converted state and the sampler's live state pair up
by name.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
import threading
from typing import Any, TypeVar

from absl import logging
import jax
from tunix.dist_rl.common import env_utils
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.weight_sync import ici_transport
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.weight_sync import weight_utils

_T = TypeVar("_T")


def _named_arrays(state: Any, label: str) -> dict[str, Any]:
  """Returns `{canonical key: array}` for transferable weights of `state`."""
  names, arrays = weight_utils.filter_bindable(
      *weight_utils.flatten_weights(state), label=label, strict=True
  )
  keyed: dict[str, Any] = {}
  origin: dict[str, str] = {}
  for name, array in zip(names, arrays, strict=True):
    key = weight_utils.param_key(name)
    if not key:
      raise ValueError(f"{label}: leaf {name!r} has an empty canonical key")
    if key in keyed:
      raise ValueError(
          f"{label}: leaves {origin[key]!r} and {name!r} both canonicalise to"
          f" {key!r}"
      )
    keyed[key] = array
    origin[key] = name
  if not keyed:
    raise ValueError(f"{label}: state has no transferable weights")
  return keyed


def _uniform_process_indices(
    arrays: Mapping[str, Any], label: str
) -> tuple[int, ...]:
  """Returns sorted process indices shared identically by every leaf."""
  expected: frozenset[int] | None = None
  expected_name = ""
  for name, array in arrays.items():
    sharding = getattr(array, "sharding", None)
    if sharding is None or not hasattr(sharding, "device_set"):
      raise ValueError(
          f"{label}: leaf {name!r} has no valid JAX sharding: {sharding!r}"
      )
    procs = frozenset(int(d.process_index) for d in sharding.device_set)
    if expected is None:
      expected = procs
      expected_name = name
    elif procs != expected:
      raise ValueError(
          f"{label}: leaf {name!r} spans processes {sorted(procs)}, which"
          f" differs from leaf {expected_name!r} on {sorted(expected)}"
      )
  if not expected:
    raise ValueError(f"{label}: no device processes found across weights")
  return tuple(sorted(expected))


def _work_unit_metadata(
    unit: weight_sync.WorkUnitId,
    arrays: Mapping[str, Any],
    process_indices: Sequence[int],
    checksums: Mapping[str, float | int] | None = None,
) -> weight_sync.WorkUnitMetadata:
  """Builds wire-safe `WorkUnitMetadata` describing `arrays`."""
  variables = tuple(
      weight_sync.TensorMetadata(
          name=name,
          shape=tuple(array.shape),
          mesh_shape=(1,) * array.ndim,
          layout=tuple(reversed(range(array.ndim))),
          item_size=array.dtype.itemsize,
          layer_idx=index,
          dtype=str(array.dtype),
      )
      for index, (name, array) in enumerate(sorted(arrays.items()))
  )
  return weight_sync.WorkUnitMetadata(
      unit=unit,
      variables=variables,
      transport_mode=ici_handler.TRANSPORT_MODE,
      use_ffi=False,
      process_indices=tuple(process_indices),
      devices_per_process=len(ici_transport.local_devices()),
      checksums=dict(checksums) if checksums is not None else None,
  )


def _full_checksums(arrays: Mapping[str, Any]) -> dict[str, float | int]:
  """Returns the checksums of every array in `arrays`, by canonical key."""
  names = sorted(arrays)
  return weight_utils.compute_checksums(
      names, [arrays[n] for n in names], sample=None, canonical_keys=True
  )


def _source_checksums(sync_request: Any) -> dict[str, float | int]:
  """Returns the one checksum dict every source unit of the round published.

  Raises:
    RuntimeError: If no source published checksums or two sources disagree.
  """
  raw = getattr(sync_request, "source_metadata", None) or ()
  entries = (
      [raw] if isinstance(raw, (weight_sync.WorkUnitMetadata, dict)) else raw
  )
  published = [
      dict(meta.checksums)
      for meta in (weight_sync.WorkUnitMetadata.from_dict(e) for e in entries)
      if meta.checksums is not None
  ]
  if not published:
    raise RuntimeError(
        "VERIFY_WEIGHTS is set but no source published weight checksums; set"
        " VERIFY_WEIGHTS on the trainer too"
    )
  if any(p != published[0] for p in published[1:]):
    raise RuntimeError("source units published conflicting weight checksums")
  return published[0]


def _report() -> dict[str, Any]:
  """Returns this worker's process index and JAX world view for `check`."""
  return {
      "process": int(jax.process_index()),
      "world": sorted({int(d.process_index) for d in jax.devices()}),
  }


class IciSourceSynchronizer:
  """Trainer-side ICI endpoint that publishes and sends converted weights."""

  def __init__(self, job_name: str):
    """Initializes the synchronizer for `job_name`."""
    self.job_name = job_name
    self._arrays: dict[str, Any] = {}
    self._metrics: dict[str, float] = {}

  @property
  def bound(self) -> bool:
    """Whether converted weight arrays are currently bound for a sync round."""
    return bool(self._arrays)

  def bind(self, state: Any) -> None:
    """Binds this round's weights, already converted to the rollout layout."""
    self._arrays = _named_arrays(state, f"ici[{self.job_name}]")
    logging.info(
        "%s bound %d arrays for ICI weight sync",
        self.job_name,
        len(self._arrays),
    )

  def d2h(self, sync_request: Any = None) -> None:
    """No-op: ICI transfers keep weights on device without host staging."""
    del sync_request

  def checksums(self, sample: int = 3) -> Mapping[str, float | int]:
    """Returns sample L1 checksums of the currently bound arrays."""
    names = sorted(self._arrays)
    return weight_utils.compute_checksums(
        names, [self._arrays[n] for n in names], sample=sample
    )

  def metrics(self) -> Mapping[str, float]:
    """Returns timing and byte-count metrics from the last ICI transfer."""
    return dict(self._metrics)

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    """Returns wire-safe metadata for the currently bound source weights."""
    if not self.bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before metadata")
    processes = _uniform_process_indices(self._arrays, f"ici[{self.job_name}]")
    return _work_unit_metadata(
        weight_sync.WorkUnitId(job_name=self.job_name),
        self._arrays,
        processes,
        # Destinations compare every tensor against these after the transfer.
        _full_checksums(self._arrays)
        if env_utils.parse_env_bool("VERIFY_WEIGHTS")
        else None,
    )

  def check(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reports this process and validates its part of `request["plan"]`."""
    plan = request["plan"]
    if plan is not None:
      ici_transport.check_plan(plan, source=self._source_for(plan))
    return _report()

  def transfer(self, request: Mapping[str, Any]) -> Mapping[str, float]:
    """Executes the ICI transfer collectives as a source or bystander."""
    plan = request["plan"]
    self._metrics = dict(
        ici_transport.execute(plan, source=self._source_for(plan))
    )
    return self._metrics

  def _source_for(
      self, plan: ici_transport.TransferPlan
  ) -> Mapping[str, Any] | None:
    # A trainer process outside this round's source role joins as a bystander.
    if jax.process_index() in plan.source_processes:
      return self._arrays
    return None

  def release(self, sync_request: Any = None) -> None:
    """Drops this round's references so converted copies can be freed."""
    del sync_request
    self._arrays = {}

  def abort(self, sync_request: Any = None) -> None:
    """Releases bound references on an aborted sync round."""
    self.release(sync_request)

  def close(self) -> None:
    """Releases bound references on teardown."""
    self.release()


class _SamplerSink:
  """Replaces serving weights in a sampler chunk-by-chunk as they arrive."""

  def __init__(
      self,
      job_name: str,
      sampler: Any,
      live: Mapping[str, Any],
  ):
    self._job_name = job_name
    self._sampler = sampler
    self._live = dict(live)
    self._shardings = {
        name: array.sharding.with_memory_kind("device")
        for name, array in live.items()
    }
    self.touched = False

  def target_sharding(self, name: str) -> jax.sharding.Sharding:
    try:
      return self._shardings[name]
    except KeyError:
      raise ValueError(
          f"{self._job_name}: sampler has no weight named {name!r}"
      ) from None

  def release(self, names: Sequence[str]) -> None:
    self.touched = True
    weight_utils.delete_array_buffers([self._live[name] for name in names])

  def install(self, arrays: Mapping[str, jax.Array]) -> None:
    self.touched = True
    names = list(arrays)
    values = [arrays[n] for n in names]
    weight_utils.apply_to_sampler(
        self._job_name,
        names,
        values,
        self._sampler,
        delete_replaced_buffers=False,
    )


class IciWeightSyncDelegate:
  """Rollout-side endpoint of an ICI weight sync round.

  Each chunk's serving leaves are freed right before their replacements arrive
  so peak device memory is one copy of weights plus one chunk. A round that
  fails after the transfer started leaves the sampler corrupted and fails fast:
  `abort_weight_sync` and subsequent `check` calls raise until the worker is
  restarted.
  """

  def __init__(
      self,
      *args: Any,
      server_id: str = "rollout",
      partial_rollout: bool = False,
      free_kv_cache: bool = True,
      **kwargs: Any,
  ):
    """Initializes the rollout ICI weight sync delegate.

    Args:
      *args: Unused positional arguments for constructor compatibility.
      server_id: Logical identifier for this rollout server.
      partial_rollout: Whether partial rollout mode is active; the KV cache
        is kept across the sync.
      free_kv_cache: Without partial rollout, whether to free the KV cache
        around the sync (`rollout_vllm_free_kv_cache_during_weight_sync`)
        instead of only resetting its prefix entries.
      **kwargs: Unused keyword arguments for constructor compatibility.
    """
    del args, kwargs
    self._server_id = server_id
    self._sampler: Any = None
    self.partial_rollout = bool(partial_rollout)
    self._free_kv_cache = bool(free_kv_cache)
    self._version = 0
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    # Serializes phase bodies and tracker transitions: a timed-out phase or
    # transfer can still be running when an abort for the same round arrives.
    self._lock = threading.Lock()
    self._sink: _SamplerSink | None = None
    self._transferred_generation: int | None = None
    self._transfer_completed = False
    self._serving_corrupted = False
    self._cache_deleted = False

  def is_bounded(self) -> bool:
    """Whether `bind_weight_sync` has attached a sampler."""
    return self._sampler is not None

  async def bind_weight_sync(
      self,
      sync_request: Any = None,
      state: Any = None,
      sampler: Any = None,
      **kwargs: Any,
  ) -> bool:
    """Attaches `sampler`; live weights are re-read from it on every round."""
    del sync_request, state, kwargs
    if sampler is not None:
      self._sampler = sampler
    return True

  def _live_arrays(self) -> dict[str, Any]:
    if self._sampler is None:
      raise RuntimeError("bind_weight_sync must run before ICI weight sync")
    arrays = _named_arrays(
        self._sampler.transformer_state, f"ici[{self._server_id}]"
    )
    me = int(jax.process_index())
    procs = _uniform_process_indices(arrays, f"ici[{self._server_id}]")
    if procs != (me,):
      raise ValueError(
          f"ici[{self._server_id}]: rollout weights span processes"
          f" {list(procs)}; an ICI destination must stay local to process {me}"
      )
    return arrays

  async def get_weight_sync_metadata(
      self, **kwargs: Any
  ) -> Sequence[weight_sync.WorkUnitMetadata]:
    """Returns wire-safe ICI work-unit metadata for this rollout replica."""
    del kwargs
    return [
        _work_unit_metadata(
            weight_sync.WorkUnitId(job_name=self._server_id),
            self._live_arrays(),
            [int(jax.process_index())],
        )
    ]

  def _corrupted_error(self) -> RuntimeError:
    return RuntimeError(
        f"{self._server_id}: the ICI transfer already replaced serving weights"
        " in place (drop_old_weights); this round cannot be rolled back and"
        " the worker must be restarted"
    )

  def _run_phase_locked(
      self,
      sync_request: Any,
      phase: str,
      body: Callable[[], _T],
      noop_return: _T,
  ) -> _T:
    with self._lock:
      has_round = weight_sync_coordinator.has_round(sync_request)
      if has_round and not self._tracker.admit(sync_request, phase):
        return noop_return
      result = body()
      if has_round:
        self._tracker.complete(sync_request, phase)
      return result

  async def _run_tracked_phase(
      self,
      sync_request: Any,
      phase: str,
      body: Callable[[], _T],
      noop_return: _T,
  ) -> _T:
    return await asyncio.to_thread(
        self._run_phase_locked, sync_request, phase, body, noop_return
    )

  def _pre_sync_body(self) -> bool:
    if self._sampler is None:
      raise RuntimeError("Sampler is not available for weight sync")
    if not self.partial_rollout:
      if self._free_kv_cache:
        self._sampler.delete_cache()
        self._cache_deleted = True
      else:
        self._sampler.reset_prefix_cache()
    jax.effects_barrier()
    return True

  async def pre_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> bool:
    """Frees or resets the KV cache before the weights change."""
    del kwargs
    return await self._run_tracked_phase(
        sync_request,
        "prepared",
        self._pre_sync_body,
        noop_return=True,
    )

  def check(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validates this worker's part of the plan (no collectives)."""
    plan = request["plan"]
    with self._lock:
      if self._serving_corrupted:
        raise self._corrupted_error()
      if plan is None:
        return _report()
      self._sink = None
      self._transferred_generation = None
      self._transfer_completed = False
      if jax.process_index() not in plan.destination_processes:
        # In the world but not receiving this round (e.g. a replica that is
        # not ACTIVE yet): joins the collectives with a throwaway payload.
        ici_transport.check_plan(plan)
        return _report()
      sink = _SamplerSink(
          self._server_id,
          self._sampler,
          self._live_arrays(),
      )
      ici_transport.check_plan(plan, sink=sink)
      self._sink = sink
    return _report()

  def transfer(self, request: Mapping[str, Any]) -> Mapping[str, float]:
    """Enters the collectives and receives weights if this is a destination."""
    plan = request["plan"]
    with self._lock:
      if self._serving_corrupted:
        raise self._corrupted_error()
      if jax.process_index() not in plan.destination_processes:
        return ici_transport.execute(plan)
      if self._sink is None:
        raise RuntimeError("ici_check must run before ici_transfer")
      try:
        timings = ici_transport.execute(plan, sink=self._sink)
      finally:
        if self._sink.touched:
          self._serving_corrupted = True
      self._transferred_generation = request["generation"]
      self._transfer_completed = True
    return timings

  def _validate_and_verify_transfer(self, sync_request: Any) -> int:
    sink = self._sink
    if sink is None or not self._transfer_completed:
      raise RuntimeError(
          f"{self._server_id}: weight_sync without a completed ICI transfer"
      )
    if weight_sync_coordinator.has_round(sync_request):
      # A tracked round always carries the coordinator's uuid; a missing one
      # (on either side) must not install an unverified transfer.
      expected = sync_request.extra_config.get("uuid")
      held = self._transferred_generation
      if expected is None or held is None or int(expected) != int(held):
        raise RuntimeError(
            f"{self._server_id}: holding the transfer of round {held}, asked"
            f" to install round {expected}"
        )
    if env_utils.parse_env_bool("VERIFY_WEIGHTS"):
      dst_checksums = _full_checksums(self._live_arrays())
      weight_utils.verify_weight_checksums(
          _source_checksums(sync_request), dst_checksums
      )
      logging.info(
          "%s verified %d tensors against the source checksums",
          self._server_id,
          dst_checksums["__tensor_count__"],
      )
    version = getattr(sync_request, "policy_version", None)
    self._version = (
        int(version) if version is not None else self._version + 1
    )
    return self._version

  async def weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> int:
    """Validates the received transfer before cache rebuild."""
    del kwargs
    return await self._run_tracked_phase(
        sync_request,
        "h2d_done",
        lambda: self._validate_and_verify_transfer(sync_request),
        noop_return=self._version,
    )

  def _commit_and_rebuild_cache(self) -> bool:
    if self._sampler is None:
      raise RuntimeError("Sampler is not available for weight sync")
    if self._cache_deleted:
      self._sampler.reinitialize_cache()
      self._cache_deleted = False
    else:
      self._sampler.refresh_state_leaves()
    self._sink = None
    self._transferred_generation = None
    self._transfer_completed = False
    self._serving_corrupted = False
    return True

  async def post_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> bool:
    """Refreshes sampler state leaves and rebuilds the KV cache."""
    del kwargs
    return await self._run_tracked_phase(
        sync_request,
        "committed",
        self._commit_and_rebuild_cache,
        noop_return=True,
    )

  def _abort_body(self) -> bool:
    sink, self._sink = self._sink, None
    self._transferred_generation = None
    self._transfer_completed = False
    corrupted = self._serving_corrupted or (
        sink is not None and sink.touched
    )
    if corrupted:
      self._serving_corrupted = True
      raise self._corrupted_error()
    if self._cache_deleted and self._sampler is not None:
      self._sampler.reinitialize_cache()
      self._cache_deleted = False
    return True

  async def abort_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> bool:
    """Discards staged weights and restores KV cache; refuses if corrupted."""
    del kwargs
    return await self._run_tracked_phase(
        sync_request,
        "aborted",
        self._abort_body,
        noop_return=False,
    )

  def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Returns the worker round tracker's status snapshot."""
    return self._tracker.report()
