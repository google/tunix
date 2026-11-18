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

from collections.abc import Mapping, Sequence
import os
import threading
from typing import Any

from absl import logging
import jax
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.weight_sync import ici_transport
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.weight_sync import weight_utils

DROP_OLD_WEIGHTS_ENV = "ICI_DROP_OLD_WEIGHTS"


def _named_arrays(state: Any, label: str) -> dict[str, Any]:
  """`{canonical key: array}` for the weights of `state`."""
  names, arrays = weight_utils.filter_bindable(
      *weight_utils.flatten_weights(state), label=label
  )
  keyed: dict[str, Any] = {}
  origin: dict[str, str] = {}
  for name, array in zip(names, arrays):
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


def _work_unit_metadata(
    unit: weight_sync.WorkUnitId,
    arrays: Mapping[str, Any],
    process_indices: Sequence[int],
) -> weight_sync.WorkUnitMetadata:
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
  )


def _report() -> dict[str, Any]:
  """Where this worker sits in the JAX world; what `check` returns."""
  return {
      "process": int(jax.process_index()),
      "world": sorted({int(d.process_index) for d in jax.devices()}),
  }


class IciSourceSynchronizer:
  """Trainer side: publishes the converted weights and sends them."""

  def __init__(self, job_name: str = "trainer"):
    self.job_name = job_name
    self._arrays: dict[str, Any] = {}
    self._metrics: dict[str, Any] = {}

  @property
  def bound(self) -> bool:
    return bool(self._arrays)

  def bind(self, state: Any) -> None:
    """Takes this round's weights, already in the rollout's layout."""
    self._arrays = _named_arrays(state, f"ici[{self.job_name}]")
    logging.info(
        "%s bound %d arrays for ICI weight sync",
        self.job_name,
        len(self._arrays),
    )

  def d2h(self, sync_request: Any = None) -> None:
    """No-op: the weights never leave the devices."""
    del sync_request

  def checksums(self, sample: int = 3) -> dict[str, float | int]:
    names = sorted(self._arrays)
    return weight_utils.compute_checksums(
        names, [self._arrays[n] for n in names], sample=sample
    )

  def metrics(self) -> dict[str, Any]:
    return dict(self._metrics)

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    if not self.bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before metadata")
    processes = sorted({
        device.process_index
        for array in self._arrays.values()
        for device in array.sharding.device_set
    })
    return _work_unit_metadata(
        weight_sync.WorkUnitId(job_name=self.job_name), self._arrays, processes
    )

  def check(self, request: Mapping[str, Any]) -> dict[str, Any]:
    """Reports this process; validates its part of the plan when given one."""
    plan = request.get("plan")
    if plan is not None:
      ici_transport.check_plan(plan, source=self._source_for(plan))
    return _report()

  def transfer(self, request: Mapping[str, Any]) -> dict[str, float]:
    plan = request["plan"]
    self._metrics = ici_transport.execute(plan, source=self._source_for(plan))
    return self._metrics

  def _source_for(self, plan: ici_transport.TransferPlan) -> Any:
    # A trainer process outside this round's source role (none today, since a
    # trainer is one process) joins as a bystander.
    if jax.process_index() in plan.source_processes:
      return self._arrays
    return None

  def release(self, sync_request: Any = None) -> None:
    """Drops this round's references so converted copies can be freed."""
    del sync_request
    self._arrays = {}

  def abort(self, sync_request: Any = None) -> None:
    self.release(sync_request)

  def close(self) -> None:
    self.release()


class _SamplerSink:
  """Installs received chunks into a sampler, in place or staged."""

  def __init__(self, job_name: str, sampler: Any, live: Mapping[str, Any]):
    self._job_name = job_name
    self._sampler = sampler
    self._live = dict(live)
    self._shardings = {
        name: array.sharding.with_memory_kind("device")
        if hasattr(array.sharding, "with_memory_kind")
        else array.sharding
        for name, array in live.items()
    }
    self.drop_old = False
    self.staged: dict[str, Any] = {}
    self.touched = False

  def target_sharding(self, name: str) -> Any:
    try:
      return self._shardings[name]
    except KeyError:
      raise ValueError(
          f"{self._job_name}: sampler has no weight named {name!r}"
      ) from None

  def release(self, names: Sequence[str]) -> None:
    if not self.drop_old:
      return
    self.touched = True
    weight_utils.delete_array_buffers([self._live[name] for name in names])

  def install(self, arrays: Mapping[str, Any]) -> None:
    if self.drop_old:
      self.touched = True
      self.apply(arrays, delete_replaced=False)
    else:
      self.staged.update(arrays)

  def apply(self, arrays: Mapping[str, Any], *, delete_replaced: bool) -> None:
    names = list(arrays)
    values = [arrays[n] for n in names]
    # A vLLM sampler serves from its runner's state; a plain sampler from its
    # transformer state.
    runner = getattr(self._sampler, "model_runner", None)
    if runner is not None:
      weight_utils.apply_to_runner(
          self._job_name,
          names,
          values,
          runner,
          delete_replaced_buffers=delete_replaced,
      )
    else:
      weight_utils.apply_to_state(
          self._job_name,
          names,
          values,
          self._sampler.transformer_state,
          delete_replaced_buffers=delete_replaced,
      )

  def discard(self) -> None:
    weight_utils.delete_array_buffers(list(self.staged.values()))
    self.staged = {}


class IciWeightSyncDelegate:
  """Rollout side of an ICI weight sync round.

  Two install modes:

  * staged (default): received weights are held next to the serving ones and
    swapped in by `weight_sync()`. An aborted round leaves the sampler exactly
    as it was. Needs device memory for two copies of the weights.
  * drop-old (`drop_old_weights=True`): each chunk's serving leaves are freed
    right before their replacements arrive. Peak memory is one copy plus one
    chunk, which is what large models need, but a round that fails after the
    transfer started leaves the sampler without usable weights:
    `abort_weight_sync` raises and the worker needs a restart. This is the
    same exposure as the Raiden destination, which writes the serving copy in
    place.
  """

  def __init__(
      self,
      *args,
      server_id: str = "rollout",
      partial_rollout: bool = False,
      drop_old_weights: bool | None = None,
      **kwargs,
  ):
    del args, kwargs
    self._server_id = server_id
    self._sampler: Any = None
    self.partial_rollout = bool(partial_rollout)
    if drop_old_weights is None:
      drop_old_weights = (
          os.environ.get(DROP_OLD_WEIGHTS_ENV, "").lower() in ("1", "true")
      )
    self._drop_old = bool(drop_old_weights)
    self._version = 0
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    # Serializes phase bodies: a timed-out transfer can still be running when
    # the abort for the same round arrives.
    self._lock = threading.Lock()
    self._sink: _SamplerSink | None = None
    self._transferred_generation: int | None = None

  def is_bounded(self) -> bool:
    return self._sampler is not None

  async def bind_weight_sync(
      self,
      sync_request: Any = None,
      state: Any = None,
      sampler: Any = None,
      **kwargs,
  ) -> Any:
    """Remembers the sampler; the live state is re-read every round."""
    del sync_request, state, kwargs
    if sampler is not None:
      self._sampler = sampler
    return True

  def _live_arrays(self) -> dict[str, Any]:
    if self._sampler is None:
      raise RuntimeError("bind_weight_sync must run before ICI weight sync")
    return _named_arrays(
        self._sampler.transformer_state, f"ici[{self._server_id}]"
    )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    del kwargs
    return [
        _work_unit_metadata(
            weight_sync.WorkUnitId(job_name=self._server_id),
            self._live_arrays(),
            [jax.process_index()],
        )
    ]

  def _has_round(self, sync_request: Any) -> bool:
    extra = getattr(sync_request, "extra_config", None) or {}
    return extra.get("req_id") is not None

  def _use_partial(self, kwargs: Mapping[str, Any]) -> bool:
    config = getattr(self._sampler, "config", None)
    flag = getattr(config, "partial_rollout", None)
    default = bool(kwargs.get("partial_rollout", self.partial_rollout))
    return bool(default or (isinstance(flag, bool) and flag))

  def _free_kv_cache(self) -> bool:
    config = getattr(self._sampler, "config", None)
    return bool(getattr(config, "free_kv_cache_during_weight_sync", True))

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Frees (or resets) the KV cache before the weights change."""
    if self._has_round(sync_request):
      if not self._tracker.admit(sync_request, "prepared"):
        return True
      self._tracker.complete(sync_request, "prepared")
    if self._sampler is None:
      raise RuntimeError("Sampler is not available for weight sync")
    if self._use_partial(kwargs):
      pass
    elif self._free_kv_cache():
      self._sampler.delete_cache()
    else:
      self._sampler.reset_prefix_cache()
    jax.effects_barrier()
    return True

  def check(self, request: Mapping[str, Any]) -> dict[str, Any]:
    """Validates this worker's part of the plan. No collectives."""
    plan = request.get("plan")
    if plan is None:
      return _report()
    with self._lock:
      if self._sink is not None:
        self._sink.discard()
      self._sink = None
      self._transferred_generation = None
      if jax.process_index() not in plan.destination_processes:
        # In the world but not receiving this round (e.g. a replica that is
        # not ACTIVE yet): joins the collectives with a throwaway payload.
        ici_transport.check_plan(plan)
        return _report()
      sink = _SamplerSink(self._server_id, self._sampler, self._live_arrays())
      sink.drop_old = self._drop_old
      ici_transport.check_plan(plan, sink=sink)
      self._sink = sink
    return _report()

  def transfer(self, request: Mapping[str, Any]) -> dict[str, float]:
    """Enters the collectives; receives the weights if this is a destination."""
    plan = request["plan"]
    with self._lock:
      if jax.process_index() not in plan.destination_processes:
        return ici_transport.execute(plan)
      if self._sink is None:
        raise RuntimeError("ici_check must run before ici_transfer")
      timings = ici_transport.execute(plan, sink=self._sink)
      self._transferred_generation = request.get("generation")
    return timings

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Publishes the received weights (a pointer swap in staged mode)."""
    del kwargs
    if self._has_round(sync_request):
      if not self._tracker.admit(sync_request, "h2d_done"):
        return self._version
    with self._lock:
      sink = self._sink
      if sink is None or self._transferred_generation is None:
        raise RuntimeError(
            f"{self._server_id}: weight_sync without a completed ICI transfer"
        )
      extra = getattr(sync_request, "extra_config", None) or {}
      expected = extra.get("uuid")
      if expected is not None and int(expected) != int(
          self._transferred_generation
      ):
        raise RuntimeError(
            f"{self._server_id}: holding the transfer of round"
            f" {self._transferred_generation}, asked to install round"
            f" {expected}"
        )
      if sink.staged:
        sink.apply(sink.staged, delete_replaced=True)
        sink.staged = {}
      self._sink = None
    if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
      live = self._live_arrays()
      names = sorted(live)
      logging.info(
          "destination checksums: %s",
          weight_utils.compute_checksums(names, [live[n] for n in names]),
      )
    version = getattr(sync_request, "policy_version", None)
    self._version = int(version) if version is not None else self._version + 1
    if self._has_round(sync_request):
      self._tracker.complete(sync_request, "h2d_done")
    return self._version

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Rebuilds the KV cache after the new weights are in place."""
    if self._has_round(sync_request):
      if not self._tracker.admit(sync_request, "committed"):
        return True
    if self._sampler is None:
      raise RuntimeError("Sampler is not available for weight sync")
    if self._free_kv_cache() and not self._use_partial(kwargs):
      self._sampler.reinitialize_cache()
    else:
      self._sampler.refresh_state_leaves()
    if self._has_round(sync_request):
      self._tracker.complete(sync_request, "committed")
    return True

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Discards the round; refuses if the serving weights were overwritten."""
    del kwargs
    with self._lock:
      sink, self._sink = self._sink, None
      self._transferred_generation = None
      if sink is not None:
        if sink.touched:
          raise RuntimeError(
              f"{self._server_id}: the ICI transfer already replaced serving"
              " weights in place (drop_old_weights); this round cannot be"
              " rolled back and the worker must be restarted"
          )
        sink.discard()
    if self._has_round(sync_request):
      if not self._tracker.admit(sync_request, "aborted"):
        return False
      self._tracker.complete(sync_request, "aborted")
    return True

  def get_weight_sync_status(self) -> Mapping[str, Any]:
    return self._tracker.report()
