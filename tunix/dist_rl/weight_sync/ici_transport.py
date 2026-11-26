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

"""Device-to-device weight transport over JAX collectives (TPU ICI).

Trainer and rollout workers that share ONE multi-process JAX world (a single
`jax.distributed` runtime spanning one TPU slice) can move weights without a
host round trip: a `pmap`'d `lax.ppermute` sends shard k of a source process
to shard k of a destination process over the accelerator interconnect.

This module is the data plane only. It has no notion of workers, RPCs or
rounds; `ici_handler.IciHandler` drives it. Every participating process calls
`execute` with the SAME `TransferPlan`; the collectives line up because the
plan fixes their order.

How one plan runs (identically on every participant):

  for each chunk of leaves:                  # bounded scratch memory
    source role stages the chunk             # canonical per-process layout
    for each relay round:                    # log2 fan-out
      one ppermute over all participants     # holders -> waiting destinations
    destinations reshard + install the chunk

Constraints:

* Every participant owns the same number of local devices (shard k maps to
  shard k).
* A destination is one process. The source role may span several processes;
  staging then gathers a full copy onto each of them (a collective every
  source-role process enters).
* EVERY process of the JAX world enters every collective, including ones
  that neither send nor receive (they join with a throwaway payload).
* A participant that fails mid-plan leaves its peers blocked inside a
  collective. Callers must validate with `check_plan` on every participant
  BEFORE any of them calls `execute`, and bound `execute` with a timeout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import time
from typing import Any, Protocol

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_CHUNK_BYTES = 2 << 30

_AXIS = "ici"
_CHUNK_CACHE_MAXSIZE = 1024
_MOVER_CACHE_MAXSIZE = 64


@dataclasses.dataclass(frozen=True)
class LeafSpec:
  """One logical tensor, as both sides agree to call and shape it.

  Attributes:
    name: Canonical weight name agreed upon by source and destination.
    shape: Global tensor shape (all dimensions non-negative).
    dtype: Canonical NumPy/JAX dtype string (for example, `"bfloat16"`).
  """

  name: str
  shape: tuple[int, ...]
  dtype: str

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("LeafSpec.name must not be empty")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
        for dim in self.shape
    ):
      raise ValueError(
          f"LeafSpec {self.name!r} has invalid shape {self.shape}; dimensions"
          " must be non-negative ints"
      )
    try:
      jnp.dtype(self.dtype)
    except (TypeError, ValueError) as exc:
      raise ValueError(
          f"LeafSpec {self.name!r} has invalid dtype {self.dtype!r}"
      ) from exc

  @property
  def nbytes(self) -> int:
    """Global size of this leaf in bytes."""
    return (
        int(np.prod(self.shape, dtype=np.int64))
        * jnp.dtype(self.dtype).itemsize
    )


@dataclasses.dataclass(frozen=True)
class TransferPlan:
  """Everything a participant needs to run its part of a transfer.

  Plain data: it crosses process boundaries inside an RPC.

  Attributes:
    leaves: The tensors to move, in canonical order.
    chunks: Leaf indices grouped per collective.
    rounds: Relay schedule. `rounds[r]` is the `(sender, receiver)` process
      pairs of round r; a receiver holds the chunk from the next round on.
    source_processes: Every process of the source role.
    participants: Every process that enters the collectives, sorted: the whole
      JAX world, bystanders included.
    width: Local device count each participant must own.
  """

  leaves: tuple[LeafSpec, ...]
  chunks: tuple[tuple[int, ...], ...]
  rounds: tuple[tuple[tuple[int, int], ...], ...]
  source_processes: tuple[int, ...]
  participants: tuple[int, ...]
  width: int

  @property
  def destination_processes(self) -> tuple[int, ...]:
    """Destination process indices in delivery order across all rounds."""
    return tuple(dst for pairs in self.rounds for _, dst in pairs)


class Sink(Protocol):
  """Destination-side interface for receiving and installing weight chunks."""

  def target_sharding(self, name: str) -> jax.sharding.Sharding:
    """Returns the sharding onto which the installed leaf must be placed."""

  def release(self, names: Sequence[str]) -> None:
    """Called before a chunk arrives; may free the leaves it replaces."""

  def install(self, arrays: Mapping[str, jax.Array]) -> None:
    """Receives a chunk's leaves, already placed on `target_sharding`."""


def plan_chunks(
    leaves: Sequence[LeafSpec], chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> tuple[tuple[int, ...], ...]:
  """Groups leaf indices so each group's global bytes stay under a limit.

  Greedy and order-preserving, hence deterministic: equal inputs give equal
  chunks, which lets compiled collectives be reused across syncs. A leaf
  larger than the limit gets a chunk of its own.

  Args:
    leaves: Ordered leaves to partition into chunks.
    chunk_bytes: Upper bound on global bytes per chunk.

  Returns:
    Tuple of chunks, where each chunk is a tuple of indices into `leaves`.

  Raises:
    ValueError: If `chunk_bytes` is not positive.
  """
  if chunk_bytes <= 0:
    raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
  chunks: list[list[int]] = [[]]
  size = 0
  for index, leaf in enumerate(leaves):
    if chunks[-1] and size + leaf.nbytes > chunk_bytes:
      chunks.append([])
      size = 0
    chunks[-1].append(index)
    size += leaf.nbytes
  return tuple(tuple(chunk) for chunk in chunks if chunk)


def plan_rounds(
    source_processes: Sequence[int], destination_processes: Sequence[int]
) -> tuple[tuple[tuple[int, int], ...], ...]:
  """Builds a log2 relay schedule from sources to destinations.

  Every holder feeds one waiting destination per round. A process sends at
  most once and receives at most once per round, so each round is a valid
  partial permutation and holders double every round.

  Args:
    source_processes: Process indices that initially hold the weights.
    destination_processes: Process indices that need to receive the weights.

  Returns:
    Tuple of rounds, each containing `(sender, receiver)` process index pairs.
  """
  holders = list(source_processes)
  waiting = list(destination_processes)
  rounds: list[tuple[tuple[int, int], ...]] = []
  while waiting:
    # Deliberately truncates to min(len(holders), len(waiting)).
    pairs = tuple(zip(holders, waiting))
    waiting = waiting[len(pairs) :]
    holders.extend(dst for _, dst in pairs)
    rounds.append(pairs)
  return tuple(rounds)


def _validate_plan_inputs(
    leaves: Sequence[LeafSpec],
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    world: tuple[int, ...],
    width: int,
) -> None:
  """Validates `build_plan` arguments before constructing a `TransferPlan`."""
  if len(set(world)) != len(world):
    raise ValueError(f"duplicate process in world_processes {world}")
  outside = sorted({*sources, *destinations} - set(world))
  if outside:
    raise ValueError(
        f"processes {outside} are not part of the JAX world {list(world)}"
    )
  if not sources or not destinations:
    raise ValueError("an ICI transfer needs a source and a destination")
  if len(set(sources)) != len(sources) or len(set(destinations)) != len(
      destinations
  ):
    raise ValueError(
        f"duplicate process in sources {sources} or destinations {destinations}"
    )
  overlap = sorted(set(sources) & set(destinations))
  if overlap:
    raise ValueError(
        f"processes {overlap} are both source and destination; trainer and"
        " rollout must be separate JAX processes"
    )
  if width <= 0:
    raise ValueError(f"width must be positive, got {width}")
  names = [leaf.name for leaf in leaves]
  if len(set(names)) != len(names):
    raise ValueError("duplicate leaf names in an ICI transfer plan")
  if not leaves:
    raise ValueError("an ICI transfer plan needs at least one leaf")


def build_plan(
    leaves: Sequence[LeafSpec],
    *,
    source_processes: Sequence[int],
    destination_processes: Sequence[int],
    width: int,
    world_processes: Sequence[int] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> TransferPlan:
  """Builds the transfer plan executed concurrently by every participant.

  Args:
    leaves: The tensors to move, in the order both sides agreed on.
    source_processes: JAX process indices of the source role.
    destination_processes: One JAX process index per destination.
    width: Local device count of every participant.
    world_processes: Every process index of the JAX world. When None, defaults
      to the sorted union of `source_processes` and `destination_processes`.
    chunk_bytes: Upper bound on the global bytes moved per collective.

  Returns:
    A deterministic `TransferPlan` shared across all participants.

  Raises:
    ValueError: If roles overlap, contain duplicates, reference processes
      outside `world_processes`, have non-positive `width`, or contain empty or
      duplicate leaf names.
  """
  sources = tuple(int(p) for p in source_processes)
  destinations = tuple(int(p) for p in destination_processes)
  world = (
      tuple(sorted({*sources, *destinations}))
      if world_processes is None
      else tuple(sorted(int(p) for p in world_processes))
  )
  _validate_plan_inputs(leaves, sources, destinations, world, width)
  return TransferPlan(
      leaves=tuple(leaves),
      chunks=plan_chunks(leaves, chunk_bytes),
      rounds=plan_rounds(sources, destinations),
      source_processes=sources,
      participants=world,
      width=int(width),
  )


def local_devices(process: int | None = None) -> Sequence[jax.Device]:
  """Returns one process's devices in ascending device-ID order.

  Args:
    process: Target JAX process index. When None, uses `jax.process_index()`.

  Returns:
    Tuple of devices owned by `process`, sorted by `device.id`.
  """
  target = jax.process_index() if process is None else process
  return tuple(
      sorted(
          (d for d in jax.devices() if d.process_index == target),
          key=lambda d: d.id,
      )
  )


def _participant_devices(plan: TransferPlan) -> tuple[jax.Device, ...]:
  members = set(plan.participants)
  return tuple(
      sorted(
          (d for d in jax.devices() if d.process_index in members),
          key=lambda d: (d.process_index, d.id),
      )
  )


def _split_axis(shape: Sequence[int], width: int) -> int | None:
  """Returns the first axis `width` divides, or None if none does."""
  for axis, size in enumerate(shape):
    if size >= width and size % width == 0:
      return axis
  return None


def _shard_shape(shape: Sequence[int], width: int) -> tuple[int, ...]:
  """Returns the per-device shard shape under `canonical_spec(shape, width)`."""
  axis = _split_axis(shape, width)
  if axis is None:
    return tuple(shape)
  return (*shape[:axis], shape[axis] // width, *shape[axis + 1 :])


def canonical_spec(shape: Sequence[int], width: int) -> tuple[str | None, ...]:
  """Splits the first axis `width` divides; replicates when none does.

  Args:
    shape: Global tensor shape.
    width: Number of local devices per participant.

  Returns:
    A tuple of `PartitionSpec` axis names (each `_AXIS` or None) with length
    matching `len(shape)`.
  """
  spec: list[str | None] = [None] * len(shape)
  axis = _split_axis(shape, width)
  if axis is not None:
    spec[axis] = _AXIS
  return tuple(spec)


def _check_source_arrays(
    plan: TransferPlan,
    source: Mapping[str, Any] | None,
    me: int,
) -> None:
  """Validates source leaf shapes, dtypes, and process placement."""
  if source is None:
    raise ValueError(f"source process {me} has no source arrays")
  source_proc_set = set(plan.source_processes)
  expected_multihost_devices = (
      {d for p in plan.source_processes for d in local_devices(p)}
      if len(source_proc_set) > 1
      else None
  )
  for leaf in plan.leaves:
    array = source.get(leaf.name)
    if array is None:
      raise ValueError(f"source is missing leaf {leaf.name!r}")
    if tuple(array.shape) != leaf.shape or str(array.dtype) != leaf.dtype:
      raise ValueError(
          f"source leaf {leaf.name!r} is {tuple(array.shape)}"
          f" {array.dtype}; the plan wants {leaf.shape} {leaf.dtype}"
      )
    sharding = getattr(array, "sharding", None)
    if sharding is None or not hasattr(sharding, "device_set"):
      raise ValueError(
          f"source leaf {leaf.name!r} has no valid JAX sharding: {sharding!r}"
      )
    array_procs = {int(d.process_index) for d in sharding.device_set}
    if not array_procs <= source_proc_set:
      raise ValueError(
          f"source leaf {leaf.name!r} spans processes {sorted(array_procs)},"
          f" outside plan source_processes {list(plan.source_processes)}"
      )
    if expected_multihost_devices is not None:
      if (
          array.is_fully_addressable
          or array_procs != source_proc_set
          or sharding.device_set != expected_multihost_devices
      ):
        raise ValueError(
            f"multi-host source leaf {leaf.name!r} must span all devices of"
            f" source_processes {list(plan.source_processes)}; got processes"
            f" {sorted(array_procs)} with {len(sharding.device_set)} device(s)"
            f" (is_fully_addressable={array.is_fully_addressable})"
        )
    elif not array.is_fully_addressable:
      raise ValueError(
          f"single-host source leaf {leaf.name!r} must be fully addressable"
          f" on process {me}"
      )


def _check_destination_sink(
    plan: TransferPlan,
    sink: Sink | None,
    me: int,
) -> None:
  """Validates destination sink shardings belong strictly to process `me`."""
  if sink is None:
    raise ValueError(f"destination process {me} has no sink")
  for leaf in plan.leaves:
    target_shd = sink.target_sharding(leaf.name)
    if not hasattr(target_shd, "device_set"):
      raise ValueError(
          f"destination leaf {leaf.name!r} has invalid target sharding"
          f" {target_shd!r}"
      )
    target_procs = {int(d.process_index) for d in target_shd.device_set}
    if target_procs != {me}:
      raise ValueError(
          f"destination leaf {leaf.name!r} target sharding spans processes"
          f" {sorted(target_procs)}; an ICI destination must stay local to"
          f" process {me}"
      )


def check_plan(
    plan: TransferPlan,
    *,
    source: Mapping[str, Any] | None = None,
    sink: Sink | None = None,
) -> None:
  """Raises unless this process can run its part of `plan` (no collectives).

  Run on every participant before any of them calls `execute`: a participant
  that fails inside `execute` strands the others in a collective.

  Args:
    plan: The shared `TransferPlan` to validate.
    source: `{leaf_name: array}` mapping on source processes, or None on
      non-source processes.
    sink: Destination `Sink` on destination processes, or None on
      non-destination processes.

  Raises:
    ValueError: If this process is not in `plan.participants`, the local JAX
      world disagrees with `plan.participants`, local device counts disagree
      with `plan.width`, or any source/destination leaf fails shape, dtype, or
      placement validation.
  """
  me = jax.process_index()
  if me not in plan.participants:
    raise ValueError(
        f"process {me} is not a participant of this plan {plan.participants}"
    )
  world = sorted({d.process_index for d in jax.devices()})
  if list(plan.participants) != world:
    raise ValueError(
        f"plan participants {list(plan.participants)} are not this JAX world"
        f" {world}; trainer and rollout must share one jax.distributed runtime"
        " for ICI weight sync, and every process of it must participate"
    )
  for process in plan.participants:
    count = len(local_devices(process))
    if count != plan.width:
      raise ValueError(
          f"process {process} owns {count} device(s) but the plan is"
          f" {plan.width} wide; every participant must own the same number"
      )
  if me in plan.source_processes:
    _check_source_arrays(plan, source, me)
  if me in plan.destination_processes:
    _check_destination_sink(plan, sink, me)


def _device_ids(devices: Sequence[jax.Device]) -> tuple[int, ...]:
  return tuple(int(d.id) for d in devices)


def _devices_by_ids(device_ids: tuple[int, ...]) -> list[jax.Device]:
  by_id = {d.id: d for d in jax.devices()}
  return [by_id[i] for i in device_ids]


def _canonical_shardings(
    leaves: Sequence[LeafSpec], devices: Sequence[jax.Device]
) -> tuple[jax.sharding.NamedSharding, ...]:
  """Builds canonical local-device shardings sharing one Mesh for `leaves`."""
  width = len(devices)
  mesh = jax.sharding.Mesh(np.asarray(devices), (_AXIS,))
  return tuple(
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*canonical_spec(leaf.shape, width))
      )
      for leaf in leaves
  )


def _local_payload_shardings(
    ndims: Sequence[int], device_ids: tuple[int, ...]
) -> tuple[jax.sharding.NamedSharding, ...]:
  devices = _devices_by_ids(device_ids)
  mesh = jax.sharding.Mesh(np.asarray(devices), ("local",))
  return tuple(
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec("local", *([None] * ndim))
      )
      for ndim in ndims
  )


@functools.lru_cache(maxsize=_MOVER_CACHE_MAXSIZE)
def _mover(
    pairs: tuple[tuple[int, int], ...], device_ids: tuple[int, ...]
) -> Callable[[Any], Any]:
  """A pmap'd ppermute over the participants' devices.

  Cached per (permutation, device set); pmap's own cache then keys on the
  chunk's shapes, so each distinct chunk compiles once per relay round.
  """
  devices = _devices_by_ids(device_ids)

  def move(tree: Any) -> Any:
    return jax.tree.map(lambda v: jax.lax.ppermute(v, _AXIS, pairs), tree)

  return jax.pmap(
      move, axis_name=_AXIS, axis_size=len(devices), devices=devices
  )


@functools.lru_cache(maxsize=_CHUNK_CACHE_MAXSIZE)
def _resharder(
    out_shardings: tuple[jax.sharding.Sharding, ...],
) -> Callable[[Any], Any]:
  """A jitted identity that lands a tuple of leaves on `out_shardings`."""
  return jax.jit(lambda values: values, out_shardings=out_shardings)


def _device_pairs(
    pairs: Sequence[tuple[int, int]],
    devices: Sequence[jax.Device],
    width: int,
) -> tuple[tuple[int, int], ...]:
  """Process-level pairs as positions in the participants' device list."""
  first: dict[int, int] = {}
  for position, device in enumerate(devices):
    first.setdefault(device.process_index, position)
  result: list[tuple[int, int]] = []
  for src, dst in pairs:
    for k in range(width):
      result.append((first[src] + k, first[dst] + k))
  return tuple(result)


def _pack_canonical_leaf(
    array: jax.Array, shape: tuple[int, ...], width: int
) -> jax.Array:
  """Reshapes a canonical-sharded leaf into `(width, *shard_shape)`."""
  axis = _split_axis(shape, width)
  if axis is None:
    return jnp.broadcast_to(array[None, ...], (width, *shape))
  split_shape = (
      *shape[:axis],
      width,
      shape[axis] // width,
      *shape[axis + 1 :],
  )
  return jnp.moveaxis(array.reshape(split_shape), axis, 0)


def _unpack_payload_leaf(
    payload: jax.Array, shape: tuple[int, ...], width: int
) -> jax.Array:
  """Inverse of `_pack_canonical_leaf`: `(width, *shard_shape)` -> `shape`."""
  axis = _split_axis(shape, width)
  if axis is None:
    return payload[0]
  return jnp.moveaxis(payload, 0, axis).reshape(shape)


@functools.lru_cache(maxsize=_CHUNK_CACHE_MAXSIZE)
def _zero_payloads_fn(
    specs: tuple[tuple[tuple[int, ...], str], ...],
    device_ids: tuple[int, ...],
) -> Callable[[], tuple[jax.Array, ...]]:
  """Allocates a chunk's zero payloads on each local device in one jit."""
  width = len(device_ids)
  shard_shapes = tuple(
      (width, *_shard_shape(shape, width)) for shape, _ in specs
  )
  dtypes = tuple(jnp.dtype(dtype_str) for _, dtype_str in specs)
  out_shardings = _local_payload_shardings(
      [len(shape) for shape, _ in specs], device_ids
  )

  def make_zeros() -> tuple[jax.Array, ...]:
    return tuple(
        jnp.zeros(s, dtype=dt)
        for s, dt in zip(shard_shapes, dtypes, strict=True)
    )

  return jax.jit(make_zeros, out_shardings=out_shardings)


def _zero_payloads(
    leaves: Sequence[LeafSpec], devices: Sequence[jax.Device]
) -> tuple[jax.Array, ...]:
  specs = tuple((leaf.shape, leaf.dtype) for leaf in leaves)
  return _zero_payloads_fn(specs, _device_ids(devices))()


@functools.lru_cache(maxsize=_CHUNK_CACHE_MAXSIZE)
def _chunk_packer(
    shapes: tuple[tuple[int, ...], ...],
    device_ids: tuple[int, ...],
) -> Callable[[tuple[jax.Array, ...]], tuple[jax.Array, ...]]:
  """Packs a chunk of local arrays into `(width, *shard_shape)` payloads."""
  width = len(device_ids)
  out_shardings = _local_payload_shardings(
      [len(shape) for shape in shapes], device_ids
  )

  def pack(arrays: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(
        _pack_canonical_leaf(a, s, width)
        for a, s in zip(arrays, shapes, strict=True)
    )

  return jax.jit(pack, out_shardings=out_shardings)


@functools.lru_cache(maxsize=_CHUNK_CACHE_MAXSIZE)
def _chunk_installer(
    shapes: tuple[tuple[int, ...], ...],
    target_shardings: tuple[jax.sharding.Sharding, ...],
    width: int,
) -> Callable[[tuple[jax.Array, ...]], tuple[jax.Array, ...]]:
  """Unpacks a chunk's pmap payloads and reshards onto `target_shardings`."""

  def install(payloads: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(
        _unpack_payload_leaf(p, s, width)
        for p, s in zip(payloads, shapes, strict=True)
    )

  return jax.jit(install, out_shardings=target_shardings)


def _sharding_device_order(sharding: Any) -> tuple[int, ...] | None:
  """Returns the ordered device IDs of `sharding`, or None if unavailable."""
  mesh = getattr(sharding, "mesh", None)
  if mesh is not None and hasattr(mesh, "devices"):
    return tuple(int(d.id) for d in np.asarray(mesh.devices).ravel())
  return None


def _all_match_device_order(
    shardings: Sequence[Any], expected_device_ids: tuple[int, ...]
) -> bool:
  return all(
      _sharding_device_order(sh) == expected_device_ids for sh in shardings
  )


def _staging_shardings(
    leaves: Sequence[LeafSpec],
    source_processes: Sequence[int],
    width: int,
) -> tuple[jax.sharding.NamedSharding, ...]:
  """Shardings that give EVERY source-role process a whole copy of each leaf.

  The source mesh is `(process, local device)`: replicated over the process
  axis, split `width` ways over the local one. Each process then holds the leaf
  in exactly the canonical per-device pieces.
  """
  rows = [local_devices(p) for p in source_processes]
  if any(len(row) != width for row in rows):
    raise RuntimeError(
        f"cannot stage: source processes own {[len(r) for r in rows]} mesh"
        f" devices but the transfer is {width} wide"
    )
  mesh = jax.sharding.Mesh(np.asarray(rows), ("ici_process", _AXIS))
  return tuple(
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*canonical_spec(leaf.shape, width))
      )
      for leaf in leaves
  )


def _stage_multiprocess(
    leaves: Sequence[LeafSpec],
    arrays: tuple[jax.Array, ...],
    devices: Sequence[jax.Device],
    source_processes: Sequence[int],
) -> tuple[jax.Array, ...]:
  """Gathers a full canonical copy onto each source process, then packs it."""
  width = len(devices)
  shapes = tuple(leaf.shape for leaf in leaves)
  device_ids = _device_ids(devices)
  shardings = _staging_shardings(leaves, source_processes, width)
  staging_device_ids = _device_ids(list(shardings[0].mesh.devices.flat))
  if _all_match_device_order([a.sharding for a in arrays], staging_device_ids):
    staged = _resharder(shardings)(arrays)
  else:
    # Input arrays live on the same device set in a different device order
    # (e.g., MaxText's 2D TPU snake mesh vs. process-grouped staging mesh).
    staged = jax.device_put(arrays, shardings)
  jax.block_until_ready(staged)
  canonicals = _canonical_shardings(leaves, devices)
  local_canonicals = []
  for leaf, array, canonical in zip(leaves, staged, canonicals, strict=True):
    # Refuse rather than assume: if staging handed this process's devices
    # other slices than the canonical view, shards would arrive at the wrong
    # offsets -- silent corruption, not a crash.
    if array.sharding.addressable_devices_indices_map(
        leaf.shape
    ) != canonical.addressable_devices_indices_map(leaf.shape):
      raise RuntimeError(
          f"staging {leaf.name!r} {leaf.shape} gave this process's devices"
          " different slices than the canonical layout"
      )
    by_device = {s.device: s.data for s in array.addressable_shards}
    local_canonicals.append(
        jax.make_array_from_single_device_arrays(
            leaf.shape,
            canonical,
            [by_device[d] for d in devices],
            dtype=jnp.dtype(leaf.dtype),
        )
    )
  payloads = _chunk_packer(shapes, device_ids)(tuple(local_canonicals))
  jax.block_until_ready(payloads)
  return payloads


def _stage(
    leaves: Sequence[LeafSpec],
    source: Mapping[str, Any],
    devices: Sequence[jax.Device],
    source_processes: Sequence[int],
) -> tuple[jax.Array, ...]:
  """Puts a chunk into the canonical pmap payload layout in one batched step.

  A collective when the source role spans processes: every one of them must
  call it.
  """
  arrays = tuple(source[leaf.name] for leaf in leaves)
  shapes = tuple(leaf.shape for leaf in leaves)
  device_ids = _device_ids(devices)
  if len(source_processes) == 1 and all(a.is_fully_addressable for a in arrays):
    # Single-process source: fuse reshard-to-canonical + pack into one jit call
    # when input arrays already match `devices` order; otherwise align via
    # `jax.device_put` (e.g. TPU 2D snake-ordered meshes).
    if not _all_match_device_order([a.sharding for a in arrays], device_ids):
      arrays = jax.device_put(arrays, _canonical_shardings(leaves, devices))
    payloads = _chunk_packer(shapes, device_ids)(arrays)
    jax.block_until_ready(payloads)
    return payloads
  return _stage_multiprocess(leaves, arrays, devices, source_processes)


def _install_chunk(
    leaves: Sequence[LeafSpec],
    payloads: Sequence[jax.Array],
    devices: Sequence[jax.Device],
    sink: Sink,
) -> None:
  """Unpacks `payloads` and installs the leaves onto `sink`."""
  shapes = tuple(leaf.shape for leaf in leaves)
  device_ids = _device_ids(devices)
  target_shardings = tuple(sink.target_sharding(leaf.name) for leaf in leaves)
  if _all_match_device_order(target_shardings, device_ids):
    unpacked = _chunk_installer(shapes, target_shardings, len(devices))(
        tuple(payloads)
    )
  else:
    canonicals = _canonical_shardings(leaves, devices)
    canonical_arrays = _chunk_installer(shapes, canonicals, len(devices))(
        tuple(payloads)
    )
    unpacked = jax.device_put(canonical_arrays, target_shardings)
  jax.block_until_ready(unpacked)
  sink.install(
      {leaf.name: arr for leaf, arr in zip(leaves, unpacked, strict=True)}
  )


def _receive_round(plan: TransferPlan, process: int) -> int | None:
  """Returns the round index in which `process` receives a chunk, or None."""
  for round_idx, pairs in enumerate(plan.rounds):
    if any(dst == process for _, dst in pairs):
      return round_idx
  return None


def _execute_chunk(
    leaves: Sequence[LeafSpec],
    *,
    plan: TransferPlan,
    devices: Sequence[jax.Device],
    movers: Sequence[Callable[[Any], Any]],
    source: Mapping[str, Any] | None,
    sink: Sink | None,
    receive_round: int | None,
) -> tuple[float, float, float, int]:
  """Runs staging, relay rounds, and installation for one chunk of leaves."""
  t0 = time.perf_counter()
  if source is not None:
    payloads = _stage(leaves, source, devices, plan.source_processes)
  else:
    if sink is not None:
      sink.release([leaf.name for leaf in leaves])
    payloads = _zero_payloads(leaves, devices)
  t1 = time.perf_counter()
  for index, mover in enumerate(movers):
    moved = mover(payloads)
    jax.block_until_ready(moved)
    if index == receive_round:
      # From here on this process relays what it received.
      payloads = moved
    del moved
  t2 = time.perf_counter()
  received = 0
  if sink is not None:
    _install_chunk(leaves, payloads, devices, sink)
    received = sum(leaf.nbytes for leaf in leaves)
  del payloads
  t3 = time.perf_counter()
  return t1 - t0, t2 - t1, t3 - t2, received


def _warn_if_cache_exceeded(plan: TransferPlan) -> None:
  """Warns when a plan has more chunks or rounds than the JIT cache holds."""
  if len(plan.chunks) > _CHUNK_CACHE_MAXSIZE:
    logging.warning(
        "ICI transfer plan has %d chunks (> cache limit %d); consider raising"
        " chunk_bytes to avoid XLA recompilation.",
        len(plan.chunks),
        _CHUNK_CACHE_MAXSIZE,
    )
  if len(plan.rounds) > _MOVER_CACHE_MAXSIZE:
    logging.warning(
        "ICI transfer plan has %d rounds (> mover cache limit %d).",
        len(plan.rounds),
        _MOVER_CACHE_MAXSIZE,
    )


def execute(
    plan: TransferPlan,
    *,
    source: Mapping[str, Any] | None = None,
    sink: Sink | None = None,
) -> Mapping[str, float]:
  """Runs this process's part of `plan`. Blocks until the plan is done.

  Every participant must call this with an equal plan, concurrently.

  Args:
    plan: The shared plan.
    source: `{leaf name: array}` on a source-role process; arrays must match
      the plan's shapes and dtypes. They are read, never donated or deleted.
      Pass None on non-source processes.
    sink: Where a destination process installs what it receives. Pass None on
      non-destination processes.

  Returns:
    Mapping of wall-clock seconds spent per stage (`stage`, `move`, `install`,
    `total`) and `bytes` received by this process.

  Raises:
    ValueError: If pre-transfer validation via `check_plan` fails.
  """
  check_plan(plan, source=source, sink=sink)
  me = jax.process_index()
  receive_round = _receive_round(plan, me)
  active_source: Mapping[str, Any] | None = (
      source if me in plan.source_processes else None
  )
  active_sink: Sink | None = sink if receive_round is not None else None
  _warn_if_cache_exceeded(plan)

  devices = local_devices()
  world_devices = _participant_devices(plan)
  device_ids = _device_ids(world_devices)
  movers = [
      _mover(_device_pairs(pairs, world_devices, plan.width), device_ids)
      for pairs in plan.rounds
  ]
  timings = {"stage": 0.0, "move": 0.0, "install": 0.0}
  received = 0
  started = time.perf_counter()
  for chunk in plan.chunks:
    leaves = [plan.leaves[i] for i in chunk]
    stage_s, move_s, install_s, chunk_bytes = _execute_chunk(
        leaves,
        plan=plan,
        devices=devices,
        movers=movers,
        source=active_source,
        sink=active_sink,
        receive_round=receive_round,
    )
    timings["stage"] += stage_s
    timings["move"] += move_s
    timings["install"] += install_s
    received += chunk_bytes
  timings["total"] = time.perf_counter() - started
  timings["bytes"] = float(received)
  logging.info(
      "ici transfer on process %d: %d leaves in %d chunk(s) x %d round(s),"
      " %.2f GiB received, stage %.2fs move %.2fs install %.2fs",
      me,
      len(plan.leaves),
      len(plan.chunks),
      len(plan.rounds),
      received / 2**30,
      timings["stage"],
      timings["move"],
      timings["install"],
  )
  return timings
