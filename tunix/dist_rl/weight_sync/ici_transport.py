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

Design points carried over from the single-controller (Pathways) fast sync:

* Conform once. The source hands over leaves that already have the
  destination's names, shapes and dtypes; nothing is converted per replica.
* Chunked, cached programs. A collective moves a whole chunk of leaves, not
  one leaf, so a sync is a few dozen programs rather than leaves x replicas
  of them. Chunking is deterministic, so the programs compiled on the first
  sync are reused by every later one.
* One fan-out, not one transfer per replica. A destination that has received
  a chunk relays it in the next round, so R replicas take ceil(log2(R + S))
  rounds from S source processes instead of R.
* Drop old before new. A destination may free a chunk's serving leaves right
  before receiving their replacements (`Sink.release`), for models whose old
  and new weights do not fit in device memory together.

Constraints:

* Every participant owns the same number of local devices (shard k maps to
  shard k).
* A destination is one process. The source role may span several processes;
  staging then gathers a full copy onto each of them (a collective every
  source-role process enters).
* EVERY process of the JAX world enters every collective, including ones
  that neither send nor receive (they join with a throwaway payload). A
  process that sits a collective out does not error; it blocks the ones that
  did not. Scoping the `pmap` to the devices of the sending and receiving
  processes does not lift this.
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


@dataclasses.dataclass(frozen=True)
class LeafSpec:
  """One logical tensor, as both sides agree to call and shape it."""

  name: str
  shape: tuple[int, ...]
  dtype: str

  @property
  def nbytes(self) -> int:
    return int(np.prod(self.shape, dtype=np.int64)) * jnp.dtype(
        self.dtype
    ).itemsize


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
    return tuple(dst for pairs in self.rounds for _, dst in pairs)

  @property
  def nbytes(self) -> int:
    return sum(leaf.nbytes for leaf in self.leaves)


class Sink(Protocol):
  """Where a destination puts what it receives."""

  def target_sharding(self, name: str) -> Any:
    """The sharding the installed leaf must have."""

  def release(self, names: Sequence[str]) -> None:
    """Called before a chunk arrives; may free the leaves it replaces."""

  def install(self, arrays: Mapping[str, Any]) -> None:
    """Receives a chunk's leaves, already on `target_sharding`."""


def plan_chunks(
    leaves: Sequence[LeafSpec], chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> tuple[tuple[int, ...], ...]:
  """Groups leaf indices so each group's global bytes stay under a limit.

  Greedy and order-preserving, hence deterministic: equal inputs give equal
  chunks, which is what lets compiled collectives be reused across syncs. A
  leaf larger than the limit gets a chunk of its own.
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
  """Relay schedule: every holder feeds one waiting destination per round.

  A process sends at most once and receives at most once per round, so each
  round is a valid (partial) permutation. Holders double every round.
  """
  holders = list(source_processes)
  waiting = list(destination_processes)
  rounds: list[tuple[tuple[int, int], ...]] = []
  while waiting:
    pairs = tuple(zip(holders, waiting))
    waiting = waiting[len(pairs) :]
    holders.extend(dst for _, dst in pairs)
    rounds.append(pairs)
  return tuple(rounds)


def build_plan(
    leaves: Sequence[LeafSpec],
    *,
    source_processes: Sequence[int],
    destination_processes: Sequence[int],
    width: int,
    world_processes: Sequence[int] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> TransferPlan:
  """Builds the plan every participant executes.

  Args:
    leaves: The tensors to move, in the order both sides agreed on.
    source_processes: JAX process indices of the source role.
    destination_processes: One JAX process index per destination.
    width: Local device count of every participant.
    world_processes: Every process index of the JAX world. Defaults to the
      sources and destinations, which is only right when they ARE the world.
    chunk_bytes: Upper bound on the global bytes moved per collective.
  """
  sources = tuple(int(p) for p in source_processes)
  destinations = tuple(int(p) for p in destination_processes)
  world = (
      tuple(sorted({*sources, *destinations}))
      if world_processes is None
      else tuple(sorted(int(p) for p in world_processes))
  )
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
        f"duplicate process in sources {sources} or destinations"
        f" {destinations}"
    )
  if set(sources) & set(destinations):
    raise ValueError(
        f"processes {sorted(set(sources) & set(destinations))} are both source"
        " and destination; trainer and rollout must be separate JAX processes"
    )
  if width <= 0:
    raise ValueError(f"width must be positive, got {width}")
  names = [leaf.name for leaf in leaves]
  if len(set(names)) != len(names):
    raise ValueError("duplicate leaf names in an ICI transfer plan")
  if not leaves:
    raise ValueError("an ICI transfer plan needs at least one leaf")
  return TransferPlan(
      leaves=tuple(leaves),
      chunks=plan_chunks(leaves, chunk_bytes),
      rounds=plan_rounds(sources, destinations),
      source_processes=sources,
      participants=world,
      width=int(width),
  )


def local_devices(process: int | None = None) -> list[Any]:
  """This (or another) process's devices, in the order shards map onto."""
  process = jax.process_index() if process is None else process
  return sorted(
      (d for d in jax.devices() if d.process_index == process),
      key=lambda d: d.id,
  )


def _participant_devices(plan: TransferPlan) -> list[Any]:
  members = set(plan.participants)
  return sorted(
      (d for d in jax.devices() if d.process_index in members),
      key=lambda d: (d.process_index, d.id),
  )


def canonical_spec(shape: Sequence[int], width: int) -> tuple[Any, ...]:
  """Splits the first axis `width` divides; replicates when none does."""
  spec: list[Any] = [None] * len(shape)
  for axis, size in enumerate(shape):
    if size >= width and size % width == 0:
      spec[axis] = _AXIS
      break
  return tuple(spec)


def _canonical_sharding(leaf: LeafSpec, devices: Sequence[Any]) -> Any:
  mesh = jax.sharding.Mesh(np.asarray(devices), (_AXIS,))
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(*canonical_spec(leaf.shape, len(devices)))
  )


def check_plan(
    plan: TransferPlan,
    *,
    source: Mapping[str, Any] | None = None,
    sink: Sink | None = None,
) -> None:
  """Raises unless this process can run its part of `plan`. No collectives.

  Run on every participant before any of them calls `execute`: a participant
  that fails inside `execute` strands the others in a collective.
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
    if source is None:
      raise ValueError(f"source process {me} has no source arrays")
    for leaf in plan.leaves:
      array = source.get(leaf.name)
      if array is None:
        raise ValueError(f"source is missing leaf {leaf.name!r}")
      if tuple(array.shape) != leaf.shape or str(array.dtype) != leaf.dtype:
        raise ValueError(
            f"source leaf {leaf.name!r} is {tuple(array.shape)}"
            f" {array.dtype}; the plan wants {leaf.shape} {leaf.dtype}"
        )
  if me in plan.destination_processes:
    if sink is None:
      raise ValueError(f"destination process {me} has no sink")
    for leaf in plan.leaves:
      sink.target_sharding(leaf.name)


@functools.lru_cache(maxsize=8)
def _mover(pairs: tuple[tuple[int, int], ...], device_ids: tuple[int, ...]):
  """A pmap'd ppermute over the participants' devices.

  Cached per (permutation, device set); pmap's own cache then keys on the
  chunk's shapes, so each distinct chunk compiles once per relay round.
  """
  by_id = {d.id: d for d in jax.devices()}
  devices = [by_id[i] for i in device_ids]

  def move(tree):
    return jax.tree.map(lambda v: jax.lax.ppermute(v, _AXIS, pairs), tree)

  return jax.pmap(
      move, axis_name=_AXIS, axis_size=len(devices), devices=devices
  )


@functools.lru_cache(maxsize=64)
def _resharder(out_shardings: tuple[Any, ...]):
  """A jitted identity that lands a tuple of leaves on `out_shardings`.

  `device_put` rejects arrays that are not fully addressable; jit with
  `out_shardings` is the ordinary multi-process reshard.
  """
  return jax.jit(lambda values: values, out_shardings=out_shardings)


def _device_pairs(
    pairs: Sequence[tuple[int, int]], devices: Sequence[Any], width: int
) -> tuple[tuple[int, int], ...]:
  """Process-level pairs as positions in the participants' device list."""
  first = {}
  for position, device in enumerate(devices):
    first.setdefault(device.process_index, position)
  return tuple(
      (first[src] + k, first[dst] + k)
      for src, dst in pairs
      for k in range(width)
  )


def _as_payload(shards: Sequence[Any], devices: Sequence[Any]) -> Any:
  """Per-device shards as this process's slice of a pmap argument."""
  mesh = jax.sharding.Mesh(np.asarray(devices), ("local",))
  shard_shape = tuple(shards[0].shape)
  return jax.make_array_from_single_device_arrays(
      (len(devices), *shard_shape),
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec("local", *([None] * len(shard_shape)))
      ),
      [shard[None, ...] for shard in shards],
  )


def _zero_payload(leaf: LeafSpec, devices: Sequence[Any]) -> Any:
  shard_shape = _canonical_sharding(leaf, devices).shard_shape(leaf.shape)
  dtype = jnp.dtype(leaf.dtype)
  return _as_payload(
      [jax.device_put(jnp.zeros(shard_shape, dtype), d) for d in devices],
      devices,
  )


def _staging_shardings(
    leaves: Sequence[LeafSpec], sample: Any, width: int
) -> tuple[Any, ...]:
  """Shardings that give EVERY source-role process a whole copy of each leaf.

  The source mesh is re-read as (process, local device): replicated over the
  process axis, split `width` ways over the local one. Each process then holds
  the leaf in exactly the canonical per-device pieces.
  """
  devices = sorted(
      sample.sharding.mesh.devices.reshape(-1),
      key=lambda d: (d.process_index, d.id),
  )
  processes = sorted({d.process_index for d in devices})
  rows = [[d for d in devices if d.process_index == p] for p in processes]
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


def _stage(
    leaves: Sequence[LeafSpec],
    source: Mapping[str, Any],
    devices: Sequence[Any],
) -> list[Any]:
  """Puts a chunk into the canonical layout; returns one payload per leaf.

  A collective when the source role spans processes: every one of them must
  call it.
  """
  arrays = [source[leaf.name] for leaf in leaves]
  width = len(devices)
  if all(a.is_fully_addressable for a in arrays):
    staged = [
        jax.device_put(a, _canonical_sharding(leaf, devices))
        for leaf, a in zip(leaves, arrays)
    ]
  else:
    shardings = _staging_shardings(leaves, arrays[0], width)
    staged = list(_resharder(shardings)(tuple(arrays)))
  jax.block_until_ready(staged)
  payloads = []
  for leaf, array in zip(leaves, staged):
    # Refuse rather than assume: if staging handed this process's devices
    # other slices than the canonical view, shards would arrive at the wrong
    # offsets -- silent corruption, not a crash.
    canonical = _canonical_sharding(leaf, devices)
    if array.sharding.addressable_devices_indices_map(
        leaf.shape
    ) != canonical.addressable_devices_indices_map(leaf.shape):
      raise RuntimeError(
          f"staging {leaf.name!r} {leaf.shape} gave this process's devices"
          " different slices than the canonical layout"
      )
    by_device = {s.device: s.data for s in array.addressable_shards}
    payloads.append(_as_payload([by_device[d] for d in devices], devices))
  return payloads


def _install_chunk(
    leaves: Sequence[LeafSpec],
    payloads: Sequence[Any],
    devices: Sequence[Any],
    sink: Sink,
) -> None:
  installed = {}
  for leaf, payload in zip(leaves, payloads):
    canonical = jax.make_array_from_single_device_arrays(
        leaf.shape,
        _canonical_sharding(leaf, devices),
        [payload.addressable_data(k)[0] for k in range(len(devices))],
        dtype=jnp.dtype(leaf.dtype),
    )
    installed[leaf.name] = jax.device_put(
        canonical, sink.target_sharding(leaf.name)
    )
  jax.block_until_ready(list(installed.values()))
  sink.install(installed)


def execute(
    plan: TransferPlan,
    *,
    source: Mapping[str, Any] | None = None,
    sink: Sink | None = None,
) -> dict[str, float]:
  """Runs this process's part of `plan`. Blocks until the plan is done.

  Every participant must call this with an equal plan, concurrently.

  Args:
    plan: The shared plan.
    source: `{leaf name: array}` on a source-role process; arrays must match
      the plan's shapes and dtypes. They are read, never donated or deleted.
    sink: Where a destination process installs what it receives.

  Returns:
    Wall-clock seconds spent per stage and the bytes this process received.
  """
  check_plan(plan, source=source, sink=sink)
  me = jax.process_index()
  devices = local_devices()
  is_source = me in plan.source_processes
  receive_round = next(
      (
          r
          for r, pairs in enumerate(plan.rounds)
          if any(dst == me for _, dst in pairs)
      ),
      None,
  )
  world_devices = _participant_devices(plan)
  device_ids = tuple(d.id for d in world_devices)
  movers = [
      _mover(_device_pairs(pairs, world_devices, plan.width), device_ids)
      for pairs in plan.rounds
  ]
  timings = {"stage": 0.0, "move": 0.0, "install": 0.0}
  received = 0
  started = time.perf_counter()
  for chunk in plan.chunks:
    leaves = [plan.leaves[i] for i in chunk]
    t0 = time.perf_counter()
    if is_source:
      payloads = tuple(_stage(leaves, source, devices))
    else:
      if receive_round is not None:
        sink.release([leaf.name for leaf in leaves])
      payloads = tuple(_zero_payload(leaf, devices) for leaf in leaves)
    t1 = time.perf_counter()
    for index, mover in enumerate(movers):
      moved = mover(payloads)
      jax.block_until_ready(moved)
      if index == receive_round:
        # From here on this process relays what it received.
        payloads = moved
      del moved
    t2 = time.perf_counter()
    if receive_round is not None:
      _install_chunk(leaves, payloads, devices, sink)
      received += sum(leaf.nbytes for leaf in leaves)
    del payloads
    t3 = time.perf_counter()
    timings["stage"] += t1 - t0
    timings["move"] += t2 - t1
    timings["install"] += t3 - t2
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
