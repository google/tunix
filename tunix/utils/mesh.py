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

"""Shared mesh device allocation helpers.

Typical usage:

  allocations = allocate_named_mesh_device_slices([
      ("actor", 8),
      ("rollout", 4),
  ])

The keys are arbitrary mesh names chosen by the caller. The integer is the
number of devices that mesh should receive.
"""

import collections
import dataclasses
from typing import Any, Sequence

from absl import logging
import jax
import numpy as np
from tunix.utils import topology

MeshRequirement = tuple[str, int]


def create_mesh(
    axis_shapes: tuple[int, ...],
    axis_names: tuple[str, ...],
    devices: Sequence[Any] | None = None,
):
  """Builds a JAX mesh from parsed axis metadata."""
  if len(axis_shapes) != len(axis_names):
    raise ValueError(
        f"mesh.shape {axis_shapes} and mesh.axis_names {axis_names} "
        "must have the same length."
    )

  num_devices = len(devices) if devices is not None else jax.device_count()
  required_devices = int(np.prod(axis_shapes))
  if required_devices > num_devices:
    raise ValueError(
        f"Mesh shape {axis_shapes} requires {required_devices} devices, "
        f"but found {num_devices}."
    )
  if devices is not None:
    if required_devices != num_devices:
      raise ValueError(
          f"Mesh shape {axis_shapes} requires {required_devices} devices, "
          f"but was assigned {num_devices}."
      )
    return jax.sharding.Mesh(
        np.array(list(devices)).reshape(axis_shapes),
        axis_names,
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )
  return jax.make_mesh(
      axis_shapes,
      axis_names,
      axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
  )


@dataclasses.dataclass(frozen=True)
class CoordTopology:
  """Normalized coord metadata for a device pool.

  Attributes:
    coord_to_device: Mapping from physical coords to device objects.
    all_coords: Normalized coord tuples for all devices.
    num_dims: Number of coord dimensions.
    max_shape: Bounding-box shape of the device pool.
  """

  coord_to_device: dict[tuple[int, ...], Any]
  all_coords: tuple[tuple[int, ...], ...]
  num_dims: int
  max_shape: tuple[int, ...]
  chip_coord_to_coords: dict[tuple[int, ...], tuple[tuple[int, ...], ...]]


@dataclasses.dataclass(frozen=True)
class DeviceAllocationState:
  """Tracks the remaining device pool across sequential mesh allocations.

  This state object exists so `allocate_devices()` can be the lowest-level
  public API while still supporting multi-mesh allocation. Callers that only
  need one mesh can pass `devices=` directly to `allocate_devices()`. Callers
  that need multiple meshes can create state once and repeatedly allocate from
  it, which is exactly what `allocate_named_mesh_device_slices()` does.

  Attributes:
    remaining_devices: Flat view of devices that have not yet been assigned.
    remaining_host_groups: Optional per-host buckets used by the host-aware
      fallback path. This becomes `None` once allocation falls back to a purely
      flat remaining-device pool.
    full_devices_per_host: Original per-host capacity derived from host groups.
    host_bound_shape: Per-host physical topology shape, such as `(2, 2, 1)`.
    host_bound_device_count: Device count implied by `host_bound_shape`.
    total_device_count: Size of the original device pool.
    used_device_count: Number of devices already assigned.
  """

  remaining_devices: tuple[Any, ...]
  remaining_host_groups: tuple[tuple[Any, ...], ...] | None
  full_devices_per_host: int
  host_bound_shape: tuple[int, ...] | None
  host_bound_device_count: int | None
  total_device_count: int
  used_device_count: int = 0


def device_attr(device: Any, attr_name: str) -> Any:
  """Returns a raw device attribute, calling it first if JAX exposes it lazily.

  Args:
    device: A JAX device or test double.
    attr_name: Attribute name such as "coords" or "process_index".

  Returns:
    The attribute value, or None if the attribute does not exist.
  """
  value = getattr(device, attr_name, None)
  return value() if callable(value) else value


def device_host_key(device: Any) -> tuple[Any, ...] | None:
  """Returns a stable host grouping key for topology-aware allocation.

  Args:
    device: A JAX device or test double.

  Returns:
    A tuple of (slice_id, task_id) when that metadata is available, otherwise
    None.
  """
  task_id = None
  for attr_name in ("logical_task", "task_id", "process_index"):
    task_id = device_attr(device, attr_name)
    if task_id is not None:
      break
  if task_id is None:
    return None

  slice_id = None
  for attr_name in ("slice_index", "slice"):
    slice_id = device_attr(device, attr_name)
    if slice_id is not None:
      break
  return (slice_id, task_id)


def device_slice_id(device: Any) -> Any:
  """Returns the slice identifier when the runtime exposes one.

  This is intentionally narrower than `device_host_key()`: it captures only the
  slice boundary, not the host/task within that slice. Slice-aware allocation
  uses this to prefer satisfying a mesh from one slice before spilling into the
  next slice.
  """
  for attr_name in ("slice_index", "slice"):
    slice_id = device_attr(device, attr_name)
    if slice_id is not None:
      return slice_id
  return None


def device_mesh_coords(device: Any) -> tuple[int, ...] | None:
  """Returns physical mesh coordinates for topology-aware allocation.

  Args:
    device: A JAX device or test double.

  Returns:
    A tuple like (x, y, z) or (x, y, z, core) when the runtime exposes device
    coordinates, otherwise None.
  """
  coords = device_attr(device, "coords")
  if coords is None:
    return None

  coords = tuple(coords)
  if not coords:
    return None

  normalized_coords = tuple(int(coord) for coord in coords)
  core_on_chip = device_attr(device, "core_on_chip")
  if core_on_chip is None:
    return normalized_coords
  return normalized_coords + (int(core_on_chip),)


def infer_core_on_chip_count(devices: Sequence[Any]) -> int | None:
  """Returns the per-chip core count when the runtime exposes it consistently."""
  chip_to_cores = collections.defaultdict(set)
  saw_any_core = False

  for device in devices:
    coords = device_attr(device, "coords")
    core_on_chip = device_attr(device, "core_on_chip")
    if coords is None:
      return None
    if core_on_chip is None:
      continue
    saw_any_core = True
    chip_to_cores[tuple(int(coord) for coord in coords)].add(int(core_on_chip))

  if not saw_any_core:
    return None

  core_counts = {len(core_ids) for core_ids in chip_to_cores.values()}
  if len(core_counts) != 1:
    return None
  return next(iter(core_counts))


def summarize_devices_for_logging(devices: Sequence[Any]) -> list[dict[str, Any]]:
  """Builds compact log-friendly summaries for a device list.

  Args:
    devices: Devices to summarize.

  Returns:
    A list of dictionaries containing device id, coords, and inferred host key.
  """
  summaries = []
  for device in devices:
    summaries.append({
        "id": device_attr(device, "id"),
        "coords": device_mesh_coords(device),
        "host": device_host_key(device),
    })
  return summaries


def summarize_devices_for_debug_logging(
    devices: Sequence[Any],
    limit: int = 16,
) -> list[dict[str, Any]]:
  """Builds richer device summaries for topology debugging.

  Args:
    devices: Devices to summarize.
    limit: Maximum number of devices to include.

  Returns:
    A list of dictionaries with raw device topology metadata.
  """
  summaries = []
  for device in devices[:limit]:
    summaries.append({
        "id": device_attr(device, "id"),
        "coords": device_attr(device, "coords"),
        "core_on_chip": device_attr(device, "core_on_chip"),
        "process_index": device_attr(device, "process_index"),
        "logical_task": device_attr(device, "logical_task"),
        "task_id": device_attr(device, "task_id"),
        "slice_index": device_attr(device, "slice_index"),
        "slice": device_attr(device, "slice"),
        "host": device_host_key(device),
    })
  return summaries


def summarize_host_groups_for_logging(devices: Sequence[Any]) -> dict[tuple[Any, ...], int]:
  """Summarizes device counts per derived host key for debug logging."""
  host_counts = collections.Counter()
  for device in devices:
    host_key = device_host_key(device)
    host_counts[host_key] += 1
  return dict(sorted(host_counts.items(), key=lambda item: str(item[0])))


def group_devices_by_slice(devices: Sequence[Any]) -> list[list[Any]] | None:
  """Groups devices by slice while preserving first-seen slice order.

  Returns `None` when slice metadata is unavailable for any device. The order of
  groups matches the first appearance of each slice in `devices`, which lets the
  allocator prefer earlier slices before spilling into later ones.
  """
  slice_to_devices = {}
  for device in devices:
    slice_id = device_slice_id(device)
    if slice_id is None:
      return None
    slice_to_devices.setdefault(slice_id, []).append(device)
  return list(slice_to_devices.values())


def group_devices_by_host(devices: Sequence[Any]) -> list[list[Any]] | None:
  """Groups devices by host/task when that metadata is available.

  Args:
    devices: Candidate devices to partition.

  Returns:
    A list of equal-sized per-host device lists, or None if host metadata is
    missing or inconsistent.
  """
  host_to_devices = {}
  for device in devices:
    host_key = device_host_key(device)
    if host_key is None:
      return None
    host_to_devices.setdefault(host_key, []).append(device)

  host_sizes = {len(host_devices) for host_devices in host_to_devices.values()}
  if len(host_sizes) != 1:
    logging.warning(
        "Falling back to flat device allocation because host sizes differ: %s",
        sorted(host_sizes),
    )
    return None
  return list(host_to_devices.values())


def host_mesh_shape(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Returns the per-host physical box shape when coords are available.

  Args:
    devices: Devices spanning one or more hosts.

  Returns:
    The shape of one host in physical coords, such as (2, 2, 1), or None when
    it cannot be inferred reliably.
  """
  host_to_coords = collections.defaultdict(list)
  for device in devices:
    host_key = device_host_key(device)
    coords = device_mesh_coords(device)
    if host_key is None or coords is None:
      return None
    host_to_coords[host_key].append(coords)

  host_shapes = set()
  for coords_list in host_to_coords.values():
    ndim = len(coords_list[0])
    mins = tuple(min(coords[i] for coords in coords_list) for i in range(ndim))
    maxs = tuple(max(coords[i] for coords in coords_list) for i in range(ndim))
    shape = tuple(max_coord - min_coord + 1 for min_coord, max_coord in zip(mins, maxs))
    if int(np.prod(shape)) != len(coords_list):
      return None
    host_shapes.add(shape)

  if len(host_shapes) != 1:
    return None
  return next(iter(host_shapes))


def get_coord_topology(devices: Sequence[Any]) -> CoordTopology | None:
  """Builds normalized coord metadata for a device pool.

  Args:
    devices: Candidate devices to inspect.

  Returns:
    A CoordTopology describing the device coords and overall bounding box, or
    None when the devices do not expose a consistent coord layout.
  """
  if not devices:
    return None

  coord_to_device = {}
  all_coords = []
  for device in devices:
    coords = device_mesh_coords(device)
    if coords is None:
      logging.info(
          "Coord topology unavailable because device lacks coords: %s",
          summarize_devices_for_debug_logging([device]),
      )
      return None
    if all_coords and len(coords) != len(all_coords[0]):
      logging.info(
          "Coord topology unavailable because coord rank differs: existing_rank=%d device=%s",
          len(all_coords[0]),
          summarize_devices_for_debug_logging([device]),
      )
      return None
    if coords in coord_to_device:
      logging.info(
          "Coord topology unavailable because multiple devices share coords %s: %s",
          coords,
          summarize_devices_for_debug_logging([coord_to_device[coords], device]),
      )
      return None
    coord_to_device[coords] = device
    all_coords.append(coords)

  num_dims = len(all_coords[0])
  chip_coord_to_coords = collections.defaultdict(list)
  for coords in all_coords:
    chip_coord_to_coords[coords[:-1]].append(coords)
  max_shape = tuple(
      max(coords[dim] for coords in all_coords)
      - min(coords[dim] for coords in all_coords)
      + 1
      for dim in range(num_dims)
  )
  return CoordTopology(
      coord_to_device=coord_to_device,
      all_coords=tuple(all_coords),
      num_dims=num_dims,
      max_shape=max_shape,
      chip_coord_to_coords={
          chip_coord: tuple(sorted(group_coords))
          for chip_coord, group_coords in chip_coord_to_coords.items()
      },
  )


def candidate_uses_whole_chips(
    coord_topology: CoordTopology,
    candidate_coords: Sequence[tuple[int, ...]],
) -> bool:
  """Returns whether a candidate includes all logical devices for each chip.

  When multiple logical devices share the same physical chip coordinates, a
  valid Pathways subslice must include either all of them or none of them.
  This rejects candidates that split `core_on_chip` siblings across meshes.
  """
  if coord_topology.num_dims <= 1:
    return True

  selected_coords = set(candidate_coords)
  selected_chip_coords = {coords[:-1] for coords in selected_coords}
  for chip_coord in selected_chip_coords:
    chip_group = coord_topology.chip_coord_to_coords.get(chip_coord, ())
    if any(coords not in selected_coords for coords in chip_group):
      return False
  return True


def known_host_mesh_shape(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Returns known host bounds from static topology metadata when available.

  Args:
    devices: Devices from a single TPU slice.

  Returns:
    A known per-host physical bound such as (1, 1, 1) or (2, 2, 1), or None if
    the accelerator family is unknown.
  """
  bounds = topology.infer_chips_per_host_bounds(devices)
  if bounds is None:
    return None

  coords = device_mesh_coords(devices[0]) if devices else None
  if coords is None:
    return None

  if len(coords) == len(bounds):
    return bounds

  if len(coords) == len(bounds) + 1:
    core_count = infer_core_on_chip_count(devices)
    if core_count is None:
      return None
    return bounds + (core_count,)

  return None


def resolve_per_host_mesh_shape(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Resolves per-host shape and validates inferred vs known topology.

  Args:
    devices: Devices spanning one or more hosts.

  Returns:
    The inferred per-host shape when available, otherwise the known static host
    bounds.

  Raises:
    ValueError: If runtime-inferred host shape disagrees with known static host
      bounds for the device family.
  """
  inferred_shape = host_mesh_shape(devices)
  static_shape = known_host_mesh_shape(devices)
  if (
      inferred_shape is not None
      and static_shape is not None
      and inferred_shape != static_shape
  ):
    raise ValueError(
        "Inferred per-host device shape "
        f"{inferred_shape} does not match known host bounds {static_shape}."
    )
  return inferred_shape or static_shape


def _divisors(value: int) -> list[int]:
  divisors = set()
  for candidate in range(1, int(np.sqrt(value)) + 1):
    if value % candidate == 0:
      divisors.add(candidate)
      divisors.add(value // candidate)
  return sorted(divisors)


def _enumerate_box_shapes(
    required_devices: int,
    max_shape: tuple[int, ...],
) -> list[tuple[int, ...]]:
  """Enumerates box shapes whose volume matches the requested device count."""
  shapes = []
  num_dims = len(max_shape)

  def build(dim_index: int, remaining: int, prefix: tuple[int, ...]):
    if dim_index == num_dims - 1:
      if remaining <= max_shape[dim_index]:
        shapes.append(prefix + (remaining,))
      return

    for size in _divisors(remaining):
      if size > max_shape[dim_index]:
        continue
      build(dim_index + 1, remaining // size, prefix + (size,))

  build(0, required_devices, ())
  return shapes


def _coord_box_score(
    start: tuple[int, ...],
    shape: tuple[int, ...],
    host_shape: tuple[int, ...] | None,
) -> tuple[Any, ...]:
  """Builds a lexicographic sort key for candidate coord boxes.

  The returned tuple is ordered so Python tuple comparison implements the
  desired ranking policy directly:

  1. Prefer host-aligned boxes when host_shape is known.
  2. Prefer boxes with a smaller maximum dimension.
  3. Prefer more compact overall shapes.
  4. Prefer earlier start coordinates as a stable tiebreaker.

  Args:
    start: Candidate box origin.
    shape: Candidate box shape.
    host_shape: Per-host physical shape such as (2, 2, 1).

  Returns:
    A tuple sort key suitable for lexicographic comparison.
  """
  chip_host_alignment = 1
  full_host_alignment = 1
  if host_shape is not None:
    chip_dims = min(3, len(shape), len(host_shape))
    chip_aligned = all(
        start[dim] % host_shape[dim] == 0
        and shape[dim] % host_shape[dim] == 0
        for dim in range(chip_dims)
        if host_shape[dim] > 1
    )
    fully_aligned = all(
        start[dim] % host_shape[dim] == 0
        and shape[dim] % host_shape[dim] == 0
        for dim in range(len(shape))
        if host_shape[dim] > 1
    )
    chip_host_alignment = 0 if chip_aligned else 1
    full_host_alignment = 0 if fully_aligned else 1
  return (
      chip_host_alignment,
      full_host_alignment,
      max(shape),
      tuple(sorted(shape, reverse=True)),
      tuple(-dim for dim in shape),
      start,
  )


def select_best_candidate_coords(
    candidate_boxes: Sequence[
        tuple[tuple[int, ...], tuple[int, ...], Sequence[tuple[int, ...]]]
    ],
    host_shape: tuple[int, ...] | None,
) -> list[tuple[int, ...]] | None:
  """Selects the best candidate coord box using the mesh heuristic.

  Args:
    candidate_boxes: Sequence of (start, shape, candidate_coords) tuples.
      `start` is the box origin, `shape` is the physical box shape, and
      `candidate_coords` are the device coords inside that box.
    host_shape: Per-host physical shape such as (2, 2, 1), used to prefer
      host-aligned boxes when available.

  Returns:
    The candidate coord list for the best-ranked box, or None when there are no
    candidates.

  Notes:
    Candidate boxes are ranked by `_coord_box_score()`, which uses a
    lexicographic sort key instead of a single numeric score. This makes the
    priority order explicit and avoids arbitrary weighting between ranking
    factors.
  """
  best_candidate_coords = None
  best_score = None
  for start, shape, candidate_coords in candidate_boxes:
    score = _coord_box_score(start, shape, host_shape)
    if best_score is None or score < best_score:
      best_score = score
      best_candidate_coords = list(candidate_coords)
  return best_candidate_coords


def find_candidate_coord_boxes(
    coord_topology: CoordTopology,
    required_devices: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]]:
  """Finds contiguous candidate coord boxes for a requested device count.

  Args:
    coord_topology: Normalized coord metadata for the candidate device pool.
    required_devices: Number of devices needed for one mesh.

  Returns:
    A list of (start, shape, candidate_coords) tuples representing contiguous
    coord boxes whose volume matches required_devices.

  Notes:
    This function only enumerates valid contiguous boxes that exist in the
    current device pool. It does not choose among them; ranking is handled by
    `select_best_candidate_coords()`.
  """
  candidate_boxes = []
  for shape in _enumerate_box_shapes(required_devices, coord_topology.max_shape):
    for start in coord_topology.coord_to_device:
      candidate_coords = []
      for offset in np.ndindex(shape):
        candidate_coord = tuple(
            start[dim] + offset[dim] for dim in range(coord_topology.num_dims)
        )
        if candidate_coord not in coord_topology.coord_to_device:
          break
        candidate_coords.append(candidate_coord)
      else:
        if candidate_uses_whole_chips(coord_topology, candidate_coords):
          candidate_boxes.append((start, shape, tuple(candidate_coords)))
  return candidate_boxes


def find_host_aligned_candidate_coord_boxes(
    coord_topology: CoordTopology,
    required_devices: int,
    host_shape: tuple[int, ...],
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]]:
  """Finds contiguous candidate boxes that exactly respect host bounds.

  Args:
    coord_topology: Normalized coord metadata for the candidate device pool.
    required_devices: Number of devices needed for one mesh.
    host_shape: Known per-host physical shape such as (2, 2, 1) or
      (2, 2, 1, 2).

  Returns:
    A list of valid coord boxes whose shape is an exact multiple of host_shape.
  """
  if len(host_shape) != coord_topology.num_dims:
    return []

  host_volume = int(np.prod(host_shape))
  if host_volume <= 0 or required_devices % host_volume != 0:
    return []

  host_grid_shape = tuple(
      coord_topology.max_shape[dim] // host_shape[dim]
      for dim in range(coord_topology.num_dims)
  )
  required_host_boxes = required_devices // host_volume

  candidate_boxes = []
  for host_box_shape in _enumerate_box_shapes(required_host_boxes, host_grid_shape):
    physical_shape = tuple(
        host_box_shape[dim] * host_shape[dim]
        for dim in range(coord_topology.num_dims)
    )
    for start in coord_topology.coord_to_device:
      if any(
          start[dim] % host_shape[dim] != 0
          for dim in range(coord_topology.num_dims)
          if host_shape[dim] > 1
      ):
        continue

      candidate_coords = []
      for offset in np.ndindex(physical_shape):
        candidate_coord = tuple(
            start[dim] + offset[dim] for dim in range(coord_topology.num_dims)
        )
        if candidate_coord not in coord_topology.coord_to_device:
          break
        candidate_coords.append(candidate_coord)
      else:
        if candidate_uses_whole_chips(coord_topology, candidate_coords):
          candidate_boxes.append((start, physical_shape, tuple(candidate_coords)))
  return candidate_boxes


def allocate_devices_by_coords(
    devices: Sequence[Any],
    required_devices: int,
) -> list[Any] | None:
  """Allocates a contiguous physical box of devices when coords exist.

  Args:
    devices: Candidate devices to allocate from.
    required_devices: Number of devices needed for one mesh.

  Returns:
    A list of devices forming the best contiguous physical box, or None if the
    devices do not expose usable coordinates.

  Notes:
    This helper runs in three stages:

    1. Build normalized coord metadata with `get_coord_topology()`.
    2. Enumerate valid contiguous candidate boxes with
       `find_candidate_coord_boxes()`.
    3. Rank those candidates with `select_best_candidate_coords()` and map the
       winning coords back to device objects.
  """
  coord_topology = get_coord_topology(devices)
  if coord_topology is None:
    return None
  per_host_shape = resolve_per_host_mesh_shape(devices)

  candidate_boxes = []
  if per_host_shape is not None:
    candidate_boxes = find_host_aligned_candidate_coord_boxes(
        coord_topology,
        required_devices,
        per_host_shape,
    )
  if not candidate_boxes:
    candidate_boxes = find_candidate_coord_boxes(coord_topology, required_devices)

  best_candidate_coords = select_best_candidate_coords(
      candidate_boxes,
      per_host_shape,
  )
  if best_candidate_coords is None:
    return None

  selected_coords = set(best_candidate_coords)
  return [
      device
      for device in devices
      if device_mesh_coords(device) in selected_coords
  ]


def _create_device_allocation_state(
    devices: Sequence[Any] | None = None,
    *,
    log_summary: bool = True,
) -> DeviceAllocationState:
  """Builds reusable allocator state for one or more mesh allocations.

  This is intentionally private because callers should not need to understand
  the allocator internals to request one mesh. The public entry point is
  `allocate_devices()`, which accepts either raw `devices` for one-shot use or
  an existing `allocation_state` for incremental use.
  """
  all_devices = tuple(jax.devices() if devices is None else devices)
  if log_summary:
    logging.info(
        "Mesh allocator raw device sample: %s",
        summarize_devices_for_debug_logging(all_devices),
    )
    logging.info(
        "Mesh allocator derived host groups: %s",
        summarize_host_groups_for_logging(all_devices),
    )
  remaining_host_groups = group_devices_by_host(all_devices)
  full_devices_per_host = (
      len(remaining_host_groups[0]) if remaining_host_groups else 0
  )
  host_bound_shape = _infer_host_bound_shape(all_devices)
  host_bound_device_count = _infer_host_bound_device_count(
      host_bound_shape,
      full_devices_per_host,
  )
  if remaining_host_groups and (
      host_bound_shape is None or not host_bound_device_count
  ):
    raise ValueError(
        "Host-group allocation requires an inferable host-bound shape and "
        "device count."
    )
  return DeviceAllocationState(
      remaining_devices=all_devices,
      remaining_host_groups=(
          tuple(tuple(group) for group in remaining_host_groups)
          if remaining_host_groups
          else None
      ),
      full_devices_per_host=full_devices_per_host,
      host_bound_shape=host_bound_shape,
      host_bound_device_count=host_bound_device_count,
      total_device_count=len(all_devices),
  )


def _allocate_devices_from_pool(
    required_devices: int,
    remaining_devices: list[Any],
    remaining_host_groups: list[list[Any]] | None,
    full_devices_per_host: int,
    host_bound_shape: tuple[int, ...] | None,
    host_bound_device_count: int | None,
    mesh_name: str,
) -> tuple[list[Any], list[Any], list[list[Any]] | None]:
  """Allocates one mesh from a concrete device pool without slice policy.

  This helper contains the pool-local allocation strategy used after any
  slice-level decision has already been made.
  """
  assigned_devices = allocate_devices_by_coords(remaining_devices, required_devices)
  if assigned_devices is not None:
    remaining_devices = _remove_devices_by_identity(
        remaining_devices,
        assigned_devices,
    )
    remaining_host_groups = None
    return assigned_devices, remaining_devices, remaining_host_groups

  if remaining_host_groups:
    assigned_devices, remaining_host_groups = _allocate_from_host_groups(
        remaining_host_groups,
        required_devices,
        full_devices_per_host,
        host_bound_shape,
        host_bound_device_count or 0,
        mesh_name,
    )
    remaining_devices = _remove_devices_by_identity(
        remaining_devices,
        assigned_devices,
    )
    return assigned_devices, remaining_devices, remaining_host_groups

  if required_devices > len(remaining_devices):
    raise ValueError(
        f"Mesh allocation requires {required_devices} devices for {mesh_name}, "
        f"but only {len(remaining_devices)} remain available."
    )
  assigned_devices = remaining_devices[:required_devices]
  remaining_devices = remaining_devices[required_devices:]
  return assigned_devices, remaining_devices, remaining_host_groups


def allocate_devices(
    required_devices: int,
    devices: Sequence[Any] | None = None,
    *,
    mesh_name: str = "allocated_mesh",
    allocation_state: DeviceAllocationState | None = None,
    return_state: bool = False,
) -> list[Any] | tuple[list[Any], DeviceAllocationState]:
  """Allocates devices for a single mesh request.

  This is the lowest-level public allocation API. It handles exactly one mesh
  request and applies the allocator policy in priority order:

  1. Prefer a contiguous coord-aligned box when device coords are available.
  2. Otherwise, use host-aware allocation without illegally breaking host
     topology.
  3. Otherwise, fall back to a flat prefix of the remaining devices.

  There are two intended calling modes:

  1. One-shot allocation: pass `devices=` and receive a single allocation.
  2. Incremental allocation: pass `allocation_state=` and, when
     `return_state=True`, receive the updated remaining pool for the next call.

  `allocate_named_mesh_device_slices()` is implemented as a thin loop around
  this function.

  Args:
    required_devices: Number of devices to allocate for this mesh.
    devices: Raw device pool for one-shot use. Mutually exclusive with
      `allocation_state`.
    mesh_name: Name used only for diagnostics and error messages.
    allocation_state: Existing state for incremental allocation.
    return_state: Whether to return the updated allocation state alongside the
      assigned devices.

  Returns:
    Either the assigned device list, or `(assigned_devices, next_state)` when
    `return_state=True`.

  Raises:
    ValueError: If both `devices` and `allocation_state` are provided, or if
      the request cannot be satisfied from the remaining device pool.
  """
  if devices is not None and allocation_state is not None:
    raise ValueError(
        "Pass either devices or allocation_state to allocate_devices, not both."
    )

  owns_state = allocation_state is None
  state = allocation_state or _create_device_allocation_state(devices)
  remaining_devices = list(state.remaining_devices)
  remaining_host_groups = (
      [list(group) for group in state.remaining_host_groups]
      if state.remaining_host_groups
      else None
  )
  assigned_devices = None

  slice_groups = group_devices_by_slice(remaining_devices)
  if slice_groups and len(slice_groups) > 1:
    # Prefer staying within one slice when a single slice can satisfy the whole
    # request. This avoids accidental cross-slice meshes when slice metadata is
    # available.
    for slice_devices in slice_groups:
      if len(slice_devices) < required_devices:
        continue
      slice_state = _create_device_allocation_state(
          slice_devices,
          log_summary=False,
      )
      assigned_devices, _, _ = _allocate_devices_from_pool(
          required_devices,
          list(slice_state.remaining_devices),
          (
              [list(group) for group in slice_state.remaining_host_groups]
              if slice_state.remaining_host_groups
              else None
          ),
          slice_state.full_devices_per_host,
          slice_state.host_bound_shape,
          slice_state.host_bound_device_count,
          mesh_name,
      )
      remaining_devices = _remove_devices_by_identity(
          remaining_devices,
          assigned_devices,
      )
      remaining_host_groups = None
      break

    # If no single slice is large enough, consume slices in order. This makes a
    # cross-slice mesh grow by exhausting one slice before spilling into the
    # next one.
    if assigned_devices is None and len(remaining_devices) >= required_devices:
      slice_order = [device_slice_id(group[0]) for group in slice_groups]
      assigned_devices = []
      remaining_required = required_devices
      for slice_id in slice_order:
        if remaining_required == 0:
          break
        current_slice_devices = [
            device for device in remaining_devices if device_slice_id(device) == slice_id
        ]
        if not current_slice_devices:
          continue
        slice_request = min(remaining_required, len(current_slice_devices))
        slice_state = _create_device_allocation_state(
            current_slice_devices,
            log_summary=False,
        )
        partial_devices, _, _ = _allocate_devices_from_pool(
            slice_request,
            list(slice_state.remaining_devices),
            (
                [list(group) for group in slice_state.remaining_host_groups]
                if slice_state.remaining_host_groups
                else None
            ),
            slice_state.full_devices_per_host,
            slice_state.host_bound_shape,
            slice_state.host_bound_device_count,
            mesh_name,
        )
        assigned_devices.extend(partial_devices)
        remaining_devices = _remove_devices_by_identity(
            remaining_devices,
            partial_devices,
        )
        remaining_required -= len(partial_devices)
      remaining_host_groups = None

  if assigned_devices is None:
    assigned_devices, remaining_devices, remaining_host_groups = _allocate_devices_from_pool(
        required_devices,
        remaining_devices,
        remaining_host_groups,
        state.full_devices_per_host,
        state.host_bound_shape,
        state.host_bound_device_count,
        mesh_name,
    )

  next_state = dataclasses.replace(
      state,
      remaining_devices=tuple(remaining_devices),
      remaining_host_groups=(
          tuple(tuple(group) for group in remaining_host_groups)
          if remaining_host_groups
          else None
      ),
      used_device_count=state.used_device_count + len(assigned_devices),
  )
  logging.info(
      "Allocated devices for %s: %s",
      mesh_name,
      summarize_devices_for_logging(assigned_devices),
  )

  if owns_state and not return_state:
    unused_device_count = next_state.total_device_count - next_state.used_device_count
    if unused_device_count > 0:
      logging.warning(
        "Mesh allocation used %d of %d devices; %d devices remain unused.",
        next_state.used_device_count,
        next_state.total_device_count,
        unused_device_count,
      )

  if return_state:
    return assigned_devices, next_state
  return assigned_devices


def _remove_devices_by_identity(
    devices: Sequence[Any],
    assigned_devices: Sequence[Any],
) -> list[Any]:
  assigned_device_ids = {id(device) for device in assigned_devices}
  return [device for device in devices if id(device) not in assigned_device_ids]


def _infer_host_bound_device_count(
    host_bound_shape: tuple[int, ...] | None,
    full_devices_per_host: int,
) -> int | None:
  """Infers the smallest host-aligned device-count unit when possible."""
  if host_bound_shape is None or full_devices_per_host <= 0:
    return None

  host_bound_device_count = int(np.prod(host_bound_shape))
  if host_bound_device_count <= 0:
    return None
  if host_bound_device_count > full_devices_per_host:
    return None
  return host_bound_device_count


def _infer_host_bound_shape(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Infers the per-host bound shape when topology metadata is available."""
  return resolve_per_host_mesh_shape(devices)


def _allocate_devices_within_host_group(
    host_devices: Sequence[Any],
    required_devices: int,
    host_bound_shape: tuple[int, ...],
) -> list[Any] | None:
  """Allocates devices from one host bucket using coord-aware selection."""
  coord_topology = get_coord_topology(host_devices)
  if coord_topology is None:
    return None

  candidate_boxes = find_host_aligned_candidate_coord_boxes(
      coord_topology,
      required_devices,
      host_bound_shape,
  )
  if not candidate_boxes:
    candidate_boxes = find_candidate_coord_boxes(coord_topology, required_devices)

  best_candidate_coords = select_best_candidate_coords(
      candidate_boxes,
      host_bound_shape,
  )
  if best_candidate_coords is None:
    return None

  selected_coords = set(best_candidate_coords)
  return [
      device
      for device in host_devices
      if device_mesh_coords(device) in selected_coords
  ]


def _satisfies_host_bound_shape(
    host_devices: Sequence[Any],
    host_bound_shape: tuple[int, ...] | None,
    host_bound_device_count: int,
) -> bool:
  if host_bound_shape is None or host_bound_device_count <= 0:
    raise ValueError(
        "host_bound_shape and host_bound_device_count must be set for "
        "host-group allocation."
    )
  return (
      _allocate_devices_within_host_group(
          host_devices,
          len(host_devices),
          host_bound_shape,
      )
      is not None
  )


def _allocate_partial_host_group(
    host_groups: Sequence[Sequence[Any]],
    required_devices: int,
    host_bound_shape: tuple[int, ...],
    host_bound_device_count: int,
    mesh_name: str,
) -> tuple[list[Any], list[list[Any]] | None] | None:
  """Allocates a request from one host bucket if a compatible bucket exists.

  This helper deliberately does not merge fragments from different hosts. The
  policy is to satisfy a partial request from exactly one remaining host bucket
  and to keep the leftover from that same bucket only if the leftover still
  forms a host-valid shape. If taking the prefix would leave an invalid host
  fragment behind, this host bucket is skipped and the allocator tries the next
  one.
  """
  for host_index, host_devices in enumerate(host_groups):
    if len(host_devices) < required_devices:
      continue
    if not _satisfies_host_bound_shape(
        host_devices,
        host_bound_shape,
        host_bound_device_count,
    ):
      logging.info(
          "Skipping remaining host group for %s because %d devices do not "
          "satisfy inferred host-bound shape %s.",
          mesh_name,
          len(host_devices),
          host_bound_shape,
      )
      continue
    assigned_devices = list(host_devices[:required_devices])
    remaining_devices_for_host = list(host_devices[required_devices:])
    if remaining_devices_for_host and not _satisfies_host_bound_shape(
        remaining_devices_for_host,
        host_bound_shape,
        host_bound_device_count,
    ):
      logging.info(
          "Skipping remaining host group for %s because taking %d devices would "
          "leave %d devices that do not satisfy host-bound shape %s.",
          mesh_name,
          required_devices,
          len(remaining_devices_for_host),
          host_bound_shape,
      )
      continue
    remaining_host_groups = [list(group) for group in host_groups]
    if remaining_devices_for_host:
      remaining_host_groups[host_index] = remaining_devices_for_host
    else:
      del remaining_host_groups[host_index]
    return assigned_devices, remaining_host_groups or None
  return None


def _allocate_from_host_groups(
    host_groups: Sequence[Sequence[Any]],
    required_devices: int,
    full_devices_per_host: int,
    host_bound_shape: tuple[int, ...] | None,
    host_bound_device_count: int,
    mesh_name: str,
) -> tuple[list[Any], list[list[Any]] | None]:
  """Allocates from remaining per-host buckets while preserving leftovers.

  This path is used only after coord allocation fails.

  Why this exists:

  1. Some environments expose enough host metadata to preserve host boundaries
    even when a full coord topology is unavailable or unsuitable.
  2. We want to allow partial-host reuse across meshes, but only when the
    leftover fragment still has a valid host-bounded shape.
  3. We do not want to silently assemble one logical "host" out of unrelated
    fragments taken from multiple different hosts.

  Policy:

  1. Allocate the whole-host portion first when `required_devices` spans one or
    more full hosts.
  2. Allocate any remainder from exactly one remaining host bucket.
  3. Reject the request if that remainder cannot be taken without leaving an
    invalid host fragment behind.
  """
  if host_bound_shape is None or host_bound_device_count <= 0:
    raise ValueError(
        "Host-group allocation requires an inferable host-bound shape and "
        "device count."
    )
  if full_devices_per_host <= 0:
    raise ValueError(
        "Host-group allocation requires a positive full host device count."
    )

  remaining_host_groups = [list(group) for group in host_groups]
  assigned_devices = []

  required_full_hosts = required_devices // full_devices_per_host
  remainder_devices = required_devices % full_devices_per_host

  if required_full_hosts:
    full_host_indices = [
      index
      for index, host_devices in enumerate(remaining_host_groups)
      if len(host_devices) == full_devices_per_host
    ]
    if required_full_hosts > len(full_host_indices):
      raise ValueError(
          f"Mesh allocation requires {required_full_hosts} hosts for {mesh_name}, "
          f"but only {len(full_host_indices)} are available."
      )

    selected_host_indices = set(full_host_indices[:required_full_hosts])
    assigned_devices.extend([
        device
        for index, host_devices in enumerate(remaining_host_groups)
        if index in selected_host_indices
        for device in host_devices
    ])
    remaining_host_groups = [
        list(host_devices)
        for index, host_devices in enumerate(remaining_host_groups)
        if index not in selected_host_indices
    ]

  if remainder_devices:
    partial_allocation = _allocate_partial_host_group(
        remaining_host_groups,
        remainder_devices,
        host_bound_shape,
        host_bound_device_count,
        mesh_name,
    )
    if partial_allocation is None:
      raise ValueError(
          f"Mesh allocation for {mesh_name} requires {required_devices} devices, "
          f"but no remaining host group can satisfy the remaining {remainder_devices} devices."
      )
    partial_devices, remaining_host_groups = partial_allocation
    assigned_devices.extend(partial_devices)

  if not assigned_devices:
    partial_allocation = _allocate_partial_host_group(
        remaining_host_groups,
        required_devices,
        host_bound_shape,
        host_bound_device_count,
        mesh_name,
    )
    if partial_allocation is None:
      raise ValueError(
          f"Mesh allocation for {mesh_name} requires {required_devices} devices, "
          "but no remaining host has enough capacity to satisfy that request."
      )
    assigned_devices, remaining_host_groups = partial_allocation

  return assigned_devices, remaining_host_groups or None


def allocate_named_mesh_device_slices(
    mesh_requirements: Sequence[MeshRequirement],
    devices: Sequence[Any] | None = None,
) -> dict[str, list[Any]]:
  """Allocates device subsets for named meshes.

  This is a convenience wrapper over `allocate_devices()` for callers that want
  several named allocations from one shared device pool.

  The function builds one `DeviceAllocationState`, then calls
  `allocate_devices()` once per `(mesh_name, required_devices)` pair. That
  keeps the single-mesh allocation policy centralized in one public API instead
  of duplicating decision logic here.

  Args:
    mesh_requirements: Sequence of (mesh_name, required_devices) pairs.
      Example: [("actor", 8), ("rollout", 4)]. The mesh_name is only used for
      logging and as the key in the returned dictionary.
    devices: Optional explicit device list. When omitted, this uses
      jax.devices().

  Returns:
    A dictionary mapping each mesh name to the list of devices assigned to it.

  Raises:
    ValueError: If a requested mesh cannot be assigned enough devices or if a
      host-based allocation would split hosts illegally.
  """
  state = _create_device_allocation_state(devices)
  allocations = {}

  for mesh_name, required_devices in mesh_requirements:
    assigned_devices, state = allocate_devices(
        required_devices,
        mesh_name=mesh_name,
        allocation_state=state,
        return_state=True,
    )
    allocations[mesh_name] = assigned_devices

  unused_device_count = state.total_device_count - state.used_device_count
  if unused_device_count > 0:
    logging.warning(
        "Mesh allocation used %d of %d devices; %d devices remain unused.",
        state.used_device_count,
        state.total_device_count,
        unused_device_count,
    )
  logging.info(
      "Mesh device allocation: %s",
      {mesh_name: len(assigned_devices) for mesh_name, assigned_devices in allocations.items()},
  )
  return allocations
