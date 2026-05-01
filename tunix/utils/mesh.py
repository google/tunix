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
  ], allocation_policy="COMPACT")

The keys are arbitrary mesh names chosen by the caller. The integer is the
number of devices that mesh should receive.

Supported allocation policies:

* ``COMPACT``: pack meshes into the smallest fitting remaining coord region.
* ``PERFORMANCE``: prefer more cubical supported extracted shapes.
"""

import collections
import dataclasses
from typing import Any, Sequence

from absl import logging
import jax
import numpy as np
from tunix.utils import topology

MeshRequirement = tuple[str, int]
_COMPACT_ALLOCATION_POLICY = "COMPACT"
_PERFORMANCE_ALLOCATION_POLICY = "PERFORMANCE"
_SUPPORTED_ALLOCATION_POLICIES = {
  _COMPACT_ALLOCATION_POLICY,
  _PERFORMANCE_ALLOCATION_POLICY,
}


def create_mesh(
    axis_shapes: tuple[int, ...],
    axis_names: tuple[str, ...],
    devices: Sequence[Any] | None = None,
):
  """Builds a JAX mesh from parsed axis metadata.

  Args:
    axis_shapes: Mesh dimension sizes such as ``(2, 4)``.
    axis_names: Mesh axis names such as ``("data", "model")``.
    devices: Optional explicit device assignment. When omitted, Tunix uses the
      default JAX device set.

  Returns:
    A ``jax.sharding.Mesh`` with the requested logical shape.

  Raises:
    ValueError: If the axis metadata is inconsistent or the requested mesh
      shape does not match the available device count.
  """
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
    min_coords: Minimum coord on each axis for the device pool.
    max_shape: Bounding-box shape of the device pool.
  """

  coord_to_device: dict[tuple[int, ...], Any]
  all_coords: tuple[tuple[int, ...], ...]
  num_dims: int
  min_coords: tuple[int, ...]
  max_shape: tuple[int, ...]
  chip_coord_to_coords: dict[tuple[int, ...], tuple[tuple[int, ...], ...]]
  has_core_on_chip_dimension: bool


@dataclasses.dataclass(frozen=True)
class CoordRegion:
  """Axis-aligned coord region tracked across sequential allocations."""

  start: tuple[int, ...]
  shape: tuple[int, ...]


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
    full_devices_per_host: Original per-host capacity derived from host groups.
    host_bound_shape: Per-host physical topology shape, such as `(2, 2, 1)`.
    total_device_count: Size of the original device pool.
    remaining_coord_regions_by_slice: Remaining coord regions keyed by slice id.
      When slice metadata is absent, the key is `None`.
    full_devices_per_slice: Original per-slice capacity keyed by slice id. This
      lets later allocations tell whether a remaining slice is still whole or
      has already been partially consumed.
    used_device_count: Number of devices already assigned.
  """

  remaining_devices: tuple[Any, ...]
  full_devices_per_host: int
  host_bound_shape: tuple[int, ...] | None
  total_device_count: int
  allocation_policy: str = _COMPACT_ALLOCATION_POLICY
  remaining_coord_regions_by_slice: dict[Any, tuple[CoordRegion, ...]] | None = None
  full_devices_per_slice: dict[Any, int] | None = None
  used_device_count: int = 0


def _normalize_allocation_policy(allocation_policy: str | None) -> str:
  """Validates and normalizes the requested allocation policy."""
  normalized_policy = (
      _COMPACT_ALLOCATION_POLICY
      if allocation_policy is None
      else allocation_policy.upper()
  )
  if normalized_policy not in _SUPPORTED_ALLOCATION_POLICIES:
    raise ValueError(
        "allocation_policy must be one of "
        f"{sorted(_SUPPORTED_ALLOCATION_POLICIES)}, got {allocation_policy!r}."
    )
  return normalized_policy


def normalize_allocation_policy(allocation_policy: str | None) -> str:
  """Validates and normalizes a user-facing mesh allocation policy.

  Args:
    allocation_policy: Policy name provided by a caller or config. ``None``
      selects the default policy.

  Returns:
    The normalized policy string.

  Raises:
    ValueError: If the policy is unsupported.
  """
  return _normalize_allocation_policy(allocation_policy)


def device_attr(device: Any, attr_name: str) -> Any:
  """Returns a raw device attribute, calling it first if JAX exposes it lazily.

  Args:
    device: A JAX device or test double.
    attr_name: Attribute name such as "coords" or "process_index".

  Returns:
    The attribute value, or None if the attribute does not exist.
  """
  return topology._device_attr(device, attr_name)


def device_host_key(device: Any) -> tuple[Any, ...] | None:
  """Returns a stable host grouping key for topology-aware allocation.

  Args:
    device: A JAX device or test double.

  Returns:
    A tuple of (slice_id, task_id) when that metadata is available, otherwise
    None.
  """
  return topology._device_host_key(device)


def device_slice_id(device: Any) -> Any:
  """Returns the slice identifier when the runtime exposes one.

  This is intentionally narrower than `device_host_key()`: it captures only the
  slice boundary, not the host/task within that slice. Slice-aware allocation
  uses this to prefer satisfying a mesh from one slice before spilling into the
  next slice.
  """
  return device_attr(device, "slice_index")


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


def _pool_allows_2d_chip_coords(devices: Sequence[Any]) -> bool:
  """Returns whether raw 2D chip coords are valid for this device pool."""
  return _device_family(devices) in {"v5e", "v6e"}


def _canonical_device_chip_coords(
    device: Any,
    *,
    allow_2d: bool,
) -> tuple[int, int, int] | None:
  """Returns canonical 3D chip coords for one device when available."""
  parsed = topology._device_coords(device)
  if parsed is None:
    return None
  if len(parsed) == 2 and allow_2d:
    return topology._canonicalize_chip_shape_to_3d(parsed)
  if len(parsed) == 3:
    return parsed
  return None


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


def summarize_coord_regions_for_logging(
    coord_regions_by_slice: dict[Any, tuple[CoordRegion, ...]] | None,
) -> dict[Any, list[dict[str, tuple[int, ...]]]] | None:
  """Builds a compact log-friendly summary of remaining coord regions.

  Args:
    coord_regions_by_slice: Remaining coord-region partition keyed by slice id.

  Returns:
    A JSON-like summary suitable for logging, or ``None`` when no coord-region
    state is available.
  """
  if coord_regions_by_slice is None:
    return None
  return {
      slice_id: [
          {"start": region.start, "shape": region.shape}
          for region in regions
      ]
      for slice_id, regions in sorted(
          coord_regions_by_slice.items(),
          key=lambda item: str(item[0]),
      )
  }


def _optional_int_sort_key(value: Any) -> tuple[int, int]:
  """Builds a deterministic sort key component for optional integer metadata."""
  if value is None:
    return (1, 0)
  return (0, int(value))


def _host_sort_key(host_key: tuple[Any, ...] | None) -> tuple[tuple[int, Any], ...]:
  """Builds a stable sort key for derived host identifiers.

  Args:
    host_key: Host grouping key such as `(slice_id, task_id)`, or None when
      host metadata is unavailable.

  Returns:
    A tuple suitable for deterministic sorting. Missing host metadata sorts
    after concrete integer values.
  """
  if host_key is None:
    return (_optional_int_sort_key(None),)
  return tuple(_optional_int_sort_key(value) for value in host_key)


def allocation_device_sort_key(device: Any) -> tuple[Any, ...]:
  """Sort key for deterministic allocation order: slice, host, coords, id."""
  host_key = device_host_key(device)
  host_id = host_key[1] if host_key is not None and len(host_key) > 1 else None
  coords = device_mesh_coords(device)
  return (
      _optional_int_sort_key(device_slice_id(device)),
      _optional_int_sort_key(host_id),
      coords or (),
      _optional_int_sort_key(device_attr(device, "id")),
  )


def group_devices_by_slice(devices: Sequence[Any]) -> list[list[Any]] | None:
  """Groups devices by slice while preserving first-seen slice order.

  Devices without slice metadata are treated as belonging to one shared slice.
  The order of groups matches the first appearance of each slice in `devices`,
  which lets the allocator prefer earlier slices before spilling into later
  ones.
  """
  slice_to_devices = {}
  for device in devices:
    slice_id = device_slice_id(device)
    slice_to_devices.setdefault(slice_id, []).append(device)
  return list(slice_to_devices.values())


def group_devices_by_host(devices: Sequence[Any]) -> list[list[Any]] | None:
  """Groups devices by host/task when that metadata is available.

  Args:
    devices: Candidate devices to partition.

  Returns:
    A list of equal-sized per-host device lists, or None if host metadata is
    missing or inconsistent.

  Notes:
    Host buckets are returned in deterministic host-key order, and devices
    inside each bucket are sorted with `allocation_device_sort_key()` so host
    grouping remains reproducible for callers and validations that still use
    this helper.
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
  ordered_host_keys = sorted(host_to_devices, key=_host_sort_key)
  return [
      sorted(host_to_devices[host_key], key=allocation_device_sort_key)
      for host_key in ordered_host_keys
  ]


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
  has_core_on_chip_dimension = False
  for device in devices:
    if device_attr(device, "core_on_chip") is not None:
      has_core_on_chip_dimension = True
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
  min_coords = tuple(
      min(coords[dim] for coords in all_coords)
      for dim in range(num_dims)
  )
  chip_coord_to_coords = collections.defaultdict(list)
  for coords in all_coords:
    chip_coord = coords[:-1] if has_core_on_chip_dimension else coords
    chip_coord_to_coords[chip_coord].append(coords)
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
      min_coords=min_coords,
      max_shape=max_shape,
      chip_coord_to_coords={
          chip_coord: tuple(sorted(group_coords))
          for chip_coord, group_coords in chip_coord_to_coords.items()
      },
      has_core_on_chip_dimension=has_core_on_chip_dimension,
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
  if not coord_topology.has_core_on_chip_dimension:
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
    A known per-host physical bound in canonical 3D chip form, optionally with
    a trailing core-count dimension.
  """
  bounds = topology.infer_chips_per_host_bounds(devices)
  if bounds is None:
    return None

  allow_2d = _pool_allows_2d_chip_coords(devices)
  chip_coords = (
      _canonical_device_chip_coords(devices[0], allow_2d=allow_2d)
      if devices
      else None
  )
  if chip_coords is None:
    return None

  if device_attr(devices[0], "core_on_chip") is not None:
    core_count = infer_core_on_chip_count(devices)
    if core_count is None:
      return None
    return bounds + (core_count,)

  return bounds


def resolve_per_host_mesh_shape(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Returns the known per-host shape and validates each host against it.

  Args:
    devices: Devices spanning one or more hosts.

  Returns:
    The known per-host shape when static topology metadata is available.

  Raises:
    ValueError: If any discovered host bucket does not match the known per-host
      shape for the device family.
  """
  static_shape = known_host_mesh_shape(devices)
  if static_shape is None:
    return None

  host_groups = group_devices_by_host(devices)
  if host_groups is None:
    return static_shape

  for host_devices in host_groups:
    if not _satisfies_host_bound_shape(
        host_devices,
        static_shape,
        len(host_devices),
    ):
      raise ValueError(
          "Observed host devices do not match known host bounds "
          f"{static_shape}."
      )
  return static_shape


def _divisors(value: int) -> list[int]:
  """Returns the positive divisors of `value` in ascending order."""
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


def _device_family(devices: Sequence[Any]) -> str | None:
  """Returns the normalized accelerator family for a device pool."""
  if not devices:
    return None
  device_kind = device_attr(devices[0], "device_kind")
  if not isinstance(device_kind, str):
    return None
  return topology._resolve_family(device_kind)


def _supported_coord_box_shapes(
    devices: Sequence[Any],
    coord_topology: CoordTopology,
    required_devices: int,
    available_coord_shape: Sequence[int] | None = None,
) -> list[tuple[int, ...]] | None:
  """Returns topology-valid physical box shapes for the current device pool.

  When the accelerator family is known, this narrows box search to the exact
  TPU topology shapes that can legally realize `required_devices` on the
  current cluster. For unknown families, callers should fall back to generic
  contiguous-box enumeration.
  """
  family = _device_family(devices)
  if family is None:
    return None

  core_count = infer_core_on_chip_count(devices) or 1
  if required_devices % core_count != 0:
    return []

  chip_rank = coord_topology.num_dims - (1 if coord_topology.has_core_on_chip_dimension else 0)
  if chip_rank <= 0:
    return []
  available_shape = tuple(available_coord_shape or coord_topology.max_shape)
  available_chip_shape = available_shape[:chip_rank]
  required_chips = required_devices // core_count

  candidate_chip_shapes = [
      chip_shape
      for chip_shape in topology.best_topology_shapes_for_chip_count(
        family,
        required_chips,
        chip_rank=chip_rank,
        available_chip_shape=available_chip_shape,
      )
  ]

  if not candidate_chip_shapes:
    return []
  if coord_topology.has_core_on_chip_dimension:
    return [chip_shape + (core_count,) for chip_shape in candidate_chip_shapes]
  return candidate_chip_shapes


def _full_coord_region(coord_topology: CoordTopology) -> CoordRegion:
  """Returns the one bounding coord region for a concrete device pool."""
  return CoordRegion(
      start=coord_topology.min_coords,
      shape=coord_topology.max_shape,
  )


def _create_coord_regions(devices: Sequence[Any]) -> tuple[CoordRegion, ...] | None:
  """Builds the initial coord-region partition for one device pool."""
  coord_topology = get_coord_topology(devices)
  if coord_topology is None:
    return None
  return (_full_coord_region(coord_topology),)


def _create_coord_regions_by_slice(
    devices: Sequence[Any],
) -> dict[Any, tuple[CoordRegion, ...]] | None:
  """Builds initial coord-region partitions keyed by slice id when present."""
  slice_groups = group_devices_by_slice(devices)
  if not slice_groups:
    coord_regions = _create_coord_regions(devices)
    if coord_regions is None:
      return None
    return {None: coord_regions}

  coord_regions_by_slice = {}
  for slice_devices in slice_groups:
    if not slice_devices:
      continue
    coord_regions = _create_coord_regions(slice_devices)
    if coord_regions is None:
      continue
    coord_regions_by_slice[device_slice_id(slice_devices[0])] = coord_regions
  return coord_regions_by_slice or None


def _split_coord_region(
    region: CoordRegion,
    allocated_start: tuple[int, ...],
    allocated_shape: tuple[int, ...],
) -> tuple[CoordRegion, ...]:
  """Splits a consumed region into remaining z/y/x-style guillotine regions.

  Args:
    region: Region being carved up.
    allocated_start: Start coordinate of the allocated box.
    allocated_shape: Shape of the allocated box.

  Returns:
    The remaining regions after subtracting the allocated box.

  Raises:
    ValueError: If the allocated box rank does not match the region or falls
      outside the region bounds.
  """
  if len(region.shape) != len(allocated_shape) or len(region.shape) != len(allocated_start):
    raise ValueError(
        "Coord region rank does not match allocated box: "
        f"region={region.shape} start={allocated_start} allocated={allocated_shape}."
    )
  region_end = tuple(
      region.start[dim] + region.shape[dim]
      for dim in range(len(region.shape))
  )
  allocated_end = tuple(
      allocated_start[dim] + allocated_shape[dim]
      for dim in range(len(region.shape))
  )
  if any(
      allocated_start[dim] < region.start[dim]
      or allocated_end[dim] > region_end[dim]
      for dim in range(len(region.shape))
  ):
    raise ValueError(
        "Allocated box does not fit within coord region: "
        f"region_start={region.start} region_shape={region.shape} "
        f"allocated_start={allocated_start} allocated_shape={allocated_shape}."
    )

  remaining_regions = []
  num_dims = len(region.shape)
  for dim in range(num_dims - 1, -1, -1):
    before_dim = allocated_start[dim] - region.start[dim]
    after_dim = region_end[dim] - allocated_end[dim]

    if before_dim > 0:
      start = list(region.start)
      shape = list(region.shape)
      for earlier_dim in range(dim):
        start[earlier_dim] = allocated_start[earlier_dim]
        shape[earlier_dim] = allocated_shape[earlier_dim]
      shape[dim] = before_dim
      remaining_regions.append(CoordRegion(tuple(start), tuple(shape)))

    if after_dim > 0:
      start = list(region.start)
      shape = list(region.shape)
      for earlier_dim in range(dim):
        start[earlier_dim] = allocated_start[earlier_dim]
        shape[earlier_dim] = allocated_shape[earlier_dim]
      start[dim] = allocated_end[dim]
      shape[dim] = after_dim
      remaining_regions.append(CoordRegion(tuple(start), tuple(shape)))
  return tuple(remaining_regions)


def _region_contains_box(
    region: CoordRegion,
    start: tuple[int, ...],
    shape: tuple[int, ...],
) -> bool:
  """Returns whether a candidate box lies fully within a coord region."""
  return all(
      region.start[dim] <= start[dim]
      and start[dim] + shape[dim] <= region.start[dim] + region.shape[dim]
      for dim in range(len(region.shape))
  )


def _build_candidate_coords(
    coord_topology: CoordTopology,
    start: tuple[int, ...],
    shape: tuple[int, ...],
) -> list[tuple[int, ...]] | None:
  """Builds the coord list for one candidate box if the full box exists."""
  candidate_coords = []
  for offset in np.ndindex(shape):
    candidate_coord = tuple(
        start[dim] + offset[dim] for dim in range(coord_topology.num_dims)
    )
    if candidate_coord not in coord_topology.coord_to_device:
      return None
    candidate_coords.append(candidate_coord)
  return candidate_coords


def _coord_box_score(
    start: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[Any, ...]:
  """Builds a lexicographic sort key for candidate coord boxes.

  The returned tuple is ordered so Python tuple comparison implements the
  desired ranking policy directly:

  1. Prefer boxes with a smaller maximum dimension.
  2. Prefer more compact overall shapes.
  3. Prefer earlier start coordinates as a stable tiebreaker.

  Args:
    start: Candidate box origin.
    shape: Candidate box shape.

  Returns:
    A tuple sort key suitable for lexicographic comparison.
  """
  return (
      max(shape),
      tuple(sorted(shape, reverse=True)),
      tuple(-dim for dim in shape),
      start,
  )


def _order_candidate_regions(
    candidate_regions: Sequence[tuple[CoordRegion, Sequence[tuple[int, ...]]]],
    allocation_policy: str,
) -> list[tuple[CoordRegion, Sequence[tuple[int, ...]]]]:
  """Orders candidate regions according to the requested allocation policy.

  Args:
    candidate_regions: Candidate remaining regions paired with the supported
      shapes each region can realize for the current request.
    allocation_policy: ``COMPACT`` or ``PERFORMANCE``.

  Returns:
    Candidate regions ordered according to the requested policy.
  """
  if allocation_policy == _COMPACT_ALLOCATION_POLICY:
    return sorted(
        candidate_regions,
        key=lambda item: (
          _coord_box_score(item[0].start, item[0].shape),
          _coord_box_score(item[0].start, item[1][0]),
        ),
    )
  return sorted(
      candidate_regions,
      key=lambda item: (
        _coord_box_score(item[0].start, item[1][0]),
        _coord_box_score(item[0].start, item[0].shape),
      ),
  )


def _find_best_candidate_box(
    coord_topology: CoordTopology,
    candidate_shapes: Sequence[tuple[int, ...]],
    *,
    region: CoordRegion | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, ...]]] | None:
  """Finds the best valid candidate box, optionally constrained to a region.

  This is the shared internal scan used by both region-aware allocation and
  whole-topology fallback allocation. Unlike `find_best_candidate_coords()`,
  it returns the full winning box metadata so callers can both assign devices
  and update remaining-region state.

  Args:
    coord_topology: Normalized coord metadata for the candidate device pool.
    candidate_shapes: Exact box shapes to consider.
    region: Optional remaining coord region that candidate boxes must fit
      inside. When omitted, the scan considers the whole topology.

  Returns:
    ``(start, shape, coords)`` for the best-ranked valid box, or ``None`` when
    no candidate fits.
  """
  best_candidate = None
  best_score = None

  for shape in candidate_shapes:
    for start in sorted(coord_topology.coord_to_device):
      if region is not None and not _region_contains_box(region, start, shape):
        continue
      candidate_coords = _build_candidate_coords(coord_topology, start, shape)
      if candidate_coords is None:
        continue
      if not candidate_uses_whole_chips(coord_topology, candidate_coords):
        continue
      score = _coord_box_score(start, shape)
      if best_score is None or score < best_score:
        best_score = score
        best_candidate = (start, shape, candidate_coords)

  return best_candidate


def find_best_candidate_coords(
    coord_topology: CoordTopology,
    required_devices: int,
    candidate_shapes: Sequence[tuple[int, ...]] | None = None,
) -> list[tuple[int, ...]] | None:
  """Returns only the coord list for the best candidate box.

  Args:
    coord_topology: Normalized coord metadata for the candidate device pool.
    required_devices: Number of devices needed for one mesh.
    candidate_shapes: Optional exact physical shapes to scan instead of
      enumerating every factorization of `required_devices`.

  Returns:
    The coord list for the best-ranked candidate box, or ``None`` when no
    valid box exists.

  Notes:
    This is a thin convenience wrapper over `_find_best_candidate_box()` for
    callers that only need the selected coordinates. It intentionally discards
    the winning box start and shape, which the region-aware allocator still
    needs in order to split remaining regions.
  """
  shapes = candidate_shapes or _enumerate_box_shapes(
      required_devices,
      coord_topology.max_shape,
  )
  best_candidate = _find_best_candidate_box(
      coord_topology,
      shapes,
  )
  if best_candidate is None:
    return None
  return list(best_candidate[2])


def _allocate_devices_by_coords(
    devices: Sequence[Any],
    required_devices: int,
    coord_regions: Sequence[CoordRegion] | None = None,
    allocation_policy: str = _COMPACT_ALLOCATION_POLICY,
) -> tuple[list[Any] | None, tuple[CoordRegion, ...] | None]:
  """Allocates a contiguous physical box of devices when coords exist.

  Args:
    devices: Candidate devices to allocate from.
    required_devices: Number of devices needed for one mesh.
    coord_regions: Optional remaining-region partition to respect during
      incremental allocations.
    allocation_policy: ``COMPACT`` prefers the smallest fitting remaining
      region; ``PERFORMANCE`` prefers the most cubical supported extracted
      shape.

  Returns:
    A tuple of `(assigned_devices, next_coord_regions)`. `assigned_devices` is
    the best contiguous physical box, or None if the devices do not expose
    usable coordinates. `next_coord_regions` preserves a guillotine-style
    partition when allocation consumed one tracked region from its origin.

  Notes:
    This helper runs in three stages:

    1. Build normalized coord metadata with `get_coord_topology()`.
    2. Derive preferred physical shapes for the device family.
    3. First try the tracked remaining coord regions, which preserves the
      guillotine-style region partition used across incremental allocations.
    4. If no tracked region can realize a valid box, fall back to a
      whole-topology scan with `find_best_candidate_coords()`. This exists
      because the tracked region partition is conservative bookkeeping: it is
      useful for incremental allocation, but it does not represent every
      contiguous box that may still exist in the remaining device pool.
  """
  coord_topology = get_coord_topology(devices)
  if coord_topology is None:
    return None, None

  allocation_policy = _normalize_allocation_policy(allocation_policy)
  regions = tuple(coord_regions or (_full_coord_region(coord_topology),))
  candidate_regions = []
  for region in regions:
    candidate_shapes = _supported_coord_box_shapes(
        devices,
        coord_topology,
        required_devices,
        available_coord_shape=region.shape,
    )
    if not candidate_shapes:
      continue
    candidate_regions.append((region, candidate_shapes))

  candidate_regions = _order_candidate_regions(candidate_regions, allocation_policy)
  for region, candidate_shapes in candidate_regions:
    best_region_candidate = _find_best_candidate_box(
        coord_topology,
        candidate_shapes,
        region=region,
    )
    if best_region_candidate is None:
      continue
    candidate_start, candidate_shape, candidate_coords = best_region_candidate
    selected_coords = set(candidate_coords)
    assigned_devices = [
        device
        for device in devices
        if device_mesh_coords(device) in selected_coords
    ]
    next_coord_regions = tuple(
        remaining_region
        for existing_region in regions
        for remaining_region in (
            _split_coord_region(existing_region, candidate_start, candidate_shape)
            if existing_region == region
            else (existing_region,)
        )
    )
    return assigned_devices, next_coord_regions

  candidate_shapes = _supported_coord_box_shapes(
      devices,
      coord_topology,
      required_devices,
  )
  best_candidate_coords = find_best_candidate_coords(
      coord_topology,
      required_devices,
      candidate_shapes=candidate_shapes,
  )
  if best_candidate_coords is None:
    return None, tuple(regions)

  selected_coords = set(best_candidate_coords)
  return [
      device
      for device in devices
      if device_mesh_coords(device) in selected_coords
  ], _create_coord_regions([
      device
      for device in devices
      if device_mesh_coords(device) not in selected_coords
  ])


def _create_device_allocation_state(
    devices: Sequence[Any] | None = None,
    *,
    allocation_policy: str = _COMPACT_ALLOCATION_POLICY,
    log_summary: bool = True,
) -> DeviceAllocationState:
  """Builds reusable allocator state for one or more mesh allocations.

  This is intentionally private because callers should not need to understand
  the allocator internals to request one mesh. The public entry point is
  `allocate_devices()`, which accepts either raw `devices` for one-shot use or
  an existing `allocation_state` for incremental use.

  Args:
    devices: Optional explicit device pool. When omitted, uses ``jax.devices()``.
    allocation_policy: Allocation policy carried forward for later incremental
      allocations.
    log_summary: Whether to emit debug summaries for the initial pool.

  Returns:
    A state object containing the remaining device pool plus cached host and
    slice capacity metadata used by later allocation calls.
  """
  allocation_policy = _normalize_allocation_policy(allocation_policy)
  all_devices = tuple(
      sorted(
        jax.devices() if devices is None else devices,
        key=allocation_device_sort_key,
      )
  )
  if log_summary:
    logging.info(
        "Mesh allocator raw device sample: %s",
        summarize_devices_for_debug_logging(all_devices),
    )
    logging.info(
        "Mesh allocator derived host groups: %s",
        summarize_host_groups_for_logging(all_devices),
    )
  host_groups = group_devices_by_host(all_devices)
  slice_groups = group_devices_by_slice(all_devices)
  remaining_coord_regions_by_slice = _create_coord_regions_by_slice(all_devices)
  if log_summary:
    logging.info(
        "Mesh allocator derived coord regions: %s",
        summarize_coord_regions_for_logging(remaining_coord_regions_by_slice),
    )
  full_devices_per_host = (
      len(host_groups[0]) if host_groups else 0
  )
  full_devices_per_slice = None
  if slice_groups is not None:
    full_devices_per_slice = {
        device_slice_id(slice_devices[0]): len(slice_devices)
        for slice_devices in slice_groups
        if slice_devices
    }
  host_bound_shape = resolve_per_host_mesh_shape(all_devices)
  return DeviceAllocationState(
      remaining_devices=all_devices,
      full_devices_per_host=full_devices_per_host,
      host_bound_shape=host_bound_shape,
      total_device_count=len(all_devices),
      allocation_policy=allocation_policy,
      remaining_coord_regions_by_slice=remaining_coord_regions_by_slice,
      full_devices_per_slice=full_devices_per_slice,
  )


def _allocate_devices_from_pool(
    required_devices: int,
    remaining_devices: list[Any],
    mesh_name: str,
    coord_regions: Sequence[CoordRegion] | None = None,
    allocation_policy: str = _COMPACT_ALLOCATION_POLICY,
) -> tuple[list[Any], list[Any], tuple[CoordRegion, ...] | None]:
  """Allocates one mesh from a concrete device pool without slice policy.

  This helper contains the pool-local allocation strategy used after any
  slice-level decision has already been made.

  Coord-box allocation is mandatory here. If the remaining devices cannot form
  a valid coord-based box for the request, allocation fails instead of falling
  back to host buckets or flat prefix assignment.

  Args:
    required_devices: Number of devices requested.
    remaining_devices: Current flat device pool.
    mesh_name: Name used for diagnostics.
    coord_regions: Optional remaining-region partition for coord allocation.
    allocation_policy: Coord-allocation policy to apply.

  Returns:
    A tuple of assigned devices, the remaining flat pool, and the updated
    coord-region partition.
  """
  if required_devices > len(remaining_devices):
    raise ValueError(
        f"Mesh allocation requires {required_devices} devices for {mesh_name}, "
        f"but only {len(remaining_devices)} remain available."
    )

  assigned_devices, next_coord_regions = _allocate_devices_by_coords(
      remaining_devices,
      required_devices,
      coord_regions,
      allocation_policy,
  )
  if assigned_devices is None:
    raise ValueError(
        f"Mesh allocation requires {required_devices} devices for {mesh_name}, "
        "but coord-based allocation could not construct a valid box from the "
        "remaining devices."
    )

  remaining_devices = _remove_devices_by_identity(
      remaining_devices,
      assigned_devices,
  )
  return assigned_devices, remaining_devices, next_coord_regions


def allocate_devices(
    required_devices: int,
    devices: Sequence[Any] | None = None,
    *,
    mesh_name: str = "allocated_mesh",
    allocation_state: DeviceAllocationState | None = None,
    allocation_policy: str | None = None,
    return_state: bool = False,
) -> list[Any] | tuple[list[Any], DeviceAllocationState]:
  """Allocates devices for a single mesh request.

  This is the lowest-level public allocation API. It handles exactly one mesh
  request and applies the allocator policy in priority order:

  1. Reuse or create allocation state for the current remaining device pool.
  2. When one slice group remains, allocate directly from that slice.
  3. When multiple slice groups remain, first try satisfying the request
     within one slice.
  4. If no single slice worked, a request may span slices only by consuming
     whole remaining slices.

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
    allocation_policy: Optional allocation policy for one-shot use. When an
      existing `allocation_state` is provided, any explicit policy must match
      the policy stored in that state.
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
  if allocation_state is None:
    state = _create_device_allocation_state(
        devices,
        allocation_policy=_normalize_allocation_policy(allocation_policy),
    )
  else:
    state = allocation_state
    if allocation_policy is not None:
      normalized_policy = _normalize_allocation_policy(allocation_policy)
      if normalized_policy != state.allocation_policy:
        raise ValueError(
            "allocation_policy must match allocation_state.allocation_policy, "
            f"got {normalized_policy!r} and {state.allocation_policy!r}."
        )
  remaining_devices = list(state.remaining_devices)
  remaining_coord_regions_by_slice = (
      dict(state.remaining_coord_regions_by_slice)
      if state.remaining_coord_regions_by_slice is not None
      else None
  )
  assigned_devices = None
  allocation_error_prefix = (
      f"Mesh allocation requires {required_devices} devices for {mesh_name}, "
  )

  slice_groups = group_devices_by_slice(remaining_devices)
  if assigned_devices is None and slice_groups:
    for slice_devices in slice_groups:
      if len(slice_devices) < required_devices:
        continue
      slice_state = _create_device_allocation_state(
          slice_devices,
          log_summary=False,
      )
      slice_id = device_slice_id(slice_devices[0])
      slice_coord_regions = None
      if remaining_coord_regions_by_slice is not None:
        slice_coord_regions = remaining_coord_regions_by_slice.get(slice_id)
      assigned_devices, _, next_slice_coord_regions = _allocate_devices_from_pool(
          required_devices,
          list(slice_state.remaining_devices),
          mesh_name,
          coord_regions=slice_coord_regions,
          allocation_policy=state.allocation_policy,
      )
      remaining_devices = _remove_devices_by_identity(
          remaining_devices,
          assigned_devices,
      )
      if remaining_coord_regions_by_slice is not None:
        remaining_coord_regions_by_slice[slice_id] = next_slice_coord_regions or ()
      break

    # If no single slice is large enough, a cross-slice mesh must consume
    # whole slices in order. This avoids partial-slice allocation and keeps
    # cross-slice policy simple.
    if (
      assigned_devices is None
      and len(slice_groups) > 1
      and len(remaining_devices) >= required_devices
    ):
      assigned_devices = []
      assigned_slice_groups = []
      assigned_device_count = 0
      for slice_devices in slice_groups:
        if assigned_device_count >= required_devices:
          break
        slice_id = device_slice_id(slice_devices[0])
        if (
            state.full_devices_per_slice is not None
            and len(slice_devices) != state.full_devices_per_slice.get(slice_id)
        ):
          continue
        assigned_slice_groups.append(slice_devices)
        assigned_devices.extend(slice_devices)
        assigned_device_count += len(slice_devices)

      if assigned_device_count == required_devices:
        for slice_devices in assigned_slice_groups:
          remaining_devices = _remove_devices_by_identity(
              remaining_devices,
              slice_devices,
          )
          if remaining_coord_regions_by_slice is not None:
            remaining_coord_regions_by_slice.pop(device_slice_id(slice_devices[0]), None)
      else:
        raise ValueError(
              allocation_error_prefix
              + "but cross-slice allocation only supports whole slices."
        )

  if assigned_devices is None:
    raise ValueError(
          allocation_error_prefix
          + f"but only {len(remaining_devices)} remain available."
    )

  next_state = dataclasses.replace(
      state,
      remaining_devices=tuple(remaining_devices),
      remaining_coord_regions_by_slice=remaining_coord_regions_by_slice,
      used_device_count=state.used_device_count + len(assigned_devices),
  )
  logging.info(
      "Allocated devices for %s: %s",
      mesh_name,
      summarize_devices_for_logging(assigned_devices),
  )
  logging.info(
      "Remaining coord regions after %s: %s",
      mesh_name,
      summarize_coord_regions_for_logging(remaining_coord_regions_by_slice),
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
  """Removes assigned devices from a pool using object identity.

  Identity-based removal avoids ambiguity when test doubles or device objects
  compare equal by value but still represent distinct runtime objects.
  """
  assigned_device_ids = {id(device) for device in assigned_devices}
  return [device for device in devices if id(device) not in assigned_device_ids]


def _satisfies_host_bound_shape(
    host_devices: Sequence[Any],
    host_bound_shape: tuple[int, ...] | None,
    host_bound_device_count: int,
) -> bool:
  """Returns whether one host bucket matches the expected host topology.

  Args:
    host_devices: Devices believed to belong to one host.
    host_bound_shape: Expected per-host chip shape, optionally with a trailing
      logical-devices-per-chip dimension.
    host_bound_device_count: Expected number of logical devices per host.

  Returns:
    True when the devices occupy exactly the expected host bounds and device
    multiplicity, otherwise False.
  """
  if host_bound_shape is None or host_bound_device_count <= 0:
    raise ValueError(
        "host_bound_shape and host_bound_device_count must be set for "
        "host-group allocation."
    )
  if len(host_devices) != host_bound_device_count:
    return False

  chip_coords = []
  allow_2d = _pool_allows_2d_chip_coords(host_devices)
  for device in host_devices:
    canonical_coords = _canonical_device_chip_coords(device, allow_2d=allow_2d)
    if canonical_coords is None:
      return False
    chip_coords.append(canonical_coords)

  if not chip_coords:
    return False

  num_chip_dims = 3
  if len(host_bound_shape) not in (num_chip_dims, num_chip_dims + 1):
    return False
  if any(len(coords) != num_chip_dims for coords in chip_coords):
    return False

  unique_chip_coords = set(chip_coords)
  mins = tuple(min(coords[dim] for coords in unique_chip_coords) for dim in range(num_chip_dims))
  maxs = tuple(max(coords[dim] for coords in unique_chip_coords) for dim in range(num_chip_dims))
  chip_shape = tuple(
      maxs[dim] - mins[dim] + 1 for dim in range(num_chip_dims)
  )
  expected_chip_shape = host_bound_shape[:num_chip_dims]
  if chip_shape != expected_chip_shape:
    return False
  if int(np.prod(chip_shape)) != len(unique_chip_coords):
    return False

  chip_counts = collections.Counter(chip_coords)
  if len(host_bound_shape) == num_chip_dims:
    return len(set(chip_counts.values())) == 1

  expected_devices_per_chip = host_bound_shape[-1]
  return all(count == expected_devices_per_chip for count in chip_counts.values())


def allocate_named_mesh_device_slices(
    mesh_requirements: Sequence[MeshRequirement],
    devices: Sequence[Any] | None = None,
    *,
    allocation_policy: str = _COMPACT_ALLOCATION_POLICY,
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
    allocation_policy: Allocation policy shared by all requested meshes in this
      pass. ``COMPACT`` packs into the smallest fitting remaining region.
      ``PERFORMANCE`` prefers more cubical extracted shapes.

  Returns:
    A dictionary mapping each mesh name to the list of devices assigned to it.

  Raises:
    ValueError: If a requested mesh cannot be assigned enough devices or if a
      host-based allocation would split hosts illegally.
  """
  state = _create_device_allocation_state(
      devices,
      allocation_policy=allocation_policy,
  )
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
