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

"""Accelerator topology helpers used by Tunix mesh allocation.

This module captures two distinct layers of TPU topology knowledge:

1. Per-host bounds, such as `(2, 2, 1)` chips per host for multi-host fish
   families.
2. Supported pod-level physical topology shapes, such as `2x2x4` or `8x16x16`.

For fish families (`v4`, `v5p`, `tpu7x`), the supported physical pod shapes are
treated as:

1. A small explicit sequence before the first full cube: `2x2x1`, `2x2x2`,
   `2x2x4`, `2x4x4`.
2. Any canonical `4i x 4j x 4k` shape once the topology reaches `4x4x4`, with
   `i <= j <= k` in the requested axis order.

For `v5e` and `v6e`, supported physical shapes are canonicalized to 3D with a
trailing singleton `z`, so an edge shape like `8x16` is treated as `8x16x1`.

Source references:

- v4: https://docs.cloud.google.com/tpu/docs/v4
- v5e: https://docs.cloud.google.com/tpu/docs/v5e
- v5p: https://docs.cloud.google.com/tpu/docs/v5p
- v6e: https://docs.cloud.google.com/tpu/docs/v6e
- v7x: https://docs.cloud.google.com/tpu/docs/v7x
"""

import ast
import collections
import functools
import math
import re
from typing import Any, Sequence

_SINGLE_HOST_BOUNDS = (1, 1, 1)
_MULTI_HOST_BOUNDS = (2, 2, 1)
_SUPPORTED_FAMILIES = {
  "v4",
  "v5e",
  "v5p",
  "v6e",
  "tpu7x",
}
_FISH_SUPPORTED_SUB_CUBE_SHAPES = (
  (2, 2, 1),
  (2, 2, 2),
  (2, 2, 4),
  (2, 4, 4),
)
_FISH_CUBE_GRANULARITY = 4
_EDGE_SUPPORTED_SHAPES = (
  (1, 1, 1),
  (2, 2, 1),
  (2, 4, 1),
  (4, 4, 1),
  (4, 8, 1),
  (8, 8, 1),
  (8, 16, 1),
  (16, 16, 1),
)


def _enumerate_single_host_fish_shapes(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> list[tuple[int, int, int]]:
  """Returns compact fish-family subslice shapes that fit within one host."""
  host_shape = _MULTI_HOST_BOUNDS
  if available_chip_shape is not None:
    if len(available_chip_shape) != 3:
      return []
    host_shape = tuple(
        min(int(limit), host_limit)
        for limit, host_limit in zip(available_chip_shape, _MULTI_HOST_BOUNDS)
    )

  shapes = []
  for x in range(1, host_shape[0] + 1):
    for y in range(1, host_shape[1] + 1):
      for z in range(1, host_shape[2] + 1):
        shape = (x, y, z)
        if math.prod(shape) == required_chips:
          shapes.append(shape)
  return sorted(
      shapes,
      key=lambda shape: (
        max(shape),
        tuple(sorted(shape, reverse=True)),
          tuple(-dim for dim in shape),
        shape,
      ),
  )

def _topology_shape_sort_key(shape: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
  """Ranks valid shapes from more cubical to less cubical."""
  return (
      max(shape),
      tuple(sorted(shape, reverse=True)),
      shape,
  )


@functools.cache
def _is_pathways_backend_used() -> bool:
  """Returns whether the current process is attached to a Pathways backend."""
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    return bool(pathwaysutils.is_pathways_backend_used())
  except ImportError:
    return False


def _pathways_device_host_attr(device: Any, attr_name: str) -> Any:
  match = re.search(
      rf"(?:^|[,(]){re.escape(attr_name)}=(\[[^\]]*\]|[^,)]+)",
      repr(device),
  )
  if match is None:
    return None

  raw_value = match.group(1).strip()
  try:
    return ast.literal_eval(raw_value)
  except (SyntaxError, ValueError):
    return raw_value

def _device_attr(device: Any, attr_name: str, default: Any = None) -> Any:
  """Returns a raw device attribute, calling it first when exposed lazily."""
  value = getattr(device, attr_name, default)
  return value() if callable(value) else value


def _normalize_device_kind(device_kind: str) -> str | None:
  device_kind = device_kind.lower()
  if "v7" in device_kind:
    return "tpu7x"
  if "v6 lite" in device_kind or "v6e" in device_kind or "v6" in device_kind:
    return "v6e"
  if "v5 lite" in device_kind or "v5e" in device_kind:
    return "v5e"
  if "v5" in device_kind:
    return "v5p"
  if "v4" in device_kind:
    return "v4"
  return None


def _resolve_family(device_kind_or_family: str) -> str | None:
  """Resolves a raw device kind or normalized family key to a known family."""
  family = _normalize_device_kind(device_kind_or_family)
  if family is not None:
    return family
  normalized = device_kind_or_family.lower()
  if normalized in _SUPPORTED_FAMILIES:
    return normalized
  return None


def _device_host_key(device: Any) -> tuple[Any, ...] | None:
  """Returns a stable per-host key when runtime metadata exposes one."""
  task_id = None
  if _is_pathways_backend_used():
    task_id = _pathways_device_host_attr(device, "logical_task")
  else:
    for attr_name in ("task_id", "process_index"):
      task_id = _device_attr(device, attr_name)
      if task_id is not None:
        break
  if task_id is None:
    task_id = 0

  slice_id = _device_attr(device, "slice_index", None)
  if slice_id is None:
    slice_id = 0
  return (slice_id, task_id)


def _device_coords(device: Any) -> tuple[int, ...] | None:
  coords = _device_attr(device, "coords")
  if coords is None:
    return None
  return tuple(int(coord) for coord in coords)


def _canonicalize_chip_shape_to_3d(shape: Sequence[int]) -> tuple[int, int, int] | None:
  """Canonicalizes a chip topology shape to `(x, y, z)` form.

  Shapes may come from edge runtimes that expose 2D chip coordinates. Those
  are normalized to 3D by appending a trailing singleton `z` dimension.
  """
  parsed = tuple(int(dim) for dim in shape)
  if len(parsed) == 2:
    return parsed + (1,)
  if len(parsed) == 3:
    return parsed
  return None


def _infer_host_shape_from_runtime(devices: Sequence[Any]) -> tuple[int, ...] | None:
  """Infers per-host chip bounds from runtime host and coord metadata."""
  host_to_coords = collections.defaultdict(list)
  for device in devices:
    host_key = _device_host_key(device)
    coords = _device_coords(device)
    if host_key is None or coords is None:
      return None
    host_to_coords[host_key].append(coords)

  if not host_to_coords:
    return None

  host_shapes = set()
  for coords_list in host_to_coords.values():
    if not coords_list:
      return None
    rank = len(coords_list[0])
    if any(len(coords) != rank for coords in coords_list):
      return None
    unique_coords = set(coords_list)
    mins = tuple(min(coords[dim] for coords in unique_coords) for dim in range(rank))
    maxs = tuple(max(coords[dim] for coords in unique_coords) for dim in range(rank))
    shape = tuple(maxs[dim] - mins[dim] + 1 for dim in range(rank))
    canonical_shape = _canonicalize_chip_shape_to_3d(shape)
    if canonical_shape is None:
      return None
    if int(math.prod(shape)) != len(unique_coords):
      return None
    host_shapes.add(canonical_shape)

  if len(host_shapes) != 1:
    return None
  return next(iter(host_shapes))


def best_topology_shapes_for_chip_count(
    device_kind_or_family: str,
    required_chips: int,
    *,
    chip_rank: int = 3,
    available_chip_shape: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
  """Returns the best legal topology shape(s) for a requested chip count.

  Shapes are ranked from more cubical to less cubical. For fish families this
  helper returns only the best-ranked 3D shape(s), which is enough for the
  current allocator. For edge families, callers may request either 2D or 3D
  shapes.

  Raises:
    ValueError: If a fish-family request at or above `4x4x4` is not divisible
      by the cube granularity volume.
  """
  if required_chips <= 0:
    return []

  family = _resolve_family(device_kind_or_family)
  if family is None:
    return []

  parsed_available_shape = None
  if available_chip_shape is not None:
    parsed_available_shape = tuple(available_chip_shape)

  if family in {"v5e", "v6e"}:
    canonical_edge_available_shape = None
    if parsed_available_shape is not None:
      canonical_edge_available_shape = _canonicalize_chip_shape_to_3d(
          parsed_available_shape
      )
    matching_shapes = []
    for shape in _EDGE_SUPPORTED_SHAPES:
      if math.prod(shape) != required_chips:
        continue
      if canonical_edge_available_shape is not None:
        if any(
            dim > limit
            for dim, limit in zip(shape, canonical_edge_available_shape)
        ):
          continue
      matching_shapes.append(shape)
    if chip_rank == 2:
      return [shape[:2] for shape in matching_shapes]
    if chip_rank == 3:
      return matching_shapes
    return []

  if chip_rank != 3:
    return []

  if parsed_available_shape is not None and len(parsed_available_shape) != 3:
    return []

  best_shape = None
  best_shape_key = None

  def consider_shape(shape: tuple[int, int, int]):
    nonlocal best_shape, best_shape_key
    shape_key = _topology_shape_sort_key(shape)
    if best_shape_key is None or shape_key < best_shape_key:
      best_shape = shape
      best_shape_key = shape_key

  for shape in _FISH_SUPPORTED_SUB_CUBE_SHAPES:
    if math.prod(shape) != required_chips:
      continue
    if parsed_available_shape is not None:
      if any(dim > limit for dim, limit in zip(shape, parsed_available_shape)):
        continue
    best_shape = shape
    best_shape_key = _topology_shape_sort_key(shape)
    break

  if best_shape is None:
    for shape in _enumerate_single_host_fish_shapes(
        required_chips,
        parsed_available_shape,
    ):
      consider_shape(shape)
      break

  if required_chips >= _FISH_CUBE_GRANULARITY**3:
    cube_units, remainder = divmod(required_chips, _FISH_CUBE_GRANULARITY**3)
    if remainder != 0:
      raise ValueError(
          "Fish-family topology requests at or above 4x4x4 must be divisible "
          f"by {_FISH_CUBE_GRANULARITY**3} chips, got {required_chips}."
      )
    max_i = cube_units
    max_j = cube_units
    max_k = cube_units
    if parsed_available_shape is not None:
      max_i = min(max_i, parsed_available_shape[0] // _FISH_CUBE_GRANULARITY)
      max_j = min(max_j, parsed_available_shape[1] // _FISH_CUBE_GRANULARITY)
      max_k = min(max_k, parsed_available_shape[2] // _FISH_CUBE_GRANULARITY)
    i = 1
    while i <= max_i and i * i <= cube_units:
      j = i
      while j <= max_j and i * j <= cube_units:
        k, extra = divmod(cube_units, i * j)
        if extra == 0 and j <= k and k <= max_k:
          consider_shape(
              (
                  _FISH_CUBE_GRANULARITY * i,
                  _FISH_CUBE_GRANULARITY * j,
                  _FISH_CUBE_GRANULARITY * k,
              )
          )
        j += 1
      i += 1

  if best_shape is None:
    return []
  return [best_shape]


def infer_chips_per_host_bounds(
    devices: Sequence[Any],
) -> tuple[int, ...] | None:
  if not devices:
    return None

  device_kind = _device_attr(devices[0], "device_kind")
  if not isinstance(device_kind, str):
    return None

  family = _normalize_device_kind(device_kind)
  if family is None:
    return None

  runtime_host_shape = _infer_host_shape_from_runtime(devices)
  if family in {"v5e", "v6e"} and runtime_host_shape is not None:
    return runtime_host_shape

  device_count = len(devices)
  if family in {"v5e", "v6e"} and device_count == 1:
    return _SINGLE_HOST_BOUNDS
  if family == "tpu7x" and device_count == 2:
    return _SINGLE_HOST_BOUNDS
  return _MULTI_HOST_BOUNDS
