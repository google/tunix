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

This module resolves a device pool's accelerator family and the supported
physical topology shapes (such as `2x2x4` or `8x16x16`) that can realize a
requested chip count on that family.

For fish families (`v4`, `v5p`, `v7x`), the supported physical pod shapes are
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
import functools
import math
import re
from typing import Any, Sequence

_MULTI_HOST_BOUNDS = (2, 2, 1)
# Edge families expose a 2D chip torus; fish families expose a 3D torus. The two
# groups follow different supported-shape rules, so keep them as named sets and
# derive the full supported set from them.
_EDGE_FAMILIES = frozenset({"v5e", "v6e"})
_FISH_FAMILIES = frozenset({"v4", "v5p", "v7x"})
_SUPPORTED_FAMILIES = _EDGE_FAMILIES | _FISH_FAMILIES
# Supported fish-family (v4/v5p/v7x) sub-cube chip shapes below the first full
# cube, keyed by chip count. Each volume is unique, so a chip count maps to
# exactly one shape. At or above the first full cube (4x4x4), multiple axis
# arrangements share a volume, so those are enumerated separately.
_FISH_SUB_CUBE_SHAPE_BY_CHIP_COUNT = {
    4: (2, 2, 1),
    8: (2, 2, 2),
    16: (2, 2, 4),
    32: (2, 4, 4),
}
_FISH_CUBE_GRANULARITY = 4
# Supported edge-family (v5e/v6e) chip shapes, keyed by chip count. Volumes are
# unique, so a chip count maps to exactly one shape.
_EDGE_SHAPE_BY_CHIP_COUNT = {
    1: (1, 1, 1),
    4: (2, 2, 1),
    8: (2, 4, 1),
    16: (4, 4, 1),
    32: (4, 8, 1),
    64: (8, 8, 1),
    128: (8, 16, 1),
    256: (16, 16, 1),
}


def _best_single_host_fish_shape(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> tuple[int, int, int] | None:
  """Returns the most cubical fish-family subslice that fits within one host.

  Below the first full cube a request may be a fraction of a host (e.g. 1 or 2
  chips). This searches the ``(x, y, z)`` shapes within the per-host bound that
  hold exactly ``required_chips`` and returns the most cubical one, packing into
  earlier axes first.

  Args:
    required_chips: Number of chips requested.
    available_chip_shape: Optional 3D per-axis bound; the per-host bound is
      further capped to it. When its rank is not 3, no shape is returned.

  Returns:
    The best fitting within-host shape, or None when none holds the count.
  """
  supported_shapes = _supported_single_host_fish_shapes(
      required_chips, available_chip_shape
  )
  if not supported_shapes:
    return None
  return supported_shapes[0]


def _supported_single_host_fish_shapes(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> list[tuple[int, int, int]]:
  """Returns all exact single-host fish shapes ranked by compactness.

  Args:
    required_chips: Number of chips requested.
    available_chip_shape: Optional 3D per-axis bound; the per-host bound is
      further capped to it. When its rank is not 3, no shape is returned.

  Returns:
    All matching shapes sorted from more cubical to less cubical.
  """
  host_shape = _MULTI_HOST_BOUNDS
  if available_chip_shape is not None:
    if len(available_chip_shape) != 3:
      return []
    host_shape = tuple(
        min(int(limit), host_limit)
        for limit, host_limit in zip(available_chip_shape, _MULTI_HOST_BOUNDS)
    )

  supported_shapes = []
  for x in range(1, host_shape[0] + 1):
    for y in range(1, host_shape[1] + 1):
      for z in range(1, host_shape[2] + 1):
        shape = (x, y, z)
        if math.prod(shape) != required_chips:
          continue
        supported_shapes.append(shape)

  return sorted(
      supported_shapes,
      key=lambda shape: (
          max(shape),
          tuple(sorted(shape, reverse=True)),
          tuple(-dim for dim in shape),
          shape,
      ),
  )


def _supported_single_host_edge_shapes(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> list[tuple[int, int, int]]:
  """Returns all exact single-host edge shapes ranked by compactness.

  Edge families expose a 2D chip torus, but small requests may still occupy a
  fraction of one host. This enumerates all 2D rectangular factorizations of
  ``required_chips`` that fit within the per-host ``2x2`` bound, then
  canonicalizes them to ``(x, y, z=1)``.

  Args:
    required_chips: Number of chips requested.
    available_chip_shape: Optional 2D/3D per-axis bound; when malformed it is
      ignored to preserve the existing edge-family behavior.

  Returns:
    All matching shapes sorted from more cubical to less cubical, breaking ties
    by preferring earlier axes first.
  """
  host_shape = _MULTI_HOST_BOUNDS
  if available_chip_shape is not None:
    canonical_available_shape = _canonicalize_chip_shape_to_3d(
        available_chip_shape
    )
    if canonical_available_shape is not None:
      host_shape = tuple(
          min(int(limit), host_limit)
          for limit, host_limit in zip(
              canonical_available_shape, _MULTI_HOST_BOUNDS
          )
      )

  supported_shapes = []
  for x in range(1, host_shape[0] + 1):
    for y in range(1, host_shape[1] + 1):
      shape = (x, y, 1)
      if math.prod(shape) != required_chips:
        continue
      supported_shapes.append(shape)

  return sorted(
      supported_shapes,
      key=lambda shape: (
          max(shape),
          tuple(sorted(shape, reverse=True)),
          tuple(-dim for dim in shape),
          shape,
      ),
  )


def _topology_shape_sort_key(
    shape: tuple[int, ...],
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
  """Ranks valid shapes from more cubical to less cubical.

  Args:
    shape: Dimensions of the shape.

  Returns:
    A comparison tuple where shapes with smaller maximum dimensions sort
    earlier.
  """
  return (
      max(shape),
      tuple(sorted(shape, reverse=True)),
      shape,
  )


def _shape_fits_within(
    shape: Sequence[int], available_shape: Sequence[int]
) -> bool:
  """Returns whether `shape` fits axis-by-axis within `available_shape`.

  Args:
    shape: Tuple/sequence of dimensions to check.
    available_shape: Tuple/sequence of bound limits.

  Returns:
    True if the shape fits within the limits for all axes, otherwise False.
  """
  return all(dim <= limit for dim, limit in zip(shape, available_shape))


@functools.cache
def _is_pathways_backend_used() -> bool:
  """Returns whether the current process is attached to a Pathways backend.

  Returns:
    True if a Pathways backend is used, otherwise False.
  """
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    return bool(pathwaysutils.is_pathways_backend_used())
  except ImportError:
    return False


def _pathways_device_host_attr(device: Any, attr_name: str) -> Any:
  """Parses a named attribute out of a Pathways device's repr string.

  Pathways does not expose per-host metadata (such as ``logical_task``) as a
  Python attribute, so it is recovered from ``repr(device)``, which looks like
  ``device(0,TPU,...,logical_task=11,slice=3,...)``. The match is anchored on a
  preceding delimiter so a name is not matched as a substring of another field.

  Args:
    device: The device whose repr is parsed.
    attr_name: The field name to extract, e.g. ``"logical_task"``.

  Returns:
    The parsed value (``ast.literal_eval`` of the captured token, e.g. an int or
    list), the raw string when it is not a Python literal, or None when the
    field is absent from the repr.
  """
  # (?:^|[,(])  -> attr must follow start-of-string, a comma, or "(" so e.g.
  #               "logical_task" is not matched inside "xlogical_task".
  # name=       -> the literal "<attr>=".
  # (\[...\]|[^,)]+) -> capture either a [...] list or a run up to the next
  #                     "," or ")".
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
  """Returns a raw device attribute, calling it first when exposed lazily.

  Args:
    device: JAX device or test double.
    attr_name: Name of attribute to read.
    default: Default value if the attribute does not exist.

  Returns:
    The attribute value, or the default value.
  """
  value = getattr(device, attr_name, default)
  return value() if callable(value) else value


def _normalize_device_kind(device_kind: str) -> str | None:
  """Maps a raw JAX ``device_kind`` string to a normalized family key.

  Args:
    device_kind: The runtime device kind, e.g. ``"TPU v5 lite"`` or ``"TPU
      v7"``.

  Returns:
    The normalized family key (``"v4"``, ``"v5e"``, ``"v5p"``, ``"v6e"``, or
    ``"v7x"``), or None when the kind does not name a supported TPU family.
  """
  device_kind = device_kind.lower()
  if "v7" in device_kind:
    return "v7x"
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
  """Resolves a raw device kind or normalized family key to a known family.

  Args:
    device_kind_or_family: Raw device kind or normalized family string.

  Returns:
    The normalized family key, or None if the family is not recognized.
  """
  family = _normalize_device_kind(device_kind_or_family)
  if family is not None:
    return family
  normalized = device_kind_or_family.lower()
  if normalized in _SUPPORTED_FAMILIES:
    return normalized
  return None


def _device_family(devices: Sequence[Any]) -> str | None:
  """Returns the normalized accelerator family for a device pool.

  Args:
    devices: Devices to inspect. The family is read from the first device's
      ``device_kind``.

  Returns:
    The normalized family key (e.g. ``"v5e"`` or ``"v7x"``), or None when the
    pool is empty or the device kind is missing or unrecognized.
  """
  if not devices:
    return None
  device_kind = _device_attr(devices[0], "device_kind")
  if not isinstance(device_kind, str):
    return None
  return _resolve_family(device_kind)


def _device_host_key(device: Any) -> tuple[Any, ...] | None:
  """Returns a stable per-host key when runtime metadata exposes one.

  Pathways does not expose a per-host task attribute, so the task component is
  parsed from the device repr; the slice component is still read from the
  ``slice_index`` attribute, which Pathways does expose. On non-Pathways
  backends the task component is the ``process_index`` attribute.

  Args:
    device: JAX device or test double.

  Returns:
    A (slice_id, task_id) tuple, or None if no task metadata is available.
  """
  if _is_pathways_backend_used():
    task_id = _pathways_device_host_attr(device, "logical_task")
  else:
    task_id = _device_attr(device, "process_index")
  if task_id is None:
    return None

  slice_id = _device_attr(device, "slice_index", None)
  return (slice_id, task_id)


def _canonicalize_chip_shape_to_3d(
    shape: Sequence[int],
) -> tuple[int, int, int] | None:
  """Canonicalizes a chip topology shape to `(x, y, z)` form.

  Shapes may come from edge runtimes that expose 2D chip coordinates. Those
  are normalized to 3D by appending a trailing singleton `z` dimension.

  Args:
    shape: Input shape to canonicalize.

  Returns:
    The canonical 3D shape, or None if shape is invalid.
  """
  parsed = tuple(int(dim) for dim in shape)
  if len(parsed) == 2:
    return parsed + (1,)
  if len(parsed) == 3:
    return parsed
  return None


def _best_fish_cube_shape(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> tuple[int, int, int] | None:
  """Returns the most cubical fish-family ``4i x 4j x 4k`` shape, if any.

  At or above the first full cube (``4x4x4`` = 64 chips), fish topologies are
  multiples of the ``4x4x4`` cube, and several axis arrangements can share a
  chip count. This enumerates the valid ``(4i, 4j, 4k)`` shapes with
  ``i <= j <= k`` that fit within ``available_chip_shape`` and returns the most
  cubical one (ranked by `_topology_shape_sort_key`).

  Args:
    required_chips: Total chip count requested.
    available_chip_shape: Optional per-axis chip bound the shape must fit
      within. When None, only the cube-count constraint applies. Expected to be
      a 3-tuple when provided.

  Returns:
    The most cubical valid cube-multiple shape, or None when `required_chips` is
    below one full cube or no arrangement fits the available bound.

  Raises:
    ValueError: If `required_chips` is at or above one full cube but not an
      exact multiple of the cube volume.
  """
  supported_shapes = _supported_fish_cube_shapes(
      required_chips, available_chip_shape
  )
  if not supported_shapes:
    return None
  return supported_shapes[0]


def _supported_fish_cube_shapes(
    required_chips: int,
    available_chip_shape: Sequence[int] | None,
) -> list[tuple[int, int, int]]:
  """Returns all valid fish-family cube-multiple shapes, best first.

  Args:
    required_chips: Total chip count requested.
    available_chip_shape: Optional per-axis chip bound the shape must fit
      within. When None, only the cube-count constraint applies. Expected to be
      a 3-tuple when provided.

  Returns:
    All valid shapes sorted from more cubical to less cubical.

  Raises:
    ValueError: If `required_chips` is at or above one full cube but not an
      exact multiple of the cube volume.
  """
  cube_volume = _FISH_CUBE_GRANULARITY**3
  if required_chips < cube_volume:
    return []

  cube_units, remainder = divmod(required_chips, cube_volume)
  if remainder != 0:
    raise ValueError(
        "Fish-family topology requests at or above 4x4x4 must be divisible "
        f"by {cube_volume} chips, got {required_chips}."
    )

  max_i = max_j = max_k = cube_units
  if available_chip_shape is not None:
    max_i = min(max_i, available_chip_shape[0] // _FISH_CUBE_GRANULARITY)
    max_j = min(max_j, available_chip_shape[1] // _FISH_CUBE_GRANULARITY)
    max_k = min(max_k, available_chip_shape[2] // _FISH_CUBE_GRANULARITY)

  # Enumerate factorizations cube_units = i * j * k with i <= j <= k. The
  # canonical i <= j <= k ordering means each multiset of factors is visited
  # once; `i * i <= cube_units` and `i * j <= cube_units` bound the loops to
  # that ordering (once i exceeds the cube root, no valid j >= i remains). The
  # actual shape multiplies each factor by the 4-chip cube edge.
  supported_shapes = []
  i = 1
  while i <= max_i and i * i <= cube_units:
    j = i
    while j <= max_j and i * j <= cube_units:
      k, extra = divmod(cube_units, i * j)
      if extra == 0 and j <= k and k <= max_k:  # exact factor, keeps i<=j<=k
        shape = (
            _FISH_CUBE_GRANULARITY * i,
            _FISH_CUBE_GRANULARITY * j,
            _FISH_CUBE_GRANULARITY * int(k),
        )
        supported_shapes.append(shape)
      j += 1
    i += 1
  return sorted(supported_shapes, key=_topology_shape_sort_key)


def supported_topology_shapes_for_chip_count(
    device_kind_or_family: str,
    required_chips: int,
    *,
    chip_rank: int = 3,
    available_chip_shape: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
  """Returns all legal topology shapes for a requested chip count.

  Shapes are ranked from more cubical to less cubical. This is the full-shape
  counterpart to `best_topology_shapes_for_chip_count()`, which returns only
  the first entry.

  Args:
    device_kind_or_family: Either a raw device kind (``"TPU v5 lite"``) or an
      already-normalized family key (``"v5e"``).
    required_chips: Number of physical chips the shape must contain.
    chip_rank: Rank of the returned shapes. Edge families support 2 or 3; fish
      families support only 3.
    available_chip_shape: Optional per-axis chip bound the shape must fit
      within.

  Returns:
    A possibly-empty list of all legal shapes, sorted best-first.

  Raises:
    ValueError: If a fish-family request at or above ``4x4x4`` is not divisible
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

  if family in _EDGE_FAMILIES:
    supported_shapes = _supported_single_host_edge_shapes(
        required_chips, parsed_available_shape
    )
    if not supported_shapes:
      shape = _EDGE_SHAPE_BY_CHIP_COUNT.get(required_chips)
      if shape is None:
        return []
      if parsed_available_shape is not None:
        canonical_edge_available_shape = _canonicalize_chip_shape_to_3d(
            parsed_available_shape
        )
        if canonical_edge_available_shape is not None and not _shape_fits_within(
            shape, canonical_edge_available_shape
        ):
          return []
      supported_shapes = [shape]
    if chip_rank == 2:
      return [shape[:2] for shape in supported_shapes]
    if chip_rank == 3:
      return supported_shapes
    return []

  if chip_rank != 3:
    return []

  if parsed_available_shape is not None and len(parsed_available_shape) != 3:
    return []

  sub_cube_shape = _FISH_SUB_CUBE_SHAPE_BY_CHIP_COUNT.get(required_chips)
  if sub_cube_shape is not None and (
      parsed_available_shape is None
      or _shape_fits_within(sub_cube_shape, parsed_available_shape)
  ):
    return [sub_cube_shape]

  single_host_shapes = _supported_single_host_fish_shapes(
      required_chips, parsed_available_shape
  )
  if single_host_shapes:
    return single_host_shapes

  cube_shapes = _supported_fish_cube_shapes(required_chips, parsed_available_shape)
  if cube_shapes:
    return cube_shapes

  return []


def best_topology_shapes_for_chip_count(
    device_kind_or_family: str,
    required_chips: int,
    *,
    chip_rank: int = 3,
    available_chip_shape: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
  """Returns the best legal topology shape(s) for a requested chip count.

  Shapes are ranked from more cubical to less cubical. For fish families this
  helper returns only the best-ranked 3D shape. For edge families it returns the
  single supported shape for the count, projected to the requested
  ``chip_rank``.

  Args:
    device_kind_or_family: Either a raw device kind (``"TPU v5 lite"``) or an
      already-normalized family key (``"v5e"``).
    required_chips: Number of physical chips the shape must contain.
    chip_rank: Rank of the returned shapes. Edge families support 2 or 3; fish
      families support only 3.
    available_chip_shape: Optional per-axis chip bound the shape must fit within
      (e.g. the remaining region of a partially consumed slice).

  Returns:
    A list with the best shape, or an empty list when no supported shape matches
    the count, rank, or available bound (or the family is unknown).

  Raises:
    ValueError: If a fish-family request at or above ``4x4x4`` is not divisible
      by the cube granularity volume.

  Examples:
    Edge family, 2D vs 3D projection::

        best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=2)  # [(2,
        4)]
        best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=3)  # [(2,
        4, 1)]

    Fish family picks the most cubical arrangement, and honors the bound::

        best_topology_shapes_for_chip_count("TPU v7", 256)              # [(4,
        8, 8)]
        best_topology_shapes_for_chip_count(
            "TPU v7", 576, available_chip_shape=(4, 12, 16))            # [(4,
            12, 12)]
  """
  supported_shapes = supported_topology_shapes_for_chip_count(
      device_kind_or_family,
      required_chips,
      chip_rank=chip_rank,
      available_chip_shape=available_chip_shape,
  )
  if not supported_shapes:
    return []
  return [supported_shapes[0]]
