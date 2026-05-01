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

"""Minimal accelerator topology helpers used by Tunix mesh allocation."""

from typing import Any, Sequence

_SINGLE_HOST_BOUNDS = (1, 1, 1)
_MULTI_HOST_BOUNDS = (2, 2, 1)


def _device_attr(device: Any, attr_name: str) -> Any:
  """Returns a raw device attribute, calling it first when exposed lazily."""
  value = getattr(device, attr_name, None)
  return value() if callable(value) else value


def _normalize_device_kind(device_kind: str) -> str | None:
  device_kind = device_kind.lower()
  if "v7" in device_kind:
    return "tpu7x"
  if "v6e" in device_kind or "v6" in device_kind:
    return "v6e"
  if "v5e" in device_kind:
    return "v5e"
  if "v5" in device_kind:
    return "v5p"
  if "v4" in device_kind:
    return "v4"
  return None


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

  device_count = len(devices)
  if family in {"v5e", "v6e"} and device_count == 1:
    return _SINGLE_HOST_BOUNDS
  if family == "tpu7x" and device_count == 2:
    return _SINGLE_HOST_BOUNDS
  return _MULTI_HOST_BOUNDS
