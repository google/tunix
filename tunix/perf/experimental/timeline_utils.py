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

from __future__ import annotations

from collections.abc import Sequence
import threading
from typing import Any

import numpy as np
from tunix.perf.experimental import timeline


JaxDevice = Any


# Utility functions for handling timeline IDs.


def generate_host_timeline_id() -> str:
  """Generates a string ID for a host timeline.

  Returns:
    A string ID formatted as 'host-<thread_id>'.
  """
  return f"host-{threading.get_ident()}"


def is_host_timeline(tl_id: str) -> bool:
  """Checks if the timeline ID corresponds to a host timeline.

  Args:
    tl_id: The timeline ID to check.

  Returns:
    True if the timeline ID starts with 'host-', False otherwise.
  """
  return tl_id.startswith("host-")


def generate_device_timeline_id(device_id: str | JaxDevice) -> str:
  """Generates a string ID for a device timeline.

  Args:
    device_id: A string ID or a JAX device object.

  Returns:
    A string representation of the device ID. For a JAX device object, it will
    be the platform name followed by the device ID, e.g., "tpu0".

  Raises:
    ValueError: If the input device_id type is not supported. Only string and
      JAX device objects (with platform and id attributes) are supported.
  """

  if isinstance(device_id, str):
    return device_id
  elif hasattr(device_id, "platform") and hasattr(device_id, "id"):
    # if it's a JAX device object, convert to string
    return f"{device_id.platform}{device_id.id}"
  else:
    raise ValueError(f"Unsupported id type: {type(device_id)}")


def generate_device_timeline_ids(
    devices: Sequence[str | JaxDevice] | np.ndarray | None,
) -> Sequence[str]:
  """Generates a list of string IDs for a list of devices.

  Args:
    devices: A sequence of devices, a numpy array of devices, or None.
      Devices can be represented as strings or JAX device objects.

  Returns:
    A list of string representations of the device IDs.
  """
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    device_list = devices.flatten().tolist()
  else:
    device_list = devices
  return [generate_device_timeline_id(device) for device in device_list]


# Utility functions for handling spans.


def is_timeline_only_of_allowed_type(
    tl: timeline.Timeline, allowed_span_names: Sequence[str]
) -> bool:
  """Checks if all spans in a timeline are of allowed types.

  Args:
    tl: The timeline to check.
    allowed_span_names: A sequence of allowed span names.

  Returns:
    True if the timeline has spans and all spans have a name in
    `allowed_span_names`, False otherwise.
  """
  if not tl.spans:
    return False
  return all(span.name in allowed_span_names for span in tl.spans.values())
