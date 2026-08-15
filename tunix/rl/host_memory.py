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

"""Small, dependency-free host-memory telemetry helpers."""

from __future__ import annotations

import gc
import os
from typing import Callable, Mapping


def _read_optional_text(path: str) -> str | None:
  try:
    with open(path, encoding="utf-8") as stream:
      return stream.read().strip()
  except (FileNotFoundError, OSError):
    return None


def _parse_optional_bytes(value: str | None) -> int | None:
  if value in (None, "", "max"):
    return None
  try:
    parsed = int(value)
  except ValueError:
    return None
  return parsed if parsed >= 0 else None


def snapshot(
    *,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_status_path: str = "/proc/self/status",
) -> dict[str, int | None]:
  """Returns best-effort cgroup and process host-memory counters in bytes."""
  current = _parse_optional_bytes(
      _read_optional_text(os.path.join(cgroup_root, "memory.current"))
  )
  peak = _parse_optional_bytes(
      _read_optional_text(os.path.join(cgroup_root, "memory.peak"))
  )
  limit = _parse_optional_bytes(
      _read_optional_text(os.path.join(cgroup_root, "memory.max"))
  )
  if current is None:
    memory_root = os.path.join(cgroup_root, "memory")
    current = _parse_optional_bytes(
        _read_optional_text(os.path.join(memory_root, "memory.usage_in_bytes"))
    )
    peak = _parse_optional_bytes(
        _read_optional_text(
            os.path.join(memory_root, "memory.max_usage_in_bytes")
        )
    )
    limit = _parse_optional_bytes(
        _read_optional_text(os.path.join(memory_root, "memory.limit_in_bytes"))
    )

  rss_bytes = None
  high_water_bytes = None
  status = _read_optional_text(proc_status_path)
  if status is not None:
    for line in status.splitlines():
      fields = line.split()
      if len(fields) < 2:
        continue
      try:
        value_bytes = int(fields[1]) * 1024
      except ValueError:
        continue
      if fields[0] == "VmRSS:":
        rss_bytes = value_bytes
      elif fields[0] == "VmHWM:":
        high_water_bytes = value_bytes
  return {
      "cgroup_current_bytes": current,
      "cgroup_peak_bytes": peak,
      "cgroup_limit_bytes": limit,
      "process_rss_bytes": rss_bytes,
      "process_hwm_bytes": high_water_bytes,
  }


def contract(environ: Mapping[str, str]) -> tuple[bool, int]:
  """Returns the fail-closed P45 telemetry and cyclic-GC contract."""
  enabled_raw = environ.get("CANON_P45_HOST_MEMORY_TELEMETRY", "0")
  if enabled_raw not in ("0", "1"):
    raise ValueError(
        "CANON_P45_HOST_MEMORY_TELEMETRY must be exactly 0 or 1"
    )
  if enabled_raw == "0":
    return False, 0
  interval_raw = environ.get("CANON_P45_HOST_GC_INTERVAL", "")
  try:
    interval = int(interval_raw)
  except ValueError as exc:
    raise ValueError(
        "CANON_P45_HOST_GC_INTERVAL must be a positive integer"
    ) from exc
  if interval <= 0:
    raise ValueError("CANON_P45_HOST_GC_INTERVAL must be a positive integer")
  return True, interval


def record(
    *,
    phase: str,
    step: int,
    gc_collected: int | None = None,
) -> dict[str, int | str | None]:
  return {
      "schema": "canon.p45.host-memory.v1",
      "phase": phase,
      "step": int(step),
      "gc_collected": gc_collected,
      **snapshot(),
  }


def maybe_collect_garbage(
    *, step: int, interval: int, collector: Callable[[], int] = gc.collect
) -> int | None:
  """Runs cyclic GC only on the configured committed-step cadence."""
  if interval <= 0:
    raise ValueError("P45 host GC interval must be positive")
  return int(collector()) if step % interval == 0 else None
