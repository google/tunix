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

"""Compatibility shim for legacy MaxText commits importing `tunix.experimental`."""

from typing import Any, Optional

from tunix.dist_rl.weight_sync import raiden_synchronizer
from tunix.dist_rl.weight_sync import weight_sync as _weight_sync
from tunix.dist_rl.weight_sync.weight_sync import *  # pylint: disable=wildcard-import,unused-wildcard-import


# TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Remove this shim module and
# `create_weight_synchronizer` once pinned `--maxtext-commit=a2cf3bc8b` images
# are retired and all callers construct `raiden_synchronizer.RaidenSynchronizer`
# directly via `tunix.dist_rl.weight_sync`.
def create_weight_synchronizer(
    mode: _weight_sync.WeightSyncMode | str,
    job_name: str,
    state: Any = None,
    *,
    worker_index: int = 0,
    staging_dir: Optional[str] = None,
    **kwargs: Any,
) -> raiden_synchronizer.RaidenSynchronizer:
  """Legacy shim for older MaxText commits; prefer `RaidenSynchronizer`.

  Args:
    mode: Weight synchronization transport mode (`WeightSyncMode.RAIDEN` or
      `"raiden"`, case-insensitive). Legacy `"gcs"` and `"file"` modes are
      unsupported and raise `ValueError`.
    job_name: Logical role name for work-unit registration (e.g. `"trainer"`).
    state: Optional initial model weight state to bind immediately.
    worker_index: Zero-based worker replica index.
    staging_dir: Accepted for compatibility with older MaxText commits and
      ignored by the Raiden transport.
    **kwargs: Additional keyword arguments forwarded to `RaidenSynchronizer`.

  Returns:
    A configured `RaidenSynchronizer` instance.

  Raises:
    ValueError: If `mode` is not `WeightSyncMode.RAIDEN` or `"raiden"`.
  """
  del staging_dir
  if isinstance(mode, _weight_sync.WeightSyncMode):
    resolved_mode = mode
  elif isinstance(mode, str):
    try:
      resolved_mode = _weight_sync.WeightSyncMode(mode.strip().lower())
    except ValueError as exc:
      raise ValueError(
          f"Unsupported WeightSyncMode {mode!r} for WeightSynchronizer; only"
          " 'raiden' is supported."
      ) from exc
  else:
    raise ValueError(
        f"Unsupported WeightSyncMode {mode!r} for WeightSynchronizer; only"
        " 'raiden' is supported."
    )
  if resolved_mode == _weight_sync.WeightSyncMode.RAIDEN:
    return raiden_synchronizer.RaidenSynchronizer(
        job_name=job_name,
        state=state,
        worker_index=worker_index,
        **kwargs,
    )
  raise ValueError(
      f"Unsupported WeightSyncMode {resolved_mode!r} for WeightSynchronizer;"
      " only WeightSyncMode.RAIDEN is supported."
  )
