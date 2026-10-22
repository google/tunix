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

"""Test helpers for `WorkerRegistry`."""

from tunix.dist_rl.orchestrator import worker_registry


def set_status(
    registry: worker_registry.WorkerRegistry,
    worker_id: str,
    status: worker_registry.AdmissionStatus,
) -> None:
  """Moves `worker_id` to `status` from whatever status it is in now.

  Stands in for `LifecycleDriver` and `DistributedRLEngine` in tests that build
  a registry by hand.

  Args:
    registry: The registry holding `worker_id`.
    worker_id: The worker to update.
    status: The new status.

  Raises:
    RuntimeError: If the worker's handle or status changed concurrently.
  """
  if not registry.set_status(
      worker_id,
      status,
      expected_handle=registry.handle(worker_id),
      expected_status=registry.status(worker_id),
  ):
    raise RuntimeError(f"worker {worker_id!r} changed while setting its status")
