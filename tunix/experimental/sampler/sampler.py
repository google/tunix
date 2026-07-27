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

"""Sampler abstractions for Distributed-RL."""

from typing import Any, Protocol, Sequence

from tunix.experimental.common import datatypes


class Sampler(Protocol):
  """Protocol defining standard lifecycle, sampling, and weight-sync interface for worker slices."""

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling server engine or local loop."""
    ...

  async def stop(self, **kwargs) -> str | None | Any:
    """Terminates sampler execution and closes local connections."""
    ...

  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    ...

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    ...

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    ...

  # --- Inference ---
  async def sample(
      self,
      sampling_requests: (
          datatypes.SamplingRequest
          | Sequence[datatypes.SamplingRequest]
          | Any
          | Sequence[Any]
      ),
      **kwargs,
  ) -> list[datatypes.SamplingResult] | Any:
    """Generates completions for a batch of prompt conversations concurrently."""
    ...

  # --- Weight Synchronization ---
  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sharding specs and layout metadata across devices for policy model weights."""
    ...

  async def pre_weight_sync(
      self, sync_request: datatypes.WeightSyncRequest | Any = None, **kwargs
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update from the specified controller."""
    ...

  async def weight_sync(
      self, sync_request: datatypes.WeightSyncRequest | Any = None, **kwargs
  ) -> str | None | Any:
    """Updates model weights in-place from the specified controller."""
    ...

  async def post_weight_sync(
      self, sync_request: datatypes.WeightSyncRequest | Any = None, **kwargs
  ) -> str | None | Any:
    """Finalizes and switches active policy weights after transfer completion."""
    ...

  async def get_transfer_status(
      self, req_id: str | Any, **kwargs
  ) -> str | Any:
    """Queries the status of an ongoing weight transfer or KV-cache migration request."""
    ...
