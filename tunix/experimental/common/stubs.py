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

"""Stub implementations for Tunix experimental workers and samplers."""

from typing import Any


class MockWeightSynchronizer:
  """A mock weight synchronizer for local vLLM rollout workers."""


class RaidenSamplerServer:
  """A dummy sampler server wrapping a vLLM endpoint."""

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs

  def initialize(self) -> None:
    pass


class StubEnvironmentPool:
  """A dummy environment pool for rollout collection."""

  def __init__(self, pool_size: int) -> None:
    self.pool_size = pool_size


class StubAgent:
  """A dummy agent for rollout collection."""
