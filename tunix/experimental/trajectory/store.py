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

"""Protocols and factory registry for Trajectory Store implementations."""

from tunix.experimental.trajectory import base_store
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import sql_store

# Ensure built-in backends are imported so their __init_subclass__ hooks
# register them in TrajectoryStore._REGISTRY before TrajectoryStore.from_config
# lookup.
_BUILTIN_BACKENDS = (
    file_store.FileTrajectoryStore,
    in_memory_store.InMemoryTrajectoryStore,
    sql_store.SqlTrajectoryStore,
)

MetadataT = base_store.MetadataT
TrajectoryNotFoundError = base_store.TrajectoryNotFoundError
TrajectoryMetadataNotFoundError = base_store.TrajectoryMetadataNotFoundError
TrajectoryReader = base_store.TrajectoryReader
TrajectoryWriter = base_store.TrajectoryWriter
TrajectoryStore = base_store.TrajectoryStore
TunixTrajectoryStore = base_store.TunixTrajectoryStore
