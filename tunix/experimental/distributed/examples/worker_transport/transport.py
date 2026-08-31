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

"""Generic transport helper functions for worker actor creation.

This module provides helper functions to initialize worker actor handles across
three deployment cases:

  1. Same-process Worker (In-process):
       local_handle = local(Worker, name="local")

  2. Same-host Worker (Separate local OS process):
       remote_handle = remote(Worker, address="grpc://localhost:12345")

  3. Remote-host Worker (Distributed network process / K8s pod):
       remote_handle = remote(Worker, address=discovered_address)
"""

from typing import Any

from tunix.experimental.worker import remote_execution


def local(cls: Any, *args: Any, **kwargs: Any) -> remote_execution.ActorHandle:
  """Creates a local, in-process ActorHandle for target class `cls` instantiated with `*args, **kwargs`."""
  return remote_execution.remote(cls, transport="inprocess").remote(
      *args, **kwargs
  )


def remote(cls: Any, address: str) -> remote_execution.ActorHandle:
  """Creates a gRPC remote ActorHandle for target class `cls` at network address `address`."""
  return remote_execution.remote(
      cls, transport="grpc", address=address
  ).remote()
