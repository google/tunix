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

"""Turning a worker that announced itself into one the orchestrator can call.

The runtime can tell a process when a peer registers, and a worker can
announce itself. What has been missing is everything between: the announcement
carries opaque bytes with no agreed shape, and nothing turns "a peer appeared
at this host and port" into a handle in the registry. Without that seam the
orchestrator has to be told where its workers are, which is why every run so
far has had ports written into it.

This supplies both halves. An announcement is a named record rather than an
ad-hoc dictionary, so the two sides cannot disagree about whether the field is
called the port or the service port. And a registrar builds the right kind of
handle for the announced role and registers it, so a worker that starts joins
the fleet by saying so.

Ordering matters and is the worker's responsibility: announce only once the
port accepts connections, or the orchestrator will build a handle to something
that is not listening yet.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping, Optional

from absl import logging

ROLE_TRAINER = "trainer"
ROLE_ROLLOUT = "rollout"
ROLE_INFERENCE = "inference"

_ENCODING = "utf-8"


@dataclasses.dataclass(frozen=True)
class WorkerAnnouncement:
  """What a worker says about itself when it joins.

  Attributes:
    role: Which role it serves.
    worker_id: Its identity in the registry.
    port: The port its server accepts connections on.
    host: Where to reach it; filled in by the receiver when the transport
      already knows.
    resources: What it declares about its configuration, for the startup
      agreement check.
  """

  role: str
  worker_id: str
  port: int
  host: str = "localhost"
  resources: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def encode(self) -> bytes:
    """Serializes for the discovery channel, which carries bytes."""
    return json.dumps(dataclasses.asdict(self)).encode(_ENCODING)

  @classmethod
  def decode(cls, payload: bytes) -> "WorkerAnnouncement":
    """Rebuilds an announcement.

    Args:
      payload: What `encode` produced.

    Returns:
      The announcement.

    Raises:
      ValueError: If the payload is not an announcement this version
        understands. Better to say so than to register a handle pointing
        somewhere arbitrary.
    """
    try:
      fields = json.loads(payload.decode(_ENCODING))
      return cls(
          role=fields["role"],
          worker_id=fields["worker_id"],
          port=int(fields["port"]),
          host=fields.get("host", "localhost"),
          resources=fields.get("resources", {}),
      )
    except (ValueError, KeyError, UnicodeDecodeError) as e:
      raise ValueError(
          f"Unrecognized worker announcement: {e}. Every worker joining this"
          " fleet must announce in the shared format."
      ) from e

  def address(self, host: Optional[str] = None) -> str:
    """The URI to build a handle from, preferring what the transport saw."""
    return f"grpc://{host or self.host}:{self.port}"


class DiscoveryRegistrar:
  """Registers workers into a fleet as they announce themselves."""

  def __init__(
      self,
      registry: Any,
      handle_factories: Mapping[str, Callable[[str, str], Any]],
      *,
      on_registered: Optional[Callable[[WorkerAnnouncement, Any], None]] = None,
  ):
    """Initializes the registrar.

    Args:
      registry: Receives each worker that joins.
      handle_factories: Role to a callable taking `(address, worker_id)` and
        returning the handle to register. A role with no factory is ignored,
        loudly: a fleet should not silently drop a worker that showed up.
      on_registered: Optional notification after a worker joins.
    """
    self._registry = registry
    self._handle_factories = dict(handle_factories)
    self._on_registered = on_registered
    self.registered: list[WorkerAnnouncement] = []

  def subscribe(self, discovery: Any) -> None:
    """Starts listening for peers on a discovery context."""
    discovery.on_register(self.on_peer_registered)

  def on_peer_registered(
      self, hostname: str, port: int, metadata: bytes
  ) -> None:
    """Handles one announcement, as the runtime delivers it.

    Args:
      hostname: Where the announcement came from.
      port: The discovery port it came through, which is not the port the
        worker serves on -- that is in the announcement.
      metadata: The encoded announcement.
    """
    del port
    try:
      announcement = WorkerAnnouncement.decode(metadata)
    except ValueError as e:
      logging.error("Ignoring an announcement that could not be read: %s", e)
      return
    self.register(announcement, hostname)

  def register(
      self, announcement: WorkerAnnouncement, hostname: Optional[str] = None
  ) -> Any:
    """Builds a handle for an announced worker and adds it to the registry.

    Args:
      announcement: What the worker said about itself.
      hostname: Where it was seen, preferred over what it claimed.

    Returns:
      The registered handle, or None when the role has no factory.
    """
    factory = self._handle_factories.get(announcement.role)
    if factory is None:
      logging.error(
          "Worker %r announced role %r, which this fleet has no handle for;"
          " it will not receive work.",
          announcement.worker_id,
          announcement.role,
      )
      return None

    handle = factory(
        announcement.address(hostname), announcement.worker_id
    )
    self._registry.register(handle)
    self.registered.append(announcement)
    logging.info(
        "Registered %s worker %r at %s",
        announcement.role,
        announcement.worker_id,
        announcement.address(hostname),
    )
    if self._on_registered is not None:
      self._on_registered(announcement, handle)
    return handle


def announce(
    discovery: Any, announcement: WorkerAnnouncement
) -> None:
  """Tells the fleet this worker exists.

  Call only after the server is accepting connections: the orchestrator builds
  a handle as soon as it hears, and a handle to a port that is not listening
  yet fails on first use.

  Args:
    discovery: The discovery context to announce through.
    announcement: What to say.
  """
  discovery.register(announcement.encode())
