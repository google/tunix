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

"""A worker joins the fleet by announcing itself, not by being configured in."""

from typing import Any

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_discovery
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker


class _Handle(abstract_worker.Worker):
  """A stand-in for whatever the factory builds."""

  def __init__(self, address: str, worker_id: str, role: str = "rollout"):
    self.address = address
    self._worker_id = worker_id
    self._role = role

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({self._role})
    )

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    return datatypes.Response()

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=datatypes.WorkerState.READY)


class _Discovery:
  """The discovery context, as the runtime presents it."""

  def __init__(self):
    self.callback = None
    self.announced: list[bytes] = []

  def on_register(self, callback):
    self.callback = callback

  def register(self, metadata: bytes) -> None:
    self.announced.append(metadata)


def _registrar(registry):
  return worker_discovery.DiscoveryRegistrar(
      registry,
      {
          worker_discovery.ROLE_ROLLOUT: (
              lambda address, worker_id: _Handle(address, worker_id, "rollout")
          ),
          worker_discovery.ROLE_TRAINER: (
              lambda address, worker_id: _Handle(address, worker_id, "trainer")
          ),
      },
  )


class AnnouncementTest(absltest.TestCase):

  def test_it_round_trips(self):
    original = worker_discovery.WorkerAnnouncement(
        role="rollout",
        worker_id="r0",
        port=1234,
        host="worker-3",
        resources={"tokenizer_hash": "sha"},
    )

    restored = worker_discovery.WorkerAnnouncement.decode(original.encode())

    self.assertEqual(restored, original)
    self.assertEqual(restored.address(), "grpc://worker-3:1234")

  def test_where_it_was_seen_beats_what_it_claimed(self):
    announcement = worker_discovery.WorkerAnnouncement(
        role="rollout", worker_id="r0", port=99, host="stale-name"
    )

    self.assertEqual(announcement.address("10.0.0.4"), "grpc://10.0.0.4:99")

  def test_an_unreadable_announcement_is_rejected(self):
    with self.assertRaises(ValueError):
      worker_discovery.WorkerAnnouncement.decode(b"not an announcement")

  def test_an_announcement_missing_a_field_is_rejected(self):
    with self.assertRaises(ValueError):
      worker_discovery.WorkerAnnouncement.decode(b'{"role": "rollout"}')


class DiscoveryRegistrarTest(absltest.TestCase):

  def test_an_announced_worker_joins_the_registry(self):
    registry = worker_registry.WorkerRegistry()
    registrar = _registrar(registry)
    discovery = _Discovery()
    registrar.subscribe(discovery)

    worker_discovery.announce(
        discovery,
        worker_discovery.WorkerAnnouncement(
            role="rollout", worker_id="r0", port=5001
        ),
    )
    # The runtime delivers what was announced.
    discovery.callback("10.0.0.1", 7000, discovery.announced[0])

    self.assertEqual(registry.worker_ids(), ["r0"])
    self.assertEqual(
        registry.group("rollout").members()[0].address, "grpc://10.0.0.1:5001"
    )

  def test_each_role_gets_the_handle_built_for_it(self):
    registry = worker_registry.WorkerRegistry()
    registrar = _registrar(registry)

    registrar.register(
        worker_discovery.WorkerAnnouncement(
            role="trainer", worker_id="t0", port=1
        )
    )
    registrar.register(
        worker_discovery.WorkerAnnouncement(
            role="rollout", worker_id="r0", port=2
        )
    )

    self.assertEqual(registry.roles(), {"trainer", "rollout"})

  def test_a_role_the_fleet_cannot_serve_is_not_silently_dropped(self):
    registry = worker_registry.WorkerRegistry()
    registrar = _registrar(registry)

    handle = registrar.register(
        worker_discovery.WorkerAnnouncement(
            role="reward_model", worker_id="x0", port=3
        )
    )

    self.assertIsNone(handle)
    self.assertEmpty(registry.worker_ids())

  def test_a_malformed_announcement_does_not_take_the_listener_down(self):
    registry = worker_registry.WorkerRegistry()
    registrar = _registrar(registry)
    discovery = _Discovery()
    registrar.subscribe(discovery)

    discovery.callback("10.0.0.1", 7000, b"garbage")
    discovery.callback(
        "10.0.0.2",
        7000,
        worker_discovery.WorkerAnnouncement(
            role="rollout", worker_id="r1", port=5002
        ).encode(),
    )

    # The good one still got through.
    self.assertEqual(registry.worker_ids(), ["r1"])

  def test_declared_resources_travel_with_the_announcement(self):
    """So the startup agreement check has something to compare."""
    registry = worker_registry.WorkerRegistry()
    registrar = _registrar(registry)

    registrar.register(
        worker_discovery.WorkerAnnouncement(
            role="rollout",
            worker_id="r0",
            port=1,
            resources={"tokenizer_hash": "sha", "pad_id": 0},
        )
    )

    self.assertEqual(
        registrar.registered[0].resources["tokenizer_hash"], "sha"
    )


if __name__ == "__main__":
  absltest.main()
