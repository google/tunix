# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for gRPC peer discovery server and registration helper."""

import threading
from unittest import mock

from absl.testing import absltest
import grpc
import portpicker
from tunix.experimental.distributed.runtime.discovery import discovery


class DiscoveryTest(absltest.TestCase):

  def test_start_unconfigured_mode_raises(self):
    server = discovery.DiscoveryServer()
    with self.assertRaises(RuntimeError):
      server.start(8888)

  def test_start_with_zero_port_raises(self):
    server = discovery.DiscoveryServer()
    server.on_register(lambda h, p, m: None)
    with self.assertRaises(ValueError):
      server.start(0)

  @mock.patch.object(grpc, "server")
  def test_start_twice_raises(self, mock_grpc_server):
    server = discovery.DiscoveryServer()
    port = 8888
    server.on_register(lambda h, p, m: None)
    server.start(port)
    try:
      with self.assertRaises(RuntimeError):
        server.start(port)
    finally:
      server.stop()

  @mock.patch.object(grpc, "server")
  def test_server_register_rpc(self, mock_grpc_server):
    server = discovery.DiscoveryServer()
    port = 8888
    received = {}

    def callback(hostname, p, metadata):
      received["hostname"] = hostname
      received["port"] = p
      received["metadata"] = metadata

    server.on_register(callback)
    server.start(port)
    try:
      self.assertTrue(server.is_started())
      mock_grpc_server.return_value.add_insecure_port.assert_called_once_with(
          f"[::]:{port}"
      )
      mock_grpc_server.return_value.start.assert_called_once()
    finally:
      server.stop()

  def test_register_empty_address_raises(self):
    with self.assertRaises(ValueError):
      discovery.register("", "node-0", 1234, b"meta")

  @mock.patch.object(grpc, "insecure_channel")
  @mock.patch("time.sleep")
  def test_register_retry_on_unavailable(self, mock_sleep, mock_channel):
    mock_stub_cls = mock.MagicMock()
    mock_stub = mock_stub_cls.return_value

    unavailable_error = grpc.RpcError()
    unavailable_error.code = lambda: grpc.StatusCode.UNAVAILABLE
    unavailable_error.details = lambda: "unavailable"

    mock_stub.Register.side_effect = [unavailable_error, None]

    with mock.patch(
        "tunix.experimental.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        return_value=mock_stub,
    ):
      discovery.register("localhost:9999", "node-0", 1234, b"meta")

    self.assertEqual(mock_stub.Register.call_count, 2)
    mock_sleep.assert_called_once()

  @mock.patch.object(grpc, "insecure_channel")
  def test_register_non_retryable_error_raises(self, mock_channel):
    mock_stub = mock.MagicMock()
    invalid_error = grpc.RpcError()
    invalid_error.code = lambda: grpc.StatusCode.INVALID_ARGUMENT
    invalid_error.details = lambda: "invalid arg"
    mock_stub.Register.side_effect = invalid_error

    with mock.patch(
        "tunix.experimental.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        return_value=mock_stub,
    ):
      with self.assertRaises(RuntimeError):
        discovery.register("localhost:9999", "node-0", 1234, b"meta")

  def test_connect_initial_connection(self):
    port = portpicker.pick_unused_port()
    server = discovery.DiscoveryServer(heartbeat_sec=1)
    server_connected_events = []

    server.on_connect(
        on_client_connected=lambda cid, h, p, m, rec: server_connected_events.append(
            (cid, h, p, m, rec)
        )
    )
    server.start(port, heartbeat_sec=1)

    client_connected_events = []
    try:
      client = discovery.connect(
          f"localhost:{port}",
          "node-0",
          1234,
          b"meta-data",
          client_id="node-0",
          on_connected=lambda epoch, rec: client_connected_events.append(
              (epoch, rec)
          ),
      )

      self.assertEqual(len(client_connected_events), 1)
      epoch, is_reconnect = client_connected_events[0]
      self.assertFalse(is_reconnect)

      self.assertEqual(len(server_connected_events), 1)
      cid, h, p, m, is_rec = server_connected_events[0]
      self.assertEqual(cid, "node-0")
      self.assertFalse(is_rec)

      client.stop()
    finally:
      server.stop()

  def test_connect_reconnect_on_server_restart(self):
    port = portpicker.pick_unused_port()
    server = discovery.DiscoveryServer(heartbeat_sec=1)
    server_connected_events = []

    server.on_connect(
        on_client_connected=lambda cid, h, p, m, rec: server_connected_events.append(
            (cid, h, p, m, rec)
        )
    )
    server.start(port, heartbeat_sec=1)

    client_connected_events = []
    client_disconnected_events = []
    reconnected_event = threading.Event()

    def on_connected(epoch, rec):
      client_connected_events.append((epoch, rec))
      if rec:
        reconnected_event.set()

    try:
      client = discovery.connect(
          f"localhost:{port}",
          "node-0",
          1234,
          b"meta-data",
          client_id="node-0",
          on_connected=on_connected,
          on_disconnected=lambda epoch, reason: client_disconnected_events.append(
              (epoch, reason)
          ),
      )

      initial_epoch = client_connected_events[0][0]

      # Simulate server restart by changing servicer epoch
      server._servicer._server_epoch = "rebooted-epoch-1234"

      # Wait for heartbeat loop to detect epoch mismatch and reconnect
      self.assertTrue(reconnected_event.wait(timeout=5.0))

      self.assertGreaterEqual(len(client_disconnected_events), 1)
      self.assertEqual(client_disconnected_events[0][0], initial_epoch)
      self.assertEqual(client_disconnected_events[0][1], "epoch_mismatch")

      self.assertGreaterEqual(len(client_connected_events), 2)
      new_epoch, is_reconnected = client_connected_events[1]
      self.assertTrue(is_reconnected)
      self.assertEqual(new_epoch, "rebooted-epoch-1234")

      self.assertGreaterEqual(len(server_connected_events), 2)
      self.assertTrue(server_connected_events[1][4])  # is_reconnect=True

      client.stop()
    finally:
      server.stop()

  def test_server_lease_eviction_on_heartbeat_timeout(self):
    port = portpicker.pick_unused_port()
    server = discovery.DiscoveryServer(heartbeat_sec=1)
    server_disconnected_events = []
    evicted_event = threading.Event()

    def on_disconnected(cid, h, p, reason):
      server_disconnected_events.append((cid, h, p, reason))
      evicted_event.set()

    server.on_connect(on_client_disconnected=on_disconnected)
    server.start(port, heartbeat_sec=1)

    try:
      client = discovery.connect(
          f"localhost:{port}",
          "node-0",
          1234,
          b"meta-data",
          client_id="node-0",
      )
      # Stop client heartbeat thread prematurely to simulate crashed/dead client
      client._stop_event.set()

      # Wait for server eviction loop (threshold = 3 * heartbeat_sec)
      self.assertTrue(evicted_event.wait(timeout=5.0))

      self.assertGreaterEqual(len(server_disconnected_events), 1)
      cid, h, p, reason = server_disconnected_events[0]
      self.assertEqual(cid, "node-0")
      self.assertEqual(reason, "heartbeat_timeout")

      client.stop()
    finally:
      server.stop()

  def test_mode_mutual_exclusion(self):
    server = discovery.DiscoveryServer()
    server.on_register(lambda h, p, m: None)
    with self.assertRaises(RuntimeError):
      server.on_connect(lambda cid, h, p, m, rec: None)

    server2 = discovery.DiscoveryServer()
    server2.on_connect(lambda cid, h, p, m, rec: None)
    with self.assertRaises(RuntimeError):
      server2.on_register(lambda h, p, m: None)


if __name__ == "__main__":
  absltest.main()
