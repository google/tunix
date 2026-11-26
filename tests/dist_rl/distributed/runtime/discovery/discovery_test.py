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
from tunix.dist_rl.distributed.runtime.discovery import discovery


class DiscoveryTest(absltest.TestCase):

  def test_start_with_zero_port_raises(self):
    server = discovery.DiscoveryServer()
    with self.assertRaises(ValueError):
      server.start(0, lambda h, p, m: None)

  @mock.patch.object(grpc, "server")
  def test_start_twice_raises(self, mock_grpc_server):
    server = discovery.DiscoveryServer()
    port = 8888
    server.start(port, lambda h, p, m: None)
    try:
      with self.assertRaises(RuntimeError):
        server.start(port, lambda h, p, m: None)
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

    server.start(port, callback)
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
        "tunix.dist_rl.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
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
        "tunix.dist_rl.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        return_value=mock_stub,
    ):
      with self.assertRaises(RuntimeError):
        discovery.register("localhost:9999", "node-0", 1234, b"meta")

  @mock.patch.object(grpc, "insecure_channel")
  @mock.patch("time.sleep")
  def test_register_follows_moved_server(self, mock_sleep, mock_channel):
    del mock_sleep
    unavailable_error = grpc.RpcError()
    unavailable_error.code = lambda: grpc.StatusCode.UNAVAILABLE
    unavailable_error.details = lambda: "unavailable"
    stale_stub = mock.MagicMock()
    stale_stub.Register.side_effect = unavailable_error
    new_stub = mock.MagicMock()
    channels = {}

    def make_channel(address):
      channels[address] = mock.MagicMock(name=address)
      return channels[address]

    mock_channel.side_effect = make_channel

    def make_stub(channel):
      return stale_stub if channel is channels["10.0.0.1:1"] else new_stub

    with mock.patch(
        "tunix.dist_rl.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        side_effect=make_stub,
    ):
      discovery.register(
          "10.0.0.1:1",
          "node-0",
          1234,
          b"meta",
          address_resolver=lambda: "10.0.0.2:1",
      )

    self.assertEqual(
        [c.args[0] for c in mock_channel.call_args_list],
        ["10.0.0.1:1", "10.0.0.2:1"],
    )
    new_stub.Register.assert_called_once()
    channels["10.0.0.1:1"].close.assert_called_once()
    channels["10.0.0.2:1"].close.assert_called_once()

  @mock.patch.object(grpc, "insecure_channel")
  @mock.patch("time.sleep")
  def test_register_resets_backoff_for_moved_server(
      self, mock_sleep, mock_channel
  ):
    unavailable_error = grpc.RpcError()
    unavailable_error.code = lambda: grpc.StatusCode.UNAVAILABLE
    unavailable_error.details = lambda: "unavailable"
    stale_stub = mock.MagicMock()
    stale_stub.Register.side_effect = unavailable_error
    new_stub = mock.MagicMock()
    new_stub.Register.side_effect = [unavailable_error, None]
    channels = {}

    def make_channel(address):
      channels[address] = mock.MagicMock(name=address)
      return channels[address]

    mock_channel.side_effect = make_channel

    def make_stub(channel):
      return stale_stub if channel is channels["10.0.0.1:1"] else new_stub

    # The server moves after 60 failed attempts, once the backoff has doubled.
    addresses = iter(["10.0.0.1:1"] * 59 + ["10.0.0.2:1"] * 10)
    with mock.patch(
        "tunix.dist_rl.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        side_effect=make_stub,
    ):
      discovery.register(
          "10.0.0.1:1",
          "node-0",
          1234,
          b"meta",
          address_resolver=lambda: next(addresses),
      )

    delays = [c.args[0] for c in mock_sleep.call_args_list]
    self.assertEqual(delays, [1] * 61)

  @mock.patch.object(grpc, "insecure_channel")
  @mock.patch("time.sleep")
  def test_register_keeps_address_when_resolver_fails(
      self, mock_sleep, mock_channel
  ):
    del mock_sleep
    mock_stub = mock.MagicMock()
    unavailable_error = grpc.RpcError()
    unavailable_error.code = lambda: grpc.StatusCode.UNAVAILABLE
    unavailable_error.details = lambda: "unavailable"
    mock_stub.Register.side_effect = [unavailable_error, None]

    def failing_resolver():
      raise RuntimeError("BNS lookup failed")

    with mock.patch(
        "tunix.dist_rl.distributed.runtime.discovery.discovery_service_pb2_grpc.DiscoveryServiceStub",
        return_value=mock_stub,
    ):
      discovery.register(
          "10.0.0.1:1",
          "node-0",
          1234,
          b"meta",
          address_resolver=failing_resolver,
      )

    mock_channel.assert_called_once_with("10.0.0.1:1")
    self.assertEqual(mock_stub.Register.call_count, 2)

  def _start_and_get_servicer(self, callback):
    """Starts a `DiscoveryServer` on a mocked gRPC server; returns its servicer."""
    server = discovery.DiscoveryServer()
    with (
        mock.patch.object(grpc, "server"),
        mock.patch.object(
            discovery.pb2_grpc, "add_DiscoveryServiceServicer_to_server"
        ) as mock_add,
    ):
      server.start(8888, callback)
    self.addCleanup(server.stop)
    return mock_add.call_args.args[0]

  def test_registration_unavailable_error_aborts_with_unavailable(self):
    def callback(hostname, p, metadata):
      del hostname, p, metadata
      raise discovery.RegistrationUnavailableError("registration is closed")

    servicer = self._start_and_get_servicer(callback)
    context = mock.MagicMock()
    servicer.Register(
        discovery.pb2.RegisterRequest(hostname="h", port=1), context
    )

    # `register` retries `UNAVAILABLE` (`test_register_retry_on_unavailable`).
    context.abort.assert_called_once_with(
        grpc.StatusCode.UNAVAILABLE, "registration is closed"
    )

  def test_registration_callback_error_is_not_mapped_to_unavailable(self):
    def callback(hostname, p, metadata):
      del hostname, p, metadata
      raise ValueError("config mismatch")

    servicer = self._start_and_get_servicer(callback)
    context = mock.MagicMock()
    with self.assertRaisesRegex(ValueError, "config mismatch"):
      servicer.Register(
          discovery.pb2.RegisterRequest(hostname="h", port=1), context
      )
    context.abort.assert_not_called()


if __name__ == "__main__":
  absltest.main()
