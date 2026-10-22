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


if __name__ == "__main__":
  absltest.main()
