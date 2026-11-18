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

"""Tests for TcpWeightSyncHandler, TcpSourceExporter, and TcpReceiverServer."""

from __future__ import annotations

import concurrent.futures
import json
import os
import socket
import struct
import threading
from unittest import mock

from absl.testing import absltest
import ml_dtypes
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import tcp_handler
from tunix.dist_rl.weight_sync import weight_sync


class TcpNetworkingTest(absltest.TestCase):

  def test_format_parse_host_port(self):
    self.assertEqual(
        tcp_handler.format_host_port("127.0.0.1", 8080),
        "127.0.0.1:8080",
    )
    self.assertEqual(
        tcp_handler.format_host_port("::1", 8080),
        "[::1]:8080",
    )
    self.assertEqual(
        tcp_handler.format_host_port("[::1]", 8080),
        "[::1]:8080",
    )
    self.assertEqual(
        tcp_handler.parse_host_port("127.0.0.1:8080"),
        ("127.0.0.1", 8080),
    )
    self.assertEqual(
        tcp_handler.parse_host_port("[::1]:8080"),
        ("::1", 8080),
    )
    with self.assertRaises(ValueError):
      tcp_handler.parse_host_port("[::1:8080")

  def test_create_listener_socket(self):
    sock = tcp_handler._create_listener_socket(0)
    self.assertIsNotNone(sock)
    port = sock.getsockname()[1]
    self.assertGreater(port, 0)
    sock.close()

  def test_dynamic_address_picks_up_late_weight_sync_host_env(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("DIST_RL_WEIGHT_SYNC_HOST", None)
      exporter = tcp_handler.TcpSourceExporter(
          job_name="trainer_dyn", bind_port=0
      )
      receiver = tcp_handler.TcpReceiverServer(bind_port=0)
      try:
        addr1 = exporter.address
        self.assertIn(":", addr1)
        os.environ["DIST_RL_WEIGHT_SYNC_HOST"] = "worker-0.example.com"
        self.assertTrue(
            exporter.address.startswith("worker-0.example.com:")
        )
        self.assertTrue(
            receiver.address.startswith("worker-0.example.com:")
        )
      finally:
        exporter.close()
        receiver.close()


class TcpTransportSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.exporter = tcp_handler.TcpSourceExporter(job_name="trainer")
    self.receiver = tcp_handler.TcpReceiverServer(bind_port=0)
    self.handler = tcp_handler.TcpWeightSyncHandler()

  def tearDown(self):
    self.exporter.close()
    self.receiver.close()
    super().tearDown()

  def test_multi_round_sync_success(self):
    # Round 1
    t1 = np.arange(10, dtype=np.uint16)
    t2 = np.ones((4, 4), dtype=np.float32)
    hf_weights_r1 = {
        "model.layers.0.weight": tcp_handler.RawTensorPayload(
            buffer=t1, shape=(10,), dtype="bf16"
        ),
        "model.layers.1.weight": tcp_handler.RawTensorPayload(
            buffer=t2, shape=(4, 4), dtype="f32"
        ),
    }
    req1 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1}
    )

    src_units = self.exporter.prepare_weight_sync(req1, hf_weights_r1)
    self.assertLen(src_units, 1)
    self.handler.register_work_unit(src_units[0])

    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    res1 = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r1",
        generation=1,
    )
    self.assertTrue(res1.success, res1.message)

    # Receiver verifies weights
    popped_r1 = self.receiver.pop_round("r1", 1)
    np.testing.assert_array_equal(
        popped_r1["model.layers.0.weight"].buffer, t1
    )
    np.testing.assert_array_equal(
        popped_r1["model.layers.1.weight"].buffer, t2
    )

    self.exporter.release_weight_sync(req1)

    # Round 2 with updated weights
    t1_r2 = np.arange(10, 20, dtype=np.uint16)
    hf_weights_r2 = {
        "model.layers.0.weight": tcp_handler.RawTensorPayload(
            buffer=t1_r2, shape=(10,), dtype="bf16"
        )
    }
    req2 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r2", "uuid": 2}
    )

    src_units_r2 = self.exporter.prepare_weight_sync(req2, hf_weights_r2)
    self.handler.register_work_unit(src_units_r2[0])

    res2 = self.handler.transfer(
        src_units=[src_units_r2[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r2",
        generation=2,
    )
    self.assertTrue(res2.success)

    popped_r2 = self.receiver.pop_round("r2", 2)
    np.testing.assert_array_equal(
        popped_r2["model.layers.0.weight"].buffer, t1_r2
    )
    self.exporter.release_weight_sync(req2)

  def test_multi_chunk_pipelined_streaming(self):
    # Small chunk size to force multiple chunks and pipelined double-buffering
    handler = tcp_handler.TcpWeightSyncHandler(chunk_bytes=64)
    exporter = tcp_handler.TcpSourceExporter(job_name="trainer_stream")
    r1 = tcp_handler.TcpReceiverServer(bind_port=0)
    r2 = tcp_handler.TcpReceiverServer(bind_port=0)

    try:
      # 512 bytes total (256 uint16 elements) -> 8 chunks of 64 bytes
      t1 = np.arange(256, dtype=np.uint16)
      payloads = {
          "weights.layer.0": tcp_handler.RawTensorPayload(
              buffer=t1, shape=(256,), dtype="bf16"
          )
      }
      req = datatypes.WeightSyncRequest(
          request_id="1", extra_config={"req_id": "stream_r1", "uuid": 1}
      )
      src_units = exporter.prepare_weight_sync(req, payloads)
      handler.register_work_unit(src_units[0])

      dst_unit1 = weight_sync.WorkUnitMetadata(
          unit=weight_sync.WorkUnitId(job_name="rollout_1", data_replica_idx=0),
          control_plane_rpc_address=r1.address,
          variables=src_units[0].variables,
          transport_mode="tcp",
          use_ffi=False,
      )
      dst_unit2 = weight_sync.WorkUnitMetadata(
          unit=weight_sync.WorkUnitId(job_name="rollout_2", data_replica_idx=0),
          control_plane_rpc_address=r2.address,
          variables=src_units[0].variables,
          transport_mode="tcp",
          use_ffi=False,
      )
      handler.register_work_unit(dst_unit1)
      handler.register_work_unit(dst_unit2)

      res = handler.transfer(
          src_units=[src_units[0].unit],
          dst_units=[dst_unit1.unit, dst_unit2.unit],
          req_id="stream_r1",
          generation=1,
      )
      self.assertTrue(res.success, res.message)

      popped1 = r1.pop_round("stream_r1", 1)
      popped2 = r2.pop_round("stream_r1", 1)
      np.testing.assert_array_equal(popped1["weights.layer.0"].buffer, t1)
      np.testing.assert_array_equal(popped2["weights.layer.0"].buffer, t1)
    finally:
      exporter.close()
      r1.close()
      r2.close()

  def test_fault_injection_premature_eof(self):
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "err_req", "uuid": 1}
    )
    t1 = np.ones(100, dtype=np.uint16)
    src_units = self.exporter.prepare_weight_sync(
        req,
        {"w": tcp_handler.RawTensorPayload(
            buffer=t1, shape=(100,), dtype="bf16"
        )},
    )
    self.handler.register_work_unit(src_units[0])

    # Connect bad socket to receiver that closes immediately
    host, port = tcp_handler.parse_host_port(self.receiver.address)
    s = socket.create_connection((host, port))
    s.close()

    # Destination unit with invalid unreachable port
    bad_dst = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout_bad", data_replica_idx=0),
        control_plane_rpc_address="127.0.0.1:1",
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(bad_dst)

    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[bad_dst.unit],
        req_id="err_req",
        generation=1,
    )
    self.assertFalse(res.success)
    self.assertIn("TCP weight transfer failed", res.message)

  def test_fault_injection_unknown_source_staged_weights(self):
    # Exporter doesn't have the requested round staged
    t1 = np.ones(10, dtype=np.uint16)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {"w": tcp_handler.RawTensorPayload(
            buffer=t1, shape=(10,), dtype="bf16"
        )},
    )
    self.handler.register_work_unit(src_units[0])

    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    # Transfer with un-staged req_id
    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="missing_req",
        generation=1,
    )
    self.assertFalse(res.success)
    self.assertIn("Source exporter has no staged weights", res.message)

  def test_discard_before_receive_prevents_late_stage_leak(self):
    t1 = np.ones(10, dtype=np.uint16)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_abort", "uuid": 7}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {"w": tcp_handler.RawTensorPayload(
            buffer=t1, shape=(10,), dtype="bf16"
        )},
    )
    self.handler.register_work_unit(src_units[0])

    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    # Discard before transfer completes
    self.receiver.discard_round("r_abort", 7)

    # Now run transfer
    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r_abort",
        generation=7,
    )
    self.assertTrue(res.success)

    # Ensure round was NOT staged in receiver
    with self.assertRaises(RuntimeError):
      self.receiver.pop_round("r_abort", 7)

  def test_empty_dst_units_returns_failure(self):
    res = self.handler.transfer(
        src_units=[
            weight_sync.WorkUnitId(job_name="trainer", data_replica_idx=0)
        ],
        dst_units=[],
        req_id="r_empty_dst",
        generation=1,
    )
    self.assertFalse(res.success)
    self.assertIn("at least 1 dst_unit", res.message)

  def test_oversized_header_rejected_and_server_recovers(self):
    host, port = tcp_handler.parse_host_port(self.exporter.address)
    with socket.create_connection((host, port), timeout=5.0) as sock:
      sock.sendall(struct.pack(">Q", 32 * 1024 * 1024))
    r_host, r_port = tcp_handler.parse_host_port(self.receiver.address)
    with socket.create_connection((r_host, r_port), timeout=5.0) as sock:
      sock.sendall(struct.pack(">Q", 32 * 1024 * 1024))

    t1 = np.arange(4, dtype=np.uint16)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_recover", "uuid": 99}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {"w": tcp_handler.RawTensorPayload(
            buffer=t1, shape=(4,), dtype="bf16"
        )},
    )
    self.handler.register_work_unit(src_units[0])
    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r_recover",
        generation=99,
    )
    self.assertTrue(res.success, msg=res.message)
    staged = self.receiver.pop_round("r_recover", 99)
    np.testing.assert_array_equal(staged["w"].buffer, t1)

  def test_receiver_rejects_mismatched_num_bytes(self):
    r_host, r_port = tcp_handler.parse_host_port(self.receiver.address)
    header_bytes = json.dumps({
        "req_id": "r_bad_bytes",
        "generation": 5,
        "tensors": [{
            "name": "w",
            "shape": [4],
            "dtype": "bf16",
            "num_bytes": 999,
        }],
    }).encode("utf-8")
    closed_without_ack = False
    with socket.create_connection((r_host, r_port), timeout=5.0) as sock:
      sock.sendall(struct.pack(">Q", len(header_bytes)))
      sock.sendall(header_bytes)
      try:
        sock.sendall(np.zeros((4,), dtype=np.uint16).tobytes())
        closed_without_ack = not sock.recv(1)
      except (ConnectionResetError, BrokenPipeError):
        closed_without_ack = True
    self.assertTrue(closed_without_ack)
    with self.assertRaises(RuntimeError):
      self.receiver.pop_round("r_bad_bytes", 5)

  def test_transfer_rejects_oversized_source_header(self):
    listener = tcp_handler._create_listener_socket(0)
    port = listener.getsockname()[1]

    def _serve_oversized():
      try:
        conn, _ = listener.accept()
        with conn:
          raw_len = tcp_handler._recv_exact(conn, 8)
          (q_len,) = struct.unpack(">Q", raw_len)
          tcp_handler._recv_exact(conn, q_len)
          conn.sendall(
              struct.pack(">Q", tcp_handler._MAX_HEADER_BYTES + 1)
          )
      except OSError:
        pass

    t = threading.Thread(target=_serve_oversized, daemon=True)
    t.start()
    try:
      src_id = weight_sync.WorkUnitId(job_name="bad_src", data_replica_idx=0)
      dst_id = weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0)
      self.handler.register_work_unit(
          weight_sync.WorkUnitMetadata(
              unit=src_id,
              control_plane_rpc_address=f"127.0.0.1:{port}",
              variables=(),
              transport_mode="tcp",
              use_ffi=False,
          )
      )
      self.handler.register_work_unit(
          weight_sync.WorkUnitMetadata(
              unit=dst_id,
              control_plane_rpc_address=self.receiver.address,
              variables=(),
              transport_mode="tcp",
              use_ffi=False,
          )
      )
      res = self.handler.transfer(
          src_units=[src_id],
          dst_units=[dst_id],
          req_id="r_oversized",
          generation=1,
      )
      self.assertFalse(res.success)
      self.assertIn("exceeds limit", res.message)
    finally:
      listener.close()
      t.join(timeout=2.0)

  def test_receiver_rejects_huge_and_negative_shapes_before_alloc(self):
    r_host, r_port = tcp_handler.parse_host_port(self.receiver.address)
    bad_headers = [
        ("r_huge_shape", 10, [10**12], 8),
        ("r_neg_shape", 11, [-1], 4),
    ]
    with mock.patch.object(
        tcp_handler.np, "empty", wraps=tcp_handler.np.empty
    ) as mock_empty:
      for req_id, gen, shape, num_bytes in bad_headers:
        header_bytes = json.dumps({
            "req_id": req_id,
            "generation": gen,
            "tensors": [{
                "name": "w",
                "shape": shape,
                "dtype": "f32",
                "num_bytes": num_bytes,
            }],
        }).encode("utf-8")
        closed_without_ack = False
        with socket.create_connection((r_host, r_port), timeout=5.0) as sock:
          sock.sendall(struct.pack(">Q", len(header_bytes)))
          sock.sendall(header_bytes)
          try:
            closed_without_ack = not sock.recv(1)
          except (ConnectionResetError, BrokenPipeError):
            closed_without_ack = True
        self.assertTrue(closed_without_ack)
        with self.assertRaises(RuntimeError):
          self.receiver.pop_round(req_id, gen)
      mock_empty.assert_not_called()

  def test_ml_dtypes_bfloat16_transfer_roundtrip(self):
    t_bf16 = np.array([1.5, -2.25, 3.0, 0.5], dtype=ml_dtypes.bfloat16)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_bf16", "uuid": 1}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {
            "w": tcp_handler.RawTensorPayload(
                buffer=t_bf16, shape=(4,), dtype=""
            )
        },
    )
    self.handler.register_work_unit(src_units[0])
    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r_bf16",
        generation=1,
    )
    self.assertTrue(res.success, msg=res.message)
    popped = self.receiver.pop_round("r_bf16", 1)
    self.assertEqual(popped["w"].dtype, "bf16")
    self.assertEqual(popped["w"].shape, (4,))
    np.testing.assert_array_equal(
        popped["w"].buffer.view(ml_dtypes.bfloat16), t_bf16
    )

  def test_receiver_rejects_disallowed_dtype_tag_and_recovers(self):
    r_host, r_port = tcp_handler.parse_host_port(self.receiver.address)
    for idx, bad_dtype in enumerate(("V16", "O")):
      header_bytes = json.dumps({
          "req_id": f"r_bad_dtype_{idx}",
          "generation": idx + 1,
          "tensors": [{
              "name": "w",
              "shape": [1],
              "dtype": bad_dtype,
              "num_bytes": 16,
          }],
      }).encode("utf-8")
      closed_without_ack = False
      with socket.create_connection((r_host, r_port), timeout=5.0) as sock:
        sock.sendall(struct.pack(">Q", len(header_bytes)))
        sock.sendall(header_bytes)
        try:
          closed_without_ack = not sock.recv(1)
        except (ConnectionResetError, BrokenPipeError):
          closed_without_ack = True
      self.assertTrue(closed_without_ack)
      with self.assertRaises(RuntimeError):
        self.receiver.pop_round(f"r_bad_dtype_{idx}", idx + 1)

    # Listener remains healthy for a subsequent valid transfer.
    t_ok = np.array([1.0, 2.0], dtype=np.float32)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_after_bad_dtype", "uuid": 9}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {
            "w": tcp_handler.RawTensorPayload(
                buffer=t_ok, shape=(2,), dtype="f32"
            )
        },
    )
    self.handler.register_work_unit(src_units[0])
    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)
    res = self.handler.transfer(
        src_units=[src_units[0].unit],
        dst_units=[dst_unit.unit],
        req_id="r_after_bad_dtype",
        generation=9,
    )
    self.assertTrue(res.success, msg=res.message)
    popped = self.receiver.pop_round("r_after_bad_dtype", 9)
    np.testing.assert_array_equal(popped["w"].buffer, t_ok)

  def test_receiver_enforces_max_staged_rounds_and_recovers(self):
    t = np.array([1.0], dtype=np.float32)
    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=(),
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    def _send_round(req_id: str, gen: int) -> weight_sync.TransferResult:
      req = datatypes.WeightSyncRequest(
          extra_config={"req_id": req_id, "uuid": gen}
      )
      src_units = self.exporter.prepare_weight_sync(
          req,
          {
              "w": tcp_handler.RawTensorPayload(
                  buffer=t, shape=(1,), dtype="f32"
              )
          },
      )
      self.handler.register_work_unit(src_units[0])
      try:
        return self.handler.transfer(
            src_units=[src_units[0].unit],
            dst_units=[dst_unit.unit],
            req_id=req_id,
            generation=gen,
        )
      finally:
        self.exporter.release_weight_sync(req)

    for i in range(tcp_handler._MAX_STAGED_ROUNDS):
      res = _send_round(f"r_cap_{i}", i + 1)
      self.assertTrue(res.success, msg=res.message)

    # 17th unconsumed round must be rejected before staging.
    res_overflow = _send_round("r_cap_overflow", 17)
    self.assertFalse(res_overflow.success)
    with self.assertRaises(RuntimeError):
      self.receiver.pop_round("r_cap_overflow", 17)

    # Popping one round frees capacity.
    self.receiver.pop_round("r_cap_0", 1)
    res_after_pop = _send_round("r_cap_after_pop", 18)
    self.assertTrue(res_after_pop.success, msg=res_after_pop.message)

    # Discarding one round also frees capacity.
    self.receiver.discard_round("r_cap_1", 2)
    res_after_discard = _send_round("r_cap_after_discard", 19)
    self.assertTrue(res_after_discard.success, msg=res_after_discard.message)

  def test_extract_round_id_rejects_empty_request_id(self):
    with self.assertRaisesRegex(
        ValueError, "missing both req_id and request_id"
    ):
      tcp_handler._extract_round_id(
          datatypes.WeightSyncRequest(request_id="", extra_config={})
      )
    self.assertEqual(
        tcp_handler._extract_round_id(
            datatypes.WeightSyncRequest(request_id="req-1", extra_config={})
        ),
        "req-1",
    )

  def test_stalled_half_open_sockets_do_not_block_transfer(self):
    exp_host, exp_port = tcp_handler.parse_host_port(self.exporter.address)
    rec_host, rec_port = tcp_handler.parse_host_port(self.receiver.address)
    stalled_sockets: list[socket.socket] = []
    try:
      for _ in range(2):
        stalled_sockets.append(
            socket.create_connection((exp_host, exp_port), timeout=5.0)
        )
        stalled_sockets.append(
            socket.create_connection((rec_host, rec_port), timeout=5.0)
        )

      t = np.arange(8, dtype=np.float32)
      req = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_nonblocking", "uuid": 1}
      )
      src_units = self.exporter.prepare_weight_sync(
          req,
          {
              "w": tcp_handler.RawTensorPayload(
                  buffer=t, shape=(8,), dtype="f32"
              )
          },
      )
      self.handler.register_work_unit(src_units[0])
      dst_unit = weight_sync.WorkUnitMetadata(
          unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
          control_plane_rpc_address=self.receiver.address,
          variables=src_units[0].variables,
          transport_mode="tcp",
          use_ffi=False,
      )
      self.handler.register_work_unit(dst_unit)

      res = self.handler.transfer(
          src_units=[src_units[0].unit],
          dst_units=[dst_unit.unit],
          req_id="r_nonblocking",
          generation=1,
      )
      self.assertTrue(res.success, msg=res.message)
      popped = self.receiver.pop_round("r_nonblocking", 1)
      np.testing.assert_array_equal(popped["w"].buffer, t)
    finally:
      for sock in stalled_sockets:
        try:
          sock.close()
        except OSError:
          pass

  def test_receiver_rejects_exceeding_max_round_bytes_before_alloc(self):
    with self.assertRaisesRegex(ValueError, "max_round_bytes must be positive"):
      tcp_handler.TcpReceiverServer(bind_port=0, max_round_bytes=0)

    bounded_receiver = tcp_handler.TcpReceiverServer(
        bind_port=0, max_round_bytes=64
    )
    try:
      r_host, r_port = tcp_handler.parse_host_port(bounded_receiver.address)
      # Two 36-byte tensors (72 bytes total > 64 bytes budget).
      oversized_header = json.dumps({
          "req_id": "r_over_budget",
          "generation": 1,
          "tensors": [
              {"name": "w0", "shape": [9], "dtype": "f32", "num_bytes": 36},
              {"name": "w1", "shape": [9], "dtype": "f32", "num_bytes": 36},
          ],
      }).encode("utf-8")

      with mock.patch.object(
          tcp_handler.np, "empty", wraps=tcp_handler.np.empty
      ) as mock_empty:
        closed_without_ack = False
        with socket.create_connection((r_host, r_port), timeout=5.0) as sock:
          sock.sendall(struct.pack(">Q", len(oversized_header)))
          sock.sendall(oversized_header)
          try:
            closed_without_ack = not sock.recv(1)
          except (ConnectionResetError, BrokenPipeError):
            closed_without_ack = True
        self.assertTrue(closed_without_ack)
        mock_empty.assert_not_called()

      with self.assertRaises(RuntimeError):
        bounded_receiver.pop_round("r_over_budget", 1)

      # Subsequent transfer within 64-byte budget succeeds.
      t_ok = np.arange(16, dtype=np.float32)  # 64 bytes
      req = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_within_budget", "uuid": 2}
      )
      src_units = self.exporter.prepare_weight_sync(
          req,
          {
              "w": tcp_handler.RawTensorPayload(
                  buffer=t_ok, shape=(16,), dtype="f32"
              )
          },
      )
      self.handler.register_work_unit(src_units[0])
      dst_unit = weight_sync.WorkUnitMetadata(
          unit=weight_sync.WorkUnitId(
              job_name="rollout_bounded", data_replica_idx=0
          ),
          control_plane_rpc_address=bounded_receiver.address,
          variables=src_units[0].variables,
          transport_mode="tcp",
          use_ffi=False,
      )
      self.handler.register_work_unit(dst_unit)

      res = self.handler.transfer(
          src_units=[src_units[0].unit],
          dst_units=[dst_unit.unit],
          req_id="r_within_budget",
          generation=2,
      )
      self.assertTrue(res.success, msg=res.message)
      popped = bounded_receiver.pop_round("r_within_budget", 2)
      np.testing.assert_array_equal(popped["w"].buffer, t_ok)
    finally:
      bounded_receiver.close()

  def test_concurrent_register_work_unit_and_transfer(self):
    t = np.arange(16, dtype=np.float32)
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_concurrent", "uuid": 1}
    )
    src_units = self.exporter.prepare_weight_sync(
        req,
        {"w": tcp_handler.RawTensorPayload(buffer=t, shape=(16,), dtype="f32")},
    )
    self.handler.register_work_unit(src_units[0])
    dst_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="rollout", data_replica_idx=0),
        control_plane_rpc_address=self.receiver.address,
        variables=src_units[0].variables,
        transport_mode="tcp",
        use_ffi=False,
    )
    self.handler.register_work_unit(dst_unit)

    stop_event = threading.Event()
    errors: list[Exception] = []

    def _register_worker(worker_idx: int) -> None:
      try:
        idx = 0
        while not stop_event.is_set() and idx < 200:
          self.handler.register_work_unit(
              weight_sync.WorkUnitMetadata(
                  unit=weight_sync.WorkUnitId(
                      job_name=f"extra_{worker_idx}_{idx}",
                      data_replica_idx=0,
                  ),
                  control_plane_rpc_address="127.0.0.1:1",
                  variables=(),
                  transport_mode="tcp",
                  use_ffi=False,
              )
          )
          idx += 1
      except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      reg_futs = [pool.submit(_register_worker, i) for i in range(2)]
      transfer_futs = []
      for gen in range(1, 9):
        transfer_futs.append(
            pool.submit(
                self.handler.transfer,
                [src_units[0].unit],
                [dst_unit.unit],
                "r_concurrent",
                gen,
            )
        )
      results = [fut.result() for fut in transfer_futs]
      stop_event.set()
      for fut in reg_futs:
        fut.result()

    self.assertEmpty(errors)
    for gen, res in enumerate(results, start=1):
      self.assertTrue(res.success, msg=res.message)
      popped = self.receiver.pop_round("r_concurrent", gen)
      np.testing.assert_array_equal(popped["w"].buffer, t)


if __name__ == "__main__":
  absltest.main()
