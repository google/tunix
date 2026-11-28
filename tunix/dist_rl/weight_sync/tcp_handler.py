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

"""TCP weight sync transport handler and source/receiver servers."""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import json
import logging
import math
import os
import socket
import struct
import threading
import time
from typing import Mapping, Sequence
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync

_ACK_BYTE = b"\x01"
_STREAM_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_DISCARDED_TOMBSTONES = 256
_MAX_STAGED_ROUNDS = 16
_MAX_ACCEPT_WORKERS = 4
_SOCKET_IO_TIMEOUT_S = 300.0
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_ROUND_BYTES = 256 * 1024 * 1024 * 1024
_ALLOWED_DTYPE_TAGS: dict[str, np.dtype[np.generic]] = {
    "bf16": np.dtype(np.uint16),
    "bfloat16": np.dtype(np.uint16),
    "f16": np.dtype(np.float16),
    "float16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
    "i8": np.dtype(np.int8),
    "int8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "int16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "int32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "int64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
    "uint8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "uint16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "uint32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
    "uint64": np.dtype(np.uint64),
    "bool": np.dtype(np.bool_),
    "bool_": np.dtype(np.bool_),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class RawTensorPayload:
  """Framework-neutral raw memoryview/ndarray payload for a single tensor."""

  buffer: np.ndarray
  shape: tuple[int, ...]
  dtype: str


def _extract_round_id(sync_request: datatypes.WeightSyncRequest) -> str:
  """Extracts the round / request identifier from a sync request."""
  if not isinstance(sync_request, datatypes.WeightSyncRequest):
    raise TypeError(f"Expected WeightSyncRequest, got {type(sync_request)}")
  if (
      "req_id" in sync_request.extra_config
      and sync_request.extra_config["req_id"]
  ):
    return str(sync_request.extra_config["req_id"])
  if sync_request.request_id:
    return str(sync_request.request_id)
  raise ValueError("WeightSyncRequest missing both req_id and request_id")


def format_host_port(host: str, port: int) -> str:
  """Formats a host string and port, bracketing IPv6 addresses."""
  if ":" in host and not (host.startswith("[") and host.endswith("]")):
    return f"[{host}]:{port}"
  return f"{host}:{port}"


def parse_host_port(addr: str) -> tuple[str, int]:
  """Parses a host:port string, handling bracketed IPv6 addresses."""
  if addr.startswith("["):
    idx = addr.find("]:")
    if idx == -1:
      raise ValueError(f"Invalid bracketed IPv6 address format: {addr!r}")
    host = addr[1:idx]
    port = int(addr[idx + 2 :])
  else:
    host, port_str = addr.rsplit(":", 1)
    port = int(port_str)
  return host, port


def _default_advertised_host() -> str:
  """Probes local routing table to determine outbound IP without blocking."""
  for family, target in (
      (socket.AF_INET, ("8.8.8.8", 80)),
      (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
  ):
    try:
      with socket.socket(family, socket.SOCK_DGRAM) as s:
        s.connect(target)
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
          return ip
    except (socket.error, OSError):
      continue
  return "127.0.0.1"


def _create_listener_socket(bind_port: int) -> socket.socket:
  """Creates a listening TCP socket supporting IPv6 with IPv4 dual-stack fallback."""
  sock = None
  try:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", bind_port))
    sock.listen(128)
    return sock
  except (socket.error, OSError):
    if sock is not None:
      try:
        sock.close()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", bind_port))
    sock.listen(128)
    return sock


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
  """Reads exactly num_bytes from sock or raises ConnectionError on EOF."""
  buf = bytearray(num_bytes)
  view = memoryview(buf)
  pos = 0
  while pos < num_bytes:
    n = sock.recv_into(view[pos:])
    if n == 0:
      raise ConnectionError(
          f"Unexpected EOF waiting for {num_bytes - pos} bytes"
      )
    pos += n
  return bytes(buf)


def _recv_exact_into(
    sock: socket.socket, view: memoryview  # pylint: disable=g-bare-generic
) -> None:
  """Reads exactly len(view) bytes from sock into view or raises ConnectionError."""
  total = len(view)
  pos = 0
  while pos < total:
    n = sock.recv_into(view[pos:])
    if n == 0:
      raise ConnectionError(
          f"Unexpected EOF waiting for {total - pos} bytes"
      )
    pos += n


class TcpSourceExporter:
  """TCP server embedded in trainer worker staging and serving weights to handlers."""

  def __init__(
      self,
      job_name: str = "trainer_0",
      bind_port: int | None = None,
      advertised_host: str | None = None,
  ):
    self._job_name = job_name
    self._bind_port = (
        bind_port
        if bind_port is not None
        else int(os.environ.get("DIST_RL_WEIGHT_SYNC_PORT", "0"))
    )
    self._advertised_host = advertised_host
    self._rounds: dict[str, dict[str, RawTensorPayload]] = {}
    self._rounds_lock = threading.Lock()
    self._closed = False

    self._listener = _create_listener_socket(self._bind_port)
    self._port = self._listener.getsockname()[1]
    self._conn_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_ACCEPT_WORKERS,
        thread_name_prefix=f"{job_name}_exporter_conn",
    )

    self._thread = threading.Thread(
        target=self._server_loop, daemon=True, name=f"{job_name}_exporter"
    )
    self._thread.start()

  @property
  def address(self) -> str:
    """Dynamically resolves the advertised host:port address."""
    host = (
        self._advertised_host
        or os.environ.get("DIST_RL_WEIGHT_SYNC_HOST")
        or _default_advertised_host()
    )
    return format_host_port(host, self._port)

  def _handle_connection(self, conn: socket.socket) -> None:
    """Serves a single incoming weight pull request over TCP."""
    try:
      with conn:
        conn.settimeout(_SOCKET_IO_TIMEOUT_S)
        raw_len = _recv_exact(conn, 8)
        (req_len,) = struct.unpack(">Q", raw_len)
        if req_len > _MAX_HEADER_BYTES:
          raise ValueError(
              f"TCP request header length {req_len} exceeds limit"
              f" {_MAX_HEADER_BYTES}"
          )
        req_json = _recv_exact(conn, req_len)
        query = json.loads(req_json.decode("utf-8"))
        req_id = str(query["req_id"])

        with self._rounds_lock:
          weights = self._rounds.get(req_id)

        if weights is None:
          header = json.dumps({"found": False, "req_id": req_id}).encode(
              "utf-8"
          )
          conn.sendall(struct.pack(">Q", len(header)))
          conn.sendall(header)
          return

        tensors_meta = []
        for name, payload in sorted(weights.items()):
          tensors_meta.append({
              "name": name,
              "shape": list(payload.shape),
              "dtype": payload.dtype,
              "num_bytes": int(payload.buffer.nbytes),
          })

        header_dict = {
            "found": True,
            "req_id": req_id,
            "tensors": tensors_meta,
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        conn.sendall(struct.pack(">Q", len(header_bytes)))
        conn.sendall(header_bytes)

        for _, payload in sorted(weights.items()):
          conn.sendall(memoryview(payload.buffer.view(np.uint8)))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not self._closed:
        logging.warning("TcpSourceExporter handling connection: %s", exc)

  def _server_loop(self) -> None:
    """Runs background socket listener loop serving weight requests."""
    while not self._closed:
      try:
        conn, _ = self._listener.accept()
      except (socket.error, OSError):
        break
      try:
        self._conn_executor.submit(self._handle_connection, conn)
      except RuntimeError:
        conn.close()
        break

  def prepare_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest,
      hf_weights: Mapping[str, RawTensorPayload],
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Stages exported weights in host RAM and returns work unit metadata."""
    req_id = _extract_round_id(sync_request)
    staged: dict[str, RawTensorPayload] = {}
    variables = []
    for name, payload in sorted(hf_weights.items()):
      contig = np.ascontiguousarray(payload.buffer)
      if contig.dtype.name == "bfloat16":
        contig = contig.view(np.uint16)
        dtype_tag = payload.dtype or "bf16"
      elif contig.dtype == np.uint16 and not payload.dtype:
        dtype_tag = "bf16"
      else:
        dtype_tag = payload.dtype or str(contig.dtype)
      shape = tuple(int(d) for d in payload.shape)
      if any(d < 0 for d in shape):
        raise ValueError(
            f"Tensor {name!r} has negative dimension in shape {shape}"
        )
      expected_bytes = math.prod(shape) * contig.dtype.itemsize
      if contig.nbytes != expected_bytes:
        raise ValueError(
            f"Tensor {name!r} nbytes mismatch: buffer={contig.nbytes},"
            f" shape*dtype={expected_bytes}"
        )
      staged[name] = RawTensorPayload(
          buffer=contig,
          shape=shape,
          dtype=dtype_tag,
      )
      variables.append(
          weight_sync.TensorMetadata(
              name=name,
              shape=shape,
              mesh_shape=(1,) * len(shape),
              layout=tuple(range(len(shape) - 1, -1, -1)),
              item_size=int(contig.dtype.itemsize),
          )
      )

    with self._rounds_lock:
      self._rounds[req_id] = staged

    unit = weight_sync.WorkUnitId(
        job_name=self._job_name,
        data_replica_idx=0,
    )
    return [
        weight_sync.WorkUnitMetadata(
            unit=unit,
            control_plane_rpc_address=self.address,
            variables=tuple(variables),
            transport_mode="tcp",
            use_ffi=False,
        )
    ]

  def release_weight_sync(
      self, sync_request: datatypes.WeightSyncRequest
  ) -> None:
    """Releases staged round weights from host RAM."""
    req_id = _extract_round_id(sync_request)
    with self._rounds_lock:
      self._rounds.pop(req_id, None)

  def close(self) -> None:
    """Shuts down listener and releases all staged rounds."""
    self._closed = True
    try:
      self._listener.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    self._conn_executor.shutdown(wait=False, cancel_futures=True)
    with self._rounds_lock:
      self._rounds.clear()


class TcpReceiverServer:
  """TCP server running on rollout worker receiving and staging weights in host RAM."""

  def __init__(
      self,
      bind_port: int | None = None,
      advertised_host: str | None = None,
      max_round_bytes: int = _DEFAULT_MAX_ROUND_BYTES,
  ):
    if max_round_bytes <= 0:
      raise ValueError("max_round_bytes must be positive")
    self._max_round_bytes = int(max_round_bytes)
    self._bind_port = (
        bind_port
        if bind_port is not None
        else int(os.environ.get("DIST_RL_WEIGHT_SYNC_PORT", "0"))
    )
    self._advertised_host = advertised_host
    self._staged_rounds: dict[
        tuple[str, int], dict[str, RawTensorPayload]
    ] = {}
    self._discarded_rounds: collections.OrderedDict[tuple[str, int], None] = (
        collections.OrderedDict()
    )
    self._lock = threading.Lock()
    self._closed = False

    self._listener = _create_listener_socket(self._bind_port)
    self._port = self._listener.getsockname()[1]
    self._conn_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_ACCEPT_WORKERS,
        thread_name_prefix="tcp_receiver_conn",
    )

    self._thread = threading.Thread(
        target=self._server_loop, daemon=True, name="tcp_weight_receiver"
    )
    self._thread.start()

  @property
  def address(self) -> str:
    """Dynamically resolves the advertised host:port address."""
    host = (
        self._advertised_host
        or os.environ.get("DIST_RL_WEIGHT_SYNC_HOST")
        or _default_advertised_host()
    )
    return format_host_port(host, self._port)

  def _handle_connection(self, conn: socket.socket) -> None:
    """Handles a single incoming weight push connection over TCP."""
    try:
      with conn:
        conn.settimeout(_SOCKET_IO_TIMEOUT_S)
        raw_len = _recv_exact(conn, 8)
        (header_len,) = struct.unpack(">Q", raw_len)
        if header_len > _MAX_HEADER_BYTES:
          raise ValueError(
              f"TCP push header length {header_len} exceeds limit"
              f" {_MAX_HEADER_BYTES}"
          )
        header_bytes = _recv_exact(conn, header_len)
        header = json.loads(header_bytes.decode("utf-8"))

        req_id = str(header["req_id"])
        generation = int(header["generation"])
        tensors_meta = header["tensors"]

        with self._lock:
          if (
              (req_id, generation) not in self._staged_rounds
              and len(self._staged_rounds) >= _MAX_STAGED_ROUNDS
          ):
            raise BufferError(
                "TcpReceiverServer staged rounds limit"
                f" ({_MAX_STAGED_ROUNDS}) exceeded"
            )

        validated_descriptors: list[
            tuple[str, tuple[int, ...], str, np.dtype[np.generic]]
        ] = []
        total_round_bytes = 0
        for t_meta in tensors_meta:
          name = str(t_meta["name"])
          shape = tuple(int(d) for d in t_meta["shape"])
          if any(d < 0 for d in shape):
            raise ValueError(
                f"Tensor {name!r} has negative dimension in shape {shape}"
            )
          dtype_tag = str(t_meta["dtype"])
          if dtype_tag not in _ALLOWED_DTYPE_TAGS:
            raise ValueError(f"Unsupported tensor dtype tag: {dtype_tag!r}")
          np_dtype = _ALLOWED_DTYPE_TAGS[dtype_tag]
          expected_bytes = int(t_meta["num_bytes"])
          computed_bytes = math.prod(shape) * np_dtype.itemsize
          if expected_bytes != computed_bytes:
            raise ValueError(
                f"Tensor {name!r} num_bytes mismatch:"
                f" header={expected_bytes}, shape*dtype={computed_bytes}"
            )
          total_round_bytes += computed_bytes
          if total_round_bytes > self._max_round_bytes:
            raise ValueError(
                f"TcpReceiverServer round total bytes {total_round_bytes}"
                f" exceeds max_round_bytes={self._max_round_bytes}"
            )
          validated_descriptors.append((name, shape, dtype_tag, np_dtype))

        round_weights: dict[str, RawTensorPayload] = {}
        for name, shape, dtype_tag, np_dtype in validated_descriptors:
          arr = np.empty(shape, dtype=np_dtype)
          view = memoryview(arr).cast("B")
          _recv_exact_into(conn, view)
          round_weights[name] = RawTensorPayload(
              buffer=arr,
              shape=shape,
              dtype=dtype_tag,
          )

        with self._lock:
          if (req_id, generation) not in self._discarded_rounds:
            if (
                (req_id, generation) not in self._staged_rounds
                and len(self._staged_rounds) >= _MAX_STAGED_ROUNDS
            ):
              raise BufferError(
                  "TcpReceiverServer staged rounds limit"
                  f" ({_MAX_STAGED_ROUNDS}) exceeded"
              )
            self._staged_rounds[(req_id, generation)] = round_weights

        # Send ACK only after all tensor bytes received and staged
        conn.sendall(_ACK_BYTE)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not self._closed:
        logging.warning("TcpReceiverServer handling push: %s", exc)

  def _server_loop(self) -> None:
    """Runs background socket listener loop receiving pushed weights."""
    while not self._closed:
      try:
        conn, _ = self._listener.accept()
      except (socket.error, OSError):
        break
      try:
        self._conn_executor.submit(self._handle_connection, conn)
      except RuntimeError:
        conn.close()
        break

  def pop_round(
      self, req_id: str, generation: int
  ) -> dict[str, RawTensorPayload]:
    """Pops and returns staged round weights from host RAM."""
    with self._lock:
      round_weights = self._staged_rounds.pop((req_id, int(generation)), None)
    if round_weights is None:
      raise RuntimeError(
          f"No staged weights found for req_id={req_id!r},"
          f" generation={generation}"
      )
    return round_weights

  def discard_round(self, req_id: str, generation: int) -> None:
    """Discards staged round weights and registers tombstone to prevent late receive."""
    with self._lock:
      self._staged_rounds.pop((req_id, int(generation)), None)
      key = (req_id, int(generation))
      self._discarded_rounds[key] = None
      self._discarded_rounds.move_to_end(key)
      if len(self._discarded_rounds) > _MAX_DISCARDED_TOMBSTONES:
        self._discarded_rounds.popitem(last=False)

  def close(self) -> None:
    """Shuts down receiver server and clears all staged memory."""
    self._closed = True
    try:
      self._listener.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    self._conn_executor.shutdown(wait=False, cancel_futures=True)
    with self._lock:
      self._staged_rounds.clear()
      self._discarded_rounds.clear()


class TcpWeightSyncHandler(weight_sync.WeightSyncHandler):
  """Coordinator weight sync handler using direct host-staged TCP transfers."""

  def __init__(self, chunk_bytes: int = _STREAM_CHUNK_BYTES):
    self._chunk_bytes = max(1, int(chunk_bytes))
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}
    self._units_lock = threading.Lock()

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    """Registers or updates a source or destination work unit metadata."""
    with self._units_lock:
      self._units[metadata.unit] = metadata

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: str | None = None,
      generation: int | None = None,
  ) -> weight_sync.TransferResult:
    """Pulls weights from source exporter and pushes to destination receiver servers."""
    effective_req_id = req_id or ""
    if len(src_units) != 1:
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message="TcpWeightSyncHandler requires exactly 1 src_unit",
      )
    if not dst_units:
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message="TcpWeightSyncHandler requires at least 1 dst_unit",
      )

    if generation is None or int(generation) <= 0:
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message=(
              "TcpWeightSyncHandler requires positive generation, got"
              f" {generation}"
          ),
      )

    src_id = src_units[0]
    with self._units_lock:
      if src_id not in self._units:
        return weight_sync.TransferResult(
            req_id=effective_req_id,
            success=False,
            message=(
                f"Source unit {src_id} is not registered with valid address"
            ),
        )
      src_meta = self._units[src_id]
      if not src_meta.control_plane_rpc_address:
        return weight_sync.TransferResult(
            req_id=effective_req_id,
            success=False,
            message=(
                f"Source unit {src_id} is not registered with valid address"
            ),
        )

      dst_addresses = []
      for dst_id in dst_units:
        if dst_id not in self._units:
          return weight_sync.TransferResult(
              req_id=effective_req_id,
              success=False,
              message=(
                  f"Destination unit {dst_id} is not registered with valid"
                  " address"
              ),
          )
        dst_meta = self._units[dst_id]
        if not dst_meta.control_plane_rpc_address:
          return weight_sync.TransferResult(
              req_id=effective_req_id,
              success=False,
              message=(
                  f"Destination unit {dst_id} is not registered with valid"
                  " address"
              ),
          )
        dst_addresses.append(dst_meta.control_plane_rpc_address)

    t_start = time.perf_counter()
    try:
      with contextlib.ExitStack() as stack:
        src_host, src_port = parse_host_port(src_meta.control_plane_rpc_address)
        src_sock = stack.enter_context(
            socket.create_connection((src_host, src_port), timeout=30.0)
        )
        src_sock.settimeout(_SOCKET_IO_TIMEOUT_S)
        query_bytes = json.dumps({"req_id": effective_req_id}).encode("utf-8")
        src_sock.sendall(struct.pack(">Q", len(query_bytes)))
        src_sock.sendall(query_bytes)

        raw_len = _recv_exact(src_sock, 8)
        (header_len,) = struct.unpack(">Q", raw_len)
        if header_len > _MAX_HEADER_BYTES:
          raise ValueError(
              f"TCP response header length {header_len} exceeds limit"
              f" {_MAX_HEADER_BYTES}"
          )
        header_bytes = _recv_exact(src_sock, header_len)
        resp_header = json.loads(header_bytes.decode("utf-8"))

        if not resp_header["found"]:
          return weight_sync.TransferResult(
              req_id=effective_req_id,
              success=False,
              message=(
                  "Source exporter has no staged weights for req_id="
                  f"{effective_req_id!r}"
              ),
          )

        tensors_meta = resp_header["tensors"]
        for t in tensors_meta:
          if int(t["num_bytes"]) < 0:
            raise ValueError(
                f"Negative num_bytes ({t['num_bytes']}) for tensor"
                f" {t['name']!r}"
            )
        total_bytes = sum(int(t["num_bytes"]) for t in tensors_meta)

        # 2. Build push_header once and open all destination sockets
        push_header = {
            "req_id": effective_req_id,
            "generation": int(generation),
            "tensors": tensors_meta,
        }
        push_header_bytes = json.dumps(push_header).encode("utf-8")
        push_header_prefix = struct.pack(">Q", len(push_header_bytes))

        dst_socks: list[tuple[str, socket.socket]] = []
        for addr in dst_addresses:
          d_host, d_port = parse_host_port(addr)
          d_sock = stack.enter_context(
              socket.create_connection((d_host, d_port), timeout=30.0)
          )
          d_sock.settimeout(_SOCKET_IO_TIMEOUT_S)
          d_sock.sendall(push_header_prefix)
          d_sock.sendall(push_header_bytes)
          dst_socks.append((addr, d_sock))

        # 3. Double-buffered pipelined chunk streaming
        executor = stack.enter_context(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, max(1, len(dst_socks)))
            )
        )
        buf_size = min(max(1, total_bytes), self._chunk_bytes)
        pull_buf = bytearray(buf_size)
        push_buf = bytearray(buf_size)
        remaining = total_bytes
        pending_futures: list[concurrent.futures.Future[None]] = []

        while remaining > 0:
          n = min(remaining, self._chunk_bytes)
          _recv_exact_into(src_sock, memoryview(pull_buf)[:n])
          for fut in pending_futures:
            fut.result()
          pull_buf, push_buf = push_buf, pull_buf
          chunk_view = memoryview(push_buf)[:n]
          if dst_socks:
            pending_futures = [
                executor.submit(d_sock.sendall, chunk_view)
                for _, d_sock in dst_socks
            ]
          remaining -= n

        for fut in pending_futures:
          fut.result()

        # 4. Verify ACKs from all destinations
        for addr, d_sock in dst_socks:
          ack = _recv_exact(d_sock, 1)
          if ack != _ACK_BYTE:
            raise ConnectionError(
                f"Invalid ACK from destination {addr}: {ack!r}"
            )

        total_elapsed = time.perf_counter() - t_start
        total_mb = total_bytes / (1024 * 1024)
        gb_per_sec = (total_bytes / (1024 * 1024 * 1024)) / max(
            total_elapsed, 1e-6
        )
        logging.info(
            "[TCP] transfer req_id=%s gen=%s: %d tensors (%.2f MB) in %.3fs"
            " (%.2f GB/s)",
            effective_req_id,
            generation,
            len(tensors_meta),
            total_mb,
            total_elapsed,
            gb_per_sec,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning(
          "[TCP] transfer req_id=%s gen=%s failed: %s",
          effective_req_id,
          generation,
          exc,
      )
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message=f"TCP weight transfer failed: {exc}",
      )

    return weight_sync.TransferResult(req_id=effective_req_id, success=True)
