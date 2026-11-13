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

"""Basic weight sync transport: whole canonical HF tensors staged in host RAM.

Streams over TCP; built for simplicity and generality across framework pairs,
not peak performance.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import errno
import hmac
import ipaddress
import json
import logging
import math
import os
import secrets
import socket
import struct
import threading
import time
from typing import Mapping, Sequence, TypedDict

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync

_ACK_BYTE = b"\x01"
_AUTH_URI_PREFIX = "basic-token:"
_STREAM_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_DISCARDED_TOMBSTONES = 256
_MAX_STAGED_ROUNDS = 16
_MAX_ACCEPT_WORKERS = 4
_PRE_AUTH_TIMEOUT_S = 15.0
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


def _is_loopback_ip(ip: str) -> bool:
  """Returns True when `ip` is a loopback IPv4 or IPv6 address."""
  try:
    return ipaddress.ip_address(ip).is_loopback
  except ValueError:
    return ip.startswith("127.") or ip == "::1"


def _default_advertised_host(
    family: int = socket.AF_UNSPEC,
    *,
    ipv6_only: bool = False,
) -> str:
  """Probes local routing table for an outbound IP matching listener family."""
  probes: list[tuple[socket.AddressFamily, tuple[str, int]]] = []
  if family in (socket.AF_UNSPEC, socket.AF_INET) or (
      family == socket.AF_INET6 and not ipv6_only
  ):
    probes.append((socket.AF_INET, ("8.8.8.8", 80)))
  if family in (socket.AF_UNSPEC, socket.AF_INET6):
    probes.append((socket.AF_INET6, ("2001:4860:4860::8888", 80)))
  for probe_family, target in probes:
    try:
      with socket.socket(probe_family, socket.SOCK_DGRAM) as s:
        s.connect(target)
        ip = s.getsockname()[0]
        if ip and not _is_loopback_ip(ip):
          return ip
    except (socket.error, OSError):
      continue
  if family == socket.AF_INET6 and ipv6_only:
    return "::1"
  return "127.0.0.1"


def _listener_ipv6_only(sock: socket.socket) -> bool:
  """Returns True if `sock` is an IPv6 socket with IPV6_V6ONLY enabled."""
  if sock.family != socket.AF_INET6:
    return False
  try:
    return bool(sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY))
  except OSError:
    return True


def _extract_auth_token(artifact_uri: str | None) -> str:
  """Extracts and validates a basic weight sync capability token from `artifact_uri`."""
  if not isinstance(artifact_uri, str) or not artifact_uri.startswith(
      _AUTH_URI_PREFIX
  ):
    raise ValueError(
        "WorkUnitMetadata.artifact_uri must start with"
        f" {_AUTH_URI_PREFIX!r}, got {artifact_uri!r}"
    )
  token = artifact_uri[len(_AUTH_URI_PREFIX) :]
  if not token:
    raise ValueError("WorkUnitMetadata.artifact_uri has empty auth token")
  return token


def _shutdown_socket(sock: socket.socket) -> None:
  """Shuts down a socket with `SHUT_RDWR`, ignoring OS errors."""
  with contextlib.suppress(OSError):
    sock.shutdown(socket.SHUT_RDWR)


def _close_socket(sock: socket.socket) -> None:
  """Closes a socket FD, ignoring OS errors."""
  with contextlib.suppress(OSError):
    sock.close()


def _shutdown_and_close_socket(sock: socket.socket) -> None:
  """Shuts down and closes a socket when no other thread can use its FD."""
  _shutdown_socket(sock)
  _close_socket(sock)


def _validate_positive_shape(name: str, raw_shape: object) -> tuple[int, ...]:
  """Validates that `raw_shape` is a non-empty sequence of positive ints."""
  if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
    raise ValueError(
        f"Tensor {name!r} must have a non-empty shape, got {raw_shape!r}"
    )
  dims: list[int] = []
  for dim in raw_shape:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
      raise ValueError(
          f"Tensor {name!r} has non-positive dimension in shape {raw_shape!r}"
      )
    dims.append(dim)
  return tuple(dims)


class _WireTensorMeta(TypedDict):
  name: str
  shape: list[int]
  dtype: str
  num_bytes: int


def _parse_exporter_response(
    header_bytes: bytes,
) -> tuple[bool, list[_WireTensorMeta]]:
  """Parses and validates a source exporter JSON response header."""
  resp_header = json.loads(header_bytes.decode("utf-8"))
  if (
      not isinstance(resp_header, dict)
      or "found" not in resp_header
      or not isinstance(resp_header["found"], bool)
  ):
    raise ValueError("Malformed response header from source exporter")
  if not resp_header["found"]:
    return False, []
  if "tensors" not in resp_header or not isinstance(
      resp_header["tensors"], list
  ):
    raise ValueError(
        "Malformed response header from source exporter: missing tensors list"
    )
  tensors_meta: list[_WireTensorMeta] = []
  for entry in resp_header["tensors"]:
    if (
        not isinstance(entry, dict)
        or "name" not in entry
        or not isinstance(entry["name"], str)
        or not entry["name"]
        or "shape" not in entry
        or "dtype" not in entry
        or not isinstance(entry["dtype"], str)
        or "num_bytes" not in entry
        or isinstance(entry["num_bytes"], bool)
        or not isinstance(entry["num_bytes"], int)
        or entry["num_bytes"] < 0
    ):
      raise ValueError(
          f"Malformed tensor metadata in response header: {entry!r}"
      )
    shape = _validate_positive_shape(entry["name"], entry["shape"])
    tensors_meta.append({
        "name": entry["name"],
        "shape": list(shape),
        "dtype": entry["dtype"],
        "num_bytes": entry["num_bytes"],
    })
  return True, tensors_meta


def _create_listener_socket(bind_port: int) -> socket.socket:
  """Creates a listening TCP socket supporting IPv6 with IPv4 fallback."""
  sock: socket.socket | None = None
  try:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with contextlib.suppress(OSError):
      sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", bind_port))
  except OSError as exc:
    if sock is not None:
      sock.close()
      sock = None
    if exc.errno not in (
        errno.EAFNOSUPPORT,
        errno.EADDRNOTAVAIL,
        errno.EPROTONOSUPPORT,
    ):
      raise
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      sock.bind(("0.0.0.0", bind_port))
    except BaseException:
      sock.close()
      raise
  try:
    sock.listen(128)
    return sock
  except BaseException:
    sock.close()
    raise


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
      raise ConnectionError(f"Unexpected EOF waiting for {total - pos} bytes")
    pos += n


class BasicSourceExporter:
  """Host-RAM staging and TCP-serving source server for basic weight sync.

  Stages whole canonical HF tensors in host RAM and serves them over TCP; built
  for simplicity and generality across framework pairs, not peak performance.
  """

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
    self._active_conns: set[socket.socket] = set()
    self._closed = False
    self._auth_token = secrets.token_hex(32)

    self._listener = _create_listener_socket(self._bind_port)
    self._listener_family = self._listener.family
    self._listener_ipv6_only = _listener_ipv6_only(self._listener)
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
  def auth_uri(self) -> str:
    """Returns the capability URI required to pull weights from this server."""
    return f"{_AUTH_URI_PREFIX}{self._auth_token}"

  @property
  def address(self) -> str:
    """Dynamically resolves the advertised host:port address."""
    host = (
        self._advertised_host
        or os.environ.get("DIST_RL_WEIGHT_SYNC_HOST")
        or _default_advertised_host(
            self._listener_family, ipv6_only=self._listener_ipv6_only
        )
    )
    return format_host_port(host, self._port)

  def _handle_connection(self, conn: socket.socket) -> None:
    """Serves a single incoming weight pull request over TCP."""
    try:
      conn.settimeout(_PRE_AUTH_TIMEOUT_S)
      raw_len = _recv_exact(conn, 8)
      (req_len,) = struct.unpack(">Q", raw_len)
      if req_len > _MAX_HEADER_BYTES:
        raise ValueError(
            f"TCP request header length {req_len} exceeds limit"
            f" {_MAX_HEADER_BYTES}"
        )
      req_json = _recv_exact(conn, req_len)
      query = json.loads(req_json.decode("utf-8"))
      if (
          not isinstance(query, dict)
          or "auth_token" not in query
          or not isinstance(query["auth_token"], str)
          or not hmac.compare_digest(query["auth_token"], self._auth_token)
      ):
        raise PermissionError("Invalid basic weight sync auth token")
      conn.settimeout(_SOCKET_IO_TIMEOUT_S)
      if "req_id" not in query or not isinstance(query["req_id"], str):
        raise ValueError("Pull query missing valid req_id")
      req_id = query["req_id"]

      with self._rounds_lock:
        if self._closed:
          return
        weights = self._rounds.get(req_id)

      if weights is None:
        header = json.dumps({"found": False, "req_id": req_id}).encode("utf-8")
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
        logging.warning("BasicSourceExporter handling connection: %s", exc)
    finally:
      with self._rounds_lock:
        self._active_conns.discard(conn)
      _shutdown_and_close_socket(conn)

  def _server_loop(self) -> None:
    """Runs background socket listener loop serving weight requests."""
    while not self._closed:
      try:
        conn, _ = self._listener.accept()
      except OSError:
        break
      with self._rounds_lock:
        closed = self._closed
        if not closed:
          self._active_conns.add(conn)
      if closed:
        _shutdown_and_close_socket(conn)
        break
      try:
        self._conn_executor.submit(self._handle_connection, conn)
      except RuntimeError:
        with self._rounds_lock:
          self._active_conns.discard(conn)
        _shutdown_and_close_socket(conn)
        break

  def prepare_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest,
      hf_weights: Mapping[str, RawTensorPayload],
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Stages exported weights in host RAM and returns work unit metadata."""
    with self._rounds_lock:
      if self._closed:
        raise RuntimeError("BasicSourceExporter is closed")
    req_id = _extract_round_id(sync_request)
    staged: dict[str, RawTensorPayload] = {}
    variables = []
    for name, payload in sorted(hf_weights.items()):
      contig = np.array(payload.buffer, copy=True, order="C")
      if contig.dtype.name == "bfloat16":
        contig = contig.view(np.uint16)
        dtype_tag = payload.dtype or "bf16"
        if dtype_tag not in ("bf16", "bfloat16"):
          raise ValueError(
              f"Tensor {name!r} bfloat16 buffer incompatible with"
              f" dtype={dtype_tag!r}"
          )
      elif contig.dtype == np.uint16 and payload.dtype in (
          "",
          "bf16",
          "bfloat16",
      ):
        dtype_tag = payload.dtype or "bf16"
      else:
        dtype_tag = payload.dtype or str(contig.dtype)
        if (
            dtype_tag not in _ALLOWED_DTYPE_TAGS
            or _ALLOWED_DTYPE_TAGS[dtype_tag] != contig.dtype
        ):
          raise ValueError(
              f"Tensor {name!r} dtype mismatch or unsupported dtype:"
              f" tag={dtype_tag!r}, buffer={contig.dtype}"
          )
      shape = _validate_positive_shape(name, payload.shape)
      if contig.shape != shape:
        raise ValueError(
            f"Tensor {name!r} shape mismatch: buffer={contig.shape},"
            f" payload.shape={shape}"
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
      if self._closed:
        raise RuntimeError("BasicSourceExporter is closed")
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
            transport_mode="basic",
            use_ffi=False,
            artifact_uri=self.auth_uri,
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
    """Shuts down listener, active connections, and releases staged rounds."""
    with self._rounds_lock:
      if self._closed:
        return
      self._closed = True
      conns = list(self._active_conns)
      self._active_conns.clear()
      self._rounds.clear()
    _shutdown_socket(self._listener)
    for conn in conns:
      _shutdown_socket(conn)
    try:
      self._conn_executor.shutdown(wait=True, cancel_futures=True)
      if threading.current_thread() is not self._thread:
        self._thread.join(timeout=5.0)
    finally:
      _close_socket(self._listener)
      for conn in conns:
        _close_socket(conn)


class BasicReceiverServer:
  """Host-RAM receiving and staging server for basic weight sync over TCP."""

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
    self._staged_rounds: dict[tuple[str, int], dict[str, RawTensorPayload]] = {}
    self._discarded_rounds: collections.OrderedDict[tuple[str, int], None] = (
        collections.OrderedDict()
    )
    self._lock = threading.Lock()
    self._active_conns: set[socket.socket] = set()
    self._closed = False
    self._auth_token = secrets.token_hex(32)

    self._listener = _create_listener_socket(self._bind_port)
    self._listener_family = self._listener.family
    self._listener_ipv6_only = _listener_ipv6_only(self._listener)
    self._port = self._listener.getsockname()[1]
    self._conn_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_ACCEPT_WORKERS,
        thread_name_prefix="basic_receiver_conn",
    )

    self._thread = threading.Thread(
        target=self._server_loop, daemon=True, name="basic_weight_receiver"
    )
    self._thread.start()

  @property
  def auth_uri(self) -> str:
    """Returns the capability URI required to push weights to this server."""
    return f"{_AUTH_URI_PREFIX}{self._auth_token}"

  @property
  def address(self) -> str:
    """Dynamically resolves the advertised host:port address."""
    host = (
        self._advertised_host
        or os.environ.get("DIST_RL_WEIGHT_SYNC_HOST")
        or _default_advertised_host(
            self._listener_family, ipv6_only=self._listener_ipv6_only
        )
    )
    return format_host_port(host, self._port)

  def _handle_connection(self, conn: socket.socket) -> None:
    """Handles a single incoming weight push connection over TCP."""
    try:
      conn.settimeout(_PRE_AUTH_TIMEOUT_S)
      raw_len = _recv_exact(conn, 8)
      (header_len,) = struct.unpack(">Q", raw_len)
      if header_len > _MAX_HEADER_BYTES:
        raise ValueError(
            f"TCP push header length {header_len} exceeds limit"
            f" {_MAX_HEADER_BYTES}"
        )
      header_bytes = _recv_exact(conn, header_len)
      header = json.loads(header_bytes.decode("utf-8"))
      if (
          not isinstance(header, dict)
          or "auth_token" not in header
          or not isinstance(header["auth_token"], str)
          or not hmac.compare_digest(header["auth_token"], self._auth_token)
      ):
        raise PermissionError("Invalid basic weight sync auth token")
      conn.settimeout(_SOCKET_IO_TIMEOUT_S)

      if (
          "req_id" not in header
          or not isinstance(header["req_id"], str)
          or not header["req_id"]
          or "generation" not in header
          or isinstance(header["generation"], bool)
          or not isinstance(header["generation"], int)
          or "tensors" not in header
          or not isinstance(header["tensors"], list)
      ):
        raise ValueError("Malformed push header from transfer handler")
      req_id = header["req_id"]
      generation = header["generation"]
      tensors_meta = header["tensors"]

      with self._lock:
        if self._closed:
          return
        if (req_id, generation) not in self._staged_rounds and len(
            self._staged_rounds
        ) >= _MAX_STAGED_ROUNDS:
          raise BufferError(
              "BasicReceiverServer staged rounds limit"
              f" ({_MAX_STAGED_ROUNDS}) exceeded"
          )

      validated_descriptors: list[
          tuple[str, tuple[int, ...], str, np.dtype[np.generic]]
      ] = []
      total_round_bytes = 0
      for t_meta in tensors_meta:
        if (
            not isinstance(t_meta, dict)
            or "name" not in t_meta
            or not isinstance(t_meta["name"], str)
            or not t_meta["name"]
            or "shape" not in t_meta
            or "dtype" not in t_meta
            or not isinstance(t_meta["dtype"], str)
            or "num_bytes" not in t_meta
            or isinstance(t_meta["num_bytes"], bool)
            or not isinstance(t_meta["num_bytes"], int)
        ):
          raise ValueError(
              f"Malformed tensor metadata in push header: {t_meta!r}"
          )
        name = t_meta["name"]
        shape = _validate_positive_shape(name, t_meta["shape"])
        dtype_tag = t_meta["dtype"]
        if dtype_tag not in _ALLOWED_DTYPE_TAGS:
          raise ValueError(f"Unsupported tensor dtype tag: {dtype_tag!r}")
        np_dtype = _ALLOWED_DTYPE_TAGS[dtype_tag]
        expected_bytes = t_meta["num_bytes"]
        computed_bytes = math.prod(shape) * np_dtype.itemsize
        if expected_bytes != computed_bytes:
          raise ValueError(
              f"Tensor {name!r} num_bytes mismatch:"
              f" header={expected_bytes}, shape*dtype={computed_bytes}"
          )
        total_round_bytes += computed_bytes
        if total_round_bytes > self._max_round_bytes:
          raise ValueError(
              f"BasicReceiverServer round total bytes {total_round_bytes}"
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

      should_ack = False
      with self._lock:
        if self._closed:
          return
        if (req_id, generation) not in self._discarded_rounds:
          if (req_id, generation) not in self._staged_rounds and len(
              self._staged_rounds
          ) >= _MAX_STAGED_ROUNDS:
            raise BufferError(
                "BasicReceiverServer staged rounds limit"
                f" ({_MAX_STAGED_ROUNDS}) exceeded"
            )
          self._staged_rounds[(req_id, generation)] = round_weights
        should_ack = True

      # Send ACK only after all tensor bytes received and staged
      if should_ack:
        conn.sendall(_ACK_BYTE)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not self._closed:
        logging.warning("BasicReceiverServer handling push: %s", exc)
    finally:
      with self._lock:
        self._active_conns.discard(conn)
      _shutdown_and_close_socket(conn)

  def _server_loop(self) -> None:
    """Runs background socket listener loop receiving pushed weights."""
    while not self._closed:
      try:
        conn, _ = self._listener.accept()
      except OSError:
        break
      with self._lock:
        closed = self._closed
        if not closed:
          self._active_conns.add(conn)
      if closed:
        _shutdown_and_close_socket(conn)
        break
      try:
        self._conn_executor.submit(self._handle_connection, conn)
      except RuntimeError:
        with self._lock:
          self._active_conns.discard(conn)
        _shutdown_and_close_socket(conn)
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
    with self._lock:
      if self._closed:
        return
      self._closed = True
      conns = list(self._active_conns)
      self._active_conns.clear()
      self._staged_rounds.clear()
      self._discarded_rounds.clear()
    _shutdown_socket(self._listener)
    for conn in conns:
      _shutdown_socket(conn)
    try:
      self._conn_executor.shutdown(wait=True, cancel_futures=True)
      if threading.current_thread() is not self._thread:
        self._thread.join(timeout=5.0)
    finally:
      _close_socket(self._listener)
      for conn in conns:
        _close_socket(conn)


class BasicWeightSyncHandler(weight_sync.WeightSyncHandler):
  """Coordinator handler staging whole canonical HF tensors in host RAM over TCP.

  Built for simplicity and generality across framework pairs, not peak
  performance.
  """

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
          message="BasicWeightSyncHandler requires exactly 1 src_unit",
      )
    if not dst_units:
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message="BasicWeightSyncHandler requires at least 1 dst_unit",
      )

    if generation is None or int(generation) <= 0:
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message=(
              "BasicWeightSyncHandler requires positive generation, got"
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

      try:
        src_token = _extract_auth_token(src_meta.artifact_uri)
        dst_targets: list[tuple[str, str]] = []
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
          dst_targets.append((
              dst_meta.control_plane_rpc_address,
              _extract_auth_token(dst_meta.artifact_uri),
          ))
      except ValueError as exc:
        return weight_sync.TransferResult(
            req_id=effective_req_id,
            success=False,
            message=str(exc),
        )

    t_start = time.perf_counter()
    try:
      with contextlib.ExitStack() as close_stack:
        with contextlib.ExitStack() as io_stack:
          executor = io_stack.enter_context(
              concurrent.futures.ThreadPoolExecutor(
                  max_workers=min(32, max(1, len(dst_targets)))
              )
          )
          src_host, src_port = parse_host_port(
              src_meta.control_plane_rpc_address
          )
          src_sock = socket.create_connection(
              (src_host, src_port), timeout=30.0
          )
          close_stack.callback(_close_socket, src_sock)
          io_stack.callback(_shutdown_socket, src_sock)
          src_sock.settimeout(_SOCKET_IO_TIMEOUT_S)
          query_bytes = json.dumps(
              {"req_id": effective_req_id, "auth_token": src_token}
          ).encode("utf-8")
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
          found, tensors_meta = _parse_exporter_response(header_bytes)

          if not found:
            return weight_sync.TransferResult(
                req_id=effective_req_id,
                success=False,
                message=(
                    "Source exporter has no staged weights for req_id="
                    f"{effective_req_id!r}"
                ),
            )

          total_bytes = sum(t["num_bytes"] for t in tensors_meta)

          # 2. Build per-destination push headers and connect
          dst_socks: list[tuple[str, socket.socket]] = []
          for addr, dst_token in dst_targets:
            d_host, d_port = parse_host_port(addr)
            d_sock = socket.create_connection((d_host, d_port), timeout=30.0)
            close_stack.callback(_close_socket, d_sock)
            io_stack.callback(_shutdown_socket, d_sock)
            d_sock.settimeout(_SOCKET_IO_TIMEOUT_S)
            push_header = {
                "req_id": effective_req_id,
                "generation": int(generation),
                "auth_token": dst_token,
                "tensors": tensors_meta,
            }
            push_header_bytes = json.dumps(push_header).encode("utf-8")
            d_sock.sendall(struct.pack(">Q", len(push_header_bytes)))
            d_sock.sendall(push_header_bytes)
            dst_socks.append((addr, d_sock))

          # 3. Double-buffered pipelined chunk streaming
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
              "[BASIC] transfer req_id=%s gen=%s: %d tensors (%.2f MB) in"
              " %.3fs (%.2f GB/s)",
              effective_req_id,
              generation,
              len(tensors_meta),
              total_mb,
              total_elapsed,
              gb_per_sec,
          )
    except (OSError, ValueError, BufferError, struct.error) as exc:
      logging.warning(
          "[BASIC] transfer req_id=%s gen=%s failed: %s",
          effective_req_id,
          generation,
          exc,
      )
      return weight_sync.TransferResult(
          req_id=effective_req_id,
          success=False,
          message=f"Basic weight transfer failed: {exc}",
      )

    return weight_sync.TransferResult(req_id=effective_req_id, success=True)
