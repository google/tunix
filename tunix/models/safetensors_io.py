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

"""Shared Safetensors wire codec, `/dev/shm` arena, and storage transport."""

from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
import concurrent.futures
import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import http.client
import io
import itertools
import json
import logging
import math
import mmap
import multiprocessing
import os
import re
import selectors
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, ContextManager, Protocol, TypeGuard, cast
import urllib.parse

from etils import epath
import ml_dtypes
import numpy as np

_g3_mp_lib: Any = None

READ_WORKERS = 16
# 288 = 96 GCS worker procs x 3 streams/proc (saturates multi-VIP GCS read
# bandwidth while supporting 16 distributed-FS procs x 16 streams = 256).
MAX_STREAMS = 288
READ_CHUNK_BYTES = 16 * 1024 * 1024
DIRECT_SLICE_BYTES = 4 * 1024 * 1024
MAX_COALESCE_GAP = 2 * 1024 * 1024
MAX_COALESCE_SPAN = 64 * 1024 * 1024
SHM_BUDGET_BYTES = 0
MAX_READ_ARENA_BYTES = 192 * 1024 * 1024 * 1024
MAX_HEADER_BYTES = 100_000_000
GCS_PRIMARY_TIMEOUT_S = 4.0
GCS_HEDGE_TIMEOUT_S = 2.5
GCS_URGENT_HEDGE_DELAY_S = 0.60
GCS_STREAMS_PER_PROC = 3
GCS_MAX_READ_PROCS = 96
GCS_VIP_CACHE_TTL_S = 300.0
SIDECAR_HEADERS_FILENAME = '.tunix_headers.json'
_PREFAULT_WORKERS = 32
_PREFAULT_PAGE_BYTES = 4096
_PREFAULT_BLOCK_BYTES = 64 * 1024 * 1024

GCS_HOSTS: tuple[str, ...] = (
    'storage.googleapis.com',
    'storage-download.googleapis.com',
    'storage-upload.googleapis.com',
    'private.googleapis.com',
)
SAFETENSORS_DTYPES: Mapping[str, np.dtype] = {
    'BOOL': np.dtype(np.bool_),
    'U8': np.dtype(np.uint8),
    'I8': np.dtype(np.int8),
    'U16': np.dtype(np.uint16),
    'I16': np.dtype(np.int16),
    'U32': np.dtype(np.uint32),
    'I32': np.dtype(np.int32),
    'U64': np.dtype(np.uint64),
    'I64': np.dtype(np.int64),
    'F8_E4M3': np.dtype(ml_dtypes.float8_e4m3fn),
    'F8_E5M2': np.dtype(ml_dtypes.float8_e5m2),
    'BF16': np.dtype(ml_dtypes.bfloat16),
    'F16': np.dtype(np.float16),
    'F32': np.dtype(np.float32),
    'F64': np.dtype(np.float64),
}
NP_TO_SAFETENSORS_DTYPE: Mapping[np.dtype, str] = {
    dt: name for name, dt in SAFETENSORS_DTYPES.items()
}

SHARED_POOL = concurrent.futures.ThreadPoolExecutor(MAX_STREAMS)
_CPU_COUNT = os.cpu_count()
PACK_WORKERS = min(
    64, max(READ_WORKERS, (16 if _CPU_COUNT is None else _CPU_COUNT) // 2)
)
PACK_POOL = concurrent.futures.ThreadPoolExecutor(PACK_WORKERS)
_REQ_COUNTER = itertools.count(1)
_SHM_ID_COUNTER = itertools.count(1)


class _ProcState:
  """Per-worker reply demux state avoiding global notify_all() wake storms."""

  def __init__(self) -> None:
    self.lock = threading.Lock()
    self.waiters: dict[int, threading.Event] = {}
    self.replies: dict[int, int] = {}
    self.reader_active = False


_PROC_STATES: dict[int, _ProcState] = {}
_PROC_STATES_LOCK = threading.Lock()
_GCS_TOKEN_LOCK = threading.RLock()
GCS_TOKEN_GEN: list[int] = [0]
GCS_TOKEN_CACHE: list[tuple[float, str]] = []
GCS_VIP_CACHE: list[tuple[float, tuple[str, ...]]] = []
_GCS_VIP_LOCK = threading.Lock()
GCS_RAPID_BUCKET_CACHE: dict[str, bool] = {}
_GCS_RAPID_LOCK = threading.Lock()
GCS_SOURCE_IPS_CACHE: dict[str, tuple[str, ...]] = {}
_GCS_SOURCE_IPS_LOCK = threading.Lock()
_GCS_THREAD_VIP_COUNTER = itertools.count()
_GCS_THREAD_VIP_IDX: dict[int, int] = {}
THREAD_CONNS: dict[tuple[Any, ...], http.client.HTTPSConnection] = {}
READ_PROCS: list[tuple[subprocess.Popen[str], threading.Lock]] = []
_READ_PROCS_LOCK = threading.RLock()


def await_worker_ready(
    proc: subprocess.Popen[str],
    deadline: float,
    error_cls: type[Exception] = RuntimeError,
    err_prefix: str = 'Worker failed startup handshake',
) -> None:
  """Waits until `proc.stdout` emits `READY` before `deadline` via epoll."""
  if proc.stdout is None:
    raise error_cls(f'{err_prefix}: missing stdout pipe')
  fd = -1
  with contextlib.suppress(OSError, ValueError, AttributeError):
    fd = proc.stdout.fileno()
  # Use DefaultSelector (epoll/kqueue) instead of select.select() so high
  # FDs (>= FD_SETSIZE=1024) on multi-host TPU workers never fail.
  is_ready = not isinstance(fd, int) or fd < 0
  if not is_ready:
    with selectors.DefaultSelector() as sel:
      sel.register(fd, selectors.EVENT_READ)
      is_ready = bool(sel.select(timeout=max(0.0, deadline - time.monotonic())))
  ready = proc.stdout.readline().strip() if is_ready else ''
  if ready != 'READY':
    raise error_cls(f'{err_prefix}: {ready!r}')


def available_shm_bytes(shm_dir: str = '') -> int:
  """Returns 80% of currently free bytes in `shm_dir` (default `/dev/shm`)."""
  root = shm_dir or (
      '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
  )
  st = os.statvfs(root)
  return max(4096, (st.f_bavail * st.f_frsize * 4) // 5)


def parse_gs_uri(gs_uri: str) -> tuple[str, str]:
  """Validates `gs://<bucket>/<object>` and returns `(bucket, object_key)`."""
  if not gs_uri.startswith('gs://') or '/' not in gs_uri[5:]:
    raise ValueError(f'Invalid GCS URI (expected gs://bucket/key): {gs_uri!r}.')
  bucket, key = gs_uri[5:].split('/', 1)
  if not bucket or not key:
    raise ValueError(f'Invalid GCS URI (empty bucket or key): {gs_uri!r}.')
  return bucket, key


def resolve_gcs_vips(vips: Sequence[str] = ()) -> tuple[str, ...]:
  """Returns caller-supplied VIPs or a 5-minute cached multi-VIP tuple."""
  if vips:
    return tuple(vips)
  now = time.monotonic()
  if GCS_VIP_CACHE and now < GCS_VIP_CACHE[0][0]:
    return GCS_VIP_CACHE[0][1]
  with _GCS_VIP_LOCK:
    now = time.monotonic()
    if GCS_VIP_CACHE and now < GCS_VIP_CACHE[0][0]:
      return GCS_VIP_CACHE[0][1]
    try:
      resolved = tuple(GsHttpPath.discover_vips())
    except OSError:
      return ()
    if resolved:
      GCS_VIP_CACHE[:] = [(time.monotonic() + GCS_VIP_CACHE_TTL_S, resolved)]
    return resolved


def pick_thread_vip(
    vips: Sequence[str],
    attempt: int = 0,
    vip_offset: int | None = None,
) -> str:
  """Assigns each calling thread a round-robin VIP and rotates on retries."""
  if not vips:
    return ''
  if vip_offset is not None:
    return vips[(vip_offset + attempt) % len(vips)]
  tid = threading.get_ident()
  idx = _GCS_THREAD_VIP_IDX.get(tid)
  if idx is None:
    idx = next(_GCS_THREAD_VIP_COUNTER)
    _GCS_THREAD_VIP_IDX[tid] = idx
  return vips[(idx + attempt) % len(vips)]


def evict_thread_https_conn(
    vip: str,
    tid: int | None = None,
    source_address: tuple[str, int] | None = None,
) -> None:
  """Closes and evicts the cached thread-local HTTPS connection for `vip`."""
  owner = threading.get_ident() if tid is None else tid
  key = (owner, vip) if source_address is None else (owner, vip, source_address)
  conn = THREAD_CONNS.pop(key, None)
  if conn is not None:
    try:
      conn.close()
    except (OSError, http.client.HTTPException):
      pass


def get_thread_https_conn(
    vip: str,
    timeout: float = 15.0,
    source_address: tuple[str, int] | None = None,
) -> http.client.HTTPSConnection:
  """Returns a thread-local Keep-Alive HTTPS connection pinned to `vip`."""
  owner = threading.get_ident()
  key = (owner, vip) if source_address is None else (owner, vip, source_address)
  conn = THREAD_CONNS.get(key)
  if conn is not None:
    if conn.sock is not None:
      conn.timeout = timeout
      conn.sock.settimeout(timeout)
      return conn
    THREAD_CONNS.pop(key, None)
    with contextlib.suppress(OSError, http.client.HTTPException):
      conn.close()
  ctx = ssl.create_default_context()
  conn = http.client.HTTPSConnection(
      'storage.googleapis.com',
      timeout=timeout,
      context=ctx,
      source_address=source_address,
  )
  if vip:
    raw_sock = socket.create_connection(
        (vip, 443), timeout=timeout, source_address=source_address
    )
    try:
      raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      # Leave SO_RCVBUF/SO_SNDBUF unset: explicit buffer sizing disables Linux
      # TCP autotuning and clamps sockets to rmem_max/wmem_max on COS pods.
      conn.sock = ctx.wrap_socket(
          raw_sock, server_hostname='storage.googleapis.com'
      )
    except BaseException:
      raw_sock.close()
      raise
  else:
    conn.connect()
  THREAD_CONNS[key] = conn
  return conn


def _probe_rapid_bucket_http(
    bucket: str, vips: tuple[str, ...], token: str
) -> tuple[bool, bool]:
  """Probes GCS bucket metadata; returns `(is_rapid, cacheable)`."""
  path = (
      f'/storage/v1/b/{urllib.parse.quote(bucket, safe="")}'
      '?fields=locationType,storageClass'
  )
  token_gen = GCS_TOKEN_GEN[0]
  for attempt in range(3):
    vip = vips[attempt % len(vips)]
    headers = {
        'Host': GCS_HOSTS[0],
        'Authorization': f'Bearer {token}',
        'Connection': 'keep-alive',
    }
    try:
      conn = get_thread_https_conn(vip, timeout=5.0)
      conn.request('GET', path, headers=headers)
      resp = conn.getresponse()
      status, body = resp.status, resp.read()
      if status == 200:
        data = json.loads(body)
        return (
            isinstance(data, dict)
            and data.get('locationType') == 'zone'
            and data.get('storageClass') == 'RAPID',
            True,
        )
    except (OSError, ValueError, http.client.HTTPException):
      evict_thread_https_conn(vip)
      continue
    evict_thread_https_conn(vip)
    if status == 403:
      logging.warning(
          'GCS bucket metadata probe denied (HTTP 403) for gs://%s; '
          'disabling Rapid Zonal auto-detect (grant storage.buckets.get '
          'to enable).',
          bucket,
      )
      return False, True
    if status == 404:
      return False, True
    if status != 401:
      continue
    refreshed = GsHttpPath.refresh_token_if_stale(token, token_gen)
    if refreshed in ('', token):
      return False, False
    token, token_gen = refreshed, GCS_TOKEN_GEN[0]
  return False, False


def is_rapid_zonal_bucket(gcs_uri: str) -> bool:
  """Returns True iff `gcs_uri` points to a Rapid Zonal GCS bucket (cached)."""
  if not gcs_uri.startswith('gs://'):
    return False
  bucket = gcs_uri[5:].split('/', 1)[0]
  if not bucket:
    return False
  with _GCS_RAPID_LOCK:
    if bucket in GCS_RAPID_BUCKET_CACHE:
      return GCS_RAPID_BUCKET_CACHE[bucket]
    token = GsHttpPath.resolve_token()
    if not token:
      return False
    is_rapid, cacheable = _probe_rapid_bucket_http(
        bucket, resolve_gcs_vips() or ('',), token
    )
    if cacheable:
      GCS_RAPID_BUCKET_CACHE[bucket] = is_rapid
    return is_rapid


# Linux AF_NETLINK NETLINK_ROUTE protocol constants (rtnetlink.h / netlink.h).
_NETLINK_ROUTE = 0  # NETLINK_ROUTE socket protocol (linux/netlink.h).
_NLM_F_REQUEST = 0x01  # Request message flag (linux/netlink.h).
_NLM_F_DUMP = 0x300  # NLM_F_ROOT | NLM_F_MATCH dump flag (linux/netlink.h).
_NLMSG_ERROR = 2  # Netlink error/ACK message type (linux/netlink.h).
_NLMSG_DONE = 3  # Multipart dump terminator type (linux/netlink.h).
_RTM_NEWADDR = 20  # Interface address response type (linux/rtnetlink.h).
_RTM_GETADDR = 22  # Interface address dump request (linux/rtnetlink.h).
_RTM_NEWROUTE = 24  # Route table entry response type (linux/rtnetlink.h).
_RTM_GETROUTE = 26  # Route lookup request type (linux/rtnetlink.h).
_IFA_ADDRESS = 1  # Interface address rtattr (linux/if_addr.h).
_IFA_LOCAL = 2  # Local address rtattr (linux/if_addr.h).
_RTA_DST = 1  # Route destination IP rtattr (linux/rtnetlink.h).
_RTA_SRC = 2  # Route source IP rtattr (linux/rtnetlink.h).
_RTA_OIF = 4  # Output interface index rtattr (linux/rtnetlink.h).
_RT_SCOPE_HOST = 254  # Host-local address scope threshold (linux/rtnetlink.h).


def _parse_nl_attrs(buf: bytes, offset: int) -> dict[int, bytes]:
  """Parses 4-byte-aligned Linux rtattrs from `buf[offset:]`."""
  attrs: dict[int, bytes] = {}
  end = len(buf)
  while offset + 4 <= end:
    rta_len, rta_type = struct.unpack_from('=HH', buf, offset)
    if rta_len < 4 or offset + rta_len > end:
      break
    attrs[rta_type] = buf[offset + 4 : offset + rta_len]
    offset += (rta_len + 3) & ~3
  return attrs


def _parse_nl_packet(
    data: bytes, expected_seq: int
) -> tuple[list[tuple[int, bytes]], str]:
  """Parses one Netlink recv buffer into `(msgs, status)`."""
  msgs: list[tuple[int, bytes]] = []
  off = 0
  end = len(data)
  while off + 16 <= end:
    nl_len, nl_type, _, nl_seq, _ = struct.unpack_from('=IHHII', data, off)
    if nl_len < 16 or off + nl_len > end:
      return [], 'error'
    payload = data[off + 16 : off + nl_len]
    off += (nl_len + 3) & ~3
    if nl_seq != expected_seq:
      continue
    if nl_type == _NLMSG_DONE:
      return msgs, 'done'
    if nl_type == _NLMSG_ERROR:
      err = (
          struct.unpack_from('=i', payload, 0)[0] if len(payload) >= 4 else -1
      )
      return (msgs, 'done') if err == 0 else ([], 'error')
    msgs.append((nl_type, payload))
  return msgs, 'continue'


def _netlink_route_exchange(
    msg_type: int, flags: int, body: bytes, *, multipart: bool
) -> list[tuple[int, bytes]]:
  """Sends one `NETLINK_ROUTE` request and collects response payloads."""
  try:
    sock_cm = socket.socket(
        socket.AF_NETLINK, socket.SOCK_RAW, _NETLINK_ROUTE
    )
  except (AttributeError, OSError):
    return []
  seq = 1
  try:
    with sock_cm as sock:
      sock.settimeout(2.0)
      sock.bind((0, 0))
      hdr = struct.pack('=IHHII', 16 + len(body), msg_type, flags, seq, 0)
      sock.send(hdr + body)
      msgs: list[tuple[int, bytes]] = []
      for _ in range(256):
        data = sock.recv(65536)
        if not data:
          break
        chunk, status = _parse_nl_packet(data, seq)
        if status == 'error':
          return []
        msgs.extend(chunk)
        if status == 'done' or (not multipart and msgs):
          break
      return msgs
  except OSError:
    return []


def _list_physical_ipv4_nics() -> list[tuple[str, str]]:
  """Returns `(dev, ipv4)` pairs for physical Linux NICs via `AF_NETLINK`."""
  req = struct.pack('=BBBBI', socket.AF_INET, 0, 0, 0, 0)
  msgs = _netlink_route_exchange(
      _RTM_GETADDR, _NLM_F_REQUEST | _NLM_F_DUMP, req, multipart=True
  )
  result: list[tuple[str, str]] = []
  seen_pairs: set[tuple[str, str]] = set()
  for msg_type, payload in msgs:
    if msg_type != _RTM_NEWADDR or len(payload) < 8:
      continue
    family, _, _, scope, ifindex = struct.unpack_from('=BBBBI', payload, 0)
    if family != socket.AF_INET or scope >= _RT_SCOPE_HOST:
      continue
    attrs = _parse_nl_attrs(payload, 8)
    raw_ip = attrs.get(_IFA_LOCAL) or attrs.get(_IFA_ADDRESS)
    if not raw_ip or len(raw_ip) != 4:
      continue
    ip = socket.inet_ntop(socket.AF_INET, raw_ip)
    if ip.startswith(('127.', '169.254.')):
      continue
    with contextlib.suppress(OSError):
      dev = socket.if_indextoname(ifindex)
      if (dev, ip) not in seen_pairs and os.path.exists(
          f'/sys/class/net/{dev}/device'
      ):
        seen_pairs.add((dev, ip))
        result.append((dev, ip))
  return result


def _route_egress_dev(gcs_vip: str, src_ip: str) -> str | None:
  """Queries kernel routing via `RTM_GETROUTE` and returns egress dev name."""
  with contextlib.suppress(OSError):
    rtm = struct.pack('=BBBBBBBBI', socket.AF_INET, 32, 32, 0, 0, 0, 0, 0, 0)
    body = (
        rtm
        + struct.pack('=HH', 8, _RTA_DST)
        + socket.inet_pton(socket.AF_INET, gcs_vip)
        + struct.pack('=HH', 8, _RTA_SRC)
        + socket.inet_pton(socket.AF_INET, src_ip)
    )
    msgs = _netlink_route_exchange(
        _RTM_GETROUTE, _NLM_F_REQUEST, body, multipart=False
    )
    payload = next(
        (p for t, p in msgs if t == _RTM_NEWROUTE and len(p) >= 12), b''
    )
    oif_raw = (
        _parse_nl_attrs(payload, 12).get(_RTA_OIF, b'') if payload else b''
    )
    if len(oif_raw) == 4:
      return socket.if_indextoname(struct.unpack('=I', oif_raw)[0])
  return None


def _probe_nic_egress(
    dev: str, src_ip: str, gcs_vip: str, seen_pcis: AbstractSet[str]
) -> tuple[str, str, int]:
  """Probes `dev`/`src_ip` egress to `gcs_vip`; returns `(status, pci, numa)`."""
  state = ''
  with contextlib.suppress(OSError):
    with open(f'/sys/class/net/{dev}/operstate', encoding='utf-8') as fh:
      state = fh.read().strip()
  pci = os.path.realpath(f'/sys/class/net/{dev}/device')
  if (
      state != 'up'
      or pci in seen_pcis
      or _route_egress_dev(gcs_vip, src_ip) != dev
  ):
    return 'skip', pci, -1
  node = -1
  with contextlib.suppress(OSError, ValueError):
    with open(f'/sys/class/net/{dev}/device/numa_node', encoding='utf-8') as fh:
      node = int(fh.read().strip())
  try:
    with contextlib.closing(
        socket.create_connection(
            (gcs_vip, 443), timeout=1.0, source_address=(src_ip, 0)
        )
    ):
      return 'ok', pci, node
  except OSError:
    return 'tcp_err', pci, node


def _probe_all_source_ips(vip: str) -> tuple[tuple[str, ...], bool]:
  """Probes candidate physical NICs; returns `(source_ips, cacheable)`."""
  nics = _list_physical_ipv4_nics()
  if len({d for d, _ in nics}) < 2:
    return (), True
  working: dict[str, tuple[int, str]] = {}
  routed_pcis: set[str] = set()
  tcp_failed = False
  for dev, ip in nics:
    status, pci, node = _probe_nic_egress(dev, ip, vip, working.keys())
    if status == 'ok':
      routed_pcis.add(pci)
      working[pci] = (node, ip)
    elif status == 'tcp_err':
      routed_pcis.add(pci)
      tcp_failed = True
  by_numa: dict[int, str] = {}
  for node, ip in working.values():
    if node >= 0:
      by_numa.setdefault(node, ip)
  if len(by_numa) >= 2:
    return tuple(by_numa.values()), True
  if len(working) >= 2:
    return tuple(ip for _, ip in working.values()), True
  return (), not (tcp_failed and len(routed_pcis) >= 2)


def discover_gcs_source_ips(gcs_vip: str = '') -> tuple[str, ...]:
  """Discovers `>= 2` distinct physical egress NIC IPv4s routed to `gcs_vip`.

  Args:
    gcs_vip: Optional target GCS IPv4 VIP to verify routing and TCP egress
      against. Defaults to the first resolved VIP from `resolve_gcs_vips()`.

  Returns:
    Tuple of verified source IPv4 strings across distinct physical NUMA/PCI
    interfaces, or `()` unless at least two distinct egress NICs work.
  """
  vip = gcs_vip or (resolve_gcs_vips() or ('',))[0]
  if not vip:
    return ()
  with _GCS_SOURCE_IPS_LOCK:
    if vip in GCS_SOURCE_IPS_CACHE:
      return GCS_SOURCE_IPS_CACHE[vip]
    ips, cacheable = _probe_all_source_ips(vip)
    if cacheable:
      GCS_SOURCE_IPS_CACHE[vip] = ips
    return ips


def _touch_shm_range(
    mm: mmap.mmap, off: int, count: int, write: bool
) -> None:
  """Touches one 4 KiB-strided page range in `mm` to populate PTEs."""
  if count <= 0:
    return
  view = np.frombuffer(mm, dtype=np.uint8, count=count, offset=off)
  if write:
    view[::_PREFAULT_PAGE_BYTES] = 0
  else:
    _ = int(view[::_PREFAULT_PAGE_BYTES].sum())


def pre_fault_shm(
    mm: mmap.mmap,
    total_bytes: int,
    *,
    offset: int = 0,
    write: bool = True,
    block_indices: Sequence[int] | None = None,
    block_bytes: int = 0,
    max_workers: int = _PREFAULT_WORKERS,
) -> None:
  """Populates `/dev/shm` page tables in parallel to bypass single-thread tmpfs stalls."""
  if block_indices is not None:
    spans = [
        (
            offset + b * block_bytes,
            min(block_bytes, total_bytes - b * block_bytes),
        )
        for b in block_indices
        if b * block_bytes < total_bytes
    ]
  else:
    if total_bytes <= 0:
      return
    blk = min(_PREFAULT_BLOCK_BYTES, max(_PREFAULT_PAGE_BYTES, total_bytes))
    n_blks = (total_bytes + blk - 1) // blk
    spans = [
        (offset + b * blk, min(blk, total_bytes - b * blk))
        for b in range(n_blks)
    ]
  if not spans:
    return
  if len(spans) == 1:
    _touch_shm_range(mm, spans[0][0], spans[0][1], write)
    return
  workers = min(max_workers, len(spans))
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futs = [
        pool.submit(_touch_shm_range, mm, off, cnt, write)
        for off, cnt in spans
    ]
    for fut in futs:
      fut.result()


_FALLOCATE_UNSUPPORTED_ERRNOS = frozenset({
    errno.EOPNOTSUPP,
    errno.ENOTSUP,
    errno.EINVAL,
    errno.ENOSYS,
    errno.EPERM,
})


def _fallocate_fd(fd: int, offset: int, length: int) -> bool:
  """Pre-allocates tmpfs backing pages via `os.posix_fallocate` if supported."""
  if length <= 0:
    return False
  try:
    os.posix_fallocate(fd, offset, length)
    return True
  except AttributeError:
    return False
  except OSError as exc:
    # Fall back to userspace page-touching only when posix_fallocate is
    # unsupported; propagate ENOSPC/ENOMEM/EIO so tmpfs exhaustion fails fast
    # instead of triggering uncatchable SIGBUS during subsequent mmap writes.
    if exc.errno is None or exc.errno in _FALLOCATE_UNSUPPORTED_ERRNOS:
      return False
    raise


def create_shm_mmap(
    shm_path: str, total_bytes: int, *, prefault: bool = True
) -> mmap.mmap:
  """Creates, truncates, mmaps, and optionally pre-faults `shm_path`."""
  fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
  fallocated = False
  try:
    os.ftruncate(fd, total_bytes)
    if prefault:
      fallocated = _fallocate_fd(fd, 0, total_bytes)
    mm = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
  except BaseException:
    os.close(fd)
    if os.path.exists(shm_path):
      os.unlink(shm_path)
    raise
  os.close(fd)
  if not prefault:
    return mm
  # Close and unlink on pre-fault failure so partial /dev/shm allocations
  # never leak across retries.
  try:
    pre_fault_shm(
        mm,
        total_bytes,
        write=True,
        max_workers=8 if fallocated else _PREFAULT_WORKERS,
    )
  except BaseException:
    mm.close()
    if os.path.exists(shm_path):
      os.unlink(shm_path)
    raise
  return mm


def terminate_proc(proc: subprocess.Popen[Any], timeout: float = 5.0) -> None:
  """Closes worker pipes and reaps `proc` cleanly without leaking zombies."""
  if proc.stdin is not None:
    try:
      proc.stdin.close()
    except OSError:
      pass
  try:
    proc.wait(timeout=timeout)
  except (subprocess.TimeoutExpired, OSError):
    proc.kill()
    try:
      proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
      pass
  if proc.stdout is not None:
    try:
      proc.stdout.close()
    except OSError:
      pass


def build_worker_cmd(
    script_file: str,
    entry_expr: str,
    flag_args: Sequence[str] = (),
    script_args: Sequence[str] = (),
    *,
    prefer_script: bool = False,
) -> list[str]:
  """Builds a zero-JAX subprocess argv for Python or bundled binaries."""
  exe = sys.executable or ''
  if _g3_mp_lib is not None:
    _g3_mp_lib.set_spawn_exe_path()
    spawn_mod = getattr(multiprocessing, 'spawn', None)
    if spawn_mod is not None:
      exe = os.fsdecode(spawn_mod.get_executable() or exe)
  if os.path.basename(exe).lower().startswith('python'):
    if (
        (prefer_script or not flag_args)
        and script_file
        and os.path.isfile(script_file)
    ):
      return [
          exe,
          '-u',
          os.path.abspath(script_file),
          *flag_args,
          *script_args,
      ]
    return [exe, '-u', '-c', entry_expr, *flag_args, *script_args]
  return [exe, '-u', *flag_args, '-c', entry_expr, *script_args]


def auto_gcs_read_procs(
    total_bytes: int, read_procs: int | None = None
) -> int:
  """Auto-scales GCS `/dev/shm` read worker count from host payload bytes."""
  if read_procs is not None:
    return max(0, min(GCS_MAX_READ_PROCS, int(read_procs)))
  if total_bytes < 64 * 1024 * 1024:
    return 0
  cpu_count = os.cpu_count()
  host_cap = min(
      GCS_MAX_READ_PROCS, max(16, 16 if cpu_count is None else cpu_count)
  )
  byte_target = max(8, total_bytes // (128 * 1024 * 1024))
  return min(host_cap, byte_target)


def _get_proc_state(proc: subprocess.Popen[str]) -> _ProcState:
  """Returns or creates the per-subprocess reply demux state for `proc`."""
  key = id(proc)
  state = _PROC_STATES.get(key)
  if state is None:
    with _PROC_STATES_LOCK:
      state = _PROC_STATES.setdefault(key, _ProcState())
  return state


def to_np_dtype(dtype: Any) -> np.dtype:
  """Converts a Safetensors dtype code or JAX/NumPy dtype to `np.dtype`."""
  if isinstance(dtype, str):
    if dtype not in SAFETENSORS_DTYPES:
      raise ValueError(f'Unsupported safetensors dtype: {dtype!r}')
    return SAFETENSORS_DTYPES[dtype]
  return np.dtype(dtype)


def from_np_dtype(dtype: Any) -> str:
  """Converts a NumPy or JAX dtype into its Safetensors wire string."""
  return NP_TO_SAFETENSORS_DTYPE[np.dtype(dtype)]


def read_safetensors_header(filepath: Any) -> tuple[int, dict[str, Any]]:
  """Reads and validates the 8-byte length prefix and JSON header of `filepath`."""
  with cast(io.BufferedIOBase, filepath.open('rb')) as fh:
    size_bytes = fh.read(8)
    if len(size_bytes) != 8:
      raise OSError(f'Truncated safetensors header in {filepath}.')
    header_size = struct.unpack('<Q', size_bytes)[0]
    if header_size == 0 or header_size > MAX_HEADER_BYTES:
      raise ValueError(
          f'Invalid safetensors header size {header_size} in {filepath}.'
      )
    header_bytes = fh.read(header_size)
    if len(header_bytes) != header_size:
      raise OSError(f'Truncated safetensors metadata in {filepath}.')
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
      raise ValueError(
          f'Safetensors header must be a JSON object in {filepath}.'
      )
    return header_size, header


def format_safetensors_header(
    tensors_meta: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, str] | None = None,
) -> bytes:
  """Serializes tensor metadata into an 8-byte-aligned Safetensors wire header."""
  header: dict[str, Any] = {}
  if metadata is not None:
    header['__metadata__'] = dict(metadata)
  for name, meta in tensors_meta.items():
    header[name] = {
        'dtype': str(meta['dtype']),
        'shape': list(meta['shape']),
        'data_offsets': list(meta['data_offsets']),
    }
  raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
  pad = (-len(raw)) % 8
  if pad:
    raw += b' ' * pad
  return struct.pack('<Q', len(raw)) + raw


# ==============================================================================
# Layer 3: Cloud Provider & Storage File-System Transport Layer
# ==============================================================================


class CachedRangeFile(io.BufferedIOBase):
  """Transport stream base with optional 256 KiB 1-RTT header caching."""

  def __init__(
      self,
      parent: Any,
      fh: io.BufferedIOBase | None = None,
      proc_entry: tuple[subprocess.Popen[str], threading.Lock] | None = None,
      proc_spec: str = '',
  ):
    super().__init__()
    self.parent = parent
    self.fh = fh
    self.proc_entry = proc_entry
    self.proc_spec = proc_spec or str(parent)
    self.pos = 0
    self.done_flags: Sequence[int] | None = None
    self.task_idx = 0

  def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
    if whence not in (io.SEEK_SET, io.SEEK_CUR):
      raise ValueError(f'Unsupported seek whence: {whence}.')
    self.pos = offset + (self.pos if whence == io.SEEK_CUR else 0)
    return self.pos

  def fetch_range(self, start: int, length: int, mv: memoryview[int]) -> int:
    del length
    assert self.fh is not None
    self.fh.seek(start)
    return self.fh.readinto(mv)

  def fetch_bytes(self, start: int, length: int) -> bytes:
    if self.fh is not None:
      self.fh.seek(start)
      return self.fh.read(length)
    mv = memoryview(bytearray(length))
    got = self.fetch_range(start, length, mv)
    return bytes(mv[:got])

  def read(self, size: int | None = -1) -> bytes:
    if size is None or size <= 0:
      return b''
    if self.fh is not None and not self.parent.use_hdr_cache:
      self.fh.seek(self.pos)
      res = self.fh.read(size)
      self.pos += len(res)
      return res
    if self.pos == 0 and size == 8 and self.parent.hdr_cache is None:
      self.parent.hdr_cache = self.fetch_bytes(0, 262144)
    hdr = self.parent.hdr_cache
    if hdr is not None and self.pos + size <= len(hdr):
      res = hdr[self.pos : self.pos + size]
      self.pos += len(res)
      return res
    res = self.fetch_bytes(self.pos, size)
    self.pos += len(res)
    return res

  def readinto(self, b) -> int:
    if self.done_flags is not None and self.done_flags[self.task_idx]:
      return 0
    # Release memoryviews deterministically so exceptions never pin `/dev/shm`.
    with memoryview(b) as raw_mv, raw_mv.cast('B') as mv:
      got = self.fetch_range(self.pos, len(mv), mv) if len(mv) else 0
    self.pos += got
    return got

  def read_shm_range(
      self,
      start: int,
      length: int,
      shm_uint8: np.ndarray,
      shm_off: int,
      shm_path: str = '',
      shm_cap: int = 0,
  ) -> int:
    """Reads `[start, start+length)` into `shm_uint8` via worker or handle."""
    if self.done_flags is not None and self.done_flags[self.task_idx]:
      return 0
    if self.proc_entry is not None and shm_path:
      cmd = (
          f'{shm_path}\t{shm_cap}\t{self.proc_spec}\t'
          f'{start}\t{length}\t{shm_off}'
      )
      n = ProcPoolPath.proc_read(self.proc_entry, cmd)
      self.pos = start + n
      return n
    self.seek(start)
    return self.readinto(shm_uint8[shm_off : shm_off + length])

  def close(self) -> None:
    if self.fh is not None:
      self.fh.close()
    super().close()


class RangeSource:
  """Uniform Layer 3 byte-range file source wrapping local or remote shards."""

  supports_hedge: bool = False
  use_hdr_cache: bool = False

  def __init__(self, raw_path: Any, name: str | None = None):
    self.raw_path = raw_path
    self.name = name if name is not None else raw_path.name
    self.hdr_cache: bytes | None = None

  def __fspath__(self) -> str:
    return os.fspath(self.raw_path)

  def __str__(self) -> str:
    return str(self.raw_path)

  def ensure_procs(self, total_bytes: int) -> int:
    """Returns the live `/dev/shm` read worker count (none by default)."""
    del total_bytes
    return 0

  def open(self, mode: str = 'rb', stream_id: int = -1) -> CachedRangeFile:
    del stream_id
    return CachedRangeFile(
        self, fh=cast(io.BufferedIOBase, self.raw_path.open(mode))
    )


class ProcPoolPath(RangeSource):
  """`epath` range source with a 256 KiB header cache and `/dev/shm` readers.

  Remote file systems registered via `register_storage_backend` subclass this
  to set `proc_env_var` and override `configure_flags`; all subclasses share
  one `READ_PROCS` pool of pipe-driven read worker subprocesses.
  """

  use_hdr_cache: bool = True
  proc_env_var: str = ''

  @classmethod
  def configure_flags(cls) -> tuple[str, ...]:
    """Applies storage client flags in-process and returns worker argv."""
    return ()

  @classmethod
  def from_files(cls, files: Sequence[Any]) -> list[RangeSource]:
    """Configures flags, honors `proc_env_var`, and wraps `files`."""
    cls.configure_flags()
    raw_procs = (
        os.environ[cls.proc_env_var].strip()
        if cls.proc_env_var and cls.proc_env_var in os.environ
        else ''
    )
    if raw_procs:
      cls.spawn_procs(int(raw_procs))
    return [cls(f) for f in files]

  def ensure_procs(self, total_bytes: int) -> int:
    """Spawns up to 16 `/dev/shm` worker processes for large remote I/O."""
    if not self.proc_env_var:
      return 0
    raw_procs = (
        os.environ[self.proc_env_var].strip()
        if self.proc_env_var in os.environ
        else ''
    )
    if raw_procs:
      target = int(raw_procs)
    elif total_bytes >= 64 * 1024 * 1024:
      target = min(16, max(1, total_bytes // (64 * 1024 * 1024)))
    else:
      target = 0
    type(self).spawn_procs(target)
    with _READ_PROCS_LOCK:
      return len(READ_PROCS)

  @staticmethod
  def open_worker_handle(spec: str, mode: str = 'rb') -> io.BufferedIOBase:
    """Opens a thread-local `epath` file handle in a read worker."""
    return cast(io.BufferedIOBase, epath.Path(spec).open(mode))

  @staticmethod
  def _close_shm_mmaps(
      shm_mmaps: dict[tuple[str, int], tuple[mmap.mmap, np.ndarray]],
      *,
      ignore_busy: bool = False,
  ) -> None:
    """Explicitly closes worker `/dev/shm` mmaps so unlinked arenas free RAM.

    Observed in 397B multi-sync runs: persistent read workers retained open
    mmaps to unlinked `/dev/shm/tunix_shm_*` files from prior weight syncs,
    leaking ~186 GiB/sync of `/dev/shm` ghost capacity across consecutive syncs.
    Dropping the NumPy buffer view before `mm.close()` returns pages to the OS
    immediately without raising `BufferError`.
    """
    # Pop one entry at a time so `del arr` drops the sole reference to the
    # NumPy buffer view before `mm.close()`; re-track any busy mmap if an
    # in-flight worker thread still holds a slice so it is closed on next UNMAP.
    busy: dict[tuple[str, int], tuple[mmap.mmap, np.ndarray]] = {}
    while shm_mmaps:
      key, (mm, arr) = shm_mmaps.popitem()
      del arr
      if ignore_busy:
        try:
          mm.close()
        except BufferError:
          busy[key] = (mm, np.frombuffer(mm, dtype=np.uint8))
      else:
        mm.close()
    shm_mmaps.update(busy)

  @staticmethod
  def _exec_worker_io(
      fields: Sequence[str],
      shm_mmaps: dict[tuple[str, int], tuple[mmap.mmap, np.ndarray]],
      file_handles: dict[tuple[int, str, str], io.BufferedIOBase],
      lock: threading.Lock,
  ) -> int:
    """Executes one worker `/dev/shm` range read operation."""
    shm_path, cap_str, spec, start_str, len_str, off_str = fields[:6]
    mode = 'rb'
    mmap_key = (shm_path, int(cap_str))
    fh_key = (threading.get_ident(), spec, mode)
    with lock:
      if mmap_key not in shm_mmaps:
        ProcPoolPath._close_shm_mmaps(shm_mmaps, ignore_busy=True)
        fd = os.open(shm_path, os.O_RDWR)
        try:
          mm = mmap.mmap(fd, mmap_key[1], access=mmap.ACCESS_WRITE)
        finally:
          os.close(fd)
        shm_mmaps[mmap_key] = (mm, np.frombuffer(mm, dtype=np.uint8))
      shm_arr = shm_mmaps[mmap_key][1]
      fh = file_handles.get(fh_key)
    if fh is None:
      fh = ProcPoolPath.open_worker_handle(spec, mode)
      with lock:
        file_handles[fh_key] = fh
    fh.seek(int(start_str))
    shm_slice = shm_arr[int(off_str) : int(off_str) + int(len_str)]
    return fh.readinto(shm_slice)

  @staticmethod
  def run_worker_cmd(
      cmd: str,
      shm_mmaps: dict[tuple[str, int], tuple[mmap.mmap, np.ndarray]],
      file_handles: dict[tuple[int, str, str], io.BufferedIOBase],
      lock: threading.Lock,
  ) -> None:
    """Executes one 16 MiB range read into `/dev/shm` on a worker thread."""
    fields = cmd.split('\t')
    n = -1
    try:
      n = ProcPoolPath._exec_worker_io(fields, shm_mmaps, file_handles, lock)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      sys.stderr.write(f'safetensors_io worker error: {exc!r}\n')
      sys.stderr.flush()
    finally:
      # Always emit a reply (`-1` on failure) so parent `_read_proc_until_reply`
      # wakes immediately instead of deadlocking forever on an orphaned req_id.
      reply = f'{fields[6]}\t{n}\n' if len(fields) > 6 else f'{n}\n'
      with lock:
        sys.stdout.write(reply)
        sys.stdout.flush()

  @classmethod
  def worker_loop(cls) -> None:
    """Subprocess loop executing 16 MiB range reads via `/dev/shm`."""
    cls.configure_flags()
    sys.stdout.write('READY\n')
    sys.stdout.flush()
    shm_mmaps: dict[tuple[str, int], tuple[mmap.mmap, np.ndarray]] = {}
    file_handles: dict[tuple[int, str, str], io.BufferedIOBase] = {}
    lock = threading.Lock()
    futs: list[concurrent.futures.Future[None]] = []
    for line in sys.stdin:
      if not (cmd := line.rstrip('\n')):
        break
      if cmd.startswith('UNMAP\t') or cmd == 'UNMAP':
        # Drain in-flight worker tasks before closing mmaps so no thread holds
        # a NumPy view into the arena; also drop old-step file wrappers while
        # preserving warm TLS sockets in THREAD_CONNS for the next weight sync.
        concurrent.futures.wait(futs)
        futs.clear()
        with lock:
          ProcPoolPath._close_shm_mmaps(shm_mmaps)
          for fh in file_handles.values():
            fh.close()
          file_handles.clear()
          parts = cmd.split('\t')
          reply = f'{parts[1]}\t0\n' if len(parts) > 1 else '0\n'
          sys.stdout.write(reply)
          sys.stdout.flush()
        continue
      futs.append(
          SHARED_POOL.submit(
              ProcPoolPath.run_worker_cmd, cmd, shm_mmaps, file_handles, lock
          )
      )
      # Observed in long-running persistent workers: unbounded `futs.append()`
      # retained Future objects across tens of thousands of range requests.
      if len(futs) >= 64:
        futs = [f for f in futs if not f.done()]
    concurrent.futures.wait(futs)
    for fh in file_handles.values():
      fh.close()
    ProcPoolPath._close_shm_mmaps(shm_mmaps, ignore_busy=True)

  @staticmethod
  def _read_proc_until_reply(
      proc: subprocess.Popen[str], state: _ProcState, req_id: int
  ) -> int:
    """Reads `proc.stdout` as sole reader until `req_id` completes."""
    assert proc.stdout is not None
    try:
      while True:
        line = proc.stdout.readline().strip()
        parts = line.split('\t') if line else ['-1']
        r_id = int(parts[0]) if len(parts) == 2 else req_id
        val = int(parts[-1])
        if r_id == req_id:
          return val
        with state.lock:
          state.replies[r_id] = val
          # Wake only the specific thread waiting for `r_id` (targeted 1-thread
          # handoff instead of a global `_PROC_COND.notify_all()` thundering
          # herd that woke ~224-288 threads on every 8-16 MiB reply).
          if (waiter := state.waiters.pop(r_id, None)) is not None:
            waiter.set()
    finally:
      with state.lock:
        state.reader_active = False
        if state.waiters:
          next(iter(state.waiters.values())).set()

  @staticmethod
  def proc_read(
      proc_entry: tuple[subprocess.Popen[str], threading.Lock],
      cmd: str,
  ) -> int:
    """Dispatches a multiplexed `/dev/shm` range command to `proc_entry`."""
    proc, lock = proc_entry
    state = _get_proc_state(proc)
    req_id = next(_REQ_COUNTER)
    with lock:
      assert proc.stdin is not None
      proc.stdin.write(f'{cmd}\t{req_id}\n')
      proc.stdin.flush()
    ev: threading.Event | None = None
    while True:
      with state.lock:
        if req_id in state.replies:
          return state.replies.pop(req_id)
        if not state.reader_active:
          state.reader_active = True
          state.waiters.pop(req_id, None)
          is_reader = True
        else:
          if ev is None:
            ev = threading.Event()
          ev.clear()
          state.waiters[req_id] = ev
          is_reader = False
      if is_reader:
        return ProcPoolPath._read_proc_until_reply(proc, state, req_id)
      assert ev is not None
      ev.wait()

  @staticmethod
  def release_proc_mmaps() -> None:
    """Unmaps `/dev/shm` arenas across all live worker procs in parallel."""
    with _READ_PROCS_LOCK:
      procs = list(READ_PROCS)
    if not procs:
      return
    list(
        SHARED_POOL.map(
            ProcPoolPath.proc_read, procs, itertools.repeat('UNMAP')
        )
    )

  @staticmethod
  def _reap_dead_procs(clear_all: bool = False) -> None:
    """Terminates all workers if `clear_all`, else prunes exited workers."""
    reap: list[subprocess.Popen[str]] = []
    with _READ_PROCS_LOCK:
      if clear_all:
        reap = [proc for proc, _ in READ_PROCS]
        READ_PROCS.clear()
      else:
        alive: list[tuple[subprocess.Popen[str], threading.Lock]] = []
        for entry in READ_PROCS:
          if isinstance(entry[0].poll(), int):
            reap.append(entry[0])
          else:
            alive.append(entry)
        READ_PROCS[:] = alive
    if not reap:
      return
    with _PROC_STATES_LOCK:
      for proc in reap:
        _PROC_STATES.pop(id(proc), None)
    for proc in reap:
      terminate_proc(proc)

  @classmethod
  def spawn_procs(cls, num_procs: int) -> None:
    """Applies storage client flags and ensures `num_procs` read workers."""
    flag_args = cls.configure_flags()
    with _READ_PROCS_LOCK:
      if num_procs <= 0:
        ProcPoolPath._reap_dead_procs(clear_all=True)
        return
      ProcPoolPath._reap_dead_procs(clear_all=False)
      need = num_procs - len(READ_PROCS)
      if need <= 0:
        return
      cmd = build_worker_cmd(
          __file__,
          f'from {cls.__module__} import {cls.__qualname__} as p;'
          ' p.worker_loop()',
          flag_args=flag_args,
          prefer_script=True,
      )
      env = {
          **os.environ,
          'CUDA_VISIBLE_DEVICES': '',
          'TPU_VISIBLE_CHIPS': '',
          'TPU_CHIPS_PER_HOST_BOUNDS': '',
      }
      # Spawn all needed workers in parallel before waiting for READY so worker
      # startup runs concurrently across processes.
      pending: list[subprocess.Popen[str]] = []
      try:
        for _ in range(need):
          pending.append(
              subprocess.Popen(
                  cmd,
                  stdin=subprocess.PIPE,
                  stdout=subprocess.PIPE,
                  text=True,
                  env=env,
              )
          )
        deadline = time.monotonic() + 60.0
        for proc in pending:
          await_worker_ready(
              proc,
              deadline,
              error_cls=OSError,
              err_prefix='Read worker subprocess failed to start',
          )
      except BaseException:
        for bad in pending:
          bad.kill()
          terminate_proc(bad)
        raise
      READ_PROCS.extend((proc, threading.Lock()) for proc in pending)

  def open(self, mode: str = 'rb', stream_id: int = -1) -> CachedRangeFile:
    with _READ_PROCS_LOCK:
      proc_entry = (
          READ_PROCS[stream_id % len(READ_PROCS)]
          if stream_id >= 0 and READ_PROCS
          else None
      )
    fh = (
        None
        if proc_entry is not None
        else cast(io.BufferedIOBase, self.raw_path.open(mode))
    )
    return CachedRangeFile(self, fh, proc_entry)


atexit.register(ProcPoolPath._reap_dead_procs, True)  # pylint: disable=protected-access


class ParallelStripes(Protocol):
  """Opens one remote file as parallel stripes written by stage workers."""

  def stripe_bytes(self, path: str) -> int:
    """Returns the stripe size for `path`, or `0` if it is not striped."""
    ...

  def open(self, path: str, num_stripes: int) -> tuple[Any, Sequence[str]]:
    """Opens `path` and returns `(handle, per-stripe destination paths)`."""
    ...

  def close(self, handle: Any) -> None:
    """Seals a handle returned by `open` once every stripe is written."""
    ...


@dataclasses.dataclass(frozen=True)
class StorageBackendSpec:
  """Hooks for a remote file system served under a registered path prefix.

  Attributes:
    path_cls: `ProcPoolPath` subclass wrapping read shards and read workers.
    validate: Raises `ValueError` on a malformed output directory.
    write_flags: Applies client write flags in-process and returns the argv
      forwarded to stage worker subprocesses.
    open_stage_file: Opens a binary sink that a stage worker streams one
      `/dev/shm` slot into.
    parallel_stripes: Optional striped writer for slots above one slot size.
  """

  path_cls: type[ProcPoolPath]
  validate: Callable[[str], None]
  write_flags: Callable[[], tuple[str, ...]]
  open_stage_file: Callable[[str], ContextManager[Any]]
  parallel_stripes: ParallelStripes | None = None


STORAGE_BACKENDS: dict[str, StorageBackendSpec] = {}


def register_storage_backend(prefix: str, backend: StorageBackendSpec) -> None:
  """Routes paths starting with `prefix` (e.g. `'/fs/'`) to `backend`."""
  if not prefix.endswith('/'):
    raise ValueError(f'Storage prefix must end with "/", got {prefix!r}.')
  STORAGE_BACKENDS[prefix] = backend


def find_storage_backend(path: Any) -> StorageBackendSpec | None:
  """Returns the longest registered prefix match for `path` (or its root)."""
  p = str(path)
  hits = [pre for pre in STORAGE_BACKENDS if p.startswith(pre) or p == pre[:-1]]
  return STORAGE_BACKENDS[max(hits, key=len)] if hits else None


class GsHttpPath(RangeSource):
  """GCP `gs://` range source with multi-VIP striping and tail hedging."""

  supports_hedge: bool = True
  use_hdr_cache: bool = True

  def __init__(self, gs_uri: str, token: str, vips: Sequence[str] = ()):
    bucket, obj = parse_gs_uri(gs_uri)
    super().__init__(gs_uri, name=obj.rsplit('/', 1)[-1])
    self.bucket = bucket
    self.obj = obj
    self.http_path = f'/{bucket}/{urllib.parse.quote(obj, safe="/")}'
    self.token = token
    self.vips = tuple(vips)

  @staticmethod
  def _fetch_metadata_token() -> tuple[str, float]:
    """Queries GCE metadata for an OAuth2 access token with bounded retries."""
    for attempt in range(3):
      try:
        with contextlib.closing(
            http.client.HTTPConnection('169.254.169.254', timeout=2.0)
        ) as conn:
          conn.request(
              'GET',
              '/computeMetadata/v1/instance/service-accounts/default/token',
              headers={'Metadata-Flavor': 'Google'},
          )
          resp = conn.getresponse()
          body = resp.read()
          if resp.status == 200:
            payload = json.loads(body)
            token = str(payload.get('access_token') or '').strip()
            raw_expires = payload.get('expires_in')
            expires_in = (
                3600.0 if raw_expires is None else float(raw_expires)
            )
            if token:
              return token, expires_in
      except (OSError, ValueError, http.client.HTTPException):
        pass
      if attempt < 2:
        time.sleep(0.05 * (attempt + 1))
    return '', 0.0

  @staticmethod
  def resolve_token(force_refresh: bool = False) -> str:
    """Resolves a GCS OAuth2 token from environment variables or GCE metadata.

    Observed in 397B multi-sync runs: an unlocked 300s cache TTL with a 0.5s
    metadata timeout caused ~192 concurrent worker threads to stampede
    169.254.169.254 simultaneously, poisoning the cache with `""` for 60s and
    triggering cascading HTTP 401 failures. Serialize refreshes under
    `_GCS_TOKEN_LOCK`, retry with a >=2.0s timeout, and never cache empty
    tokens.
    """
    env_tok = (os.environ.get('TUNIX_GCS_TOKEN') or '').strip()
    if env_tok:
      return env_tok
    now = time.monotonic()
    if not force_refresh and GCS_TOKEN_CACHE and now < GCS_TOKEN_CACHE[0][0]:
      return GCS_TOKEN_CACHE[0][1]
    with _GCS_TOKEN_LOCK:
      now = time.monotonic()
      if not force_refresh and GCS_TOKEN_CACHE and now < GCS_TOKEN_CACHE[0][0]:
        return GCS_TOKEN_CACHE[0][1]
      prev_token = GCS_TOKEN_CACHE[0][1] if GCS_TOKEN_CACHE else ''
      token, expires_in = GsHttpPath._fetch_metadata_token()
      if token:
        ttl = min(300.0, expires_in - 60.0)
        if ttl > 0.0:
          GCS_TOKEN_CACHE[:] = [(time.monotonic() + ttl, token)]
        else:
          GCS_TOKEN_CACHE.clear()
        GCS_TOKEN_GEN[0] += 1
        return token
      return prev_token

  @staticmethod
  def refresh_token_if_stale(failed_token: str, token_gen: int) -> str:
    """Deduplicates concurrent HTTP 401 token refreshes across worker threads."""
    with _GCS_TOKEN_LOCK:
      if (
          GCS_TOKEN_GEN[0] != token_gen
          and GCS_TOKEN_CACHE
          and GCS_TOKEN_CACHE[0][1] != failed_token
      ):
        return GCS_TOKEN_CACHE[0][1]
      return GsHttpPath.resolve_token(force_refresh=True)

  @staticmethod
  def discover_vips() -> list[str]:
    """Resolves IPv4 VIPs across `GCS_HOSTS` for multi-VIP socket striping."""
    vips: set[str] = set()
    for host in GCS_HOSTS:
      try:
        infos = socket.getaddrinfo(
            host, 443, socket.AF_INET, socket.SOCK_STREAM
        )
      except OSError:
        continue
      for info in infos:
        vips.add(str(info[4][0]))
    return sorted(vips)

  def open(self, mode: str = 'rb', stream_id: int = -1) -> 'GsHttpFile':
    del mode
    idx = stream_id if stream_id >= 0 else threading.get_ident()
    chosen = self.vips[idx % len(self.vips)] if self.vips else ''
    # GCS reads use either direct thread-local HTTPS or `GcsShmReadPool`, never
    # `READ_PROCS` pipe workers.
    return GsHttpFile(self, vip=chosen)


# Matches `bytes 0-99/100` -> ('0', '99', '100') or
# `bytes 0-99/*` -> ('0', '99', '*').
_CONTENT_RANGE_RE = re.compile(r'bytes (\d+)-(\d+)/(\d+|\*)')


def _expected_response_bytes(
    resp: Any, start: int, length: int
) -> int | None:
  """Parses expected response byte count from HTTP Range/Length headers."""
  get_hdr = getattr(resp, 'getheader', None)
  if not callable(get_hdr):
    return None
  cr = get_hdr('Content-Range') if resp.status == 206 else None
  if isinstance(cr, str):
    m = _CONTENT_RANGE_RE.fullmatch(cr)
    if m is None:
      return -1
    lo, hi = int(m.group(1)), int(m.group(2))
    want_hi = (
        hi if m.group(3) == '*' else min(start + length, int(m.group(3))) - 1
    )
    if lo != start or hi != want_hi or not (lo <= hi < start + length):
      return -1
    return hi - lo + 1
  cl = get_hdr('Content-Length')
  if not isinstance(cl, str) or not cl.isdigit():
    return None
  cl_val = int(cl)
  if resp.status == 206 and cl_val > length:
    return -1
  return min(length, cl_val)


class GsHttpFile(CachedRangeFile):
  """HTTP/1.1 Keep-Alive Range stream with multi-VIP socket pinning."""

  def __init__(
      self,
      parent: Any,
      vip: str = '',
      fh: io.BufferedIOBase | None = None,
      proc_entry: tuple[subprocess.Popen[str], threading.Lock] | None = None,
      proc_spec: str = '',
  ):
    super().__init__(parent, fh=fh, proc_entry=proc_entry, proc_spec=proc_spec)
    self.vip = vip

  def read_body(
      self, resp: http.client.HTTPResponse, length: int, mv: memoryview[int]
  ) -> int:
    total = 0
    done = self.done_flags
    while total < length and (done is None or not done[self.task_idx]):
      if not (n := resp.readinto(mv[total:])):
        break
      total += n
    return total

  def _reset_conn_and_maybe_refresh(
      self,
      conn: http.client.HTTPSConnection,
      resp: http.client.HTTPResponse,
      headers: dict[str, str],
      refreshed_auth: bool,
      token_gen: int = 0,
  ) -> bool:
    """Closes a failed GCS response socket and retries HTTP 401 once."""
    status = resp.status
    try:
      resp.read()
    finally:
      conn.close()
      evict_thread_https_conn(self.vip)
    # Retry HTTP 401 once after forcing a fresh GCE metadata token (e.g., when
    # a persistent worker subprocess outlives its initial token TTL).
    if status == 401 and not refreshed_auth:
      new_tok = GsHttpPath.refresh_token_if_stale(self.parent.token, token_gen)
      if new_tok:
        self.parent.token = new_tok
        headers['Authorization'] = f'Bearer {new_tok}'
        return True
    return refreshed_auth

  def _read_range_attempt(
      self,
      headers: dict[str, str],
      start: int,
      length: int,
      mv: memoryview[int],
      refreshed_auth: bool,
      token_gen: int,
  ) -> tuple[int | None, bool, bool]:
    """Executes one HTTP Range attempt; returns `(total, refreshed, fatal)`."""
    conn = get_thread_https_conn(self.vip)
    conn.request('GET', self.parent.http_path, headers=headers)
    resp = conn.getresponse()
    # Reject HTTP 200 when `start > 0`: reading offset 0 into a non-zero
    # slice would silently corrupt tensor weights.
    if not (resp.status == 206 or (resp.status == 200 and start == 0)):
      refreshed = self._reset_conn_and_maybe_refresh(
          conn, resp, headers, refreshed_auth, token_gen
      )
      return None, refreshed, resp.status in (403, 404)
    expected = _expected_response_bytes(resp, start, length)
    if expected is not None and expected < 0:
      refreshed = self._reset_conn_and_maybe_refresh(
          conn, resp, headers, refreshed_auth, token_gen
      )
      return None, refreshed, False
    total = self.read_body(resp, length, mv)
    cancelled = self.done_flags is not None and bool(
        self.done_flags[self.task_idx]
    )
    target = length if expected is None else expected
    if resp.status == 200 or total < target:
      evict_thread_https_conn(self.vip)
    if total < target and not cancelled:
      return None, refreshed_auth, False
    return total, refreshed_auth, False

  def fetch_range(self, start: int, length: int, mv: memoryview[int]) -> int:
    headers = {
        'Host': 'storage.googleapis.com',
        'Authorization': f'Bearer {self.parent.token}',
        'Range': f'bytes={start}-{start + length - 1}',
        'Connection': 'keep-alive',
    }
    refreshed_auth = False
    for attempt in range(3):
      token_gen = GCS_TOKEN_GEN[0]
      try:
        total, refreshed_auth, fatal = self._read_range_attempt(
            headers, start, length, mv, refreshed_auth, token_gen
        )
        if total is not None:
          return total
        if fatal:
          break
      except (OSError, http.client.HTTPException):
        evict_thread_https_conn(self.vip)
        if attempt == 2:
          raise
    raise OSError(f'Failed HTTP Range read at {start} of {self.parent.obj}.')


def is_not_found_error(exc: BaseException) -> bool:
  """Returns True when `exc` indicates a missing local, GCS, or remote path."""
  if isinstance(exc, FileNotFoundError):
    return True
  msg = str(exc)
  return (
      'NOT_FOUND' in msg
      or 'No such file' in msg
      or bool(re.search(r'\b404\b', msg))
  )


def is_strict_int(val: Any) -> TypeGuard[int]:
  """Returns True iff `val` is an `int` and not a `bool`."""
  return isinstance(val, int) and not isinstance(val, bool)


def validate_tensor_meta(
    filepath: Any, name: str, meta: Any
) -> tuple[int, int, np.dtype, tuple[int, ...]]:
  """Validates safetensors dtype, non-negative shape, and exact byte span."""
  if not isinstance(meta, Mapping):
    raise ValueError(f'Invalid metadata for {name!r} in {filepath}.')
  dtype_key = meta.get('dtype')
  raw_shape = meta.get('shape')
  raw_offsets = meta.get('data_offsets')
  if (
      not isinstance(dtype_key, str)
      or dtype_key not in SAFETENSORS_DTYPES
      or not isinstance(raw_shape, (list, tuple))
      or not isinstance(raw_offsets, (list, tuple))
      or len(raw_offsets) != 2
  ):
    raise ValueError(f'Invalid metadata for {name!r} in {filepath}.')
  np_dt = SAFETENSORS_DTYPES[dtype_key]
  if any(not is_strict_int(d) or d < 0 for d in raw_shape):
    raise ValueError(f'Invalid shape {raw_shape!r} for {name!r} in {filepath}.')
  shape = tuple(raw_shape)
  start, end = raw_offsets
  expected_bytes = math.prod(shape) * np_dt.itemsize
  if (
      not is_strict_int(start)
      or not is_strict_int(end)
      or start < 0
      or end - start != expected_bytes
  ):
    raise ValueError(
        f'Invalid data_offsets {(start, end)!r} for {name!r} in {filepath}.'
    )
  return start, end, np_dt, shape


def _wrap_range_sources(files: Sequence[Any]) -> list[RangeSource]:
  """Wraps resolved `.safetensors` paths into Layer 3 `RangeSource` adapters."""
  first = str(files[0])
  if (backend := find_storage_backend(first)) is not None:
    return backend.path_cls.from_files(files)
  if not first.startswith('gs://'):
    return [RangeSource(f) for f in files]
  token = GsHttpPath.resolve_token()
  if not token:
    return [RangeSource(f) for f in files]
  vips = resolve_gcs_vips()
  return [GsHttpPath(str(f), token, vips) for f in files]


def resolve_range_sources(file_dir: str) -> list[RangeSource]:
  """Resolves `.safetensors` shards into Layer 3 `RangeSource` adapters."""
  root_dir = (
      epath.Path(file_dir)
      if '://' in str(file_dir)
      else epath.Path(file_dir).expanduser()
  )
  files = sorted(root_dir.glob('*.safetensors'))
  if not files:
    raise ValueError(f'No safetensors found in {file_dir}')
  return _wrap_range_sources(files)


def build_range_sources_from_names(
    file_dir: str, shard_names: Sequence[str]
) -> list[RangeSource]:
  """Constructs `RangeSource`s from known shard names, skipping `glob`/list."""
  if not shard_names:
    raise ValueError(f'No safetensors found in {file_dir}')
  names = sorted(shard_names)
  if str(file_dir).startswith('gs://'):
    token = GsHttpPath.resolve_token()
    if token:
      base = str(file_dir).rstrip('/')
      vips = resolve_gcs_vips()
      return [GsHttpPath(f'{base}/{name}', token, vips) for name in names]
    root_dir = epath.Path(file_dir)
    return [RangeSource(root_dir / name) for name in names]
  root_dir = (
      epath.Path(file_dir)
      if '://' in str(file_dir)
      else epath.Path(file_dir).expanduser()
  )
  return _wrap_range_sources([root_dir / name for name in names])


# Validates a single-segment `.safetensors` filename with no traversal or
# control characters: "model-00001-of-00002.safetensors" -> match;
# "../evil.safetensors", "sub/a.safetensors", "a.safetensors\n" -> None.
_SIDECAR_SHARD_RE = re.compile(
    r'^(?!.*\.\.)[^/\\\x00-\x1f\x7f]+\.safetensors\Z'
)


def _parse_sidecar_shard(
    entry: Any,
    seen_keys: set[str],
) -> tuple[int, dict[str, Any]] | None:
  """Validates one `.tunix_headers.json` shard entry and returns `(len, map)`."""
  if not isinstance(entry, dict):
    return None
  header_len = entry.get('header_len')
  file_size = entry.get('file_size')
  tensors = entry.get('tensors')
  if (
      not is_strict_int(header_len)
      or not (0 < header_len <= MAX_HEADER_BYTES)
      or not is_strict_int(file_size)
      or file_size < 8 + header_len
      or not isinstance(tensors, dict)
  ):
    return None
  data_size = file_size - 8 - header_len
  for name, meta in tensors.items():
    if not isinstance(meta, dict):
      return None
    if name == '__metadata__':
      continue
    if name in seen_keys:
      return None
    try:
      _, end, _, _ = validate_tensor_meta(SIDECAR_HEADERS_FILENAME, name, meta)
    except ValueError:
      return None
    if end > data_size:
      return None
    seen_keys.add(name)
  return header_len, tensors


def sidecar_matches_weight_map(
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]],
    weight_map: Mapping[str, str],
    *,
    total_size: int | None = None,
    file_sizes: Mapping[str, int] | None = None,
) -> bool:
  """Returns True iff `shard_headers` matches `weight_map` and byte sizes."""
  if weight_map and set(shard_headers) != set(weight_map.values()):
    return False
  sidecar_map: dict[str, str] = {}
  shard_ends: dict[str, int] = {}
  actual_total = 0
  for s_name, (hdr_len, tensors) in shard_headers.items():
    data_items = {k: v for k, v in tensors.items() if k != '__metadata__'}
    sidecar_map.update(dict.fromkeys(data_items, s_name))
    actual_total += sum(
        m['data_offsets'][1] - m['data_offsets'][0] for m in data_items.values()
    )
    shard_ends[s_name] = 8 + hdr_len + max(
        (m['data_offsets'][1] for m in data_items.values()), default=0
    )
  if weight_map and sidecar_map != dict(weight_map):
    return False
  if file_sizes is not None and shard_ends != dict(file_sizes):
    return False
  return total_size is None or actual_total == total_size


def load_sidecar_headers(
    file_dir: str,
) -> dict[str, tuple[int, dict[str, Any]]] | None:
  """Loads optional `.tunix_headers.json` sidecar, or `None` if absent."""
  root_dir = (
      epath.Path(file_dir)
      if '://' in str(file_dir)
      else epath.Path(file_dir).expanduser()
  )
  try:
    payload = json.loads((root_dir / SIDECAR_HEADERS_FILENAME).read_text())
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  shards = payload.get('shards') if isinstance(payload, dict) else None
  version = payload.get('version') if isinstance(payload, dict) else None
  if (
      not isinstance(shards, dict)
      or not shards
      or not is_strict_int(version)
      or version != 1
  ):
    return None
  parsed: dict[str, tuple[int, dict[str, Any]]] = {}
  seen_keys: set[str] = set()
  for shard_name, entry in shards.items():
    if not _SIDECAR_SHARD_RE.match(shard_name):
      return None
    shard_val = _parse_sidecar_shard(entry, seen_keys)
    if shard_val is None:
      return None
    parsed[shard_name] = shard_val
  return parsed


# ==============================================================================
# Layer 1: Shared POSIX `/dev/shm` Arena & Sliding-Window Credit Lock
# ==============================================================================


class ShmArena:
  """Direction-neutral `/dev/shm` mmap arena and sliding-window credit lock."""

  def __init__(self, size_bytes: int = 0, shm_budget_bytes: int = 0):
    shm_dir = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
    env_budget = (
        int(os.environ['TUNIX_SHM_BUDGET_BYTES'])
        if 'TUNIX_SHM_BUDGET_BYTES' in os.environ
        else SHM_BUDGET_BYTES
    )
    budget = shm_budget_bytes or size_bytes or env_budget
    if budget <= 0:
      budget = available_shm_bytes(shm_dir)
    self.capacity = budget
    # Monotonic counter prevents path reuse collisions when Python reuses id().
    self.shm_path = os.path.join(
        shm_dir, f'tunix_shm_{os.getpid()}_{next(_SHM_ID_COUNTER)}'
    )
    self.mm: mmap.mmap | None = create_shm_mmap(
        self.shm_path, self.capacity, prefault=False
    )
    self.shm_uint8: np.ndarray | None = np.frombuffer(self.mm, dtype=np.uint8)
    self.base_ptr = self.shm_uint8.ctypes.data
    self.cursor = 0
    self.ring_base = 0
    self.in_flight_bytes = 0
    self.next_admit_unit = 0
    self.ring_capacity = self.capacity
    self.aborted = False
    self._used_procs = False
    self._used_gcs_pool = False
    self._prefaulted_bytes = 0
    self.file_shm_bytes: dict[int, int] = {}
    self.unit_shm_bytes: list[int] = []
    self.lock = threading.Lock()
    self.cond = threading.Condition(self.lock)

  def ensure_prefaulted(self, nbytes: int, *, touch_ptes: bool = True) -> None:
    """Monotonically pre-allocates `/dev/shm` pages (and optional write PTEs)."""
    target = min(self.capacity, max(0, nbytes))
    if self.mm is None or target <= self._prefaulted_bytes:
      return
    start = self._prefaulted_bytes
    delta = target - start
    fallocated = False
    if os.path.exists(self.shm_path):
      fd = os.open(self.shm_path, os.O_RDWR)
      try:
        fallocated = _fallocate_fd(fd, start, delta)
      finally:
        os.close(fd)
    if touch_ptes or not fallocated:
      pre_fault_shm(
          self.mm,
          delta,
          offset=start,
          write=True,
          max_workers=8 if fallocated else _PREFAULT_WORKERS,
      )
    self._prefaulted_bytes = target

  def __enter__(self) -> 'ShmArena':
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    try:
      # Isolate worker unmap failures so a broken pipe never skips releasing
      # the sibling pool, unlinking `shm_path`, or masks an active exception.
      with contextlib.suppress(OSError):
        if self._used_procs:
          ProcPoolPath.release_proc_mmaps()
      with contextlib.suppress(OSError):
        if self._used_gcs_pool:
          release_gcs_read_pool_mmaps()
    finally:
      self.shm_uint8 = None
      if self.mm is not None:
        try:
          self.mm.close()
        except BufferError:
          pass
        self.mm = None
      if os.path.exists(self.shm_path):
        os.unlink(self.shm_path)

  def alloc(self, shape: int | Sequence[int]) -> np.ndarray:
    """Allocates a 64-byte-aligned `uint8` array slice from `/dev/shm`."""
    shape_tup = (shape,) if isinstance(shape, int) else tuple(shape)
    nbytes = math.prod(shape_tup)
    shm = self.shm_uint8
    if shm is None or nbytes <= 0:
      return np.empty(shape_tup, dtype=np.uint8)
    with self.lock:
      start = (self.cursor + 63) & ~63
      end = start + nbytes
      # Reserve min(capacity // 2, 64 GiB) for the strided staging ring instead
      # of unconditionally capping direct buffers at 50%: on 918 GiB TPU hosts
      # (capacity=734 GiB), 50% (367 GiB) missed 397B@2H's 370 GiB direct slice
      # by 3 GiB, forcing a 2-batch barrier.
      ring_reserve = min(self.capacity // 2, 64 << 30)
      if end > self.capacity - ring_reserve:
        return np.empty(shape_tup, dtype=np.uint8)
      self.cursor = end
    return shm[start:end].reshape(shape_tup)

  def admit_units(self) -> None:
    """Admits waiting units whose `/dev/shm` staging spans fit in the ring."""
    while self.next_admit_unit < len(self.unit_shm_bytes):
      need = self.unit_shm_bytes[self.next_admit_unit]
      if (
          self.in_flight_bytes > 0
          and self.in_flight_bytes + need > self.ring_capacity
      ):
        return
      self.in_flight_bytes += need
      self.next_admit_unit += 1

  def release_unit(self, unit_idx: int) -> None:
    """Recycles `unit_idx`'s `/dev/shm` ring bytes immediately after consumption."""
    with self.cond:
      freed = self.unit_shm_bytes[unit_idx]
      if self.in_flight_bytes < freed:
        raise RuntimeError(
            f'ShmArena ring underflow: {self.in_flight_bytes} < {freed}.'
        )
      self.unit_shm_bytes[unit_idx] = 0
      self.in_flight_bytes -= freed
      self.admit_units()
      self.cond.notify_all()

  def release_file(self, file_idx: int, total_files: int = 0) -> None:
    """Recycles `file_idx`'s `/dev/shm` slot and wakes waiting streams."""
    del total_files
    with self.cond:
      self.file_shm_bytes.pop(file_idx, None)
      self.admit_units()
      self.cond.notify_all()


# ==============================================================================
# Layer 3b: Lock-Free `/dev/shm` GCS Read Worker Pool
# ==============================================================================

GCS_SHM_TOKEN_CAP = 4096
# 4 MiB paths table holds >16k GCS object URIs (~250 B each) for per-wave Rapid.
GCS_SHM_PATHS_CAP = 4 * 1024 * 1024
# 16,384 groups accommodates per-wave 397B Rapid checkpoints without fallback.
GCS_SHM_MAX_GROUPS = 16384
GCS_SHM_MAX_TASKS = 131072


class _GcsCtrlHeader(ctypes.Structure):
  """Shared-memory control header between coordinator and GCS read workers."""

  _fields_ = [
      ('batch_id', ctypes.c_int64),
      ('shm_gen', ctypes.c_int64),
      ('num_tasks', ctypes.c_int64),
      ('num_groups', ctypes.c_int64),
      ('task_counter', ctypes.c_int64),
      ('done_count', ctypes.c_int64),
      ('max_allowed_task', ctypes.c_int64),
      ('hedge_counter', ctypes.c_int64),
      ('urgent_group_end', ctypes.c_int64),
      ('urgent_hedge_cursor', ctypes.c_int64),
      ('ready_workers', ctypes.c_int64),
      ('finished_workers', ctypes.c_int64),
      ('unmapped_workers', ctypes.c_int64),
      ('error_flag', ctypes.c_int64),
      ('shm_cap', ctypes.c_int64),
      ('paths_len', ctypes.c_int64),
      ('token_len', ctypes.c_int64),
      ('token_gen', ctypes.c_int64),
      ('shm_path_len', ctypes.c_int64),
      ('shm_path_buf', ctypes.c_char * 256),
      ('error_msg', ctypes.c_char * 256),
  ]


class _GcsRangeTaskStruct(ctypes.Structure):
  """Fixed-width 32-byte GCS range task entry in the control mmap."""

  _fields_ = [
      ('file_idx', ctypes.c_int32),
      ('group_idx', ctypes.c_int32),
      ('start', ctypes.c_int64),
      ('length', ctypes.c_int64),
      ('shm_off', ctypes.c_int64),
  ]


_CTRL_TOKEN_OFF = 1024
_CTRL_PATHS_OFF = _CTRL_TOKEN_OFF + GCS_SHM_TOKEN_CAP
_CTRL_GROUP_REM_OFF = _CTRL_PATHS_OFF + GCS_SHM_PATHS_CAP
_CTRL_GROUP_ACTIVE_OFF = _CTRL_GROUP_REM_OFF + 8 * GCS_SHM_MAX_GROUPS
_CTRL_DONE_OFF = _CTRL_GROUP_ACTIVE_OFF + 8 * GCS_SHM_MAX_GROUPS
_CTRL_TASKS_OFF = _CTRL_DONE_OFF + GCS_SHM_MAX_TASKS
_CTRL_TOTAL_BYTES = (
    _CTRL_TASKS_OFF + ctypes.sizeof(_GcsRangeTaskStruct) * GCS_SHM_MAX_TASKS
)


class _ShmMutex:
  """Intra-process + inter-process mutex backed by `threading.Lock` + `flock`."""

  def __init__(self, lock_path: str) -> None:
    self._tlock = threading.Lock()
    self._fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)

  def __enter__(self) -> None:
    self._tlock.acquire()
    try:
      fcntl.flock(self._fd, fcntl.LOCK_EX)
    except BaseException:
      self._tlock.release()
      raise

  def __exit__(self, *_: Any) -> None:
    try:
      fcntl.flock(self._fd, fcntl.LOCK_UN)
    finally:
      self._tlock.release()

  def close(self) -> None:
    if self._fd >= 0:
      os.close(self._fd)
      self._fd = -1


class _GcsCtrlView:
  """Typed ctypes views over the shared `/dev/shm` GCS control mmap."""

  def __init__(self, mm: mmap.mmap) -> None:
    self.mm: mmap.mmap | None = mm
    self.hdr: Any = _GcsCtrlHeader.from_buffer(mm, 0)
    self.group_rem: Any = (ctypes.c_int64 * GCS_SHM_MAX_GROUPS).from_buffer(
        mm, _CTRL_GROUP_REM_OFF
    )
    self.group_active: Any = (ctypes.c_int64 * GCS_SHM_MAX_GROUPS).from_buffer(
        mm, _CTRL_GROUP_ACTIVE_OFF
    )
    self.done_flags: Any = (ctypes.c_uint8 * GCS_SHM_MAX_TASKS).from_buffer(
        mm, _CTRL_DONE_OFF
    )
    self.tasks: Any = (_GcsRangeTaskStruct * GCS_SHM_MAX_TASKS).from_buffer(
        mm, _CTRL_TASKS_OFF
    )

  def read_token(self) -> str:
    assert self.mm is not None
    return self.mm[
        _CTRL_TOKEN_OFF : _CTRL_TOKEN_OFF + int(self.hdr.token_len)
    ].decode('utf-8')

  def write_token(self, token: str) -> None:
    assert self.mm is not None
    raw = token.encode('utf-8')
    if len(raw) > GCS_SHM_TOKEN_CAP:
      raise ValueError('GCS bearer token exceeds control buffer capacity.')
    self.mm[_CTRL_TOKEN_OFF : _CTRL_TOKEN_OFF + len(raw)] = raw
    self.hdr.token_len = len(raw)

  def read_paths(self) -> list[str]:
    assert self.mm is not None
    raw = self.mm[
        _CTRL_PATHS_OFF : _CTRL_PATHS_OFF + int(self.hdr.paths_len)
    ].decode('utf-8')
    return raw.split('\0') if raw else []

  def write_paths(self, encoded_paths: bytes) -> None:
    assert self.mm is not None
    self.mm[_CTRL_PATHS_OFF : _CTRL_PATHS_OFF + len(encoded_paths)] = (
        encoded_paths
    )
    self.hdr.paths_len = len(encoded_paths)

  def close(self) -> None:
    self.hdr = None
    self.group_rem = None
    self.group_active = None
    self.done_flags = None
    self.tasks = None
    if self.mm is not None:
      try:
        self.mm.close()
      except BufferError:
        pass
      self.mm = None


class _WorkerLocalState:
  """Per-worker-process data arena mapping shared across worker threads."""

  def __init__(self) -> None:
    self.data_mm: mmap.mmap | None = None
    self.data_arr: np.ndarray | None = None
    self.shm_gen = -1
    self.last_batch_id = 0
    self.file_paths: list[str] = []
    self.running = False

  def close_data_arena(self) -> None:
    self.data_arr = None
    if self.data_mm is not None:
      self.data_mm.close()
      self.data_mm = None
    self.shm_gen = -1


def _gcs_refresh_shared_token(
    ctrl: _GcsCtrlView,
    shm_lock: _ShmMutex,
    failed_token: str,
    seen_gen: int,
) -> str:
  """Refreshes an expired bearer token once and publishes it across workers."""
  # Hold `shm_lock` across the metadata refresh so all 96 worker processes
  # serialize on a single metadata fetch and readers never observe torn tokens.
  with shm_lock:
    if ctrl.hdr.token_gen != seen_gen:
      cur = ctrl.read_token()
      if cur and cur != failed_token:
        return cur
    new_tok = GsHttpPath.refresh_token_if_stale(failed_token, GCS_TOKEN_GEN[0])
    if not new_tok:
      return failed_token
    ctrl.write_token(new_tok)
    ctrl.hdr.token_gen += 1
    return new_tok


def _gcs_read_http_once(
    ctrl: _GcsCtrlView,
    shm_lock: _ShmMutex,
    http_path: str,
    task_idx: int,
    start: int,
    length: int,
    dst_arr: np.ndarray,
    vip: str,
    timeout: float,
    source_address: tuple[str, int] | None = None,
) -> str:
  """Runs one HTTP Range GET into `dst_arr`; returns 'done'|'retry'|'fatal'."""
  # Snapshot `(token_gen, token)` under `shm_lock` to eliminate torn UTF-8 reads
  # while another worker publishes a refreshed OAuth2 bearer token.
  with shm_lock:
    seen_gen = int(ctrl.hdr.token_gen)
    token = ctrl.read_token()
  headers = {
      'Host': 'storage.googleapis.com',
      'Authorization': f'Bearer {token}',
      'Range': f'bytes={start}-{start + length - 1}',
      'Connection': 'keep-alive',
  }
  try:
    conn = get_thread_https_conn(
        vip, timeout=timeout, source_address=source_address
    )
    conn.request('GET', http_path, headers=headers)
    resp = conn.getresponse()
    if resp.status != 206 and (resp.status != 200 or start != 0):
      status = resp.status
      resp.read()
      evict_thread_https_conn(vip, source_address=source_address)
      if status == 401:
        _gcs_refresh_shared_token(ctrl, shm_lock, token, seen_gen)
      return 'fatal' if status in (403, 404) else 'retry'
    got = 0
    with memoryview(dst_arr) as mv:
      while (
          got < length
          and not ctrl.done_flags[task_idx]
          and not ctrl.hdr.error_flag
          and (n := resp.readinto(mv[got:]))
      ):
        got += n
  except (OSError, http.client.HTTPException):
    evict_thread_https_conn(vip, source_address=source_address)
    return 'retry'
  if resp.status == 200 or got < length:
    evict_thread_https_conn(vip, source_address=source_address)
  return 'done' if (got == length or ctrl.done_flags[task_idx]) else 'retry'


def _claim_next_gcs_task(
    ctrl: _GcsCtrlView, shm_lock: _ShmMutex, num_tasks: int
) -> tuple[int, bool]:
  """Claims the next task by priority: urgent > primary > hedge."""
  hdr = ctrl.hdr
  if (
      hdr.urgent_group_end <= hdr.urgent_hedge_cursor
      and hdr.max_allowed_task <= hdr.task_counter < num_tasks
  ):
    return -2, False
  with shm_lock:
    # Increment `group_active` atomically with every claim (primary or hedge) so
    # ring recycling waits until zero active writers remain for that group.
    end = min(num_tasks, int(hdr.urgent_group_end))
    while int(hdr.urgent_hedge_cursor) < end:
      cur = int(hdr.urgent_hedge_cursor)
      hdr.urgent_hedge_cursor = cur + 1
      if not ctrl.done_flags[cur]:
        ctrl.group_active[int(ctrl.tasks[cur].group_idx)] += 1
        return cur, True
    if hdr.task_counter < hdr.max_allowed_task:
      idx = int(hdr.task_counter)
      hdr.task_counter = idx + 1
      ctrl.group_active[int(ctrl.tasks[idx].group_idx)] += 1
      return idx, False
    if hdr.task_counter < num_tasks:
      return -2, False
    while hdr.hedge_counter < num_tasks:
      idx = num_tasks - 1 - int(hdr.hedge_counter)
      hdr.hedge_counter += 1
      if not ctrl.done_flags[idx]:
        ctrl.group_active[int(ctrl.tasks[idx].group_idx)] += 1
        return idx, True
  return -1, False


def _finish_gcs_task(
    ctrl: _GcsCtrlView,
    shm_lock: _ShmMutex,
    task_idx: int,
    group_idx: int,
    ok: bool,
    is_hedge: bool,
) -> bool:
  """Records task completion or primary failure; returns False to halt batch."""
  if ok:
    with shm_lock:
      if not ctrl.done_flags[task_idx]:
        ctrl.done_flags[task_idx] = 1
        ctrl.hdr.done_count += 1
        ctrl.group_rem[group_idx] -= 1
    return True
  if is_hedge or ctrl.done_flags[task_idx]:
    return True
  with shm_lock:
    if not ctrl.hdr.error_flag:
      ctrl.hdr.error_flag = 1
      ctrl.hdr.error_msg = (
          f'GCS range read failed for task {task_idx}'.encode('utf-8')[:255]
      )
  return False


def _execute_gcs_task(
    ctrl: _GcsCtrlView,
    shm_lock: _ShmMutex,
    file_paths: Sequence[str],
    data_arr: np.ndarray,
    task_idx: int,
    is_hedge: bool,
    vips_pair: tuple[str, str],
    source_address: tuple[str, int] | None = None,
) -> bool:
  """Executes one claimed range task with VIP rotation and bounded retries."""
  task = ctrl.tasks[task_idx]
  http_path = file_paths[int(task.file_idx)]
  start, length, shm_off = int(task.start), int(task.length), int(task.shm_off)
  dst_arr = data_arr[shm_off : shm_off + length]
  timeout = GCS_HEDGE_TIMEOUT_S if is_hedge else GCS_PRIMARY_TIMEOUT_S
  max_attempts = 2 if is_hedge else 6
  ok = False
  for attempt in range(max_attempts):
    # Halt retries immediately when another worker or the coordinator aborts.
    if ctrl.hdr.error_flag:
      return False
    if ctrl.done_flags[task_idx]:
      ok = True
      break
    vip = vips_pair[attempt & 1]
    # Clear NIC binding on the second half of retries so a worker pinned to a
    # secondary NIC recovers if a VIP subnet lacks a route or the NIC flaps.
    attempt_src = None if attempt >= max_attempts // 2 else source_address
    status = _gcs_read_http_once(
        ctrl,
        shm_lock,
        http_path,
        task_idx,
        start,
        length,
        dst_arr,
        vip,
        timeout,
        source_address=attempt_src,
    )
    if status != 'retry':
      ok = status == 'done'
      break
    if not is_hedge and attempt + 1 < max_attempts:
      time.sleep(0.05 * (attempt + 1))
  return _finish_gcs_task(
      ctrl, shm_lock, task_idx, int(task.group_idx), ok, is_hedge
  )


class _GcsShmWorker:
  """Zero-JAX worker process executing GCS range reads over `/dev/shm`."""

  def __init__(
      self,
      ctrl_path: str,
      wid: int,
      streams_per_proc: int,
      vips: Sequence[str],
      source_ip: str = '',
  ) -> None:
    fd = os.open(ctrl_path, os.O_RDWR)
    try:
      ctrl_mm = mmap.mmap(fd, _CTRL_TOTAL_BYTES, access=mmap.ACCESS_WRITE)
    finally:
      os.close(fd)
    self.ctrl = _GcsCtrlView(ctrl_mm)
    self.shm_lock = _ShmMutex(f'{ctrl_path}.lock')
    self.state = _WorkerLocalState()
    self.barrier = threading.Barrier(streams_per_proc)
    self.wid = wid
    self.streams_per_proc = streams_per_proc
    self.vips = tuple(vips)
    self.source_address: tuple[str, int] | None = (
        (source_ip, 0) if source_ip else None
    )
    self._init_ppid = os.getppid()

  def _prewarm_vips(self, tid: int) -> tuple[str, str]:
    """Selects `(primary_vip, hedge_vip)` for `tid` and pre-warms TLS sockets."""
    stream_id = self.wid * self.streams_per_proc + tid
    n_vips = len(self.vips)
    primary_vip = self.vips[stream_id % n_vips] if n_vips else ''
    hedge_vip = (
        self.vips[(stream_id + max(1, n_vips // 2)) % n_vips] if n_vips else ''
    )
    targets = ((primary_vip, GCS_PRIMARY_TIMEOUT_S),) + (
        ((hedge_vip, GCS_HEDGE_TIMEOUT_S),) if hedge_vip != primary_vip else ()
    )
    for vip, timeout in targets:
      if not vip:
        continue
      try:
        get_thread_https_conn(
            vip, timeout=timeout, source_address=self.source_address
        )
      except (OSError, http.client.HTTPException):
        evict_thread_https_conn(vip, source_address=self.source_address)
    return primary_vip, hedge_vip

  def _prepare_batch(self) -> bool:
    """Waits for the next control-plane batch and attaches any new data shm."""
    hdr = self.ctrl.hdr
    while True:
      while int(hdr.batch_id) == self.state.last_batch_id:
        # Exit cleanly if the coordinator process died (e.g. SIGKILL) so
        # orphaned workers never hold open `/dev/shm` mmaps indefinitely.
        if os.getppid() != self._init_ppid:
          return False
        time.sleep(0.0005)
      bid = int(hdr.batch_id)
      self.state.last_batch_id = bid
      if bid == -1:
        return False
      if bid <= -2:
        self.state.close_data_arena()
        with self.shm_lock:
          hdr.unmapped_workers += 1
        continue
      break
    # Acquire shm_lock once after observing batch_id so weakly-ordered CPUs
    # observe all coordinator metadata writes published prior to batch_id.
    with self.shm_lock:
      pass
    shm_gen = int(hdr.shm_gen)
    if shm_gen != self.state.shm_gen:
      self.state.close_data_arena()
      shm_path = bytes(hdr.shm_path_buf[: int(hdr.shm_path_len)]).decode(
          'utf-8'
      )
      fd = os.open(shm_path, os.O_RDWR)
      try:
        self.state.data_mm = mmap.mmap(
            fd, int(hdr.shm_cap), access=mmap.ACCESS_WRITE
        )
      finally:
        os.close(fd)
      self.state.data_arr = np.frombuffer(self.state.data_mm, dtype=np.uint8)
      self.state.shm_gen = shm_gen
    self.state.file_paths = self.ctrl.read_paths()
    return True

  def _run_batch(self, primary_vip: str, hedge_vip: str) -> None:
    """Drains primary and hedged GCS range tasks until the batch completes."""
    hdr = self.ctrl.hdr
    num_tasks = int(hdr.num_tasks)
    assert self.state.data_arr is not None
    while hdr.done_count < num_tasks and not hdr.error_flag:
      task_idx, is_hedge = _claim_next_gcs_task(
          self.ctrl, self.shm_lock, num_tasks
      )
      if task_idx == -1:
        return
      if task_idx == -2:
        time.sleep(0.0002)
        continue
      vips_pair = (
          (hedge_vip, primary_vip) if is_hedge else (primary_vip, hedge_vip)
      )
      group_idx = int(self.ctrl.tasks[task_idx].group_idx)
      try:
        ok = _execute_gcs_task(
            self.ctrl,
            self.shm_lock,
            self.state.file_paths,
            self.state.data_arr,
            task_idx,
            is_hedge,
            vips_pair,
            source_address=self.source_address,
        )
      finally:
        # Release the per-group active reader reference after any stalled
        # primary or hedged read exits so circular ring slots can recycle.
        with self.shm_lock:
          self.ctrl.group_active[group_idx] -= 1
      if not ok:
        return

  def _thread_loop(self, tid: int) -> None:
    """Pre-warms VIP TLS sockets and executes batches across worker threads."""
    try:
      primary_vip, hedge_vip = self._prewarm_vips(tid)
      if self.barrier.wait() == 0:
        with self.shm_lock:
          self.ctrl.hdr.ready_workers += 1
      while True:
        if tid == 0:
          self.state.running = self._prepare_batch()
        self.barrier.wait()
        if not self.state.running:
          return
        self._run_batch(primary_vip, hedge_vip)
        if self.barrier.wait() == 0:
          with self.shm_lock:
            self.ctrl.hdr.finished_workers += 1
    except threading.BrokenBarrierError:
      return
    except BaseException as exc:
      # Abort the shared barrier so sibling threads wake instead of deadlocking
      # forever, and publish the exception so the coordinator fails fast.
      with self.shm_lock:
        if self.ctrl.hdr is not None and not self.ctrl.hdr.error_flag:
          self.ctrl.hdr.error_flag = 1
          self.ctrl.hdr.error_msg = repr(exc).encode('utf-8')[:255]
      self.barrier.abort()
      raise

  def run(self) -> None:
    """Spawns stream threads, joins until exit, and closes shared mmaps."""
    threads = [
        threading.Thread(target=self._thread_loop, args=(t,), daemon=True)
        for t in range(self.streams_per_proc)
    ]
    for th in threads:
      th.start()
    try:
      for th in threads:
        th.join()
    finally:
      self.state.close_data_arena()
      self.ctrl.close()
      self.shm_lock.close()


def run_gcs_shm_worker_cli(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint for zero-JAX `--gcs-shm-worker` subprocesses."""
  args = list(sys.argv[1:] if argv is None else argv)
  idx = args.index('--gcs-shm-worker')
  ctrl_path = args[idx + 1]
  wid = int(args[idx + 2])
  streams_per_proc = int(args[idx + 3])
  vips_csv = args[idx + 4] if idx + 4 < len(args) else ''
  source_ip = args[idx + 5] if idx + 5 < len(args) else ''
  vips = tuple(v for v in vips_csv.split(',') if v)
  _GcsShmWorker(
      ctrl_path, wid, streams_per_proc, vips, source_ip=source_ip
  ).run()


class GcsShmReadPool:
  """Persistent zero-JAX subprocess pool using lock-free `/dev/shm` queues."""

  def __init__(
      self,
      num_procs: int,
      vips: Sequence[str],
      streams_per_proc: int = GCS_STREAMS_PER_PROC,
      *,
      spawn: bool = True,
      source_ips: Sequence[str] | None = None,
  ) -> None:
    shm_dir = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
    self.num_procs = num_procs
    self.streams_per_proc = streams_per_proc
    self.vips = tuple(vips)
    if source_ips is not None:
      self.source_ips = tuple(source_ips)
    elif spawn and self.vips:
      # Probe one representative VIP per /16 prefix and bind only NICs routed
      # across all prefixes (e.g. private.googleapis.com vs public VIPs).
      rep_vips = tuple(
          {v.rsplit('.', 2)[0]: v for v in reversed(self.vips)}.values()
      )
      per_prefix = [discover_gcs_source_ips(v) for v in rep_vips]
      common = tuple(
          ip for ip in per_prefix[0] if all(ip in ips for ips in per_prefix[1:])
      )
      self.source_ips = common if len(common) >= 2 else ()
    else:
      self.source_ips = ()
    self.ctrl_path = os.path.join(
        shm_dir, f'tunix_gcs_ctrl_{os.getpid()}_{next(_SHM_ID_COUNTER)}'
    )
    self.lock_path = f'{self.ctrl_path}.lock'
    self._procs: list[subprocess.Popen[Any]] = []
    self._attached_shm: tuple[str, int] | None = None
    self._batch_seq = 0
    self._unmap_seq = 0
    self._healthy = True
    ctrl_mm = create_shm_mmap(self.ctrl_path, _CTRL_TOTAL_BYTES, prefault=False)
    self.ctrl = _GcsCtrlView(ctrl_mm)
    try:
      self.shm_lock = _ShmMutex(self.lock_path)
    except BaseException:
      self.ctrl.close()
      for path in (self.ctrl_path, self.lock_path):
        if os.path.exists(path):
          os.unlink(path)
      raise
    if spawn:
      self._spawn_workers()

  def _spawn_workers(self) -> None:
    """Spawns `num_procs` zero-JAX workers and waits for TLS pre-warm."""
    env = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': '',
        'TPU_VISIBLE_CHIPS': '',
        'TPU_CHIPS_PER_HOST_BOUNDS': '',
    }
    vips_csv = ','.join(self.vips)
    entry = (
        'from tunix.models import safetensors_io as s;'
        ' s.run_gcs_shm_worker_cli()'
    )
    try:
      for wid in range(self.num_procs):
        src_ip = (
            self.source_ips[wid % len(self.source_ips)]
            if len(self.source_ips) >= 2
            else ''
        )
        script_args = (
            '--gcs-shm-worker',
            self.ctrl_path,
            str(wid),
            str(self.streams_per_proc),
            vips_csv,
            *((src_ip,) if src_ip else ()),
        )
        cmd = build_worker_cmd(
            __file__,
            entry,
            script_args=script_args,
            prefer_script=True,
        )
        self._procs.append(subprocess.Popen(cmd, env=env))
      deadline = time.monotonic() + 30.0
      while int(self.ctrl.hdr.ready_workers) < self.num_procs:
        if any(p.poll() is not None for p in self._procs):
          raise OSError('GCS /dev/shm read worker exited during startup.')
        if time.monotonic() > deadline:
          raise OSError('Timed out waiting for GCS /dev/shm read workers.')
        time.sleep(0.005)
    except BaseException:
      self.shutdown()
      raise

  @staticmethod
  def can_handle(num_tasks: int, num_groups: int, paths_len: int) -> bool:
    return (
        num_tasks <= GCS_SHM_MAX_TASKS
        and num_groups <= GCS_SHM_MAX_GROUPS
        and paths_len <= GCS_SHM_PATHS_CAP
    )

  def is_alive(self) -> bool:
    return (
        self._healthy
        and bool(self._procs)
        and all(p.poll() is None for p in self._procs)
    )

  def start_batch(
      self,
      shm_path: str,
      shm_cap: int,
      file_paths: Sequence[str],
      token: str,
      tasks: Sequence[tuple[int, int, int, int, int]],
      group_counts: Sequence[int],
      initial_max_allowed: int,
  ) -> None:
    """Publishes one batch of range tasks to all workers in zero pipe copies."""
    if self._procs and not self.is_alive():
      raise OSError('GCS /dev/shm read worker crashed before batch start.')
    hdr = self.ctrl.hdr
    encoded_paths = '\0'.join(file_paths).encode('utf-8')
    if not self.can_handle(len(tasks), len(group_counts), len(encoded_paths)):
      raise ValueError('GCS batch exceeds shared-memory control table limits.')
    if self._attached_shm != (shm_path, shm_cap):
      raw_shm = shm_path.encode('utf-8')
      if len(raw_shm) > 256:
        raise ValueError('GCS /dev/shm data path exceeds 256-byte limit.')
      hdr.shm_path_buf = raw_shm
      hdr.shm_path_len = len(raw_shm)
      hdr.shm_cap = shm_cap
      hdr.shm_gen += 1
      self._attached_shm = (shm_path, shm_cap)
    self.ctrl.write_token(token)
    self.ctrl.write_paths(encoded_paths)
    hdr.num_groups = len(group_counts)
    for g_idx, cnt in enumerate(group_counts):
      self.ctrl.group_rem[g_idx] = cnt
    ctypes.memset(
        ctypes.addressof(self.ctrl.group_active), 0, 8 * len(group_counts)
    )
    ctypes.memset(
        ctypes.addressof(self.ctrl.done_flags), 0, len(tasks)
    )
    for t_idx, (f_idx, g_idx, start, length, shm_off) in enumerate(tasks):
      slot = self.ctrl.tasks[t_idx]
      slot.file_idx = f_idx
      slot.group_idx = g_idx
      slot.start = start
      slot.length = length
      slot.shm_off = shm_off
    with self.shm_lock:
      hdr.num_tasks = len(tasks)
      hdr.task_counter = 0
      hdr.done_count = 0
      hdr.hedge_counter = 0
      hdr.urgent_group_end = 0
      hdr.urgent_hedge_cursor = 0
      hdr.finished_workers = 0
      hdr.error_flag = 0
      hdr.error_msg = b''
      hdr.max_allowed_task = initial_max_allowed
      self._batch_seq += 1
      hdr.batch_id = self._batch_seq

  def wait_batch_drained(
      self, timeout_s: float = 5.0, *, raise_on_timeout: bool = True
  ) -> None:
    """Waits until all worker threads exit the current batch before arena reuse."""
    if self.ctrl.hdr is None or self._batch_seq == 0:
      return
    deadline = time.monotonic() + timeout_s
    while int(self.ctrl.hdr.finished_workers) < self.num_procs:
      if self._procs and not self.is_alive():
        self._healthy = False
        break
      if time.monotonic() > deadline:
        # Timed-out workers may still write into `/dev/shm`; tear down and
        # evict the pool so the next batch cannot overwrite live worker memory.
        self.shutdown()
        if raise_on_timeout:
          raise OSError(
              'Timed out waiting for GCS /dev/shm read batch to drain.'
          )
        return
      time.sleep(0.001)

  def update_max_allowed(self, max_allowed: int) -> None:
    """Advances the sliding-window ring credit ceiling atomically."""
    if max_allowed <= int(self.ctrl.hdr.max_allowed_task):
      return
    with self.shm_lock:
      if max_allowed > int(self.ctrl.hdr.max_allowed_task):
        self.ctrl.hdr.max_allowed_task = max_allowed

  def trigger_urgent_hedge(self, group_end: int) -> None:
    """Marks stalled tasks up to `group_end` for immediate `2.5 s` hedging."""
    if group_end <= int(self.ctrl.hdr.urgent_group_end):
      return
    with self.shm_lock:
      if group_end > int(self.ctrl.hdr.urgent_group_end):
        self.ctrl.hdr.urgent_group_end = group_end

  def abort_batch(self) -> None:
    """Signals workers to stop the active batch immediately on consumer error."""
    if self.ctrl.hdr is None:
      return
    with self.shm_lock:
      self.ctrl.hdr.error_flag = 1
      self.ctrl.hdr.max_allowed_task = self.ctrl.hdr.num_tasks

  def check_health(self) -> None:
    """Raises `OSError` immediately if any worker reported an error or died."""
    hdr = self.ctrl.hdr
    if hdr.error_flag:
      msg = bytes(hdr.error_msg).split(b'\0', 1)[0].decode('utf-8', 'replace')
      raise OSError(msg or 'GCS /dev/shm read worker failed.')
    if self._procs and not self.is_alive():
      self.abort_batch()
      raise OSError('GCS /dev/shm read worker exited unexpectedly.')

  def unmap_data_arena(self, timeout_s: float = 5.0) -> None:
    """Unmaps the data arena across all workers so unlinked `/dev/shm` frees."""
    if self._attached_shm is None or self.ctrl.hdr is None:
      return
    if self._procs and not self.is_alive():
      self._attached_shm = None
      return
    with self.shm_lock:
      self.ctrl.hdr.unmapped_workers = 0
      self._unmap_seq += 1
      self.ctrl.hdr.batch_id = -(self._unmap_seq + 2)
    deadline = time.monotonic() + timeout_s
    while int(self.ctrl.hdr.unmapped_workers) < self.num_procs:
      if self._procs and not self.is_alive():
        self._healthy = False
        break
      if time.monotonic() > deadline:
        # Kill wedged workers so their open mmaps cannot pin unlinked
        # `/dev/shm` ghost capacity across subsequent weight syncs.
        self.shutdown()
        return
      time.sleep(0.001)
    self._attached_shm = None

  def shutdown(self) -> None:
    """Terminates all worker processes and unlinks `/dev/shm` control files."""
    self._healthy = False
    with _GCS_READ_POOL_LOCK:
      if _GCS_READ_POOL and _GCS_READ_POOL[0] is self:
        _GCS_READ_POOL.clear()
    if self.ctrl.hdr is not None:
      with self.shm_lock:
        self.ctrl.hdr.error_flag = 1
        self.ctrl.hdr.batch_id = -1
    # Bound total teardown wait across all workers with one shared deadline
    # instead of waiting up to 5s sequentially per process (96 * 5s = 480s).
    deadline = time.monotonic() + 5.0
    for proc in self._procs:
      terminate_proc(proc, timeout=max(0.05, deadline - time.monotonic()))
    self._procs.clear()
    self.ctrl.close()
    self.shm_lock.close()
    for path in (self.ctrl_path, self.lock_path):
      if os.path.exists(path):
        os.unlink(path)


_GCS_READ_POOL: list[GcsShmReadPool] = []
_GCS_READ_POOL_LOCK = threading.RLock()


def ensure_gcs_read_pool(
    num_procs: int, vips: Sequence[str]
) -> GcsShmReadPool | None:
  """Returns a live persistent `GcsShmReadPool` with `>= num_procs` workers."""
  if num_procs <= 0:
    return None
  with _GCS_READ_POOL_LOCK:
    if _GCS_READ_POOL:
      pool = _GCS_READ_POOL[0]
      if pool.is_alive() and pool.num_procs >= num_procs:
        return pool
      _GCS_READ_POOL.clear()
      pool.shutdown()
    pool = GcsShmReadPool(num_procs, vips)
    _GCS_READ_POOL.append(pool)
    return pool


def release_gcs_read_pool_mmaps() -> None:
  """Unmaps the active data arena across the singleton GCS read pool."""
  with _GCS_READ_POOL_LOCK:
    pool = _GCS_READ_POOL[0] if _GCS_READ_POOL else None
  if pool is not None:
    pool.unmap_data_arena()


def shutdown_gcs_read_pool() -> None:
  """Shuts down and cleans up the singleton GCS read pool at process exit."""
  with _GCS_READ_POOL_LOCK:
    while _GCS_READ_POOL:
      _GCS_READ_POOL.pop().shutdown()


atexit.register(shutdown_gcs_read_pool)


if __name__ == '__main__':
  if '--gcs-shm-worker' in sys.argv:
    run_gcs_shm_worker_cli()
  else:
    ProcPoolPath.worker_loop()
