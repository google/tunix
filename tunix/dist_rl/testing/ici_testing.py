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

"""Shared subprocess and JAX-Gloo helpers for ICI weight sync tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import inspect
import json
from multiprocessing import connection
import os
import socket
import subprocess
import sys  # pylint: disable=unused-import  # Used by OSS Copybara replacement.
import tempfile
import threading
import traceback
from typing import Any


DEFAULT_STARTUP_TIMEOUT_S = 120.0
DEFAULT_CALL_TIMEOUT_S = 120.0


def free_port() -> int:
  """Returns an unused local TCP port for test coordinators."""
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def gloo_available() -> bool:
  """Returns whether JAX supports CPU Gloo collectives in this environment."""
  try:
    # Lazy import avoids initializing JAX in parent test runner processes.
    import jax  # pylint: disable=g-import-not-at-top

    return hasattr(jax.config, "jax_cpu_collectives_implementation")
  except ImportError:
    return False


def clean_env() -> dict[str, str]:
  """Returns a copy of `os.environ` without proxy vars that hang Gloo."""
  return {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}


def worker_command(binary_name: str, test_file: str) -> list[str]:
  """Returns the argv prefix to launch a test worker subprocess.

  Args:
    binary_name: Bazel/Blaze `py_binary` target name (and `.py` stem in
      `test_data/`).
    test_file: `__file__` of the calling test module.

  Returns:
    Command-line argv prefix suitable for `subprocess.Popen`.
  """
  script = os.path.join(
      os.path.dirname(os.path.abspath(test_file)),
      "test_data",
      f"{binary_name}.py",
  )
  return [sys.executable, script]


def set_listener_timeout(
    listener: connection.Listener, timeout_s: float
) -> None:
  """Sets an accept timeout on a `multiprocessing.connection.Listener`."""
  # stdlib Listener does not expose a public timeout API.
  raw_sock = getattr(getattr(listener, "_listener", None), "_socket", None)
  if not isinstance(raw_sock, socket.socket):
    raise TypeError(
        f"cannot set an accept timeout on {listener!r}: it has no private"
        " socket to set it on"
    )
  raw_sock.settimeout(timeout_s)


def init_cpu_gloo_world(
    *, width: int, jax_port: int, processes: int, rank: int
) -> Any:
  """Sets CPU Gloo env vars, initializes `jax.distributed`, and returns jax."""
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={width}"
  os.environ["JAX_PLATFORMS"] = "cpu"
  # Lazy import so XLA_FLAGS and JAX_PLATFORMS take effect before JAX init.
  import jax  # pylint: disable=g-import-not-at-top

  jax.config.update("jax_cpu_collectives_implementation", "gloo")
  jax.distributed.initialize(
      coordinator_address=f"127.0.0.1:{jax_port}",
      num_processes=processes,
      process_id=rank,
  )
  # The backend only comes up once every process of the world has asked for it.
  jax.devices()
  return jax


class FakeSyncRequest:
  """Minimal `WeightSyncRequest` stand-in carrying version and round fields."""

  def __init__(self, fields: Mapping[str, Any]):
    self.policy_version = fields["policy_version"]
    self.extra_config = {
        "req_id": fields["req_id"],
        "uuid": fields["uuid"],
    }
    self.source_metadata = fields.get("source_metadata")


def serve_worker_ipc(control_port: int, rank: int, worker: Any) -> None:
  """Connects to the test controller and serves `(method, argument)` RPCs."""
  conn = connection.Client(("127.0.0.1", control_port))
  conn.send(rank)
  while True:
    try:
      method, argument = conn.recv()
    except EOFError:
      return
    if method == "exit":
      return
    try:
      result = getattr(worker, method)(argument)
      if inspect.isawaitable(result):
        result = asyncio.run(result)
      conn.send(("ok", result))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      conn.send(("error", f"{type(exc).__name__}: {exc}"))


class ProcHandle:
  """Stand-in for a worker `ActorHandle` backed by a `Connection`."""

  def __init__(
      self,
      rank: int,
      conn: connection.Connection,
      *,
      call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
  ):
    """Initializes the handle for worker `rank` over `conn`."""
    self.rank = rank
    self._conn = conn
    self._call_timeout_s = float(call_timeout_s)
    self._lock = threading.Lock()

  def submit(self, method: str, argument: Any = None) -> Any:
    """Sends `(method, argument)` and blocks until the worker replies."""
    with self._lock:
      self._conn.send((method, argument))
      if not self._conn.poll(self._call_timeout_s):
        raise TimeoutError(f"worker {self.rank} did not answer {method}")
      status, payload = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"worker {self.rank} {method} failed: {payload}")
    return payload

  def close(self) -> None:
    """Closes the underlying connection."""
    self._conn.close()


def call_concurrent(
    handles: Sequence[ProcHandle],
    method: str,
    argument: Any = None,
) -> list[Any]:
  """Invokes `method` concurrently on every handle and returns all results.

  Each `ProcHandle.submit` enforces its own call timeout.
  """
  with futures.ThreadPoolExecutor(max_workers=len(handles)) as pool:
    futs = [pool.submit(h.submit, method, argument) for h in handles]
    return [f.result() for f in futs]


class WorkerCluster:
  """Context manager spawning a multi-process CPU Gloo worker cluster."""

  def __init__(
      self,
      binary_name: str,
      test_file: str,
      configs_fn: Callable[[int, int], Sequence[Mapping[str, Any]]],
      *,
      startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
      call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
  ):
    jax_port = free_port()
    listener = connection.Listener(("127.0.0.1", 0))
    control_port = int(listener.address[1])
    configs = list(configs_fn(jax_port, control_port))
    env = clean_env()
    cmd_prefix = worker_command(binary_name, test_file)
    self._procs: list[subprocess.Popen[Any]] = []
    self._logs: list[Any] = []
    self.handles: dict[int, ProcHandle] = {}
    for cfg in configs:
      log = tempfile.TemporaryFile(mode="w+")
      self._logs.append(log)
      self._procs.append(
          subprocess.Popen(
              [*cmd_prefix, json.dumps(dict(cfg))],
              stdout=log,
              stderr=subprocess.STDOUT,
              env=env,
          )
      )
    set_listener_timeout(listener, startup_timeout_s)
    try:
      for _ in configs:
        conn = listener.accept()
        rank = int(conn.recv())
        self.handles[rank] = ProcHandle(
            rank, conn, call_timeout_s=call_timeout_s
        )
    except Exception as exc:
      logs_text = self.worker_logs()
      self.close()
      raise RuntimeError(
          "ICI test workers did not start:\n" + logs_text
      ) from exc
    finally:
      listener.close()

  def worker_logs(self, tail: int = 1500) -> str:
    """Returns the last `tail` characters of each worker's stdout/stderr log."""
    out = []
    for rank, log in enumerate(self._logs):
      if log.closed:
        continue
      log.seek(0)
      out.append(f"--- worker {rank} ---\n{log.read()[-tail:]}")
    return "\n".join(out)

  def close(self) -> None:
    """Closes connections, terminates subprocesses, and closes log files."""
    for handle in self.handles.values():
      handle.close()
    for proc in self._procs:
      if proc.poll() is None:
        proc.kill()
    for proc in self._procs:
      proc.wait()
    for log in self._logs:
      if not log.closed:
        log.close()

  def __enter__(self) -> WorkerCluster:
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()
