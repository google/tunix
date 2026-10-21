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

"""Structured run events (v1): one append-only JSONL stream per process.

Off unless `TUNIX_EVENTS` is `on` or `full`. When off, every emit site costs one
module-attribute read (`events.ON`) and nothing else; importing this module does
no I/O and starts no thread.

When on, `emit()` appends one item (its kwargs dict plus the envelope times) to
a bounded in-memory deque and returns.
One daemon writer thread per process serializes the tuples to JSON, appends them
to a local file every `FLUSH_INTERVAL_S`, and copies the file to
`<gcs_dir>/<role>-<pod>.jsonl` at each checkpoint save (`flush(upload=True)`),
every `UPLOAD_INTERVAL_S`, and at exit (atexit and SIGTERM). Nothing in this
module raises into its caller: exceptions are counted (`health()["errors"]`)
and each kind is logged once.

Twinned sites call `log_and_emit()`, which writes the site's existing text line
(same logger, level, format and arguments, and the same caller file/line in the
log prefix) and, when on, emits the event from the same values.

Envelope of every line: `v` (schema version), `t` (time.time()), `m`
(time.monotonic() in the emitting process), `ev`, `run`, `role`, `pod`, then the
event's fields.
"""

from __future__ import annotations

import atexit
import collections
import json
import logging as _std_logging
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Public state. `ON` is the off-mode guard: read it as `events.ON` (a module
# attribute), never `from ... import ON`, which would capture a stale value.
ON: bool = False

V = 1
ENV_LEVEL = "TUNIX_EVENTS"
LEVELS = ("off", "on", "full")
DEFAULT_QUEUE_SIZE = 65536
FLUSH_INTERVAL_S = 2.0
UPLOAD_INTERVAL_S = 300.0
HEALTH_INTERVAL_S = 60.0
# The writer encodes queued events every ENCODE_INTERVAL_S in chunks of
# ENCODE_CHUNK, sleeping YIELD_S between chunks so it never holds the GIL for
# long: one 2 s batch encoded at once added ~5 ms (one GIL switch interval) to
# the asyncio loop's p99 lag at 5,000 events/s (M, test step 2). The file write
# stays at FLUSH_INTERVAL_S.
ENCODE_INTERVAL_S = 0.1
ENCODE_CHUNK = 64
YIELD_S = 0.0002
EXIT_TIMEOUT_S = 20.0
# A node (trainer, rollout) that replaces the SIGTERM handler with its own calls
# node_stopping() before its drain and node_stopped() after it; each waits at
# most this long for the writer (TUNIX_EVENTS_NODE_STOP_TIMEOUT_S overrides).
NODE_STOP_TIMEOUT_S = 10.0
# The orchestrator's manifest: one listing, then the per-stream health sidecars
# read in parallel, all within MANIFEST_BUDGET_S (a partial manifest is written
# first, so an exit that cuts this short still leaves one that says so).
MANIFEST_BUDGET_S = 12.0
MANIFEST_READERS = 32
HEALTH_DIR = "_health"  # <gcs_dir>/_health/<role>-<pod>.json, per stream
MAX_EVENT_BYTES = 256 * 1024
REASON_MAX_CHARS = 200
ENVELOPE = ("v", "t", "m", "ev", "run", "role", "pod")
MANIFEST_NAME = "manifest.json"

_KT, _KM, _KE = " t", " m", " ev"  # envelope keys inside a queued item
_LOG = _std_logging.getLogger(__name__)
_FULL: bool = False
_state: "_State | None" = None
_strict: bool = False

# A twinned site's log record must name the site, not this file, in its
# `file:line` prefix. absl's logger skips registered frames; stdlib loggers get
# `stacklevel=2` (see `log_and_emit`).
try:  # absl is already imported by every caller; no I/O here.
  from absl import logging as _absl_logging  # pylint: disable=g-import-not-at-top

  _ABSL_LOGGER_CLS = _absl_logging.ABSLLogger
  _ABSL_LOGGER_CLS.register_frame_to_skip(__file__, "log_and_emit")
except Exception:  # pylint: disable=broad-except
  _absl_logging = None
  _ABSL_LOGGER_CLS = None

# Level constant -> the logging function a site calls. absl and stdlib levels
# differ (absl INFO is 0, stdlib INFO is 20); each table is keyed by its own.
_STD_LEVEL_FN = {_std_logging.DEBUG: "debug", _std_logging.INFO: "info",
                 _std_logging.WARNING: "warning", _std_logging.ERROR: "error",
                 _std_logging.CRITICAL: "critical"}
_ABSL_LEVEL_FN = {}
if _absl_logging is not None:
  _ABSL_LEVEL_FN = {_absl_logging.DEBUG: "debug", _absl_logging.INFO: "info",
                    _absl_logging.WARNING: "warning",
                    _absl_logging.ERROR: "error"}


# ---------------------------------------------------------------------------
# Schema registry.
#
# Field kinds: "int", "float", "str", "bool", "list", "obj", "repr".
# The writer thread coerces "str" (str(v), as `%s` prints it), "repr"
# (repr(v)[:200], as `%r` prints it, trimmed), "float" (int -> float) and "obj"
# (any value made JSON-safe with str() at the leaves). Nothing is coerced on the
# caller's thread.


class SchemaError(Exception):
  """A strict-mode schema violation (tests only; never raised in production)."""


class Schema:
  __slots__ = ("ev", "required", "optional", "full_only")

  def __init__(self, ev: str, required: dict[str, str],
               optional: dict[str, str] | None = None, full_only: bool = False):
    self.ev = ev
    self.required = dict(required)
    self.optional = dict(optional or {})
    self.full_only = full_only

  def kind(self, name: str) -> str | None:
    return self.required.get(name) or self.optional.get(name)


SCHEMAS: dict[str, Schema] = {}
FULL_ONLY: set[str] = set()
_KINDS = ("int", "float", "str", "bool", "list", "obj", "repr")
_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def register(ev: str, required: dict[str, str],
             optional: dict[str, str] | None = None,
             full_only: bool = False) -> Schema:
  """Registers (or replaces) the schema of event type `ev`."""
  for name, kind in {**required, **(optional or {})}.items():
    if kind not in _KINDS:
      raise ValueError(f"{ev}.{name}: unknown kind {kind!r}")
    if not _SNAKE.match(name) or name in ENVELOPE:
      raise ValueError(f"{ev}.{name}: not a snake_case non-envelope name")
    if name.endswith("_s") and kind != "float":
      raise ValueError(f"{ev}.{name}: a *_s field is a float duration")
  s = Schema(ev, required, optional, full_only)
  SCHEMAS[ev] = s
  if full_only:
    FULL_ONLY.add(ev)
  else:
    FULL_ONLY.discard(ev)
  return s


_FT = {"action": "str", "worker": "str", "req_id": "str", "retry": "int",
       "reason": "repr"}

register("run_start", {"mllog_time_ms": "int"})
register("run_stop", {"mllog_time_ms": "int"}, {"status": "str"})
register("config", {"pid": "int"},
         {"argv": "list", "env": "obj", "level": "str", "python": "str"})
register("stagger_hold", {"batch": "int", "held_s": "float"},
         {"first_s": "float", "rest_s": "float"})
register("dispatch", {"batch": "int", "prompt_idx": "int"},
         {"prompt_id": "str", "intra_batch_idx": "int", "policy_version": "int",
          "next_batch": "int", "req_ids": "list", "worker": "str",
          "submit_s": "float"})
register("arrive", {"batch": "int"},
         {"prompt_id": "str", "prompt_idx": "int", "intra_batch_idx": "int",
          "policy_version": "int", "worker": "str", "group_id": "str",
          "n_traj": "int", "dur_s": "float"})
register("pack", {"n_traj": "int"},
         {"step": "int", "mb_idx": "int", "traj_ids": "str",
          "padding_ratio": "float", "row_imbalance": "float", "rows": "int",
          "tokens": "int", "batches": "list"})
register("mb_stage", {"stage_key": "str"},
         {"step": "int", "mb_idx": "int", "bytes": "int", "stage_seq": "int"})
register("mb_send", {"stage_key": "str"},
         {"step": "int", "mb_idx": "int", "bytes": "int", "stage_seq": "int",
          "stage_cap": "int"})
register("mb_ack", {"stage_key": "str", "rpc_s": "float"},
         {"staged": "bool", "stage_seq": "int", "refused": "str",
          "step": "int", "mb_idx": "int"})
register("trainer_mb", {"mb_idx": "int"},
         {"step": "int", "stage_key": "str", "dispatch_begin": "float",
          "dispatch_end": "float", "ready": "float", "via": "str"})
register("trainer_step", {"step": "int"},
         {"policy_version": "int", "n_mb": "int", "update_s": "float",
          "rtt_s": "float", "complete": "bool", "update_present": "str"})
register("trainer_step_incomplete", {"step": "int"},
         {"n_mb": "int", "update_present": "str"})
register("trainer_timer_warning", {"kind": "str"},
         {"trainer": "str", "token": "str", "recorded": "int", "ran": "int",
          "missed": "str", "via": "str"})
register("wsync_phase", {"round": "int", "req_id": "str"},
         {"phase": "str", "elapsed_s": "float", "timeout": "str",
          "workers": "int", "units": "int", "failures": "int",
          "success": "bool", "outcome": "str", "note": "str", "n_cached": "int",
          "n_dest": "int", "uuid": "int", "unsafe_units": "str"})
register("wsync_round", {"round": "int", "req_id": "str"},
         {"policy_version": "int", "state": "str", "bind_s": "float",
          "metadata_s": "float", "source_prepare_s": "float",
          "register_s": "float", "pre_s": "float", "transfer_s": "float",
          "h2d_s": "float", "post_s": "float", "abort_s": "float",
          "release_s": "float", "prepare_write_s": "float",
          "h2d_read_s": "float", "post_verify_s": "float", "e2e_s": "float",
          "n_dest": "int", "plan_warmup": "int", "plan_s": "float",
          "plan_cache_hit": "int", "plan_cache_size": "int",
          "orch_rss_gib": "float", "orch_rss_peak_gib": "float",
          "dest_set_hash": "str", "dest_changed": "bool"})
register("wsync_timings", {"round": "int", "req_id": "str"},
         {"policy_version": "int", "state": "str", "bind_s": "float",
          "bind_timeout": "str", "metadata_s": "float",
          "metadata_timeout": "str", "source_prepare_s": "float",
          "source_prepare_timeout": "str", "register_s": "float",
          "pre_s": "float", "pre_timeout": "str", "transfer_s": "float",
          "transfer_timeout": "str", "h2d_s": "float", "h2d_timeout": "str",
          "post_s": "float", "post_timeout": "str", "abort_s": "float",
          "abort_timeout": "str", "release_s": "float",
          "release_timeout": "str", "e2e_s": "float"})
register("engine_pause", {}, {"round": "int", "policy_version": "int",
                              "mode": "str"})
register("engine_unpause", {}, {"round": "int", "policy_version": "int"})
register("worker_ft", {"action": "str"},
         {**_FT, "active_workers": "int", "incarnation": "obj", "state": "str",
          "retry_max": "int", "prompt_id": "str", "group_index": "int",
          "policy_version": "int", "requests": "int", "actor_busy_count": "int",
          "actor_weights_dirty": "str", "weights_consistent": "str",
          "max_zero_worker_wait_s": "float", "error": "repr", "evicted": "obj",
          "expired_requests": "obj", "remaining_in_flight": "int",
          "task_timeout_s": "float", "stage": "str", "counts": "obj",
          "minimum": "obj", "full": "obj", "grace_s": "float",
          "registered": "obj"})
register("turn", {"traj": "str", "turn": "int"},
         {"model_s": "float", "env_s": "float", "total_s": "float",
          "prompt_tokens": "int", "completion_tokens": "int",
          "preemptions": "int", "tps": "float", "tpot_ms": "float",
          "live": "int"}, full_only=True)
register("sampler_call", {"requests": "int", "dur_s": "float"},
         {"req_id": "str", "prompt_tokens": "int", "completion_tokens": "int",
          "preemptions": "int", "tps": "float", "tpot_ms": "float",
          "inflight": "int"}, full_only=True)
register("trajectory", {"traj": "str"},
         {"policy_version": "obj", "status": "str", "reward": "obj",
          "turns": "int", "wall_s": "float", "model_s": "float",
          "env_s": "float", "reward_s": "float", "mean_step_s": "float",
          "completion_tokens": "int", "preemptions": "int", "tps": "float",
          "tpot_ms": "float"}, full_only=True)
register("engine_stats", {"running": "int", "waiting": "int"},
         {"kv_usage": "float", "prefix_hit": "float", "prompt_tok_s": "float",
          "gen_tok_s": "float", "engine_idx": "int", "interval_s": "float"})
register("ckpt", {"step": "int"}, {"path": "str", "manifest_ts_ms": "int"})
register("events_health",
         {"emitted": "int", "dropped": "int", "errors": "int",
          "queue_hwm": "int"},
         {"written": "int", "lost": "int", "bytes": "int", "final": "bool"})


# ---------------------------------------------------------------------------
# Secrets: the `config` event carries env keys from an explicit allowlist (the
# primary control); as a backstop, any key whose name looks secret is dropped,
# and the values of all secret-named env vars are scrubbed from argv.

CONFIG_ENV_ALLOWLIST = (
    "JOB_PREFIX", "HOSTNAME", "POD_NAME", "JOBSET_NAME", "TUNIX_EVENTS",
    "TUNIX_EVENTS_GCS_DIR", "TUNIX_EVENTS_POD",
    "TUNIX_STAGE_BATCH", "TUNIX_WSYNC_PLAN_WARMUP", "TUNIX_WSYNC_PER_ENGINE",
    "TUNIX_DEBUG_INFERENCE_LOGS", "IN_FLIGHT_WEIGHT_UPDATES",
    "WEIGHT_SYNC_PARALLEL_H2H", "WEIGHT_SYNC_DISABLE_TIMEOUTS",
    "RAIDEN_BROADCAST_HOST_RATIO", "RAIDEN_BROADCAST_PIPELINE_STAGES",
    "STAGE_DEPTH", "STAGE_MAX_IN_FLIGHT", "STAGE_MAX_HOST_BYTES",
    "MIN_ROLLOUT_WORKERS", "ROLLOUT_WAIT_GRACE_S", "VERIFY_WEIGHTS",
    "REPLICATED_JOB_NAME", "JOB_INDEX", "POD_INDEX", "JAX_PLATFORMS",
    "VLLM_LOG_STATS_INTERVAL", "EMULATE", "EMU_TIME_SCALE",
)
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)


def _secret_values() -> list[str]:
  return [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and v
          and len(v) >= 4]


def _scrub(s: str, secrets: list[str]) -> str:
  for v in secrets:
    if v in s:
      s = s.replace(v, "<redacted>")
  return s


def config_fields(argv: list[str] | None = None,
                  allowlist: tuple[str, ...] = CONFIG_ENV_ALLOWLIST) -> dict:
  """Fields of the `config` event: argv and allowlisted env, secrets removed."""
  secrets = _secret_values()
  env = {}
  for k in allowlist:
    if _SECRET_NAME.search(k):  # backstop: an allowlisted secret name
      continue
    if k in os.environ:
      env[k] = _scrub(os.environ[k], secrets)
  args = []
  for a in (sys.argv if argv is None else argv):
    a = str(a)
    flag = a.split("=", 1)[0]
    if flag.startswith("-") and _SECRET_NAME.search(flag):
      args.append(flag + "=<redacted>" if "=" in a else a)
      continue
    args.append(_scrub(a, secrets))
  return {"pid": os.getpid(), "argv": args, "env": env,
          "python": sys.version.split()[0]}


# ---------------------------------------------------------------------------
# Hot path.


def enabled(full: bool = False) -> bool:
  """Whether events are on (and, with `full=True`, at the `full` level)."""
  return ON and (_FULL or not full)


def emit(ev: str, /, **fields: Any) -> None:
  """Queues one event. Never blocks, never raises; a no-op when off."""
  if not ON:
    return
  try:
    st = _state
    if ev in FULL_ONLY and not _FULL:
      return
    if _strict:
      _validate_strict(ev, fields)
    q = st.q
    n = len(q)
    if n >= st.cap:
      st.count_drop()
      return
    # The queued item is the call's own kwargs dict with the envelope added
    # (keys that no keyword argument can collide with). A dict of atomic values
    # is not tracked by the cyclic GC, where a tuple always is: 65,536 queued
    # tuples triggered full collections of 4-6 ms inside emit() (M, test 1.3).
    fields[_KT] = time.time()
    fields[_KM] = time.monotonic()
    fields[_KE] = ev
    q.append(fields)
    if n >= st.hwm:
      st.hwm = n + 1
      if n + 1 >= st.wake_at:
        st.wake.set()
  except SchemaError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    _count_error("emit", e)


def log_and_emit(logger: Any, level: int, fmt: str, ev: str, /, *,
                 text_args: tuple = (), names: tuple = (),
                 **fields: Any) -> None:
  """Writes a site's text line exactly as before, then emits its event.

  `logger.log(level, fmt, *text_args)` is the site's current statement (pass
  the level constant of the logging module the site uses: absl and stdlib
  levels differ). The record keeps the site's file, line and function. When on,
  the event's fields are `zip(names, text_args)` (a `None` name skips that
  argument) plus `**fields`; when off nothing beyond the log call runs.
  """
  # The site's own call (`logging.info(...)`, `logger.warning(...)`), so the
  # record and anything that patches that function see what they saw before.
  if logger is _absl_logging or (_ABSL_LOGGER_CLS is not None and isinstance(
      logger, _ABSL_LOGGER_CLS)):
    fn = getattr(logger, _ABSL_LEVEL_FN.get(level, ""), None)
    if fn is None:
      logger.log(level, fmt, *text_args)  # absl skips this frame (registered)
    else:
      fn(fmt, *text_args)
  else:  # a stdlib Logger, or the stdlib `logging` module itself
    fn = getattr(logger, _STD_LEVEL_FN.get(level, ""), None)
    if fn is None:
      logger.log(level, fmt, *text_args, stacklevel=2)
    else:
      fn(fmt, *text_args, stacklevel=2)
  if ON:
    try:
      if names:
        for name, value in zip(names, text_args):
          if name is not None:
            fields[name] = value
      emit(ev, **fields)
    except SchemaError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      _count_error("emit", e)


def flush(upload: bool = False, timeout: float | None = None) -> None:
  """Asks the writer thread to write now (and upload when `upload`).

  Returns at once unless `timeout` is given, in which case it waits up to
  `timeout` seconds for that work to finish. The work runs on the writer thread.
  """
  if not ON:
    return
  try:
    st = _state
    if st is not None:
      st.request(upload=upload, timeout=timeout)
  except Exception as e:  # pylint: disable=broad-except
    _count_error("flush", e)


def health() -> dict:
  """Counters of this process's stream (all zero when never configured)."""
  st = _state
  if st is None:
    return {"emitted": 0, "dropped": 0, "errors": _orphan_errors[0],
            "queue_hwm": 0, "written": 0, "lost": 0, "bytes": 0}
  return st.health()


def set_expected(**role_counts: int | None) -> None:
  """Records how many files per role the orchestrator's manifest expects.

  A count already set from `TUNIX_EVENTS_EXPECTED` (e.g. an emulation that
  packs several engines into one process) is kept.
  """
  st = _state
  if st is not None:
    for k, v in role_counts.items():
      st.expected.setdefault(k, v)


# ---------------------------------------------------------------------------
# Errors: counted, each kind logged once.

_err_lock = threading.Lock()
_orphan_errors = [0]  # errors before configure()
_logged_kinds: set[str] = set()


def _count_error(kind: str, exc: BaseException | None = None) -> None:
  try:
    with _err_lock:
      st = _state
      if st is not None:
        st.errors += 1
      else:
        _orphan_errors[0] += 1
      first = kind not in _logged_kinds
      _logged_kinds.add(kind)
    if first:
      _LOG.warning("tunix events: %s error (further %s errors are counted,"
                   " not logged): %r", kind, kind, exc)
  except Exception:  # pylint: disable=broad-except
    pass


def _validate_strict(ev: str, fields: dict) -> None:
  s = SCHEMAS.get(ev)
  if s is None:
    raise SchemaError(f"unknown event {ev!r}")
  for name in s.required:
    if name not in fields:
      raise SchemaError(f"{ev}: missing required field {name!r}")
  for name, value in fields.items():
    if not _SNAKE.match(name) or name in ENVELOPE:
      raise SchemaError(f"{ev}: bad field name {name!r}")
    kind = s.kind(name)
    if kind is None:
      raise SchemaError(f"{ev}: field {name!r} not in the schema")
    if value is None:
      continue
    if not _kind_ok(kind, value):
      raise SchemaError(f"{ev}.{name}: {type(value).__name__} is not {kind}")


def _kind_ok(kind: str, v: Any) -> bool:
  if kind == "int":
    return isinstance(v, int) and not isinstance(v, bool)
  if kind == "float":
    return isinstance(v, (int, float)) and not isinstance(v, bool)
  if kind == "bool":
    return isinstance(v, bool)
  if kind == "list":
    return isinstance(v, (list, tuple))
  return True  # str / repr / obj are coerced


def _jsonable(v: Any, depth: int = 0) -> Any:
  if v is None or isinstance(v, (bool, int, float, str)):
    return v
  if depth > 8:
    return str(v)
  if isinstance(v, dict):
    return {str(getattr(k, "value", k)): _jsonable(x, depth + 1)
            for k, x in v.items()}
  if isinstance(v, (list, tuple, set, frozenset)):
    return [_jsonable(x, depth + 1) for x in v]
  return str(v)


def _coerce(kind: str, v: Any) -> Any:
  if v is None:
    return None
  if kind == "str":
    return v if isinstance(v, str) else str(v)
  if kind == "repr":
    return repr(v)[:REASON_MAX_CHARS]
  if kind == "float":
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v
  if kind == "obj":
    return _jsonable(v)
  if kind == "list" and isinstance(v, tuple):
    return list(v)
  return v


# ---------------------------------------------------------------------------
# Storage: the local file, the upload, the manifest.


def _gcs_split(uri: str) -> tuple[str, str]:
  rest = uri[len("gs://"):]
  bucket, _, path = rest.partition("/")
  return bucket, path


def _default_upload(local_path: str, dest_uri: str) -> None:
  if not dest_uri.startswith("gs://"):
    os.makedirs(os.path.dirname(dest_uri) or ".", exist_ok=True)
    tmp = dest_uri + ".tmp"
    with open(local_path, "rb") as src, open(tmp, "wb") as dst:
      dst.write(src.read())
    os.replace(tmp, dest_uri)
    return
  bucket, path = _gcs_split(dest_uri)
  _gcs_client().bucket(bucket).blob(path).upload_from_filename(local_path)


_gcs_client_cache: dict[str, Any] = {}
# Tests patch these. Each takes (local_path, dest_uri) / (dir_uri).
_upload_fn: Callable[[str, str], None] = _default_upload


def _default_list_tails(dir_uri: str) -> dict[str, tuple[int, bytes]]:
  """{basename: (n_event_lines or -1, last ~64 KiB)} of *.jsonl under dir_uri.

  n_event_lines counts complete lines other than `events_health`; -1 when not
  counted (GCS, where only the tail is read).
  """
  out: dict[str, tuple[int, bytes]] = {}
  if not dir_uri.startswith("gs://"):
    if not os.path.isdir(dir_uri):
      return out
    for name in sorted(os.listdir(dir_uri)):
      if not name.endswith(".jsonl"):
        continue
      p = os.path.join(dir_uri, name)
      with open(p, "rb") as f:
        data = f.read()
      n_events = data.count(b"\n") - data.count(b'"ev":"events_health"')
      out[name] = (n_events, data[-65536:])
    return out
  bucket, path = _gcs_split(dir_uri.rstrip("/") + "/")
  blobs = [b for b in _gcs_client().list_blobs(bucket, prefix=path)
           if "/" not in b.name[len(path):] and b.name.endswith(".jsonl")]

  def tail_of(blob):
    size = blob.size or 0
    return blob.download_as_bytes(start=max(0, size - 65536)) if size else b""

  for blob, (ok, tail) in zip(blobs, _parallel(tail_of, blobs, None)):
    if not ok:
      raise tail
    out[blob.name[len(path):]] = (-1, tail)
  return out


def _parallel(fn: Callable[[Any], Any], items: list, timeout: float | None,
              workers: int = MANIFEST_READERS) -> list:
  """[(True, fn(x)) or (False, exception) or None (not done in time)].

  Daemon threads rather than concurrent.futures: the orchestrator builds its
  manifest from atexit, after concurrent.futures refuses new work, and a read
  that hangs must not hold the interpreter's exit.
  """
  results: list = [None] * len(items)
  nxt = [0]
  lock = threading.Lock()
  done = threading.Event()
  left = [len(items)]

  def work():
    while True:
      with lock:
        i = nxt[0]
        nxt[0] += 1
      if i >= len(items):
        return
      try:
        r = (True, fn(items[i]))
      except Exception as e:  # pylint: disable=broad-except
        r = (False, e)
      with lock:
        results[i] = r
        left[0] -= 1
        if left[0] == 0:
          done.set()
  if not items:
    return results
  for _ in range(min(workers, len(items))):
    threading.Thread(target=work, name="tunix-events-read", daemon=True).start()
  done.wait(timeout)
  with lock:
    return list(results)


def _gcs_client():
  from google.cloud import storage  # pylint: disable=g-import-not-at-top
  client = _gcs_client_cache.get("c")
  if client is None:
    client = _gcs_client_cache["c"] = storage.Client()
  return client


def _default_list_streams(dir_uri: str) -> dict[str, int]:
  """{basename: size in bytes} of the *.jsonl streams directly under dir_uri
  (one listing, no reads)."""
  if not dir_uri.startswith("gs://"):
    if not os.path.isdir(dir_uri):
      return {}
    return {n: os.path.getsize(os.path.join(dir_uri, n))
            for n in sorted(os.listdir(dir_uri)) if n.endswith(".jsonl")}
  bucket, path = _gcs_split(dir_uri.rstrip("/") + "/")
  out = {}
  for blob in _gcs_client().list_blobs(bucket, prefix=path):
    name = blob.name[len(path):]
    if "/" not in name and name.endswith(".jsonl"):
      out[name] = int(blob.size or 0)
  return out


def _default_read_health(dir_uri: str, name: str,
                         size: int | None) -> tuple[dict | None, str]:
  """(the last health record of stream `name`, where it came from).

  The writer's sidecar (`<dir>/_health/<role>-<pod>.json`) when there is one,
  else the last `events_health` line in the stream's last 64 KiB (a writer that
  never uploaded a sidecar), else (None, "absent").
  """
  rel = f"{HEALTH_DIR}/{name[:-len('.jsonl')]}.json"
  if not dir_uri.startswith("gs://"):
    p = os.path.join(dir_uri, rel)
    if os.path.exists(p):
      with open(p, encoding="utf-8") as f:
        return json.load(f), "sidecar"
    sp = os.path.join(dir_uri, name)
    if not os.path.exists(sp):
      return None, "absent"
    with open(sp, "rb") as f:
      f.seek(max(0, os.path.getsize(sp) - 65536))
      h = _last_health(f.read())
    return h, ("tail" if h is not None else "absent")
  bucket, path = _gcs_split(dir_uri.rstrip("/") + "/" + rel)
  b = _gcs_client().bucket(bucket)
  try:
    return json.loads(b.blob(path).download_as_bytes()), "sidecar"
  except Exception as e:  # pylint: disable=broad-except
    if type(e).__name__ not in ("NotFound", "KeyError"):
      raise
  _, spath = _gcs_split(dir_uri.rstrip("/") + "/" + name)
  if not size:
    return None, "absent"
  h = _last_health(b.blob(spath).download_as_bytes(start=max(0, size - 65536)))
  return h, ("tail" if h is not None else "absent")


_list_tails_fn: Callable[[str], dict[str, tuple[int, bytes]]] = \
    _default_list_tails
_list_streams_fn: Callable[[str], dict[str, int]] = _default_list_streams
_read_health_fn: Callable[[str, str, int | None],
                          tuple[dict | None, str]] = _default_read_health


def _last_health(tail: bytes) -> dict | None:
  for line in reversed(tail.splitlines()):
    try:
      rec = json.loads(line)
    except Exception:  # pylint: disable=broad-except
      continue
    if rec.get("ev") == "events_health":
      return rec
  return None


def _file_entry(h: dict | None, size: int | None, src: str) -> dict:
  if h is None:
    return {"v": V, "events": None, "dropped": None, "errors": None,
            "final": False, "lines": -1, "size": size, "health": src}
  return {
      "v": h.get("v", V), "role": h.get("role"), "pod": h.get("pod"),
      "events": int(h["emitted"]) - int(h["dropped"]) - int(h.get("lost", 0)),
      "dropped": int(h["dropped"]), "errors": int(h["errors"]),
      "final": bool(h.get("final")), "lines": -1, "size": size,
      "health": src,
  }


def build_manifest(dir_uri: str, run: str, expected: dict,
                   own: dict[str, dict] | None = None,
                   budget_s: float | None = None,
                   on_partial: Callable[[dict], None] | None = None) -> dict:
  """The manifest of the stream files under `dir_uri`.

  One listing of the streams, then each stream's health sidecar (its last
  `events_health` record, uploaded by its writer after each stream upload),
  read in parallel within `budget_s`. `own` overrides with exact in-process
  counts. `on_partial(manifest)` is called with the listing-only manifest
  (`complete: false`) before any sidecar is read. A sidecar not read in time is
  listed under `unread` and the manifest says `complete: false`.
  """
  budget_s = MANIFEST_BUDGET_S if budget_s is None else budget_s
  t0 = time.monotonic()
  sizes = _list_streams_fn(dir_uri)
  for name in own or {}:
    sizes.setdefault(name, None)
  files: dict[str, dict] = {
      n: {"v": V, "events": None, "dropped": None, "errors": None,
          "final": False, "lines": -1, "size": sz, "health": "unread"}
      for n, sz in sizes.items()}
  for name, rec in (own or {}).items():
    files[name] = {**files[name], **rec, "health": "own"}

  def man(complete, unread, note=None):
    m = {"v": V, "run": run, "written_t": time.time(),
         "expected": dict(expected), "complete": complete,
         "listed": len(sizes), "unread": sorted(unread), "files": files}
    if note:
      m["note"] = note
    return m

  todo = [n for n in sizes if n not in (own or {})]
  if on_partial is not None:
    try:
      on_partial(man(False, todo, "partial: listing only, sidecars not read"))
    except Exception as e:  # pylint: disable=broad-except
      _count_error("manifest", e)
  if not todo:
    return man(True, [])
  left = max(0.0, budget_s - (time.monotonic() - t0))
  unread = set()
  for n, r in zip(todo, _parallel(
      lambda n: _read_health_fn(dir_uri, n, sizes[n]), todo, left)):
    if r is None:
      unread.add(n)
    elif r[0]:
      files[n] = _file_entry(r[1][0], sizes[n], r[1][1])
    else:
      _count_error("manifest_read", r[1])
      files[n]["health"] = f"error: {type(r[1]).__name__}"
  note = None
  if unread:
    note = (f"partial: {len(unread)} of {len(todo)} sidecars not read within"
            f" {budget_s:.1f} s")
  return man(not unread, unread, note)


def _read_manifest(dir_uri: str) -> dict:
  if dir_uri.startswith("gs://"):
    bucket, path = _gcs_split(dir_uri.rstrip("/") + "/" + MANIFEST_NAME)
    return json.loads(_gcs_client().bucket(bucket).blob(path).download_as_bytes())
  with open(os.path.join(dir_uri, MANIFEST_NAME), encoding="utf-8") as f:
    return json.load(f)


def verify_manifest(dir_uri: str) -> dict:
  """Checks the files under `dir_uri` against its manifest.

  Returns {"manifest": ..., "problems": [...]}: a role with fewer files than
  expected, a manifest file that is absent, a file with fewer event lines than
  the manifest counts or without a final `events_health` line (truncated), and
  a file with drops or errors.
  """
  man = _read_manifest(dir_uri)
  tails = _list_tails_fn(dir_uri)
  problems = []
  if man.get("complete") is False:
    problems.append(f"partial manifest: {man.get('note') or 'incomplete'}"
                    f" (unread: {len(man.get('unread') or [])})")
  for role, n in sorted(man.get("expected", {}).items()):
    found = sum(1 for name in tails if name.startswith(role + "-"))
    if n is not None and found < n:
      problems.append(f"missing: expected {n} {role} file(s), found {found}")
  for name, rec in sorted(man.get("files", {}).items()):
    if name not in tails:
      problems.append(f"missing: {name} (in the manifest, no file)")
      continue
    n_lines, tail = tails[name]
    want = rec.get("events")
    h = _last_health(tail)
    if want is not None and n_lines >= 0 and n_lines < want:
      problems.append(f"truncated: {name} has {n_lines} event lines, the"
                      f" manifest counts {want}")
    elif h is None or not h.get("final") or not tail.endswith(b"\n"):
      problems.append(f"truncated: {name} has no final events_health line")
    if rec.get("dropped"):
      problems.append(f"dropped: {name} dropped {rec['dropped']}")
    if rec.get("errors"):
      problems.append(f"errors: {name} counted {rec['errors']}")
  return {"manifest": man, "problems": problems}


# ---------------------------------------------------------------------------
# The writer.


class _State:
  """Per-process stream state; `q` is filled by emit(), drained by the writer."""

  def __init__(self, run: str, role: str, pod: str, gcs_dir: str | None,
               local_dir: str, cap: int, flush_s: float, upload_s: float,
               health_s: float, clock: Callable[[], float]):
    self.run, self.role, self.pod = run, role, pod
    self.gcs_dir = gcs_dir.rstrip("/") if gcs_dir else None
    self.name = f"{role}-{pod}.jsonl"
    self.local_dir = local_dir
    self.local_path = os.path.join(local_dir, self.name)
    self.cap = cap
    self.wake_at = max(1, cap // 2)
    self.q: collections.deque = collections.deque()
    self.hwm = 0
    self.dropped = 0
    self.errors = 0
    self.popped = 0
    self.written = 0
    self.lost = 0
    self.bytes = 0
    self.flush_s, self.upload_s, self.health_s = flush_s, upload_s, health_s
    self.clock = clock
    self.expected: dict[str, int | None] = {}
    self.wake = threading.Event()
    self.lock = threading.Lock()  # guards dropped and the request fields
    self.req_upload = False
    self.req_stop = False
    self.req_gen = 0
    self.done_gen = 0
    self.done_cv = threading.Condition(self.lock)
    self.stopped = threading.Event()
    self.thread: threading.Thread | None = None
    self.pid = os.getpid()
    self._fh = None
    self.buf: list[str] = []
    self.encode_s = ENCODE_INTERVAL_S
    self.chunk = ENCODE_CHUNK
    self.yield_s = YIELD_S
    now = clock()
    self.next_encode = now + self.encode_s
    self.next_flush = now + flush_s
    self.next_upload = now + upload_s
    self.next_health = now + health_s

  # -- caller side ---------------------------------------------------------
  def count_drop(self) -> None:
    with self.lock:
      self.dropped += 1

  def request(self, upload: bool, timeout: float | None,
              stop: bool = False) -> bool:
    with self.lock:
      self.req_gen += 1
      gen = self.req_gen
      self.req_upload = self.req_upload or upload
      self.req_stop = self.req_stop or stop
    self.wake.set()
    if timeout is None or self.thread is None:
      return False
    if threading.current_thread() is self.thread:
      return False
    deadline = time.monotonic() + timeout
    with self.done_cv:
      while self.done_gen < gen and not self.stopped.is_set():
        left = deadline - time.monotonic()
        if left <= 0:
          return False
        self.done_cv.wait(left)
    return True

  def health(self) -> dict:
    return {"emitted": self.popped + len(self.q) + self.dropped,
            "dropped": self.dropped, "errors": self.errors,
            "queue_hwm": self.hwm, "written": self.written, "lost": self.lost,
            "bytes": self.bytes}

  # -- writer side -----------------------------------------------------------
  def start(self) -> None:
    self.thread = threading.Thread(target=self._run, name="tunix-events-writer",
                                   daemon=True)
    self.thread.start()

  def _run(self) -> None:
    while True:
      try:
        nxt = min(self.next_flush, self.next_upload, self.next_health)
        if self.q:
          nxt = min(nxt, self.next_encode)
        timeout = max(0.0, nxt - self.clock())
        self.wake.wait(timeout)
        self.wake.clear()
        if self.tick(self.clock()):
          return
      except Exception as e:  # pylint: disable=broad-except
        _count_error("writer", e)
        if self.req_stop:
          self.stopped.set()
          return
        time.sleep(0.5)

  def tick(self, now: float) -> bool:
    """One writer pass at time `now`; returns True once stopped."""
    with self.lock:
      gen = self.req_gen
      upload = self.req_upload
      stop = self.req_stop
      self.req_upload = False
    requested = gen != self.done_gen
    try:
      if now >= self.next_encode or len(self.q) >= self.wake_at:
        self.next_encode = now + self.encode_s
        self.encode_pending()
      if requested or now >= self.next_flush:
        self.next_flush = now + self.flush_s
        self.drain()
      if now >= self.next_health and not stop:
        self.next_health = now + self.health_s
        self.write_health(final=False)
      if stop:
        self._safely(self.drain)
        self._safely(lambda: self.write_health(final=True))
        self._safely(lambda: self.upload(final=True))
        self._safely(self.write_manifest)
      elif upload or now >= self.next_upload:
        self.next_upload = now + self.upload_s
        self.drain()
        self.upload()
    except Exception as e:  # pylint: disable=broad-except
      _count_error("writer", e)
    with self.done_cv:
      self.done_gen = max(self.done_gen, gen)
      self.done_cv.notify_all()
    if stop:
      self.close()
      self.stopped.set()
      with self.done_cv:
        self.done_cv.notify_all()
    return stop

  @staticmethod
  def _safely(fn: Callable[[], None]) -> None:
    try:
      fn()
    except Exception as e:  # pylint: disable=broad-except
      _count_error("writer", e)

  def _encode_item(self, item: dict) -> str:
    t = item.pop(_KT)
    m = item.pop(_KM)
    ev = item.pop(_KE)
    return self._encode(t, m, ev, item)

  def _encode(self, t: float, m: float, ev: str, fields: dict) -> str:
    rec: dict[str, Any] = {"v": V, "t": t, "m": m, "ev": ev, "run": self.run,
                           "role": self.role, "pod": self.pod}
    s = SCHEMAS.get(ev)
    bad = None
    if s is None:
      bad = "unknown_event"
    for k, v in fields.items():
      if k in rec:
        bad = bad or "envelope_collision"
        k = "f_" + k
      kind = s.kind(k) if s is not None else None
      if kind is not None:
        v = _coerce(kind, v)
      elif k.endswith("_s") and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
      rec[k] = v
    if s is not None and bad is None:
      for k in s.required:
        if k not in fields:
          bad = "schema"
          break
    try:
      line = json.dumps(rec, allow_nan=False, separators=(",", ":"))
      if len(line) > MAX_EVENT_BYTES:
        raise ValueError(f"event {ev} encodes to {len(line)} bytes")
    except Exception as e:  # pylint: disable=broad-except
      _count_error("encode", e)
      rec = {k: rec[k] for k in ENVELOPE}
      rec["encode_error"] = type(e).__name__
      return json.dumps(rec, separators=(",", ":")) + "\n"
    if bad is not None:
      _count_error(bad, ValueError(f"{bad}: {ev}"))
    return line + "\n"

  def encode_pending(self) -> None:
    """Encodes what is queued now into the write buffer, in short chunks."""
    q = self.q
    buf = self.buf
    # Bounded by the length at entry, so a steady stream of emits cannot keep
    # one pass (and the file write behind it) from finishing.
    left = len(q)
    while left > 0:
      for _ in range(min(left, self.chunk)):
        try:
          item = q.popleft()
        except IndexError:
          left = 0
          break
        self.popped += 1
        try:
          buf.append(self._encode_item(item))
        except Exception as e:  # pylint: disable=broad-except
          _count_error("encode", e)
          self.lost += 1
      left -= self.chunk
      if left > 0 and self.yield_s:
        time.sleep(self.yield_s)  # releases the GIL between chunks

  def drain(self) -> None:
    """Encodes everything queued and appends the buffer to the local file."""
    self.encode_pending()
    good, self.buf = self.buf, []
    if good:
      if self._write(good):
        self.written += len(good)
      else:
        self.lost += len(good)

  def _write(self, lines: list[str]) -> bool:
    try:
      if self._fh is None:
        os.makedirs(self.local_dir, exist_ok=True)
        self._fh = open(self.local_path, "a", encoding="ascii")
      # Lines are ASCII (json.dumps escapes the rest), so len() is the byte
      # count. Joined and written in slices to keep each GIL hold short.
      for i in range(0, len(lines), 1024):
        data = "".join(lines[i:i + 1024])
        self._fh.write(data)
        self.bytes += len(data)
      self._fh.flush()
      return True
    except Exception as e:  # pylint: disable=broad-except
      _count_error("io", e)
      try:
        if self._fh is not None:
          self._fh.close()
      except Exception:  # pylint: disable=broad-except
        pass
      self._fh = None
      return False

  def write_health(self, final: bool) -> None:
    h = self.health()
    fields = {"emitted": h["emitted"], "dropped": h["dropped"],
              "errors": h["errors"], "queue_hwm": h["queue_hwm"],
              "written": h["written"], "lost": h["lost"], "bytes": h["bytes"],
              "final": final}
    line = self._encode(time.time(), time.monotonic(), "events_health", fields)
    self._write([line])

  def upload(self, final: bool = False) -> None:
    """Uploads the stream, then its health sidecar (the sidecar is also kept
    in `<local_dir>/_health/`, which is the manifest's source without GCS)."""
    if not os.path.exists(self.local_path):
      return
    if self.gcs_dir:
      try:
        _upload_fn(self.local_path, f"{self.gcs_dir}/{self.name}")
      except Exception as e:  # pylint: disable=broad-except
        _count_error("upload", e)
        return
    # After the stream it describes: the orchestrator's manifest reads these
    # small files rather than ~160 stream tails.
    try:
      h = self.health()
      side = {"v": V, "run": self.run, "role": self.role, "pod": self.pod,
              "name": self.name, "t": time.time(), "final": final, **h}
      base = self.name[:-len(".jsonl")]
      hdir = os.path.join(self.local_dir, HEALTH_DIR)
      os.makedirs(hdir, exist_ok=True)
      local = os.path.join(hdir, f"{base}.json")
      with open(local + ".tmp", "w", encoding="utf-8") as f:
        json.dump(side, f, sort_keys=True)
      os.replace(local + ".tmp", local)
      if self.gcs_dir:
        _upload_fn(local, f"{self.gcs_dir}/{HEALTH_DIR}/{base}.json")
    except Exception as e:  # pylint: disable=broad-except
      _count_error("upload_health", e)

  def write_manifest(self) -> None:
    if self.role != "orch":
      return
    try:
      d = self.gcs_dir or self.local_dir
      h = self.health()
      own = {self.name: {"v": V, "role": self.role, "pod": self.pod,
                         "events": h["emitted"] - h["dropped"] - h["lost"],
                         "dropped": h["dropped"], "errors": h["errors"],
                         "final": True}}
      tmp = os.path.join(self.local_dir, MANIFEST_NAME)

      def put(man):
        with open(tmp + ".tmp", "w", encoding="utf-8") as f:
          json.dump(man, f, indent=1, sort_keys=True)
        os.replace(tmp + ".tmp", tmp)
        if self.gcs_dir:
          _upload_fn(tmp, f"{self.gcs_dir}/{MANIFEST_NAME}")

      budget = float(os.environ.get("TUNIX_EVENTS_MANIFEST_BUDGET_S",
                                    MANIFEST_BUDGET_S))
      put(build_manifest(d, self.run, self.expected, own, budget_s=budget,
                         on_partial=put))
    except Exception as e:  # pylint: disable=broad-except
      _count_error("manifest", e)

  def close(self) -> None:
    try:
      if self._fh is not None:
        self._fh.close()
    except Exception:  # pylint: disable=broad-except
      pass
    self._fh = None


# ---------------------------------------------------------------------------
# Configuration and shutdown.

_prev_sigterm: Any = None
_hooks_installed = False


def _parse_level(level: str | None) -> str:
  lv = (level if level is not None else os.environ.get(ENV_LEVEL, "")).strip()
  lv = lv.lower()
  if lv in ("", "0", "off", "false", "no"):
    return "off"
  if lv in ("1", "on", "true", "yes"):
    return "on"
  if lv == "full":
    return "full"
  return "invalid"


def configure(run: str, role: str, pod: str, gcs_dir: str | None,
              level: str | None = None, *, local_dir: str | None = None,
              queue_size: int | None = None,
              flush_interval_s: float = FLUSH_INTERVAL_S,
              upload_interval_s: float = UPLOAD_INTERVAL_S,
              health_interval_s: float = HEALTH_INTERVAL_S,
              start_thread: bool = True,
              clock: Callable[[], float] = time.monotonic,
              argv: list[str] | None = None) -> None:
  """Configures this process's stream once. Off (`level` off/unset): no-op."""
  global ON, _FULL, _state
  try:
    lv = _parse_level(level)
    if lv == "off":
      return
    if lv == "invalid":
      _count_error("configure", ValueError(
          f"{ENV_LEVEL} must be off, on or full, got {level!r}"))
      return
    if _state is not None:
      _count_error("configure", RuntimeError("configure() called twice; the"
                                             " first configuration is kept"))
      return
    if queue_size is None:
      queue_size = int(os.environ.get("TUNIX_EVENTS_QUEUE_SIZE",
                                      DEFAULT_QUEUE_SIZE))
    if local_dir is None:
      local_dir = os.environ.get("TUNIX_EVENTS_LOCAL_DIR", "/tmp/tunix_events")
    st = _State(run=run or "", role=role, pod=pod or "", gcs_dir=gcs_dir,
                local_dir=local_dir, cap=max(1, queue_size),
                flush_s=flush_interval_s, upload_s=upload_interval_s,
                health_s=health_interval_s, clock=clock)
    exp = os.environ.get("TUNIX_EVENTS_EXPECTED", "")
    for part in filter(None, (p.strip() for p in exp.split(","))):
      k, _, v = part.partition("=")
      st.expected[k] = int(v) if v.strip().lstrip("-").isdigit() else None
    _state = st
    _FULL = lv == "full"
    if start_thread:
      st.start()
    _install_hooks()
    ON = True
    emit("config", level=lv, **config_fields(argv))
  except Exception as e:  # pylint: disable=broad-except
    _count_error("configure", e)


_JOBSET_POD_ENV = ("JOBSET_NAME", "REPLICATED_JOB_NAME", "JOB_INDEX",
                   "POD_INDEX")


def pod_name_from_env(env: Any = None) -> str:
  """The stream's pod name: TUNIX_EVENTS_POD, else POD_NAME (downward API
  `metadata.name`), else `<jobset>-<replicated job>-<job index>-<pod index>`
  from the JobSet env the launcher already sets (the pod name without its
  random suffix), else HOSTNAME, else the host name. With hostNetwork,
  HOSTNAME is the node's name, so it is the last resort."""
  env = os.environ if env is None else env
  pod = (env.get("TUNIX_EVENTS_POD") or env.get("POD_NAME") or "").strip()
  if pod:
    return pod
  parts = [(env.get(k) or "").strip() for k in _JOBSET_POD_ENV]
  if all(parts):
    return "-".join(parts)
  pod = (env.get("HOSTNAME") or "").strip()
  if not pod:
    import socket  # pylint: disable=g-import-not-at-top
    pod = socket.gethostname()
  return pod


def configure_from_env(role: str, argv: list[str] | None = None) -> None:
  """configure() from the launcher env: JOB_PREFIX, the pod name
  (pod_name_from_env), TUNIX_EVENTS_GCS_DIR. Returns at once when
  `TUNIX_EVENTS` is off."""
  if _parse_level(None) == "off":
    return
  pod = pod_name_from_env()
  configure(run=os.environ.get("JOB_PREFIX", ""), role=role, pod=pod,
            gcs_dir=os.environ.get("TUNIX_EVENTS_GCS_DIR") or None,
            argv=argv)


def role_for_process_main(process_main: str) -> str:
  """orch / trainer / rollout from a runtime `--process_main` path."""
  p = process_main.lower()
  if "deepswe_dist" in p or "orchestrator" in p:
    return "orch"
  if "trainer" in p:
    return "trainer"
  if "rollout" in p:
    return "rollout"
  return "proc"


def _install_hooks() -> None:
  global _hooks_installed, _prev_sigterm
  if _hooks_installed:
    return
  _hooks_installed = True
  atexit.register(_on_atexit)
  if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)
  try:
    if threading.current_thread() is threading.main_thread():
      _prev_sigterm = signal.getsignal(signal.SIGTERM)
      signal.signal(signal.SIGTERM, _on_sigterm)
  except Exception as e:  # pylint: disable=broad-except
    _count_error("signal", e)


def shutdown(timeout: float = EXIT_TIMEOUT_S) -> None:
  """Final write, health line, upload and (orchestrator) manifest; then off."""
  global ON
  st = _state
  if st is None or st.pid != os.getpid() or st.stopped.is_set():
    ON = False
    return
  ON = False
  try:
    if st.thread is None or not st.thread.is_alive():
      st.request(upload=True, timeout=None, stop=True)
      st.tick(st.clock())
      return
    st.request(upload=True, timeout=timeout, stop=True)
  except Exception as e:  # pylint: disable=broad-except
    _count_error("shutdown", e)


_n_mains = 1  # process mains sharing this OS process's stream


def set_process_count(n: int) -> None:
  """How many process mains run in this OS process (runtime main sets it)."""
  global _n_mains
  try:
    _n_mains = max(1, int(n))
  except Exception:  # pylint: disable=broad-except
    pass


def _node_timeout(timeout: float | None) -> float:
  if timeout is not None:
    return timeout
  try:
    return float(os.environ.get("TUNIX_EVENTS_NODE_STOP_TIMEOUT_S",
                                NODE_STOP_TIMEOUT_S))
  except Exception:  # pylint: disable=broad-except
    return NODE_STOP_TIMEOUT_S


def node_stopping(timeout: float | None = None) -> None:
  """A node's own SIGTERM/SIGINT handler fired and its drain is about to run.

  Writes and uploads what is queued now (so a pod killed during a long drain
  keeps everything up to here), waiting at most `timeout` seconds. Never raises.
  """
  if not ON:
    return
  try:
    flush(upload=True, timeout=_node_timeout(timeout))
  except Exception as e:  # pylint: disable=broad-except
    _count_error("node_stop", e)


def node_stopped(timeout: float | None = None) -> None:
  """A node finished draining: final write, final health line and upload.

  Waits at most `timeout` seconds and never raises. When several process mains
  share this OS process, only writes and uploads (the stream stays on for the
  others; atexit writes the final line).
  """
  if not ON:
    return
  try:
    t = _node_timeout(timeout)
    if _n_mains > 1:
      flush(upload=True, timeout=t)
    else:
      shutdown(t)
  except Exception as e:  # pylint: disable=broad-except
    _count_error("node_stop", e)


def _on_atexit() -> None:
  try:
    shutdown(float(os.environ.get("TUNIX_EVENTS_EXIT_TIMEOUT_S",
                                  EXIT_TIMEOUT_S)))
  except Exception:  # pylint: disable=broad-except
    pass


def _on_sigterm(signum, frame) -> None:
  try:
    shutdown(float(os.environ.get("TUNIX_EVENTS_EXIT_TIMEOUT_S",
                                  EXIT_TIMEOUT_S)))
  except Exception:  # pylint: disable=broad-except
    pass
  prev = _prev_sigterm
  if callable(prev):
    prev(signum, frame)
    return
  if prev == signal.SIG_IGN:
    return
  # Default disposition: restore it and re-deliver, so the exit status is the
  # same as without events.
  signal.signal(signal.SIGTERM, signal.SIG_DFL)
  os.kill(os.getpid(), signal.SIGTERM)


def _after_fork_child() -> None:
  # The child does nothing: one writer per object, and no lock of the parent's
  # is touched (the state is dropped, not reused).
  global ON, _state, _err_lock
  ON = False
  _state = None
  _err_lock = threading.Lock()


def _reset_for_tests() -> None:
  """Stops the writer (if any) and returns the module to its import state."""
  global ON, _FULL, _state, _strict
  st = _state
  ON = False
  if st is not None and st.thread is not None and st.thread.is_alive():
    st.request(upload=False, timeout=10.0, stop=True)
  _state = None
  _FULL = False
  _strict = False
  _orphan_errors[0] = 0
  _logged_kinds.clear()


def _set_strict(on: bool) -> None:
  """Strict test mode: emit() raises SchemaError on a schema violation."""
  global _strict
  _strict = bool(on)
