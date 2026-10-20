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

"""Tests of tunix.experimental.events (spec Validation 1.1-1.9 and step 2).

Step numbers in the test names follow notes/events_spec.md rev 2. The
subprocess tests load events.py by path, so they do not import the tunix
package (which pulls in JAX).
"""

import ast
import asyncio
import builtins
import gc
import json
import logging
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tracemalloc
import unittest
from unittest import mock

from absl import logging as absl_logging
from tunix.experimental import events

EVENTS_PY = pathlib.Path(events.__file__).resolve()
TUNIX_ROOT = EVENTS_PY.parents[1]  # .../tunix


def _lines(path):
  with open(path, encoding="utf-8") as f:
    return [json.loads(x) for x in f.read().splitlines() if x]


def _run_child(code: str, env=None, timeout=60):
  """Runs `code` in a fresh interpreter with `events` loaded by path."""
  prelude = textwrap.dedent(f"""
      import importlib.util, sys
      _s = importlib.util.spec_from_file_location("events", {str(EVENTS_PY)!r})
      events = importlib.util.module_from_spec(_s)
      sys.modules["events"] = events
      _s.loader.exec_module(events)
  """)
  full_env = dict(os.environ)
  full_env.pop("TUNIX_EVENTS", None)
  full_env.update(env or {})
  return subprocess.Popen([sys.executable, "-c", prelude + textwrap.dedent(code)],
                          env=full_env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)


class _Base(unittest.TestCase):

  def setUp(self):
    super().setUp()
    events._reset_for_tests()
    self.tmp = tempfile.mkdtemp(prefix="events_test_")
    self._saved_upload = events._default_upload
    self._saved_list = events._default_list_tails
    # No test reaches GCS: gs:// uploads land under <tmp>/gcs, listings of a
    # gs:// directory are empty.
    gcs_root = os.path.join(self.tmp, "gcs")

    def fake_upload(src, dst):
      if dst.startswith("gs://"):
        dst = os.path.join(gcs_root, dst[len("gs://"):])
      events._default_upload(src, dst)

    def fake_list(d):
      return {} if d.startswith("gs://") else self._saved_list(d)

    events._upload_fn = fake_upload
    events._list_tails_fn = fake_list
    self._env = mock.patch.dict(os.environ, {}, clear=False)
    self._env.start()
    os.environ.pop("TUNIX_EVENTS", None)

  def tearDown(self):
    events._reset_for_tests()
    events._upload_fn = self._saved_upload
    events._list_tails_fn = self._saved_list
    self._env.stop()
    super().tearDown()

  def configure(self, level="on", **kw):
    kw.setdefault("local_dir", self.tmp)
    events.configure("run-x", kw.pop("role", "orch"), kw.pop("pod", "pod-0"),
                     kw.pop("gcs_dir", None), level, **kw)
    return events._state

  def path(self, role="orch", pod="pod-0"):
    return os.path.join(self.tmp, f"{role}-{pod}.jsonl")


# ---------------------------------------------------------------------------
# 1.1 Off mode is inert.


class OffModeTest(_Base):

  def test_1_1_1_off_calls_no_thread_open_or_gcs(self):
    self.assertFalse(events.ON)
    calls = []
    fake_storage = mock.MagicMock()
    with mock.patch.object(threading, "Thread",
                           side_effect=lambda *a, **k: calls.append("thread")), \
        mock.patch.object(builtins, "open",
                          side_effect=lambda *a, **k: calls.append("open")), \
        mock.patch.dict(sys.modules, {"google.cloud.storage": fake_storage}), \
        mock.patch.object(events, "_upload_fn",
                          side_effect=lambda *a: calls.append("upload")):
      for _ in range(100):
        events.emit("ckpt", step=1, path="p")
        events.log_and_emit(absl_logging, absl_logging.DEBUG, "x=%d", "ckpt",
                            text_args=(1,), names=("step",))
        events.flush(upload=True)
        events.configure_from_env("orch")
    self.assertEqual(calls, [])
    fake_storage.assert_not_called()
    self.assertIsNone(events._state)

  def test_1_1_1_off_levels(self):
    for lv in ("off", "", "0", "false"):
      events.configure("r", "orch", "p", None, lv, local_dir=self.tmp)
      self.assertFalse(events.ON, lv)
      self.assertIsNone(events._state)
    self.assertEqual(os.listdir(self.tmp), [])

  def test_1_1_2_off_emit_allocates_nothing(self):
    def guarded():
      if events.ON:
        events.emit("ckpt", step=1)

    def direct():
      events.emit("ckpt")

    for fn in (guarded, direct):
      fn()
      gc.collect()
      tracemalloc.start()
      snap0 = tracemalloc.take_snapshot()
      for _ in range(10000):
        fn()
      snap1 = tracemalloc.take_snapshot()
      tracemalloc.stop()
      filt = [tracemalloc.Filter(True, str(EVENTS_PY)),
              tracemalloc.Filter(True, __file__)]
      diff = snap1.filter_traces(filt).compare_to(snap0.filter_traces(filt),
                                                  "filename")
      grown = sum(d.size_diff for d in diff if d.size_diff > 0)
      # tracemalloc's own frames aside, no allocation is attributed to either
      # file over 10,000 calls.
      self.assertLess(grown, 512, (fn.__name__, diff[:3]))

  def test_1_1_2_off_emit_with_fields_allocates_nothing_in_module(self):
    gc.collect()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for i in range(10000):
      events.emit("ckpt", step=i, path="p")
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    filt = [tracemalloc.Filter(True, str(EVENTS_PY))]
    diff = snap1.filter_traces(filt).compare_to(snap0.filter_traces(filt),
                                                "filename")
    self.assertEqual(sum(d.size_diff for d in diff if d.size_diff > 0), 0)

  def test_1_1_3_import_does_no_io_and_starts_no_thread(self):
    code = textwrap.dedent(f"""
        import sys, threading, importlib.util
        # Dependencies, imported by every caller already.
        import atexit, collections, json, logging, os, re, signal, time, typing
        from absl import logging as _absl
        seen = []
        def hook(ev, args):
          if ev in ("open", "os.listdir", "os.scandir", "socket.connect",
                    "subprocess.Popen", "os.system"):
            p = str(args[0]) if args else ""
            if ev == "open" and (p.endswith((".py", ".pyc")) or "__pycache__" in p):
              return
            seen.append((ev, p))
          if ev == "_thread.start_new_thread":
            seen.append((ev, ""))
        n0 = threading.active_count()
        sys.addaudithook(hook)
        s = importlib.util.spec_from_file_location("events", {str(EVENTS_PY)!r})
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        assert m.ON is False
        print(repr(seen), threading.active_count() - n0)
    """)
    env = dict(os.environ)
    env.pop("TUNIX_EVENTS", None)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=60)
    self.assertEqual(out.returncode, 0, out.stderr)
    seen, dthreads = out.stdout.strip().rsplit(" ", 1)
    # importlib lists the module's directory once to find the .pyc cache; any
    # other open/listdir is I/O done by the module.
    allowed = {str(EVENTS_PY.parent), str(EVENTS_PY.parent / "__pycache__")}
    bad = [e for e in ast.literal_eval(seen)
           if not (e[0] in ("os.listdir", "os.scandir") and e[1] in allowed)]
    self.assertEqual(bad, [])
    self.assertEqual(int(dthreads), 0)

  def test_1_1_4_lint_direct_emit_is_guarded(self):
    problems = lint_tree(TUNIX_ROOT)
    self.assertEqual(problems, [], "\n".join(problems))

  def test_1_1_4_lint_catches_violations(self):
    bad = textwrap.dedent("""
        from tunix.experimental import events
        from tunix.experimental.events import ON
        def f(x):
          events.emit("ckpt", step=x)
          if x and events.ON:
            events.emit("ckpt", step=x)
          if events.ON:
            events.emit("ckpt", step=x)
          else:
            events.emit("ckpt", step=x)
    """)
    probs = lint_source(bad, "bad.py")
    self.assertEqual(len(probs), 3, probs)  # import ON, bare, else-branch


def _is_events_on(node) -> bool:
  return (isinstance(node, ast.Attribute) and node.attr == "ON"
          and isinstance(node.value, ast.Name) and node.value.id in
          ("events", "_events"))


def _guarded_by_on(test) -> bool:
  if _is_events_on(test):
    return True
  if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
    return any(_guarded_by_on(v) for v in test.values)
  return False


def lint_source(src: str, filename: str) -> list[str]:
  """Direct `events.emit(...)` calls not inside `if events.ON:` bodies, and
  `from ...events import ON` (which captures a stale value)."""
  tree = ast.parse(src, filename)
  problems = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
        "events") and any(a.name == "ON" for a in node.names):
      problems.append(f"{filename}:{node.lineno}: imports ON by value")

  def visit(node, guarded):
    if isinstance(node, ast.If):
      g = guarded or _guarded_by_on(node.test)
      for c in node.body:
        visit(c, g)
      for c in node.orelse:
        visit(c, guarded)
      visit(node.test, guarded)
      return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
      guarded = False  # a nested function runs later, outside the guard
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "emit" and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("events", "_events") and not guarded):
      problems.append(f"{filename}:{node.lineno}: events.emit() outside"
                      " `if events.ON:`")
    for c in ast.iter_child_nodes(node):
      visit(c, guarded)

  visit(tree, False)
  return problems


def lint_tree(root: pathlib.Path) -> list[str]:
  problems = []
  for p in sorted(root.rglob("*.py")):
    if p.name.endswith("_test.py") or p == EVENTS_PY:
      continue
    src = p.read_text(encoding="utf-8")
    if "events" not in src:
      continue
    problems += lint_source(src, str(p.relative_to(root.parent)))
  return problems


# ---------------------------------------------------------------------------
# 1.2 emit() does not raise.


class _Recursive(dict):
  pass


class NoRaiseTest(_Base):

  def _one_error_logged(self, cm_records, kind):
    msgs = [r.getMessage() for r in cm_records
            if r.name == events._LOG.name and f"{kind} error" in r.getMessage()]
    self.assertEqual(len(msgs), 1, msgs)

  def _case(self, kind, fn, n=3, pod="pod-0"):
    events._reset_for_tests()
    st = self.configure(pod=pod)
    e0 = events.health()["errors"]
    with self.assertLogs(events._LOG.name, level="WARNING") as cm:
      for _ in range(n):
        fn()  # must not raise
      events.flush(timeout=10)
      events._LOG.warning("sentinel")  # assertLogs needs >= 1 record
    self.assertGreaterEqual(events.health()["errors"] - e0, n, kind)
    self._one_error_logged(cm.records, kind)
    return st

  def test_1_2_1_unencodable_fields(self):
    rec = {}
    rec["self"] = rec
    cases = {
        "object": lambda: events.emit("ckpt", step=1, obj=object()),
        "bytes": lambda: events.emit("ckpt", step=1, blob=b"\x00\x01"),
        "nan": lambda: events.emit("ckpt", step=1, x=math.nan),
        "recursive": lambda: events.emit("ckpt", step=1, d=rec),
        "huge": lambda: events.emit("ckpt", step=1, s="x" * (10 * 2**20)),
    }
    for name, fn in cases.items():
      with self.subTest(name):
        self._case("encode", fn, pod=name)
        lines = _lines(self.path(pod=name))
        bad = [x for x in lines if x["ev"] == "ckpt"]
        self.assertEqual(len(bad), 3)
        self.assertTrue(all("encode_error" in x for x in bad))
        self.assertLess(os.path.getsize(self.path(pod=name)), 2**20)

  def test_1_2_2_unknown_event_name(self):
    self._case("unknown_event", lambda: events.emit("no_such_event", a=1))
    self.assertIn("no_such_event", [x["ev"] for x in _lines(self.path())])

  def test_1_2_2_before_configure_is_noop(self):
    events.emit("ckpt", step=1)
    events.log_and_emit(absl_logging, absl_logging.DEBUG, "x", "ckpt")
    events.flush(upload=True, timeout=1)
    self.assertEqual(events.health()["emitted"], 0)
    self.assertEqual(os.listdir(self.tmp), [])

  def test_1_2_2_second_configure(self):
    self.configure()
    with self.assertLogs(events._LOG.name, level="WARNING") as cm:
      for _ in range(3):
        events.configure("other", "trainer", "p2", None, "full",
                         local_dir=self.tmp)
    self.assertEqual(events.health()["errors"], 3)
    self._one_error_logged(cm.records, "configure")
    self.assertEqual(events._state.role, "orch")
    events.emit("ckpt", step=1)
    events.flush(timeout=10)
    self.assertEqual(_lines(self.path())[-1]["ev"], "ckpt")

  def test_1_2_3_open_error(self):
    real_open = builtins.open

    def bad_open(f, *a, **k):
      if str(f).endswith(".jsonl"):
        raise OSError(13, "Permission denied")
      return real_open(f, *a, **k)

    with mock.patch.object(builtins, "open", bad_open):
      st = self._case("io", lambda: (events.emit("ckpt", step=1),
                                     events.flush(timeout=10)))
    self.assertGreaterEqual(st.lost, 3)

  def test_1_2_3_write_error_and_full_disk(self):
    for errno_, msg in ((5, "Input/output error"), (28, "No space left")):
      with self.subTest(msg):
        events._reset_for_tests()
        st = self.configure()
        events.emit("ckpt", step=0)
        events.flush(timeout=10)  # opens the file

        class BadFile:

          def __init__(self, real):
            self.real = real

          def write(self, data):
            raise OSError(errno_, msg)

          def flush(self):
            pass

          def close(self):
            self.real.close()

        st._fh = BadFile(st._fh)
        real_open = builtins.open

        def bad_open(f, *a, **k):
          if str(f).endswith(".jsonl"):
            return BadFile(real_open(f, *a, **k))
          return real_open(f, *a, **k)

        with mock.patch.object(builtins, "open", bad_open), \
            self.assertLogs(events._LOG.name, level="WARNING") as cm:
          for i in range(3):
            events.emit("ckpt", step=i)
            events.flush(timeout=10)
          events._LOG.warning("sentinel")
        self.assertGreaterEqual(st.errors, 3)
        self.assertGreaterEqual(st.lost, 3)
        self._one_error_logged(cm.records, "io")

  def test_1_2_3_upload_error_with_gcs_dir(self):
    events._upload_fn = mock.MagicMock(side_effect=OSError("gcs down"))
    events._reset_for_tests()
    self.configure(gcs_dir="gs://b/x/events")
    with self.assertLogs(events._LOG.name, level="WARNING") as cm:
      for i in range(3):
        events.emit("ckpt", step=i)
        events.flush(upload=True, timeout=10)
      events._LOG.warning("sentinel")
    self.assertEqual(events._upload_fn.call_count, 3)
    self.assertEqual(events.health()["errors"], 3)
    self._one_error_logged(cm.records, "upload")
    self.assertEqual(len([x for x in _lines(self.path()) if x["ev"] == "ckpt"]),
                     3)

  def test_1_2_4_emit_never_raises_on_internal_error(self):
    st = self.configure()
    q, st.q = st.q, None  # any internal failure
    with self.assertLogs(events._LOG.name, level="WARNING") as cm:
      for _ in range(5):
        events.emit("ckpt", step=1)
    st.q = q
    self.assertEqual(st.errors, 5)
    self._one_error_logged(cm.records, "emit")


# ---------------------------------------------------------------------------
# 1.3 emit() does not block.


class NoBlockTest(_Base):

  def test_1_3_blocked_writer_drops_overflow_and_keeps_accepted(self):
    entered, release = threading.Event(), threading.Event()

    def blocking_upload(src, dst):
      entered.set()
      release.wait(60)

    events._upload_fn = blocking_upload
    st = self.configure(gcs_dir="gs://b/p/events")
    events.flush(upload=True)
    self.assertTrue(entered.wait(10))
    base = events.health()["emitted"]  # the config event
    cap = st.cap
    self.assertEqual(cap, events.DEFAULT_QUEUE_SIZE)
    overflow = 1000
    already = len(st.q)
    n = cap - already + overflow
    worst = 0
    for i in range(n):
      t0 = time.perf_counter_ns()
      events.emit("ckpt", step=i)
      worst = max(worst, time.perf_counter_ns() - t0)
    self.assertLess(worst, 1_000_000, f"slowest emit {worst} ns")
    self.assertEqual(events.health()["dropped"], overflow)
    release.set()
    events.flush(timeout=60)
    ckpts = [x for x in _lines(self.path()) if x["ev"] == "ckpt"]
    self.assertEqual(len(ckpts), n - overflow)
    self.assertEqual([x["step"] for x in ckpts], list(range(n - overflow)))
    self.assertEqual(events.health()["emitted"], base + n)


# ---------------------------------------------------------------------------
# 1.4 Concurrency.


class ConcurrencyTest(_Base):

  def test_1_4_threads_and_asyncio(self):
    self.configure(flush_interval_s=0.05)
    n_threads, per_thread = 16, 5000
    start = threading.Barrier(n_threads + 1)

    def worker(tid):
      start.wait()
      for i in range(per_thread):
        events.emit("ckpt", step=i, path=f"t{tid}")

    async def loop_emitter():
      for i in range(per_thread):
        events.emit("ckpt", step=i, path="loop")
        if i % 100 == 0:
          await asyncio.sleep(0)

    def run_loop():
      start.wait()
      asyncio.run(loop_emitter())

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    ts.append(threading.Thread(target=run_loop))
    for t in ts:
      t.start()
    for t in ts:
      t.join()
    events.flush(timeout=60)
    with open(self.path(), encoding="utf-8") as f:
      raw = f.read().splitlines()
    recs = [json.loads(x) for x in raw]  # raises on a torn line
    h = events.health()
    ckpts = [x for x in recs if x["ev"] == "ckpt"]
    total = n_threads * per_thread + per_thread
    self.assertEqual(len(ckpts) + 1, h["emitted"] - h["dropped"])  # + config
    self.assertEqual(h["emitted"], total + 1)
    by_src = {}
    for x in ckpts:
      by_src.setdefault(x["path"], []).append((x["step"], x["m"]))
    for src, seq in by_src.items():
      steps = [s for s, _ in seq]
      self.assertEqual(steps, sorted(steps), src)
      ms = [m for _, m in seq]
      self.assertEqual(ms, sorted(ms), src)


# ---------------------------------------------------------------------------
# 1.5 Shutdown and durability.

_CHILD_EMIT = """
    import os, signal, sys, time
    {pre}
    events.configure("run-c", "rollout", "pod-c", None, "on",
                     local_dir=sys.argv[0] if False else os.environ["D"],
                     flush_interval_s=3600)
    for i in range(500):
        events.emit("ckpt", step=i)
    print("ready", flush=True)
    {post}
"""


class ShutdownTest(_Base):

  def _child(self, pre="", post="", env=None):
    env = {"D": self.tmp, **(env or {})}
    ind = lambda c: textwrap.indent(textwrap.dedent(c), "    ").strip()
    code = _CHILD_EMIT.format(pre=ind(pre), post=ind(post))
    return _run_child(code, env=env)

  def _check_file(self):
    recs = _lines(os.path.join(self.tmp, "rollout-pod-c.jsonl"))
    self.assertEqual([x["step"] for x in recs if x["ev"] == "ckpt"],
                     list(range(500)))
    self.assertEqual(recs[-1]["ev"], "events_health")
    self.assertTrue(recs[-1]["final"])
    self.assertEqual(recs[-1]["emitted"], 501)
    return recs

  def test_1_5_1_sigterm_flushes_and_keeps_default_exit(self):
    p = self._child(post="time.sleep(60)")
    self.assertEqual(p.stdout.readline().strip(), "ready")
    p.send_signal(signal.SIGTERM)
    p.wait(60)
    self.assertEqual(p.returncode, -signal.SIGTERM)  # same as without events
    self._check_file()

  def test_1_5_2_existing_sigterm_handler_still_runs(self):
    pre = textwrap.dedent("""
        def h(signum, frame):
            print("prev-handler", flush=True)
            sys.exit(7)
        signal.signal(signal.SIGTERM, h)
    """)
    p = self._child(pre=pre, post="time.sleep(60)")
    self.assertEqual(p.stdout.readline().strip(), "ready")
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    self.assertIn("prev-handler", out)
    self.assertEqual(p.returncode, 7)
    self._check_file()

  def test_1_5_2_sig_ign_is_kept(self):
    pre = "signal.signal(signal.SIGTERM, signal.SIG_IGN)"
    p = self._child(pre=pre, post="time.sleep(2); print('alive', flush=True)")
    self.assertEqual(p.stdout.readline().strip(), "ready")
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    self.assertIn("alive", out)
    self.assertEqual(p.returncode, 0)

  def test_1_5_3_normal_exit_atexit(self):
    p = self._child(post="sys.exit(3)")
    p.communicate(timeout=60)
    self.assertEqual(p.returncode, 3)
    self._check_file()

  def test_1_5_4_fake_clock_kill_loses_at_most_one_interval(self):
    now = [1000.0]
    st = self.configure(start_thread=False, clock=lambda: now[0],
                        flush_interval_s=2.0)
    for i in range(10):
      events.emit("ckpt", step=i)
    now[0] += 1.9
    st.tick(now[0])  # before the interval: nothing written yet
    self.assertFalse(os.path.exists(self.path()))
    now[0] += 0.1
    st.tick(now[0])  # the 2 s write
    for i in range(10, 20):
      events.emit("ckpt", step=i)
    now[0] += 1.99
    st.tick(now[0])
    # A SIGKILL here loses exactly the events of the open interval.
    on_disk = [x["step"] for x in _lines(self.path()) if x["ev"] == "ckpt"]
    self.assertEqual(on_disk, list(range(10)))
    now[0] += 0.01
    st.tick(now[0])
    on_disk = [x["step"] for x in _lines(self.path()) if x["ev"] == "ckpt"]
    self.assertEqual(on_disk, list(range(20)))

  def test_1_5_4_real_sigkill(self):
    code = """
        import os, time
        events.configure("run-k", "rollout", "pod-k", None, "on",
                         local_dir=os.environ["D"], flush_interval_s=0.2)
        print("ready", flush=True)
        i = 0
        while True:
            events.emit("ckpt", step=i, path=repr(time.monotonic()))
            i += 1
            time.sleep(0.001)
    """
    p = _run_child(code, env={"D": self.tmp})
    self.assertEqual(p.stdout.readline().strip(), "ready")
    time.sleep(1.5)
    t_kill = time.monotonic()
    p.kill()
    p.wait(30)
    recs = [x for x in _lines(os.path.join(self.tmp, "rollout-pod-k.jsonl"))
            if x["ev"] == "ckpt"]
    self.assertTrue(recs)
    last_m = recs[-1]["m"]
    # time.monotonic() is system-wide on Linux, so the child's m and this
    # process's clock agree. Loss <= one 0.2 s interval plus scheduling slack.
    self.assertGreater(last_m, t_kill - 0.2 - 0.3)
    self.assertEqual([x["step"] for x in recs], list(range(len(recs))))


# ---------------------------------------------------------------------------
# 1.6 Fork safety.


class ForkTest(_Base):

  @unittest.skipUnless(hasattr(os, "fork"), "needs fork")
  def test_1_6_child_does_nothing_and_does_not_deadlock(self):
    st = self.configure()
    events.emit("ckpt", step=1)
    events.flush(timeout=10)
    st.lock.acquire()  # a parent lock held across the fork
    errlock = events._err_lock
    errlock.acquire()
    try:
      pid = os.fork()
      if pid == 0:  # child
        ok = 1
        try:
          signal.alarm(10)
          events.emit("ckpt", step=2)
          events.log_and_emit(absl_logging, absl_logging.DEBUG, "x", "ckpt")
          events.flush(upload=True, timeout=1)
          h = events.health()
          events._count_error("child", None)
          ok = 0 if (not events.ON and events._state is None
                     and h["emitted"] == 0) else 2
        finally:
          os._exit(ok)
    finally:
      st.lock.release()
      errlock.release()
    _, status = os.waitpid(pid, 0)
    self.assertTrue(os.WIFEXITED(status), status)
    self.assertEqual(os.WEXITSTATUS(status), 0)
    events.flush(timeout=10)
    steps = [x.get("step") for x in _lines(self.path()) if x["ev"] == "ckpt"]
    self.assertEqual(steps, [1])  # the child wrote nothing


# ---------------------------------------------------------------------------
# 1.7 No secrets.


class SecretsTest(_Base):

  def test_1_7_secrets_never_reach_a_file(self):
    sentinel = "SENTINEL-6f1d2c9a7b"
    os.environ["WANDB_API_KEY"] = sentinel
    os.environ["FOO_TOKEN"] = sentinel
    os.environ["JOB_PREFIX"] = f"job-{sentinel}"  # leaks via an allowed key
    allow = events.CONFIG_ENV_ALLOWLIST + ("WANDB_API_KEY", "FOO_TOKEN")
    uploaded = os.path.join(self.tmp, "gcs")
    events._upload_fn = lambda src, dst: events._default_upload(
        src, os.path.join(uploaded, os.path.basename(dst)))
    with mock.patch.object(events, "CONFIG_ENV_ALLOWLIST", allow):
      events.configure("run-s", "orch", "pod-s", "gs://b/events", "on",
                       local_dir=self.tmp,
                       argv=["prog", f"--wandb_api_key={sentinel}",
                             f"--note=x{sentinel}y", "--api-token", sentinel])
      events.emit("config", **events.config_fields(allowlist=allow))
    events.shutdown()
    found = []
    for root, _, files in os.walk(self.tmp):
      for name in files:
        with open(os.path.join(root, name), "rb") as f:
          if sentinel.encode() in f.read():
            found.append(name)
    self.assertEqual(found, [])
    cfg = [x for x in _lines(self.path("orch", "pod-s")) if x["ev"] == "config"]
    self.assertEqual(len(cfg), 2)
    self.assertNotIn("WANDB_API_KEY", cfg[0]["env"])
    self.assertNotIn("FOO_TOKEN", cfg[0]["env"])
    self.assertIn("JOB_PREFIX", cfg[0]["env"])


# ---------------------------------------------------------------------------
# 1.8 Schema.


class SchemaTest(_Base):

  def test_1_8_1_every_event_type_has_a_schema(self):
    spec_events = {
        "run_start", "run_stop", "config", "stagger_hold", "dispatch",
        "arrive", "pack", "mb_stage", "mb_send", "mb_ack", "trainer_mb",
        "trainer_step", "wsync_phase", "wsync_round", "engine_pause",
        "engine_unpause", "worker_ft", "turn", "sampler_call", "trajectory",
        "engine_stats", "ckpt", "events_health"}
    self.assertLessEqual(spec_events, set(events.SCHEMAS))
    self.assertEqual(events.FULL_ONLY, {"turn", "sampler_call", "trajectory"})
    for s in events.SCHEMAS.values():
      for name, kind in {**s.required, **s.optional}.items():
        self.assertRegex(name, r"^[a-z][a-z0-9_]*$")
        if name.endswith("_s"):
          self.assertEqual(kind, "float", (s.ev, name))

  def test_1_8_1_register_rejects_bad_schemas(self):
    with self.assertRaises(ValueError):
      events.register("x_bad", {"Bad": "int"})
    with self.assertRaises(ValueError):
      events.register("x_bad", {"dur_s": "int"})
    with self.assertRaises(ValueError):
      events.register("x_bad", {"t": "float"})

  def test_1_8_2_strict_mode_fails_on_violation(self):
    self.configure()
    events._set_strict(True)
    events.emit("ckpt", step=1, path="p")  # valid
    with self.assertRaises(events.SchemaError):
      events.emit("ckpt", path="p")  # missing required
    with self.assertRaises(events.SchemaError):
      events.emit("ckpt", step="1")  # wrong type
    with self.assertRaises(events.SchemaError):
      events.emit("ckpt", step=1, unknown_field=1)
    with self.assertRaises(events.SchemaError):
      events.emit("no_such_event")
    with self.assertRaises(events.SchemaError):
      events.log_and_emit(absl_logging, absl_logging.DEBUG, "s=%s", "ckpt",
                          text_args=("x",), names=("step",))

  def test_1_8_3_golden_envelope(self):
    self.configure(level="full")
    events.emit("wsync_phase", round=3, req_id="r3", phase="bind",
                elapsed_s=1, workers=8)
    events.emit("turn", traj="t1", turn=0, model_s=0.5)
    events.flush(timeout=10)
    recs = _lines(self.path())
    for rec in recs:
      self.assertEqual(list(rec)[:7], list(events.ENVELOPE))
      self.assertEqual(rec["v"], 1)
      self.assertIsInstance(rec["t"], float)
      self.assertIsInstance(rec["m"], float)
      self.assertEqual((rec["run"], rec["role"], rec["pod"]),
                       ("run-x", "orch", "pod-0"))
      for k, v in rec.items():
        self.assertRegex(k, r"^[a-z][a-z0-9_]*$")
        if k.endswith("_s"):
          self.assertIsInstance(v, float, (rec["ev"], k))
    ws = [r for r in recs if r["ev"] == "wsync_phase"][0]
    golden = {"ev": "wsync_phase", "round": 3, "req_id": "r3", "phase": "bind",
              "elapsed_s": 1.0, "workers": 8}
    self.assertEqual({k: ws[k] for k in golden}, golden)
    self.assertEqual(set(ws) - set(events.ENVELOPE),
                     set(golden) - {"ev"})

  def test_1_8_full_only_events_skipped_at_on(self):
    self.configure(level="on")
    self.assertTrue(events.enabled())
    self.assertFalse(events.enabled(full=True))
    events.emit("turn", traj="t", turn=1)
    events.emit("ckpt", step=1)
    events.flush(timeout=10)
    self.assertEqual([x["ev"] for x in _lines(self.path())],
                     ["config", "ckpt"])

  def test_1_8_coercion_matches_text_formatting(self):
    self.configure()

    class Actor:

      def __str__(self):
        return "actor-7"

    exc = RuntimeError("x" * 500)
    events.emit("worker_ft", action="evict", worker=Actor(), reason=exc,
                counts={events.__name__: 1}, max_zero_worker_wait_s=3)
    events.flush(timeout=10)
    ft = _lines(self.path())[-1]
    self.assertEqual(ft["worker"], "actor-7")  # as %s prints it
    self.assertEqual(ft["reason"], repr(exc)[:200])  # %r, trimmed
    self.assertEqual(ft["max_zero_worker_wait_s"], 3.0)
    self.assertEqual(events.health()["errors"], 0)


# ---------------------------------------------------------------------------
# log_and_emit: the text line and the record's location are the site's.


class LogAndEmitTest(_Base):

  def _site_absl(self, rnd, req):
    events.log_and_emit(absl_logging, absl_logging.INFO,
                        "WEIGHT_SYNC_PHASE round=%d req_id=%s phase=bind"
                        " elapsed_s=%.3f", "wsync_phase",
                        text_args=(rnd, req, 1.23456),
                        names=("round", "req_id", "elapsed_s"), phase="bind")

  def _site_std(self, logger, rnd):
    events.log_and_emit(logger, logging.INFO, "STD round=%d", "wsync_phase",
                        text_args=(rnd,), names=("round",), req_id="q")

  def _site_std_module(self, rnd):
    events.log_and_emit(logging, logging.INFO, "MOD round=%d", "wsync_phase",
                        text_args=(rnd,), names=("round",), req_id="m")

  def _capture(self, fn):
    records = []

    class H(logging.Handler):

      def emit(self, record):
        records.append(record)

    h = H(level=logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(h)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
      fn()
    finally:
      root.removeHandler(h)
      root.setLevel(old)
    return records

  def test_text_and_location_identical_off_and_on(self):
    std = logging.getLogger("events_test.std")
    for level in ("off", "on"):
      events._reset_for_tests()
      if level == "on":
        self.configure()
      recs = self._capture(lambda: (self._site_absl(4, "r4"),
                                    self._site_std(std, 5),
                                    self._site_std_module(6)))
      self.assertEqual(
          [r.getMessage() for r in recs],
          ["WEIGHT_SYNC_PHASE round=4 req_id=r4 phase=bind elapsed_s=1.235",
           "STD round=5", "MOD round=6"])
      for r in recs:
        self.assertEqual(r.filename, os.path.basename(__file__), level)
        self.assertIn(r.funcName,
                      ("_site_absl", "_site_std", "_site_std_module"), level)
      self.assertEqual(recs[0].levelno, logging.INFO)
      self.assertEqual(recs[1].levelno, logging.INFO)
    events.flush(timeout=10)
    ws = [x for x in _lines(self.path()) if x["ev"] == "wsync_phase"]
    self.assertEqual(
        [{k: x[k] for k in ("round", "req_id") if k in x} for x in ws],
        [{"round": 4, "req_id": "r4"}, {"round": 5, "req_id": "q"},
         {"round": 6, "req_id": "m"}])
    self.assertEqual(ws[0]["elapsed_s"], 1.23456)


# ---------------------------------------------------------------------------
# 1.9 Manifest.


class ManifestTest(_Base):

  def test_1_9_manifest_counts_equal_files_and_problems_reported(self):
    # Two other processes write their streams into the shared directory.
    for role, pod, n in (("rollout", "r0", 40), ("trainer", "t0", 25)):
      code = f"""
          import os
          events.configure("run-m", {role!r}, {pod!r}, None, "on",
                           local_dir=os.environ["D"], queue_size=30,
                           start_thread=False)
          for i in range({n}):
              events.emit("ckpt", step=i)
      """
      p = _run_child(code, env={"D": self.tmp})
      _, err = p.communicate(timeout=60)
      self.assertEqual(p.returncode, 0, err)
    os.environ["TUNIX_EVENTS_EXPECTED"] = "orch=1,rollout=2,trainer=1"
    self.configure(role="orch", pod="o0")
    for i in range(7):
      events.emit("ckpt", step=i)
    events.shutdown()
    man = json.load(open(os.path.join(self.tmp, events.MANIFEST_NAME)))
    self.assertEqual(man["expected"], {"orch": 1, "rollout": 2, "trainer": 1})
    for name, rec in man["files"].items():
      lines = _lines(os.path.join(self.tmp, name))
      n_events = sum(1 for x in lines if x["ev"] != "events_health")
      self.assertEqual(rec["events"], n_events, name)
      self.assertEqual(rec["dropped"], lines[-1]["dropped"], name)
    self.assertEqual(man["files"]["rollout-r0.jsonl"]["dropped"], 11)  # 41-30
    self.assertEqual(man["files"]["trainer-t0.jsonl"]["dropped"], 0)
    self.assertEqual(events.verify_manifest(self.tmp)["problems"],
                     ["missing: expected 2 rollout file(s), found 1",
                      "dropped: rollout-r0.jsonl dropped 11"])
    # Truncate one file and remove another.
    p = os.path.join(self.tmp, "trainer-t0.jsonl")
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) // 2])
    os.remove(os.path.join(self.tmp, "rollout-r0.jsonl"))
    probs = events.verify_manifest(self.tmp)["problems"]
    self.assertIn("missing: rollout-r0.jsonl (in the manifest, no file)", probs)
    self.assertTrue(any(x.startswith("truncated: trainer-t0.jsonl")
                        for x in probs), probs)


# ---------------------------------------------------------------------------
# Step 2: overhead.


def _pct(xs, q):
  xs = sorted(xs)
  return xs[min(len(xs) - 1, int(q * len(xs)))]


async def _lag_probe(duration_s, rate_per_s):
  """Overshoots of asyncio.sleep(0.01) over `duration_s`, while a second task
  emits `rate_per_s` events from the loop (as coroutine sites do)."""
  lags = []
  stop = False

  async def emitter():
    per_tick = max(1, rate_per_s // 100)
    i = 0
    while not stop:
      for _ in range(per_tick):
        if events.ON:
          events.emit("wsync_phase", round=i, req_id="r", phase="pre",
                      elapsed_s=0.1, workers=8)
        i += 1
      await asyncio.sleep(0.01)

  task = asyncio.create_task(emitter())
  t_end = time.monotonic() + duration_s
  while time.monotonic() < t_end:
    t0 = time.monotonic()
    await asyncio.sleep(0.01)
    lags.append(time.monotonic() - t0 - 0.01)
  stop = True
  await task
  return lags


class OverheadTest(_Base):
  """Spec step 2. Prints `EVENTS_OVERHEAD {json}` for the README."""

  def test_2_emit_p99_and_loop_lag(self):
    duration = float(os.environ.get("EVENTS_LAG_PROBE_S", "60"))
    uploads = [0]
    in_upload = threading.Event()

    def slow_upload(src, dst):  # an upload in progress on the writer thread
      uploads[0] += 1
      in_upload.set()
      with open(src, "rb") as f:
        f.read()
      time.sleep(0.5)

    events._upload_fn = slow_upload
    self.configure(gcs_dir="gs://b/p/events", upload_interval_s=0.0)
    events.flush(upload=True)
    self.assertTrue(in_upload.wait(10))
    n = 10**6
    durs = [0] * n
    pc = time.perf_counter_ns
    for i in range(n):
      t0 = pc()
      events.emit("wsync_phase", round=i, req_id="r", phase="pre",
                  elapsed_s=0.1, workers=8)
      durs[i] = pc() - t0
    h1 = events.health()
    p50, p99 = _pct(durs, 0.50) / 1e3, _pct(durs, 0.99) / 1e3
    # The same 10^6 calls with a queue that holds them all (no drop path).
    events._reset_for_tests()
    events._upload_fn = slow_upload
    in_upload.clear()
    self.configure(gcs_dir="gs://b/p/events", upload_interval_s=0.0,
                   queue_size=2 * n)
    events.flush(upload=True)
    self.assertTrue(in_upload.wait(10))
    durs_a = [0] * n
    for i in range(n):
      t0 = pc()
      events.emit("wsync_phase", round=i, req_id="r", phase="pre",
                  elapsed_s=0.1, workers=8)
      durs_a[i] = pc() - t0
    h1a = events.health()
    a50, a99 = _pct(durs_a, 0.50) / 1e3, _pct(durs_a, 0.99) / 1e3
    del durs_a
    events.flush(timeout=300)  # the backlog is written before the lag probe

    # Loop lag with events on (5,000 events/s) and then off.
    events._reset_for_tests()
    events._upload_fn = slow_upload
    self.configure(gcs_dir="gs://b/p/events")
    lag_on = asyncio.run(_lag_probe(duration, 5000))
    h2 = events.health()
    events._reset_for_tests()
    lag_off = asyncio.run(_lag_probe(duration, 5000))
    on99, off99 = _pct(lag_on, 0.99) * 1e3, _pct(lag_off, 0.99) * 1e3
    result = {
        "emit_n": n, "emit_p50_us": round(p50, 3), "emit_p99_us": round(p99, 3),
        "emit_max_us": round(max(durs) / 1e3, 1), "emit_dropped": h1["dropped"],
        "accepted_p50_us": round(a50, 3), "accepted_p99_us": round(a99, 3),
        "accepted_dropped": h1a["dropped"],
        "uploads_during_emit": uploads[0], "lag_probe_s": duration,
        "lag_on_p50_ms": round(_pct(lag_on, 0.5) * 1e3, 3),
        "lag_on_p99_ms": round(on99, 3),
        "lag_off_p50_ms": round(_pct(lag_off, 0.5) * 1e3, 3),
        "lag_off_p99_ms": round(off99, 3),
        "lag_p99_delta_ms": round(on99 - off99, 3),
        "lag_on_events": h2["emitted"], "lag_on_dropped": h2["dropped"],
        "lag_samples_on": len(lag_on), "lag_samples_off": len(lag_off)}
    print("EVENTS_OVERHEAD " + json.dumps(result), flush=True)
    self.assertLess(p99, 20.0, result)
    self.assertLess(a99, 20.0, result)
    self.assertLess(on99 - off99, 1.0, result)


if __name__ == "__main__":
  unittest.main()
