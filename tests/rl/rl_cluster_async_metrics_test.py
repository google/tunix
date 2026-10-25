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

"""Shutdown tests for the TUNIX_ASYNC_METRICS metrics exporter."""

import ast
import contextlib
import gc
import inspect
import os
import subprocess
import sys
import textwrap
import threading
import time
from unittest import mock
import weakref

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc

_WORKER_NAME = "tunix-async-metrics"


@contextlib.contextmanager
def _async_metrics_env(enabled: bool, queue_depth: int = 8):
  with mock.patch.dict(os.environ):
    os.environ.pop("TUNIX_ASYNC_METRICS", None)
    os.environ.pop("TUNIX_ASYNC_METRICS_QUEUE_DEPTH", None)
    if enabled:
      os.environ["TUNIX_ASYNC_METRICS"] = "1"
      os.environ["TUNIX_ASYNC_METRICS_QUEUE_DEPTH"] = str(queue_depth)
    yield


def _make_recorder(delay_s: float = 0.0):
  """External metrics logger recording (step, thread name) per flush.

  The recorder holds no reference to the engine, so it does not affect the
  collectibility checks below.

  Args:
    delay_s: Time each flush takes, to keep flushes queued at close().

  Returns:
    The recorder callable and the list it appends to.
  """
  records = []

  def recorder(metrics_buffer):
    if delay_s:
      time.sleep(delay_s)
    records.append(
        (metrics_buffer.global_steps, threading.current_thread().name)
    )

  return recorder, records


class AsyncMetricsShutdownTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    tc.safe_set_n_cpu_devices(4)

  def _make_engine(self, recorder) -> rl_engine_lib.RLEngine:
    devices = jax.devices()
    actor_mesh = jax.sharding.Mesh(
        np.array(devices[:2]).reshape(2, 1), ("fsdp", "tp")
    )
    rollout_mesh = jax.sharding.Mesh(
        np.array(devices[2:4]).reshape(1, 2), ("fsdp", "tp")
    )
    cluster_config = rl_engine_lib.ClusterConfig(
        role_to_mesh={
            rl_engine_lib.Role.ACTOR: actor_mesh,
            rl_engine_lib.Role.REFERENCE: actor_mesh,
            rl_engine_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_engine_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            kv_cache_size=1024,
            data_type=jnp.bfloat16,
        ),
    )
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    engine = rl_engine_lib.RLEngine(
        actor=model, tokenizer=vocab, cluster_config=cluster_config
    )
    engine.with_external_metrics_logger(recorder)
    return engine

  def _run_steps(self, engine, num_steps: int) -> None:
    """Buffers one metric per step; the step-`s` buffer flushes at `s + 1`."""
    for step in range(num_steps):
      engine.global_steps = step
      engine.buffer_metrics({"loss": (float(step), np.mean)})

  def test_close_terminates_worker_thread(self):
    with _async_metrics_env(enabled=True):
      recorder, _ = _make_recorder()
      engine = self._make_engine(recorder)
      self._run_steps(engine, 4)
      exporter = engine._async_metrics_exporter
      self.assertIsNotNone(exporter)
      worker = exporter._worker
      self.assertTrue(worker.is_alive())

      engine.close()

      self.assertFalse(worker.is_alive())

  def test_engine_and_exporter_collectible_after_close(self):
    with _async_metrics_env(enabled=True):
      recorder, records = _make_recorder()
      engine = self._make_engine(recorder)
      self._run_steps(engine, 3)
      engine.close()
      self.assertEqual([step for step, _ in records], [0, 1, 2])

      engine_ref = weakref.ref(engine)
      exporter_ref = weakref.ref(engine._async_metrics_exporter)
      del engine
      gc.collect()

      # Neither the worker thread nor an exit hook may keep the last flush
      # callback (bound to the engine) or the exporter alive after close().
      self.assertIsNone(engine_ref())
      self.assertIsNone(exporter_ref())

  def test_close_flushes_every_accepted_metric_in_step_order(self):
    with _async_metrics_env(enabled=True, queue_depth=8):
      # Slow flushes keep several buffers queued when close() starts.
      recorder, records = _make_recorder(delay_s=0.05)
      engine = self._make_engine(recorder)
      self._run_steps(engine, 5)
      exporter = engine._async_metrics_exporter

      engine.close()

      self.assertEqual(exporter._dropped, 0)
      self.assertEqual([step for step, _ in records], [0, 1, 2, 3, 4])
      # Completed steps flush on the worker; the open step-4 buffer is
      # flushed inline by close() after the queued ones.
      self.assertEqual([name for _, name in records[:4]], [_WORKER_NAME] * 4)
      self.assertEqual(records[4][1], threading.main_thread().name)
      self.assertEqual(exporter._queue.unfinished_tasks, 0)

  def test_close_is_idempotent(self):
    with _async_metrics_env(enabled=True):
      recorder, records = _make_recorder()
      engine = self._make_engine(recorder)
      self._run_steps(engine, 4)
      exporter = engine._async_metrics_exporter

      engine.close()
      engine.close()

      self.assertFalse(exporter._worker.is_alive())
      self.assertIs(engine._async_metrics_exporter, exporter)
      worker_steps = [step for step, name in records if name == _WORKER_NAME]
      self.assertEqual(worker_steps, [0, 1, 2])

  def test_flush_after_close_runs_inline_without_new_worker(self):
    with _async_metrics_env(enabled=True):
      recorder, records = _make_recorder()
      engine = self._make_engine(recorder)
      self._run_steps(engine, 2)
      exporter = engine._async_metrics_exporter
      engine.close()
      del records[:]

      engine.global_steps = 10
      engine.buffer_metrics({"loss": (10.0, np.mean)})

      # The step-1 buffer that close() left in place is flushed before
      # buffer_metrics() returns, on the calling thread, by the same (closed)
      # exporter owner: no new worker thread is started.
      self.assertEqual(records, [(1, threading.main_thread().name)])
      self.assertIs(engine._async_metrics_exporter, exporter)
      self.assertFalse(exporter._worker.is_alive())

  def test_flags_off_flushes_inline_without_worker(self):
    with _async_metrics_env(enabled=False):
      recorder, records = _make_recorder()
      engine = self._make_engine(recorder)
      main_name = threading.main_thread().name
      for step in range(4):
        engine.global_steps = step
        engine.buffer_metrics({"loss": (float(step), np.mean)})
        # The previous step is flushed before buffer_metrics() returns.
        self.assertEqual(records, [(s, main_name) for s in range(step)])

      engine.close()

      self.assertEqual(records, [(s, main_name) for s in range(4)])
      self.assertIsNone(engine._async_metrics_exporter)


class AsyncMetricsExporterTest(absltest.TestCase):

  def test_close_runs_accepted_flushes_and_stops_worker(self):
    ran = []
    exporter = rl_engine_lib._AsyncMetricsExporter(maxsize=4)
    gate = threading.Event()
    exporter.submit(lambda: gate.wait(5.0))
    for i in range(3):
      exporter.submit(lambda i=i: ran.append(i))
    gate.set()

    self.assertTrue(exporter.close(timeout_s=10.0))

    self.assertEqual(ran, [0, 1, 2])
    self.assertFalse(exporter._worker.is_alive())
    self.assertEqual(exporter._queue.unfinished_tasks, 0)
    self.assertTrue(exporter._queue.empty())
    # Idempotent.
    self.assertTrue(exporter.close(timeout_s=10.0))
    self.assertTrue(exporter.closed)

  def test_submit_after_close_fails_closed(self):
    exporter = rl_engine_lib._AsyncMetricsExporter(maxsize=1)
    self.assertTrue(exporter.close(timeout_s=10.0))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      exporter.submit(lambda: None)

  def test_close_releases_last_callback(self):
    class Owner:

      def flush(self):
        pass

    owner = Owner()
    owner_ref = weakref.ref(owner)
    exporter = rl_engine_lib._AsyncMetricsExporter(maxsize=1)
    exporter.submit(owner.flush)
    del owner

    self.assertTrue(exporter.close(timeout_s=10.0))
    gc.collect()

    self.assertIsNone(owner_ref())

  def test_queue_full_drops_without_blocking(self):
    ran = []
    started = threading.Event()
    release = threading.Event()
    exporter = rl_engine_lib._AsyncMetricsExporter(maxsize=1)

    def blocked():
      started.set()
      release.wait(10.0)
      ran.append("blocked")

    exporter.submit(blocked)
    self.assertTrue(started.wait(10.0))
    exporter.submit(lambda: ran.append("queued"))
    t0 = time.monotonic()
    exporter.submit(lambda: ran.append("dropped"))
    self.assertLess(time.monotonic() - t0, 1.0)
    self.assertEqual(exporter._dropped, 1)

    release.set()
    self.assertTrue(exporter.close(timeout_s=10.0))
    self.assertEqual(ran, ["blocked", "queued"])

  def test_close_timeout_discards_queued_flushes(self):
    ran = []
    started = threading.Event()
    release = threading.Event()
    exporter = rl_engine_lib._AsyncMetricsExporter(maxsize=4)

    def hung():
      started.set()
      release.wait(10.0)
      ran.append("hung")

    exporter.submit(hung)
    self.assertTrue(started.wait(10.0))
    exporter.submit(lambda: ran.append("late"))

    with self.assertLogs(level="WARNING") as logs:
      self.assertFalse(exporter.close(timeout_s=0.2))
    self.assertIn("discarded 1", "\n".join(logs.output))

    # The flush in progress cannot be interrupted, but nothing queued behind
    # it runs once close() has returned, and the worker then exits.
    release.set()
    exporter._worker.join(10.0)
    self.assertFalse(exporter._worker.is_alive())
    self.assertEqual(ran, ["hung"])
    self.assertEqual(exporter._queue.unfinished_tasks, 0)
    self.assertTrue(exporter.close(timeout_s=1.0))

  def test_process_exit_runs_accepted_flushes(self):
    # No explicit close(): the exit hook must run every accepted flush and
    # stop the worker before the interpreter goes away.
    script = textwrap.dedent("""
        import time
        from tunix.rl import rl_cluster

        exporter = rl_cluster._AsyncMetricsExporter(maxsize=4)

        def flush(i):
          time.sleep(0.1)
          print(f"flushed {i}", flush=True)

        for i in range(3):
          exporter.submit(lambda i=i: flush(i))
        print("exiting", flush=True)
        """)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    lines = result.stdout.splitlines()
    self.assertIn("exiting", lines)
    self.assertEqual(
        [line for line in lines if line.startswith("flushed")],
        ["flushed 0", "flushed 1", "flushed 2"],
    )

  def test_exporter_issues_no_collectives(self):
    # Callbacks run on one process's worker thread; any cross-process
    # aggregation must happen on the train loop before submit().
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(rl_engine_lib._AsyncMetricsExporter))
    )
    names = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Name):
        names.add(node.id)
      elif isinstance(node, ast.Attribute):
        names.add(node.attr)
    self.assertEmpty(
        names
        & {
            "jax",
            "jnp",
            "multihost_utils",
            "process_allgather",
            "sync_global_devices",
            "broadcast_one_to_all",
            "psum",
            "all_gather",
        }
    )


if __name__ == "__main__":
  absltest.main()
