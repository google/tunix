from concurrent import futures
import json
import locale
import os
import pathlib
import tempfile
import threading
import time
from typing import TypedDict
from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd
from tunix.utils import trajectory_logger


_FAKE_GCS_ROOT = 'gs://fake-bucket'


class _FakeGcsPath:
  """Emulates GCS path semantics on top of a local directory.

  GCS has no real directories: `mkdir()` is a no-op and `is_dir()` keeps
  returning False until an object exists under the prefix.
  """

  def __init__(self, uri: str, local_root: str):
    self._uri = str(uri).rstrip('/')
    self._local_root = local_root
    relative = self._uri[len(_FAKE_GCS_ROOT) :].lstrip('/')
    self._local = pathlib.Path(local_root) / relative

  @property
  def local_path(self) -> pathlib.Path:
    return self._local

  @property
  def name(self) -> str:
    return self._uri.rsplit('/', 1)[-1]

  @property
  def parent(self) -> '_FakeGcsPath':
    return _FakeGcsPath(self._uri.rsplit('/', 1)[0], self._local_root)

  def __str__(self) -> str:
    return self._uri

  def __truediv__(self, other: str) -> '_FakeGcsPath':
    return _FakeGcsPath(f'{self._uri}/{other}', self._local_root)

  def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
    del parents, exist_ok  # No-op, as on GCS.

  def is_dir(self) -> bool:
    return False

  def exists(self) -> bool:
    return self._local.exists()

  def open(self, mode: str = 'r', encoding: str | None = None):
    if any(flag in mode for flag in ('w', 'a', 'x')):
      self._local.parent.mkdir(parents=True, exist_ok=True)
    if 'b' not in mode and encoding is None:
      encoding = 'utf-8'
    return self._local.open(mode, encoding=encoding)

  def replace(self, target: '_FakeGcsPath') -> None:
    target.local_path.parent.mkdir(parents=True, exist_ok=True)
    self._local.replace(target.local_path)

  def unlink(self) -> None:
    self._local.unlink()


class TrajectoryLoggerTest(absltest.TestCase):
  def setUp(self):
    super().setUp()

  def test_log_item_with_none_log_path(self):
    """Tests that log_item with log_path=None raises ValueError."""
    item = {
        'global_step': 0,
        'trajectory_id': 't0',
        'completion': 'c0',
        'prompt': 'p0',
    }
    with self.assertRaisesRegex(
        ValueError, 'No directory for logging provided'
    ):
      trajectory_logger.log_item(None, item)

  def test_log_item_with_non_existent_dir_creates_dir(self):
    """Tests that log_item creates a non-existent directory."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name
    non_existent_path = os.path.join(temp_dir, 'non_existent')
    item = {
        'global_step': 0,
        'trajectory_id': 't0',
        'completion': 'c0',
        'prompt': 'p0',
    }
    trajectory_logger.log_item(non_existent_path, item)
    log_file = os.path.join(non_existent_path, 'trajectory_log.json')
    self.assertTrue(os.path.exists(log_file))

  def test_log_item_creates_and_writes_to_file(self):
    """Tests that log_item creates and writes to a log file (default json and csv)."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name
    item1 = dict(
        global_step=0,
        trajectory_id='t0',
        completion='c0',
        prompt='p0',
        value=np.int32(0),
    )
    trajectory_logger.log_item(temp_dir, item1)
    trajectory_logger.log_item(temp_dir, item1, file_format='csv')

    json_file = os.path.join(temp_dir, 'trajectory_log.json')
    csv_file = os.path.join(temp_dir, 'trajectory_log.csv')
    self.assertTrue(os.path.exists(json_file))
    self.assertTrue(os.path.exists(csv_file))

    item2 = dict(
        global_step=1,
        trajectory_id='t1',
        completion='c1|pipe',
        prompt='p1</reasoning>',
        value=np.int32(1),
    )
    trajectory_logger.log_item(temp_dir, item2)
    trajectory_logger.log_item(temp_dir, item2, file_format='csv')

    item3 = dict(
        global_step=2,
        trajectory_id='t2',
        completion='a, "b", c',
        prompt='a prompt with\na newline',
        value=np.int32(2),
    )
    trajectory_logger.log_item(temp_dir, item3)
    trajectory_logger.log_item(temp_dir, item3, file_format='csv')

    for label, df in (
        ('JSON default', pd.read_json(json_file)),
        ('CSV explicit', pd.read_csv(csv_file)),
    ):
      with self.subTest(label):
        self.assertLen(df, 3)
        self.assertEqual(df['trajectory_id'].tolist(), ['t0', 't1', 't2'])
        self.assertEqual(df['completion'][2], 'a, "b", c')
        self.assertEqual(df['prompt'][2], 'a prompt with\na newline')
        self.assertEqual(df['value'].tolist(), [0, 1, 2])

  def test_log_item_with_gcs_path_and_non_existent_dir(self):
    """Tests logging to a `gs://` prefix that has no objects yet."""
    temp_dir = self.create_tempdir().full_path

    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/run0'
    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      trajectory_logger.log_item(gcs_dir, dict(global_step=0, reward=0.0))
      trajectory_logger.log_item(gcs_dir, dict(global_step=1, reward=1.0))

    log_file = os.path.join(temp_dir, 'trajectories/run0/trajectory_log.json')
    self.assertTrue(os.path.exists(log_file))
    df = pd.read_json(log_file)
    self.assertEqual(df['global_step'].tolist(), [0, 1])
    self.assertEqual(df['reward'].tolist(), [0.0, 1.0])

  def test_async_trajectory_logger_logs_and_stops(self):
    """Tests that AsyncTrajectoryLogger writes items asynchronously and stops cleanly."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name

    logger = trajectory_logger.AsyncTrajectoryLogger(temp_dir)
    for i in range(5):
      logger.log_item_async({
          'global_step': i,
          'prompt_id': f'prompt_{i}',
          'reward': float(i),
      })
    logger.stop()

    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    self.assertLen(json_files, 1)
    df = pd.read_json(os.path.join(temp_dir, json_files[0]))
    self.assertLen(df, 5)
    self.assertEqual(df['global_step'].tolist(), list(range(5)))
    self.assertEqual(df['reward'].tolist(), [float(i) for i in range(5)])

  def test_async_trajectory_logger_stop_drain_race_condition(self):
    """Tests that stop() cleanly terminates without deadlock when queue has items and sentinel."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name

    logger = trajectory_logger.AsyncTrajectoryLogger(temp_dir)
    # Rapidly enqueue multiple items and immediately call stop to stress the queue batch-drain path
    for i in range(20):
      logger.log_item_async({'step': i, 'value': i * 2})
    logger.stop()

    self.assertTrue(logger._stopped)
    self.assertFalse(logger._logging_thread.is_alive())

    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    self.assertLen(json_files, 1)
    df = pd.read_json(os.path.join(temp_dir, json_files[0]))
    self.assertLen(df, 20)

  def test_async_trajectory_logger_stop_idempotent(self):
    """Tests that calling stop() multiple times is safe and idempotent."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name

    logger = trajectory_logger.AsyncTrajectoryLogger(temp_dir)
    logger.log_item_async({'step': 0})
    logger.stop()
    # Calling stop again should be a no-op
    logger.stop()
    self.assertTrue(logger._stopped)

  def test_async_trajectory_logger_log_after_stop(self):
    """Tests that logging after stop() does not crash."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name

    logger = trajectory_logger.AsyncTrajectoryLogger(temp_dir)
    logger.stop()
    # Should not raise exception
    logger.log_item_async({'step': 1})

  def test_log_item_gcs_read_stall_times_out(self):
    """Tests that a stalled GCS CSV read times out without overwriting existing rows."""
    temp_dir = self.create_tempdir().full_path
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/stall_test'

    def _stalled_read_csv(*args, **kwargs):
      del args, kwargs
      time.sleep(10.0)
      return [], []

    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      # Write initial row so file exists on fake GCS.
      trajectory_logger.log_item(
          gcs_dir, {'global_step': 0, 'reward': 0.0}, file_format='csv'
      )

      # Next call with stalled CSV read must time out in ~0.2s and not clobber
      # step 0 on remote GCS.
      start = time.monotonic()
      with mock.patch.object(
          trajectory_logger, '_read_csv_rows', side_effect=_stalled_read_csv
      ):
        trajectory_logger.log_item(
            gcs_dir,
            {'global_step': 1, 'reward': 1.0},
            file_format='csv',
            gcs_timeout_sec=0.2,
        )
      elapsed = time.monotonic() - start
      self.assertLess(elapsed, 2.0)

      # Subsequent call after GCS recovers preserves step 0 and appends step 2.
      trajectory_logger.log_item(
          gcs_dir, {'global_step': 2, 'reward': 2.0}, file_format='csv'
      )

    log_file = os.path.join(
        temp_dir, 'trajectories/stall_test/trajectory_log.csv'
    )
    self.assertTrue(os.path.exists(log_file))
    df = pd.read_csv(log_file)
    self.assertEqual(df['global_step'].tolist(), [0, 2])
    self.assertEqual(df['reward'].tolist(), [0.0, 2.0])

  def test_async_trajectory_logger_stop_does_not_deadlock_when_worker_blocked(
      self,
  ):
    """Tests that stop() returns within stop_timeout_sec when worker is blocked."""
    temp_dir = self.create_tempdir().full_path
    worker_entered = threading.Event()

    def _blocked_log_item(*args, **kwargs):
      del args, kwargs
      worker_entered.set()
      time.sleep(10.0)

    with mock.patch.object(
        trajectory_logger, 'log_item', side_effect=_blocked_log_item
    ):
      logger = trajectory_logger.AsyncTrajectoryLogger(
          temp_dir, stop_timeout_sec=0.3, gcs_timeout_sec=0.3
      )
      logger.log_item_async({'step': 0})
      self.assertTrue(worker_entered.wait(timeout=2.0))

      start = time.monotonic()
      logger.stop()
      elapsed = time.monotonic() - start

      self.assertLess(elapsed, 2.0)
      self.assertTrue(logger._stopped)

  def test_async_trajectory_logger_bounded_queue_does_not_block_or_deadlock_stop(
      self,
  ):
    """Tests that a full bounded queue neither blocks log_item_async nor deadlocks stop()."""
    temp_dir = self.create_tempdir().full_path
    worker_entered = threading.Event()

    def _blocked_log_item(*args, **kwargs):
      del args, kwargs
      worker_entered.set()
      time.sleep(10.0)

    with mock.patch.object(
        trajectory_logger, 'log_item', side_effect=_blocked_log_item
    ):
      logger = trajectory_logger.AsyncTrajectoryLogger(
          temp_dir, max_queue_size=3, stop_timeout_sec=0.2
      )
      logger.log_item_async({'step': 0})
      self.assertTrue(worker_entered.wait(timeout=2.0))

      # Enqueue more items than max_queue_size; must not block caller.
      start = time.monotonic()
      for i in range(1, 15):
        logger.log_item_async({'step': i})
      enqueue_elapsed = time.monotonic() - start
      self.assertLess(enqueue_elapsed, 1.0)

      # Calling stop() when queue is full must not block on queue.put(None).
      stop_start = time.monotonic()
      logger.stop()
      stop_elapsed = time.monotonic() - stop_start
      self.assertLess(stop_elapsed, 1.5)

  def test_async_trajectory_logger_gcs_local_staging_avoids_quadratic_reads(
      self,
  ):
    """Tests that AsyncTrajectoryLogger stages GCS CSVs locally without remote re-reading."""
    temp_dir = self.create_tempdir().full_path
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/staged_run'

    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      with mock.patch.object(
          trajectory_logger,
          '_read_gcs_csv',
          wraps=trajectory_logger._read_gcs_csv,
      ) as spy_read_gcs_csv:
        logger = trajectory_logger.AsyncTrajectoryLogger(
            gcs_dir, file_format='csv'
        )
        for step in range(5):
          # Alternate key insertion order to verify column alignment on append.
          item = (
              {'global_step': step, 'reward': float(step)}
              if step % 2 == 0
              else {'reward': float(step), 'global_step': step}
          )
          logger.log_item_async(item)
          # Allow worker to flush each step individually.
          time.sleep(0.05)
        logger.stop()
        # No remote file existed initially, so _read_gcs_csv should not be
        # called across all 5 flushes.
        self.assertEqual(spy_read_gcs_csv.call_count, 0)

    run_dir = os.path.join(temp_dir, 'trajectories/staged_run')
    csv_files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
    self.assertLen(csv_files, 1)
    df = pd.read_csv(os.path.join(run_dir, csv_files[0]))
    self.assertEqual(df['global_step'].tolist(), [0, 1, 2, 3, 4])
    self.assertEqual(df['reward'].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

  def test_log_item_gcs_staging_round_trips_non_ascii(self):
    """Tests that non-ASCII completions survive local staging under C locale."""
    temp_dir = self.create_tempdir().full_path
    staging_dir = self.create_tempdir().full_path
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/unicode_run'
    completions = ['héllo — wörld', '你好，世界', 'π ≈ 3.14 🚀']

    old_locale = locale.setlocale(locale.LC_CTYPE)
    try:
      locale.setlocale(locale.LC_CTYPE, 'C')
      with mock.patch.object(
          trajectory_logger.epath,
          'Path',
          lambda path: _FakeGcsPath(path, temp_dir),
      ):
        for step, completion in enumerate(completions):
          item = {'global_step': step, 'completion': completion}
          if step == 2:
            # Trigger schema-expansion rewrite path in local staging as well.
            item['extra_col'] = 'ñ'
          trajectory_logger.log_item(
              gcs_dir,
              item,
              file_format='csv',
              local_staging_dir=staging_dir,
          )
    finally:
      locale.setlocale(locale.LC_CTYPE, old_locale)

    log_file = os.path.join(
        temp_dir, 'trajectories/unicode_run/trajectory_log.csv'
    )
    df = pd.read_csv(log_file, encoding='utf-8')
    self.assertEqual(df['completion'].tolist(), completions)

  def test_stop_is_reentrant_from_same_thread(self):
    """Tests that re-entering stop() on the same thread does not self-deadlock."""
    temp_dir = self.create_tempdir().full_path
    logger = trajectory_logger.AsyncTrajectoryLogger(
        temp_dir, stop_timeout_sec=0.2
    )

    def _acquire_and_stop_on_same_thread():
      # Emulates a signal handler firing on the same thread while stop() holds
      # _stop_lock. With threading.Lock this blocks forever.
      with logger._stop_lock:
        logger.stop()

    t = threading.Thread(target=_acquire_and_stop_on_same_thread, daemon=True)
    t.start()
    t.join(timeout=1.0)
    self.assertFalse(
        t.is_alive(), 'stop() self-deadlocked on same-thread re-entrancy'
    )
    self.assertTrue(logger._stopped)

  def test_write_timeout_does_not_unlink_tmp_while_writer_alive(self):
    """Tests that a timed-out upload leaves the tmp file to the live writer."""
    temp_dir = self.create_tempdir().full_path
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/write_stall'
    writer_entered = threading.Event()
    unlinked = []

    real_unlink = _FakeGcsPath.unlink

    def _tracking_unlink(fake_path):
      unlinked.append(str(fake_path))
      return real_unlink(fake_path)

    def _stalled_replace(fake_path, target):
      del fake_path, target
      writer_entered.set()
      time.sleep(10.0)

    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ), mock.patch.object(
        _FakeGcsPath, 'unlink', _tracking_unlink
    ), mock.patch.object(
        _FakeGcsPath, 'replace', _stalled_replace
    ):
      trajectory_logger.log_item(
          gcs_dir, {'global_step': 0}, gcs_timeout_sec=0.2
      )
      self.assertTrue(writer_entered.wait(timeout=2.0))
      self.assertEmpty(unlinked)

  def test_make_serializable_ndarray_avoids_elementwise_recursion(self):
    """Tests that _make_serializable does not recurse on every element of numeric ndarrays."""
    routed_experts = np.arange(16 * 40 * 8, dtype=np.int32).reshape(16, 40, 8)
    with mock.patch.object(
        trajectory_logger,
        '_make_serializable',
        wraps=trajectory_logger._make_serializable,
    ) as spy_make_serializable:
      result = trajectory_logger._make_serializable(
          {'routed_experts': routed_experts, 'scalar': np.int64(7)}
      )
      # 1 call for outer dict + 1 call for 'routed_experts' ndarray
      # + 1 call for 'scalar' = 3 calls
      self.assertEqual(spy_make_serializable.call_count, 3)

    self.assertEqual(result['routed_experts'], routed_experts.tolist())
    self.assertEqual(result['scalar'], 7)

  def test_nanobind_format_ndarray_and_csv_parity(self):
    """Tests that C++ nanobind ndarray formatting matches Python list repr and CSV output."""
    arr_3d = np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2) - 10
    arr_non_contig = arr_3d[:, ::2, :]
    arr_float = np.array(
        [[0.0, 1.5, -2.25], [3.0, 1e-4, 42.0]], dtype=np.float64
    )
    arr_fp16 = np.array([[0.5, -1.25], [2.0, 3.5]], dtype=np.float16)
    arr_bool = np.array([[True, False], [False, True]], dtype=np.bool_)
    arr_empty = np.zeros((2, 0), dtype=np.int64)

    for arr in (
        arr_3d,
        arr_non_contig,
        arr_float,
        arr_fp16,
        arr_bool,
        arr_empty,
    ):
      formatted = trajectory_logger._format_ndarray_for_csv(arr)
      self.assertEqual(str(formatted), str(arr.tolist()))
      self.assertEqual(repr(formatted), repr(arr.tolist()))

    # Also verify _make_csv_serializable handles float16 arrays without
    # TypeError.
    csv_fp16 = trajectory_logger._make_csv_serializable({'fp16': arr_fp16})
    self.assertEqual(str(csv_fp16['fp16']), str(arr_fp16.tolist()))

    # Verify nested dict repr parity when containing formatted arrays.
    nested = {
        'steps': [{
            'routed_experts': trajectory_logger._format_ndarray_for_csv(arr_3d)
        }]
    }
    expected_nested = {'steps': [{'routed_experts': arr_3d.tolist()}]}
    self.assertEqual(str(nested), str(expected_nested))

  def test_dumps_json_with_large_ndarrays_and_nested_types(self):
    """Tests GIL-free C++ dumps_json on trajectory payloads with ndarrays."""

    class _CustomTag:

      def __str__(self) -> str:
        return 'custom_tag_value'

    routed_experts = np.arange(8 * 40 * 8, dtype=np.int32).reshape(8, 40, 8)
    logprobs = np.linspace(-2.5, -0.1, 16, dtype=np.float32)
    fp16_scores = np.array([0.5, -1.25, 2.0, 3.5], dtype=np.float16)
    payload = {
        'global_step': 3,
        'traj_id': 'traj_42_g0',
        'status': 'COMPLETED',
        'is_valid': True,
        'np_flag': np.bool_(False),
        'custom_obj': _CustomTag(),
        'reward': 1.0,
        'unicode_text': 'héllo — 你好 🚀 "quoted" \n newline',
        'prompt_tokens': np.array([101, 102, 103], dtype=np.int32),
        'fp16_scores': fp16_scores,
        'steps': [{
            'step_index': 0,
            'assistant_tokens': np.array([201, 202, 203, 204], dtype=np.int64),
            'assistant_logprobs': logprobs,
            'routed_experts': routed_experts,
            'flags': np.array([True, False, True], dtype=np.bool_),
        }],
    }

    compact_json = trajectory_logger.dumps_json(payload)
    pretty_json = trajectory_logger.dumps_json(payload, indent=2)

    parsed_compact = json.loads(compact_json)
    parsed_pretty = json.loads(pretty_json)
    self.assertEqual(parsed_compact, parsed_pretty)
    self.assertEqual(parsed_compact['global_step'], 3)
    self.assertTrue(parsed_compact['is_valid'])
    self.assertFalse(parsed_compact['np_flag'])
    self.assertEqual(parsed_compact['custom_obj'], 'custom_tag_value')
    self.assertEqual(parsed_compact['unicode_text'], payload['unicode_text'])
    self.assertEqual(parsed_compact['prompt_tokens'], [101, 102, 103])
    self.assertEqual(parsed_compact['fp16_scores'], fp16_scores.tolist())
    self.assertEqual(
        parsed_compact['steps'][0]['routed_experts'],
        routed_experts.tolist(),
    )
    self.assertEqual(
        parsed_compact['steps'][0]['flags'],
        [True, False, True],
    )
    np.testing.assert_allclose(
        parsed_compact['steps'][0]['assistant_logprobs'],
        logprobs.tolist(),
        rtol=1e-6,
    )

    # Verify cyclic / excessively nested payloads raise an error rather than
    # crashing.
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with self.assertRaises((ValueError, RecursionError)):
      trajectory_logger.dumps_json(cyclic)

  def test_trajectory_logger_ext_is_available(self):
    """Tests that the C++ extension is built in every build, including OSS."""
    self.assertIsNotNone(trajectory_logger._trajectory_logger_ext)

  def test_pure_python_fallback_matches_extension(self):
    """Tests that the pure-Python fallback serializes like the C++ extension."""
    arrays = {
        'routed_experts': (
            np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2) - 10
        ),
        'scores': np.array(
            [[0.0, 1.5, -2.25], [3.0, 1e-4, 42.0]], dtype=np.float64
        ),
        'fp16_scores': np.array([[0.5, -1.25], [2.0, 3.5]], dtype=np.float16),
        'flags': np.array([[True, False], [False, True]], dtype=np.bool_),
        'empty': np.zeros((2, 0), dtype=np.int64),
    }
    payload = {
        'global_step': 3,
        'is_valid': True,
        'np_flag': np.bool_(False),
        'reward': 1.0,
        'unicode_text': 'héllo — 你好 🚀 "quoted" \n newline',
        'steps': [arrays],
    }
    expected_csv = {
        name: str(trajectory_logger._format_ndarray_for_csv(arr))
        for name, arr in arrays.items()
    }
    expected_json = json.loads(trajectory_logger.dumps_json(payload))

    with mock.patch.object(trajectory_logger, '_trajectory_logger_ext', None):
      for name, arr in arrays.items():
        formatted = trajectory_logger._format_ndarray_for_csv(arr)
        self.assertIsInstance(formatted, str)
        self.assertEqual(str(formatted), expected_csv[name])
        self.assertEqual(repr(formatted), expected_csv[name])
      self.assertEqual(
          json.loads(trajectory_logger.dumps_json(payload)), expected_json
      )
      self.assertEqual(
          json.loads(trajectory_logger.dumps_json(payload, indent=2)),
          expected_json,
      )

  def test_async_trajectory_logger_large_ndarray_csv_and_concurrency(self):
    """Tests that AsyncTrajectoryLogger logs large 3D routed_experts arrays without starving main thread."""
    temp_dir = self.create_tempdir().full_path
    routed_experts = np.arange(128 * 40 * 8, dtype=np.int32).reshape(128, 40, 8)
    prompt_tokens = np.arange(256, dtype=np.int32)

    logger = trajectory_logger.AsyncTrajectoryLogger(
        temp_dir, file_format='csv'
    )
    for step in range(4):
      logger.log_item_async({
          'global_step': step,
          'prompt_tokens': prompt_tokens,
          'trajectory': {'routed_experts': routed_experts},
      })

    # Measure main-thread progress while background thread serializes arrays.
    ticks = 0
    max_tick_gap_sec = 0.0
    while not logger._logging_queue.empty():
      t0 = time.monotonic()
      time.sleep(0.002)
      gap = time.monotonic() - t0
      max_tick_gap_sec = max(max_tick_gap_sec, gap)
      ticks += 1
    logger.stop()

    self.assertLess(max_tick_gap_sec, 0.5)
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    self.assertLen(csv_files, 1)
    df = pd.read_csv(os.path.join(temp_dir, csv_files[0]))
    self.assertLen(df, 4)
    self.assertEqual(df['prompt_tokens'][0], str(prompt_tokens.tolist()))
    self.assertEqual(
        df['trajectory'][0],
        str({'routed_experts': routed_experts.tolist()}),
    )

  def test_perf_benchmark_old_vs_new_writer(self):
    """Benchmarks legacy Python serialization vs C++ nanobind GIL-free writer."""

    def _legacy_make_serializable(item):
      if isinstance(item, dict):
        return {k: _legacy_make_serializable(v) for k, v in item.items()}
      elif isinstance(item, list):
        return [_legacy_make_serializable(x) for x in item]
      elif isinstance(item, tuple):
        return tuple(_legacy_make_serializable(x) for x in item)
      elif isinstance(item, np.ndarray):
        return _legacy_make_serializable(item.tolist())
      elif isinstance(item, np.integer):
        return int(item)
      elif isinstance(item, np.floating):
        return float(item)
      elif isinstance(item, np.bool_):
        return bool(item)
      elif isinstance(item, np.str_):
        return str(item)
      elif isinstance(item, (float, int, bool, str)) or item is None:
        return item
      return str(item)

    # Simulate 16 trajectories, each with [512, 40, 8] int32 routed_experts
    # (163,840 ints per trajectory -> 2,621,440 ints total across the batch).
    num_trajs = 16
    seq_len = 512
    routed_experts = np.arange(seq_len * 40 * 8, dtype=np.int32).reshape(
        seq_len, 40, 8
    )
    prompt_tokens = np.arange(256, dtype=np.int32)
    completion_tokens = np.arange(seq_len, dtype=np.int32)
    logprobs = np.linspace(-2.0, -0.05, seq_len, dtype=np.float32)

    batch = [
        {
            'global_step': 1,
            'prompt_id': f'q_{i}',
            'group_index': i % 4,
            'status': 'COMPLETED',
            'reward': 1.0,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'trajectory': {
                'steps': [{
                    'step_index': 0,
                    'assistant_tokens': completion_tokens,
                    'assistant_logprobs': logprobs,
                    'routed_experts': routed_experts,
                }]
            },
        }
        for i in range(num_trajs)
    ]

    def _measure_with_main_thread_probe(work_fn):
      stop_probe = threading.Event()
      tick_gaps_ms: list[float] = []
      ticks = 0

      def _main_thread_loop():
        nonlocal ticks
        while not stop_probe.is_set():
          t0 = time.perf_counter()
          time.sleep(0.001)
          # Simulate light Python orchestrator work (acquiring GIL)
          _ = sum(range(50))
          tick_gaps_ms.append((time.perf_counter() - t0) * 1000.0)
          ticks += 1

      probe_thread = threading.Thread(target=_main_thread_loop, daemon=True)
      probe_thread.start()
      t_start = time.perf_counter()
      work_fn()
      wall_ms = (time.perf_counter() - t_start) * 1000.0
      stop_probe.set()
      probe_thread.join()
      max_stall_ms = max(tick_gaps_ms) if tick_gaps_ms else 0.0
      p99_stall_ms = (
          float(np.percentile(tick_gaps_ms, 99)) if tick_gaps_ms else 0.0
      )
      return wall_ms, max_stall_ms, p99_stall_ms, ticks

    temp_dir = self.create_tempdir().full_path

    # 1. Benchmark CSV Writer (log_item): Legacy vs Python Fix vs C++ Nanobind
    def _csv_legacy():
      out_file = os.path.join(temp_dir, 'legacy.csv')
      serialized = _legacy_make_serializable(batch)
      df = pd.DataFrame(serialized)
      df.to_csv(out_file, index=False)

    def _csv_py_tolist():
      out_file = os.path.join(temp_dir, 'py_tolist.csv')
      serialized = trajectory_logger._make_serializable(batch)
      df = pd.DataFrame(serialized)
      df.to_csv(out_file, index=False)

    def _csv_cpp_nanobind():
      out_file = os.path.join(temp_dir, 'cpp_nanobind.csv')
      serialized = trajectory_logger._make_csv_serializable(batch)
      df = pd.DataFrame(serialized)
      df.to_csv(out_file, index=False)

    # 2. Benchmark 8-Thread JSON Writer (simulating multi-worker JSON pool)
    def _json_8threads_legacy():
      with futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(
                lambda it: json.dumps(_legacy_make_serializable(it), indent=2),
                item,
            )
            for item in batch
        ]
        futures.wait(futs)

    def _json_8threads_py_tolist():
      with futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(
                lambda it: json.dumps(
                    trajectory_logger._make_serializable(it), indent=2
                ),
                item,
            )
            for item in batch
        ]
        futures.wait(futs)

    def _json_8threads_cpp_nanobind():
      with futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [
            ex.submit(
                lambda it: trajectory_logger.dumps_json(it, indent=2), item
            )
            for item in batch
        ]
        futures.wait(futs)

    # Warmup C++ extension
    _ = trajectory_logger.dumps_json(batch[0])
    _ = trajectory_logger._make_csv_serializable(batch[0])

    csv_leg = _measure_with_main_thread_probe(_csv_legacy)
    csv_py = _measure_with_main_thread_probe(_csv_py_tolist)
    csv_cpp = _measure_with_main_thread_probe(_csv_cpp_nanobind)

    json_leg = _measure_with_main_thread_probe(_json_8threads_legacy)
    json_py = _measure_with_main_thread_probe(_json_8threads_py_tolist)
    json_cpp = _measure_with_main_thread_probe(_json_8threads_cpp_nanobind)

    print('\n' + '=' * 88)
    print(
        'BENCHMARK: 16 Trajectories x [512, 40, 8] int32 routed_experts '
        '(2,621,440 ints / batch)'
    )
    print('=' * 88)
    print(
        f'{"Mode / Implementation":<34} | {"Wall (ms)":>10} | {"Speedup":>8} | '
        f'{"Max GIL Stall (ms)":>18} | {"P99 Stall (ms)":>14}'
    )
    print('-' * 88)
    for label, res, base_ms in [
        ('CSV: Legacy Python (recursive)', csv_leg, csv_leg[0]),
        ('CSV: Python O(1) tolist()', csv_py, csv_leg[0]),
        ('CSV: C++ nanobind (nogil)', csv_cpp, csv_leg[0]),
        ('JSON (8 threads): Legacy Python', json_leg, json_leg[0]),
        ('JSON (8 threads): Python tolist()', json_py, json_leg[0]),
        ('JSON (8 threads): C++ nanobind', json_cpp, json_leg[0]),
    ]:
      wall_ms, max_stall, p99_stall, _ = res
      speedup = base_ms / wall_ms if wall_ms > 0 else 0.0
      print(
          f'{label:<34} | {wall_ms:>10.2f} | {speedup:>7.2f}x | '
          f'{max_stall:>18.2f} | {p99_stall:>14.2f}'
      )
    print('=' * 88 + '\n')

  def test_log_item_jsonl_and_json_formats_use_dumps_json(self):
    """Tests that log_item supports jsonl and json formats with multi-worker dumps_json."""
    temp_dir = self.create_tempdir().full_path
    routed_experts = np.arange(4 * 40 * 8, dtype=np.int32).reshape(4, 40, 8)
    batch1 = [
        {
            'global_step': 0,
            'prompt_id': f'p_{i}',
            'reward': float(i),
            'routed_experts': routed_experts,
        }
        for i in range(4)
    ]
    batch2 = [
        {
            'global_step': 1,
            'prompt_id': f'p_{i}',
            'reward': float(i + 10),
            'routed_experts': routed_experts,
        }
        for i in range(2)
    ]

    with mock.patch.object(
        trajectory_logger, 'dumps_json', wraps=trajectory_logger.dumps_json
    ) as spy_dumps_json:
      trajectory_logger.log_item(
          temp_dir, batch1, 'run1', file_format='jsonl', num_workers=4
      )
      trajectory_logger.log_item(
          temp_dir, batch2, 'run1', file_format='jsonl', num_workers=4
      )
      self.assertEqual(spy_dumps_json.call_count, 6)

    jsonl_path = os.path.join(temp_dir, 'trajectory_log_run1.jsonl')
    self.assertTrue(os.path.exists(jsonl_path))
    with open(jsonl_path, 'r', encoding='utf-8') as f:
      lines = [json.loads(line) for line in f if line.strip()]
    self.assertLen(lines, 6)
    self.assertEqual([r['global_step'] for r in lines], [0, 0, 0, 0, 1, 1])
    self.assertEqual(lines[0]['routed_experts'], routed_experts.tolist())

    # Also test file_format='json' across incremental appends.
    trajectory_logger.log_item(
        temp_dir, batch1, 'run1', file_format='json', num_workers=4
    )
    trajectory_logger.log_item(
        temp_dir, batch2, 'run1', file_format='json', num_workers=4
    )
    json_path = os.path.join(temp_dir, 'trajectory_log_run1.json')
    self.assertTrue(os.path.exists(json_path))
    with open(json_path, 'r', encoding='utf-8') as f:
      records = json.load(f)
    self.assertLen(records, 6)
    self.assertEqual(
        [r['reward'] for r in records], [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]
    )
    self.assertEqual(records[-1]['routed_experts'], routed_experts.tolist())

  def test_async_trajectory_logger_jsonl_and_json_local_and_gcs(self):
    """Tests AsyncTrajectoryLogger in jsonl and json formats on local and GCS paths."""
    temp_dir = self.create_tempdir().full_path
    routed_experts = np.arange(8 * 40 * 8, dtype=np.int32).reshape(8, 40, 8)

    # 1. Local JSONL logger with 4 worker threads
    local_jsonl_dir = os.path.join(temp_dir, 'local_jsonl')
    logger_jsonl = trajectory_logger.AsyncTrajectoryLogger(
        local_jsonl_dir, file_format='jsonl', num_workers=4
    )
    for step in range(6):
      logger_jsonl.log_item_async({
          'global_step': step,
          'reward': float(step) * 0.5,
          'routed_experts': routed_experts,
      })
    logger_jsonl.stop()

    jsonl_files = [
        f for f in os.listdir(local_jsonl_dir) if f.endswith('.jsonl')
    ]
    self.assertLen(jsonl_files, 1)
    with open(
        os.path.join(local_jsonl_dir, jsonl_files[0]), 'r', encoding='utf-8'
    ) as f:
      rows = [json.loads(line) for line in f if line.strip()]
    self.assertLen(rows, 6)
    self.assertEqual([r['global_step'] for r in rows], list(range(6)))
    self.assertEqual(rows[0]['routed_experts'], routed_experts.tolist())

    # 2. Fake GCS JSON logger with local staging and 4 worker threads
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/json_staged_run'
    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      logger_json = trajectory_logger.AsyncTrajectoryLogger(
          gcs_dir, file_format='json', num_workers=4
      )
      for step in range(4):
        logger_json.log_item_async({
            'global_step': step,
            'prompt_id': f'q_{step}',
            'reward': float(step),
        })
        time.sleep(0.03)
      logger_json.stop()

    gcs_run_dir = os.path.join(temp_dir, 'trajectories/json_staged_run')
    json_files = [f for f in os.listdir(gcs_run_dir) if f.endswith('.json')]
    self.assertLen(json_files, 1)
    with open(
        os.path.join(gcs_run_dir, json_files[0]), 'r', encoding='utf-8'
    ) as f:
      gcs_rows = json.load(f)
    self.assertLen(gcs_rows, 4)
    self.assertEqual([r['global_step'] for r in gcs_rows], [0, 1, 2, 3])

  def test_invalid_file_format_or_num_workers_raises(self):
    """Tests that invalid file_format or num_workers raises ValueError."""
    temp_dir = self.create_tempdir().full_path
    invalid_fmt_parquet: object = 'parquet'
    invalid_fmt_xml: object = 'xml'
    with self.assertRaisesRegex(ValueError, 'Unsupported file_format'):
      trajectory_logger.log_item(
          temp_dir, {'step': 0}, file_format=invalid_fmt_parquet  # pyrefly: ignore[bad-argument-type]
      )
    with self.assertRaisesRegex(ValueError, 'num_workers must be >= 1'):
      trajectory_logger.log_item(temp_dir, {'step': 0}, num_workers=0)
    with self.assertRaisesRegex(ValueError, 'Unsupported file_format'):
      trajectory_logger.AsyncTrajectoryLogger(
          temp_dir, file_format=invalid_fmt_xml  # pyrefly: ignore[bad-argument-type]
      )
    with self.assertRaisesRegex(ValueError, 'num_workers must be >= 1'):
      trajectory_logger.AsyncTrajectoryLogger(temp_dir, num_workers=0)

  def test_multi_worker_parallel_execution_and_order_preservation(self):
    """Tests that multiple worker threads execute concurrently and preserve item order."""
    temp_dir = self.create_tempdir().full_path
    barrier = threading.Barrier(4, timeout=5.0)
    warmup_entered = threading.Event()
    release_warmup = threading.Event()
    observed_threads: set[int] = set()
    observed_names: set[str] = set()
    lock = threading.Lock()
    real_dumps_json = trajectory_logger.dumps_json

    def _concurrent_dumps_json(item, *, indent=None):
      if item['global_step'] == -1:
        warmup_entered.set()
        release_warmup.wait(timeout=5.0)
        return real_dumps_json(item, indent=indent)
      with lock:
        observed_threads.add(threading.get_ident())
        observed_names.add(threading.current_thread().name)
      # All 4 worker threads must reach this barrier simultaneously; if
      # execution were serial, this would raise BrokenBarrierError.
      barrier.wait()
      # Finish in reverse order (item 0 sleeps longest, item 3 finishes first)
      # to verify that output order in the file is still strictly [0, 1, 2, 3].
      time.sleep((3 - item['global_step']) * 0.01)
      return real_dumps_json(item, indent=indent)

    batch = [
        {
            'global_step': i,
            'reward': float(i),
            'tokens': np.arange(32, dtype=np.int32) + i,
        }
        for i in range(4)
    ]

    with mock.patch.object(
        trajectory_logger, 'dumps_json', side_effect=_concurrent_dumps_json
    ):
      logger = trajectory_logger.AsyncTrajectoryLogger(
          temp_dir, file_format='jsonl', num_workers=4
      )
      self.assertIsNotNone(logger._executor)
      # Hold worker on warmup item so all 4 batch items accumulate in queue.
      logger.log_item_async({'global_step': -1, 'reward': 0.0, 'tokens': []})
      self.assertTrue(warmup_entered.wait(timeout=5.0))
      for item in batch:
        logger.log_item_async(item)
      release_warmup.set()
      logger.stop()

    self.assertLen(observed_threads, 4)
    self.assertTrue(
        all(name.startswith('tunix_traj_ser') for name in observed_names),
        f'Expected tunix_traj_ser worker threads, got {observed_names}',
    )

    jsonl_files = [f for f in os.listdir(temp_dir) if f.endswith('.jsonl')]
    self.assertLen(jsonl_files, 1)
    with open(
        os.path.join(temp_dir, jsonl_files[0]), 'r', encoding='utf-8'
    ) as f:
      rows = [json.loads(line) for line in f if line.strip()]
    self.assertEqual([r['global_step'] for r in rows], [-1, 0, 1, 2, 3])

    # Also verify num_workers=1 does not allocate a ThreadPoolExecutor.
    single_dir = os.path.join(temp_dir, 'single_worker')
    single_logger = trajectory_logger.AsyncTrajectoryLogger(
        single_dir, file_format='jsonl', num_workers=1
    )
    self.assertIsNone(single_logger._executor)
    single_logger.log_item_async({'global_step': 0})
    single_logger.stop()


if __name__ == '__main__':
  absltest.main()
