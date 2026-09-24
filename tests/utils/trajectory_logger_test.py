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

  def write_text(self, data: str, encoding: str = 'utf-8') -> int:
    self._local.parent.mkdir(parents=True, exist_ok=True)
    return self._local.write_text(data, encoding=encoding)

  def read_text(self, encoding: str = 'utf-8') -> str:
    return self._local.read_text(encoding=encoding)

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
    log_file = os.path.join(non_existent_path, 'trajectory_log.csv')
    self.assertTrue(os.path.exists(log_file))

  def test_log_item_creates_and_writes_to_file(self):
    """Tests that log_item creates and writes to a log file."""
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

    log_file = os.path.join(temp_dir, 'trajectory_log.csv')
    self.assertTrue(os.path.exists(log_file))

    item2 = dict(
        global_step=1,
        trajectory_id='t1',
        completion='c1|pipe',
        prompt='p1</reasoning>',
        value=np.int32(1),
    )
    trajectory_logger.log_item(temp_dir, item2)

    item3 = dict(
        global_step=2,
        trajectory_id='t2',
        completion='a, "b", c',
        prompt='a prompt with\na newline',
        value=np.int32(2),
    )
    trajectory_logger.log_item(temp_dir, item3)

    df = pd.read_csv(log_file)
    with self.subTest('DataFrame content'):
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

    log_file = os.path.join(temp_dir, 'trajectories/run0/trajectory_log.csv')
    self.assertTrue(os.path.exists(log_file))
    df = pd.read_csv(log_file)
    self.assertEqual(df['global_step'].tolist(), [0, 1])
    self.assertEqual(df['reward'].tolist(), [0.0, 1.0])

  def test_async_trajectory_logger_logs_and_stops(self):
    """Tests that AsyncTrajectoryLogger writes JSON files asynchronously and stops cleanly."""
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
          'worker_id': 'worker0',
      })
    logger.stop()

    # CSV files must be disabled by default
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    self.assertEmpty(csv_files)

    # Verify JSON directory hierarchy: step{i}/worker0/traj_prompt_{i}_g0/
    for i in range(5):
      traj_dir = os.path.join(
          temp_dir, f'step{i}', 'worker0', f'traj_prompt_{i}_g0'
      )
      self.assertTrue(
          os.path.isdir(traj_dir), f'Expected directory {traj_dir} to exist.'
      )
      meta_file = os.path.join(traj_dir, 'metadata.json')
      step_file = os.path.join(traj_dir, 'step0.json')
      self.assertTrue(os.path.exists(meta_file))
      self.assertTrue(os.path.exists(step_file))

      with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
      self.assertEqual(meta['global_step'], i)
      self.assertEqual(meta['prompt_id'], f'prompt_{i}')
      self.assertEqual(meta['reward'], float(i))

  def test_async_trajectory_logger_csv_fallback(self):
    """Tests that AsyncTrajectoryLogger writes CSV when log_format='csv' is explicitly set."""
    try:
      temp_dir = self.create_tempdir().full_path
    except Exception:
      temp_dir = tempfile.TemporaryDirectory().name

    logger = trajectory_logger.AsyncTrajectoryLogger(
        temp_dir, log_format='csv'
    )
    for i in range(5):
      logger.log_item_async({
          'global_step': i,
          'prompt_id': f'prompt_{i}',
          'reward': float(i),
      })
    logger.stop()

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    self.assertLen(csv_files, 1)
    df = pd.read_csv(os.path.join(temp_dir, csv_files[0]))
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
      logger.log_item_async({
          'global_step': 0,
          'prompt_id': f'p_{i}',
          'worker_id': 'worker0',
          'value': i * 2,
      })
    logger.stop()

    self.assertTrue(logger._stopped)
    self.assertFalse(logger._logging_thread.is_alive())

    # Verify all 20 trajectory folders exist with metadata.json and step0.json
    for i in range(20):
      traj_dir = os.path.join(
          temp_dir, 'step0', 'worker0', f'traj_p_{i}_g0'
      )
      self.assertTrue(os.path.exists(os.path.join(traj_dir, 'metadata.json')))
      self.assertTrue(os.path.exists(os.path.join(traj_dir, 'step0.json')))

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
      return pd.DataFrame()

    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      # Write initial row so file exists on fake GCS.
      trajectory_logger.log_item(gcs_dir, {'global_step': 0, 'reward': 0.0})

      # Next call with stalled read_csv must time out in ~0.2s and not clobber
      # step 0 on remote GCS.
      start = time.monotonic()
      with mock.patch.object(pd, 'read_csv', side_effect=_stalled_read_csv):
        trajectory_logger.log_item(
            gcs_dir,
            {'global_step': 1, 'reward': 1.0},
            gcs_timeout_sec=0.2,
        )
      elapsed = time.monotonic() - start
      self.assertLess(elapsed, 2.0)

      # Subsequent call after GCS recovers preserves step 0 and appends step 2.
      trajectory_logger.log_item(gcs_dir, {'global_step': 2, 'reward': 2.0})

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
          temp_dir, log_format='csv', stop_timeout_sec=0.3, gcs_timeout_sec=0.3
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
          temp_dir, log_format='csv', max_queue_size=3, stop_timeout_sec=0.2
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
            gcs_dir, log_format='csv'
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

  def test_make_serializable_and_sanitize_path_segment(self):
    """Tests that _make_serializable and _sanitize_path_segment handle edge cases."""
    self.assertEqual(
        trajectory_logger._sanitize_path_segment('worker/0:test..'),
        'worker_0_test__',
    )
    self.assertEqual(
        trajectory_logger._sanitize_path_segment('   '), 'unknown'
    )
    self.assertEqual(
        trajectory_logger._sanitize_path_segment(None, default='fallback'),
        'fallback',
    )

    sample = {
        123: np.int64(42),
        'arr': np.array([1.0, 2.5]),
        'b': np.bool_(True),
        's': np.str_('hello'),
        'nested': {'key': None},
    }
    serialized = trajectory_logger._make_serializable(sample)
    self.assertEqual(
        serialized,
        {
            '123': 42,
            'arr': [1.0, 2.5],
            'b': True,
            's': 'hello',
            'nested': {'key': None},
        },
    )

  def test_log_trajectory_json_multiturn_conversation(self):
    """Tests that log_trajectory_json correctly extracts multi-turn conversation steps."""
    temp_dir = self.create_tempdir().full_path
    item = {
        'global_step': 1,
        'prompt_id': 'issue_42',
        'group_index': 0,
        'worker_id': 'worker_a',
        'traj_id': 'traj_custom_1',
        'status': 'RESOLVED',
        'reward': 1.0,
        'trajectory': {
            'status': 'RESOLVED',
            'conversation_text': [
                {'role': 'system', 'content': 'You are an agent.'},
                {'role': 'user', 'content': 'Solve issue #42.'},
                {'role': 'assistant', 'content': 'I will run git status.'},
                {'role': 'user', 'content': 'On branch main.'},
                {'role': 'assistant', 'content': 'I will edit the code.'},
                {'role': 'tool', 'content': 'File updated.'},
            ],
            'env_time': {
                'step_latency': [2.5, 4.1],
            },
        },
    }

    out_dir = trajectory_logger.log_trajectory_json(temp_dir, item)
    self.assertIsNotNone(out_dir)

    expected_dir = os.path.join(
        temp_dir, 'step1', 'worker_a', 'traj_custom_1'
    )
    self.assertEqual(out_dir, expected_dir)
    self.assertTrue(os.path.isdir(expected_dir))

    # Check metadata.json
    with open(os.path.join(expected_dir, 'metadata.json'), 'r') as f:
      metadata = json.load(f)
    self.assertEqual(metadata['global_step'], 1)
    self.assertEqual(metadata['worker_id'], 'worker_a')
    self.assertEqual(metadata['traj_id'], 'traj_custom_1')
    self.assertEqual(metadata['status'], 'RESOLVED')
    self.assertEqual(metadata['reward'], 1.0)
    self.assertEqual(metadata['num_steps'], 2)

    # Check step0.json
    with open(os.path.join(expected_dir, 'step0.json'), 'r') as f:
      step0 = json.load(f)
    self.assertEqual(step0['step_index'], 0)
    self.assertEqual(step0['thought_and_action'], 'I will run git status.')
    self.assertEqual(step0['observation'], 'On branch main.')
    self.assertEqual(step0['latency_sec'], 2.5)

    # Check step1.json
    with open(os.path.join(expected_dir, 'step1.json'), 'r') as f:
      step1 = json.load(f)
    self.assertEqual(step1['step_index'], 1)
    self.assertEqual(step1['thought_and_action'], 'I will edit the code.')
    self.assertEqual(step1['observation'], 'File updated.')
    self.assertEqual(step1['latency_sec'], 4.1)

  def test_log_trajectory_json_with_explicit_steps(self):
    """Tests log_trajectory_json when explicit steps list is provided."""
    temp_dir = self.create_tempdir().full_path
    item = {
        'global_step': 2,
        'prompt_id': 'test_p',
        'worker_id': 'worker1',
        'trajectory': {
            'steps': [
                {'action': 'click', 'obs': 'clicked'},
                {'action': 'type', 'obs': 'typed text'},
            ],
            'trajectory_reward': 0.8,
        },
    }

    out_dir = trajectory_logger.log_trajectory_json(temp_dir, item)
    self.assertIsNotNone(out_dir)

    expected_dir = os.path.join(
        temp_dir, 'step2', 'worker1', 'traj_test_p_g0'
    )
    self.assertTrue(os.path.isdir(expected_dir))

    with open(os.path.join(expected_dir, 'metadata.json'), 'r') as f:
      meta = json.load(f)
    self.assertEqual(meta['num_steps'], 2)
    self.assertEqual(meta['reward'], 0.8)

    with open(os.path.join(expected_dir, 'step0.json'), 'r') as f:
      s0 = json.load(f)
    self.assertEqual(s0['step_index'], 0)
    self.assertEqual(s0['action'], 'click')
    self.assertEqual(s0['obs'], 'clicked')

    with open(os.path.join(expected_dir, 'step1.json'), 'r') as f:
      s1 = json.load(f)
    self.assertEqual(s1['step_index'], 1)
    self.assertEqual(s1['action'], 'type')
    self.assertEqual(s1['obs'], 'typed text')

  def test_log_trajectory_json_gcs_path(self):
    """Tests logging JSON trajectory hierarchy to a simulated gs:// URI."""
    temp_dir = self.create_tempdir().full_path
    gcs_dir = f'{_FAKE_GCS_ROOT}/trajectories/run_gcs'

    item = {
        'global_step': 3,
        'prompt_id': 'p3',
        'worker_id': 'worker2',
        'trajectory': {
            'steps': [{'action': 'run_tool', 'output': 'ok'}],
        },
    }

    with mock.patch.object(
        trajectory_logger.epath,
        'Path',
        lambda path: _FakeGcsPath(path, temp_dir),
    ):
      out_dir = trajectory_logger.log_trajectory_json(gcs_dir, item)
      self.assertEqual(
          out_dir, f'{gcs_dir}/step3/worker2/traj_p3_g0'
      )

    local_traj_dir = os.path.join(
        temp_dir, 'trajectories/run_gcs/step3/worker2/traj_p3_g0'
    )
    self.assertTrue(os.path.exists(os.path.join(local_traj_dir, 'metadata.json')))
    self.assertTrue(os.path.exists(os.path.join(local_traj_dir, 'step0.json')))

  def test_log_trajectory_json_inference_metrics(self):
    """Tests that log_trajectory_json writes inference_metrics.json and inference_metrics.jsonl."""
    temp_dir = self.create_tempdir().full_path
    item = {
        'global_step': 5,
        'prompt_id': 'issue_99',
        'group_index': 1,
        'worker_id': 'rollout_worker_0',
        'traj_id': 'traj_issue_99_g1',
        'status': 'RESOLVED',
        'reward': 1.0,
        'trajectory': {
            'status': 'RESOLVED',
            'conversation_text': [
                {'role': 'system', 'content': 'System prompt.'},
                {'role': 'user', 'content': 'User prompt.'},
                {'role': 'assistant', 'content': 'Step 0 action.'},
                {'role': 'tool', 'content': 'Step 0 obs.'},
                {'role': 'assistant', 'content': 'Step 1 action.'},
            ],
            'model_time': {
                'step_latency': [1.2, 0.8],
                'prompt_tokens': [100, 150],
                'completion_tokens': [24, 16],
                'num_preemptions': [1, 0],
            },
            'env_time': {
                'reset_latency': 0.5,
                'step_latency': [2.0, 3.0],
                'close_latency': 0.5,
            },
        },
    }

    out_dir = trajectory_logger.log_trajectory_json(temp_dir, item)
    self.assertIsNotNone(out_dir)

    expected_dir = os.path.join(
        temp_dir, 'step5', 'rollout_worker_0', 'traj_issue_99_g1'
    )
    self.assertEqual(out_dir, expected_dir)

    # 1. Verify metadata.json has model_time
    with open(os.path.join(expected_dir, 'metadata.json'), 'r') as f:
      metadata = json.load(f)
    self.assertIn('model_time', metadata)
    self.assertEqual(metadata['model_time']['step_latency'], [1.2, 0.8])

    # 2. Verify step files have model_latency_sec
    with open(os.path.join(expected_dir, 'step0.json'), 'r') as f:
      step0 = json.load(f)
    self.assertEqual(step0['model_latency_sec'], 1.2)
    with open(os.path.join(expected_dir, 'step1.json'), 'r') as f:
      step1 = json.load(f)
    self.assertEqual(step1['model_latency_sec'], 0.8)

    # 3. Verify inference_metrics.json (summary)
    with open(os.path.join(expected_dir, 'inference_metrics.json'), 'r') as f:
      inf_summary = json.load(f)
    self.assertEqual(inf_summary['type'], 'summary')
    self.assertEqual(inf_summary['traj_id'], 'traj_issue_99_g1')
    self.assertAlmostEqual(inf_summary['total_model_time_sec'], 2.0)
    self.assertAlmostEqual(inf_summary['total_env_time_sec'], 6.0)
    self.assertAlmostEqual(inf_summary['total_time_sec'], 8.0)
    self.assertAlmostEqual(inf_summary['total_time'], 8.0)
    self.assertEqual(inf_summary['total_preemptions'], 1)
    self.assertEqual(inf_summary['preemptions'], 1)
    self.assertEqual(inf_summary['total_completion_tokens'], 40)
    self.assertAlmostEqual(inf_summary['tokens_per_second'], 20.0)
    self.assertAlmostEqual(inf_summary['tpot_ms'], 50.0)
    self.assertAlmostEqual(inf_summary['mean_step_latency_sec'], 1.0)

    # 4. Verify inference_metrics.jsonl (per-step records + summary record)
    with open(os.path.join(expected_dir, 'inference_metrics.jsonl'), 'r') as f:
      lines = [json.loads(line) for line in f if line.strip()]
    self.assertLen(lines, 3)
    # Line 0: step 0
    self.assertEqual(lines[0]['global_step'], 5)
    self.assertEqual(lines[0]['step_index'], 0)
    self.assertEqual(lines[0]['prompt_tokens'], 100)
    self.assertEqual(lines[0]['completion_tokens'], 24)
    self.assertAlmostEqual(lines[0]['latency_sec'], 1.2)
    self.assertAlmostEqual(lines[0]['model_time_sec'], 1.2)
    self.assertAlmostEqual(lines[0]['env_time_sec'], 2.0)
    self.assertAlmostEqual(lines[0]['total_time_sec'], 3.2)
    self.assertAlmostEqual(lines[0]['total_time'], 3.2)
    self.assertEqual(lines[0]['preemptions'], 1)
    self.assertAlmostEqual(lines[0]['tokens_per_second'], 20.0)
    self.assertAlmostEqual(lines[0]['tpot_ms'], 50.0)
    # Line 1: step 1
    self.assertEqual(lines[1]['step_index'], 1)
    self.assertEqual(lines[1]['prompt_tokens'], 150)
    self.assertEqual(lines[1]['completion_tokens'], 16)
    self.assertAlmostEqual(lines[1]['latency_sec'], 0.8)
    self.assertAlmostEqual(lines[1]['model_time_sec'], 0.8)
    self.assertAlmostEqual(lines[1]['env_time_sec'], 3.0)
    self.assertAlmostEqual(lines[1]['total_time_sec'], 3.8)
    self.assertAlmostEqual(lines[1]['total_time'], 3.8)
    self.assertEqual(lines[1]['preemptions'], 0)
    self.assertAlmostEqual(lines[1]['tokens_per_second'], 20.0)
    self.assertAlmostEqual(lines[1]['tpot_ms'], 50.0)
    # Line 2: summary
    self.assertEqual(lines[2]['type'], 'summary')
    self.assertAlmostEqual(lines[2]['total_model_time_sec'], 2.0)
    self.assertAlmostEqual(lines[2]['total_time_sec'], 8.0)
    self.assertAlmostEqual(lines[2]['total_time'], 8.0)
    self.assertEqual(lines[2]['total_preemptions'], 1)
    self.assertEqual(lines[2]['total_completion_tokens'], 40)


if __name__ == '__main__':
  absltest.main()
