import os
import tempfile
from typing import TypedDict

from absl.testing import absltest
import numpy as np
import pandas as pd
from tunix.utils import trajectory_logger


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
      logger.log_item_async({'step': i, 'value': i * 2})
    logger.stop()

    self.assertTrue(logger._stopped)
    self.assertFalse(logger._logging_thread.is_alive())

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    self.assertLen(csv_files, 1)
    df = pd.read_csv(os.path.join(temp_dir, csv_files[0]))
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


if __name__ == '__main__':
  absltest.main()
