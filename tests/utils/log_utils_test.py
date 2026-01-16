"""Unit tests for log_utils."""

import os

from absl.testing import absltest
from tunix.utils import log_utils


class LogUtilsTest(absltest.TestCase):
  """Tests for log_utils."""

  def test_log_item_with_none_log_path(self):
    """Tests that log_item with log_path=None does not log."""
    item = {
        'global_step': 0,
        'trajectory_id': 't0',
        'completion': 'c0',
        'prompt': 'p0',
    }
    log_utils.log_item(None, item)  # Should not raise error or write file.

  def test_log_item_with_non_existent_dir(self):
    """Tests that log_item with a non-existent dir raises AssertionError."""
    temp_dir = self.create_tempdir().full_path
    non_existent_path = os.path.join(temp_dir, 'non_existent')
    item = {
        'global_step': 0,
        'trajectory_id': 't0',
        'completion': 'c0',
        'prompt': 'p0',
    }
    with self.assertRaisesRegex(
        AssertionError, 'must be an existing directory'
    ):
      log_utils.log_item(non_existent_path, item)

  def test_log_item_with_file_as_log_path(self):
    """Tests that log_item with a file as log_path raises AssertionError."""
    temp_file = self.create_tempfile()
    item = {
        'global_step': 0,
        'trajectory_id': 't0',
        'completion': 'c0',
        'prompt': 'p0',
    }
    with self.assertRaisesRegex(
        AssertionError, 'must be an existing directory'
    ):
      log_utils.log_item(temp_file.full_path, item)

  def test_log_item_creates_and_writes_to_file(self):
    """Tests that log_item creates and writes to a log file."""
    temp_dir_obj = self.create_tempdir()
    temp_dir = temp_dir_obj.full_path
    item1 = log_utils.TrajectoryData(
        global_step=0,
        trajectory_id='t0',
        completion='c0',
        prompt='p0',
    )
    log_utils.log_item(temp_dir, item1)

    log_file = os.path.join(temp_dir, 'trajectory_log.md')
    self.assertTrue(os.path.exists(log_file))

    with open(log_file, 'r') as f:
      lines = f.readlines()

    self.assertLen(lines, 3)
    self.assertEqual(
        lines[0].strip(),
        '| completion | global_step | prompt | trajectory_id |',
    )
    self.assertEqual(lines[1].strip(), '| --- | --- | --- | --- |')
    self.assertEqual(lines[2].strip(), '| c0 | 0 | p0 | t0 |')

    item2 = {
        'global_step': 1,
        'trajectory_id': 't1',
        'completion': 'c1|pipe',
        'prompt': 'p1</reasoning>',
    }
    log_utils.log_item(temp_dir, item2)
    with open(log_file, 'r') as f:
      lines = f.readlines()

    self.assertLen(lines, 4)
    self.assertEqual(
        lines[0].strip(),
        '| completion | global_step | prompt | trajectory_id |',
    )
    self.assertEqual(lines[1].strip(), '| --- | --- | --- | --- |')
    self.assertEqual(lines[2].strip(), '| c0 | 0 | p0 | t0 |')
    self.assertEqual(
        lines[3].strip(), '| c1\\|pipe | 1 | p1</reasoning> | t1 |'
    )


if __name__ == '__main__':
  absltest.main()
