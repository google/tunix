"""Unit and negative-control tests for P34 scheduler geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("probe_scheduler_contract.py")
SPEC = importlib.util.spec_from_file_location("p34_scheduler_probe", PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot load P34 scheduler probe")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _power_of_two_paddings(*, min_token_size, max_token_size, padding_gap):
  del padding_gap
  result = []
  value = min_token_size
  while value < max_token_size:
    result.append(value)
    value *= 2
  result.append(max_token_size)
  return result


class SchedulerContractTest(unittest.TestCase):

  def test_current_geometry_is_one_bucket_and_64_requests(self):
    result = probe.evaluate(_power_of_two_paddings)
    self.assertEqual(result.current_paddings, (4096,))
    self.assertEqual(result.current_num_reqs, 64)

  def test_historical_global_as_local_geometry_is_rejected(self):
    result = probe.evaluate(_power_of_two_paddings)
    self.assertGreater(len(result.legacy_paddings), 1)
    self.assertEqual(result.legacy_paddings[-1], 131072)
    self.assertEqual(result.legacy_num_reqs, 1024)

  def test_extra_current_bucket_fails_closed(self):
    def bad_paddings(*, min_token_size, max_token_size, padding_gap):
      del min_token_size, max_token_size, padding_gap
      return [4096, 8192]

    with self.assertRaisesRegex(ValueError, "exactly one"):
      probe.evaluate(bad_paddings)


if __name__ == "__main__":
  unittest.main()
