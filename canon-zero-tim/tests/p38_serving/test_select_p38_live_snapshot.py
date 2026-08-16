#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "select_p38_live_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("p38_snapshot_selector", MODULE)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)

LIVE_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    "canon-p38-test/attempt-0/live"
)


def _objects(snapshot: str, capsule_rounds: tuple[int, ...], records: int):
  names = ["LIVE.json", "SHA256SUMS", "run.log", "pre-alignment.jsonl"]
  names.extend(
      f"p38_frozenlake_mismatch_capsule.round-{value:06d}.npz"
      for value in capsule_rounds)
  for index in range(records):
    names.append(f"p38_seam_{index:06d}.json")
    names.append(f"p38_seam_{index:06d}.npz")
  return [f"{LIVE_ROOT}/{snapshot}/{name}" for name in names]


class SelectP38LiveSnapshotTest(unittest.TestCase):

  def _listing(self, root: Path, lines: list[str]) -> Path:
    path = root / "listing.txt"
    path.write_text("\n".join(lines) + "\n")
    return path

  def test_prefers_capsule_coverage_before_snapshot_number(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      listing = self._listing(root, [
          *_objects("000019", (0, 1), 8),
          *_objects("000020", (0,), 12),
      ])
      report = selector.select_snapshot(listing, LIVE_ROOT, 2)
      self.assertTrue(report["selection_complete"])
      self.assertEqual(report["selected_snapshot"], "000019")
      self.assertEqual(report["selected_capsule_rounds"], [0, 1])

  def test_selects_latest_when_coverage_is_equal(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      listing = self._listing(root, [
          *_objects("000019", (0, 1), 8),
          *_objects("000021", (0, 1), 10),
      ])
      report = selector.select_snapshot(listing, LIVE_ROOT, 2)
      self.assertEqual(report["selected_snapshot"], "000021")

  def test_refuses_undercovered_or_unpaired_snapshots(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      lines = _objects("000020", (0,), 5)
      broken = _objects("000021", (0, 1), 5)
      broken.remove(f"{LIVE_ROOT}/000021/p38_seam_000004.npz")
      listing = self._listing(root, [*lines, *broken])
      report = selector.select_snapshot(listing, LIVE_ROOT, 2)
      self.assertFalse(report["selection_complete"])
      self.assertEqual(report["qualified_candidate_count"], 0)
      candidate = next(
          item for item in report["candidates"]
          if item["snapshot"] == "000021")
      self.assertFalse(candidate["paired_seam_records"])


if __name__ == "__main__":
  unittest.main()
