#!/usr/bin/env python3
"""Host positives and negatives for Attempt-13 flat-shard replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from replay_m15_attempt13_flat_shards import (
    Attempt13FlatReplayError,
    SOURCE_COMMIT,
    replay,
)
from stage_m15_wide_shard import _sha256, stage
from test_classify_m15_apc_wide_seam import Fixture


SOURCE = SOURCE_COMMIT


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class Attempt13FlatShardReplayTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.fixtures: list[Fixture] = []

  def tearDown(self) -> None:
    for fixture in self.fixtures:
      fixture.close()
    self.holder.cleanup()

  def _fixture(self, arm: str, *, boundary: str = "rpa_output") -> Fixture:
    fixture = Fixture(mode="full")
    self.fixtures.append(fixture)
    if arm == "off":
      fixture._write_report(ab_bytes=0, bc_bytes=0)  # pylint: disable=protected-access
      return fixture
    if boundary == "rpa_output":
      for metadata_path in fixture.capture.glob("p38_seam_*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["arm"] != "A":
          continue
        npz_path = metadata_path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as archive:
          arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        values = arrays["layer_fingerprints"]
        values.fill(0)
        checkpoint = metadata["checkpoint_names"].index("rpa_output")
        values[0, 0, checkpoint, 3] = 1
        np.savez(npz_path, **arrays)
        metadata["npz_sha256"] = _sha(npz_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return fixture

  def _arm(self, arm: str, *, boundary: str = "rpa_output") -> Path:
    fixture = self._fixture(arm, boundary=boundary)
    root = self.root / arm
    shards = root / "shards"
    shard = shards / "000000"
    inventory = stage(
        directory=fixture.capture,
        shard_root=shards,
        output=shard,
        round_index=0,
        sequence=0,
        max_records=32,
        max_bytes=256 * 1024 * 1024,
        expected_commit=SOURCE,
        runtime_commit=SOURCE,
    )
    completion = {
        "schema": "m15-wide-observer-shard-completion-v1",
        "status": "sealed-uploaded-verified",
        "claim_ceiling": (
            "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
        ),
        "sequence": 0,
        "diagnostic_round": 0,
        "record_pairs": inventory["record_pairs"],
        "payload_bytes": inventory["payload_bytes"],
        "manifest_sha256": _sha256(shard / "SHA256SUMS"),
        "archive_sha256": "a" * 64,
        "expected_source_commit": SOURCE,
        "runtime_source_commit": SOURCE,
    }
    (shard / "SHARD_COMPLETE.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    live = root / "live"
    live.mkdir(parents=True)
    shutil.copyfile(fixture.report, live / "pre-alignment.jsonl")
    shutil.copyfile(fixture.ledger, live / "m15-replay-envelope.jsonl")
    (live / "diagnostic-round.txt").write_text("0\n", encoding="ascii")
    if arm == "on":
      shutil.copyfile(fixture.capsule, live / "mismatch-capsule.round-000000.npz")
    names = sorted(path.name for path in live.iterdir())
    (live / "SHA256SUMS").write_text(
        "".join(f"{_sha(live / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    receipt = {
        "schema": "canon-p38-gcs-live-v1",
        "status": "live-snapshot",
        "source_commit": SOURCE,
        "sequence": 7,
        "files": names,
        "manifest_sha256": _sha(live / "SHA256SUMS"),
    }
    (live / "LIVE.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root

  def _replay(self, *, boundary: str = "rpa_output"):
    off = self._arm("off")
    on = self._arm("on", boundary=boundary)
    return replay(
        off_root=off,
        on_root=on,
        work=self.root / "work",
        output=self.root / "output",
        source_commit=SOURCE,
        expected_shards={"off": 1, "on": 1},
        expected_pairs={"off": 2, "on": 2},
        expected_alignment={
            "off": {"a_b_differing_bytes": 0},
            "on": {"a_b_differing_bytes": 1, "a_b_differing_elements": 1},
        },
        expected_layer=5,
    )

  def test_verified_flat_union_replays_single_attention_interval(self) -> None:
    result = self._replay()
    self.assertEqual(
        result["decision"], "SINGLE_ROUND_ATTENTION_INTERVAL_REPRODUCED"
    )
    self.assertFalse(result["numerical_repair_authorized"])
    self.assertEqual(result["arms"]["off"]["shard_union"]["record_pairs"], 2)
    self.assertFalse(result["arms"]["on"]["compact_bundle"]["returned"])
    output = self.root / "output"
    self.assertFalse(any(path.suffix == ".tar" for path in output.iterdir()))
    for line in (output / "SHA256SUMS").read_text().splitlines():
      digest, name = line.split("  ", 1)
      self.assertEqual(_sha(output / name), digest)

  def test_replayed_boundary_disagreement_is_preserved_not_overclaimed(self) -> None:
    result = self._replay(boundary="q_post_rope")
    self.assertEqual(result["decision"], "SINGLE_ROUND_OFFICIAL_REPLAY_DISAGREES")
    self.assertFalse(result["numerical_repair_authorized"])

  def test_missing_shard_sequence_is_rejected(self) -> None:
    off = self._arm("off")
    on = self._arm("on")
    with self.assertRaisesRegex(Attempt13FlatReplayError, "sequences"):
      replay(
          off_root=off,
          on_root=on,
          work=self.root / "work",
          output=self.root / "output",
          source_commit=SOURCE,
          expected_shards={"off": 2, "on": 1},
          expected_pairs={"off": 2, "on": 2},
          expected_alignment={
              "off": {"a_b_differing_bytes": 0},
              "on": {"a_b_differing_bytes": 1, "a_b_differing_elements": 1},
          },
          expected_layer=5,
      )

  def test_tampered_shard_member_is_rejected(self) -> None:
    off = self._arm("off")
    on = self._arm("on")
    member = next((off / "shards/000000").glob("p38_seam_*.npz"))
    member.write_bytes(b"tampered")
    with self.assertRaisesRegex(Attempt13FlatReplayError, "failed SHA"):
      replay(
          off_root=off,
          on_root=on,
          work=self.root / "work",
          output=self.root / "output",
          source_commit=SOURCE,
          expected_shards={"off": 1, "on": 1},
          expected_pairs={"off": 2, "on": 2},
          expected_alignment={
              "off": {"a_b_differing_bytes": 0},
              "on": {"a_b_differing_bytes": 1, "a_b_differing_elements": 1},
          },
          expected_layer=5,
      )

  def test_on_live_snapshot_without_capsule_is_rejected(self) -> None:
    off = self._arm("off")
    on = self._arm("on")
    capsule = next((on / "live").glob("*capsule*.npz"))
    capsule.unlink()
    with self.assertRaisesRegex(Attempt13FlatReplayError, "manifest|capsule"):
      replay(
          off_root=off,
          on_root=on,
          work=self.root / "work",
          output=self.root / "output",
          source_commit=SOURCE,
          expected_shards={"off": 1, "on": 1},
          expected_pairs={"off": 2, "on": 2},
          expected_alignment={
              "off": {"a_b_differing_bytes": 0},
              "on": {"a_b_differing_bytes": 1, "a_b_differing_elements": 1},
          },
          expected_layer=5,
      )


if __name__ == "__main__":
  unittest.main()
