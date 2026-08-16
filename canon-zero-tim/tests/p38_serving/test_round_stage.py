#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py"
)
SPEC = importlib.util.spec_from_file_location("stage_p38_round", MODULE)
assert SPEC and SPEC.loader
stage_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_module)


class RoundStageTest(unittest.TestCase):

  def _fixture(self, root: Path, *, tail: bool = True) -> argparse.Namespace:
    observer = root / "observer"
    observer.mkdir()
    run_log = root / "run.log"
    run_log.write_text("run\n", encoding="utf-8")
    sources = []
    for name in ("pre.jsonl", "journal.jsonl", "incident.jsonl"):
      path = root / name
      path.write_text(
          json.dumps({"diagnostic_round": 0, "name": name}) + "\n"
          + json.dumps({"diagnostic_round": 1, "name": name}) + "\n",
          encoding="utf-8",
      )
      sources.append(path)
    capsule = root / "capsule.npz"
    np.savez(root / "capsule.round-000000.npz", values=np.arange(4))
    self._record_pair(observer, "p38_seam_000000", "p38-seam-fingerprint-v1")
    if tail:
      self._record_pair(observer, "p38_tail_000000", "p38-tail-values-v1")
    return argparse.Namespace(
        round=0,
        output=root / "round",
        run_log=run_log,
        pre_alignment=sources[0],
        capsule=capsule,
        request_journal=sources[1],
        incident_ledger=sources[2],
        observer_dir=observer,
        require_seam=True,
        require_kv=False,
        require_tail=True,
    )

  def _record_pair(self, root: Path, stem: str, schema: str) -> None:
    npz = root / f"{stem}.npz"
    np.savez(npz, values=np.arange(8, dtype=np.float32))
    (root / f"{stem}.json").write_text(
        json.dumps({
            "diagnostic_round": 0,
            "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            "schema": schema,
        }),
        encoding="utf-8",
    )

  def test_stages_complete_round_and_filters_jsonl(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      result = stage_module.stage(args)
      self.assertEqual(result["seam_records"], 1)
      self.assertEqual(result["tail_records"], 1)
      for name in (
          "mismatch-capsule.npz", "pre-alignment.jsonl",
          "request-journal.jsonl", "incident-ledger.jsonl",
          "p38_seam_000000.json", "p38_seam_000000.npz",
          "p38_tail_000000.json", "p38_tail_000000.npz",
          "ROUND_INVENTORY.json",
      ):
        self.assertTrue((args.output / name).is_file(), name)
      record = json.loads(
          (args.output / "pre-alignment.jsonl").read_text(encoding="utf-8")
      )
      self.assertEqual(record["diagnostic_round"], 0)

  def test_missing_tail_and_sha_mutation_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory), tail=False)
      with self.assertRaisesRegex(ValueError, "has no tail records"):
        stage_module.stage(args)
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      seam = args.observer_dir / "p38_seam_000000.npz"
      seam.write_bytes(seam.read_bytes() + b"fault")
      with self.assertRaisesRegex(ValueError, "NPZ SHA failed"):
        stage_module.stage(args)


if __name__ == "__main__":
  unittest.main()
