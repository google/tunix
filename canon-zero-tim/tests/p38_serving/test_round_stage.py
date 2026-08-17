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

  def _fixture(
      self, root: Path, *, tail: bool = True, terminal: bool = True,
  ) -> argparse.Namespace:
    observer = root / "observer"
    observer.mkdir()
    run_log = root / "run.log"
    run_log.write_text("run\n", encoding="utf-8")
    pre = root / "pre.jsonl"
    pre.write_text(
        json.dumps({"diagnostic_round": 0, "name": "pre", "step": 0})
        + "\n"
        + json.dumps({"diagnostic_round": 1, "name": "pre", "step": 0})
        + "\n",
        encoding="utf-8",
    )
    journal = root / "journal.jsonl"
    journal.write_text(
        json.dumps({"schema": "p38-request-journal-v1", "name": "first"})
        + "\n"
        + json.dumps({"schema": "p38-request-journal-v1", "name": "second"})
        + "\n",
        encoding="utf-8",
    )
    incident = root / "incident.jsonl"
    incident.write_text(
        json.dumps({
            "diagnostic_round": 0,
            "schema": "p38-incident-ledger-v1",
        }) + "\n"
        + json.dumps({
            "diagnostic_round": 1,
            "schema": "p38-incident-ledger-v1",
        }) + "\n",
        encoding="utf-8",
    )
    capsule = root / "capsule.npz"
    np.savez(root / "capsule.round-000000.npz", values=np.arange(4))
    self._record_pair(observer, "p38_seam_000000", "p38-seam-fingerprint-v1")
    if tail:
      self._record_pair(observer, "p38_tail_000000", "p38-tail-values-v1")
    if terminal:
      self._record_pair(
          observer,
          "p38_terminal_000000",
          "p38-terminal-discriminator-v1",
      )
    return argparse.Namespace(
        round=0,
        output=root / "round",
        run_log=run_log,
        pre_alignment=pre,
        capsule=capsule,
        request_journal=journal,
        incident_ledger=incident,
        observer_dir=observer,
        require_seam=True,
        require_kv=False,
        require_tail=True,
        require_terminal=True,
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
      self.assertEqual(result["terminal_records"], 1)
      for name in (
          "mismatch-capsule.npz", "pre-alignment.jsonl",
          "request-journal.jsonl", "incident-ledger.jsonl",
          "p38_seam_000000.json", "p38_seam_000000.npz",
          "p38_tail_000000.json", "p38_tail_000000.npz",
          "p38_terminal_000000.json", "p38_terminal_000000.npz",
          "ROUND_INVENTORY.json",
      ):
        self.assertTrue((args.output / name).is_file(), name)
      record = json.loads(
          (args.output / "pre-alignment.jsonl").read_text(encoding="utf-8")
      )
      self.assertEqual(record["diagnostic_round"], 0)
      inventory = json.loads(
          (args.output / "ROUND_INVENTORY.json").read_text(encoding="utf-8")
      )
      self.assertEqual(inventory["journal_records"], 2)
      self.assertEqual(inventory["journal_scope"], "cumulative-unscoped")

  def test_missing_tail_and_sha_mutation_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory), tail=False)
      with self.assertRaisesRegex(ValueError, "has no tail records"):
        stage_module.stage(args)
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory), terminal=False)
      with self.assertRaisesRegex(
          ValueError, "has no terminal discriminator records"):
        stage_module.stage(args)
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      seam = args.observer_dir / "p38_seam_000000.npz"
      seam.write_bytes(seam.read_bytes() + b"fault")
      with self.assertRaisesRegex(ValueError, "NPZ SHA failed"):
        stage_module.stage(args)

  def test_step_is_not_accepted_as_round_scope(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      args.pre_alignment.write_text('{"step": 0}\n', encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "has no diagnostic_round"):
        stage_module.stage(args)

  def test_unscoped_incident_and_wrong_journal_schema_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      args.incident_ledger.write_text(
          '{"schema":"p38-incident-ledger-v1"}\n', encoding="utf-8"
      )
      with self.assertRaisesRegex(ValueError, "has no diagnostic_round"):
        stage_module.stage(args)
    with tempfile.TemporaryDirectory() as directory:
      args = self._fixture(Path(directory))
      args.request_journal.write_text(
          '{"schema":"wrong"}\n', encoding="utf-8"
      )
      with self.assertRaisesRegex(ValueError, "JSONL schema drifted"):
        stage_module.stage(args)


if __name__ == "__main__":
  unittest.main()
