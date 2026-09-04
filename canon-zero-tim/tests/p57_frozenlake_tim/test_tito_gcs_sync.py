#!/usr/bin/env python3
"""Tests for bounded P57 TiTO GCS evidence snapshots."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tasks/multiturn-tito-cross-workload/scripts/sync_tito_evidence_to_gcs.py"
)
SPEC = importlib.util.spec_from_file_location("p57_tito_gcs_sync", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P57 TiTO GCS sync")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def _write(path: Path, record: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
  path.chmod(0o600)


def _fixture(root: Path) -> Path:
  state = root / "canon-p57-tito-fixture"
  state.mkdir()
  _write(
      state / "p57_tito_witness/host/host-request-a.json",
      {"schema": "host", "value": 1},
  )
  _write(
      state / "p57_tito_witness/runner/runner-input-a.json",
      {"schema": "runner", "value": 1},
  )
  _write(
      state / "p57_tito_witness/diagnostic-summary.json",
      {"schema": "summary", "value": 1},
  )
  _write(
      state / "p57_tito_collection.classification.json",
      {"schema": "classification", "verdict": "PASS"},
  )
  return state


class TitoGcsSyncTest(unittest.TestCase):

  def test_full_final_inventory_requires_every_token_event_capsule(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = Path(tmp) / "state"
      state.mkdir()
      required = (
          "p57_tito_witness/single-writer.json",
          "p57_tito_witness/full-row-map.jsonl",
          "p57_tito_full_record.classification.json",
          "p33_frozenlake-dp8-tp8_full.classification.json",
          "v1_hp_p45_full.classification.json",
          "pre_alignment.jsonl",
          "alignment.jsonl",
          "updates.jsonl",
          "p57_tito_witness/journal-reconstruction.json",
          "p57_tito_gcs/orbax-probe.json",
      )
      for relative in required:
        _write(state / relative, {"fixture": relative})
      summary = state / "p57_tito_witness/full-record-summary.json"
      _write(summary, {"collection": {"token_difference_events": 1}})
      capsule = state / "token-continuity-first-diff/event-1.json"
      _write(capsule, {"event_index": 1})
      records = sync.evidence_inventory(state, final=True)
      self.assertIn(
          "token-continuity-first-diff/event-1.json",
          {record["path"] for record in records},
      )
      capsule.unlink()
      with self.assertRaisesRegex(
          ValueError, "token-difference inventory differs"
      ):
        sync.evidence_inventory(state, final=True)

  def test_live_inventory_reuses_uploaded_sha_but_final_rehashes(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      initial = sync.evidence_inventory(state, final=False)
      prior = {record["path"]: record for record in initial}
      target = state / "p57_tito_witness/host/host-request-a.json"
      original = target.read_bytes()
      target.write_bytes(b"x" * len(original))
      target.chmod(0o600)

      live = sync.evidence_inventory(
          state, final=False, prior_records=prior
      )
      live_by_path = {record["path"]: record for record in live}
      self.assertEqual(
          live_by_path["p57_tito_witness/host/host-request-a.json"],
          prior["p57_tito_witness/host/host-request-a.json"],
      )

      final_hashes = sync.evidence_inventory(state, final=False)
      final_by_path = {record["path"]: record for record in final_hashes}
      self.assertNotEqual(
          final_by_path["p57_tito_witness/host/host-request-a.json"][
              "sha256"
          ],
          prior["p57_tito_witness/host/host-request-a.json"]["sha256"],
      )

  def test_append_journals_become_immutable_complete_line_chunks(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = Path(tmp) / "state"
      state.mkdir()
      for relative in sync._APPEND_JOURNALS:
        path = state / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"step":0}\n')
        path.chmod(0o600)

      first = sync.materialize_journal_deltas(state, final=False)
      self.assertFalse(first["final"])
      self.assertEqual(len(first["journals"]), 4)
      self.assertTrue(all(len(row["chunks"]) == 1 for row in first["journals"]))

      row_map = state / "p57_tito_witness/full-row-map.jsonl"
      with row_map.open("ab") as output:
        output.write(b'{"step":1}\n{"partial":')
      second = sync.materialize_journal_deltas(state, final=False)
      row = next(
          item for item in second["journals"]
          if item["source"] == "p57_tito_witness/full-row-map.jsonl"
      )
      self.assertEqual(len(row["chunks"]), 2)
      self.assertLess(row["complete_bytes"], row["source_bytes"])
      with self.assertRaisesRegex(ValueError, "not fully reconstructable"):
        sync.materialize_journal_deltas(state, final=True)

      with row_map.open("ab") as output:
        output.write(b'true}\n')
      final = sync.materialize_journal_deltas(state, final=True)
      self.assertTrue(final["final"])
      receipt = state / "p57_tito_witness/journal-reconstruction.json"
      self.assertTrue(receipt.is_file())
      self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
      recorded = json.loads(receipt.read_text())
      self.assertEqual(recorded, final)

      chunk = next(
          (state / "p57_tito_gcs/journal-deltas").glob("**/chunk-*.jsonl")
      )
      chunk.write_bytes(b"tampered\n")
      with self.assertRaisesRegex(ValueError, "chunk chain differs"):
        sync.materialize_journal_deltas(state, final=True)

  def test_existing_snapshot_rehashes_every_payload(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      records = sync.evidence_inventory(state, final=False)
      snapshot = state / "snapshot.tar"
      sync.create_snapshot(state, records, snapshot)
      manifest = sync._manifest_payload(records)
      with tarfile.open(snapshot, mode="w") as archive:
        manifest_info = tarfile.TarInfo("SHA256SUMS")
        manifest_info.size = len(manifest)
        manifest_info.mode = 0o600
        archive.addfile(manifest_info, io.BytesIO(manifest))
        for index, record in enumerate(records):
          payload = (state / record["path"]).read_bytes()
          if index == 0:
            payload = b"x" * len(payload)
          info = tarfile.TarInfo(record["path"])
          info.size = len(payload)
          info.mode = 0o600
          archive.addfile(info, io.BytesIO(payload))
      snapshot.chmod(0o600)
      with self.assertRaisesRegex(ValueError, "snapshot payload differs"):
        sync.create_snapshot(state, records, snapshot)

  def test_incremental_and_final_snapshots_have_verified_readbacks(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      remote: dict[str, bytes] = {}

      def fake_copy(source: str, destination: str, *, no_clobber: bool) -> int:
        if source.startswith("gs://"):
          if source not in remote:
            return 1
          Path(destination).write_bytes(remote[source])
          return 0
        if no_clobber and destination in remote:
          return 1
        remote[destination] = Path(source).read_bytes()
        return 0

      prefix = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-direct"
      )
      with mock.patch.object(sync, "_run_gcloud_cp", side_effect=fake_copy):
        probe = sync.probe_gcs(state, prefix)
        live = sync.sync_once(state, prefix, final=False)
        unchanged = sync.sync_once(state, prefix, final=False)
        _write(
            state / "p57_tito_witness/host/host-request-b.json",
            {"schema": "host", "value": 2},
        )
        delta = sync.sync_once(state, prefix, final=False)
        final = sync.sync_once(state, prefix, final=True)
        final_retry = sync.sync_once(state, prefix, final=True)
      self.assertEqual(probe["status"], "PASS")
      self.assertEqual(live["status"], "PASS")
      self.assertEqual(unchanged["status"], "UNCHANGED")
      self.assertEqual(delta["files"], 1)
      self.assertEqual(
          [record["path"] for record in delta["records"]],
          ["p57_tito_witness/host/host-request-b.json"],
      )
      self.assertEqual(final["kind"], "final")
      self.assertEqual(final_retry, final)
      self.assertEqual(final["files"], 2)
      self.assertEqual(final["complete_files"], 5)
      self.assertIn(f"{prefix}/final-manifest.json", remote)
      self.assertEqual(len(remote), 5)
      manifest = state / "p57_tito_gcs/final-manifest.json"
      self.assertTrue(manifest.is_file())
      self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)

  def test_mode_prefix_missing_final_and_readback_tamper_fail(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      host = state / "p57_tito_witness/host/host-request-a.json"
      host.chmod(0o644)
      with self.assertRaisesRegex(ValueError, "mode 0600"):
        sync.evidence_inventory(state, final=False)
      host.chmod(0o600)
      (state / "p57_tito_collection.classification.json").unlink()
      with self.assertRaisesRegex(ValueError, "incomplete"):
        sync.evidence_inventory(state, final=True)
      with self.assertRaisesRegex(ValueError, "registered evidence root"):
        sync.sync_once(state, "gs://wrong/prefix", final=False)

    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      state = _fixture(root)
      host_dir = state / "p57_tito_witness/host"
      (host_dir / "host-request-a.json").unlink()
      host_dir.rmdir()
      external = root / "external-host"
      _write(external / "host-request-escaped.json", {"escaped": True})
      host_dir.symlink_to(external, target_is_directory=True)
      with self.assertRaisesRegex(ValueError, "escapes its state directory"):
        sync.evidence_inventory(state, final=False)

    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))

      def corrupt_copy(source: str, destination: str, *, no_clobber: bool) -> int:
        if source.startswith("gs://"):
          Path(destination).write_bytes(b"corrupt")
        return 0

      prefix = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-0"
      )
      with (
          mock.patch.object(sync, "_run_gcloud_cp", side_effect=corrupt_copy),
          self.assertRaisesRegex(RuntimeError, "readback SHA256 differs"),
      ):
        sync.sync_once(state, prefix, final=False)

  def test_transient_readback_retries_and_changed_delta_fails(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      remote: dict[str, bytes] = {}
      readbacks = 0

      def flaky_copy(source: str, destination: str, *, no_clobber: bool) -> int:
        nonlocal readbacks
        if source.startswith("gs://"):
          readbacks += 1
          if readbacks <= 2:
            return 1
          Path(destination).write_bytes(remote[source])
          return 0
        if no_clobber and destination in remote:
          return 1
        remote[destination] = Path(source).read_bytes()
        return 0

      prefix = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-1"
      )
      with (
          mock.patch.object(sync, "_run_gcloud_cp", side_effect=flaky_copy),
          mock.patch.object(sync.time, "sleep") as sleep,
      ):
        result = sync.sync_once(state, prefix, final=False)
      self.assertEqual(result["status"], "PASS")
      self.assertEqual(sleep.call_args_list, [mock.call(1), mock.call(2)])

      host = state / "p57_tito_witness/host/host-request-a.json"
      _write(host, {"schema": "host", "value": "changed"})
      with self.assertRaisesRegex(ValueError, "changed or disappeared"):
        sync.sync_once(state, prefix, final=False)

  def test_final_rejects_missing_prior_delta(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = _fixture(Path(tmp))
      remote: dict[str, bytes] = {}

      def fake_copy(source: str, destination: str, *, no_clobber: bool) -> int:
        if source.startswith("gs://"):
          Path(destination).write_bytes(remote[source])
          return 0
        remote[destination] = Path(source).read_bytes()
        return 0

      prefix = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-2"
      )
      with mock.patch.object(sync, "_run_gcloud_cp", side_effect=fake_copy):
        sync.sync_once(state, prefix, final=False)
        next((state / "p57_tito_gcs/snapshots").glob("*.tar")).unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
          sync.sync_once(state, prefix, final=True)

  def test_worker_periodically_snapshots_and_finalizes(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      state = _fixture(root)
      fake_bin = root / "bin"
      fake_bin.mkdir()
      fake_gcloud = fake_bin / "gcloud"
      fake_gcloud.write_text(
          """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
if args[:2] != ["storage", "cp"]:
  raise SystemExit(2)
no_clobber = "--no-clobber" in args
args = [value for value in args[2:] if value != "--no-clobber"]
source, destination = args
root = Path(os.environ["FAKE_GCS_ROOT"])
def local(value):
  return root / value.removeprefix("gs://") if value.startswith("gs://") else Path(value)
source_path = local(source)
destination_path = local(destination)
if no_clobber and destination_path.exists():
  raise SystemExit(1)
if not source_path.is_file():
  raise SystemExit(1)
destination_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source_path, destination_path)
""",
          encoding="utf-8",
      )
      fake_gcloud.chmod(0o755)
      controls = state / "p57_tito_gcs"
      ready = controls / "ready"
      stop = controls / "stop"
      finalize = controls / "finalize"
      ack = controls / "finalize.ack"
      heartbeat = controls / "heartbeat"
      prefix = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-direct"
      )
      process = subprocess.Popen(
          [
              "bash",
              str(
                  ROOT
                  / "tasks/multiturn-tito-cross-workload/scripts/"
                  "p57_tito_gcs_worker.sh"
              ),
          ],
          env={
              **os.environ,
              "PATH": f"{fake_bin}:{os.environ['PATH']}",
              "FAKE_GCS_ROOT": str(root / "remote"),
              "CANON_PKG": str(ROOT),
              "CANON_STATE": str(state),
              "CANON_P57_TITO_GCS_PREFIX": prefix,
              "CANON_P57_TITO_GCS_INTERVAL_SECONDS": "1",
              "CANON_P57_TITO_GCS_READY": str(ready),
              "CANON_P57_TITO_GCS_STOP_FILE": str(stop),
              "CANON_P57_TITO_GCS_FINALIZE_FILE": str(finalize),
              "CANON_P57_TITO_GCS_FINAL_ACK": str(ack),
              "CANON_P57_TITO_GCS_HEARTBEAT": str(heartbeat),
          },
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
      )
      deadline = time.monotonic() + 5
      receipts = controls / "receipts"
      while time.monotonic() < deadline and not list(receipts.glob("*.json")):
        time.sleep(0.05)
      self.assertEqual(ready.read_text(), "action=ready status=PASS\n")
      finalize.touch(mode=0o600)
      output, _ = process.communicate(timeout=8)
      self.assertEqual(process.returncode, 0, output)
      self.assertEqual(ack.read_text(), "action=finalize status=PASS\n")
      self.assertIn("status=PASS", heartbeat.read_text())
      remote_manifest = (
          root
          / "remote/yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
          "canon-p57-tito-fixture/attempt-direct/final-manifest.json"
      )
      self.assertTrue(remote_manifest.is_file())


if __name__ == "__main__":
  unittest.main()
