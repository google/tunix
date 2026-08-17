#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "tasks/p38-pathways-decode-prefill-carrier/scripts"
REDUCER = SCRIPTS / "reduce_p38_seam_tail_evidence.py"
AUDITOR = SCRIPTS / "audit_p38_seam_tail_reduction.py"
CLASSIFIER = SCRIPTS / "classify_p38_seam.py"
WRAPPER = SCRIPTS / "run_reduce_p38s18r2_round0_on_gcp.sh"
SOURCE_URI = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    "canon-p38-test/attempt-0/rounds/000000"
)
SOURCE_COMMIT = "1" * 40
ANALYSIS_COMMIT = "2" * 40
TAIL_CHECKPOINTS = [
    "raw_target_logit",
    "raw_log_normalizer",
    "processed_target_logit",
    "processed_log_normalizer",
    "observer_target_logprob",
    "production_target_logprob",
]


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix(values: np.ndarray) -> bytes:
  return hashlib.sha256(
      np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest().encode()


class ReduceP38SeamTailEvidenceTest(unittest.TestCase):

  def _write_fake_tools(self, directory: Path) -> None:
    gcloud = directory / "gcloud"
    gcloud.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(r'''
        import os
        from pathlib import Path
        import shutil
        import sys

        root = Path(os.environ["FAKE_GCS_ROOT"])
        def local(value):
          return root / value[5:] if value.startswith("gs://") else Path(value)

        args = sys.argv[1:]
        with open(os.environ["FAKE_GCS_LOG"], "a", encoding="utf-8") as stream:
          stream.write(" ".join(args) + "\n")
        if args[:2] == ["storage", "ls"]:
          target = args[-1]
          target = target[:-3] if target.endswith("/**") else target
          path = local(target)
          if not path.exists():
            raise SystemExit(1)
          if path.is_file():
            print(target)
          else:
            for item in sorted(path.rglob("*")):
              if item.is_file():
                print("gs://" + item.relative_to(root).as_posix())
        elif args[:2] == ["storage", "rsync"]:
          source, target = args[-2:]
          source_path, target_path = local(source), local(target)
          target_path.mkdir(parents=True, exist_ok=True)
          for item in source_path.rglob("*"):
            if item.is_file():
              destination = target_path / item.relative_to(source_path)
              destination.parent.mkdir(parents=True, exist_ok=True)
              shutil.copy2(item, destination)
        elif args[:2] == ["storage", "cp"]:
          source, target = args[-2:]
          source_path, target_path = local(source), local(target)
          target_path.parent.mkdir(parents=True, exist_ok=True)
          shutil.copy2(source_path, target_path)
        else:
          raise SystemExit(2)
        '''))
    gcloud.chmod(0o755)
    git = directory / "git"
    git.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(f'''
        import sys
        args = sys.argv[1:]
        if args[-2:] == ["rev-parse", "--show-toplevel"]:
          print("/tmp/p38s18r2-test-checkout")
        elif args[-2:] == ["rev-parse", "HEAD"]:
          print("{ANALYSIS_COMMIT}")
        elif args[-2:] == ["status", "--short"]:
          pass
        else:
          raise SystemExit(2)
        '''))
    git.chmod(0o755)

  def _write_seam(
      self,
      root: Path,
      index: int,
      arm: str,
      prefixes: list[bytes],
      positions: list[int],
      tokens: list[int],
      *,
      mutate_row: int | None = None,
  ) -> None:
    rows = len(prefixes)
    layers = np.zeros((rows, 2, 2, 8), dtype=np.uint32)
    if mutate_row is not None:
      layers[mutate_row, 1, 1, 3] = 1
    npz = root / f"p38_seam_{index:06d}.npz"
    np.savez(
        npz,
        row_indices=np.arange(rows, dtype=np.int32) + 200,
        positions=np.asarray(positions, dtype=np.int32),
        token_ids=np.asarray(tokens, dtype=np.int32),
        request_ordinals=np.zeros(rows, dtype=np.int32),
        token_prefix_sha256=np.asarray(prefixes, dtype="S64"),
        layer_fingerprints=layers,
        final_norm_fingerprints=np.zeros((rows, 8), dtype=np.uint32),
    )
    (root / f"p38_seam_{index:06d}.json").write_text(json.dumps({
        "schema": "p38-seam-fingerprint-v1",
        "record_index": index,
        "arm": arm,
        "diagnostic_round": 0,
        "observer_mode": "layer",
        "checkpoint_names": ["layer_input", "layer_output"],
        "layer_indices": [0, 1],
        "call_index": index + 100,
        "program_path": "standard",
        "requests": [],
        "npz_sha256": _sha256(npz),
    }))

  def _write_tail(
      self,
      root: Path,
      index: int,
      arm: str,
      prefixes: list[bytes],
      positions: list[int],
      source_tokens: list[int],
      target_ids: list[int],
      production_values: list[float],
      *,
      mutate_row: int | None = None,
  ) -> None:
    rows = len(prefixes)
    values = np.zeros((rows, len(TAIL_CHECKPOINTS)), dtype=np.float32)
    values[:, -1] = np.asarray(production_values, dtype=np.float32)
    if arm == "B":
      values[:, 0] = np.float32(0.25)
    if mutate_row is not None:
      values[mutate_row, 2] = np.float32(7.0)
    npz = root / f"p38_tail_{index:06d}.npz"
    np.savez(
        npz,
        row_indices=np.arange(rows, dtype=np.int32) + 200,
        positions=np.asarray(positions, dtype=np.int32),
        token_ids=np.asarray(source_tokens, dtype=np.int32),
        request_ordinals=np.zeros(rows, dtype=np.int32),
        token_prefix_sha256=np.asarray(prefixes, dtype="S64"),
        logit_row_indices=np.asarray(positions, dtype=np.int32),
        target_ids=np.asarray(target_ids, dtype=np.int32),
        tail_values=values,
    )
    (root / f"p38_tail_{index:06d}.json").write_text(json.dumps({
        "schema": "p38-tail-values-v1",
        "record_index": index,
        "arm": arm,
        "diagnostic_round": 0,
        "checkpoint_names": TAIL_CHECKPOINTS,
        "call_index": index + 200,
        "program_path": "standard",
        "requests": [],
        "npz_sha256": _sha256(npz),
    }))

  def _write_capsule(self, root: Path, red_points: int) -> tuple[Path, dict]:
    prompt = np.asarray([[11]], dtype=np.int32)
    completion = np.arange(21, 21 + red_points + 1, dtype=np.int32)[None, :]
    decode = np.arange(red_points + 1, dtype=np.float32)[None, :]
    prefill = decode.copy()
    prefill[0, :red_points] += np.float32(0.5)
    action = np.ones_like(completion, dtype=np.bool_)
    capsule = root / "p38_frozenlake_mismatch_capsule.round-000000.npz"
    np.savez(
        capsule,
        metadata_json=np.frombuffer(
            json.dumps({"diagnostic_round": 0}).encode(), dtype=np.uint8),
        selected_rows=np.asarray([255], dtype=np.int32),
        prompt_ids=prompt,
        prompt_mask=np.ones_like(prompt, dtype=np.bool_),
        completion_ids=completion,
        completion_valid_mask=np.ones_like(completion, dtype=np.bool_),
        action_mask=action,
        s_decode=decode,
        s_prefill=prefill,
    )
    tokens = np.concatenate((prompt[0], completion[0]))
    positions = list(range(red_points))
    return capsule, {
        "prefixes": [_prefix(tokens[:position + 1]) for position in positions],
        "positions": positions,
        "source_tokens": [int(tokens[position]) for position in positions],
        "target_ids": [int(completion[0, position]) for position in positions],
        "decode": [float(decode[0, position]) for position in positions],
        "prefill": [float(prefill[0, position]) for position in positions],
    }

  def _seal(self, root: Path) -> tuple[Path, dict]:
    seam_records = len(list(root.glob("p38_seam_*.json")))
    tail_records = len(list(root.glob("p38_tail_*.json")))
    (root / "pre-alignment.jsonl").write_text('{"diagnostic_round":0}\n')
    (root / "run.log").write_text("partial round\n")
    (root / "ROUND_INVENTORY.json").write_text(json.dumps({
        "schema": "canon-p38-round-stage-v1",
        "diagnostic_round": 0,
        "seam_records": seam_records,
        "tail_records": tail_records,
    }))
    inputs = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.name not in ("ROUND_COMPLETE.json", "SHA256SUMS")
    )
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in inputs))
    manifest_sha = _sha256(manifest)
    (root / "ROUND_COMPLETE.json").write_text(json.dumps({
        "schema": "canon-p38-round-completion-v1",
        "diagnostic_round": 0,
        "status": "sealed-and-verified",
        "source_commit": SOURCE_COMMIT,
        "manifest_sha256": manifest_sha,
    }))
    listing = root.parent / "OBJECT_LISTING.txt"
    objects = sorted(path.name for path in root.iterdir() if path.is_file())
    listing.write_text("".join(f"{SOURCE_URI}/{name}\n" for name in objects))
    return listing, {
        "seam_records": seam_records,
        "tail_records": tail_records,
        "manifest_files": len(inputs),
        "object_count": len(objects),
        "manifest_sha": manifest_sha,
    }

  def _fixture(
      self,
      root: Path,
      *,
      red_points: int = 1,
      seam_alias: bool = False,
      seam_conflict: bool = False,
      tail_alias: bool = False,
      tail_conflict: bool = False,
      omit_seam_b: bool = False,
      omit_tail_b: bool = False,
  ) -> tuple[Path, Path, dict]:
    source = root / "source"
    source.mkdir()
    capsule, data = self._write_capsule(source, red_points)
    self._write_seam(
        source, 10, "A", data["prefixes"], data["positions"],
        data["source_tokens"])
    if not omit_seam_b:
      self._write_seam(
          source, 20, "B", data["prefixes"], data["positions"],
          data["source_tokens"])
    if seam_alias or seam_conflict:
      self._write_seam(
          source, 11, "A", data["prefixes"], data["positions"],
          data["source_tokens"], mutate_row=0 if seam_conflict else None)
    self._write_tail(
        source, 30, "A", data["prefixes"], data["positions"],
        data["source_tokens"], data["target_ids"], data["decode"])
    if not omit_tail_b:
      self._write_tail(
          source, 40, "B", data["prefixes"], data["positions"],
          data["source_tokens"], data["target_ids"], data["prefill"])
    if tail_alias or tail_conflict:
      self._write_tail(
          source, 31, "A", data["prefixes"], data["positions"],
          data["source_tokens"], data["target_ids"], data["decode"],
          mutate_row=0 if tail_conflict else None)
    listing, contract = self._seal(source)
    contract["capsule"] = capsule
    return source, listing, contract

  def _run(self, source: Path, listing: Path, contract: dict, output: Path):
    return subprocess.run([
        sys.executable, str(REDUCER),
        "--source-dir", str(source),
        "--source-gcs-uri", SOURCE_URI,
        "--object-listing", str(listing),
        "--capsule", str(contract["capsule"]),
        "--output-dir", str(output),
        "--mode", "layer",
        "--analysis-source-commit", ANALYSIS_COMMIT,
        "--expected-source-commit", SOURCE_COMMIT,
        "--expected-manifest-sha256", contract["manifest_sha"],
        "--expected-diagnostic-round", "0",
        "--expected-seam-records", str(contract["seam_records"]),
        "--expected-tail-records", str(contract["tail_records"]),
        "--expected-object-count", str(contract["object_count"]),
        "--expected-manifest-files", str(contract["manifest_files"]),
        "--expected-red-points", str(contract.get("red_points", 1)),
        "--expected-rounds", "3",
    ], text=True, capture_output=True, check=False)

  def _audit(self, output: Path, audit: Path):
    return subprocess.run([
        sys.executable, str(AUDITOR), "--bundle-dir", str(output),
        "--output", str(audit),
    ], text=True, capture_output=True, check=False)

  def _reseal_bundle(self, output: Path) -> None:
    manifest = output / "SHA256SUMS"
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path != manifest)
    manifest.write_text("".join(
        f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
        for path in files))

  def test_unique_seam_and_tail_are_reclassified_and_audited(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      report = json.loads((output / "classification.json").read_text())
      self.assertTrue(report["tail_observer_required_and_joined"])
      self.assertEqual(report["joined_red_points"], 1)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertEqual(manifest["matched_seam_keys"], 2)
      self.assertEqual(manifest["matched_tail_keys"], 2)
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 0, audit.stderr)

  def test_equivalent_seam_and_tail_aliases_are_enumerated(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(
          root, seam_alias=True, tail_alias=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertEqual(len(manifest["equivalent_alias_keys"]), 1)
      self.assertEqual(len(manifest["tail_equivalent_alias_keys"]), 1)
      self.assertTrue((output / "candidates/p38_seam_000011.npz").is_file())
      self.assertTrue((output / "candidates/p38_tail_000031.npz").is_file())
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 0, audit.stderr)

  def test_seam_payload_conflict_remains_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, seam_conflict=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      self.assertFalse((output / "classification.json").exists())
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 0, audit.stderr)

  def test_tail_payload_conflict_remains_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, tail_conflict=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 0, audit.stderr)

  def test_missing_tail_arm_remains_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, omit_tail_b=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertEqual(manifest["matched_tail_keys"], 1)

  def test_missing_seam_arm_remains_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, omit_seam_b=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 4, result.stderr)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      self.assertEqual(manifest["matched_seam_keys"], 1)

  def test_direct_classifier_negative_control_detects_duplicate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, _, contract = self._fixture(root, seam_alias=True)
      result = subprocess.run([
          sys.executable, str(CLASSIFIER), "--directory", str(source),
          "--capsule", str(contract["capsule"]), "--mode", "layer",
          "--require-tail", "--output", str(root / "classification.json"),
      ], text=True, capture_output=True, check=False)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("duplicate seam token-prefix record", result.stderr)

  def test_32_red_points_join_all_64_seam_and_tail_keys(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, red_points=32)
      contract["red_points"] = 32
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      manifest = json.loads((output / "REDUCTION_MANIFEST.json").read_text())
      report = json.loads((output / "classification.json").read_text())
      self.assertEqual(manifest["required_arm_keys"], 64)
      self.assertEqual(manifest["matched_seam_keys"], 64)
      self.assertEqual(manifest["matched_tail_keys"], 64)
      self.assertEqual(report["joined_red_points"], 32)

  def test_empty_source_uri_and_bundle_tamper_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root)
      command = [
          sys.executable, str(REDUCER), "--source-dir", str(source),
          "--source-gcs-uri", "", "--object-listing", str(listing),
          "--capsule", str(contract["capsule"]), "--output-dir",
          str(root / "bad"), "--mode", "layer",
          "--analysis-source-commit", ANALYSIS_COMMIT,
          "--expected-source-commit", SOURCE_COMMIT,
          "--expected-manifest-sha256", contract["manifest_sha"],
          "--expected-diagnostic-round", "0", "--expected-seam-records",
          str(contract["seam_records"]), "--expected-tail-records",
          str(contract["tail_records"]), "--expected-object-count",
          str(contract["object_count"]), "--expected-manifest-files",
          str(contract["manifest_files"]), "--expected-red-points", "1",
      ]
      empty = subprocess.run(command, text=True, capture_output=True, check=False)
      self.assertEqual(empty.returncode, 2)
      output = root / "output"
      good = self._run(source, listing, contract, output)
      self.assertEqual(good.returncode, 0, good.stderr)
      with (output / "records/p38_tail_000030.npz").open("ab") as stream:
        stream.write(b"tamper")
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 2)
      self.assertIn("bundle SHA failed", audit.stderr)

  def test_each_required_bundle_class_is_sha_protected(self):
    targets = (
        "SOURCE_ROUND_INVENTORY.json",
        "records/p38_seam_000010.npz",
        "capsules/p38_frozenlake_mismatch_capsule.round-000000.npz",
        "REDUCTION_MANIFEST.json",
        "classification.json",
    )
    for target in targets:
      with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, listing, contract = self._fixture(root)
        output = root / "output"
        result = self._run(source, listing, contract, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / target).open("ab") as stream:
          stream.write(b"tamper")
        audit = self._audit(output, root / "audit.json")
        self.assertEqual(audit.returncode, 2)
        self.assertIn("bundle SHA failed", audit.stderr)

  def test_resealed_forged_classifier_output_is_semantically_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      classification = json.loads((output / "classification.json").read_text())
      classification["classification"] = "forged"
      (output / "classification.json").write_text(
          json.dumps(classification, sort_keys=True, indent=2) + "\n")
      self._reseal_bundle(output)
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 2)
      self.assertIn("classifier output differs", audit.stderr)

  def test_resealed_forged_alias_selection_is_semantically_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, listing, contract = self._fixture(root, seam_alias=True)
      output = root / "output"
      result = self._run(source, listing, contract, output)
      self.assertEqual(result.returncode, 0, result.stderr)
      manifest_path = output / "REDUCTION_MANIFEST.json"
      manifest = json.loads(manifest_path.read_text())
      alias_entry = next(
          entry for entry in manifest["join_entries"]
          if entry["resolution"] == "equivalent_alias")
      alias_entry["selected"] = alias_entry["candidates"][1]
      manifest_path.write_text(
          json.dumps(manifest, sort_keys=True, indent=2) + "\n")
      self._reseal_bundle(output)
      audit = self._audit(output, root / "audit.json")
      self.assertEqual(audit.returncode, 2)
      self.assertIn("alias decisions differ", audit.stderr)

  def test_gcs_wrapper_runs_fixed_contract_and_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      fixture_root = root / "fixture"
      fixture_root.mkdir()
      source, _, values = self._fixture(fixture_root)
      gcs = root / "gcs"
      gcs_source = gcs / SOURCE_URI.removeprefix("gs://")
      gcs_source.parent.mkdir(parents=True)
      shutil.copytree(source, gcs_source)
      destination = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          "canon-p38-test/attempt-0/derived/"
          "p38s18r2-round0-seam-tail-reduction-v2"
      )
      contract = root / "contract.json"
      contract.write_text(json.dumps({
          "schema": "p38s18r2-round0-reduction-contract-v1",
          "source_gcs_uri": SOURCE_URI,
          "destination_gcs_uri": destination,
          "expected_source_commit": SOURCE_COMMIT,
          "expected_source_manifest_sha256": values["manifest_sha"],
          "expected_diagnostic_round": 0,
          "expected_seam_records": values["seam_records"],
          "expected_tail_records": values["tail_records"],
          "expected_object_count": values["object_count"],
          "expected_manifest_files": values["manifest_files"],
          "expected_red_points": 1,
          "expected_rounds": 3,
          "mode": "layer",
          "require_tail": True,
          "max_output_bytes": 180000000,
      }))
      bin_dir = root / "bin"
      bin_dir.mkdir()
      self._write_fake_tools(bin_dir)
      env = dict(os.environ)
      env["FAKE_GCS_ROOT"] = str(gcs)
      env["FAKE_GCS_LOG"] = str(root / "fake-gcs.log")
      env["PATH"] = f"{bin_dir}:{env['PATH']}"
      returned = root / "returned"
      command = [
          "bash", str(WRAPPER), str(contract), str(root), str(returned)
      ]
      result = subprocess.run(
          command, text=True, capture_output=True, check=False, env=env)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("matched_seam_keys=2 matched_tail_keys=2", result.stdout)
      self.assertIn(f"analysis_source_commit={ANALYSIS_COMMIT}", result.stdout)
      destination_root = gcs / destination.removeprefix("gs://")
      self.assertTrue((destination_root / "files/SHA256SUMS").is_file())
      self.assertTrue((destination_root / "bundle-audit.json").is_file())
      self.assertTrue((returned / "files/SHA256SUMS").is_file())
      self.assertTrue((returned / "bundle-audit.json").is_file())
      audit = json.loads((destination_root / "bundle-audit.json").read_text())
      self.assertEqual(audit["bundle_integrity"], "PASS")
      storage_ops = (root / "fake-gcs.log").read_text().splitlines()
      self.assertTrue(storage_ops[-1].endswith(
          f"{destination}/files/SHA256SUMS"))
      repeated = subprocess.run(
          ["bash", str(WRAPPER), str(contract), str(root)],
          text=True, capture_output=True, check=False, env=env)
      self.assertEqual(repeated.returncode, 3)
      self.assertIn("destination already exists", repeated.stderr)


if __name__ == "__main__":
  unittest.main()
