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
WRAPPER = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "run_reduce_p38s18l_on_gcp.sh"
)
AUDITOR = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "audit_p38_seam_reduction.py"
)
RUN_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    "canon-p38-test/attempt-0"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_gcs(root: Path, uri: str) -> Path:
  assert uri.startswith("gs://")
  return root / uri.removeprefix("gs://")


class ReduceP38GcpWrapperTest(unittest.TestCase):

  def _write_snapshot(
      self, gcs: Path, snapshot: str, rounds: tuple[int, ...]
  ) -> None:
    uri = f"{RUN_ROOT}/live/{snapshot}"
    root = _local_gcs(gcs, uri)
    root.mkdir(parents=True)
    record_index = 0
    pre_alignment = []
    for diagnostic_round in rounds:
      prompt = np.asarray([[11, 12]], dtype=np.int32)
      completion = np.asarray([[21 + diagnostic_round, 22, 23]], dtype=np.int32)
      tokens = np.concatenate((prompt[0], completion[0]))
      prefix = hashlib.sha256(
          np.ascontiguousarray(tokens[:3], dtype="<i8").tobytes()
      ).hexdigest().encode()
      for arm in ("A", "B"):
        arrays = {
            "row_indices": np.asarray([255], dtype=np.int32),
            "positions": np.asarray([2], dtype=np.int32),
            "token_ids": np.asarray([tokens[2]], dtype=np.int32),
            "request_ordinals": np.asarray([0], dtype=np.int32),
            "token_prefix_sha256": np.asarray([prefix], dtype="S64"),
            "layer_fingerprints": np.zeros((1, 2, 2, 8), dtype=np.uint32),
            "final_norm_fingerprints": np.zeros((1, 8), dtype=np.uint32),
        }
        npz = root / f"p38_seam_{record_index:06d}.npz"
        np.savez(npz, **arrays)
        (root / f"p38_seam_{record_index:06d}.json").write_text(json.dumps({
            "schema": "p38-seam-fingerprint-v1",
            "record_index": record_index,
            "arm": arm,
            "diagnostic_round": diagnostic_round,
            "observer_mode": "layer",
            "checkpoint_names": ["layer_input", "layer_output"],
            "layer_indices": [0, 1],
            "call_index": record_index,
            "program_path": "standard",
            "requests": [],
            "npz_sha256": _sha256(npz),
        }))
        record_index += 1
      capsule = root / (
          "p38_frozenlake_mismatch_capsule."
          f"round-{diagnostic_round:06d}.npz")
      np.savez(
          capsule,
          metadata_json=np.frombuffer(json.dumps({
              "diagnostic_round": diagnostic_round}).encode(), dtype=np.uint8),
          selected_rows=np.asarray([255], dtype=np.int32),
          prompt_ids=prompt,
          prompt_mask=np.ones_like(prompt, dtype=np.bool_),
          completion_ids=completion,
          completion_valid_mask=np.ones_like(completion, dtype=np.bool_),
          action_mask=np.ones((1, 3), dtype=np.bool_),
          s_decode=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
          s_prefill=np.asarray([[0.0, 1.5, 2.0]], dtype=np.float32),
      )
      pre_alignment.append(json.dumps({"diagnostic_round": diagnostic_round}))
    (root / "pre-alignment.jsonl").write_text("\n".join(pre_alignment) + "\n")
    (root / "run.log").write_text("partial run\n")
    (root / "LIVE.json").write_text(json.dumps({
        "schema": "canon-p38-gcs-live-v1", "prefix": uri}))
    inputs = sorted(
        path for path in root.iterdir()
        if path.name not in ("LIVE.json", "SHA256SUMS"))
    (root / "SHA256SUMS").write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in inputs))

  def _write_fake_gcloud(self, directory: Path) -> None:
    script = directory / "gcloud"
    script.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(r'''
        import os
        from pathlib import Path
        import shutil
        import sys

        root = Path(os.environ["FAKE_GCS_ROOT"])

        def local(uri):
          if not uri.startswith("gs://"):
            return Path(uri)
          return root / uri[5:]

        args = sys.argv[1:]
        if args[:2] == ["storage", "ls"]:
          recursive = "--recursive" in args
          target = args[-1]
          target = target[:-3] if target.endswith("/**") else target
          path = local(target)
          if not path.exists():
            raise SystemExit(1)
          if path.is_file():
            print(target)
          elif recursive:
            for item in sorted(path.rglob("*")):
              if item.is_file():
                print("gs://" + item.relative_to(root).as_posix())
          else:
            for item in sorted(path.iterdir()):
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
          print("unsupported fake gcloud command", args, file=sys.stderr)
          raise SystemExit(2)
        '''))
    script.chmod(0o755)

  def test_wrapper_selects_two_round_snapshot_and_returns_auditable_bundle(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      gcs = root / "gcs"
      self._write_snapshot(gcs, "000019", (0, 1))
      self._write_snapshot(gcs, "000020", (0,))
      bin_dir = root / "bin"
      bin_dir.mkdir()
      self._write_fake_gcloud(bin_dir)
      derived = f"{RUN_ROOT}/derived/p38s18l-seam-reduction-v2"
      env = dict(os.environ)
      env["FAKE_GCS_ROOT"] = str(gcs)
      env["PATH"] = f"{bin_dir}:{env['PATH']}"
      result = subprocess.run(
          ["bash", str(WRAPPER), f"{RUN_ROOT}/live", derived, str(root)],
          text=True,
          capture_output=True,
          check=False,
          env=env,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("snapshot=000019", result.stdout)
      self.assertIn("red_points=2 matched_arm_keys=4", result.stdout)
      bundle = _local_gcs(gcs, derived) / "files"
      self.assertTrue((bundle / "SNAPSHOT_SELECTION.json").is_file())
      self.assertTrue((bundle / "AMBIGUITY_AUDIT.json").is_file())
      audit = subprocess.run(
          [
              sys.executable, str(AUDITOR),
              "--bundle-dir", str(bundle),
              "--output", str(root / "bundle-audit.json"),
          ],
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(audit.returncode, 0, audit.stderr)
      self.assertIn("red_points=2 matched_arm_keys=4", audit.stdout)


if __name__ == "__main__":
  unittest.main()
