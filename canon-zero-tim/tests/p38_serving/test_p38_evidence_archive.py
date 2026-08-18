#!/usr/bin/env python3
"""Contract tests for bounded-object P38 evidence transport."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _ROOT
    / "tasks/p38-pathways-decode-prefill-carrier/scripts"
    / "p38_evidence_archive.py"
)
_SPEC = importlib.util.spec_from_file_location("p38_evidence_archive", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
archive_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(archive_module)


class P38EvidenceArchiveTest(unittest.TestCase):

  def _fixture(self, root: Path, count: int) -> Path:
    records = []
    for index in range(count):
      name = f"p38_terminal_{index:06d}.npz"
      payload = f"record-{index}\n".encode()
      (root / name).write_bytes(payload)
      records.append((name, hashlib.sha256(payload).hexdigest()))
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in records),
        encoding="utf-8",
    )
    return manifest

  def test_5246_logical_files_have_one_deterministic_archive(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp) / "root"
      root.mkdir()
      manifest = self._fixture(root, 5246)
      first = Path(temp) / "first.tar"
      second = Path(temp) / "second.tar"
      first_count, first_sha = archive_module.create_archive(root, manifest, first)
      second_count, second_sha = archive_module.create_archive(root, manifest, second)
      self.assertEqual(first_count, 5246)
      self.assertEqual(second_count, 5246)
      self.assertEqual(first_sha, second_sha)
      self.assertEqual(first.read_bytes(), second.read_bytes())
      verified_count, verified_sha = archive_module.verify_archive(
          first, first_sha
      )
      self.assertEqual(verified_count, 5246)
      self.assertEqual(verified_sha, first_sha)

  def test_extract_reconstructs_manifest_exactly(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp) / "root"
      root.mkdir()
      manifest = self._fixture(root, 4)
      archive = Path(temp) / "round.tar"
      _, archive_sha = archive_module.create_archive(root, manifest, archive)
      output = Path(temp) / "extracted"
      count, extracted_sha = archive_module.extract_archive(archive, output)
      self.assertEqual(count, 4)
      self.assertEqual(extracted_sha, archive_sha)
      self.assertEqual((output / "SHA256SUMS").read_bytes(), manifest.read_bytes())
      for name in manifest.read_text(encoding="utf-8").splitlines():
        self.assertTrue((output / name.split("  ", 1)[1]).is_file())

  def test_archive_bit_flip_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp) / "root"
      root.mkdir()
      manifest = self._fixture(root, 2)
      archive = Path(temp) / "round.tar"
      _, expected_sha = archive_module.create_archive(root, manifest, archive)
      payload = bytearray(archive.read_bytes())
      payload[-1] ^= 1
      archive.write_bytes(payload)
      with self.assertRaisesRegex(ValueError, "archive SHA failed"):
        archive_module.verify_archive(archive, expected_sha)

  def test_missing_manifest_member_is_rejected_before_creation(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp) / "root"
      root.mkdir()
      manifest = self._fixture(root, 2)
      (root / "p38_terminal_000001.npz").unlink()
      with self.assertRaisesRegex(ValueError, "absent or unsafe"):
        archive_module.create_archive(root, manifest, Path(temp) / "round.tar")


if __name__ == "__main__":
  unittest.main()
