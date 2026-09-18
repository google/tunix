#!/usr/bin/env python3
"""Negative controls for P38 serving-archive transport."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


_MODULE_PATH = Path(__file__).with_name("extract_p38_serving_archive.py")
_SPEC = importlib.util.spec_from_file_location("extract_p38_serving_archive", _MODULE_PATH)
assert _SPEC and _SPEC.loader
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def _log(directory: Path, payload: bytes, *, sha: str | None = None) -> Path:
  digest = sha or hashlib.sha256(payload).hexdigest()
  encoded = base64.b64encode(payload).decode("ascii")
  lines = [
      f"[CANON_P38_SERVING_ARCHIVE] path=/tmp/capture.tar bytes={len(payload)} sha256={digest} encoding=base64",
      f"[CANON_P38_SERVING_ARCHIVE_B64] {encoded}",
  ]
  path = directory / "run.log"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


class ExtractP38ServingArchiveTest(unittest.TestCase):

  def test_extracts_verified_payload(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      payload = b"bounded serving archive"
      output = directory / "capture.tar"
      result = module.extract(_log(directory, payload), output)
      self.assertEqual(output.read_bytes(), payload)
      self.assertEqual(result["bytes"], len(payload))

  def test_rejects_sha_mismatch(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
        module.extract(
            _log(directory, b"payload", sha="0" * 64),
            directory / "capture.tar",
        )

  def test_rejects_missing_payload(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      log = directory / "run.log"
      log.write_text(
          "[CANON_P38_SERVING_ARCHIVE] path=/tmp/x bytes=1 "
          f"sha256={'0' * 64} encoding=base64\n",
          encoding="utf-8",
      )
      with self.assertRaisesRegex(RuntimeError, "payload is missing"):
        module.extract(log, directory / "capture.tar")

  def test_rejects_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      output = directory / "capture.tar"
      output.write_bytes(b"existing")
      with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
        module.extract(_log(directory, b"new"), output)


if __name__ == "__main__":
  unittest.main()
