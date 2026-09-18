#!/usr/bin/env python3
"""Tests for the P38 capsule log recovery tool."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


_SCRIPT = Path(__file__).with_name("extract_p38_capsule.py")
_SPEC = importlib.util.spec_from_file_location("extract_p38_capsule", _SCRIPT)
extractor = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(extractor)


class ExtractP38CapsuleTest(unittest.TestCase):

  def _log(self, root: Path, *, corrupt: bool = False) -> Path:
    capsule = root / "source.npz"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    metadata = {
        "schema": "p38-frozenlake-mismatch-capsule-v1",
        "arrays": {
            "s_decode": {
                "sha256": hashlib.sha256(values.tobytes()).hexdigest()
            }
        },
    }
    np.savez_compressed(
        capsule,
        selected_rows=np.asarray([1], dtype=np.int32),
        metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
        s_decode=values,
    )
    payload = capsule.read_bytes()
    encoded = base64.b64encode(payload).decode()
    if corrupt:
      encoded = encoded[:-1] + ("A" if encoded[-1] != "A" else "B")
    log = root / "pod.log"
    chunks = [encoded[index:index + 76] for index in range(0, len(encoded), 76)]
    log.write_text(
        "\n".join([
            "prefix",
            "[CANON_P38_CAPSULE_ARTIFACT] path=/tmp/capsule.npz "
            f"bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()} "
            "encoding=base64",
            *(f"[CANON_P38_CAPSULE_B64] {chunk}" for chunk in chunks),
        ])
        + "\n",
        encoding="utf-8",
    )
    return log

  def test_recovers_and_verifies_capsule(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      output = root / "recovered.npz"
      result = extractor.recover(self._log(root), output)
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["selected_rows"], [1])
      self.assertTrue(output.is_file())

  def test_rejects_corrupt_transport(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      with self.assertRaises(ValueError):
        extractor.recover(
            self._log(root, corrupt=True), root / "recovered.npz"
        )


if __name__ == "__main__":
  unittest.main()
