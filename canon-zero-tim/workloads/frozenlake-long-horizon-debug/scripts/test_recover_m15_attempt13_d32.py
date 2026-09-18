#!/usr/bin/env python3
"""Transport-contract tests for the Attempt-13 bucket wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("recover_m15_attempt13_d32.sh")


class Attempt13D32TransportTest(unittest.TestCase):

  def test_wrapper_queries_flat_shards_not_new_multiround_layout(self) -> None:
    with tempfile.TemporaryDirectory() as holder:
      root = Path(holder)
      binary = root / "bin"
      binary.mkdir()
      log = root / "gcloud.log"
      fake = binary / "gcloud"
      fake.write_text(
          "#!/usr/bin/env bash\n"
          "set -euo pipefail\n"
          "printf '%s\\n' \"$*\" >> \"$FAKE_GCLOUD_LOG\"\n"
          "test \"${1:-}\" = storage\n"
          "test \"${2:-}\" = ls\n"
          "exit 1\n",
          encoding="utf-8",
      )
      fake.chmod(0o755)
      environment = dict(os.environ)
      environment["PATH"] = f"{binary}:/usr/bin:/bin"
      environment["FAKE_GCLOUD_LOG"] = str(log)
      result = subprocess.run(
          ["bash", str(SCRIPT), str(root / "return"), str(root)],
          check=False,
          capture_output=True,
          text=True,
          env=environment,
      )
      self.assertNotEqual(result.returncode, 0)
      calls = log.read_text(encoding="utf-8")
      self.assertIn("/wide/shards/*/SHARD_COMPLETE.json", calls)
      self.assertNotIn("/wide/rounds/", calls)

  def test_wrapper_does_not_fabricate_a_three_round_contract(self) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    self.assertNotIn("CANON_P38_DIAGNOSTIC_ROUNDS", source)
    self.assertNotIn("run_m15_multiround_gcs_return.sh", source)
    self.assertIn("rounds=1", source)


if __name__ == "__main__":
  unittest.main()
