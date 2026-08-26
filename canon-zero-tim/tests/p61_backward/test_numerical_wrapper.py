#!/usr/bin/env python3
"""Contract tests for durable P61 numerical A/B terminal evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
WRAPPER = (
    PACKAGE
    / "tasks/p61-backward-numerical-oracle/scripts/"
    "run_onehost_dp4_numerical_ab.sh"
)


def _executable(path: Path, source: str) -> None:
  path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
  path.chmod(path.stat().st_mode | stat.S_IXUSR)


class NumericalWrapperTest(unittest.TestCase):

  def test_classified_reject_is_manifested_and_never_green(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      evidence = root / "evidence"
      baseline = root / "tier1.json"
      baseline.write_text("{}\n", encoding="utf-8")
      runner = root / "runner.py"
      comparator = root / "comparator.py"
      _executable(
          runner,
          """
          #!/usr/bin/env bash
          set -euo pipefail
          mode="$1"
          label="$2"
          arm="$P61_NUMERICAL_EVIDENCE_ROOT/p59_dp4_${mode}_${label}"
          mkdir -p "$arm/train/p61_numerical"
          printf '{}\\n' >"$arm/train/updates.jsonl"
          printf '{}\\n' >"$arm/train/classification.json"
          printf 'fake-arm-manifest\\n' >"$arm/SHA256SUMS"
          """,
      )
      _executable(
          comparator,
          """
          #!/usr/bin/env python3
          import json
          from pathlib import Path
          import sys

          output = Path(sys.argv[sys.argv.index("--output") + 1])
          output.write_text(json.dumps({"verdict": "NUMERICAL_REJECT"}) + "\\n")
          raise SystemExit(1)
          """,
      )
      env = os.environ.copy()
      env.update({
          "P61_NUMERICAL_TEST_MODE": "1",
          "P61_NUMERICAL_TEST_RUNNER": str(runner),
          "P61_NUMERICAL_TEST_COMPARATOR": str(comparator),
          "P61_NUMERICAL_EVIDENCE_ROOT": str(evidence),
      })
      completed = subprocess.run(
          [
              "bash",
              str(WRAPPER),
              "serial_r1",
              "parallel_r1",
              "bundle_r1",
              str(baseline),
          ],
          cwd=PACKAGE.parent,
          env=env,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertEqual(completed.returncode, 1, completed.stderr)
      bundle = evidence / "p61_dp4_numerical_ab_bundle_r1"
      driver = (bundle / "driver.log").read_text(encoding="utf-8")
      self.assertIn("verdict=NUMERICAL_REJECT", driver)
      self.assertIn("classification=CLASSIFIED_NON_KEEP", driver)
      self.assertNotIn("GREEN", driver)
      self.assertTrue((bundle / "numerical_ab.json").is_file())
      self.assertTrue((bundle / "SHA256SUMS").is_file())
      manifest_check = subprocess.run(
          ["sha256sum", "-c", str(bundle / "SHA256SUMS")],
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertEqual(manifest_check.returncode, 0, manifest_check.stderr)
      result = json.loads((bundle / "numerical_ab.json").read_text())
      self.assertEqual(result["verdict"], "NUMERICAL_REJECT")


if __name__ == "__main__":
  unittest.main()
