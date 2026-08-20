"""CPU contracts for P46 attempt logs and completion postflight."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
RUN = PKG / "cluster" / "steps" / "90_run.sh"


class P46PostflightTest(unittest.TestCase):

  def _run(
      self,
      *,
      full_campaign: bool,
      lines: list[str],
      first_pass_census: bool = False,
  ):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    state = root / "state"
    logs = root / "logs"
    state.mkdir()
    workload = root / "workload.sh"
    workload.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "".join(f"printf '%s\\n' {line!r}\n" for line in lines),
        encoding="utf-8",
    )
    workload.chmod(0o755)
    base_log = logs / "campaign.log"
    (state / "env.sh").write_text(
        "\n".join((
            f"export CANON_RUN_CMD={str(workload)!r}",
            f"export CANON_RUN_CWD={str(root)!r}",
            f"export CANON_RUN_LOG={str(base_log)!r}",
            "export CANON_P46_EVALUATION=1",
            f"export CANON_P46_FULL_CAMPAIGN={int(full_campaign)}",
            f"export CANON_P46_CENSUS_FIRST_PASS={int(first_pass_census)}",
            "export CANON_P46_RESUME_TAG=resume-test",
            "export CANON_RUN_ID=launch-test",
            "",
        )),
        encoding="utf-8",
    )
    environ = os.environ.copy()
    environ.update({"CANON_STATE": str(state), "CANON_PKG": str(PKG)})
    result = subprocess.run(
        ["bash", str(RUN)],
        cwd=ROOT,
        env=environ,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result, base_log

  def test_full_campaign_uses_immutable_attempt_log_and_new_marker(self):
    lines = [
        f"P46_EVAL_CAMPAIGN_LOGICAL_PASS logical_shard={index}"
        for index in range(58)
    ]
    lines.append(
        "P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 "
        "valid_trajectories=29616 logical_shards=58 summary_sha256=" + "a" * 64
    )
    result, base_log = self._run(full_campaign=True, lines=lines)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("[P46.EVAL.POSTFLIGHT] PASS", result.stdout)
    self.assertFalse(base_log.exists())
    attempts = list(base_log.parent.glob("campaign.attempt-*.log"))
    self.assertEqual(len(attempts), 1)
    self.assertIn("P46_EVAL_CAMPAIGN_PASS", attempts[0].read_text())

  def test_full_campaign_rejects_timeout_even_with_a_pass_marker(self):
    lines = [
        f"P46_EVAL_CAMPAIGN_LOGICAL_PASS logical_shard={index}"
        for index in range(58)
    ]
    lines.extend((
        "P46_EVAL_CAMPAIGN_WAVE_TIMEOUT logical_shard=0 physical_shard=0 "
        "resume_tag=resume-test resume_same_tag=1",
        "P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 "
        "valid_trajectories=29616 logical_shards=58 summary_sha256=" + "b" * 64,
    ))
    result, _ = self._run(full_campaign=True, lines=lines)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("full-campaign completion marker contract failed", result.stdout)

  def test_census_accepts_deferred_invalid_and_bounded_wave_timeout(self):
    lines = [
        f"P46_EVAL_CENSUS_LOGICAL_COMPLETE logical_shard={index}"
        for index in range(58)
    ]
    lines.extend((
        "P46_EVAL_SHARD_TIMEOUT completed=63/64 deadline=3600s",
        "P46_EVAL_CENSUS_PASS tasks=1851 scheduled_identities=29616 "
        "attempted_identities=29616 valid_identities=29584 "
        "deferred_invalid=32 unattempted=0 q4_learnable=609 "
        "logical_shards=58 summary_sha256=" + "c" * 64,
    ))
    result, _ = self._run(
        full_campaign=True,
        first_pass_census=True,
        lines=lines,
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("[P46.EVAL.POSTFLIGHT] PASS mode=census", result.stdout)

  def test_census_incomplete_marker_is_not_a_pass(self):
    lines = [
        f"P46_EVAL_CENSUS_LOGICAL_COMPLETE logical_shard={index}"
        for index in range(58)
    ]
    lines.append(
        "P46_EVAL_CENSUS_INCOMPLETE tasks=1851 "
        "scheduled_identities=29616 attempted_identities=29615"
    )
    result, _ = self._run(
        full_campaign=True,
        first_pass_census=True,
        lines=lines,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("census completion marker contract failed", result.stdout)

  def test_legacy_single_wave_marker_remains_supported(self):
    result, base_log = self._run(
        full_campaign=False,
        lines=["P46_EVAL_SUBSHARD_PASS tag=test"],
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertTrue(base_log.exists())


if __name__ == "__main__":
  unittest.main()
