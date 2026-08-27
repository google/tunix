from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".claude/skills/manage-canon-flags/scripts/audit_flag_registry.py"
)
SPEC = importlib.util.spec_from_file_location("audit_flag_registry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AddedFlagPathsTest(unittest.TestCase):

  def test_marker_inventory_is_separate_from_settable_flags(self):
    with tempfile.TemporaryDirectory() as tmp:
      flags = Path(tmp) / "FLAGS.md"
      marker = "CANON_" + "RUNTIME_MARKER"
      flag = "CANON_" + "RUNTIME_FLAG"
      flags.write_text(
          "## MARKERS\n\n"
          f"`[{marker}]` is observational.\n\n"
          "## Appendix\n\n"
          "Count: 1 settable names\n\n"
          f"```text\n{flag}\n```\n"
      )
      self.assertEqual(
          AUDIT._marker_inventory(flags), {marker}
      )
      self.assertEqual(
          AUDIT._inventory(flags), ([flag], 1)
      )

  def test_ignores_documentation_and_immutable_evidence_markers(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp)
      runtime_flag = "CANON_" + "RUNTIME_FLAG"
      documentation_marker = "CANON_" + "DOCUMENTATION_MARKER"
      evidence_marker = "CANON_" + "EVIDENCE_MARKER"
      debug_log_marker = "CANON_" + "DEBUG_LOG_MARKER"
      subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
      (repo / "runtime.py").write_text("VALUE = 1\n")
      subprocess.run(["git", "add", "."], cwd=repo, check=True)
      subprocess.run(
          [
              "git",
              "-c",
              "user.name=flag-audit-test",
              "-c",
              "user.email=flag-audit-test@example.invalid",
              "commit",
              "-qm",
              "Create test base",
          ],
          cwd=repo,
          check=True,
      )
      base = subprocess.run(
          ["git", "rev-parse", "HEAD"],
          cwd=repo,
          check=True,
          text=True,
          capture_output=True,
      ).stdout.strip()

      (repo / "runtime.py").write_text(
          f'VALUE = os.environ.get("{runtime_flag}")\n'
      )
      (repo / "notes.md").write_text(f"[{documentation_marker}]\n")
      evidence = repo / "tasks/example/evidence"
      evidence.mkdir(parents=True)
      (evidence / "run.log").write_text(f"[{evidence_marker}]\n")
      debug_logs = repo / "debug_logs"
      debug_logs.mkdir()
      (debug_logs / "raw.log").write_text(f"[{debug_log_marker}]\n")

      self.assertEqual(
          AUDIT._added_flags(repo, base),
          {runtime_flag},
      )


if __name__ == "__main__":
  unittest.main()
