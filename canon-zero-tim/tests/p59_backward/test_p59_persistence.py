#!/usr/bin/env python3
"""Local transport gates for the P59 in-pod persistence wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = (
    ROOT
    / "canon-zero-tim"
    / "tasks"
    / "p59-dp16-parallel-backward"
    / "scripts"
    / "run_and_persist.sh"
)


class P59PersistenceTest(unittest.TestCase):

  def _environment(self, root: Path, *, require_xprof: bool) -> dict[str, str]:
    state = root / "state"
    state.mkdir()
    fake_bin = root / "bin"
    fake_bin.mkdir()
    fake_gcs = root / "gcs"
    fake_gcs.mkdir()
    fake_pkg = root / "canon-zero-tim"
    classifier = fake_pkg / "tests" / "p59_backward" / "classify_and_analyze.py"
    classifier.parent.mkdir(parents=True)
    classifier.write_text(
        textwrap.dedent(
            """\
            import argparse, json, pathlib
            p = argparse.ArgumentParser()
            p.add_argument('--kind'); p.add_argument('--run-log')
            p.add_argument('--pre-alignment-report'); p.add_argument('--update-report')
            p.add_argument('--alignment-report'); p.add_argument('--output')
            a = p.parse_args()
            pathlib.Path(a.output).write_text(
                json.dumps({'verdict': 'PASS'}) + '\\n', encoding='utf-8'
            )
            """
        ),
        encoding="utf-8",
    )
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            test "$1" = storage
            op="$2"
            map_path() {
              case "$1" in
                gs://*) printf '%s/%s' "$P59_FAKE_GCS_ROOT" "${1#gs://}" ;;
                *) printf '%s' "$1" ;;
              esac
            }
            case "$op" in
              ls)
                target="$(map_path "$3")"
                test -e "$target"
                ;;
              cp)
                source="$(map_path "$3")"
                target="$(map_path "$4")"
                mkdir -p "$(dirname "$target")"
                cp -- "$source" "$target"
                ;;
              *) exit 64 ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    inner = (
        "printf '{}\\n' > \"$CANON_PRE_ALIGN_REPORT\"; "
        "printf '{}\\n' > \"$CANON_ALIGN_REPORT\"; "
        "printf '{}\\n' > \"$CANON_UPDATE_REPORT\""
    )
    if require_xprof:
      inner += (
          "; mkdir -p \"$CANON_XPROF_DIR/plugins/profile/test\"; "
          "printf trace > \"$CANON_XPROF_DIR/plugins/profile/test/xplane.pb\""
      )
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "P59_FAKE_GCS_ROOT": str(fake_gcs),
        "JOBSET_RESTART_ATTEMPT": "0",
        "CANON_STATE": str(state),
        "CANON_PKG": str(fake_pkg),
        "CANON_EXPECT_COMMIT": "a" * 40,
        "CANON_WANDB_RUN_NAME": "canon-p59-test",
        "CANON_P59_INNER_RUN_CMD": inner,
        "CANON_P59_GCS_PREFIX": (
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p59/"
            "canon-p59-test/attempt-0"
        ),
        "CANON_P59_REQUIRE_XPROF": "1" if require_xprof else "0",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_PRE_ALIGN_REPORT": str(state / "pre.jsonl"),
        "CANON_ALIGN_REPORT": str(state / "align.jsonl"),
        "CANON_UPDATE_REPORT": str(state / "updates.jsonl"),
        "CANON_XPROF_DIR": str(state / "xprof"),
        "JAX_PLATFORMS": "cpu",
    })
    return env

  def test_persists_reports_and_refuses_label_reuse(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      env = self._environment(root, require_xprof=False)
      first = subprocess.run(
          ["bash", str(WRAPPER)], env=env, text=True, capture_output=True
      )
      self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
      remote = (
          root
          / "gcs"
          / "yuxzhang-tunix-models"
          / "canon-zero-tim"
          / "evidence"
          / "p59"
          / "canon-p59-test"
          / "attempt-0"
      )
      self.assertTrue((remote / "EVIDENCE.tar").is_file())
      self.assertTrue((remote / "COMPLETE.json").is_file())
      shutil.rmtree(env["CANON_STATE"])
      Path(env["CANON_STATE"]).mkdir()
      second = subprocess.run(
          ["bash", str(WRAPPER)], env=env, text=True, capture_output=True
      )
      self.assertNotEqual(second.returncode, 0)
      self.assertIn("remote label has already been used", second.stderr)

  def test_persists_required_xprof(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      env = self._environment(root, require_xprof=True)
      result = subprocess.run(
          ["bash", str(WRAPPER)], env=env, text=True, capture_output=True
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      remote = (
          root
          / "gcs"
          / "yuxzhang-tunix-models"
          / "canon-zero-tim"
          / "evidence"
          / "p59"
          / "canon-p59-test"
          / "attempt-0"
      )
      self.assertTrue((remote / "XPROF.tar").is_file())
      self.assertTrue((remote / "XPROF.sha256").is_file())


if __name__ == "__main__":
  unittest.main()
