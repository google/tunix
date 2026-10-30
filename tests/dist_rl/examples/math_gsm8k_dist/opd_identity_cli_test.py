# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Caller-declared teacher revisions reach OPD without endpoint inference."""

import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from tunix.dist_rl.distillation import identity
from tunix.dist_rl.distillation import teacher
from tunix.dist_rl.examples.math_gsm8k_dist import run_gsm8k_dist_grpo as entry


class OPDIdentityCliTest(unittest.TestCase):

  def _kwargs(self, *extra):
    args = entry._parse_args([
        "--opd_teacher_url=http://teacher:30001/revision-looking-url",
        "--opd_teacher_tokenizer_path=/teacher/tokenizer",
        *extra,
    ])
    scorer = mock.Mock(tokenizer_fingerprint="semantic-tokenizer-digest")
    with mock.patch.object(entry.AutoTokenizer, "from_pretrained"):
      with mock.patch.object(
          teacher, "SGLangTeacherScorer", return_value=scorer
      ):
        return entry._opd_program_kwargs(args, object())

  def test_explicit_revision_and_semantic_fingerprint_reach_program(self):
    kwargs = self._kwargs("--opd_teacher_revision=sha256:fixed-weights")
    self.assertEqual(
        kwargs["opd_teacher_identity"],
        identity.TeacherIdentity(
            declared_revision="sha256:fixed-weights",
            tokenizer_fingerprint="semantic-tokenizer-digest",
        ),
    )

  def test_endpoint_does_not_supply_missing_revision(self):
    teacher_identity = self._kwargs()["opd_teacher_identity"]
    self.assertIsNone(teacher_identity.declared_revision)
    self.assertEqual(
        teacher_identity.tokenizer_fingerprint, "semantic-tokenizer-digest"
    )

  def test_empty_revision_declaration_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "declared_revision"):
      self._kwargs("--opd_teacher_revision=  ")

  def test_launcher_forwards_declared_revision_as_one_argument(self):
    launcher = (
        Path(__file__).resolve().parents[4]
        / "tunix/dist_rl/examples/math_gsm8k_dist/launcher.sh"
    ).read_text()
    start = launcher.index("  ORCHESTRATOR_CMD=(")
    end = launcher.index("  export JAX_PLATFORMS=cpu", start)
    script = (
        launcher[: launcher.index("TRAINER_TPU_CHIPS=")]
        + launcher[start:end]
        + '\nprintf "%s\\0" "${ORCHESTRATOR_CMD[@]}"\n'
    )
    result = subprocess.run(
        ["bash"],
        input=script.encode(),
        env={
            "PATH": os.environ["PATH"],
            "OPD_TEACHER_URL": "http://teacher:30001",
            "OPD_TEACHER_TOKENIZER_PATH": "/teacher/tokenizer",
            "OPD_TEACHER_REVISION": "immutable snapshot digest",
        },
        capture_output=True,
        check=True,
    )
    self.assertIn(
        "--opd_teacher_revision=immutable snapshot digest",
        result.stdout.decode().rstrip("\0").split("\0"),
    )


if __name__ == "__main__":
  unittest.main()
