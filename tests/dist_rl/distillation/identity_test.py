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

"""OPD checkpoint objective and caller-declared teacher identity contracts."""

import copy
import json
import unittest

from tunix.dist_rl.distillation import identity
from tunix.dist_rl.distillation import on_policy


def _manifest(**overrides):
  options = dict(
      config=on_policy.OnPolicyDistillationConfig(),
      sampling_contract={
          "temperature": 1.0,
          "top_p": 1.0,
          "top_k": 0,
          "return_logprobs": True,
      },
      teacher_identity=identity.TeacherIdentity(
          declared_revision="sha256:weights-a",
          tokenizer_fingerprint="tokenizer-a",
      ),
  )
  options.update(overrides)
  return identity.build_manifest(**options)


class IdentityTest(unittest.TestCase):

  def test_round_trip_and_disabled_baseline(self):
    manifest = _manifest()
    metadata = json.loads(json.dumps(identity.checkpoint_metadata(manifest)))
    identity.validate_resume_manifest(manifest, metadata)
    self.assertTrue(identity.has_resumable_identity(manifest))
    self.assertEqual(identity.checkpoint_metadata(None), {})
    identity.validate_resume_manifest(None, {"step": 3})
    identity.validate_resume_manifest(None, None)
    self.assertIsNone(identity.build_manifest(None, sampling_contract={}))

  def test_contract_and_checkpoint_values_are_detached(self):
    sampling = {"top_k": 0, "nested": {"value": 1}}
    manifest = _manifest(sampling_contract=sampling)
    sampling["nested"]["value"] = 2
    metadata = identity.checkpoint_metadata(manifest)
    metadata[identity.METADATA_KEY]["sampling"]["nested"]["value"] = 3
    self.assertEqual(manifest["sampling"]["nested"]["value"], 1)

  def test_equivalent_disabled_top_k(self):
    first = _manifest(sampling_contract={"top_k": -1})
    second = _manifest(sampling_contract={"top_k": 0})
    identity.validate_resume_manifest(
        first, identity.checkpoint_metadata(second)
    )

  def test_changed_objective_is_rejected_with_field_name(self):
    original = _manifest()
    for field, value in (
        (("coefficient",), 0.5),
        (("task_reward_coefficient",), 1.0),
        (("sampling", "return_logprobs"), False),
        (("sampling", "seed"), 42),
        (("teacher", "declared_revision"), "sha256:weights-b"),
        (("teacher", "tokenizer_fingerprint"), "tokenizer-b"),
    ):
      with self.subTest(field=field):
        changed = copy.deepcopy(original)
        target = changed
        for part in field[:-1]:
          target = target[part]
        target[field[-1]] = value
        with self.assertRaisesRegex(ValueError, r"\.".join(field)):
          identity.validate_resume_manifest(
              original, identity.checkpoint_metadata(changed)
          )

  def test_enabling_or_disabling_opd_on_resume_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "no opd_manifest"):
      identity.validate_resume_manifest(_manifest(), {"step": 1})
    with self.assertRaisesRegex(ValueError, "OPD disabled"):
      identity.validate_resume_manifest(
          None, identity.checkpoint_metadata(_manifest())
      )

  def test_incomplete_teacher_allows_manifest_but_not_resume(self):
    for teacher in (
        None,
        identity.TeacherIdentity(declared_revision="sha256:weights-a"),
        identity.TeacherIdentity(tokenizer_fingerprint="tokenizer-a"),
    ):
      with self.subTest(teacher=teacher):
        manifest = _manifest(teacher_identity=teacher)
        self.assertFalse(identity.has_resumable_identity(manifest))
        with self.assertRaisesRegex(ValueError, "declared teacher revision"):
          identity.validate_resume_manifest(
              manifest, identity.checkpoint_metadata(manifest)
          )

  def test_zero_coefficient_has_no_teacher_dependency(self):
    first = _manifest(
        config=on_policy.OnPolicyDistillationConfig(
            coefficient=0, task_reward_coefficient=1
        ),
        teacher_identity=None,
    )
    second = _manifest(
        config=on_policy.OnPolicyDistillationConfig(
            coefficient=0, task_reward_coefficient=1
        ),
    )
    self.assertIsNone(first["teacher"])
    self.assertTrue(identity.has_resumable_identity(first))
    identity.validate_resume_manifest(
        first, identity.checkpoint_metadata(second)
    )

  def test_bad_manifest_schema_is_rejected(self):
    for stored in (
        None,
        [],
        {},
        {"schema_version": 2},
        {"schema_version": True},
    ):
      with self.subTest(stored=stored):
        with self.assertRaisesRegex(ValueError, "invalid opd_manifest"):
          identity.validate_resume_manifest(
              _manifest(), {identity.METADATA_KEY: stored}
          )

  def test_invalid_declared_fields_and_disabled_configuration(self):
    for value in ("", "  ", 7, False):
      for name in ("declared_revision", "tokenizer_fingerprint"):
        with self.subTest(name=name, value=value):
          with self.assertRaisesRegex(ValueError, name):
            identity.TeacherIdentity(**{name: value})
    with self.assertRaisesRegex(ValueError, "requires an OPD"):
      _manifest(config=None)


if __name__ == "__main__":
  unittest.main()
